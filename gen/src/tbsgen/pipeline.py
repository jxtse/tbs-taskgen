"""Pipeline orchestrator: resumable stepping over item state machines, budget stops, daily report.

    P = Pipeline(cfg, clients={"teacher": FakeClient(...)})   # or real clients from cfg
    P.ingest(candidates)          # S1: one item dir per paper (dedup by item id)
    P.run(max_steps=...)          # advance every active item until it is terminal / waiting / paused
A BudgetStop during a stage pauses the item (status "paused", reason budget:<why>) and stops the run for
that model (other items needing the same model are not attempted in this run).  `resume` reactivates
paused items (e.g. next day / after raising a budget / after resetting a breaker).
ModelError from a stage counts as a stage failure: up to 2 tries per stage per run, then the item is parked.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from . import blocklist, intake, screen_jev, stages_a as A, stages_b as B
from .budget import BudgetStop, Ledger, today
from .config import resolve_role
from .models import ModelError, get_client
from .state import STAGES, Store, read_json, write_json

HANDLERS = {"S1": A.s1, "S2": A.s2, "S3": A.s3, "S4": A.s4, "S5": A.s5, "S6A": A.s6a, "S6A_run": A.s6a_run, "S6M": A.s6m,
            "S6B": A.s6b, "S6C": B.s6c, "S7": B.s7, "S8": B.s8, "PKG": B.pkg, "S9": B.s9, "S9R": B.s9r, "S10": B.s10}
MODEL_STAGES = {"S5": "spec", "S6A": "gen", "S6M": "gen", "S6C": "tests", "S8": "repair"}
# sub-stages that are not in state.STAGES (for `run --until`): S6A_run is part of S6A; S6M (a model call) comes
# after S6B, so `--until S6B` does not start it
_SUBSTAGE = {"S6A_run": ("S6A", 0.0), "S6M": ("S6B", 0.5)}
_STOP = {"a", "an", "the", "of", "for", "and", "in", "on", "with", "by", "to", "from", "using", "via", "based",
         "new", "novel", "fast", "efficient", "method", "methods", "approach", "algorithm", "data", "analysis"}


def _stage_pos(stage: str) -> float:
    s, off = _SUBSTAGE.get(stage, (stage, 0.0))
    return STAGES.index(s) + off if s in STAGES else float(len(STAGES))


def _title_words(title: str) -> frozenset:
    import re
    return frozenset(w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 2 and w not in _STOP)


class Pipeline:
    def __init__(self, cfg: dict, clients: dict | None = None, fetcher=None):
        root = cfg["run"]["root"]
        if not root:
            raise ValueError("cfg.run.root is required")
        self.cfg = cfg
        self.store = Store(root)
        self.ledger = Ledger(Path(root) / "ledger.sqlite", cfg)
        self._clients = dict(clients or {})
        self.fetcher = fetcher
        self._tbs = None
        self._jev = None
        self.jev_transport = None           # tests inject a fake HTTP transport here
        self.stopped_models: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def client(self, role: str):
        if role in self._clients:
            return self._clients[role]
        alias = resolve_role(self.cfg, role)
        if alias in self._clients:
            return self._clients[alias]
        c = get_client(alias, self.cfg, self.ledger)
        if c is None:
            raise ModelError("config", f"role {role} has no model (alias {alias})")
        self._clients[role] = c
        return c

    def fulltext_fetcher(self):
        """The S3 Fetcher (one per pipeline: one ledger connection, one egress check).  A test-injected
        fulltext.Fetcher takes precedence."""
        from .fulltext import Fetcher
        if self.fetcher is None:
            self.fetcher = Fetcher(self.cfg)
        return self.fetcher

    def jev_client(self) -> screen_jev.JevClient:
        if self._jev is None:
            self._jev = screen_jev.JevClient(self.ledger, self.cfg, transport=self.jev_transport)
        return self._jev

    def prefetch_screen(self, only: list[str] | None = None) -> int:
        """roles.screen = jev: ask Jev about every active S2 item with an abstract, `screen.jev.concurrency`
        requests at a time; S2 then reads the cached answers.  Failures are left for S2's fallback."""
        if resolve_role(self.cfg, "screen") != "jev":
            return 0
        its = [i for i in self.store.items() if i.state["status"] == "active" and i.state["stage"] == "S2"
               and (not only or i.item_id in only)]
        rules = self.cfg["screen"].get("blocklist")
        if rules:                           # blocked papers are dropped by S2 without a screen call (a rule the
            lifted = blocklist.lifted_rules(blocklist.load_state(self.store))     # recheck lifted blocks nothing;
            its = [i for i in its                                                # trial papers are screened by S2)
                   if blocklist.match(read_json(i.p("S1_candidate.json")) or {}, rules, lifted) is None]
        if not its:
            return 0
        try:
            self.jev_client().key()
        except ModelError:
            return 0                        # no key: S2 falls back per item (and records why)
        return screen_jev.prefetch(self.jev_client(), its, self.cfg)

    def tbs_index(self):
        if self._tbs is None:
            root = self.cfg["probe"].get("tbs_tasks_root")
            self._tbs = intake.TBSIndex(Path(root)) if root and Path(root).exists() else None
        return self._tbs

    def item_cost(self, item_id: str) -> dict:
        with self.ledger._db() as db:
            r = db.execute("SELECT count(*), SUM(COALESCE(usd,0)), SUM(in_tok), SUM(out_tok) FROM calls WHERE item=?",
                           (item_id,)).fetchone()
        return dict(calls=r[0], usd=round(r[1] or 0, 4), in_tok=r[2] or 0, out_tok=r[3] or 0)

    # ------------------------------------------------------------------ intake
    def ingest(self, candidates: list[dict]) -> list[str]:
        ids = []
        for c in candidates:
            iid = intake.item_id_for(c)
            it = self.store.item(iid)
            if not it.p("S1_candidate.json").exists():
                write_json(it.p("S1_candidate.json"), c)
                it.save()
                self.store.event("ingest", item=iid, doi=c.get("doi"))
            ids.append(iid)
        return ids

    # ------------------------------------------------------------------ run loop
    def step(self, it) -> str:
        st = it.state["stage"]
        role = MODEL_STAGES.get(st)
        alias = resolve_role(self.cfg, role) if role else None
        if role:
            name = getattr(self._clients.get(role) or self._clients.get(alias), "name", alias)
            if name in self.stopped_models or alias in self.stopped_models:
                return "skipped"
        self.store.event("stage_start", item=it.item_id, stage=st)
        t0 = time.time()
        try:
            HANDLERS[st](self, it)
        except BudgetStop as e:
            it.finish("paused", f"budget:{e.reason}", detail=e.detail, paused_stage=st)
            self.stopped_models.add(alias if role else st)
            self.store.event("budget_stop", item=it.item_id, stage=st, reason=e.reason, detail=e.detail)
            return "paused"
        except ModelError as e:
            n = it.bump(f"err:{st}")
            self.store.event("stage_error", item=it.item_id, stage=st, cls=e.cls, err=str(e)[:300])
            if e.cls in ("quota", "auth", "model_mismatch", "config"):
                it.finish("paused", f"model:{e.cls}", paused_stage=st)
                self.stopped_models.add(alias if role else st)
                return "paused"
            if n >= 2:
                it.finish("parked", f"model_error:{st}:{e.cls}")
            return "error"
        except Exception as e:  # noqa: BLE001  (bug or unexpected data: park the item, keep the run going)
            self.store.event("stage_crash", item=it.item_id, stage=st, err=f"{type(e).__name__}: {e}",
                             tb=traceback.format_exc()[-1500:])
            n = it.bump(f"crash:{st}")
            if n >= 2:
                it.finish("parked", f"crash:{st}:{type(e).__name__}")
            return "crash"
        self.store.event("stage_end", item=it.item_id, stage=st, to=it.state["stage"], status=it.state["status"],
                         sec=round(time.time() - t0, 2))
        return "ok"

    def order(self, items: list) -> list:
        """Items in stepping order (schedule.order): with a daily model-call cap only the first papers in this order
        reach S5 on a given day.  round_robin: targets take turns (target with the best paper first), within a
        target the S4 Jev feasibility score first when there is one (roles.probe=jev), else the S2 score;
        near-duplicate titles (schedule.similar_title_jaccard) move to the end."""
        sc = self.cfg.get("schedule") or {}
        mode = sc.get("order", "round_robin")
        if mode == "id":
            return list(items)
        info = {}
        for it in items:
            c = read_json(it.p("S1_candidate.json"), {}) or {}
            s4 = (read_json(it.p("S4_probe.json"), {}) or {}).get("jev", {}).get("score")
            s = (read_json(it.p("S2_screen.json"), {}) or {}).get("score")
            # (has an S4 score, S4 score, S2 score): S4 and S2 scores are on different scales, so a paper with an
            # S4 Jev score ranks above every paper that only has an S2 score; ties / no S4 fall back to S2
            s2 = float(s) if isinstance(s, (int, float)) else float("-inf")
            key = (1.0, float(s4), s2) if isinstance(s4, (int, float)) else (0.0, float("-inf"), s2)
            info[it.item_id] = (c.get("_target") or "", key, c.get("title") or "")
        by_score = sorted(items, key=lambda i: (tuple(-x for x in info[i.item_id][1]), i.item_id))
        if mode == "score":
            return by_score
        groups: dict[str, list] = {}
        for it in by_score:
            groups.setdefault(info[it.item_id][0], []).append(it)
        jac = float(sc.get("similar_title_jaccard", 0.5) or 0)
        for t, g in groups.items():
            if jac > 0:
                placed, later, words = [], [], []
                for it in g:
                    w = _title_words(info[it.item_id][2])
                    if w and any(len(w & p) / len(w | p) >= jac for p in words):
                        later.append(it)
                    else:
                        placed.append(it)
                        words.append(w)
                groups[t] = placed + later
        out, k = [], 0
        rows = sorted(groups.values(), key=lambda g: (tuple(-x for x in info[g[0].item_id][1]), info[g[0].item_id][0]))
        while any(k < len(g) for g in rows):
            out += [g[k] for g in rows if k < len(g)]
            k += 1
        return out

    def run(self, max_steps: int = 10000, only: list[str] | None = None, until: str | None = None) -> dict:
        """until: last stage to run (e.g. "S4"); items that reach a later stage are left active there, untouched."""
        if until is not None and until not in STAGES:
            raise ValueError(f"unknown stage {until!r}")
        last = _stage_pos(until) if until else None
        steps = 0
        progress = True
        if last is None or last >= _stage_pos("S2"):    # the S2 Jev prefetch is S2 work
            if self.cfg["screen"].get("blocklist"):
                # trial papers the fetch queue delivered since S3 parked them count as successes now; then this
                # week's trial quota picks papers an earlier week's S2 dropped by a rule
                blocklist.reevaluate(self.store, self.cfg, self.store.event)
                if not only:
                    blocklist.revive_for_trials(self.store, self.cfg, self.store.event)
            self.prefetch_screen(only)

        rewoken: set[str] = set()

        def runnable(it) -> bool:
            if only and it.item_id not in only:
                return False
            if last is not None and _stage_pos(it.state["stage"]) > last:
                return False
            if it.state["status"] == "active":
                return True
            # S9 waiting with nothing running (prepared while allow_submit was off, or blocked): once per run
            return it.item_id not in rewoken and B.reschedulable(self.cfg, it)

        while progress and steps < max_steps:
            progress = False
            # filter first, then order: skipped items must not take a round-robin slot (review)
            for it in self.order([i for i in self.store.items() if runnable(i)]):
                if not runnable(it):            # its status may have changed since the snapshot
                    continue
                if it.state["status"] == "waiting":
                    rewoken.add(it.item_id)
                    if not B.rewake(self.cfg, it):      # re-checked under the item lock
                        continue
                r = self.step(it)
                steps += 1
                progress |= r in ("ok", "error", "crash")
                if steps >= max_steps:
                    break
        out = self.status()
        if until:
            out["until"] = until
        return out

    def resume(self, reasons_prefix: tuple[str, ...] = ("budget:", "model:"), only: list[str] | None = None,
               stages: list[str] | None = None) -> list[str]:
        """Re-activate paused items whose reason starts with one of reasons_prefix; only: just these item ids;
        stages: just items paused at these stages."""
        out = []
        for it in self.store.items():
            if only and it.item_id not in only:
                continue
            hist = it.state.get("history") or [{}]
            if stages and hist[-1].get("paused_stage", it.state["stage"]) not in stages:
                continue
            if it.state["status"] == "paused" and str(it.state.get("reason", "")).startswith(reasons_prefix):
                it.state["stage"] = hist[-1].get("paused_stage", it.state["stage"])
                it.state["status"] = "active"
                it.save()
                out.append(it.item_id)
        self.stopped_models.clear()
        return out

    def status(self) -> dict:
        from collections import Counter
        items = self.store.items()
        return dict(n=len(items), by_status=dict(Counter(i.state["status"] for i in items)),
                    by_stage=dict(Counter(f"{i.state['stage']}/{i.state['status']}" for i in items)),
                    reasons=dict(Counter(i.state.get("reason") for i in items if i.state.get("reason"))),
                    ledger_today=self.ledger.summary(today()))
