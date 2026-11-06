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

from . import intake, stages_a as A, stages_b as B
from .budget import BudgetStop, Ledger, today
from .config import resolve_role
from .models import ModelError, get_client
from .state import Store, write_json

HANDLERS = {"S1": A.s1, "S2": A.s2, "S3": A.s3, "S4": A.s4, "S5": A.s5, "S6A": A.s6a, "S6A_run": A.s6a_run,
            "S6B": A.s6b, "S6C": B.s6c, "S7": B.s7, "S8": B.s8, "PKG": B.pkg, "S9": B.s9, "S10": B.s10}
MODEL_STAGES = {"S5": "spec", "S6A": "gen", "S6C": "tests", "S8": "repair"}


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

    def tbs_index(self):
        if self._tbs is None:
            root = Path(self.cfg["probe"]["tbs_tasks_root"])
            self._tbs = intake.TBSIndex(root) if root.exists() else None
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
            if e.cls in ("quota", "auth", "model_mismatch"):
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

    def run(self, max_steps: int = 10000, only: list[str] | None = None) -> dict:
        steps = 0
        progress = True
        while progress and steps < max_steps:
            progress = False
            for it in self.store.items():
                if only and it.item_id not in only:
                    continue
                if it.state["status"] != "active":
                    continue
                r = self.step(it)
                steps += 1
                progress |= r in ("ok", "error", "crash")
                if steps >= max_steps:
                    break
        return self.status()

    def resume(self, reasons_prefix: tuple[str, ...] = ("budget:", "model:")) -> list[str]:
        out = []
        for it in self.store.items():
            if it.state["status"] == "paused" and str(it.state.get("reason", "")).startswith(reasons_prefix):
                it.state["stage"] = it.state["history"][-1].get("paused_stage", it.state["stage"])
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
