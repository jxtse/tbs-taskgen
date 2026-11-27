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

from . import blocklist, intake, prereview_flow, screen_jev, stages_a as A, stages_b as B, taskimage
from .budget import EXCLUDE_UNSTARTED_JOBS, BudgetStop, Ledger, harden_scope, today, variant_scope
from .calib import migrate_manual_commands, pending_launcher
from .config import resolve_role
from .models import ModelError, get_client
from .state import STAGES, Store, read_json, write_json

HANDLERS = {"S1": A.s1, "S2a": A.s2a, "S3b": A.s3b, "S3c": A.s3c, "S4c": A.s4c, "S5G": A.s5g, "S2": A.s2, "S3": A.s3, "S4": A.s4, "S5": A.s5, "S6I": A.s6i, "S6A": A.s6a, "S6A_run": A.s6a_run, "S6M": A.s6m,
            "S6B": A.s6b, "S6C": B.s6c, "S7": B.s7, "S8": B.s8, "PKG": B.pkg, "S9": B.s9, "S9R": B.s9r, "S10": B.s10}
DEFAULT_RESUME_REASONS = ("budget:", "model:", "parked:probe_unreviewed", *B.FORMAL_REVIEW_REASONS)
AGENT_STAGES = ("S4c", "S5G")     # code-grounded Harbor agent jobs: the item waits while its job runs
MODEL_STAGES = {"S5": "spec", "S6A": "gen", "S6M": "gen", "S6C": "tests", "S8": "repair"}
# sub-stages that are not in state.STAGES (for `run --until`): S6A_run is part of S6A; S6M (a model call) comes
# after S6B, so `--until S6B` does not start it
_SUBSTAGE = {"S6A_run": ("S6A", 0.0), "S6M": ("S6B", 0.5)}
_STOP = {"a", "an", "the", "of", "for", "and", "in", "on", "with", "by", "to", "from", "using", "via", "based",
         "new", "novel", "fast", "efficient", "method", "methods", "approach", "algorithm", "data", "analysis"}


STAGE_GROUPS = {
    "intake": {"S1", "S2a", "S2", "S3", "S4"},
    "code": {"S3b", "S3c"},
    "agent": set(AGENT_STAGES),
    "build": {"S6I"},
    "sandbox": {"S6A_run", "S6B", "S6C", "S7", "S8", "PKG"},
    "calib": {"S9", "S9R", "S10"},
    "harden": {"harden"},
}


def parse_stages(value: str) -> set[str]:
    selected = set()
    for name in map(str.strip, value.split(",")):
        if name in STAGE_GROUPS:
            selected.update(STAGE_GROUPS[name])
        elif name in HANDLERS:
            selected.add(name)
        else:
            raise ValueError(f"unknown worker stage or group: {name!r}")
    return selected


def _stage_pos(stage: str) -> float:
    s, off = _SUBSTAGE.get(stage, (stage, 0.0))
    return STAGES.index(s) + off if s in STAGES else float(len(STAGES))


def _title_words(title: str) -> frozenset:
    import re
    return frozenset(w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 2 and w not in _STOP)


class Pipeline:
    def __init__(self, cfg: dict, clients: dict | None = None, fetcher=None, *, init_deadline=None):
        root = cfg["run"]["root"]
        if not root:
            raise ValueError("cfg.run.root is required")
        self.cfg = cfg
        self.store = Store(root)
        self.ledger = Ledger(Path(root) / "ledger.sqlite", cfg, init_deadline=init_deadline)
        migrate_manual_commands(self.store, cfg.get("config_path"), deadline=init_deadline)
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

    def fulltext_fetcher(self, *, deadline=None):
        """The S3 Fetcher (one per pipeline: one ledger connection, one egress check).  A test-injected
        fulltext.Fetcher takes precedence."""
        from .fulltext import Fetcher
        if self.fetcher is None:
            self.fetcher = Fetcher(self.cfg, init_deadline=deadline)
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
            r = db.execute("SELECT count(*), SUM(COALESCE(usd,0)), SUM(in_tok), SUM(out_tok) FROM calls "
                           f"WHERE item=? AND {EXCLUDE_UNSTARTED_JOBS}",
                           (item_id,)).fetchone()
        return dict(calls=r[0], usd=round(r[1] or 0, 4), in_tok=r[2] or 0, out_tok=r[3] or 0)

    # ------------------------------------------------------------------ intake
    def ingest(self, candidates: list[dict]) -> list[str]:
        ids = []
        for c in candidates:
            iid = intake.item_id_for(c)
            it = self.store.item(iid)
            if not it.p("S1_candidate.json").exists() or not it.p("state.json").exists():
                with it.operator_lock():
                    if not it.p("S1_candidate.json").exists():
                        write_json(it.p("S1_candidate.json"), c)
                    if not it.p("state.json").exists():
                        it.load().save()
                        self.store.event("ingest", item=iid, doi=c.get("doi"))
            ids.append(iid)
        return ids

    # ------------------------------------------------------------------ run loop
    def step(self, it) -> str:
        expected = it.state
        with it.step_lock() as held:
            if not held or it.load().state != expected:
                return "skipped"
            if B.legacy_report_id(it):
                return self._step_with_budget(it)
            if it.state["status"] not in ("active", "waiting") and not B.harden_needs_step(it):
                return "skipped"
            if (it.state["status"] == "waiting" and B.job_stage(it) in ("S9", "S9R")
                    and not B.harden_needs_step(it)):
                action = B.poll_calibration(self, it)
                if action:
                    return "skipped" if action["action"] == "ingest_error" else "ok"
                if not B.rewake(self.cfg, it):
                    return "skipped"
            return self._step_with_budget(it)

    def _step_with_budget(self, it) -> str:
        # a hardening round's reservations are tagged with the round (budget.item_harden_usd_cap); evaluated at each
        # reservation from it.state, which handlers re-load under the item lock (another run / watch may have moved it)
        def hk():
            s = it.state
            if not s.get("harden_active") or s.get("harden_kind") == "variant":
                return 0
            return int(s.get("harden_round", 0) or 0)

        def vk():
            s = it.state
            if not s.get("harden_active") or s.get("harden_kind") != "variant":
                return 0
            return int(s.get("harden_round", 0) or 0)
        with harden_scope(hk), variant_scope(vk):
            return self._step(it)

    def _finish_failed_stage(self, it, stage, status, reason, **info):
        if stage in ("S4c", "S5G", "S9", "S9R"):
            previous_status = it.state["status"]
            with B.item_lock(it):
                it.load()
                if it.state["stage"] == stage and it.state["status"] == previous_status:
                    B.finish_job_stage(self, it, status, reason, **info)
        else:
            it.finish(status, reason, **info)

    def _migrate_probe_report(self, it):
        try:
            return "ok" if B.migrate_probe_report(self, it) else None
        except Exception as exc:
            with B.item_lock(it):
                it.load()
                stage = it.state["stage"]
                self.store.event("stage_crash", item=it.item_id, stage=stage, err=f"{type(exc).__name__}: {exc}",
                                 tb=traceback.format_exc()[-1500:])
                if B.legacy_report_id(it):
                    it.finish("paused", f"probe migration: {exc}", paused_stage=stage)
            return "paused"

    def _restore_task_image(self, it, stage):
        if stage in ("S5G", "S6I") or not taskimage.is_task(it.state):
            return False
        if stage in ("S9", "S9R") and any(pending_launcher(run, self.ledger)
                                          for run in B._job_runs(self.cfg, it, stage)):
            # The submitted job owns its reservation and stage until reconciliation.
            return False
        if taskimage.needs_restore(self.cfg, it.dir, it.state):
            it.state["image_resume_stage"] = stage
            it.goto("S6I", "restore or reverify task image before " + stage)
            return True
        if stage not in ("S9", "S9R", "S10"):
            return False
        prov = read_json(it.p("pkg/provenance.json"), {})
        if (not prov.get("image_manifest") and prov.get("image_digest") == it.state["image_digest"]
                and prov.get("image_tag") == it.state["image_tag"]):
            # Pre-OCI calibration builds did not attest the dependency layers used by the teacher/student.
            if any(r.get("outcome") is None for role_dir in ("calib", "review")
                   for r in read_json(it.p(role_dir, "runs.json"), [])):
                raise taskimage.ImageUnavailable("pre-OCI package has outstanding jobs; reconcile before regate")
            # Pipeline.step already holds the outer step lock.
            with B.item_lock(it):
                B._regate(self, it, "migrate package to verified S6I base", recalibrate=True)
            return True
        if not taskimage.package_matches(it.dir):
            raise taskimage.ImageUnavailable("package Dockerfile differs from verified S6I recipe")
        return False

    def _step(self, it) -> str:
        migrated = self._migrate_probe_report(it)
        if migrated:
            return migrated
        st = it.state["stage"]
        role = MODEL_STAGES.get(st)
        alias = resolve_role(self.cfg, role) if role else None
        advisory = bool(it.state.get("prereview_pending")) and st in {"S7", "S8", "PKG"}
        stopped = False
        if role:
            name = getattr(self._clients.get(role) or self._clients.get(alias), "name", alias)
            stopped = name in self.stopped_models or alias in self.stopped_models
            if stopped and not advisory:
                return "skipped"
        self.store.event("stage_start", item=it.item_id, stage=st)
        t0 = time.time()
        try:
            if advisory and prereview_flow.resume_edit(it, self.ledger):
                return "ok"
            if stopped:
                prereview_flow.reject(it, "S8 repair model unavailable")
            elif B.harden_needs_step(it):
                B.harden_step(self, it)         # finish an interrupted hardening start / revert first
            elif not self._restore_task_image(it, st):
                if advisory:
                    try:
                        HANDLERS[st](self, it)
                    except Exception as exc:
                        prereview_flow.reject(it, f"{st}: {type(exc).__name__}")
                else:
                    HANDLERS[st](self, it)
                if B.harden_needs_step(it):
                    B.harden_step(self, it)     # a hardened version that failed to build: back to the old one
        except taskimage.ImageUnavailable as e:
            it.finish("paused", f"task image unavailable: {e}", paused_stage=st)
            return "paused"
        except BudgetStop as e:
            if not e.cancelled_unstarted and e.reason in ("item_usd", "item_harden_usd", "item_variant_usd"):
                # this paper's authoring allowance (budget.item_usd_cap, item_harden_usd_cap in a hardening round,
                # item_variant_usd_cap in an S9b variant round) is used up: final for the paper / the round, and
                # the model stays available for the others
                self._finish_failed_stage(it, st, "dropped", e.reason + "_cap", detail=e.detail)
                if B.harden_needs_step(it):
                    B.harden_step(self, it)     # a hardened version: back to the previous one
                self.store.event("budget_stop", item=it.item_id, stage=st, reason=e.reason, detail=e.detail)
                return "ok"
            it.finish("paused", f"budget:{e.reason}", detail=e.detail, paused_stage=st, **e.reservation)
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
                self._finish_failed_stage(it, st, "parked", f"model_error:{st}:{e.cls}")
            return "error"
        except Exception as e:  # noqa: BLE001  (bug or unexpected data: park the item, keep the run going)
            self.store.event("stage_crash", item=it.item_id, stage=st, err=f"{type(e).__name__}: {e}",
                             tb=traceback.format_exc()[-1500:])
            n = it.bump(f"crash:{st}")
            if n >= 2:
                self._finish_failed_stage(it, st, "parked", f"crash:{st}:{type(e).__name__}")
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

    def run(self, max_steps: int = 10000, only: list[str] | None = None, until: str | None = None,
            stages: set[str] | None = None, stopping=None, poll_calib=False) -> dict:
        """until: last stage to run (e.g. "S4"); items that reach a later stage are left active there, untouched."""
        if until is not None and until not in STAGES:
            raise ValueError(f"unknown stage {until!r}")
        poll_calib = poll_calib or stages is not None
        last = _stage_pos(until) if until else None
        steps = 0
        progress = True
        made_progress = False
        screen_worker = (stages is None or "S2" in stages) and (last is None or last >= _stage_pos("S2"))
        if screen_worker:    # the S2 Jev prefetch is S2 work
            if self.cfg["screen"].get("blocklist"):
                # trial papers the fetch queue delivered since S3 parked them count as successes now; then this
                # week's trial quota picks papers an earlier week's S2 dropped by a rule
                blocklist.reevaluate(self.store, self.cfg, self.store.event)
                if not only:
                    made_progress |= bool(blocklist.revive_for_trials(self.store, self.cfg, self.store.event))
            made_progress |= bool(self.prefetch_screen(only))

        skipped: set[str] = set()
        rewoken: set[str] = set()
        polled: set[str] = set()

        # A remote_unknown agent item is parked, so the loop would never look at it again.  Reconcile the ones
        # whose launcher has demonstrably ended before filtering (failclass / calib.remote_end_evidence).
        for it in self.store.items():
            if only and it.item_id not in only:
                continue
            if not (it.state.get("status") == "parked" and it.state.get("stage") in AGENT_STAGES
                    and str(it.state.get("reason") or "").startswith("remote_unknown: ")):
                continue
            try:
                made_progress |= bool(A.reconcile_parked_agent(self, it))
            except Exception as exc:  # noqa: BLE001 - a bad job dir must not stop the run
                self.store.event("remote_unknown_reconcile_error", item=it.item_id,
                                 err=f"{type(exc).__name__}: {exc}")

        def runnable(it) -> bool:
            if it.item_id in skipped:
                return False
            if stages is not None and it.state["stage"] not in stages and not (
                    "harden" in stages and B._in_hardening(it)):
                return False
            if only and it.item_id not in only:
                return False
            if last is not None and _stage_pos(it.state["stage"]) > last:
                return False
            if it.state["status"] == "active" or B.legacy_report_id(it):
                return True
            if B.harden_needs_step(it):         # an interrupted hardening transaction (any status)
                return it.item_id not in polled
            if A.image_retry_due(it):
                return True
            # code-grounded agent job (S4c / S5G) running on the build host: poll it once per run (not while it
            # waits out a relay refusal: calib.preflight_retry_s)
            if it.state["status"] == "waiting" and it.state["stage"] in AGENT_STAGES:
                return it.item_id not in polled and not A.relay_waiting(self.cfg, it)
            # S9 waiting with nothing running (prepared while allow_submit was off, or blocked): once per run
            return it.item_id not in rewoken and (B.reschedulable(self.cfg, it) or
                    (poll_calib and B.job_stage(it) in ("S9", "S9R") and it.state["status"] == "waiting"))

        while progress and steps < max_steps:
            progress = False
            # filter first, then order: skipped items must not take a round-robin slot (review)
            ordered = self.order([i for i in self.store.items() if runnable(i)])
            if stopping and not steps:
                # A bounded loop must eventually poll past the first N waiting jobs.
                cursor = getattr(self, "_loop_cursor", None)
                offset = next((i + 1 for i, it in enumerate(ordered) if it.item_id == cursor), 0)
                ordered = ordered[offset:] + ordered[:offset]
            for it in ordered:
                if stopping and stopping():
                    break
                if not runnable(it):            # its status may have changed since the snapshot
                    continue
                if B.harden_needs_step(it):
                    polled.add(it.item_id)
                elif it.state["status"] == "waiting" and it.state["stage"] in AGENT_STAGES:
                    polled.add(it.item_id)
                elif it.state["status"] == "waiting" and not B.legacy_report_id(it):
                    rewoken.add(it.item_id)
                before = (it.state["stage"], it.state["status"], it.state.get("reason"))
                r = self.step(it)
                if r == "skipped":
                    skipped.add(it.item_id)
                    continue
                after = (it.state["stage"], it.state["status"], it.state.get("reason"))
                made_progress |= before != after or it.state["status"] == "active"
                steps += 1
                progress |= r in ("ok", "error", "crash")
                if steps >= max_steps:
                    if stopping:
                        self._loop_cursor = it.item_id
                    break
        if screen_worker and steps and self.cfg["screen"].get("blocklist"):
            blocklist.reevaluate(self.store, self.cfg, self.store.event)
        out = self.status()
        out["progress"] = made_progress
        if until:
            out["until"] = until
        return out

    def requeue_image(self, item_id: str) -> bool:
        it = next((i for i in self.store.items() if i.item_id == item_id), None)
        if it is None:
            return False
        with it.operator_lock():
            result = read_json(it.p("S6I_image.json"), {})
            if (it.state["stage"] != "S6I" or not isinstance(result, dict) or
                    result.get("ok") is not False or not taskimage.infra_failure(result)):
                return False
            why = f"image_build_failed:{result['failed_gate']}"
            if it.state["status"] == "dropped" and it.state.get("reason") == "s5g_jobs_exhausted:" + why:
                it.state.pop("image_retry_used", None)
                feedback = read_json(it.p("S6_failure.json"), {})
                if isinstance(feedback, dict) and feedback.get("why") == why and feedback.get("finding") == result["error"]:
                    it.p("S6_failure.json").unlink()
            elif (it.state["status"], it.state.get("reason")) != ("parked", "parked:infra"):
                return False
            it.goto("S6I", "operator requeued image infrastructure failure")
            return True

    def resume(self, reasons_prefix: tuple[str, ...] = DEFAULT_RESUME_REASONS, only: list[str] | None = None,
               stages: list[str] | None = None) -> list[str]:
        """Re-activate paused items whose reason starts with one of reasons_prefix; only: just these item ids;
        stages: just items paused at these stages."""
        out = []
        for it in self.store.items():
            if only and it.item_id not in only:
                continue
            if it.state["status"] not in ("paused", "parked") and not B.legacy_report_id(it):
                continue
            with it.step_lock(timeout=60):
                it.load()
                hist = it.state.get("history") or [{}]
                if stages and hist[-1].get("paused_stage", it.state["stage"]) not in stages:
                    continue
                migrated = self._migrate_probe_report(it)
                if migrated:
                    if migrated == "ok":
                        out.append(it.item_id)
                    continue
                if (str(it.state.get("reason", "")).startswith(reasons_prefix)
                        and B.resume_formal_review(self.cfg, it)):
                    out.append(it.item_id)
                    continue
                probe_park = (it.state["status"] == "parked" and it.state.get("reason") == "parked:probe_unreviewed"
                              and (self.cfg.get("review") or {}).get("enabled"))
                if (it.state["status"] == "paused" or probe_park) and str(it.state.get("reason", "")).startswith(reasons_prefix):
                    if not self.ledger.window_allows(hist[-1]):
                        continue
                    it.state["stage"] = hist[-1].get("paused_stage", it.state["stage"])
                    it.state.setdefault("history", []).append(dict(stage=it.state["stage"], result="resume", ts=time.time()))
                    it.state.pop("reason", None)
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
                    remote_unknown=sum(r.get("state") == "remote_unknown" and r.get("outcome") is None
                                       for i in items for directory in ("calib", "review")
                                       for r in read_json(i.p(directory, "runs.json"), []))
                                   + sum(read_json(i.p(stage, "run.json"), {}).get("state") == "remote_unknown"
                                         for i in items for stage in AGENT_STAGES),
                    ledger_today=self.ledger.summary(today()), window_caps=self.ledger.window_usage())
