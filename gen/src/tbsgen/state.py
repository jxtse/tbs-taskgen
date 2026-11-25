"""Per-item state directories and the stage state machine (resumable).

Layout of a run (cfg.run.root):
    ledger.sqlite              budget ledger (models)
    events.jsonl               append-only event log (stage start/finish/fail, budget stops)
    reports/daily-<day>.md     daily report
    items/<item_id>/
        state.json             {"item_id", "stage", "status", "history": [...], "attempts": {stage: n}, ...}
        S1_candidate.json  S2_screen.json  S3_fulltext.json  fulltext.txt  S4_probe.json
        _spec/{depgraph,spec,coverage,provenance}.json
        task/                  working task tree (generator/, solution/, baselines/, tests/, instruction.md)
        runs/<tag>/            sandbox run artefacts (gen<seed>/, out-*/, logs-*)
        S6A_measure.json  thresholds.json  gates/attempt-<n>.json  repair/round-<n>/
        pkg/                   Harbor task (only this goes to Harbor)
        calib/                 submit manifests + collected results
Item status values: "active" (has a next stage), "waiting" (S9 submitted, awaiting results),
"paused" (budget/breaker stop; resumes as-is), "done", "dropped" (terminal, reason code), "parked" (needs human).
A stage writes its outputs first and then commits state.json atomically, so a crash in the middle of a
stage simply re-runs that stage.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ["S1", "S2a", "S2", "S3", "S3b", "S3c", "S4", "S4c", "S5", "S5G", "S6A", "S6B", "S6C", "S7", "S8", "PKG", "S9", "S9R", "S10"]
TERMINAL = {"done", "dropped", "parked"}


def write_json(path: Path, obj, atomic: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path) if atomic else tmp.rename(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def gate_attempts(directory: Path) -> list[Path]:
    """Numeric order of S7 report files."""
    return sorted((p for p in directory.glob("attempt-*.json") if p.stem[8:].isdigit()),
                  key=lambda p: int(p.stem[8:]))


def latest_gate_report(directory: Path) -> dict:
    attempts = gate_attempts(directory)
    return read_json(attempts[-1], {}) if attempts else {}


def passed_gate_attempt(state: dict) -> int | None:
    if "gates_attempt" in state:
        return state["gates_attempt"]
    # Before the explicit binding, Item.advance recorded the passing attempt in history.
    revert = None
    for h in reversed(state.get("history", [])):
        if revert is not None:
            if (h.get("to") == "S5G" and h.get("result") in ("harden", "variant")
                    and revert.get("round") in (None, h.get("round"))):
                revert = None
        elif h.get("result") == "harden_revert":
            # Older revert entries omitted the round. Its start is the preceding harden/variant transition.
            revert = h
        elif h.get("stage") == "S7" and h.get("result") == "ok":
            return h.get("attempt")
    return None


def passed_gate_report(directory: Path, state: dict) -> dict:
    attempt = passed_gate_attempt(state)
    if attempt is None:
        return {}
    path = directory / f"attempt-{attempt}.json"
    return json.loads(path.read_text())


class StepLockTimeout(TimeoutError):
    pass


@dataclass
class Item:
    root: Path
    item_id: str
    state: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return self.root / "items" / self.item_id

    def p(self, *parts) -> Path:
        return self.dir.joinpath(*parts)

    @contextlib.contextmanager
    def step_lock(self, timeout: float = 0):
        """Outer worker lock. Never acquire this while holding calib/.lock."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.p(".step.lock").open("a") as lock:
            deadline = time.monotonic() + timeout
            if not _try_lock(lock, deadline):
                if timeout:
                    raise StepLockTimeout(f"timed out waiting for {self.item_id} step lock")
                yield False
                return
            yield True

    @contextlib.contextmanager
    def operator_lock(self, timeout: float = 60, *, deadline=None):
        if deadline is not None:
            timeout = min(timeout, max(0, deadline - time.monotonic()))
        with self.step_lock(timeout) as held:
            if not held:
                raise TimeoutError(f"busy item: {self.item_id}")
            with item_lock(self, deadline=deadline):
                self.load()
                yield

    def load(self) -> "Item":
        self.state = read_json(self.p("state.json"))
        if not self.state:
            candidate = self.p("S1_candidate.json")
            self.state = dict(item_id=self.item_id, stage="S1", status="active", history=[], attempts={},
                              created=candidate.stat().st_mtime if candidate.exists() else 0.0)
        return self

    def save(self):
        self.state["updated"] = time.time()
        write_json(self.p("state.json"), self.state)

    # transitions --------------------------------------------------------------------------
    def advance(self, done_stage: str, next_stage: str | None, status: str = "active", **info):
        self.state["history"].append(dict(stage=done_stage, result="ok", ts=time.time(), **info))
        if next_stage:
            self.state["stage"] = next_stage
        self.state["status"] = status
        if status in ("active", "done"):
            self.state.pop("reason", None)
        self.save()

    def goto(self, stage: str, reason: str):
        self.state["history"].append(dict(stage=self.state["stage"], result="goto", to=stage, reason=reason,
                                          ts=time.time()))
        self.state["stage"] = stage
        self.state["status"] = "active"
        self.state.pop("reason", None)
        self.save()

    def finish(self, status: str, reason: str, **info):
        assert status in TERMINAL | {"paused", "waiting"}
        self.state["history"].append(dict(stage=self.state["stage"], result=status, reason=reason, ts=time.time(),
                                          **info))
        self.state["status"] = status
        self.state["reason"] = reason
        self.save()

    def bump(self, stage: str) -> int:
        a = self.state.setdefault("attempts", {})
        a[stage] = a.get(stage, 0) + 1
        self.save()
        return a[stage]


def _try_lock(fh, deadline):
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))


@contextlib.contextmanager
def item_lock(it, deadline=None):
    """Per-item S9 lock (flock on calib/.lock): `run`, `calib watch` and `calib-ingest` may run at the same
    time; S9 steps and ingests of one item are serialised so an intent is never launched twice and a result is
    never attached twice.  Not re-entrant: never nest."""
    p = it.p("calib", ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        if deadline is None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        elif not _try_lock(fh, deadline):
            raise StepLockTimeout(f"timed out waiting for {it.item_id} calib lock")
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        (self.root / "items").mkdir(parents=True, exist_ok=True)

    def item(self, item_id: str) -> Item:
        return Item(self.root, item_id).load()

    def items(self) -> list[Item]:
        return [Item(self.root, d.name).load() for d in sorted((self.root / "items").iterdir())
                if (d / "state.json").exists() or (d / "S1_candidate.json").exists()]

    def event(self, kind: str, **data):
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(dict(ts=time.time(), kind=kind, **data), ensure_ascii=False, default=str) + "\n")

    def events(self) -> list[dict]:
        p = self.root / "events.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
