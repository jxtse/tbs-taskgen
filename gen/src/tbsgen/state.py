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

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ["S1", "S2", "S3", "S4", "S5", "S6A", "S6B", "S6C", "S7", "S8", "PKG", "S9", "S10"]
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

    def load(self) -> "Item":
        self.state = read_json(self.p("state.json"), None) or dict(
            item_id=self.item_id, stage="S1", status="active", history=[], attempts={}, created=time.time())
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


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        (self.root / "items").mkdir(parents=True, exist_ok=True)

    def item(self, item_id: str) -> Item:
        return Item(self.root, item_id).load()

    def items(self) -> list[Item]:
        return [Item(self.root, d.name).load() for d in sorted((self.root / "items").iterdir()) if d.is_dir()]

    def event(self, kind: str, **data):
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(dict(ts=time.time(), kind=kind, **data), ensure_ascii=False, default=str) + "\n")

    def events(self) -> list[dict]:
        p = self.root / "events.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
