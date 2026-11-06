"""Leak-gate back-test on tasks from an earlier generation batch (no LLM, no model inference).
The task data is not part of this repository; the paths below point into the author's workspace.

Positives (from the human review decisions of that batch):
  * 6 "drop" tasks (public oracle object leaks the optimum)
  * doi-196db5a3b2c31394 pre-fix (design covariance invertible)
Controls:
  * every reviewed-accepted task of the batch (use_v2_seed0 / use_catchup_latest; these passed human
    review, so any flag is counted as a false positive *candidate*)
  * doi-196db5a3b2c31394 post-fix

    uv run python -m tbsgen.backtest --out eval/leak_backtest [--limit N] [--no-introspect]

Data locations (required, no defaults):
  TBS_BACKTEST_REVIEWED  reviewed batch: REVIEW_DECISIONS.jsonl, harbor-v07-new244/, harbor-v07-new244-fix/
  TBS_BACKTEST_CORPUS    packaged pre-review tasks (one Harbor task dir per id) for the positives
  TBS_BACKTEST_HOLDOUT   held-out packaged tasks (--set holdout81 only)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os

from .config import load_config
from .leakcheck import leak_report

COV_ID = "doi-196db5a3b2c31394"


def _dir(var: str) -> Path:
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"{var} is not set (back-test data is not part of the repository; see module docstring)")
    return Path(v)


def cases():
    FINAL, CORPUS = _dir("TBS_BACKTEST_REVIEWED"), _dir("TBS_BACKTEST_CORPUS")
    rows = [json.loads(x) for x in (FINAL / "REVIEW_DECISIONS.jsonl").read_text().splitlines() if x.strip()]
    out = []
    for r in rows:
        if r["action"] == "drop":
            out.append(dict(id=r["task_id"], label="leak", why="drop: public oracle object", dir=CORPUS / r["task_id"]))
    out.append(dict(id=COV_ID + "@prefix", label="leak", why="fix: invertible design covariance", dir=CORPUS / COV_ID))
    out.append(dict(id=COV_ID + "@postfix", label="clean", why="fixed covariance", dir=FINAL / "harbor-v07-new244-fix" / COV_ID))
    for r in rows:
        if r["action"] in ("use_v2_seed0", "use_catchup_latest"):
            d = FINAL / "harbor-v07-new244" / r["task_id"]
            if d.exists():
                out.append(dict(id=r["task_id"], label="clean", why=r["action"], dir=d))
    return out


def holdout_cases():
    """81 accepted tasks from an earlier batch.  NOT looked at while tuning the
    rules (tuning used the 231 controls above), so their flag rate is an out-of-sample false-positive estimate.
    They passed Harbor oracle/nop but only part of them had human review."""
    HOLD = _dir("TBS_BACKTEST_HOLDOUT")
    return [dict(id=d.name, label="clean", why="holdout81", dir=d) for d in sorted(HOLD.iterdir())
            if (d / "instruction.md").exists()]


def one(c, cfg, introspect):
    d = c["dir"]
    t0 = time.time()
    try:
        rep = leak_report(d / "environment" / "data", [d / "tests" / "truth"],
                          (d / "instruction.md").read_text(errors="replace"), cfg,
                          introspect="docker" if introspect else "skip")
    except Exception as e:  # noqa: BLE001
        rep = dict(verdict="error", classes=[], findings=[], info=dict(error=f"{type(e).__name__}: {e}"))
    return dict(id=c["id"], label=c["label"], why=c["why"], verdict=rep["verdict"], classes=rep["classes"],
                warn_classes=rep.get("warn_classes", []),
                sec=round(time.time() - t0, 2), findings=rep["findings"][:12], info=rep["info"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/leak_backtest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-introspect", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", default="main", choices=["main", "holdout81"])
    a = ap.parse_args(argv)
    cfg = load_config()
    cs = cases() if a.set == "main" else holdout_cases()
    if a.limit:
        cs = cs[:7] + cs[7:7 + a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda c: one(c, cfg, not a.no_introspect), cs))
    with (out / "results.jsonl").open("w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    pos = [r for r in res if r["label"] == "leak"]
    neg = [r for r in res if r["label"] == "clean"]
    tp = sum(r["verdict"] == "leak" for r in pos)
    fp = [r for r in neg if r["verdict"] == "leak"]
    rv_pos = sum(r["verdict"] in ("leak", "review") for r in pos)
    rv_neg = [r for r in neg if r["verdict"] == "review"]
    summ = dict(n_pos=len(pos), n_neg=len(neg), recall=f"{tp}/{len(pos)}",
                false_positive=f"{len(fp)}/{len(neg)}",
                recall_incl_review=f"{rv_pos}/{len(pos)}", review_only_neg=f"{len(rv_neg)}/{len(neg)}",
                review_by_class=dict(Counter(c for r in rv_neg for c in r["warn_classes"])),
                fp_by_class=dict(Counter(c for r in fp for c in r["classes"])),
                pos_classes={r["id"]: r["classes"] for r in pos},
                errors=[r["id"] for r in res if r["verdict"] == "error"],
                fp_ids=[r["id"] for r in fp])
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
