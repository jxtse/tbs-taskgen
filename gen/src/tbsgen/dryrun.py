"""End-to-end dry run with the fake model (no real model calls, no Harbor jobs).

    uv run python -m tbsgen.dryrun --out runs/dryrun-<date> [--candidates FILE]

Candidates: by default one real candidate record taken from the retrieval layer's recorded eval output
(retrieval/eval/raw/*.json, read-only) with full text from the old literature cache, plus a synthetic
"toy" candidate whose full text is the toy fixture description.  The fake model returns the toy fixture
files for every candidate, so the generator / reference / baselines are real executable code run in Docker.
Then: S1 -> S9 (calibration configs written, not submitted) -> simulated Harbor results -> S10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .budget import Ledger
from .config import PIPELINE_V2, load_config
from .fakes import toy_script
from .models import FakeClient
from .pipeline import Pipeline
from .report import daily_report

FIX = PIPELINE_V2 / "gen" / "tests" / "fixtures" / "toy_task"
TOY_TEXT = ("Robust estimation of exponential decay constants from detector afterglow traces. We fit "
            "amplitude and rate by weighted log-linear least squares with heteroscedastic weights, and "
            "reject one-sided spikes from cosmic rays with an iterative robust estimator. Numerical "
            "simulation and Monte Carlo calibration of the algorithm on a benchmark dataset. ") * 60


def make_toy_candidate(root: Path) -> dict:
    cache = root / "fulltext_cache"
    cache.mkdir(parents=True, exist_ok=True)
    doi = "10.0000/tbsgen.toy.decay"
    (cache / "10.0000_tbsgen.toy.decay.txt").write_text("Robust exponential decay fitting\n" + TOY_TEXT)
    return dict(doi=doi, title="Robust exponential decay fitting",
                abstract="We estimate decay rates with a weighted least squares algorithm and robust spike "
                         "rejection; simulation benchmark, open-source python implementation on github.",
                code_links=["https://github.com/example/decay"])


def build(out: Path, fake_kw: dict | None = None, budget: dict | None = None):
    cfg = load_config(None, {"run": {"root": str(out)},
                             "fulltext": {"cache_dirs": [str(out / "fulltext_cache")] +
                                          load_config()["fulltext"]["cache_dirs"], "allow_network": False},
                             "budget": budget or {}})
    led = Ledger(out / "ledger.sqlite", cfg)
    fake = FakeClient(led, cfg, toy_script(FIX, **(fake_kw or {})), name="fake-opus")
    P = Pipeline(cfg, clients={"teacher": fake, "spec": fake, "gen": fake, "tests": fake, "repair": fake})
    return P


def simulate_harbor(P, item_id: str, outcomes: list[str]):
    """Feed fake Harbor trial results into S9 (no jobs launched)."""
    from .state import read_json, write_json
    it = P.store.item(item_id)
    for oc in outcomes:
        runs = read_json(it.p("calib", "runs.json"), [])
        pend = [r for r in runs if r.get("outcome") is None]
        if not pend:
            break
        pend[-1].update(outcome=oc, minutes=42.0, trial=dict(simulated=True))
        write_json(it.p("calib", "runs.json"), runs)
        it.state["stage"], it.state["status"] = "S9", "active"
        it.save()
        P.run()
        it = P.store.item(item_id)
    return it.state


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--retrieval-json", default=str(PIPELINE_V2 / "retrieval" / "eval" / "raw" /
                                                   "method__multivariate-fay-herriot-hierarchical-small-area-models.json"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    P = build(out)
    cands = [make_toy_candidate(out)]
    rj = Path(a.retrieval_json)
    if rj.exists():
        d = json.loads(rj.read_text())
        sel = d.get("selected") or []
        cands += [dict(p, _target=d["target"]["id"]) for p in sel[:3]]
    ids = P.ingest(cands)
    st = P.run()
    print(json.dumps(st, indent=1, ensure_ascii=False))
    toy = ids[0]
    final = simulate_harbor(P, toy, ["fail", "pass"])      # GLM#1 fail -> teacher pass -> hard -> S10
    print("toy:", final["stage"], final["status"], final.get("calib_label"))
    print(daily_report(P))


if __name__ == "__main__":
    main()
