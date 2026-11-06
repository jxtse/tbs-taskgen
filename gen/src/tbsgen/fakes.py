"""Scripted fake model for dry runs and tests: plays S5 / S6A / S6C / S8 for the toy fixture
(tests/fixtures/toy_task).  S6C reads the thresholds out of its prompt, the way a real model would, so the
test file always matches S6B's measured thresholds.  Knobs let tests inject failures:

    toy_script(fixture_dir, break_reference=False, leak_in_public=False, s6c_drop_exploit=False)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SPEC = {
    "title": "Spike-robust exponential decay fitting for detector afterglow traces",
    "domain": "physics", "field": "signal processing / detector calibration", "form": "ANS",
    "expert_time_hours": 0.5,
    "competencies": ["log-linear least squares with heteroscedastic weights", "one-sided outlier rejection"],
    "grounding_vs_extension": "toy fixture (no paper)",
    "depgraph": {
        "deliverables": ["D1"],
        "nodes": [
            {"id": "D1", "kind": "deliverable", "desc": "per-series amplitude and rate", "provision": "given_instruction"},
            {"id": "C1", "kind": "compute", "desc": "y^2-weighted log-linear fit", "provision": "derive", "key": True,
             "secret_names": []},
            {"id": "C2", "kind": "compute", "desc": "one-sided spike rejection", "provision": "derive", "key": True},
            {"id": "P1", "kind": "premise", "desc": "series", "provision": "given_file", "carrier": "data/series.csv"},
            {"id": "H1", "kind": "premise", "desc": "true amp/rate", "provision": "hidden_truth",
             "secret_names": ["params.csv"]}],
        "edges": [["P1", "C1"], ["P1", "C2"], ["C2", "C1"], ["C1", "D1"]],
        "mid_drop": ["C1"], "naive_drop": ["C1", "C2"],
        "difficulty_knobs": {"derive_nodes": 2, "chain_len": 3, "files_crossed": 1, "nonstd_tool": False}},
    "metrics": [{"name": "amp_median_rel_error", "direction": "<=", "on": "public"},
                {"name": "rate_p90_rel_error", "direction": "<=", "on": "public"},
                {"name": "coverage", "direction": ">=", "on": "public"}],
    "public_files": ["series.csv"], "hidden_files": [], "truth_files": ["params.csv"],
    "why_naive_fails": "unweighted log fit + spikes", "mid_missing_step": "C1 weighting"}


def _files(fx: Path, rels) -> str:
    return "".join(f"=== FILE: {r} ===\n{(fx / r).read_text()}" for r in rels) + "=== END ===\n"


def test_file(thr: dict, metrics_src: str) -> str:
    calls = "\n".join(f"    check({k!r}, m[{k!r}], {v['threshold']!r}, {v['direction']!r})" for k, v in thr.items())
    return (metrics_src + "\nfrom _metrics import check\n\n\ndef test_metrics():\n    m = metrics()\n" +
            "    failures = []\n" + "\n".join(
                f"    try:\n        check({k!r}, m[{k!r}], {v['threshold']!r}, {v['direction']!r})\n"
                f"    except AssertionError as e:\n        failures.append(str(e))" for k, v in thr.items()) +
            "\n    assert not failures, failures\n") if calls else ""


def instruction(thr: dict, leak_text: str = "") -> str:
    crit = "\n".join(f"- `{k}` {v['direction']} {v['threshold']}" for k, v in thr.items())
    return f"""# Afterglow decay constants from spiky detector traces

`/app/data/series.csv` holds 48 detector afterglow traces (`series_id`, `t` in µs, `y` in ADC counts).
Each trace is a single exponential decay `y = amp * exp(-rate * t)` with ~1% multiplicative read noise,
plus occasional positive spikes (cosmic-ray hits) that add 30-80% of the local signal.
{leak_text}
## Deliverable
`/app/output/estimates.csv` with columns `series_id,amp,rate`, one row per trace.

## Pass criteria (relative errors against the true parameters; all must hold)
{crit}

`amp_median_rel_error` is the median over traces of |amp_hat-amp|/amp; `rate_p90_rel_error` is the 90th
percentile of |rate_hat-rate|/rate; `coverage` is the share of traces with finite estimates.
Time budget: 2 hours.
"""


EXPLOIT = """import csv, os
rows = list(csv.DictReader(open('/app/data/series.csv')))
first = {}
for r in rows:
    first.setdefault(r['series_id'], float(r['y']))
os.makedirs('/app/output', exist_ok=True)
with open('/app/output/estimates.csv', 'w') as f:
    f.write('series_id,amp,rate\\n')
    for k, y in first.items():
        f.write(f'{k},{y},0.5\\n')
"""


def toy_script(fx: Path, break_reference: bool = False, leak_in_public: bool = False,
               s6c_drop_exploit: bool = False, repair_fixes: bool = True):
    fx = Path(fx)
    metrics_src = (fx / "metrics_src.py").read_text()
    state = {"s8": 0}

    def script(stage, system, user):
        if stage == "S5":
            return "Here is the spec:\n```json\n" + json.dumps(SPEC) + "\n```"
        if stage == "S6A":
            out = _files(fx, ["generator/generate.py", "solution/solve.py", "baselines/mid.py", "baselines/naive.py",
                              "tests/measure.py"])
            if break_reference:
                out = out.replace("write({sid: robust(t, y) for sid, (t, y) in load().items()})",
                                  "raise SystemExit('reference crashed')")
            if leak_in_public:
                out = out.replace('w = csv.writer(f); w.writerow(["series_id", "t", "y"]); w.writerows(rows)',
                                  'w = csv.writer(f); w.writerow(["series_id", "t", "y"]); w.writerows(rows)\n'
                                  'with open("environment/data/meta.csv", "w", newline="") as f:\n'
                                  '    w = csv.writer(f); w.writerow(["series_id", "gain", "tau"]); w.writerows(truth)')
            return out
        if stage == "S6C":
            m = re.search(r"Thresholds fixed from those measurements \(do not change them\): (\{.*?\})\n", user)
            thr = json.loads(m.group(1))
            files = {"tests/test_outputs.py": test_file(thr, metrics_src), "instruction.md": instruction(thr)}
            if not s6c_drop_exploit:
                files["baselines/exploit_first_point.py"] = EXPLOIT
            return "".join(f"=== FILE: {k} ===\n{v}" for k, v in files.items()) + "=== END ===\n"
        if stage == "S8":
            state["s8"] += 1
            if repair_fixes and "reference crashed" in user:
                return "=== FILE: solution/solve.py ===\n" + (fx / "solution/solve.py").read_text() + "=== END ===\n"
            if repair_fixes and "meta.csv" in user:
                g = (fx / "generator/generate.py").read_text()
                return "=== FILE: generator/generate.py ===\n" + g + "=== END ===\n"
            return "=== FILE: instruction.md ===\nno real fix\n=== END ===\n"
        raise ValueError(stage)

    return script
