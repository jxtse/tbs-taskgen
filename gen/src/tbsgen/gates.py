"""S7 mechanical gates.  No model calls (L4b cheap-agent inversion is 待定 and reported as skipped).

gates/attempt-<n>.json = {"gates": {name: bool}, "blocking": [names], "details": {...}}
  determinism   generator: same seed twice -> identical env/tests hashes
  oracle        reference passes test_outputs.py on every seed
  oracle_margin reference clears each threshold (re-measured values, relative margin >= min_ref_margin/2)
  nop           empty /app/output fails
  naive         naive fails >= naive_fail_min checks on every seed
  mid           mid fails 1..2 checks and all checks were evaluated
  exploit       every baselines/exploit_*.py fails
  hidden_shift  PROG: tests/hidden exists, differs from public, >= 2 hidden checks recorded for the reference
  thresholds_stated  every threshold appears numerically in instruction.md (old G7)
  leak          L1-L4a (leakcheck.leak_report, with depgraph)
  isolation     no tests/, truth, generator under environment/; no python oracle module in public data for
                non-INTERACT forms that reads a non-public file
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from . import runner
from .leakcheck import leak_report


def numbers_in(text: str) -> set[float]:
    vals = set()
    t = text.replace("−", "-").replace("\u2009", "").replace("\\,", "")
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:[eE]|[×x·]\s*10\^?)\s*\{?([-+]?\d+)\}?", t):
        vals.add(float(m.group(1)) * 10 ** int(m.group(2)))
    for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+|\.\d+)(\s*%)?", t):
        v = float(m.group(1).replace(",", ""))
        vals.add(v)
        if m.group(2):
            vals.add(v / 100)
    return vals


def run_gates(cfg: dict, item_dir: Path, spec: dict, thresholds: dict, attempt: int) -> dict:
    task = item_dir / "task"
    work = item_dir / "runs" / f"s7-{attempt}"
    seeds = cfg["gen"]["seeds"]
    g, det = {}, {}
    # determinism
    a = runner.generate(cfg, task, seeds[0], work / "detA")
    b = runner.generate(cfg, task, seeds[0], work / "detB")
    g["determinism"] = a["rc"] == 0 and b["rc"] == 0 and a["hash_env"] == b["hash_env"] and \
        a["hash_tests"] == b["hash_tests"]
    det["determinism"] = dict(a={k: a[k] for k in ("rc", "hash_env", "hash_tests", "err")},
                              b={k: b[k] for k in ("rc", "hash_env", "hash_tests")})
    ok = dict(oracle=True, nop=True, naive=True, mid=True, exploit=True, hidden_shift=True, oracle_margin=True)
    exploits = sorted(str(p.relative_to(task)) for p in (task / "baselines").glob("exploit_*.py"))
    det["seeds"] = {}
    for s in seeds:
        gen = work / "detA" if s == seeds[0] else work / f"gen{s}"
        if s != seeds[0]:
            r = runner.generate(cfg, task, s, gen)
            if r["rc"] != 0:
                ok = {k: False for k in ok}
                det["seeds"][s] = dict(gen=r)
                continue
        sd = {}
        for tag, prog in [("ref", "solution/solve.py"), ("nop", None), ("mid", "baselines/mid.py"),
                          ("naive", "baselines/naive.py")] + [(Path(e).stem, e) for e in exploits]:
            sv = runner.solve(cfg, task, gen, prog, work, f"{tag}{s}")
            t = runner.test(cfg, task, gen, Path(sv["outdir"]), work, f"{tag}{s}")
            sd[tag] = dict(solve_rc=sv["rc"], solve_sec=sv["sec"], solve_err=sv["err"][-300:], passed=t["passed"],
                           n_checks=t["n_checks"], n_failed=t["n_failed"], failed=t["failed"],
                           metrics=[{k: m.get(k) for k in ("name", "value", "threshold", "direction", "passed")}
                                    for m in t["metrics"]])
        ref = sd["ref"]
        ok["oracle"] &= ref["passed"] and ref["solve_rc"] == 0
        ok["nop"] &= not sd["nop"]["passed"]
        ok["naive"] &= (not sd["naive"]["passed"]) and sd["naive"]["n_failed"] >= cfg["gates"]["naive_fail_min"]
        ok["mid"] &= (not sd["mid"]["passed"]) and 1 <= sd["mid"]["n_failed"] <= 2 and \
            sd["mid"]["n_checks"] >= ref["n_checks"]
        ok["exploit"] &= all(not sd[Path(e).stem]["passed"] for e in exploits) and bool(exploits)
        for m in ref["metrics"]:
            if m.get("threshold") is None or m.get("value") is None or m["name"].startswith("_"):
                continue
            room = (m["threshold"] - m["value"]) if m["direction"] == "<=" else (m["value"] - m["threshold"])
            if room < 0.5 * cfg["thresholds"]["min_ref_margin"] * max(abs(m["threshold"]), 1e-12):
                ok["oracle_margin"] = False
        if spec.get("form") == "PROG":
            hid = gen / "tests" / "hidden"
            pub = gen / "environment" / "data"
            hfiles = [p for p in hid.rglob("*") if p.is_file()]
            differs = bool(hfiles) and runner.tree_hash(hid) != runner.tree_hash(pub)
            hid_names = {m["name"] for m in spec.get("metrics", []) if m.get("on") == "hidden"}
            n_hidden = sum(1 for m in ref["metrics"] if not m.get("aux") and
                           (m["name"].startswith("hidden_") or m["name"] in hid_names))
            ok["hidden_shift"] &= differs and n_hidden >= 2
        det["seeds"][s] = sd
    g.update(ok)
    # thresholds stated
    ins = (task / "instruction.md").read_text() if (task / "instruction.md").exists() else ""
    nums = numbers_in(ins)
    unstated = [k for k, t in thresholds.items()
                if not any(math.isclose(v, abs(t["threshold"]), rel_tol=1e-6, abs_tol=1e-12) for v in nums)]
    g["thresholds_stated"] = not unstated
    det["unstated"] = unstated
    # leak gates on seed 0 artefacts
    gen0 = work / "detA"
    lr = leak_report(gen0 / "environment" / "data", [gen0 / "tests" / "truth"], ins, cfg,
                     depgraph=spec.get("depgraph"),
                     introspect="docker" if cfg["sandbox"]["backend"] == "docker" else "skip")
    g["leak"] = lr["verdict"] != "leak"          # "review" (warn-only findings) is recorded, not blocking
    det["leak"] = lr
    # isolation: nothing answer-bearing under environment/
    bad = [str(p.relative_to(gen0)) for p in (gen0 / "environment").rglob("*")
           if p.is_file() and re.search(r"(^|/)(tests|truth|generator)(/|$)", str(p.relative_to(gen0 / "environment")))]
    g["isolation"] = not bad
    det["isolation"] = bad
    blocking = [k for k, v in g.items() if not v and k in _blocking(cfg)]
    nonblocking = [k for k, v in g.items() if not v and k not in _blocking(cfg)]
    return dict(attempt=attempt, gates=g, blocking=blocking, nonblocking=nonblocking, ok=all(g.values()),
                details=det)


def _blocking(cfg) -> set[str]:
    b = set(cfg["gates"]["blocking"])
    if "oracle" in b:
        b.add("oracle_margin")
    return b | {"nop", "naive", "mid", "exploit"}
