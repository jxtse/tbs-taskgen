"""S7 mechanical gates.  No model calls (L4b cheap-agent inversion is 待定 and reported as skipped).

gates/attempt-<n>.json = {"gates": {name: bool}, "blocking": [names], "details": {...}}
  determinism   generator: same seed twice -> identical env/tests hashes
  oracle        reference passes test_outputs.py on every seed
  oracle_margin reference clears each threshold (re-measured values, relative margin >= min_ref_margin/2)
  nop           empty /app/output fails
  naive         naive fails >= naive_fail_min checks on every seed
  mid           mid fails 1..2 checks and all checks were evaluated
  exploit       every baselines/exploit_*.py fails
  exploit_valid every exploit really ran under the same invocation and produced scorable output (v2.5): its solve
                step exited 0, it wrote something to /app/output, every solution_cli.py run (PROG) exited 0, every
                check the reference got was evaluated, and not every metric sits at the empty-output (nop) value.
                An exploit that crashes (e2e-prod-20261109: single-case hidden inputs) tests nothing, so "it failed"
                is not a pass of the exploit gate (blocking -> S8)
  hidden_shift  PROG: tests/hidden exists, differs from public, >= 2 hidden checks recorded for the reference
  truth_isolation PROG: a probe solution_cli.py (reads every file under /tests except hidden inputs) must be run
                via _metrics.run_solution and read nothing; catches verifiers that exec agent code themselves
                (as root, truth readable) -- found in e2e-iter1 where a truth-reading CLI got reward 1
  thresholds_stated  every threshold appears numerically in instruction.md
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
from .spec import metric_family


CLI_RE = re.compile(r"solution_cli\.py\s+<input_dir>\s+<output_dir>")


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


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b or (math.isnan(a) and math.isnan(b))
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


def exploit_validity(sd_exp: dict, sd_ref: dict, sd_nop: dict, form: str | None) -> dict:
    """Did an exploit baseline really run and get scored?  sd_* = per-seed arm records of run_gates (with
    solve_rc, out_files, n_checks, metrics incl. aux `_solution_cli_run` rows).  -> {ok, why: [...]}"""
    why = []
    if sd_exp.get("solve_rc") != 0:
        why.append(f"solve step exited {sd_exp.get('solve_rc')}")
    if not sd_exp.get("out_files"):
        why.append("wrote nothing to /app/output")
    runs = [m for m in sd_exp.get("metrics", []) if m.get("name") == "_solution_cli_run"]
    if form == "PROG":
        if not runs:
            why.append("solution_cli.py was never run by the verifier")
        bad = [m.get("rc") for m in runs if m.get("rc") != 0]
        if bad:
            why.append(f"solution_cli.py exited {bad[:3]} under the verifier's invocation")
    if sd_exp.get("n_checks", 0) < sd_ref.get("n_checks", 0):
        why.append(f"only {sd_exp.get('n_checks')} of {sd_ref.get('n_checks')} checks evaluated")
    ev = {m["name"]: m.get("value") for m in sd_exp.get("metrics", []) if not m.get("aux") and "name" in m}
    nv = {m["name"]: m.get("value") for m in sd_nop.get("metrics", []) if not m.get("aux") and "name" in m}
    common = [k for k in ev if k in nv]
    if common and all(_same(ev[k], nv[k]) for k in common):
        why.append("every metric equals the empty-output (nop) value: no scorable output")
    return dict(ok=not why, why=why)


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
    ok = dict(oracle=True, nop=True, naive=True, mid=True, exploit=True, exploit_valid=True, hidden_shift=True,
              oracle_margin=True)
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
            od = Path(sv["outdir"])
            sd[tag] = dict(solve_rc=sv["rc"], solve_sec=sv["sec"], solve_err=sv["err"][-300:], passed=t["passed"],
                           reward=t.get("reward"), n_checks=t["n_checks"], n_failed=t["n_failed"], failed=t["failed"],
                           out_files=sum(1 for p in od.rglob("*") if p.is_file()) if od.exists() else 0,
                           metrics=[{k: m.get(k) for k in ("name", "value", "threshold", "direction", "passed", "aux",
                                                            "rc", "wall_times") if k in m}
                                    for m in t["metrics"]])
        ref = sd["ref"]
        ok["oracle"] &= ref["passed"] and ref["solve_rc"] == 0
        ok["nop"] &= not sd["nop"]["passed"]
        ok["naive"] &= (not sd["naive"]["passed"]) and sd["naive"]["n_failed"] >= cfg["gates"]["naive_fail_min"]
        n_mid = len({metric_family(n) for n in sd["mid"]["failed"]})   # public/hidden twins = one criterion
        ok["mid"] &= (not sd["mid"]["passed"]) and 1 <= n_mid <= 2 and \
            sd["mid"]["n_checks"] >= ref["n_checks"]
        ok["exploit"] &= all(not sd[Path(e).stem]["passed"] for e in exploits) and bool(exploits)
        for e in exploits:
            v = exploit_validity(sd[Path(e).stem], ref, sd["nop"], spec.get("form"))
            sd[Path(e).stem]["valid"] = v
            ok["exploit_valid"] &= v["ok"]
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
            if s == seeds[0]:
                tp = runner.truth_probe(cfg, task, gen, work / f"out-ref{s}", work, f"truthprobe{s}")
                g["truth_isolation"] = tp["ok"]
                det["truth_isolation"] = tp
        det["seeds"][s] = sd
    g.update(ok)
    # thresholds stated
    ins = (task / "instruction.md").read_text() if (task / "instruction.md").exists() else ""
    nums = numbers_in(ins)
    unstated = [k for k, t in thresholds.items()
                if not any(math.isclose(v, abs(t["threshold"]), rel_tol=1e-6, abs_tol=1e-12) for v in nums)]
    g["thresholds_stated"] = not unstated
    det["unstated"] = unstated
    if spec.get("form") == "PROG":
        # e2e-iter1: instruction said `solution_cli.py --data <csv> --out <dir>` while the verifier calls
        # `solution_cli.py <input_dir> <output_dir>`: an agent following the text fails every hidden check
        g["cli_contract"] = bool(CLI_RE.search(ins)) and not re.search(r"solution_cli\.py\s+--\w", ins)
        det["cli_contract"] = dict(found=bool(CLI_RE.search(ins)))
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
    # TDM terms: public task files must not copy long passages of an Elsevier / Wiley TDM full text
    from . import tdmgate
    tv = tdmgate.check(cfg, item_dir, task, gen0 / "environment" / "data")
    det["tdm_verbatim"] = tv
    if tv["applies"]:
        g["tdm_verbatim"] = tv["ok"]
    # ready-made implementation of the paper in the image, the reference or a baseline (prerelease.py)
    from . import prerelease
    from .state import read_json
    rm = prerelease.readymade_check(task, read_json(item_dir / "S1_candidate.json", {}), spec,
                                    public_env=gen0 / "environment")
    g["readymade"] = rm["ok"]
    det["readymade"] = rm
    blocking = [k for k, v in g.items() if not v and k in _blocking(cfg)]
    nonblocking = [k for k, v in g.items() if not v and k not in _blocking(cfg)]
    return dict(attempt=attempt, gates=g, blocking=blocking, nonblocking=nonblocking, ok=all(g.values()),
                details=det)


def _blocking(cfg) -> set[str]:
    b = set(cfg["gates"]["blocking"])
    if "oracle" in b:
        b.add("oracle_margin")
    return b | {"nop", "naive", "mid", "exploit", "exploit_valid"}
