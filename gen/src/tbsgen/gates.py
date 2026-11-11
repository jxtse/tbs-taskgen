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
  output_contract every CSV header the statement gives for an output file equals (as a column set) the header the
                reference wrote there -- the verifier reads the reference's format, so a mismatch fails every agent
                that follows the statement (v2.7, blocking -> S8)
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
# a declaration is "Header: `a,b,c`" / "Columns: `a,b,c`": the keyword, a colon, then directly the backticked list
_DECL = re.compile(r"(?:^|[\s(*_|>-])(?:header|columns)\s*:\s*`([^`\n]+)`", re.I)
def _split_header(text: str) -> list[str] | None:
    """A backticked CSV header `a,b,c` -> ["a", "b", "c"] (csv rules; surrounding spaces trimmed, inner kept);
    None unless it is at least two plain column names."""
    import csv
    try:
        cols = next(csv.reader([text]))
    except (StopIteration, csv.Error):
        return None
    cols = [c.strip() for c in cols]
    if len(cols) < 2 or any(not c or len(c) > 64 or not re.match(r"^[A-Za-z_][\w .\-()/]*$", c) for c in cols):
        return None
    return cols


# any file-like token: name.ext, including multi-part extensions (input.csv.gz) and <output_dir>/ prefixes
_FILE_TOK = re.compile(r"(?<![\w./<>-])(?:<\w+>/)?[\w./-]*\w\.[A-Za-z][\w]*(?:\.[A-Za-z][\w]*)*(?![\w-])")
_PARTIAL = re.compile(r"(?:\b(?:includ\w*|at least|additional|extra|optional|among|such as|for example|subset|"
                      r"other columns|more columns|may also|derived from|same as|like|whose|corresponds?)\b|"
                      r"\be\.g\.|\bi\.e\.)", re.I)
_OUT_PREFIX = re.compile(r"^(?:\./|/app/output/|output/|<output_dir>/)")


def _norm_path(tok: str) -> str:
    return _OUT_PREFIX.sub("", tok.strip())


def stated_csv_headers(ins: str, names: list[str]) -> dict[str, list[dict]]:
    """Explicit, exhaustive CSV header declarations in the statement, per output file.  Strict on purpose (a false
    positive would block a good task):
      - anchor: a line that names exactly one of the outputs (relative path; ./, /app/output/, output/,
        <output_dir>/ stripped; or its basename when unique among the outputs).  File tokens are any name.ext.
      - declaration: `Header: `a,b,c`` / `Columns: `a,b,c`` -- the keyword, a colon, then directly ONE backticked
        comma list -- on the anchor line (only if it names no other file) or on one of the next 2 lines, which
        must name no file at all.
      - the anchor line and the declaration line must not use partial / attributing wording (includes, at least,
        additional, e.g., i.e., derived from, same as, like, whose, corresponds ...).
    -> {file: [{"cols": [...], "line": n}, ...]} -- every declaration in the window."""
    lines = ins.splitlines()
    by_base: dict[str, list[str]] = {}
    for n in names:
        by_base.setdefault(n.rsplit("/", 1)[-1], []).append(n)

    def resolve(tok: str) -> str | None:
        t = _norm_path(tok)
        if t in names:
            return t
        same = by_base.get(t.rsplit("/", 1)[-1]) or []
        return same[0] if "/" not in t and len(same) == 1 else None

    files_on = [{_norm_path(m.group(0)) for m in _FILE_TOK.finditer(line)} for line in lines]
    out: dict[str, list[dict]] = {}
    for k, toks in enumerate(files_on):
        outs = {r for r in (resolve(t) for t in toks) if r is not None}
        if len(outs) != 1 or _PARTIAL.search(lines[k]):
            continue
        n = outs.pop()
        for j in range(k, min(k + 3, len(lines))):
            if j > k and files_on[j]:
                break                                   # another file is named: its own declaration window
            if j == k and len(toks) > 1:
                continue                                # the anchor line names an input too
            if _PARTIAL.search(lines[j]):
                continue
            decl = [c for c in (_split_header(x) for x in _DECL.findall(lines[j])) if c]
            if len(decl) == 1:
                out.setdefault(n, []).append(dict(cols=decl[0], line=j + 1))
    return out


def output_contract(ins: str, ref_out: Path) -> dict:
    """The statement's explicit CSV header declarations must match what the reference writes (and the verifier,
    which the reference passes, reads), as column sets.  e2e-batch2 (nano): the statement said
    `material,rate,depth_nm,H_fit_GPa` while the scorer merged on obs_id / H_fit, so every agent that followed the
    text scored inf -- including Opus.  -> {ok, mismatch: {file: {stated, line, reference}}, checked}"""
    import csv
    ref_out = Path(ref_out)
    files = sorted(p for p in ref_out.rglob("*.csv") if p.is_file()) if ref_out.exists() else []
    rel = {str(p.relative_to(ref_out)): p for p in files}
    stated = stated_csv_headers(ins, sorted(rel))
    mismatch = {}
    for name, decls in stated.items():
        try:
            with open(rel[name], newline="") as fh:
                hdr = [c.strip() for c in next(csv.reader(fh), [])]
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
        bad = [d for d in decls if set(d["cols"]) != set(hdr)]
        if bad:
            mismatch[name] = dict(stated=bad[0]["cols"], line=bad[0]["line"], reference=hdr)
    return dict(ok=not mismatch, mismatch=mismatch, checked=sorted(stated))


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


def exploit_patch_problems(task: Path, patch: dict) -> list[str]:
    """An exploit-only S8 repair may rewrite existing baselines/exploit_*.py files and nothing else."""
    bad = []
    for rel in patch:
        if not re.fullmatch(r"baselines/exploit_[\w.-]+\.py", rel) or ".." in rel:
            bad.append(f"not an exploit baseline: {rel}")
        elif not (task / rel).is_file():
            bad.append(f"new file: {rel}")
    return bad


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
                                                            "rc", "wall_times", "gated", "threshold_effective",
                                                            "speed_ratio") if k in m}
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
            if m.get("gated") is False:
                continue        # recorded-only resource metric: never decides a run, no margin to demand
            th = m.get("threshold_effective", m["threshold"])          # machine-speed scaled time threshold
            room = (th - m["value"]) if m["direction"] == "<=" else (m["value"] - th)
            if room < 0.5 * cfg["thresholds"]["min_ref_margin"] * max(abs(th), 1e-12):
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
    # the verifier-time note PKG generates is not the task author's text: never evidence for a stated threshold
    from .package import strip_time_note
    ins = strip_time_note(ins)
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
    # output contract: CSV headers in the statement == the reference's (= what the verifier reads)
    oc = output_contract(ins, work / f"out-ref{seeds[0]}")
    g["output_contract"] = oc["ok"]
    det["output_contract"] = oc
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
    spec_a = prerelease.with_author_code(item_dir, spec)
    rm = prerelease.readymade_check(task, read_json(item_dir / "S1_candidate.json", {}), spec_a,
                                    public_env=gen0 / "environment")
    g["readymade"] = rm["ok"]
    det["readymade"] = rm
    if (item_dir / "_spec" / "author_code.json").exists():
        al = prerelease.author_name_leaks(task, spec_a, public_env=gen0 / "environment")
        g["author_leak"] = al["ok"]
        det["author_leak"] = al
    blocking = [k for k, v in g.items() if not v and k in _blocking(cfg)]
    nonblocking = [k for k, v in g.items() if not v and k not in _blocking(cfg)]
    return dict(attempt=attempt, gates=g, blocking=blocking, nonblocking=nonblocking, ok=all(g.values()),
                details=det)


def _blocking(cfg) -> set[str]:
    b = set(cfg["gates"]["blocking"])
    if "oracle" in b:
        b.add("oracle_margin")
    return b | {"nop", "naive", "mid", "exploit", "exploit_valid"}
