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

import hashlib
import math
import re
from pathlib import Path

from . import runner
from .leakcheck import leak_report
from .spec import _requires_assumptions, is_resource_metric, metric_family, requires_hidden_provenance
from .state import read_json
from .statement import FILE_TOKEN, assumptions_stated, output_relative_path, statement_clean, threshold_format


PROBE_POLICY_VERSION = "s7-probes-v1"


def current_probe_report(report):
    return isinstance(report, dict) and report.get("probe_policy_version") == PROBE_POLICY_VERSION


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


_PARTIAL = re.compile(r"(?:\b(?:includ\w*|at least|additional|extra|optional|among|such as|for example|subset|"
                      r"other columns|more columns|may also|derived from|same as|like|whose|corresponds?)\b|"
                      r"\be\.g\.|\bi\.e\.)", re.I)


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
        t = output_relative_path(tok)
        if t in names:
            return t
        same = by_base.get(t.rsplit("/", 1)[-1]) or []
        return same[0] if "/" not in t and len(same) == 1 else None

    files_on = [{output_relative_path(m.group(0)) for m in FILE_TOKEN.finditer(line)} for line in lines]
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


def hidden_duplicates(pub: Path, hid: Path, shared_inputs=(), copied_share_max: float = 0.5) -> dict:
    """Compare file bytes across splits and relative-path/digest trees across hidden regimes."""
    def index(root):
        result = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                with p.open("rb") as f:
                    result[p.relative_to(root)] = (p.stat().st_size, hashlib.file_digest(f, "sha256").hexdigest())
        return result

    public = index(pub)
    by_digest, any_digest = {}, {d for _, d in public.values()}
    for path, (size, digest) in public.items():
        if size > 1024:
            by_digest.setdefault(digest, []).append(str(path))
    hidden = index(hid)
    copies = {str(path): by_digest[digest] for path, (size, digest) in hidden.items()
              if size > 1024 and digest in by_digest and path.name not in shared_inputs}
    sets = sorted(d.name for d in hid.iterdir() if d.is_dir()) if hid.is_dir() else []
    trees = {name: {p.relative_to(name): v for p, v in hidden.items() if p.parts[0] == name} for name in sets}
    same_sets = [[a, b] for i, a in enumerate(sets) for b in sets[i+1:] if trees[a] == trees[b]]
    same_public = [name for name, tree in {".": hidden, **trees}.items() if tree and tree == public]
    # Allow individual shared inputs, such as a training table or time grid.
    mostly = {}
    for name, tree in ({".": hidden} if not sets else trees).items():
        total = sum(size for size, _ in tree.values())
        # every byte-identical file counts here, small ones too (a hidden set of copied 800-byte case files);
        # empty files carry nothing
        copied = sum(size for p, (size, digest) in tree.items()
                     if size and digest in any_digest and Path(p).name not in shared_inputs)
        if total and copied >= copied_share_max * total:
            mostly[name] = round(copied / total, 3)
    return dict(ok=bool(hidden) and not (same_sets or same_public or mostly), copied_files=copies,
                identical_sets=same_sets, identical_public=same_public, mostly_public=mostly)


def hidden_parity(metrics: list, spec: dict, hidden_sets=None) -> dict:
    declared = {m["name"]: m for m in spec.get("metrics", [])}
    families = {"public": set(), "hidden": set()}
    # The gate supplies the full layout for per-set checks and an empty tuple for historical aggregate checks.
    # Standalone reports with no layout can infer the measured scopes from the rows.
    if hidden_sets is None:
        hidden_sets = sorted({scope for row in metrics for scope in row.get("hidden_sets", [])})
    by_set = {name: set() for name in hidden_sets}
    exempt = set()
    for row in metrics:
        name = row.get("name")
        if not name or row.get("aux") or name.startswith("_") or row.get("gated") is False:
            continue        # recorded-only resource metrics decide nothing, on either split
        m = declared.get(name, {})
        if is_resource_metric(m) or is_resource_metric({"name": name}):
            continue
        on = m.get("on")
        if on not in families:
            on = "hidden" if name.startswith("hidden_") or name.endswith("_hidden") else "public"
        family = metric_family(name)
        if on == "hidden" and by_set:
            for scope in row.get("hidden_sets") or ():
                if scope in by_set:
                    by_set[scope].add(family)
        families[on].add(family)
        reason = m.get("public_only_reason")
        if on == "public" and m.get("public_only") is True and isinstance(reason, str) and reason.strip():
            exempt.add(family)
    # Ignore aggregation and split qualifiers when matching a public family to its hidden counterpart.
    def covered(fam, others):
        want = _core_tokens(fam)
        return any(want and want <= _core_tokens(o) | _tokens(o) for o in others)

    def missing(measured):
        return sorted(f for f in families["public"] - exempt if not covered(f, measured))

    missing_public = sorted(f for f in families["hidden"] if not covered(f, families["public"]))
    if by_set:
        gaps = {name: missing(measured) for name, measured in by_set.items()}
        return dict(missing_hidden={k: v for k, v in gaps.items() if v}, missing_public=missing_public,
                    evaluated_hidden={k: sorted(v) for k, v in by_set.items()})
    return dict(missing_hidden=missing(families["hidden"]), missing_public=missing_public)


_AGG = {"mean", "median", "max", "min", "worst", "avg", "average", "best", "p95", "p90", "total", "overall", "all",
        "set", "a", "b", "c", "d", "regime", "case", "field", "split"}


def _tokens(name: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _core_tokens(name: str) -> set:
    return {t for t in _tokens(name) if t not in _AGG and not re.fullmatch(r"\d|set\w|h\d+", t)}


def realism(spec, public, runs, inventory=None):
    times = [v["solve_sec"] for k, v in runs.items() if re.fullmatch(r"ref\d+", k) and
             v.get("solve_rc") == 0 and isinstance(v.get("solve_sec"), (int, float))]
    hours = next((spec[k] for k in ("expert_time_estimate_hours", "expert_hours", "expert_time_hours")
                  if k in spec), 0)
    files = [p for p in public.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    formats = sorted({p.suffix.lower() for p in files})
    try:
        long_task = float(hours) >= 2
    except (TypeError, ValueError):
        long_task = False
    effort = bool(str(spec.get("effort_note") or "").strip()) or not times or max(times) >= 30 or not long_task
    names = {p["name"].lower() for p in inventory or ()}
    reader_formats = {Path(p.name.lower().removesuffix(".gz")).suffix for p in files}
    readers = {ext: sorted(matches) for ext in sorted(reader_formats)
               if (matches := runner.taskimage.READERS.get(ext, set()) & names)}
    data = bool(readers) or bool(str(spec.get("data_realism_note") or "").strip()) or size >= 2_000_000 or bool(
        set(formats) - {".csv", ".npy", ".json", ".txt"})
    detail = dict(bytes=size, extensions=formats)
    if inventory is not None:
        detail["readers"] = readers
    return {"effort_floor": (effort, dict(solve_sec=times, expert_hours=hours)),
            "data_realism": (data, detail)}


def run_gates(cfg: dict, item_dir: Path, spec: dict, thresholds: dict, attempt: int) -> dict:
    task = item_dir / "task"
    work = item_dir / "runs" / f"s7-{attempt}"
    seeds = cfg["gen"]["seeds"]
    versions = (spec.get("prompt_version"), read_json(item_dir / "_spec/provenance.json", {}).get("prompt_version"))
    review_rules = any(_requires_assumptions(v) for v in versions)
    per_set = any(requires_hidden_provenance(v) for v in versions)
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
    hidden_names = {m["name"] for m in spec.get("metrics", []) if m.get("on") == "hidden"}
    if spec.get("form") == "PROG":
        ok["hidden_parity"] = True
        det["hidden_parity"], det["hidden_shift"] = {}, {}
    det["oracle_margin"] = {}
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
                           verifier_rc=t["rc"], reward=t.get("reward"), n_checks=t["n_checks"], n_failed=t["n_failed"],
                           failed=t["failed"],
                           out_files=sum(1 for p in od.rglob("*") if p.is_file()) if od.exists() else 0,
                           metrics=[{k: m.get(k) for k in ("name", "value", "threshold", "direction", "passed", "aux",
                                                            "rc", "wall_times", "gated", "threshold_effective",
                                                            "speed_ratio", "hidden_sets") if k in m}
                                    for m in t["metrics"]])
            if s == seeds[0] and tag == "ref":
                sd[tag]["verifier"] = {k: t[k] for k in ("out", "err", "metrics") if k in t}
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
            hidden_metric = m["name"].startswith("hidden_") or m["name"] in hidden_names
            half = 0.5 * cfg["thresholds"]["min_ref_margin"]
            hm = cfg["thresholds"].get("min_ref_margin_hidden")
            # unset (None) = the old rule for every metric, also for configs that customised min_ref_margin
            fraction = (half if hm is None else float(hm)) if hidden_metric else half
            required = fraction * max(abs(th), 1e-12)
            if room < required and not math.isclose(room, required, rel_tol=1e-12):
                det["oracle_margin"].setdefault(s, []).append(m["name"])
                ok["oracle_margin"] = False
        if spec.get("form") == "PROG":
            hid = gen / "tests" / "hidden"
            pub = gen / "environment" / "data"
            dup = hidden_duplicates(pub, hid, spec.get("shared_inputs", []))
            n_hidden = sum(1 for m in ref["metrics"] if not m.get("aux") and
                           (m["name"].startswith("hidden_") or m["name"] in hidden_names))
            ok["hidden_shift"] &= dup["ok"] and n_hidden >= 2
            det["hidden_shift"][s] = dict(dup, n_hidden=n_hidden)
            parity = hidden_parity(ref["metrics"], spec, runner.hidden_set_names(gen) if per_set else ())
            det["hidden_parity"][s] = parity
            # Earlier specs did not require public_only declarations. Record their gaps for review.
            parity["enforced"] = review_rules
            if review_rules:
                ok["hidden_parity"] &= not parity["missing_hidden"]
        det["seeds"][s] = sd
    seed = seeds[0]
    gen = work / "detA"
    ref_out = work / f"out-ref{seed}"
    scored = det["seeds"][seed]["ref"]
    reference = {k: scored.get(k) for k in ("solve_rc", "verifier_rc", "reward", "verifier")}
    reference_ok = (reference.get("solve_rc") == 0 and reference.get("verifier_rc") == 0
                    and reference.get("reward") == "1")
    prog = spec.get("form") == "PROG"
    arms = [("coverage", "one_confident")]
    if prog:
        tp = (runner.truth_probe(cfg, task, gen, ref_out, work, f"truthprobe{seed}") if reference_ok
              else dict(ok=False, error=f"seed {seed} reference failed"))
        g["truth_isolation"], det["truth_isolation"] = tp["ok"], tp
        arms.append(("hidden_garbage", "hidden_garbage"))
    isolated = g.get("truth_isolation", True) is not False      # never run the reference wrapper with truth readable
    for gate, arm in arms:
        if reference_ok and isolated:
            result = runner.output_probe(cfg, task, gen, ref_out, work, arm, prog)
            if not prog:
                result["control"] = reference
        else:
            reason = f"seed {seed} reference failed" if not reference_ok else "not run: truth isolation failed"
            scopes = {scope: dict(status="inconclusive", error=reason, transformed_files=[])
                      for scope, _ in runner.probe_scopes(arm, prog, gen)}
            result = dict(ok=True, inconclusive=True, reference=reference, error=reason, scopes=scopes)
        det["seeds"][seed][f"exploit__{arm}"] = result
        g[gate] = result["ok"]
        det[gate] = result
    g.update(ok)
    # thresholds stated
    ins = (task / "instruction.md").read_text() if (task / "instruction.md").exists() else ""
    # the verifier-time note PKG generates is not the task author's text: never evidence for a stated threshold
    from .package import strip_time_note
    ins = strip_time_note(ins)
    missing_quotes = assumptions_stated(ins, spec.get("generator_assumptions", []), require_long_quote=review_rules)
    for name, issues in (("threshold_format", threshold_format(ins, thresholds)), ("statement_clean", statement_clean(ins)),
                         ("assumptions_stated", missing_quotes)):
        g[name], det[name] = not issues, issues
    measured = read_json(item_dir / "S6A_measure.json", {}).get("runs", {})
    state = read_json(item_dir / "state.json", {})
    inventory = state.get("inventory", []) if runner.taskimage.is_task(state) else None
    for name, (passed, detail) in realism(spec, work / "detA/environment/data", measured, inventory).items():
        g[name], det[name] = passed, detail
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
    # Inspect the generated files used by PKG.
    gen0 = work / "detA"
    lr = leak_report(gen0 / "environment" / "data", [gen0 / "tests" / "truth"], ins, cfg,
                     depgraph=spec.get("depgraph"), image=runner.taskimage.image_for(cfg, item_dir) if inventory is not None else None,
                     introspect="docker" if cfg["sandbox"]["backend"] == "docker" or inventory is not None else "skip")
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
    spec_a = prerelease.with_author_code(item_dir, spec)
    rm = prerelease.readymade_check(task, read_json(item_dir / "S1_candidate.json", {}), spec_a,
                                    public_env=gen0 / "environment")
    g["readymade"] = rm["ok"]
    det["readymade"] = rm
    if (item_dir / "_spec" / "author_code.json").exists():
        al = prerelease.author_name_leaks(task, spec_a, public_env=gen0 / "environment")
        g["author_leak"] = al["ok"]
        det["author_leak"] = al
    if (item_dir / "_spec" / "difficulty_design.json").exists() or (cfg.get("difficulty") or {}).get("profile_path"):
        # the difficulty targeting (profile ids / wording) must not reach the statement or any public file
        from . import difficulty
        try:
            prof = difficulty.load_profile(cfg)
        except (OSError, ValueError):
            prof = None
        tl = difficulty.targeting_leaks([task / "instruction.md", gen0 / "environment"], prof,
                                        read_json(item_dir / "_spec" / "difficulty_design.json", None))
        g["targeting_leak"] = not tl
        det["targeting_leak"] = tl
    return dict(attempt=attempt, probe_policy_version=PROBE_POLICY_VERSION, gates=g, details=det, **gate_policy(cfg, g))


def gate_policy(cfg, g):
    required = _blocking(cfg)
    blocking = [k for k, v in g.items() if not v and k in required]
    nonblocking = [k for k, v in g.items() if not v and k not in required]
    advisory = OUTPUT_PROBES | {"effort_floor", "data_realism"}
    return dict(blocking=blocking, nonblocking=nonblocking, ok=not blocking and set(nonblocking) <= advisory)


OUTPUT_PROBES = {"coverage", "one_confident", "hidden_garbage"}

# gates added after the 2026-11-23 colleague review: blocking even when a run config pins an older gates.blocking
# list (otherwise a fixable failure lands in nonblocking and S7 drops the item without an S8 round)
REVIEW_GATES = {"threshold_format", "statement_clean", "hidden_parity", "assumptions_stated"}


def _blocking(cfg) -> set[str]:
    b = (set(cfg["gates"]["blocking"]) - OUTPUT_PROBES) | REVIEW_GATES
    if "oracle" in b:
        b.add("oracle_margin")
    return b | {"nop", "naive", "mid", "exploit", "exploit_valid", "targeting_leak"}
