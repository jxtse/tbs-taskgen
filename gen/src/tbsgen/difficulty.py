"""Difficulty targeting for code-grounded tasks (S5G): a weakness profile of the calibration student, the
difficulty design an authoring agent records for its task, a light integrity check of that design against the
measured reference / mid / naive separation, and the hardening loop that sends a task the student solved on its
first attempt back to S5G (hardening rounds; `[difficulty]` in config.py).

The profile is an internal markdown file (difficulty.profile_path).  This module only loads it, takes its id
(front matter `profile_id:` or the file stem), its sha256 and the weakness ids (`## <ID> <title>` headings).  The
repository ships a generic example only (gen/examples/weakness_profile.example.md).

difficulty_design.json (written by the S5G agent next to the spec; never shown to solvers):
    {"profile": "<profile id>",
     "levers": [{"weakness_id": "<id from the profile>", "lever": "what in this task exercises it",
                 "paper_evidence": "section / equation / figure of the paper",
                 "code_evidence": "file:line or function of the authors' code",
                 "why_naive_fails": "...", "why_mid_fails": "...", "reference_handles": "...",
                 "metrics": ["<scored metric names where the lever shows>"]}],
     "artificial_difficulty_check": "why none of the forbidden shortcuts is used",
     "hardening": {"round": k, "changes": "...", "why_previous_solution_fails_now": "..."}   # hardening rounds only
    }
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from .state import read_json, write_json

DESIGN_FILE = "difficulty_design.json"
_ID_RE = re.compile(r"^##\s+([A-Z][A-Z0-9_]*\d+)\b", re.M)
_FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
LEVER_KEYS = ("weakness_id", "lever", "paper_evidence", "code_evidence", "why_mid_fails", "reference_handles")
# words that must never reach the solver's statement (they would reveal the targeting)
_STATEMENT_BANNED = re.compile(r"weakness[ _-]?profile|difficulty[ _-]?design|\bweak point|\btargeted weakness",
                               re.I)


def dcfg(cfg: dict) -> dict:
    return cfg.get("difficulty") or {}


# ------------------------------------------------------------------ profile
def load_profile(cfg: dict) -> dict | None:
    """None when no profile is configured; raises FileNotFoundError / ValueError for a configured but unusable one."""
    p = dcfg(cfg).get("profile_path")
    if not p:
        return None
    path = Path(p).expanduser()
    text = path.read_text()
    ids = list(dict.fromkeys(_ID_RE.findall(text)))
    if not ids:
        raise ValueError(f"weakness profile {path.name}: no '## <ID> <title>' entries")
    pid = dcfg(cfg).get("profile_id")
    m = _FM_RE.match(text)
    if not pid and m:
        for line in m.group(1).splitlines():
            if line.strip().startswith("profile_id:"):
                pid = line.split(":", 1)[1].strip()
    return dict(id=pid or path.stem, sha=hashlib.sha256(text.encode()).hexdigest()[:16], ids=ids, text=text)


def profile_problem(cfg: dict) -> str | None:
    try:
        load_profile(cfg)
    except (OSError, ValueError) as e:
        return f"difficulty_profile:{type(e).__name__}"
    return None


def provenance(prof: dict | None) -> dict | None:
    return None if not prof else dict(id=prof["id"], sha=prof["sha"])


# ------------------------------------------------------------------ design validation (S5G acceptance)
def validate_design(design, prof: dict, spec: dict, harden_round: int = 0) -> list[str]:
    errs = []
    if not isinstance(design, dict):
        return ["difficulty_design.json is not an object"]
    if design.get("profile") != prof["id"]:
        errs.append(f"profile must be {prof['id']!r}")
    levers = design.get("levers")
    if not isinstance(levers, list) or not 1 <= len(levers) <= 3:
        errs.append("levers must be a list of 1-3 entries")
        levers = levers if isinstance(levers, list) else []
    names = {m.get("name") for m in spec.get("metrics") or [] if isinstance(m, dict)}
    for i, lv in enumerate(levers):
        if not isinstance(lv, dict):
            errs.append(f"levers[{i}] is not an object")
            continue
        if lv.get("weakness_id") not in prof["ids"]:
            errs.append(f"levers[{i}].weakness_id {lv.get('weakness_id')!r} not in the profile")
        for k in LEVER_KEYS:
            if not str(lv.get(k) or "").strip():
                errs.append(f"levers[{i}].{k} is empty")
        ms = lv.get("metrics")
        if not isinstance(ms, list) or not ms:
            errs.append(f"levers[{i}].metrics must name at least one scored metric")
        else:
            # spec metric names are base names; thresholds may prefix public_/hidden_
            bad = [m for m in ms if not any(str(m) == n or str(m).endswith("_" + str(n)) for n in names)]
            if bad:
                errs.append(f"levers[{i}].metrics {bad} are not metrics of the spec")
    if not str(design.get("artificial_difficulty_check") or "").strip():
        errs.append("artificial_difficulty_check is empty")
    if harden_round:
        h = design.get("hardening")
        if not isinstance(h, dict) or not str(h.get("changes") or "").strip() or \
                not str(h.get("why_previous_solution_fails_now") or "").strip():
            errs.append("hardening.changes / hardening.why_previous_solution_fails_now are required in a "
                        "hardening round")
    return errs


# ------------------------------------------------------------------ S7 integrity check (measured separation)
def _fails(v, t: dict) -> bool:
    if v is None or t.get("threshold") is None:
        return True
    try:
        v = float(v)
    except (TypeError, ValueError):
        return True
    if v != v:                                  # NaN
        return True
    return v < t["threshold"] if t.get("direction") == ">=" else v > t["threshold"]


def _metric_keys(m: str, th: dict) -> list[str]:
    if m in th:
        return [m]
    return [k for k in th if k.endswith("_" + m)]


def check_lever_separation(item_dir: Path, spec: dict, thresholds: dict, prof: dict | None = None) -> dict:
    """Light integrity check run with the S7 gates on a task that has a difficulty design: every lever names
    scored, non-resource metrics on which the reference passes on every seed while mid or naive fails on every
    seed (the lever is real, not a claim); the statement does not reveal the targeting; in a hardening round the
    task differs from the previous version."""
    from .spec import is_resource_metric
    design = read_json(item_dir / "_spec" / DESIGN_FILE, None)
    meas = read_json(item_dir / "S6A_measure.json", {}) or {}
    problems, per = [], []
    if not isinstance(design, dict):
        return dict(ok=False, problems=["no difficulty_design.json"], levers=[])
    resource = {m["name"] for m in spec.get("metrics") or [] if isinstance(m, dict) and is_resource_metric(m)}
    for i, lv in enumerate(design.get("levers") or []):
        if not isinstance(lv, dict):
            continue
        rec = dict(weakness_id=lv.get("weakness_id"), metrics={})
        shown = False
        for m in lv.get("metrics") or []:
            keys = _metric_keys(str(m), thresholds)
            if not keys:
                rec["metrics"][str(m)] = "no threshold"
                continue
            for k in keys:
                base = k.split("_", 1)[1] if k not in {x.get("name") for x in spec.get("metrics") or []} and \
                    "_" in k else k
                if k in resource or base in resource:
                    rec["metrics"][k] = "resource metric (time / memory) cannot carry a lever"
                    continue
                t = thresholds[k]
                ref = meas.get("ref") or {}
                seeds = sorted(ref.keys())
                ref_ok = bool(seeds) and all(not _fails((ref.get(s) or {}).get(k), t) for s in seeds)
                mid_f = bool(seeds) and all(_fails(((meas.get("mid") or {}).get(s) or {}).get(k), t) for s in seeds)
                nai_f = bool(seeds) and all(_fails(((meas.get("naive") or {}).get(s) or {}).get(k), t) for s in seeds)
                rec["metrics"][k] = dict(ref_passes=ref_ok, mid_fails=mid_f, naive_fails=nai_f)
                shown |= ref_ok and (mid_f or nai_f)
        rec["shown"] = shown
        if not shown:
            problems.append(f"lever {i} ({lv.get('weakness_id')}): no named metric where the reference passes and "
                            "mid or naive fails on every seed")
        per.append(rec)
    if not per:
        problems.append("no levers")
    leaks = targeting_leaks([item_dir / "task" / "instruction.md"], prof, design)
    if leaks:
        problems.append(f"the statement reveals the difficulty targeting: {leaks[:3]}")
    return dict(ok=not problems, problems=problems, levers=per)


def check_hardening(item_dir: Path, rnd: int) -> list[str]:
    """A hardening round must change the science (generator / reference / metric script), not only the baselines
    (that would only move the derived thresholds) or the wording."""
    rec = read_json(item_dir / "harden" / f"round-{rnd}" / "record.json", {}) or {}
    prev = item_dir / "harden" / f"round-{rnd}" / "prev" / "task"
    if not prev.is_dir():
        return [f"hardening round {rnd}: previous version missing"]
    if science_sha(prev) == science_sha(item_dir / "task"):
        return [f"hardening round {rnd}: generator, reference and metric script are unchanged (only baselines / "
                "wording changed): thresholds would merely move"]
    _ = rec
    return []


def targeting_leaks(paths: list[Path], prof: dict | None, design: dict | None) -> list[str]:
    """Files under `paths` (the statement and every public file a solver gets) that reveal the difficulty
    targeting: the profile id, a weakness id, or profile / design wording.  Every file is scanned as raw bytes in
    overlapping 1 MB chunks whatever its extension or size; gzip / bz2 / xz files and zip members are scanned
    decompressed too.  A file that cannot be read is reported."""
    ids = set((prof or {}).get("ids") or [])
    ids |= {str(lv.get("weakness_id")) for lv in (design or {}).get("levers") or []
            if isinstance(lv, dict) and lv.get("weakness_id")}
    pid = (prof or {}).get("id") or (design or {}).get("profile")
    pats = [re.compile(rb"(?<![A-Za-z0-9])" + re.escape(i.encode()) + rb"(?![A-Za-z0-9])") for i in sorted(ids) if i]
    if pid:
        pats.append(re.compile(re.escape(str(pid).encode()), re.I))
    pats.append(re.compile(_STATEMENT_BANNED.pattern.encode(), re.I))
    out = []
    for root in paths:
        files = [root] if root.is_file() else (sorted(x for x in root.rglob("*") if x.is_file()) if root.is_dir()
                                               else [])
        for f in files:
            if f.is_symlink():
                out.append(f"{f.name}: symlink")
                continue
            try:
                with f.open("rb") as fh:
                    hit = _scan_stream(fh, pats)
                if hit is None:
                    hit = _scan_compressed(f, pats)
            except Exception as e:  # noqa: BLE001  (unreadable / corrupt archive: report, never pass silently)
                out.append(f"{f.name}: unreadable ({type(e).__name__})")
                continue
            if hit:
                out.append(f"{f.name}: {hit!r}")
    return out


def _scan_compressed(f: Path, pats: list) -> str | None:
    import bz2
    import gzip
    import lzma
    import zipfile
    with f.open("rb") as fh:
        magic = fh.read(6)
    opener = {b"\x1f\x8b": gzip.open, b"BZh": bz2.open, b"\xfd7zXZ": lzma.open}
    for m, op in opener.items():
        if magic.startswith(m):
            with op(f, "rb") as fh:
                return _scan_stream(fh, pats)
    if magic.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(f) as z:
            for name in z.namelist():
                with z.open(name) as fh:
                    hit = _scan_stream(fh, pats)
                if hit:
                    return f"{name}: {hit}"
    return None


def _scan_stream(fh, pats: list) -> str | None:
    tail = b""
    while True:
        chunk = fh.read(1 << 20)
        if not chunk:
            return None
        buf = tail + chunk
        for pat in pats:
            m = pat.search(buf)
            if m:
                return m.group(0).decode(errors="replace")
        tail = buf[-256:]


def science_sha(task: Path) -> str:
    """Hash of what defines the science of a version: generator (and assets), reference, metric script."""
    return _sha_parts(task, ("generator", "solution", "tests/measure.py"))


def task_sha(task: Path) -> str:
    """Science plus the baselines (generator, reference, metric script, mid / naive)."""
    return _sha_parts(task, ("generator", "solution", "tests/measure.py", "baselines/mid.py", "baselines/naive.py"))


def version_diff(prev: Path, cur: Path, limit: int = 200_000) -> str:
    """Unified diff of the authored files of two versions (for the S9R reviewer)."""
    import difflib
    out = []
    rels = set()
    for root in (prev, cur):
        for sub in ("instruction.md", "generator", "solution", "tests/measure.py", "baselines/mid.py",
                    "baselines/naive.py"):
            p = root / sub
            if p.is_file():
                rels.add(sub)
            elif p.is_dir():
                rels |= {str(x.relative_to(root)) for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts}
    for rel in sorted(rels):
        a, b = prev / rel, cur / rel
        try:
            ta = a.read_text(errors="replace").splitlines(True) if a.is_file() else []
            tb = b.read_text(errors="replace").splitlines(True) if b.is_file() else []
        except OSError:
            continue
        if ta != tb:
            out += list(difflib.unified_diff(ta, tb, f"prev/{rel}", f"new/{rel}"))
    text = "".join(out)
    return text if len(text) <= limit else text[:limit] + f"\n... [diff truncated, {len(text)} chars]\n"


def _sha_parts(task: Path, parts) -> str:
    h = hashlib.sha256()
    for sub in parts:
        p = task / sub
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else ([p] if p.is_file() else [])
        for f in files:
            if "__pycache__" in f.parts:
                continue
            h.update(str(f.relative_to(task)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ hardening loop
def harden_enabled(cfg: dict, it) -> bool:
    d = dcfg(cfg)
    if not d.get("harden", True) or it.item_id in (d.get("harden_exclude") or []):
        return False
    return it.p("_spec", "author_code.json").exists()          # code-grounded (S5G) items only


def decisive(runs: list[dict], cfg: dict) -> list[dict] | None:
    """Decisive calibration runs, None when the timeout policy parks the item."""
    from .calib import _Park, decisive_runs
    try:
        return decisive_runs(runs, cfg)
    except _Park:
        return None


def rounds_left(cfg: dict, it) -> bool:
    return not it.state.get("harden_exhausted") and \
        int(it.state.get("harden_round", 0)) < int(dcfg(cfg).get("harden_max_rounds", 2))


def should_harden(cfg: dict, it, runs: list[dict]) -> bool:
    """The student passed its first decisive attempt of this calibration epoch and nothing else ran yet."""
    if not harden_enabled(cfg, it) or not rounds_left(cfg, it):
        return False
    if it.state.get("rescore_pending") or it.state.get("thresholds_locked"):
        return False        # a re-score / regate in progress decides first
    if read_json(it.p("review", "runs.json"), []):
        return False        # an S9R-reviewed item: review job names follow the review round (not supported)
    real = decisive(runs, cfg)
    return bool(real) and len(real) == 1 and real[0]["who"] == "glm" and real[0]["outcome"] == "pass"


def _trial_dir(rec: dict) -> Path | None:
    t = rec.get("trial") or {}
    jd = Path(t.get("job_dir") or rec.get("job_dir") or "/nonexistent")
    if not jd.is_dir():
        return None
    name = t.get("trial")
    if name and (jd / name).is_dir():
        return jd / name
    tr = sorted(p for p in jd.iterdir() if (p / "result.json").exists())
    return tr[0] if tr else None


def _final_message(trial: Path | None, limit: int = 4000) -> str:
    if trial is None:
        return ""
    p = trial / "agent" / "claude-code.txt"
    if not p.exists() or not _inside_trial(trial, p):
        return ""
    last = ""
    try:
        for line in p.read_text(errors="replace").splitlines():
            if '"type":"result"' in line.replace(" ", "") or '"type": "result"' in line:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "result" and d.get("result"):
                    last = str(d["result"])
    except OSError:
        return ""
    return last if len(last) <= limit else last[:limit] + f" ...[+{len(last) - limit} chars]"


def feedback(rec: dict, allowed: dict) -> dict:
    """What the authoring agent may learn from the student's passing run: its own summary of the method (final
    message), elapsed minutes, cost, the solution's run time, and for each scored metric of this version
    (`allowed` = thresholds.json "thresholds": name -> {threshold, direction}) the student's aggregate value and its
    relative margin.  Only names in `allowed`, only finite numbers, one value per metric; the threshold and the
    direction come from `allowed` (the author's own file), not from the verifier.  Never: the verifier's stdout,
    unknown or structured metric rows, the student's output files, per-object results, hidden inputs or truth."""
    import math
    trial = _trial_dir(rec)
    vals, runtime = {}, None
    mp = trial / "verifier" / "metrics.jsonl" if trial else None
    if mp and mp.exists() and _inside_trial(trial, mp):
        for line in mp.read_text(errors="replace").splitlines():
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if not isinstance(m, dict):
                continue
            name, v = m.get("name"), m.get("value")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                continue
            if name == "_solution_cli_run" and runtime is None:
                runtime = round(float(v), 3)
            elif isinstance(name, str) and name in allowed and name not in vals:
                vals[name] = float(v)
    margins = []
    for name in sorted(vals):
        t = allowed[name] or {}
        thr, d = t.get("threshold"), t.get("direction")
        rel = None
        if isinstance(thr, (int, float)) and thr != 0 and d in (">=", "<="):
            rel = round(((vals[name] - thr) if d == ">=" else (thr - vals[name])) / abs(thr), 4)
        margins.append(dict(metric=name, value=round(vals[name], 6), threshold=thr, direction=d,
                            relative_margin=rel))
    t = rec.get("trial") or {}
    mins = rec.get("minutes", t.get("minutes"))
    cost = rec.get("cost_usd", t.get("cost_usd"))
    return dict(job=rec.get("job_name"), model=rec.get("model"), outcome=rec.get("outcome"),
                minutes=round(float(mins), 1) if isinstance(mins, (int, float)) else None,
                cost_usd=round(float(cost), 2) if isinstance(cost, (int, float)) else None,
                solution_runtime_sec=runtime, metric_margins=margins,
                student_final_message=_final_message(trial))


def _inside_trial(trial: Path, p: Path) -> bool:
    try:
        cur = p
        while cur != trial:
            if cur.is_symlink():
                return False
            cur = cur.parent
        return True
    except OSError:
        return False


VERSION_FILES = ("task", "_spec", "pkg", "gates", "thresholds.json", "S6A_measure.json", "S6B_decisions.json",
                 "S6C_calls.json", "S6_failure.json", "S5G_result.json", "PKG_check.json", "S5G/report.md",
                 "S5G/run.json")
# files of the old version that must not look like the new version's (removed after the snapshot)
STALE_FILES = ("S6A_measure.json", "S6B_decisions.json", "S6_failure.json", "PKG_check.json")


def snapshot_version(it, dest: Path) -> None:
    """Copy the files of the current task version under dest (idempotent: dest/DONE marks a complete copy)."""
    if (dest / "DONE").exists():
        return
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel in VERSION_FILES:
        src = it.p(rel)
        if src.is_dir():
            shutil.copytree(src, dest / rel, symlinks=True)
        elif src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
    (dest / "DONE").write_text(str(time.time()))


def restore_version(it, src: Path, failed: Path) -> None:
    """Put a snapshot back as the current version (idempotent): the current files move to `failed` once
    (failed/DONE marks it), then every snapshot file replaces the current one."""
    if not (failed / "DONE").exists():
        failed.mkdir(parents=True, exist_ok=True)
        for rel in VERSION_FILES:
            cur = it.p(rel)
            if cur.exists():
                (failed / rel).parent.mkdir(parents=True, exist_ok=True)
                if (failed / rel).is_dir():
                    shutil.rmtree(failed / rel)
                elif (failed / rel).exists():
                    (failed / rel).unlink()
                shutil.move(str(cur), str(failed / rel))
        (failed / "DONE").write_text(str(time.time()))
    for rel in VERSION_FILES:
        cur, s = it.p(rel), src / rel
        if cur.is_dir():
            shutil.rmtree(cur)
        elif cur.exists():
            cur.unlink()
        if s.is_dir():
            shutil.copytree(s, cur, symlinks=True)
        elif s.is_file():
            cur.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, cur)


def write_round(it, k: int, **kw) -> dict:
    p = it.p("harden", f"round-{k}", "record.json")
    rec = read_json(p, {}) or {}
    rec.update(kw)
    write_json(p, rec)
    return rec
