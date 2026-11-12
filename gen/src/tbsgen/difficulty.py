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
    ins = item_dir / "task" / "instruction.md"
    text = ins.read_text(errors="replace") if ins.exists() else ""
    ids = (prof or {}).get("ids") or [lv.get("weakness_id") for lv in design.get("levers") or []
                                      if isinstance(lv, dict) and lv.get("weakness_id")]
    named = sorted({i for i in ids if i and re.search(rf"\b{re.escape(str(i))}\b", text)})
    if named or _STATEMENT_BANNED.search(text):
        problems.append(f"the statement reveals the difficulty targeting ({named or 'profile wording'})")
    h = design.get("hardening")
    if isinstance(h, dict) and h.get("prev_task_sha"):
        if h["prev_task_sha"] == task_sha(item_dir / "task"):
            problems.append("hardening round: the task is unchanged from the previous version")
    return dict(ok=not problems, problems=problems, levers=per)


def task_sha(task: Path) -> str:
    """Hash of the authored task files that define the science (generator, reference, metric script, baselines);
    the statement and the S6C tests are excluded (S6C rewrites them)."""
    h = hashlib.sha256()
    for sub in ("generator", "solution", "tests/measure.py", "baselines/mid.py", "baselines/naive.py"):
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
    if not p.exists():
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


def feedback(rec: dict) -> dict:
    """What the authoring agent may learn from the student's passing run: its own summary of the method (final
    message), elapsed minutes, cost, and per-metric value / threshold / relative margin (aggregates the author
    could compute itself from its own truth).  Never: the verifier's stdout, the student's output files, per-object
    results, hidden inputs or truth."""
    trial = _trial_dir(rec)
    margins = []
    mp = trial / "verifier" / "metrics.jsonl" if trial else None
    runtime = None
    if mp and mp.exists():
        for line in mp.read_text(errors="replace").splitlines():
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("aux"):
                if m.get("name") == "_solution_cli_run":
                    runtime = m.get("value")
                continue
            v, t, d = m.get("value"), m.get("threshold"), m.get("direction")
            rel = None
            if isinstance(v, (int, float)) and isinstance(t, (int, float)) and t != 0:
                rel = round(((v - t) if d == ">=" else (t - v)) / abs(t), 4)
            margins.append(dict(metric=m.get("name"), value=v, threshold=t, direction=d, passed=m.get("passed"),
                                relative_margin=rel))
    t = rec.get("trial") or {}
    return dict(job=rec.get("job_name"), model=rec.get("model"), outcome=rec.get("outcome"),
                minutes=rec.get("minutes", t.get("minutes")), cost_usd=rec.get("cost_usd", t.get("cost_usd")),
                solution_runtime_sec=runtime, metric_margins=margins,
                student_final_message=_final_message(trial))


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
    """Put a snapshot back as the current version; the current files move to `failed` first."""
    failed.mkdir(parents=True, exist_ok=True)
    for rel in VERSION_FILES:
        cur = it.p(rel)
        if cur.exists():
            (failed / rel).parent.mkdir(parents=True, exist_ok=True)
            if (failed / rel).exists():
                (shutil.rmtree if (failed / rel).is_dir() else Path.unlink)(failed / rel)
            shutil.move(str(cur), str(failed / rel))
        s = src / rel
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
