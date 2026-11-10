"""S9R task review: an agent that can run code audits a calibrated task before release.

Runs after S9 for every task S9 would keep (calib.keep_labels) and, as a diagnosis, for tasks whose teacher runs all
failed (review.review_parked).  The reviewer is a Claude Code agent (review.model, default Opus) started in a fresh
context inside the task's own container image (a derived Harbor task built here, under items/<id>/review/r<k>/task):
the whole task package, the extra generation files (baselines, generator), the threshold record, the S6B/S7 records,
and a digest of every calibration run of the current calibration epoch (verifier output + trace digest, student and
teacher).  Its network is that of the teacher: the model endpoint during the agent phase only.

The agent writes /app/review_out/verdict.json:
    {"verdict": "pass" | "revise" | "drop",
     "checklist": {"a_consistency": {"status": "ok"|"issue"|"na", "note": ...}, ... "g_calibration": {...}},
     "issues": [{"id", "category", "severity": "blocking"|"minor", "fix_kind": "local"|"fundamental", "title",
                 "evidence": [...], "fix"}],
     "label_assessment": {"label": ..., "trustworthy": true|false|null, "reason": ...},
     "summary": "..."}
normalize() enforces the contract mechanically: a blocking issue without evidence is downgraded to minor; an issue
marked "leak": true (ground truth / answers readable or copyable, or scoring tamperable) with evidence is blocking
whatever severity the reviewer gave it.  Every blocking issue carries a fix_kind: "local" (fixable by editing the
statement, the format notes, the test / scoring scripts, or an exploit baseline, without loosening a threshold and
without touching the reference) or "fundamental" (the reference itself is wrong, the truth must be regenerated, or
the task's core does not hold).  The final verdict is derived from the blocking issues, not taken from the reviewer:
any "fundamental" blocking issue -> drop; only "local" ones -> revise; none -> pass.  Each blocking issue names its
problem class (PROBLEM_CLASSES); the class, not the reviewer's own fix_kind, decides local vs fundamental.  A
malformed verdict (unknown verdict or problem class, blocking issue without a class, verifier reward != 1) is
invalid and the review run is retried (review.infra_retries), not routed anywhere.

Routing (stages_b.s9r): pass -> S10 (minor issues in the manifest) for kept labels / stays parked for the parked
ones; revise -> S8 review revision (review.max_revisions, default 3; thresholds locked) -> S6A_run/S6B/S6C/S7/PKG ->
S9 -> S9R again; drop -> dropped.  There is no "human" exit: after the revision limit, or when a revised package
keeps failing, the item is dropped.  human stays a valid output only behind review.allow_human (default off), for
installations that want an operator in the loop.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

from . import calib
from .state import read_json

VERDICTS = ("pass", "revise", "drop")
ALLOWED_HUMAN = ("human",)          # accepted only behind review.allow_human (legacy verdicts stay readable)
SEVERITIES = ("blocking", "minor")
FIX_KINDS = ("local", "fundamental")
# what a blocking issue is about (issue["problem"]): local problems a review revision can fix -> revise;
# fundamental ones it cannot -> drop.  The code derives fix_kind from the problem class and the verdict from the
# blocking issues, so a "revise" whose blocking issues are all fundamental is routed as drop.
PROBLEM_CLASSES = {
    "statement": "local",           # wording / missing or contradictory condition in instruction.md
    "format_notes": "local",        # the data format notes (generator README)
    "grader_io": "local",           # the scoring scripts' inputs / outputs (paths, columns, which file is read)
    "exploit_baseline": "local",    # an exploit / cheating baseline is missing or wrong
    "test_gap": "local",            # the tests miss a check (or check something the statement does not ask)
    "timing_gate": "local",         # resource-metric policy / timing margin (thresholds stay locked)
    "reference_wrong": "fundamental",   # the reference solution itself is wrong
    "truth_regen": "fundamental",       # the truth / data must be regenerated
    "core_invalid": "fundamental",      # the task's core does not hold (not separable, ill-posed, ...)
    "calibration": "fundamental",       # the label cannot be trusted and no local edit changes that
}
CHECKLIST = ("a_consistency", "b_missing_conditions", "c_thresholds", "d_stability", "e_shortcuts_leaks",
             "f_network_readymade", "g_calibration", "h_timing_gate")
CATEGORIES = {"a": "consistency", "b": "missing_conditions", "c": "thresholds", "d": "stability",
              "e": "shortcuts_leaks", "f": "network_readymade", "g": "calibration", "h": "timing_gate"}
VERDICT_PATH = "/app/review_out/verdict.json"
# files a review revision (S8) may change: the statement, the data format notes (written by the generator), the
# test / scoring scripts, and the exploit baselines.  Never the reference solution, the baselines' mid / naive,
# _spec/ or thresholds.json.
REVISION_PATHS = ("instruction.md", "tests/test_outputs.py", "tests/measure.py", "generator/generate.py")
REVISION_BASELINES = ("baselines/exploit_",)        # prefix: existing exploit_*.py may be rewritten
# a revision that changes only these is scoring-only: data, thresholds and the reference are untouched, so the
# kept calibration can be re-scored with the new grader instead of being archived (review.rescore)
GRADER_PATHS = ("tests/test_outputs.py", "tests/measure.py")
# run record result types that produced no deliverable at all: no re-scoring can change them (the S7 nop gate
# proves the new grader fails an empty output)
NO_OUTPUT_TYPES = ("timeout_no_output", "no_output")


def revision_path_ok(rel: str) -> bool:
    if rel in REVISION_PATHS:
        return True
    return rel.startswith(REVISION_BASELINES) and ".." not in rel and rel.endswith(".py")


# ------------------------------------------------------------------ config
def review_cfg(cfg: dict) -> dict:
    """A copy of cfg whose calib section routes role "review" (any non-"glm" role reads the teacher_* keys) to the
    review settings: model, agent, environment, allowed hosts, launcher root, launcher, jobs dir, reservation.
    Unset review keys fall back to the teacher route."""
    r = cfg.get("review") or {}
    out = copy.deepcopy(cfg)
    c = out["calib"]
    src = cfg["calib"]
    c["teacher_model"] = r.get("model") or src["teacher_model"]
    c["teacher_agent"] = r.get("agent") or src.get("teacher_agent")
    c["teacher_environment"] = r.get("environment") or calib.role(src, "teacher", "environment")
    hosts = r.get("agent_allowed_hosts")
    c["teacher_agent_allowed_hosts"] = list(hosts) if hosts is not None else \
        list(calib.role(src, "teacher", "agent_allowed_hosts") or [])
    c["teacher_harbor_root"] = r.get("harbor_root") or calib.role(src, "teacher", "harbor_root")
    c["teacher_launcher"] = r.get("launcher") or src.get("teacher_launcher")
    c["teacher_prepared_tasks_dir"] = ""            # the review task is built per round; never a prepared copy
    c["jobs_dir"] = r.get("jobs_dir") or src["jobs_dir"]
    c["est_usd_per_run"] = float(r.get("est_usd_per_run") if r.get("est_usd_per_run") is not None
                                 else src["est_usd_per_run"])
    return out


def reviewable(cfg: dict, label: str | None) -> bool:
    r = cfg.get("review") or {}
    if not r.get("enabled") or not label:
        return False
    return label in cfg["calib"]["keep_labels"] or label in (r.get("review_parked") or [])


# ------------------------------------------------------------------ trace digest
def _short(s, n):
    s = str(s or "")
    return s if len(s) <= n else s[:n] + f" …[+{len(s) - n} chars]"


def trace_digest(trial: Path, max_chars: int = 150_000) -> str:
    """Readable digest of a Claude Code stream-json trace (agent/claude-code.txt): every tool call (truncated),
    every assistant text, the last tool results, and the final result message."""
    f = trial / "agent" / "claude-code.txt"
    if not f.exists():
        return "(no agent/claude-code.txt in this trial)\n"
    calls, results, final = [], [], None
    seen = set()
    with f.open(errors="replace") as fh:
        for line in fh:
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = o.get("type")
            if t == "result":
                final = o
                continue
            m = o.get("message") or {}
            for b in m.get("content") or [] if isinstance(m.get("content"), list) else []:
                if not isinstance(b, dict):
                    continue
                key = (m.get("id"), b.get("id") or b.get("text", "")[:80] or b.get("tool_use_id"))
                if t == "assistant" and key in seen:
                    continue
                seen.add(key)
                if t == "assistant" and b.get("type") == "text" and b.get("text", "").strip():
                    calls.append(("say", _short(b["text"], 1500)))
                elif t == "assistant" and b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    desc = inp.get("command") or inp.get("file_path") or inp.get("pattern") or json.dumps(inp)[:300]
                    if b.get("name") in ("Write", "Edit") and inp.get("content"):
                        desc = f"{inp.get('file_path')} ({len(inp['content'])} chars)"
                    calls.append(("tool", f"{b.get('name')}: {_short(desc, 400)}"))
                elif t == "user" and b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    results.append(_short(c, 600))
    out = [f"# Trace digest of {trial.name}", "", f"{sum(1 for k, _ in calls if k == 'tool')} tool calls, "
           f"{sum(1 for k, _ in calls if k == 'say')} assistant texts.", "", "## Steps (in order)"]
    for k, v in calls:
        out.append(("- [tool] " if k == "tool" else "- [assistant] ") + v.replace("\n", "\n    "))
    out += ["", "## Last tool results (up to 30)"]
    out += [f"- {r}".replace("\n", "\n    ") for r in results[-30:]]
    if final:
        out += ["", "## Final result message",
                f"subtype={final.get('subtype')} is_error={final.get('is_error')} num_turns={final.get('num_turns')} "
                f"api_error_status={final.get('api_error_status')}", "", _short(final.get("result"), 8000)]
    txt = "\n".join(out) + "\n"
    if len(txt) > max_chars:                    # keep the head (approach) and the tail (outcome)
        txt = txt[: max_chars // 2] + "\n\n…[digest truncated]…\n\n" + txt[-max_chars // 2:]
    return txt


def _tail(p: Path, n: int) -> str:
    if not p.exists():
        return ""
    s = p.read_text(errors="replace")
    return s[-n:]


# ------------------------------------------------------------------ bundle / review task
def _calib_records(item_dir: Path) -> list[dict]:
    try:
        return json.loads((item_dir / "calib" / "runs.json").read_text())
    except (OSError, ValueError):
        return []


def write_bundle(item_dir: Path, dest: Path, state: dict, digest_kb: int = 150) -> dict:
    """Everything the reviewer may read, under dest (becomes /review in the container)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    pkg = item_dir / "pkg"
    shutil.copytree(pkg, dest / "pkg", symlinks=False)
    task = item_dir / "task"
    ex = dest / "extra"
    for sub in ("baselines", "generator"):
        if (task / sub).is_dir():
            shutil.copytree(task / sub, ex / sub)
    rec = dest / "records"
    rec.mkdir()
    for name in ("thresholds.json", "S6A_measure.json", "S6B_decisions.json", "PKG_check.json",
                 "thresholds.relock.json"):
        if (item_dir / name).exists():
            shutil.copy2(item_dir / name, rec / name)
    if (item_dir / "_spec" / "spec.json").exists():
        shutil.copy2(item_dir / "_spec" / "spec.json", rec / "spec.json")
    gates = sorted((item_dir / "gates").glob("attempt-*.json")) if (item_dir / "gates").exists() else []
    if gates:
        (rec / "s7_gates").mkdir()
        for g in gates:
            shutil.copy2(g, rec / "s7_gates" / g.name)
    reps = sorted((item_dir / "repair").glob("round-*/review.json")) if (item_dir / "repair").exists() else []
    if reps:
        (rec / "s8_repairs").mkdir()
        for r in reps:
            shutil.copy2(r, rec / "s8_repairs" / f"{r.parent.name}-review.json")
    revs = sorted((item_dir / "repair").glob("review-*")) if (item_dir / "repair").exists() else []
    for r in revs:                       # earlier review revisions: what changed (before / patch) and why
        shutil.copytree(r, rec / "review_revisions" / r.name)
    prev = sorted((item_dir / "review").glob("r*/verdict.json")) if (item_dir / "review").exists() else []
    for p in prev:
        if p.parent.name != f"r{state.get('review_round', 1)}":
            (rec / "previous_reviews").mkdir(exist_ok=True)
            shutil.copy2(p, rec / "previous_reviews" / f"{p.parent.name}-verdict.json")
    hist = [{k: h.get(k) for k in ("stage", "result", "to", "reason") if h.get(k) is not None}
            for h in state.get("history") or []]
    (rec / "pipeline_history.json").write_text(json.dumps(dict(
        item_id=item_dir.name, calib_label=state.get("calib_label"), calib_epoch=state.get("calib_epoch", 0),
        review_round=state.get("review_round", 1), history=hist), indent=1))
    runs = _calib_records(item_dir)
    cal = dest / "calibration"
    cal.mkdir()
    summary = []
    for r in runs:
        t = r.get("trial") or {}
        s = {k: r.get(k) for k in ("who", "n", "model", "job_name", "outcome", "result_type", "minutes",
                                    "cost_usd", "state", "error")}
        s.update(reward=t.get("reward"), exception=t.get("exception"), api_status=t.get("api_status"),
                 delivered=t.get("delivered"))
        if r.get("operator_import"):
            s["environment"] = (r["operator_import"] or {}).get("environment")
        summary.append(s)
        jd = Path(t.get("job_dir") or r.get("job_dir") or "/nonexistent")
        trials = sorted(p for p in jd.iterdir() if (p / "result.json").exists()) if jd.is_dir() else []
        if not trials:
            continue
        d = cal / r["job_name"]
        d.mkdir()
        for tr in trials:
            pre = "" if len(trials) == 1 else f"{tr.name}."
            res = json.loads((tr / "result.json").read_text())
            (d / f"{pre}result_summary.json").write_text(json.dumps(dict(
                trial=tr.name, reward=((res.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
                exception=(res.get("exception_info") or {}).get("exception_type"),
                exception_message=_short((res.get("exception_info") or {}).get("exception_message"), 2000),
                agent_execution=res.get("agent_execution"), verifier=res.get("verifier")), indent=1, default=str))
            (d / f"{pre}verifier_test_stdout.txt").write_text(_tail(tr / "verifier" / "test-stdout.txt", 20000))
            (d / f"{pre}verifier_metrics.jsonl").write_text(_tail(tr / "verifier" / "metrics.jsonl", 20000))
            (d / f"{pre}trace_digest.md").write_text(trace_digest(tr, digest_kb * 1000))
    (cal / "runs.json").write_text(json.dumps(summary, indent=1, default=str))
    return dict(calib_runs=len(runs), digests=sorted(p.name for p in cal.iterdir() if p.is_dir()))


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def build_task(cfg: dict, item_dir: Path, state: dict, rnd: int, out: Path | None = None) -> Path:
    """The derived Harbor task the reviewer runs in: the item's image (pkg/environment) plus /review (bundle),
    /tests and /solution baked in so the reference and the verifier can be run as in calibration."""
    import tomllib
    r = cfg.get("review") or {}
    out = out or item_dir / "review" / f"r{rnd}" / "task"
    if out.exists():
        shutil.rmtree(out)
    pkg = item_dir / "pkg"
    (out / "environment").mkdir(parents=True)
    env_src = pkg / "environment"
    for p in env_src.iterdir():
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, out / "environment" / p.name)
    info = write_bundle(item_dir, out / "environment" / "review", state, int(r.get("trace_digest_kb") or 150))
    df = (out / "environment" / "Dockerfile")
    df.write_text(df.read_text().rstrip() + "\n# S9R review bundle (tbsgen review.build_task)\nCOPY review /review\n"
                  "RUN mkdir -p /tests /solution /app/review_out /logs/verifier && cp -r /review/pkg/tests/. /tests/ && "
                  "cp -r /review/pkg/solution/. /solution/\n")
    t = tomllib.loads((pkg / "task.toml").read_text())
    env = dict(t.get("environment") or {})
    env["network_mode"] = env.get("network_mode") or "public"
    lines = ['schema_version = "1.0"', "", "[task]", f'name = "tbsgen-v2-review/{item_dir.name}-r{rnd}"',
             f'description = {json.dumps("S9R review of " + item_dir.name)}', "", "[metadata]",
             'generator = "tbsgen v2 S9R"', "", "[verifier]", "timeout_sec = 300.0", 'network_mode = "no-network"',
             "", "[agent]", f"timeout_sec = {float(r.get('timeout_sec') or 3600)}",
             "# no internet while the reviewer works except the model endpoint added per run", 
             'network_mode = "allowlist"', "allowed_hosts = []", "", "[environment]"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in env.items()]
    (out / "task.toml").write_text("\n".join(lines) + "\n")
    (out / "instruction.md").write_text(instruction(cfg, item_dir, state, rnd, info))
    (out / "tests").mkdir()
    (out / "tests" / "s9r_validate.py").write_text(VALIDATOR.replace("{problems}", repr(sorted(PROBLEM_CLASSES))))
    (out / "tests" / "test.sh").write_text(TEST_SH)
    (out / "tests" / "test.sh").chmod(0o755)
    (out / "solution").mkdir()
    (out / "solution" / "solve.sh").write_text(ORACLE_SH)
    (out / "solution" / "solve.sh").chmod(0o755)
    return out


TEST_SH = """#!/bin/bash
# S9R review task verifier: keep the reviewer's verdict and check its structure (reward 1 = a well-formed verdict;
# it says nothing about the task under review).
mkdir -p /logs/verifier
cp -f /app/review_out/verdict.json /logs/verifier/verdict.raw.json 2>/dev/null || true
cp -f /app/review_out/report.md /logs/verifier/report.md 2>/dev/null || true
if python3 /tests/s9r_validate.py /app/review_out/verdict.json /logs/verifier/verdict.json; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

ORACLE_SH = """#!/bin/bash
# oracle of the review task (harness smoke test only): a well-formed "pass" verdict
mkdir -p /app/review_out
cat > /app/review_out/verdict.json <<'EOF'
{"verdict": "pass", "issues": [], "checklist": {}, "label_assessment": {"label": null, "trustworthy": null,
 "reason": "oracle smoke test"}, "summary": "oracle smoke test of the S9R harness"}
EOF
"""

VALIDATOR = r'''"Structure check of /app/review_out/verdict.json (S9R).  Writes the parsed verdict to argv[2]."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
try:
    v = json.load(open(src))
except Exception as e:
    print("verdict.json missing or not JSON:", e); sys.exit(1)
errs = []
PROBLEMS = {problems}
if not isinstance(v, dict):
    print("verdict.json is not an object"); sys.exit(1)
if v.get("verdict") not in ("pass", "revise", "drop"):
    errs.append("verdict must be pass|revise|drop")
if not isinstance(v.get("issues", []), list):
    errs.append("issues must be a list")
for i, it in enumerate(v.get("issues") or [] if isinstance(v.get("issues", []), list) else []):
    if not isinstance(it, dict) or it.get("severity") not in ("blocking", "minor"):
        errs.append(f"issues[{i}].severity must be blocking|minor")
    elif it.get("severity") == "blocking" and it.get("problem") not in PROBLEMS:
        errs.append(f"issues[{i}].problem must be one of {sorted(PROBLEMS)} on a blocking issue")
json.dump(v, open(dst, "w"), indent=1)
print("\n".join(errs) or "ok")
sys.exit(1 if errs else 0)
'''


def _fmt_thresholds(item_dir: Path) -> str:
    try:
        th = json.loads((item_dir / "thresholds.json").read_text())["thresholds"]
    except (OSError, ValueError, KeyError):
        return "(thresholds.json missing)"
    rows = []
    for k, v in th.items():
        rest = {a: b for a, b in v.items() if a not in ("direction", "threshold")}
        rows.append(f"- `{k}` {v.get('direction')} {v.get('threshold')}  ({json.dumps(rest, default=str)})")
    return "\n".join(rows) + "\n(ref_worst / mid_best: worst reference and best mid-baseline value over the seeds; " \
        "full measurements in records/S6A_measure.json)"


def instruction(cfg: dict, item_dir: Path, state: dict, rnd: int, info: dict) -> str:
    label = state.get("calib_label")
    runs = _calib_records(item_dir)
    lines = []
    for r in runs:
        t = r.get("trial") or {}
        lines.append(f"- `{r.get('job_name')}` ({r.get('who')}, {r.get('model')}): outcome={r.get('outcome')} "
                     f"result_type={r.get('result_type')} reward={t.get('reward')} exception={t.get('exception')} "
                     f"minutes={None if r.get('minutes') is None else round(r['minutes'], 1)}")
    prev = ""
    if rnd > 1:
        prev = ("\nThis is review round {rnd}: the task was revised after an earlier review (records/previous_reviews/, "
                "records/review_revisions/<n>/{{before,patch}}/, records/thresholds.relock.json) and recalibrated from "
                "scratch.  Check that the revision fixed the earlier blocking issues without loosening the task, and "
                "judge the task as it is now.\n").format(rnd=rnd)
    return REVIEW_PROMPT.format(item=item_dir.name, label=label, rnd=rnd, prev=prev,
                                max_revisions=int((cfg.get("review") or {}).get("max_revisions", 3)),
                                runs="\n".join(lines) or "- (no calibration runs recorded)",
                                thresholds=_fmt_thresholds(item_dir), verdict_path=VERDICT_PATH,
                                digests=", ".join(info.get("digests") or []) or "none")


REVIEW_PROMPT = """# Task review (S9R) — {item}

You are an independent reviewer of a benchmark task that an automatic pipeline generated from a scientific paper.
Agents (a student model and a stronger teacher model) have already attempted it; the calibration label is
**`{label}`**.  Decide whether the task is fit for release, and give evidence.  You did not write this task; do not
assume it is correct, and do not assume it is broken.{prev}

## What you have (read-only copies; work in /app or /tmp)

- `/review/pkg/` — the released task package exactly as agents saw it plus its hidden parts:
  `instruction.md` (the statement agents get), `environment/` (Dockerfile + `data/`, which is `/app/data` here),
  `tests/` (verifier: `test.sh`, `test_outputs.py` with the thresholds, `measure.py` computing the metrics,
  `_metrics.py` helper, `hidden/` inputs, `truth/`), `solution/` (reference solution), `task.toml`.
- The same tests and reference are installed at `/tests` and `/solution`, as in a real run.  To reproduce the
  reference end to end: `bash /solution/solve.sh && bash /tests/test.sh; cat /logs/verifier/reward.txt` (metrics
  in `/logs/verifier/metrics.jsonl`; clear `/app/output` between runs).
- `/review/extra/` — generation-time files not shipped to agents: `baselines/` (mid = reference minus one key
  step, naive, exploit_* attack baselines that must fail) and `generator/generate.py` (makes data, hidden inputs,
  truth, and the data format notes in `environment/data/`).
- `/review/records/` — `thresholds.json` (how each threshold was set from the reference / mid / naive measurements;
  S6B), `S6A_measure.json` (raw measurements), `S6B_decisions.json`, `s7_gates/` (quality gate reports: oracle,
  hidden shift, leak, isolation, determinism, …), `s8_repairs/`, `spec.json` (the task specification), and
  `pipeline_history.json`.
- `/review/calibration/` — every calibration run of the current calibration: `runs.json`, and per job
  ({digests}): `result_summary.json`, `verifier_test_stdout.txt`, `verifier_metrics.jsonl` (each metric's
  value/threshold/pass) and `trace_digest.md` (the agent's tool calls, messages and final message).

Calibration runs:
{runs}

Thresholds:
{thresholds}

You have no internet.  numpy/scipy/pandas (whatever the image has) are available.  Note: this container's CPU may
differ from the machines that ran calibration (the student ran on a cloud sandbox, the teacher on a Xeon server);
when you judge timing, measure the reference here and reason about the margin, not just the absolute value.

## Checklist — give a conclusion with evidence for every item

a. **Consistency.**  Do `instruction.md`, the data format notes in `environment/data/`, `tests/measure.py` /
   `test_outputs.py` and the reference agree on file names, output paths, column names, array shapes, units, formats
   and the CLI contract?  An agent that follows the statement literally must be scored correctly.
b. **Missing conditions.**  Does the statement give every convention needed to hit the thresholds (integration
   ranges, boundary / tail handling, truncation, normalisation, random seeds, tie-breaking, …)?  Is a convention the
   reference relies on, and that changes a metric by more than its threshold allows, left unstated?  Do the tests
   check anything the statement does not ask for?
c. **Thresholds.**  Are thresholds derived from the reference's measured values (records) and reasonable — tight
   enough that the mid / naive / exploit baselines fail, not so tight that an equally valid method fails?  A
   threshold must never be loosened just to let a run pass; if you think one is wrong, say what the evidence is.
d. **Stability.**  Timing gates with a small margin over the reference (measure it), dependence on CPU type, thread
   count, nondeterminism or seeds.
e. **Shortcuts and leaks.**  Can the task be passed without doing the work (hard-coding, reading test files or
   truth, echoing inputs, bypassing the computation)?  Does the statement or data leak the answer or key constants?
   Can the agent program tamper with the scoring (e.g. leave a process running that rewrites
   `/logs/verifier/reward.txt` or `metrics.jsonl`)?
f. **Network / ready-made code.**  Does solving it need the internet or the paper's published implementation?
g. **Calibration evidence.**  Using the traces: did failures come from the task's difficulty, from a task defect
   (ambiguity, contradiction), or from the harness / infrastructure (timeouts from a slow model, crashes, API
   errors)?  Did passing runs really solve the task?  Is the label `{label}` trustworthy?
h. **Timing in the verdict.**  Should the resource metric(s) (wall time / memory) count toward the reward
   (`resource_metric_gating` in the spec: true = they gate, false = they are recorded but never decide a run)?
   They should gate only when speed itself is the paper's contribution; otherwise a slower machine would fail a
   correct solver.  Say which setting this task has and whether it is right.

Run things.  Reproduce the reference score; when you suspect a defect, test it (e.g. follow the statement literally
and see what the verifier says; compare the reference's convention with an alternative and see which metric moves).

## Rules for the verdict

- `pass`: no blocking issue.  Minor issues are fine and are recorded.
- `revise`: at least one **blocking** issue that a local edit of the statement, the format notes, the test /
  scoring scripts or an exploit baseline can fix *without loosening any threshold and without changing the
  reference solution or the data*.  Up to {max_revisions} revision rounds are allowed per task; prefer revise
  whenever a local fix exists.
- `drop`: a blocking issue that such an edit cannot fix.  Any fundamental blocking issue means drop, whatever the
  rest says.
- Every blocking issue carries a `problem` class; the pipeline derives `fix_kind` and the routed verdict from it
  (a `revise` whose blocking issues are all fundamental is routed as drop):
  - local (-> revise): `statement` (wording, a missing or contradictory condition in instruction.md),
    `format_notes` (the data format notes), `grader_io` (what the scoring scripts read / write: paths, columns,
    which file is trusted), `exploit_baseline` (a cheating baseline missing or wrong), `test_gap` (the tests miss
    a check or check something not asked), `timing_gate` (resource-metric policy or timing margin; thresholds
    stay locked);
  - fundamental (-> drop): `reference_wrong` (the reference solution itself is wrong), `truth_regen` (the truth
    or data must be regenerated), `core_invalid` (the task's core does not hold), `calibration` (the label cannot
    be trusted and no local edit changes that).
- **blocking** = would make a correct, careful solver fail (or let a non-solver pass), or makes the label wrong.
- **Any leak of ground truth or answers is blocking, never minor** — on public or hidden data alike: a file, the
  statement, a data note or an object in the image from which the expected answer of *any* scored metric (or a
  value that can be copied to get its score) can be read or reconstructed without doing the work.  "It only
  affects a public metric" does not make it minor: every check counts toward the reward.  The same holds for any
  way to tamper with the verifier's scoring (reward.txt, metrics.jsonl).  Mark such an issue `"leak": true`
  (category e); a leak issue with evidence is always counted as blocking.
  Every blocking issue needs concrete evidence: file + line numbers, a command you ran and its output, or a quoted
  trace excerpt.  A suspicion without such evidence is **minor**.  Do not inflate severity.

## Output (required)

Write `{verdict_path}` (JSON, exactly these keys) and optionally `/app/review_out/report.md` (your notes):

```json
{{
  "verdict": "pass | revise | drop",
  "checklist": {{
    "a_consistency": {{"status": "ok | issue | na", "note": "..."}},
    "b_missing_conditions": {{"status": "...", "note": "..."}},
    "c_thresholds": {{"status": "...", "note": "..."}},
    "d_stability": {{"status": "...", "note": "..."}},
    "e_shortcuts_leaks": {{"status": "...", "note": "..."}},
    "f_network_readymade": {{"status": "...", "note": "..."}},
    "g_calibration": {{"status": "...", "note": "..."}},
    "h_timing_gate": {{"status": "...", "note": "..."}}
  }},
  "issues": [
    {{"id": "I1", "category": "a|b|c|d|e|f|g|h", "severity": "blocking | minor",
      "problem": "statement | format_notes | grader_io | exploit_baseline | test_gap | timing_gate | reference_wrong | truth_regen | core_invalid | calibration",
      "leak": false, "title": "...",
      "evidence": ["instruction.md:53 says ... ; tests/measure.py:42 reads ...", "ran X -> output Y"],
      "fix": "the smallest edit that fixes it (which file, what change)"}}
  ],
  "label_assessment": {{"label": "{label}", "trustworthy": true, "reason": "..."}},
  "reference_check": {{"reward": 1.0, "notes": "your reproduction of the reference, incl. timing margins"}},
  "summary": "two or three sentences"
}}
```

Finish by writing the file; the run ends when you stop.  Do not modify anything under /review.
"""


# ------------------------------------------------------------------ verdict
def _has_evidence(ev) -> bool:
    if isinstance(ev, str):
        return len(ev.strip()) >= 8
    if isinstance(ev, list):
        return any(_has_evidence(x) for x in ev)
    if isinstance(ev, dict):
        return any(_has_evidence(v) for v in ev.values())
    return False


def normalize(raw: dict | None, reward: float | None = None, allow_human: bool = False) -> dict:
    """Apply the contract to a raw verdict: returns {verdict, raw_verdict, issues, blocking, minor, downgraded,
    label_assessment, checklist, problems}.  verdict None = invalid (structural problem: unknown verdict or
    severity, non-object issue, blocking issue without a fix_kind, verifier reward != 1 when known) — the caller
    retries the review run; it is not a routing outcome.  The routed verdict is derived from the blocking issues:
    any fundamental one -> drop, otherwise any blocking one -> revise, none -> pass.  The only silent rewrites are
    the contract's own rules (a blocking issue without evidence counts as minor; a leak with evidence is
    blocking); a raw verdict that disagrees with its issues is recorded in problems but does not make the verdict
    invalid, the derived verdict wins."""
    problems = []
    if not isinstance(raw, dict):
        return dict(verdict=None, raw_verdict=None, issues=[], blocking=[], minor=[], downgraded=[], escalated=[],
                    label_assessment={}, checklist={}, summary="", problems=["no verdict"])
    rv = raw.get("verdict")
    issues, downgraded, escalated, hard = [], [], [], []      # hard: structural problems -> invalid verdict
    if not isinstance(raw.get("issues", []), list):
        problems.append("issues is not a list")
        hard.append("issues")
    for i, it in enumerate(raw.get("issues") or [] if isinstance(raw.get("issues"), list) else []):
        if not isinstance(it, dict):
            problems.append(f"issues[{i}] not an object")
            hard.append(f"issues[{i}]")
            continue
        it = dict(it)
        it.setdefault("id", f"I{i + 1}")
        sev = it.get("severity")
        if sev not in SEVERITIES:
            problems.append(f"{it['id']}: severity {sev!r} not blocking|minor")
            hard.append(it["id"])
            it["severity"] = "minor"
        elif sev == "blocking" and not _has_evidence(it.get("evidence")):
            it["severity"] = "minor"
            it["downgraded"] = "blocking without evidence"
            downgraded.append(it["id"])
        if it.get("leak") is True and it["severity"] != "blocking" and _has_evidence(it.get("evidence")) and \
                sev in SEVERITIES:
            # a leak of ground truth / answers (any metric, public or hidden) or a way to tamper with the scoring
            # is blocking by contract (S9R validation of e2e-prod-20261109: a public-truth leak was rated minor)
            it["severity"] = "blocking"
            it["escalated"] = "leak is blocking"
            escalated.append(it["id"])
        if it["severity"] == "blocking":
            pc = it.get("problem")
            fk = it.get("fix_kind")
            if pc in PROBLEM_CLASSES:
                if fk is not None and fk != PROBLEM_CLASSES[pc]:
                    problems.append(f"{it['id']}: fix_kind {fk!r} contradicts problem {pc!r} -> "
                                    f"{PROBLEM_CLASSES[pc]}")
                it["fix_kind"] = PROBLEM_CLASSES[pc]       # the class decides fixability, not the label
            elif pc is not None:
                problems.append(f"{it['id']}: problem {pc!r} not one of {sorted(PROBLEM_CLASSES)}")
                hard.append(it["id"])
            elif fk not in FIX_KINDS:
                problems.append(f"{it['id']}: blocking issue without a problem class (or fix_kind)")
                hard.append(it["id"])
        issues.append(it)
    if reward is not None and reward != 1:
        problems.append(f"review verifier reward {reward} (verdict.json failed the structure check)")
        hard.append("reward")
    blocking = [i for i in issues if i["severity"] == "blocking"]
    if rv not in VERDICTS and not (allow_human and rv in ALLOWED_HUMAN):
        problems.append(f"verdict {rv!r} not one of {VERDICTS}")
        v = None
    elif hard:
        v = None
    elif allow_human and rv in ALLOWED_HUMAN:
        v = "human"
    elif rv in ("revise", "drop") and not blocking:
        # the reviewer judged a defect but no blocking issue survives the contract (none, or all without
        # evidence): never turned into a pass by the code -> invalid, the review runs again
        problems.append(f"raw verdict {rv!r} without an evidenced blocking issue")
        v = None
    else:
        # the verdict follows from the blocking issues, not from the reviewer's label
        v = "drop" if any(i.get("fix_kind") == "fundamental" for i in blocking) else \
            ("revise" if blocking else "pass")
        if v != rv:
            problems.append(f"raw verdict {rv!r} overridden by the blocking issues' fix_kinds -> {v}")
    return dict(verdict=v, raw_verdict=rv, issues=issues, blocking=[i["id"] for i in blocking],
                minor=[i["id"] for i in issues if i["severity"] == "minor"], downgraded=downgraded,
                escalated=escalated, label_assessment=raw.get("label_assessment") or {},
                checklist=raw.get("checklist") or {}, reference_check=raw.get("reference_check") or {},
                summary=str(raw.get("summary") or "")[:2000], problems=problems)


def read_verdict(trial: Path) -> dict | None:
    for name in ("verdict.json", "verdict.raw.json"):
        p = trial / "verifier" / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except ValueError:
                continue
    return None


def classify(job_dir: Path, model: str | None, prices: dict, allow_human: bool = False) -> dict:
    """A finished review job: outcome "done" (a verdict file came back; verdict may still be None = invalid,
    retried by routing) or "infra"; cost as for calibration."""
    trials = calib.collect(job_dir, model, prices)
    usd, kind, usage = calib.job_cost(trials)
    raw, trial = None, None
    for t in sorted(p for p in Path(job_dir).iterdir() if (p / "result.json").exists()):
        v = read_verdict(t)
        if v is not None:
            raw, trial = v, t
    last = trials[-1] if trials else {}
    reward = None
    if trial is not None:
        rw = ((json.loads((trial / "result.json").read_text()).get("verifier_result") or {}).get("rewards") or {})
        reward = rw.get("reward")
        reward = float(reward) if isinstance(reward, (int, float)) else 0.0
    return dict(outcome="done" if raw is not None else "infra", verdict_raw=raw,
                verdict=normalize(raw, reward, allow_human) if raw is not None else None,
                trial=dict(last, job_dir=str(Path(job_dir).resolve()),
                           verdict_trial=trial.name if trial else None),
                minutes=last.get("minutes"), cost_usd=usd, cost_kind=kind, usage=usage, n_trials=len(trials))


# ------------------------------------------------------------------ run records
def route_problem(cfg: dict) -> str | None:
    """A review route that inherits a teacher environment running prepared task copies cannot run a review task
    (built per round from the Dockerfile, never prepared)."""
    src = cfg["calib"]
    if not (cfg.get("review") or {}).get("environment") and calib.role(src, "teacher", "prepared_tasks_dir"):
        return ("the review route would inherit the teacher's environment, which runs prepared task copies "
                "(calib.teacher_prepared_tasks_dir / prepared_tasks_dir); a review task is built per round and has "
                "none: set review.environment / review.launcher / review.harbor_root to a route that builds the "
                "Dockerfile")
    return None


def job_name(item_id: str, rnd: int, attempt: int) -> str:
    return f"tbsgen-{item_id}-review{rnd}" + (f"-a{attempt}" if attempt > 1 else "")


def prepare(cfg: dict, item_dir: Path, task: Path, rnd: int, attempt: int, epoch: int = 0) -> dict:
    """Intent record for one review run (job config written next to the review task)."""
    rc = review_cfg(cfg)
    c = rc["calib"]
    name = job_name(item_dir.name, rnd, attempt)
    conf = calib.job_config(rc, task, "review", name)
    p = item_dir / "review" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(conf, indent=1))
    p = p.resolve()
    root, launcher = c.get("teacher_harbor_root"), c.get("teacher_launcher")
    command = f"cd {root} && ./{launcher} run {p}" if launcher and root else f"harbor run --config {p} --yes"
    rec = dict(who="review", n=rnd, attempt=attempt, item=item_dir.name, model=c["teacher_model"], job_name=name,
               config=str(p), command=command, state="intent", submitted=False, ts=time.time(),
               job_dir=str(Path(c["jobs_dir"]) / name), task=str(task.resolve()), outcome=None,
               ledger_stage="S9R", pkg_sha=calib.pkg_sha(item_dir / "pkg"), calib_epoch=epoch)
    why = calib.network_problem(rc, task, conf) or route_problem(cfg)
    if why:
        rec["blocked" if c.get("allow_submit") else "warning"] = why
    if c.get("allow_submit") and not (launcher and root):
        rec["blocked"] = "calib.allow_submit needs review.launcher/harbor_root (or the teacher's)"
    return rec


# ------------------------------------------------------------------ S8 review revision
def check_calls(src: str) -> list[tuple] | None:
    """(metric, threshold, direction) of every check(...) call in test_outputs.py, each a literal; None when any
    check() argument for name / threshold / direction is not a literal (cannot be proven unchanged) or the source
    does not parse."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "check":
            if len(n.args) < 4:
                return None
            vals = []
            for a in (n.args[0], n.args[2], n.args[3]):
                try:
                    vals.append(ast.literal_eval(a))
                except ValueError:
                    return None
            name, thr, d = vals
            if not isinstance(name, str) or not isinstance(thr, (int, float)) or d not in ("<=", ">="):
                return None
            out.append((name, float(thr), d))
    return sorted(out)


def rebinds_check(src: str) -> bool:
    """True unless the only binding of the name `check` (and the only use of `_metrics`) is the module-level
    `from _metrics import check`: any other binding — def / class / assignment / parameter / lambda arg / import /
    alias / global, attribute writes on _metrics — could rescale thresholds at run time while the call literals stay
    the same."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True
    ok_import = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if (a.asname or a.name) == "check" or a.name == "check" or a.name == "*":
                    if n.module == "_metrics" and a.name == "check" and a.asname in (None, "check") and \
                            n in tree.body:
                        ok_import += 1
                    else:
                        return True
            if n.module == "_metrics" and any(a.name != "check" for a in n.names):
                return True
        elif isinstance(n, ast.Import):
            if any((a.asname or a.name).split(".")[0] in ("check", "_metrics") or a.name == "_metrics" for a in n.names):
                return True
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == "check":
            return True
        elif isinstance(n, ast.arg) and n.arg == "check":
            return True
        elif isinstance(n, ast.Name) and n.id in ("check", "_metrics") and not isinstance(n.ctx, ast.Load):
            return True
        elif isinstance(n, ast.Name) and n.id == "_metrics":
            return True
        elif isinstance(n, (ast.Global, ast.Nonlocal)) and "check" in n.names:
            return True
        elif isinstance(n, ast.alias) and n.asname == "check" and n.name != "check":
            return True
        elif isinstance(n, ast.Attribute) and n.attr == "check" and not isinstance(n.ctx, ast.Load):
            return True
        elif isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("setattr", "globals", "exec", "eval",
                                                                         "__import__", "vars", "locals"):
            return True
    return ok_import != 1


def runtime_thresholds(metrics_jsonl: Path) -> list[tuple] | None:
    """(metric, threshold, direction) as the verifier's check() recorded them at run time (non-aux lines)."""
    try:
        rows = [json.loads(x) for x in metrics_jsonl.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return None
    return sorted({(r["name"], float(r["threshold"]), r["direction"]) for r in rows
                   if not r.get("aux") and "threshold" in r})


def pkg_data_hash(pkg: Path) -> str | None:
    """Hash of the packaged data, hidden inputs and truth (environment/data, tests/hidden, tests/truth) without
    README* format notes: a review revision may rewrite the notes, never the data."""
    import hashlib
    if not pkg.is_dir():
        return None
    h = hashlib.sha256()
    for sub in ("environment/data", "tests/hidden", "tests/truth"):
        root = pkg / sub
        for p in sorted(root.rglob("*")) if root.exists() else []:
            if p.is_file() and not p.name.startswith("README"):
                h.update(str(p.relative_to(pkg)).encode())
                h.update(b"\0")
                h.update(p.read_bytes())
                h.update(b"\0")
    return h.hexdigest()[:16]


def locked_signature(thresholds: dict) -> list[tuple]:
    return sorted((k, float(v["threshold"]), v["direction"]) for k, v in thresholds.items())


def revision_review(task: Path, patch: dict[str, str], depgraph: dict, thresholds: dict | None = None) -> dict:
    """Mechanical preservation review of a review-driven revision.  Paths are checked first (nothing is written for
    a patch with a bad path); then the S8 rules (repair.review); then: only REVISION_PATHS may change, and the
    check() calls of the revised test_outputs.py must be literal and equal to the locked thresholds.json, metric by
    metric (threshold value and direction)."""
    from . import repair
    reasons = []
    for rel in patch:
        if not revision_path_ok(rel):
            reasons.append(f"path not allowed in a review revision: {rel!r}")
    if reasons:
        return dict(ok=False, reasons=reasons, files=sorted(patch), method="review-revision-v2")
    rv = repair.review(task, patch, depgraph)
    reasons = list(rv["reasons"])
    t = "tests/test_outputs.py"
    if t in patch and rebinds_check(patch[t]):
        reasons.append("test_outputs.py defines, wraps or re-imports check (only `from _metrics import check`)")
    if t in patch:
        sig = check_calls(patch[t])
        if sig is None:
            reasons.append("test_outputs.py check() arguments are not all literals (thresholds cannot be verified)")
        elif thresholds is not None and sig != locked_signature(thresholds):
            reasons.append("test_outputs.py check() metrics/thresholds/directions differ from the locked thresholds")
        elif thresholds is None and (task / t).exists() and sig != check_calls((task / t).read_text()):
            reasons.append("test_outputs.py check() metrics/thresholds/directions changed")
    return dict(rv, ok=not reasons, reasons=reasons, method="review-revision-v2")


def files_for_revision(item_dir: Path, max_chars: int = 90000) -> str:
    task = item_dir / "task"
    want = ["instruction.md", "tests/test_outputs.py", "tests/measure.py", "generator/generate.py",
            "solution/solve.py"]
    out, n = [], 0
    for rel in want:
        p = task / rel
        if p.exists():
            body = p.read_text()
            tag = " (read-only: do not change)" if rel == "solution/solve.py" else ""
            out.append(f"=== FILE: {rel}{tag} ===\n{body}")
            n += len(body)
            if n > max_chars:
                break
    pd = item_dir / "pkg" / "environment" / "data"
    for p in sorted(pd.glob("README*")) if pd.exists() else []:
        out.append(f"=== FILE (generated by generator/generate.py, read-only copy): environment/data/{p.name} ===\n"
                   + p.read_text()[:12000])
    return "\n".join(out)


def lock_compare(locked: dict, fresh: dict) -> dict:
    """Per metric: locked vs freshly derived threshold, and whether the fresh one would be looser."""
    out = {}
    for k, v in (locked or {}).items():
        f = (fresh or {}).get(k) or {}
        a, b, d = v.get("threshold"), f.get("threshold"), v.get("direction")
        looser = None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            looser = (b > a) if d == "<=" else (b < a)
        out[k] = dict(locked=a, fresh=b, direction=d, fresh_would_be_looser=looser)
    return out

# ------------------------------------------------------- scoring-only revision / re-score
def scoring_only(patch: dict) -> bool:
    """True when a review revision (or a shared-verifier regate) changed only the grader: the statement, the data,
    the thresholds and the reference solution are untouched, so the kept calibration can be re-scored with the
    new grader instead of being archived (the deliverables do not depend on the grader)."""
    return bool(patch) and all(rel in GRADER_PATHS for rel in patch)


def save_deliverable(item_dir: Path, job_name: str, src: Path, note: str = "") -> Path:
    """Register a run's saved deliverable (its /app/output, e.g. re-played from the agent trace by the operator)
    for re-scoring: items/<id>/calib/deliverables/<job_name>/ + provenance.json."""
    d = item_dir / "calib" / "deliverables" / job_name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d, symlinks=False)
    (d / ".." / f"{job_name}.provenance.json").write_text(json.dumps(
        dict(job_name=job_name, src=str(Path(src).resolve()), note=note[:500], ts=time.time()), indent=1))
    return d


def rescore_run(cfg: dict, item_dir: Path, run: dict) -> dict:
    """Run the packaged verifier (pkg/tests, the new grader) on a run's saved deliverable in the local sandbox.
    -> {ok, reward, passed} — ok=False when the sandbox could not run (infrastructure, not a label change)."""
    from . import runner
    pkg = item_dir / "pkg"
    d = item_dir / "calib" / "deliverables" / run["job_name"]
    if not d.is_dir() or not (pkg / "tests" / "test.sh").exists():
        return dict(ok=False, why="no saved deliverable", reward=None, passed=None)
    work = item_dir / "runs" / f"rescore-{run['job_name']}"
    tests = work / "tests"
    if tests.exists():
        shutil.rmtree(tests)
    shutil.copytree(pkg / "tests", tests)
    logs = work / "logs"
    if logs.exists():
        shutil.rmtree(logs)
    logs.mkdir(parents=True)
    logs.chmod(0o777)
    settle = float(cfg["gen"].get("verifier_settle_sec", 1.0))
    from .spec import is_resource_metric
    spec = runner.task_spec(item_dir / "task") or {}
    budget = runner.timing_budget(cfg, spec)
    r = runner.sandbox.run(cfg, [(d, "/src/output", "ro"), (tests, "/src/tests", "ro"),
                                 (logs, "/logs/verifier", "rw"),
                                 (pkg / "environment" / "data", "/src/data", "ro")],
                           runner.VERIFIER_PRE + f"cd /tests && bash /tests/test.sh; rc=$?; sleep {settle:g}; exit $rc",
                           int(cfg["gen"]["test_timeout"] + budget), f"tbsgen-rescore-{run['job_name']}", root=True)
    rf = logs / "reward.txt"
    reward = rf.read_text().strip() if rf.exists() and not rf.is_symlink() else None
    if reward is None:                  # the verifier never finished (sandbox / docker trouble): not a re-score
        return dict(ok=False, why=f"verifier wrote no reward (rc {r.get('rc')})", reward=None, passed=None,
                    rc=r.get("rc"), err=(r.get("err") or "")[-300:])
    return dict(ok=True, reward=reward, passed=reward == "1", rc=r.get("rc"), err=(r.get("err") or "")[-300:])


def rescore_item(cfg: dict, item_dir: Path, runs: list) -> dict:
    """Re-score every decisive run of the kept calibration with the new grader.  A run without a saved
    deliverable keeps its outcome only when it produced no deliverable at all (NO_OUTPUT_TYPES — the S7 nop gate
    proves the new grader fails an empty output).  -> {keep, checked, missing, changed, details}"""
    checked, missing, changed, details = 0, [], [], []
    for r in runs or []:
        if r.get("outcome") not in ("pass", "fail"):
            continue                                    # infra / unsettled runs never decided the label
        was_pass = r.get("outcome") == "pass"
        d = item_dir / "calib" / "deliverables" / r.get("job_name", "")
        if not d.is_dir():
            if not was_pass and r.get("result_type") in NO_OUTPUT_TYPES:
                details.append(dict(job=r.get("job_name"), kept="no deliverable was produced", outcome=r.get("outcome")))
                continue
            missing.append(r.get("job_name"))
            continue
        res = rescore_run(cfg, item_dir, r)
        checked += 1
        if not res.get("ok"):
            missing.append(r.get("job_name"))
            details.append(dict(job=r.get("job_name"), error=res.get("why")))
            continue
        if res["passed"] != was_pass:
            changed.append(dict(job=r.get("job_name"), was=was_pass, now=res["passed"], reward=res.get("reward")))
        else:
            details.append(dict(job=r.get("job_name"), outcome=r.get("outcome"), reward=res.get("reward")))
    return dict(keep=not missing and not changed, checked=checked, missing=missing, changed=changed, details=details)


# ------------------------------------------------------------------ batch summary
def summary(P, sample_n: int = 3, seed: int = 0) -> dict:
    """`tbsgen review summary`: this batch's review outcome.  Verdicts are the routed (derived) verdicts; alerts
    flag anything an operator should look at (drop rate over review.summary_alerts.drop_rate, verdicts of the
    same package that disagree, invalid verdicts, review:infra parks, label changes)."""
    import random as _random
    cfg = P.cfg
    alert_cfg = (cfg.get("review") or {}).get("summary_alerts") or {}
    per_item, verdict_counts, fix_kinds, drop_reasons, rounds, categories = {}, {}, {}, {}, {}, {}
    invalid = contradictory = 0
    for it in P.store.items():
        st = it.state
        rd = it.p("review")
        if not rd.exists():
            continue
        verdicts = sorted(rd.glob("r*/*.verdict.json"))
        if not verdicts:
            continue
        v = read_json(it.p("review", "verdict.json"), None)
        nv = (v or {}).get("_normalized") or {}
        verdict = nv.get("verdict") or v.get("verdict") if v else None
        per_item[it.item_id] = dict(stage=st.get("stage"), status=st.get("status"), label=st.get("calib_label"),
                                    reason=st.get("reason"), verdict=verdict,
                                    rounds=st.get("review_round", 0), revisions=st.get("review_revisions", 0))
        verdict_counts[verdict or "none"] = verdict_counts.get(verdict or "none", 0) + 1
        n = max(1, int(st.get("review_round") or 1))
        rounds[n] = rounds.get(n, 0) + 1
        n_inv = sum(1 for x in read_json(it.p("review", "runs.json"), []) or [] if x.get("outcome") == "invalid")
        per_item[it.item_id]["invalid_verdicts"] = n_inv
        if n_inv:
            invalid += 1
        # same package, disagreeing routed verdicts (two jobs of one round)
        by_pkg = {}
        for p in verdicts:
            vv = read_json(p, {}) or {}
            by_pkg.setdefault(vv.get("pkg_sha"), []).append((vv.get("_normalized") or {}).get("verdict"))
        for sha, vs in by_pkg.items():
            if len(set(vs)) > 1:
                contradictory += 1
        for i in nv.get("issues") or []:
            if i.get("severity") == "blocking":
                k = i.get("fix_kind") or "unknown"
                fix_kinds[k] = fix_kinds.get(k, 0) + 1
                cat = CATEGORIES.get(str(i.get("category") or "")[:1], "other")
                categories[cat] = categories.get(cat, 0) + 1
        if st.get("status") == "dropped" and str(st.get("reason") or "").startswith("review:"):
            drop_reasons[str(st.get("reason"))[:120]] = drop_reasons.get(str(st.get("reason"))[:120], 0) + 1
    reviewed = len(per_item)
    dropped = sum(1 for x in per_item.values() if x["status"] == "dropped")
    rate = (dropped / reviewed) if reviewed else 0.0
    alerts = []
    if reviewed and rate > float(alert_cfg.get("drop_rate", 0.5)):
        alerts.append(f"drop rate {dropped}/{reviewed} ({rate:.0%}) over review.summary_alerts.drop_rate")
    if contradictory:
        alerts.append(f"{contradictory} item(s) with disagreeing verdicts for the same package")
    if invalid:
        alerts.append(f"{invalid} item(s) with invalid (retried) verdicts")
    infra = sum(1 for x in per_item.values() if str(x["reason"] or "").startswith("review:infra"))
    if infra:
        alerts.append(f"{infra} item(s) parked review:infra (review infrastructure)")
    passes = [i for i, x in per_item.items() if x["verdict"] == "pass"]
    sample = sorted(_random.Random(seed).sample(passes, min(sample_n, len(passes)))) if passes else []
    # blocklist auto-recheck decisions (blocklist-trials.json + events)
    from . import blocklist as _bl
    bst = _bl.load_state(P.store)
    bl = {name: dict(lifted=bool(r.get("lifted")), lifted_rate=r.get("lifted_rate"),
                     trials={o: sum(1 for t in r.get("trials") or [] if t.get("outcome") == o)
                             for o in ("ok", "fail", "pending", "void")},
                     last_kept=(r.get("kept") or [None])[-1])
          for name, r in (bst.get("rules") or {}).items()}
    bl_events = [e for e in P.store.events() if e.get("kind") in ("blocklist_lifted", "blocklist_kept")][-20:]
    return dict(items=reviewed, verdicts=verdict_counts, drop_rate=round(rate, 4), alerts=alerts,
                blocking_fix_kinds=fix_kinds, blocking_categories=categories, drop_reasons=drop_reasons,
                rounds_histogram=rounds, blocklist=bl, blocklist_events=bl_events,
                invalid=invalid, contradictory=contradictory, review_infra_parked=infra,
                pass_sample=[dict(item=i, verdict_path=str(P.store.root / "items" / i / "review" / "verdict.json"))
                             for i in sample], per_item=per_item)

