"""Prompt templates for the Opus stages (S5 spec, S6A generator+reference, S6C baselines+tests+instruction,
S8 local repair).  Outputs are machine-parsed: S5 -> one JSON object; S6A/S6C/S8 -> multi-file blocks
"=== FILE: <rel path> ===" ... "=== END ===".  Kept short on purpose: the contract (file layout + rules)
matters more than prose.  Version strings go into provenance so prompt changes are traceable.
"""
PROMPT_VERSION = "v2.7-2026-11-11"

METRICS_API = """Verifier helper module `_metrics` (provided, importable from tests/; exact signatures):
  check(name: str, value: float, threshold: float, direction: "<=" | ">=") -> None   # records + asserts
  hidden_sets() -> list[str]          # absolute dirs: subdirectories of /tests/hidden, or [/tests/hidden]
  run_solution(timeout_sec: float, input_dir: str = "/tests/hidden") -> tuple[str, float, int]
      # runs `python /app/output/solution_cli.py <input_dir> <out_dir>` as user nobody with /tests/truth locked;
      # returns (out_dir, wall_seconds, returncode); cached per input_dir.  All calls of one verifier session
      # share a fixed total time cap (the verifier's own timeout): a larger timeout_sec is clamped to what is
      # left, and a call with no time left returns rc -9 / wall_seconds inf like a timeout
Example (PROG):  for d in hidden_sets(): out_dir, sec, rc = run_solution(600, d)"""


SYSTEM = (
    "You design verifiable scientific-computing training tasks for terminal agents, grounded in one research "
    "paper. Every task: public inputs in /app/data, deliverables in /app/output, a hidden verifier with "
    "numeric criteria. Ground truth never appears in public files or objects; no rule-only protection "
    "(the agent is root: 'do not read X' protects nothing). No black-box query budgets with an in-container "
    "oracle. Output exactly the requested format.")

S5_SPEC = """Paper (excerpt, DOI {doi}):
<<<
{excerpt}
>>>
Feasibility card: {probe}

Fixed harness contract (the spec must use exactly these locations; do not invent others):
- public inputs: files under /app/data (the generator writes them to environment/data/); list them in
  "public_files" as paths relative to /app/data.
- ground truth: files under /tests/truth (generator writes tests/truth/); readable only by the verifier.
- PROG: the deliverable program is /app/output/solution_cli.py, run by the verifier as
  `python /app/output/solution_cli.py <input_dir> <output_dir>` on shifted hidden inputs under /tests/hidden
  (generator writes tests/hidden/, laid out exactly like /app/data); the agent also writes its outputs on the
  public data to /app/output.  Every metric computed on hidden data has "on": "hidden" and a name starting
  with "hidden_".
- metrics compare deliverables with the truth files only; the verifier never re-runs a reference solution.
  If a metric needs an oracle quantity (e.g. oracle CRPS), the generator precomputes it from the true
  parameters into tests/truth.
- the data are synthetic (simulated by a deterministic generator from the paper's model); numpy/scipy/pandas/
  scikit-learn/statsmodels only, 2 CPUs, no network, generator < 5 min, reference solution < 15 min.

Task size: a real multi-step piece of scientific work an expert needs {hours_lo}-{hours_hi} hours for (derive,
implement, debug, validate), not a template fill-in or a single library call.  "expert_time_hours" is your honest
estimate for a domain expert starting from the instruction.

Write the task specification as ONE JSON object with keys:
- "title", "domain", "field", "form" (one of {forms}), "expert_time_hours",
- "competencies": [..], "grounding_vs_extension": str,
- "depgraph": {{"deliverables": [...], "nodes": [{{"id","kind":"deliverable|compute|premise|shift",
   "desc", "provision":"given_instruction|given_file|derive|internal|hidden_truth", "key": bool,
   "secret_names": [identifiers that must not appear publicly], "carrier": file or null}}], "edges": [[a,b],..],
   "mid_drop": [one key derive node], "naive_drop": [>=2 nodes],
   "difficulty_knobs": {{"derive_nodes": int, "chain_len": int, "files_crossed": int, "nonstd_tool": bool}}}},
- "metrics": [{{"name", "direction": "<=" or ">=", "on": "public|hidden", "desc"}}]  (3-8 metrics, no thresholds yet),
- "resource_metric_gating": false, or true ONLY when speed / memory itself is the paper's contribution (wall time
  and memory metrics are then asserted at a machine-speed-scaled threshold; otherwise they are recorded only and
  never decide a run, so they cannot be the metric that separates mid or naive),
- "public_files": [..], "hidden_files": [..], "truth_files": [..],
- "hidden_shift": str (PROG only), "why_naive_fails": str, "mid_missing_step": str,
- "expected_separation": {{"<metric name>": "ref passes / mid fails / naive fails", ...}} for every metric: which
  of the three is expected to pass it.
- "method_names": [names of the paper's method and of its published software/package, if any] (used to check that
  the task does not install or import that implementation).
Thresholds are NOT chosen now; they are set from measured reference vs mid results.

Baselines (they decide whether the task survives; most rejected specs failed here):
- mid = the reference minus exactly ONE key derive step (mid_drop), with the simplest sensible default in its
  place; everything else identical to the reference.  It must be a NEAR miss: it should fail only 1-2 of the
  metrics (criteria; a public_X / hidden_X pair counts once) and pass the rest.  Pick a step whose omission
  degrades a specific part of the output, not one that breaks everything (if dropping it makes every metric
  fail, it is too fundamental: choose a more local step).
- naive = the simplest reasonable approach a competent non-expert would try first (e.g. the textbook / default
  library method, no paper-specific steps): drops naive_drop (which includes mid_drop).  It must fail at least 2
  metrics, and the reference must beat it on EVERY metric: a metric on which a correct reference can be worse
  than the naive approach (e.g. a selection score that an all-or-nothing answer maximises) is not a valid metric.
- Choose metrics so that the expected separation holds by construction: ref passes all; mid fails 1-2 criteria
  (the ones its missing step affects); naive fails >= 2.  State it in "expected_separation".
Keep the JSON compact (short descs): code is written in a later step."""

S6A_GEN = """Specification:
{spec}

Write the generator and the reference solution.  Files:
- generator/generate.py: `python generator/generate.py <seed>` run from the task root writes
  environment/data/** (public), tests/truth/** (truth, any format), tests/hidden/** (PROG: shifted hidden inputs
  laid out exactly like /app/data; their truth also goes to tests/truth/).  Deterministic in seed.
  numpy/scipy/pandas/sklearn/statsmodels only, < 5 min, 2 CPUs.  Public files must not contain truth values or
  anything from which they follow trivially (no true-parameter columns, no noise-free copies).
- solution/solve.py: reads /app/data, writes deliverables to /app/output (for PROG also writes
  /app/output/solution_cli.py taking <input_dir> <output_dir>, e.g. a copy of its own source).  Must use only
  public inputs; < 15 min on 2 CPUs.
- baselines/mid.py: the reference minus exactly the mid_drop node (naive default instead);
  baselines/naive.py: also drops the naive_drop nodes.  Same I/O as solve.py (PROG: they write their own
  self-contained solution_cli.py).  Keep the baselines compact.  Each baseline is run ALONE (only its own file
  is mounted): it must not import or read solution/solve.py or any other task file -- copy what it needs.
- tests/measure.py: defines `compute() -> dict` returning {{"<metric name>": float}} for every metric in the spec
  (reads /app/output and /tests/truth).  PROG: execute the agent program ONLY through the provided helper
  `from _metrics import run_solution, hidden_sets`; `for d in hidden_sets(): out_dir, sec, rc = run_solution(t, d)`
  runs `python /app/output/solution_cli.py <d> <out_dir>` as an unprivileged user with /tests/truth locked
  (hidden_sets() = subdirectories of /tests/hidden, or /tests/hidden itself).  Never subprocess/exec agent code
  yourself (the verifier runs as root, so that would expose the truth).
{metrics_api}
  Under
  `if __name__ == "__main__":` prints ONE JSON line of compute().  No thresholds, no asserts.  Missing or
  malformed deliverables must give the worst value (e.g. inf / 0.0), never an exception.
Format: === FILE: <path> === blocks, then === END ===.  Write the files in the order listed."""

S6A_REWRITE = """

A previous implementation of this spec was rejected: on these metrics the reference solution was worse than the
naive baseline by the factors shown ({ratios}), i.e. the reference is wrong (units, normalisation, sign, a
mis-scaled output, or a metric that compares the wrong quantities).  Write ALL files again from scratch and check
the reference against the generator's truth on these metrics before you finish."""

S6C_TESTS = """Specification:
{spec}
Measured metrics (reference / mid / naive, per seed): {measure}
Thresholds fixed from those measurements (do not change them): {thresholds}

Write (generator, solution and mid/naive baselines already exist and are fixed):
- tests/test_outputs.py: pytest; `from measure import compute` and `from _metrics import check`; compute once
  (module-level cache) and call check(name, value, threshold, direction) once per metric with EXACTLY the
  thresholds above (one test function per metric).  Do not re-implement the metrics.
- baselines/exploit_*.py: at least one shortcut attempt (constant/default output, public-data echo, skipping the
  {mid_drop} step entirely).  Each exploit is run ALONE like the baselines (`python baselines/exploit_x.py`,
  no arguments, only its own file mounted: no imports of other task files) and must have the same I/O as
  solve.py: read /app/data, write every deliverable to /app/output (PROG: also a self-contained
  /app/output/solution_cli.py taking <input_dir> <output_dir> that handles every hidden input layout, e.g. a
  single case).  S7 rejects an exploit that crashes or produces no scorable output, since it tests nothing.
- instruction.md: the task statement for an expert agent.  States the scientific problem and the data files
  (columns, units); deliverables and file formats exact (must agree with what tests/measure.py reads and what
  solution/solve.py writes: give each CSV output's header as `Header: `col1,col2,...`` listing exactly the
  reference's columns -- S7 checks it); for PROG
  include verbatim the line `python /app/output/solution_cli.py <input_dir> <output_dir>` (positional args,
  <input_dir> laid out like /app/data) and say what may change on the hidden inputs (qualitatively, no hidden
  parameter values); every metric defined precisely and
  every threshold above with its direction.  Nothing from derive/internal/hidden_truth nodes: do not give the
  method recipe, formulas or parameter values the agent must derive; no "do not read" rules; time budget
  {hours} h.
Format: === FILE: <path> === blocks, then === END ==="""

S8_REPAIR = """The task failed blocking gates.  Gate log:
{gates}
{metrics_api}
Dependency graph (fixed, do not change): {depgraph}
Current files:
{files}
Return ONLY the files you change (same paths; no new files, no deleted criteria, thresholds unchanged unless the
gate log says they were re-derived).  Format: === FILE: <path> === blocks, then === END ==="""


S8_EXPLOIT = """S7 found exploit baselines that did not really run, so they tested nothing (details per seed: solve_rc,
solve_err, valid.why):
{gates}
{metrics_api}
How S7 runs an exploit: `cd /app && python /sol/exploit_x.py` -- no arguments, only that one file mounted at /sol,
public data at /app/data (read-only), deliverables to /app/output.  Then the verifier (tests/measure.py) scores
/app/output; for PROG it calls `python /app/output/solution_cli.py <input_dir> <output_dir>` once per hidden set
(hidden_sets(); a set may hold a single case), so the exploit must also write a self-contained solution_cli.py.
Context (read-only, do not return these): the statement, the scoring script and the reference solution:
{context}
Exploit baselines to fix:
{files}
Rewrite each listed exploit so it runs to the end under exactly that invocation and writes every deliverable in the
right format, while still taking its shortcut (it must stay a cheat that should fail the thresholds, not become a
real solution).  Return ONLY the exploit files you change (same paths).  Format: === FILE: <path> === blocks, then
=== END ==="""


S8_REVIEW = """An independent reviewer audited the packaged task after calibration and asked for a revision.  Blocking issues
come first, minor ones after (fix a minor one only if the same edit fixes it for free):
{issues}
Thresholds (FIXED: never change, loosen or remove any of them; check() calls keep the same metric, threshold and
direction): {thresholds}
Current files (the reference solution is shown read-only; it must not change):
{files}
Fix the blocking issues with the smallest local edit of the statement, the data format notes, or the test /
scoring scripts.  Files you may change: {allowed}.  The data format notes in environment/data/ (README*) are written
by generator/generate.py: edit the text there.  Make the statement, the format notes, measure.py and the reference
agree (when they disagree, follow what the reference solution actually does and what measure.py scores), and state
any convention the reference relies on that a careful solver needs (ranges, boundary handling, units, names).  Never
reveal the reference's algorithm, hidden-truth values or constants it is supposed to derive.
Timing / memory metrics decide a run only when the spec's resource_metric_gating is true (now: {gating}); a timing
issue (should the runtime gate at all?) is fixed by returning the pseudo-file _spec/resource_metric_gating with the
single word true or false (thresholds stay as they are).
Return ONLY the files you change (same paths; no new files).  Format: === FILE: <path> === blocks, then === END ==="""


S6M_MID = """Specification:
{spec}
Reference solution (fixed):
=== FILE: solution/solve.py ===
{solve}
=== END ===
Measured metrics: reference {ref}; current mid baseline (drops {old_drop}) {mid}.
Thresholds sit between the reference and the mid baseline, and the mid baseline should be a NEAR miss: it must
fail only 1-2 of the metrics.  The current mid fails too many ({n_fail} of {n_metrics}): the dropped step is too
fundamental.  Pick ONE other key derive node from the depgraph (not in {tried}) whose omission costs accuracy on
only part of the metrics -- prefer a node of naive_drop {naive_drop}, so the naive baseline (which drops all of
those) stays a superset of the mid baseline -- and write baselines/mid.py = the reference minus exactly that node
(simple default instead), same I/O as solve.py (PROG: it writes its own self-contained /app/output/solution_cli.py).
mid.py is run ALONE (only its own file is mounted): it must not import or read solution/solve.py -- copy the code.
If the node you pick is NOT in naive_drop, also rewrite baselines/naive.py so it drops naive_drop + that node.
First line of your answer: MID_DROP: <node id>.  Then === FILE: baselines/mid.py === ... (and naive.py if needed)
=== END ==="""
