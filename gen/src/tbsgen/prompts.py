"""Prompt templates for the Opus stages (S5 spec, S6A generator+reference, S6C baselines+tests+instruction,
S8 local repair).  Outputs are machine-parsed: S5 -> one JSON object; S6A/S6C/S8 -> multi-file blocks
"=== FILE: <rel path> ===" ... "=== END ===".  Kept short on purpose: the contract (file layout + rules)
matters more than prose.  Version strings go into provenance so prompt changes are traceable.
"""
PROMPT_VERSION = "v2.1-2026-11-06"

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
- "public_files": [..], "hidden_files": [..], "truth_files": [..],
- "hidden_shift": str (PROG only), "why_naive_fails": str, "mid_missing_step": str.
Thresholds are NOT chosen now; they are set from measured reference vs mid results.  Keep the JSON compact
(short descs): code is written in a later step."""

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
  self-contained solution_cli.py).  Keep the baselines compact.
- tests/measure.py: defines `compute() -> dict` returning {{"<metric name>": float}} for every metric in the spec
  (reads /app/output and /tests/truth).  PROG: execute the agent program ONLY through the provided helper
  `from _metrics import run_solution, hidden_sets`; `for d in hidden_sets(): out_dir, sec, rc = run_solution(t, d)`
  runs `python /app/output/solution_cli.py <d> <out_dir>` as an unprivileged user with /tests/truth locked
  (hidden_sets() = subdirectories of /tests/hidden, or /tests/hidden itself).  Never subprocess/exec agent code
  yourself (the verifier runs as root, so that would expose the truth).  Under
  `if __name__ == "__main__":` prints ONE JSON line of compute().  No thresholds, no asserts.  Missing or
  malformed deliverables must give the worst value (e.g. inf / 0.0), never an exception.
Format: === FILE: <path> === blocks, then === END ===.  Write the files in the order listed."""

S6C_TESTS = """Specification:
{spec}
Measured metrics (reference / mid / naive, per seed): {measure}
Thresholds fixed from those measurements (do not change them): {thresholds}

Write (generator, solution and mid/naive baselines already exist and are fixed):
- tests/test_outputs.py: pytest; `from measure import compute` and `from _metrics import check`; compute once
  (module-level cache) and call check(name, value, threshold, direction) once per metric with EXACTLY the
  thresholds above (one test function per metric).  Do not re-implement the metrics.
- baselines/exploit_*.py: at least one shortcut attempt (constant/default output, public-data echo, skipping the
  {mid_drop} step entirely).
- instruction.md: the task statement for an expert agent.  States the scientific problem and the data files
  (columns, units); deliverables and file formats exact (must agree with what tests/measure.py reads); for PROG
  include verbatim the line `python /app/output/solution_cli.py <input_dir> <output_dir>` (positional args,
  <input_dir> laid out like /app/data) and say what may change on the hidden inputs (qualitatively, no hidden
  parameter values); every metric defined precisely and
  every threshold above with its direction.  Nothing from derive/internal/hidden_truth nodes: do not give the
  method recipe, formulas or parameter values the agent must derive; no "do not read" rules; time budget
  {hours} h.
Format: === FILE: <path> === blocks, then === END ==="""

S8_REPAIR = """The task failed blocking gates.  Gate log:
{gates}
Dependency graph (fixed, do not change): {depgraph}
Current files:
{files}
Return ONLY the files you change (same paths; no new files, no deleted criteria, thresholds unchanged unless the
gate log says they were re-derived).  Format: === FILE: <path> === blocks, then === END ==="""
