"""Prompt templates for the Opus stages (S5 spec, S6A generator+reference, S6C baselines+tests+instruction,
S8 local repair).  Outputs are machine-parsed: S5 -> one JSON object; S6A/S6C/S8 -> multi-file blocks
"=== FILE: <rel path> ===" ... "=== END ===".  Kept short on purpose: the contract (file layout + rules)
matters more than prose.  Version strings go into provenance so prompt changes are traceable.
"""
PROMPT_VERSION = "v2.0-2026-11-05"

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

Write the task specification as ONE JSON object with keys:
- "title", "domain", "field", "form" (one of {forms}), "expert_time_hours" (0.5-4),
- "competencies": [..], "grounding_vs_extension": str,
- "depgraph": {{"deliverables": [...], "nodes": [{{"id","kind":"deliverable|compute|premise|shift",
   "desc", "provision":"given_instruction|given_file|derive|internal|hidden_truth", "key": bool,
   "secret_names": [identifiers that must not appear publicly], "carrier": file or null}}], "edges": [[a,b],..],
   "mid_drop": [one key derive node], "naive_drop": [>=2 nodes],
   "difficulty_knobs": {{"derive_nodes": int, "chain_len": int, "files_crossed": int, "nonstd_tool": bool}}}},
- "metrics": [{{"name", "direction": "<=" or ">=", "on": "public|hidden", "desc"}}]  (3-8 metrics, no thresholds yet),
- "public_files": [..], "hidden_files": [..], "truth_files": [..],
- "hidden_shift": str (PROG only), "why_naive_fails": str, "mid_missing_step": str.
Thresholds are NOT chosen now; they are set from measured reference vs mid results."""

S6A_GEN = """Specification:
{spec}

Write the generator and the reference solution.  Files:
- generator/generate.py: `python generator/generate.py <seed>` run from the task root writes
  environment/data/** (public), tests/truth/** (truth, any format), tests/hidden/** (PROG: shifted hidden inputs
  laid out like /app/data).  Deterministic in seed.  numpy/scipy/pandas/sklearn/statsmodels only, < 5 min, 2 CPUs.
- solution/solve.py: reads /app/data, writes deliverables to /app/output (for PROG also /app/output/solution_cli.py
  taking <input_dir> <output_dir>).  Must use only public inputs.
- baselines/mid.py: the reference minus exactly the mid_drop node (naive default instead);
  baselines/naive.py: also drops the naive_drop nodes.  Same I/O as solve.py.
- tests/measure.py: `python /tests/measure.py` reads /app/output and /tests/truth (+ /tests/hidden via
  `from _metrics import run_solution`) and prints ONE JSON line {{"<metric name>": value, ...}} for every metric in the
  spec (no thresholds, no asserts).
Format: === FILE: <path> === blocks, then === END ==="""

S6C_TESTS = """Specification:
{spec}
Measured metrics (reference / mid / naive, per seed): {measure}
Thresholds fixed from those measurements (do not change them): {thresholds}

Write (generator, solution and mid/naive baselines already exist and are fixed):
- tests/test_outputs.py: pytest; import `check` from _metrics and call check(name, value, threshold, direction) once
  per metric with EXACTLY the thresholds above; compute values the same way as tests/measure.py.
- baselines/exploit_*.py: at least one shortcut attempt (constant/default output, public-data echo, skipping the
  {mid_drop} step entirely).
- instruction.md: the task statement.  Every threshold above appears with its direction; deliverables and file
  formats exact; nothing from derive/internal/hidden_truth nodes; no "do not read" rules; time budget {hours} h.
Format: === FILE: <path> === blocks, then === END ==="""

S8_REPAIR = """The task failed blocking gates.  Gate log:
{gates}
Dependency graph (fixed, do not change): {depgraph}
Current files:
{files}
Return ONLY the files you change (same paths; no new files, no deleted criteria, thresholds unchanged unless the
gate log says they were re-derived).  Format: === FILE: <path> === blocks, then === END ==="""
