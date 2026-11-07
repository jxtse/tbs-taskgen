# S2 reference-label rubric (written before any screen was run on the set)

Purpose of S2: decide, from the bibliographic record alone (title, abstract or the first ~4000 characters of the full
text when there is no abstract, venue, year, code/data links), whether a paper is worth the next, more expensive
stages (full-text download and a feasibility probe) on the way to a **terminal-based computational task**: an agent
re-implements or applies the paper's computation on data, and the result is checked numerically.

A paper is **SUITABLE** when all of these plausibly hold:

1. **Computational core.** It presents or uses a concrete computational procedure (algorithm, estimator, numerical
   scheme, model fitting, signal/image/data processing, simulation, optimisation) that could be re-implemented from
   the paper and its references. Applying a standard method in a non-trivial way to a well-defined data problem counts.
2. **Numerical check.** The output of that computation can be compared with something numerical: a known truth,
   held-out data, a simulation, a published table.
3. **Data.** Input data are public, released with the paper, or could be simulated realistically from a model the
   paper describes. Statistical methods applied to sample-survey data count; "survey" in that sense is not a review.
4. **Scale.** The core computation runs on one ordinary machine in hours: no GPU cluster or large-model training,
   no supercomputer-scale run, no wet-lab work or special instrument needed to reproduce the computation itself.

A paper is **UNSUITABLE** when it is mainly one of: a literature review / survey of methods / tutorial / perspective;
a clinical trial, cohort or qualitative study whose contribution is not a computational method; a purely experimental
or observational study where computation is incidental; pure theory with nothing to compute on data; a method whose
reproduction needs large-scale training or special hardware; a method that only works by interactive querying of a
live experiment or expensive black box under a budget (these are flagged, not dropped, by S2 — see below).

Topic relevance to a particular retrieval target does **not** matter here.

**Borderline**: information is insufficient (e.g. a very short abstract) or the criteria conflict. The reference label
then records the lean (`keep` / `drop`) and `confidence: low`.

Interactive / black-box-budget methods (Bayesian optimisation of a live experiment, active-learning loops, bandits)
get `interactive: true`; their keep/drop follows the four criteria above (S4, not S2, bans the form).

Output per paper: `{"id", "label": "keep" | "drop", "confidence": "high" | "low", "interactive": bool,
"why": "<= 25 words"}`.

Metrics: positive class = `keep`. Recall = kept by the screen among reference-keep; precision = reference-keep among
kept by the screen; keep rate = kept / all. Primary numbers on the test half; thresholds and any rule changes are
tuned on the dev half only (split = SHA-256 of the DOI/title key, even -> dev).

These are **model reference labels (Claude Opus), not human gold labels**.
