---
profile_id: example-weakness-v0
student: <calibration student model and harness>
---
# Example weakness profile (generic; replace with your own internal profile)

The pipeline reads `difficulty.profile_path`.  Entries are `## <ID> <title>` headings; the id is what an authoring
agent cites in `_spec/difficulty_design.json`.  Write each entry from *your own* evidence about the student model
(failure attributions, comparisons with stronger solvers), keep it abstract, and never copy names, data, methods,
constants or examples of the evaluation tasks that evidence came from.  Keep the real profile outside the
repository and record its version in the run configuration.

Each entry should state: when the model tends to fail; a legitimate scientific lever an author can build on; how to
show with the reference / mid / naive baselines that the lever works; and which shortcuts are forbidden.

## EX01 Residual structure is dismissed
- When it fails: a plausible first model leaves a structured residual above the noise level; the solver calls it
  noise and stops.
- Lever: the data contain a real secondary component that the paper's method models; the score depends on it.
- Show it: mid = reference without the extension fails the affected metric on every seed.
- Forbidden: structure below the noise floor or invisible in the provided data.

## EX02 Generic method with a structural ceiling
- When it fails: a generic method is tuned instead of being replaced by one derived from the stated principles.
- Lever: the paper fixes a real structural limitation of the common approach.
- Show it: the tuned generic baseline stays below threshold; the principled reference clears it with margin.
- Forbidden: banning standard tools; thresholds at the reference's own edge.

## EX03 Self-consistent but unrepresentative validation
- When it fails: validation on the solver's own simulator or on the fitted data misses a bias.
- Lever: hidden cases differ along a disclosed, scientifically motivated axis.
- Show it: a public-tuned baseline passes public metrics and fails hidden ones.
- Forbidden: undisclosed shifts; hidden cases needing information not in the statement or data.

## Forbidden artificial difficulty (all entries)
Ambiguous statements, hidden requirements, conventions that cannot be inferred, output-format traps, thresholds at
the reference's numerical floor or decided by luck, pure time pressure, and any mention of the profile in task
files.
