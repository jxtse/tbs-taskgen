"""Instructions for the code-grounded agent jobs (S4c runnability, S5G task authoring).  Versioned like prompts.py."""

CODE_PROMPT_VERSION = "cg-v1-2026-11-12"

S4C_RUNNABILITY = """# Runnability check of a paper's published code ({item})

You are preparing the authors' own code of a scientific paper so that a later step can build a benchmark task on it.
Work in this container (you are root; network access is allowed while you install things).

## Inputs (read-only copies)
- `/work/paper.txt` — the paper's full text; `/work/candidate.json` — title / DOI / abstract.
- `/work/repo/` — the authors' repository (or unpacked archive), pinned: `/work/code.json` (commit / archive sha256,
  languages, dependency files, licence file).

## Your job (time limit about {minutes} minutes; be efficient)
1. Read the README / docs / dependency files.  Decide the language(s) and how the authors intend the code to be run.
2. Install what is needed **inside this container** (apt, pip, conda/mamba, R `install.packages` / `BiocManager`,
   Julia `Pkg`, cargo, make / cmake ...).  Any language is fine.  Use CPU only: never install CUDA.
   MATLAB code: try GNU Octave (`apt-get install octave`) first.
3. Run **one** real example of the authors' code end to end: their example / demo / test suite / main entry point on
   the data shipped with the repository or on small synthetic input you create.  It counts only if the authors'
   own code computes a numerical result (an import that succeeds is not enough).  Keep the run under 15 minutes on
   4 CPUs; reduce sizes / iterations if the documented example is bigger.
4. Write the results:
   - `/app/out/runnability.json`:
     `{{"runnable": true|false, "reason": "<one of: ok, needs_gpu, needs_commercial_license, needs_mathematica,
       matlab_octave_incompatible, build_failed, install_failed, missing_data, crashes, too_slow, no_entry_point,
       other>", "language": "...", "entry": "<command you ran>", "seconds": <runtime of the example>,
       "install": ["<shell commands that set up the environment, in order>"],
       "outputs": "<what the run produced: files / printed numbers>", "notes": "..."}}`
   - `/app/out/run.log` — the full log of the successful (or last failed) run.
   - `/app/out/HOWTO.md` — for the next agent: how to install and run the authors' code, the main functions /
     classes and what they compute (file:line), the inputs they expect (formats, shapes, units), where real or
     example data lives in the repo, simulators / data generators the authors provide, runtime of a typical call,
     pitfalls you hit.  Be concrete; this is the only memory the next step gets.
   - `/app/out/setup.sh` — a shell script that reproduces the installation from a fresh `python:3.11-slim`
     container (the commands you actually used; it may download packages).
5. Not runnable (`runnable: false`) when: it needs a GPU, MATLAB without a working Octave fallback, Mathematica, a
   commercial licence or solver, data you cannot obtain, or it still fails after reasonable effort.  Say why.

Do not modify `/work/repo` in place (copy it to /app/repo if you need to patch something, and list the patches in
HOWTO.md).  Do not upload anything anywhere.
"""

S5G_AUTHOR = """# Build a benchmark task from a paper and its authors' code ({item})

You are an expert task author for a scientific-computing benchmark (Terminal-Bench Science style).  You write a task
that a strong AI agent must solve in an offline container **without** access to the authors' code.  You have the
paper, the authors' code (installed and runnable here) and the harness contract.  Work in this container (root;
network allowed for installing).  Time limit about {minutes} minutes; spend it on getting a task that truly
separates a correct expert solution from simpler approaches.

## Inputs (read-only)
- `/work/paper.txt`, `/work/candidate.json` — the paper.
- `/work/repo/` — the authors' code, pinned (`/work/code.json`; `.tbsgen-remote-repo` names the pin).  `/work/HOWTO.md`, `/work/runnability.json` — how to
  install and run it (from the previous step: follow `install` / `setup.sh` there first).
- `/work/contract/S5_SPEC.txt` and `/work/contract/S6A_GEN.txt` — the harness contract the pipeline uses everywhere
  (`{{...}}` are template slots: ignore them).  Your files must satisfy it exactly: the same locations
  (/app/data, /app/output, /tests/truth, /tests/hidden, solution_cli.py for PROG), the same spec JSON keys, the same
  baselines rules, `tests/measure.py` with `compute()` using only `_metrics` helpers (`/work/contract/METRICS_API.txt`,
  helper source `/work/contract/_metrics_template.py`).
- Allowed forms: {forms}.  Expert time {hours_lo}-{hours_hi} hours.

## Deliverable: `/app/out/` with exactly this layout
```
/app/out/_spec/spec.json           the S5 spec JSON (all keys of S5_SPEC; "method_names" must list the paper's
                                   method name AND the authors' repository / package names)
/app/out/_spec/author_code.json    {{"repo": "...", "pin": "...", "package_names": [...], "main_functions": [...],
                                    "reference_uses": "<which author functions the reference calls or ports>",
                                    "data_from": "<repo data / author simulator / own generator>",
                                    "notes": "..."}}
/app/out/task/instruction.md       the statement the solver sees
/app/out/task/generator/generate.py
/app/out/task/solution/solve.py
/app/out/task/tests/measure.py
/app/out/task/baselines/mid.py
/app/out/task/baselines/naive.py
/app/out/report.md                 what you built, the measured reference / mid / naive metric values per seed
```

## Grounding in the authors' code (this is the point of this pipeline)
- The **reference solution** should call or port the authors' implementation of the paper's method (port = copy the
  needed functions into `solution/solve.py`, self-contained; credit the source file in a comment).  It must run in
  the task image: python 3.11 with exactly numpy 2.1.3, scipy 1.14.1, pandas 2.2.3, scikit-learn 1.5.2,
  statsmodels 0.14.4 (no other packages, no network) -- **never** the authors' repository or package (the solver
  must not get it).  Test that in a fresh venv with only those pins.  If the authors' code is not Python or needs
  other packages, use it at **build time only**: run it here to produce ground truth / reference outputs for fixed
  inputs, store them under `generator/assets/` (small, < 20 MB, with `generator/assets/PROVENANCE.md`: exact
  commands, commit), have `generate.py` read them, and write the reference as a faithful Python re-implementation
  checked against those assets.
- **Data**: prefer the real data shipped with the repository or the authors' own simulator / data generator; vary
  seeds / parameters for the hidden (PROG) inputs.  Synthetic data from your own generator is the fallback.
- `generator/generate.py <seed>` must be deterministic in the seed and run with the same pinned stack (it runs in a
  network-less container with only that stack; it cannot call the authors' code).

## No leaks of the authors' implementation
- `instruction.md` must not name the repository, the package, the authors' function / class names, or their file
  names.  Describe the method scientifically (as the paper does), not by pointing at code.
- Do not write `task/environment/` (the pipeline builds the task image itself and discards that directory).  The
  solver's environment never gets `/work/repo` or the authors' package.

## Iterate until it works (you can run everything here)
Run, from `/app/out/task`: `python generator/generate.py 0` and `1`, the reference and both baselines on the
generated data (copy environment/data to /app/data, outputs to /app/output, truth to /tests/truth exactly as the
harness does), then `tests/measure.py`.  Required: the reference beats naive on every metric; mid fails 1-2
criteria and naive fails >= 2 at thresholds halfway (log scale) between reference and mid / naive; the reference
finishes in < 15 min on 2 CPUs; the generator in < 5 min.  If they do not separate, change the task (metrics,
mid_drop, hidden shift), not the numbers.  Record the measured values in report.md.

Do not write thresholds; the pipeline measures and sets them.  Do not upload anything.
"""
