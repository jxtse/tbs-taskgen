# tbs-taskgen

[![CI](https://github.com/jxtse/tbs-taskgen/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jxtse/tbs-taskgen/actions/workflows/ci.yml)

**Turn scientific papers into executable, Terminal-Bench-style scientific computing tasks.**

tbs-taskgen is a research pipeline that reads a method paper and produces a
[Harbor](https://github.com/harbor-framework/harbor)-format task in the style of
[Terminal-Bench-Science](https://github.com/harbor-framework/terminal-bench-science): a task
statement, a seeded data generator, a reference solution, mid/naive baselines and hidden tests.
The tasks are meant for evaluating terminal agents and for building agent training data.

- **Input:** a retrieval target (a method topic plus a few keyword queries), or your own list of
  candidate papers as JSON/JSONL.
- **Output:** one directory per paper under `runs/<run>/items/<id>/`. Papers that pass every gate
  get a Harbor task in `pkg/` (`task.toml`, `instruction.md`, `environment/` with Dockerfile and
  public data, `solution/`, `tests/`). Each run also has a per-item state machine, a spend ledger
  and `release/manifest.jsonl` listing the tasks that passed difficulty calibration.
- **Scope:** computational tasks whose deliverable is a numeric answer or a program and can be
  graded automatically (task forms `ANS` and `PROG`). File-format-heavy, interactive and proof
  tasks are not supported yet.

> **Status: research code, not production.** The pipeline runs end to end on real papers, but it
> has been tried on only a handful of them and has not been used for bulk generation. See
> [Status and limitations](#status-and-limitations).

## How it works

```
retrieval/ (tbsretr)                       gen/ (tbsgen)
────────────────────                       ──────────────────────────────────────────────────────────
OpenAlex · Semantic Scholar · Exa          S1 ingest → S2 screen → S3 full text → S4 feasibility + dedup
Crossref · arXiv · Europe PMC              → S5 spec (teacher) → S6 generate files (teacher), set thresholds
(Tavily adapter, off by default)             from measured reference / mid / naive scores
  → dedup → RRF fusion → rule prefilter    → S7 mechanical gates, incl. 4-layer leak check
  → ranked candidate papers  ──────────▶   → S8 bounded local repair → PKG Harbor package
                                           → S9 difficulty calibration → S10 release manifest
```

1. **Multi-source literature retrieval** (`retrieval/`). Sources are queried in parallel; results
   are normalised, deduplicated across DOI / arXiv / OpenAlex / S2 IDs / title, fused with
   reciprocal-rank fusion and rule-prefiltered. URLs of the TB-Science benchmark itself are dropped by
   default, and papers that TB-Science tasks were built from are dropped when you provide that
   list (`TBS_LEAK_REFS`). This layer calls no LLM.
2. **Teacher-model task authoring** (`gen/`, S5–S6). A teacher model writes a spec with an explicit
   information-dependency graph, then the generator, reference solution, baselines, tests and
   statement. Pass thresholds are set from the *measured* scores of the reference and the mid
   baseline in a Docker sandbox, not chosen by the model.
3. **Leak checks** (S7). L1: truth or secret names in the statement or public text, and rules that
   "protect" public files only by asking the agent not to read them. L2: truth values in public
   files, including permuted, negated or scaled copies. L3: introspection of public Python objects
   inside Docker. L4a: rank correlation of truth columns with public columns. A `truth_isolation`
   gate checks that the agent's program cannot read `/tests` during grading.
4. **Reference and empty-submission checks** (S7). Same seed gives the same output; the reference
   passes every seed with margin; an empty submission (`nop`) fails; the naive baseline fails at
   least two criteria and the mid baseline one or two; scripted exploit attempts fail; `PROG`
   tasks are graded on hidden, distribution-shifted inputs. Blocking failures go to a local repair
   loop (S8, at most two rounds).
5. **Difficulty calibration** (S9). A student agent runs the packaged task in Harbor; if it fails, a
   teacher agent tries. A task is kept if it is *unstable* (the student passes one of two runs) or
   *hard* (the student fails and the teacher passes). By default S9 only writes job configs and
   launch commands; submitting jobs must be enabled explicitly.
6. **Packaging** (PKG, S10). Harbor task layout with a self-contained Dockerfile; kept tasks are
   appended to `release/manifest.jsonl`.

Every real model call goes through a sqlite budget ledger with daily and per-run caps and a
circuit breaker. Real calls are refused until a daily budget is configured. Items resume from
their last completed stage after a crash or a budget stop.

## Quick start

### Hand this to your coding agent

> Clone https://github.com/jxtse/tbs-taskgen. Make sure `uv` and Docker are installed. Build the
> sandbox image from `gen/docker/Dockerfile.base` with the tag `tbsgen-base:py311-sci-v2`. Run the
> offline tests with `uv run pytest -q` in `retrieval/` and then in `gen/`. Then run the fake-model
> dry run `uv run --directory gen python -m tbsgen.dryrun --out runs/dryrun`. Expected result: all
> tests pass, and the dry run prints `toy: S10 done hard` and leaves a packaged Harbor task in
> `gen/runs/dryrun/items/*/pkg/`. Do not set any API keys or budgets; none of these steps should
> call a paid API. Afterwards, read the Configuration section of the README and tell me which
> environment variables and `config.toml` keys I must set to run the pipeline on my own papers.

### Manually

You need [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed) and Docker for the
sandbox and the integration tests.

```bash
git clone https://github.com/jxtse/tbs-taskgen.git
cd tbs-taskgen

# Sandbox image for generators, references, baselines and tests (python 3.11 + pinned scientific stack).
# Add --build-arg PIP_INDEX_URL=<mirror>/simple if PyPI is slow from your network.
docker build -t tbsgen-base:py311-sci-v2 -f gen/docker/Dockerfile.base gen/docker

# Offline tests: recorded API responses, no keys, no model calls
cd retrieval && uv sync && uv run pytest -q && cd ..
cd gen && uv sync && uv run pytest -q && cd ..   # Docker integration tests are skipped without Docker or the image

# End-to-end dry run with a scripted fake teacher: S1 → S10, no real model calls, no Harbor jobs
uv run --directory gen python -m tbsgen.dryrun --out runs/dryrun
```

The dry run prints the run status and then `toy: S10 done hard`; the packaged task is in
`gen/runs/dryrun/items/<id>/pkg/`. To inspect a run:

```bash
cd gen
uv run tbsgen status --run runs/dryrun
uv run tbsgen ledger --run runs/dryrun
uv run tbsgen report --run runs/dryrun    # writes runs/dryrun/reports/daily-<date>.md
```

### On real papers

The first two steps make free API calls (OpenAlex, Crossref, arXiv) and no model calls. They also
need [`jq`](https://jqlang.org/). Without a key, OpenAlex may answer with HTTP 429; the source then
cools down and the run finishes with the remaining sources.

```bash
# 1. Retrieve candidates for one of the bundled targets (sources without a key are skipped)
uv run --directory retrieval tbsretr status
uv run --directory retrieval tbsretr run --targets eval/targets10.json \
    --only task:ambient-rna-correction --sources openalex,crossref,arxiv --seed-mode none \
    --out ../runs/retrieval

# 2. Turn the top candidates into pipeline items and run the stages that need no model
jq -c '.papers[:2][]' runs/retrieval/task__ambient-rna-correction.json > runs/candidates.jsonl
uv run --directory gen tbsgen ingest --run runs/R --candidates ../runs/candidates.jsonl
uv run --directory gen tbsgen run    --run runs/R --config ../my.toml
uv run --directory gen tbsgen status --run runs/R
```

On a fresh machine both items stop at S3 as `parked: fulltext_unavailable` (listed in
`S3_queue.json`): the pipeline needs the paper's full text, from a local cache
(`fulltext.cache_dirs`) or an open-access PDF link. With full text, a teacher endpoint and a
budget in `my.toml` (see below), `tbsgen run` continues into the model stages. The other commands
(`resume`, `reset`, `breaker`, `calib-ingest`, `harbor-local`, `ping`) are documented in
[`gen/README.md`](gen/README.md) §3.

## Configuration

Both packages start from built-in defaults, then apply an optional TOML file, then the environment
variables listed below. Pass the file with `tbsgen <command> --config FILE` or
`tbsretr --config FILE <command>`; `tbsretr` also reads `retrieval/config.toml` if it exists. Only
the keys you want to change need to be in the file.

### LLM provider (teacher model)

The teacher client uses the Anthropic Messages API (`POST …/v1/messages`). Point it at any
compatible endpoint and pick the model yourself:

```toml
# my.toml
[budget]
daily_usd_total = 5.0                                  # required: real model calls are refused while this is unset
daily_calls_per_model = { "my-teacher-model" = 10 }

[models.opus]                                          # historical alias name; any compatible model works
url     = "https://your-endpoint.example.com/v1/messages"
model   = "my-teacher-model"
key_env = "MY_TEACHER_API_KEY"                         # name of the environment variable holding your key

[models.prices]                                        # USD per million tokens, for the ledger
"my-teacher-model" = { in = 3.0, out = 15.0 }
```

The key is read from the named variable at call time, sent only in a request header and redacted
from error messages. Without `key_env`, no key is sent (useful behind a local gateway that holds
credentials). The built-in price table is a placeholder; set prices for your model.

### Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `TBS_OPUS_URL` | gen | Teacher endpoint URL (Anthropic-compatible `/v1/messages`). No default: set it here or with `models.opus.url`; real model stages pause with `model:config` until it is set. |
| *your `key_env`* | gen | Teacher API key, if `models.opus.key_env` is set. |
| `TBS_OLD_PIPE` | gen, retrieval | Data from an earlier pipeline: a full-text cache (`literature/txt`) and TB-Science task texts used for deduplication (`tasks_text`). Unset: no full-text cache and no deduplication. |
| `TBS_LEAK_REFS` | retrieval | JSON file of papers that TB-Science tasks were built from; these are excluded from retrieval. Defaults to `$TBS_OLD_PIPE/literature/tbs_task_refs.json` when `TBS_OLD_PIPE` is set; otherwise no exclusion list. |
| `TBS_HARBOR_ROOT` | gen (S9) | Directory with your Harbor job launch scripts for calibration (`calib.harbor_root`). Unset: S9 prints a plain `harbor run --config ...` command. |
| `TBS_HARBOR_BIN` | gen | Path to the `harbor` binary for `tbsgen harbor-local` (otherwise looked up on `PATH`). |
| `TBS_SEARCH_POOL_URL` | retrieval | Search-pool service for Exa (`sources.exa.backend = "pool"`, the default). No default: unset means Exa is skipped. Or set `backend = "direct"` and use `EXA_API_KEY_SECONDARY`. |
| `OPENALEX_API_KEY` | retrieval | Optional; OpenAlex also works without a key, at lower rate limits. |
| `SEMANTIC_SCHOLAR_API_KEY` | retrieval | Required for Semantic Scholar; the source is skipped without it. |
| `EXA_API_KEY_SECONDARY` | retrieval | Exa key, used with `sources.exa.backend = "direct"`. |
| `TAVILY_API_KEY_SECONDARY` | retrieval | Tavily key; the Tavily source is disabled by default. |
| `TBSGEN_METRICS`, `TBSGEN_ENTRY`, `TBSGEN_HIDDEN` | generated tests | Set by the local test runner inside the sandbox; you do not set these. |

Crossref, arXiv and Europe PMC need no key. `uv run tbsretr status` lists which sources are usable;
it shows whether a key is present, never its value.

### Optional external settings

There are no machine-specific defaults. Unset settings switch the corresponding feature off:
the teacher URL (real model stages pause until it is set), the Exa search pool (Exa skipped),
`TBS_OLD_PIPE` (no full-text cache, no deduplication against TB-Science tasks), and the S9 launcher
settings under `[calib]` (`harbor_root`, `student_launcher`, `teacher_launcher`). Without launchers,
S9 writes the job config and prints a plain `harbor run --config <file> --yes` command, and
`allow_submit = true` refuses to launch. Calibration jobs use Harbor's built-in `claude-code` agent
and `docker` environment unless `student_agent`, `teacher_agent` or `environment` name a custom
`module:Class`. None of this affects the tests or the dry run.

All other keys and their defaults are in [`gen/src/tbsgen/config.py`](gen/src/tbsgen/config.py) and
[`retrieval/src/tbsretr/config.py`](retrieval/src/tbsretr/config.py), and are explained in the two
package READMEs. Commonly changed ones: `fulltext.cache_dirs` (directories of full texts named
after the DOI), `fulltext.allow_network` (open-access PDF download), `calib.allow_submit`
(default `false`: only write calibration job configs), `calib.prepared_tasks_dir`,
`sandbox.image`, and the per-source daily budgets of the retrieval layer.

## Status and limitations

Research code under active development.

**Tests** (run by [GitHub Actions CI](.github/workflows/ci.yml) on every push to `main` and every pull request)

- `retrieval/`: 47 offline unit tests that replay recorded API responses.
- `gen/`: 69 tests: 53 unit tests and 16 integration tests that run task code in Docker. Locally
  they are skipped when Docker or the sandbox image is missing; CI builds the image and fails if
  any of them is skipped.
- Fake-model dry run: S1 → S10 on a synthetic toy task, including the mechanical gates and
  simulated calibration results.

**Evaluations** (separate from the tests)

- Leak-gate backtest on previously generated tasks, no LLM calls
  ([`gen/eval/LEAK_BACKTEST.md`](gen/eval/LEAK_BACKTEST.md)): 7/7 known leaking tasks caught;
  0/231 blocking false positives in-sample (the rules were tuned on this set) and 1/81 on a
  held-out set.
- Retrieval evaluation on 10 targets ([`retrieval/eval/REPORT.md`](retrieval/eval/REPORT.md)).
  Relevance labels come from a single annotator and are subjective.
- End-to-end runs on 6 real papers: 3 became packaged
  Harbor tasks where the reference solution scores 1 and an empty submission scores 0; 2 were
  dropped because the mid baseline could not be separated from the reference; 1 was still in
  repair. Each of the 3 packaged tasks was calibrated with a single student run, which is not
  enough to assign difficulty labels.

**Known limitations**

- Packaged tasks currently use `network_mode = "public"`, so an agent could install an existing
  implementation of the paper's method. Offline-by-default and a check for ready-made
  implementations are planned, not implemented.
- The teacher-agent branch of calibration has not been run on real tasks.
- The L4b inversion attack, the cheap-model roles, harder-variant generation (S9b) and the
  file-format and interactive task forms are not implemented.
- The expert-time estimate in `task.toml` is the teacher model's own estimate.
- Full text comes only from a local cache or open-access PDF links.
- Automatic S9 job submission needs your own Harbor launch scripts (`calib.*_launcher`).

## Repository layout

```
retrieval/                 tbsretr: retrieval layer (standard library only)
  src/tbsretr/             source adapters, fusion, rate-limited and budgeted HTTP ledger, CLI
  tests/                   offline tests and recorded responses (keys stripped)
  eval/                    10-target retrieval evaluation, REPORT.md
  scripts/                 fixture re-recording
gen/                       tbsgen: task-generation pipeline
  src/tbsgen/              stages, gates, leak checks, sandbox runner, packaging, calibration, CLI
  docker/Dockerfile.base   sandbox image
  tests/                   unit and Docker integration tests; fixtures/toy_task is a synthetic task
  eval/                    leak-gate backtest results
```

The package READMEs ([`retrieval/README.md`](retrieval/README.md), [`gen/README.md`](gen/README.md))
are development notes in Chinese. They contain the full stage-by-stage specification and the
configuration reference. Internal working notes (design drafts, run logs) are kept out of the
repository; `docs/internal/` is git-ignored for that purpose.

Not in the repository: run directories (`runs/`, `state/`), raw retrieval responses, paper full
texts and TB-Science task texts.

## License

[MIT](LICENSE) © 2026 Jinxiang Xie
