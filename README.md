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
                                             from measured reference / mid / naive scores
  → dedup → RRF fusion → rule prefilter    → S7 mechanical gates, incl. 4-layer leak check
  → ranked candidate papers  ──────────▶   → S8 bounded local repair → PKG Harbor package
                                           → S9 difficulty calibration → S9R task review
                                           → S10 release manifest
```

1. **Multi-source literature retrieval** (`retrieval/`). Sources are queried in parallel; results
   are normalised, deduplicated across DOI / arXiv / OpenAlex / S2 IDs / title, fused with
   reciprocal-rank fusion and rule-prefiltered. URLs of the TB-Science benchmark itself are dropped by
   default, and papers that TB-Science tasks were built from are dropped when you provide that
   list (`leak_refs` in the retrieval config). This layer calls no LLM.
2. **Teacher-model task authoring** (`gen/`, S5–S6). A teacher model writes a spec with an explicit
   information-dependency graph, then the generator, reference solution, baselines, tests and
   statement. Pass thresholds are set from the *measured* scores of the reference and the mid
   baseline in a Docker sandbox, not chosen by the model.
3. **Leak checks** (S7). L1: truth or secret names in the statement or public text, and rules that
   "protect" public files only by asking the agent not to read them. L2: truth values in public
   files, including permuted, negated or scaled copies. L3: introspection of public Python objects
   inside Docker. L4a: rank correlation of truth columns with public columns. A `truth_isolation`
   gate checks that the agent's program cannot read `/tests` during grading. The agent's program also cannot
   tamper with the score: the verifier runs it in its own session as `nobody`, kills its process group and
   every remaining `nobody` process after each call, keeps `/logs/verifier` unwritable for it, and `test.sh`
   writes `reward.txt` last (`_metrics.py --finalize-reward`), only after no such process is left.
4. **Reference and empty-submission checks** (S7). Same seed gives the same output; the reference
   passes every seed with margin; an empty submission (`nop`) fails; the naive baseline fails at
   least two criteria and the mid baseline one or two; scripted exploit attempts fail *and* each
   exploit really ran under the verifier's invocation and produced scorable output (`exploit_valid`; a
   crashing exploit tests nothing and goes to repair); resource metrics (runtime, memory) are timed as the
   median of `thresholds.timing_repeats` runs (default 3) both when thresholds are set and when grading; the
   agent program's scored runs share one total cap that, with the repeats' budget and
   `package.verifier_overhead_sec`, ends before the task's verifier timeout (checked at packaging); `PROG`
   tasks are graded on hidden, distribution-shifted inputs. Pre-release checks: the task image, the
   reference and the baselines must not use the paper's own published implementation, and public
   task files must not copy long passages of a TDM full text. Blocking failures go to a local repair
   loop (S8, at most two rounds).
5. **Difficulty calibration** (S9). A student agent runs the packaged task in Harbor; if it fails, a
   teacher agent tries. A task is kept if it is *unstable* (the student passes one of two runs) or
   *hard* (the student fails and the teacher passes). By default S9 only writes job configs and
   launch commands; submitting jobs must be enabled explicitly.
5b. **Task review** (S9R). Every task S9 would keep, and every task whose teacher runs all failed (as a
   diagnosis), is audited by a fresh code-executing agent (Claude Code, Opus by default) inside the task's own
   image, with the model endpoint as its only network. It sees the whole package, baselines, generator,
   threshold and gate records and a digest of every calibration trace, reruns the reference and tests, and
   returns `pass` / `revise` / `drop` with evidence per checklist item (consistency, missing conditions,
   thresholds, stability, shortcuts and leaks, network / ready-made code, calibration evidence, and whether
   timing metrics should gate at all). Every blocking issue names a problem class: local ones (statement, format
   notes, grader inputs/outputs, exploit baseline, test gap, timing gate) can be revised; fundamental ones
   (reference wrong, truth must be regenerated, task core invalid, untrustworthy calibration) mean drop. The
   pipeline derives the verdict from those classes, so a `revise` whose blocking issues are all fundamental is
   routed as `drop`. A blocking issue without evidence is downgraded to minor; a leak of ground truth or answers or
   a way to tamper with the scoring is always blocking. There is no human exit by default: each task's fate is
   decided by the review agent (`review.allow_human`, off by default, restores an operator park). Only `pass`
   releases a task. `revise` sends it back to a local repair of the statement / format notes / test scripts with
   **thresholds locked** (never loosened), up to `review.max_revisions` (default 3) rounds, then drop. A revision
   that only changed the grader first re-scores the kept student and teacher deliverables with the new grader;
   if no outcome changes the calibration is kept and the task is reviewed again, otherwise it is recalibrated.
   `tbsgen review summary` reports verdict counts, issue classes, revision rounds per task, a random sample of
   passed tasks with their review record paths for spot checks, and alerts (drop rate, contradictory reviews).
   Details: [`gen/README.md`](gen/README.md) §5b.
   Timing metrics (runtime, memory) are recorded but only decide a run when the spec sets
   `resource_metric_gating = true` (speed is the paper's contribution); time thresholds are then scaled by the
   grading machine's speed relative to the machine that set them (a fixed CPU benchmark recorded in
   `thresholds.json`, ratio clamped to 0.25–8). Details: [`gen/README.md`](gen/README.md) §5c.
6. **Packaging** (PKG, S10). Harbor task layout with a self-contained Dockerfile and a network
   policy that keeps the agent off the internet while it works (see
   [Network of packaged tasks](#network-of-packaged-tasks)); kept tasks are appended to
   `release/manifest.jsonl`.

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

On a fresh machine both items may stop at S3 as `parked: fulltext_unavailable` (listed in
`S3_queue.json` and in the run's full-text queue): the pipeline needs the paper's full text. S3 tries a
configurable cascade (see [Full text (S3)](#full-text-s3)); by default only a local cache and open-access copies
are used. `tbsgen fetch --run runs/R` retries the queue later, hands the text to every waiting item (also items of
other runs sharing the queue) and moves them on to S4; an entry leaves the queue only after that. Only a paper whose
channels actually sent a request and got nothing backs off (doubling from `fulltext.queue_backoff_s`, per channel);
a paper that was only skipped because of a cap, cooldown or breaker (or the egress check) is not charged a try and
becomes due again when the earliest of those limits lifts. `tbsgen fetch --only DOI[,DOI]` (DOIs, arXiv ids or queue
idents) processes just those entries, due or not. Only one `tbsgen fetch` drains a queue at a time, and only one
process at a time uses a browser TaskSpace (a second one skips the browser channel). With full text, a teacher endpoint and a
budget in `my.toml` (see below), `tbsgen run` continues into the model stages. The other commands
(`resume`, `reset`, `breaker`, `calib-ingest`, `calib watch`, `calib prepare`, `review build`, `harbor-local`, `ping`)
are documented in [`gen/README.md`](gen/README.md) §3; calibrating only some items, the environment-preparation hook
(`calib.prepare_command`), teacher-only routing (`calib.teacher_*`) and the S9R review settings (`review.*`, on by
default; routed like the teacher unless set) are in §4–§5b there.

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

[models.teacher]                                       # any Anthropic-compatible model
url     = "https://your-endpoint.example.com/v1/messages"
model   = "my-teacher-model"
key_env = "MY_TEACHER_API_KEY"                         # name of the environment variable holding your key

[models.prices]                                        # USD per million tokens, for the ledger
"my-teacher-model" = { in = 3.0, out = 15.0 }
```

The key is read from the named variable at call time, sent only in a request header and redacted
from error messages. Without `key_env`, no key is sent (useful behind a local gateway that holds
credentials). The built-in price table is a placeholder; set prices for your model.

### Code-grounded generation (`code.grounded`)

With `[code] grounded = true` a task is built from the paper **and its authors' published code**; a paper without
the authors' code is dropped. Two filters run before any model call: **S2a** (before the full-text download) keeps a
paper only if some code signal exists — code links in the metadata, the retrieval layer's `code_signals` (OpenAlex
full-text search for code hosts, the frozen Papers-with-Code archive, GitHub repositories whose README cites the
DOI / arXiv id) or an OpenAlex full-text flag; **S3b** (after the full text) scores every code link for ownership
(Code Availability section, link context, reference list, README citing the paper), asks the cheap Jev model only
for ambiguous links (`roles.code_owner = "jev"`), resolves the authors' links on their host and drops with a
specific reason (`code_on_request`, `third_party_code_only`, `code_repo_unavailable`, `data_only_no_code`,
`code_host_unsupported`, `no_code_link`). Licences are recorded, not filtered. **S3c** clones (or downloads) the authors' code on a build host over ssh and pins it (commit SHA / archive sha256; size
cap, no LFS); third-party code never runs on the orchestrating machine. **S4c** is a runnability gate: a Claude Code
agent job on the build host installs the code (package hosts allowlisted during install), runs one of the authors'
examples / tests and writes a "how to run it" note, or the paper is dropped (`code_not_runnable`, `code_needs_gpu`,
`code_needs_commercial_license`, ...). **S5G** replaces the single-call S5 + S6A: a Claude Code agent with the paper,
the repository, that note and the harness contract writes exactly the same artefacts (`_spec/spec.json`,
`task/{instruction.md, generator, solution, tests/measure.py, baselines}`) and iterates in its container until the
reference and the baselines separate; everything from S6A_run on is unchanged (`code.authoring = "single_call"`
restores the old path). The task image never contains the authors' repository or package: S7's ready-made check
includes the authors' repository / package / main function names, a new blocking `author_leak` gate checks the
statement and public files for them, and the S9R checklist asks whether the task is just a re-run of the authors'
repository. `sandbox.backend = "ssh"` runs the generator / reference / baselines in the build host's Docker. The retrieval side
(`retrieval/`, `code.enabled`) adds an OpenAlex full-text code channel, widens the fused head
(`code.top_n_factor`, default 2.75) and prefers papers from `code.min_year` (default 2018) on. Details:
[`gen/README.md`](gen/README.md) §0.

**Difficulty targeting** (`[difficulty]`, code-grounded items only). `difficulty.profile_path` points to an
internal weakness profile of the calibration student (abstract failure mechanisms with legitimate authoring levers;
a generic example is `gen/examples/weakness_profile.example.md`). S5G then asks the agent to find the paper's real
difficulties first and target 1-3 matching weaknesses, recorded in `_spec/difficulty_design.json` (internal); S7
checks that each lever shows in the measured reference / mid / naive values, and the S9R checklist treats artificial
difficulty (vague wording, hidden requirements, format traps, edge thresholds, time pressure) as blocking. With
`difficulty.harden` (default on, `harden_max_rounds = 2`) a task the student solves on its first S9 attempt goes
back to S5G with the previous version and a sanitised summary of the passing run (its own final message, minutes,
per-metric margins; never verifier output or truth), is rebuilt and re-gated from S6A_run, and is recalibrated from
scratch; a hardened version that fails to build reverts to the previous one. `tbsgen harden --item ID` starts a
round by hand. Details: [`gen/README.md`](gen/README.md) §0b (Chinese).

### S2 abstract screen

S2 decides from the bibliographic record (title, abstract, venue, year, code/data links) whether a paper is worth
downloading and probing. Two modes, chosen with `models.roles.screen`:

- `"rules"` (default): keyword cues in title and abstract, each cue stem counted once (`rules-v2`; keep at
  `screen.min_score`). No network, no cost.
- `"jev"`: one request per paper to TypeSafe Jev (`https://api.typesafe.ai/v1/systemone`). Ten narrow questions are
  answered in parallel (paper type, whether there is a re-implementable computational method, whether it runs on one
  machine, whether data are public or can be simulated, black-box query budget, review, enough information); code in
  [`gen/src/tbsgen/screen_jev.py`](gen/src/tbsgen/screen_jev.py) combines the probabilities into keep, drop or
  uncertain. Uncertain papers are kept and flagged by default. The rules screen is the fallback when the key is
  missing, the circuit breaker is open or the API is down (`method = "rules-v2-fallback"` in `S2_screen.json`).
  Answers are cached per item, so a resumed run does not call again.

In both modes, papers without an abstract are passed on and re-screened at S4 on the first 4000 characters of the
full text. Evaluation: [`gen/eval/screen_jev/`](gen/eval/screen_jev/).

Before either mode, S2 drops papers from sources where the S3 channels have rarely or never obtained the full text
(`screen.blocklist`, matched on DOI prefix and optionally publication year): Elsevier before 2000, SAGE, the
American Meteorological Society, Powder Diffraction, ASCE and SPIE. Such papers get `method = "blocklist"` and
`reason = "blocklist:<rule>"` in `S2_screen.json` and cost no screen call. Set `screen.blocklist = []` to disable,
or edit the list to add or remove publishers. A rule that blocks more than 10% of a target's candidates gets 5 trial
papers a week through the normal download path (`screen.blocklist_recheck`); at ≥ 5 decided trials with ≥ 60%
success the rule is lifted automatically. Lifts and keeps are logged as events and shown by `tbsgen review summary`.

```toml
[models.roles]
screen = "jev"
[screen.jev]
model = "jev-1.14.0"        # pinned: thresholds were set on this version; the response model is checked
uncertain = "keep"          # or "drop"
on_unavailable = "rules"    # or "park" (item paused with model:jev_unavailable, `tbsgen resume` retries)
concurrency = 4
```

Jev is billed on input tokens only ($0.042 per million at the time of writing; about 1,400 tokens, i.e. well under
$0.001, per paper). Calls are recorded in the budget ledger like any other model call, so `budget.daily_usd_total`
must be set.

### Full text (S3)

S3 tries these channels in the order given by `fulltext.cascade`; the first one that returns a usable text wins.
Every network channel has its own switch, and only `cache` and `oa` are on by default.

| Channel | Source | Needs | Default |
|---|---|---|---|
| `cache` | `<doi_key>.txt` in `fulltext.cache_dirs` (read-only) or in `fulltext.write_cache_dir` | nothing | on |
| `oa` | Europe PMC full-text XML (open-access PMC articles), then the PDF links in the retrieval record, then Unpaywall OA locations if `fulltext.oa.unpaywall_email` is set | nothing | on |
| `elsevier` | Elsevier Article Retrieval API (full-text XML), DOI prefixes `fulltext.elsevier.prefixes` (`10.1016/`) | API key in `ELSEVIER_API_KEY` (optional `ELSEVIER_INSTTOKEN`) and an institutional entitlement | off |
| `wiley` | Wiley TDM API (PDF), prefixes `10.1002/`, `10.1111/`, `10.1029/`, `10.1113/` | token in `WILEY_TDM_TOKEN`, accepted Wiley TDM licence | off |
| `inst` | Publisher PDF URLs your institution's IP range is entitled to, configured as `fulltext.inst.routes` (`{prefix, url}`); empty by default | network on the institution's IP range | off |
| `browser` | A local [ego-browser](https://ego.app) (Ego Lite) session with your signed-in profile, driven by the reviewed script [`browser_fetch.mjs`](gen/src/tbsgen/browser_fetch.mjs) | `ego-browser` on `PATH` | off; when on, S3 only queues and `tbsgen fetch` runs it |

Checks: a download is accepted only if it is a PDF (or full-text XML / an HTML article page with section
headings), its text has at least `min_chars` characters, and at least `title_match_min` of the title words appear
in its first 6,000 characters. HTML wrapper pages, abstract-only API answers and bot-wall pages are rejected.
Accepted texts go to `fulltext.write_cache_dir` (default `<run>/fulltext_cache/`) as `<doi_key>.txt` with a
`<doi_key>.json` provenance record (channel, licence class `elsevier_tdm` / `wiley_tdm` / `oa` / `inst` /
`browser`, URL without query string, time, bytes, SHA-256, title match). Read-only caches are never written.

Not entitled: when the Elsevier or Wiley API answers with metadata or an abstract only (`short text N`: the
institution does not subscribe to that volume), the (channel, DOI) pair is recorded in the ledger table
`not_entitled` and that channel never asks for the DOI again (`fulltext.not_entitled`: `enabled`, `channels`,
`ttl_days`, default no expiry). Such answers do not count towards the channel's failure breaker.

Limits: every download is booked in a SQLite ledger (`fulltext.ledger_path`, default
`<run>/fulltext_ledger.sqlite`) before it is sent. Per channel: successes and attempts per day, successes per hour,
a minimum gap with random jitter between two requests to the same host, and per-host daily and hourly attempt caps;
plus a global daily success cap. Defaults are in
[`gen/src/tbsgen/dlledger.py`](gen/src/tbsgen/dlledger.py) (`DEFAULT_LIMITS`, `DEFAULT_GLOBAL`, `DEFAULT_HOST_CAPS`)
and are small fractions of the providers' published limits. The browser and `inst` channels, which use
subscription access on the institution's IP range, have much lower caps than the TDM APIs (40 papers a day each;
the browser waits 60–120 s between two papers of one publisher and runs different publishers in parallel,
`browser.concurrency`, default 2). HTTP 429 cools the host for 6 hours; a challenge page (Cloudflare, captcha,
Akamai "Access Denied") cools it for 24 hours and is never clicked through; a plain paywall 403 is not a block.
By default a channel's breaker opens for the day after 3 block signals or 8 failures in a row; the `oa` channel,
whose links point at many unrelated hosts, opens only after 10 distinct blocked sites (one site that keeps blocking
is handled by its own cooldown; `breaker_count_hosts`) and never on failures alone (configurable per channel:
`breaker_blocks_per_day`, `breaker_consecutive_fail`). One blocked response counts once. After a redirect the
cooldown lands on the final site; resolvers such as doi.org are never cooled. A cooldown also stops requests that
were already waiting for their slot, and a longer cooldown is never shortened by a later, shorter one. All DOI
prefixes of one publisher share the browser's pacing, caps and cooldowns, and a block on one of the publisher's web
hosts stops both the HTTP channels and the browser for it. The TDM API hosts are kept separate: an API throttle
does not stop the browser, and a website block does not stop the API. Redirect hops to other hosts are paced and
counted against those hosts. Metadata lookups (Europe PMC search,
Unpaywall) are paced at 1 request per second per host and never count as downloads. The ledger uses the
tables `attempts`, `cooldown` and `hosts`, so another tool with the same schema can share the file and the usage
of both adds up. `tbsgen fetch --status` prints today's usage.

Network: HTTP requests ignore `HTTP(S)_PROXY` unless `fulltext.use_env_proxy = true`. Redirects are followed hop by
hop (at most 5, never from HTTPS to HTTP, not into a host in cooldown), and API credentials are sent only to the
original host. Logged errors, ledger notes and provenance records drop URL query strings and mask credentials. The
browser process gets only a short allowlist of environment variables. With
`fulltext.egress_check.enabled`, the public IP is looked up before the first download (and every `ttl_s`) and all
network channels are refused unless it lies in `fulltext.egress_check.allow_cidrs`; the browser's egress is checked
through the browser. Use this when subscription access depends on the IP range.

#### Remote mode (HTTP channels on another host)

When the machine running the pipeline is not on the institution's network but another host you can reach with
`ssh` is, set `fulltext.remote` (`enabled`, `host`, `python`, `workdir`). `tbsgen fetch` then sends the due papers
as one job to that host, which runs the same tbsgen code (`python -m tbsgen.remote`) through the `oa` / `elsevier` /
`wiley` / `inst` channels, and copies the texts back into the local `write_cache_dir`. The local side makes no
publisher request.

- Same code on both sides: before leasing anything, `tbsgen fetch` refuses to start a job if the local package
  (`gen/src/tbsgen`) has uncommitted changes (`remote.require_clean_tree`) or if the remote host's code
  fingerprint (a hash of the package sources) differs (`remote.preflight`). It then exits with code 3, prints the
  reason and points at `tbsgen remote deploy`; the papers stay due and are not charged a try. A job the remote side
  refuses at submit is reported the same way (`remote_error` in the printed stats).
- `tbsgen remote deploy [--dry-run]` bundles the local `HEAD` (`git bundle`), copies it with `scp` to
  `remote.code_dir`, runs `git fetch` + `git checkout --detach <sha>` and `remote.deploy_sync` (default
  `uv sync -q --no-dev`) there, then checks that the remote `HEAD` and fingerprint equal the local ones. It refuses
  while the package has uncommitted changes or a remote job is not collected yet (`--force` overrides the latter).
  `--dry-run` prints the plan and contacts nothing. `tbsgen remote status` prints both sides' `HEAD` and
  fingerprint and the remote jobs (exit 2 when the code differs).
- The remote host must have `egress_check` on with `allow_cidrs`; the job is refused otherwise.
- Secrets (`ELSEVIER_API_KEY`, `WILEY_TDM_TOKEN`) are read locally (environment, or the command in
  `fulltext.remote.secrets_cmd`, which gets the wanted names on stdin and prints a JSON object) and sent on the
  ssh connection's stdin. The remote worker keeps them in memory; they are never in a command line, file or log.
- One ledger: the local `ledger_path` stays the only authority, and one remote job at a time uses it. Before a job, the local ledger books one lease row
  per paper (counts against the global daily cap, so the browser cannot use the same quota) and sends a snapshot of
  today's usage, live cooldowns and pacing state; the remote job works on a copy seeded from it, so channel caps,
  breakers and cooldowns include the local usage. Afterwards the remote job's new rows are merged into the local
  ledger exactly once and the lease is dropped (a lease holds its quota until then, however long that takes).
  Cooldowns set locally while a job runs reach the remote side with the next job (`max_job_s` bounds the delay).
- Interruptions: the remote worker is detached, so an ssh drop does not stop it. The local job file
  (`<queue>.remote/<job>.json`) records the stage; the next `tbsgen fetch` first collects unfinished jobs. Results
  are written per paper (a text saved just before a crash is recovered from its provenance), so a worker that dies
  keeps what it finished; the rest stays queued and due. `fulltext.remote.ca_file` / `fulltext.ca_file` set a CA
  bundle when the interpreter has none.
- `fulltext.pdftotext` points at a `pdftotext` outside `PATH` (e.g. installed in a user directory).

#### Browser batches

`fulltext.browser.batch_only = true` keeps the browser out of normal `tbsgen fetch` runs; it then only runs in
`tbsgen fetch --browser`, one batch sized to an egress window that is open for a limited time (`browser.batch`):

```bash
uv run tbsgen fetch --run runs/R --config config.toml --browser --wait-for-egress   # then open the egress
```

- The browser's egress is read from an IP-echo page (`batch.watch_urls`, e.g. a publisher's `/cdn-cgi/trace`)
  opened in the browser itself, every `poll_s` (>= 5) seconds, each echo host at most every `probe_host_gap_s`.
  Probes are not downloads. Without `--wait-for-egress` one probe decides and an outside egress refuses the batch
  (exit 2); with it the command waits up to `max_wait_s` and exits 0 if the egress never becomes allowed.
- The window starts at the first allowed probe and ends `window_s - margin_s` later. The batch takes at most
  `window / (est_script_s + agent_fraction * est_agent_s)` papers (and `batch_max`, `per_publisher_max`, the
  ledger's remaining browser quota), publishers interleaved. During the batch the egress is checked every
  `recheck_s` and after every download; when it leaves `allow_cidrs` the batch stops, files obtained since the
  last good probe are discarded and the remaining papers stay queued.
- Each paper is first tried with the reviewed script. With `browser.agent.enabled`, the papers the script could
  not get go to one Hermes CLI run (`hermes chat --oneshot`, `agent.provider` / `model` / `max_turns` /
  `run_budget_s`) with a prompt generated from [`browser_agent_prompt.md`](gen/src/tbsgen/browser_agent_prompt.md).
  The agent writes no browser code. Its only tool is `tbsgen-browser N <action>` (plus `vision_analyze` for the
  batch's own screenshots), each call running the reviewed action script
  [`browser_agent_action.mjs`](gen/src/tbsgen/browser_agent_action.mjs) in a TaskSpace of its own per paper.

  | Action | What it does |
  |---|---|
  | `look [full] [@REF] [OFFSET]` | page summary: URL (no query), title, PDF? verification page? login form?, text excerpt, numbered links / buttons / fields of the page, same-origin iframes and (via the ego page snapshot) cross-origin iframes; `full` whole page, `@REF` one subtree |
  | `open URL` | navigate (allowed domains only) |
  | `click I` | click item I (also inside same-origin iframes and links inside cross-origin iframes; a button inside a cross-origin iframe after opening the frame item; `_blank` links by a native click, new tabs tracked) |
  | `click_xy X Y` | native click at viewport coordinates (the element there is checked first) |
  | `hover I` | native pointer onto item I (dropdown menus) |
  | `scroll up\|down [PX]` | scroll (wheel fallback) |
  | `select I VALUE` / `type I TEXT` | choose an option / type text (never into password, one-time-code, account or e-mail fields) |
  | `screenshot` | PNG of the viewport with every field masked; the agent views it with `vision_analyze` |
  | `tabs` / `switch LABEL` | list / select the paper's pages (never your own tabs) |
  | `back` | one history step, same-origin and allowed host only |
  | `wait_for url\|sel ...` | wait for a URL / selector (<= 60 s) |
  | `save_pdf` | save the current PDF page |
  | `sso_login` | on `browser.sso_login.domains`: press Login on the browser-autofilled form (values never read; at most `max_submissions` per batch, counted before the click, no retry after a failure) |
  | `handoff_status` / `finish` | poll a hand-off / close the workspaces |

  Hard rules, enforced in code:
  - **credential values never reach the model, logs or anything outside the browser**: field values are never
    read in the page; the ego page snapshot is reduced inside the action script before use (every field's text
    is dropped); screenshots mask every field and are refused on login pages and on pages with fields the mask
    cannot reach (cross-origin frames, closed shadow roots); `type` / `select` refuse credential fields; every
    URL in the tool's output and in the agent's stderr loses its userinfo, query and fragment; stderr records
    with credential-like keys are dropped;
  - **no automatic CAPTCHA / Cloudflare / Turnstile solving**: verification pages and widgets are never clicked;
    with `browser.handoff.enabled` they are handed to you (macOS notification, optional `handoff.bark_cmd`);
  - **navigation stays inside the paper's `allowed_domains`** (publisher, `doi.org`, PDF CDNs,
    `browser.sso.domains`, `browser.sso_login.domains`): `open`, link / form / frame targets, `back` and
    downloads are checked before the request, and a page that ends up outside (redirect) is reset to
    about:blank as soon as it lands;
  - **files are validated and booked by code**, never by the agent's report; screenshots are never deliverables;
  - budgets: at most `agent.max_calls_per_paper` (25) calls and `agent.max_navigations_per_paper` (12)
    page-changing calls per paper, page loads of one publisher `agent.nav_gap_s` (2.5 s) apart; one hand-off per
    paper and per publisher per batch; a hand-off that is not cleared in time (or whose call lost its result)
    closes the paper and cools the publisher; budgets are reserved under a lock before the action script runs.
  - On pages with a login signal (password / one-time-code / account fields, autofill, login wording, SAML
    hand-back) `click`, `click_xy`, `hover`, `select` and `type` are refused; only `sso_login` (on the SSO
    domain) or the hand-off act there.

  The run uses an isolated Hermes home (no user memory, rules, plugins or credentials; only Hermes' own
  dependency environments, `installs/`, are linked from your Hermes data root so the CLI can start, or from
  `agent.hermes_installs_dir`) whose `pre_tool_call` hook blocks every other command before it runs; repeated
  attempts end the run. The agent's stderr is kept (tail, credential-like strings masked; a multi-line record,
  such as a traceback, a JSON object or a multi-line value, with a credential-like field is dropped whole, and
  after a sensitive value whose end cannot be known the rest is dropped) in the batch directory
  as `agent-stderr.log` and in the `agent_end` log event. Saved files are validated and booked by code; the
  agent's own report is not used. Papers cut by an egress change or the window's end stay queued without
  backoff. Model usage per run is appended to `<queue>.browser/usage.jsonl`.

  Known limitations (residual risk of the full action set, accepted by design):
  - a cross-domain navigation the page itself starts (script or HTTP redirect) cannot be intercepted with the
    browser SDK; it is reset only after it lands (nothing from it is saved);
  - `click_xy`, native clicks on `_blank` links and actions inside cross-origin iframes act through the browser's
    hit-testing / snapshot refs: the element is checked right before, but a page can move content between the
    check and the native event;
  - a click / submit handler of an allowed page can still do anything after the checks (e.g. `fetch`, rewrite a
    form); form submissions to allowed hosts are allowed;
  - login detection is heuristic; a page it misses is still protected by the field rules (`type` refuses
    credential fields, values are never read), not by the hand-off;
  - `sso_login` presses the autofilled Login button after checking the form posts (POST) to an SSO login
    domain; the page's own handlers could change the target after that check (it then counts as a failed
    login and is not retried);
  - the ego SDK returns the page snapshot (used for `look` items inside cross-origin frames and for the screenshot
    field check) only as full text to the action script's Node process: it is reduced line by line inside one
    function and never returned, printed or logged, but the raw text does exist briefly in that local process;
  - not run end to end on real publisher or institution sites yet.

```toml
[fulltext.browser.handoff]
enabled = true                 # hand verification / login pages to the person at this machine
[fulltext.browser.sso]
domains = ["carsi.edu.cn", "openathens.net", "shibboleth.net"]   # institution / WAYF
[fulltext.browser.sso_login]
enabled = true
domains = ["login.example.edu"]   # the institution's own login host (HTTPS); autofill-only login
max_submissions = 2
[fulltext.browser.agent]
enabled = true
```

Terms: use only content your own subscriptions and API licences allow. No channel bypasses access control.

#### TDM compliance (Elsevier, Wiley)

The [Elsevier TDM provisions](https://dev.elsevier.com/policy/TDM-provisions.pdf) and the
[Wiley TDM terms](https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining) allow non-commercial
text mining by subscribers. Among their conditions: output may quote at most 200 characters of a mined article
(Elsevier, with a DOI link); the content must not be redistributed or substantially reproduced; and the downloaded
content must be deleted when the project or the agreement ends. How this pipeline follows them:

- Full texts stay on the machine that runs the pipeline, as grounding input for the teacher model. They are never
  part of a packaged task or of this repository (run directories are git-ignored).
- Every text's provenance record names its channel and licence class (`elsevier_tdm`, `wiley_tdm`, `oa`, `inst`,
  `browser`); items record it in `S3_fulltext.json` (a cache hit records the channel that filled the cache).
- S7 gate `tdm_verbatim` (blocking): for a paper whose text came from a TDM channel
  (`gates.tdm_verbatim_channels`), the longest verbatim overlap between the full text and the public task files
  (`instruction.md` and the text files under `environment/data`), compared after whitespace normalisation, must not
  exceed `gates.tdm_verbatim_max_chars` (200). A longer overlap goes to S8 repair like any other blocking gate.
- `tbsgen fulltext purge --run DIR --channel elsevier,wiley [--before YYYY-MM-DD] [--dry-run]` deletes the texts
  and raw PDF/XML files of those channels from the write cache, and the copies held by finished items of that run.
  The ledger rows and provenance records stay (they hold no article text; the record gets `purged_at`). Items
  still before S6 keep their copy and are listed; read-only caches are never written, matches there are counted
  so their owner can delete them.

```toml
[fulltext]
allow_network = true
write_cache_dir = "/data/fulltext"
[fulltext.elsevier]
enabled = true                 # key from env ELSEVIER_API_KEY
[fulltext.wiley]
enabled = true                 # token from env WILEY_TDM_TOKEN
[fulltext.inst]
enabled = true
routes = [{prefix = "10.1007/", url = "https://link.springer.com/content/pdf/{doi}.pdf"}]
[fulltext.browser]
enabled = true
concurrency = 2
[fulltext.limits.elsevier]
daily_ok = 100                 # tighten any cap per channel
```

### Network of packaged tasks

An agent that can reach the internet while it solves a task can `pip install` the paper's own library or
download a finished implementation, which defeats the task. Harbor (checked against 0.23.0: `harbor.models.task.config`,
`harbor.environments`, `harbor run --help`; [github.com/laude-institute/harbor](https://github.com/laude-institute/harbor))
lets a task set a network policy per phase in `task.toml`: `network_mode = "public" | "no-network" | "allowlist"` with `allowed_hosts`, in
`[environment]` (the default) and in `[agent]` / `[verifier]` (overrides for those phases), and the launcher can
add hosts for the agent phase only (`harbor run --allow-agent-host HOST`, or `extra_allowed_hosts` of an agent in
a job config). Installed agents such as Claude Code or Codex run inside the task container and must reach their
model API, so plain `no-network` would stop them.

`package.network.mode` (default `"agent-allowlist"`) writes:

```toml
[verifier]
network_mode = "no-network"
[agent]
network_mode = "allowlist"       # while the agent works: only package.network.agent_hosts ...
allowed_hosts = []               # ... plus the model endpoint the launcher adds per run
[environment]
network_mode = "public"          # agent setup (installing the agent CLI) only
```

Dependencies are installed when the image is built. S9 adds `calib.agent_allowed_hosts` (your model endpoint's
host) to the job config; with `calib.allow_submit = true` and no host configured, S9 refuses to launch rather than
start an agent that cannot reach its model. `"no-network"` (offline throughout; oracle checks only) and `"public"`
(the old behaviour, for comparison) are also available. `package.network.disable_index = true` additionally strips
pip / uv / conda index settings from the image after the build (off by default because some agent setups
pip-install their own runtime; it is a speed bump, the allowlist is the barrier).

Enforcement depends on the Harbor environment. In Harbor 0.23.0 the local `docker` environment enforces
allowlists and per-phase changes with an egress-control sidecar (tested here on Docker Desktop for macOS: during the
agent phase PyPI and GitHub were unreachable and `pip download` failed; with `--allow-agent-host pypi.org` only
PyPI opened); the E2B, Novita, Beam and Runta environments declare the same capability, others only static
allowlists or none. Before a trial starts, Harbor refuses a policy the environment cannot enforce. A custom
environment (`calib.environment`) must declare and honour these capabilities itself.

A separate gate, `readymade` (S7, blocking), checks that the task does not hand the agent the paper's own
implementation: the task image's declared dependencies (requirements files, `pip install` lines in a
Dockerfile), imports of the reference solution and the baselines, and code shipped under `environment/` are
matched against the repository and package names in the paper's `code_links` / `data_links` and the method
names of the S5 spec. A hit goes to S8 repair (drop the dependency, implement the method, or change the task).
Known companion repositories are written to the task's `provenance.json` (`companion_code`) for calibration and
human review.

Scope and known limits: `readymade` is a name-based rule check. It recognises requirement specifiers (`name==v`,
`name @ url`, `-e`, `#egg=`), VCS and direct archive URLs/files (version and `#sha256=` fragments removed),
GitHub/Bitbucket/Codeberg/Hugging Face `owner/repo`, GitLab `group/subgroup/.../repo`, PyPI `/project/<name>`,
CRAN `package=<name>` and `/web/packages/<name>`, and shipped source, archive and compiled files (`.so`, `.pyd`,
`.pyc`, ...) whose name matches, ignoring case, `-`/`_`/`.`, version or branch suffixes and `py` prefixes or
suffixes. It does not promise to find an implementation that was renamed, vendored under another name,
obfuscated, or fetched at run time some other way; calibration and human review cover those.

### Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `TBS_TEACHER_URL` | gen | Teacher endpoint URL (Anthropic-compatible `/v1/messages`). No default: set it here or with `models.teacher.url`; real model stages pause with `model:config` until it is set. |
| *your `key_env`* | gen | Teacher API key, if `models.teacher.key_env` is set. |
| `TYPESAFE_API_KEY` | gen (S2) | TypeSafe key for the Jev screen (`models.roles.screen = "jev"`). The variable name is `screen.jev.key_env`. Unset: S2 falls back to the rules screen (or parks, see `screen.jev.on_unavailable`). |
| `ELSEVIER_API_KEY`, `ELSEVIER_INSTTOKEN` | gen (S3) | Elsevier TDM channel (`fulltext.elsevier.key_env` / `insttoken_env`). Unset: the channel is skipped. |
| `WILEY_TDM_TOKEN` | gen (S3) | Wiley TDM channel (`fulltext.wiley.token_env`). Unset: the channel is skipped. |
| `TBS_HARBOR_ROOT` | gen (S9) | Directory with your Harbor job launch scripts for calibration (`calib.harbor_root`). Unset: S9 prints a plain `harbor run --config ...` command. |
| `TBS_HARBOR_BIN` | gen | Path to the `harbor` binary for `tbsgen harbor-local` (otherwise looked up on `PATH`). |
| `TBS_SEARCH_POOL_URL` | retrieval | Search-pool service for Exa (`sources.exa.backend = "pool"`, the default). No default: unset means Exa is skipped. Or set `backend = "direct"` and use `EXA_API_KEY_SECONDARY`. |
| `OPENALEX_API_KEY` | retrieval | Optional; OpenAlex also works without a key, at lower rate limits. |
| `SEMANTIC_SCHOLAR_API_KEY` | retrieval | Required for Semantic Scholar; the source is skipped without it. |
| `EXA_API_KEY_SECONDARY` | retrieval | Exa key, used with `sources.exa.backend = "direct"`. |
| `TBSGEN_METRICS`, `TBSGEN_ENTRY`, `TBSGEN_HIDDEN` | generated tests | Set by the local test runner inside the sandbox; you do not set these. |

Crossref, arXiv and Europe PMC need no key. `uv run tbsretr status` lists which sources are usable;
it shows whether a key is present, never its value.

### Optional external settings

There are no machine-specific defaults. Unset settings switch the corresponding feature off:
the teacher URL (real model stages pause until it is set), the Exa search pool (Exa skipped),
`fulltext.cache_dirs` (read-only full-text caches; none by default), `probe.tbs_tasks_root` (TB-Science task
texts for deduplication; none: no deduplication), `leak_refs` in the retrieval config (papers TB-Science tasks
were built from, excluded from retrieval; none: no exclusion), and the S9 launcher
settings under `[calib]` (`harbor_root`, `student_launcher`, `teacher_launcher`). Without launchers,
S9 writes the job config and prints a plain `harbor run --config <file> --yes` command, and
`allow_submit = true` refuses to launch. Calibration jobs use Harbor's built-in `claude-code` agent
and `docker` environment unless `student_agent`, `teacher_agent` or `environment` name a custom
`module:Class`. None of this affects the tests or the dry run.

All other keys and their defaults are in [`gen/src/tbsgen/config.py`](gen/src/tbsgen/config.py) and
[`retrieval/src/tbsretr/config.py`](retrieval/src/tbsretr/config.py), and are explained in the two
package READMEs. Commonly changed ones: `fulltext.cache_dirs` (directories of full texts named
after the DOI), `fulltext.allow_network` (all network channels of S3) and the per-channel `fulltext.*` switches, `calib.allow_submit`
(default `false`: only write calibration job configs), `calib.prepared_tasks_dir`,
`sandbox.image`, and the per-source daily budgets of the retrieval layer.

## Status and limitations

Research code under active development.

**Tests** (run by [GitHub Actions CI](.github/workflows/ci.yml) on every push to `main` and every pull request)

- `retrieval/`: 47 offline unit tests that replay recorded API responses.
- `gen/`: 269 tests: 238 offline unit tests (18 for the Jev screen, 6 for the S2 publisher blocklist and 161 for
  the S3 full-text channels and browser batches, using recorded and scripted responses and fake `ego-browser` /
  `hermes` executables; no network, no real browser or model), 16 integration tests that run task code in
  Docker, and 15 live tests of the browser agent's action script against local pages in a real ego-browser
  (no website is contacted; skipped where ego-browser is not installed, as in CI). Locally the integration
  tests are skipped when Docker or the sandbox image is missing; CI builds the image and fails if any of them
  is skipped.
- Fake-model dry run: S1 → S10 on a synthetic toy task, including the mechanical gates and
  simulated calibration results.

**Evaluations** (separate from the tests)

- Leak-gate backtest on previously generated tasks, no LLM calls
  ([`gen/eval/LEAK_BACKTEST.md`](gen/eval/LEAK_BACKTEST.md)): 7/7 known leaking tasks caught;
  0/231 blocking false positives in-sample (the rules were tuned on this set) and 1/81 on a
  held-out set.
- S2 screen evaluation ([`gen/eval/screen_jev/`](gen/eval/screen_jev/README.md)): rules vs Jev on 200 sampled
  abstracts with reference labels from a teacher model (not human labels), and on a larger set of papers whose
  downstream outcome in an earlier pipeline is known.
- Retrieval evaluation on 10 targets ([`retrieval/eval/REPORT.md`](retrieval/eval/REPORT.md)).
  Relevance labels come from a single annotator and are subjective. Historical evaluation: it includes the
  Tavily source, which has since been removed from the code.
- End-to-end runs on 6 real papers: 3 became packaged
  Harbor tasks where the reference solution scores 1 and an empty submission scores 0; 2 were
  dropped because the mid baseline could not be separated from the reference; 1 was still in
  repair. Each of the 3 packaged tasks was calibrated with a single student run, which is not
  enough to assign difficulty labels.

**Known limitations**

- Packaged tasks keep the agent off the internet with Harbor's per-phase network policy; this needs a Harbor
  environment that can enforce it (Docker with Harbor's egress sidecar, or Harbor's E2B-based sandboxes).
  Other environments refuse the trial instead of running it open. The model endpoint the agent needs stays
  reachable, as does anything else on the allow list. Agent setup (installing the agent CLI) runs with network.
- The ready-made-implementation check is rule-based: it matches the paper's code/data links, method names and
  declared dependencies. A package or repository with an unrelated name, or code fetched through an allowed host,
  is not detected.
- The teacher-agent branch of calibration has not been run on real tasks.
- The L4b inversion attack, the cheap-model roles other than the S2 screen, harder-variant generation (S9b) and the
  file-format and interactive task forms are not implemented.
- The expert-time estimate in `task.toml` is the teacher model's own estimate.
- The S3 publisher channels (TDM APIs, institution IP, browser) are off by default and need your own licences;
  their offline tests use recorded or scripted responses only. The browser agent's exploration actions,
  hand-off and institution sign-in are tested against local pages only; on real publisher and institution
  sites they have not been run end to end yet; see "Known limitations" of the browser agent above.
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
