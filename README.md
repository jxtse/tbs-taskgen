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

On a fresh machine both items may stop at S3 as `parked: fulltext_unavailable` (listed in
`S3_queue.json` and in the run's full-text queue): the pipeline needs the paper's full text. S3 tries a
configurable cascade (see [Full text (S3)](#full-text-s3)); by default only a local cache and open-access copies
are used. `tbsgen fetch --run runs/R` retries the queue later (failed entries back off, doubling from
`fulltext.queue_backoff_s`), hands the text to every waiting item (also items of other runs sharing the queue) and
moves them on to S4; an entry leaves the queue only after that. Only one `tbsgen fetch` drains a queue at a time, and only one
process at a time uses a browser TaskSpace (a second one skips the browser channel). With full text, a teacher endpoint and a
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

### S2 abstract screen

S2 decides from the bibliographic record (title, abstract, venue, year, code/data links) whether a paper is worth
downloading and probing. Two modes, chosen with `models.roles.screen`:

- `"rules"` (default): keyword cues in title and abstract (`screen.rules_version = "v2"`, keep at
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

Before either mode, S2 drops papers whose full text the S3 channels have never been able to obtain
(`screen.blocklist`, matched on DOI prefix and optionally publication year): Elsevier before 2000, SAGE, the
American Meteorological Society, Powder Diffraction, ASCE and SPIE. Such papers get `method = "blocklist"` and
`reason = "blocklist:<rule>"` in `S2_screen.json` and cost no screen call. Set `screen.blocklist = []` to disable,
or edit the list to add or remove publishers.

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
| `wiley` | Wiley TDM API (PDF), prefixes `10.1002/`, `10.1111/` | token in `WILEY_TDM_TOKEN`, accepted Wiley TDM licence | off |
| `inst` | Publisher PDF URLs your institution's IP range is entitled to, configured as `fulltext.inst.routes` (`{prefix, url}`); empty by default | network on the institution's IP range | off |
| `browser` | A local [ego-browser](https://ego.app) (Ego Lite) session with your signed-in profile, driven by the reviewed script [`browser_fetch.mjs`](gen/src/tbsgen/browser_fetch.mjs) | `ego-browser` on `PATH` | off; when on, S3 only queues and `tbsgen fetch` runs it |

Checks: a download is accepted only if it is a PDF (or full-text XML / an HTML article page with section
headings), its text has at least `min_chars` characters, and at least `title_match_min` of the title words appear
in its first 6,000 characters. HTML wrapper pages, abstract-only API answers and bot-wall pages are rejected.
Accepted texts go to `fulltext.write_cache_dir` (default `<run>/fulltext_cache/`) as `<doi_key>.txt` with a
`<doi_key>.json` provenance record (channel, URL without query string, time, bytes, SHA-256, title match).
Read-only caches are never written.

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
whose links point at many unrelated hosts, opens only after 10 block signals and never on failures alone (both
configurable per channel: `breaker_blocks_per_day`, `breaker_consecutive_fail`). A cooldown also stops requests that
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
as one job to that host, which runs the same tbsgen code (`python -m tbsgen.remote`; both sides compare a hash of
the package sources and the job is refused on a mismatch) through the `oa` / `elsevier` / `wiley` / `inst`
channels, and copies the texts back into the local `write_cache_dir`. The local side makes no publisher request.

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
  The agent writes no browser code. Its only tool is `tbsgen-browser N look | open URL | click I | save_pdf` (and
  `finish`), each call running the reviewed action script
  [`browser_agent_action.mjs`](gen/src/tbsgen/browser_agent_action.mjs): the paper is booked in the ledger before
  its first call; only the paper's publisher domains are opened or clicked; nothing is typed; verification pages,
  verification widgets and login forms are never interacted with; a bot wall cools the publisher at once.
  Known limit: URLs, links and form targets are checked before the request, but a cross-domain navigation that
  the page itself starts (script or HTTP redirect) cannot be intercepted with the browser SDK; such a page is
  reset to about:blank as soon as it lands and reported, and nothing from it is saved. The run
  uses an isolated Hermes home (no user memory, rules, plugins or credentials) whose `pre_tool_call` hook blocks
  every other command before it runs; repeated attempts end the run. Saved files are validated and booked by
  code; the agent's own report is not used. Papers cut by an egress change or the window's end stay queued
  without backoff. Model usage per run is appended to `<queue>.browser/usage.jsonl`.

Terms: use only content your own subscriptions and API licences allow. The Elsevier and Wiley TDM licences are
for non-commercial text mining by subscribers; do not redistribute full texts. This pipeline only uses the text
to ground generated tasks and does not copy it into task files. No channel bypasses access control.

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

### Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `TBS_OPUS_URL` | gen | Teacher endpoint URL (Anthropic-compatible `/v1/messages`). No default: set it here or with `models.opus.url`; real model stages pause with `model:config` until it is set. |
| *your `key_env`* | gen | Teacher API key, if `models.opus.key_env` is set. |
| `TYPESAFE_API_KEY` | gen (S2) | TypeSafe key for the Jev screen (`models.roles.screen = "jev"`). The variable name is `screen.jev.key_env`. Unset: S2 falls back to the rules screen (or parks, see `screen.jev.on_unavailable`). |
| `ELSEVIER_API_KEY`, `ELSEVIER_INSTTOKEN` | gen (S3) | Elsevier TDM channel (`fulltext.elsevier.key_env` / `insttoken_env`). Unset: the channel is skipped. |
| `WILEY_TDM_TOKEN` | gen (S3) | Wiley TDM channel (`fulltext.wiley.token_env`). Unset: the channel is skipped. |
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
after the DOI), `fulltext.allow_network` (all network channels of S3) and the per-channel `fulltext.*` switches, `calib.allow_submit`
(default `false`: only write calibration job configs), `calib.prepared_tasks_dir`,
`sandbox.image`, and the per-source daily budgets of the retrieval layer.

## Status and limitations

Research code under active development.

**Tests** (run by [GitHub Actions CI](.github/workflows/ci.yml) on every push to `main` and every pull request)

- `retrieval/`: 47 offline unit tests that replay recorded API responses.
- `gen/`: 156 tests: 140 unit tests (17 of them for the Jev screen and 70 for the S3 full-text channels, using
  recorded and scripted responses; no network, no browser) and 16 integration tests that run task code in Docker. Locally
  they are skipped when Docker or the sandbox image is missing; CI builds the image and fails if
  any of them is skipped.
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
- The L4b inversion attack, the cheap-model roles other than the S2 screen, harder-variant generation (S9b) and the
  file-format and interactive task forms are not implemented.
- The expert-time estimate in `task.toml` is the teacher model's own estimate.
- The S3 publisher channels (TDM APIs, institution IP, browser) are off by default and need your own licences;
  their tests use recorded or scripted responses only.
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
