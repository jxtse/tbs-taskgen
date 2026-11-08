"""Configuration for the v2 task-generation pipeline.

Every value not decided yet ("待定") is a config key with a conservative default; the
README lists them.  Budgets default to None: a *real* model cannot be called until the daily budget
for it is set explicitly (fake models are exempt).

Load order: DEFAULTS <- config.toml (optional) <- explicit overrides dict.
"""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]          # repo root (gen/ and retrieval/ live here)
# Optional external locations.  No machine-specific defaults: unset -> the feature that needs them is off
# (no full-text cache, no TB-Science dedup, no launcher-based calibration submit).  Set them in config.toml
# (fulltext.cache_dirs, probe.tbs_tasks_root, calib.harbor_root; TBS_HARBOR_ROOT also works for the last).
HARBOR_ROOT = os.environ.get("TBS_HARBOR_ROOT") or None     # directory holding your Harbor launcher scripts

DEFAULTS: dict[str, Any] = {
    "run": {
        "root": None,                       # run directory; required
        "tasks_per_paper": 1,               # MVP: 1 task per paper (variants = S9b, disabled)
    },
    # ---------------------------------------------------------------- budget / breaker
    "budget": {
        # USD per UTC+8 day.  None = not configured -> real model calls refuse to start.
        "daily_usd_total": None,
        "daily_usd_per_model": {},          # e.g. {"claude-opus-5-5": 50.0}
        "daily_usd_per_stage": {},          # e.g. {"S6A": 20.0}
        "daily_calls_per_model": {},        # hard call caps, e.g. {"claude-opus-5-5": 6}
        "run_usd_total": None,              # whole-run caps (all days; daily caps reset at 0:00 UTC+8)
        "run_calls_per_model": {},
        "max_calls_per_item": 24,           # all stages, all attempts, one item
        "breaker_consecutive_errors": 4,    # N consecutive failed calls -> breaker opens for that model
        "breaker_open_on": ["quota", "auth"],  # these error classes open the breaker immediately
        "est_usd_per_call": 1.5,            # pre-call reservation used for the "would exceed" check
    },
    # ---------------------------------------------------------------- models
    "models": {
        "cheap": None,                      # 待定: cheap-model role; None -> rules-only fallbacks
        "roles": {                          # stage role -> model alias ("teacher"/"cheap"/"rules"/explicit)
            "screen": "rules",              # S2: "rules" (rules-v2, intake.screen) | "jev" (TypeSafe Jev, screen.jev)
            "probe": "rules",               # S4 (待定 cheap agent; sandbox repro not implemented in MVP)
            "spec": "teacher",              # S5
            "gen": "teacher",               # S6A
            "tests": "teacher",             # S6C
            "repair": "teacher",            # S8
            "review": "rules",              # S8 preservation review (mechanical diff; cheap model 待定)
            "invert": None,                 # S7 L4b cheap-agent inversion attack (待定; disabled)
        },
        # The teacher model (S5 / S6 / S8): an Anthropic-compatible /v1/messages endpoint.  No default URL: set
        # models.teacher.url or TBS_TEACHER_URL.
        "teacher": {"model": "claude-opus-5-5", "url": os.environ.get("TBS_TEACHER_URL") or None,
                 "effort": "medium", "max_tokens": 32000, "timeout": 1800, "retries": 2,
                 # per-stage effort override, e.g. {"S6A": "low"}; empty -> "effort" for every stage
                 "effort_by_stage": {},
                 # empty answer with stop_reason=max_tokens (adaptive thinking used the whole output budget, seen on
                 # S6A at 64k with effort medium): one retry at this effort if it is lower than the call's; None -> no retry
                 "effort_on_thinking_exhausted": "low"},
        # USD per million tokens.  PLACEHOLDERS: set real prices for your model.
        "prices": {"claude-opus-5-5": {"in": 5.0, "out": 25.0, "cache_read": 0.5},
                   # TypeSafe list price: input tokens only, $0.042 per million (output not billed)
                   "jev-1.14.0": {"in": 0.042, "out": 0.0, "cache_read": 0.0},
                   "jev-latest": {"in": 0.042, "out": 0.0, "cache_read": 0.0},
                   "fake": {"in": 0.0, "out": 0.0}},
    },
    # ---------------------------------------------------------------- stages
    "screen": {
        # rules (rules-v2): each cue stem counts once, "survey" is negative only as a literature survey.  Keep
        # threshold picked on the dev half of the S2 reference set (gen/eval/screen_jev/).
        "min_score": 1,
        # Papers whose full text S3 has rarely or never obtained (history per rule below), dropped at S2 before
        # any screen call.  Match = DOI prefix [+ published before `before_year`].  Set to [] to disable.
        "blocklist": [
            {"name": "elsevier-pre2000", "doi_prefix": "10.1016", "before_year": 2000,
             "why": "Elsevier TDM returns metadata only for pre-2000 volumes (0/11)"},
            {"name": "sage", "doi_prefix": "10.1177", "why": "no TDM/OA route; 0/3"},
            {"name": "ams-meteorology", "doi_prefix": "10.1175", "why": "American Meteorological Society; 0/3"},
            {"name": "powder-diffraction", "doi_prefix": "10.1154", "why": "Powder Diffraction (Cambridge/ICDD); 0/3"},
            {"name": "asce", "doi_prefix": "10.1061", "why": "Cloudflare-walled, no TDM; 1/4"},
            {"name": "spie", "doi_prefix": "10.1117", "why": "mostly proceedings, no TDM/OA; 1/4"},
        ],
        "jev": {
            "model": "jev-1.14.0",          # pinned version (thresholds were set on it); "jev-latest" = alias
            "url": "https://api.typesafe.ai/v1/systemone",
            "key_env": "TYPESAFE_API_KEY",  # env var holding the key; inject from your secret store
            "proxy": "env",                 # "env": honour HTTPS_PROXY; "none": direct connection
            "timeout": 30,                  # seconds per HTTP attempt
            "max_retries": 4,               # 429 / 529 / 5xx / network: exponential backoff + retry-after
            "backoff_base": 1.0, "backoff_max": 30.0, "retry_after_max": 120,
            "concurrency": 4,               # parallel requests when prefetching a batch of S2 items
            "uncertain": "keep",            # "keep" (flagged uncertain=true) | "drop"
            "on_unavailable": "rules",      # no key / breaker open / API down: "rules" (rules fallback) | "park"
            "est_usd_per_call": 0.001,      # ledger reservation per call (actual ~1.3k input tokens ~ $0.00006)
            "max_chars": 4000,              # full-text opening sent when the record has no abstract (S4 re-screen)
            "thresholds": {},               # overrides of screen_jev.THRESHOLDS
        },
    },
    "fulltext": {
        # Cascade (first success wins).  Every network channel has its own switch; only cache + oa are on by
        # default.  See fulltext.py and README (S3).
        "cascade": ["cache", "oa", "elsevier", "wiley", "inst", "browser"],
        "cache_dirs": [],                   # read-only full-text caches (<key>.txt), e.g. an earlier pipeline's
        "write_cache_dir": None,            # downloaded texts + provenance json; None -> <run.root>/fulltext_cache
        "keep_raw": True,                   # also keep the raw PDF/XML under <write_cache_dir>/raw/
        "ledger_path": None,                # download ledger (sqlite); None -> <run.root>/fulltext_ledger.sqlite
        "queue_path": None,                 # browser queue (jsonl); None -> <run.root>/fulltext_queue.jsonl
        # a queue entry whose channels made a request and got nothing waits this long (doubling, max 8x).  Papers
        # only refused by the ledger (caps, cooldowns, breakers) or the egress check are not backed off: they are
        # due again when the earliest refusal lifts (see fulltext.classify).
        "queue_backoff_s": 21600,
        "min_chars": 8000,
        "title_match_min": 0.6,
        "max_pdf_mb": 40,
        "timeout": 90,
        "allow_network": True,              # False: cache only (every network channel off)
        "use_env_proxy": False,             # False: urllib ignores HTTP(S)_PROXY (ProxyHandler({}))
        "mailto": None,                     # contact address put in the User-Agent
        # Refuse network downloads unless the public egress IP is inside allow_cidrs (e.g. the institution's
        # ranges, so subscription access and IP-based TDM entitlements apply).  Checked once per ttl_s, for
        # urllib and (separately, through the browser) for the browser channel.  When a proxy routes per domain,
        # list a publisher-domain URL that echoes the client IP (e.g. a Cloudflare-fronted publisher's
        # "/cdn-cgi/trace") first, so the check sees the route the downloads take.
        "egress_check": {"enabled": False, "allow_cidrs": [], "ttl_s": 600, "timeout": 15,
                         "urls": ["https://api.ipify.org", "https://checkip.amazonaws.com"]},
        "oa": {"enabled": True, "europepmc": True, "unpaywall_email": None, "max_urls": 3, "skip_hosts": []},
        "elsevier": {"enabled": False, "prefixes": ["10.1016/"], "key_env": "ELSEVIER_API_KEY",
                     "insttoken_env": "ELSEVIER_INSTTOKEN"},
        "wiley": {"enabled": False, "prefixes": ["10.1002/", "10.1111/"], "token_env": "WILEY_TDM_TOKEN"},
        # Publisher PDF URLs the institution's IP range is entitled to, e.g.
        # {prefix = "10.1007/", url = "https://link.springer.com/content/pdf/{doi}.pdf"}.  Empty by default.
        "inst": {"enabled": False, "routes": []},
        "browser": {
            "enabled": False,               # needs a local ego-browser (Ego Lite) with a logged-in profile
            "inline": False,                # False: S3 only queues; `tbsgen fetch` drains the queue
            "binary": "ego-browser",
            "space": "tbsgen-s3-fulltext",  # one fixed TaskSpace, reused
            "concurrency": 2,               # parallel publishers (one Page each); never 2 of one publisher
            "challenge_wait_s": 40,         # a challenge page must clear by itself within this, else cooldown
            "challenge_cooldown_s": 86400,
            "nav_timeout_s": 45, "download_timeout_s": 120, "script_timeout_s": 420, "max_navigations": 5,
            "download_wait_s": 6,           # extra wait for a download after the navigation settled
            "batch_only": False,            # true: the browser runs only in egress-gated batches (--browser)
            # Batch mode (`tbsgen fetch --browser`): the reviewed script first, then (agent.enabled) one Hermes CLI
            # agent run for the papers the script could not get.  See browser_agent.py.
            "batch": {
                "window_s": 600,            # egress window opened by the user (e.g. a VPN client paused for 10 min)
                "margin_s": 60,             # stop this long before the window ends
                "batch_max": 8,
                "per_publisher_max": 2,
                "require_http_first": True, # only papers the HTTP channels already tried (browser quota is small)
                "est_script_s": 60,         # planning estimates per paper (script run, agent run)
                "min_script_s": 20,         # no script is started with less time than this left in the window
                "est_agent_s": 120,
                "agent_fraction": 0.5,      # expected share of papers the script cannot get
                "poll_s": 5,                # --wait-for-egress: egress probe interval (>= 5 s; not a download)
                "max_wait_s": 1800,         # --wait-for-egress: give up (exit 0) after this
                "recheck_s": 30,            # egress re-check interval during a batch
                "probe_host_gap_s": 15,     # per echo host; rotate over several watch_urls for 5 s polling
                "watch_urls": [],           # IP-echo pages for the probes (rotated); [] -> egress_check.urls
                "refuse_hint": "",          # extra text printed when the browser egress is outside allow_cidrs
            },
            # Verification (bot wall) and login pages are never solved by the agent: with handoff.enabled the
            # tool hands the paper's TaskSpace to the user (macOS notification), waits up to wait_s for the
            # user to finish and continue (the agent works on other papers meanwhile), then re-checks the
            # page.  One hand-off per paper and per publisher per batch; the wait counts against the window.
            "handoff": {
                "enabled": False,           # repo default off; enable on the machine in front of the browser
                "wait_s": 120,              # max wait for the user (also capped by the window deadline)
                "notify": True,             # macOS notification when the control is handed to the user
                "bark_cmd": [],             # optional local notification command; off by default, never group chats
            },
            # Institution / WAYF domains every paper may navigate to (they are added to allowed_domains).
            "sso": {
                "domains": ["carsi.edu.cn", "openathens.net", "shibboleth.net"],
            },
            # Browser-autofill login on the institution's own SSO domains (also allowed for every paper):
            # `tbsgen-browser N sso_login` checks that Chromium autofilled the form (its :autofill state - values
            # are never read, returned or logged) and clicks Login; POST to the SSO domain only.  The submission
            # is counted BEFORE the click; no retry after a failure; at most max_submissions per batch; a
            # captcha / slider / SMS / MFA code or missing autofill hands the page to the user instead.
            "sso_login": {
                "enabled": False,           # repo default off; enable on the machine with the logged-in profile
                "domains": [],              # e.g. ["authserver.nju.edu.cn"]; HTTPS only
                "max_submissions": 2,       # per batch
            },
            "agent": {
                "enabled": False,
                "hermes_bin": "hermes",
                "provider": None,           # hermes --provider
                "model": None,              # hermes -m
                "provider_config": None,    # provider entry for the agent's isolated Hermes home (no api_key)
                "skills": [],               # skills linked into the agent's isolated Hermes home
                "max_violations": 3,        # blocked commands before the run is ended
                "max_turns": 80,
                "run_budget_s": 420,        # hermes --run-budget (also capped by the egress window)
                "extra_domains": ["doi.org"],   # always allowed besides the queued papers' publisher domains
                "gate_max_wait_s": 150,     # the per-paper gate waits at most this long for the publisher's slot
                "prompt_template": None,    # None -> packaged browser_agent_prompt.md
                "max_calls_per_paper": 25,  # ego-browser calls the agent may make per paper
                "max_navigations_per_paper": 12,   # page-changing calls per paper (open / click / click_xy / hover /
                                                   # scroll / select / type / back / sso_login / save_pdf)
                "vision": True,             # the agent may view the screenshots it saved (vision_analyze, batch only)
                "nav_gap_s": 2.5,           # min seconds between two page loads of the same publisher
                "call_timeout_s": 150,      # one ego-browser call of the agent
                "keep_batch_dir": False,    # keep prompt / wrapper state after the batch (debugging)
                "hermes_installs_dir": None,  # Hermes' dependency state; None -> <HERMES_HOME or ~/.hermes>/installs
            },
        },
        # Remote mode: run the HTTP channels (oa / elsevier / wiley / inst) on another host over ssh, e.g. a
        # machine on the institution's network.  The local ledger stays the only persistent ledger; see remote.py.
        "remote": {
            "enabled": False,
            "host": None,                   # ssh destination (alias from ~/.ssh/config)
            "python": None,                 # remote interpreter with tbsgen installed (absolute path)
            "workdir": None,                # remote data dir (job dirs live in <workdir>/jobs)
            "pdftotext": None,              # remote pdftotext binary (None -> "pdftotext" on PATH)
            "ca_file": None,                # remote CA bundle (None -> Python default, else a system bundle)
            "ssh": ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"],
            # a remote job starts no new paper after this; it also bounds how stale the job's copy of the ledger
            # state (cooldowns set meanwhile by the local browser) can get
            "max_job_s": 1800,
            "batch_max": 50,
            # Command printing a JSON object of secrets on stdout; it gets the wanted names (JSON list) on stdin.
            # None -> read them from this process's environment.  Secrets go to the remote worker on ssh stdin.
            "secrets_cmd": None,
            # Deployment (`tbsgen remote deploy` / `status`): the remote git checkout of this repository and the
            # dependency sync run in its gen/ directory after the checkout.
            "code_dir": None,               # e.g. "~/tbsgen/code"
            "deploy_sync": "uv sync -q --no-dev",
            "scp": ["scp", "-q", "-o", "BatchMode=yes"],
            # `tbsgen fetch` refuses to start a remote job (before anything is leased or sent) when the local
            # package has uncommitted changes or the remote code fingerprint differs; fix with `remote deploy`.
            "require_clean_tree": True,
            "preflight": True,
        },
        # A TDM API (elsevier / wiley) answering 200 with too little text (metadata / abstract only: the
        # institution does not subscribe to that volume): that (channel, DOI) is never asked again.
        # ttl_days: forget such records after this many days (None: never).
        "not_entitled": {"enabled": True, "channels": ["elsevier", "wiley"], "ttl_days": None},
        "pdftotext": None,                  # pdftotext binary; None -> "pdftotext" on PATH
        "ca_file": None,                    # CA bundle for HTTPS; None -> Python default, else a system bundle
        # Caps / pacing (dlledger.DEFAULT_LIMITS etc.); dicts merge per channel / host.
        "limits": {},
        "global_limits": {},
        "host_caps": {},
    },
    "probe": {
        "tbs_tasks_root": None,             # TB-Science task texts (<task>/instruction.md ...); None -> no dedup
        "decontam_cos_max": 0.30,
        "min_method_hits": 2,
    },
    "spec": {
        "allowed_forms": ["ANS", "PROG"],   # MVP; DOMF needs per-format readers; INTERACT banned until isolation (3rd phase)
        "prog_quota_frac": 0.5,             # 待定: max share of PROG among released tasks
        "spec_retries": 1,
        "excerpt_chars": 60000,
        "expert_hours": [2, 6],             # target expert time stated in the S5 prompt
        "max_tokens": 32000,
    },
    "gen": {
        "seeds": [0, 1],
        "gen_timeout": 300, "solve_timeout": 900, "baseline_timeout": 600, "test_timeout": 900,
        "max_public_mb": 50,
        "back_to_spec_max": 1,              # S6B "mid not separable" -> back to S5 at most once
        "s6a_max_tokens": 64000,            # S6A emits 5 files + adaptive thinking; 32k truncated in e2e-iter1
        "s6c_max_tokens": 48000,
        "mid_rework_max": 1,                # S6B mid not separable -> S6M rewrites only mid.py (before back to S5)
        "continue_calls": 1,                # missing files after a truncated multi-file answer -> 1 continuation
    },
    "thresholds": {
        "frac": 0.5,                        # 待定: threshold position between ref and mid (log-midpoint)
        "min_ref_margin": 0.10,             # reference must clear each threshold by >=10% (relative)
        "loose_margin": 0.50,               # non-separating metrics: threshold = ref worsened by 50%
        "max_mid_fail": 2, "min_mid_fail": 1,
        "sig_digits": 2,
    },
    "gates": {
        "naive_fail_min": 2,
        "leak_rtol": 1e-6,
        "l4a_spearman": 0.995,
        "l4b_minutes": 20,                  # cheap-agent inversion (待定, disabled)
        "max_repair_rounds": 2,
        "blocking": ["oracle", "hidden_shift", "leak", "isolation", "infeasible", "determinism", "truth_isolation",
                     "cli_contract", "tdm_verbatim", "readymade"],
        # readymade (prerelease.readymade_check): the task image installs, or the reference / a baseline imports,
        # the paper's own implementation (code_links repo names, method_names of the S5 spec).  Rules only.
        # TDM terms (Elsevier: quotes <= 200 chars; Wiley: no redistribution): longest verbatim overlap between
        # a TDM full text and the public task files (whitespace-normalised).  See tdmgate.py.
        "tdm_verbatim_max_chars": 200,
        "tdm_verbatim_channels": ["elsevier", "wiley"],
        "tdm_verbatim_unknown": False,      # True: a full text of unknown channel is checked too
    },
    "sandbox": {
        "backend": "docker",                # only docker is supported
        "image": "tbsgen-base:py311-sci-v2",
        "cpus": 2, "memory": "4g", "pids": 512,
    },
    # ---------------------------------------------------------------- calibration (S9)
    "calib": {
        "harness": "claude-code", "cc_version": "2.1.280",      # decision 2026-11-05
        "student_model": "glm-5.3",
        # Agent / environment: None -> Harbor's built-in "claude-code" agent and "docker" environment;
        # otherwise an import path "module:Class" for a custom Harbor agent / environment.
        "student_agent": None,
        "teacher_model": "claude-opus-5-5",
        "teacher_agent": None,
        "environment": None,
        # Launchers: scripts under harbor_root called as `./<launcher> run <job-config.json>`.  Unset -> the
        # printed command is a plain `harbor run --config ...`, and allow_submit=true refuses to launch.
        "harbor_root": HARBOR_ROOT,
        "jobs_dir": str(REPO_ROOT / "gen" / "state" / "harbor-jobs"),
        "student_launcher": None,
        "teacher_launcher": None,
        "allow_submit": False,              # must be flipped explicitly; True = reserve in ledger + tmux launch
        "est_usd_per_run": 20.0,            # ledger reservation per Harbor run; settled with the job's cost_usd
        "teacher_available": True,          # False: GLM#1 fail -> parked:teacher_unavailable (no teacher run)
        "prepared_tasks_dir": None,         # sandbox-prepared task copies (docker_image set), one dir per item
        # Hosts the agent may reach while it works when the task's agent phase is an allowlist
        # (package.network.mode = "agent-allowlist"): the model endpoint(s).  Written to the job config as
        # agents[].extra_allowed_hosts (= harbor run --allow-agent-host).  Empty with an allowlisted task: S9 does
        # not submit (the agent could not reach its model).
        "agent_allowed_hosts": [],
        "glm_runs_on_first_fail": 1,        # first GLM fail -> teacher (no 2nd GLM run)
        "teacher_runs_max": 2,
        "quick_pass_minutes": None,         # 待定: 1st-pass shortcut to "easy" (e.g. 30); None=off
        "keep_labels": ["hard", "unstable"],  # decision 2026-11-05: GLM 1/2, or GLM 0 + teacher pass
        "infra_retries": 2,                 # API quota / sandbox errors do not count as attempts
        "variant_depth": 0,                 # S9b harder variants; 待定 (e.g. <=2); 0 = off in MVP
    },
    "package": {
        "dockerfile_mode": "selfcontained",  # "selfcontained" (pip in Dockerfile, portable) | "local_base"
        "agent_hours": 2,
        "verifier_timeout": 900,
        # Network of the packaged task (prerelease.py).  "agent-allowlist": no internet while the agent works
        # except `agent_hosts` + the model endpoint added per run (calib.agent_allowed_hosts / --allow-agent-host);
        # agent setup runs before that with network (installs the agent CLI); the verifier is offline.
        # "no-network": offline throughout (oracle-only checks).  "public": old behaviour, comparison runs only.
        # disable_index: also strip pip/uv/conda indexes from the image (off: agent setups that pip-install their
        # own runtime would break; the allowlist already blocks package downloads).
        "network": {"mode": "agent-allowlist", "agent_hosts": [], "disable_index": False},
    },
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if p.exists():
            cfg = _merge(cfg, tomllib.loads(p.read_text()))
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def resolve_role(cfg: dict, role: str) -> str | None:
    """role -> model alias: 'rules' / None / 'fake:*' / 'jev' / 'teacher' / another models.<name> client table."""
    v = cfg["models"]["roles"].get(role, "rules")
    if v == "cheap":
        v = cfg["models"]["cheap"]
    return v
