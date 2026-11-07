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
# (no full-text cache, no TB-Science dedup, no launcher-based calibration submit).  Set the env vars or the
# corresponding config.toml keys.
_OLD_PIPE = os.environ.get("TBS_OLD_PIPE") or None          # earlier pipeline data: literature/txt, tasks_text
OLD_PIPE = Path(_OLD_PIPE) if _OLD_PIPE else None
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
        "teacher": "opus",                  # decision 2026-11-05: Opus
        "cheap": None,                      # 待定: cheap-model role; None -> rules-only fallbacks
        "roles": {                          # stage role -> model alias ("teacher"/"cheap"/"rules"/explicit)
            "screen": "rules",              # S2: "rules" (screen.rules_version) | "jev" (TypeSafe Jev, see screen.jev)
            "probe": "rules",               # S4 (待定 cheap agent; sandbox repro not implemented in MVP)
            "spec": "teacher",              # S5
            "gen": "teacher",               # S6A
            "tests": "teacher",             # S6C
            "repair": "teacher",            # S8
            "review": "rules",              # S8 preservation review (mechanical diff; cheap model 待定)
            "invert": None,                 # S7 L4b cheap-agent inversion attack (待定; disabled)
        },
        # url: Anthropic-compatible /v1/messages endpoint.  No default: set TBS_OPUS_URL or models.opus.url.
        "opus": {"model": "claude-opus-5-5", "url": os.environ.get("TBS_OPUS_URL") or None,
                 "effort": "medium", "max_tokens": 32000, "timeout": 1800, "retries": 2},
        # USD per million tokens.  PLACEHOLDERS: set real prices for your model.
        "prices": {"claude-opus-5-5": {"in": 5.0, "out": 25.0, "cache_read": 0.5},
                   # TypeSafe list price: input tokens only, $0.042 per million (output not billed)
                   "jev-1.14.0": {"in": 0.042, "out": 0.0, "cache_read": 0.0},
                   "jev-latest": {"in": 0.042, "out": 0.0, "cache_read": 0.0},
                   "fake": {"in": 0.0, "out": 0.0}},
    },
    # ---------------------------------------------------------------- stages
    "screen": {
        "rules_version": "v2",              # v2: each cue stem once, "survey" negative only as literature survey
        # rules keep threshold.  rules-v2 counts each cue stem once, so its scores are lower than rules-v1's for
        # the same text; 1 was picked on the dev half of the S2 reference set (gen/eval/screen_jev/).
        "min_score": 1,
        "min_score_v1": 2,                  # rules-v1 threshold (unchanged)
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
        "cache_dirs": [str(OLD_PIPE / "literature" / "txt")] if OLD_PIPE else [],   # read-only caches
        "write_cache_dir": None,            # downloaded texts + provenance json; None -> <run.root>/fulltext_cache
        "keep_raw": True,                   # also keep the raw PDF/XML under <write_cache_dir>/raw/
        "ledger_path": None,                # download ledger (sqlite); None -> <run.root>/fulltext_ledger.sqlite
        "queue_path": None,                 # browser queue (jsonl); None -> <run.root>/fulltext_queue.jsonl
        "queue_backoff_s": 21600,           # a failed queue entry waits this long (doubling, max 8x) before retry
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
        },
        # Caps / pacing (dlledger.DEFAULT_LIMITS etc.); dicts merge per channel / host.
        "limits": {},
        "global_limits": {},
        "host_caps": {},
    },
    "probe": {
        "tbs_tasks_root": str(OLD_PIPE / "tasks_text") if OLD_PIPE else None,   # None -> no TBS dedup
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
                     "cli_contract"],
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
    """role -> model alias: 'rules' / None / 'fake:*' / 'opus' / ..."""
    v = cfg["models"]["roles"].get(role, "rules")
    if v in ("teacher", "cheap"):
        v = cfg["models"][v]
    return v
