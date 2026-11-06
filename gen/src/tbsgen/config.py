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

PIPELINE_V2 = Path(__file__).resolve().parents[3]          # repo root (gen/ and retrieval/ live here)
# Machine-specific locations.  Override with env vars (or config.toml) on another machine.
WORKSPACE = Path(os.environ.get("TBS_WORKSPACE", "/Volumes/Storage/hermes/workspace"))
OLD_PIPE = Path(os.environ.get("TBS_OLD_PIPE", str(WORKSPACE / "tbs-datapipe-20261015")))
HARBOR_ROOT = Path(os.environ.get("TBS_HARBOR_ROOT", str(WORKSPACE / "tbs-aliyun")))

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
            "screen": "rules",              # S2 (待定 cheap model)
            "probe": "rules",               # S4 (待定 cheap agent; sandbox repro not implemented in MVP)
            "spec": "teacher",              # S5
            "gen": "teacher",               # S6A
            "tests": "teacher",             # S6C
            "repair": "teacher",            # S8
            "review": "rules",              # S8 preservation review (mechanical diff; cheap model 待定)
            "invert": None,                 # S7 L4b cheap-agent inversion attack (待定; disabled)
        },
        "opus": {"model": "claude-opus-5-5", "url": os.environ.get("TBS_OPUS_URL", "http://127.0.0.1:3000/v1/messages"),
                 "effort": "medium", "max_tokens": 32000, "timeout": 1800, "retries": 2},
        # USD per million tokens.  PLACEHOLDERS (same as DESIGN.md §3.1) until real bills replace them.
        "prices": {"claude-opus-5-5": {"in": 5.0, "out": 25.0, "cache_read": 0.5},
                   "fake": {"in": 0.0, "out": 0.0}},
    },
    # ---------------------------------------------------------------- stages
    "screen": {"min_score": 2, "keep_top": 20},
    "fulltext": {
        "mode": "oa_and_cache",             # cache (old literature/txt, read-only) then OA PDF link; else browser queue
        "cache_dirs": [str(OLD_PIPE / "literature" / "txt")],
        "min_chars": 8000,
        "title_match_min": 0.6,
        "max_pdf_mb": 40,
        "timeout": 90,
        "allow_network": True,
    },
    "probe": {
        "tbs_tasks_root": str(OLD_PIPE / "tasks_text"),
        "decontam_cos_max": 0.30,
        "min_method_hits": 2,
    },
    "spec": {
        "allowed_forms": ["ANS", "PROG"],   # MVP; DOMF needs per-format readers; INTERACT banned until isolation (3rd phase)
        "prog_quota_frac": 0.5,             # 待定: max share of PROG among released tasks
        "spec_retries": 1,
        "excerpt_chars": 60000,
        "expert_hours": [2, 6],             # target expert time stated in the S5 prompt (old pipeline: 0.4-0.6 h)
        "max_tokens": 32000,
    },
    "gen": {
        "seeds": [0, 1],
        "gen_timeout": 300, "solve_timeout": 900, "baseline_timeout": 600, "test_timeout": 900,
        "max_public_mb": 50,
        "back_to_spec_max": 1,              # S6B "mid not separable" -> back to S5 at most once
        "s6a_max_tokens": 64000,            # S6A emits 5 files + adaptive thinking; 32k truncated in e2e-iter1
        "s6c_max_tokens": 48000,
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
        "student_agent": "harbor_ext.glm65_agents:ClaudeCodeGlm65",
        "teacher_model": "claude-opus-5-5",
        "teacher_agent": "harbor_ext.tbs_agent:ClaudeCodeTbs",
        "environment": "harbor_ext.sandbox:SandboxEnvironment",
        "harbor_root": str(HARBOR_ROOT),
        "jobs_dir": str(PIPELINE_V2 / "gen" / "state" / "harbor-jobs"),
        "student_launcher": "glm53-official65/run-glm65.sh",
        "teacher_launcher": "run-job-opus.sh",
        "allow_submit": False,              # must be flipped explicitly; MVP never submits
        "glm_runs_on_first_fail": 1,        # DESIGN §3.5: first GLM fail -> teacher (no 2nd GLM run)
        "teacher_runs_max": 2,
        "quick_pass_minutes": None,         # 待定: 1st-pass shortcut to "easy" (DESIGN suggests 30); None=off
        "keep_labels": ["hard", "unstable"],  # decision 2026-11-05: GLM 1/2, or GLM 0 + teacher pass
        "infra_retries": 2,                 # API quota / sandbox errors do not count as attempts
        "variant_depth": 0,                 # S9b harder variants; 待定 (DESIGN suggests <=2); 0 = off in MVP
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
