"""Configuration: built-in conservative defaults, optionally overridden by a TOML file and CLI flags.

Nothing in here is secret. API keys are only ever read from environment variables (names configurable below),
which your secret manager / shell should export at process start.
"""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]

DEFAULTS: dict = {
    "state_dir": str(PROJECT / "state"),          # ledger.sqlite + cache.sqlite
    "mailto": "jxtse@users.noreply.github.com",    # Crossref polite pool / OpenAlex / arXiv UA
    "user_agent": "tbs-retrieval/0.2 (TB-Science method-literature search; mailto:{mailto})",
    "cache_ttl_days": 30,
    "parallel_sources": True,                      # one thread per source inside a target (each source keeps its own pacing)
    "timeout_s": 60,
    # TBS task-source papers (the data a TBS task was built from) must never be proposed. Read-only.
    "leak_refs": os.environ.get("TBS_LEAK_REFS", os.path.join(os.environ.get("TBS_OLD_PIPE", "/Volumes/Storage/hermes/workspace/tbs-datapipe-20261015"), "literature", "tbs_task_refs.json")),
    # Resources (code/data/docs) whose URL contains any of these substrings are dropped: the TBS benchmark itself
    # leaks task statements/solutions (both Exa and Tavily returned it on 3/10 targets in the 2026-11-05 eval).
    "leak_url_substrings": ["harbor-framework/terminal-bench-science", "terminal-bench-science", "tbench.ai"],
    "fusion": {
        "rrf_k": 60,
        "top_n": 60,                               # papers handed to the (later, separate) LLM screen
        "multi_source_bonus": 0.0,                 # extra score per additional distinct source (0 = pure RRF)
        "weights": {                               # per ranked-list weight in RRF (subjective defaults)
            "openalex.search": 1.0, "openalex.snowball": 1.0, "openalex.semantic": 0.6,
            "s2.search": 1.0, "s2.recommend": 0.8,
            "exa.paper": 0.8, "tavily.paper": 0.5,
            # papers that show up in the code/data/docs channels count, but weakly
            "exa.code": 0.3, "exa.data": 0.3, "exa.docs": 0.3, "tavily.code": 0.3, "tavily.data": 0.3, "tavily.docs": 0.3,
            "crossref.search": 0.7, "arxiv.search": 0.7, "europepmc.search": 0.5,
        },
    },
    "prefilter": {
        "drop_review": True, "drop_non_english": True, "drop_leak": True,
        # 2026-11-05 eval: a hard no-abstract drop removed 7/55 old-source core papers and 4/35 selected papers
        # (mostly Crossref/Elsevier records) -> keep them, down-weighted, flagged abstract_missing.
        "drop_no_abstract": False, "no_abstract_factor": 0.5,
        "min_year": 1985, "min_abstract_chars": 200,
    },
    # Pseudo-seeds for snowball / S2 recommendations when the target has no seed DOIs: the top-k pre-filtered
    # papers of the OpenAlex keyword lists alone (OpenAlex is in every ablation arm, so ablations stay comparable).
    "pseudo_seeds": {"k": 3},
    "sources": {
        "openalex": {
            "enabled": True, "key_env": "OPENALEX_API_KEY", "require_key": False,
            "min_interval_s": 0.2, "per_page": 50, "max_queries": 4,
            "filter": "type:article|preprint|book-chapter,language:en,is_retracted:false",
            "snowball": True, "snowball_min_shared": 2, "snowball_max": 50,
            "semantic": False,                     # search.semantic was unstable in an earlier benchmark (0/10 on 2 targets)
            "backfill_top": 120,                    # DOI batch lookup to fill abstracts / OA for fused papers
            "cost_usd": {"search": 0.001, "semantic": 0.001, "list": 0.0001, "get": 0.0},
            "daily_budget_usd": 0.50, "monthly_budget_usd": None, "daily_max_calls": 2000,
            "min_remaining_usd": 0.10,             # stop if X-RateLimit-Remaining-USD drops below this
            "retries": 4, "cooldown_429_s": 3600,
        },
        "s2": {
            "enabled": True, "key_env": "SEMANTIC_SCHOLAR_API_KEY", "require_key": True,   # no key -> skipped
            "min_interval_s": 3.0,                 # key quota is 1 rps across ALL endpoints; observed sporadic 429s even
                                                   # at 0.25-0.5 rps (2026-11-05), so pace at 1 req/3 s + 429 backoff
            "backoff_429_s": 5.0,
            "limit": 20, "max_queries": 3, "recommend_limit": 30, "batch_lookup": True,
            # 2026-11-05 eval: recommendations hit 1/21 held-out selected papers with pseudo seeds and 0/18 with
            # real seeds; only 15 recommended papers reached any head-60 (2 core, 0 unique) -> off by default.
            # Set recommend=True (+ recommend_pseudo=True to also use pseudo seeds) to re-enable.
            "recommend": False, "recommend_pseudo": False,
            "daily_max_calls": 1500, "retries": 5, "cooldown_429_s": 900,
        },
        "exa": {
            # backend "pool": local search-pool service that holds the Exa keys (POST {pool_url}/v1/search, engine=exa,
            # no Tavily fallback; env TBS_SEARCH_POOL_URL overrides pool_url).  backend "direct": api.exa.ai + key_env.
            "backend": "pool", "pool_url": "http://127.0.0.1:18150",
            "enabled": True, "key_env": "EXA_API_KEY_SECONDARY", "require_key": True,
            "min_interval_s": 0.3, "num_results": 10, "type": "auto",
            # 2026-11-05 eval: usable resources per 100 results: code 65, docs 53, data 39. The paper channel is
            # kept: removing it lowers top-20 core papers 119 -> 114 (it re-ranks, rarely adds unique papers).
            "channels": ["code", "docs", "data", "paper"],
            "highlights_chars": 400,
            "est_cost_usd": 0.012,                 # used only when the response lacks costDollars
            "daily_max_calls": 80, "daily_budget_usd": 0.60, "monthly_budget_usd": 9.0,   # 4 calls/target -> 20 targets/day
            "retries": 3, "cooldown_429_s": 3600,
        },
        "tavily": {
            # 2026-11-05 decision: Tavily removed from the default pipeline (low yield vs Exa); adapter kept for reruns.
            "enabled": False, "key_env": "TAVILY_API_KEY_SECONDARY", "require_key": True,
            "min_interval_s": 1.0, "max_results": 10, "search_depth": "basic",
            # 2026-11-05 eval: Tavily found 35 target-usable resources vs Exa 137 (13 Tavily-only), 41 aggregator/
            # forum URLs; paper channel added 0 papers that no other source had -> backup code channel only.
            "channels": ["code"],
            "daily_max_calls": 30, "daily_max_credits": 30, "monthly_max_credits": 900,   # free tier = 1000/month
            "retries": 3, "cooldown_429_s": 3600,
        },
        "crossref": {
            "enabled": True, "min_interval_s": 0.34, "rows": 20, "max_queries": 3,   # polite pool 3 rps
            "metadata_lookup": True, "daily_max_calls": 3000, "retries": 4, "cooldown_429_s": 1800,
        },
        "arxiv": {
            "enabled": True, "min_interval_s": 3.1, "max_results": 20, "max_queries": 2,   # ToU: 1 req / 3 s, 1 conn
            "daily_max_calls": 1000, "retries": 3, "cooldown_429_s": 3600,
        },
        "europepmc": {
            # biomedical targets only (routed); 2026-11-05 eval: +2 core papers in head-60, 0 not found elsewhere
            "enabled": True, "min_interval_s": 0.3, "page_size": 20, "max_queries": 2,
            "daily_max_calls": 2000, "retries": 3, "cooldown_429_s": 1800,
        },
    },
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else (PROJECT / "config.toml")
    if p.exists():
        cfg = _merge(cfg, tomllib.loads(p.read_text()))
    if overrides:
        cfg = _merge(cfg, overrides)
    cfg["user_agent"] = cfg["user_agent"].format(mailto=cfg["mailto"])
    return cfg
