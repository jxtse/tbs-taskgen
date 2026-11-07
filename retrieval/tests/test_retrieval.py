"""Offline tests on recorded real responses (tests/fixtures, recorded 2026-11-05 by scripts/record_fixtures.py)."""
import json
import os
from pathlib import Path

import pytest
from conftest import Replay, load_fixtures, make_sources

from tbsretr.fusion import Pool, fuse, fuse_resources, prefilter_reason
from tbsretr.ledger import BudgetExceeded, Ledger, SourceCooling
from tbsretr.model import Paper, RankedList, Target, route_arxiv, is_biomed
from tbsretr.pipeline import Retriever
from tbsretr.util import (classify_url, doi_from_text, norm_doi, redact, resource_key, strip_secret_params)

T = Target(id="task:ambient-rna-correction", queries=["ambient RNA removal droplet single-cell RNA-seq"],
           description="Remove ambient RNA (soup) contamination from droplet scRNA-seq count matrices using empty droplets.",
           name="ambient RNA correction", libs=["SoupX"])


# ------------------------------------------------------------------ parsers on real responses
def test_openalex_parse(cfg, keys):
    S, led, _ = make_sources(cfg, Replay())
    rl = S["openalex"].search(T.queries[0], T)
    assert len(rl.papers) >= 20
    p = rl.papers[0]
    assert p.title and p.openalex and p.openalex.startswith("W")
    assert sum(bool(x.abstract) for x in rl.papers) >= 0.7 * len(rl.papers)
    assert any(x.referenced_works for x in rl.papers)
    assert any(x.oa_url for x in rl.papers)
    assert led.usage("openalex")["usd"] == pytest.approx(0.001)


def test_s2_search_recommend_and_429_backoff(cfg, keys):
    rp = Replay()
    S, led, _ = make_sources(cfg, rp)
    rl = S["s2"].search(T.queries[0], T)          # recorded: 429, 429, 200
    assert len(rl.papers) == 20 and all(p.s2 for p in rl.papers)
    assert sum(1 for c in rp.seen if "paper/search" in c[1]) == 3
    assert any(p.oa_url for p in rl.papers)
    assert all(c[2].get("x-api-key", "").startswith("SECRET-") for c in rp.seen)
    rec = S["s2"].recommend(rl.papers[:2], T)
    assert len(rec.papers) >= 10 and rec.name == "s2.recommend"
    one = S["s2"].recommend(rl.papers[:1], T)      # forpaper endpoint
    assert len(one.papers) >= 10
    n429 = sum(1 for r in led.summary("test") if r["source"] == "s2" for _ in range(r["n429"] or 0))
    assert n429 >= 2


def test_s2_skipped_without_key(cfg, monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    rt = Retriever(cfg, transport=Replay(), sources={"s2"})
    assert rt.status()["s2"]["available"] is False
    out = rt.retrieve(T, seed_mode="none")
    assert out["sources"]["s2"]["status"].startswith("skipped")
    assert out["papers"] == []


def test_exa_paper_entities(cfg, keys):
    S, led, _ = make_sources(cfg, Replay())
    rl = S["exa"].channel("paper", T)
    assert len(rl.papers) >= 8
    with_doi = [p for p in rl.papers if p.doi]
    assert len(with_doi) >= 6
    assert any(len(p.abstract) > 200 for p in rl.papers)
    assert "| Oxford Academic" not in " ".join(p.title for p in rl.papers)
    assert led.usage("exa")["usd"] == pytest.approx(0.007)   # costDollars from the response


def test_tavily_code_channel(cfg, keys):
    S, led, _ = make_sources(cfg, Replay())
    rl = S["tavily"].channel("code", T)
    kinds = {r.kind for r in rl.resources}
    assert "code" in kinds
    keys_ = {resource_key(r.url) for r in rl.resources}
    assert "github.com/broadinstitute/cellbender" in keys_
    assert led.usage("tavily")["credits"] == 1


def test_crossref_arxiv_epmc(cfg, keys):
    S, _, _ = make_sources(cfg, Replay())
    cr = S["crossref"].search("SoupX ambient RNA", T)
    assert cr.papers and cr.papers[0].doi
    look = S["crossref"].by_dois(["10.1093/gigascience/giaa151"])
    assert look and look[0].doi == "10.1093/gigascience/giaa151" and look[0].venue
    ax = S["arxiv"].search("neural quantum states stochastic reconfiguration", T, ["quant-ph", "cond-mat"])
    assert ax.papers and all(p.arxiv for p in ax.papers) and ax.papers[0].abstract
    ep = S["europepmc"].search(T.queries[0], T)
    assert ep.papers and sum(bool(p.doi) for p in ep.papers) >= 10


# ------------------------------------------------------------------ secrets
def test_no_secret_in_ledger_cache_or_fixtures(cfg, keys, tmp_path):
    S, led, _ = make_sources(cfg, Replay())
    S["openalex"].search(T.queries[0], T)
    S["exa"].channel("paper", T)
    S["tavily"].channel("code", T)
    blob = (led.path.read_bytes() + led.cache_path.read_bytes()).decode("latin1")
    for k in ("OPENALEX_API_KEY", "EXA_API_KEY_SECONDARY", "TAVILY_API_KEY_SECONDARY"):
        assert os.environ[k] not in blob
    assert "api_key" not in blob
    for fx in load_fixtures():
        assert "api_key=" not in fx["url"]


def test_strip_and_redact(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "abcdef123456")
    u = "https://api.openalex.org/works?search=x&api_key=abcdef123456&mailto=a%40b"
    assert "api_key" not in strip_secret_params(u) and "mailto" in strip_secret_params(u)
    assert "abcdef123456" not in redact("GET " + u)
    assert "abcdef123456" not in redact('{"Authorization": "Bearer abcdef123456"}')


# ------------------------------------------------------------------ ledger behaviour
def test_cache_hit_no_second_call(cfg, keys):
    rp = Replay()
    S, led, _ = make_sources(cfg, rp)
    S["openalex"].search(T.queries[0], T)
    S["openalex"].search(T.queries[0], T)
    assert len(rp.seen) == 1
    assert led.usage("openalex")["calls"] == 1


def test_budget_stops_source(cfg, keys):
    cfg["sources"]["openalex"]["daily_budget_usd"] = 0.0015
    rp = Replay(extra={})
    S, led, _ = make_sources(cfg, rp)
    S["openalex"].search(T.queries[0], T)
    with pytest.raises(BudgetExceeded):
        S["openalex"].search("another query", T)
    cfg["sources"]["exa"]["daily_max_calls"] = 0
    with pytest.raises(BudgetExceeded):
        S["exa"].channel("paper", T)


def test_monthly_credit_cap(cfg, keys):
    cfg["sources"]["tavily"]["monthly_max_credits"] = 1
    S, _, _ = make_sources(cfg, Replay())
    S["tavily"].channel("code", T)
    with pytest.raises(BudgetExceeded):
        S["tavily"].channel("data", T)


def test_429_cooldown_persists(cfg, keys):
    cfg["sources"]["exa"]["retries"] = 1

    def always429(m, u, h, b):
        return 429, {}, b'{"error":"rate"}'
    S, led, _ = make_sources(cfg, always429)
    with pytest.raises(SourceCooling):
        S["exa"].channel("code", T)
    assert led.cooling("exa") > 0
    led2 = Ledger(cfg["state_dir"])       # another process sees the cooldown
    assert led2.cooling("exa") > 0


def test_openalex_remaining_usd_guard(cfg, keys):
    cfg["sources"]["openalex"]["min_remaining_usd"] = 0.5

    def tr(m, u, h, b):
        return 200, {"X-RateLimit-Remaining-USD": "0.2"}, b'{"results": []}'
    S, _, _ = make_sources(cfg, tr)
    S["openalex"].search("q1", T)
    with pytest.raises(BudgetExceeded):
        S["openalex"].search("q2", T)


def test_5xx_retry_then_ok(cfg, keys):
    calls = []

    def tr(m, u, h, b):
        calls.append(1)
        return (503, {}, b"busy") if len(calls) < 3 else (200, {}, b'{"results": []}')
    cfg["sources"]["openalex"]["retries"] = 3
    import tbsretr.ledger as L
    orig = L.time.sleep
    L.time.sleep = lambda s: None
    try:
        S, _, _ = make_sources(cfg, tr)
        S["openalex"].search("q", T)
    finally:
        L.time.sleep = orig
    assert len(calls) == 3


# ------------------------------------------------------------------ fusion / filters
def _p(**kw):
    base = dict(title="A sufficiently long and distinctive method paper title", abstract="x " * 150, language="en")
    base.update(kw)
    return Paper(**base)


def test_dedupe_across_ids():
    pool = Pool()
    pool.add_list(RankedList("openalex.search", "q", [_p(doi="10.1/ABC", openalex="W1")]))
    pool.add_list(RankedList("s2.search", "q", [_p(doi="https://doi.org/10.1/abc", s2="s2x", oa_url="http://x.pdf")]))
    pool.add_list(RankedList("arxiv.search", "q", [_p(arxiv="2101.00001v2", title="Other title entirely here ok fine")]))
    pool.add_list(RankedList("crossref.search", "q", [_p(doi="10.48550/arxiv.2101.00001", title="Third distinct title for crossref copy")]))
    roots = pool.roots()
    assert len(roots) == 2
    p = next(pool.papers[i] for i in roots if pool.papers[i].doi == "10.1/abc")
    assert p.s2 == "s2x" and p.openalex == "W1" and p.oa_url


def test_title_dedupe_and_rrf_order(cfg):
    pool = Pool()
    a = _p(title="Decontamination of ambient RNA in single-cell RNA-seq with DecontX")
    b = _p(title="Something else entirely about another topic in genomics")
    pool.add_list(RankedList("openalex.search", "q", [b, a]))
    pool.add_list(RankedList("s2.search", "q", [_p(title="Decontamination of ambient RNA in single-cell RNA-seq with DecontX.")]))
    pool.add_list(RankedList("exa.paper", "q", [_p(title="DECONTAMINATION of ambient RNA in single-cell RNA-seq with DecontX")]))
    kept, _ = fuse(pool, cfg, set())
    assert kept[0]["title"].startswith("Decontamination") and kept[0]["n_sources"] == 3


def test_rrf_many_queries_one_source_do_not_swamp(cfg):
    pool = Pool()
    x = _p(title="Paper X found by one source in many query lists")
    y = _p(title="Paper Y found by three different sources at rank two")
    for q in range(4):
        pool.add_list(RankedList("crossref.search", f"q{q}", [x]))
    for s in ("openalex.search", "s2.search", "arxiv.search"):
        pool.add_list(RankedList(s, "q", [_p(title="filler paper with long enough title " + s), y]))
    kept, _ = fuse(pool, cfg, set())
    order = [k["title"][:7] for k in kept]
    assert order.index("Paper Y") < order.index("Paper X")


def test_prefilter_rules(cfg):
    assert prefilter_reason(_p(title="A review of ambient RNA correction methods"), cfg, set(), set()) == "review/overview"
    assert prefilter_reason(_p(type="review"), cfg, set(), set()) == "review/overview"
    assert prefilter_reason(_p(language="zh"), cfg, set(), set()).startswith("non-english")
    assert prefilter_reason(_p(title="单细胞测序中的环境RNA去除方法研究与应用分析", abstract="环境" * 200), cfg, set(), set()).startswith("non-english")
    assert prefilter_reason(_p(abstract="short"), cfg, set(), set()) is None          # soft by default
    cfg["prefilter"]["drop_no_abstract"] = True
    assert prefilter_reason(_p(abstract="short"), cfg, set(), set()) == "no-abstract"
    cfg["prefilter"]["drop_no_abstract"] = False
    assert prefilter_reason(_p(doi="10.5067/atlas/atl08.007"), cfg, {"10.5067/atlas/atl08.007"}, set()).startswith("leak")
    assert prefilter_reason(_p(arxiv="2101.00001"), cfg, {"arxiv:2101.00001"}, set()).startswith("leak")
    assert prefilter_reason(_p(), cfg, set(), set()) is None


def test_no_abstract_demoted_not_dropped(cfg):
    pool = Pool()
    a = _p(title="Paper with an abstract and a long enough title")
    b = _p(title="Paper without abstract but ranked first in the list", abstract="")
    pool.add_list(RankedList("openalex.search", "q", [b, a]))
    kept, dropped = fuse(pool, cfg, set())
    assert [k["title"][:9] for k in kept] == ["Paper wit", "Paper wit"] and not dropped
    assert kept[0]["abstract_missing"] is False and kept[1]["abstract_missing"] is True


def test_ablation_allowed_sources(cfg):
    pool = Pool()
    pool.add_list(RankedList("openalex.search", "q", [_p(title="Only openalex has this one, title long enough")]))
    pool.add_list(RankedList("crossref.search", "q", [_p(title="Only crossref has this one, title long enough")]))
    kept, _ = fuse(pool, cfg, set(), allowed_sources={"openalex"})
    assert [k["title"][:13] for k in kept] == ["Only openalex"]


def test_resources_fuse_and_classify():
    assert classify_url("https://github.com/broadinstitute/CellBender") == "code"
    assert classify_url("https://zenodo.org/records/123") == "data"
    assert classify_url("https://cellbender.readthedocs.io/en/latest/") == "docs"
    assert classify_url("https://arxiv.org/abs/2101.00001") == "paper"
    assert resource_key("https://github.com/Foo/Bar/issues/3") == "github.com/foo/bar"
    from tbsretr.model import Resource
    l1 = RankedList("exa.code", "q", resources=[Resource("https://github.com/Foo/Bar", "code")])
    l2 = RankedList("tavily.code", "q", resources=[Resource("https://github.com/foo/bar/blob/main/x.py", "code")])
    res = fuse_resources([l1, l2], [{"code_links": ["github.com/foo/bar"], "fused_rank": 1, "doi": "10.1/x"}])
    assert len(res) == 1 and set(res[0]["found_by"]) == {"exa.code", "tavily.code"} and res[0]["from_papers"]


def test_resource_leak_filter():
    from tbsretr.model import Resource
    l1 = RankedList("exa.code", "q", resources=[
        Resource("https://github.com/harbor-framework/terminal-bench-science/tree/main/tasks/x", "code"),
        Resource("https://github.com/ok/repo", "code")])
    res = fuse_resources([l1], [], leak_urls=("terminal-bench-science",))
    assert [r["key"] for r in res] == ["github.com/ok/repo"]


def test_ids():
    assert norm_doi("https://doi.org/10.1000/ABC.") == "10.1000/abc"
    assert doi_from_text("https://link.springer.com/article/10.1186/s13059-020-1950-6") == "10.1186/s13059-020-1950-6"
    assert doi_from_text("https://onlinelibrary.wiley.com/doi/full/10.1002/qj.1234") == "10.1002/qj.1234"


def test_routing():
    assert "astro-ph" in route_arxiv("astronomy / cosmology 3x2pt weak lensing")
    assert "quant-ph" in route_arxiv("stochastic reconfiguration neural quantum states")
    assert "stat" in route_arxiv("multivariate Fay-Herriot small area")
    assert route_arxiv("ambient RNA removal droplet single-cell") == []
    assert is_biomed("biology / single-cell-genomics") and not is_biomed("vehicle routing column generation")


def test_crossref_exact_queries(cfg):
    rt = Retriever(cfg, transport=Replay(), sources={"crossref"})
    t = Target(id="x", queries=["CellBender background correction empty droplets",
                                "SoupX cell free mRNA contamination estimation", "ambient RNA"], libs=["numpy"])
    qs = rt.crossref_queries(t)
    assert len(qs) == 3 and qs[0].startswith("CellBender") and qs[1].startswith("SoupX")


# ------------------------------------------------------------------ end to end (offline)
def test_retrieve_end_to_end_offline(cfg, keys):
    """Full pipeline on fixtures: unknown requests return 599 -> recorded as errors, never crash."""
    cfg["sources"]["exa"]["channels"] = ["paper"]
    cfg["sources"]["tavily"]["channels"] = ["code"]
    cfg["sources"]["openalex"]["retries"] = 0
    cfg["sources"]["s2"]["retries"] = 0
    for s in cfg["sources"].values():
        s["retries"] = 0
    cfg["sources"]["s2"]["recommend"] = True
    cfg["sources"]["s2"]["recommend_pseudo"] = True
    rt = Retriever(cfg, transport=Replay())
    out = rt.retrieve(T, seed_mode="pseudo")
    assert len(out["papers"]) >= 20
    top = out["papers"][0]
    assert {"doi", "title", "abstract", "rrf", "ranks", "sources", "fused_rank"} <= set(top)
    assert any(r["kind"] == "code" for r in out["resources"])
    assert out["sources"]["openalex"]["lists"] >= 1 and out["sources"]["exa"]["lists"] == 1
    json.dumps({k: v for k, v in out.items() if not k.startswith("_")})


# ------------------------------------------------------------------ exa via local search pool
def _pool_replay(fx_name="exa_6.json", provider="exa"):
    """Wrap a recorded direct Exa response the way the search-pool service returns it."""
    rec = json.loads((Path(__file__).parent / "fixtures" / fx_name).read_text())
    exa = json.loads(rec["response"])
    rows = [{"position": i + 1, "title": r.get("title"), "url": r.get("url"), "content": "", "raw": r}
            for i, r in enumerate(exa["results"])]
    out = {"object": "search.results", "provider": provider, "results": rows,
           "usage": {"result_count": len(rows), "cost_usd": exa.get("costDollars", {}).get("total")},
           "account": {"provider": provider, "index": 1, "name": "SLOT"}}
    seen = []

    def transport(method, url, headers, body):
        seen.append((method, url, headers, body))
        return 200, {}, json.dumps(out).encode()
    return transport, seen, rec


def test_exa_pool_backend_same_parse_as_direct(cfg, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY_SECONDARY", raising=False)     # pool backend needs no key in this process
    monkeypatch.delenv("TBS_SEARCH_POOL_URL", raising=False)
    cfg["sources"]["exa"]["backend"] = "pool"
    cfg["sources"]["exa"]["pool_url"] = "http://127.0.0.1:18150"
    tr, seen, rec = _pool_replay()
    S, led, _ = make_sources(cfg, tr)
    assert S["exa"].available() == (True, "")
    rl = S["exa"].channel("paper", T)
    m, url, hdr, body = seen[0]
    assert url == "http://127.0.0.1:18150/v1/search" and m == "POST"
    assert "x-api-key" not in {k.lower() for k in hdr}
    assert body["engine"] == "exa" and body["limit"] == 10 and body["include_contents"] is False
    assert body["exa_options"] == {"type": "auto", "highlights_max_characters": 400, "category": "publication"}
    assert body["query"] == rec["body"]["query"]
    assert led.usage("exa")["usd"] == pytest.approx(0.007)          # cost_usd reported by the pool
    # identical parsing to the direct path on the same Exa rows
    S2, _, _ = make_sources(dict(cfg, sources=dict(cfg["sources"], exa=dict(cfg["sources"]["exa"], backend="direct"))),
                            Replay())
    monkeypatch.setenv("EXA_API_KEY_SECONDARY", "SECRET-x")
    direct = S2["exa"].channel("paper", T)
    assert [(p.doi, p.title) for p in rl.papers] == [(p.doi, p.title) for p in direct.papers]
    assert len(rl.papers) >= 8 and sum(bool(p.doi) for p in rl.papers) >= 6


def test_exa_pool_rejects_other_provider_and_env_url(cfg, monkeypatch):
    monkeypatch.setenv("TBS_SEARCH_POOL_URL", "http://127.0.0.1:9/")
    cfg["sources"]["exa"]["backend"] = "pool"
    tr, seen, _ = _pool_replay(provider="tavily")
    S, led, _ = make_sources(cfg, tr)
    assert S["exa"]._safe(S["exa"].channel, "code", T) is None
    assert seen[0][1] == "http://127.0.0.1:9/v1/search"
    assert "provider='tavily'" in S["exa"].errors[0]
    u = led.usage("exa")
    assert u["calls"] == 1 and u["usd"] == pytest.approx(0.007)        # real cost kept ...
    # ... but the bad 200 is not cached: once the pool is fixed, the same query is fetched again
    tr2, seen2, _ = _pool_replay(provider="exa")
    S2, _, _ = make_sources(cfg, tr2)
    rl = S2["exa"].channel("code", T)
    assert len(seen2) == 1 and (rl.papers or rl.resources)


def test_exa_pool_404_is_an_error_not_empty(cfg, monkeypatch):
    monkeypatch.delenv("TBS_SEARCH_POOL_URL", raising=False)
    cfg["sources"]["exa"]["backend"] = "pool"
    S, _, _ = make_sources(cfg, lambda m, u, h, b: (404, {}, b'{"error": {"type": "not_found"}}'))
    assert S["exa"]._safe(S["exa"].channel, "paper", T) is None
    assert S["exa"].errors and "404" in S["exa"].errors[0]


@pytest.mark.parametrize("n,ok", [(0, False), (1, True), (20, True), (21, False), ("10", False), (True, False)])
def test_exa_pool_num_results_range(cfg, n, ok):
    cfg["sources"]["exa"]["backend"] = "pool"
    cfg["sources"]["exa"]["num_results"] = n
    if ok:
        make_sources(cfg, Replay())
    else:
        with pytest.raises(ValueError):
            make_sources(cfg, Replay())


@pytest.mark.parametrize("url,ok", [
    ("http://127.0.0.1:18150/v1/search", True), ("http://localhost:18150/x", True), ("http://LOCALHOST./x", True),
    ("http://127.0.0.2/x", True), ("http://[::1]:18150/x", True), ("http://[0:0:0:0:0:0:0:1]/x", True),
    ("http://[::ffff:127.0.0.1]/x", True), ("https://api.exa.ai/search", False), ("http://10.0.0.1/x", False),
    ("http://localhost.example.com/x", False)])
def test_is_loopback(url, ok):
    from tbsretr.ledger import is_loopback
    assert is_loopback(url) is ok


def test_exa_pool_unset_url_skips(cfg, monkeypatch):
    monkeypatch.delenv("TBS_SEARCH_POOL_URL", raising=False)
    cfg["sources"]["exa"]["backend"] = "pool"
    cfg["sources"]["exa"]["pool_url"] = ""
    S, _, _ = make_sources(cfg, Replay())
    assert S["exa"].available()[0] is False


def test_no_machine_specific_defaults(monkeypatch):
    """Default pool URL and leak-refs path are unset unless the environment provides them."""
    import importlib
    import tbsretr.config as rc
    for k in ("TBS_SEARCH_POOL_URL", "TBS_LEAK_REFS", "TBS_OLD_PIPE"):
        monkeypatch.delenv(k, raising=False)
    c = importlib.reload(rc).load_config(path="/nonexistent.toml")
    assert c["sources"]["exa"]["pool_url"] is None and c["leak_refs"] is None
    monkeypatch.setenv("TBS_OLD_PIPE", "/data/old")
    assert importlib.reload(rc).load_config(path="/nonexistent.toml")["leak_refs"] == "/data/old/literature/tbs_task_refs.json"
    monkeypatch.setenv("TBS_LEAK_REFS", "/x/refs.json")
    assert importlib.reload(rc).load_config(path="/nonexistent.toml")["leak_refs"] == "/x/refs.json"
    monkeypatch.undo()
    importlib.reload(rc)
