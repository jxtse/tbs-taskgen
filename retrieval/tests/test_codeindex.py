"""Code-grounded retrieval (codeindex.py): OpenAlex code channel + flags, PwC join, GitHub README search, merge."""
import gzip
import json
import urllib.parse

from conftest import make_sources

from tbsretr import codeindex
from tbsretr.model import Target
from tbsretr.pipeline import Retriever

T = Target(id="method:enks", queries=["ensemble Kalman smoother", "iterative ensemble smoother"],
           description="ensemble smoothers")


def _work(doi, title, year=2021):
    return {"id": "https://openalex.org/W" + str(abs(hash(doi)) % 10**8), "doi": "https://doi.org/" + doi,
            "title": title, "publication_year": year, "type": "article", "language": "en",
            "abstract_inverted_index": {w: [i] for i, w in enumerate(("word " * 60).split())}}


class FakeOA:
    """Transport: OpenAlex search / code search / DOI flags, GitHub search."""

    def __init__(self):
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append(url)
        u = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qs(u.query)
        if u.netloc == "api.openalex.org":
            f = q.get("filter", [""])[0]
            if "fulltext.search" in f and f.startswith("doi:"):
                return 200, {}, json.dumps({"results": [{"doi": "https://doi.org/10.1/code"}]}).encode()
            if "fulltext.search" in f:
                assert "publication_year:>2017" in f
                return 200, {}, json.dumps({"results": [_work("10.1/code", "Code paper with smoother", 2022)]}).encode()
            if f.startswith("doi:") or f.startswith("openalex:"):
                return 200, {}, json.dumps({"results": []}).encode()
            return 200, {}, json.dumps({"results": [_work("10.1/nocode", "No code smoother paper long title", 2015),
                                                    _work("10.1/code", "Code paper with smoother", 2022)]}).encode()
        if u.netloc == "api.github.com":
            return 200, {}, json.dumps({"items": [{"full_name": "alice/smoother"}]}
                                       if "10.1/code" in q["q"][0] else {"items": []}).encode()
        return 404, {}, b""


def _cfg(cfg, **code):
    c = json.loads(json.dumps(cfg))
    c["code"].update(dict(enabled=True, github_client="rest"), **code)
    c["parallel_sources"] = False
    for n in c["sources"]:
        c["sources"][n]["min_interval_s"] = 0.0
    return c


def test_code_disabled_by_default(cfg):
    assert cfg["code"]["enabled"] is False


def test_retrieve_with_code_signals(cfg, tmp_path):
    c = _cfg(cfg, pwc_path=None)
    tr = FakeOA()
    rt = Retriever(c, transport=tr, sources={"openalex"})
    out = rt.retrieve(T, seed_mode="none")
    names = {rl.name for rl in out["_lists"]}
    assert "openalex.code" in names
    by = {p["doi"]: p for p in out["papers"]}
    assert by["10.1/code"]["code_signals"]["openalex_fulltext"] is True
    assert by["10.1/code"]["code_signals"]["gh_readme"] == ["https://github.com/alice/smoother"]
    assert "https://github.com/alice/smoother" in by["10.1/code"]["code_links"]
    assert by["10.1/code"]["code_signals"]["any"] and not by["10.1/nocode"]["code_signals"]["any"]
    # year bias: the 2015 paper is ranked after the newer one
    assert out["papers"][-1]["doi"] == "10.1/nocode" and "year_bias" in out["papers"][-1]
    assert out["code"]["with_signal"] == 1 and out["code"]["github_searched"] == 2


def test_top_n_widened(cfg):
    c = _cfg(cfg, github_search=False)
    c["fusion"]["top_n"] = 1
    rt = Retriever(c, transport=FakeOA(), sources={"openalex"})
    out = rt.retrieve(T, seed_mode="none")
    assert len(out["papers"]) == 2               # round(1 * 2.75) = 3 >= 2 pool papers


def test_pwc_index(tmp_path):
    p = tmp_path / "pwc.tsv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("arxiv\ttitle_key\trepo_url\tis_official\tmentioned_in_paper\n")
        fh.write("2101.00001\tsomelongpapertitleaboutsmoothersandfilters\thttps://github.com/x/off\t1\t0\n")
        fh.write("2101.00001\tsomelongpapertitleaboutsmoothersandfilters\thttps://github.com/y/fan\t0\t0\n")
    ix = codeindex.PwcIndex(p)
    assert [h["repo"] for h in ix.lookup("2101.00001v2", None)] == ["https://github.com/x/off", "https://github.com/y/fan"]
    assert ix.lookup(None, "Some long paper title about smoothers and filters")[0]["official"]
    papers = [{"arxiv": "2101.00001", "title": "t", "code_links": []}]
    codeindex.annotate(papers, oa_flags=set(), pwc=ix, gh_hits={})
    assert papers[0]["code_signals"]["pwc"] == ["https://github.com/x/off"]
    assert papers[0]["code_signals"]["pwc_mentioned"] == []
    assert papers[0]["code_signals"]["pwc_unofficial"] == ["https://github.com/y/fan"]
    assert papers[0]["code_signals"]["any"]


def test_github_gh_client_cached_and_cooldown(cfg):
    c = _cfg(cfg)
    _, led, http = make_sources(c, FakeOA())
    calls = []

    def gh(path, params):
        calls.append(params["q"])
        return 200, {"items": [{"full_name": "a/b"}]}

    p = {"doi": "10.1/code"}
    assert codeindex.github_readme_search(http, c, p, gh=gh) == ["a/b"]
    assert codeindex.github_readme_search(http, c, p, gh=gh) == ["a/b"]
    assert len(calls) == 1                        # second time from cache

    def gh403(path, params):
        return 403, {"message": "rate limit"}
    assert codeindex.github_readme_search(http, c, {"doi": "10.1/other"}, gh=gh403) == []
    assert led.cooling("github") > 0


def test_meta_code_detection():
    assert codeindex.has_meta_code({"code_links": ["github.com/a/b"]})
    assert codeindex.has_meta_code({"code_links": ["https://zenodo.org/records/1"]})
    assert not codeindex.has_meta_code({"code_links": ["https://example.org/x"]})


def test_gh_branch_respects_offline(cfg):
    c = _cfg(cfg)
    _, led, http = make_sources(c, FakeOA())
    http.offline = True
    calls = []
    assert codeindex.github_readme_search(http, c, {"doi": "10.1/zzz"}, gh=lambda p, q: calls.append(q)) == []
    assert calls == []


def test_code_channel_hit_counts_as_signal():
    papers = [{"doi": "10.1/x", "title": "t", "code_links": [], "ranks": {"openalex.code": 3}}]
    codeindex.annotate(papers, oa_flags=set(), pwc=codeindex.PwcIndex(None), gh_hits={})
    assert papers[0]["code_signals"]["any"]
