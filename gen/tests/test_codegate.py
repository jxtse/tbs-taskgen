"""Code-grounded intake: S2a (code signal before download) and S3b (own code confirmed + resolved)."""
import json

import pytest

from tbsgen import codegate
from tbsgen.config import load_config
from tbsgen.pipeline import Pipeline
from tbsgen.state import read_json, write_json

FT_OWN = """Fast smoothing for data assimilation
Abstract. We propose a smoother ...
Methods ... The computations were performed with numpy [12].
Code availability
The code used in this study is available at https://github.com/alice/fastsmoother and archived on Zenodo
(https://doi.org/10.5281/zenodo.1234567).
References
[12] Harris et al. Array programming with NumPy. https://github.com/numpy/numpy
"""
FT_REQUEST = """A method paper
Code availability
The code is available from the corresponding author upon reasonable request.
References
"""
FT_THIRD = """Another paper
We used the CLASS Boltzmann code (https://github.com/lesgourg/class_public) as in [3].
References
[3] Lesgourgues. https://github.com/lesgourg/class_public
"""


def fake_get(table):
    def get(url):
        for k, v in table.items():
            if k in url:
                return v
        return 404, None
    return get


def fake_gh(table):
    def gh(path):
        return table.get(path, (404, None))
    return gh


def test_extract_and_decide_own_github():
    c = {"title": "Fast smoothing for data assimilation"}
    links, paper = codegate.extract_links(FT_OWN, c)
    by = {l["key"]: l for l in links}
    assert by["github.com/alice/fastsmoother"]["own_guess"] == "own"
    assert by["github.com/numpy/numpy"]["generic"]
    assert paper["code_availability"]
    gh = fake_gh({"repos/alice/fastsmoother": (200, {"size": 120, "language": "Python", "license": {"spdx_id": "MIT"},
                                                     "html_url": "https://github.com/alice/fastsmoother",
                                                     "default_branch": "main"})})
    get = fake_get({"zenodo.org/api/records/1234567": (200, {"metadata": {"resource_type": {"type": "software"}},
                                                              "files": [{"key": "fastsmoother-1.0.zip", "size": 9}]})})
    rec = codegate.s3b_record(FT_OWN, c, {"code": {}}, get=get, gh=gh, gh_raw=fake_gh({}))
    assert rec["keep"] and rec["primary"]["url"] == "https://github.com/alice/fastsmoother"
    assert rec["primary"]["license"] == "MIT"                 # recorded, never filtered


def test_on_request_and_third_party_and_404():
    c = {"title": "A method paper"}
    r = codegate.s3b_record(FT_REQUEST, c, {"code": {}}, get=fake_get({}), gh=fake_gh({}), gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "code_on_request"
    r = codegate.s3b_record(FT_THIRD, {"title": "Another paper"}, {"code": {}}, get=fake_get({}), gh=fake_gh({}),
                            gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "third_party_code_only"
    r = codegate.s3b_record(FT_OWN, {"title": "Fast smoothing"}, {"code": {}}, get=fake_get({}), gh=fake_gh({}),
                            gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "code_repo_unavailable"


def test_data_only_and_empty_repo():
    ft = "Paper\nData and code availability\nOur data are available at https://zenodo.org/records/7777.\nReferences\n"
    get = fake_get({"api/records/7777": (200, {"metadata": {"resource_type": {"type": "dataset"}},
                                               "files": [{"key": "obs.csv", "size": 5}]})})
    r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=get, gh=fake_gh({}), gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "data_only_no_code"
    gh = fake_gh({"repos/alice/fastsmoother": (200, {"size": 0})})
    r = codegate.s3b_record(FT_OWN, {"title": "Fast smoothing"}, {"code": {}}, get=fake_get({}), gh=gh,
                            gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "code_repo_unavailable"


def test_ambiguous_goes_to_jev_not_teacher():
    ft = "Paper X\nResults were computed with the scripts at https://github.com/bob/xtools.\nReferences\n"
    seen = []

    def jev(link):
        seen.append(link["key"])
        return 0.1
    r = codegate.s3b_record(ft, {"title": "Paper X"}, {"code": {}}, jev=jev, get=fake_get({}), gh=fake_gh({}),
                            gh_raw=fake_gh({}))
    assert seen == ["github.com/bob/xtools"] and not r["keep"] and r["reason"] == "third_party_code_only"


def test_readme_citing_paper_is_a_signal_not_a_verdict():
    ft = "Paper X\nResults were computed with the scripts at https://github.com/bob/xtools.\nReferences\n"
    gh_raw = fake_gh({"repos/bob/xtools/readme": (200, "Code for doi 10.1/xyz")})
    gh = fake_gh({"repos/bob/xtools": (200, {"size": 5, "language": "R"})})
    r = codegate.s3b_record(ft, {"title": "Paper X", "doi": "10.1/xyz"}, {"code": {}}, jev=lambda l: 0.0,
                            get=fake_get({}), gh=gh, gh_raw=gh_raw)
    assert not r["keep"]                      # Jev says third party: the README alone does not decide
    r = codegate.s3b_record(ft, {"title": "Paper X", "doi": "10.1/xyz"}, {"code": {}}, jev=lambda l: 0.9,
                            get=fake_get({}), gh=gh, gh_raw=gh_raw)
    assert r["keep"] and r["primary"]["language"] == "R"


def test_maybe_without_jev_follows_policy_and_jev_down_pauses():
    ft = "Paper X\nResults were computed with the scripts at https://github.com/bob/xtools.\nReferences\n"
    gh = fake_gh({"repos/bob/xtools": (200, {"size": 5})})
    r = codegate.s3b_record(ft, {"title": "Paper X"}, {"code": {}}, get=fake_get({}), gh=gh, gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "third_party_code_only"
    r = codegate.s3b_record(ft, {"title": "Paper X"}, {"code": {}}, jev=lambda l: None, get=fake_get({}), gh=gh,
                            gh_raw=fake_gh({}))
    assert not r["keep"] and r["pause"] == "code_owner_undecided"


def test_on_request_paper_not_rescued_by_reference_link():
    ft = ("Paper\nCode availability\nThe code is available upon reasonable request from the corresponding author."
          "\nReferences\n[1] Smith. Tool. https://github.com/smith/tool\n" + "x " * 50)
    r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=fake_get({}),
                            gh=fake_gh({"repos/smith/tool": (200, {"size": 9})}), gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "code_on_request"


def test_private_transient_bioconductor_zenodo_archive():
    c = {"title": "Fast smoothing"}
    r = codegate.s3b_record(FT_OWN, c, {"code": {}}, get=fake_get({}),
                            gh=fake_gh({"repos/alice/fastsmoother": (200, {"size": 9, "private": True})}),
                            gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "code_repo_unavailable"
    r = codegate.s3b_record(FT_OWN, c, {"code": {}}, get=fake_get({"zenodo": (503, None)}),
                            gh=fake_gh({"repos/alice/fastsmoother": (403, None)}), gh_raw=fake_gh({}))
    assert not r["keep"] and r["pause"] == "code_host_unavailable"
    ft = "Paper\nCode availability\nOur code is available at https://bioconductor.org/packages/fakepkg.\n"
    r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=fake_get({}), gh=fake_gh({}),
                            gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "code_repo_unavailable"
    ft = "Paper\nCode availability\nOur code and data are available at https://zenodo.org/records/5555.\n"
    get = fake_get({"api/records/5555": (200, {"metadata": {"resource_type": {"type": "dataset"}},
                                                "files": [{"key": "observations.zip", "size": 5}]})})
    r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=get, gh=fake_gh({}), gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "data_only_no_code"


def test_repo_names_kept():
    links, _ = codegate.extract_links("see https://github.com/a/solver.v2 and https://github.com/b/Foo.jl and "
                                      "https://github.com/d/Solver.Core and "
                                      "https://github.com/c/tool.Acknowledgments and "
                                      "https://gitlab.com/grp/sub/deeper/proj/-/tree/main", {})
    keys = {l["key"] for l in links}
    assert {"github.com/a/solver.v2", "github.com/b/foo.jl", "github.com/c/tool.acknowledgments",
            "github.com/d/solver.core",
            "gitlab.com/grp/sub/deeper/proj"} <= keys


def test_s2a_pwc_reaches_s3b(tmp_path, monkeypatch):
    P = _pipe(tmp_path)
    [a] = P.ingest([{"doi": "10.1/a", "title": "Some paper"}])
    it = P.store.item(a)
    write_json(it.p("S2a_code_signal.json"), {"pwc": ["https://github.com/alice/fastsmoother"], "any": True})
    it.state["stage"] = "S3b"
    it.save()
    it.p("fulltext.txt").write_text("Some paper\nNo links in this text.\n")
    seen = {}

    def fake_record(txt, c, cfg, jev=None, **kw):
        seen["cs"] = c["code_signals"]
        return dict(keep=False, reason="x", links=[], paper={})
    monkeypatch.setattr(codegate, "s3b_record", fake_record)
    P.run(max_steps=1)
    assert seen["cs"]["pwc"] == ["https://github.com/alice/fastsmoother"]


def test_fetch_delivery_uses_subscriber_next_stage(tmp_path):
    P = _pipe(tmp_path)
    [a] = P.ingest([{"doi": "10.1/a", "title": "Some paper"}])
    it = P.store.item(a)
    it.state["stage"] = "S3"
    it.save()
    P.run(max_steps=1)                         # no full text offline -> parked, queue entry written
    q = read_json(it.p("S3_queue.json"))
    assert P.store.item(a).state["status"] == "parked" and q["next_stage"] == "S3b"


def test_pdf_broken_url_joined():
    ft = "Code availability\nOur code is available at https://github.com/broadinstitute/\nCellBender\n"
    links, _ = codegate.extract_links(ft, {"title": "CellBender"})
    assert links[0]["key"] == "github.com/broadinstitute/cellbender"


def _pipe(tmp_path, **code):
    cfg = load_config(None, {"run": {"root": str(tmp_path)}, "code": dict(grounded=True, **code),
                             "fulltext": {"allow_network": False, "cache_dirs": []}})
    return Pipeline(cfg)


def test_s2a_drops_without_signal_and_routes(tmp_path, monkeypatch):
    P = _pipe(tmp_path)
    monkeypatch.setattr(codegate, "openalex_fulltext_flag", lambda c, cfg: False)
    [a, b] = P.ingest([{"doi": "10.1/a", "title": "No code paper about smoothing"},
                       {"doi": "10.1/b", "title": "Code paper", "code_links": ["github.com/x/y"]}])
    P.run(max_steps=4, until="S2a")
    sa, sb = P.store.item(a).state, P.store.item(b).state
    assert sa["status"] == "dropped" and sa["reason"] == "no_code_signal"
    assert sb["stage"] == "S2" and sb["status"] == "active"
    assert read_json(P.store.item(b).p("S2a_code_signal.json"))["meta"] == ["github.com/x/y"]


def test_s2a_off_without_grounded(tmp_path):
    cfg = load_config(None, {"run": {"root": str(tmp_path)}})
    P = Pipeline(cfg)
    [a] = P.ingest([{"doi": "10.1/a", "title": "x"}])
    P.run(max_steps=1, until="S1")
    assert P.store.item(a).state["stage"] == "S2"


def test_s3b_stage(tmp_path, monkeypatch):
    P = _pipe(tmp_path)
    [a] = P.ingest([{"doi": "10.1/a", "title": "A method paper"}])
    it = P.store.item(a)
    it.state["stage"] = "S3b"
    it.save()
    it.p("fulltext.txt").write_text(FT_REQUEST)
    P.run(max_steps=1)
    st = P.store.item(a).state
    assert st["status"] == "dropped" and st["reason"] == "code_on_request"
    assert read_json(it.p("S3b_code.json"))["paper"]["on_request"]


def test_r2_regressions():
    # candidate signals alone (metadata + README + title) never make a link "own" without Jev
    ft = "Fastsmoother paper\nWe compare with https://github.com/bob/fastsmoother in Section 3.\nReferences\n"
    c = {"title": "Fastsmoother paper", "code_links": ["https://github.com/bob/fastsmoother"],
         "code_signals": {"gh_readme": ["https://github.com/bob/fastsmoother"]}}
    gh = fake_gh({"repos/bob/fastsmoother": (200, {"size": 5})})
    r = codegate.s3b_record(ft, c, {"code": {}}, get=fake_get({}), gh=gh, gh_raw=fake_gh({}))
    assert not r["keep"]
    # code availability ends at the next heading: an acknowledgement link does not inherit it
    ft = ("Paper\nCode availability\nCode is available on request from the corresponding author.\n"
          "Acknowledgements\nWe thank the authors of https://github.com/x/tool for figures.\n")
    r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=fake_get({}),
                            gh=fake_gh({"repos/x/tool": (200, {"size": 5})}), gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "code_on_request"
    # unknown Jev + an own link that is private -> pause, not drop
    ft2 = FT_OWN.replace("References", "Acknowledgements\nScripts from https://github.com/bob/xtools.\nReferences")
    gh = fake_gh({"repos/alice/fastsmoother": (200, {"size": 9, "private": True}),
                  "repos/bob/xtools": (200, {"size": 5})})
    r = codegate.s3b_record(ft2, {"title": "Fast smoothing"}, {"code": {}}, jev=lambda l: None,
                            get=fake_get({"zenodo": (404, None)}), gh=gh, gh_raw=fake_gh({}))
    assert r.get("pause") == "code_owner_undecided"
    # every 5xx is transient
    for st in (507, 520, 522, 529):
        r = codegate.s3b_record(FT_OWN, {"title": "Fast smoothing"}, {"code": {}}, get=fake_get({"zenodo": (st, None)}),
                                gh=fake_gh({"repos/alice/fastsmoother": (st, None)}), gh_raw=fake_gh({}))
        assert r.get("pause") == "code_host_unavailable", st
    # a package whose source repo is private keeps the public package as primary
    ft = "Pkg paper\nCode availability\nOur code is available at https://pypi.org/project/fastpkg.\n"
    get = fake_get({"pypi.org/pypi/fastpkg": (200, {"info": {"name": "fastpkg", "project_urls":
                                                             {"Source": "https://github.com/a/fastpkg"}}})})
    gh = fake_gh({"repos/a/fastpkg": (200, {"size": 5, "private": True})})
    r = codegate.s3b_record(ft, {"title": "Pkg paper"}, {"code": {}}, get=get, gh=gh, gh_raw=fake_gh({}))
    assert r["keep"] and r["primary"]["kind"] == "package" and not r["primary"].get("private")


def test_r3_regressions():
    # numbered / colon headings end the Code Availability section and the ownership context
    for hdr in ("6 Acknowledgements", "Acknowledgements:", "6. Acknowledgments"):
        ft = ("Paper\nCode availability\nCode is available on request from the corresponding author.\n" + hdr +
              "\nWe thank the authors of https://github.com/x/tool for figures.\n")
        r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=fake_get({}),
                                gh=fake_gh({"repos/x/tool": (200, {"size": 5})}), gh_raw=fake_gh({}))
        assert not r["keep"] and r["reason"] == "code_on_request", hdr
    # a dotted name is kept; the glued-word cut is only tried when the full name 404s
    ft = "Paper\nCode availability\nOur code is available at https://github.com/a/Solver.Data\n"
    gh = fake_gh({"repos/a/Solver.Data": (200, {"size": 5}), "repos/a/Solver": (200, {"size": 5})})
    r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=fake_get({}), gh=gh, gh_raw=fake_gh({}))
    assert r["keep"] and r["primary"]["key"] == "github.com/a/solver.data" and not r["primary"].get("alt_used")
    ft = "Paper\nCode availability\nOur code is available at https://github.com/a/tool.Acknowledgments\n"
    gh = fake_gh({"repos/a/tool": (200, {"size": 5})})
    r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=fake_get({}), gh=gh, gh_raw=fake_gh({}))
    assert r["keep"] and r["primary"]["alt_used"] == "tool"
    # unsupported hosts drop, they do not pause
    ft = "Paper\nCode availability\nOur code is available at https://codeocean.com/capsule/1234567\n"
    r = codegate.s3b_record(ft, {"title": "Paper"}, {"code": {}}, get=fake_get({}), gh=fake_gh({}),
                            gh_raw=fake_gh({}))
    assert not r["keep"] and r["reason"] == "code_host_unsupported" and not r.get("pause")
