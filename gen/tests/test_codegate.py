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


def test_readme_citing_paper_settles_maybe():
    ft = "Paper X\nResults were computed with the scripts at https://github.com/bob/xtools.\nReferences\n"
    gh_raw = fake_gh({"repos/bob/xtools/readme": (200, "Code for doi 10.1/xyz")})
    gh = fake_gh({"repos/bob/xtools": (200, {"size": 5, "language": "R"})})
    r = codegate.s3b_record(ft, {"title": "Paper X", "doi": "10.1/xyz"}, {"code": {}}, jev=lambda l: 0.0,
                            get=fake_get({}), gh=gh, gh_raw=gh_raw)
    assert r["keep"] and r["primary"]["language"] == "R"


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
