from tbsgen import blocklist
from tbsgen.config import load_config
from tbsgen.pipeline import Pipeline
from tbsgen.state import read_json

RULES = load_config(None)["screen"]["blocklist"]


def test_default_rules_cover_the_five_publishers_and_old_elsevier():
    names = {r["name"] for r in RULES}
    assert names == {"elsevier-pre2000", "sage", "ams-meteorology", "powder-diffraction", "asce", "spie"}


def test_prefix_match_is_exact_registrant():
    assert blocklist.match({"doi": "10.1177/14759217251415195"}, RULES)["name"] == "sage"
    assert blocklist.match({"doi": "https://doi.org/10.1117/12.2280807"}, RULES)["name"] == "spie"
    assert blocklist.match({"doi": "10.11770/abc"}, RULES) is None          # different registrant
    assert blocklist.match({"doi": "10.1080/0950083"}, RULES) is None       # T&F not blocked
    assert blocklist.match({"title": "no doi"}, RULES) is None


def test_elsevier_year_rule_uses_year_then_pii():
    assert blocklist.match({"doi": "10.1016/j.x.2021.1", "year": 1999}, RULES)["name"] == "elsevier-pre2000"
    assert blocklist.match({"doi": "10.1016/j.x.2021.1", "year": "2000"}, RULES) is None
    assert blocklist.match({"doi": "10.1016/s0022-5096(97)00086-0"}, RULES)["name"] == "elsevier-pre2000"
    assert blocklist.match({"doi": "10.1016/s0022-5096(02)00086-0"}, RULES) is None
    assert blocklist.match({"doi": "10.1016/0377-0427(88)90342-1"}, RULES)["name"] == "elsevier-pre2000"
    assert blocklist.match({"doi": "10.1016/j.procs.2016.05.501"}, RULES) is None   # unknown year: never blocked


def test_invalid_year_falls_back_and_doi_variants_normalise():
    # implausible / malformed years are ignored (then the PII year, if any, decides)
    assert blocklist.match({"doi": "10.1016/j.x.2021.1", "year": 0}, RULES) is None
    assert blocklist.match({"doi": "10.1016/j.x.2021.1", "year": "20"}, RULES) is None
    assert blocklist.match({"doi": "10.1016/s0022-5096(97)00086-0", "year": "n.d."}, RULES)["name"] == \
        "elsevier-pre2000"
    assert blocklist.paper_year({"year": "1997-03-01"}) == 1997
    # "doi: " with a space, URL forms and percent-encoding
    assert blocklist.match({"doi": "doi: 10.1177/x"}, RULES)["name"] == "sage"
    assert blocklist.match({"doi": "https://doi.org/10.1016/S0022-5096%2897%2900086-0"}, RULES)["name"] == \
        "elsevier-pre2000"


def test_s2_drops_blocked_paper_without_screen_call(tmp_path):
    cfg = load_config(None, {"run": {"root": str(tmp_path / "run")}, "models": {"roles": {"screen": "rules"}}})
    P = Pipeline(cfg)
    a, b = P.ingest([
        {"doi": "10.1177/1475921", "title": "Damage localization", "abstract": "We propose an algorithm."},
        {"doi": "10.1016/s0010-4655(96)00141-5", "title": "Old CPC", "abstract": "A simulation code.",
         "year": 1997},
    ])
    P.run(only=[a, b], max_steps=4)
    for iid, name in ((a, "sage"), (b, "elsevier-pre2000")):
        it = P.store.item(iid)
        rec = read_json(it.p("S2_screen.json"))
        assert it.state["status"] == "dropped" and rec["reason"] == f"blocklist:{name}"
        assert rec["method"] == "blocklist"


def test_blocklist_can_be_disabled(tmp_path):
    cfg = load_config(None, {"run": {"root": str(tmp_path / "run")}, "models": {"roles": {"screen": "rules"}},
                             "screen": {"blocklist": []}})
    P = Pipeline(cfg)
    (a,) = P.ingest([{"doi": "10.1177/1475921", "title": "t", "abstract": "We propose an algorithm."}])
    P.run(only=[a], max_steps=2)
    assert read_json(P.store.item(a).p("S2_screen.json"))["method"] != "blocklist"
