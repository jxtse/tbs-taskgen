"""S2 Jev screen: offline tests (recorded / scripted System One responses; no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tbsgen import screen_jev
from tbsgen.budget import Ledger
from tbsgen.config import load_config
from tbsgen.intake import screen
from tbsgen.models import ModelError
from tbsgen.pipeline import Pipeline
from tbsgen.state import read_json, write_json

FX = json.loads((Path(__file__).parent / "fixtures" / "jev_responses.json").read_text())
FAY = dict(doi="10.1007/s10182-025-00550-5",
           title="Small area estimation of poverty indicators under bivariate Fay–Herriot model with correlated time "
                 "effects",
           abstract="This paper presents an area-level temporal bivariate linear mixed model, incorporating correlated "
                    "time effects for estimating socioeconomic indicators in small areas. The model is applied through "
                    "the residual maximum likelihood method, leading to the derivation of empirical best linear "
                    "unbiased predictors for these indicators. Additionally, an approximation of the mean square error "
                    "matrix (MSE) is provided and four MSE estimators are proposed. The first estimator involves a "
                    "plug-in approach to the MSE approximation, while the remaining estimators are based on parametric "
                    "bootstrap procedures. To assess the performance of the fitting algorithm, predictors, and MSE "
                    "estimators, three simulation experiments are carried out. An application to real data from the "
                    "2016 to 2022 Spanish Living Conditions Survey is conducted.",
           venue="AStA Advances in Statistical Analysis", year=2025)
NOABS = dict(doi="10.1557/x", title="Thermal drift in room temperature nanoindentation experiments: Measurement and "
                                   "correction")
REVIEW = dict(doi="10.1/review", title="A review of things", abstract="This review surveys perspectives.")


def answers(**p) -> dict:
    """Build an `answers` object from probabilities; paper_type=(choice, prob)."""
    base = dict(paper_type=("new_method", 0.95), paper_type_fits=0.9, computational_method=0.9,
                quantitative_check=0.8, single_machine=0.9, data_public=0.5, data_simulatable=0.9,
                black_box_queries=0.05, is_review=0.03, enough_info=0.9)
    base.update(p)
    out = {}
    for k, v in base.items():
        if k == "paper_type":
            probs = {o: 0.0 for o in screen_jev.PAPER_TYPES}
            probs[v[0]] = v[1]
            out[k] = {"type": "choice", "choice": v[0], "probabilities": probs, "confidence": v[1]}
        else:
            out[k] = {"type": "noul", "noul": v}
    return out


def resp(ans: dict, model="jev-1.14.0", tokens=1300) -> bytes:
    return json.dumps({"model": model, "answers": ans, "usage": {"input_tokens": tokens, "output_tokens": 250}}).encode()


class Transport:
    """Scripted HTTP: list of (status, headers, body) consumed in order; the last entry repeats."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def __call__(self, url, body, headers, timeout):
        self.calls.append(dict(url=url, body=json.loads(body), auth=headers.get("Authorization"), timeout=timeout))
        s = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(s, Exception):
            raise s
        return s


def jcfg(tmp_path, **screen_jev_over):
    return load_config(None, {"run": {"root": str(tmp_path)}, "budget": {"daily_usd_total": 1.0},
                              "models": {"roles": {"screen": "jev"}},
                              "screen": {"jev": dict({"backoff_base": 0.0}, **screen_jev_over)}})


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")


# ------------------------------------------------------------------ combination rule
def test_combine_keep_drop_uncertain():
    P = screen_jev.probabilities
    assert screen_jev.combine(P(answers()))["decision"] == "keep"
    d = screen_jev.combine(P(answers(paper_type=("review", 0.97), computational_method=0.1, is_review=0.98)))
    assert d["decision"] == "drop" and d["reason"] == "jev_type_review"
    d = screen_jev.combine(P(answers(paper_type=("clinical_or_qualitative", 0.99), computational_method=0.2)))
    assert d["decision"] == "drop" and d["reason"] == "jev_type_clinical_or_qualitative"
    assert screen_jev.combine(P(answers(computational_method=0.2)))["reason"] == "jev_no_computational_method"
    assert screen_jev.combine(P(answers(single_machine=0.05)))["reason"] == "jev_needs_heavy_compute"
    assert screen_jev.combine(P(answers(data_public=0.1, data_simulatable=0.2)))["reason"] == "jev_no_data"
    d = screen_jev.combine(P(answers(computational_method=0.55, enough_info=0.2)))
    assert d["decision"] == "uncertain" and "computational_method" in d["open"] and "not_enough_info" in d["open"]
    # several questions leaning no at once: joint probability below joint_no -> drop
    d = screen_jev.combine(P(answers(computational_method=0.5, single_machine=0.6, data_simulatable=0.6)))
    assert d["decision"] == "drop" and d["reason"].startswith("jev_low_joint")
    # a review-type answer with a strong method signal is not dropped on type alone
    d = screen_jev.combine(P(answers(paper_type=("review", 0.8), computational_method=0.85)))
    assert d["decision"] == "keep"
    # a choice that "fits" poorly does not drive a drop
    d = screen_jev.combine(P(answers(paper_type=("review", 0.9), paper_type_fits=0.2, computational_method=0.45)))
    assert d["decision"] == "uncertain"
    assert screen_jev.combine(P(answers(black_box_queries=0.9)))["form_hint"] == "INTERACT"


def test_thresholds_override_from_config():
    p = screen_jev.probabilities(answers(computational_method=0.6))
    assert screen_jev.combine(p)["decision"] == "uncertain"
    assert screen_jev.combine(p, {"method_no": 0.65})["reason"] == "jev_no_computational_method"
    assert screen_jev.combine(p, {"joint_no": 0.6})["reason"].startswith("jev_low_joint")


def test_recorded_fay_herriot_kept(tmp_path):
    rec = screen_jev.record_from_answer(dict(FX["fay_herriot"], usd=0.0), jcfg(tmp_path), False)
    assert rec["keep"] and rec["decision"] == "keep" and rec["form_hint"] is None
    assert rec["jev_model"] == "jev-1.14.0" and rec["input_tokens"] == 1300
    assert set(screen_jev.QUESTIONS) <= set(rec["probabilities"]) | {"paper_type"}


def test_uncertain_policy(tmp_path):
    r = dict(answers=answers(computational_method=0.6), model="jev-1.14.0", usage={"input_tokens": 1})
    keep = screen_jev.record_from_answer(r, jcfg(tmp_path), False)
    assert keep["keep"] and keep["uncertain"] and keep["reason"] is None
    drop = screen_jev.record_from_answer(r, jcfg(tmp_path, uncertain="drop"), False)
    assert not drop["keep"] and drop["reason"] == "jev_uncertain_dropped"


# ------------------------------------------------------------------ state
def test_state_shape():
    st = screen_jev.build_state(dict(FAY, code_links=[], data_links=None))
    assert set(st["paper"]) == {"title", "abstract", "venue", "year"}          # empty lists left out
    st = screen_jev.build_state(NOABS, fulltext_head="x" * 9000)
    assert "abstract" not in st["paper"] and len(st["paper"]["full_text_opening"]) == 4000


# ------------------------------------------------------------------ client: retries, errors, ledger
def test_retry_429_529_honours_retry_after(tmp_path, key):
    cfg = jcfg(tmp_path)
    slept = []
    tr = Transport([(429, {"Retry-After": "2"}, b"slow down"), (529, {}, b"overloaded"),
                    (200, {}, resp(answers()))])
    cli = screen_jev.JevClient(Ledger(tmp_path / "l.sqlite", cfg), cfg, transport=tr, sleep=slept.append)
    r = cli.ask(screen_jev.build_state(FAY), stage="S2", item="i1")
    assert r["attempts"] == 3 and slept[0] == 2.0 and len(slept) == 2
    assert tr.calls[0]["auth"] == "Bearer test-key-not-real" and tr.calls[0]["body"]["model"] == "jev-1.14.0"
    assert set(tr.calls[0]["body"]["questions"]) == set(screen_jev.QUESTIONS)
    s = cli.ledger.summary()
    assert s["calls_total"] == 1 and s["rows"][0]["in_tok"] == 1300
    assert abs(s["usd_total"] - round(1300 * 0.042 / 1e6, 4)) < 1e-9


def test_retry_cap_and_no_retry_on_4xx(tmp_path, key):
    cfg = jcfg(tmp_path, max_retries=2)
    led = Ledger(tmp_path / "l.sqlite", cfg)
    tr = Transport([(429, {}, b"rate")])
    with pytest.raises(ModelError) as e:
        screen_jev.JevClient(led, cfg, transport=tr, sleep=lambda s: None).ask({"paper": {}}, stage="S2")
    assert e.value.cls == "rate" and len(tr.calls) == 3
    tr = Transport([(422, {}, b'{"detail":"bad question"}')])
    with pytest.raises(ModelError) as e:
        screen_jev.JevClient(led, cfg, transport=tr, sleep=lambda s: None).ask({"paper": {}}, stage="S2")
    assert e.value.cls == "other" and len(tr.calls) == 1


def test_auth_opens_breaker_and_model_pin(tmp_path, key):
    cfg = jcfg(tmp_path)
    led = Ledger(tmp_path / "l.sqlite", cfg)
    with pytest.raises(ModelError):
        screen_jev.JevClient(led, cfg, transport=Transport([(401, {}, b"bad key")])).ask({"paper": {}}, stage="S2")
    assert led.breaker("jev-1.14.0")["open"]
    led.reset_breaker("jev-1.14.0")
    with pytest.raises(ModelError) as e:
        screen_jev.JevClient(led, cfg, transport=Transport([(200, {}, resp(answers(), model="jev-1.15.0"))])
                             ).ask({"paper": {}}, stage="S2")
    assert e.value.cls == "model_mismatch" and led.breaker("jev-1.14.0")["open"]


def test_missing_key_raises_without_ledger_row(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    cfg = jcfg(tmp_path)
    led = Ledger(tmp_path / "l.sqlite", cfg)
    with pytest.raises(ModelError) as e:
        screen_jev.JevClient(led, cfg, transport=Transport([(200, {}, b"")])).ask({"paper": {}}, stage="S2")
    assert e.value.cls == "auth" and led.summary()["calls_total"] == 0


# ------------------------------------------------------------------ pipeline integration
def mkp(tmp_path, script, **over):
    P = Pipeline(jcfg(tmp_path, **over))
    P.jev_transport = Transport(script)
    return P


def test_pipeline_s2_jev_keep_drop_and_resume_is_idempotent(tmp_path, key):
    bad = answers(paper_type=("review", 0.99), computational_method=0.05, is_review=0.99)
    script = [(200, {}, resp(answers())), (200, {}, resp(bad))]
    P = mkp(tmp_path, script, concurrency=1)
    a, b = P.ingest([FAY, REVIEW])
    P.run(only=[a, b], max_steps=4)                 # S1 for both, then S2 for both (prefetched)
    P.run(only=[a, b], max_steps=2)                 # prefetch at run start: answers already cached
    sa, sb = (P.store.item(i) for i in (a, b))
    ra, rb = read_json(sa.p("S2_screen.json")), read_json(sb.p("S2_screen.json"))
    assert ra["method"] == rb["method"] == "jev"
    assert ra["keep"] and not rb["keep"] and rb["reason"] == "jev_type_review"
    assert sb.state["status"] == "dropped" and sa.state["stage"] != "S2"
    assert len(P.jev_transport.calls) == 2
    # re-running S2 on the same items (resume) must use the cache, not the API
    for it in (sa, sb):
        it.state.update(stage="S2", status="active")
        it.save()
    P2 = Pipeline(P.cfg)
    P2.jev_transport = Transport([(500, {}, b"must not be called")])
    P2.run(only=[a, b], max_steps=2)
    assert P2.jev_transport.calls == [] and read_json(sa.p("S2_screen.json"))["cached"]
    assert read_json(sa.p("S2_screen.json"))["keep"] == ra["keep"]


def test_pipeline_fallback_to_rules_when_key_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    P = mkp(tmp_path, [(200, {}, resp(answers()))])
    (i,) = P.ingest([FAY])
    P.run(only=[i], max_steps=3)
    rec = read_json(P.store.item(i).p("S2_screen.json"))
    assert rec["method"] == "rules-v2-fallback" and rec["fallback_reason"].startswith("auth")
    assert rec["keep"]                                  # rules-v2: "Living Conditions Survey" is not a review
    assert P.jev_transport.calls == []


def test_pipeline_fallback_when_breaker_open(tmp_path, key):
    P = mkp(tmp_path, [(200, {}, resp(answers()))])
    P.ledger.open_breaker("jev-1.14.0", "test")
    (i,) = P.ingest([FAY])
    P.run(only=[i], max_steps=3)
    rec = read_json(P.store.item(i).p("S2_screen.json"))
    assert rec["method"] == "rules-v2-fallback" and "breaker_open" in rec["fallback_reason"]


def test_consecutive_errors_open_breaker_then_fallback(tmp_path, key):
    P = mkp(tmp_path, [(500, {}, b"boom")], max_retries=0, concurrency=1)
    ids = P.ingest([dict(FAY, doi=f"10.9/{k}") for k in range(6)])
    P.run(only=ids, max_steps=24)
    assert P.ledger.breaker("jev-1.14.0")["open"]
    # 4 consecutive failed logical calls open the breaker; later items do not call at all
    assert len(P.jev_transport.calls) == 4
    recs = [read_json(P.store.item(i).p("S2_screen.json")) for i in ids]
    assert all(r["method"] == "rules-v2-fallback" for r in recs)


def test_pipeline_park_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    P = mkp(tmp_path, [(200, {}, resp(answers()))], on_unavailable="park")
    (i,) = P.ingest([FAY])
    P.run(only=[i], max_steps=3)
    st = P.store.item(i).state
    assert st["status"] == "paused" and st["reason"].startswith("model:jev_unavailable")
    assert P.resume() == [i] and P.store.item(i).state["stage"] == "S2"


def test_deferred_no_abstract_then_s4_asks_jev_on_fulltext(tmp_path, key):
    from tbsgen import stages_a
    P = mkp(tmp_path, [(200, {}, resp(answers()))])
    (i,) = P.ingest([NOABS])
    it = P.store.item(i)
    stages_a.s1(P, it)
    stages_a.s2(P, it)
    rec = read_json(it.p("S2_screen.json"))
    assert rec["deferred"] and rec["keep"] and rec["method"] == "jev" and P.jev_transport.calls == []
    # S4 re-screen path (as stages_a.s4 calls it)
    rec2 = stages_a.screen_paper(P, it, NOABS, "S4", fulltext_head="We propose a drift correction algorithm ...")
    assert rec2["method"] == "jev" and rec2["input"] == "fulltext_head" and rec2["keep"]
    body = P.jev_transport.calls[0]["body"]
    assert "full_text_opening" in body["state"]["paper"] and "abstract" not in body["state"]["paper"]
    assert P.ledger.summary()["rows"][0]["stage"] == "S4"


def test_legacy_s2_record_still_read(tmp_path, key):
    """An S2_screen.json written by rules-v1 (old run dir) keeps working: s4 reads only keep/deferred/form_hint."""
    from tbsgen import intake
    legacy = {"score": 5, "pos": ["estimat"], "neg": [], "keep": True, "form_hint": "INTERACT", "method": "rules-v1",
              "reason": None}
    r = intake.probe(FAY, "We propose an estimation algorithm. " * 400, legacy, load_config(), None)
    assert "interact_form_banned_until_isolation" in r["reasons"]
    P = mkp(tmp_path, [(500, {}, b"must not be called")])
    (i,) = P.ingest([NOABS])
    it = P.store.item(i)
    write_json(it.p("S2_screen.json"), legacy)          # not deferred -> S4 must not re-screen
    it.state.update(stage="S4", status="active")
    it.save()
    it.p("fulltext.txt").write_text("We propose an estimation algorithm with simulation. " * 400)
    from tbsgen import stages_a
    stages_a.s4(P, it)
    assert P.jev_transport.calls == []


# ------------------------------------------------------------------ rules-v2 fixes
def test_rules_v2_counts_stems_once_and_survey_is_not_a_review():
    v1, v2 = load_config(overrides={"screen": {"rules_version": "v1"}}), load_config()
    t = dict(title="x", abstract="estimating estimation estimator estimators")
    assert screen(t, v1)["score"] == 4 and screen(t, v2)["score"] == 1
    r1, r2 = screen(FAY, v1), screen(FAY, v2)
    assert "survey" in r1["neg"] and not r2["neg"]
    assert len(r2["pos"]) < len(r1["pos"])          # variants (estimating / estimators ...) counted once
    assert r1["method"] == "rules-v1" and r2["method"] == "rules-v2"
    lit = dict(title="Methods for X", abstract="This survey of the literature covers estimation methods.")
    assert screen(lit, v2)["neg"]
    assert not screen(REVIEW, v2)["keep"]
