"""assumptions_stated: format-only differences pass, rewording fails; S6C / S8 / review revisions are told the exact
S5G quotes S7 checks (a paraphrase of a correctly disclosed assumption dropped 6 papers on 2026-11-27)."""
import json
from types import SimpleNamespace

from tbsgen import stages_b
from tbsgen.config import load_config
from tbsgen.statement import assumptions_stated, quote_block, quote_key, stated_quotes

A = dict(assumption="match rule", stated=True,
         where="take the predicted nucleus that contains the\n  largest number of its particles")


def test_markdown_and_typography_do_not_hide_a_quote():
    assert not assumptions_stated("Gravity: it points **down** and has norm 9.81 m/s^2.",
                                  [dict(A, where="it points **down** and has norm 9.81")])
    assert not assumptions_stated("Gravity: it points down and has norm 9.81.",
                                  [dict(A, where="it points **down** and has norm 9.81")])
    # the real 2026-11-27 case (doi-5837...): the quote opens a list item, so S6C capitalized it
    assert not assumptions_stated("   - It points **down** and has norm 9.81 m/s\u00b2.",
                                  [dict(A, where="it points **down** and has norm 9.81")])
    assert not assumptions_stated("the rectifier ReLU(z) = max(z, 0) (named in meta.json)",
                                  [dict(A, where="the rectifier `ReLU(z) = max(z, 0)` (named in `meta.json`)")])
    assert not assumptions_stated("the water\u2019s sound speed", [dict(A, where="the water's sound speed")])
    assert not assumptions_stated("each crystallite: *take the predicted nucleus that contains the largest number "
                                  "of its particles*.", [A])


def test_reworded_quote_still_fails():
    # the real S6C rewrite of 2026-11-27 (doi-ed0a...): same meaning, different words -> the gate keeps failing
    text = "each true crystallite is matched to *the predicted nucleus that contains the largest number of its particles*"
    assert assumptions_stated(text, [A]) == [A["where"]]
    assert assumptions_stated("standard deviation", [dict(A, where="Gaussian additive error with std")])
    assert assumptions_stated("Other noise", [dict(A, where="Additive Gaussian noise")])


def test_quote_key_keeps_words_and_math():
    assert quote_key("`z = e + s\u00b7\u03b4\u00b7y`") == "z = e + s\u00b7\u03b4\u00b7y"
    assert quote_key("a \u2212 b") == "a - b" and quote_key("Sigma_w") == "sigmaw"


def test_quote_block_lists_only_stated_quotes():
    spec_a = [A, dict(assumption="noise", stated=False, where="recovered from data: residual variance"),
              dict(assumption="bad", stated=True, where="  ")]
    assert stated_quotes(spec_a) == [" ".join(A["where"].split())]
    blk = quote_block(spec_a)
    assert "take the predicted nucleus that contains the largest number of its particles" in blk
    assert "recovered from data" not in blk and "word for word" in blk
    assert quote_block([]) == "" and quote_block(None) == ""


def _item(tmp_path, blocking):
    (tmp_path / "gates").mkdir()
    (tmp_path / "gates" / "attempt-1.json").write_text(json.dumps(dict(
        gates={"assumptions_stated": False}, blocking=blocking,
        details={"assumptions_stated": [A["where"]]})))
    (tmp_path / "_spec").mkdir()
    (tmp_path / "_spec" / "spec.json").write_text(json.dumps(dict(depgraph={}, generator_assumptions=[A])))
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "instruction.md").write_text("statement\n")
    saved = {}
    it = SimpleNamespace(state=dict(repair_from="gates:" + ",".join(blocking)), item_id="doi-x",
                         p=lambda *a: tmp_path.joinpath(*a), save=lambda: None,
                         goto=lambda *a: saved.setdefault("goto", a), finish=lambda *a, **k: saved.setdefault("finish", a))
    return it, saved


def _run_s8(tmp_path, monkeypatch, blocking):
    it, saved = _item(tmp_path, blocking)
    seen = {}

    class Client:
        def chat(self, system, user, **kw):
            seen["user"] = user
            return SimpleNamespace(text="", stop_reason="end_turn")
    monkeypatch.setattr(stages_b, "save_raw", lambda *a, **k: None)
    P = SimpleNamespace(cfg=load_config(None), client=lambda role: Client())
    stages_b.s8(P, it)
    return seen["user"], saved


def test_s8_repair_prompt_names_the_exact_quotes(tmp_path, monkeypatch):
    user, saved = _run_s8(tmp_path, monkeypatch, ["assumptions_stated"])
    assert "take the predicted nucleus that contains the largest number of its particles" in user
    assert "word for word" in user and saved["goto"][0] == "S8"     # empty patch -> rejected, as before


def test_s8_repair_prompt_without_the_gate_is_unchanged(tmp_path, monkeypatch):
    user, _ = _run_s8(tmp_path, monkeypatch, ["effort_floor_x"])
    assert "word for word" not in user


def test_s6c_prompt_names_the_exact_quotes(tmp_path, monkeypatch):
    seen = {}

    def chat_files(P, it, role, stage, user, need, **kw):
        seen["user"] = user
        return {}, []
    monkeypatch.setattr(stages_b, "chat_files", chat_files)
    (tmp_path / "_spec").mkdir()
    (tmp_path / "_spec" / "spec.json").write_text(json.dumps(dict(depgraph={}, generator_assumptions=[A])))
    (tmp_path / "thresholds.json").write_text(json.dumps(dict(thresholds={"m": dict(threshold=1.0, direction="max")})))
    (tmp_path / "S6A_measure.json").write_text(json.dumps(dict(ref={}, mid={}, naive={})))
    (tmp_path / "task" / "tests").mkdir(parents=True)
    fin = {}
    it = SimpleNamespace(state={}, item_id="doi-x", p=lambda *a: tmp_path.joinpath(*a), save=lambda: None,
                         finish=lambda *a, **k: fin.setdefault("finish", a))
    stages_b.s6c(SimpleNamespace(cfg=load_config(None)), it)
    assert "take the predicted nucleus that contains the largest number of its particles" in seen["user"]
    assert fin["finish"][1] == "s6c_missing_files"
