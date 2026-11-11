"""S7 output_contract gate (v2.7): CSV headers stated in the instruction must match the reference's outputs.
Regression for e2e-batch2 (nano indentation task): the statement gave `material,rate,depth_nm,H_fit_GPa` while the
scorer read obs_id / H_fit, so an agent following the text (Opus included) scored inf on 4 of 5 metrics."""
from tbsgen import gates
from tbsgen.config import load_config

NANO_INS = """## Deliverables
Write all four CSV files, each with exactly the header shown.

1. `fitted_curves.csv`: your model's fitted hardness at every row of `measurements.csv`.
   Header: `material,rate,depth_nm,H_fit_GPa`
2. `predictions.csv`: predicted hardness at every row of `queries_curve.csv`.
   Header: `material,rate,depth_nm,H_pred_GPa`
3. `categories.csv`: one row per material.
   Header: `material,category`
"""


def _out(tmp_path, files):
    d = tmp_path / "out-ref0"
    d.mkdir()
    for n, hdr in files.items():
        (d / n).write_text(hdr + "\n1,2,3,4\n")
    return d


def _cols(h):
    return {k: [d["cols"] for d in v] for k, v in h.items()}


def test_stated_headers_parsed_per_file():
    h = gates.stated_csv_headers(NANO_INS, ["fitted_curves.csv", "predictions.csv", "categories.csv", "x.csv"])
    assert _cols(h) == {"fitted_curves.csv": [["material", "rate", "depth_nm", "H_fit_GPa"]],
                        "predictions.csv": [["material", "rate", "depth_nm", "H_pred_GPa"]],
                        "categories.csv": [["material", "category"]]}


def test_mismatch_flagged(tmp_path):
    d = _out(tmp_path, {"fitted_curves.csv": "obs_id,material,rate,depth_nm,H_fit",
                        "predictions.csv": "material,rate,depth_nm,H_pred_GPa",
                        "categories.csv": "material,category"})
    r = gates.output_contract(NANO_INS, d)
    assert not r["ok"] and list(r["mismatch"]) == ["fitted_curves.csv"]


def test_match_in_any_column_order_passes(tmp_path):
    d = _out(tmp_path, {"fitted_curves.csv": "rate,material,depth_nm,H_fit_GPa",
                        "predictions.csv": "material,rate,depth_nm,H_pred_GPa",
                        "categories.csv": "material,category"})
    assert gates.output_contract(NANO_INS, d)["ok"]


def test_no_stated_header_or_no_csv_is_not_judged(tmp_path):
    d = _out(tmp_path, {"other.csv": "a,b"})
    r = gates.output_contract("Write `other.csv` with your results.\n", d)
    assert r["ok"] and r["checked"] == []
    assert gates.output_contract(NANO_INS, tmp_path / "missing")["ok"]


def test_header_of_a_neighbouring_file_is_not_borrowed(tmp_path):
    ins = "- `a.csv`: per row.\n- `b.csv`: per material. Header: `material,category`\n"
    assert _cols(gates.stated_csv_headers(ins, ["a.csv", "b.csv"])) == {"b.csv": [["material", "category"]]}


def test_output_contract_is_blocking_by_default():
    assert "output_contract" in gates._blocking(load_config(None))


def test_input_column_lists_without_header_keyword_are_not_judged(tmp_path):
    """Codex r1: a comma list describing INPUT columns next to an output name must not count as its header."""
    ins = "- `out.csv`: one row per row of `measurements.csv` (`obs_id,material,rate`).\n"
    d = _out(tmp_path, {"out.csv": "obs_id,H"})
    assert gates.output_contract(ins, d) == dict(ok=True, mismatch={}, checked=[])


def test_two_lists_after_keyword_or_two_outputs_on_a_line_are_ambiguous():
    ins = ("- `a.csv` header: `x,y` (same as `u,v`)\n"
           "- `b.csv` and `c.csv`, Header: `p,q`\n")
    assert gates.stated_csv_headers(ins, ["a.csv", "b.csv", "c.csv"]) == {}


def test_csv_parsing_keeps_inner_spaces_and_strips_quotes(tmp_path):
    ins = "- `t.csv`. Header: `sample id, value`\n"
    d = tmp_path / "o"
    d.mkdir()
    (d / "t.csv").write_text('"sample id","value"\n1,2\n')
    r = gates.output_contract(ins, d)
    assert r["ok"] and r["checked"] == ["t.csv"], r


def test_same_basename_in_two_dirs_needs_the_path(tmp_path):
    d = tmp_path / "o"
    (d / "a").mkdir(parents=True)
    (d / "b").mkdir()
    (d / "a" / "out.csv").write_text("x,y\n")
    (d / "b" / "out.csv").write_text("p,q\n")
    ins = "- `a/out.csv`: Header: `x,y`\n- `b/out.csv`: Header: `p,q`\n- `out.csv` Header: `z,w`\n"
    r = gates.output_contract(ins, d)
    assert r["ok"] and r["checked"] == ["a/out.csv", "b/out.csv"], r


def test_every_declaration_is_checked(tmp_path):
    ins = "- `o.csv`: Header: `a,b`\n\n## Format\n`o.csv` columns: `a,c`\n"
    d = _out(tmp_path, {"o.csv": "a,b"})
    r = gates.output_contract(ins, d)
    assert not r["ok"] and r["mismatch"]["o.csv"]["stated"] == ["a", "c"], r


def test_markdown_table_with_separate_backticked_columns_is_not_judged(tmp_path):
    ins = "| file | columns |\n|---|---|\n| `o.csv` | `id`, `wrong` |\n"
    d = _out(tmp_path, {"o.csv": "id,value"})
    assert gates.output_contract(ins, d)["checked"] == []


def test_real_nano_statement_shape_is_flagged(tmp_path):
    d = _out(tmp_path, {"fitted_curves.csv": "obs_id,material,rate,depth_nm,H_fit",
                        "predictions.csv": "query_id,material,rate,depth_nm,H_pred",
                        "categories.csv": "material,category"})
    r = gates.output_contract(NANO_INS, d)
    assert sorted(r["mismatch"]) == ["fitted_curves.csv", "predictions.csv"]
    assert r["mismatch"]["fitted_curves.csv"]["reference"][0] == "obs_id"


def test_s8_slim_carries_the_mismatch_to_the_repair_prompt():
    from tbsgen.stages_b import _slim
    g = dict(gates={"output_contract": False}, blocking=["output_contract"],
             details={"output_contract": {"ok": False, "mismatch": {"o.csv": {"stated": ["a"], "reference": ["b"]}}}})
    out = _slim(g)
    assert out["details"]["output_contract"]["mismatch"]["o.csv"]["reference"] == ["b"] and "fix_hint" in out


def test_input_attributed_and_partial_declarations_are_skipped(tmp_path):
    """Codex r2: an input file's columns on the output's line, and partial lists, are not exhaustive headers."""
    d = _out(tmp_path, {"out.csv": "id,value,diag"})
    for ins in ("`out.csv`: derived from `input.csv` (columns: `id,x`)\n",
                "`out.csv` includes columns `id,value`; additional columns are allowed\n",
                "`out.csv`: header at least `id,value`\n"):
        assert gates.output_contract(ins, d)["checked"] == [], ins


def test_prefixed_paths_resolve_and_compressed_names_do_not(tmp_path):
    d = tmp_path / "o"
    (d / "results").mkdir(parents=True)
    (d / "results" / "out.csv").write_text("id,value\n")
    for ins in ("Write `./results/out.csv`. Header: `id,wrong`\n",
                "Write `/app/output/results/out.csv`. Header: `id,wrong`\n",
                "Write `out.csv`. Header: `id,wrong`\n"):
        assert not gates.output_contract(ins, d)["ok"], ins
    assert gates.output_contract("Write `out.csv.gz`. Header: `id,wrong`\n", d)["checked"] == []


def test_keyword_without_list_does_not_hide_a_later_declaration(tmp_path):
    ins = "`out.csv` has a header row.\nHeader: `id,wrong`\n"
    d = _out(tmp_path, {"out.csv": "id,value"})
    r = gates.output_contract(ins, d)
    assert not r["ok"] and r["mismatch"]["out.csv"]["line"] == 2


def test_locked_retry_feedback_carries_full_mismatch(tmp_path, monkeypatch):
    """After a review revision (thresholds locked) an output_contract failure goes back to the revision with the
    file / stated / reference columns in the feedback (not truncated to the gate name)."""
    import json
    from types import SimpleNamespace
    from tbsgen import stages_b
    seen = {}
    monkeypatch.setattr(stages_b, "_revision_retry", lambda P, it, why: seen.setdefault("why", why))
    (tmp_path / "gates").mkdir()
    wide = ["c%02d" % i for i in range(30)]
    (tmp_path / "gates" / "attempt-3.json").write_text(json.dumps(dict(details=dict(output_contract=dict(
        ok=False, mismatch={"o.csv": dict(stated=wide[:-1], line=9, reference=wide)})))))
    it = SimpleNamespace(state=dict(repair_from="gates:output_contract", thresholds_locked=True),
                         p=lambda *a: tmp_path.joinpath(*a))
    stages_b.s8(SimpleNamespace(cfg=load_config(None)), it)
    assert "c29" in seen["why"] and '"line": 9' in seen["why"]


def test_codex_r3_false_positive_shapes_are_not_judged(tmp_path):
    d = _out(tmp_path, {"out.csv": "id,value,extra"})
    for ins in ("`out.csv`: one row per `input.feather` (columns: `id,x`)\n",
                "`out.csv`: one row per `input.csv.gz` (columns: `id,x`)\n",
                "`out.csv` corresponds to `input.csv`, whose\ncolumns are `id,x`\n",
                "`out.csv` corresponds to the input, whose\nColumns: `id,x`\n",
                "`out.csv` has these columns, e.g. `id,value`\n",
                "`out.csv`: columns `id` and `value`; accepted labels are `low,high`\n"):
        assert gates.output_contract(ins, d)["checked"] == [], ins


def test_output_dir_placeholder_prefix_resolves(tmp_path):
    d = tmp_path / "o"
    (d / "results").mkdir(parents=True)
    (d / "results" / "out.csv").write_text("id,value\n")
    assert not gates.output_contract("Write `<output_dir>/results/out.csv`. Header: `id,wrong`\n", d)["ok"]


def test_latest_attempt_is_chosen_numerically(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tbsgen import stages_b
    seen = {}
    monkeypatch.setattr(stages_b, "_revision_retry", lambda P, it, why: seen.setdefault("why", why))
    (tmp_path / "gates").mkdir()
    (tmp_path / "gates" / "attempt-9.json").write_text(json.dumps(dict(details=dict(output_contract=dict(
        ok=False, mismatch={"stale.csv": dict(stated=["a", "b"], line=1, reference=["a", "c"])})))))
    (tmp_path / "gates" / "attempt-10.json").write_text(json.dumps(dict(details=dict(output_contract=dict(
        ok=False, mismatch={"new.csv": dict(stated=["a", "b"], line=2, reference=["a", "z"])})))))
    it = SimpleNamespace(state=dict(repair_from="gates:output_contract", thresholds_locked=True),
                         p=lambda *a: tmp_path.joinpath(*a))
    stages_b.s8(SimpleNamespace(cfg=load_config(None)), it)
    assert "new.csv" in seen["why"] and "stale.csv" not in seen["why"] and '"z"' in seen["why"]
