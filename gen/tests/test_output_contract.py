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


def test_stated_headers_parsed_per_file():
    h = gates.stated_csv_headers(NANO_INS, ["fitted_curves.csv", "predictions.csv", "categories.csv", "x.csv"])
    assert h == {"fitted_curves.csv": ["material", "rate", "depth_nm", "H_fit_GPa"],
                 "predictions.csv": ["material", "rate", "depth_nm", "H_pred_GPa"],
                 "categories.csv": ["material", "category"]}


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
    assert gates.stated_csv_headers(ins, ["a.csv", "b.csv"]) == {"b.csv": ["material", "category"]}


def test_output_contract_is_blocking_by_default():
    assert "output_contract" in gates._blocking(load_config(None))
