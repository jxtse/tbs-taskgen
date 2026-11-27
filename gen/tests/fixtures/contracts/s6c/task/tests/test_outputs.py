"""Output-side checks of one agent run (fixture: shape of the S6C verifier after the S6C rewrite)."""
import csv
import json
import math
from pathlib import Path

OUT = Path("/app/output")


def _rows(name):
    with open(OUT / name, newline="") as fh:
        return list(csv.DictReader(fh))


def test_matches_header_and_rows():
    rows = _rows("matches.csv")
    assert rows and set(rows[0]) == {"crystallite_id", "orientation_deg", "size_nm"}
    assert [r["crystallite_id"] for r in rows] == sorted(r["crystallite_id"] for r in rows)
    for r in rows:
        assert 0.0 <= float(r["orientation_deg"]) < 180.0
        assert float(r["size_nm"]) > 0.0


def test_metrics():
    metrics = json.loads((OUT / "metrics.json").read_text())
    assert metrics["match_f1"] >= 0.85
    assert math.isfinite(metrics["median_angle_err_deg"])
