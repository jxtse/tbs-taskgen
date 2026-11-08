"""Build eval/targets10.json from <baseline>/targets_sample.json (same 10 targets, same 4 queries as the earlier
OpenAlex/Crossref/arXiv benchmark) + read-only metadata from <earlier pipeline>/literature/methodlit/targets.json
(libs, method text).  Neither input is part of this repository:

    uv run python eval/build_targets.py --sample <baseline>/targets_sample.json --methodlit <...>/targets.json

`topic` (short English phrase for the Exa resource channels) is hand-written here for the 5 task targets because
task ids are terse; in production this phrase comes from the upstream query writer together with the queries.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

TOPIC = {
    "task:ambient-rna-correction": "ambient RNA contamination removal for droplet single-cell RNA-seq",
    "task:3x2pt-inference": "3x2pt cosmic shear and galaxy clustering cosmological parameter inference",
    "task:nanoindentation-property-extraction": "nanoindentation load-displacement curve analysis Oliver-Pharr",
    "task:tumor-immune-interface": "tumor-immune spatial interface analysis in imaging mass cytometry",
    "task:qsm-reconstruction": "quantitative susceptibility mapping QSM reconstruction from multi-echo GRE MRI",
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--methodlit", required=True)
    a = ap.parse_args()
    sample = json.loads(Path(a.sample).read_text())["targets"]
    ml = {t["id"]: t for t in json.loads(Path(a.methodlit).read_text())}
    out = []
    for s in sample:
        m = ml.get(s["id"], {})
        libs = m.get("libs") or []
        if isinstance(libs, str):
            libs = ast.literal_eval(libs)
        if s["kind"] == "task":
            name = s["id"].split(":", 1)[1].replace("-", " ")
            desc = f"Methods and software for {TOPIC[s['id']]}: " + "; ".join(s["queries"][:3]) + "."
        else:
            name = m.get("name") or s["id"].split(":", 1)[1].replace("-", " ")
            desc = name + ". " + (m.get("what") or s.get("what") or "")
        out.append({"id": s["id"], "queries": s["queries"], "name": name, "topic": TOPIC.get(s["id"], name),
                    "field": s.get("field") or "", "description": desc[:900], "libs": libs})
    (HERE / "targets10.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(len(out), "targets ->", HERE / "targets10.json")


if __name__ == "__main__":
    main()
