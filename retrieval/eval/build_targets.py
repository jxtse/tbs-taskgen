"""Build eval/targets10.json from search_api/targets_sample.json (same 10 targets, same 4 queries as the old
OpenAlex/Crossref/arXiv benchmark) + read-only metadata from tbs-datapipe methodlit/targets.json (libs, method text).

`topic` (short English phrase for Exa/Tavily resource channels) is hand-written here for the 5 task targets because
task ids are terse; in production this phrase comes from the upstream query writer together with the queries.
"""
import ast
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE.parents[2] / "search_api/targets_sample.json"
ML = Path(os.environ.get("TBS_OLD_PIPE", "/Volumes/Storage/hermes/workspace/tbs-datapipe-20261015")) / "literature/methodlit/targets.json"

TOPIC = {
    "task:ambient-rna-correction": "ambient RNA contamination removal for droplet single-cell RNA-seq",
    "task:3x2pt-inference": "3x2pt cosmic shear and galaxy clustering cosmological parameter inference",
    "task:nanoindentation-property-extraction": "nanoindentation load-displacement curve analysis Oliver-Pharr",
    "task:tumor-immune-interface": "tumor-immune spatial interface analysis in imaging mass cytometry",
    "task:qsm-reconstruction": "quantitative susceptibility mapping QSM reconstruction from multi-echo GRE MRI",
}


def main():
    sample = json.loads(SAMPLE.read_text())["targets"]
    ml = {t["id"]: t for t in json.loads(ML.read_text())}
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
