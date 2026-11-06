"""Re-run only the arXiv lists of eval/raw/*.json with the fixed query form and patch them in place
(the first eval run used a strict AND query that returned 0 hits for most targets). Everything else in the raw
records is untouched; analyze.py re-fuses from the lists.
   uv run python eval/rerun_arxiv.py
"""
import json
from dataclasses import asdict
from pathlib import Path

from tbsretr.config import load_config
from tbsretr.model import Target
from tbsretr.pipeline import Retriever

HERE = Path(__file__).resolve().parent


def main():
    cfg = load_config()
    cfg["sources"]["arxiv"]["daily_max_calls"] = 60   # eval cap for arXiv
    rt = Retriever(cfg, run="eval10-arxivfix", sources={"arxiv"})
    for f in sorted((HERE / "raw").glob("*.json")):
        rec = json.loads(f.read_text())
        t = Target.from_dict(rec["target"])
        new = rt._stage1_source("arxiv", t)
        rec["stage1"] = [l for l in rec["stage1"] if l["name"] != "arxiv.search"] + [
            {"name": rl.name, "query": rl.query, "papers": [asdict(p) for p in rl.papers],
             "resources": [asdict(r) for r in rl.resources]} for rl in new]
        rec["arxiv_rerun"] = True
        f.write_text(json.dumps(rec, ensure_ascii=False))
        print(t.id[:50], [len(rl.papers) for rl in new], rt.src["arxiv"].errors[-1:] if rt.src["arxiv"].errors else "")


if __name__ == "__main__":
    main()
