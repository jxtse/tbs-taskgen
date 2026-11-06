"""Record one real response per source endpoint into tests/fixtures (keys never written: auth is in headers,
OpenAlex api_key stripped; every saved body passes util.redact). Goes through the ledger like any other call.
   uv run python scripts/record_fixtures.py
"""
import json, sys
from pathlib import Path
from tbsretr.config import load_config
from tbsretr.pipeline import Retriever
from tbsretr.model import Target
from tbsretr.util import redact, strip_secret_params

FX = Path(__file__).resolve().parents[1] / "tests/fixtures"
FX.mkdir(parents=True, exist_ok=True)
rec = []

def recording(inner):
    def tr(method, url, headers, body):
        st, hd, b = inner(method, url, headers, body)
        rec.append({"method": method, "url": strip_secret_params(url), "body": body, "status": st,
                    "headers": {k: v for k, v in hd.items() if k.lower().startswith("x-ratelimit") or k.lower() == "content-type"},
                    "response": redact(b.decode("utf8", "replace"))})
        return st, hd, b
    return tr

cfg = load_config(overrides={"cache_ttl_days": 0})   # force live
rt = Retriever(cfg, run="fixtures")
rt.http.transport = recording(rt.http._urllib)
t = Target(id="task:ambient-rna-correction", queries=["ambient RNA removal droplet single-cell RNA-seq"],
           description="Remove ambient RNA (soup) contamination from droplet scRNA-seq count matrices using empty droplets.",
           name="ambient RNA correction", libs=["SoupX"])
S = rt.src
which = sys.argv[1:] or ["openalex", "s2", "exa", "tavily", "crossref", "arxiv", "europepmc"]
if "openalex" in which: S["openalex"].search(t.queries[0], t)
if "s2" in which:
    l = S["s2"].search(t.queries[0], t); S["s2"].recommend(l.papers[:2], t); S["s2"].recommend(l.papers[:1], t)
if "exa" in which: S["exa"].channel("paper", t)
if "tavily" in which: S["tavily"].channel("code", t)
if "crossref" in which:
    S["crossref"].search("SoupX ambient RNA", t); S["crossref"].by_dois(["10.1093/gigascience/giaa151"], t.id)
if "arxiv" in which: S["arxiv"].search("neural quantum states stochastic reconfiguration", t, ["quant-ph", "cond-mat"])
if "europepmc" in which: S["europepmc"].search(t.queries[0], t)
for r in rec:
    name = r["url"].split("//")[1].split("/")[0].split(".")[-2] + "_" + str(len(list(FX.glob("*.json")))) \
        + ("_" + str(r["status"]) if r["status"] != 200 else "")
    for k in ("OPENALEX_API_KEY","SEMANTIC_SCHOLAR_API_KEY","EXA_API_KEY_SECONDARY","TAVILY_API_KEY_SECONDARY"):
        import os; v = os.environ.get(k)
        assert not v or v not in json.dumps(r), "secret leaked into fixture"
    (FX / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(name, r["status"], len(r["response"]))
