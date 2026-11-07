"""Reference labels for the S2 set from the teacher model (Opus), following RUBRIC.md.

Uses tbsgen's own OpusClient + budget ledger.  The ledger lives next to the output file and enforces a whole-run
USD cap (`--usd-cap`, priced at models.prices placeholders); the script stops when a reservation would exceed it.
Papers are sent in batches (one call labels `--batch` papers) to keep the cost down; resumable (labels already in
the output file are skipped).

    TBS_OPUS_URL=... uv run python eval/screen_jev/label_opus.py --set <set.jsonl> --out <labels.jsonl>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tbsgen.budget import BudgetStop, Ledger
from tbsgen.config import load_config
from tbsgen.models import ModelError, OpusClient, extract_json

HERE = Path(__file__).resolve().parent


def paper_view(r: dict) -> dict:
    v = {"id": r["key"], "title": r.get("title")}
    if r.get("abstract"):
        v["abstract"] = r["abstract"]
    elif r.get("fulltext_head"):
        v["full_text_opening"] = r["fulltext_head"]
    for k in ("venue", "year", "code_links", "data_links"):
        if r.get(k):
            v[k] = r[k]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--usd-cap", type=float, default=5.0)
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.set).read_text().splitlines() if x.strip()]
    out = Path(a.out)
    done = {json.loads(x)["id"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    todo = [r for r in rows if r["key"] not in done]
    cfg = load_config(overrides={"budget": {"daily_usd_total": a.usd_cap, "run_usd_total": a.usd_cap,
                                            "est_usd_per_call": 0.4, "max_calls_per_item": None},
                                 "models": {"opus": {"effort": a.effort, "max_tokens": 16000}}})
    led = Ledger(out.with_suffix(".ledger.sqlite"), cfg)
    cli = OpusClient(led, cfg)
    rubric = (HERE / "RUBRIC.md").read_text()
    system = ("You label scientific papers for a dataset-construction pipeline. Follow the rubric exactly. Judge "
              "only from the record given. Answer with one JSON array and nothing else.\n\n" + rubric)
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        user = ("Label each paper. Return a JSON array with one object per paper, in the same order, with keys "
                "id, label, confidence, interactive, why.\n\n" + json.dumps([paper_view(r) for r in chunk], indent=1))
        try:
            res = cli.chat(system, user, stage="label", item="")
        except BudgetStop as e:
            print("STOP budget:", e.reason)
            break
        except ModelError as e:
            print("model error:", e.cls, str(e)[:200])
            continue
        try:
            arr = extract_json(res.text)
        except ValueError:
            print("unparseable batch", i)
            continue
        ids = {r["key"] for r in chunk}
        with out.open("a") as f:
            for o in arr:
                if o.get("id") in ids and o.get("label") in ("keep", "drop"):
                    f.write(json.dumps(dict(o, model=res.model)) + "\n")
        print(f"batch {i // a.batch}: {len(arr)} labels, usd {res.usd:.4f}, total {led.summary()['usd_total']}")
    print(json.dumps(led.summary(), indent=1))


if __name__ == "__main__":
    main()
