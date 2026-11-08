"""Build the S2 reference set (<= 200 papers) by stratified sampling from three existing sources.

Sources (paths come from the environment; nothing machine-specific is hard-coded):
  * $TBS_CORPUS_RUN - a run directory of the earlier pipeline (manifest.json + tasks/*/_record.json): full-text
    papers and whether each became an accepted task (validated + oracle reward 1).  Outcome = downstream success
    on an already screened and selected corpus, i.e. mostly positives.  `--corpus-out` also writes the whole
    corpus (for `run_eval.py corpus`).
  * retrieval/eval/raw/*.json or $TBS_RETR_RAW (+ $TBS_EVAL_BASELINE/review and retrieval/eval/labels_new.json) - every paper the
    retrieval eval saw for 10 targets, with human topic-relevance labels (0/1/2) for the judged top-20.
    Relevance to a target is NOT suitability for a computational task.
  * --old-pipe DIR: <DIR>/literature/methodlit/cands/*.jsonl - candidates scored by the earlier pipeline's LLM abstract
    screen (exclude flag, 0-10 sub-scores).  Model labels, not human.
  * $TBS_E2E_ITEMS (optional) - the e2e-iter1 items (S1_candidate.json + fulltext.txt).

Papers without an abstract go to a separate "deferred" stratum, screened on the first 4000 characters of the full
text (the S4 re-screen path); only papers whose full text is available locally are eligible.

    uv run python eval/screen_jev/build_set.py --out <dir outside the repo>/set.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEED = 20261107
N = dict(corpus_ok=36, corpus_fail=12, retr_rel2=28, retr_rel1=26, retr_rel0=18, retr_unl=10,
         cands_excl=26, cands_low=16, cands_high=12, deferred=10)


def nd(d):
    return (d or "").lower().strip()


def split_of(key: str) -> str:
    return "dev" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def lst(v):
    return v if isinstance(v, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-out", default=None)
    ap.add_argument("--old-pipe", required=True, help="earlier pipeline's data directory (not in this repo)")
    a = ap.parse_args()
    old = Path(a.old_pipe)
    base = Path(os.environ["TBS_EVAL_BASELINE"])
    rng = random.Random(SEED)
    rows, seen = [], set()

    def add(r, stratum, n_needed=None):
        k = nd(r.get("doi")) or nd(r.get("title"))
        if not k or k in seen:
            return False
        seen.add(k)
        r = dict(r, key=k, stratum=stratum, split=split_of(k))
        rows.append(r)
        return True

    # e2e-iter1 items (always included)
    e2e = os.environ.get("TBS_E2E_ITEMS")
    if e2e:
        for d in sorted(Path(e2e).glob("*/")):
            c = json.loads((d / "S1_candidate.json").read_text())
            ft = (d / "fulltext.txt")
            r = dict(doi=c.get("doi"), title=c.get("title"), abstract=c.get("abstract") or None, venue=c.get("venue"),
                     year=c.get("year"), code_links=lst(c.get("code_links")), data_links=lst(c.get("data_links")),
                     src="e2e", item_id=d.name)
            if not r["abstract"]:
                r["fulltext_head"] = ft.read_text()[:4000] if ft.exists() else None
            add(r, "e2e_deferred" if not r["abstract"] else "e2e")

    # earlier-pipeline corpus
    run = Path(os.environ["TBS_CORPUS_RUN"])
    m = json.loads((run / "manifest.json").read_text())
    recs = {}
    for f in glob.glob(str(run / "tasks/*/_record.json")):
        r = json.loads(Path(f).read_text())
        recs[nd(r["doi"])] = r
    corpus = []
    for e in m["entries"]:
        r = recs.get(nd(e["doi"]))
        ok = e["state"] == "reused" or bool(r and r.get("validated") and r.get("oracle_reward") == 1)
        row = e["row"]
        corpus.append(dict(doi=row["doi"], title=row["title"], abstract=row.get("abstract") or None,
                           venue=row.get("venue"), year=row.get("year"), code_links=lst(row.get("code_links")),
                           src="corpus", outcome="task_ok" if ok else ("blocked" if e["state"] == "blocked"
                                                                          else "task_failed"), text=e["text"]))
    if a.corpus_out:
        with open(a.corpus_out, "w") as f:
            for c in corpus:
                r = {x: y for x, y in c.items() if x != "text"}
                r.update(key=nd(c["doi"]), split="corpus", stratum="corpus")
                if not r["abstract"] and Path(c["text"]).exists():
                    r["fulltext_head"] = Path(c["text"]).read_text(errors="replace")[:4000]
                f.write(json.dumps(r) + "\n")
    rng.shuffle(corpus)
    for stratum, pred, n in (("corpus_ok", lambda c: c["outcome"] == "task_ok", N["corpus_ok"]),
                             ("corpus_fail", lambda c: c["outcome"] != "task_ok", N["corpus_fail"])):
        k = 0
        for c in corpus:
            if k < n and c["abstract"] and pred(c) and add({x: y for x, y in c.items() if x != "text"}, stratum):
                k += 1
    k = 0
    for c in corpus:            # deferred stratum: no abstract, full text available
        if k < N["deferred"] and not c["abstract"] and c["outcome"] != "blocked" and Path(c["text"]).exists():
            r = {x: y for x, y in c.items() if x != "text"}
            r["fulltext_head"] = Path(c["text"]).read_text(errors="replace")[:4000]
            if add(r, "deferred"):
                k += 1

    # retrieval eval
    lab = collections.defaultdict(dict)
    oldlab = json.loads((base / "review/labels.json").read_text())
    for f in (base / "review").glob("top10_*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            v = oldlab.get(f"{r['api']}|{r['target']}|{r['rank']}")
            if v is not None:
                kk = r["key"]
                lab[r["target"]]["arxiv:" + kk[3:] if kk.startswith("ax:") else kk] = v
    for t, d in json.loads((REPO / "retrieval/eval/labels_new.json").read_text()).items():
        if not t.startswith("_"):
            lab[t].update(d)
    retr = []
    raw = Path(os.environ.get("TBS_RETR_RAW") or REPO / "retrieval/eval/raw")      # git-ignored recorded runs
    for f in sorted(raw.glob("*.json")):
        d = json.loads(f.read_text())
        tid = d["target"]["id"]
        for p in d["enriched"]:
            kk = "doi:" + nd(p.get("doi")) if p.get("doi") else ("arxiv:" + p["arxiv"] if p.get("arxiv") else None)
            retr.append(dict(doi=p.get("doi"), arxiv=p.get("arxiv"), title=p.get("title"),
                             abstract=p.get("abstract") or None, venue=p.get("venue"), year=p.get("year"),
                             code_links=lst(p.get("code_links")), data_links=lst(p.get("data_links")),
                             src="retrieval_eval", target=tid, relevance=lab[tid].get(kk) if kk else None))
    rng.shuffle(retr)
    for stratum, rel in (("retr_rel2", 2), ("retr_rel1", 1), ("retr_rel0", 0), ("retr_unl", None)):
        k = 0
        for r in retr:
            if k < N[stratum] and r["abstract"] and r["relevance"] == rel and add(r, stratum):
                k += 1

    # methodlit candidates (earlier LLM screen)
    cands = []
    for f in sorted((old / "literature/methodlit/cands").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            c = json.loads(line)
            s = c.get("screen") if isinstance(c.get("screen"), dict) else {}
            cands.append(dict(doi=c.get("doi"), title=c.get("title"), abstract=c.get("abstract") or None,
                              venue=c.get("venue"), year=c.get("year"), src="methodlit_cands",
                              old_exclude=s.get("exclude"), old_total=s.get("total")))
    rng.shuffle(cands)
    for stratum, pred in (("cands_excl", lambda c: c["old_exclude"] is True),
                          ("cands_low", lambda c: c["old_exclude"] is False and (c["old_total"] or 0) < 25),
                          ("cands_high", lambda c: c["old_exclude"] is False and (c["old_total"] or 0) >= 35)):
        k = 0
        for c in cands:
            if k < N[stratum] and c["abstract"] and pred(c) and add(c, stratum):
                k += 1

    assert len(rows) <= 200, len(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), dict(collections.Counter(r["stratum"] for r in rows)),
          dict(collections.Counter(r["split"] for r in rows)))


if __name__ == "__main__":
    main()
