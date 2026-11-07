"""Run the three S2 screens over the reference set and report against the reference labels.

    # 1. call Jev once per paper (+ repeats for stability); answers are cached in --jev-cache, so re-running
    #    analysis never calls the API again.  Needs TYPESAFE_API_KEY in the environment.
    uv run python eval/screen_jev/run_eval.py call --set set.jsonl --jev-cache jev.jsonl --repeats 3 --repeat-n 20
    # 2. metrics: rules-v1, rules-v2, jev on dev and test halves
    uv run python eval/screen_jev/run_eval.py report --set set.jsonl --labels labels.jsonl --jev-cache jev.jsonl
    # 3. downstream check: papers with a known pipeline outcome (no model labels): how many each screen keeps
    uv run python eval/screen_jev/run_eval.py corpus --set corpus.jsonl --jev-cache jev_corpus.jsonl

Jev calls go through tbsgen's JevClient and a budget ledger next to the cache with a hard USD cap (`--usd-cap`).
The reference labels are model labels (see RUBRIC.md).  Results files contain DOIs and titles of public papers and
internal sampling details: keep them outside the repository.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tbsgen import intake, screen_jev
from tbsgen.budget import BudgetStop, Ledger
from tbsgen.config import load_config
from tbsgen.models import ModelError


def load(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def cand(r: dict) -> dict:
    return {k: r.get(k) for k in ("title", "abstract", "venue", "year", "code_links", "data_links")}


def cfg_for(usd_cap: float, th: dict | None = None, uncertain: str = "keep") -> dict:
    return load_config(overrides={"budget": {"daily_usd_total": usd_cap, "run_usd_total": usd_cap,
                                             "max_calls_per_item": None},
                                  "screen": {"jev": {"proxy": "none", "uncertain": uncertain,
                                                     "thresholds": th or {}}}})


# ------------------------------------------------------------------------------------------- call
def cmd_call(a):
    rows = load(a.set)
    cache = Path(a.jev_cache)
    have = {}
    if cache.exists():
        for x in load(cache):
            have.setdefault(x["key"], []).append(x)
    cfg = cfg_for(a.usd_cap)
    led = Ledger(cache.with_suffix(".ledger.sqlite"), cfg)
    cli = screen_jev.JevClient(led, cfg)
    jobs = []
    rep_keys = {r["key"] for r in rows[: a.repeat_n]} if a.repeats > 1 else set()
    for r in rows:
        want = a.repeats if r["key"] in rep_keys else 1
        for i in range(len(have.get(r["key"], [])), want):
            jobs.append((r, i))

    def one(job):
        r, i = job
        head = None if r.get("abstract") else r.get("fulltext_head")
        st = screen_jev.build_state(cand(r), head)
        try:
            res = cli.ask(st, stage="S2" if not head else "S4", item=r["key"])
        except (ModelError, BudgetStop) as e:
            return dict(key=r["key"], rep=i, error=str(e)[:200])
        return dict(key=r["key"], rep=i, model=res["model"], answers=res["answers"], usage=res["usage"],
                    usd=res["usd"], latency_s=res["latency_s"], attempts=res["attempts"], ts=time.time())
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex, cache.open("a") as f:
        for out in ex.map(one, jobs):
            if "error" in out:
                print("error", out["key"], out["error"])
                continue
            f.write(json.dumps(out) + "\n")
            f.flush()
    s = led.summary()
    print(json.dumps(dict(calls=s["calls_total"], usd=s["usd_total"], breakers=s["breakers_open"])))


# ------------------------------------------------------------------------------------------- report
def confusion(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    tn = sum((not p) and (not g) for p, g in zip(pred, gold))
    n = len(gold)
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn, recall=round(tp / (tp + fn), 3) if tp + fn else None,
                precision=round(tp / (tp + fp), 3) if tp + fp else None, keep_rate=round((tp + fp) / n, 3) if n else None,
                accuracy=round((tp + tn) / n, 3) if n else None)


def decide_rules(r: dict, ver: str, min_score: int | None = None) -> dict:
    cfg = load_config(overrides={"screen": {"rules_version": ver, **({"min_score": min_score} if min_score else {})}})
    rec = intake.screen(cand(r), cfg)
    if rec.get("deferred"):           # S4 re-screen on the full-text opening (as the pipeline does)
        rec = intake.screen(cand(r), cfg, fulltext_head=r.get("fulltext_head") or "")
    return rec


def decide_jev(ans: dict, th: dict | None = None, uncertain: str = "keep") -> dict:
    cfg = cfg_for(0, th, uncertain)
    return screen_jev.record_from_answer(dict(answers=ans["answers"], model=ans["model"], usage=ans["usage"]), cfg,
                                         False)


def cmd_report(a):
    rows = load(a.set)
    lab = {x["id"]: x for x in load(a.labels)}
    jev = {}
    for x in load(a.jev_cache):
        jev.setdefault(x["key"], []).append(x)
    th = json.loads(a.thresholds) if a.thresholds else None
    out = dict(n_set=len(rows), n_labelled=sum(r["key"] in lab for r in rows), splits={}, latency={}, cost={},
               stability={}, per_paper=[])
    per = []
    for r in rows:
        if r["key"] not in lab or r["key"] not in jev:
            continue
        g = lab[r["key"]]
        v1, v2 = decide_rules(r, "v1"), decide_rules(r, "v2")
        j = decide_jev(jev[r["key"]][0], th, a.uncertain)
        per.append(dict(key=r["key"], title=r.get("title"), split=r["split"], stratum=r["stratum"],
                        gold=g["label"] == "keep", gold_conf=g.get("confidence"), gold_why=g.get("why"),
                        interactive_ref=g.get("interactive"), v1=v1["keep"], v1_score=v1["score"],
                        v2=v2["keep"], v2_score=v2["score"], jev=j["keep"], jev_decision=j["decision"],
                        jev_reason=j["reason"] or j.get("uncertain_reason"), jev_p=j["probabilities"],
                        jev_form=j["form_hint"], v1_form=v1["form_hint"], no_abstract=not r.get("abstract")))
    for split in ("dev", "test", "all"):
        sub = [p for p in per if split == "all" or p["split"] == split]
        gold = [p["gold"] for p in sub]
        out["splits"][split] = {m: confusion([p[m] for p in sub], gold) for m in ("v1", "v2", "jev")}
        out["splits"][split]["jev_decisions"] = {d: sum(p["jev_decision"] == d for p in sub)
                                                 for d in ("keep", "uncertain", "drop")}
        hi = [p for p in sub if p["gold_conf"] == "high"]
        out["splits"][split]["high_conf_only"] = {m: confusion([p[m] for p in hi], [p["gold"] for p in hi])
                                                  for m in ("v1", "v2", "jev")}
    lats = [x["latency_s"] for xs in jev.values() for x in xs]
    toks = [x["usage"]["input_tokens"] for xs in jev.values() for x in xs]
    q = statistics.quantiles(lats, n=20) if len(lats) >= 20 else [None] * 19
    out["latency"] = dict(n=len(lats), p50=round(statistics.median(lats), 3), p95=round(q[18], 3) if q[18] else None,
                          max=round(max(lats), 3))
    out["cost"] = dict(calls=len(toks), input_tokens_mean=round(statistics.mean(toks)),
                       input_tokens_p95=sorted(toks)[int(0.95 * len(toks)) - 1],
                       usd_per_paper=round(statistics.mean(toks) * 0.042 / 1e6, 7),
                       usd_total=round(sum(x.get("usd") or 0 for xs in jev.values() for x in xs), 6))
    flips, maxdev = 0, []
    reps = {k: xs for k, xs in jev.items() if len(xs) > 1}
    for k, xs in reps.items():
        ds = [decide_jev(x, th, a.uncertain)["decision"] for x in xs]
        flips += len(set(ds)) > 1
        ps = [screen_jev.probabilities(x["answers"]) for x in xs]
        for q_ in screen_jev.QUESTIONS:
            vals = [p.get(q_ if q_ != "paper_type" else "paper_type_p") for p in ps]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if len(vals) > 1:
                maxdev.append(max(vals) - min(vals))
    out["stability"] = dict(papers=len(reps), repeats=max((len(v) for v in reps.values()), default=0),
                            decision_changed=flips, prob_range_max=round(max(maxdev), 4) if maxdev else None,
                            prob_range_mean=round(statistics.mean(maxdev), 4) if maxdev else None)
    out["per_paper"] = per
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_paper"}, indent=1))


def cmd_corpus(a):
    rows = load(a.set)
    jev = {x["key"]: x for x in load(a.jev_cache)}
    th = json.loads(a.thresholds) if a.thresholds else None
    out: dict = {}
    for r in rows:
        if r["key"] not in jev:
            continue
        j = decide_jev(jev[r["key"]], th)
        o = out.setdefault(r["outcome"], dict(n=0, v1=0, v2=0, jev=0, jev_no_joint=0))
        o["n"] += 1
        o["v1"] += decide_rules(r, "v1")["keep"]
        o["v2"] += decide_rules(r, "v2")["keep"]
        o["jev"] += j["keep"]
        o["jev_no_joint"] += decide_jev(jev[r["key"]], dict(th or {}, joint_no=0.0))["keep"]
    lats = [x["latency_s"] for x in jev.values()]
    out["_latency"] = dict(n=len(lats), p50=round(statistics.median(lats), 3),
                           p95=round(statistics.quantiles(lats, n=20)[18], 3))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("call")
    c.add_argument("--set", required=True)
    c.add_argument("--jev-cache", required=True)
    c.add_argument("--repeats", type=int, default=1)
    c.add_argument("--repeat-n", type=int, default=20)
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--usd-cap", type=float, default=1.0)
    r = sp.add_parser("report")
    r.add_argument("--set", required=True)
    r.add_argument("--labels", required=True)
    r.add_argument("--jev-cache", required=True)
    r.add_argument("--thresholds", default=None, help="JSON overrides of screen_jev.THRESHOLDS")
    r.add_argument("--uncertain", default="keep")
    r.add_argument("--out", required=True)
    k = sp.add_parser("corpus")
    k.add_argument("--set", required=True)
    k.add_argument("--jev-cache", required=True)
    k.add_argument("--thresholds", default=None)
    a = ap.parse_args()
    {"call": cmd_call, "report": cmd_report, "corpus": cmd_corpus}[a.cmd](a)


if __name__ == "__main__":
    main()
