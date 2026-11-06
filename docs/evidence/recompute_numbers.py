"""Read-only recomputation of the old-pipeline numbers cited in ../DESIGN.md.

No LLM calls, no writes outside stdout. Run:  python3 recompute_numbers.py
"""
import collections
import glob
import hashlib
import json
import os
import sqlite3
import statistics

WS = os.environ.get("TBS_WORKSPACE", "/Volumes/Storage/hermes/workspace").rstrip("/") + "/"
RUN = WS + "tbs-datapipe-20261015/runs/st-corpus410-20261027/"
JOBS = WS + "tbs-aliyun/jobs/"


def calls_summary():
    con = sqlite3.connect(f"file:{RUN}guard/calls.sqlite?mode=ro", uri=True)
    n = inp = out = think = 0
    stops = collections.Counter()
    for (usage, stop) in con.execute("select usage, stop from calls"):
        n += 1
        stops[stop] += 1
        u = json.loads(usage) if usage else None
        if not u:
            continue
        inp += u.get("input_tokens", 0) or 0
        out += u.get("output_tokens", 0) or 0
        think += (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0
    return dict(calls=n, input=inp, output=out, thinking=think, stops=dict(stops))


def events_summary():
    ev = [json.loads(l) for l in open(RUN + "events.jsonl")]
    rec = {}
    for p in glob.glob(RUN + "tasks/*/_record.json"):
        r = json.load(open(p))
        rec[r["task_id"]] = r
    acc = {t for t, r in rec.items() if r.get("validated") and r.get("oracle_reward") == 1}
    by_ev = collections.defaultdict(lambda: [0, 0, 0])
    tout = collections.Counter()
    for e in ev:
        u = (e.get("llm") or {}).get("usage")
        if u:
            s = by_ev[e["ev"]]
            s[0] += 1
            s[1] += u.get("input_tokens", 0)
            s[2] += u.get("output_tokens", 0)
            tout[e["task"]] += u.get("output_tokens", 0)
    first = {}
    for e in ev:
        if e["ev"] == "S3_validate" and e["task"] not in first:
            first[e["task"]] = e
    gate_fail = collections.Counter(k for e in first.values() for k, v in e["gates"].items() if not v)
    # repair output by trigger
    last, rep = {}, collections.Counter()
    for e in ev:
        if e["ev"] in ("S2b_static", "S2c_leak", "S3_validate"):
            last[e["task"]] = e
        if e["ev"] == "S2_repair":
            k = {"S2b_static": "decontam", "S2c_leak": "leak"}.get(last[e["task"]]["ev"], "validate")
            rep[k] += e["llm"]["usage"]["output_tokens"]
    return dict(
        papers=len(rec), accepted=len(acc),
        by_event={k: dict(calls=v[0], input=v[1], output=v[2]) for k, v in by_ev.items()},
        out_on_accepted=sum(tout[t] for t in acc),
        out_on_rejected=sum(v for t, v in tout.items() if t not in acc),
        first_validate=len(first), first_validate_failed=sum(1 for e in first.values() if not e["ok"]),
        first_gate_failures=dict(gate_fail), repair_output_by_trigger=dict(rep),
        repairs_used_accepted=dict(collections.Counter(rec[t].get("repairs_used") for t in acc)),
    ), rec, tout


def glm_new244():
    trials = collections.defaultdict(list)
    for f in glob.glob(JOBS + "glm53-n244all-cc280/*/result.json"):
        d = json.load(open(f))
        if "trial_name" not in d:
            continue
        r = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        ar = d.get("agent_result") or {}
        trials[d["task_name"].split("/")[-1]].append((d.get("started_at") or "", r, ar.get("n_output_tokens"), ar.get("n_input_tokens")))
    flat = [x for v in trials.values() for x in v]
    passes = sum(1 for x in flat if (x[1] or 0) >= 1)
    dist = collections.Counter(sum(1 for x in v if (x[1] or 0) >= 1) for v in trials.values())
    seq = [tuple(int((x[1] or 0) >= 1) for x in sorted(v)) for v in trials.values()]
    seq_runs = sum(1 if s[0] == 0 else 2 for s in seq) / len(seq)
    first_two_pass = [s for s in seq if s[:2] == (1, 1)]
    return dict(
        trials=len(flat), passes=passes, pass_rate=round(passes / len(flat), 4), per_task=dict(dist),
        median_out=statistics.median([x[2] for x in flat if x[2]]),
        median_in=statistics.median([x[3] for x in flat if x[3]]),
        sequential_runs_per_task=round(seq_runs, 3),
        first_two_pass=len(first_two_pass), first_two_pass_then_3of3=sum(1 for s in first_two_pass if s[2] == 1),
    ), trials


def teacher_on_glm0(trials):
    zero = {t for t, v in trials.items() if not any((x[1] or 0) >= 1 for x in v)}
    op = collections.defaultdict(list)
    for f in glob.glob(JOBS + "opus-teacher-glm-missing46-cc280/*/result.json"):
        d = json.load(open(f))
        if "trial_name" in d:
            op[d["task_name"].split("/")[-1]].append(((d.get("verifier_result") or {}).get("rewards") or {}).get("reward"))
    return dict(glm0=len(zero), teacher_attempted=sum(1 for t in zero if t in op),
                teacher_pass=sum(1 for t in zero if any((x or 0) >= 1 for x in op.get(t, []))))


def official65():
    res = {}
    for f in ["glm53-official65/state/slots-live.json", "glm53-official65-dsh/state/slots-live.json"]:
        for v in json.load(open(WS + "tbs-aliyun/" + f)).values():
            tr = (v.get("trials") or [{}])[-1]
            res[(v["harness"], v["task"])] = (tr.get("reward"), tr.get("agent_minutes"))
    out = {}
    for h in sorted({h for h, _ in res}):
        rs = [v for (hh, _), v in res.items() if hh == h]
        out[h] = dict(slots=len(rs), with_reward=sum(1 for r in rs if r[0] is not None),
                      passed=sum(1 for r in rs if (r[0] or 0) >= 1),
                      median_minutes=statistics.median([r[1] for r in rs if r[1]]))
    union = sorted({t for (h, t), v in res.items() if (v[0] or 0) >= 1})
    out["tasks_passed_by_any_harness"] = union
    return out


def ab20(rec, trials, tout):
    strata = collections.defaultdict(list)
    for t, r in rec.items():
        if t in trials:
            k = sum(1 for x in trials[t] if (x[1] or 0) >= 1)
            s = "easy3" if k == 3 else ("unstable" if k else "glm0")
        elif r.get("validated") and r.get("oracle_reward") == 1:
            s = "review_drop"
        else:
            s = "old_fail"
        strata[s].append(t)
    quota = {"easy3": 8, "unstable": 5, "glm0": 2, "old_fail": 4, "review_drop": 1}
    sel = []
    for s, n in quota.items():
        L = sorted(strata[s], key=lambda t: hashlib.sha256(("tbs-v2-ab20|" + t).encode()).hexdigest())
        sel += [(s, t, rec[t]["doi"], tout.get(t, 0)) for t in L[:n]]
    return {k: len(v) for k, v in strata.items()}, sel


if __name__ == "__main__":
    print(json.dumps(calls_summary(), indent=1))
    ev, rec, tout = events_summary()
    print(json.dumps(ev, indent=1))
    g, trials = glm_new244()
    print(json.dumps(g, indent=1))
    print(json.dumps(teacher_on_glm0(trials), indent=1))
    print(json.dumps(official65(), indent=1))
    strata, sel = ab20(rec, trials, tout)
    print(json.dumps(strata))
    for row in sel:
        print(row)
