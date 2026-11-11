"""Stage handlers S1-S6B.  Each handler gets (P, item) and returns via item.advance / item.goto / item.finish.
P is the Pipeline (cfg, store, ledger, clients).  Handlers write outputs before committing state."""
from __future__ import annotations

import hashlib
import time
import json
import shutil
import re
from pathlib import Path

from . import blocklist, codegate, intake, runner, screen_jev
from .budget import BudgetStop
from .config import resolve_role
from .models import ModelError, extract_json, parse_files  # noqa: F401
from .prompts import METRICS_API, PROMPT_VERSION, S5_SPEC, S6A_GEN, S6A_REWRITE, S6M_MID, SYSTEM
from .spec import crashed_runs, s6b_decision, set_thresholds, validate_spec
from .state import read_json, write_json


def s1(P, it):
    c = read_json(it.p("S1_candidate.json"))
    if not c:
        return it.finish("dropped", "no_candidate")
    it.advance("S1", codegate.after_s1(P.cfg))


def s2a(P, it):
    """Code-grounded pre-screen (before any full-text download): any code signal, else drop no_code_signal."""
    c = read_json(it.p("S1_candidate.json")) or {}
    sig = codegate.s2a_signals(c, P.cfg)
    write_json(it.p("S2a_code_signal.json"), dict(sig, ts=time.time()))
    if not sig["any"]:
        return it.finish("dropped", "no_code_signal")
    it.advance("S2a", "S2", signals=[k for k in ("meta", "retrieval", "pwc", "openalex_fulltext") if sig.get(k)])


def s3b(P, it):
    """Code-grounded confirmation on the full text: the authors' own code exists and resolves on its host."""
    c = read_json(it.p("S1_candidate.json")) or {}
    txt = it.p("fulltext.txt").read_text(errors="replace")
    jev = None
    if resolve_role(P.cfg, "code_owner") == "jev":
        client = P.jev_client()

        def jev(link):
            try:
                return codegate.jev_own(client, it, c, link)
            except (ModelError, BudgetStop) as e:
                P.store.event("code_owner_jev_unavailable", item=it.item_id, err=str(e)[:200])
                return None
    rec = codegate.s3b_record(txt, c, P.cfg, jev=jev)
    write_json(it.p("S3b_code.json"), rec)
    if not rec["keep"]:
        return it.finish("dropped", rec["reason"])
    p = rec["primary"]
    it.advance("S3b", "S4", repo=p.get("url"),
               license=p.get("license"), language=p.get("language"))


def screen_paper(P, it, c: dict, stage: str, fulltext_head: str | None = None) -> dict | None:
    """S2 decision for one paper with the configured screen (rules or Jev).  Returns the S2_screen.json record,
    or None when Jev is unavailable and screen.jev.on_unavailable="park" (the item is then paused with a
    model:jev_unavailable reason, which `tbsgen resume` reactivates)."""
    if resolve_role(P.cfg, "screen") != "jev":
        return intake.screen(c, P.cfg, fulltext_head=fulltext_head)
    if not c.get("abstract") and fulltext_head is None:
        return dict(intake.screen(c, P.cfg), method="jev")          # deferred: S4 asks Jev on the full text
    try:
        return screen_jev.cached_call(P.jev_client(), it, c, P.cfg, stage=stage, fulltext_head=fulltext_head)
    except (ModelError, BudgetStop) as e:
        why = f"{getattr(e, 'cls', None) or 'budget'}:{getattr(e, 'reason', '') or ''}".rstrip(":")
        P.store.event("screen_jev_unavailable", item=it.item_id, stage=stage, why=why, err=str(e)[:200])
        if P.cfg["screen"]["jev"].get("on_unavailable", "rules") == "park":
            it.finish("paused", f"model:jev_unavailable:{why}", paused_stage=stage)
            return None
        rec = intake.screen(c, P.cfg, fulltext_head=fulltext_head)
        return dict(rec, method=rec["method"] + "-fallback", fallback_reason=why)


def s2(P, it):
    c = read_json(it.p("S1_candidate.json")) or {}
    rules = P.cfg["screen"].get("blocklist")
    c.setdefault("item_id", it.item_id)
    rule = blocklist.match(c, rules, blocklist.lifted_rules(blocklist.load_state(P.store)))
    if rule is not None:                    # full text unobtainable: drop before spending a screen call —
        d = blocklist.decide(P.store, P.cfg, c, rules, rule, P.store.event)  # unless it is a recheck trial
        if d.get("action") != "trial":
            rec = blocklist.screen_record(c, rule)
            write_json(it.p("S2_screen.json"), rec)
            return it.finish("dropped", rec["reason"])
        it.state["blocklist_trial"] = rule.get("name") or rule.get("doi_prefix")
        it.save()
    rec = screen_paper(P, it, c, "S2")
    if rec is None:
        return
    write_json(it.p("S2_screen.json"), rec)
    if not rec["keep"]:
        if it.state.get("blocklist_trial"):
            blocklist.resolve(P.store, it, void=f"S2 screen: {rec['reason']}")   # never reached the download
        return it.finish("dropped", rec["reason"])
    it.advance("S2", "S3", score=rec["score"], screen=rec["method"], uncertain=bool(rec.get("uncertain")))


def s3(P, it):
    c = read_json(it.p("S1_candidate.json")) or {}
    txt, rec = intake.fetch_fulltext(c, P.cfg, P.fulltext_fetcher())
    write_json(it.p("S3_fulltext.json"), rec)
    if txt is None:
        from . import fulltext
        entry = fulltext.queue_entry(c, subscribers=[[str(P.store.root.resolve()), it.item_id]],
                                     failed_channels=sorted(failed_channels(rec)),
                                     queued_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        # 1. intent record, 2. park, 3. queue.  A crash between 2 and 3 is repaired by `tbsgen fetch`, which
        # first re-queues every item parked with fulltext_unavailable from its S3_queue.json (fulltext.reconcile).
        write_json(it.p("S3_queue.json"), dict(entry, route="tbsgen fetch", errors=rec.get("errors"),
                                               skipped=rec.get("skipped")))
        it.finish("parked", "fulltext_unavailable")
        if it.state.get("blocklist_trial"):
            blocklist.resolve(P.store, it)        # trial decided: no full text (stays pending if still queued)
            blocklist.reevaluate(P.store, P.cfg, P.store.event)
        if entry["ident"]:
            fulltext.enqueue(fulltext.queue_path(P.cfg), entry)
        return None
    it.p("fulltext.txt").write_text(txt)
    if it.state.get("blocklist_trial"):
        blocklist.resolve(P.store, it)            # trial decided: full text obtained
        blocklist.reevaluate(P.store, P.cfg, P.store.event)
    it.advance("S3", codegate.after_s3(P.cfg), source=rec.get("source"), chars=len(txt))


def failed_channels(rec: dict) -> set[str]:
    """Channels that made a request and got a definitive non-text answer (not skipped / capped)."""
    return {t.split(":", 1)[0] for t in rec.get("tried") or [] if not t.startswith("cache:")}


def s4(P, it):
    c = read_json(it.p("S1_candidate.json"))
    if read_json(it.p("S2_screen.json"), {}).get("deferred"):
        rec = screen_paper(P, it, c, "S4", fulltext_head=it.p("fulltext.txt").read_text()[:4000])
        if rec is None:
            return
        write_json(it.p("S2_screen.json"), dict(rec, rescreened_at="S4"))
        if not rec["keep"]:
            return it.finish("dropped", rec["reason"])
    fulltext = it.p("fulltext.txt").read_text()
    rec = intake.probe(c, fulltext, read_json(it.p("S2_screen.json"), {}), P.cfg, P.tbs_index())
    if rec["feasible"] and resolve_role(P.cfg, "probe") == "jev":
        try:
            j = screen_jev.s4_feasibility(P.jev_client(), it, c, fulltext, P.cfg)
        except (ModelError, BudgetStop) as e:
            why = f"{getattr(e, 'cls', None) or 'budget'}:{getattr(e, 'reason', '') or ''}".rstrip(":")
            P.store.event("probe_jev_unavailable", item=it.item_id, stage="S4", why=why, err=str(e)[:200])
            if (P.cfg["probe"].get("jev") or {}).get("on_unavailable", "rules") == "park":
                return it.finish("paused", f"model:jev_unavailable:{why}", paused_stage="S4")
            j = dict(feasible=True, reasons=[], score=None, unavailable=why)
        rec["jev"] = {k: v for k, v in j.items() if k != "cached"}
        rec["method"] = rec["method"] + " + jev"
        if not j["feasible"]:
            rec["reasons"] = rec["reasons"] + j["reasons"]
            rec["feasible"] = False
    write_json(it.p("S4_probe.json"), rec)
    if not rec["feasible"]:
        return it.finish("dropped", ",".join(rec["reasons"]))
    it.advance("S4", "S5")


def s5(P, it):
    c = read_json(it.p("S1_candidate.json"))
    client = P.client("spec")
    excerpt = it.p("fulltext.txt").read_text()[: P.cfg["spec"]["excerpt_chars"]]
    probe = {k: v for k, v in read_json(it.p("S4_probe.json"), {}).items() if k != "decontam"}
    avoid = it.state.get("avoid_mid_drop", [])
    lo, hi = P.cfg["spec"].get("expert_hours", [2, 6])
    user = S5_SPEC.format(doi=c.get("doi"), excerpt=excerpt, probe=json.dumps(probe),
                          forms=P.cfg["spec"]["allowed_forms"], hours_lo=lo, hours_hi=hi)
    if avoid:
        user += f"\nA previous spec used mid_drop {avoid}: it did not separate reference and mid. Choose another."
    errs = ["not run"]
    for _ in range(1 + P.cfg["spec"]["spec_retries"]):
        r = client.chat(SYSTEM, user, stage="S5", item=it.item_id, max_tokens=P.cfg["spec"].get("max_tokens"))
        save_raw(it, "S5", r)
        try:
            spec = extract_json(r.text)
        except ValueError as e:
            errs = [f"unparseable: {e}"]
            continue
        errs = validate_spec(spec, P.cfg)
        if not errs:
            break
        user += f"\nYour previous answer failed validation: {errs}. Fix and return the full JSON."
    write_json(it.p("_spec", "spec_errors.json"), errs)
    if errs:
        return it.finish("dropped", "spec_invalid")
    for k in ("depgraph",):
        write_json(it.p("_spec", f"{k}.json"), spec[k])
    write_json(it.p("_spec", "spec.json"), spec)
    write_json(it.p("_spec", "coverage.json"), dict(form=spec.get("form"), domain=spec.get("domain"),
                                                    field=spec.get("field"), competencies=spec.get("competencies")))
    write_json(it.p("_spec", "provenance.json"), dict(doi=c.get("doi"), title=c.get("title"),
                                                      target=c.get("_target"), prompt_version=PROMPT_VERSION,
                                                      fulltext=read_json(it.p("S3_fulltext.json")),
                                                      fulltext_sha=hashlib.sha256(excerpt.encode()).hexdigest()[:16]))
    it.advance("S5", "S6A", form=spec.get("form"))


def _write_files(root: Path, files: dict[str, str], allowed_prefixes: tuple[str, ...]):
    bad = [f for f in files if not f.startswith(allowed_prefixes) or ".." in f]
    for rel, body in files.items():
        if rel in bad:
            continue
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return bad


def save_raw(it, stage: str, r) -> Path:
    """Keep every raw model response (debugging truncation / parse failures needs the text, not just usage)."""
    n = it.bump(f"raw:{stage}")
    p = it.p("raw", f"{stage}-{n}.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# stop_reason={r.stop_reason} usage={json.dumps(r.usage)}\n" + (r.text or ""))
    return p


def chat_files(P, it, role: str, stage: str, user: str, need: set[str], max_tokens: int | None = None,
               need_prefix: tuple[str, ...] = ()) -> tuple[dict, list[dict]]:
    """One multi-file call; if required files are missing (typically stop_reason=max_tokens), at most
    `continue_calls` continuation calls that show the files already received and ask only for the missing ones."""
    client = P.client(role)
    r = client.chat(SYSTEM, user, stage=stage, item=it.item_id, max_tokens=max_tokens)
    save_raw(it, stage, r)
    files = parse_files(r.text, truncated=r.stop_reason == "max_tokens")
    log = [dict(stop=r.stop_reason, out_tok=r.usage.get("output_tokens"), got=sorted(files))]

    def missing():
        m = need - set(files)
        if need_prefix and not any(f.startswith(need_prefix) for f in files):
            m |= {need_prefix[0] + "*"}
        return m

    for _ in range(P.cfg["gen"].get("continue_calls", 1)):
        if not missing():
            break
        shown = "\n".join(f"=== FILE: {k} ===\n{v}" for k, v in files.items())
        cont = (user + "\n\nYour previous answer was cut off or incomplete.  Files already received (fixed, do NOT "
                "repeat them):\n" + shown + "\n=== END ===\n\nNow write ONLY these missing files, consistent with the "
                f"files above: {sorted(missing())}.  Keep code compact.  Format: === FILE: <path> === blocks, then "
                "=== END ===")
        r = client.chat(SYSTEM, cont, stage=stage, item=it.item_id, max_tokens=max_tokens)
        save_raw(it, stage, r)
        more = parse_files(r.text, truncated=r.stop_reason == "max_tokens")
        files.update({k: v for k, v in more.items() if k not in files})
        log.append(dict(stop=r.stop_reason, out_tok=r.usage.get("output_tokens"), got=sorted(more), cont=True))
    return files, log


def s6a(P, it):
    spec = read_json(it.p("_spec", "spec.json"))
    task = it.p("task")
    if not (task / "generator" / "generate.py").exists() or it.state.get("regen_6a"):
        need = {"generator/generate.py", "solution/solve.py", "tests/measure.py", "baselines/mid.py",
                "baselines/naive.py"}
        user = S6A_GEN.format(spec=json.dumps(spec, indent=1), metrics_api=METRICS_API)
        fb = it.state.get("s6a_feedback")
        if fb:
            user += S6A_REWRITE.format(ratios=json.dumps(fb.get("ratios")))
        files, log = chat_files(P, it, "gen", "S6A", user, need, max_tokens=P.cfg["gen"].get("s6a_max_tokens"))
        write_json(it.p("S6A_calls.json"), log)
        if not need <= set(files):
            write_json(it.p("S6A_error.json"), dict(missing=sorted(need - set(files)), got=sorted(files), calls=log))
            return it.finish("dropped", "s6a_missing_files")
        _write_files(task, files, ("generator/", "solution/", "tests/", "baselines/"))
        it.state.pop("regen_6a", None)
        it.state.pop("s6a_feedback", None)
        it.save()
    it.advance("S6A", "S6A_run")


def s6a_run(P, it):
    """Measure the reference (and mid/naive if S6C already produced them) on each seed."""
    task = it.p("task")
    work = it.p("runs", "s6")
    meas = {"ref": {}, "mid": {}, "naive": {}, "nop": {}, "runs": {}}
    for s in P.cfg["gen"]["seeds"]:
        gen = work / f"gen{s}"
        g = runner.generate(P.cfg, task, s, gen)
        meas["runs"][f"gen{s}"] = {k: g[k] for k in ("rc", "sec", "err", "public_mb")}
        if g["rc"] != 0:
            write_json(it.p("S6A_measure.json"), meas)
            return _to_repair(P, it, "generator_failed", meas)
        # nop (empty /app/output) gives each metric's missing-output value: a reference equal to it on some
        # metric did not produce that deliverable (review #7: replaces the global |value| >= 1e5 heuristic)
        nd = work / f"out-nop{s}"
        if nd.exists():
            shutil.rmtree(nd)
        nd.mkdir(parents=True)
        nd.chmod(0o777)
        mn = runner.measure(P.cfg, task, gen, nd, work, f"nop{s}")
        meas.setdefault("nop", {})[str(s)] = mn["values"]
        meas.setdefault("speed_bench", {})[str(s)] = runner.speed_bench(P.cfg, task, gen, work, s)
        for who, prog in (("ref", "solution/solve.py"), ("mid", "baselines/mid.py"), ("naive", "baselines/naive.py")):
            if not (task / prog).exists():
                continue
            sv = runner.solve(P.cfg, task, gen, prog, work, f"{who}{s}")
            m = runner.measure(P.cfg, task, gen, Path(sv["outdir"]), work, f"{who}{s}")
            meas["runs"][f"{who}{s}"] = dict(solve_rc=sv["rc"], solve_sec=sv["sec"], measure_rc=m["rc"],
                                             err=(sv["err"] or m["err"])[-400:])
            meas[who][str(s)] = m["values"]
    write_json(it.p("S6A_measure.json"), meas)
    if any(meas["runs"][f"ref{s}"]["solve_rc"] != 0 for s in P.cfg["gen"]["seeds"]):
        return _to_repair(P, it, "reference_failed", meas)
    it.advance("S6A_run", "S6B")


def s6b(P, it):
    """Thresholds from the measurement, then spec.s6b_decision: S6C / S8 / S6M / S5 / S6A / drop.  Every decision
    is appended to S6B_decisions.json; an early stop (decision.rule set) is also an `early_stop` event."""
    spec = read_json(it.p("_spec", "spec.json"))
    meas = read_json(it.p("S6A_measure.json"))
    if it.state.get("thresholds_locked"):
        return _s6b_locked(P, it, spec, meas)
    th = set_thresholds(spec, meas, P.cfg)
    bench = [v for v in (meas.get("speed_bench") or {}).values() if isinstance(v, (int, float))]
    if bench:                      # machine-speed reference for the verifier's time-threshold scaling
        th["speed_bench_sec"] = sorted(bench)[len(bench) // 2]
    dec = s6b_decision(th, meas, P.cfg, it.state)
    th["decision"] = dec
    write_json(it.p("thresholds.json"), th)
    log = read_json(it.p("S6B_decisions.json"), []) or []
    log.append(dict(ts=time.time(), status=th["status"], **dec))
    write_json(it.p("S6B_decisions.json"), log)
    if dec["rule"]:
        P.store.event("early_stop", item=it.item_id, stage="S6B", rule=dec["rule"], action=dec["action"],
                      reason=dec["reason"], detail=dec["detail"])
    act = dec["action"]
    if act == "S6C":
        return it.advance("S6B", "S6C", separating=th["separating"])
    if act == "S8":
        return _to_repair(P, it, th["status"], th)
    if act == "S6M":
        # v2.2: a cheaper fix than a new spec: rewrite only baselines/mid.py around another drop node
        return it.goto("S6M", dec["reason"])
    if act == "S5":
        it.state["back_to_spec"] = it.state.get("back_to_spec", 0) + 1
        it.state.setdefault("avoid_mid_drop", []).extend(spec["depgraph"].get("mid_drop", []))
        it.state["regen_6a"] = True
        it.state["regen_6c"] = True
        return it.goto("S5", dec["reason"])     # one save: counters and stage change land together
    if act == "S6A":
        # the reference is grossly wrong (orders of magnitude worse than the naive baseline): regenerate the code
        # once from the same spec instead of patching it in S8
        it.state["ref_rewrite"] = it.state.get("ref_rewrite", 0) + 1
        it.state["regen_6a"] = True
        it.state["regen_6c"] = True
        it.state["s6a_feedback"] = dict(why=dec["reason"], ratios=dec["detail"].get("ratios"))
        # one save (in goto): a crash cannot leave ref_rewrite spent while the item still sits at S6B.  S8 rounds
        # already spent stay spent (gates.max_repair_rounds is per item)
        return it.goto("S6A", dec["reason"])
    return it.finish("dropped", dec["reason"], early_stop=dec["rule"])


def _s6b_locked(P, it, spec, meas):
    """S6B after an S9R review revision or a regate: the thresholds stay those of thresholds.json (never
    re-derived, never loosened).  The fresh derivation from the new measurement is recorded next to them
    (thresholds.relock.json) for the reviewer; the reference must still clear every locked threshold on every
    seed.  A threshold set before the machine-speed reference was recorded (thresholds.speed_bench_sec) gets it
    backfilled from this measurement — thresholds are set on this host, so its bench time is the reference.  A
    reference that misses a locked threshold fails the revision: the review revision is retried with the failure
    fed back (stages_b._revision_retry) while rounds remain, else the item is dropped."""
    from . import review
    old = read_json(it.p("thresholds.json"))
    fresh = set_thresholds(spec, meas, P.cfg)
    cmp_ = review.lock_compare(old["thresholds"], fresh["thresholds"])
    bench = [v for v in (meas.get("speed_bench") or {}).values() if isinstance(v, (int, float))]
    here = sorted(bench)[len(bench) // 2] if bench else None
    base = old.get("speed_bench_sec") if isinstance(old.get("speed_bench_sec"), (int, float)) else None
    if here and base is None:
        old["speed_bench_sec"] = base = here
        old["speed_bench_source"] = "backfilled by the locked re-measurement on this host"
        write_json(it.p("thresholds.json"), old)
    # the verifier's policy, applied to the re-measurement: resource metrics decide only with
    # spec.resource_metric_gating, and time thresholds scale with this host's speed (the locked literal stays)
    kinds = runner.resource_metric_kinds(spec) or {}
    gating = runner.resource_gating(P.cfg, spec)
    ratio = runner.speed_ratio(P.cfg, here, base)
    fails = {}
    for k, v in old["thresholds"].items():
        kind = kinds.get(k)
        if kind and not gating:
            continue                                    # recorded only: never fails the revision
        thr = v["threshold"]
        if kind == "time":
            thr = thr * ratio if v["direction"] == "<=" else thr / ratio
        for s, vals in (meas.get("ref") or {}).items():
            x = (vals or {}).get(k)
            ok = isinstance(x, (int, float)) and (x <= thr if v["direction"] == "<=" else x >= thr)
            if not ok:
                fails.setdefault(k, {})[s] = x
    write_json(it.p("thresholds.relock.json"), dict(ts=time.time(), locked=old["thresholds"], fresh=fresh["thresholds"],
                                                    compare=cmp_, ref_fails_locked=fails))
    log = read_json(it.p("S6B_decisions.json"), []) or []
    log.append(dict(ts=time.time(), status="locked", action="S6C" if not fails else "failed", rule="thresholds_locked",
                    reason="review revision: thresholds kept", detail=dict(ref_fails_locked=fails)))
    write_json(it.p("S6B_decisions.json"), log)
    if fails:
        from .stages_b import _revision_retry
        return _revision_retry(P, it, "reference fails a locked threshold after the revision: "
                                      + ",".join(sorted(fails)))
    return it.advance("S6B", "S6C", locked=True)


def s6m(P, it):
    """Rewrite baselines/mid.py around a different mid_drop node (one model call), then re-measure."""
    spec = read_json(it.p("_spec", "spec.json"))
    meas = read_json(it.p("S6A_measure.json"))
    th = read_json(it.p("thresholds.json"))
    task = it.p("task")
    g = spec["depgraph"]
    tried = list(dict.fromkeys(g.get("mid_drop", []) + it.state.get("avoid_mid_drop", [])))
    n_fail = max(th.get("mid_fail", {}).values() or [0])
    user = S6M_MID.format(spec=json.dumps(spec, indent=1), solve=(task / "solution" / "solve.py").read_text(),
                          naive_drop=g.get("naive_drop"),
                          ref=json.dumps(meas["ref"]), mid=json.dumps(meas["mid"]), old_drop=g.get("mid_drop"),
                          n_fail=n_fail, n_metrics=len(spec.get("metrics", [])), tried=tried)
    crashed = crashed_runs(meas, "mid")
    if crashed:
        err = next(iter(crashed.values()))[-300:]
        user = user.replace("The current mid fails too many", "The current mid CRASHED (non-zero exit; stderr tail: "
                            + json.dumps(err) + "), so it fails everything; before it was").replace(
            "is too\nfundamental", "may or may not be too\nfundamental")
    elif n_fail < P.cfg["thresholds"]["min_mid_fail"]:
        user = user.replace("The current mid fails too many", "The current mid fails NONE of them (the dropped "
                            "step does not matter enough); before it was").replace(
            "is too\nfundamental", "has too little effect")
    it.state["mid_rework"] = it.state.get("mid_rework", 0) + 1
    files, log = chat_files(P, it, "gen", "S6M", user, {"baselines/mid.py"},
                            max_tokens=P.cfg["gen"].get("s6c_max_tokens"))
    write_json(it.p(f"S6M_calls-{it.state['mid_rework']}.json"), log)
    raw = sorted(it.p("raw").glob("S6M-*.txt"))
    m = re.search(r"MID_DROP:\s*([A-Za-z0-9_\-]+)", raw[-1].read_text()) if raw else None
    if "baselines/mid.py" not in files:
        return it.finish("dropped", "s6m_missing_file")
    _write_files(task, {"baselines/mid.py": files["baselines/mid.py"]}, ("baselines/mid.py",))
    if "baselines/naive.py" in files and m and m.group(1) not in g.get("naive_drop", []):
        _write_files(task, {"baselines/naive.py": files["baselines/naive.py"]}, ("baselines/naive.py",))
    if m:
        new = m.group(1)
        cand = json.loads(json.dumps(spec))
        cand["depgraph"]["mid_drop"] = [new]
        if new not in cand["depgraph"].get("naive_drop", []):
            # the naive baseline already drops a superset of key derive nodes; keep naive_drop ⊇ mid_drop by
            # adding the node (naive.py is regenerated below only if the node really is new to it) (review #8)
            cand["depgraph"]["naive_drop"] = list(cand["depgraph"].get("naive_drop", [])) + [new]
        errs = validate_spec(cand, P.cfg)
        if errs:
            write_json(it.p("S6M_error.json"), dict(mid_drop=new, errors=errs))
            return it.finish("dropped", "s6m_invalid_spec:" + ";".join(errs)[:200])
        it.state.setdefault("avoid_mid_drop", []).extend(g.get("mid_drop", []))
        if cand["depgraph"]["naive_drop"] != spec["depgraph"].get("naive_drop"):
            it.state["naive_needs_node"] = new
        spec = cand
        write_json(it.p("_spec", "spec.json"), spec)
    it.save()
    it.advance("S6M", "S6A_run", mid_drop=m.group(1) if m else None)


def _to_repair(P, it, why: str, info):
    write_json(it.p("S6_failure.json"), dict(why=why, info=info))
    it.state["repair_from"] = why
    it.save()
    it.goto("S8", why)
