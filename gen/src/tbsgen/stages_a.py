"""Stage handlers S1-S6B.  Each handler gets (P, item) and returns via item.advance / item.goto / item.finish.
P is the Pipeline (cfg, store, ledger, clients).  Handlers write outputs before committing state."""
from __future__ import annotations

import contextlib
import hashlib
import time
import json
import shutil
import re
from pathlib import Path

from . import blocklist, codegate, failclass, intake, runner, screen_jev, taskimage
from .budget import BudgetStop
from .config import resolve_role
from .models import ModelError, extract_json, parse_files  # noqa: F401
from .prompts import METRICS_API, PROMPT_VERSION, S5_SPEC, S6A_GEN, S6A_REWRITE, S6M_MID, SYSTEM
from .spec import crashed_runs, image_prompt, s6b_decision, set_thresholds, validate_spec
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
        if sig.get("openalex_fulltext_error"):     # OpenAlex unreachable / rate-limited: undecided, retry later
            return it.finish("paused", "code:openalex_unavailable", paused_stage="S2a")
        return it.finish("dropped", "no_code_signal")
    it.advance("S2a", "S2", signals=[k for k in ("meta", "retrieval", "pwc", "openalex_fulltext") if sig.get(k)])


def s3b(P, it):
    """Code-grounded confirmation on the full text: the authors' own code exists and resolves on its host."""
    c = read_json(it.p("S1_candidate.json")) or {}
    # S2a's findings (e.g. a PwC archive repo) are evidence for S3b too
    sig = read_json(it.p("S2a_code_signal.json"), {}) or {}
    cs = dict(c.get("code_signals") or {})
    for k in ("pwc", "pwc_mentioned"):
        cs[k] = sorted(set(cs.get(k) or []) | set(sig.get(k) or []))
    c = dict(c, code_signals=cs)
    txt = it.p("fulltext.txt").read_text(errors="replace")
    jev = None
    if resolve_role(P.cfg, "code_owner") == "jev":
        client = P.jev_client()

        def jev(link):
            try:
                return codegate.jev_own(client, it, c, link)
            except (ModelError, BudgetStop) as e:
                if isinstance(e, BudgetStop) and e.reason.startswith("window:"):
                    raise
                P.store.event("code_owner_jev_unavailable", item=it.item_id, err=str(e)[:200])
                return None
    rec = codegate.s3b_record(txt, c, P.cfg, jev=jev)
    write_json(it.p("S3b_code.json"), rec)
    if rec.get("pause"):          # host / Jev could not answer now: retry later (`tbsgen resume --reason code:`)
        return it.finish("paused", f"code:{rec['pause']}", paused_stage="S3b")
    if not rec["keep"]:
        return it.finish("dropped", rec["reason"])
    p = rec["primary"]
    it.advance("S3b", _code_next_after_s3b(P), repo=p.get("url"),
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
        if isinstance(e, BudgetStop) and e.reason.startswith("window:"):
            raise
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
    rule = blocklist.match(c, rules)
    if rule is not None:                    # full text unobtainable: drop before spending a screen call —
        d = blocklist.decide(P.store, P.cfg, c, rules, rule, P.store.event)  # unless it is a recheck trial
        if d["action"] == "block":
            rec = blocklist.screen_record(c, d["rule"])
            write_json(it.p("S2_screen.json"), rec)
            return it.finish("dropped", rec["reason"])
        if d["action"] == "trial":
            it.state["blocklist_trial"] = d["rule"].get("name") or d["rule"].get("doi_prefix")
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
                                               skipped=rec.get("skipped"), next_stage=codegate.after_s3(P.cfg)))
        it.finish("parked", "fulltext_unavailable")
        if entry["ident"]:
            fulltext.enqueue(fulltext.queue_path(P.cfg), entry)
    else:
        it.p("fulltext.txt").write_text(txt)
        it.advance("S3", codegate.after_s3(P.cfg), source=rec.get("source"), chars=len(txt))
    if it.state.get("blocklist_trial"):
        blocklist.reevaluate(P.store, P.cfg, P.store.event)


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
            if isinstance(e, BudgetStop) and e.reason.startswith("window:"):
                raise
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
    it.advance("S4", "S4c" if codegate.grounded(P.cfg) and codegate.ccfg(P.cfg).get("runnability", True) else "S5")


def s5(P, it):
    if _agent_reauthor(P, it, "back_to_spec"):
        return
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
        if isinstance(spec, dict):
            spec["prompt_version"] = PROMPT_VERSION
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
    if it.state.get("regen_6a") and _agent_reauthor(P, it, "regenerate_reference"):
        return
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


def _format_warning(public, formats):
    declared = {ext.lower().lstrip(".") for ext in formats}
    observed = set()
    for path in public.rglob("*"):
        if path.is_file() and path.suffix:
            name = path.name.lower()
            matches = [ext for ext in declared if name.endswith("." + ext)]
            observed.add(max(matches, key=len) if matches else path.suffix[1:].lower())
    if declared != observed:
        return dict(missing=sorted(declared - observed), undeclared=sorted(observed - declared))


def s6a_run(P, it):
    """Measure the reference (and mid/naive if S6C already produced them) on each seed."""
    if it.p("_spec", "author_code.json").exists():
        from . import codeagent
        why = codeagent.env_ok(P.cfg)
        if why:                                 # a reference that ports the authors' code runs on the build host only
            return it.finish("paused", f"code:config:{why}", paused_stage="S6A_run")
    task = it.p("task")
    work = it.p("runs", "s6")
    spec = read_json(it.p("_spec", "spec.json"), {})
    meas = {"ref": {}, "mid": {}, "naive": {}, "nop": {}, "runs": {}}
    for s in P.cfg["gen"]["seeds"]:
        gen = work / f"gen{s}"
        g = runner.generate(P.cfg, task, s, gen)
        meas["runs"][f"gen{s}"] = {k: g[k] for k in ("rc", "sec", "err", "public_mb")}
        if g["rc"] != 0:
            write_json(it.p("S6A_measure.json"), meas)
            return _to_repair(P, it, "generator_failed", meas)
        if image_prompt(spec.get("prompt_version")):
            warning = _format_warning(gen / "environment/data", spec["data_formats"])
            if warning:
                meas["runs"][f"gen{s}"]["data_formats_warning"] = warning
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
        errs = validate_spec(cand, P.cfg, read_json(it.p("_spec", "provenance.json"), {}).get("prompt_version"))
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


# =============================================================================================== code-grounded S3c/S4c/S5G
def _code_next_after_s3b(P) -> str:
    return "S3c" if codegate.ccfg(P.cfg).get("fetch", True) else "S4"


def s3c(P, it):
    """Fetch the authors' code on the build host and pin it (commit SHA / archive sha256)."""
    from . import codeagent
    rec = read_json(it.p("S3b_code.json"), {}) or {}
    prim = rec.get("primary") or {}
    if not prim:
        return it.finish("dropped", "no_code_primary")
    why = codeagent.env_ok(P.cfg)
    if why:
        return it.finish("paused", f"code:config:{why}", paused_stage="S3c")
    r = codeagent.s3c_fetch(P.cfg, it.item_id, prim)
    write_json(it.p("S3c_code.json"), dict(r, primary_url=prim.get("url"), license=prim.get("license"),
                                           ts=time.time()))
    if not r.get("ok"):
        if r.get("transient") or r.get("why") in ("clone_failed", "download_failed") and it.bump("S3c:fetch") < 3:
            return it.finish("paused", f"code:fetch:{r.get('why')}", paused_stage="S3c")
        reason = {"too_large": "code_too_large", "unsupported": "code_host_unsupported"}.get(r.get("why"),
                                                                                          "code_fetch_failed")
        return it.finish("dropped", reason)
    it.advance("S3c", "S4", pin=r.get("pin"), size_mb=r.get("size_mb"), languages=r.get("languages"))


def _observe_remote_agent(P, it, stage):
    """Drain marker-backed work before any stage-specific preparation or terminal transition."""
    from . import calib, codeagent
    from .stages_b import finish_job_stage
    path = it.p(stage, "run.json")
    rec = codeagent.read_agent_run(P.cfg, it, stage) or {}
    finish = it.state.get("pending_job_finish")
    if rec.get("state") in ("intent", "reserved", "launched", "remote_unknown") and calib.remote_job(rec, P.ledger):
        with _agent_slot_lock(P):
            action = codeagent.reconcile_remote_agent(P.cfg, it, stage, rec, P.ledger)
            _agent_admit(P, it, stage)
        if action == "remote_unknown":
            it.finish("parked", f"remote_unknown: {rec['job_name']}")
            return True
        if action == "not_started":
            # the launcher ended and the job dir proves Harbor never ran: a zero-model-call job, not a park
            rec = codeagent.read_agent_run(P.cfg, it, stage) or rec
            if rec.get("not_started"):
                with _agent_slot_lock(P):
                    _withdraw_admission(P, it, stage, job=rec.get("job_name"), why="not_launched")
                    _record_nostart(it, stage, rec.get("job_name"))
            it.state["infra_backoff_until"] = time.time() + float(P.cfg["code"].get("infra_backoff_s") or 600)
            it.state["status"] = "active"
            it.state.pop("reason", None)
            it.save()
            return False
        if action == "waiting":
            reason = f"{stage}:agent_running"
            if it.state["status"] != "waiting" or it.state.get("reason") != reason:
                it.finish("waiting", reason, job=rec["job_name"])
            return True
        calib.check_job_identity(Path(rec["job_dir"]), rec)
        rec = codeagent.collect_agent(P.cfg, path, rec, P.ledger)
    if finish and rec.get("state") in ("finished", "launch_failed"):
        it.state.pop("pending_job_finish")
        finish_job_stage(P, it, finish["status"], finish["reason"], **finish["info"])
        return True
    return False


def reconcile_parked_agent(P, it):
    """Unpark a remote_unknown S4c / S5G item whose launcher has ended (failclass / launch evidence).

    A parked remote job is never stepped again, so a launcher that finished hours ago used to hold an agent
    slot forever.  When calib's end evidence is positive (launch log .exit present / tmux session gone AND a
    finished result.json or a job dir that proves Harbor never ran) the same reconcile path resolves it: a
    finished job is collected by the normal stage flow, a never-started job is accounted as a zero-model job
    and retried.  Anything ambiguous stays parked for the operator.  Returns True if the item was unparked."""
    from . import codeagent
    from .stages_b import item_lock
    with item_lock(it):
        it.load()
        stage = it.state.get("stage")
        if (it.state.get("status") != "parked" or stage not in ("S4c", "S5G")
                or not str(it.state.get("reason") or "").startswith("remote_unknown: ")):
            return False
        rec = codeagent.read_agent_run(P.cfg, it, stage) or {}
        if rec.get("state") != "remote_unknown":
            return False
        with _agent_slot_lock(P):
            action = codeagent.reconcile_remote_agent(P.cfg, it, stage, rec, P.ledger)
            if action == "not_started" and rec.get("not_started"):
                _withdraw_admission(P, it, stage, job=rec.get("job_name"), why="not_launched")
                _record_nostart(it, stage, rec.get("job_name"))
        if action == "remote_unknown":
            return False
        if action == "not_started":
            it.state["infra_backoff_until"] = time.time() + float(P.cfg["code"].get("infra_backoff_s") or 600)
        it.state["history"].append(dict(stage=stage, result="reconciled", prev_reason=it.state.get("reason"),
                                        action=action, ts=time.time()))
        it.state["status"] = "active"
        it.state.pop("reason", None)
        it.save()
        P.store.event("remote_unknown_reconciled", item=it.item_id, stage=stage, job=rec.get("job_name"),
                      action=action)
        return True


def _agent_stage(P, it, kind: str, instruction: str, need: list[str], timeout_s: float, max_usd: float | None,
                 est_usd: float, with_contract: bool, rnd: dict | None = None):
    """Shared S4c / S5G driver: build + launch the job once, then (later steps) poll it.  Returns the finished
    run record, or None while it runs (the item is left `waiting`)."""
    from . import codeagent
    recp = it.p(kind, "run.json")
    rec = read_json(recp, None)
    if rec is not None and rec.get("state") == "finished" and it.state.get(f"{kind}_rerun"):
        rec["retry"] = True                     # S6B sent the task back: author it again (with feedback)
        write_json(recp, rec)                   # persisted before the flag is dropped (crash / pause safe)
        it.state.pop(f"{kind}_rerun", None)
        it.save()
    if rnd is not None and rec is not None and rec.get("state") == "finished" and not rec.get("retry") and \
            rec.get("harden_round", 0) != rnd["round"]:
        rec["retry"] = True                     # a finished job of another version never counts for this round
        write_json(recp, rec)
    if rec is not None and rec.get("state") in ("launch_failed", "cancelled_unstarted"):
        if rec.get("failure_class") == failclass.NO_MODEL_CALL:
            parked = _record_infra_signature(P, it, kind, rec)   # same deterministic start failure again -> park
            if parked:
                it.finish("parked", parked)
                return None
        if rec.get("not_started"):
            with _agent_slot_lock(P):
                _withdraw_admission(P, it, kind, job=rec.get("job_name"), why="not_launched")
        rec["retry"] = True                     # infrastructure: a resumed item launches a fresh job
        rec["state"] = "finished"
        if rec.get("not_started"):
            _record_nostart(it, kind, rec.get("job_name"))     # tmux never started it: no job of the paper ran
        write_json(recp, rec)                   # persist the transition before a possible backoff return
    if rec is not None and rec.get("state") == "finished" and rec.get("retry"):
        _account_refusal(P, it, kind, rec)      # idempotent: completes a refusal's accounting cut short by a crash
    if rec is None or (rec.get("state") == "finished" and rec.get("retry")):
        until = float(it.state.get("infra_backoff_until") or 0)
        if until > time.time():
            reason = f"{kind}:infra_backoff"
            if it.state.get("status") != "waiting" or it.state.get("reason") != reason:
                it.finish("waiting", reason, job=(rec or {}).get("job_name"), backoff_until=until)
            return None
        from .calib import preflight_wait
        wait = preflight_wait(P.cfg, rec)
        if wait:                                # the relay refused the last job: wait before the next one
            if it.state.get("status") != "waiting" or it.state.get("reason") != wait:
                it.finish("waiting", wait, job=(rec or {}).get("job_name"))
            return None
        why = codeagent.env_ok(P.cfg, agent=True)
        if why:
            it.finish("paused", f"code:config:{why}", paused_stage=kind)
            return None
        # per-paper caps are cumulative over all jobs of this stage.  <kind>/spent.json maps job name -> charge and is
        # written (atomically) before anything else, so a crash and resume recomputes the same total (idempotent).
        sp = it.p(kind, "spent.json")
        charges = read_json(sp, {}) or {}
        if rec is not None and rec.get("job_name") and rec["job_name"] not in charges:
            if rec.get("not_started") or not rec.get("launch_attempted"):
                usd, sec, cost_kind = 0.0, 0.0, "not_launched"   # confirmed never started
            else:
                # failclass: only a job that provably made no model call (no agent files, no token usage, no
                # reported cost) is charged 0; every other unknown-cost job keeps counting its whole allowance.
                verdict = failclass.classify(rec.get("job_dir"), rec)
                rec["failure_class"], rec["error_signature"] = verdict.kind, verdict.signature
                rec["error_signature_short"] = verdict.short
                write_json(recp, rec)
                if verdict.kind == failclass.NO_MODEL_CALL:
                    usd, sec, cost_kind = 0.0, 0.0, "not_started"
                elif _failed_before_harbor(rec):
                    # explicit pre-start failure evidence in this job's own launcher log (the relay preflight
                    # refused, so Harbor never ran and no model was called): no model cost.  A refusal handled as
                    # such (preflight_refused) used no agent time either; an older record keeps its wall time
                    usd = 0.0
                    sec = 0.0 if rec.get("preflight_refused") else \
                        max(0.0, float(rec.get("finished_ts") or time.time()) - float(rec["launch_attempted"]))
                    cost_kind = "relay_refused" if rec.get("preflight_refused") else "not_launched"
                else:
                    # a job that may have run: unknown cost counts as its whole allowance, unknown time as wall
                    # time since the launch (never as zero)
                    b = rec.get("budget") or {}
                    usd = rec.get("cost_usd")
                    usd = float(usd) if usd is not None else float(b.get("max_usd") or max_usd or 0.0)
                    sec = 60.0 * float(rec["minutes"]) if rec.get("minutes") is not None else \
                        max(0.0, float(rec.get("finished_ts") or time.time()) - float(rec["launch_attempted"]))
                    cost_kind = rec.get("cost_kind") if rec.get("cost_usd") is not None else "reservation"
            charges[rec["job_name"]] = dict(usd=usd, sec=sec)
            write_json(sp, charges)
        spent = dict(usd=sum(c["usd"] for c in charges.values()), sec=sum(c["sec"] for c in charges.values()))
        it.state[f"{kind}_spent"] = spent
        if rnd is not None:
            # a hardening round (difficulty.py) has its own allowance: spend since the round started
            base = rnd["base"]
            spent = dict(usd=spent["usd"] - float(base["usd"]), sec=spent["sec"] - float(base["sec"]))
            timeout_s, max_usd = float(rnd["timeout_s"]), rnd["max_usd"]
        left_s = timeout_s - spent["sec"]
        left_usd = None if max_usd is None else float(max_usd) - spent["usd"]
        floor = (600.0, 1.0) if spent["sec"] or spent["usd"] else (1.0, 0.01)   # first job: the configured caps
        from .budget import item_usd_left
        cap_left = item_usd_left(P.ledger, P.cfg, it.item_id) if kind in \
            (P.cfg["budget"].get("item_usd_cap_stages") or []) else None
        if cap_left is not None:                # budget.item_usd_cap: the paper's whole authoring allowance
            left_usd = cap_left if left_usd is None else min(left_usd, cap_left)
            floor = (floor[0], max(floor[1], 1.0))
        if left_s < floor[0] or (left_usd is not None and left_usd < floor[1]):
            it.finish("dropped", f"{kind.lower()}_budget_exhausted", spent=spent, item_usd_left=cap_left)
            return None
        timeout_s, max_usd, est_usd = left_s, left_usd, min(est_usd, left_usd) if left_usd is not None else est_usd
        if cap_left is not None and max_usd is not None:
            # with a per-paper cap the ledger reserves the job's whole allowance: an unknown final cost then counts
            # as that allowance in the ledger too (the same bound spent.json uses), never as a smaller estimate
            est_usd = float(max_usd)
        it.state.pop("infra_backoff_until", None)
        with _agent_slot_lock(P):
            # admission check, job number, intent, reservation and launch under one run-level lock: two processes
            # stepping different papers cannot both take the last slot / daily admission
            checked_day = _cst_day()
            why = _agent_admission(P, it, kind)
            if why:
                it.finish(_admission_status(why), why, paused_stage=kind)
                return None
            n = it.bump(f"{kind}:job")
            work = it.p(kind, "work")
            codeagent.stage_inputs(P.cfg, it, work, with_contract)
            images = kind == "S5G" and taskimage.author_images(P.cfg, it.state)
            task = codeagent.build_agent_task(P.cfg, kind, it.item_id, work, it.p(kind, "task"), instruction, need,
                                              timeout_s, image_outputs=images)
            jobs = Path(codeagent._c(P.cfg).get("agent_jobs_dir") or it.p(kind, "jobs"))
            conf = codeagent.agent_job_config(P.cfg, task, f"tbsgen-{it.item_id}-{kind.lower()}{n}", jobs, max_usd,
                                              kind=kind)
            extra = dict(budget=dict(timeout_s=timeout_s, max_usd=max_usd), est_usd=est_usd)
            if kind == "S5G":
                from .prompts_code import CODE_PROMPT_VERSION, LEGACY_CODE_PROMPT_VERSION
                extra["prompt_version"] = CODE_PROMPT_VERSION if images else LEGACY_CODE_PROMPT_VERSION
                if images:
                    extra["image_outputs"] = True
            if rnd is not None:
                extra["harden_round"] = rnd["round"]
            rec = _launch_admitted_agent(P, it, kind, conf, est_usd, checked_day, extra=extra)
            rec.update(extra)
            write_json(recp, rec)
        if rec.get("state") == "remote_unknown":
            it.finish("parked", f"remote_unknown: {rec['job_name']}")
            return None
        if rec.get("state") == "launch_failed":
            if rec.get("not_started"):
                _record_nostart(it, kind, rec.get("job_name"))   # tmux never started it (also in a hardening round)
            return _launch_failed(it, kind)
        reason = "launch_grace" if "confirm_launch_after" in rec else f"{kind}:agent_running"
        it.finish("waiting", reason, job=rec["job_name"])
        return None
    if rec.get("state") in ("intent", "reserved", "cancelled_unstarted"):
        with _agent_slot_lock(P):
            _settle_orphans(P, recp, rec, "diversity: reservation released before launch")
            started = codeagent.launch_evidence(P.cfg, rec)
            if started is None:
                it.finish("waiting", "launch_grace", job=rec.get("job_name"))
                return None
            checked_day = _cst_day()
            if not started and not rec.get("launch_attempted"):
                # Only this job's first-paper admission is provisional. An earlier successful job of the
                # same paper still exempts its retries from the daily new-paper cap.
                if rec.get("job_name") and any(
                        row.get("job") == rec["job_name"] and row.get("item") == it.item_id and row.get("day") != checked_day
                        for row in _admissions(P, kind)):
                    _withdraw_admission(P, it, kind, job=rec["job_name"], why="unstarted_day_rollover")
                why = _agent_admission(P, it, kind)
                if why:
                    if (why.startswith(("budget:", "diversity:")) and rec["state"] == "intent"
                            and not rec.get("reservation_key") and rec.get("config")):
                        conf = read_json(Path(rec["config"]), {})
                        if conf:
                            codeagent.recover_legacy_agent_call(it, kind, rec, conf, P.ledger)
                    if why.startswith("budget:"):
                        P.ledger.cancel_unstarted(rec)
                        rec.update(state="cancelled_unstarted", ledger_call=None)
                        write_json(recp, rec)
                    elif why.startswith("diversity:"):
                        if rec.get("job_name"):
                            _withdraw_admission(P, it, kind, job=rec["job_name"], why="diversity_parked")
                        # Persist the orphan before releasing its reservation, so recovery can finish the release.
                        orphan = rec.get("ledger_call")
                        key = rec.pop("reservation_key", None)
                        if key and not orphan:  # DB commit survived, reserved run.json did not
                            orphan = P.ledger.reserved_call(key)
                        rec.update(state="intent", ledger_call=None,
                                   orphan_calls=list(rec.get("orphan_calls") or []) + ([orphan] if orphan else []))
                        write_json(recp, rec)
                        _settle_orphans(P, recp, rec, why)
                    it.finish(_admission_status(why), why, paused_stage=kind)
                    return None
            rec = _launch_admitted_agent(P, it, kind, read_json(Path(rec["config"])),
                                         float(rec.get("est_usd", est_usd)), checked_day, started=started)
        if rec.get("state") == "remote_unknown":
            it.finish("parked", f"remote_unknown: {rec['job_name']}")
            return None
        if rec.get("state") == "launch_failed":
            if rec.get("not_started"):
                _record_nostart(it, kind, rec.get("job_name"))   # tmux never started it (also in a hardening round)
            return _launch_failed(it, kind)
        reason = "launch_grace" if "confirm_launch_after" in rec else f"{kind}:agent_running"
        it.finish("waiting", reason, job=rec.get("job_name"))
        return None
    if rec.get("state") == "launched":
        if codegate.ccfg(P.cfg).get({"S4c": "daily_new_runnability", "S5G": "daily_new_authoring"}[kind]) is not None:
            with _agent_slot_lock(P):
                _agent_admit(P, it, kind)       # idempotent: a crash between "launched" and the admission line
        rec = codeagent.poll_agent(P.cfg, it, kind, P.ledger)
        if rec and rec.get("state") == "remote_unknown":
            it.finish("parked", f"remote_unknown: {rec['job_name']}")
            return None
        if rec and rec.get("not_started"):
            with _agent_slot_lock(P):
                _withdraw_admission(P, it, kind, job=rec["job_name"], why="not_launched")
            _record_nostart(it, kind, rec["job_name"])
            return _launch_failed(it, kind)
        if rec is None or rec.get("state") != "finished":
            reason = "launch_grace" if "confirm_launch_after" in (rec or {}) else f"{kind}:agent_running"
            if it.state.get("status") != "waiting" or it.state.get("reason") != reason:
                it.finish("waiting", reason, job=(rec or {}).get("job_name"))
            return None
    if rec is not None and _account_refusal(P, it, kind, rec):
        from .calib import preflight_wait
        it.finish("waiting", preflight_wait(P.cfg, rec) or "relay_down: preflight refused", job=rec.get("job_name"))
        return None
    return rec


def _launch_admitted_agent(P, it, kind, conf, est_usd, checked_day, extra=None, started=None):
    """Called under the admission lock, including the launch-intent write and any rollback."""
    from . import codeagent
    from .budget import BudgetStop

    def record_intent(rec):
        day = _cst_day(rec["launch_attempted"])
        if day != checked_day:
            # Preparation may span local midnight while we hold the admission lock.
            _withdraw_admission(P, it, kind, job=rec["job_name"], why="unstarted_day_rollover")
            why = _daily_paper_cap(P, it, kind, day)
            if why:
                P.ledger.cancel_unstarted(rec)
                rec.pop("launch_attempted", None)  # admission refused before tmux
                rec.update(state="cancelled_unstarted", ledger_call=None)
                write_json(it.p(kind, "run.json"), rec)
                raise BudgetStop(why.removeprefix("budget:"))
        _agent_admit(P, it, kind)

    try:
        rec = codeagent.launch_agent(P.cfg, it, kind, conf, P.ledger, est_usd, extra=extra,
                                     on_launch_intent=record_intent, started=started)
    except BudgetStop as exc:
        if exc.cancelled_unstarted:
            _withdraw_admission(P, it, kind, job=conf["job_name"], why="budget_refused")
        raise
    if rec.get("not_started"):
        _withdraw_admission(P, it, kind, job=conf["job_name"], why="not_launched")
    elif rec["state"] in ("launched", "remote_unknown"):
        _agent_admit(P, it, kind)  # backfill a recovered launch
    return rec


def _settle_orphans(P, recp, rec, note: str) -> None:
    """Close ledger rows of reservations a diversity park released (run.json is back at intent).  Idempotent: the
    row id leaves orphan_calls only after its settle committed; settling a closed row again is harmless.
    cost_kind diversity_parked = no model call was made (excluded from the call-count caps)."""
    for call in list(rec.get("orphan_calls") or []):
        P.ledger.settle(call, 0., status="error", kind="diversity_parked", note=note[:300])
        rec["orphan_calls"].remove(call)
        write_json(recp, rec)


def relay_waiting(cfg: dict, it) -> bool:
    """A waiting S4c / S5G item whose last job the relay refused less than calib.preflight_retry_s ago: there is
    nothing to poll (pipeline.run skips it so it never takes a scheduling step)."""
    from .calib import preflight_wait
    st = it.state.get("stage")
    if st not in ("S4c", "S5G") or not str(it.state.get("reason") or "").startswith("relay_down"):
        return False
    return bool(preflight_wait(cfg, read_json(it.p(st, "run.json"), None)))


def _nostart(it, kind: str) -> list[str]:
    """Jobs of this stage that are known never to have started (relay preflight refused / tmux failed), by name."""
    return list(((it.state.get("agent_nostart") or {}).get(kind)) or [])


def _record_nostart(it, kind: str, job: str | None, save: bool = True) -> None:
    if job and job not in _nostart(it, kind):
        it.state.setdefault("agent_nostart", {}).setdefault(kind, []).append(job)
    if save:
        it.save()


def _none_started(it, kind: str) -> bool:
    """Every job number this paper was given for `kind` is a job known never to have started.  A number without
    such evidence (a job that ran, a record from before this rule, a crash between numbering and launch) counts as
    started: the paper keeps its daily admission (the conservative side of the cap)."""
    n = int((it.state.get("attempts") or {}).get(f"{kind}:job", 0))
    ns = set(_nostart(it, kind))
    return all(f"tbsgen-{it.item_id}-{kind.lower()}{k}" in ns for k in range(1, n + 1))


def _account_refusal(P, it, kind: str, rec: dict) -> bool:
    """A finished job of `kind` that the model relay refused at its preflight (explicit evidence in its own
    launcher log, no Harbor job): nothing ran, so it is not an infra failure of the paper -- no <kind>:infra retry,
    no authoring_max_jobs job, no daily admission while none of the paper's jobs started, 0 USD / 0 s.  Every step
    is idempotent and the whole accounting is re-checked on each call, so a crash (or another process) between any
    two steps is completed by the next step instead of being counted as an infra failure.  True = refused (the
    caller waits calib.preflight_retry_s; the next launch is a fresh job)."""
    if rec.get("state") != "finished" or not _failed_before_harbor(rec):
        return False
    job = rec.get("job_name")
    with _agent_slot_lock(P):
        if not rec.get("preflight_refused"):
            rec.update(preflight_refused=True, refused_ts=time.time(), retry=True, not_started=True)
            write_json(it.p(kind, "run.json"), rec)
        if rec.get("ledger_call"):              # idempotent: the refused reservation is no call (budget call caps)
            P.ledger.settle(rec["ledger_call"], 0.0, status="error", kind="relay_refused",
                            note="relay preflight refused: no Harbor job, no model call")
        first = job not in _nostart(it, kind)
        _record_nostart(it, kind, job, save=False)
        if _none_started(it, kind):
            _withdraw_admission(P, it, kind)        # no job of this paper started: give the daily slot back
        it.save()                               # last: a crash before it redoes the (idempotent) steps above
    if first:
        P.store.event("relay_preflight_refused", item=it.item_id, stage=kind, job=job)
    return True


def s4c(P, it):
    """Serialised per item like s5g: two processes stepping the same paper (run + a second run / watch) never poll,
    account and relaunch its job at the same time."""
    from .stages_b import item_lock
    with item_lock(it):
        it.load()
        if it.state.get("stage") != "S4c" or it.state.get("status") not in ("active", "waiting"):
            return None
        if _observe_remote_agent(P, it, "S4c"):
            return None
        return _s4c(P, it)


def _record_infra_signature(P, it, stage, rec):
    """Count consecutive identical error signatures for one item+stage; park when deterministic and repeated.

    Returns the park reason or None.  Stored on state.infra_signatures ({stage: {last, count}}) so a resumed
    step sees the same count; the signature never contains job ids, paths or numbers."""
    if not rec.get("error_signature"):
        return None
    verdict = failclass.Verdict(kind=rec.get("failure_class") or failclass.MODEL_FAILED,
                                signature=rec["error_signature"], short=rec.get("error_signature_short") or "sig")
    sigs = it.state.setdefault("infra_signatures", {}).setdefault(stage, {})
    if sigs.get("last") == verdict.signature:
        sigs["count"] = int(sigs.get("count") or 0) + 1
    else:
        sigs.update(last=verdict.signature, count=1)
    repeat = max(1, int(P.cfg["code"].get("infra_repeat_park") or 2))
    if failclass.repeat_is_deterministic(verdict) and sigs["count"] >= repeat:
        it.save()
        P.store.event("agent_infra_repeat", item=it.item_id, stage=stage, job=rec.get("job_name"),
                      signature=verdict.signature, failure_class=verdict.kind)
        return f"infra_repeat:{verdict.short}"
    return None


def _retry_agent_infra(P, it, stage, rec):
    """Retry a stage job whose agent or launcher failed before it delivered: at most twice for transient agent errors
    (API, timeout, non-zero exit), once for launcher_failed / no_result. A job that delivered complete outputs is used
    as is even if the agent ended with an error.

    failclass: a job that provably made no model call does not count as an infra attempt -- it is retried after
    code.infra_backoff_s.  The same error signature code.infra_repeat_park times in a row (deterministic failures
    only; model failures keep their old budget) parks the item as ``infra_repeat:<signature>``.  Returns
    ("retry"|"park"|"", reason)."""
    outcome = rec.get("outcome")
    if outcome not in ("agent_error", "launcher_failed", "no_result"):
        return "", ""
    verdict = failclass.classify(rec.get("job_dir"), rec)
    rec["failure_class"], rec["error_signature"] = verdict.kind, verdict.signature
    rec["error_signature_short"] = verdict.short
    write_json(it.p(stage, "run.json"), rec)
    parked = _record_infra_signature(P, it, stage, rec)
    if parked:
        return "park", parked
    counter = f"{stage}:infra"
    if verdict.kind == failclass.NO_MODEL_CALL:
        # zero-spend infra failure: no stage infra attempt, plain retry after a backoff (the lifetime of the
        # failure is bounded by the repeat-signature rule above, not by the dropped-paper rule)
        backoff = float(P.cfg["code"].get("infra_backoff_s") or 600)
        it.state["infra_backoff_until"] = time.time() + backoff
        it.state.pop("last_agent_infra_job", None)
        _record_nostart(it, stage, rec.get("job_name"), save=False)   # nor an authoring_max_jobs job
    else:
        # A resumed step can see the same finished job after the counter was saved.
        if it.state.get("last_agent_infra_job") != rec["job_name"]:
            it.state["last_agent_infra_job"] = rec["job_name"]
            it.bump(counter)
        limit = 2 if outcome == "agent_error" else 1
        if it.state.get("attempts", {}).get(counter, 0) > limit:
            return "", ""
    rec["retry"] = True
    write_json(it.p(stage, "run.json"), rec)
    it.save()
    P.store.event("agent_infra_retry", item=it.item_id, stage=stage, job=rec.get("job_name"),
                  failure_class=verdict.kind, signature=verdict.signature, backoff=verdict.kind == failclass.NO_MODEL_CALL)
    return "retry", verdict.short


def _s4c(P, it):
    """Runnability gate: a Claude Code agent installs and runs the authors' code once on the build host."""
    from . import codeagent
    from .prompts_code import S4C_RUNNABILITY
    c = codegate.ccfg(P.cfg)
    tmo = float(c.get("runnability_timeout_s", 3600))
    rec = _agent_stage(P, it, "S4c", S4C_RUNNABILITY.format(item=it.item_id, minutes=int(tmo // 60)),
                       codeagent.agent_output_files("S4c"), tmo, c.get("runnability_max_usd", 8.0),
                       float(c.get("runnability_est_usd", 6.0)), with_contract=False)
    if rec is None:
        return
    out = codeagent.deliverable(rec)
    res = None
    if out:
        for f in ("runnability.json", "HOWTO.md", "setup.sh"):
            body = codeagent.safe_read(out / f, root=Path(rec["trial"]))
            if body is not None:
                it.p("S4c", f).write_text(body)
        try:
            res = json.loads(codeagent.safe_read(out / "runnability.json", root=Path(rec["trial"])) or "null")
        except ValueError:
            res = None
    receipts = []
    if out:
        receipts = codeagent.run_receipts(out)
        write_json(it.p("S4c", "receipts.json"), receipts)
    if isinstance(res, dict) and res.get("runnable") is True:
        # a runnable verdict needs: reason "ok", the how-to, a verifier pass and a wrapper receipt with exit code 0
        # whose command is the reported entry (and its log)
        ok_runs = [r for r in receipts if r.get("rc") == 0 and str(r.get("cmd", "")).strip() ==
                   str(res.get("entry", "")).strip() and out and codeagent.safe_read(out / str(r.get("log", "")),
                                                                                    root=out) is not None]
        bad = [k for k, ok in (("reason", res.get("reason") == "ok"), ("entry", bool(str(res.get("entry", "")).strip())),
                               ("HOWTO.md", it.p("S4c", "HOWTO.md").exists()), ("reward", rec.get("reward") == 1),
                               ("receipt", bool(ok_runs))) if not ok]
        if bad:
            res = dict(res, runnable=False, reason="invalid_result", invalid=bad)
        else:
            it.p("S4c", "run.log").write_text(codeagent.safe_read(out / str(ok_runs[-1]["log"]), root=out) or "")
    elif isinstance(res, dict) and not isinstance(res.get("runnable"), bool):
        res = dict(res, runnable=False, reason="invalid_result", invalid=["runnable"])
    write_json(it.p("S4c_runnability.json"), dict(job=rec.get("job_name"), outcome=rec.get("outcome"),
                                                  exception=rec.get("exception"), cost_usd=rec.get("cost_usd"),
                                                  minutes=rec.get("minutes"), result=res))
    if not isinstance(res, dict) or (rec.get("outcome") == "agent_error" and res.get("reason") == "invalid_result"):
        # No verdict, or the agent failed (API error, timeout) before it could back its verdict with evidence.
        act, why = _retry_agent_infra(P, it, "S4c", rec)
        if act == "park":
            return it.finish("parked", why)
        if act == "retry":
            return it.goto("S4c", "runnability job infra failure: retry")
        if not isinstance(res, dict):
            return it.finish("dropped", "code_runnability_agent_failed")
    if res.get("runnable") is not True:
        why = str(res.get("reason") or "other")
        reason = {"needs_gpu": "code_needs_gpu", "needs_commercial_license": "code_needs_commercial_license",
                  "needs_mathematica": "code_needs_commercial_license",
                  "matlab_octave_incompatible": "code_matlab_octave_incompatible"}.get(why, "code_not_runnable")
        return it.finish("dropped", reason, detail=why)
    it.advance("S4c", "S5G" if codegate.ccfg(P.cfg).get("authoring", "agent") == "agent" else "S5",
               language=res.get("language"), seconds=res.get("seconds"))


def _launch_failed(it, kind):
    """A launch that failed: paused for the operator; in a hardening round the hardened version is given up instead
    (dropped -> harden_revert puts the previous version back; an infra revert does not use up a round, and
    harden_max_attempts bounds the retries)."""
    if kind == "S5G" and it.state.get("harden_active"):
        return it.finish("dropped", f"code:{kind}:launch_failed")
    it.finish("paused", f"code:{kind}:launch_failed", paused_stage=kind)
    return None


def s5g(P, it):
    """Serialised per item (`run` and `calib watch` may both step it): the job number, inputs, launch and the
    acceptance of the result happen under the item lock, on a freshly loaded state."""
    from .stages_b import item_lock
    with item_lock(it):
        it.load()
        if it.state.get("stage") != "S5G" or it.state.get("status") not in ("active", "waiting"):
            return None
        if _observe_remote_agent(P, it, "S5G"):
            return None
        return _s5g(P, it)


def _s5g(P, it):
    """Task authoring by a Claude Code agent on the build host (replaces S5 + S6A; same artefacts).  With a
    weakness profile (difficulty.profile_path) the agent also targets 1-3 of the student's weak points that the
    paper really supports and records them in _spec/difficulty_design.json; in a hardening round (difficulty.py)
    it gets the previous version and the student's passing-run summary and makes the task harder."""
    from . import codeagent, difficulty
    from .prompts_code import (LEGACY_AUTHOR, S5G_AUTHOR, S5G_DIFFICULTY, S5G_HARDEN, S5G_HARDEN_PREFERRED,
                               S5G_PREFERRED, S5G_VARIANT)
    c = codegate.ccfg(P.cfg)
    d = difficulty.dcfg(P.cfg)
    tmo = float(c.get("authoring_timeout_s", 7200))
    lo, hi = P.cfg["spec"].get("expert_hours", [2, 6])
    try:
        prof = difficulty.load_profile(P.cfg)
    except (OSError, ValueError) as e:
        return it.finish("paused", f"code:config:difficulty_profile:{type(e).__name__}", paused_stage="S5G")
    hround = int(it.state.get("harden_round", 0)) if it.state.get("harden_active") else 0
    hkind = difficulty.round_kind(it) if hround else None
    images = taskimage.author_images(P.cfg, it.state)
    prompt = S5G_AUTHOR if images else LEGACY_AUTHOR
    instruction = prompt.format(item=it.item_id, minutes=int(tmo // 60), forms=P.cfg["spec"]["allowed_forms"],
                                    hours_lo=lo, hours_hi=hi)
    if images:
        catalog = taskimage.base_catalog(taskimage.catalog_path(P.cfg))[:10]
        reuse = [dict(base=r["base"], tasks=len(r["tasks"]), verified_python=r["python"], size=r["size"])
                 for r in catalog]
        instruction += (f"\nPublic data cap: {P.cfg['gen']['max_public_mb']} MB. "
                        f"Image size cap: {P.cfg['taskimage']['max_gb']} GB. "
                        f"Previously verified bases (measurements from a task using each base): {json.dumps(reuse)}\n")
        if it.state.get("image_retry_used"):
            instruction += "Read /work/previous_attempt.json and resolve its image finding.\n"
    need = codeagent.agent_output_files("S5G", with_profile=bool(prof))
    if prof:
        pref = ", ".join(prof.get("preferred") or [])
        instruction += S5G_DIFFICULTY.format(profile_id=prof["id"],
                                             preferred=S5G_PREFERRED.format(ids=pref) if pref else "")
    if hround:
        pref = ", ".join((prof or {}).get("preferred") or [])
        # the round as counted against its own depth (an infra-reverted attempt does not count)
        if hkind == difficulty.KIND_VARIANT:
            instruction += S5G_VARIANT.format(round=max(1, difficulty.rounds_used_kind(it, hkind)),
                                              max_rounds=difficulty.max_rounds(P.cfg, it, hkind),
                                              preferred=S5G_HARDEN_PREFERRED.format(ids=pref) if pref else "")
        else:
            instruction += S5G_HARDEN.format(round=max(1, difficulty.rounds_used(it)),
                                             max_rounds=int(d.get("harden_max_rounds", 2)),
                                             preferred=S5G_HARDEN_PREFERRED.format(ids=pref) if pref else "")
    rnd = None
    if hround:
        ht, hu = d.get("harden_authoring_timeout_s"), d.get("harden_authoring_max_usd")
        base = it.state.get("harden_base") or {}
        if base.get("round") != hround:
            from .stages_b import finish_job_stage
            return finish_job_stage(P, it, "parked", f"harden:no_round_base:{hround}")
        rnd = dict(round=hround, base=base, timeout_s=float(tmo if ht is None else ht),
                   max_usd=c.get("authoring_max_usd", 20.0) if hu is None else hu)
    rec = _agent_stage(P, it, "S5G", instruction, need, tmo, c.get("authoring_max_usd", 20.0),
                       float(c.get("authoring_est_usd", 15.0)), with_contract=True, rnd=rnd)
    if rec is None:
        return
    version = rec.get("prompt_version", "cg-v3-2026-11-12")  # jobs already launched before the change
    # A resumed launcher may have authored its result under the previous flag setting.
    images = image_prompt(version) or taskimage.is_task(it.state)
    # Both PRs briefly used cg-v5. The staged contract identifies in-flight image-authoring jobs.
    if not images and version == "cg-v5-2026-11-24":
        contract = it.p("S5G/work/contract/S5_SPEC.txt")
        images = contract.is_file() and "use the authored task image" in contract.read_text()
    out = codeagent.deliverable(rec)
    if images and out and (out / "task/environment/Dockerfile").is_file():
        need.append("report.md")
    miss = [f for f in need if not (out and codeagent.safe_read(out / f, root=Path(rec["trial"])) is not None)]
    write_json(it.p("S5G_result.json"), dict(job=rec.get("job_name"), outcome=rec.get("outcome"),
                                             exception=rec.get("exception"), cost_usd=rec.get("cost_usd"),
                                             cost_kind=rec.get("cost_kind"), minutes=rec.get("minutes"),
                                             missing=miss, prompt_version=version,
                                             difficulty_profile=difficulty.provenance(prof), harden_round=hround,
                                             harden_kind=hkind))
    if miss:
        act, why = _retry_agent_infra(P, it, "S5G", rec)
        if act == "park":
            return it.finish("parked", why)
        if act == "retry":
            return it.goto("S5G", "authoring job infra failure: retry")
        return it.finish("dropped", "s5g_missing_files", missing=miss[:5])
    try:
        spec = json.loads(codeagent.safe_read(out / "_spec" / "spec.json", root=Path(rec["trial"])) or "")
    except (OSError, ValueError) as e:
        return it.finish("dropped", f"s5g_spec_unparseable:{type(e).__name__}")
    env = out / "task/environment"
    authored = images and (env / "Dockerfile").exists()
    if taskimage.is_task(it.state) and not authored:
        return _image_failure(P, it, taskimage.lint_failure(["existing task image requires an authored Dockerfile"]))
    if isinstance(spec, dict):
        spec.pop("image_kind", None)
        if images:
            if isinstance(spec.get("image"), dict):
                spec["image"].pop("base_resolved_from", None)
            spec["image_kind"] = "task" if authored else "legacy"
        spec["prompt_version"] = version
    errs = validate_spec(spec, P.cfg)
    write_json(it.p("_spec", "spec_errors.json"), errs)
    if errs:
        return it.finish("dropped", "spec_invalid")
    design = None
    if prof:
        try:
            design = json.loads(codeagent.safe_read(out / "_spec" / difficulty.DESIGN_FILE,
                                                    root=Path(rec["trial"])) or "")
        except ValueError:
            design = None
        derrs = difficulty.validate_design(design, prof, spec, hround)
        write_json(it.p("_spec", "difficulty_errors.json"), derrs)
        if derrs:
            write_json(it.p("S6_failure.json"), dict(why="difficulty_design_invalid", info=dict(errors=derrs)))
            P.store.event("difficulty_design_invalid", item=it.item_id, errors=derrs[:5], harden_round=hround)
            _agent_reauthor(P, it, "difficulty_design_invalid", force=True)
            return
    if images and (env.exists() or env.is_symlink()):
        if not env.is_dir() or not codeagent._inside(Path(rec["trial"]), env):
            return it.finish("dropped", "s5g_unsafe_files", refused=["environment"])
        # Check names and bounds before copying any environment bytes from the trial.
        problems = taskimage.context_errors(env) if authored else []
        if problems:
            return _image_failure(P, it, taskimage.lint_failure(problems))
    task = it.p("task")
    if task.exists():
        shutil.rmtree(task)
    refused = codeagent.safe_copytree(out / "task", task)
    if refused:
        shutil.rmtree(task, ignore_errors=True)
        return it.finish("dropped", "s5g_unsafe_files", refused=refused[:10])
    if not authored and (task / "environment").is_dir():
        shutil.rmtree(task / "environment")
    for key in taskimage.STATE_FIELDS:
        it.state.pop(key, None)
    if images:
        it.state["image_kind"] = spec["image_kind"]
    # A new version invalidates the image receipt, but keeps earlier build diagnostics.
    receipt = it.p("S6I_image.json")
    history = read_json(receipt, {}).get("history", [])
    if history:
        write_json(receipt, {"history": history})
    else:
        receipt.unlink(missing_ok=True)
    # S6C products (tests with the thresholds, exploit baselines) are always rebuilt for a new authored version:
    # an agent that started from an earlier version must not carry its thresholds into the verifier
    if (task / "tests" / "test_outputs.py").exists():
        (task / "tests" / "test_outputs.py").unlink()
    for x in (task / "baselines").glob("exploit_*") if (task / "baselines").is_dir() else []:
        x.unlink()
    it.state["regen_6c"] = True
    try:
        author = json.loads(codeagent.safe_read(out / "_spec" / "author_code.json", root=Path(rec["trial"])) or "")
    except ValueError:
        return it.finish("dropped", "s5g_author_code_unparseable")
    s3c = read_json(it.p("S3c_code.json"), {}) or {}
    author.update(pin=s3c.get("pin"), repo_url=s3c.get("primary_url"), license=s3c.get("license"))
    write_json(it.p("_spec", "author_code.json"), author)
    write_json(it.p("_spec", "spec.json"), spec)
    write_json(it.p("_spec", "depgraph.json"), spec["depgraph"])
    if design is not None:
        if hround:
            hr = read_json(it.p("harden", f"round-{hround}", "record.json"), {}) or {}
            design.setdefault("hardening", {})
            design["hardening"].update(round=hround, prev_task_sha=hr.get("prev_task_sha"))
        write_json(it.p("_spec", difficulty.DESIGN_FILE), design)
    elif it.p("_spec", difficulty.DESIGN_FILE).exists():
        it.p("_spec", difficulty.DESIGN_FILE).unlink()          # never keep a design of an earlier version
    write_json(it.p("_spec", "coverage.json"), dict(form=spec.get("form"), domain=spec.get("domain"),
                                                    field=spec.get("field"), competencies=spec.get("competencies")))
    c1 = read_json(it.p("S1_candidate.json"), {}) or {}
    write_json(it.p("_spec", "provenance.json"), dict(
        doi=c1.get("doi"), title=c1.get("title"), target=c1.get("_target"), prompt_version=version,
        authoring="agent", agent_job=rec.get("job_name"), fulltext=read_json(it.p("S3_fulltext.json")),
        fulltext_sha=hashlib.sha256(it.p("fulltext.txt").read_bytes()).hexdigest()[:16],
        author_code=dict(repo=s3c.get("primary_url"), pin=s3c.get("pin"), license=s3c.get("license")),
        difficulty_profile=difficulty.provenance(prof),
        difficulty_levers=[lv.get("weakness_id") for lv in (design or {}).get("levers") or [] if isinstance(lv, dict)],
        harden_round=hround))
    rep = codeagent.safe_read(out / "report.md", root=Path(rec["trial"]))
    if rep is not None:
        it.p("S5G", "report.md").write_text(rep)
    if hround:
        from .stages_b import harden_authored
        harden_authored(P, it)
    it.advance("S5G", "S6I" if authored else "S6A_run", form=spec.get("form"),
               cost_usd=rec.get("cost_usd"), minutes=rec.get("minutes"))


def _record_image_result(it, result):
    receipt = it.p("S6I_image.json")
    previous = read_json(receipt, {})
    history = previous.get("history", [])
    if not history and previous.get("failed_gate"):  # pre-history receipt, including legacy drops
        history.append(dict(gate=previous["failed_gate"], error=previous.get("error")))
    if not result["ok"]:
        history.append(dict(gate=result["failed_gate"], error=result["error"], ts=time.time()))
    write_json(receipt, dict(result, history=history))


def image_retry_due(it):
    st = it.state
    if not (st["stage"] == "S6I" and st["status"] == "waiting" and st.get("reason", "").startswith("S6I:infra:")):
        return False
    retry_at = next((h.get("retry_at") for h in reversed(st.get("history") or []) if "retry_at" in h), None)
    return retry_at is None or retry_at <= time.time()


def _image_failure(P, it, result):
    _record_image_result(it, result)
    gate = result["failed_gate"]
    if taskimage.infra_failure(result):
        attempts = it.state.setdefault("attempts", {})
        n = attempts["S6I:infra"] = attempts.get("S6I:infra", 0) + 1
        if n >= P.cfg["taskimage"]["infra_retries"]:
            return it.finish("parked", "parked:infra")
        delay = (600, 1800, 3600)[min(n, 3) - 1]
        return it.finish("waiting", f"S6I:infra:{gate}", retry_at=time.time() + delay)
    why = "readymade_image" if gate == "readymade_image" else f"image_build_failed:{gate}"
    feedback = dict(why=why, finding=result["error"], action="Choose another base or drop that package."
                    if gate == "readymade_image" else "Correct the image and retest the declared pins.")
    write_json(it.p("S6_failure.json"), feedback)
    if it.state.get("image_retry_used"):
        return it.finish("dropped", why)
    it.state.pop("image_resume_stage", None)
    it.state["image_retry_used"] = True
    _agent_reauthor(P, it, why, force=True)


def s6i(P, it):
    from .stages_b import item_lock
    with item_lock(it):
        it.load()
        if it.state.get("stage") != "S6I" or (it.state["status"] != "active" and not image_retry_due(it)):
            return
        result = taskimage.build(P.cfg, it.dir, state=it.state)
        if not result["ok"]:
            if it.state.get("image_resume_stage"):
                raise taskimage.ImageUnavailable(f"S6I restore failed at {result['failed_gate']}: {result['error']}")
            return _image_failure(P, it, result)
        _record_image_result(it, result)
        taskimage.record_base(taskimage.catalog_path(P.cfg), result["base"], it.item_id, result)
        it.state.update({k: result[k] for k in taskimage.STATE_FIELDS if k in result}, image_kind="task")
        it.p("taskimage-evicted.json").unlink(missing_ok=True)
        it.advance("S6I", it.state.pop("image_resume_stage", "S6A_run"), image_tag=result["image_tag"])


# (the relay-preflight evidence rule lives in calib.relay_preflight_refused, shared with S9 / S9R)


def _failed_before_harbor(rec: dict) -> bool:
    """True only with explicit evidence that the job never reached Harbor: launcher_failed (no Harbor config.json)
    and calib.relay_preflight_refused (the job's OWN launcher log, exit status 3, the relay-preflight refusal).  A
    missing config.json alone is not enough (a failed copy-back of a job that really ran looks the same)."""
    from .calib import relay_preflight_refused
    return rec.get("outcome") == "launcher_failed" and relay_preflight_refused(rec)


def _jobs_used(it, kind: str) -> int:
    """Jobs of this stage that count against code.authoring_max_jobs: every job number handed out except those
    known never to have started (relay preflight refused, tmux failed)."""
    return int((it.state.get("attempts") or {}).get(f"{kind}:job", 0)) - len(set(_nostart(it, kind)))


def _admissions_path(P, kind: str) -> Path:
    return P.store.root / f"{kind.lower()}-admissions.jsonl"


@contextlib.contextmanager
def _agent_slot_lock(P):
    import fcntl
    with open(P.store.root / ".agent-slots.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _admissions(P, kind: str) -> list[dict]:
    path = _admissions_path(P, kind)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _cst_day(ts=None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime((time.time() if ts is None else ts) + 8 * 3600))


def _admission_status(why):
    if why.startswith("diversity:"):
        return "parked"
    return "paused" if why.startswith("budget:") else "waiting"


def _agent_admission(P, it, kind: str) -> str | None:
    """Production throttles before a new S4c / S5G job (call under _agent_slot_lock): code.max_concurrent_agent_jobs
    (S4c + S5G records of this run in reserved / launched; "queued:" -> the item waits and is polled
    again) and code.daily_new_runnability / daily_new_authoring (papers whose first launch intent was recorded on
    this Asia/Shanghai day; "budget:daily_*" -> paused until the next day's `resume --reason budget:daily`).  A
    paper is recorded before starting the launcher (_agent_admit); later jobs of the same paper (re-authoring, hardening
    rounds) are not counted again."""
    if kind == "S5G":
        from . import diversity
        why = diversity.admission(P, it, kind)
        if why:
            return why
    c = codegate.ccfg(P.cfg)
    # code.agent_job_caps {"S4c": n, ...}: a stage with its own cap counts only its own running jobs against it and
    # is left out of the shared code.max_concurrent_agent_jobs pool (e.g. S4c on another model route)
    own_caps = {k: v for k, v in (c.get("agent_job_caps") or {}).items() if k in ("S4c", "S5G") and v is not None}
    if kind in own_caps:
        cap, kinds = own_caps[kind], (kind,)
    else:
        cap, kinds = c.get("max_concurrent_agent_jobs"), tuple(k for k in ("S4c", "S5G") if k not in own_caps)
    if cap is not None:
        running = 0
        for other in P.store.items():
            if other.item_id == it.item_id:
                continue
            for k in kinds:
                r = read_json(other.p(k, "run.json"), None) or {}
                if r.get("state") in ("reserved", "launched", "remote_unknown"):
                    running += 1
        if running >= int(cap):
            return f"{kind}:queued: agent slots full ({running}/{cap})"
    return _daily_paper_cap(P, it, kind, _cst_day())


def _daily_paper_cap(P, it, kind, day):
    lim = codegate.ccfg(P.cfg).get({"S4c": "daily_new_runnability", "S5G": "daily_new_authoring"}[kind])
    if lim is None:
        return None
    _backfill_admissions(P, kind)
    rows = _admissions(P, kind)
    if any(r.get("item") == it.item_id for r in rows):
        return None
    today = [r for r in rows if r.get("day") == day]
    if len(today) >= int(lim):
        return f"budget:daily_{kind.lower()}_papers ({len(today)}/{lim} on {day})"
    return None


def _backfill_admissions(P, kind: str) -> None:
    """Papers whose job of `kind` started (launch_attempted, state launched / finished) but whose admission line is
    missing (crash in between): record them on the day of their launch."""
    have = {r.get("item") for r in _admissions(P, kind)}
    for other in P.store.items():
        if other.item_id in have:
            continue
        r = read_json(other.p(kind, "run.json"), None) or {}
        if r.get("launch_attempted") and r.get("state") in ("launched", "finished") and not r.get("not_started"):
            day = _cst_day(float(r["launch_attempted"]))
            with _admissions_path(P, kind).open("a") as f:
                f.write(json.dumps(dict(day=day, item=other.item_id, ts=time.time(), backfilled=True)) + "\n")


def _withdraw_admission(P, it, kind: str, *, job=None, why="relay_preflight_refused") -> None:
    """Remove this paper's admission line of `kind` (call under _agent_slot_lock): its only job was refused by the
    relay preflight, so no job of the paper started and it must not use up a daily_new_* slot."""
    path = _admissions_path(P, kind)
    rows = _admissions(P, kind)
    keep = [r for r in rows if r.get("item") != it.item_id or (job and r.get("job") != job)]
    if len(keep) == len(rows):
        return
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
    tmp.replace(path)
    P.store.event(f"{kind.lower()}_admission_withdrawn", item=it.item_id, why=why)


def _agent_admit(P, it, kind: str) -> None:
    """Record that this paper's first job of `kind` started (idempotent; call under _agent_slot_lock).  The day is
    the Asia/Shanghai day of the job's launch (launch_attempted), whenever the record is written."""
    if any(r.get("item") == it.item_id for r in _admissions(P, kind)):
        return
    r = read_json(it.p(kind, "run.json"), None) or {}
    ts = float(r.get("launch_attempted") or time.time())
    day = _cst_day(ts)
    with _admissions_path(P, kind).open("a") as f:
        f.write(json.dumps(dict(day=day, item=it.item_id, job=r.get("job_name"), ts=time.time())) + "\n")
    P.store.event(f"{kind.lower()}_admitted", item=it.item_id, day=day)


def _agent_reauthor(P, it, why: str, force: bool = False) -> bool:
    """Code-grounded items authored by the S5G agent go back to S5G (not to the single-call S5 / S6A) when S6B
    asks for a new spec or a regenerated reference; at most code.authoring_max_jobs jobs per item (plus that many
    per hardening round).  force: called from S5G itself (no author_code.json yet)."""
    if not force and (not it.p("_spec", "author_code.json").exists() or
                      codegate.ccfg(P.cfg).get("authoring", "agent") != "agent"):
        return False
    n = _jobs_used(it, "S5G")
    per = int(codegate.ccfg(P.cfg).get("authoring_max_jobs", 2))
    if it.state.get("harden_active"):               # each hardening round gets its own authoring_max_jobs
        n -= int((it.state.get("harden_base") or {}).get("jobs", 0))
    if n >= per:
        it.finish("dropped", f"s5g_jobs_exhausted:{why}")
        return True
    it.state.pop("regen_6a", None)
    it.state["S5G_rerun"] = True
    it.goto("S5G", f"re-author by agent: {why}")
    return True
