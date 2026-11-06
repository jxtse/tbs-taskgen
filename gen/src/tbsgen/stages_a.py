"""Stage handlers S1-S6B.  Each handler gets (P, item) and returns via item.advance / item.goto / item.finish.
P is the Pipeline (cfg, store, ledger, clients).  Handlers write outputs before committing state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import intake, runner
from .models import extract_json, parse_files
from .prompts import PROMPT_VERSION, S5_SPEC, S6A_GEN, SYSTEM
from .spec import set_thresholds, validate_spec
from .state import read_json, write_json


def s1(P, it):
    c = read_json(it.p("S1_candidate.json"))
    if not c:
        return it.finish("dropped", "no_candidate")
    it.advance("S1", "S2")


def s2(P, it):
    c = read_json(it.p("S1_candidate.json"))
    rec = intake.screen(c, P.cfg)
    write_json(it.p("S2_screen.json"), rec)
    if not rec["keep"]:
        return it.finish("dropped", rec["reason"])
    it.advance("S2", "S3", score=rec["score"])


def s3(P, it):
    c = read_json(it.p("S1_candidate.json"))
    txt, rec = intake.fetch_fulltext(c, P.cfg, P.fetcher)
    write_json(it.p("S3_fulltext.json"), rec)
    if txt is None:
        write_json(it.p("S3_queue.json"), dict(doi=c.get("doi"), route="browser_fetch (manual, not wired in MVP)"))
        return it.finish("parked", "fulltext_unavailable")
    it.p("fulltext.txt").write_text(txt)
    it.advance("S3", "S4", source=rec.get("source"), chars=len(txt))


def s4(P, it):
    c = read_json(it.p("S1_candidate.json"))
    rec = intake.probe(c, it.p("fulltext.txt").read_text(), read_json(it.p("S2_screen.json"), {}), P.cfg, P.tbs_index())
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
    user = S5_SPEC.format(doi=c.get("doi"), excerpt=excerpt, probe=json.dumps(probe),
                          forms=P.cfg["spec"]["allowed_forms"])
    if avoid:
        user += f"\nA previous spec used mid_drop {avoid}: it did not separate reference and mid. Choose another."
    errs = ["not run"]
    for _ in range(1 + P.cfg["spec"]["spec_retries"]):
        r = client.chat(SYSTEM, user, stage="S5", item=it.item_id)
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


def s6a(P, it):
    spec = read_json(it.p("_spec", "spec.json"))
    task = it.p("task")
    if not (task / "generator" / "generate.py").exists() or it.state.get("regen_6a"):
        r = P.client("gen").chat(SYSTEM, S6A_GEN.format(spec=json.dumps(spec, indent=1)), stage="S6A", item=it.item_id)
        files = parse_files(r.text)
        need = {"generator/generate.py", "solution/solve.py", "tests/measure.py", "baselines/mid.py",
                "baselines/naive.py"}
        if not need <= set(files):
            write_json(it.p("S6A_error.json"), dict(missing=sorted(need - set(files)), got=sorted(files)))
            return it.finish("dropped", "s6a_missing_files")
        _write_files(task, files, ("generator/", "solution/", "tests/", "baselines/"))
        it.state.pop("regen_6a", None)
        it.save()
    it.advance("S6A", "S6A_run")


def s6a_run(P, it):
    """Measure the reference (and mid/naive if S6C already produced them) on each seed."""
    task = it.p("task")
    work = it.p("runs", "s6")
    meas = {"ref": {}, "mid": {}, "naive": {}, "runs": {}}
    for s in P.cfg["gen"]["seeds"]:
        gen = work / f"gen{s}"
        g = runner.generate(P.cfg, task, s, gen)
        meas["runs"][f"gen{s}"] = {k: g[k] for k in ("rc", "sec", "err", "public_mb")}
        if g["rc"] != 0:
            write_json(it.p("S6A_measure.json"), meas)
            return _to_repair(P, it, "generator_failed", meas)
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
    spec = read_json(it.p("_spec", "spec.json"))
    meas = read_json(it.p("S6A_measure.json"))
    th = set_thresholds(spec, meas, P.cfg)
    write_json(it.p("thresholds.json"), th)
    if th["status"] == "not_separable":
        back = it.state.get("back_to_spec", 0)
        if back < P.cfg["gen"]["back_to_spec_max"]:
            it.state["back_to_spec"] = back + 1
            it.state.setdefault("avoid_mid_drop", []).extend(spec["depgraph"].get("mid_drop", []))
            it.state["regen_6a"] = True
            it.state["regen_6c"] = True
            it.save()
            return it.goto("S5", "mid_not_separable")
        return it.finish("dropped", "mid_not_separable")
    if th["status"] != "ok":
        return _to_repair(P, it, th["status"], th)
    it.advance("S6B", "S6C", separating=th["separating"])


def _to_repair(P, it, why: str, info):
    write_json(it.p("S6_failure.json"), dict(why=why, info=info))
    it.state["repair_from"] = why
    it.save()
    it.goto("S8", why)
