"""One advisory review before an epoch's first calibration. The original PKG remains available for rollback."""
from __future__ import annotations

import hashlib
import json
import shutil

from . import calib, repair, review
from .runner import format_note_bindings, format_notes, regular_file, tree_hash
from .state import passed_gate_attempt, read_json, write_json


def start(P, it):
    """Called under S9's lock, before a first run intent exists. A saved claim prevents duplicate model calls."""
    try:
        cfg = P.cfg.get("prereview") or {}
        mode = cfg.get("mode", "off")
        context = "review_revision" if it.state.get("review_revisions") else "base"
        if (mode not in ("shadow", "enforce") or it.state.get("harden_round") or it.state.get("variant_round")
                or context not in cfg.get("allowed_contexts", ["base", "review_revision"])):
            return False
        epoch = int(it.state.get("calib_epoch", 0))
        rnd = review.next_round(it.dir, it.state)
        key = f"e{epoch}-r{rnd}"
        rec = _check_once(P, it, key, mode, epoch, rnd, context)
    except Exception as exc:
        P.store.event("prereview_error", item=it.item_id, error=type(exc).__name__)
        return False
    if isinstance(rec, dict) and rec.get("status") == "ready" and mode == "enforce":
        it.state["prereview_pending"] = key
        it.goto("S8", "statement revision")
        return True
    return False


def _check_once(P, it, key, mode, epoch, rnd, context):
    path = it.p("prereview", key + ".json")
    rec = read_json(path)
    if not isinstance(rec, dict) and path.exists():
        raise ValueError("invalid pre-review record")
    if rec is None:
        rec = dict(mode=mode, epoch=epoch, round=rnd, context=context, packet_hash=None, raw=[], validated=[],
                   discarded=0, usd=0.0, latency_s=0.0, revision_usd=0.0, revision_applied=False, status="checking")
        write_json(path, rec)
        try:
            from .prereview import run_prereview
            rec.update(run_prereview(it.p("pkg"), lambda: P.client("prereview"), item=it.item_id,
                                     generator_dir=it.p("task", "generator")))
        except Exception as exc:
            rec["error"] = type(exc).__name__
        rec["status"] = "ready" if mode == "enforce" and rec["validated"] and not rec.get("error") else "done"
        write_json(path, rec)
        P.store.event("prereview", item=it.item_id, epoch=epoch, round=rnd, mode=mode,
                      validated=len(rec["validated"]), error=rec.get("error"))
    elif isinstance(rec, dict) and rec.get("status") == "checking":
        rec.update(status="done", error="interrupted pre-review")
        write_json(path, rec)
    return rec


def _record(it):
    return it.p("prereview", it.state["prereview_pending"] + ".json")


def candidate_dir(it):
    # PKG checks discover item state and image inventory from the package parent.
    return it.p(".prereview-candidate")


def _finish(it, rec, *, applied=False, reason=None):
    rec.update(status="done", revision_applied=applied, revision_error=reason)
    write_json(_record(it), rec)
    it.state.pop("prereview_pending")
    if applied:
        # Even the first epoch can now have an out-of-date prebuilt calibration task.
        it.state["require_prepared_receipt"] = True
        it.advance("PKG", "S9")
    else:
        it.goto("S9", "continue calibration")


def _source(it, rel):
    # Keep generated-document overrides outside the authored image build context.
    root = it.p("task") if rel == "instruction.md" else it.p("_spec/format_notes")
    return root / rel


def reject(it, reason):
    """Restore the source documents and original passing gate binding. PKG only copies verified edits on success."""
    rec = read_json(_record(it))
    if rec["status"] in {"done", "committing"}:
        return resume_edit(it)
    if rec["status"] != "rejecting":
        rec.update(status="rejecting", revision_error=reason)
        write_json(_record(it), rec)
    _rollback(it, rec)


def _rollback(it, rec):
    for rel, original in rec.get("before", {}).items():
        dest = _source(it, rel)
        if original is None:
            dest.unlink(missing_ok=True)
        else:
            dest.write_bytes(original.encode())
    if "notes_before" in rec:
        old = rec["notes_before"]
        path = it.p("_spec", "format_notes.json")
        if old is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(old.encode())
    if "gate_attempt" in rec:
        it.state["gates_attempt"] = rec["gate_attempt"]
    if "difficulty_before" in rec:
        # S7 writes this unversioned check, and the Opus bundle reads it directly.
        path = it.p("gates", "difficulty.json")
        if rec["difficulty_before"] is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(rec["difficulty_before"].encode())
    shutil.rmtree(candidate_dir(it), ignore_errors=True)
    _finish(it, rec, reason=rec["revision_error"])


def revise(P, it, request_patch):
    path = _record(it)
    rec = read_json(path)
    if rec["status"] == "requesting":
        return reject(it, "interrupted S8 request")
    pkg, task = it.p("pkg"), it.p("task")
    if rec["status"] == "ready":
        rec["status"] = "requesting"
        write_json(path, rec)
        try:
            allowed = set()
            docs = []
            remaining = 60000
            for rel in ["instruction.md", *format_notes(pkg)]:
                source = task if rel == "instruction.md" else pkg
                if not regular_file(source, rel) or (source / rel).stat().st_size > remaining:
                    continue
                body = (source / rel).read_text()
                docs.append(f"=== FILE: {rel} ===\n" + body)
                allowed.add(rel)
                remaining -= len(body.encode())
            if not docs:
                return reject(it, "no eligible document within the S8 prompt budget")
            prompt = ("Revise only these task documents to address the verified findings. Preserve every other "
                      "condition. Return only changed whole files in === FILE: path === blocks followed by "
                      "=== END ===. Do not include review commentary.\n" + json.dumps(rec["validated"]) +
                      "\n" + "\n".join(docs))

            def record_cost(usd):
                rec["revision_usd"] = usd
                write_json(path, rec)

            patch = request_patch(P, it, prompt, record_cost=record_cost)
            if not patch or not set(patch) <= allowed:
                return reject(it, "empty patch or file outside the allowed documents")
            for rel in patch:
                dest = _source(it, rel)
                if (not regular_file(pkg, rel) or dest.resolve() != it.dir.resolve() / dest.relative_to(it.dir)
                        or (dest.exists() and dest.stat().st_nlink > 1) or (pkg / rel).stat().st_nlink > 1):
                    return reject(it, "linked document")
            spec = read_json(it.p("_spec", "spec.json"))
            chk = repair.review(pkg, patch, spec["depgraph"])
            if not chk["ok"]:
                return reject(it, "; ".join(chk["reasons"]))
            rec.update(status="editing", patch=patch, gate_attempt=passed_gate_attempt(it.state),
                       threshold_hash=hashlib.sha256(it.p("thresholds.json").read_bytes()).hexdigest(),
                       before={rel: _source(it, rel).read_bytes().decode() if _source(it, rel).is_file() else None
                               for rel in patch})
            difficulty_report = it.p("gates", "difficulty.json")
            rec["difficulty_before"] = difficulty_report.read_bytes().decode() if difficulty_report.is_file() else None
            notes = set(patch) - {"instruction.md"}
            if notes:
                binding_path = it.p("_spec", "format_notes.json")
                old = binding_path.read_bytes().decode() if binding_path.exists() else None
                bindings = format_note_bindings(task)
                for rel in notes:
                    if rel not in bindings:
                        bindings[rel] = hashlib.sha256((pkg / rel).read_bytes()).hexdigest()
                rec.update(notes_before=old, notes_after={"files": bindings,
                                                         "generator": tree_hash(task / "generator")})
            write_json(path, rec)
        except Exception as exc:
            if getattr(exc, "usd", 0):
                rec["revision_usd"] = exc.usd
                write_json(path, rec)
            return reject(it, f"S8: {getattr(exc, 'cls', type(exc).__name__)}")
    for rel, body in rec["patch"].items():
        if rel == "instruction.md":
            repair.apply(task, {rel: body})
        else:
            repair.apply(it.p("_spec/format_notes"), {rel: body})
    if "notes_after" in rec:
        write_json(it.p("_spec", "format_notes.json"), rec["notes_after"])
    it.goto("S7", "validate revised documents")


def finish_package(it, out, validation):
    rec = read_json(_record(it))
    pkg = it.p("pkg")
    patch = rec["patch"]
    if not validation["ok"]:
        return reject(it, "PKG: " + str(validation.get("errors"))[:500])
    # Provenance includes the build timestamp and cumulative ledger spend. Keep the original bytes.
    skip = {"provenance.json", *patch}
    if (hashlib.sha256(it.p("thresholds.json").read_bytes()).hexdigest() != rec["threshold_hash"]
            or calib.pkg_sha(out, exclude=skip) != calib.pkg_sha(pkg, exclude=skip)):
        return reject(it, "protected files changed during rebuild")
    if any(not regular_file(out, rel) or not regular_file(pkg, rel) for rel in patch):
        return reject(it, "rebuild omitted a revised document")
    approved = {rel: (out / rel).read_bytes().decode() for rel in patch}
    if any(approved[rel] != body for rel, body in patch.items() if rel != "instruction.md"):
        return reject(it, "generated format notes changed since the reviewed package")
    if all((pkg / rel).read_bytes() == body.encode() for rel, body in approved.items()):
        return reject(it, "no document changed")
    original_check = read_json(it.p("PKG_check.json"), {})
    if "verifier_timeout_fix" in original_check:
        validation = dict(validation, verifier_timeout_fix=original_check["verifier_timeout_fix"])
    rec.update(status="committing", approved=approved, package_check=validation)
    write_json(_record(it), rec)
    resume_edit(it)


def resume_edit(it):
    """Complete a saved commit or rollback before the interrupted stage runs again."""
    rec = read_json(_record(it))
    if rec["status"] == "done":
        _finish(it, rec, applied=rec["revision_applied"], reason=rec.get("revision_error"))
        return True
    if rec["status"] == "rejecting":
        _rollback(it, rec)
        return True
    if rec["status"] != "committing":
        return False
    for rel, body in rec["approved"].items():
        it.p("pkg", rel).write_bytes(body.encode())
    write_json(it.p("PKG_check.json"), rec["package_check"])
    shutil.rmtree(candidate_dir(it), ignore_errors=True)
    _finish(it, rec, applied=True)
    return True
