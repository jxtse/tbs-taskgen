from itertools import zip_longest
from math import ceil
from pathlib import Path

from .profile import configured_profiles, required_profile
from .core import digest, dumps, file_hash, loads, read
from .select import image_counts
from .wire import Counter, check, export_modes, image_policy_errors, render, source_check, reference_shape


def distribution(counts):
    ordered = sorted(counts)
    return {
        k: (ordered[ceil(len(ordered) * p) - 1] if ordered else 0)
        for k, p in (("p50", 0.5), ("p90", 0.9), ("max", 1))
    }


def export(ctx):
    cfg = ctx.cfg["export"]
    path = ctx.path(cfg["tokenizer"])
    target, images = export_modes(cfg)
    profiles = configured_profiles(ctx)

    def build():
        counts = []
        screened = list(ctx.rows("screen"))
        decisions = [dict(trial_id=r["trial_id"], task_id=r["task_id"], segment=i,
                          keep=False, reason="image_content")
                     for r in screened for i in r.get("image_dropped_segments", [])]
        token_counter = Counter(path)
        with (
            (ctx.work / "train.jsonl").open("w", encoding="utf-8") as output,
            (ctx.work / "train.sources.jsonl").open("w", encoding="utf-8") as sources,
        ):
            for r in ctx.rows("selection"):
                profile = profiles.get(r["harness"])
                if r["harness"] == "claude-code":
                    required_profile(profile, r["harness"], r.get("agent_version"))
                if r.get("wire_context_hash") != digest([profile, target, cfg.get("codex_reasoning", "drop"), images]):
                    raise ValueError("wire context changed: rerun screen and select")
                dropped = set(r.get("image_dropped_segments", []))
                for i, segment in enumerate(read(ctx.out / "segments" / r["source"])):
                    if i in dropped:
                        continue
                    meta = {k: r[k] for k in ("task_id", "trial_id", "harness", "tier", "domain")}
                    meta.update(segment=i, compaction=segment.get("compaction", False))
                    row = render(segment, meta)
                    errors = check(row, allow_masked=target == "loss_mask")
                    errors.extend(image_policy_errors(segment, images))
                    if errors:
                        raise ValueError(f"invalid screened row: {errors}")
                    tokens, _ = token_counter.counts(row)
                    keep = tokens <= cfg.get("max_length", 194560)
                    decisions.append(
                        dict(**meta, tokens=tokens, keep=keep, reason="exported" if keep else "max_length")
                    )
                    if keep:
                        counts.append(tokens)
                        output.write(dumps(row, sort_keys=False) + "\n")
                        sources.write(dumps({"meta": meta, "segment": segment}, sort_keys=False) + "\n")
        ctx.emit("export.ledger", decisions)
        return {
            "count": len(counts),
            "length": distribution(counts),
            "image_content": image_counts(screened),
            "dropped": len(decisions) - len(counts),
            "train_on_what": target,
            "note": "Managed all_assistant_messages supervises every assistant message and ignores loss_mask.",
        }

    return ctx.stage("export", [ctx.dependency("select"), file_hash(path), cfg, digest(profiles)], build)


def validate(ctx, file, allow_masked=False, reference=None, *, allow_legacy_provenance=False):
    cfg = ctx.cfg["export"]
    target, images = export_modes(cfg)
    tokenizer = ctx.path(cfg["tokenizer"])
    file = Path(file).absolute()
    exemplar = loads(reference.read_bytes()) if reference else None
    if reference and (errors := check(exemplar, allow_masked=True)):
        raise ValueError(f"invalid platform reference: {errors}")
    sources = file.with_suffix(".sources.jsonl")

    def build():
        token_counter = Counter(tokenizer)
        failed = warned = count = masked_assistants = max_tokens = 0
        required_fields = {"messages", "role", "content"}
        with (
            file.open("rb") as stream,
            (ctx.work / "validate.ledger.jsonl").open("w", encoding="utf-8") as ledger,
        ):
            source_stream = sources.open("rb") if sources.exists() else None
            try:
                for count, (raw, src) in enumerate(zip_longest(stream, source_stream or ()), 1):
                    errors = []
                    warnings = []
                    n = None
                    try:
                        row = loads(raw.decode("utf-8")) if raw is not None else None
                        errors = check(
                            row,
                            allow_masked=allow_masked
                            or target == "loss_mask",
                        )
                        if not errors and reference:
                            errors.extend(reference_shape(row, exemplar))
                        if not errors:
                            n, _ = token_counter.counts(row)
                            max_tokens = max(max_tokens, n)
                            if row["tools"]:
                                required_fields.add("tools")
                            for m in row["messages"]:
                                required_fields.update(k for k in ("reasoning_content", "tool_calls", "tool_call_id")
                                                       if m.get(k))
                            masked_assistants += sum(
                                m["role"] == "assistant" and loss == 0
                                for m, loss in zip(row["messages"], row["loss_mask"], strict=True)
                            )
                            if n > cfg.get("max_length", 194560):
                                errors.append("max_length")
                        if src is None:
                            errors.append("source_missing")
                        elif not errors:
                            issues = source_check(loads(src.decode("utf-8")), row, images)
                            if allow_legacy_provenance and issues == ["unverifiable_response_boundaries"]:
                                warnings = issues
                            else:
                                errors.extend(issues)
                    except UnicodeDecodeError:
                        errors.append("utf8")
                    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
                        errors.append("invalid_record: " + str(exc))
                    failed += bool(errors)
                    warned += bool(warnings)
                    reason = "rejected" if errors else "warning" if warnings else "valid"
                    ledger.write(dumps(dict(row=count, tokens=n, errors=errors, warnings=warnings, reason=reason)) + "\n")
                if not count:
                    failed = 1
                    ledger.write(dumps(dict(row=0, errors=["empty_dataset"], reason="rejected")) + "\n")
            finally:
                if source_stream:
                    source_stream.close()
        if masked_assistants:
            required_fields.add("loss_mask")
        return {
            "required_fields": sorted(required_fields),
            "count": count, "failed": failed, "warned": warned, "valid": not failed,
            "max_tokens": max_tokens, "masked_assistants": masked_assistants,
        }

    return ctx.stage(
        "validate",
        [
            str(file),
            file_hash(file),
            file_hash(sources) if sources.is_file() else None,
            file_hash(tokenizer),
            cfg,
            allow_masked,
            allow_legacy_provenance,
            file_hash(reference) if reference else None,
        ],
        build,
    )
