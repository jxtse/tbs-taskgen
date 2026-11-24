"""Platform row construction, token accounting, and strict local checks."""

import json

from tokenizers import Tokenizer

from .core import dumps, loads
from .native import function_schemas, string_content, validate_pairs


def render(source, meta):
    validate_pairs(source["messages"])
    schemas = list(function_schemas(source["tools"]))
    messages = []
    masks = []
    pending = []
    results = {}
    deferred = []

    def add(m, loss=0):
        messages.append(m)
        masks.append(loss)

    def flush():
        if set(pending) != set(results):
            raise ValueError("unanswered_tool_call")
        for tid in pending:
            add(results[tid])
        for m in deferred:
            add(m)
        pending.clear()
        results.clear()
        deferred.clear()

    last = max((i for i, m in enumerate(source["messages"]) if m["role"] == "assistant"), default=-1)
    tail = source["messages"][last+1:]
    dropped = sum(sum(b["type"] == "tool_result" for b in m["content"])
                  + int(any(b["type"] != "tool_result" for b in m["content"])) for m in tail)
    for m in source["messages"][:last+1]:
        role = m["role"]
        if role == "assistant":
            flush()
            content = []
            reasoning = []
            calls = []
            for b in m["content"]:
                kind = b["type"]
                if b.get("hint"):
                    raise ValueError("unremoved_hint")
                if kind == "text":
                    content.append(b["text"])
                elif kind == "thinking":
                    reasoning.append(b["thinking"])
                elif kind == "tool_use":
                    tid, name = b["id"], b["name"]
                    args = json.dumps(b["input"], ensure_ascii=False, allow_nan=False)
                    calls.append(dict(id=tid, type="function", function=dict(name=name, arguments=args)))
                    pending.append(tid)
                elif kind == "image":
                    content.append("[image omitted in export]")
                elif kind != "redacted_thinking":
                    raise ValueError("unsupported_assistant_block")
            wire = {"role": role, "content": "\n".join(content)}
            if reasoning or "codex_reasoning" in source:
                wire["reasoning_content"] = "\n\n".join(reasoning)
            if calls:
                wire["tool_calls"] = calls
            add(wire, m["loss"])
        elif role in ("user", "system"):
            texts = []
            for b in m["content"]:
                if b["type"] == "tool_result":
                    tid = b["tool_use_id"]
                    if tid not in pending or tid in results:
                        raise ValueError("orphan_tool_result")
                    results[tid] = {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": string_content(b["content"]),
                    }
                else:
                    texts.append(string_content([b]))
            if texts:
                entry = {"role": role, "content": "\n\n".join(texts)}
                if pending:
                    deferred.append(entry)
                else:
                    add(entry)
        else:
            raise ValueError("unknown_role")
    return dict(messages=messages, loss_mask=masks, tools=schemas,
                meta={**meta, **{k: source[k] for k in ("codex_reasoning", "reasoning_source", "summary_model", "summary_model_source") if k in source},
                      "render_notes": {"dropped_tail": dropped}})


class Counter:
    def __init__(self, path):
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def counts(self, row):
        def encode(text):
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

        total = 1 + encode(dumps(row["tools"]))
        supervised = 0
        for m, loss in zip(row["messages"], row["loss_mask"], strict=True):
            text = "<|" + m["role"] + "|>"
            if "reasoning_content" in m:
                text += "<think>" + m["reasoning_content"] + "</think>"
            text += m["content"]
            if m.get("tool_calls"):
                text += "<tool_calls>" + dumps(m["tool_calls"]) + "</tool_calls>"
            if "tool_call_id" in m:
                text += m["tool_call_id"]
            n = encode(text) + 1
            total += n
            if loss:
                supervised += n
        return total, supervised


def check(row, *, allow_masked=False):
    """Return one stable reason per failed invariant, including malformed nested data."""
    errors = []
    try:
        dumps(row)  # Also rejects numeric overflow such as 1e999.
        ms, mask, tools = row["messages"], row["loss_mask"], row["tools"]
        if set(row) != {"meta", "messages", "tools", "loss_mask"}:
            errors.append("row_keys")
        if not isinstance(row.get("meta"), dict):
            errors.append("meta")
        if not isinstance(ms, list) or not ms:
            return ["messages"]
        if ms[0]["role"] != "system":
            errors.append("first_role")
        last = ms[-1]
        if (
            last["role"] != "assistant"
            or not isinstance(last.get("content"), str)
            or not (last["content"].strip() or last.get("reasoning_content") or last.get("tool_calls"))
        ):
            errors.append("no_final_answer")
        if not isinstance(mask, list) or len(mask) != len(ms):
            errors.append("mask_length")
        elif 1 not in mask:
            errors.append("no_supervision")
        if not isinstance(tools, list):
            return errors + ["tools_schema"]
        names = set()
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name")
            if (
                set(t) != {"type", "function"}
                or set(fn) != {"name", "description", "parameters"}
                or not isinstance(fn.get("description"), str)
                or not fn.get("parameters")
                or t.get("type") != "function"
                or not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(fn.get("parameters"), dict)
            ):
                errors.append("tools_schema")
            names.add(name)
        seen = set()
        pending = []
        for i, m in enumerate(ms):
            role = m["role"]
            if not isinstance(m.get("content"), str):
                errors.append("content_type")
            if isinstance(mask, list) and i < len(mask):
                v = mask[i]
                if type(v) is not int or v not in (0, 1) or (v and role != "assistant"):
                    errors.append("invalid_loss")
                if v == 0 and role == "assistant" and not allow_masked:
                    errors.append("masked_assistant")
            if pending and role != "tool":
                errors.append("tool_response_missing")
                pending.clear()
            if role == "tool":
                if set(m) != {"role", "tool_call_id", "content"}:
                    errors.append("tool_keys")
                if not pending or pending.pop(0) != m.get("tool_call_id"):
                    errors.append("tool_order")
            elif role == "assistant":
                if set(m) - {"role", "content", "tool_calls", "reasoning_content"}:
                    errors.append("assistant_keys")
                if not isinstance(m.get("reasoning_content", ""), str):
                    errors.append("reasoning_type")
                calls = m.get("tool_calls", [])
                if not isinstance(calls, list) or ("tool_calls" in m and not calls):
                    errors.append("tool_calls")
                    if not isinstance(calls, list):
                        continue
                if not calls and not (
                    str(m.get("content", "")).strip() or str(m.get("reasoning_content", "")).strip()
                ):
                    errors.append("empty_assistant")
                for c in calls:
                    fn = c["function"]
                    if (
                        set(c) != {"id", "type", "function"}
                        or c["type"] != "function"
                        or set(fn) != {"name", "arguments"}
                    ):
                        errors.append("tool_call_schema")
                    tid = c["id"]
                    if not isinstance(tid, str) or not tid or tid in seen:
                        errors.append("tool_id")
                    seen.add(tid)
                    pending.append(tid)
                    if fn["name"] not in names:
                        errors.append("missing_tool_schema")
                    try:
                        if not isinstance(fn["arguments"], str) or not isinstance(
                            loads(fn["arguments"]), dict
                        ):
                            errors.append("arguments")
                    except ValueError:
                        errors.append("arguments")
            elif role in ("system", "user"):
                if set(m) != {"role", "content"}:
                    errors.append("message_keys")
            else:
                errors.append("role")
        # A rendered final call may lose its matched result only at the recorded tail cut.
        notes = row["meta"].get("render_notes", {}) if isinstance(row["meta"], dict) else {}
        if pending and "dropped_tail" in notes and len(pending) > notes["dropped_tail"]:
            errors.append("tool_response_missing")
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        errors.append("malformed_row")
    return sorted(set(errors))


def source_check(source, row):
    """Verify source blocks against their corresponding role, call ID and output field."""
    expected = render(source["segment"], source["meta"])
    if expected != row:
        return ["provenance"]
    # Independently walk the input blocks. Renderer regressions must not silently erase text.
    fields = [m["content"] for m in row["messages"]]
    thoughts = [m.get("reasoning_content", "") for m in row["messages"] if m["role"] == "assistant"]
    source_messages = source["segment"]["messages"]
    end = max(i for i, m in enumerate(source_messages) if m["role"] == "assistant")
    for m in source_messages[:end+1]:
        for b in m["content"]:
            kind = b["type"]
            if kind == "thinking" and not any(b["thinking"] in s for s in thoughts):
                return ["provenance"]
            if kind == "text" and not any(b["text"] in s for s in fields):
                return ["provenance"]
            if kind == "tool_result" and not any(
                v.get("tool_call_id") == b["tool_use_id"] and string_content(b["content"]) == v["content"]
                for v in row["messages"]
            ):
                return ["provenance"]
    return []


def reference_shape(row, reference):
    """The reference defines allowed field types by role, independent of turn counts."""
    def field_types(message):
        return {(key, type(value)) for key, value in message.items()}

    fields = {}
    for message in reference["messages"]:
        fields.setdefault(message["role"], set()).update(field_types(message))
    if field_types(row) != field_types(reference):
        return ["reference_row_shape"]
    for message in row["messages"]:
        if not field_types(message) <= fields.get(message["role"], set()):
            return ["reference_message_shape"]
    return []
