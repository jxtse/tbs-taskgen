"""Reconstruct the active Claude Code context from its native event tree."""

from collections import defaultdict
from copy import deepcopy


def is_boundary(event):
    return event["type"] == "system" and event.get("subtype") == "compact_boundary"


def text_block(text):
    return {"type": "text", "text": text}


def attachment_parts(record):
    for item in record.get("rendered") or []:
        value = item.get("content")
        if isinstance(value, str) and value.strip():
            yield value
        elif isinstance(value, list):
            yield from (b["text"] for b in value if b.get("type") == "text" and b.get("text"))


def unwrap(text):
    if text.startswith("<system-reminder>"):
        text = text.removeprefix("<system-reminder>").removeprefix("\n")
    if text.endswith("</system-reminder>"):
        text = text.removesuffix("</system-reminder>").removesuffix("\n")
    return text


def native_content(record, preserved):
    content = record["message"]["content"]
    if isinstance(content, str):
        return content
    blocks = []
    for b in content:
        kind = b["type"]
        if kind == "text":
            if not b.get("text") and not b.get("hint"):
                continue
            blocks.append(text_block(b["text"]))
        elif kind == "thinking":
            blocks.append(dict(type=kind, thinking=b.get("thinking") or "", signature=b.get("signature", "")))
        elif kind == "tool_use":
            blocks.append(dict(type=kind, id=b["id"], name=b["name"], input=deepcopy(b["input"])))
        else:
            block = deepcopy(b)
            if preserved and kind == "tool_result" and isinstance(block.get("content"), str):
                block["content"] = block["content"].lstrip()
            blocks.append(block)
        if b.get("hint"):
            blocks[-1]["hint"] = b["hint"]
    return blocks


class Turns:
    def __init__(self, compacted):
        self.messages = []
        self.pending = []
        self.assistants = {}
        self.compacted = compacted
        self.from_string = None
        self.early_attributions = []

    def flush(self):
        if self.pending:
            self.messages.append(dict(role="system", content="\n\n".join(self.pending)))
            self.pending.clear()

    def last_user(self):
        last = next((m for m in reversed(self.messages) if m["role"] != "system"), None)
        return last if last and last["role"] == "user" else None

    def attribute(self, raw):
        user = self.last_user()
        if user is None:
            user = next((m for m in self.messages if m["role"] == "user"), None)
            if user is None:
                self.early_attributions.append(raw)
                return
            prepend = True
        else:
            prepend = not self.compacted and not any(
                m["role"] == "assistant" for m in self.messages
            )
        c = user["content"]
        if prepend:
            if isinstance(c, str):
                c = user["content"] = [text_block(c)]
                self.from_string = user
            c.insert(0, text_block(raw + "\n"))
            user["prepended"] = user.get("prepended", 0) + 1
        elif isinstance(c, str):
            user["content"] += "\n\n" + raw
        elif c and c[-1]["type"] == "tool_result":
            result = c[-1]["content"]
            if isinstance(result, str):
                c[-1]["content"] = result.rstrip() + "\n\n" + raw
            else:
                result.append(text_block(raw))
        else:
            c.append(text_block(raw))

    def feed(self, record, preserved=False, historical=False):
        role = record["type"]
        if role == "attachment":
            kind = record.get("attachment", {}).get("type")
            parts = list(attachment_parts(record))
            if parts and kind == "remote_session_change":
                self.attribute("\n".join(parts))
            elif parts and kind not in {"atis_latch", "prompt_snapshot"}:
                self.pending.append("\n".join(map(unwrap, parts)))
            return
        if role not in {"assistant", "user"}:
            return
        content = native_content(record, preserved)
        mid = record["message"].get("id")
        if role == "user" and isinstance(content, list) and any(b["type"] == "tool_result" for b in content):
            self.assistants.clear()
        if role == "assistant" and mid in self.assistants:
            self.assistants[mid]["content"].extend(content)
            return
        self.flush()
        user = self.last_user() if role == "user" else None
        if user and isinstance(content, str) and self.messages[-1] is user and isinstance(user["content"], str):
            user["content"] += "\n\n" + content
            return
        if user and isinstance(content, list):
            if isinstance(user["content"], str):
                user["content"] = [text_block(user["content"] + "\n")]
            elif user is self.from_string:
                user["content"][-1]["text"] += "\n"
                self.from_string = None
            user["content"].extend(content)
            return
        msg = dict(role=role, content=content)
        self.messages.append(msg)
        if role == "assistant":
            msg["loss"] = int(not (preserved or historical))
            if mid and isinstance(content, list):
                self.assistants[mid] = msg
        elif self.early_attributions:
            for raw in self.early_attributions:
                self.attribute(raw)
            self.early_attributions.clear()


class Session:
    def __init__(self, records):
        self.index = {}
        self.children = defaultdict(list)
        self.results = {}
        for r in records:
            uid = r.get("uuid")
            if not uid or uid in self.index or r.get("isSidechain"):
                continue
            self.index[uid] = r
            self.children[r.get("parentUuid")].append(uid)
            c = r.get("message", {}).get("content")
            if r["type"] == "user" and isinstance(c, list):
                for b in c:
                    if b["type"] == "tool_result":
                        self.results.setdefault(b["tool_use_id"], b)
        if not self.index:
            raise ValueError("session has no main-chain UUIDs")
        self.leaf = next((u for u in reversed(self.index) if not self.children[u]), None)
        if self.leaf is None:
            raise ValueError("cyclic parentUuid history")
        self.boundaries = [r for r in self.index.values() if is_boundary(r)]

    def chain(self, leaf):
        path = []
        seen = set()
        while leaf:
            if leaf in seen or leaf not in self.index:
                raise ValueError("broken parentUuid chain")
            seen.add(leaf)
            event = self.index[leaf]
            path.append(event)
            leaf = event.get("parentUuid")
        return path[::-1]

    def summary(self, boundary):
        anchor = boundary["compactMetadata"].get("preservedMessages", {}).get("anchorUuid")
        summary = self.index.get(anchor, {})
        if not summary.get("isCompactSummary"):
            raise ValueError("missing_compaction_summary")
        return summary

    def resume(self, marker):
        matches = [r["uuid"] for r in self.index.values() if r["type"] == "user"
                   and isinstance(r["message"]["content"], str)
                   and marker and r["message"]["content"].startswith(marker)]
        if len(matches) > 1:
            raise ValueError("unresolved_resume_prefix")
        return next(iter(matches), None)

    def after_attachments(self, uid):
        while len(self.children[uid]) == 1:
            child, = self.children[uid]
            if self.index[child]["type"] != "attachment":
                break
            uid = child
        return uid

    def windows(self):
        """Trace logical parents across compactions, omitting abandoned branches."""
        ends = []
        leaf = self.leaf
        seen = set()
        while leaf:
            if leaf in seen:
                raise ValueError("cyclic compaction history")
            seen.add(leaf)
            ends.append(leaf)
            boundary = next((r for r in reversed(self.chain(leaf)) if is_boundary(r)), None)
            leaf = boundary.get("logicalParentUuid") if boundary else None
        return ends[::-1]

    def state(self, leaf, historical=()):
        path = self.chain(leaf)
        start = next((i for i in reversed(range(len(path))) if is_boundary(path[i])), None)
        turns = Turns(start is not None)
        if start is not None:
            boundary = path[start]
            path = path[start + 1:]
            summary = self.summary(boundary)
            pos = next((i for i, r in enumerate(path) if r["uuid"] == summary["uuid"]), None)
            if pos is None:
                raise ValueError("compaction summary outside boundary chain")
            envelope = [r for r in path[:pos] if r.get("attachment", {}).get("type") in {"session_context", "date"}]
            turns.feed(summary)
            for r in envelope:
                turns.feed(r)
            segment = boundary["compactMetadata"].get("preservedSegment") or {}
            skip = {summary["uuid"], *(r["uuid"] for r in envelope)}
            if segment.get("headUuid"):
                preserved = self.chain(segment["tailUuid"])
                head = next((i for i, r in enumerate(preserved) if r["uuid"] == segment["headUuid"]), None)
                if head is None:
                    raise ValueError("invalid preserved segment")
                for r in preserved[head:]:
                    turns.feed(r, preserved=True)
                    skip.add(r["uuid"])
            path = [r for r in path if r["uuid"] not in skip]
        for r in path:
            turns.feed(r, historical=r["uuid"] in historical)
        turns.flush()
        self.repair_results(turns.messages)
        return turns.messages

    def repair_results(self, messages):
        present = {b["tool_use_id"] for m in messages if isinstance(m["content"], list)
                   for b in m["content"] if b["type"] == "tool_result"}
        for assistant, user in zip(messages, messages[1:]):
            if assistant["role"] != "assistant" or user["role"] != "user":
                continue
            c = user["content"]
            if not isinstance(c, list) or not any(b["type"] == "tool_result" for b in c):
                continue
            missing = [b["id"] for b in assistant["content"] if isinstance(b, dict)
                       and b["type"] == "tool_use" and b["id"] in self.results and b["id"] not in present]
            c[:0] = [deepcopy(self.results[tid]) for tid in missing]
            present.update(missing)


def request(session, leaf, profile, options, cache=True):
    messages = [{k: m[k] for k in ("role", "content")} for m in session.state(leaf)]
    if cache and messages:
        last = messages[-1]
        if isinstance(last["content"], str):
            last["content"] = [text_block(last["content"])]
        if last["content"]:
            last["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return {**options, "system": profile["system"], "tools": profile["tools"], "messages": messages}
