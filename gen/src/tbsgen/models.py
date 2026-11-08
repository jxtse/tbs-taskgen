"""Pluggable model-call layer.

    client = get_client(alias, cfg, ledger)     # alias: "teacher" | "fake:<script.json>" | FakeClient instance
    r = client.chat(system, user, stage="S5", item="doi-...", max_tokens=...)
    r.text, r.usage, r.usd, r.model

* TeacherClient: Anthropic /v1/messages request to an Anthropic-compatible endpoint (by default a local gateway;
  model claude-opus-5-5, adaptive thinking, effort medium (per stage: models.teacher.effort_by_stage), SSE for long
  outputs).  An empty answer cut at max_tokens (thinking used the whole output budget) is billed and retried once at
  models.teacher.effort_on_thinking_exhausted (default "low").  No key in this process: the
  gateway holds the credentials.  If cfg.models.teacher.key_env is set, the key is read from that env var
  (injected by your secret manager) and only placed in a request header; never logged.
* FakeClient: scripted responses for tests / dry runs.  A script is a list of rules
  {"stage": "S5", "match": "<substring of user prompt>"?, "text": "...", "usage": {...}?, "error": "quota"?}
  consumed in order per stage; or a python callable(stage, system, user) -> str.
* RecordingClient: wraps a real client and appends (stage, prompt sha, response text, usage) to a JSONL so a
  real run can later be replayed through FakeClient.

Every call goes through Ledger.reserve/finish.  Errors are classified so the breaker can act:
quota (HTTP 429 with quota wording / 402), auth (401/403), rate (429 other), overload (529/503),
server (5xx), stream (SSE error / incomplete read), network, empty, other.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .budget import BudgetStop, Ledger


@dataclass
class ModelResult:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    usd: float = 0.0
    stop_reason: str | None = None
    latency_s: float = 0.0
    call_id: int | None = None


class ModelError(RuntimeError):
    def __init__(self, cls: str, msg: str, usage: dict | None = None, stop_reason: str | None = None):
        super().__init__(f"{cls}: {msg}")
        self.cls = cls
        # usage of a call that did reach the model (e.g. an empty answer after all output tokens went to
        # thinking): billed and recorded in the ledger like a successful call
        self.usage = usage or {}
        self.stop_reason = stop_reason

    @property
    def thinking_exhausted(self) -> bool:
        """Empty answer cut at max_tokens: adaptive thinking used up the whole output budget."""
        return self.cls == "empty" and self.stop_reason == "max_tokens"


EFFORT_RANK = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4}

QUOTA_PAT = re.compile(r"额度|quota|insufficient|balance|credit|billing|exceeded your", re.I)


def classify_http(status: int, body: str) -> str:
    if status in (401, 403):
        return "auth"
    if status == 402 or (status == 429 and QUOTA_PAT.search(body or "")):
        return "quota"
    if status == 429:
        return "rate"
    if status in (503, 529):
        return "overload"
    if status >= 500:
        return "server"
    return "other"


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def _redact(s: str) -> str:
    return re.sub(r"(sk-[A-Za-z0-9_-]{6})[A-Za-z0-9_-]+|([A-Za-z0-9_-]{8})[A-Za-z0-9_-]{24,}", r"\1\2***", s or "")


# ---------------------------------------------------------------------------------------------- base
class _Base:
    real = True
    name = "base"

    def __init__(self, ledger: Ledger, cfg: dict):
        self.ledger, self.cfg = ledger, cfg

    def chat(self, system: str, user: str, *, stage: str, item: str = "", max_tokens: int | None = None,
             effort: str | None = None) -> ModelResult:
        """effort: per-call override of the teacher's reasoning effort (None: models.<alias>.effort_by_stage
        for this stage, else models.<alias>.effort)."""
        cid = self.ledger.reserve(self.name, stage, item, self.real, _sha(system, user))
        t0 = time.time()
        kw = {"effort": effort} if effort else {}
        try:
            text, usage, resp_model, stop = self._call(system, user, stage=stage, max_tokens=max_tokens, **kw)
        except ModelError as e:
            # a call that reached the model is billed even when it failed (empty answer after thinking)
            e.usd = self.ledger.finish(cid, e.usage or None, error_class=e.cls, error=_redact(str(e)),
                                       latency=time.time() - t0)
            low = self._fallback_effort(stage, effort) if e.thinking_exhausted else None
            if low is None:
                raise
            # adaptive thinking used the whole max_tokens budget without writing any answer: one retry at a lower
            # effort (a new, separately budgeted call; the failed one stays billed in the ledger)
            return self.chat(system, user, stage=stage, item=item, max_tokens=max_tokens, effort=low)
        except BudgetStop:
            raise
        except Exception as e:  # noqa: BLE001
            self.ledger.finish(cid, error_class="other", error=_redact(f"{type(e).__name__}: {e}"),
                               latency=time.time() - t0)
            raise ModelError("other", f"{type(e).__name__}: {e}") from e
        if resp_model and self.real and not (resp_model.startswith(self.name) or resp_model == "dev-" + self.name):
            self.ledger.finish(cid, usage, "model_mismatch", f"asked {self.name} got {resp_model}", time.time() - t0,
                               resp_model)
            self.ledger.open_breaker(self.name, f"responding model {resp_model}")
            raise ModelError("model_mismatch", resp_model)
        usd = self.ledger.finish(cid, usage, latency=time.time() - t0, resp_model=resp_model)
        return ModelResult(text, resp_model or self.name, usage, usd, stop, round(time.time() - t0, 2), cid)

    def _teacher_cfg(self) -> dict:
        return self.cfg["models"].get("teacher") or {}

    def effort_for(self, stage: str, effort: str | None = None) -> str:
        m = self._teacher_cfg()
        return effort or (m.get("effort_by_stage") or {}).get(stage) or m.get("effort", "medium")

    def _fallback_effort(self, stage: str, effort: str | None) -> str | None:
        """Effort for one retry after an empty max_tokens answer, or None (no retry).  The retry only LOWERS the
        effort: a call already at (or below) the fallback effort is not repeated -- the same request would most
        likely exhaust the same budget again and cost another full max_tokens.  Unknown effort names: no retry."""
        low = self._teacher_cfg().get("effort_on_thinking_exhausted")
        cur = self.effort_for(stage, effort)
        if not low or low not in EFFORT_RANK or cur not in EFFORT_RANK:
            return None
        return low if EFFORT_RANK[low] < EFFORT_RANK[cur] else None

    def _call(self, system, user, *, stage, max_tokens, effort=None):  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------- teacher
def _read_sse(r):
    text, model, usage, stop = [], None, {}, None
    try:
        for raw in r:
            stop, model = _sse_line(raw, text, usage, stop, model)
    except ModelError as e:
        e.usage = e.usage or dict(usage)      # tokens already reported by the stream are billed (lower bound)
        raise
    except (AttributeError, TypeError, ValueError) as e:     # malformed event (e.g. "delta": null)
        raise ModelError("stream", f"malformed SSE event: {type(e).__name__}: {e}", usage=dict(usage)) from None
    except (TimeoutError, ConnectionError, OSError, http.client.HTTPException) as e:
        raise ModelError("network", f"{type(e).__name__} mid-stream: {e}", usage=dict(usage)) from None
    if stop is None:
        raise ModelError("stream", "stream ended without message_delta/stop_reason", usage=dict(usage))
    return "".join(text), model, usage, stop


def _sse_line(raw, text, usage, stop, model):
    """One SSE line: appends text, updates usage in place; returns (stop, model)."""
    line = raw.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        return stop, model
    try:
        ev = json.loads(line[5:].strip())
    except json.JSONDecodeError:
        return stop, model
    t = ev.get("type")
    if t == "message_start":
        msg = ev.get("message", {})
        model = msg.get("model")
        usage.update(msg.get("usage") or {})
    elif t == "content_block_delta":
        d = ev.get("delta", {})
        if d.get("type") == "text_delta":
            text.append(d.get("text", ""))
    elif t == "message_delta":
        stop = (ev.get("delta") or {}).get("stop_reason") or stop
        usage.update(ev.get("usage") or {})
    elif t == "error":
        raise ModelError("stream", json.dumps(ev.get("error"))[:300])
    return stop, model


class TeacherClient(_Base):
    real = True

    def __init__(self, ledger: Ledger, cfg: dict, alias: str = "teacher"):
        super().__init__(ledger, cfg)
        self.m = cfg["models"][alias]
        self.name = self.m["model"]
        if not self.m.get("url"):
            raise ModelError("config", f"models.{alias}.url is not set (set models.{alias}.url or TBS_TEACHER_URL)")

    def _teacher_cfg(self) -> dict:
        return self.m

    def _call(self, system, user, *, stage, max_tokens, effort=None):
        m = self.m
        mt = int(max_tokens or m.get("max_tokens", 32000))
        stream = mt > 8000
        body = {"model": m["model"], "max_tokens": mt,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user}], "stream": stream,
                "output_config": {"effort": self.effort_for(stage, effort)}, "thinking": {"type": "adaptive"}}
        hdr = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if m.get("key_env"):
            k = os.environ.get(m["key_env"])
            if not k:
                raise ModelError("auth", f"env {m['key_env']} not set (export it in the environment)")
            hdr["x-api-key"] = k
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(m["url"], data=json.dumps(body).encode(), headers=hdr)
        try:
            with opener.open(req, timeout=int(m.get("timeout", 1800))) as r:
                if stream:
                    txt, mresp, usage, stop = _read_sse(r)
                else:
                    d = json.loads(r.read())
                    txt = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
                    mresp, usage, stop = d.get("model"), d.get("usage") or {}, d.get("stop_reason")
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "replace")[:600]
            raise ModelError(classify_http(e.code, body_txt), f"HTTP {e.code} {_redact(body_txt)}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ModelError("network", f"{type(e).__name__}: {e}") from None
        if not txt and stop != "refusal":
            raise ModelError("empty", f"empty response stop_reason={stop} output_tokens={usage.get('output_tokens')}",
                             usage=usage, stop_reason=stop)
        return txt, usage, mresp, stop


# ---------------------------------------------------------------------------------------------- fake
class FakeClient(_Base):
    """Scripted model.  Never touches the network."""
    real = False

    def __init__(self, ledger: Ledger, cfg: dict, script: list[dict] | Callable | str | Path, name: str = "fake"):
        super().__init__(ledger, cfg)
        self.name = name
        if isinstance(script, (str, Path)):
            script = json.loads(Path(script).read_text())
        self.script = script
        self.used: list[int] = []
        self.log: list[dict] = []

    def _call(self, system, user, *, stage, max_tokens, effort=None):
        self.log.append(dict(stage=stage, sha=_sha(system, user), effort=self.effort_for(stage, effort)))
        if callable(self.script):
            out = self.script(stage, system, user)
            if isinstance(out, dict):
                if out.get("error"):
                    raise ModelError(out["error"], out.get("text", "scripted error"), usage=out.get("usage"),
                                     stop_reason=out.get("stop"))
                return out.get("text", ""), out.get("usage", {}), self.name, out.get("stop", "end_turn")
            return out, {"input_tokens": len(system + user) // 4, "output_tokens": len(out) // 4}, self.name, "end_turn"
        for i, rule in enumerate(self.script):
            if i in self.used or rule.get("stage") not in (stage, "*"):
                continue
            if rule.get("match") and rule["match"] not in user:
                continue
            if not rule.get("sticky"):
                self.used.append(i)
            if rule.get("error"):
                raise ModelError(rule["error"], rule.get("text", "scripted error"), usage=rule.get("usage"),
                                 stop_reason=rule.get("stop"))
            txt = rule["text"]
            return txt, rule.get("usage", {"input_tokens": len(system + user) // 4, "output_tokens": len(txt) // 4}), \
                self.name, "end_turn"
        raise ModelError("other", f"fake script exhausted for stage {stage}")


class RecordingClient:
    """Wrap a client; append each successful response to a replayable JSONL (no prompts, no secrets)."""

    def __init__(self, inner: _Base, path: str | Path):
        self.inner, self.path = inner, Path(path)
        self.real, self.name = inner.real, inner.name

    def chat(self, system, user, **kw) -> ModelResult:
        r = self.inner.chat(system, user, **kw)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(dict(stage=kw.get("stage"), sha=_sha(system, user), text=r.text, usage=r.usage,
                                    model=r.model)) + "\n")
        return r


def get_client(alias, cfg: dict, ledger: Ledger, fake_script=None):
    if alias is None or alias == "rules":
        return None
    if isinstance(alias, _Base) or isinstance(alias, RecordingClient):
        return alias
    if alias.startswith("fake"):
        return FakeClient(ledger, cfg, fake_script if fake_script is not None else alias.split(":", 1)[1])
    if alias in cfg["models"] and isinstance(cfg["models"][alias], dict) and "url" in cfg["models"][alias]:
        return TeacherClient(ledger, cfg, alias)
    raise ValueError(f"unknown model alias {alias!r}")


def extract_json(text: str):
    """First JSON object/array in text (handles ```json fences)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in model output")
    obj, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return obj


FILE_RE = re.compile(r"^=== FILE: (?P<path>[^\n=]+?) ===\n(?P<body>.*?)(?=^=== FILE: |^=== END ===|\Z)", re.S | re.M)


def parse_files(text: str, truncated: bool = False) -> dict[str, str]:
    """Multi-file model output:  === FILE: rel/path ===\\n<content>  ...  === END ===

    truncated=True (stop_reason == max_tokens): the last block is cut mid-file unless an END/next-FILE marker
    follows it, so it is dropped instead of being written as a broken file."""
    out = {}
    cut_marker = False
    ms = list(FILE_RE.finditer(text))
    if truncated and ms:
        # The last block is complete iff its body is followed by a marker line, possibly itself cut short
        # ("=== FILE: b.py" / "=== END" / "===").  Only a body that runs to the very end of the text was cut
        # (review #10: a cut inside the next FILE header used to drop the previous, complete file).
        last = ms[-1]
        rest = text[last.end("body"):]
        if not re.match(r"\A\s*={1,3}(?: ?(?:F|E)[A-Z]*)?", rest) or not rest.strip():
            body = last.group("body")
            lines = body.rstrip("\n").split("\n")
            tail_line = lines[-1] if lines else ""
            if re.fullmatch(r"={1,3}(?: ?(?:FILE|END|F|E|FI|FIL|EN)?.*)?", tail_line.strip()) and tail_line.strip():
                cut_marker = True     # body ends with a cut marker line: the file itself was complete
            else:
                ms = ms[:-1]
    for idx, m in enumerate(ms):
        p = m.group("path").strip()
        body = m.group("body")
        if cut_marker and idx == len(ms) - 1:
            body = body.rstrip("\n").rsplit("\n", 1)[0] + "\n"     # drop the cut marker line
        body = re.sub(r"^```[a-zA-Z0-9]*\n", "", body)
        body = re.sub(r"\n```\s*$", "\n", body.rstrip() + "\n")
        out[p] = body

    return out
