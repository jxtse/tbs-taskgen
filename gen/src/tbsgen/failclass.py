"""One place that classifies a finished / failed agent or calibration job.

Three classes, decided from the job's own evidence (never from a guess):

* ``no_model_call``    -- provably no model ran: no trial agent files, no token usage, no reported cost.
  Such a job must not be charged, must not use up infra retries and must not count as a stage infra
  attempt (``infra_after_model`` and ``model_failed`` keep the old accounting).
* ``model_failed``     -- the agent ran and the run ended the normal way (API error, timeout, non-zero
  exit, safety refusal, a plain failed reward).
* ``infra_after_model``-- the agent ran, then infrastructure broke (image / tests-dir / file IO failure).

Every verdict carries an ``error_signature``: the exception class plus the first meaningful message
line with digits, ids and paths stripped.  Two consecutive jobs of one item+stage with the same
signature are a deterministic failure, not something a third retry can fix (``infra_repeat``).

The module reads a Harbor job dir defensively: every missing or unparsable file is just absent evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

NO_MODEL_CALL = "no_model_call"
MODEL_FAILED = "model_failed"
INFRA_AFTER_MODEL = "infra_after_model"

# Agent exceptions that are the run's own outcome (the model ran, this is how it ended).
MODEL_EXCEPTIONS = frozenset({
    "AgentTimeoutError", "NonZeroAgentExitCodeError", "AgentSafetyRefusalError",
    "ContextLengthExceededError", "OutputLengthExceededError",
})

# Infrastructure failures that happen after (or instead of) the model call.
INFRA_EXCEPTIONS = frozenset({
    "AddTestsDirError", "ReadError", "WriteError", "CopyError", "DeleteError", "MountError",
    "EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "AgentInstallError",
    "NetworkConnectionError", "CancelledError", "DockerBuildError", "BuildError",
    "ImageBuildError", "PermissionError", "FileNotFoundError", "OSError",
})

_INFRA_NAME = re.compile(r"(read|write|copy|delete|mount|file|disk|build|docker|image|setup|install|"
                         r"environment|network|sandbox|testsdir|permission|pull|fetch|cancel)", re.I)
_INFRA_TEXT = re.compile(r"(pull access denied|access denied|permission denied|no space left|"
                         r"cannot connect|connection refused|network_mode|is not supported|"
                         r"timed out waiting|apt(-get)? |docker|image build|ENOSPC|"
                         r"input/output error|read error|write error|tests dir|testsdir)", re.I)
_ERR_LINE = re.compile(r"(error|exception|failed|failure|denied|not supported|timed out|refused|"
                       r"quota|429|cannot|could not|no such|missing)", re.I)
# Launcher-log lines that can only appear before the agent starts (environment build / Harbor preflight).  An
# empty local job dir alone is no proof: the copy-back may have failed after a real run (it keeps the old
# whole-allowance charge).
PRE_AGENT = re.compile(r"(network_mode\s*=?\s*'?\w+'? is not supported|egress[- ]control|allowlist.*not supported|"
                       r"pull access denied|failed to solve|error building image|image build failed|"
                       r"docker build .*failed|failed to fetch .*\b429\b|\b429 too many requests\b|"
                       r"E: (?:Failed to fetch|Unable to fetch)|EnvironmentStartTimeoutError|"
                       r"AgentSetupTimeoutError|AgentInstallError|environment (?:setup|start) failed)", re.I)
_PATH = re.compile(r"(?:[A-Za-z]:)?(?:/[\w.\-+@%]+){2,}/?")
_HEX = re.compile(r"\b[0-9a-f]{8,}\b", re.I)
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_WS = re.compile(r"\s+")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_TOKEN = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens",
          "n_input_tokens", "n_output_tokens")
_NO_COST_KINDS = frozenset({"not_launched", "not_started", "relay_refused", "cancelled_unstarted",
                            "expired_unsent", "diversity_parked", "probe_migration_cancelled"})


@dataclass(frozen=True)
class Verdict:
    kind: str
    signature: str
    short: str
    exception: str | None = None
    message: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def no_model_call(self) -> bool:
        return self.kind == NO_MODEL_CALL

    def as_dict(self) -> dict:
        return dict(kind=self.kind, error_signature=self.signature, error_signature_short=self.short,
                    exception=self.exception, message=self.message, evidence=self.evidence)


def normalize_message(text: str | None, *, limit: int = 160) -> str:
    """Digits, hex ids, uuids and paths stripped; whitespace collapsed; lower case."""
    if not text:
        return ""
    line = _first_meaningful(text)
    line = _UUID.sub("<id>", line)
    line = _PATH.sub("<path>", line)
    line = _HEX.sub("<id>", line)
    line = _NUM.sub("<n>", line)
    line = _WS.sub(" ", line).strip().strip("\"'").lower()
    return line[:limit]


def _first_meaningful(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if _ERR_LINE.search(line) and not line.startswith(("Traceback", "File ")):
            return line
    skip = re.compile(r"^(Traceback|\s*File \"|\s*at |Exception traceback|====|----)")
    for line in reversed(lines):
        if not skip.match(line):
            return line
    return lines[-1]


def signature(exception: str | None, message: str | None) -> tuple[str, str]:
    """(full signature, short digest).  Stable across runs: ids / numbers / paths do not enter it."""
    norm = normalize_message(message)
    exc = (exception or "").strip() or "error"
    full = f"{exc}:{norm}" if norm else exc
    digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:12]
    return full[:220], digest


def _read_text(path: Path, limit: int = 8000) -> str:
    try:
        if path.is_file() and path.stat().st_size:
            with open(path, errors="replace") as fh:
                return fh.read()[-limit:]
    except OSError:
        pass
    return ""


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _positive_usage(usage) -> bool:
    if not isinstance(usage, dict):
        return False
    for k in _TOKEN:
        try:
            if int(usage.get(k) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _trial_dirs(job_dir: Path) -> list[Path]:
    try:
        return sorted(p.parent for p in job_dir.glob("*/result.json"))
    except OSError:
        return []


def _has_agent_files(trial: Path) -> bool:
    agent = trial / "agent"
    try:
        if not agent.is_dir():
            return False
        for p in agent.rglob("*"):
            if p.is_file() and p.stat().st_size > 0:
                return True
    except OSError:
        return False
    return False


def _trial_evidence(trial: Path, result: dict | None) -> dict:
    """Model evidence of one trial: agent files, positive tokens, a positive reported cost, an agent start time;
    plus whether Harbor recorded the trial as finished (exception or finished_at) -- only then does the absence
    of agent evidence prove anything."""
    r = result if isinstance(result, dict) else {}
    ar = r.get("agent_result") or {}
    usage = dict(r.get("usage") or {})
    for k in _TOKEN:
        usage[k] = ar.get(k, usage.get(k))
    cost = r.get("cost_usd", ar.get("cost_usd"))
    try:
        cost_usd = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost_usd = None
    ae = r.get("agent_execution") or {}
    return dict(agent_files=_has_agent_files(trial), usage=_positive_usage(usage),
                reported_cost=cost_usd is not None and cost_usd > 0,
                agent_started=bool(isinstance(ae, dict) and ae.get("started_at")),
                recorded=bool(r.get("exception_info") or r.get("finished_at")),
                trial=str(trial))


def _exception_of(result) -> tuple[str | None, str | None]:
    if not isinstance(result, dict):
        return None, None
    ei = result.get("exception_info") or {}
    if not isinstance(ei, dict):
        ei = {}
    return ei.get("exception_type"), ei.get("exception_message")


def _exception_file(*dirs) -> str:
    for d in dirs:
        if not d:
            continue
        for name in ("exception.txt", "exception.log"):
            text = _read_text(Path(d) / name)
            if text.strip():
                return text
        try:
            for p in d.glob("*/exception.txt"):
                text = _read_text(p)
                if text.strip():
                    return text
        except OSError:
            pass
    return ""


def _infraish(exc: str | None, message: str) -> bool:
    if not exc:
        return bool(message and _INFRA_TEXT.search(message))
    if exc in MODEL_EXCEPTIONS:
        return False
    if exc.endswith("ApiError"):
        return False
    if exc in INFRA_EXCEPTIONS or _INFRA_NAME.search(exc):
        return True
    return bool(message and _INFRA_TEXT.search(message))


def classify(job_dir=None, rec: dict | None = None, *, trials=None, log_text: str | None = None) -> Verdict:
    """Classify one finished / failed job.

    job_dir: the Harbor job directory (may not exist).  rec: the run record (run.json of S4c / S5G, or the
    S9 / S9R run record).  trials: per-trial records already parsed (calib.classify_trial output or
    ``{cost_usd, usage, exception, ...}``); the job dir is still read for agent files when given.
    log_text: launcher log tail to use when no job file records an exception.
    """
    rec = rec or {}
    jd = Path(job_dir) if job_dir else None
    trial_res: list[tuple[Path | None, dict]] = []
    if jd is not None:
        for t in _trial_dirs(jd):
            r = _read_json(t / "result.json") or {}
            trial_res.append((t, r))
    for t in trials or []:
        if isinstance(t, dict):
            trial_res.append((None, t))

    ev = dict(agent_files=False, usage=False, reported_cost=False,
              trials=max(len(_trial_dirs(jd)) if jd is not None else 0, len(trials or [])))
    for tdir, r in trial_res:
        if tdir is not None:
            e = _trial_evidence(tdir, r)
            ev["agent_files"] |= e["agent_files"]
            ev["usage"] |= e["usage"]
            ev["reported_cost"] |= e["reported_cost"]
        else:
            ev["usage"] |= _positive_usage(r.get("usage"))
            ev["reported_cost"] |= _result_cost(r)
    # the run record itself: a positive reported cost / usage means the agent ran
    ev["reported_cost"] |= _result_cost(rec)
    ev["usage"] |= _positive_usage(rec.get("usage"))
    if rec.get("not_started") or rec.get("no_model_call"):
        ev["not_started"] = True

    exc = None
    msg = None
    for _, r in trial_res:
        exc, m = _exception_of(r)
        if not exc:
            exc, m = (r.get("exception"), None)
        if not m:
            m = r.get("exception_message") or r.get("error") or r.get("api_msg")
        if exc or m:
            msg = m
            break
    job_result = _read_json(jd / "result.json") if jd is not None else None
    if not exc:
        exc, msg = _exception_of(job_result)
    if not exc and isinstance(rec.get("exception_info"), dict):
        exc, msg = _exception_of(rec)
    if not exc:
        exc = rec.get("exception")
    if not msg:
        msg = rec.get("exception_message")
    if not msg and not exc:
        msg = _exception_file(jd, Path(rec["trial"]) if rec.get("trial") else None,
                              Path(rec["job_dir"]) if rec.get("job_dir") else None)
    if not msg and not exc:
        msg = rec.get("error")
    if not msg or not exc:
        tail = log_text if log_text is not None else (_read_text(Path(rec["log"])) if rec.get("log") else "")
        if not msg and tail:
            msg = tail
        if not exc and tail:
            m = re.search(r"(\w+(?:\.\w+)*(?:Error|Exception|Exit|Timeout))", tail)
            exc = m.group(1) if m else None
    if not msg and jd is not None:
        msg = _exception_file(jd)

    msg = _first_meaningful(str(msg)) if msg else ""
    tdirs = [(t, _trial_evidence(t, r)) for t, r in trial_res if t is not None]
    agent_started = any(e["agent_started"] for _, e in tdirs)
    started = bool(ev["agent_files"] or ev["usage"] or ev["reported_cost"] or agent_started)
    jd_present = bool(jd is not None and jd.is_dir())
    # Positive proof that nothing reached the model.  An empty or missing local job dir alone is not proof (the
    # copy-back may have failed after a real run: the old whole-allowance charge stays).  Proof is one of:
    #   * the record says the launch never started (not_started / no_model_call),
    #   * Harbor recorded the trial's failure (exception_info) and the trial has no agent start, files, usage or cost,
    #   * the launcher log / exception text names a failure that can only happen before the agent runs
    #     (egress-control probe, image build / pull, apt rate limit, environment start).
    blob = "\n".join(x for x in (
        exc or "", msg or "", str(rec.get("error") or ""),
        _read_text(Path(rec["log"]), 4000) if rec.get("log") else "",
        log_text or "", _exception_file(jd) if jd is not None else "") if x)
    recorded = any(isinstance(r, dict) and r.get("exception_info") for t, r in trial_res if t is not None)
    pre_agent = bool(PRE_AGENT.search(blob))
    proved = bool(ev.get("not_started")) or recorded or pre_agent
    if ev.get("not_started") or (not started and proved):
        kind = NO_MODEL_CALL
    elif not started:
        kind = MODEL_FAILED                 # unknown: keep the old accounting (cost counts, infra retries count)
    else:
        kind = INFRA_AFTER_MODEL if _infraish(exc, msg) else MODEL_FAILED
    full, short = signature(exc, msg)
    ev.update(started=started, agent_started=agent_started, job_dir_present=jd_present, harbor_recorded=recorded,
              pre_agent_text=pre_agent)
    return Verdict(kind=kind, signature=full, short=short, exception=exc, message=msg[:300],
                   evidence=ev)


def _result_cost(result) -> bool:
    """A positive cost reported on the run / trial record; explicit no-cost kinds never count."""
    r = result if isinstance(result, dict) else {}
    if str(r.get("cost_kind") or "") in _NO_COST_KINDS:
        return False
    ar = r.get("agent_result") or {}
    cost = r.get("cost_usd", ar.get("cost_usd"))
    try:
        return cost is not None and float(cost) > 0
    except (TypeError, ValueError):
        return False


def repeat_is_deterministic(verdict: Verdict) -> bool:
    """Should two consecutive identical signatures park instead of retrying?

    A model failure (the agent ran and ended the normal way) keeps its old retry budget.  Every infra failure --
    before or after the model -- that repeats with the same normalized signature is treated as deterministic:
    retrying the same image / config again is what burned 55-99 minutes per job in production.  The park reason
    names the signature, so a transient storm (e.g. apt 429) is easy to find and resume.
    """
    return verdict.kind != MODEL_FAILED


# Failures that usually clear on their own (rate limits, dropped connections, gateway errors).  Two in a row is not
# yet evidence of a deterministic fault: a 30-minute apt 429 storm hit every job at once on 2026-11-27.  Such a
# signature parks only after code/calib.infra_repeat_park_transient identical repeats (default 3), each one after the
# normal backoff.
TRANSIENT = re.compile(r"(\b429\b|too many requests|throttl|flow control|rate limit|\bReadError\b|"
                       r"RemoteProtocolError|ConnectError|ConnectTimeout|ReadTimeout|\b50[234]\b|"
                       r"temporarily unavailable|connection reset|service unavailable|bad gateway)", re.I)


def is_transient(signature: str | None) -> bool:
    return bool(signature) and bool(TRANSIENT.search(signature))


def repeat_limit(signature: str | None, section: dict) -> int:
    """How many identical consecutive signatures park the item: infra_repeat_park (default 2), or
    infra_repeat_park_transient (default 3) for a signature that looks transient."""
    base = max(1, int(section.get("infra_repeat_park") or 2))
    if is_transient(signature):
        return max(base, int(section.get("infra_repeat_park_transient") or 3))
    return base


def note(verdict: Verdict) -> str:
    """Short human note for a ledger row / spent.json charge."""
    if verdict.kind == NO_MODEL_CALL:
        return f"no model call ({verdict.signature[:80]})"
    return f"{verdict.kind} ({verdict.signature[:80]})"
