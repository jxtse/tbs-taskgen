"""S2 screen with TypeSafe Jev (System One API).  All question wording and thresholds live in this file.

One request per paper.  The request carries one shared state (the paper's bibliographic record) and a set of
narrow questions that Jev answers in parallel; code combines the answer probabilities into keep / drop /
uncertain (see `combine`).  Jev only judges; the decision rule is ours and is versioned (`QUESTIONS_VERSION`,
`RULE_VERSION`).

API (https://docs.typesafe.ai): POST {url} with {"model", "state", "questions"}; the answer for a Noul is a
probability of "yes", for a Choice a chosen option plus per-option probabilities.  Billing is per input token.

Transport: standard-library urllib, timeout on every request; 429 / 529 (and 5xx / network errors) are retried
with exponential backoff that honours `retry-after`, up to `max_retries`; other 4xx are not retried.  Every
logical call (all its HTTP attempts) is one row in the budget ledger (stage S2 or S4), priced at
`models.prices[<jev model>]`, and feeds the ledger's consecutive-error breaker.

The key is read only from the environment variable named by `screen.jev.key_env` (default TYPESAFE_API_KEY)
and only placed in the Authorization header.  It is never logged or written to disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .budget import BudgetStop, Ledger
from .models import ModelError, _redact, classify_http

QUESTIONS_VERSION = "jev-q1"
RULE_VERSION = "jev-r2"

# ------------------------------------------------------------------------------------------- questions
# State shape:  {"paper": {"title", "abstract" | "full_text_opening", "venue"?, "year"?, "code_links"?,
#                          "data_links"?}}.  Empty fields are left out (an empty list can read as a signal).
PAPER_TYPES = {
    "new_method": "Proposes or improves a computational method: an algorithm, estimator, numerical scheme, "
                  "model-fitting or data-processing procedure",
    "software_tool": "Presents a software package or tool",
    "dataset_or_benchmark": "Releases a dataset or a benchmark",
    "applied_study": "Applies existing computational methods to answer a question about a specific system or "
                     "dataset",
    "experimental_study": "Reports laboratory, field, or instrument measurements; computation plays a minor role",
    "clinical_or_qualitative": "A clinical trial, patient cohort, interview, questionnaire, or other qualitative "
                               "study",
    "review": "Summarises earlier literature: a review, a survey of methods, a tutorial, or a perspective",
    "theory": "Proves theorems or derives analytic results, with no computational procedure tested on data",
}

QUESTIONS: dict[str, dict] = {
    # Choice + companion Noul (a Choice always picks something; the Noul says whether any option fits).
    "paper_type": {
        "type": "choice",
        "instructions": "What kind of contribution does the paper in `paper` mainly make?",
        "criteria": PAPER_TYPES,
    },
    "paper_type_fits": {
        "type": "noul",
        "instructions": "Does one of these describe the main contribution of `paper`: a new computational method, "
                        "a software tool, a dataset or benchmark, an applied computational study, an experimental "
                        "study, a clinical or qualitative study, a literature review, or a theory paper?",
    },
    "computational_method": {
        "type": "noul",
        "instructions": "Does `paper` describe a computational procedure that a skilled programmer could "
                        "re-implement from the paper?",
        "criteria": {
            "true": "The paper presents an algorithm, estimator, numerical scheme, model-fitting or data-processing "
                    "procedure, with enough description to code it (possibly with the full text).",
            "false": "The paper reports measurements, observations, opinions, or a summary of other work, and "
                     "contains no procedure one could code up.",
        },
    },
    "quantitative_check": {
        "type": "noul",
        "instructions": "Does `paper` check the results of its computation with numbers, such as error against a "
                        "known truth, accuracy on held-out data, or agreement with simulations?",
    },
    "single_machine": {
        "type": "noul",
        "instructions": "Could the core computation in `paper` be run on one ordinary computer within a few hours?",
        "criteria": {
            "true": "Fitting, estimation, signal processing, simulation, or optimisation at a scale a laptop or "
                    "workstation handles.",
            "false": "Results depend on a GPU cluster, training a large neural network from scratch, a "
                     "supercomputer run, a wet-lab experiment, or a special instrument.",
        },
    },
    "data_public": {
        "type": "noul",
        "instructions": "Does `paper` use or release data that anyone can download?",
    },
    "data_simulatable": {
        "type": "noul",
        "instructions": "Could realistic test data for the computation in `paper` be generated by a simulation or "
                        "a synthetic model?",
    },
    "black_box_queries": {
        "type": "noul",
        "instructions": "Does the method in `paper` work by repeatedly querying an expensive black-box function, "
                        "an oracle, or a live experiment under a limited query budget?",
        "criteria": {
            "true": "For example Bayesian optimisation, an active-learning loop, bandits, or derivative-free "
                    "optimisation of an expensive simulator.",
            "false": "The method processes a fixed dataset or solves a fully specified model.",
        },
    },
    "is_review": {
        "type": "noul",
        "instructions": "Is `paper` a review or survey of earlier literature?",
        "criteria": {
            "true": "A literature review, a survey of methods, a tutorial, or a perspective article.",
            "false": "Original research. This includes statistical methods for sample-survey data, such as "
                     "estimators applied to a household or living-conditions survey.",
        },
    },
    "enough_info": {
        "type": "noul",
        "instructions": "Does `paper` say enough about what was done to tell whether it contains a computational "
                        "method?",
    },
}

# ------------------------------------------------------------------------------------------- S4 feasibility
# S4 asks a second question set about task-ability, on the full text's opening (title + abstract + introduction):
# S2 (abstract only) says "is this a computational method paper"; every paper that reaches S4 already is one, and
# in e2e-prod-20261109 the S2 probabilities of the 3 released and the 9 failed papers were indistinguishable
# (P(method) 0.87-0.93, P(single machine) 0.91-0.97).  These questions target why generated tasks failed.
S4_QUESTIONS_VERSION = "jev-s4-q1"
S4_QUESTIONS: dict[str, dict] = {
    "self_contained_core": {
        "type": "noul",
        "instructions": "Could the core algorithm of `paper` be re-implemented from scratch in a few hundred lines of "
                        "Python with numpy/scipy, without the authors' code or another large software package?",
        "criteria": {
            "true": "The method is a self-contained algorithm or estimator (e.g. a fitting procedure, a numerical "
                    "transform, a filtering or unwrapping algorithm) described in enough detail to code.",
            "false": "The contribution lives inside a large code base (a full Boltzmann solver, an N-body or "
                     "hydrodynamics simulation, a deep network), or the paper mainly reports results of such code.",
        },
    },
    "synthetic_truth": {
        "type": "noul",
        "instructions": "Could a short program simulate input data for the method in `paper` together with the true "
                        "answer, so that the method's output can be checked against that known truth?",
    },
    "numeric_score": {
        "type": "noul",
        "instructions": "Does the method in `paper` produce numbers (estimates, reconstructions, spectra, labels) whose "
                        "accuracy against a known truth can be measured objectively with an error or accuracy metric?",
    },
    "graded_steps": {
        "type": "noul",
        "instructions": "Does the method in `paper` combine several distinct steps such that leaving out one key step "
                        "still gives a usable but measurably less accurate result?",
    },
    "simple_baseline_worse": {
        "type": "noul",
        "instructions": "Is there a simple standard approach to the same problem as `paper` (a textbook or default "
                        "method) that gives clearly less accurate results than the paper's method?",
    },
    "needs_restricted_resources": {
        "type": "noul",
        "instructions": "Does reproducing the core computation of `paper` need proprietary or restricted data, a "
                        "special instrument, a GPU cluster or supercomputer, or many hours of compute?",
    },
}
# Decision (combine_s4): drop when the probability that the paper is task-able is low on a hard requirement.
# Defaults are conservative, set on e2e-prod-20261109 (111 papers that had passed the rule probe; offline run with
# real Jev answers, gen/README.md): they drop 10/111 (deep-learning QSM nets, pure theory, a pharmacovigilance study,
# a DES-Y3 data analysis, a QC-report tool), none of the 3 released or the 9 failed generated tasks.  The failed
# tasks are NOT separable by these answers (their S4 score is not lower than the released ones'): S4 removes
# clearly unsuitable papers, the S6B early stops handle the rest.  The soft questions (graded_steps,
# simple_baseline_worse) only enter the S4 score (recorded, not used for decisions).
S4_THRESHOLDS = {
    "core_no": 0.20,           # released task 836a87 (PT extension of CLASS) scored 0.25-0.26: keep a margin
    "synthetic_no": 0.40,      # 0.35-0.38 only on two pure-theory papers; next lowest 0.62
    "numeric_no": 0.30,
    "restricted_yes": 0.70,    # 0.76 on a DES-Y3 data analysis; next highest 0.65
}
S4_HARD = (("self_contained_core", "core_no", "jev_s4_no_self_contained_core"),
           ("synthetic_truth", "synthetic_no", "jev_s4_no_synthetic_truth"),
           ("numeric_score", "numeric_no", "jev_s4_no_numeric_score"))


def combine_s4(p: dict, th: dict | None = None) -> dict:
    """S4 probabilities -> dict(feasible, reasons, score).  A missing answer never drops."""
    th = dict(S4_THRESHOLDS, **(th or {}))
    reasons = [why for q, k, why in S4_HARD if p.get(q) is not None and p[q] <= th[k]]
    if p.get("needs_restricted_resources") is not None and p["needs_restricted_resources"] >= th["restricted_yes"]:
        reasons.append("jev_s4_needs_restricted_resources")
    vals = [p.get(q) for q in ("self_contained_core", "synthetic_truth", "numeric_score", "graded_steps",
                               "simple_baseline_worse")]
    score = None
    if all(v is not None for v in vals) and p.get("needs_restricted_resources") is not None:
        score = 1.0
        for v in vals:
            score *= v
        score = round(score * (1 - p["needs_restricted_resources"]), 4)
    return dict(feasible=not reasons, reasons=reasons, score=score)


def build_s4_state(c: dict, fulltext: str, max_chars: int = 6000) -> dict:
    """Title + the full text's opening (whitespace collapsed: PDF text has long runs of spaces)."""
    # collapse whitespace first, then cut (PDF text can be mostly spaces: cutting first lost the method text)
    raw = fulltext or ""
    words, n = [], 0
    for w in raw[: max_chars * 50].split():
        words.append(w)
        n += len(w) + 1
        if n > max_chars:
            break
    head = " ".join(words)[:max_chars]
    return {"paper": {"title": c.get("title") or "", "full_text_opening": head}}


S4_CACHE = "S4_jev.json"


def s4_feasibility(client: "JevClient", it, c: dict, fulltext: str, cfg: dict) -> dict:
    """Jev S4 record for one item (cached per request key in S4_jev.json; may raise ModelError / BudgetStop)."""
    from .state import read_json, write_json
    pj = cfg["probe"].get("jev") or {}
    state = build_s4_state(c, fulltext, int(pj.get("max_chars", 6000)))
    rk = request_key(client.name, state, S4_QUESTIONS_VERSION)
    cache = read_json(it.p(S4_CACHE), {}) or {}
    if rk in cache:
        # the cache keeps the answers; the verdict is recomputed with the current thresholds (review: a cached
        # verdict would ignore a threshold change)
        hit = cache[rk]
        return dict(hit, **combine_s4(hit["probabilities"], pj.get("thresholds")), cached=True)
    r = client.ask(state, stage="S4", item=it.item_id, questions=S4_QUESTIONS)
    p = probabilities(r["answers"])
    dec = combine_s4(p, pj.get("thresholds"))
    rec = dict(dec, probabilities=p, jev_model=r["model"], questions_version=S4_QUESTIONS_VERSION,
               input_tokens=r["usage"]["input_tokens"], usd=r.get("usd"), request_key=rk)
    cache[rk] = rec
    write_json(it.p(S4_CACHE), cache)
    return rec


# ------------------------------------------------------------------------------------------- thresholds
# Noul probabilities: >= YES reads as yes, <= NO reads as no, between = not settled.  Start values follow the
# TypeSafe guidance (0.7 / 0.3) and were not tuned.  The one tuned value is `joint_no`: chosen on the dev half
# of the reference set (gen/eval/screen_jev/) as the highest cut-off that kept dev recall >= 0.98; the
# reported numbers come from the other half.
THRESHOLDS = {
    "yes": 0.7,
    "no": 0.3,
    "method_no": 0.3,          # computational_method <= this -> drop (no_computational_method)
    "compute_no": 0.3,         # single_machine <= this -> drop (needs_heavy_compute)
    "data_no": 0.3,            # data_public <= and data_simulatable <= this -> drop (no_data)
    "type_drop": 0.7,          # paper_type in DROP_TYPES with probability >= this, fits >= yes, and
    "type_drop_method_max": 0.5,   # ... computational_method < this -> drop (review / clinical)
    "review_drop": 0.7,        # is_review >= this and computational_method < type_drop_method_max -> drop
    "interact": 0.7,           # black_box_queries >= this -> form_hint INTERACT (S4 bans the form)
    "joint_no": 0.35,          # not settled, and P(method) * P(single machine) * P(some data) < this -> drop
}
DROP_TYPES = ("review", "clinical_or_qualitative")


def probabilities(answers: dict) -> dict:
    """Flatten an API `answers` object to {question: probability} (+ paper_type / paper_type_p)."""
    out: dict = {}
    for name, a in (answers or {}).items():
        if a.get("type") == "choice" or "choice" in a:
            out[name] = a.get("choice")
            out[name + "_p"] = round(float((a.get("probabilities") or {}).get(a.get("choice"), 0.0)), 4)
        elif "noul" in a:
            out[name] = round(float(a["noul"]), 4)
    return out


def combine(p: dict, th: dict | None = None) -> dict:
    """Probabilities -> decision.  Returns dict(decision in {keep, drop, uncertain}, reason, form_hint, flags).

    Order: clear drops first (review / clinical type, no computational method, heavy compute, no data), then a
    clear keep (method, compute and some data source all read yes); a paper that is neither is dropped when the
    joint probability P(method) * P(single machine) * P(some data) is below `joint_no` (several questions lean
    no at once), otherwise it is uncertain with the open questions listed.  "Not enough information" never drops
    on its own: it only explains an uncertain verdict."""
    th = dict(THRESHOLDS, **(th or {}))
    m, sm = p.get("computational_method"), p.get("single_machine")
    dp, ds = p.get("data_public"), p.get("data_simulatable")
    form_hint = "INTERACT" if (p.get("black_box_queries") or 0) >= th["interact"] else None
    ptype, ptype_p, fits = p.get("paper_type"), p.get("paper_type_p") or 0.0, p.get("paper_type_fits") or 0.0
    if None in (m, sm, dp, ds):
        return dict(decision="uncertain", reason="jev_missing_answer", form_hint=form_hint, open=[])
    if ptype in DROP_TYPES and ptype_p >= th["type_drop"] and fits >= th["yes"] and m < th["type_drop_method_max"]:
        return dict(decision="drop", reason=f"jev_type_{ptype}", form_hint=form_hint, open=[])
    if (p.get("is_review") or 0) >= th["review_drop"] and m < th["type_drop_method_max"]:
        return dict(decision="drop", reason="jev_type_review", form_hint=form_hint, open=[])
    if m <= th["method_no"]:
        return dict(decision="drop", reason="jev_no_computational_method", form_hint=form_hint, open=[])
    if sm <= th["compute_no"]:
        return dict(decision="drop", reason="jev_needs_heavy_compute", form_hint=form_hint, open=[])
    if dp <= th["data_no"] and ds <= th["data_no"]:
        return dict(decision="drop", reason="jev_no_data", form_hint=form_hint, open=[])
    open_q = [k for k, v in (("computational_method", m), ("single_machine", sm)) if v < th["yes"]]
    if max(dp, ds) < th["yes"]:
        open_q.append("data")
    if not open_q:
        return dict(decision="keep", reason=None, form_hint=form_hint, open=[])
    if m * sm * max(dp, ds) < th["joint_no"]:
        return dict(decision="drop", reason="jev_low_joint:" + ",".join(open_q), form_hint=form_hint, open=open_q)
    if (p.get("enough_info") is not None) and p["enough_info"] <= th["no"]:
        open_q.append("not_enough_info")
    return dict(decision="uncertain", reason="jev_uncertain:" + ",".join(open_q), form_hint=form_hint, open=open_q)


def score_of(p: dict) -> float | None:
    """Single number for S2_screen.json `score` (shared with the rules screen): P(method) * P(single machine) * P(some data)."""
    try:
        return round(p["computational_method"] * p["single_machine"] * max(p["data_public"], p["data_simulatable"]), 4)
    except (KeyError, TypeError):
        return None


# ------------------------------------------------------------------------------------------- state
def build_state(c: dict, fulltext_head: str | None = None, max_chars: int = 4000) -> dict:
    paper: dict = {"title": c.get("title") or ""}
    if c.get("abstract"):
        paper["abstract"] = c["abstract"]
    elif fulltext_head:
        paper["full_text_opening"] = fulltext_head[:max_chars]
    for k in ("venue", "year"):
        if c.get(k):
            paper[k] = c[k]
    code = c.get("code_links") or c.get("code_urls")
    data = c.get("data_links") or c.get("data_urls")
    if code:
        paper["code_links"] = list(code)[:5]
    if data:
        paper["data_links"] = list(data)[:5]
    return {"paper": paper}


def request_key(model: str, state: dict, questions_version: str = QUESTIONS_VERSION) -> str:
    h = hashlib.sha256(json.dumps([model, questions_version, state], sort_keys=True).encode()).hexdigest()
    return h[:16]


# ------------------------------------------------------------------------------------------- client
def _default_transport(proxy: str):
    handlers = [urllib.request.ProxyHandler({})] if proxy == "none" else []
    opener = urllib.request.build_opener(*handlers)

    def send(url: str, body: bytes, headers: dict, timeout: float):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with opener.open(req, timeout=timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), e.read()
    return send


RETRY_STATUS = {429, 529, 500, 502, 503, 504}


class JevClient:
    """One logical call = one ledger row; HTTP retries happen inside it."""

    def __init__(self, ledger: Ledger, cfg: dict, transport: Callable | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.ledger, self.cfg = ledger, cfg
        self.j = cfg["screen"]["jev"]
        self.name = self.j["model"]
        self.transport = transport or _default_transport(self.j.get("proxy", "env"))
        self.sleep = sleep

    def key(self) -> str:
        k = os.environ.get(self.j.get("key_env") or "TYPESAFE_API_KEY")
        if not k:
            raise ModelError("auth", f"env {self.j.get('key_env')} not set (inject it from your secret store)")
        return k

    def ask(self, state: dict, *, stage: str, item: str = "", questions: dict | None = None) -> dict:
        key = self.key()                 # missing key: no ledger row, caller falls back / parks
        qs = QUESTIONS if questions is None else questions
        qv = QUESTIONS_VERSION if questions is None else S4_QUESTIONS_VERSION if questions is S4_QUESTIONS else "custom"
        body = json.dumps({"model": self.name, "state": state, "questions": qs}).encode()
        est = float(self.j.get("est_usd_per_call", 0.001))
        cid = self.ledger.reserve(self.name, stage, item, True, request_key(self.name, state, qv), est_usd=est)
        t0 = time.time()
        hdr = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        attempts, max_retries = 0, int(self.j.get("max_retries", 4))
        base, cap = float(self.j.get("backoff_base", 1.0)), float(self.j.get("backoff_max", 30.0))
        try:
            while True:
                attempts += 1
                try:
                    status, rh, raw = self.transport(self.j["url"], body, hdr, float(self.j.get("timeout", 30)))
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                    status, rh, raw = None, {}, f"{type(e).__name__}: {e}".encode()
                if status == 200:
                    break
                text = raw.decode("utf-8", "replace")[:600] if isinstance(raw, (bytes, bytearray)) else str(raw)
                cls = "network" if status is None else classify_http(status, text)
                retryable = status is None or status in RETRY_STATUS
                if cls in ("quota", "auth") or not retryable or attempts > max_retries:
                    raise ModelError(cls, f"HTTP {status} after {attempts} attempt(s): {_redact(text)[:300]}")
                ra = {k.lower(): v for k, v in (rh or {}).items()}.get("retry-after")
                try:
                    wait = float(ra) if ra is not None else None
                except ValueError:
                    wait = None
                if wait is None:
                    wait = min(cap, base * 2 ** (attempts - 1)) * (0.5 + random.random() / 2)
                self.sleep(min(max(wait, 0.0), float(self.j.get("retry_after_max", 120))))
            d = json.loads(raw)
            if not isinstance(d.get("answers"), dict) or not d["answers"]:
                raise ModelError("empty", "response without answers")
        except ModelError as e:
            self.ledger.finish(cid, error_class=e.cls, error=str(e), latency=time.time() - t0)
            raise
        except BudgetStop:
            raise
        except Exception as e:  # noqa: BLE001
            self.ledger.finish(cid, error_class="other", error=_redact(f"{type(e).__name__}: {e}"),
                               latency=time.time() - t0)
            raise ModelError("other", f"{type(e).__name__}: {e}") from e
        usage = {"input_tokens": int((d.get("usage") or {}).get("input_tokens", 0) or 0),
                 "output_tokens": int((d.get("usage") or {}).get("output_tokens", 0) or 0)}
        resp_model = d.get("model")
        lat = time.time() - t0
        if self.name != "jev-latest" and resp_model and resp_model != self.name:
            self.ledger.finish(cid, usage, "model_mismatch", f"asked {self.name} got {resp_model}", lat, resp_model)
            self.ledger.open_breaker(self.name, f"responding model {resp_model}")   # thresholds are per version
            raise ModelError("model_mismatch", f"asked {self.name} got {resp_model}")
        # price: billed on input tokens only
        usd = self.ledger.finish(cid, {"input_tokens": usage["input_tokens"]}, latency=lat, resp_model=resp_model)
        return dict(answers=d["answers"], model=resp_model, usage=usage, usd=usd, latency_s=round(lat, 3),
                    attempts=attempts, call_id=cid)


# ------------------------------------------------------------------------------------------- record
def record_from_answer(r: dict, cfg: dict, deferred_input: bool) -> dict:
    j = cfg["screen"]["jev"]
    p = probabilities(r["answers"])
    dec = combine(p, j.get("thresholds"))
    uncertain = dec["decision"] == "uncertain"
    if dec["decision"] == "keep":
        keep = True
    elif dec["decision"] == "drop":
        keep = False
    else:
        keep = j.get("uncertain", "keep") != "drop"
    reason = None if keep else ("jev_uncertain_dropped" if uncertain else dec["reason"])
    return dict(score=score_of(p), keep=keep, form_hint=dec["form_hint"], method="jev", reason=reason,
                deferred=False, decision=dec["decision"], uncertain=uncertain, uncertain_reason=dec["reason"]
                if uncertain else None, drop_reason=dec["reason"] if dec["decision"] == "drop" else None,
                probabilities=p, jev_model=r["model"], questions_version=QUESTIONS_VERSION,
                rule_version=RULE_VERSION, input_tokens=r["usage"]["input_tokens"], usd=r.get("usd"),
                latency_s=r.get("latency_s"), attempts=r.get("attempts"), input="fulltext_head" if deferred_input
                else "abstract")


CACHE = "S2_jev.json"      # per item: {request_key: record}; resume reads it instead of calling again


def jev_screen(client: JevClient, c: dict, cfg: dict, *, stage: str, item: str,
               fulltext_head: str | None = None, cache: dict | None = None) -> dict:
    """Jev S2 record for one paper (may raise ModelError / BudgetStop).  A record already in `cache` under the
    same request key (model + question version + state) is returned without a call."""
    state = build_state(c, fulltext_head, int(cfg["screen"]["jev"].get("max_chars", 4000)))
    rk = request_key(client.name, state)
    if cache and rk in cache:
        return dict(cache[rk], cached=True)
    r = client.ask(state, stage=stage, item=item)
    return dict(record_from_answer(r, cfg, fulltext_head is not None and not c.get("abstract")), request_key=rk)


def cached_call(client: JevClient, it, c: dict, cfg: dict, *, stage: str, fulltext_head: str | None = None) -> dict:
    from .state import read_json, write_json
    cache = read_json(it.p(CACHE), {}) or {}
    rec = jev_screen(client, c, cfg, stage=stage, item=it.item_id, fulltext_head=fulltext_head, cache=cache)
    if not rec.get("cached"):
        cache[rec["request_key"]] = rec
        write_json(it.p(CACHE), cache)
    return rec


def prefetch(client: JevClient, items: list, cfg: dict, concurrency: int | None = None) -> int:
    """Screen the abstracts of many S2 items concurrently (bounded thread pool) before the sequential state
    machine reaches them; results land in each item's cache.  Errors are left for the per-item S2 handler
    (fallback / park).  Returns the number of new calls that succeeded."""
    from .state import read_json
    todo = []
    for it in items:
        c = read_json(it.p("S1_candidate.json"))
        if not c or not c.get("abstract"):
            continue            # no abstract -> deferred to S4, no call now
        st = build_state(c, None, int(cfg["screen"]["jev"].get("max_chars", 4000)))
        if request_key(client.name, st) in (read_json(it.p(CACHE), {}) or {}):
            continue
        todo.append((it, c))
    if not todo:
        return 0
    n = max(1, int(concurrency or cfg["screen"]["jev"].get("concurrency", 4)))

    def one(pair):
        it, c = pair
        try:
            cached_call(client, it, c, cfg, stage="S2")
        except (ModelError, BudgetStop):
            return 0
        return 1
    with ThreadPoolExecutor(max_workers=n) as ex:
        return sum(ex.map(one, todo))
