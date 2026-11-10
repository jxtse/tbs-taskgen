"""S2 publisher blocklist: papers whose full text we cannot obtain are dropped before screening.

A rule matches on the DOI prefix (registrant, e.g. "10.1177") and optionally on the publication year
(`before_year`: only papers published strictly before that year).  The year comes from the candidate record;
for Elsevier-style DOIs without a year the two-digit year embedded in the PII ("...(97)00086-0") is used.
A paper with no DOI, or with an unknown year under a year-limited rule, is never blocked.

Auto-recheck (config screen.blocklist_recheck): a rule that blocks more than min_share of a target's
candidates gets a small weekly trial quota — that many of its papers go through the normal S2 screen -> S3
fetch path; their outcome (full text obtained or not) is recorded in the run's blocklist-trials.json.  Once a
rule has min_trials decided trials, a success rate >= lift_rate lifts it (it stops blocking; recorded with an
event, `tbsgen review summary` shows it), else it stays and the weekly quota keeps probing.  State is kept in
<run>/blocklist-trials.json; `resolve()` turns settled trials into ok/fail, `decide()` is called on every
match.
"""
from __future__ import annotations

import datetime
import json
import re
import time
import urllib.parse
from pathlib import Path

_PII_YEAR = re.compile(r"\((\d{2})\)")
_YEAR = re.compile(r"^\s*(\d{4})(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\s*$")   # 1997, 1997-03, 1997-03-01


def _norm_doi(doi: str | None) -> str:
    d = urllib.parse.unquote(str(doi or "")).strip()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)\s*", "", d, flags=re.I)
    return d.strip().lower()


def paper_year(c: dict) -> int | None:
    m = _YEAR.match(str(c.get("year") or ""))
    if m and 1800 <= int(m.group(1)) <= 2100:
        return int(m.group(1))
    doi = _norm_doi(c.get("doi"))
    if doi.startswith("10.1016/") and (m := _PII_YEAR.search(doi)):
        yy = int(m.group(1))
        return 1900 + yy if yy >= 50 else 2000 + yy
    return None


def match(c: dict, rules: list[dict] | None, lifted: set[str] | None = None) -> dict | None:
    """The first rule that blocks candidate `c`, or None.  Rules in `lifted` (auto-recheck decision) don't block."""
    doi = _norm_doi(c.get("doi"))
    if not doi or not rules:
        return None
    for r in rules:
        name = r.get("name") or r.get("doi_prefix")
        if lifted and name in lifted:
            continue
        pre = str(r.get("doi_prefix") or "").lower().rstrip("/")
        if not pre or not (doi == pre or doi.startswith(pre + "/")):
            continue
        if r.get("before_year") is not None:
            y = paper_year(c)
            if y is None or y >= int(r["before_year"]):
                continue
        return r
    return None


def screen_record(c: dict, rule: dict) -> dict:
    """S2_screen.json record for a blocked paper (same shape as a rules/Jev drop)."""
    name = rule.get("name") or rule.get("doi_prefix")
    return dict(score=None, keep=False, method="blocklist", reason=f"blocklist:{name}",
                blocklist=dict(name=name, doi_prefix=rule.get("doi_prefix"), before_year=rule.get("before_year"),
                               why=rule.get("why"), year=paper_year(c)))


# ---------------------------------------------------------------- auto-recheck
def _iso_week() -> str:
    d = datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).isocalendar()
    return f"{d[0]}-W{d[1]:02d}"


def state_path(store) -> Path:
    return Path(store.root) / "blocklist-trials.json"


def load_state(store) -> dict:
    try:
        return json.loads(state_path(store).read_text())
    except (OSError, ValueError):
        return dict(rules={})


def save_state(store, st: dict) -> None:
    state_path(store).write_text(json.dumps(st, indent=1, sort_keys=True))


# per-process cache of target shares (never persisted: its keys are tuples and it is only valid for one run)
_SHARE_CACHE: dict[int, dict] = {}


def lifted_rules(st: dict) -> set[str]:
    return {k for k, v in (st.get("rules") or {}).items() if v.get("lifted")}


def target_share(store, c: dict, rules: list[dict], rule: dict, lifted: set[str], cache: dict) -> float | None:
    """The share of this target's candidates that `rule` blocks (dropped or not).  Cached per target + rule."""
    tgt = str(c.get("_target") or c.get("target") or "")
    if not tgt:
        return None
    key = (tgt, rule.get("name"))
    if key in cache:
        return cache[key]
    n = m = 0
    for it in store.items():
        cc = json.loads(it.p("S1_candidate.json").read_text()) if it.p("S1_candidate.json").exists() else {}
        if str(cc.get("_target") or cc.get("target") or "") != tgt or not cc.get("doi"):
            continue
        n += 1
        r = match(cc, rules, lifted - {rule.get("name")})      # this rule counts even when lifted elsewhere
        if r is not None and (r.get("name") or r.get("doi_prefix")) == (rule.get("name") or rule.get("doi_prefix")):
            m += 1
    share = (m / n) if n else None
    cache[key] = share
    return share


def decide(store, cfg: dict, c: dict, rules: list[dict], rule: dict, event) -> dict:
    """Called from S2 with a matching rule: {action: block | trial}.  A trial goes through the normal path; the
    paper's fetch outcome is recorded (resolve) and once enough trials decided, a lift decision is made."""
    rc = (cfg.get("screen") or {}).get("blocklist_recheck") or {}
    name = rule.get("name") or rule.get("doi_prefix")
    st = load_state(store)
    r = st.setdefault("rules", {}).setdefault(name, dict(trials=[], lifted=False))
    if r.get("lifted"):
        return dict(action="block", lifted=True)          # caller should not have matched it; block is stale
    if c.get("item_id") and any(t.get("item") == c.get("item_id") for t in r.get("trials") or []):
        return dict(action="trial", repeat=True)          # S2 re-run of a trial paper: no second quota slot
    quota = int(rc.get("weekly_quota", 5))
    week = _iso_week()
    r.setdefault("weeks", {})
    used = int(r["weeks"].get(week, 0))
    share = target_share(store, c, rules, rule, lifted_rules(st), _SHARE_CACHE.setdefault(id(store), {}))
    if share is not None and share > float(rc.get("min_share", 0.10)) and used < quota:
        r["weeks"][week] = used + 1
        r["trials"].append(dict(item=c.get("item_id"), doi=_norm_doi(c.get("doi")), ts=time.time(),
                                outcome="pending"))
        save_state(store, st)
        event("blocklist_trial", rule=name, item=c.get("item_id"), doi=_norm_doi(c.get("doi")),
              share=round(share, 4) if share is not None else None, week=week)
        return dict(action="trial", share=share)
    save_state(store, st)
    return dict(action="block", share=share)


def resolve(store, item, void: str | None = None) -> None:
    """Settle this item's pending trial: ok when its full text exists (S3 delivered), fail when S3 parked it
    fulltext_unavailable (a later delivery by the fetch queue upgrades it to ok in reevaluate()); `void`: the paper
    never reached the download (e.g. the S2 screen dropped it) -> the trial does not count either way."""
    st = load_state(store)
    changed = False
    for name, r in (st.get("rules") or {}).items():
        for t in r.get("trials") or []:
            if t.get("outcome") != "pending" or t.get("item") != item.item_id:
                continue
            if void:
                t["outcome"], t["settled"], t["void"] = "void", time.time(), str(void)[:200]
                changed = True
            elif item.p("fulltext.txt").exists():
                t["outcome"], t["settled"] = "ok", time.time()
                changed = True
            elif item.state.get("stage") == "S3" and item.state.get("status") == "parked" and \
                    item.state.get("reason") == "fulltext_unavailable":
                t["outcome"], t["settled"] = "fail", time.time()
                changed = True
    if changed:
        save_state(store, st)


def reevaluate(store, cfg: dict, event) -> dict:
    """Lift / keep decisions for every rule with enough decided trials; an {rule: decision} report.  A trial that
    failed at S3 but whose paper was delivered later by the fetch queue is upgraded to ok first."""
    rc = (cfg.get("screen") or {}).get("blocklist_recheck") or {}
    min_trials, lift_rate = int(rc.get("min_trials", 5)), float(rc.get("lift_rate", 0.6))
    st = load_state(store)
    upgraded = False
    for r in (st.get("rules") or {}).values():
        for t in r.get("trials") or []:
            if t.get("outcome") not in ("pending", "fail") or not t.get("item"):
                continue
            try:
                it = store.item(t["item"])
                if it.p("fulltext.txt").exists():
                    t["outcome"], t["settled"] = "ok", time.time()
                    upgraded = True
            except Exception:
                pass
    out = {}
    for name, r in (st.get("rules") or {}).items():
        if r.get("lifted"):
            out[name] = "lifted"
            continue
        dec = [t for t in (r.get("trials") or []) if t.get("outcome") in ("ok", "fail")]
        if len(dec) < min_trials:
            continue
        if (r.get("kept") or [{}])[-1].get("n") == len(dec) and not upgraded:
            out[name] = "kept"                            # nothing new since the last decision: no new event
            continue
        ok = sum(1 for t in dec if t["outcome"] == "ok")
        rate = ok / len(dec)
        if rate >= lift_rate:
            r["lifted"], r["lifted_at"], r["lifted_rate"] = True, time.time(), round(rate, 3)
            out[name] = "lifted"
            event("blocklist_lifted", rule=name, ok=ok, n=len(dec), rate=round(rate, 3))
        else:
            r.setdefault("kept", []).append(dict(ts=time.time(), ok=ok, n=len(dec), rate=round(rate, 3)))
            out[name] = "kept"
            event("blocklist_kept", rule=name, ok=ok, n=len(dec), rate=round(rate, 3))
    if out or upgraded:
        save_state(store, st)
    return out
