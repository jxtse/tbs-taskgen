"""S5 spec validation (rules) and S6B thresholds-from-measurement (no model).

Spec rules (DESIGN §2.3):
  * form in cfg.spec.allowed_forms; INTERACT/DESIGN/CODE/MULTI/PROOF rejected in MVP
  * depgraph: unique ids; edges reference nodes; >=1 deliverable; mid_drop = exactly 1 key derive/compute node;
    naive_drop >= 2 nodes and contains mid_drop; >=1 hidden_truth node; PROG needs a shift node + hidden metrics
  * 3..8 metrics with direction <= / >=
Thresholds (DESIGN §3.1 S6B): for each metric, from per-seed values of ref and mid:
  * worst ref value over seeds = r, best mid value over seeds = m.
  * separating metric (mid worse than ref on every seed by > min_ref_margin relative):
      threshold = geometric (or linear if signs differ / zero) interpolation at cfg.thresholds.frac between r and m
  * other metrics: threshold = r loosened by loose_margin (reference must pass with margin; mid passes too)
  * the mid baseline must fail between min_mid_fail and max_mid_fail metrics; otherwise "not_separable"
    -> back to S5 once with a different mid_drop node.
"""
from __future__ import annotations

import math

FORMS_ALL = {"ANS", "PROG", "DOMF", "CODE", "MULTI", "DESIGN", "INTERACT", "TRAJ", "PROOF"}


def validate_spec(spec: dict, cfg: dict) -> list[str]:
    errs = []
    form = spec.get("form")
    if form not in cfg["spec"]["allowed_forms"]:
        errs.append(f"form {form!r} not allowed in this phase")
    g = spec.get("depgraph") or {}
    nodes = g.get("nodes") or []
    ids = [n.get("id") for n in nodes]
    if len(set(ids)) != len(ids) or not ids:
        errs.append("depgraph node ids missing or duplicated")
    by = {n.get("id"): n for n in nodes}
    for e in g.get("edges") or []:
        if len(e) != 2 or e[0] not in by or e[1] not in by:
            errs.append(f"bad edge {e}")
    if not any(n.get("kind") == "deliverable" for n in nodes):
        errs.append("no deliverable node")
    if not any(n.get("provision") == "hidden_truth" for n in nodes):
        errs.append("no hidden_truth node")
    md = g.get("mid_drop") or []
    if len(md) != 1 or md[0] not in by or not (by[md[0]].get("kind") == "compute"
                                                  or by[md[0]].get("provision") == "derive"):
        errs.append("mid_drop must be exactly one derive/compute node")
    nd = g.get("naive_drop") or []
    if len(nd) < 2 or (md and md[0] not in nd) or any(x not in by for x in nd):
        errs.append("naive_drop must hold >=2 nodes including mid_drop")
    mets = spec.get("metrics") or []
    if not 3 <= len(mets) <= 8:
        errs.append("need 3..8 metrics")
    for m in mets:
        if m.get("direction") not in ("<=", ">=") or not m.get("name"):
            errs.append(f"bad metric {m}")
    if form == "PROG":
        if not any(n.get("kind") == "shift" for n in nodes):
            errs.append("PROG needs a shift node")
        if not any(m.get("on") == "hidden" for m in mets):
            errs.append("PROG needs hidden metrics")
    return errs


def _worse(direction: str, a: float, b: float) -> bool:
    """is a worse than b"""
    return a > b if direction == "<=" else a < b


def _round_sig(x: float, sig: int, direction: str, toward_loose: bool) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x))) - sig + 1
    q = 10.0 ** e
    f = math.ceil if (direction == "<=") == toward_loose else math.floor
    return round(f(x / q) * q, max(0, -e))


def set_thresholds(spec: dict, measure: dict, cfg: dict) -> dict:
    """measure = {"ref": {seed: {metric: value}}, "mid": {...}, "naive": {...}} -> thresholds report"""
    tc = cfg["thresholds"]
    out, sep = {}, []
    for m in spec["metrics"]:
        name, d = m["name"], m["direction"]
        r_vals = [v.get(name) for v in measure["ref"].values()]
        m_vals = [v.get(name) for v in measure["mid"].values()]
        if any(x is None or not math.isfinite(x) for x in r_vals):
            out[name] = dict(direction=d, threshold=None, error="reference metric missing/non-finite")
            continue
        r = max(r_vals) if d == "<=" else min(r_vals)
        mm = [x for x in m_vals if x is not None and math.isfinite(x)]
        m_best = (min(mm) if d == "<=" else max(mm)) if mm else None
        margin = tc["min_ref_margin"]
        separating = m_best is not None and _worse(d, m_best, r) and \
            abs(m_best - r) > margin * max(abs(r), 1e-12)
        if separating:
            if r > 0 and m_best > 0:
                thr = math.exp(math.log(r) + tc["frac"] * (math.log(m_best) - math.log(r)))
            else:
                thr = r + tc["frac"] * (m_best - r)
            sep.append(name)
        else:
            thr = r * (1 + tc["loose_margin"]) if (d == "<=") == (r >= 0) else r * (1 - tc["loose_margin"])
            if r == 0:
                thr = 1e-9 if d == "<=" else -1e-9
        thr = _round_sig(thr, tc["sig_digits"], d, toward_loose=True)
        if separating and not _worse(d, m_best, thr):   # rounding must not let the mid baseline pass
            thr = _round_sig(thr, tc["sig_digits"] + 2, d, toward_loose=False)
        ok_ref = not _worse(d, r, thr)
        out[name] = dict(direction=d, threshold=thr, ref_worst=r, mid_best=m_best, separating=separating,
                         ref_passes=ok_ref)
    mid_fail = {s: sum(1 for k, t in out.items() if t.get("threshold") is not None and v.get(k) is not None
                       and _worse(t["direction"], v[k], t["threshold"])) for s, v in measure["mid"].items()}
    naive_fail = {s: sum(1 for k, t in out.items() if t.get("threshold") is not None and v.get(k) is not None
                         and _worse(t["direction"], v[k], t["threshold"])) + sum(1 for k in out if k not in v)
                  for s, v in measure.get("naive", {}).items()}
    lo, hi = tc["min_mid_fail"], tc["max_mid_fail"]
    status = "ok"
    if any(t.get("threshold") is None for t in out.values()):
        status = "ref_metric_missing"
    elif not all(t["ref_passes"] for t in out.values()):
        status = "ref_fails"
    elif not mid_fail or not all(lo <= n <= hi for n in mid_fail.values()):
        status = "not_separable"
    return dict(status=status, thresholds=out, separating=sep, mid_fail=mid_fail, naive_fail=naive_fail)
