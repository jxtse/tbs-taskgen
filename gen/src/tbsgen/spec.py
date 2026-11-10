"""S5 spec validation (rules) and S6B thresholds-from-measurement (no model).

Spec rules:
  * form in cfg.spec.allowed_forms; INTERACT/DESIGN/CODE/MULTI/PROOF rejected in MVP
  * depgraph: unique ids; edges reference nodes; >=1 deliverable; mid_drop = exactly 1 key derive/compute node;
    naive_drop >= 2 nodes and contains mid_drop; >=1 hidden_truth node; PROG needs a shift node + hidden metrics
  * 3..8 metrics with direction <= / >=
Thresholds (S6B): for each metric, from per-seed values of ref and mid:
  * worst ref value over seeds = r, best mid value over seeds = m.
  * separating metric (mid worse than ref on every seed by > min_ref_margin relative):
      threshold = geometric (or linear if signs differ / zero) interpolation at cfg.thresholds.frac between r and m
  * other metrics: threshold = r loosened by loose_margin (reference must pass with margin; mid passes too)
  * the mid baseline must fail between min_mid_fail and max_mid_fail metrics; otherwise "not_separable"
    -> back to S5 once with a different mid_drop node.
"""
from __future__ import annotations

import math
import re

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
    if "resource_metric_gating" in spec and not isinstance(spec.get("resource_metric_gating"), bool):
        errs.append("resource_metric_gating must be a boolean")
    return errs


def metric_family(name: str) -> str:
    """public_rmse / hidden_rmse / rmse_public / rmse_hidden -> rmse"""
    n = re.sub(r"^(public|hidden)_", "", name)
    return re.sub(r"_(public|hidden)$", "", n)


# Resource metrics (wall time, memory): an accurate reference is legitimately slower / bigger than a trivial naive
# baseline, so "reference worse than naive" does not mean the metric fails to measure the method (e2e-prod-20261109:
# 2/2 specs had a hidden runtime metric, ref 0.58 s vs naive 0.40 s, and the item was dropped after 2 S8 rounds that
# could not make the reference faster than the shortcut).  Name-based; a spec may also mark {"kind": "resource"}.
# Unambiguous compute-resource tokens only.  Units and generic words ("seconds", "time_s", "latency", "elapsed",
# "rss" = residual sum of squares, "mb", "mem") are left out on purpose: a quality metric such as a delay-estimation
# error in seconds misread as a resource metric would lose this check.  Anything else: declare kind="resource".
RESOURCE_METRIC = re.compile(r"(?:^|_)(?:runtime|wall_?time|walltime|cpu_time|peak_memory|peak_rss|max_rss|peak_mem)"
                             r"(?:_|$)", re.I)


def is_resource_metric(m: dict) -> bool:
    return m.get("kind") == "resource" or bool(RESOURCE_METRIC.search(m.get("name", "")))


def _noise(a: float, b: float, floor_abs: float) -> bool:
    """Both values below the absolute floating-point noise floor (thresholds.noise_floor_abs): their difference is
    rounding, not a difference in method.  This drops ref/mid separation whenever both sit below the floor, even if
    naive is above it -- a metric whose natural scale is that small needs noise_floor_abs = 0 (or smaller)."""
    return abs(a) < floor_abs and abs(b) < floor_abs


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
    floor_abs = float(tc.get("noise_floor_abs", 1e-9))
    out, sep = {}, []
    resource = {m["name"] for m in spec["metrics"] if is_resource_metric(m)}
    # spec.resource_metric_gating (default false): wall time / memory are recorded but never decide a run, so
    # they cannot separate the mid baseline either (only speed-contribution papers gate on them)
    gating = bool(spec.get("resource_metric_gating"))
    ungated = resource if not gating else set()
    for m in spec["metrics"]:
        name, d = m["name"], m["direction"]
        r_vals = [v.get(name) for v in measure["ref"].values()]
        m_vals = [v.get(name) for v in measure["mid"].values()]
        if name in ungated and any(x is None or not math.isfinite(x) for x in r_vals):
            fin = [x for x in r_vals if x is not None and math.isfinite(x)]
            if not fin:                     # recorded-only: no value at all -> no threshold, decides nothing
                out[name] = dict(direction=d, threshold=None, error="reference metric missing/non-finite",
                                 ungated=True)
                continue
            r_vals = fin
        if any(x is None or not math.isfinite(x) for x in r_vals):
            out[name] = dict(direction=d, threshold=None, error="reference metric missing/non-finite")
            continue
        r = max(r_vals) if d == "<=" else min(r_vals)
        # v2.3: the reference sitting exactly on the metric's empty-output value (measured on nop) means it did not
        # produce that deliverable (e2e-iter1 IEnKS: ref filter_rmse_public = 1e6 -> threshold 1.5e6).  Works in
        # both directions, for negative / zero / large-scale metrics (review #7).
        nop_vals = [v.get(name) for v in measure.get("nop", {}).values()]
        hit = [x for x in r_vals for nv in nop_vals if nv is not None and math.isfinite(nv) and
               math.isclose(x, nv, rel_tol=1e-12, abs_tol=1e-15)]
        if hit and name not in ungated:      # a recorded-only resource value may equal nop's (e.g. same RSS)
            out[name] = dict(direction=d, threshold=None,
                             error=f"reference value {hit[0]:g} equals the empty-output value of this metric")
            continue
        mm = [x for x in m_vals if x is not None and math.isfinite(x)]
        m_best = (min(mm) if d == "<=" else max(mm)) if mm else None
        margin = tc["min_ref_margin"]
        naive_capped = False
        n_best = None
        separating = m_best is not None and _worse(d, m_best, r) and \
            abs(m_best - r) > margin * max(abs(r), 1e-12) and not _noise(r, m_best, floor_abs)
        if name in ungated:
            separating = False                 # recorded only: never decides a run, so it separates nothing
        if separating:
            if r > 0 and m_best > 0:
                thr = math.exp(math.log(r) + tc["frac"] * (math.log(m_best) - math.log(r)))
            else:
                thr = r + tc["frac"] * (m_best - r)
            sep.append(name)
        else:
            # resource metrics (wall time / memory) are noisy run to run (e2e-prod-20261109: reference 0.58 s in S6A,
            # 0.93 s when S7 re-measured it -> threshold ref x1.5 = 0.88 failed the oracle gate): wider margin
            lm = tc.get("resource_loose_margin", 2.0) if name in resource else tc["loose_margin"]
            thr = r * (1 + lm) if (d == "<=") == (r >= 0) else r * (1 - lm)
            if r == 0:
                thr = 1e-9 if d == "<=" else -1e-9
            # v2.1: a non-separating metric must still reject the naive baseline when the naive is clearly worse
            # (e2e-iter1: oracle-relative RMSE ratio, ref 1.00, naive 1.48-1.58, loose threshold 1.6 let naive pass)
            nv = [v.get(name) for v in measure.get("naive", {}).values()]
            nv = [x for x in nv if x is not None and math.isfinite(x)]
            n_best = (min(nv) if d == "<=" else max(nv)) if nv else None
            # v2.2: also when naive is worse by less than min_ref_margin (e2e-iter1: Pearson ref 0.996, naive 0.89,
            # loose rule gave 0.49), but never tighter than the reference worsened by min_ref_margin
            if n_best is not None and _worse(d, n_best, r):
                mid_n = math.exp(math.log(r) + tc["frac"] * (math.log(n_best) - math.log(r))) \
                    if r > 0 and n_best > 0 else r + tc["frac"] * (n_best - r)
                fm = tc.get("resource_loose_margin", 2.0) if name in resource else margin
                floor = r * (1 + fm) if (d == "<=") == (r >= 0) else r * (1 - fm)
                if _worse(d, floor, mid_n):          # midpoint tighter than the reference margin allows
                    mid_n = floor
                if _worse(d, thr, mid_n):
                    thr, naive_capped = mid_n, True
        # raw values (None / inf / nan included): a missing or non-finite value is not "at the floor"
        all_vals = r_vals + m_vals + [v.get(name) for v in measure.get("naive", {}).values()]
        if not separating and d == "<=" and all(x is not None and math.isfinite(x) and 0 <= x < floor_abs
                                                 for x in all_vals):
            # every measured solution (ref, mid, naive) sits at the floating-point noise floor of an error metric
            # (e2e-prod-20261109: ref 4.3e-11, mid 2.7e-11, naive 4.3e-11): the metric separates nothing, and ref x1.5
            # would fail a correct implementation that merely rounds differently
            thr, naive_capped = floor_abs, False
            at_floor = True
        else:
            at_floor = False
        exact = thr
        thr = exact if at_floor else _round_sig(exact, tc["sig_digits"], d, toward_loose=True)   # floor: kept exact
        if separating and not _worse(d, m_best, thr):   # rounding must not let the mid baseline pass
            thr = _round_sig(exact, tc["sig_digits"] + 2, d, toward_loose=False)
        if naive_capped and not _worse(d, n_best, thr):  # nor the naive baseline (review #9)
            for extra in (1, 2, 3):
                thr = _round_sig(exact, tc["sig_digits"] + extra, d, toward_loose=True)
                if _worse(d, n_best, thr):
                    break
            else:
                thr = exact
        ok_ref = not _worse(d, r, thr)
        out[name] = dict(direction=d, threshold=thr, ref_worst=r, mid_best=m_best, separating=separating,
                         ref_passes=ok_ref, naive_capped=naive_capped)
    # v2.2: the 1..2 "near miss" rule counts failed criteria, not metric names: the same quantity measured on public
    # and on hidden data (public_X / hidden_X / X_public ...) is one criterion (e2e-iter1 RCESPP: mid failed
    # public_optimal_frac + hidden_optimal_frac + hidden_mean_rel_gap = 2 criteria, was counted as 3 and dropped)
    mid_fail = {s: len({metric_family(k) for k, t in out.items() if k not in ungated
                        and t.get("threshold") is not None
                        and v.get(k) is not None and _worse(t["direction"], v[k], t["threshold"])})
                for s, v in measure["mid"].items()}
    naive_fail = {s: sum(1 for k, t in out.items() if k not in ungated and t.get("threshold") is not None
                         and v.get(k) is not None
                         and _worse(t["direction"], v[k], t["threshold"])) + sum(1 for k in out if k not in v
                                                                                 and k not in ungated)
                  for s, v in measure.get("naive", {}).items()}
    # v2.2: the reference must not be clearly worse than the naive baseline on any metric; otherwise that metric
    # does not measure the method (e2e-iter1 IEnKS: on the hidden regime the reference diverged, filter RMSE 4.19
    # vs naive 3.39-3.52, and the loose rule still produced thresholds that both passed)
    ref_worse = []
    for name, t in out.items():
        if t.get("threshold") is None or name in resource:
            continue
        nv = [v.get(name) for v in measure.get("naive", {}).values()]
        nv = [x for x in nv if x is not None and math.isfinite(x)]
        if nv:
            n_best = min(nv) if t["direction"] == "<=" else max(nv)
            if _worse(t["direction"], t["ref_worst"], n_best) and \
                    abs(t["ref_worst"] - n_best) > tc["min_ref_margin"] * max(abs(n_best), 1e-12) and \
                    not _noise(t["ref_worst"], n_best, floor_abs):     # both at the noise floor: not "worse"
                ref_worse.append(name)
    lo, hi = tc["min_mid_fail"], tc["max_mid_fail"]
    status = "ok"
    decides = {k: t for k, t in out.items() if k not in ungated}     # recorded-only metrics decide nothing
    if any(t.get("threshold") is None for t in decides.values()):
        status = "ref_metric_missing"
    elif ref_worse:
        status = "ref_worse_than_naive"
    elif not all(t["ref_passes"] for t in decides.values()):
        status = "ref_fails"
    elif not mid_fail or not all(lo <= n <= hi for n in mid_fail.values()):
        status = "not_separable"
    return dict(status=status, thresholds=out, separating=sep, mid_fail=mid_fail, naive_fail=naive_fail,
                ref_worse_than_naive=ref_worse, resource_metric_gating=gating,
                ungated_resource=sorted(ungated))


# ------------------------------------------------------------------------------------------------ S6B early stops
def crashed_runs(measure: dict, who: str) -> dict:
    """{run tag: stderr tail} of `who` (mid / naive / ref) runs that exited non-zero (crash / timeout)."""
    return {k: (v.get("err") or "")[-400:] for k, v in (measure.get("runs") or {}).items()
            if re.fullmatch(rf"{who}\d+", k) and v.get("solve_rc") not in (0, None)}


def ref_naive_ratios(th: dict, measure: dict, cfg: dict) -> dict:
    """For each metric flagged ref_worse_than_naive: how many times worse the reference is than the best naive seed
    (error metrics "<=": ref / naive; score metrics ">=": naive / ref).  Metrics where the denominator is at the
    noise floor or the values have mixed signs are left out (no meaningful ratio: e.g. naive FDR 0 because the
    naive baseline selects nothing)."""
    floor_abs = float(cfg["thresholds"].get("noise_floor_abs", 1e-9))
    out = {}
    for name in th.get("ref_worse_than_naive") or []:
        t = th["thresholds"][name]
        nv = [v.get(name) for v in (measure.get("naive") or {}).values()]
        nv = [x for x in nv if x is not None and math.isfinite(x)]
        if not nv:
            continue
        r = t["ref_worst"]
        if t["direction"] == "<=":
            n = min(nv)
            num, den = r, n
        else:
            n = max(nv)
            num, den = n, r
        if den is None or num is None or den <= floor_abs or num <= 0:
            continue
        out[name] = num / den
    return out


def s6b_decision(th: dict, measure: dict, cfg: dict, state: dict) -> dict:
    """What S6B does with a thresholds report: dict(action, reason, rule, detail).

    action: "S6C" (ok) | "S8" (repair) | "S6M" (rewrite mid only) | "S5" (new spec) | "S6A" (regenerate code) |
    "drop".  rule is set when an early-stop rule decided (also logged as an `early_stop` event):
      ref_gross      reference worse than naive by >= gen.ref_gross_ratio -> S6A once, then drop
      mid_crashed    mid exited non-zero -> S6M (code bug, the count of failed criteria means nothing)
      mid_far        mid fails far more criteria than allowed -> drop without S6M / S5
    Counters (state): mid_rework, back_to_spec, ref_rewrite.
    A crashed mid only matters when the item is not separable: a mid that wrote its outputs and then exited
    non-zero but still fails 1..max_mid_fail criteria is a valid near miss (status ok -> S6C, as before)."""
    g = cfg["gen"]
    st = th["status"]
    if st == "ok":
        return dict(action="S6C", reason="ok", rule=None, detail={})
    if st == "ref_worse_than_naive":
        ratios = ref_naive_ratios(th, measure, cfg)
        gross = {k: v for k, v in ratios.items() if v >= float(g.get("ref_gross_ratio", 10.0))}
        detail = dict(ratios={k: float(f"{v:.3g}") for k, v in ratios.items()}, gross_ratio=g.get("ref_gross_ratio"))
        if gross:
            if state.get("ref_rewrite", 0) < int(g.get("ref_rewrite_max", 1)):
                return dict(action="S6A", reason="ref_grossly_wrong", rule="ref_gross", detail=detail)
            return dict(action="drop", reason="ref_grossly_wrong", rule="ref_gross", detail=detail)
        return dict(action="S8", reason=st, rule=None, detail=detail)
    if st != "not_separable":
        return dict(action="S8", reason=st, rule=None, detail={})
    lo, hi = cfg["thresholds"]["min_mid_fail"], cfg["thresholds"]["max_mid_fail"]
    n_crit = len({metric_family(k) for k in th["thresholds"]})
    fails = list((th.get("mid_fail") or {}).values())
    crashed = crashed_runs(measure, "mid")
    detail = dict(mid_fail=th.get("mid_fail"), criteria=n_crit, allowed=[lo, hi], mid_crashed=sorted(crashed))
    rework_left = state.get("mid_rework", 0) < g.get("mid_rework_max", 1)
    if crashed:
        if rework_left:
            return dict(action="S6M", reason="mid_crashed", rule="mid_crashed", detail=detail)
    elif fails:
        n = min(fails)          # every seed must be far: one seed near the band keeps the item
        far = n >= hi + int(g.get("mid_far_above", 2)) or (n >= hi + 1 and n >= n_crit)
        if far:
            return dict(action="drop", reason="mid_far_from_ref", rule="mid_far", detail=detail)
    if rework_left:
        return dict(action="S6M", reason="mid_not_separable", rule=None, detail=detail)
    # back to S5 (new spec + all code) only for a near miss: mid ran, failed at least one criterion and at most one
    # more than allowed.  A mid that fails nothing after S6M (the dropped steps do not matter) or far too much is
    # a content verdict, not a near miss.
    near = not crashed and bool(fails) and min(fails) >= 1 and max(fails) <= hi + 1
    if near and state.get("back_to_spec", 0) < g["back_to_spec_max"]:
        return dict(action="S5", reason="mid_not_separable", rule=None, detail=detail)
    return dict(action="drop", reason="mid_not_separable", rule=None if near or not g["back_to_spec_max"]
                else "not_near_miss", detail=detail)
