import csv, json, os
import numpy as np


def metrics(out="/app/output/estimates.csv", truth="/tests/truth/params.csv"):
    T = {r["series_id"]: (float(r["amp"]), float(r["rate"])) for r in csv.DictReader(open(truth))}
    E = {}
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            try:
                E[r["series_id"]] = (float(r["amp"]), float(r["rate"]))
            except (KeyError, ValueError):
                pass
    ea, ek, cov = [], [], 0
    for sid, (A, k) in T.items():
        a, r = E.get(sid, (np.nan, np.nan))
        if np.isfinite(a) and np.isfinite(r):
            cov += 1
        ea.append(abs(a - A) / A if np.isfinite(a) else 1.0)
        ek.append(abs(r - k) / k if np.isfinite(r) else 1.0)
    ea, ek = np.array(ea), np.array(ek)
    return {"rate_median_rel_error": float(np.median(ek)), "amp_median_rel_error": float(np.median(ea)),
            "rate_p90_rel_error": float(np.quantile(ek, 0.9)), "coverage": cov / len(T)}
