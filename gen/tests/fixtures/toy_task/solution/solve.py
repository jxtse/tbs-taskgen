import csv, os
import numpy as np


def load(path="/app/data/series.csv"):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d.setdefault(r["series_id"], ([], []))
            d[r["series_id"]][0].append(float(r["t"]))
            d[r["series_id"]][1].append(float(r["y"]))
    return {k: (np.array(a), np.array(b)) for k, (a, b) in d.items()}


def wfit(t, y, w):
    X = np.c_[np.ones_like(t), -t]
    ly = np.log(np.clip(y, 1e-9, None))
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(X * sw[:, None], ly * sw, rcond=None)
    return float(np.exp(c[0])), float(c[1])


def write(est, out="/app/output/estimates.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["series_id", "amp", "rate"])
        for k, (A, r) in sorted(est.items()):
            w.writerow([k, f"{A:.8g}", f"{r:.8g}"])


def robust(t, y):
    # C2: one-sided spike rejection.  Local test against the running median of neighbours, then refit/iterate.
    n = t.size
    med = np.array([np.median(y[max(0, i - 2):min(n, i + 3)]) for i in range(n)])
    keep = y <= med * 1.15 + 0.002 * np.max(med)
    for _ in range(6):
        A, k = wfit(t[keep], y[keep], np.clip(y[keep], 1e-9, None) ** 2)   # C1: weights for log transform
        r = y - A * np.exp(-k * t)
        s = 1.4826 * np.median(np.abs(r[keep])) + 1e-12
        new = r < 4 * s
        if (new == keep).all():
            break
        keep = new
    return A, k


write({sid: robust(t, y) for sid, (t, y) in load().items()})
