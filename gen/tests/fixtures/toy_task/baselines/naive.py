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

# naive: unweighted log-linear least squares, no spike rejection (drops C1 and C2)
write({sid: wfit(t, y, np.ones_like(t)) for sid, (t, y) in load().items()})
