"""Toy fixture: decaying signals with sparse positive spikes.  python generator/generate.py <seed>"""
import csv, os, sys
import numpy as np

seed = int(sys.argv[1])
rng = np.random.default_rng(1000 + seed)
os.makedirs("environment/data", exist_ok=True)
os.makedirs("tests/truth", exist_ok=True)
os.makedirs("tests/hidden", exist_ok=True)
t = np.round(np.linspace(0.5, 6.0, 36), 4)
rows, truth = [], []
for i in range(48):
    sid = f"S{i:03d}"
    A = float(rng.uniform(2.0, 20.0))
    k = float(rng.uniform(0.15, 0.9))
    y = A * np.exp(-k * t) + rng.normal(0, 0.01 * A, t.size)
    spikes = rng.random(t.size) < 0.12
    y[spikes] += A * rng.uniform(0.3, 0.8, spikes.sum())
    rows += [(sid, f"{tt:.4f}", f"{yy:.6g}") for tt, yy in zip(t, y)]
    truth.append((sid, f"{A:.9g}", f"{k:.9g}"))
with open("environment/data/series.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series_id", "t", "y"]); w.writerows(rows)
with open("tests/truth/params.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series_id", "amp", "rate"]); w.writerows(truth)
