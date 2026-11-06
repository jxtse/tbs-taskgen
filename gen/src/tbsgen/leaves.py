"""Leaf extraction for the leak gates: turn files / Python objects into named numeric leaves.

A leaf is (source, name, values: np.ndarray[float]).  Names are normalised (problem-id prefixes like
"P01_" / "H03." stripped) so "same name, same value" echoes (x0, lower, budget copied into truth) can be told
apart from answer-bearing values that sit under a different name in a public file.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_FILE_MB = 200
MAX_LEAVES = 200000
TEXT_EXT = {".txt", ".dat", ".tsv", ".csv"}


@dataclass
class Leaf:
    source: str          # relative file path (or "introspect:<module>")
    name: str            # dotted key path, normalised
    values: np.ndarray   # float64, 1-D
    raw_name: str = ""


_PID = re.compile(r"^(?:[A-Za-z]{1,3}\d{1,4}|p\d+|problem_?\d+|\d+)(?:[_.\-/]|$)", re.I)


def norm_name(path: str) -> str:
    parts = [p for p in re.split(r"[./\[\]]+", path) if p and not p.isdigit()]
    out = []
    for p in parts:
        q = _PID.sub("", p)
        if q:
            out.append(q.lower())
    return out[-1] if out else ""


def _num_list(x):
    try:
        a = np.asarray(x, dtype=float).ravel()
        return a if a.size else None
    except (TypeError, ValueError):
        return None


def walk_json(obj, prefix: str, out: list[tuple[str, np.ndarray]]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_json(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            out.append((prefix, np.asarray(obj, float)))
        elif obj and all(isinstance(x, list) for x in obj):
            a = _num_list(obj) if all(all(isinstance(y, (int, float)) for y in x) for x in obj) else None
            if a is not None:
                out.append((prefix, a))
            else:
                for i, x in enumerate(obj):
                    walk_json(x, f"{prefix}[{i}]", out)
        else:
            for i, x in enumerate(obj):
                walk_json(x, f"{prefix}[{i}]", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append((prefix, np.asarray([obj], float)))


def _csv_leaves(text: str, delim: str):
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if len(rows) < 2:
        return []
    head = rows[0]
    has_head = any(re.search(r"[A-Za-z]", h or "") for h in head)
    if not has_head:
        head = [f"c{i}" for i in range(len(rows[0]))]
        body = rows
    else:
        body = rows[1:]
    out = []
    for j, h in enumerate(head):
        col = []
        for r in body:
            v = r[j] if j < len(r) else ""
            try:
                col.append(float(v) if v not in ("", "NA", "nan", "NaN") else np.nan)
            except ValueError:
                col = None
                break
        if col is not None and col:
            out.append((h, np.asarray(col, float)))
    return out


def file_leaves(path: Path, rel: str) -> list[Leaf]:
    """Numeric leaves of one file.  Unknown / binary formats -> []."""
    try:
        if path.stat().st_size > MAX_FILE_MB * 1e6:
            return []
    except OSError:
        return []
    suf = "".join(path.suffixes[-2:]).lower() if path.suffix == ".gz" else path.suffix.lower()
    pairs: list[tuple[str, np.ndarray]] = []
    try:
        if suf in (".npz",):
            with np.load(path, allow_pickle=False) as z:
                for k in z.files:
                    a = z[k]
                    if a.dtype.kind in "fiub":
                        pairs.append((k, a.astype(float).ravel()))
        elif suf == ".npy":
            a = np.load(path, allow_pickle=False)
            if a.dtype.kind in "fiub":
                pairs.append(("", a.astype(float).ravel()))
                if a.ndim == 2 and a.shape[1] <= 64:
                    pairs += [(f"col{j}", a[:, j].astype(float)) for j in range(a.shape[1])]
        elif suf in (".json", ".jsonl", ".json.gz"):
            raw = gzip.decompress(path.read_bytes()) if suf.endswith(".gz") else path.read_bytes()
            txt = raw.decode("utf-8", "replace")
            objs = [json.loads(x) for x in txt.splitlines() if x.strip()] if suf == ".jsonl" else [json.loads(txt)]
            for o in objs:
                walk_json(o, "", pairs)
        elif suf in (".csv", ".tsv", ".csv.gz", ".tsv.gz", ".txt", ".dat"):
            raw = gzip.decompress(path.read_bytes()) if suf.endswith(".gz") else path.read_bytes()
            txt = raw.decode("utf-8", "replace")
            delim = "\t" if "tsv" in suf or ("\t" in txt[:2000] and "," not in txt[:2000]) else ","
            if suf in (".txt", ".dat") and delim == "," and "," not in txt[:2000]:
                delim = " "
                txt = re.sub(r"[ \t]+", " ", txt)
            pairs = _csv_leaves(txt, delim)
    except Exception:  # noqa: BLE001  (corrupt / exotic file: no leaves; the gate reports unreadable files)
        return []
    return [Leaf(rel, norm_name(f"{rel}.{k}" if k else Path(rel).stem), v, k) for k, v in pairs if v.size]


def dir_leaves(root: Path, prefix: str = "") -> tuple[list[Leaf], list[str]]:
    leaves, unread = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = prefix + str(p.relative_to(root))
        got = file_leaves(p, rel)
        if got:
            leaves += got
        elif p.suffix.lower() not in (".py", ".md", ".sh", ".toml", ".yaml", ".yml"):
            unread.append(rel)
        if len(leaves) > MAX_LEAVES:
            break
    return leaves, unread


def csv_tables(root: Path) -> dict[str, list[dict]]:
    """Small CSV/TSV tables as list-of-rows (for the L4a join probe)."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".csv", ".tsv") and p.stat().st_size < 50e6:
            delim = "\t" if p.suffix.lower() == ".tsv" else ","
            try:
                with p.open(newline="", errors="replace") as f:
                    out[str(p.relative_to(root))] = list(csv.DictReader(f, delimiter=delim))
            except Exception:  # noqa: BLE001
                pass
    return out
