"""Deterministic leak gates (no LLM).  Used by S7 and by the back-test on old tasks.

L1  text / isolation
    a. depgraph mode: names and values of derive/internal/hidden_truth nodes must not appear in the
       instruction or in public text files.
    b. forbidden-public rule: the instruction must not declare any public file/object off-limits
       ("do not read X", "opaque", "internal state", "black box ... blob").  A file the agent is told not to
       read is still readable (the agent is root in the container): rule-based protection is treated as a leak
       of whatever that file carries.  INTERACT tasks with in-container oracles fail here by construction.
    c. hidden-truth files present under environment/ (wrong place) or truth file names reused in public data.
L2  numeric scan: every truth leaf (tests/truth/**) is searched in every public leaf (environment/data/**):
    strict value containment (9 significant digits, abs) and permutation / sign / scale equality of
    equal-size arrays (sorted, rtol 1e-5).  "Echo" matches are ignored: the public leaf is documented in the
    instruction (backticked identifier) and carries the same name as the truth leaf, or the truth leaf name is
    documented too (an input that the grader keeps a copy of).
L3  introspection (sandboxed, see introspect.py): import every public .py, instantiate its classes for every
    problem id, walk attributes (private, closures, globals); the harvested numbers go through L2.
L4a join/rank probe: for truth tables keyed like a public table, rank-correlate each truth column with every
    public column and every pairwise product/ratio of public columns (first/last row per key).  |Spearman| >=
    0.995 on >=10 keys -> "invertible" (catches the design-covariance case).
L4b cheap-agent inversion attack: not implemented (model role 待定); reported as "skipped".
"""
from __future__ import annotations

import itertools
from pathlib import Path
import math
import re
from pathlib import Path

import numpy as np

from .leaves import Leaf, csv_tables, dir_leaves

FORBID = re.compile(
    r"(?:do not|don't|must not|must never|never|may not|not allowed to|is not allowed|are not allowed|forbidden to)"
    r"(?![^.\n]{0,40}\boutside\b)[^.\n]{0,60}?(?:read|open|load|decode|inspect|access|parse|reimplement|re-implement)"
    r"[^.\n]{0,60}?(?:`[^`]+/`|`[^`]+\.(?:npz|npy|bin|pkl|json|h5)`|\bblobs?\b|\bparams/|\bmachines/|internal|private"
    r"|underscore|\bit\b)"
    r"|(?:opaque (?:simulator |oracle )?(?:payload|blob|state|object|definitions?|storage|file)s?)"
    r"|(?:(?:it|blob|state|payload|file|storage) is opaque)"
    r"|(?:internal (?:machine |oracle |simulator )?(?:state|storage|parameters))"
    r"|(?:treat (?:the |these |it |them )?[^.\n]{0,40}as (?:a )?black[- ]box)"
    r"|(?:underscore (?:methods|attributes))|(?:private attributes)", re.I)
TICK = re.compile(r"`([^`\n]{1,80})`")


def declared_names(instruction: str) -> set[str]:
    out = set()
    for m in TICK.finditer(instruction):
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1)):
            if len(tok) >= 2 or tok.isalpha():
                out.add(tok.lower())
    return out


def _informative(v: np.ndarray) -> np.ndarray:
    v = v[np.isfinite(v)]
    v = v[np.abs(v) > 1e-12]
    keep = np.abs(v - np.round(v, 3)) > 1e-9 * np.maximum(1.0, np.abs(v))   # > 3 decimals of information
    return v[keep]


def _key9(v: np.ndarray) -> np.ndarray:
    a = np.abs(v)
    e = np.floor(np.log10(a))
    return np.round(a / 10.0 ** e, 8) * 10.0 ** e


class PublicIndex:
    def __init__(self, leaves: list[Leaf]):
        self.leaves = leaves
        keys, owner = [], []
        for i, lf in enumerate(leaves):
            inf = _informative(lf.values)
            if inf.size:
                k = np.unique(_key9(inf))
                keys.append(k)
                owner.append(np.full(k.size, i))
        self.keys = np.concatenate(keys) if keys else np.zeros(0)
        self.owner = np.concatenate(owner) if owner else np.zeros(0, int)
        self.uniq = np.unique(self.keys)
        self.by_size: dict[int, list[int]] = {}
        for i, lf in enumerate(leaves):
            if lf.values.size >= 3:
                self.by_size.setdefault(lf.values.size, []).append(i)

    def contain(self, t: Leaf):
        inf = _informative(t.values)
        if inf.size == 0:
            return None
        k = np.unique(_key9(inf))
        hit = np.isin(k, self.uniq)
        cov = float(hit.mean())
        if cov < 0.9:
            return None
        owners = np.unique(self.owner[np.isin(self.keys, k[hit])])
        return dict(kind="contain", coverage=round(cov, 3), n_values=int(k.size), public=[int(o) for o in owners[:8]])

    def permuted(self, t: Leaf):
        v = t.values
        if v.size < 3 or not np.all(np.isfinite(v)) or np.unique(v).size < 3:
            return None
        sv = np.sort(v)
        out = []
        for j in self.by_size.get(v.size, []):
            p = self.leaves[j].values
            if not np.all(np.isfinite(p)):
                continue
            sp = np.sort(p)
            for tag, a, b in (("perm", sv, sp), ("neg", sv, np.sort(-p)),
                              ("scale", sv / (np.max(np.abs(sv)) or 1), sp / (np.max(np.abs(sp)) or 1))):
                if np.allclose(a, b, rtol=1e-5, atol=1e-12):
                    out.append(dict(kind=tag, public=[j]))
                    break
        return out[0] if out else None


def numeric_scan(truth: list[Leaf], public: list[Leaf], declared: set[str], max_findings: int = 50,
                 echo_same_name: bool = True):
    """Returns (findings, n_echoes).  finding["severity"]: "block" | "warn".
    Ignored as echoes: a public leaf with the same normalised name carries the value (truth dirs keep copies of
    inputs).  Blocking: value containment (>=3 informative values, or scalar equal to a *scalar* public leaf) and
    permutation/sign matches of non-integer arrays under a different name.  Warn: scale-only matches."""
    idx = PublicIndex(public)
    findings, echoes = [], 0
    for t in truth:
        m = idx.contain(t) or idx.permuted(t)
        if not m:
            continue
        pubs = [idx.leaves[j] for j in m["public"]]
        if echo_same_name and any(p.name == t.name for p in pubs):
            echoes += 1
            continue
        # whole-array copy of a documented public input kept in the truth dir (e.g. truth/public_inputs.npz:bp ==
        # bandpowers.npy): echo.  An undocumented public array identical to a truth array stays a finding.
        if any(p.values.size == t.values.size and (p.name in declared or _file_declared(p.source, declared))
               for p in pubs) and m["kind"] in ("contain", "perm"):
            echoes += 1
            continue
        sev = "block"
        if m["kind"] == "contain" and m["n_values"] < 3:
            # a lone scalar inside a big public table is usually coincidence; equal to a public *scalar* is not
            if not any(p.values.size <= 3 for p in pubs):
                continue
        if m["kind"] in ("perm", "neg", "scale") and np.allclose(t.values, np.round(t.values)):
            continue                      # ids / counts / integer grids
        if m["kind"] == "scale":
            sev = "warn"
        # documented public carrier (column / key / file named in the instruction): the truth dir keeps a copy
        # of a stated input -> review, not block.  Undocumented carriers (machines/*.npz, blobs) stay blocking.
        if any(p.name in declared or (p.raw_name or "").lower() in declared or _file_declared(p.source, declared)
               for p in pubs):
            sev = "warn"
        # a few truth values found inside a much larger public array (cut-offs equal to observed values etc.)
        if m["kind"] == "contain" and all(p.values.size > 20 * max(1, m["n_values"]) for p in pubs):
            sev = "warn"
        findings.append(dict(severity=sev, truth=f"{t.source}:{t.raw_name or t.name}", truth_name=t.name,
                             match=m["kind"], coverage=m.get("coverage"), n_values=m.get("n_values", int(t.values.size)),
                             public=[f"{p.source}:{p.raw_name or p.name}" for p in pubs][:4],
                             public_declared=[p.name in declared for p in pubs][:4]))
        if len(findings) >= max_findings:
            break
    return findings, echoes


GENERIC_TOK = {"app", "data", "output", "input", "tests", "csv", "tsv", "json", "jsonl", "npz", "npy", "txt", "gz",
               "dat", "bin", "pkl", "h5", "md", "py", "id"}


def _file_declared(source: str, declared: set[str]) -> bool:
    """file is named in the instruction: its stem (e.g. 'meta' for data/meta.csv) is a backticked token"""
    path = source.split(":")[0]
    stem = Path(path).name.split(".")[0].lower()
    return stem not in GENERIC_TOK and len(stem) >= 3 and stem in declared


ANCILLARY = re.compile(r"(?:^|_)(?:var|variance|cov|covariance|se|sd|std|stderr|sigma|err|error|weight|w|n|count|"
                       r"nobs|size|deff|id|index|idx|seed|key|order|rank|row|rows)(?:_|$|\d)", re.I)


def _ancillary(feature: str) -> bool:
    parts = re.split(r"[*/]", feature)
    return all(ANCILLARY.search(p) for p in parts)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return 0.0
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def join_probe(truth_tables: dict, public_tables: dict, rho_min: float = 0.995, max_cols: int = 16):
    findings = []
    for tname, trows in truth_tables.items():
        if len(trows) < 10:
            continue
        tcols = list(trows[0].keys())
        for pname, prows in public_tables.items():
            if not prows:
                continue
            pcols = list(prows[0].keys())
            for key in [c for c in tcols if c in pcols]:
                tkeys = {r[key] for r in trows}
                pkeys = {r[key] for r in prows}
                if len(tkeys) < 10 or len(tkeys & pkeys) < 0.8 * len(tkeys):
                    continue
                if all(not math.isnan(_fnum(k)) for k in list(tkeys)[:20]):
                    continue   # numeric "keys" (ids that are numbers / measurements) are not join keys
                groups: dict[str, list[dict]] = {}
                for r in prows:
                    groups.setdefault(r[key], []).append(r)
                num_p = [c for c in pcols if c != key and
                         np.mean([not math.isnan(_fnum(r[c])) for r in prows[:200]]) > 0.5][:max_cols]
                num_t = [c for c in tcols if c != key and
                         np.mean([not math.isnan(_fnum(r[c])) for r in trows[:200]]) > 0.5]
                ks = [r[key] for r in trows if r[key] in groups]
                for pos in ("last", "first"):
                    pick = {k: groups[k][-1 if pos == "last" else 0] for k in ks}
                    P = {c: np.array([_fnum(pick[k][c]) for k in ks]) for c in num_p}
                    feats = dict(P)
                    for a, b in itertools.combinations(num_p, 2):
                        with np.errstate(all="ignore"):
                            feats[f"{a}*{b}"] = P[a] * P[b]
                            feats[f"{a}/{b}"] = P[a] / P[b]
                            feats[f"{b}/{a}"] = P[b] / P[a]
                    tv = {r[key]: r for r in trows}
                    for c in num_t:
                        y = np.array([_fnum(tv[k][c]) for k in ks])
                        fy = y[np.isfinite(y)]
                        if np.unique(fy).size < 10 or np.allclose(fy, np.round(fy)):
                            continue          # constant-ish or integer (index / count) truth columns
                        # same-named public column = the truth table keeps a copy of an input
                        cand = {n: f for n, f in feats.items() if c not in re.split(r"[*/]", n)}
                        rho, fname = max(((abs(_spearman(y, f)), n) for n, f in cand.items()),
                                         default=(0.0, ""))
                        if rho >= rho_min:
                            findings.append(dict(truth=f"{tname}:{c}", public=f"{pname}[{pos} per {key}]",
                                                 feature=fname, rho=round(rho, 4),
                                                 severity="block" if _ancillary(fname) else "warn"))
    # one finding per truth column (best)
    best: dict[str, dict] = {}
    for f in findings:
        if f["truth"] not in best or f["rho"] > best[f["truth"]]["rho"]:
            best[f["truth"]] = f
    return list(best.values())
