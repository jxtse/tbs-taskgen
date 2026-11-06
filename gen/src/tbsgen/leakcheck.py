"""Leak gate driver: run L1-L4a on a task tree and return one report.

    leak_report(public_dir, truth_dirs, instruction, cfg, depgraph=None, introspect="docker"|"skip")

Verdict: "leak" if any blocking finding, else "clean".  Finding classes:
    L1a depgraph_text   value/name of a derive/internal/hidden_truth node found in instruction/public text
    L1b forbidden_public instruction declares a public file/object off-limits (rule-only protection)
    L1c truth_in_public  truth-like file placed in public data (name collision with tests/truth)
    L2  numeric          truth values found in public files (not a documented echo)
    L3  introspect       truth values reachable through public Python objects
    L4a invertible       truth column is a monotone function of public columns (rank corr >= threshold)
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np

from . import sandbox
from .leakgate import FORBID, declared_names, join_probe, numeric_scan
from .leaves import Leaf, csv_tables, dir_leaves, norm_name

PROBE = Path(__file__).with_name("introspect_probe.py")
TEXT_SUFFIX = {".md", ".txt", ".json", ".csv", ".tsv", ".py", ".yaml", ".yml", ".toml"}


def _public_text(public_dir: Path, limit=5_000_000) -> str:
    out, n = [], 0
    for p in sorted(public_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIX and p.stat().st_size < 2e6:
            t = p.read_text(errors="replace")
            out.append(t)
            n += len(t)
            if n > limit:
                break
    return "\n".join(out)


def l1_depgraph(depgraph: dict, instruction: str, public_text: str) -> list[dict]:
    out = []
    hay = (instruction + "\n" + public_text).lower()
    for node in depgraph.get("nodes", []):
        if node.get("provision") not in ("derive", "internal", "hidden_truth"):
            continue
        for k in ("value", "values"):
            v = node.get(k)
            vals = v if isinstance(v, list) else ([v] if v is not None else [])
            for x in vals:
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    s = f"{x:.6g}"
                    if len(s.replace(".", "").replace("-", "")) >= 4 and re.search(rf"(?<![\d.]){re.escape(s)}", hay):
                        out.append(dict(cls="L1a", node=node["id"], value=s))
                elif isinstance(x, str) and len(x) >= 6 and x.lower() in hay:
                    out.append(dict(cls="L1a", node=node["id"], value=x))
        for nm in node.get("secret_names", []):
            if re.search(rf"\b{re.escape(nm.lower())}\b", hay):
                out.append(dict(cls="L1a", node=node["id"], name=nm))
    return out


def l1_forbidden(instruction: str) -> list[dict]:
    hits = []
    for m in FORBID.finditer(instruction):
        s = instruction[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
        hits.append(dict(cls="L1b", text=s.strip()[:200]))
    return hits[:10]


def _forbidden_mentions(rel: str, forbidden: list[dict]) -> bool:
    parts = [x for x in re.split(r"[/]", rel) if x]
    keys = {parts[0] + "/", Path(rel).name} if len(parts) > 1 else {Path(rel).name}
    return any(k in f["text"] for f in forbidden for k in keys)


def l1_placement(public_dir: Path, truth_dirs: list[Path], forbidden: list[dict] | None = None) -> list[dict]:
    forbidden = forbidden or []
    """Truth files copied into public data (identical content), or answer-like file names in public data."""
    import hashlib

    def h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 200e6 else ""

    thash = {h(p): str(p) for d in truth_dirs for p in d.rglob("*") if p.is_file() and p.stat().st_size > 64}
    out = []
    for p in public_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(public_dir))
        if p.stat().st_size > 64 and h(p) in thash:
            # truth dirs often keep copies of public inputs: only a warning unless the file is declared off-limits
            out.append(dict(cls="L1c", file=rel, same_as=Path(thash[h(p)]).name,
                            severity="block" if _forbidden_mentions(rel, forbidden) else "warn"))
        elif re.search(r"(?:^|[_-])(?:truth|answers?|oracle_ref|expected|gold)(?:[_.-]|$)", p.name, re.I):
            out.append(dict(cls="L1c", file=rel, name_pattern=True, severity="block"))
    return out[:20]


def l3_introspect(cfg: dict, public_dir: Path, timeout: int = 180) -> tuple[list[Leaf], dict]:
    pys = list(public_dir.rglob("*.py"))
    if not pys:
        return [], dict(status="no_python")
    with tempfile.TemporaryDirectory(prefix="tbsgen-l3-") as td:
        td = Path(td)
        (td / "probe").mkdir()
        (td / "probe" / "introspect_probe.py").write_text(PROBE.read_text())
        (td / "out").mkdir()
        (td / "out").chmod(0o777)
        r = sandbox.run(cfg, [(public_dir, "/data", "ro"), (td / "probe", "/probe", "ro"), (td / "out", "/out", "rw")],
                        "python /probe/introspect_probe.py /data /out/introspect.json", timeout, "tbsgen-l3")
        f = td / "out" / "introspect.json"
        if not f.exists():
            return [], dict(status="probe_failed", rc=r["rc"], err=r["err"][-500:])
        rep = json.loads(f.read_text())
    leaves = [Leaf(f"introspect:{n.split(':')[0]}", norm_name(n.split(":", 1)[-1]), np.asarray(v, float), n)
              for n, v in rep.pop("leaves", [])]
    rep.update(status="ok", n_leaves=len(leaves))
    return leaves, rep


def leak_report(public_dir: Path, truth_dirs: list[Path], instruction: str, cfg: dict,
                depgraph: dict | None = None, introspect: str = "docker") -> dict:
    public_dir = Path(public_dir)
    truth_dirs = [Path(d) for d in truth_dirs if Path(d).exists()]
    rep: dict = dict(findings=[], info={})
    ptext = _public_text(public_dir)
    if depgraph:
        rep["findings"] += l1_depgraph(depgraph, instruction, ptext)
    fb = l1_forbidden(instruction)
    rep["findings"] += [dict(f, severity="block") for f in fb]
    rep["findings"] += l1_placement(public_dir, truth_dirs, fb)
    declared = declared_names(instruction)
    pub, unread = dir_leaves(public_dir)
    truth: list[Leaf] = []
    for d in truth_dirs:
        t, _ = dir_leaves(d, prefix=d.name + "/")
        truth += t
    rep["info"].update(n_public_leaves=len(pub), n_truth_leaves=len(truth), unreadable_public=unread[:20])
    f2, echoes = numeric_scan(truth, pub, declared)
    rep["info"]["l2_echoes_ignored"] = echoes
    for f in f2:   # a carrier the instruction declares off-limits is rule-only protection -> always blocking
        if any(_forbidden_mentions(p.split(":")[0], fb) for p in f["public"]):
            f["severity"] = "block"
    rep["findings"] += [dict(cls="L2", **f) for f in f2]
    rep["findings"] = [dict(f, severity=f.get("severity", "block")) for f in rep["findings"]]
    if introspect == "docker":
        il, irep = l3_introspect(cfg, public_dir)
        rep["info"]["l3"] = irep
        if il:
            # only truth values that are NOT already in public files (those are L2's business); no name-based echo
            # exemption: a private attribute holding a truth value is exactly the black-box leak
            from .leakgate import PublicIndex
            pidx = PublicIndex(pub)
            hidden_only = [t for t in truth if not (pidx.contain(t) or pidx.permuted(t))]
            f3, _ = numeric_scan(hidden_only, il, declared=set(), echo_same_name=False)
            rep["findings"] += [dict(cls="L3", **dict(f, severity="block")) for f in f3 if f["severity"] == "block"]
    else:
        rep["info"]["l3"] = dict(status="skipped")
    tt = {}
    for d in truth_dirs:
        tt.update({f"{d.name}/{k}": v for k, v in csv_tables(d).items()})
    rep["findings"] += [dict(cls="L4a", **f) for f in join_probe(tt, csv_tables(public_dir),
                                                                 cfg["gates"].get("l4a_spearman", 0.995))]
    rep["info"]["l4b"] = "skipped (cheap-agent inversion: model role 待定)"
    blocking = sorted({f["cls"] for f in rep["findings"] if f.get("severity") == "block"})
    warns = sorted({f["cls"] for f in rep["findings"] if f.get("severity") != "block"})
    rep["classes"] = blocking
    rep["warn_classes"] = warns
    rep["verdict"] = "leak" if blocking else ("review" if warns else "clean")
    return rep
