"""Offline area and method-template inventory, shared by admission and reporting."""
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import re

from .spec import is_resource_metric, metric_family
from .state import TERMINAL, read_json


# Prefer the stated domain, then the field, then the title. Specific imaging terms disambiguate tissue from MRI.
TAXONOMY = {
    "astronomy & cosmology": r"astronom|cosmolog|astrophys|astroinform|astrodynam|exoplanet|stellar|light curve|photometr",
    "earth & climate": r"earth|climat|geosci|geodes|hydrolog|seismolog|remote sensing|cryosphere|atmospheric",
    "fluids & continuum mechanics": r"fluid|continuum|hydrodynam|turbulen|aerodynam|navier",
    "structural engineering": r"structural (?:engineering|health|mechanics)|damage locali|guided wave|civil engineering",
    "materials": r"material|crystall|nanoindent|metallurg|alloy|diffraction",
    "chemistry": r"chemistr|chemical|spectrometr|chromatograph|catalys|electrochem",
    "molecular & sequence bioinformatics": r"bioinform|molecular|genom|transcriptom|epigenom|proteom|sequence alignment|dna|rna",
    "cell & tissue imaging": r"cell.*imag|tissue.*imag|bioimage|biomedical imag|histolog|microscop|nucle[ui]|pathology",
    "medical imaging & clinical": r"medical imag|clinical|radiolog|tomograph|mri\b|dicom",
    "neuroscience": r"neurosci|neuronal|brain|electrophysiol",
    "ecology & evolution": r"ecolog|evolution|biodivers|plant phenotyp|population genetics",
    "statistics & causal inference": r"statistic|causal|bayesian|semiparametric",
    "physics & optics": r"physic|optic|photon|lithograph|quantum|mechanics|electromagnet",
    "numerical methods & computing": r"numerical|computing|computer|machine learning|linear algebra|point.cloud|"
                                    r"applied math|dynamical systems|system identification|robotic",
}


def classify(domain, field, title):
    for text in (domain, field, title):
        for area, words in TAXONOMY.items():
            if re.search(r"\b(?:" + words + ")", str(text or ""), re.I):
                return area
    return "other"


_STOP = set("a an and or by on for of in to with from the under using method task".split())


def _tokens(strings):
    return frozenset(t for s in strings for t in re.findall(r"[a-z]+[0-9]*", str(s).lower()) if t not in _STOP)


@dataclass(frozen=True)
class Template:
    id: str
    area: str
    target: str
    signature: frozenset
    families: frozenset


def template(item_id, spec, prov):
    target = str(prov.get("target") or "").strip().casefold()
    nodes = [n["id"] for n in (spec.get("depgraph") or {}).get("nodes", []) if isinstance(n, dict) and "id" in n]
    words = _tokens([*(spec.get("competencies") or []), *nodes, target])
    families = frozenset(metric_family(m["name"]) for m in spec.get("metrics", [])
                         if m.get("name") and not is_resource_metric(m))
    return Template(item_id, classify(spec.get("domain"), spec.get("field"), spec.get("title") or target),
                    target, words, families)


def related(a, b, early=False):
    # Before authoring, compare both sides on the only method evidence available: target tokens.
    x, y = (_tokens([a.target]), _tokens([b.target])) if early else (a.signature, b.signature)
    return bool(x and y and 2 * len(x & y) >= len(x | y) or
                a.target and a.target == b.target and a.families & b.families)


def clusters(records, early=False):
    """Connected components, including singletons. The least member ID names a cluster reproducibly."""
    parent = {r.id: r.id for r in records}
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in combinations(records, 2):
        if related(a, b, early):
            x, y = sorted((root(a.id), root(b.id)))
            parent[y] = x
    groups = {}
    for i in sorted(parent):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def item_template(item, early=False):
    candidate = read_json(item / "S1_candidate.json", {})
    spec = {} if early else read_json(item / "_spec/spec.json", {})
    prov = {} if early else read_json(item / "_spec/provenance.json", {})
    target = prov.get("target") or candidate.get("_target") or ""
    title = spec.get("title") or candidate.get("title") or ""
    return template(item.name, dict(spec, title=title, field=spec.get("field") or target), {"target": target})


def released_ids(run):
    manifest = run / "release/manifest.jsonl"
    live = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # same last-valid-record semantics as the release stage
            if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
                continue
            if row.get("withdrawn"):
                live.discard(row["item_id"])
            else:
                live.add(row["item_id"])
    return live


def summarize(released, inflight):
    counts = Counter(r.area for r in released)
    pending = Counter(r.area for r in inflight)
    groups = clusters([*released, *inflight])
    live = {r.id for r in released}
    templates = [dict(cluster=g[0], count=len(g), released=len(live.intersection(g)), members=g) for g in groups]
    return dict(released=len(live), in_flight=len(inflight),
                released_by_area={a: counts[a] for a in (*TAXONOMY, "other")},
                in_flight_by_area={a: pending[a] for a in (*TAXONOMY, "other")},
                top_templates=templates, reskin_clusters=[g for g in groups if len(g) > 1],
                gaps=[a for a in TAXONOMY if not counts[a]])


def report(run: Path, runs_root: Path | None = None):
    runs = [run]
    if runs_root:
        runs.extend(d for d in sorted(runs_root.iterdir()) if (d / "items").is_dir() and d.resolve() != run.resolve())
    live, pending = {}, {}
    for d in runs:
        ids = released_ids(d)
        for i in sorted(ids - live.keys()):
            live[i] = item_template(d / "items" / i)
        for state in sorted((d / "items").glob("*/state.json")):
            i = state.parent.name
            if i not in live and i not in pending and read_json(state, {}).get("status") not in TERMINAL:
                pending[i] = item_template(state.parent)
    return summarize(list(live.values()), [r for i, r in pending.items() if i not in live])


def released_labels(run):
    """{item_id: label} of the live releases (same last-record semantics as released_ids)."""
    manifest = run / "release/manifest.jsonl"
    live = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
                continue
            if row.get("withdrawn"):
                live.pop(row["item_id"], None)
            else:
                live[row["item_id"]] = row.get("label")
    return live


def o70_quota(P, it):
    """Hard tasks already held for the candidate's official-task group, from an operator-maintained inventory.

    diversity.o70_inventory is a JSON file with target_group {S1 target -> official task}, library {official task ->
    {"hard": n, ...}} and library_items [item ids already counted in library].  Live hard releases of this run that
    the library has not counted yet are added.  Returns None when no inventory is configured; an unreadable inventory
    or an unmapped target never blocks.
    """
    cfg = P.cfg["diversity"]
    path = cfg.get("o70_inventory")
    if not path:
        return None
    inv = read_json(Path(path), None)
    if not isinstance(inv, dict) or not isinstance(inv.get("target_group"), dict):
        return dict(error="inventory_unreadable", reason=None)
    groups = inv["target_group"]
    target = (read_json(it.p("S1_candidate.json"), {}) or {}).get("_target") or ""
    group = groups.get(target)
    if not group:
        return dict(target=target, group=None, reason=None)
    counted = set(inv.get("library_items") or [])
    library_hard = int(((inv.get("library") or {}).get(group) or {}).get("hard", 0))
    released = []
    for i, label in sorted(released_labels(P.store.root).items()):
        if label != "hard" or i == it.item_id or i in counted:
            continue
        t = (read_json(P.store.root / "items" / i / "S1_candidate.json", {}) or {}).get("_target")
        if groups.get(t) == group:
            released.append(i)
    cap = int(cfg.get("o70_max_hard", 4))
    hard = library_hard + len(released)
    return dict(target=target, group=group, library_hard=library_hard, released_hard=released, hard=hard, cap=cap,
                reason=f"diversity:o70_full:{group}" if hard >= cap else None)


def admission(P, it, stage):
    """Record the same decision in observation and enforcement modes; return a reason only when enforced.

    The official-task quota (diversity.o70_*) is checked before a paper's first authoring job only (S5G): the tier of
    a new task is not known yet, and a group that already holds o70_max_hard hard tasks cannot use another task of
    any tier.  It is enforced by diversity.o70_enforce, independently of the area / template rules."""
    cfg = P.cfg["diversity"]
    # only a paper's first authoring: re-authoring after a finished S5G job and hardening rounds are already invested
    first = stage == "S5G" and not it.state.get("harden_active") and not it.p("S5G_result.json").exists()
    o70 = o70_quota(P, it) if first else None
    candidate = item_template(it.dir, early=stage == "S5G")
    ids = released_ids(P.store.root) - {it.item_id}
    live = [item_template(P.store.root / "items" / i) for i in sorted(ids)]
    share = sum(r.area == candidate.area for r in live) / len(live) if live else 0.
    group = next(g for g in clusters([candidate, *live], early=stage == "S5G") if candidate.id in g)
    members = [i for i in group if i != candidate.id]
    reason = None
    # an area share means nothing over a handful of releases
    if len(live) >= int(cfg.get("min_released", 20)) and share > cfg["max_area_share"]:
        reason = f"diversity:area_full:{candidate.area}"
    elif len(members) >= cfg["max_per_template"]:
        cluster = members[0] if members else candidate.id
        reason = f"diversity:template_full:{cluster}"
    decision = dict(area=candidate.area, released=len(live), share=share, template_members=members, reason=reason,
                    enforce=cfg["enforce"])
    if o70 is not None:
        decision["o70"] = dict(o70, enforce=bool(cfg.get("o70_enforce")))
    history = it.state.setdefault("diversity", {})
    if history.get(stage) != decision:
        history[stage] = decision
        it.save()
        P.store.event("diversity_admission", item=it.item_id, stage=stage, **decision)
    if o70 and o70.get("reason") and cfg.get("o70_enforce"):
        return o70["reason"]
    return reason if cfg["enforce"] else None


def format_report(result):
    lines = ["area\treleased\tin flight"]
    for area, count in result["released_by_area"].items():
        lines.append(f"{area}\t{count}\t{result['in_flight_by_area'][area]}")
    lines.append("gaps: " + ", ".join(result["gaps"]))
    for t in result["top_templates"]:
        lines.append(f"template {t['cluster']} ({t['count']}, released {t['released']}): " + ", ".join(t["members"]))
    return "\n".join(lines)
