"""Mechanical checks of the solver-facing statement."""
import math
import re

from .package import strip_time_note
from .spec import assumption_quote

# Scientific homonyms stay allowed: "patch" (sky patch, tissue patch), "internal standard", "baseline-free" (structural
# health monitoring) all occur in real task statements; only their process senses are flagged.
# Bare "gate" (gate voltage, logic gate), "hardening" (strain hardening), "lever" (lever arm), "audit" (convergence
# audit) and "revision" (data revision) are science too: only their editorial / pipeline phrasings are flagged.
PROCESS_WORDS = (r"audit\s+(?:copy|revision|note|round|fix|finding)s?", r"(?:task|statement|audit|review)\s+revision",
                 r"revision\s+(?:note|history|round)s?", r"reviewer(?:'s)?\s+(?:note|comment|request)s?",
                 r"review\s+round", r"S9R", r"S[5-9][A-Z]?\s+(?:gate|stage|repair|review)", r"(?:quality|S7|leak)\s+gates?",
                 r"tbsgen", r"TODO", r"FIXME", r"XXX", r"internal\s+(?:note|review|use|only)",
                 r"hardening\s+(?:round|lever|pass)s?", r"difficulty\s+levers?", r"round-\d+",
                 r"(?:hot)?fix\s+log", r"patch(?:ed)?\s+(?:notes?|version|round)")
_PROCESS = re.compile(r"\b(?:" + "|".join(PROCESS_WORDS) + r")\b", re.I)
_HEADING_WORDS = re.compile(r"\bexploit\b", re.I)
_BASELINE = re.compile(r"\b(?:mid|naive)[ -]+baseline\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+|(?=\.\d))(?:\.\d+)?(?:[eE][-+]?\d+)?")
_CUTOFF = re.compile(r"(?:[<>]=?|[≤≥]|\\(?:leq?|geq?)\b|\bat (?:least|most)|\b(?:less|greater) than|\b(?:above|below|under|over)|"
                     r"\bthreshold(?:\s+(?:of|is))?)\s*[:=`$* ]*$", re.I)


# any file-like token: name.ext, including multi-part extensions (input.csv.gz) and <output_dir>/ prefixes
FILE_TOKEN = re.compile(r"(?<![\w./<>-])(?:<\w+>/)?[\w./-]*\w\.[A-Za-z][\w]*(?:\.[A-Za-z][\w]*)*(?![\w-])")
_OUT_PREFIX = re.compile(r"^(?:\./|/app/output/|output/|<output_dir>/)")


def output_relative_path(tok: str) -> str:
    return _OUT_PREFIX.sub("", tok.strip())


def declared_outputs(spec, statement):
    """Output paths explicitly named by the spec or statement. An empty set means unknown."""
    declared = set()

    def add_entries(entries):
        if isinstance(entries, str):
            if FILE_TOKEN.fullmatch(entries.strip()):
                declared.add(output_relative_path(entries))
        elif isinstance(entries, list):
            for entry in entries:
                add_entries(entry)
        elif isinstance(entries, dict):
            for key, value in entries.items():
                if key in ("files", "deliverables", "outputs", "path", "file", "filename", "name"):
                    add_entries(value)
                elif FILE_TOKEN.fullmatch(key):
                    declared.add(output_relative_path(key))

    for key in ("output_contract", "deliverables", "output_files", "outputs"):
        add_entries(spec.get(key))
    graph = spec.get("depgraph") or {}
    add_entries(graph.get("deliverables"))
    for node in graph.get("nodes", []):
        if node.get("kind") == "deliverable":
            add_entries(node.get("carrier"))
    in_outputs = False
    for line in statement.splitlines():
        if line.lstrip().startswith("#"):
            in_outputs = bool(re.search(r"\b(?:outputs?|deliverables?)\b", line, re.I))
        list_end = None
        input_clause = False
        prev_end = 0
        for match in FILE_TOKEN.finditer(line):
            for cue in re.finditer(r"\b(from|using|given|read|write|generate|produce|submit|save)\b|[.;]", line[prev_end:match.start()], re.I):
                input_clause = cue[0].lower() in ("from", "using", "given", "read")
            prev_end = match.end()
            if input_clause:
                list_end = None
                continue
            path = match[0]
            explicit_path = path.startswith(("/app/output/", "<output_dir>/", "output/"))
            relative = not path.startswith(("/", "data/"))
            declaration = re.search(r"\b(?:write|generate|produce|submit|save|deliverables?:|outputs?:)\s+[`*]*$",
                                    line[:match.start()], re.I)
            continuation = list_end is not None and re.fullmatch(
                r"[`*]*\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or)\s+)[`*]*", line[list_end:match.start()], re.I)
            if explicit_path or (relative and (in_outputs or declaration or continuation)):
                declared.add(output_relative_path(path))
                list_end = match.end() if declaration or continuation or explicit_path else None
            else:
                list_end = None
    return declared


def statement_clean(text: str) -> list[str]:
    text = strip_time_note(text)
    return [line for line in text.splitlines() if "<!--" in line or _PROCESS.search(line) or _BASELINE.search(line)
            or (line.lstrip().startswith("#") and _HEADING_WORDS.search(line))]


MAX_THRESHOLD_DIGITS = 6     # a threshold (known value or a number right after a cutoff word) as written
MAX_ANY_DIGITS = 12          # any literal: 13+ significant digits is a printed float, not a physical constant
                             # (GM = 398600.4418, J2 = 1.08262668e-3 are legitimate and stay allowed)


def threshold_format(text: str, thresholds: dict | None = None) -> list[str]:
    """Flag float-repr literals anywhere, and over-precise thresholds at known values or explicit cutoffs."""
    values = {abs(t["threshold"]) for t in (thresholds or {}).values() if t.get("threshold") is not None}
    bad = []
    for line in strip_time_note(text).splitlines():
        for m in _NUMBER.finditer(line):
            literal = m[0]
            mantissa = re.split("[eE]", literal)[0].lstrip("+-−").replace(",", "")
            whole, dot, fraction = mantissa.partition(".")
            digits = mantissa.replace(".", "").lstrip("0") or fraction
            if not dot and not re.search("[eE]", literal):
                # Trailing zeros of an integer carry no precision: "n >= 1,000,000" is a round cutoff.
                digits = whole.lstrip("0").rstrip("0") or "0"
            if len(digits) <= MAX_THRESHOLD_DIGITS:
                continue
            value = abs(float(literal.replace(",", "").replace("−", "-")))
            suffix = line[m.end():].lstrip()
            power = re.match(r"(?:[×x·]|\\times)\s*10\s*\^\s*\{?([-+−]?\d+)\}?", suffix)
            if power:
                value *= float("1e" + power[1].replace("−", "-"))
                suffix = suffix[power.end():].lstrip()
            scales = [value, value / 100] if suffix.startswith("%") else [value]
            if len(digits) > MAX_ANY_DIGITS or _CUTOFF.search(line[:m.start()]) or any(
                    math.isclose(v, t, rel_tol=1e-6) for v in scales for t in values):
                bad.append(literal)
    return list(dict.fromkeys(bad))


_MARKUP = re.compile(r"[*_`]+")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2212": "-",
                         "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def quote_key(text: str) -> str:
    """Comparison form of a statement quote: Markdown emphasis / code marks (* _ `) dropped, typographic quotes and
    dashes folded to ASCII, case folded (a quote that starts a sentence is capitalized), whitespace collapsed.  A
    quote rendered in bold or code font in the statement still counts as present; any change of words does not."""
    return " ".join(_MARKUP.sub("", text.translate(_QUOTES)).split()).casefold()


def assumptions_stated(text: str, assumptions: list, require_long_quote: bool = True) -> list[str]:
    plain = " ".join(strip_time_note(text).split())
    keyed = quote_key(plain)
    missing = []
    for a in assumptions:
        quote = assumption_quote(a["where"], require_long_quote)
        if a["stated"] and (not quote or (quote not in plain and quote_key(quote) not in keyed)):
            missing.append(a["where"])
    return missing


def stated_quotes(assumptions: list | None) -> list[str]:
    """Whitespace-normalized statement quotes the S7 assumptions_stated gate looks for (spec generator_assumptions
    with stated=true)."""
    return [" ".join(a["where"].split()) for a in assumptions or []
            if isinstance(a, dict) and a.get("stated") and isinstance(a.get("where"), str) and a["where"].strip()]


def quote_block(assumptions: list | None) -> str:
    """Prompt block for a stage that (re)writes instruction.md: the exact quotes S7 requires, one per line."""
    quotes = stated_quotes(assumptions)
    if not quotes:
        return ""
    lines = "\n".join(f"- {json_quote(q)}" for q in quotes)
    return ("\nThe S7 gate assumptions_stated requires each of these exact statement quotes (from the spec's "
            "generator_assumptions[].where) to appear word for word in instruction.md.  Keep every one verbatim in "
            "the sentence that discloses that assumption (Markdown emphasis may be added around it; do not reword, "
            "split, abbreviate or re-punctuate it):\n" + lines + "\n")


def json_quote(q: str) -> str:
    import json
    return json.dumps(q, ensure_ascii=False)
