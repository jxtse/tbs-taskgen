"""Mechanical checks of the solver-facing statement."""
import math
import re

from .package import strip_time_note
from .spec import assumption_quote

# Scientific homonyms stay allowed: "patch" (sky patch, tissue patch), "internal standard", "baseline-free" (structural
# health monitoring) all occur in real statements of the 2026-11-23 share; only their process senses are flagged.
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
            fraction = mantissa.partition(".")[2]
            digits = mantissa.replace(".", "").lstrip("0") or fraction
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


def assumptions_stated(text: str, assumptions: list, require_long_quote: bool = True) -> list[str]:
    normalized = " ".join(strip_time_note(text).split())
    missing = []
    for a in assumptions:
        quote = assumption_quote(a["where"], require_long_quote)
        if a["stated"] and (not quote or quote not in normalized):
            missing.append(a["where"])
    return missing
