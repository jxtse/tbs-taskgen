"""Mechanical checks of the solver-facing statement."""
import math
import re

from .package import strip_time_note

# Scientific homonyms stay allowed: "patch" (sky patch, tissue patch), "internal standard", "baseline-free" (structural
# health monitoring) all occur in real statements of the 2026-11-23 share; only their process senses are flagged.
PROCESS_WORDS = (r"audit", r"revision", r"reviewer", r"review\s+round", r"S9R", r"S7", r"S8", r"gate",
                 r"tbsgen", r"TODO", r"FIXME", r"XXX", r"internal\s+(?:note|review|use|only)", r"hardening",
                 r"round-\d+", r"lever", r"(?:hot)?fix\s+log", r"patch(?:ed)?\s+(?:notes?|version|round)")
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


def assumptions_stated(text: str, assumptions: list) -> list[str]:
    normalized = " ".join(strip_time_note(text).split())
    missing = []
    for a in assumptions:
        quote = " ".join(a["where"].split())
        if a["stated"] and (len(quote) < 12 or quote not in normalized):
            missing.append(a["where"])
    return missing
