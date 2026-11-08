"""S2 publisher blocklist: papers whose full text we cannot obtain are dropped before screening.

A rule matches on the DOI prefix (registrant, e.g. "10.1177") and optionally on the publication year
(`before_year`: only papers published strictly before that year).  The year comes from the candidate record;
for Elsevier-style DOIs without a year the two-digit year embedded in the PII ("...(97)00086-0") is used.
A paper with no DOI, or with an unknown year under a year-limited rule, is never blocked.
"""
from __future__ import annotations

import re
import urllib.parse

_PII_YEAR = re.compile(r"\((\d{2})\)")
_YEAR = re.compile(r"^\s*(\d{4})(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\s*$")   # 1997, 1997-03, 1997-03-01


def _norm_doi(doi: str | None) -> str:
    d = urllib.parse.unquote(str(doi or "")).strip()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)\s*", "", d, flags=re.I)
    return d.strip().lower()


def paper_year(c: dict) -> int | None:
    m = _YEAR.match(str(c.get("year") or ""))
    if m and 1800 <= int(m.group(1)) <= 2100:
        return int(m.group(1))
    doi = _norm_doi(c.get("doi"))
    if doi.startswith("10.1016/") and (m := _PII_YEAR.search(doi)):
        yy = int(m.group(1))
        return 1900 + yy if yy >= 50 else 2000 + yy
    return None


def match(c: dict, rules: list[dict] | None) -> dict | None:
    """The first rule that blocks candidate `c`, or None."""
    doi = _norm_doi(c.get("doi"))
    if not doi or not rules:
        return None
    for r in rules:
        pre = str(r.get("doi_prefix") or "").lower().rstrip("/")
        if not pre or not (doi == pre or doi.startswith(pre + "/")):
            continue
        if r.get("before_year") is not None:
            y = paper_year(c)
            if y is None or y >= int(r["before_year"]):
                continue
        return r
    return None


def screen_record(c: dict, rule: dict) -> dict:
    """S2_screen.json record for a blocked paper (same shape as a rules/Jev drop)."""
    name = rule.get("name") or rule.get("doi_prefix")
    return dict(score=None, keep=False, method="blocklist", reason=f"blocklist:{name}",
                blocklist=dict(name=name, doi_prefix=rule.get("doi_prefix"), before_year=rule.get("before_year"),
                               why=rule.get("why"), year=paper_year(c)))
