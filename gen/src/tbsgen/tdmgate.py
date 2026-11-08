"""S7 gate `tdm_verbatim`: public task files must not copy long passages of a TDM-licensed full text.

Elsevier's TDM provisions allow quoting at most 200 characters of the mined text (with a DOI link) and forbid
substantial or systematic reproduction; Wiley's TDM terms forbid redistributing the content.  For a paper whose
full text came through a TDM channel (`gates.tdm_verbatim_channels`, default elsevier and wiley -- also when it
was served from a cache that one of them filled), the longest verbatim overlap between the full text and every
public task file (instruction.md and the text files under environment/data) must not exceed
`gates.tdm_verbatim_max_chars` (200), compared after whitespace normalisation and case folding.

A text whose channel is unknown is treated as TDM only when `gates.tdm_verbatim_unknown` is true (default false:
the earlier pipeline's caches carry a channel in their provenance).  A paper from a TDM channel whose full text
is no longer present (purged) fails closed: nothing can show the files are clean.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .leakcheck import TEXT_SUFFIX

_WS = re.compile(r"\s+")
_B = 1_000_003
_M = (1 << 61) - 1


def normalise(s: str) -> str:
    return _WS.sub(" ", s or "").strip().casefold()


def _hashes(s: str, k: int) -> list[int]:
    if len(s) < k:
        return []
    h, out = 0, []
    p = pow(_B, k - 1, _M)
    for i, ch in enumerate(s):
        if i >= k:
            h = (h - ord(s[i - k]) * p) % _M
        h = (h * _B + ord(ch)) % _M
        if i >= k - 1:
            out.append(h)
    return out


def longest_overlap(public: str, source: str, at_least: int) -> tuple[int, str]:
    """Length of the longest common substring of the two (normalised) texts if it is >= at_least, else 0, plus
    the start of that passage.  Rolling-hash windows of `at_least` chars find candidate positions; each hit is
    verified and extended in both texts."""
    a, b = normalise(public), normalise(source)
    k = max(1, int(at_least))
    if len(a) < k or len(b) < k:
        return 0, ""
    src = {}
    for j, h in enumerate(_hashes(b, k)):
        src.setdefault(h, []).append(j)
    best, snip, i = 0, "", 0
    ha = _hashes(a, k)
    while i < len(ha):
        hit = 0
        for j in src.get(ha[i], ())[:50]:
            if a[i:i + k] != b[j:j + k]:
                continue
            n = k
            while i + n < len(a) and j + n < len(b) and a[i + n] == b[j + n]:
                n += 1
            hit = max(hit, n)
            m = len(a[i:i + n].strip())          # a shared space at either end is not quoted text
            if m > best:
                best, snip = m, a[i:i + n].strip()[:120]
        i += max(1, hit - k + 1) if hit else 1
    return best, snip


def tdm_channel(item_dir: Path) -> str | None:
    """The channel the item's full text came through (a cache hit names the channel that filled the cache)."""
    p = item_dir / "S3_fulltext.json"
    try:
        rec = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    ch = rec.get("source")
    if ch == "cache":
        ch = rec.get("cached_channel")
    return ch


def public_files(task: Path, data_dir: Path) -> list[tuple[str, str]]:
    out = []
    ins = task / "instruction.md"
    if ins.exists():
        out.append(("instruction.md", ins.read_text(errors="replace")))
    for p in sorted(data_dir.rglob("*")) if data_dir.exists() else []:
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIX and p.stat().st_size < 2e6:
            out.append((f"environment/data/{p.relative_to(data_dir)}", p.read_text(errors="replace")))
    return out


def check(cfg: dict, item_dir: Path, task: Path, data_dir: Path) -> dict:
    g = cfg["gates"]
    limit = int(g.get("tdm_verbatim_max_chars", 200))
    chans = set(g.get("tdm_verbatim_channels") or ("elsevier", "wiley"))
    ch = tdm_channel(item_dir)
    applies = ch in chans or (ch is None and bool(g.get("tdm_verbatim_unknown", False)))
    out = dict(channel=ch, applies=applies, max_chars=limit, findings=[])
    if not applies:
        out["ok"] = True
        return out
    ft = item_dir / "fulltext.txt"
    if not ft.exists():
        out.update(ok=False, error="full text not available (purged?): cannot verify the public files")
        return out
    src = ft.read_text(errors="replace")
    for rel, txt in public_files(task, data_dir):
        n, snip = longest_overlap(txt, src, limit + 1)
        if n > limit:
            out["findings"].append(dict(file=rel, chars=n, starts=snip))
    out["ok"] = not out["findings"]
    return out
