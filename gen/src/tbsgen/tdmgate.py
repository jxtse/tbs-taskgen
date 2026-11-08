"""S7 gate `tdm_verbatim`: public task files must not copy long passages of a TDM-licensed full text.

Elsevier's TDM provisions allow quoting at most 200 characters of the mined text (with a DOI link) and forbid
substantial or systematic reproduction; Wiley's TDM terms forbid redistributing the content.  For a paper whose
full text came through a TDM channel (`gates.tdm_verbatim_channels`, default elsevier and wiley -- also when it
was served from a cache that one of them filled), the longest verbatim overlap between the full text and every
public task file (instruction.md and the text files under environment/data) must not exceed
`gates.tdm_verbatim_max_chars` (200), compared after whitespace normalisation and case folding.  Every textual
file is checked whole (exact longest common substring via a suffix automaton of the full text).

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


def normalise(s: str) -> str:
    return _WS.sub(" ", s or "").strip().casefold()


class _SAM:
    """Suffix automaton of one text: exact longest common substring with another text in linear time."""

    def __init__(self, s: str):
        self.nxt: list[dict] = [{}]
        self.link = [-1]
        self.len = [0]
        last = 0
        for ch in s:
            cur = len(self.len)
            self.nxt.append({})
            self.len.append(self.len[last] + 1)
            self.link.append(0)
            p = last
            while p != -1 and ch not in self.nxt[p]:
                self.nxt[p][ch] = cur
                p = self.link[p]
            if p != -1:
                q = self.nxt[p][ch]
                if self.len[p] + 1 == self.len[q]:
                    self.link[cur] = q
                else:
                    clone = len(self.len)
                    self.nxt.append(dict(self.nxt[q]))
                    self.len.append(self.len[p] + 1)
                    self.link.append(self.link[q])
                    while p != -1 and self.nxt[p].get(ch) == q:
                        self.nxt[p][ch] = clone
                        p = self.link[p]
                    self.link[q] = self.link[cur] = clone
            last = cur

    def longest_in(self, t: str) -> tuple[int, int, int]:
        """(length without a space at either end, start, end) of the longest substring of t that occurs in the
        automaton's text.  At each end position only the longest match matters: its shorter suffixes lose at
        most the same leading space."""
        v, n, best, start, end = 0, 0, 0, 0, 0
        for i, ch in enumerate(t):
            while v and ch not in self.nxt[v]:
                v = self.link[v]
                n = self.len[v]
            if ch in self.nxt[v]:
                v = self.nxt[v][ch]
                n += 1
            if n:
                eff = n - (t[i - n + 1] == " ") - (ch == " " and n > 1)
                if eff > best:
                    best, start, end = eff, i - n + 1, i + 1
        return best, start, end


def longest_overlap(public: str, source: str, at_least: int = 1, sam: _SAM | None = None) -> tuple[int, str]:
    """Length of the longest common substring of the two normalised texts (shared spaces at either end of the
    passage are not counted), and its start; (0, "") when shorter than `at_least`."""
    a = normalise(public)
    sam = sam or _SAM(normalise(source))
    n, start, end = sam.longest_in(a)
    passage = a[start:end].strip()
    if n < max(1, int(at_least)):
        return 0, ""
    return n, passage[:120]


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


def _textual(p: Path) -> bool:
    """Any file that could carry prose: a known text suffix, or no NUL byte in its first 8 KB."""
    if p.suffix.lower() in TEXT_SUFFIX:
        return True
    with p.open("rb") as fh:
        return b"\0" not in fh.read(8192)


def public_files(task: Path, data_dir: Path) -> list[tuple[str, str]]:
    """instruction.md and every textual file under environment/data, whole (no size limit).  Symbolic links are
    followed (files and directories), as packaging copies their targets."""
    import os
    out = []
    ins = task / "instruction.md"
    if ins.exists():
        out.append(("instruction.md", ins.read_text(errors="replace")))
    seen: set[str] = set()
    for root, dirs, files in os.walk(data_dir, followlinks=True) if data_dir.exists() else []:
        real = os.path.realpath(root)
        if real in seen:                       # a link cycle
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        for name in sorted(files):
            p = Path(root) / name
            if p.is_file() and _textual(p):
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
    sam = _SAM(normalise(ft.read_text(errors="replace")))
    for rel, txt in public_files(task, data_dir):
        n, snip = longest_overlap(txt, "", limit + 1, sam=sam)
        if n > limit:
            out["findings"].append(dict(file=rel, chars=n, starts=snip))
    out["ok"] = not out["findings"]
    return out
