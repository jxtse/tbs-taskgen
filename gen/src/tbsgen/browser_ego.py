"""Browser channel for S3: drive a local ego-browser (Ego Lite) with the user's logged-in profile.

`run_one(job)` runs the reviewed script template `browser_fetch.mjs` through `ego-browser nodejs` (stdin) with
one injected `const INPUT = {...}` line, and parses the `TBSGEN_RESULT {...}` line it prints.  No LLM is
involved.  The script uses one fixed TaskSpace (`fulltext.browser.space`) and one managed Page per parallel
slot; it never touches the user's own tabs and never clicks through a challenge page.

Pacing is per publisher (DOI prefix, ledger key "chrome:<prefix>"): `fetch_queue` runs at most
`browser.concurrency` publishers at a time and never two papers of the same publisher at once; the ledger
enforces the 60-120 s gap and the daily caps.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

SCRIPT = Path(__file__).with_name("browser_fetch.mjs")

# Known direct-PDF URL patterns by DOI prefix ({doi}, {suffix}); the landing page is used when none fits.
PDF_HINTS = {
    "10.1109/": None,                                        # IEEE: landing page -> stamp.jsp -> iframe PDF
    "10.1021/": "https://pubs.acs.org/doi/pdf/{doi}",
    "10.1080/": "https://www.tandfonline.com/doi/pdf/{doi}?download=true",
    "10.1177/": "https://journals.sagepub.com/doi/pdf/{doi}?download=true",
    "10.1287/": "https://pubsonline.informs.org/doi/pdf/{doi}?download=true",
    "10.1137/": "https://epubs.siam.org/doi/pdf/{doi}?download=true",
    "10.1061/": "https://ascelibrary.org/doi/pdf/{doi}?download=true",
    "10.2514/": "https://arc.aiaa.org/doi/pdf/{doi}?download=true",
    "10.1002/": "https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true",
    "10.1111/": "https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true",
    "10.1029/": "https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true",
    "10.1098/": "https://royalsocietypublishing.org/doi/pdf/{doi}?download=true",
    "10.1063/": None, "10.1093/": None, "10.1017/": None, "10.1016/": None, "10.1117/": None, "10.1088/": None,
}


def publisher_key(doi: str) -> str:
    """Ledger key of a browser download (the earlier pipeline's "chrome:<DOI prefix>"); see
    fulltext.publisher_of for the aliases that share its pacing and cooldowns."""
    return "chrome:" + (doi.split("/", 1)[0] if "/" in doi else doi)


# The ego-browser CLI needs only these; credentials in the parent environment never reach the browser process.
ENV_ALLOW = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
             "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "NO_PROXY", "no_proxy")


def child_env() -> dict:
    return {k: v for k, v in os.environ.items() if k in ENV_ALLOW}


def pdf_url_hint(doi: str, c: dict | None = None) -> list[str]:
    urls = []
    for pre, pat in PDF_HINTS.items():
        if doi.startswith(pre) and pat:
            urls.append(pat.format(doi=doi, suffix=doi.split("/", 1)[1]))
    for k in ("pdf_url",):
        if c and c.get(k):
            urls.append(c[k])
    return urls


def binary(cfg: dict) -> str | None:
    b = cfg.get("binary") or "ego-browser"
    return b if os.path.isabs(b) and os.access(b, os.X_OK) else shutil.which(b)


def render(inp: dict) -> str:
    return "const INPUT = " + json.dumps(inp) + ";\n" + SCRIPT.read_text()


def _exec(cfg: dict, inp: dict, timeout: float, cancel=None) -> dict:
    """Run the script; `cancel` (a threading.Event) kills it early (browser batches: the egress changed)."""
    exe = binary(cfg)
    if not exe:
        return dict(status="unavailable", error="ego-browser binary not found (fulltext.browser.binary)")
    p = subprocess.Popen([exe, "nodejs"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=child_env())
    t_end = time.monotonic() + timeout
    so = se = ""
    try:
        so, se = p.communicate(render(inp), timeout=min(1.0, timeout))
    except subprocess.TimeoutExpired:
        while True:
            if cancel is not None and cancel.is_set():
                p.kill()
                p.communicate()
                return dict(status="cancelled", error="cancelled (batch stopped)")
            if time.monotonic() > t_end:
                p.kill()
                p.communicate()
                return dict(status="timeout", error=f"ego-browser script exceeded {timeout:.0f}s")
            try:
                so, se = p.communicate(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                continue
    p = subprocess.CompletedProcess(p.args, p.returncode, so or "", se or "")
    for line in reversed((p.stdout + "\n" + p.stderr).splitlines()):     # the CLI may print to either stream
        if line.startswith("TBSGEN_RESULT "):
            try:
                return json.loads(line[len("TBSGEN_RESULT "):])
            except json.JSONDecodeError:
                break
    return dict(status="error", error=(p.stderr or p.stdout)[-300:])   # caller redacts before logging


def run_one(job: dict) -> dict:
    """job: doi, title, pdf_url (list), slot, cfg (fulltext.browser), out_dir (absolute); or mode=egress."""
    cfg = job["cfg"]
    base = dict(space=cfg.get("space", "tbsgen-s3-fulltext"), slot=int(job.get("slot", 0)),
                nav_timeout_ms=int(cfg.get("nav_timeout_s", 45) * 1000),
                download_timeout_ms=int(cfg.get("download_timeout_s", 120) * 1000),
                challenge_wait_ms=int(cfg.get("challenge_wait_s", 40) * 1000),
                download_wait_ms=int(cfg.get("download_wait_s", 6) * 1000),
                max_navigations=int(cfg.get("max_navigations", 5)))
    if job.get("mode") == "egress":                 # on a temporary Page of its own, closed afterwards
        return _exec(cfg, dict(base, mode="egress", urls=job["urls"]), 120)
    if job.get("mode") == "probe":                  # egress watcher: own TaskSpace (cfg.space), reused Page
        return _exec(cfg, dict(base, mode="probe", urls=job["urls"], probe_timeout_ms=int(job.get("timeout_ms", 8000))),
                     float(job.get("timeout_s", 30)))
    if job.get("mode") == "finish":
        return finish(cfg)
    out_file = str(Path(job["out_dir"]).resolve() / f"dl-{uuid.uuid4().hex[:12]}.bin")
    inp = dict(base, mode="fetch", doi=job["doi"], pdf_urls=job.get("pdf_url") or [], out_file=out_file,
               max_bytes=int(job.get("max_bytes") or 40e6))
    if cfg.get("landing_base"):                     # tests: a local stand-in for https://doi.org/
        inp["landing_base"] = cfg["landing_base"]
    res = _exec(cfg, inp, float(cfg.get("script_timeout_s", 420)), job.get("cancel"))
    res["out_file"] = out_file                     # the caller removes it in every outcome
    return res


def ensure_pages(cfg: dict, n: int) -> dict:
    return _exec(cfg, dict(space=cfg.get("space", "tbsgen-s3-fulltext"), mode="ensure_pages", n=n), 120)


def finish(cfg: dict) -> dict:
    """Close the TaskSpace (keep no Pages)."""
    return _exec(cfg, dict(space=cfg.get("space", "tbsgen-s3-fulltext"), mode="finish"), 120)
