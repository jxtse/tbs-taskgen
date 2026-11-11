"""Code-grounded intake: S2a (code signal before any full-text download) and S3b (the authors' own code, confirmed
from the full text and resolved on its host).

Tasks are generated from a paper AND its authors' published code (cfg.code.grounded).  A paper without code is
dropped, never routed elsewhere:

S2a  (before S3, costs no download quota).  Signals: code links in the retrieval metadata, the retrieval
     `code_signals` record (OpenAlex full text / PwC archive / GitHub README search, tbsretr.codeindex), the
     Papers-with-Code archive (cfg.code.pwc_path) and one OpenAlex full-text flag call per paper.  None ->
     dropped `no_code_signal`.  Measured on 224 papers: recall 92 %, precision 82 % for own code.
S3b  (after the full text).  Every code-host link in the full text and the metadata is scored for ownership
     (Code Availability section, "our code is available at", link context, reference list, title tokens,
     README citing the paper) - rule scores from the 2026-11-11 code-grounding study.  Ambiguous links go to
     Jev (cheap) when roles.code_owner = "jev", never to the teacher model.  Own links are resolved on their host
     (GitHub, GitLab, Bitbucket, Zenodo, CRAN, Bioconductor, PyPI, SourceForge).  Drop reasons:
       code_on_request           the paper offers code only on request
       third_party_code_only     links only to tools the authors used / cite
       code_repo_unavailable     own repository 404 / private / empty
       data_only_no_code         own deposit holds data, no code
       code_host_unsupported     own code only on a host we cannot fetch (Code Ocean capsule, Figshare ...)
       no_code_link              no code link at all in the full text and metadata
     Licences are recorded, never filtered (decision 2026-11-11).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

# ----------------------------------------------------------------------------------------------- config helpers


def ccfg(cfg: dict) -> dict:
    return cfg.get("code") or {}


def grounded(cfg: dict) -> bool:
    return bool(ccfg(cfg).get("grounded"))


def after_s1(cfg: dict) -> str:
    return "S2a" if grounded(cfg) and ccfg(cfg).get("prescreen", True) else "S2"


def after_s3(cfg: dict) -> str:
    return "S3b" if grounded(cfg) else "S4"


# ----------------------------------------------------------------------------------------------- HTTP
def _opener(proxy: str):
    return urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if proxy == "none" else []))


def http_json(url: str, cfg: dict, timeout: float = 30) -> tuple[int, dict | list | None]:
    """GET -> (status, parsed json | None).  status 0 = network error.  No credentials are ever sent."""
    req = urllib.request.Request(url, headers={"User-Agent": "tbsgen-code/0.1 (mailto:%s)" % ccfg(cfg).get(
        "mailto", "noreply@example.org"), "Accept": "application/json"})
    try:
        with _opener(ccfg(cfg).get("proxy", "env")).open(req, timeout=timeout) as r:
            body = r.read()
            try:
                return r.status, json.loads(body)
            except ValueError:
                return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:  # noqa: BLE001
        return 0, None


def gh_json(path: str, cfg: dict, raw: bool = False) -> tuple[int, dict | list | str | None]:
    """GitHub REST via the gh CLI (its own auth; this process never handles a token), else anonymous REST."""
    if ccfg(cfg).get("github_client", "gh") == "gh":
        env = {k: v for k, v in os.environ.items() if k.lower() not in ("https_proxy", "http_proxy", "all_proxy")} \
            if ccfg(cfg).get("proxy", "env") == "none" else dict(os.environ)
        args = ["gh", "api", "-i", path] + (["-H", "Accept: application/vnd.github.raw"] if raw else [])
        try:
            r = subprocess.run(args, capture_output=True, text=True, env=env, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return 0, None
        sep = "\r\n\r\n" if "\r\n\r\n" in r.stdout else "\n\n"
        head, _, body = r.stdout.partition(sep)
        m = re.match(r"HTTP/[\d.]+ (\d+)", head)
        st = int(m.group(1)) if m else (200 if r.returncode == 0 else 0)
        if raw:
            return st, body if st == 200 else None
        try:
            return st, json.loads(body) if body.strip() else None
        except ValueError:
            return st, None
    if raw:
        st, _ = 0, None
        req = urllib.request.Request("https://api.github.com/" + path, headers={
            "Accept": "application/vnd.github.raw", "User-Agent": "tbsgen-code/0.1"})
        try:
            with _opener(ccfg(cfg).get("proxy", "env")).open(req, timeout=30) as r:
                return r.status, r.read().decode("utf8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, None
        except Exception:  # noqa: BLE001
            return 0, None
    return http_json("https://api.github.com/" + path, cfg)


# ----------------------------------------------------------------------------------------------- S2a
CODE_HOST_RE = re.compile(r"(github\.com|gitlab\.|bitbucket\.org|codeberg\.org|zenodo\.org|10\.5281/zenodo|"
                          r"sourceforge\.net|cran\.r-project\.org|pypi\.org|bioconductor\.org|codeocean\.com)", re.I)


def _norm_doi(d: str | None) -> str | None:
    if not d:
        return None
    d = urllib.parse.unquote(str(d)).strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d if d.startswith("10.") else None


def openalex_fulltext_flag(c: dict, cfg: dict) -> bool | None:
    """Does OpenAlex's full-text index of this DOI mention a code host?  None = unknown (no DOI / error)."""
    doi = _norm_doi(c.get("doi"))
    if not doi or doi.startswith("10.48550"):
        return None
    terms = "|".join(ccfg(cfg).get("fulltext_terms") or ["github.com", "zenodo.org", "gitlab"])
    q = urllib.parse.urlencode({"filter": f"doi:{doi},fulltext.search:{terms}", "select": "doi",
                                "mailto": ccfg(cfg).get("mailto", "noreply@example.org")})
    st, d = http_json("https://api.openalex.org/works?" + q, cfg)
    if st != 200 or not isinstance(d, dict):
        return None
    return bool(d.get("results"))


def s2a_signals(c: dict, cfg: dict, oa_flag: Callable | None = None) -> dict:
    links = [str(u) for u in (c.get("code_links") or []) if CODE_HOST_RE.search(str(u))]
    rs = c.get("code_signals") or {}
    sig: dict = dict(meta=links[:10], retrieval=bool(rs.get("any")),
                     retrieval_detail={k: v for k, v in rs.items() if k != "any"} if rs else None)
    pw: list = []
    if ccfg(cfg).get("pwc_path"):
        from tbsretr.codeindex import pwc_index
        hits = pwc_index(ccfg(cfg)["pwc_path"]).lookup(c.get("arxiv"), c.get("title"))
        pw = [h["repo"] for h in hits if h["official"] or h["mentioned"]][:5]
    sig["pwc"] = pw
    sig["openalex_fulltext"] = None
    if not (links or sig["retrieval"] or pw) and ccfg(cfg).get("openalex_flag", True):
        sig["openalex_fulltext"] = (oa_flag or openalex_fulltext_flag)(c, cfg)
    sig["any"] = bool(links or sig["retrieval"] or pw or sig["openalex_fulltext"])
    return sig


# ----------------------------------------------------------------------------------------------- S3b: links
def norm_text(s: str) -> str:
    s = s.replace("\u00ad", "")
    # PDF text breaks URLs at line ends: "https://github.com/foo/\nbar"
    return re.sub(r"((?:github|gitlab|bitbucket|zenodo|codeocean|figshare|osf)\S*?[/\-_.])\s*\n\s*(?=[A-Za-z0-9])",
                  r"\1", s)


HOST_RES = [
    ("github", re.compile(r"github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)", re.I)),
    ("gitlab", re.compile(r"gitlab\.[a-z.]+/([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+)", re.I)),
    ("bitbucket", re.compile(r"bitbucket\.org/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)", re.I)),
    ("zenodo", re.compile(r"(?:zenodo\.org/(?:record|records|doi/10\.5281/zenodo\.)/?|10\.5281/zenodo\.)(\d{4,9})",
                          re.I)),
    ("codeocean", re.compile(r"(?:codeocean\.com/capsule/(\d+)|10\.24433/CO\.([0-9a-f.\-]+))", re.I)),
    ("figshare", re.compile(r"(?:figshare\.com/\S+|10\.6084/m9\.figshare\.(\d+))", re.I)),
    ("osf", re.compile(r"osf\.io/([a-z0-9]{5})\b", re.I)),
    ("cran", re.compile(r"cran\.r-project\.org/(?:web/packages/|package=)([A-Za-z0-9.]+)", re.I)),
    ("pypi", re.compile(r"pypi\.org/project/([A-Za-z0-9_.\-]+)", re.I)),
    ("bioconductor", re.compile(r"bioconductor\.org/packages/(?:release/bioc/html/|devel/bioc/html/)?"
                                r"([A-Za-z0-9.]+)", re.I)),
    ("sourceforge", re.compile(r"sourceforge\.net/projects/([A-Za-z0-9_.\-]+)", re.I)),
]
OWN_POS = re.compile(r"(code availability|software availability|data and code availability|availability of "
                     r"(data|code)|(our|the) (source )?codes? (is|are|has been|have been|will be)|source codes? "
                     r"(is|are|and|for)|(code|codes|software|implementation|scripts?|package|toolbox|notebooks?) "
                     r"(used|developed|written)? ?(in|for) this (study|paper|work|article|manuscript)|(is|are) "
                     r"(publicly |freely |openly )?available (at|from|on|via|in)|we (provide|release|share|make)|"
                     r"(our|the proposed|this) (implementation|method|package|code|software|framework)|(made|make) "
                     r"(publicly )?available|github repository|can be (found|downloaded|accessed) at|open[- ]source "
                     r"(implementation|code|package)|reproduc)", re.I)
OWN_STRONG = re.compile(r"(code availability|software availability|our (source )?code|code (used|developed|written) "
                        r"(in|for) this|(scripts?|codes?|software) (used|developed) (in|for) this (study|paper|work)|"
                        r"(the )?(source )?codes? (and [a-z ]{1,30})?(is|are) (publicly |freely |openly )?available|"
                        r"implementation (of|for) (our|the proposed|this)|we (have )?(made|make|release|provide) "
                        r".{0,40}(code|implementation|package|software))", re.I)
THIRD = re.compile(r"(we used|using the|was used|were used|implemented (in|with|using)|et al\.|\[\d+\]|was performed "
                   r"(with|using)|was computed (with|using)|package \(|toolbox \(|library \()", re.I)
REF_HDR = re.compile(r"\n\s*(references|bibliography|literature cited|works cited|reference list)\s*\n", re.I)
CODE_AVAIL = re.compile(r"(code|software|data and code|code and data|data and software) availability", re.I)
ON_REQUEST = re.compile(r"(code|codes|software|scripts?|implementation)[^.]{0,160}?(available|provided|shared)"
                        r"[^.]{0,60}?(up)?on (reasonable )?request|(available|provided) (up)?on (reasonable )?request "
                        r"from the (corresponding )?authors?", re.I)
GENERIC_REPOS = {"pytorch/pytorch", "tensorflow/tensorflow", "numpy/numpy", "scipy/scipy", "jax-ml/jax", "google/jax",
                 "scikit-learn/scikit-learn", "pandas-dev/pandas", "matplotlib/matplotlib", "deepmind/ferminet",
                 "netket/netket", "cosmosis/cosmosis", "google/flax", "lesgourg/class_public", "cmbant/camb",
                 "joezuntz/cosmosis", "10xgenomics/cellranger", "lh3/seqtk", "samtools/samtools",
                 "broadinstitute/gatk", "rordenlab/dcm2niix", "illumina/hap.py"}
KEEP_SUFFIX = (".jl", ".py", ".github.io", ".r", ".js", ".rs", ".m")
FETCHABLE = {"github", "gitlab", "bitbucket", "zenodo", "cran", "pypi", "bioconductor", "sourceforge"}


def _clean_repo(r: str) -> str:
    # PDF text glues the next word onto the URL ("repo.Acknowledgments"): keep only a known package suffix
    r = re.sub(r"[.,;:)\]]+$", "", r)
    low = r.lower()
    for suf in KEEP_SUFFIX:
        i = low.find(suf)
        if i > 0 and (i + len(suf) == len(low) or low[i + len(suf)] == "."):
            return r[:i + len(suf)]
    return r.split(".")[0] if "." in r else r


def canon(kind: str, m: re.Match) -> tuple[str, dict]:
    g = [x for x in m.groups() if x]
    if kind in ("github", "bitbucket"):
        o, r = g[0], _clean_repo(g[1])
        if r.lower().endswith(".git"):
            r = r[:-4]
        host = "github.com" if kind == "github" else "bitbucket.org"
        return f"{host}/{o}/{r}".lower(), dict(owner=o, repo=r, url=f"https://{host}/{o}/{r}")
    if kind == "gitlab":
        p = re.sub(r"[.,;:)\]]+$", "", g[0]).split("/-/")[0]
        parts = p.split("/")
        parts[-1] = _clean_repo(parts[-1])
        path = "/".join(parts[:3])
        hm = re.search(r"gitlab\.[a-z.]+", m.group(0), re.I)
        host = hm.group(0).lower().rstrip(".") if hm else "gitlab.com"
        return f"{host}/{path}".lower(), dict(path=path, gl_host=host, url=f"https://{host}/{path}")
    if kind in ("cran", "pypi", "bioconductor", "sourceforge"):
        name = g[0].removesuffix(".html").rstrip(".,;)")
        return f"{kind}:{name}".lower(), dict(name=name)
    if kind == "zenodo":
        return f"zenodo:{g[0]}", dict(record=g[0], url=f"https://zenodo.org/records/{g[0]}")
    return f"{kind}:{(g[0] if g else m.group(0))[:80].rstrip('.,;)')}".lower(), {}


def _title_tokens(t: str | None) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", t or "")} - {
        "with", "from", "using", "based", "model", "models", "data", "method", "methods", "analysis", "through",
        "their", "that", "this", "towards", "into", "via"}


def extract_links(fulltext: str, cand: dict) -> tuple[list[dict], dict]:
    """Code-host links of the full text + metadata with ownership features and rule score (own_score >= 3 own,
    1..2 maybe, <= 0 third party).  Also returns paper-level facts (Code Availability text, on-request)."""
    s = norm_text(fulltext or "")
    refm = list(REF_HDR.finditer(s))
    ref_start = refm[-1].start() if refm and refm[-1].start() > len(s) * 0.4 else len(s)
    ca = CODE_AVAIL.search(s)
    found: dict[str, dict] = {}
    for kind, rx in HOST_RES:
        for m in rx.finditer(s):
            key, info = canon(kind, m)
            if kind == "github" and info["owner"].lower() in ("orgs", "topics", "features", "about", "sponsors"):
                continue
            e = found.setdefault(key, dict(key=key, host=kind, n=0, ctxs=[], in_refs_only=True, pos=0, strong=0,
                                           third=0, in_code_avail=False, from_meta=False, **info))
            e["n"] += 1
            if len(e["ctxs"]) < 3:
                e["ctxs"].append(re.sub(r"\s+", " ", s[max(0, m.start() - 350): m.end() + 150]))
            e["in_refs_only"] = e["in_refs_only"] and m.start() >= ref_start
            near = s[max(0, m.start() - 250): m.start()]
            e["pos"] += bool(OWN_POS.search(near))
            e["strong"] += bool(OWN_STRONG.search(near))
            e["third"] += bool(THIRD.search(s[max(0, m.start() - 120): m.start()]))
            if ca and 0 <= m.start() - ca.start() < 1500:
                e["in_code_avail"] = True
    for u in list(cand.get("code_links") or []) + list((cand.get("code_signals") or {}).get("pwc") or []) + \
            list((cand.get("code_signals") or {}).get("gh_readme") or []):
        for kind, rx in HOST_RES:
            m = rx.search(str(u))
            if m:
                key, info = canon(kind, m)
                e = found.setdefault(key, dict(key=key, host=kind, n=0, ctxs=[], in_refs_only=False, pos=0, strong=0,
                                               third=0, in_code_avail=False, from_meta=False, **info))
                e["from_meta"] = True
                break
    gh_readme = {str(u).lower().rstrip("/").removeprefix("https://") for u in
                 (cand.get("code_signals") or {}).get("gh_readme") or []}
    pwc = {str(u).lower().rstrip("/").removeprefix("https://") for u in
           (cand.get("code_signals") or {}).get("pwc") or []}
    tt = _title_tokens(cand.get("title"))
    for e in found.values():
        name = e["key"].split("/")[-1].split(":")[-1]
        e["name_in_title"] = bool(name) and any(len(w) >= 4 and (w in name.lower() or name.lower() in w) for w in tt)
        e["generic"] = "/".join(e["key"].split("/")[1:3]) in GENERIC_REPOS
        e["readme_cites_paper"] = e["key"] in gh_readme
        e["pwc_official"] = e["key"] in pwc
        sc = 3 * (e["strong"] > 0) + 2 * e["in_code_avail"] + (e["pos"] > 0) + e["name_in_title"] + e["from_meta"] \
            + 3 * e["readme_cites_paper"] + 2 * e["pwc_official"] - 2 * e["in_refs_only"] \
            - (e["third"] > 0 and e["strong"] == 0) - 4 * e["generic"]
        e["own_score"] = sc
        # thresholds from the 2026-11-11 study (224 papers, own code by reading the contexts): own >= 3; a
        # non-reference link with score 0..2 is "maybe" (README check, then Jev or keep).  Paper level, maybe->own:
        # recall 0.96, precision 0.98.
        e["own_guess"] = "own" if sc >= 3 else ("maybe" if sc >= 0 and not e["in_refs_only"] else "third")
    paper = dict(code_availability=re.sub(r"\s+", " ", s[ca.start(): ca.start() + 1200]) if ca else None,
                 on_request=bool(ON_REQUEST.search(s[ca.start(): ca.start() + 1500] if ca else s)),
                 ref_start_frac=round(ref_start / max(1, len(s)), 2))
    return sorted(found.values(), key=lambda e: -e["own_score"]), paper


# ----------------------------------------------------------------------------------------------- S3b: resolve
CODE_EXT = re.compile(r"\.(py|r|jl|m|c|cc|cpp|h|hpp|f90|f|rs|go|java|ipynb|sh|zip|tar\.gz|tgz|tar)$", re.I)
DATA_EXT = re.compile(r"\.(csv|tsv|txt|h5|hdf5|nc|mat|npy|npz|json|xlsx?|fits|parquet|vcf|bam|fastq|gz|nii)$", re.I)


def resolve(link: dict, cfg: dict, get=None, gh=None) -> dict:
    """Host metadata of one link: exists, empty, kind (code / data / package), license, language, size_kb, ..."""
    get = get or (lambda u: http_json(u, cfg))
    gh = gh or (lambda p: gh_json(p, cfg))
    h = link["host"]
    out: dict = dict(host=h, key=link["key"], url=link.get("url"), exists=False, empty=None, kind="code",
                     license=None, language=None, size_kb=None)
    try:
        if h == "github":
            st, d = gh(f"repos/{link['owner']}/{link['repo']}")
            if st == 200 and isinstance(d, dict):
                out.update(exists=True, empty=(d.get("size") or 0) == 0, size_kb=d.get("size"),
                           language=d.get("language"), archived=d.get("archived"), fork=d.get("fork"),
                           default_branch=d.get("default_branch"), stars=d.get("stargazers_count"),
                           license=(d.get("license") or {}).get("spdx_id"), full_name=d.get("full_name"),
                           url=d.get("html_url") or out["url"], private=d.get("private"))
            out["status"] = st
        elif h == "gitlab":
            gl_host = link.get("gl_host") or "gitlab.com"
            st, d = get(f"https://{gl_host}/api/v4/projects/{urllib.parse.quote(link['path'], safe='')}?license=true")
            if st == 200 and isinstance(d, dict):
                out.update(exists=True, empty=bool(d.get("empty_repo")), default_branch=d.get("default_branch"),
                           license=(d.get("license") or {}).get("key"), stars=d.get("star_count"),
                           url=d.get("web_url") or out["url"])
            out["status"] = st
        elif h == "bitbucket":
            st, d = get(f"https://api.bitbucket.org/2.0/repositories/{link['owner']}/{link['repo']}")
            if st == 200 and isinstance(d, dict):
                out.update(exists=True, empty=(d.get("size") or 0) == 0, size_kb=(d.get("size") or 0) // 1024,
                           language=d.get("language"), url=((d.get("links") or {}).get("html") or {}).get("href")
                           or out["url"])
            out["status"] = st
        elif h == "zenodo":
            st, d = get(f"https://zenodo.org/api/records/{link['record']}")
            if st == 200 and isinstance(d, dict):
                md = d.get("metadata") or {}
                files = [(f.get("key") or "", int(f.get("size") or 0)) for f in d.get("files") or []]
                rtype = ((md.get("resource_type") or {}).get("type") or "").lower()
                code_files = [f for f in files if CODE_EXT.search(f[0])]
                kind = "code" if rtype == "software" or (code_files and not all(DATA_EXT.search(f[0])
                                                                                 for f in code_files)) else "data"
                out.update(exists=True, empty=not files, kind=kind, resource_type=rtype,
                           license=(md.get("license") or {}).get("id") if isinstance(md.get("license"), dict)
                           else md.get("license"), size_kb=sum(s for _, s in files) // 1024,
                           files=[f[0] for f in files][:20], related=[r.get("identifier") for r in
                                                                      md.get("related_identifiers") or []][:10])
            out["status"] = st
        elif h == "cran":
            st, d = get(f"https://crandb.r-pkg.org/{link['name']}")
            if st == 200 and isinstance(d, dict) and d.get("Package"):
                out.update(exists=True, empty=False, kind="package", license=d.get("License"), language="R",
                           repo_urls=[u.strip() for u in re.split(r"[,\s]+", str(d.get("URL") or "")) if
                                      "git" in u][:3], url=f"https://cran.r-project.org/package={d['Package']}")
            out["status"] = st
        elif h == "pypi":
            st, d = get(f"https://pypi.org/pypi/{link['name']}/json")
            if st == 200 and isinstance(d, dict):
                info = d.get("info") or {}
                urls = [u for u in (info.get("project_urls") or {}).values() if u and "git" in u]
                out.update(exists=True, empty=False, kind="package", license=info.get("license") or None,
                           language="Python", repo_urls=urls[:3], url=f"https://pypi.org/project/{info.get('name')}")
            out["status"] = st
        elif h == "bioconductor":
            # no light-weight JSON endpoint per package: S3c clones git.bioconductor.org and decides there
            out.update(exists=True, empty=False, kind="package", language="R",
                       url=f"https://bioconductor.org/packages/{link['name']}",
                       git=f"https://git.bioconductor.org/packages/{link['name']}", unverified=True)
        elif h == "sourceforge":
            st, d = get(f"https://sourceforge.net/rest/p/{link['name']}")
            out.update(exists=st == 200, empty=False if st == 200 else None)
            out["status"] = st
        else:
            out.update(kind="unsupported", exists=None)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def readme_cites(link: dict, cand: dict, cfg: dict, gh=None) -> bool | None:
    """GitHub README mentions the paper's DOI / arXiv id / title (strong ownership evidence)."""
    if link["host"] != "github":
        return None
    gh = gh or (lambda p: gh_json(p, cfg, raw=True))
    st, body = gh(f"repos/{link['owner']}/{link['repo']}/readme")
    if st != 200 or not isinstance(body, str):
        return None
    low = body.lower()
    doi = _norm_doi(cand.get("doi"))
    ax = str(cand.get("arxiv") or "").lower()
    title = re.sub(r"\W+", " ", (cand.get("title") or "").lower()).strip()
    return bool((doi and doi in low) or (ax and ax in low) or (len(title) > 25 and title[:60] in
                                                                 re.sub(r"\W+", " ", low)))


JEV_OWN_Q = {
    "own_code": {
        "type": "noul",
        "instructions": "Is `link.url` the authors' own implementation (code the authors of `paper` wrote and "
                        "released for this paper or for the method it introduces), as opposed to a third-party tool, "
                        "library or dataset they used or cite?  Use `link.contexts` (text around the link in the "
                        "paper).",
        "criteria": {"true": "The paper says its code / software / implementation is available at this link, or the "
                             "repository is named after the paper's method and released by its authors.",
                     "false": "The link is a tool, library, database or dataset by others that the paper used, "
                              "compared with, or cites, or a data-only deposit."},
    }
}


def jev_own(client, it, cand: dict, link: dict) -> float | None:
    state = {"paper": {"title": cand.get("title") or "", **({"abstract": cand["abstract"][:1500]}
                                                             if cand.get("abstract") else {})},
             "link": {"url": link.get("url") or link["key"], "contexts": [c[:500] for c in link.get("ctxs", [])][:3]}}
    r = client.ask(state, stage="S3b", item=it.item_id, questions=JEV_OWN_Q)
    a = (r.get("answers") or {}).get("own_code") or {}
    return float(a["noul"]) if "noul" in a else None


def decide(links: list[dict], paper: dict, resolved: dict[str, dict]) -> dict:
    """links carry verdict own|third (after rules / Jev).  -> dict(keep, reason, primary, own)."""
    own = [l for l in links if l.get("verdict") == "own"]
    if not links:
        return dict(keep=False, reason="code_on_request" if paper.get("on_request") else "no_code_link",
                    primary=None, own=[])
    if not own:
        return dict(keep=False, reason="code_on_request" if paper.get("on_request") else "third_party_code_only",
                    primary=None, own=[])
    ok = []
    for l in own:
        r = resolved.get(l["key"]) or {}
        if l["host"] not in FETCHABLE:
            continue
        if r.get("exists") and not r.get("empty") and r.get("kind") in ("code", "package"):
            ok.append((l, r))
    if not ok:
        rs = [resolved.get(l["key"]) or {} for l in own]
        if any(r.get("exists") and r.get("kind") == "data" for r in rs) and \
                not any(r.get("exists") and r.get("kind") in ("code", "package") for r in rs):
            reason = "data_only_no_code"
        elif all(l["host"] not in FETCHABLE for l in own):
            reason = "code_host_unsupported"
        else:
            reason = "code_repo_unavailable"
        return dict(keep=False, reason=reason, primary=None, own=[l["key"] for l in own])
    pref = {"github": 0, "gitlab": 1, "bitbucket": 2, "zenodo": 3, "cran": 4, "bioconductor": 4, "pypi": 5,
            "sourceforge": 6}
    ok.sort(key=lambda lr: (pref.get(lr[0]["host"], 9), -lr[0]["own_score"]))
    l, r = ok[0]
    return dict(keep=True, reason=None, primary=dict(r, own_score=l["own_score"], verdict_by=l.get("verdict_by")),
                own=[x[0]["key"] for x in ok])


def s3b_record(fulltext: str, cand: dict, cfg: dict, *, jev: Callable | None = None, get=None, gh=None,
               gh_raw=None) -> dict:
    """The whole S3b decision for one paper.  jev(link) -> probability of own code, or None (rules decide)."""
    t0 = time.time()
    links, paper = extract_links(fulltext, cand)
    th = ccfg(cfg).get("jev_own_threshold", 0.5)
    resolved: dict[str, dict] = {}
    for l in links:
        if l["generic"]:
            l["verdict"], l["verdict_by"] = "third", "rules:generic"
            continue
        if l["own_guess"] == "maybe" and l["host"] == "github":
            # README citing the paper settles an ambiguous GitHub link without a model call
            rc = readme_cites(l, cand, cfg, gh=gh_raw)
            l["readme_cites_paper"] = l["readme_cites_paper"] or bool(rc)
            if rc:
                l["own_score"] += 3
                l["own_guess"] = "own"
        if l["own_guess"] == "own":
            l["verdict"], l["verdict_by"] = "own", "rules"
        elif l["own_guess"] == "third":
            l["verdict"], l["verdict_by"] = "third", "rules"
        else:
            p = jev(l) if jev else None
            l["jev_own_p"] = p
            if p is None:
                l["verdict"], l["verdict_by"] = "own", "rules:maybe->own"
            else:
                l["verdict"], l["verdict_by"] = ("own" if p >= th else "third"), "jev"
    for l in links:
        if l["verdict"] == "own" and l["key"] not in resolved:
            resolved[l["key"]] = resolve(l, cfg, get=get, gh=gh)
            for u in resolved[l["key"]].get("repo_urls") or []:      # package -> its source repository
                m = HOST_RES[0][1].search(u)
                if m:
                    key, info = canon("github", m)
                    if key not in resolved:
                        sub = dict(key=key, host="github", **info)
                        resolved[key] = resolve(sub, cfg, get=get, gh=gh)
                        resolved[l["key"]].setdefault("source_repo", resolved[key])
    dec = decide(links, paper, resolved)
    if dec["keep"] and dec["primary"].get("kind") == "package" and \
            (dec["primary"].get("source_repo") or {}).get("exists"):
        dec["primary"] = dict(dec["primary"]["source_repo"], package=dec["primary"].get("url"),
                              package_license=dec["primary"].get("license"), own_score=dec["primary"]["own_score"],
                              verdict_by=dec["primary"].get("verdict_by"))
    slim = [{k: v for k, v in l.items() if k != "ctxs"} | {"ctx": (l.get("ctxs") or [""])[0][:400]} for l in links]
    return dict(dec, paper=paper, links=slim, resolved=resolved, sec=round(time.time() - t0, 2),
                method="rules+jev" if jev else "rules")


def summary_line(rec: dict) -> str:
    p = rec.get("primary") or {}
    return f"{p.get('url')} license={p.get('license')} lang={p.get('language')} size_kb={p.get('size_kb')}"


def write_cache(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
