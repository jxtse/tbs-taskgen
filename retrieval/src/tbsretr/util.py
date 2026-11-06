"""Small pure helpers: identifier normalisation, text cleanup, link classification, secret redaction."""
from __future__ import annotations

import html
import os
import re
import urllib.parse

# ------------------------------------------------------------------ identifiers
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>|]+)", re.I)
_ARXIV_NEW = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv[:\s]\s*|10\.48550/arxiv\.)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:\s]\s*)([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.I)


def norm_doi(d: str | None) -> str | None:
    if not d:
        return None
    d = urllib.parse.unquote(str(d)).strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    d = d.rstrip(".,;)")
    return d if d.startswith("10.") else None


def doi_from_text(s: str | None) -> str | None:
    if not s:
        return None
    m = _DOI_RE.search(urllib.parse.unquote(s))
    if not m:
        return None
    d = m.group(1)
    d = re.sub(r"(/(full|abstract|pdf|epdf|meta|html|fulltext))+/?$", "", d, flags=re.I)
    d = re.sub(r"\.(pdf|html?)$", "", d, flags=re.I)
    return norm_doi(d)


def arxiv_from_text(s: str | None) -> str | None:
    if not s:
        return None
    m = _ARXIV_NEW.search(s) or _ARXIV_OLD.search(s)
    return m.group(1).lower() if m else None


def arxiv_from_doi(doi: str | None) -> str | None:
    if doi and doi.startswith("10.48550/arxiv."):
        return doi.split("arxiv.", 1)[1]
    return None


def title_key(t: str | None) -> str:
    t = html.unescape(t or "").lower()
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"[^a-z0-9]", "", t)[:80]


def clean(s: str | None) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def inv2text(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {p: w for w, ps in inv.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def latin_ratio(s: str) -> float:
    s = s or ""
    return sum(ch.isascii() for ch in s) / max(1, len(s))


def strip_site_suffix(title: str) -> str:
    """'Foo bar | Nature Methods' / 'Foo - ScienceDirect' / 'GitHub - x/y: desc' -> core title."""
    t = clean(title)
    t = re.sub(r"\s+[|\-–—]\s+(ScienceDirect|PubMed|PMC|Nature[^|]*|Springer[^|]*|Wiley[^|]*|arXiv[^|]*|"
               r"IOPscience|ResearchGate|Oxford Academic|Semantic Scholar|bioRxiv|medRxiv|NCBI|"
               r"Taylor & Francis[^|]*|IEEE Xplore|ACM Digital Library|Europe PMC|SAGE Journals|PLOS[^|]*|"
               r"Physical Review[^|]*|APS[^|]*|Cell Press|ADS|NASA/ADS)\s*$", "", t, flags=re.I)
    t = re.sub(r"^\[\d{4}\.\d{4,5}\]\s*", "", t)  # arXiv '[2101.00001] Title'
    return t.strip()


# ------------------------------------------------------------------ code / data clues
CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "sourceforge.net", "codeocean.com",
              "git.", "pypi.org", "cran.r-project.org", "bioconductor.org", "juliahub.com", "crates.io",
              "anaconda.org", "huggingface.co/spaces")
DATA_HOSTS = ("zenodo.org", "figshare.com", "datadryad.org", "dryad", "kaggle.com/datasets", "huggingface.co/datasets",
              "physionet.org", "openneuro.org", "ncbi.nlm.nih.gov/geo", "ebi.ac.uk/arrayexpress", "ebi.ac.uk/ena",
              "ebi.ac.uk/biostudies", "osf.io", "dataverse", "pangaea.de", "data.gov", "earthdata.nasa.gov",
              "archive.ics.uci.edu", "registry.opendata.aws", "lambda.gsfc.nasa.gov", "cellxgene", "10xgenomics.com/datasets",
              "idr.openmicroscopy.org", "materialsproject.org", "nomad-lab.eu", "copernicus.eu", "ecmwf.int",
              "mast.stsci.edu", "irsa.ipac", "sdss.org", "des.ncsa", "lsst", "ukdataservice", "census.gov", "ipums")
DOCS_HINT = ("readthedocs", "docs.", "/docs", "/documentation", "/manual", "/tutorial", "/userguide", "/user-guide",
             ".github.io", "/vignettes", "/articles/", "/wiki")
PAPER_HOSTS = ("arxiv.org", "doi.org", "nature.com", "sciencedirect.com", "springer.com", "wiley.com", "iop.org",
               "aps.org", "pnas.org", "biorxiv.org", "medrxiv.org", "ncbi.nlm.nih.gov/pmc", "pubmed.ncbi",
               "europepmc.org", "academic.oup.com", "ieeexplore", "dl.acm.org", "tandfonline.com", "sagepub.com",
               "jstor.org", "plos.org", "frontiersin.org", "mdpi.com", "cell.com", "science.org", "aip.org",
               "aanda.org", "semanticscholar.org", "openreview.net", "proceedings.mlr.press", "neurips.cc",
               "jmlr.org", "siam.org", "informs.org", "ametsoc.org", "agupubs", "copernicus.org/articles",
               "biomedcentral.com", "elifesciences.org", "acs.org", "rsc.org", "researchgate.net",
               "projecteuclid.org", "cambridge.org", "royalsocietypublishing.org", "journals.", "osti.gov",
               "papers.ssrn.com", "hal.science", "inria.hal", "escholarship.org")
LINK_RE = re.compile(r"https?://[^\s)\]>\"',;]+", re.I)
CODE_LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|"
                          r"codeocean\.com|sourceforge\.net)/[^\s)\]>\"',;]+", re.I)
DATA_LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:zenodo\.org|figshare\.com|datadryad\.org|osf\.io|"
                          r"physionet\.org|openneuro\.org|kaggle\.com/datasets|huggingface\.co/datasets)/[^\s)\]>\"',;]+",
                          re.I)
CODE_WORDS = re.compile(r"\b(open[- ]source|source code|code is (publicly )?available|software package|"
                        r"R package|Python package|toolbox|implemented in|github)\b", re.I)
DATA_WORDS = re.compile(r"\b(publicly available data|data are (publicly )?available|open data|dataset is available|"
                        r"benchmark data(set)?)\b", re.I)


def code_data_clues(*texts: str | None) -> dict:
    s = " ".join(t for t in texts if t)
    code = sorted({u.rstrip(".") for u in CODE_LINK_RE.findall(s)})
    data = sorted({u.rstrip(".") for u in DATA_LINK_RE.findall(s)})
    return {"code_links": code[:10], "data_links": data[:10],
            "code_words": bool(CODE_WORDS.search(s)), "data_words": bool(DATA_WORDS.search(s))}


def host_path(url: str) -> tuple[str, str]:
    p = urllib.parse.urlparse(url)
    return (p.netloc or "").lower().removeprefix("www."), p.path or "/"


def classify_url(url: str) -> str:
    """paper | code | data | docs | web"""
    h, path = host_path(url)
    hp = h + path.lower()
    if any(x in hp for x in DATA_HOSTS):
        return "data"
    if h in ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org") or h.startswith("git."):
        return "code"
    if any(x in hp for x in PAPER_HOSTS) or doi_from_text(url):
        return "paper"
    if h.endswith("github.io") or any(x in hp for x in DOCS_HINT):
        return "docs"
    if any(h == x or h.endswith("." + x) or hp.startswith(x) for x in CODE_HOSTS):
        return "code"
    return "web"


def resource_key(url: str) -> str:
    """Canonical key: GitHub/GitLab -> host/owner/repo; others -> host + path without query/fragment."""
    h, path = host_path(url)
    parts = [x for x in path.split("/") if x]
    if h in ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org") and len(parts) >= 2:
        return f"{h}/{parts[0].lower()}/{parts[1].lower().removesuffix('.git')}"
    if h in ("huggingface.co",) and len(parts) >= 3 and parts[0] in ("datasets", "spaces"):
        return f"{h}/{parts[0]}/{parts[1].lower()}/{parts[2].lower()}"
    if h == "zenodo.org" and len(parts) >= 2 and parts[0] in ("record", "records"):
        return f"zenodo.org/records/{parts[1]}"
    if h == "pypi.org" and len(parts) >= 2 and parts[0] == "project":
        return f"pypi.org/project/{parts[1].lower()}"
    return h + "/" + "/".join(parts).lower().rstrip("/")


# ------------------------------------------------------------------ secrets
SECRET_PARAMS = {"api_key", "apikey", "key", "token", "access_token", "x-api-key"}
_KEY_ENV_NAMES = ("OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "EXA_API_KEY", "EXA_API_KEY_SECONDARY",
                  "TAVILY_API_KEY", "TAVILY_API_KEY_SECONDARY")


def strip_secret_params(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True) if k.lower() not in SECRET_PARAMS]
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, urllib.parse.urlencode(q), ""))


def redact(s: str, extra_env: tuple[str, ...] = ()) -> str:
    """Remove every known key value (from env) and any api_key=... parameter from a string."""
    if not s:
        return s
    for name in _KEY_ENV_NAMES + tuple(extra_env):
        v = os.environ.get(name)
        if v and len(v) >= 6:
            s = s.replace(v, "***")
    return re.sub(r"(?i)(api_key|apikey|x-api-key|authorization)([\"']?\s*[=:]\s*[\"']?)(Bearer\s+)?[^&\s\"',}]+",
                  r"\1\2***", s)
