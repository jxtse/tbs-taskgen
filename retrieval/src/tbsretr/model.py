"""Data model: retrieval target, normalised paper candidate, non-paper resource; query planning & routing (no LLM)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field

from .util import arxiv_from_doi, code_data_clues, norm_doi, title_key

# ------------------------------------------------------------------ routing
# arXiv only for these archives (decision 2026-11-05); Europe PMC only for biomedical targets.
_ARXIV_RULES = [
    (r"astro|cosmolog|galax|stellar|exoplanet|transit|weak lensing|3x2pt|cmb\b", ["astro-ph"]),
    (r"quantum|spin|heisenberg|neural quantum|variational monte carlo|wave ?function|qubit", ["quant-ph", "cond-mat"]),
    (r"material|indentation|crystal|fracture|alloy|condensed|phonon|dft\b|lattice", ["cond-mat"]),
    (r"\bmri\b|susceptibility mapping|\bqsm\b|tomograph|medical imaging|ultrasound", ["physics"]),
    (r"fluid|plasma|optic|acoustic|geophys|atmospher|ocean|climate|seism|data assimilation|4d-?var|kalman|"
     r"reactor|nuclear|accelerator|physics", ["physics"]),
    (r"statist|bayes|regression|small[- ]area|survey|fay|herriot|causal|mediation|likelihood|estimator", ["stat"]),
    (r"optimi[sz]ation|integer program|column generation|branch-and|vehicle routing|combinator|graph|"
     r"numerical analysis|pde|finite element|proof|lean\b|theorem|algebra", ["math", "cs"]),
    (r"machine learning|deep learning|neural network|computer vision|segmentation|reinforcement|nlp|algorithm",
     ["cs", "stat"]),
]
_BIOMED = re.compile(r"biolog|medic|clinic|genom|single[- ]cell|rna|dna|protein|cell|tumou?r|cancer|immun|neuro|"
                     r"brain|mri|susceptibility mapping|qsm|patholog|sequenc|variant|somatic|epidem|pharma|ecolog|"
                     r"microb|spatial-oncology|imaging mass cytometry", re.I)
_ARXIV_ARCHIVES = {"astro-ph", "physics", "cond-mat", "quant-ph", "stat", "math", "cs"}


def route_arxiv(text: str) -> list[str]:
    t = text.lower()
    cats: list[str] = []
    for pat, cs in _ARXIV_RULES:
        if re.search(pat, t):
            cats += [c for c in cs if c not in cats]
    return [c for c in cats if c in _ARXIV_ARCHIVES][:3]


def is_biomed(text: str) -> bool:
    return bool(_BIOMED.search(text or ""))


_STOP = set("a an and or of for the in on with to from by via using use based its into at as is are be "
            "method methods approach analysis estimation model models".split())
_EXACT = re.compile(r"\b(?:[A-Za-z]*[A-Z][A-Za-z\d]*-[A-Z][A-Za-z\d]*|[A-Za-z]*[A-Z][a-z]+[A-Z][A-Za-z]*|[A-Z]{2,}[a-z]*[A-Z]*\d*|"
                    r"[A-Za-z]+\d+[A-Za-z\d]*|[A-Za-z]+\.(?:rs|jl|py)|[A-Z][a-z]+[A-Z]+)\b")
_GENERIC_ACR = {"RNA", "DNA", "MRI", "GRE", "PCA", "MCMC", "LP", "MILP", "DP", "AR", "EB", "AF", "OK", "CSV", "PT",
                "API", "GPU", "CPU", "ML", "AI", "DL", "NIfTI", "BIDS", "JSON", "MAD", "ROI", "QC", "SNR", "ODE", "PDE"}


def exact_terms(texts: list[str], libs: list[str]) -> list[str]:
    """Tool / author / model names worth an exact Crossref query (SoupX, CellBender, Oliver-Pharr, HMcode...)."""
    out: list[str] = []
    for t in texts:
        for m in _EXACT.findall(t or ""):
            if m not in _GENERIC_ACR and len(m) >= 3 and m.lower() not in {x.lower() for x in out}:
                out.append(m)
    # drop terms contained in a longer extracted term (FAST in FAST-PT, SHARP in V-SHARP)
    out = [x for x in out if not any(x != y and x in y for y in out)]
    _common = {"numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "scikit-image", "tifffile", "emcee",
               "nibabel", "networkx", "torch", "jax", "sympy", "statsmodels", "numba", "h5py", "xarray"}
    for lib in libs or []:
        name = re.sub(r"\s*\(.*?\)", "", lib).strip()
        if name and name.lower() not in _common and name.lower() not in {x.lower() for x in out} \
                and not name.startswith("Rust"):
            out.append(name)
    return out[:8]


def informative_words(q: str, n: int = 5) -> list[str]:
    ws = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", q) if w.lower() not in _STOP and len(w) > 2]
    return ws[:n]


# ------------------------------------------------------------------ target
@dataclass
class Target:
    id: str
    queries: list[str]                       # 2–6 short keyword queries (from the upstream query writer)
    description: str = ""                    # one-paragraph English method/task description (Exa paper channel)
    field: str = ""
    name: str = ""
    topic: str = ""                          # short English phrase for resource (code/data/docs) queries
    libs: list[str] = dc_field(default_factory=list)
    seed_dois: list[str] = dc_field(default_factory=list)
    exclude_dois: list[str] = dc_field(default_factory=list)   # e.g. task-source papers for this target
    openalex_filter: str = ""                # optional extra OpenAlex filter, e.g. primary_topic.subfield.id:3104
    arxiv_cats: list[str] | None = None      # None -> routed from text
    biomed: bool | None = None               # None -> routed from text

    def text(self) -> str:
        return " ".join([self.id, self.name, self.field, self.description, " ".join(self.queries), " ".join(self.libs)])

    def routed_arxiv(self) -> list[str]:
        return self.arxiv_cats if self.arxiv_cats is not None else route_arxiv(self.text())

    def routed_biomed(self) -> bool:
        return self.biomed if self.biomed is not None else is_biomed(self.text())

    def exact(self) -> list[str]:
        return exact_terms(self.queries + [self.name], self.libs)

    def topic_phrase(self) -> str:
        """Short English phrase naming the method/task (for Exa resource queries)."""
        if self.topic:
            return self.topic
        if self.name:
            return self.name
        return self.queries[0] if self.queries else self.id.split(":", 1)[-1].replace("-", " ")

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


# ------------------------------------------------------------------ candidates
@dataclass
class Paper:
    title: str
    doi: str | None = None
    arxiv: str | None = None
    openalex: str | None = None
    s2: str | None = None
    pmid: str | None = None
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    type: str = ""
    language: str = ""
    oa_url: str | None = None
    is_oa: bool | None = None
    cited_by: int | None = None
    url: str | None = None
    referenced_works: list[str] = dc_field(default_factory=list)
    code_links: list[str] = dc_field(default_factory=list)
    data_links: list[str] = dc_field(default_factory=list)
    code_words: bool = False
    data_words: bool = False
    has_data_flag: bool = False              # Europe PMC hasData etc.
    origin: str = ""                         # source that produced this record

    def __post_init__(self):
        self.doi = norm_doi(self.doi)
        if not self.arxiv:
            self.arxiv = arxiv_from_doi(self.doi)
        if self.arxiv:
            self.arxiv = re.sub(r"v\d+$", "", self.arxiv.lower())

    def keys(self) -> list[str]:
        ks = []
        if self.doi:
            ks.append("doi:" + self.doi)
        if self.arxiv:
            ks.append("arxiv:" + self.arxiv)
        if self.openalex:
            ks.append("oa:" + self.openalex)
        if self.s2:
            ks.append("s2:" + self.s2)
        tk = title_key(self.title)
        if len(tk) >= 25:
            ks.append("t:" + tk)
        return ks

    def add_clues(self, *extra: str | None):
        c = code_data_clues(self.abstract, *extra)
        self.code_links = sorted(set(self.code_links) | set(c["code_links"]))
        self.data_links = sorted(set(self.data_links) | set(c["data_links"]))
        self.code_words = self.code_words or c["code_words"]
        self.data_words = self.data_words or c["data_words"]
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("referenced_works", None)
        return d


@dataclass
class Resource:
    url: str
    kind: str                                # code | data | docs | web | paper(unresolved)
    title: str = ""
    snippet: str = ""
    published: str | None = None
    origin: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RankedList:
    """One ranked result list from one call; the unit RRF fuses."""
    name: str                                # e.g. openalex.search, exa.code
    query: str
    papers: list[Paper] = dc_field(default_factory=list)
    resources: list[Resource] = dc_field(default_factory=list)
