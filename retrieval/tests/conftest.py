"""Offline test helpers: replay recorded responses (tests/fixtures/*.json) by (method, secret-stripped URL, body)."""
import json
from pathlib import Path

import pytest

from tbsretr.config import load_config
from tbsretr.ledger import Http, Ledger
from tbsretr.sources import ALL
from tbsretr.util import strip_secret_params

FX = Path(__file__).parent / "fixtures"


def load_fixtures():
    return [json.loads(p.read_text()) for p in sorted(FX.glob("*.json"))]


class Replay:
    """Transport that serves recorded responses. Same request may be recorded several times (e.g. 429 then 200):
    they are served in recording order. Unknown requests -> 599 so tests fail loudly."""

    def __init__(self, fixtures=None, extra=None):
        self.queue = {}
        for r in fixtures if fixtures is not None else load_fixtures():
            self.queue.setdefault(self._k(r["method"], r["url"], r["body"]), []).append(r)
        self.extra = extra or {}
        self.seen = []

    @staticmethod
    def _k(method, url, body):
        return method, strip_secret_params(url), json.dumps(body, sort_keys=True)

    def __call__(self, method, url, headers, body):
        self.seen.append((method, url, headers, body))
        k = self._k(method, url, body)
        if k in self.extra:
            st, hd, b = self.extra[k]
            return st, hd, b
        q = self.queue.get(k)
        if not q:
            return 599, {}, b"no fixture"
        r = q.pop(0) if len(q) > 1 else q[0]
        return r["status"], r.get("headers", {}), r["response"].encode()


@pytest.fixture
def cfg(tmp_path):
    c = load_config(overrides={"state_dir": str(tmp_path / "state"), "leak_refs": None})
    for s in c["sources"].values():          # no real sleeping in tests
        s["min_interval_s"] = 0.0
        s["backoff_429_s"] = 0.0
    return c


@pytest.fixture
def keys(monkeypatch):
    for k in ("OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "EXA_API_KEY_SECONDARY", "TAVILY_API_KEY_SECONDARY"):
        monkeypatch.setenv(k, "SECRET-" + k + "-1234567890")
    return True


def make_sources(cfg, transport) -> tuple[dict, Ledger, Http]:
    led = Ledger(cfg["state_dir"], run="test")
    http = Http(cfg, led, transport=transport)
    srcs: dict = {n: c(cfg, http) for n, c in ALL.items()}
    return srcs, led, http
