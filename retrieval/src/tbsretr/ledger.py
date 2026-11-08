"""SQLite ledger (calls, cost, budgets, cooldowns) + response cache + paced HTTP client.

Guarantees
  * Every outbound request is booked in `calls` with source, endpoint kind, status, latency, cost, cache flag.
  * Daily budgets (UTC calendar day: OpenAlex resets at midnight UTC) are checked *before* a request: calls,
    USD (OpenAlex/Exa) or credits (any credit-billed source). Exceeding a budget raises BudgetExceeded and the source stops for the run.
  * Monthly caps (USD or credits) are checked the same way.
  * 429 -> per-source cooldown persisted in sqlite (other processes respect it). 5xx / network -> exponential backoff.
  * Pacing: min interval between requests of the same source, shared across threads *and* processes via sqlite.
  * Cache key and stored URL never contain secrets: auth goes in headers, except OpenAlex where `api_key` is a query
    parameter; it is stripped by util.strip_secret_params before hashing/logging. Error bodies are redacted.
"""
from __future__ import annotations

import gzip
import hashlib
import ipaddress
import json
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .util import redact, strip_secret_params


class BudgetExceeded(RuntimeError):
    pass


class SourceCooling(RuntimeError):
    pass


class HTTPFailure(RuntimeError):
    def __init__(self, status: int, msg: str):
        super().__init__(f"HTTP {status}: {msg}")
        self.status = status


def utc_day(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), timezone.utc).strftime("%Y-%m-%d")


def utc_month(ts: float | None = None) -> str:
    return utc_day(ts)[:7]


SCHEMA = """
CREATE TABLE IF NOT EXISTS calls(
  id INTEGER PRIMARY KEY, ts REAL, day TEXT, source TEXT, kind TEXT, url TEXT, status INTEGER,
  latency_s REAL, cost_usd REAL DEFAULT 0, credits REAL DEFAULT 0, cached INTEGER DEFAULT 0,
  attempt INTEGER DEFAULT 0, run TEXT, target TEXT, note TEXT);
CREATE INDEX IF NOT EXISTS calls_day ON calls(day, source);
CREATE TABLE IF NOT EXISTS cooldown(source TEXT PRIMARY KEY, until REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS pace(source TEXT PRIMARY KEY, next_ok REAL);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""
CACHE_SCHEMA = "CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, ts REAL, source TEXT, url TEXT, status INTEGER, body BLOB)"


class Ledger:
    def __init__(self, state_dir: str | Path, run: str = ""):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.sqlite"
        self.cache_path = self.dir / "cache.sqlite"
        self.run = run
        self._lock = threading.Lock()
        with self._db() as db:
            db.executescript(SCHEMA)
        with self._cdb() as db:
            db.execute(CACHE_SCHEMA)

    def _db(self):
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        return _Closing(db)

    def _cdb(self):
        db = sqlite3.connect(self.cache_path, timeout=60, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        return _Closing(db)

    # ---------------------------------------------------------------- accounting
    def usage(self, source: str, day: str | None = None, month: bool = False) -> dict:
        col, val = ("substr(day,1,7)", utc_month()) if month else ("day", day or utc_day())
        with self._db() as db:
            r = db.execute(f"SELECT COUNT(*), COALESCE(SUM(cost_usd),0), COALESCE(SUM(credits),0) FROM calls "
                           f"WHERE {col}=? AND source=? AND cached=0", (val, source)).fetchone()
        return {"calls": r[0], "usd": r[1], "credits": r[2]}

    def check_budget(self, source: str, scfg: dict, next_cost_usd: float = 0.0, next_credits: float = 0.0):
        u = self.usage(source)
        if scfg.get("daily_max_calls") is not None and u["calls"] + 1 > scfg["daily_max_calls"]:
            raise BudgetExceeded(f"{source}: daily call cap {scfg['daily_max_calls']} reached")
        if scfg.get("daily_budget_usd") is not None and u["usd"] + next_cost_usd > scfg["daily_budget_usd"] + 1e-12:
            raise BudgetExceeded(f"{source}: daily USD budget {scfg['daily_budget_usd']} reached ({u['usd']:.4f})")
        if scfg.get("daily_max_credits") is not None and u["credits"] + next_credits > scfg["daily_max_credits"]:
            raise BudgetExceeded(f"{source}: daily credit cap {scfg['daily_max_credits']} reached")
        if scfg.get("monthly_budget_usd") is not None or scfg.get("monthly_max_credits") is not None:
            m = self.usage(source, month=True)
            if scfg.get("monthly_budget_usd") is not None and m["usd"] + next_cost_usd > scfg["monthly_budget_usd"]:
                raise BudgetExceeded(f"{source}: monthly USD budget {scfg['monthly_budget_usd']} reached")
            if scfg.get("monthly_max_credits") is not None and m["credits"] + next_credits > scfg["monthly_max_credits"]:
                raise BudgetExceeded(f"{source}: monthly credit cap {scfg['monthly_max_credits']} reached")
        rem = self.get_meta(f"{source}.remaining_usd")
        if rem is not None and scfg.get("min_remaining_usd") is not None:
            v, ts = json.loads(rem)
            if utc_day(ts) == utc_day() and v < scfg["min_remaining_usd"]:
                raise BudgetExceeded(f"{source}: provider-reported remaining USD {v} < {scfg['min_remaining_usd']}")

    def book(self, source, kind, url, status, latency, cost_usd=0.0, credits=0.0, cached=False, attempt=0,
             target="", note=""):
        with self._db() as db:
            db.execute("INSERT INTO calls(ts,day,source,kind,url,status,latency_s,cost_usd,credits,cached,attempt,run,"
                       "target,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (time.time(), utc_day(), source, kind, strip_secret_params(url)[:500], status, latency, cost_usd,
                        credits, int(cached), attempt, self.run, target, redact(note)[:300]))

    def get_meta(self, k):
        with self._db() as db:
            r = db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def set_meta(self, k, v):
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, v))

    # ---------------------------------------------------------------- cooldown / pacing
    def cooling(self, source: str) -> float:
        with self._db() as db:
            r = db.execute("SELECT until FROM cooldown WHERE source=?", (source,)).fetchone()
        return max(0.0, (r[0] - time.time())) if r else 0.0

    def set_cooldown(self, source: str, seconds: float, reason: str):
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO cooldown VALUES(?,?,?)", (source, time.time() + seconds, reason[:200]))

    def wait_turn(self, source: str, min_interval: float):
        """Reserve the next slot for `source` (cross-process via sqlite) and sleep until it."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            r = db.execute("SELECT next_ok FROM pace WHERE source=?", (source,)).fetchone()
            now = time.time()
            slot = max(now, r[0] if r else 0.0)
            db.execute("INSERT OR REPLACE INTO pace VALUES(?,?)", (source, slot + min_interval))
            db.execute("COMMIT")
        if slot > now:
            time.sleep(slot - now)

    # ---------------------------------------------------------------- cache
    @staticmethod
    def cache_key(method: str, url: str, body: dict | None) -> str:
        u = strip_secret_params(url)
        b = json.dumps(body, sort_keys=True) if body is not None else ""
        return hashlib.sha256(f"{method} {u} {b}".encode()).hexdigest()

    def cache_get(self, k: str, ttl_days: float):
        with self._cdb() as db:
            r = db.execute("SELECT ts, status, body FROM cache WHERE k=?", (k,)).fetchone()
        if r and time.time() - r[0] < ttl_days * 86400:
            return r[1], gzip.decompress(r[2])
        return None

    def cache_put(self, k: str, source: str, url: str, status: int, body: bytes):
        with self._cdb() as db:
            db.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?,?,?,?)",
                       (k, time.time(), source, strip_secret_params(url)[:500], status, gzip.compress(body)))

    # ---------------------------------------------------------------- report
    def summary(self, run: str | None = None) -> list[dict]:
        q = ("SELECT source, kind, SUM(cached=0), SUM(cached=1), SUM(status=200 AND cached=0), SUM(status=429), "
             "ROUND(SUM(cost_usd),5), SUM(credits), ROUND(AVG(CASE WHEN cached=0 THEN latency_s END),3) FROM calls ")
        args = ()
        if run:
            q += "WHERE run=? "
            args = (run,)
        q += "GROUP BY source, kind ORDER BY source, kind"
        with self._db() as db:
            rows = db.execute(q, args).fetchall()
        keys = ["source", "kind", "live", "cached", "ok", "n429", "usd", "credits", "avg_latency_s"]
        return [dict(zip(keys, r)) for r in rows]


class _Closing:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, *a):
        self.db.close()


# -------------------------------------------------------------------- HTTP
@dataclass
class Response:
    status: int
    body: bytes
    headers: dict
    latency_s: float
    cached: bool

    def json(self):
        return json.loads(self.body)


_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_loopback(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback
_ENV = urllib.request.build_opener()


class Http:
    """One instance per run. `transport` can be replaced in tests (callable(method,url,headers,body)->(st,hdr,bytes))."""

    def __init__(self, cfg: dict, ledger: Ledger, transport=None, offline: bool = False):
        self.cfg = cfg
        self.ledger = ledger
        self.transport = transport or self._urllib
        self.offline = offline  # cache-only: never hit the network

    def _urllib(self, method, url, headers, body):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        # loopback services (local search pool) must bypass the system proxy; everything else follows it
        opener = _DIRECT if is_loopback(url) else _ENV
        try:
            with opener.open(req, timeout=self.cfg.get("timeout_s", 60)) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), (e.read() if e.fp else b"")
        except Exception as e:  # noqa: BLE001
            return 0, {}, redact(repr(e)).encode()

    def request(self, source: str, kind: str, url: str, *, method: str = "GET", headers: dict | None = None,
                body: dict | None = None, cost_usd: float = 0.0, credits: float = 0.0, target: str = "",
                cost_from_response=None, ok_statuses=(200,), validate=None) -> Response:
        """validate(body_bytes) -> None | str: called on ok-status responses before caching.  A returned string
        rejects the response: it is booked with its real cost and the reason, never cached, and raised as
        HTTPFailure (so a bad 200 cannot be served from cache later)."""
        scfg = self.cfg["sources"][source]
        key = Ledger.cache_key(method, url, body)
        hit = self.ledger.cache_get(key, self.cfg.get("cache_ttl_days", 30))
        if hit:
            st, b = hit
            self.ledger.book(source, kind, url, st, 0.0, cached=True, target=target)
            return Response(st, b, {}, 0.0, True)
        if self.offline:
            raise HTTPFailure(0, f"offline and not cached: {source}/{kind}")
        cool = self.ledger.cooling(source)
        if cool > 0:
            raise SourceCooling(f"{source} cooling down for {cool:.0f}s more")
        self.ledger.check_budget(source, scfg, cost_usd, credits)
        hdrs = {"User-Agent": self.cfg["user_agent"], "Accept": "application/json", **(headers or {})}
        if body is not None:
            hdrs["Content-Type"] = "application/json"
        retries = scfg.get("retries", 3)
        last: tuple[int, bytes] = (0, b"no attempt")
        for attempt in range(retries + 1):
            self.ledger.wait_turn(source, scfg.get("min_interval_s", 1.0))
            t0 = time.time()
            st, hd, b = self.transport(method, url, hdrs, body)
            dt = time.time() - t0
            c_usd, c_cr = (cost_from_response(st, hd, b) if cost_from_response else (cost_usd, credits)) \
                if st in ok_statuses else (0.0, 0.0)
            rejected = validate(b) if (validate and st in ok_statuses) else None
            note = ("rejected: " + rejected) if rejected else ("" if st in ok_statuses else b[:200].decode("utf8", "replace"))
            self.ledger.book(source, kind, url, -st if rejected else st, round(dt, 3), c_usd, c_cr, attempt=attempt,
                             target=target, note=note[:200])
            if rejected:
                raise HTTPFailure(st, redact(rejected[:300]))
            rem = {k.lower(): v for k, v in (hd or {}).items()}.get("x-ratelimit-remaining-usd")
            if rem is not None:
                try:
                    self.ledger.set_meta(f"{source}.remaining_usd", json.dumps([float(rem), time.time()]))
                except ValueError:
                    pass
            if st in ok_statuses or st == 404:
                if st in ok_statuses:
                    self.ledger.cache_put(key, source, url, st, b)
                return Response(st, b, hd, dt, False)
            last = (st, b)
            if st == 429:
                ra = {k.lower(): v for k, v in (hd or {}).items()}.get("retry-after")
                base = scfg.get("backoff_429_s", 4.0)
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else base * 2 ** attempt
                if attempt >= retries or wait > 300:   # 429 persists after all retries -> persistent cooldown
                    self.ledger.set_cooldown(source, scfg.get("cooldown_429_s", 3600), "429")
                    raise SourceCooling(f"{source}: repeated 429, cooling {scfg.get('cooldown_429_s')}s")
                time.sleep(min(wait, 120) + random.random())
                continue
            if st in (0, 500, 502, 503, 504) and attempt < retries:
                time.sleep(min(60, 2 ** (attempt + 1)) + random.random())
                continue
            break
        st, b = last
        raise HTTPFailure(st, redact(b[:300].decode("utf8", "replace")))
