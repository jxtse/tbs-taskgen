"""SQLite budget ledger for model calls: per model / stage / item tokens, USD, call counts; daily budgets;
circuit breaker.  Shared across processes (WAL + BEGIN IMMEDIATE).

Rules
- A *real* model call must `reserve()` first.  Reservation fails (BudgetStop) when
  * no daily budget is configured (budget.daily_usd_total is None) -> "budget_unset";
  * spent + reserved + est_usd_per_call would exceed the total / per-model / per-stage daily USD budget;
  * the per-model daily call cap or the per-item call cap is reached;
  * the model's breaker is open.
- Fake models also go through the ledger (usd=0) so tests exercise the same path, but are exempt from
  the "budget_unset" rule.
- `finish()` records usage+cost; error classes in budget.breaker_open_on (quota/auth) open the breaker at
  once; N consecutive errors of any class open it too.  The breaker stays open until `reset_breaker()`
  (manual, after a human has looked) — this is what would have stopped the 2026-11-02 teacher job after
  the first "额度已用完" 429 instead of 100 failed trials.
- Day boundary: Asia/Shanghai (UTC+8) calendar day.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import time
from pathlib import Path

CN = dt.timezone(dt.timedelta(hours=8))


class BudgetStop(RuntimeError):
    """Raised instead of making a call; carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def today(now: float | None = None) -> str:
    return dt.datetime.fromtimestamp(now or time.time(), CN).strftime("%Y-%m-%d")


SCHEMA = """
CREATE TABLE IF NOT EXISTS calls(
  id INTEGER PRIMARY KEY, day TEXT, started REAL, ended REAL, model TEXT, real INTEGER, stage TEXT, item TEXT,
  status TEXT, est_usd REAL, usd REAL, in_tok INTEGER, out_tok INTEGER, cache_read INTEGER, cache_write INTEGER,
  error_class TEXT, error TEXT, latency REAL, resp_model TEXT, prompt_sha TEXT);
CREATE INDEX IF NOT EXISTS calls_day ON calls(day, model);
CREATE TABLE IF NOT EXISTS breaker(model TEXT PRIMARY KEY, open INTEGER, reason TEXT, ts REAL, consecutive INTEGER);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, data TEXT);
"""


def price_usd(prices: dict, model: str, usage: dict) -> float:
    p = prices.get(model) or prices.get(model.split(":")[0]) or {}
    if not p:
        return 0.0
    i = usage.get("input_tokens", 0) or 0
    o = usage.get("output_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    cw = usage.get("cache_creation_input_tokens", 0) or 0
    return (i * p.get("in", 0) + o * p.get("out", 0) + cr * p.get("cache_read", p.get("in", 0) * 0.1)
            + cw * p.get("cache_write", p.get("in", 0) * 1.25)) / 1e6


class Ledger:
    def __init__(self, path: str | Path, cfg: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self):
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # ------------------------------------------------------------------ reserve / finish
    def reserve(self, model: str, stage: str, item: str, real: bool, prompt_sha: str = "",
                est_usd: float | None = None) -> int:
        b = self.cfg["budget"]
        est = (float(b.get("est_usd_per_call", 0.0)) if est_usd is None else float(est_usd)) if real else 0.0
        day = today()
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT open, reason FROM breaker WHERE model=?", (model,)).fetchone()
            if row and row[0]:
                raise BudgetStop("breaker_open", f"{model}: {row[1]}")
            if real:
                if b.get("daily_usd_total") is None:
                    raise BudgetStop("budget_unset", "set budget.daily_usd_total explicitly before real model calls")

                def spent(where: str, *args) -> float:
                    q = f"SELECT COALESCE(SUM(COALESCE(usd, est_usd)),0) FROM calls WHERE day=? AND real=1 {where}"
                    return db.execute(q, (day, *args)).fetchone()[0]

                if spent("") + est > float(b["daily_usd_total"]) + 1e-12:
                    raise BudgetStop("daily_total", f"{b['daily_usd_total']} USD")
                if b.get("run_usd_total") is not None:     # whole-run cap across days (daily caps reset at 0:00)
                    allrun = db.execute("SELECT COALESCE(SUM(COALESCE(usd, est_usd)),0) FROM calls WHERE real=1"
                                        ).fetchone()[0]
                    if allrun + est > float(b["run_usd_total"]) + 1e-12:
                        raise BudgetStop("run_total", f"{b['run_usd_total']} USD")
                cap_m = (b.get("daily_usd_per_model") or {}).get(model)
                if cap_m is not None and spent("AND model=?", model) + est > float(cap_m) + 1e-12:
                    raise BudgetStop("daily_model", f"{model} {cap_m} USD")
                cap_s = (b.get("daily_usd_per_stage") or {}).get(stage)
                if cap_s is not None and spent("AND stage=?", stage) + est > float(cap_s) + 1e-12:
                    raise BudgetStop("daily_stage", f"{stage} {cap_s} USD")
            cap_r = (b.get("run_calls_per_model") or {}).get(model)
            if cap_r is not None:
                n = db.execute("SELECT count(*) FROM calls WHERE model=? AND real=1", (model,)).fetchone()[0]
                if n >= int(cap_r):
                    raise BudgetStop("run_calls", f"{model} {cap_r} calls")
            cap_c = (b.get("daily_calls_per_model") or {}).get(model)
            if cap_c is not None:
                n = db.execute("SELECT count(*) FROM calls WHERE day=? AND model=?", (day, model)).fetchone()[0]
                if n >= int(cap_c):
                    raise BudgetStop("daily_calls", f"{model} {cap_c} calls")
            cap_i = b.get("max_calls_per_item")
            if cap_i is not None and item:
                n = db.execute("SELECT count(*) FROM calls WHERE item=?", (item,)).fetchone()[0]
                if n >= int(cap_i):
                    raise BudgetStop("item_calls", f"{item} {cap_i} calls")
            cur = db.execute(
                "INSERT INTO calls(day,started,model,real,stage,item,status,est_usd,prompt_sha) VALUES(?,?,?,?,?,?,?,?,?)",
                (day, time.time(), model, int(real), stage, item, "reserved", est, prompt_sha))
            db.execute("COMMIT")
            return cur.lastrowid
        except BaseException:
            db.execute("ROLLBACK") if db.in_transaction else None
            raise
        finally:
            db.close()

    def finish(self, call_id: int, usage: dict | None = None, error_class: str | None = None,
               error: str | None = None, latency: float | None = None, resp_model: str | None = None) -> float:
        usage = usage or {}
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            model, real = db.execute("SELECT model, real FROM calls WHERE id=?", (call_id,)).fetchone()
            usd = price_usd(self.cfg["models"].get("prices", {}), model, usage) if real else 0.0
            db.execute("UPDATE calls SET ended=?,status=?,usd=?,in_tok=?,out_tok=?,cache_read=?,cache_write=?,"
                       "error_class=?,error=?,latency=?,resp_model=? WHERE id=?",
                       (time.time(), "error" if error_class else "ok", usd, usage.get("input_tokens", 0),
                        usage.get("output_tokens", 0), usage.get("cache_read_input_tokens", 0),
                        usage.get("cache_creation_input_tokens", 0), error_class, (error or "")[:500], latency,
                        resp_model, call_id))
            row = db.execute("SELECT consecutive FROM breaker WHERE model=?", (model,)).fetchone()
            consec = (row[0] if row else 0) + 1 if error_class else 0
            b = self.cfg["budget"]
            opened = None
            if error_class and (error_class in b.get("breaker_open_on", [])
                                or consec >= int(b.get("breaker_consecutive_errors", 4))):
                opened = f"{error_class} x{consec}: {(error or '')[:200]}"
            if opened:
                db.execute("INSERT OR REPLACE INTO breaker VALUES(?,?,?,?,?)", (model, 1, opened, time.time(), consec))
                db.execute("INSERT INTO events(ts,kind,data) VALUES(?,?,?)",
                           (time.time(), "breaker_open", json.dumps({"model": model, "reason": opened})))
            else:
                db.execute("INSERT INTO breaker(model,open,reason,ts,consecutive) VALUES(?,0,'',?,?) "
                           "ON CONFLICT(model) DO UPDATE SET consecutive=excluded.consecutive",
                           (model, time.time(), consec))
            db.execute("COMMIT")
            return usd
        finally:
            db.close()

    def settle(self, call_id: int, usd: float | None, status: str = "ok", note: str = "") -> None:
        """Close a non-token reservation (e.g. an S9 Harbor run) with its recorded cost; usd=None keeps the
        reserved estimate as the cost (unknown actual cost is never booked as 0)."""
        db = self._db()
        try:
            db.execute("UPDATE calls SET ended=?, status=?, usd=COALESCE(?, est_usd), error=? WHERE id=?",
                       (time.time(), status, usd, note[:500], call_id))
            db.commit()
        finally:
            db.close()

    # ------------------------------------------------------------------ breaker / reports
    def breaker(self, model: str) -> dict | None:
        with self._db() as db:
            r = db.execute("SELECT open, reason, ts, consecutive FROM breaker WHERE model=?", (model,)).fetchone()
        return dict(open=bool(r[0]), reason=r[1], ts=r[2], consecutive=r[3]) if r else None

    def open_breaker(self, model: str, reason: str):
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO breaker VALUES(?,?,?,?,?)", (model, 1, reason, time.time(), 0))

    def reset_breaker(self, model: str):
        with self._db() as db:
            db.execute("UPDATE breaker SET open=0, reason='', consecutive=0 WHERE model=?", (model,))
            db.execute("INSERT INTO events(ts,kind,data) VALUES(?,?,?)", (time.time(), "breaker_reset", model))

    def summary(self, day: str | None = None) -> dict:
        q = ("SELECT model, stage, real, count(*), SUM(status='error'), SUM(in_tok), SUM(out_tok), SUM(cache_read), "
             "SUM(COALESCE(usd,est_usd,0)) FROM calls {} GROUP BY model, stage, real ORDER BY model, stage")  # in-flight = reserved est
        with self._db() as db:
            rows = db.execute(q.format("WHERE day=?" if day else ""), (day,) if day else ()).fetchall()
            br = db.execute("SELECT model, open, reason FROM breaker WHERE open=1").fetchall()
        out = [dict(model=r[0], stage=r[1], real=bool(r[2]), calls=r[3], errors=r[4] or 0, in_tok=r[5] or 0,
                    out_tok=r[6] or 0, cache_read=r[7] or 0, usd=round(r[8] or 0, 4)) for r in rows]
        return dict(day=day, rows=out, usd_total=round(sum(r["usd"] for r in out), 4),
                    calls_total=sum(r["calls"] for r in out),
                    breakers_open=[dict(model=m, reason=rs) for m, o, rs in br])
