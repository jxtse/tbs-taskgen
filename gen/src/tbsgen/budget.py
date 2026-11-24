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
  (manual, after a human has looked), so a quota error stops the run at once instead of after many failed
  calls.
- Day boundary: Asia/Shanghai (UTC+8) calendar day.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import json
import sqlite3
import time
from pathlib import Path

CN = dt.timezone(dt.timedelta(hours=8))
EXCLUDE_UNSTARTED_JOBS = "COALESCE(cost_kind,'') NOT IN ('probe_migration_cancelled','cancelled_unstarted','expired_unsent')"
# hardening round of the item being stepped (Pipeline.step sets it while state.harden_active; 0 = not a hardening
# round).  Every reservation is tagged with it (calls.harden), so budget.item_harden_usd_cap can keep a hardening
# round's authoring spend (S5G, S6A, S6M, S6C, S7, S8 ...) apart from the first version's budget.item_usd_cap.
# the hardening round being stepped: an int, or a callable evaluated at reservation time (pipeline.step passes one
# that reads the item's current state, so a handler that re-loads the item under its lock tags with the fresh round)
HARDEN_ROUND: contextvars.ContextVar = contextvars.ContextVar("tbsgen_harden_round", default=0)


def harden_round_now() -> int:
    v = HARDEN_ROUND.get()
    return int((v() if callable(v) else v) or 0)


@contextlib.contextmanager
def harden_scope(k):
    tok = HARDEN_ROUND.set(k if callable(k) else int(k or 0))
    try:
        yield
    finally:
        HARDEN_ROUND.reset(tok)


def _harden_stages(b: dict) -> list[str]:
    """Stages charged to budget.item_harden_usd_cap in a hardening round: the authoring stages plus S7."""
    st = list(b.get("item_usd_cap_stages") or [])
    return st + [s for s in ("S7",) if s not in st]


# S9b variant round of the item being stepped (calib.variant_depth; 0 = not a variant round).  Tagged apart from
# hardening rounds (calls.variant instead of calls.harden) so budget.item_variant_usd_cap is a separate per-item
# allowance: an item whose hardening budget is spent can still get variant rounds.
VARIANT_ROUND: contextvars.ContextVar = contextvars.ContextVar("tbsgen_variant_round", default=0)


def variant_round_now() -> int:
    v = VARIANT_ROUND.get()
    return int((v() if callable(v) else v) or 0)


@contextlib.contextmanager
def variant_scope(k):
    tok = VARIANT_ROUND.set(k if callable(k) else int(k or 0))
    try:
        yield
    finally:
        VARIANT_ROUND.reset(tok)


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
  error_class TEXT, error TEXT, latency REAL, resp_model TEXT, prompt_sha TEXT, harden INTEGER, variant INTEGER);
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
            db.execute("BEGIN IMMEDIATE")
            cols = {r[1] for r in db.execute("PRAGMA table_info(calls)")}
            if "cost_kind" not in cols:     # added for S9 runs: reported / tokens / reservation (older ledgers: NULL)
                db.execute("ALTER TABLE calls ADD COLUMN cost_kind TEXT")
            if "harden" not in cols:        # hardening round of the call (older ledgers: NULL = first version)
                db.execute("ALTER TABLE calls ADD COLUMN harden INTEGER")
            if "variant" not in cols:       # S9b variant round of the call (older ledgers: NULL = not a variant)
                db.execute("ALTER TABLE calls ADD COLUMN variant INTEGER")
            if "reservation_key" not in cols:
                db.execute("ALTER TABLE calls ADD COLUMN reservation_key TEXT")
            # A cross-day replacement shares the key of the expired reservation.
            db.execute("DROP INDEX IF EXISTS calls_reservation")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS calls_open_reservation ON calls(reservation_key) "
                       "WHERE reservation_key IS NOT NULL AND status='reserved'")

    def _db(self):
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # ------------------------------------------------------------------ reserve / finish
    def reserved_call(self, key: str) -> int | None:
        with self._db() as db:
            return self._reserved_call(db, key)

    @staticmethod
    def _reserved_call(db, key):
        row = db.execute("SELECT id FROM calls WHERE reservation_key=? AND status='reserved'", (key,)).fetchone()
        return row[0] if row else None

    def cancel_unstarted_job(self, item, stage, key, saved_id, kind):
        """Cancel all open calls for this launch, including a replacement missing from the job record."""
        with contextlib.closing(self._db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id FROM calls WHERE item=? AND stage=? AND status IN ('reserved','sent') "
                              "AND (reservation_key=? OR id=?) ORDER BY id", (item, stage, key, saved_id)).fetchall()
            for (call,) in rows:
                self._settle(db, call, 0.0, "error", "", kind, None)
            return rows[-1][0] if rows else saved_id

    def adopt_started_job(self, model, stage, item, job_name, launch_key, estimate, cost):
        """Track a job that already started. Admission limits apply only before launch."""
        key = f"harbor-adopt:{job_name}"
        with contextlib.closing(self._db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM calls WHERE item=? AND stage=? AND "
                             "(reservation_key=? OR (reservation_key=? AND status IN ('reserved','sent','ok'))) "
                             "ORDER BY id DESC LIMIT 1", (item, stage, key, launch_key)).fetchone()
            if row:
                call = row[0]
            else:
                call = db.execute(
                    "INSERT INTO calls(day,started,model,real,stage,item,status,est_usd,reservation_key) "
                    "VALUES(?,?,?,1,?,?,'reserved',?,?)",
                    (today(), time.time(), model, stage, item, estimate, key)).lastrowid
            if cost is not None:
                usd, kind, usage = cost
                self._settle(db, call, usd, "ok", "adopted completed Harbor job", kind, usage)
            return call

    def adopt_legacy_agent_call(self, key, item, stage, model, intent_ts):
        """Attach a job's stable key to its outstanding pre-key reservation, if present."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            held = self._reserved_call(db, key)
            if held:
                return held  # upgrade interrupted before run.json was saved
            rows = db.execute("SELECT id FROM calls WHERE item=? AND stage=? AND model=? AND started>=? "
                              "AND status='reserved' AND real=1 AND prompt_sha='' AND reservation_key IS NULL",
                              (item, stage, model, intent_ts)).fetchall()
            if len(rows) > 1:
                raise BudgetStop("ambiguous_reservation", f"{item} {stage}: multiple outstanding legacy calls")
            if rows:
                call = rows[0][0]
                db.execute("UPDATE calls SET reservation_key=? WHERE id=?", (key, call))
                return call
            return None

    def reserve(self, model: str, stage: str, item: str, real: bool, prompt_sha: str = "",
                est_usd: float | None = None, reservation_key: str | None = None) -> int:
        b = self.cfg["budget"]
        est = (float(b.get("est_usd_per_call", 0.0)) if est_usd is None else float(est_usd)) if real else 0.0
        hk = harden_round_now()
        vk = variant_round_now()
        day = today()
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            if reservation_key is not None:
                held = self._reserved_call(db, reservation_key)
                if held:
                    db.execute("COMMIT")
                    return held
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
            # call-count caps count calls that were made: a reservation the model relay refused at its preflight
            # (cost_kind relay_refused, calib.relay_preflight_refused: no Harbor job, no model call) is no call
            made = f"AND COALESCE(cost_kind,'') NOT IN ('relay_refused', 'diversity_parked') AND {EXCLUDE_UNSTARTED_JOBS}"
            if cap_r is not None:
                n = db.execute(f"SELECT count(*) FROM calls WHERE model=? AND real=1 {made}", (model,)).fetchone()[0]
                if n >= int(cap_r):
                    raise BudgetStop("run_calls", f"{model} {cap_r} calls")
            cap_c = (b.get("daily_calls_per_model") or {}).get(model)
            if cap_c is not None:
                n = db.execute(f"SELECT count(*) FROM calls WHERE day=? AND model=? {made}", (day, model)).fetchone()[0]
                if n >= int(cap_c):
                    raise BudgetStop("daily_calls", f"{model} {cap_c} calls")
            cap_u, cap_h = b.get("item_usd_cap"), b.get("item_harden_usd_cap")
            cap_v = b.get("item_variant_usd_cap")
            if item and hk and cap_h is not None:
                # a hardening round with its own allowance: all its authoring spend, never the first version's
                st = _harden_stages(b)
                if stage in st:
                    used = _item_used(db, item, st, "AND COALESCE(harden,0)>0")
                    if used + est > float(cap_h) + 1e-9:
                        raise BudgetStop("item_harden_usd", f"{item} {used:.2f}+{est:.2f} > {cap_h} USD (hardening)")
            elif item and vk and cap_v is not None:
                # an S9b variant round with its own allowance: separate from the hardening rounds' spend and from
                # the first version's, so an item that used up item_harden_usd_cap still gets variant rounds
                st = _harden_stages(b)
                if stage in st:
                    used = _item_used(db, item, st, "AND COALESCE(variant,0)>0")
                    if used + est > float(cap_v) + 1e-9:
                        raise BudgetStop("item_variant_usd",
                                         f"{item} {used:.2f}+{est:.2f} > {cap_v} USD (S9b variant)")
            elif cap_u is not None and item and stage in (b.get("item_usd_cap_stages") or []):
                st = list(b.get("item_usd_cap_stages") or [])
                # without item_harden_usd_cap / item_variant_usd_cap the rounds share item_usd_cap (the original
                # rule); a set cap keeps that round's spend out of the first version's bucket
                extra = ("AND COALESCE(harden,0)=0" if cap_h is not None else "") + \
                        (" AND COALESCE(variant,0)=0" if cap_v is not None else "")
                used = _item_used(db, item, st, extra)
                if used + est > float(cap_u) + 1e-9:
                    raise BudgetStop("item_usd", f"{item} {used:.2f}+{est:.2f} > {cap_u} USD")
            cap_i = b.get("max_calls_per_item")
            # authoring calls only: S9 runs have their own caps; S2/S4 screening calls (Jev, ~$0.0001) do not
            # use up an item's authoring budget
            if cap_i is not None and item and stage not in ("S9", "S2", "S4", "S3b", "S4c", "S5G"):
                n = db.execute("SELECT count(*) FROM calls WHERE item=? AND stage NOT IN "
                               f"('S9','S2','S4','S3b','S4c','S5G') {made}",
                               (item,)).fetchone()[0]
                if n >= int(cap_i):
                    raise BudgetStop("item_calls", f"{item} {cap_i} calls")
            cur = db.execute(
                "INSERT INTO calls(day,started,model,real,stage,item,status,est_usd,prompt_sha,harden,variant,reservation_key) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (day, time.time(), model, int(real), stage, item, "reserved", est, prompt_sha, hk or None,
                 vk or None, reservation_key))
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

    def settle(self, call_id: int, usd: float | None, status: str = "ok", note: str = "",
               kind: str | None = None, usage: dict | None = None) -> None:
        """Close a non-token reservation (e.g. an S9 Harbor run) with its recorded cost; usd=None keeps the
        reserved estimate as the cost (unknown actual cost is never booked as 0) and marks the row
        cost_kind="reservation".  kind: "reported" (default for a given usd) or "tokens" (estimated from token
        usage); usage (if known) is stored in the token columns."""
        with contextlib.closing(self._db()) as db, db:
            self._settle(db, call_id, usd, status, note, kind, usage)

    @staticmethod
    def _settle(db, call_id, usd, status, note, kind, usage):
        kind = "reservation" if usd is None else (kind or "reported")
        u = usage or {}
        db.execute("UPDATE calls SET ended=?, status=?, usd=COALESCE(?, est_usd), error=?, cost_kind=?, "
                   "in_tok=COALESCE(?, in_tok), out_tok=COALESCE(?, out_tok), cache_read=COALESCE(?, cache_read), "
                   "cache_write=COALESCE(?, cache_write) WHERE id=?",
                   (time.time(), status, usd, note[:500], kind, u.get("input_tokens"), u.get("output_tokens"),
                    u.get("cache_read_input_tokens"), u.get("cache_creation_input_tokens"), call_id))

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
             f"SUM(COALESCE(usd,est_usd,0)) FROM calls WHERE {EXCLUDE_UNSTARTED_JOBS} {{}} GROUP BY model, stage, real "
             "ORDER BY model, stage")
        with self._db() as db:
            rows = db.execute(q.format("AND day=?" if day else ""), (day,) if day else ()).fetchall()
            br = db.execute("SELECT model, open, reason FROM breaker WHERE open=1").fetchall()
        out = [dict(model=r[0], stage=r[1], real=bool(r[2]), calls=r[3], errors=r[4] or 0, in_tok=r[5] or 0,
                    out_tok=r[6] or 0, cache_read=r[7] or 0, usd=round(r[8] or 0, 4)) for r in rows]
        return dict(day=day, rows=out, usd_total=round(sum(r["usd"] for r in out), 4),
                    calls_total=sum(r["calls"] for r in out),
                    breakers_open=[dict(model=m, reason=rs) for m, o, rs in br],
                    usd_by_cost_kind=self.cost_kinds(day))

    def cost_kinds(self, day: str | None = None) -> dict:
        """USD split by how the cost is known: "token_priced" (model calls: tokens x prices), "reported" (S9: Harbor
        cost_usd), "tokens" (S9: estimated from the trial's token usage), "reservation" (S9: unknown, the reserved
        estimate stands in; an upper bound, not a bill), "in_flight" (still reserved)."""
        q = ("SELECT CASE WHEN status='reserved' THEN 'in_flight' ELSE COALESCE(cost_kind, 'token_priced') END, "
             "SUM(COALESCE(usd, est_usd, 0)) FROM calls WHERE real=1 {} GROUP BY 1")
        with self._db() as db:
            rows = db.execute(q.format("AND day=?" if day else ""), (day,) if day else ()).fetchall()
        return {k: round(v or 0, 4) for k, v in rows}


def _item_used(db, item: str, stages: list[str], extra: str = "") -> float:
    if not stages:
        return 0.0
    q = (f"SELECT COALESCE(SUM(COALESCE(usd, est_usd)),0) FROM calls WHERE item=? AND real=1 AND "
         f"stage IN ({','.join('?' * len(stages))}) {extra}")
    return float(db.execute(q, (item, *stages)).fetchone()[0])


def item_usd_left(ledger, cfg: dict, item: str, harden: int | None = None) -> float | None:
    """What is left of the item's authoring allowance (settled + reserved spend; None = no cap).  harden: the
    hardening round (None = the round being stepped, budget.HARDEN_ROUND).  In a hardening round with
    budget.item_harden_usd_cap set: that cap minus the spend of all hardening rounds; in an S9b variant round with
    budget.item_variant_usd_cap set: that cap minus the spend of all variant rounds; otherwise budget.item_usd_cap
    minus the first version's spend (minus everything when the round caps are unset: the original rule)."""
    b = cfg["budget"]
    h = harden_round_now() if harden is None else int(harden)
    v = variant_round_now() if harden is None else 0
    cap_u, cap_h = b.get("item_usd_cap"), b.get("item_harden_usd_cap")
    cap_v = b.get("item_variant_usd_cap")
    with ledger._db() as db:
        if h and cap_h is not None:
            return float(cap_h) - _item_used(db, item, _harden_stages(b), "AND COALESCE(harden,0)>0")
        if v and cap_v is not None:
            return float(cap_v) - _item_used(db, item, _harden_stages(b), "AND COALESCE(variant,0)>0")
        if cap_u is None:
            return None
        extra = ("AND COALESCE(harden,0)=0" if cap_h is not None else "") + \
                (" AND COALESCE(variant,0)=0" if cap_v is not None else "")
        return float(cap_u) - _item_used(db, item, list(b.get("item_usd_cap_stages") or []), extra)
