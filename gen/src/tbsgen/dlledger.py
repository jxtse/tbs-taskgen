"""Download ledger for S3 full-text fetching: daily / hourly caps, per-host pacing, cooldowns, breakers.

Every outbound full-text download goes through `DownloadLedger.acquire(channel, host)`, which, inside one
`BEGIN IMMEDIATE` transaction,
  1. refuses hosts in cooldown and channels whose breaker is open,
  2. enforces caps (successes and attempts per channel per day and per hour, a global daily success cap,
     per-host daily / hourly attempt caps),
  3. books a `pending` attempt and the next free time slot for that host (min gap + random jitter),
then sleeps until the slot and, in a second transaction, re-checks cooldowns and breakers (they may have been
set while it slept; a refused reservation is finalised as "fail" with note "cancelled").  `record(...,
reservation=id)` finalises the row.  Two processes sharing the file cannot both take the last slot of a cap or the
same host slot.  Caps are counted over the day of the booked slot (a slot after midnight counts for the next day),
and a reservation is never booked more than `max_wait_s` ahead.  A pending row keeps its quota until it is
finalised, or until it is older than PENDING_TTL_S (a crashed run).  That is safe because every download is
bounded well below it: HTTP requests have a total deadline per hop (fulltext.timeout) and at most 5 redirect hops
with at most `hop_max_wait_s` pacing each; a browser run is killed after `browser.script_timeout_s`.
Status values: ok / fail / pending (note "booked" while waiting for its slot) / hop (a redirect to another host).

Schema: `attempts` (one row per download attempt), `cooldown` (host -> until), `hosts` (next free slot per
host), `blocks` (block signals, for the breaker), `meta_calls` (metadata lookups such as Unpaywall / Europe PMC
search, which never count against download caps and are paced under the host key "meta:<host>") and
`not_entitled` ((channel, DOI) pairs a TDM API answered with metadata / abstract only: the subscription does not
cover the paper, so that channel never asks for it again).  Browser downloads are keyed "chrome:<DOI prefix>";
callers may pass `related` keys (e.g. the publisher's web host, or all DOI prefixes of one publisher) whose
cooldowns also apply.  The day is the local calendar day; set TZ for the process if that matters.

Block signals: 429 -> 6 h host cooldown; a challenge page (Cloudflare, captcha, Akamai "Access Denied")
-> 24 h.  A plain paywall 403 is not a block.  Breakers stop a whole channel: `breaker_blocks_per_day`
block signals on one channel in a day, or `breaker_consecutive_fail` failures in a row (e.g. a revoked key;
cancelled reservations and not-entitled answers do not count: the key worked).  A channel whose limits set
`count_blocks_in_breaker = false` (smoke / test batches) still cools the host on a block signal, but the block is
booked as "<channel>:uncounted" and does not count towards the breaker.
Every LimitReached carries `until`: the earliest time the refusal can lift (cooldown end, the hour window
moving on, local midnight for daily caps and breakers) or None if unknown; the queue keeps such papers due
instead of backing them off.  Nothing here reads secrets.
"""
from __future__ import annotations

import random
import re
import sqlite3
import threading
import time
from pathlib import Path

COOLDOWN_S = 24 * 3600
COOLDOWN_429_S = 6 * 3600
PENDING_TTL_S = 3 * 3600      # > browser script timeout + max wait; older pending rows belong to crashed runs
MAX_WAIT_S = 1800             # never book a slot further ahead than this
# note prefix of a reservation the ledger cancelled before anything was sent (cooldown / breaker set while it
# waited, no paced slot): it counts against no cap and no breaker.  Other "cancelled: ..." rows (a remote worker
# that died, a stopped browser batch, a refused redirect target) may have sent a request and still count as attempts.
NOT_SENT = "cancelled (not sent): "
_SENT = "note not like 'cancelled (not sent)%'"

# Defaults are deliberately small fractions of the providers' published limits (see README, S3).
DEFAULT_LIMITS: dict[str, dict] = {
    #             ok/day  attempts/day  ok/hour  gap s  jitter s  per-host attempts/day  per-host attempts/hour
    # OA links point at many unrelated repositories: dead links and single-host blocks are normal, so the OA
    # breaker only trips on many block signals and never on a run of failures.
    "oa":       dict(daily_ok=300, daily_attempts=500, hourly_ok=60, gap_s=4.0, jitter_s=3.0,
                     host_daily_attempts=60, host_hourly_attempts=30,
                     breaker_blocks_per_day=10, breaker_consecutive_fail=0),
    "elsevier": dict(daily_ok=200, daily_attempts=300, hourly_ok=60, gap_s=3.0, jitter_s=2.0,
                     host_daily_attempts=300, host_hourly_attempts=80),
    "wiley":    dict(daily_ok=150, daily_attempts=200, hourly_ok=40, gap_s=10.0, jitter_s=4.0,   # Wiley: 10 s
                     host_daily_attempts=200, host_hourly_attempts=50),
    # campus-IP subscription access and the logged-in browser: far below the API channels (whole-campus IP
    # blocks are the failure mode, not a per-key quota)
    "inst":     dict(daily_ok=40, daily_attempts=60, hourly_ok=10, gap_s=45.0, jitter_s=45.0,
                     host_daily_attempts=30, host_hourly_attempts=8),
    "browser":  dict(daily_ok=40, daily_attempts=60, hourly_ok=12, gap_s=60.0, jitter_s=60.0,
                     host_daily_attempts=12, host_hourly_attempts=4),
}
META_LIMITS = dict(daily_calls=2000, hourly_calls=400, gap_s=1.0, jitter_s=0.5)   # per metadata host
DEFAULT_GLOBAL = dict(daily_ok=500, per_host_daily_attempts=60, breaker_blocks_per_day=3,
                      breaker_consecutive_fail=8)
# hosts that serve many papers through one API / mirror get their own per-host daily attempt cap
DEFAULT_HOST_CAPS = {"api.elsevier.com": 300, "api.wiley.com": 200, "www.ebi.ac.uk": 300, "arxiv.org": 200,
                     "export.arxiv.org": 200}

_CHALLENGE_TITLE = re.compile(
    rb"<title[^>]*>\s*(just a moment|attention required|access denied|are you a robot|verify you are human|"
    rb"please verify|security check|human verification|captcha)", re.I)
_CHALLENGE_MARKERS = (b"cf-chl", b"cf_chl_opt", b"_cf_chl", b"challenge-platform", b"px-captcha",
                      b"/cdn-cgi/challenge", b"perfdrive.com", b"radware")
CHALLENGE_TITLE_TEXT = re.compile(r"just a moment|verify you are human|attention required|access denied|"
                                  r"captcha|are you a robot|security check|请稍候", re.I)


class LimitReached(RuntimeError):
    """A cap, cooldown or breaker refuses the request.  `kind` is cap | cooldown | breaker; `scope` is "host"
    (only this host / publisher is refused: try other sources) or "channel" (stop the channel).  `until`: the
    earliest time the refusal can lift (None: unknown).  Nothing was sent: callers must not count it as a failed
    attempt."""

    def __init__(self, msg: str, kind: str = "cap", scope: str = "channel", until: float | None = None):
        super().__init__(msg)
        self.kind = kind
        self.scope = scope
        self.until = until


def challenge_page(body: bytes) -> bool:
    """A bot-wall page (served with any status).  Title / challenge-script markers only: ordinary article
    pages often embed a reCAPTCHA widget."""
    head = (body or b"")[:20000]
    return bool(_CHALLENGE_TITLE.search(head)) or any(m in head.lower() for m in _CHALLENGE_MARKERS)


def block_signal(status: int | None, body: bytes = b"") -> float | None:
    """Cooldown seconds when a response is a real block, else None.  429 -> 6 h.  A challenge page (title or
    challenge-script markers; a reCAPTCHA widget on a normal article page is not one) or an Akamai-style
    "Access Denied" 403 -> 24 h.  A plain paywall 403 is not a block."""
    if status == 429:
        return COOLDOWN_429_S
    if challenge_page(body):
        return COOLDOWN_S
    if status == 403 and re.search(rb"<title[^>]*>\s*access denied", (body or b"")[:20000], re.I):
        return COOLDOWN_S
    return None


def local_day(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts if ts is not None else time.time()))


def next_midnight(ts: float) -> float:
    """Start of the local day after `ts`."""
    lt = time.localtime(ts)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + 1, 0, 0, 0, 0, 0, -1))


NOT_ENTITLED_NOTE = "not_entitled"


class DownloadLedger:
    def __init__(self, path: str | Path, limits: dict | None = None, global_limits: dict | None = None,
                 host_caps: dict | None = None, clock=time.time, sleep=time.sleep):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = {k: dict(v) for k, v in DEFAULT_LIMITS.items()}
        for ch, v in (limits or {}).items():
            self.limits[ch] = {**self.limits.get(ch, DEFAULT_LIMITS["oa"]), **v}
        self.g = {**DEFAULT_GLOBAL, **(global_limits or {})}
        self.host_caps = {**DEFAULT_HOST_CAPS, **(host_caps or {})}
        self.meta = dict(META_LIMITS)
        self._exclude: int | None = None
        self.clock, self.sleep = clock, sleep
        self._local = threading.local()      # one sqlite connection per thread (browser workers run in threads)
        self.db.executescript("""
            create table if not exists attempts(ts real, day text, channel text, host text, doi text,
                                                status text, http int, bytes int, note text);
            create table if not exists cooldown(host text primary key, until real, reason text);
            create table if not exists hosts(host text primary key, next_ts real);
            create index if not exists a_day on attempts(day, channel);
            create table if not exists blocks(ts real, day text, channel text, host text, reason text);
            create table if not exists meta_calls(ts real, day text, host text, status text, http int);
            create table if not exists not_entitled(channel text, doi text, ts real, note text,
                                                    primary key(channel, doi));
        """)

    @property
    def db(self) -> sqlite3.Connection:
        c = getattr(self._local, "db", None)
        if c is None:
            c = self._local.db = sqlite3.connect(self.path, timeout=120, isolation_level=None)
        return c

    # ------------------------------------------------------------------ helpers
    def _tx(self):
        db = self.db

        class _T:
            def __enter__(s):
                db.execute("begin immediate")
                return db

            def __exit__(s, exc_type, *_):
                db.execute("rollback" if exc_type else "commit")
                return False
        return _T()

    def today(self) -> str:
        return local_day(self.clock())

    def _n(self, where: str, args: tuple = ()) -> int:
        if self._exclude is not None:                       # the reservation being re-checked
            where, args = f"({where}) and rowid != ?", (*args, self._exclude)
        return self.db.execute(f"select count(*) from attempts where {where}", args).fetchone()[0]

    @staticmethod
    def _in(keys: tuple) -> str:
        return "host in (" + ",".join("?" * len(keys)) + ")"

    def _ok_or_pending(self, where: str, args: tuple, day: str | None = None, since_ts: float | None = None) -> int:
        """Successes plus live reservations (a pending row may still succeed) on `day` or since `since_ts`."""
        now = self.clock()
        time_cond, targs = ("day=?", (day or self.today(),)) if since_ts is None else ("ts>?", (since_ts,))
        # remote-job leases (channel 'lease') hold quota until the job is collected, however long that takes
        return self._n(f"{time_cond} and (status='ok' or (status='pending' and (ts>? or channel='lease'))) and {where}",
                       (*targs, now - PENDING_TTL_S, *args))

    def cooling(self, host: str, *related: str) -> str | None:
        for h in (host, *related):
            row = self.db.execute("select until, reason from cooldown where host=?", (h,)).fetchone()
            if row and row[0] > self.clock():
                return f"{h} in cooldown until {time.strftime('%F %T', time.localtime(row[0]))}: {row[1]}"
        return None

    def cooling_until(self, *hosts: str) -> float | None:
        """Latest live cooldown end among `hosts` (the time all of them are clear), None if none is cooling."""
        now = self.clock()
        ends = [r[0] for h in hosts if (r := self.db.execute("select until from cooldown where host=?", (h,)).fetchone())
                and r[0] > now]
        return max(ends) if ends else None

    def _hour_reset(self, where: str, args: tuple) -> float:
        """When the oldest row counted by an hourly cap leaves the window (the cap lifts then at the earliest)."""
        now = self.clock()
        t = self.db.execute(f"select min(ts) from attempts where ts>? and {where}", (now - 3600, *args)).fetchone()[0]
        return (t or now) + 3600

    def _count_blocks(self, channel: str) -> bool:
        return bool(self.limits.get(channel, {}).get("count_blocks_in_breaker", True))

    def breaker(self, channel: str, day: str | None = None) -> str | None:
        lim = self.limits.get(channel, {})
        max_blocks = lim.get("breaker_blocks_per_day", self.g["breaker_blocks_per_day"])
        n_blocks = self.db.execute("select count(*) from blocks where day=? and channel=?",
                                   (day or self.today(), channel)).fetchone()[0]
        if max_blocks and n_blocks >= max_blocks:
            return f"{channel} breaker open: {n_blocks} block signals today"
        k = int(lim.get("breaker_consecutive_fail", self.g["breaker_consecutive_fail"]) or 0)
        rows = self.db.execute("select status from attempts where day=? and channel=? and status in ('ok','fail') "
                               "and note not like 'cancelled%' and note not like 'not_entitled%' "
                               "order by ts desc limit ?",
                               (day or self.today(), channel, k)).fetchall()
        if k and len(rows) >= k and all(r[0] == "fail" for r in rows):
            return f"{channel} breaker open: last {k} attempts failed"
        return None

    # ------------------------------------------------------------------ checks
    def check(self, channel: str, host: str, related: tuple = (), slot: float | None = None, group: tuple = ()):
        """Raise LimitReached unless one more request on (channel, host) fits, counting over the day of `slot`.
        `group`: further keys that share the host's caps and pacing (one publisher under several keys)."""
        if channel not in self.limits:
            raise LimitReached(f"unknown channel {channel}")
        lim = self.limits[channel]
        now = self.clock()
        day = local_day(slot if slot is not None else now)
        mid = next_midnight(slot if slot is not None else now)
        if (msg := self.cooling(f"channel:{channel}")):
            raise LimitReached(msg, "cooldown", until=self.cooling_until(f"channel:{channel}"))
        if (msg := self.cooling(host, *related, *group)):
            raise LimitReached(msg, "cooldown", "host", until=self.cooling_until(host, *related, *group))
        if (msg := self.breaker(channel)):                  # today's breaker resets at today's midnight
            raise LimitReached(msg, "breaker", until=next_midnight(now))
        if day != self.today() and (msg := self.breaker(channel, day)):
            raise LimitReached(msg, "breaker", until=mid)
        if self._ok_or_pending("1", (), day=day) >= self.g["daily_ok"]:
            raise LimitReached(f"global daily cap {self.g['daily_ok']} reached", until=mid)
        if self._ok_or_pending("channel=?", (channel,), day=day) >= lim["daily_ok"]:
            raise LimitReached(f"{channel} daily success cap {lim['daily_ok']} reached", until=mid)
        if self._ok_or_pending("channel=?", (channel,), since_ts=now - 3600) >= lim["hourly_ok"]:
            raise LimitReached(f"{channel} hourly success cap {lim['hourly_ok']} reached",
                               until=self._hour_reset("channel=? and status in ('ok','pending')", (channel,)))
        if self._n(f"day=? and channel=? and status!='hop' and {_SENT}", (day, channel)) >= lim["daily_attempts"]:
            raise LimitReached(f"{channel} daily attempt cap {lim['daily_attempts']} reached", until=mid)
        keys = (host, *group)
        host_day = self.host_caps.get(host, lim.get("host_daily_attempts") or self.g["per_host_daily_attempts"])
        if self._n(f"day=? and {self._in(keys)} and {_SENT}", (day, *keys)) >= host_day:
            raise LimitReached(f"{host} daily attempt cap {host_day} reached", "cap", "host", until=mid)
        if self._n(f"ts>? and {self._in(keys)} and {_SENT}", (now - 3600, *keys)) >= lim["host_hourly_attempts"]:
            raise LimitReached(f"{host} hourly attempt cap {lim['host_hourly_attempts']} reached", "cap", "host",
                               until=self._hour_reset(self._in(keys), keys))

    def acquire_meta(self, host: str) -> None:
        """Pace + cap a metadata lookup (not a download).  Sleeps until the host's meta slot."""
        key = f"meta:{host}"
        with self._tx():
            if (msg := self.cooling(host)):
                raise LimitReached(msg, "cooldown", "host", until=self.cooling_until(host))
            now = self.clock()
            n_day = self.db.execute("select count(*) from meta_calls where day=? and host=?",
                                    (self.today(), host)).fetchone()[0]
            n_hour = self.db.execute("select count(*) from meta_calls where ts>? and host=?",
                                     (now - 3600, host)).fetchone()[0]
            if n_day >= self.meta["daily_calls"]:
                raise LimitReached(f"metadata cap reached for {host}", "cap", "host", until=next_midnight(now))
            if n_hour >= self.meta["hourly_calls"]:
                t0 = self.db.execute("select min(ts) from meta_calls where ts>? and host=?",
                                     (now - 3600, host)).fetchone()[0]
                raise LimitReached(f"metadata cap reached for {host}", "cap", "host", until=(t0 or now) + 3600)
            nxt = self.db.execute("select next_ts from hosts where host=?", (key,)).fetchone()
            slot = max(now, nxt[0] if nxt else 0.0)
            self.db.execute("insert or replace into hosts values(?,?)",
                            (key, slot + self.meta["gap_s"] + random.uniform(0, self.meta["jitter_s"])))
            self.db.execute("insert into meta_calls values(?,?,?,?,?)", (slot, local_day(slot), host, "sent", None))
        if (wait := slot - self.clock()) > 0:
            self.sleep(wait)
        if (msg := self.cooling(host)):                      # blocked while we waited
            raise LimitReached(msg, "cooldown", "host", until=self.cooling_until(host))

    def _last_sent(self, keys: tuple, exclude: int | None = None) -> float:
        """Latest send time on these keys: finished rows and in-flight ones, not rows still waiting for their
        slot (note 'booked') and not cancelled ones."""
        q = (f"select max(ts) from attempts where {self._in(keys)} and not (status='pending' and note='booked') "
             "and note not like 'cancelled%'")
        args: tuple = keys
        if exclude is not None:
            q += " and rowid != ?"
            args = (*keys, exclude)
        return self.db.execute(q, args).fetchone()[0] or 0.0

    def _next_slot(self, keys: tuple, gap: float) -> float:
        nxt = max((r[0] for r in self.db.execute(f"select next_ts from hosts where {self._in(keys)}", keys)),
                  default=0.0)
        booked = self.db.execute(f"select max(ts) from attempts where {self._in(keys)} and status='pending' "
                                 "and note='booked'", keys).fetchone()[0] or 0.0
        return max(self.clock(), nxt, self._last_sent(keys) + gap, booked + gap if booked else 0.0)

    def acquire(self, channel: str, host: str, doi: str = "", related: tuple = (), group: tuple = ()) -> int:
        """Check caps, book a pending attempt and the next slot of `host` (+ `group`: keys sharing its caps and
        pacing), sleep until the slot, then — in a transaction — re-check cooldowns / breakers, the gap to the
        last real send on the group (another booking may have woken late) and, if the day changed while
        waiting, the caps of the new day.  Returns a reservation id for record().  `related`: further cooldown
        keys that must also be clear."""
        lim = self.limits[channel]
        keys = (host, *group)
        gap = float(lim["gap_s"])
        with self._tx():
            now = self.clock()
            slot = self._next_slot(keys, gap)
            if slot - now > float(lim.get("max_wait_s", MAX_WAIT_S)):
                raise LimitReached(f"{host}: next slot in {slot - now:.0f}s exceeds max_wait_s", "cap", "host",
                                   until=slot - float(lim.get("max_wait_s", MAX_WAIT_S)))
            self.check(channel, host, related, slot, group)
            self.db.execute("insert or replace into hosts values(?,?)",
                            (host, slot + gap + random.uniform(0, lim["jitter_s"])))
            rid = self.db.execute("insert into attempts values(?,?,?,?,?,?,?,?,?)",
                                  (slot, local_day(slot), channel, host, doi, "pending", None, 0,
                                   "booked")).lastrowid
        booked_day = local_day(slot)
        for _ in range(50):                                     # re-paced at most a few times in practice
            if (wait := slot - self.clock()) > 0:
                self.sleep(wait)
            with self._tx():
                msg, kind, scope = self.cooling(host, *related, *group), "cooldown", "host"
                until = self.cooling_until(host, *related, *group) if msg else None
                if not msg:
                    msg, scope = self.cooling(f"channel:{channel}"), "channel"
                    until = self.cooling_until(f"channel:{channel}") if msg else None
                if not msg and (msg := self.breaker(channel)):
                    kind, until = "breaker", next_midnight(self.clock())
                now = self.clock()
                if not msg:
                    last = self._last_sent(keys, exclude=rid)
                    if last and now < last + gap:              # another booking woke late and just sent
                        slot = last + gap + random.uniform(0, lim["jitter_s"])
                        if slot - now > float(lim.get("max_wait_s", MAX_WAIT_S)):
                            msg, kind, scope = f"{host}: re-paced slot too far ahead", "cap", "host"
                        else:
                            continue
                if not msg and local_day(now) != booked_day:   # woke after midnight: count on the new day
                    self._exclude = rid
                    try:
                        self.check(channel, host, related, now, group)
                    except LimitReached as e:
                        msg, kind, scope, until = str(e), e.kind, e.scope, e.until
                    finally:
                        self._exclude = None
                if msg:
                    self.db.execute("update attempts set status='fail', note=? where rowid=?",
                                    (f"{NOT_SENT}{msg}"[:300], rid))
                else:
                    self.db.execute("update attempts set ts=?, day=?, note='' where rowid=?",
                                    (now, local_day(now), rid))
            if msg:                                     # raised after the commit, so the cancellation is kept
                raise LimitReached(msg, kind, scope, until=until)
            return rid
        self.record(channel, host, doi, "fail", None, 0, f"{NOT_SENT}could not get a paced slot", reservation=rid)
        raise LimitReached(f"{host}: could not get a paced slot", "cap", "host")

    def acquire_hop(self, channel: str, host: str, doi: str = "", related: tuple = ()) -> None:
        """A redirect hop to another host inside one download: checks the target's cooldown and per-host caps,
        paces it, and books it as status 'hop' (counts as an attempt on that host, not against the channel's
        daily attempts, never as a success).  After the pacing sleep the cooldown and the gap are re-checked
        like a normal reservation."""
        lim = self.limits[channel]
        gap = float(lim["gap_s"])
        max_wait = float(lim.get("hop_max_wait_s", 120))
        with self._tx():
            if (msg := self.cooling(host, *related)):
                raise LimitReached(msg, "cooldown", "host", until=self.cooling_until(host, *related))
            now = self.clock()
            slot = self._next_slot((host,), gap)
            if slot - now > max_wait:
                raise LimitReached(f"{host}: redirect target busy for {slot - now:.0f}s", "cap", "host",
                                   until=slot - max_wait)
            day = local_day(slot)
            cap = self.host_caps.get(host, lim.get("host_daily_attempts") or self.g["per_host_daily_attempts"])
            if self._n(f"day=? and host=? and {_SENT}", (day, host)) >= cap:
                raise LimitReached(f"{host}: daily attempt cap reached (redirect target)", "cap", "host",
                                   until=next_midnight(slot))
            if self._n(f"ts>? and host=? and {_SENT}", (now - 3600, host)) >= lim["host_hourly_attempts"]:
                raise LimitReached(f"{host}: hourly attempt cap reached (redirect target)", "cap", "host",
                                   until=self._hour_reset(f"host=? and {_SENT}", (host,)))
            self.db.execute("insert or replace into hosts values(?,?)",
                            (host, slot + gap + random.uniform(0, lim["jitter_s"])))
            rid = self.db.execute("insert into attempts values(?,?,?,?,?,?,?,?,?)",
                                  (slot, day, channel, host, doi, "pending", None, 0, "booked")).lastrowid
        for _ in range(20):
            if (wait := slot - self.clock()) > 0:
                self.sleep(wait)
            with self._tx():
                now = self.clock()
                msg = self.cooling(host, *related)
                if not msg:
                    last = self._last_sent((host,), exclude=rid)
                    if last and now < last + gap:
                        slot = last + gap + random.uniform(0, lim["jitter_s"])
                        if slot - now <= max_wait:
                            continue
                        msg = f"{host}: redirect target busy"
                if msg:
                    self.db.execute("update attempts set status='fail', note=? where rowid=?",
                                    (f"{NOT_SENT}{msg}"[:300], rid))
                else:
                    self.db.execute("update attempts set ts=?, day=?, status='hop', note='redirect' where rowid=?",
                                    (now, local_day(now), rid))
            if msg:
                raise LimitReached(msg, "cooldown", "host")
            return
        self.record(channel, host, doi, "fail", None, 0, f"{NOT_SENT}could not pace redirect", reservation=rid)
        raise LimitReached(f"{host}: could not pace redirect", "cap", "host")

    def record(self, channel: str, host: str, doi: str, status: str, http: int | None = None, nbytes: int = 0,
               note: str = "", reservation: int | None = None):
        with self._tx():
            if reservation is not None:
                self.db.execute("update attempts set doi=?, status=?, http=?, bytes=?, note=? where rowid=?",
                                (doi, status, http, nbytes, note[:300], reservation))
            else:
                self.db.execute("insert into attempts values(?,?,?,?,?,?,?,?,?)",
                                (self.clock(), self.today(), channel, host, doi, status, http, nbytes, note[:300]))

    def finalize_pending(self, reservation: int, channel: str, host: str, doi: str, note: str):
        row = self.db.execute("select status from attempts where rowid=?", (reservation,)).fetchone()
        if row and row[0] == "pending":
            self.record(channel, host, doi, "fail", None, 0, note, reservation=reservation)

    def block(self, host: str, reason: str, seconds: float = COOLDOWN_S, channel: str | None = None):
        """Cool `host` for `seconds`; an existing longer cooldown is kept (responses may arrive out of order)."""
        until = self.clock() + seconds
        with self._tx():
            self.db.execute("insert into cooldown values(?,?,?) on conflict(host) do update set "
                            "reason = case when excluded.until > until then excluded.reason else reason end, "
                            "until = max(until, excluded.until)", (host, until, reason[:200]))
            if channel:
                # uncounted (smoke batches): kept for the record under another channel name, so no breaker sees it
                ch = channel if self._count_blocks(channel) else f"{channel}:uncounted"
                self.db.execute("insert into blocks values(?,?,?,?,?)",
                                (self.clock(), self.today(), ch, host, reason[:200]))

    # ------------------------------------------------------------------ not entitled
    def mark_not_entitled(self, channel: str, doi: str, note: str = ""):
        """The channel's API answered for `doi` with metadata / abstract only: never ask that channel again."""
        if not doi:
            return
        with self._tx():
            self.db.execute("insert into not_entitled values(?,?,?,?) on conflict(channel, doi) do update set "
                            "ts = excluded.ts, note = excluded.note", (channel, doi, self.clock(), note[:200]))

    def not_entitled(self, channel: str, doi: str, ttl_days: float | None = None) -> str | None:
        """Why `channel` must not ask for `doi` (a not-entitled answer recorded earlier), else None.  ttl_days:
        forget records older than this (None: never)."""
        row = self.db.execute("select ts, note from not_entitled where channel=? and doi=?", (channel, doi)).fetchone()
        if not row or (ttl_days is not None and self.clock() - row[0] > float(ttl_days) * 86400):
            return None
        return f"{channel}: not entitled to {doi} (recorded {time.strftime('%F', time.localtime(row[0]))}: {row[1]})"

    def usage(self) -> dict:
        out = {}
        for ch, lim in self.limits.items():
            ok = self._n("day=? and channel=? and status='ok'", (self.today(), ch))
            att = self._n(f"day=? and channel=? and status!='hop' and {_SENT}", (self.today(), ch))
            out[ch] = dict(ok=ok, attempts=att, cap_ok=lim.get("daily_ok"), cap_attempts=lim.get("daily_attempts"),
                           breaker=self.breaker(ch))
        out["_meta_calls"] = self.db.execute("select count(*) from meta_calls where day=?", (self.today(),)).fetchone()[0]
        out["_global_ok"] = self._n("day=? and status='ok'", (self.today(),))
        out["_global_cap"] = self.g["daily_ok"]
        out["_cooldown"] = [dict(host=h, until=time.strftime("%F %T", time.localtime(u)), reason=r)
                            for h, u, r in self.db.execute("select host, until, reason from cooldown where until>?",
                                                           (self.clock(),))]
        out["_not_entitled"] = dict(self.db.execute("select channel, count(*) from not_entitled group by channel"))
        return out

    def remaining(self, channel: str) -> int | None:
        lim = self.limits[channel]
        return max(0, min(lim["daily_ok"] - self._ok_or_pending("channel=?", (channel,)),
                          self.g["daily_ok"] - self._ok_or_pending("1", ())))
