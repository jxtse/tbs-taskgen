"""Remote mode for the S3 HTTP channels (oa / elsevier / wiley / inst): run them on another host over ssh.

Use case: the local machine's egress is not on the institution's network, another host (reachable with ssh) is.
The local process only orchestrates; the remote host runs the same tbsgen code (`python -m tbsgen.remote ...`).

Flow of one job (`run_job`):
  1. local: `lease` — in one ledger transaction, book `lease_n` pending rows (channel "lease", host
     "lease:<job>") in the LOCAL ledger and snapshot the ledger state the caps need (today's and the last hour's
     attempts, live cooldowns / host slots, today's blocks and metadata calls).  Lease rows count against the
     global daily cap like any live reservation, so a concurrent local channel (the browser) cannot use the same
     global quota; they never count against a channel's cap or a breaker.
  2. local -> remote over ssh: `submit <workdir> <job>`; stdin line 1 = job spec (config subset, entries, seed,
     code fingerprint), line 2 = secrets (JSON).  The remote side checks that its code fingerprint equals the
     local one, writes the spec (never the secrets), seeds a job-local ledger from the snapshot, and forks a
     detached worker that keeps the secrets in memory only.  `submit` returns at once; an ssh drop after it
     does not stop the worker.  Re-submitting a job id that already ran is a no-op (`already_started`).
  3. the worker refuses to start unless `egress_check` is enabled with a non-empty `allow_cidrs`, then fetches
     the entries with the normal Fetcher (cascade restricted to the HTTP channels) against the seeded ledger,
     so caps, pacing, cooldowns and breakers see the local usage of the day.  The job's global daily cap is
     (seeded global usage + lease_n).  One result file per entry, written atomically; status.json heartbeat.
     It starts no new paper after `max_job_s` or after local midnight.
  4. local: poll `status` until the worker is done or dead, then `export` (tar on stdout: results, cache files,
     ledger rows added after the seed) and `import`: write texts into the local write cache (same collision-safe
     naming), merge the remote ledger rows into the local ledger and drop the lease (one transaction, idempotent
     through the `remote_merged` table), then `cleanup` the remote job dir.
  5. the local job file (<queue>.remote/<job>.json) records the stage, so an interrupted run is finished by the
     next `tbsgen fetch` (it collects outstanding jobs before starting a new one).  One job at a time per queue.

Only one ledger is authoritative: the local one.  The remote ledger is a per-job working copy seeded from it;
its new rows come back exactly once.  Cooldowns set on one side reach the other at the next job (bounded by
max_job_s); see README.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

HTTP_CHANNELS = ("oa", "elsevier", "wiley", "inst")
PKG = Path(__file__).resolve().parent
_TEST_TRANSPORT = None          # tests: fake HTTP for the worker


# ------------------------------------------------------------------------------------------- shared
def code_fingerprint() -> str:
    """sha256 over the package's source files (both sides must run the same code)."""
    h = hashlib.sha256()
    for p in sorted(PKG.glob("*.py")) + sorted(PKG.glob("*.mjs")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def _atomic(path: Path, data: str | bytes):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    (tmp.write_bytes if isinstance(data, bytes) else tmp.write_text)(data)
    os.replace(tmp, path)


def posix_tz(ts: float) -> str:
    """The local UTC offset at `ts` as a POSIX TZ string (both sides count days in the same zone; DST rules are
    not carried over, so keep jobs short around a DST change)."""
    off = int(time.localtime(ts).tm_gmtoff)
    sign = "-" if off >= 0 else "+"          # POSIX: hours WEST of UTC
    off = abs(off)
    return f"TBS{sign}{off // 3600}:{off % 3600 // 60:02d}"


def _midnight_after(ts: float) -> float:
    lt = time.localtime(ts)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + 1, 0, 0, 0, 0, 0, -1))


# ------------------------------------------------------------------------------------------- local: ledger
def lease_and_snapshot(ledger, job: str, n: int, now: float | None = None) -> dict:
    """Book `n` lease rows and snapshot what the remote ledger needs, in one transaction."""
    from .dlledger import PENDING_TTL_S, local_day
    now = ledger.clock() if now is None else now
    day = local_day(now)
    with ledger._tx() as db:
        for _ in range(n):
            db.execute("insert into attempts values(?,?,?,?,?,?,?,?,?)",
                       (now, day, "lease", f"lease:{job}", "", "pending", None, 0, f"lease:{job}"))
        att = db.execute("select ts, day, channel, host, doi, status, http, bytes, note from attempts where "
                         "(day=? or ts>? or (status='pending' and ts>?)) and note != ?",
                         (day, now - 3600, now - PENDING_TTL_S, f"lease:{job}")).fetchall()
        snap = dict(
            attempts=att,
            cooldown=db.execute("select host, until, reason from cooldown where until>?", (now,)).fetchall(),
            hosts=db.execute("select host, next_ts from hosts where next_ts>?", (now - 3600,)).fetchall(),
            blocks=db.execute("select ts, day, channel, host, reason from blocks where day=?", (day,)).fetchall(),
            meta_calls=db.execute("select ts, day, host, status, http from meta_calls where day=? or ts>?",
                                  (day, now - 3600)).fetchall())
    return dict(day=day, ts=now, rows=snap)


def release_lease(ledger, job: str):
    with ledger._tx() as db:
        db.execute("delete from attempts where note=? and channel='lease'", (f"lease:{job}",))


def merge_rows(ledger, job: str, rows: dict) -> dict:
    """Merge the remote job's new ledger rows into the local ledger and drop the job's lease, atomically and at
    most once per remote row (table remote_merged).  Remote rows still pending (worker died) become
    'fail' / 'cancelled: remote worker ended' (they still count as attempts, never towards a breaker)."""
    n = dict(attempts=0, blocks=0, meta_calls=0, cooldown=0, hosts=0)
    with ledger._tx() as db:
        db.execute("create table if not exists remote_merged(job text, kind text, rid int, "
                   "primary key(job, kind, rid))")
        for kind, cols in (("attempts", 9), ("blocks", 5), ("meta_calls", 5)):
            for r in rows.get(kind) or []:
                rid, vals = r[0], list(r[1:])
                if db.execute("insert or ignore into remote_merged values(?,?,?)", (job, kind, rid)).rowcount != 1:
                    continue
                if kind == "attempts" and vals[5] == "pending":
                    vals[5], vals[8] = "fail", "cancelled: remote worker ended"
                db.execute(f"insert into {kind} values({','.join('?' * cols)})", vals[:cols])
                n[kind] += 1
        for host, until, reason in rows.get("cooldown") or []:
            db.execute("insert into cooldown values(?,?,?) on conflict(host) do update set "
                       "reason = case when excluded.until > until then excluded.reason else reason end, "
                       "until = max(until, excluded.until)", (host, until, reason))
            n["cooldown"] += 1
        for host, nxt in rows.get("hosts") or []:
            db.execute("insert into hosts values(?,?) on conflict(host) do update set "
                       "next_ts = max(next_ts, excluded.next_ts)", (host, nxt))
            n["hosts"] += 1
        db.execute("delete from attempts where note=? and channel='lease'", (f"lease:{job}",))
    return n


# ------------------------------------------------------------------------------------------- local: transport
def default_ssh_runner(rcfg: dict):
    base = list(rcfg.get("ssh") or ["ssh"]) + [rcfg["host"]]

    def run(remote_cmd: str, stdin: bytes = b"", timeout: float = 300) -> tuple[int, bytes, bytes]:
        try:
            p = subprocess.run(base + [remote_cmd], input=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 255, b"", b"ssh timeout"
        return p.returncode, p.stdout, p.stderr
    return run


def _remote_cmd(rcfg: dict, *args: str) -> str:
    py = rcfg.get("python") or "python3"
    return " ".join(shlex.quote(x) for x in (py, "-m", "tbsgen.remote", *args))


def load_secrets(fetcher, names: list[str]) -> dict:
    """Secrets for the remote worker, in memory only: from `remote.secrets_cmd` (a command printing a JSON object;
    it reads the wanted names as a JSON list on stdin) or else this process's environment / injected secrets."""
    rcfg = fetcher.ft.get("remote") or {}
    out: dict = {}
    if rcfg.get("secrets_cmd"):
        p = subprocess.run(list(rcfg["secrets_cmd"]), input=json.dumps(names).encode(), capture_output=True,
                           timeout=120)
        if p.returncode != 0:
            raise RuntimeError(f"remote.secrets_cmd failed (exit {p.returncode})")
        try:
            got = json.loads(p.stdout.decode() or "{}")
        except json.JSONDecodeError:
            raise RuntimeError("remote.secrets_cmd did not print a JSON object") from None
        out = {k: v for k, v in got.items() if k in names and isinstance(v, str) and v}
        del got, p
    for k in names:
        if k not in out and (v := fetcher.secret(k)):
            out[k] = v
    return out


def secret_names(ft: dict, channels) -> list[str]:
    names = []
    if "elsevier" in channels and (ft.get("elsevier") or {}).get("enabled"):
        names += [ft["elsevier"].get("key_env"), ft["elsevier"].get("insttoken_env")]
    if "wiley" in channels and (ft.get("wiley") or {}).get("enabled"):
        names += [ft["wiley"].get("token_env")]
    return [n for n in names if n]


# ------------------------------------------------------------------------------------------- local: jobs
def jobs_dir(cfg: dict) -> Path:
    from .fulltext import queue_path
    q = queue_path(cfg)
    return q.with_name(q.name + ".remote")


def _job_state(cfg, job) -> Path:
    return jobs_dir(cfg) / f"{job}.json"


def outstanding_jobs(cfg: dict) -> list[dict]:
    d = jobs_dir(cfg)
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        with contextlib.suppress(OSError, ValueError):
            j = json.loads(p.read_text())
            if j.get("stage") not in ("collected", "dropped"):
                out.append(j)
    return out


def remote_spec(fetcher, job: str, entries: list[dict], channels: list[str], lease_n: int, snap: dict,
                now: float) -> dict:
    import copy
    ft = copy.deepcopy(fetcher.ft)
    rcfg = ft.pop("remote")
    ft.pop("browser", None)
    wd = rcfg["workdir"].rstrip("/") + f"/jobs/{job}"
    ft.update(cache_dirs=[], write_cache_dir=f"{wd}/cache", ledger_path=f"{wd}/ledger.sqlite",
              queue_path=f"{wd}/queue.jsonl", cascade=[c for c in ft.get("cascade") or HTTP_CHANNELS
                                                        if c in channels],
              pdftotext=rcfg.get("pdftotext"), ca_file=rcfg.get("ca_file"))
    from .dlledger import DownloadLedger
    led: DownloadLedger = fetcher.ledger
    from .dlledger import PENDING_TTL_S
    # count exactly like DownloadLedger._ok_or_pending on the remote copy (live reservations only)
    seeded_global = sum(1 for r in snap["rows"]["attempts"]
                        if r[1] == snap["day"] and (r[5] == "ok" or (r[5] == "pending" and r[0] > now - PENDING_TTL_S)))
    gl = dict(ft.get("global_limits") or {})
    gl["daily_ok"] = min(int(led.g["daily_ok"]), seeded_global + lease_n)
    ft["global_limits"] = gl
    ft["limits"] = {ch: dict(v) for ch, v in led.limits.items()}       # the local effective limits
    ft["host_caps"] = dict(led.host_caps)
    deadline = min(now + float(rcfg.get("max_job_s", 7200)), _midnight_after(now) - 60)
    return dict(job=job, fingerprint=code_fingerprint(), tz=posix_tz(now), fulltext=ft, entries=entries, channels=channels, deadline=deadline, seed=snap)


def run_job(fetcher, entries: list[dict], channels: list[str] | None = None, *, log=print, ssh_runner=None,
            poll_s: float = 15.0, sleep=time.sleep, max_poll_s: float | None = None) -> dict:
    """Fetch `entries` on the remote host.  Returns {ident: Outcome-like dict} for collected results plus stats.
    Outstanding older jobs are collected first."""
    rcfg = fetcher.ft["remote"]
    run = ssh_runner or default_ssh_runner(rcfg)
    import fcntl
    jobs_dir(fetcher.cfg).mkdir(parents=True, exist_ok=True)
    with open(jobs_dir(fetcher.cfg) / ".lock", "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(results={}, jobs=[], busy=True, note="another `tbsgen fetch` is running a remote job")
        try:
            return _run_job_locked(fetcher, entries, channels, run, log, poll_s, sleep, max_poll_s)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _run_job_locked(fetcher, entries, channels, run, log, poll_s, sleep, max_poll_s) -> dict:
    from .fulltext import paper_ident
    rcfg = fetcher.ft["remote"]
    out: dict = dict(results={}, jobs=[])
    for j in outstanding_jobs(fetcher.cfg):                 # finish what an interrupted run left
        r = collect(fetcher, j["job"], run, log=log, poll_s=poll_s, sleep=sleep, max_poll_s=max_poll_s)
        out["jobs"].append(r)
        out["results"].update(r.get("results") or {})
        if r.get("stage") not in ("collected", "dropped"):
            out["note"] = f"remote job {j['job']} still running; not starting a new one"
            return out
    channels = [c for c in (channels or HTTP_CHANNELS) if c in HTTP_CHANNELS
                and (c == "oa" and (fetcher.ft.get("oa") or {}).get("enabled", True)
                     or (fetcher.ft.get(c) or {}).get("enabled"))]
    entries = [e for e in entries if paper_ident(e) not in out["results"]][: int(rcfg.get("batch_max", 50))]
    if not entries or not channels:
        return out
    if not fetcher.ft.get("allow_network"):
        out["note"] = "allow_network=false: remote job not started"
        return out
    led = fetcher.ledger
    gl_left = max(0, int(led.g["daily_ok"]) - led._ok_or_pending("1", ()))
    lease_n = min(len(entries), gl_left)
    if lease_n <= 0:
        out["note"] = "global daily cap reached"
        return out
    job = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    jobs_dir(fetcher.cfg).mkdir(parents=True, exist_ok=True)
    now = led.clock()
    # the job file is written before the lease, so a crash right after leasing is still visible
    _atomic(_job_state(fetcher.cfg, job), json.dumps(dict(job=job, stage="leasing", created=now, n=len(entries),
                                                          idents=[paper_ident(e) for e in entries])))
    snap = lease_and_snapshot(led, job, lease_n, now)
    spec = remote_spec(fetcher, job, entries, channels, lease_n, snap, now)
    _atomic(_job_state(fetcher.cfg, job), json.dumps(dict(job=job, stage="leased", created=now, n=len(entries),
                                                          lease_n=lease_n,
                                                          idents=[paper_ident(e) for e in entries])))
    names = secret_names(fetcher.ft, channels)
    secrets = load_secrets(fetcher, names)
    payload = (json.dumps(spec) + "\n" + json.dumps(secrets) + "\n").encode()
    del secrets
    rc, so, se = run(_remote_cmd(rcfg, "submit", rcfg["workdir"], job), payload, 120)
    del payload
    msg = _last_json(so)
    if rc == 0 and msg.get("ok"):
        _set_stage(fetcher.cfg, job, "submitted")
        log(json.dumps(dict(remote_job=job, submitted=len(entries), lease=lease_n, channels=channels)))
    elif msg.get("error"):                       # refused by the remote side (fingerprint, bad spec): no work done
        release_lease(led, job)
        _set_stage(fetcher.cfg, job, "dropped", error=fetcher._r(str(msg.get("error")))[:300])
        out["jobs"].append(dict(job=job, stage="dropped", error=fetcher._r(str(msg["error"]))[:300]))
        return out
    else:                                        # ssh dropped: the job may or may not have started; collect decides
        _set_stage(fetcher.cfg, job, "submitted", note=f"submit rc={rc}")
        log(json.dumps(dict(remote_job=job, submit_uncertain=rc, stderr=fetcher._r(se.decode("utf-8", "replace"))[-200:])))
    r = collect(fetcher, job, run, log=log, poll_s=poll_s, sleep=sleep, max_poll_s=max_poll_s)
    out["jobs"].append(r)
    out["results"].update(r.get("results") or {})
    return out


def _last_json(b: bytes) -> dict:
    for line in reversed((b or b"").decode("utf-8", "replace").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            with contextlib.suppress(ValueError):
                return json.loads(line)
    return {}


def _set_stage(cfg, job, stage, **kw):
    p = _job_state(cfg, job)
    j = json.loads(p.read_text()) if p.exists() else dict(job=job)
    j.update(stage=stage, **kw)
    _atomic(p, json.dumps(j))


def collect(fetcher, job: str, run, *, log=print, poll_s: float = 15.0, sleep=time.sleep,
            max_poll_s: float | None = None) -> dict:
    """Wait for the remote job to end, import its results and ledger rows, clean up.  Safe to repeat."""
    rcfg = fetcher.ft["remote"]
    st_path = _job_state(fetcher.cfg, job)
    js = json.loads(st_path.read_text()) if st_path.exists() else dict(job=job)
    if js.get("stage") in ("collected", "dropped"):
        return dict(job=job, stage=js["stage"])
    t0 = time.monotonic()
    limit = float(max_poll_s if max_poll_s is not None else rcfg.get("max_job_s", 7200) + 600)
    status: dict = {}
    while True:
        rc, so, _ = run(_remote_cmd(rcfg, "status", rcfg["workdir"], job), b"", 60)
        status = _last_json(so) if rc == 0 else {}
        if status.get("state") in ("done", "dead", "missing"):
            break
        if time.monotonic() - t0 > limit:
            return dict(job=job, stage=js.get("stage"), note="still running", status=status)
        sleep(poll_s)
    if status["state"] == "missing":
        # `submit` writes status.json before it answers and the job dir is only removed after the local stage
        # became "collected": a missing job never started, so nothing was used
        release_lease(fetcher.ledger, job)
        _set_stage(fetcher.cfg, job, "dropped", error="job not found on remote host")
        return dict(job=job, stage="dropped", error="job not found on remote host")
    rc, so, se = run(_remote_cmd(rcfg, "export", rcfg["workdir"], job), b"", 600)
    if rc != 0:
        return dict(job=job, stage=js.get("stage"), error=f"export failed rc={rc}")
    res = import_export(fetcher, job, so)
    _set_stage(fetcher.cfg, job, "collected", merged=res["merged"], n_results=len(res["results"]))
    run(_remote_cmd(rcfg, "cleanup", rcfg["workdir"], job), b"", 60)
    log(json.dumps(dict(remote_job=job, collected=len(res["results"]), ok=sum(r["ok"] for r in res["results"].values()),
                        merged=res["merged"], worker=status.get("state"))))
    return dict(job=job, stage="collected", results=res["results"], merged=res["merged"], status=status)


def import_export(fetcher, job: str, blob: bytes) -> dict:
    """Unpack the export tar (only regular files under results/ and cache/, plus ledger.json), save texts to the
    local write cache, merge ledger rows.  Returns {results: {ident: dict(ok, text, rec)}, merged}."""
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tf:
        for m in tf.getmembers():
            name = m.name
            if not m.isreg() or name.startswith("/") or ".." in Path(name).parts:
                continue
            if not (name == "ledger.json" or name.startswith(("results/", "cache/"))):
                continue
            files[name] = tf.extractfile(m).read()
    results: dict = {}
    for name, data in sorted(files.items()):
        if not (name.startswith("results/") and name.endswith(".json")):
            continue
        r = json.loads(data)
        rec = r.get("rec") or {}
        text = None
        if r.get("ok") and r.get("text_file") and f"cache/{r['text_file']}" in files:
            text = files[f"cache/{r['text_file']}"].decode("utf-8", "replace")
            raw = files.get(f"cache/raw/{r.get('raw_file')}") if r.get("raw_file") else None
            prov = dict(rec.get("provenance") or {}, via=f"remote:{job}")
            prov.pop("path", None)
            saved = fetcher.save_imported(r["entry"], prov, text, raw)
            prov = dict(prov, path=saved.get("path"))
            rec = dict(rec, provenance=prov, remote_job=job)
        rec = dict(rec, remote_seconds=r.get("seconds"))
        results[r["ident"]] = dict(ok=text is not None, text=text, rec=rec)
    rows = json.loads(files.get("ledger.json", b"{}"))
    merged = merge_rows(fetcher.ledger, job, rows)
    return dict(results=results, merged=merged)


# ------------------------------------------------------------------------------------------- remote side
def _job_dir(workdir: str, job: str) -> Path:
    if not job or "/" in job or job.startswith("."):
        raise ValueError("bad job id")
    return Path(workdir).expanduser() / "jobs" / job


def _seed_ledger(path: Path, seed: dict) -> dict:
    from .dlledger import DownloadLedger
    DownloadLedger(path)                                  # creates the schema
    db = sqlite3.connect(path, isolation_level=None)
    db.execute("begin immediate")
    rows = seed.get("rows") or {}
    for kind, n in (("attempts", 9), ("cooldown", 3), ("hosts", 2), ("blocks", 5), ("meta_calls", 5)):
        for r in rows.get(kind) or []:
            db.execute(f"insert or replace into {kind} values({','.join('?' * n)})", r)
    marks = {k: db.execute(f"select coalesce(max(rowid),0) from {k}").fetchone()[0]
             for k in ("attempts", "blocks", "meta_calls")}
    db.execute("commit")
    db.close()
    return marks


def cmd_submit(workdir: str, job: str, stdin, stdout, daemon: bool = True) -> int:
    d = _job_dir(workdir, job)
    spec_line = stdin.readline()
    sec_line = stdin.readline()
    try:
        spec = json.loads(spec_line)
        secrets = json.loads(sec_line or "{}")
    except ValueError:
        print(json.dumps(dict(ok=False, error="bad spec")), file=stdout)
        return 2
    if (d / "status.json").exists():
        st = json.loads((d / "status.json").read_text())
        print(json.dumps(dict(ok=True, already_started=True, state=st.get("state"))), file=stdout)
        return 0
    if spec.get("fingerprint") != code_fingerprint():
        print(json.dumps(dict(ok=False, error="code fingerprint mismatch: deploy the same tbsgen revision")),
              file=stdout)
        return 2
    ec = (spec.get("fulltext") or {}).get("egress_check") or {}
    if not ec.get("enabled") or not ec.get("allow_cidrs"):
        print(json.dumps(dict(ok=False, error="egress_check must be enabled with allow_cidrs")), file=stdout)
        return 2
    d.mkdir(parents=True, exist_ok=True)
    (d / "results").mkdir(exist_ok=True)
    marks = _seed_ledger(d / "ledger.sqlite", spec.get("seed") or {})
    spec.pop("seed", None)
    spec["marks"] = marks
    _atomic(d / "spec.json", json.dumps(spec))
    _atomic(d / "status.json", json.dumps(dict(state="starting", ts=time.time(), done=0, n=len(spec["entries"]))))
    if not daemon:
        print(json.dumps(dict(ok=True, pid=os.getpid(), inline=True)), file=stdout)
        stdout.flush()
        return work(d, spec, secrets)
    pid = os.fork()
    if pid:
        print(json.dumps(dict(ok=True, pid=pid)), file=stdout)
        stdout.flush()
        return 0
    os.setsid()                                 # detach: an ssh drop does not stop the worker
    if os.fork():
        os._exit(0)
    fd = os.open(str(d / "worker.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    try:
        rc = work(d, spec, secrets)
    finally:
        os._exit(0)
    return rc


def work(d: Path, spec: dict, secrets: dict, stop_after: int | None = None) -> int:
    """The remote worker: fetch every entry, one atomic result file each.  `stop_after` (tests): simulate a
    worker that dies after that many entries (it leaves status 'running' and no further results)."""
    from . import fulltext
    from .config import load_config
    if spec.get("tz") and spec["tz"] != posix_tz(time.time()):
        os.environ["TZ"] = spec["tz"]          # count days in the orchestrator's zone
        with contextlib.suppress(AttributeError):
            time.tzset()
    ft = spec["fulltext"]
    if ft.get("pdftotext"):
        fulltext.PDFTOTEXT = ft["pdftotext"]
    cfg = load_config(None, {"run": {"root": str(d)}, "fulltext": ft})
    cfg["fulltext"]["remote"]["enabled"] = False
    cfg["fulltext"]["browser"]["enabled"] = False
    st = dict(state="running", pid=os.getpid(), ts=time.time(), done=0, n=len(spec["entries"]))
    _atomic(d / "status.json", json.dumps(st))
    f = fulltext.Fetcher(cfg, secrets=secrets, transport=_TEST_TRANSPORT)
    del secrets
    for i, e in enumerate(spec["entries"]):
        ident = fulltext.paper_ident(e)
        rp = d / "results" / f"{i:04d}.json"
        if rp.exists():
            st["done"] += 1
            continue
        if stop_after is not None and i >= stop_after:
            return 1
        if time.time() > float(spec["deadline"]):
            break
        t0 = time.time()
        try:
            out = f.fetch(e, browser=False, channels=list(cfg["fulltext"]["cascade"]))
        except Exception as ex:  # noqa: BLE001
            out = fulltext.Outcome(None, dict(errors=[f._r(f"{type(ex).__name__}: {ex}")], skipped=[], source=None))
        res = dict(ident=ident, entry=e, ok=out.text is not None, rec=out.rec, seconds=round(time.time() - t0, 1))
        prov = out.rec.get("provenance") or {}
        if out.text is not None and prov.get("path"):
            p = Path(prov["path"])
            res["text_file"] = p.name
            raws = sorted((p.parent / "raw").glob(p.stem + ".*")) if (p.parent / "raw").exists() else []
            if raws:
                res["raw_file"] = raws[0].name
        _atomic(rp, json.dumps(res, ensure_ascii=False))
        st.update(done=st["done"] + 1, ts=time.time())
        _atomic(d / "status.json", json.dumps(st))
        print(json.dumps(dict(i=i, ident=ident, ok=res["ok"], source=out.rec.get("source"), seconds=res["seconds"],
                              errors=out.rec.get("errors", [])[-2:]), ensure_ascii=False), flush=True)
    st.update(state="done", ts=time.time(), egress=getattr(f, "_egress", None))
    _atomic(d / "status.json", json.dumps(st, default=str))
    return 0


def cmd_status(workdir: str, job: str, stdout) -> int:
    d = _job_dir(workdir, job)
    if not (d / "status.json").exists():
        print(json.dumps(dict(state="missing")), file=stdout)
        return 0
    st = json.loads((d / "status.json").read_text())
    if st.get("state") in ("running", "starting"):
        alive = False
        if st.get("pid"):
            try:
                os.kill(int(st["pid"]), 0)
                alive = True
            except (OSError, ValueError):
                alive = False
        if not alive and time.time() - float(st.get("ts") or 0) > 30:
            st["state"] = "dead"
    print(json.dumps(st, default=str), file=stdout)
    return 0


def cmd_export(workdir: str, job: str, stdout) -> int:
    d = _job_dir(workdir, job)
    spec = json.loads((d / "spec.json").read_text())
    marks = spec.get("marks") or {}
    db = sqlite3.connect(d / "ledger.sqlite")
    rows = {k: db.execute(f"select rowid, * from {k} where rowid > ?", (marks.get(k, 0),)).fetchall()
            for k in ("attempts", "blocks", "meta_calls")}
    rows["cooldown"] = db.execute("select host, until, reason from cooldown").fetchall()
    rows["hosts"] = db.execute("select host, next_ts from hosts").fetchall()
    db.close()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        def add(name, data: bytes):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(data))
        add("ledger.json", json.dumps(rows).encode())
        for p in sorted((d / "results").glob("*.json")):
            add(f"results/{p.name}", p.read_bytes())
        cache = d / "cache"
        if cache.exists():
            for p in sorted(cache.rglob("*")):
                if p.is_file() and not p.name.startswith("."):
                    add("cache/" + str(p.relative_to(cache)), p.read_bytes())
    out = getattr(stdout, "buffer", stdout)
    out.write(buf.getvalue())
    out.flush()
    return 0


def cmd_cleanup(workdir: str, job: str, stdout) -> int:
    import shutil
    d = _job_dir(workdir, job)
    if d.exists():
        shutil.rmtree(d)
    print(json.dumps(dict(ok=True)), file=stdout)
    return 0


def main(argv=None, stdin=None, stdout=None, daemon: bool = True) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    if len(argv) != 3 or argv[0] not in ("submit", "status", "export", "cleanup"):
        print("usage: python -m tbsgen.remote {submit|status|export|cleanup} WORKDIR JOB", file=sys.stderr)
        return 2
    cmd, wd, job = argv
    if cmd == "submit":
        return cmd_submit(wd, job, stdin, stdout, daemon)
    return dict(status=cmd_status, export=cmd_export, cleanup=cmd_cleanup)[cmd](wd, job, stdout)


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
