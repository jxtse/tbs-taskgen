"""Browser batches for S3 (`tbsgen fetch --browser [--wait-for-egress]`): egress-gated, time-boxed.

The browser runs on the local machine and must reach publishers from an allowed egress (fulltext.egress_check.
allow_cidrs).  When that egress is only available in short windows the user opens by hand (for example while a
secure-access VPN client is paused for 10 minutes), a batch is sized to fit the window:

  1. egress: an IP-echo page (e.g. a publisher's /cdn-cgi/trace) opened through the browser itself, on its own
     TaskSpace.  Probes are not downloads (no ledger row) but have their own pacing: at least `batch.poll_s`
     (>= 5 s) apart overall and `probe_host_gap_s` per echo host, rotating over `batch.watch_urls`.
     --wait-for-egress polls until the egress is allowed (or `batch.max_wait_s` passes: exit 0, nothing done);
     without it a single probe decides, and an outside egress refuses the batch.
  2. window: starts at the first allowed probe; deadline = start + window_s - margin_s.  The batch is planned
     from the time left (plan_batch), papers of different publishers interleaved.
  3. script phase: the reviewed script (browser_fetch.mjs) for every planned paper, through the normal
     Fetcher / ledger path (pacing, caps, cooldowns, validation, cache).  A paper is only started when its
     publisher slot is free before the deadline.
  4. agent phase (`browser.agent.enabled`): one Hermes CLI run (`hermes chat --oneshot`, the configured
     provider / model / skills) for the papers the script could not get, with a prompt generated from the
     reviewed template browser_agent_prompt.md.  The agent reaches the browser only through a per-batch
     `ego-browser` wrapper (first on PATH) that books the ledger slot of a paper before its first browser
     call, refuses scripts using denied APIs and prepends browser_agent_guard.mjs (domain allow-list, no
     typing, no verification widgets, files only under the batch directory).  The agent's claims are not
     trusted: after the run, code validates every saved file (PDF / text length / title match) and books the
     outcome in the ledger.
  5. a watchdog re-probes the egress every `batch.recheck_s` during the batch; when it leaves allow_cidrs the
     batch stops at once (the running script / agent process is killed, a result obtained meanwhile is
     discarded) and every paper not finished stays in the queue, still due.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import string
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import browser_ego
from .dlledger import LimitReached

PROMPT = Path(__file__).with_name("browser_agent_prompt.md")
GUARD = Path(__file__).with_name("browser_agent_guard.mjs")
REFUSE = "browser egress is outside fulltext.egress_check.allow_cidrs; open the allowed egress first"
# Scripts the agent may not run at all (the guard also blocks these at run time).
DENY_SCRIPT = re.compile(r"claimTaskSpace|takeOverTaskSpace|listTaskSpaces|userPage|handOff|\.adopt\(|\.release\(|"
                         r"\.tabs\(|\.cdp\(|keyboard\.|mouse\.|setInputFiles|waitForFileChooser|document\.cookie|"
                         r"localStorage|sessionStorage|clearing-state|child_process|node:net|node:http|"
                         r"node:https|globalThis\.fetch|\bprocess\.env\b|GUARD\s*=")
# The agent's shell is an allowlist (checked live on the stream; the run is killed on anything else): one
# `ego-browser nodejs` call with the script as a quoted heredoc and nothing after it, `ego-browser nodejs
# --help`, `ls` of a path, or `sleep N`.  Hermes may run its terminal with its own secrets in the environment,
# so no other command (env, printenv, echo $VAR, cat, curl, ...) may run.
_HEREDOC = re.compile(r"^ego-browser nodejs\s*<<-?\s*'([A-Za-z_]+)'\s*$")
_SIMPLE = re.compile(r"^(ego-browser nodejs --help|ego-browser skill|ls(\s+-[al1]+)?(\s+[A-Za-z0-9_./~-]+)*|sleep\s+\d{1,3})\s*$")
DENY_TEXT = re.compile(r"sci-?hub|libgen|z-lib|annas-archive", re.I)


def shell_allowed(cmd: str) -> str | None:
    """None if the agent's terminal command is allowed, else the reason."""
    if not isinstance(cmd, str) or not cmd.strip():
        return "empty command"
    if DENY_TEXT.search(cmd):
        return "unauthorized source"
    lines = cmd.strip("\n").split("\n")
    first = lines[0].strip()
    if _SIMPLE.match(first):
        return None if len(lines) == 1 else "extra lines after a simple command"
    m = _HEREDOC.match(first)
    if not m:
        return f"command not allowed: {first[:60]}"
    term = m.group(1)
    try:
        end = next(i for i, ln in enumerate(lines[1:], 1) if ln.strip() == term)
    except StopIteration:
        return "unterminated heredoc"
    if any(ln.strip() for ln in lines[end + 1:]):
        return "commands after the heredoc"
    return None


class EgressLost(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- egress probes
@dataclass
class Prober:
    """Paced IP-echo probes through the browser (or an injected function for tests)."""
    ft: dict
    probe_fn: Callable | None = None             # (url) -> body text; default: ego-browser on the probe TaskSpace
    clock: Callable = time.time
    sleep: Callable = time.sleep
    last: float = 0.0
    last_host: dict = field(default_factory=dict)
    i: int = 0
    n: int = 0

    def __post_init__(self):
        b = self.ft.get("browser") or {}
        bb = b.get("batch") or {}
        ec = self.ft.get("egress_check") or {}
        self.urls = list(bb.get("watch_urls") or ec.get("urls") or [])
        self.poll_s = max(5.0, float(bb.get("poll_s", 5)))
        self.host_gap = float(bb.get("probe_host_gap_s", 15))
        self.lock = threading.Lock()
        self.cfg = dict(b, space=str(b.get("space", "tbsgen-s3-fulltext")) + "-probe")

    def _body(self, url: str) -> str:
        if self.probe_fn is not None:
            return self.probe_fn(url) or ""
        r = browser_ego.run_one(dict(mode="probe", urls=[url], cfg=self.cfg, timeout_ms=8000, timeout_s=30))
        return ((r or {}).get("bodies") or {}).get(url, "")

    def probe(self) -> dict:
        """One paced probe: dict(ok, ip, via, why)."""
        from .fulltext import EgressRefused, Resp, check_egress
        if not self.urls:
            return dict(ok=False, ip=None, why="no egress probe URLs (browser.batch.watch_urls / egress_check.urls)")
        with self.lock:
            host_of = lambda u: re.sub(r"^https?://([^/]+).*$", r"\1", u)  # noqa: E731
            u = self.urls[self.i % len(self.urls)]
            for _ in range(len(self.urls)):                 # next URL whose host is not probed too recently
                u = self.urls[self.i % len(self.urls)]
                self.i += 1
                if self.clock() - self.last_host.get(host_of(u), -1e9) >= self.host_gap:
                    break
            wait = max(self.last + self.poll_s - self.clock(),
                       self.last_host.get(host_of(u), -1e9) + self.host_gap - self.clock())
            if wait > 0:
                self.sleep(wait)
            self.last = self.last_host[host_of(u)] = self.clock()
            self.n += 1
            body = self._body(u)

        def tr(url, headers, timeout, max_bytes):
            return Resp(200 if body else 0, {}, body.encode(), url)
        try:
            r = check_egress(dict(self.ft.get("egress_check") or {}, urls=[u]), tr)
            return dict(ok=True, ip=r["ip"], via=r["via"])
        except EgressRefused as e:
            m = re.search(r"egress IP (\S+)", str(e))
            return dict(ok=False, ip=m.group(1) if m else None, why=str(e))


class Watchdog:
    """Re-probes the egress every `recheck_s` in a thread; `lost` is set (and on_lost called) on the first
    probe outside allow_cidrs.  An unknown IP (probe failed) counts as lost too."""

    def __init__(self, prober: Prober, recheck_s: float, on_lost=None):
        self.prober, self.recheck_s, self.on_lost = prober, recheck_s, on_lost     # probes pace themselves
        self.lost = threading.Event()
        self.stop_ev = threading.Event()
        self.why = ""
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_ev.wait(self.recheck_s):
            r = self.prober.probe()
            if not r["ok"]:
                self.why = r.get("why") or "egress lost"
                self.lost.set()
                if self.on_lost:
                    with contextlib.suppress(Exception):
                        self.on_lost()
                return

    def check_now(self) -> bool:
        if self.lost.is_set():
            return False
        r = self.prober.probe()
        if not r["ok"]:
            self.why = r.get("why") or "egress lost"
            self.lost.set()
            return False
        return True

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()


# ------------------------------------------------------------------------------------------- planning
def plan_batch(entries: list[dict], seconds: float, bb: dict, agent: bool, remaining: int | None) -> list[dict]:
    """Pick papers for a window of `seconds`: round-robin over publishers (no two papers of one publisher in a
    row), at most per_publisher_max each, n <= seconds / (est_script_s + agent_fraction * est_agent_s), the
    ledger's remaining browser quota and batch_max."""
    from .fulltext import publisher_of
    per = float(bb.get("est_script_s", 60)) + (float(bb.get("agent_fraction", 0.5)) * float(bb.get("est_agent_s", 120))
                                                if agent else 0.0)
    n = int(max(0.0, seconds) // max(1.0, per))
    n = min(n, int(bb.get("batch_max", 8)))
    if remaining is not None:
        n = min(n, remaining)
    groups: dict[str, list[dict]] = {}
    for e in entries:
        if e.get("doi"):
            groups.setdefault(publisher_of(e["doi"])[0], []).append(e)
    cap = int(bb.get("per_publisher_max", 2))
    out: list[dict] = []
    for r in range(cap):
        for g in groups.values():
            if r < len(g) and len(out) < n:
                out.append(g[r])
    return out


def registrable(host: str) -> str:
    parts = host.lower().strip(".").split(".")
    if len(parts) >= 3 and parts[-2] in ("ac", "co", "com", "edu", "gov", "org", "net") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


# Extra web domains a publisher serves PDFs from (CDNs, platform hosts).
PUBLISHER_EXTRA = {
    "elsevier": ["sciencedirect.com", "sciencedirectassets.com", "elsevier.com", "els-cdn.com"],
    "wiley": ["wiley.com"],
    "springernature": ["springer.com", "nature.com", "springernature.com", "biomedcentral.com", "springeropen.com"],
    "oup": ["oup.com", "silverchair.com", "silverchair-cdn.com"],
    "tandf": ["tandfonline.com"],
    "iop": ["iop.org"],
    "aip": ["aip.org", "silverchair.com"],
    "cambridge": ["cambridge.org"],
}


def allowed_domains(doi: str, landing: str | None, extra: list[str]) -> list[str]:
    from .fulltext import publisher_of
    name, _, hosts = publisher_of(doi)
    d = {registrable(h) for h in hosts} | set(PUBLISHER_EXTRA.get(name, [])) | set(extra or [])
    if landing:
        h = re.sub(r"^https?://([^/:]+).*$", r"\1", landing)
        if h and h != landing:
            d.add(registrable(h))
    for u in browser_ego.pdf_url_hint(doi):
        d.add(registrable(re.sub(r"^https?://([^/:]+).*$", r"\1", u)))
    return sorted(x for x in d if x)


# ------------------------------------------------------------------------------------------- agent wrapper
WRAPPER = r'''#!{python}
# tbsgen browser batch: ego-browser wrapper for the agent (generated; removed with the batch directory)
import sys
sys.path.insert(0, {pkg_parent!r})
from tbsgen.browser_agent import wrapper_main
sys.exit(wrapper_main({batch_dir!r}, sys.argv[1:]))
'''


def wrapper_main(batch_dir: str, argv: list[str]) -> int:
    """The agent's `ego-browser`: `ego-browser nodejs [-e CODE]` (stdin) with a leading `// paper: N` line."""
    from .dlledger import DownloadLedger
    from .fulltext import publisher_of
    bd = Path(batch_dir)
    st = json.loads((bd / "batch.json").read_text())
    if (bd / "STOP").exists():
        print("tbsgen: STOP — the batch was stopped (egress changed or deadline). Do not retry; end now.")
        return 3
    if argv[:1] == ["skill"]:
        print("tbsgen: the ego-browser guide is already loaded; use `ego-browser nodejs` only.")
        return 0
    if argv[:1] != ["nodejs"]:
        print("tbsgen: only `ego-browser nodejs` is available in this batch")
        return 2
    code = argv[2] if len(argv) >= 3 and argv[1] == "-e" else sys.stdin.read()
    if re.match(r"\s*//\s*finish\s*$", code.strip().splitlines()[0] if code.strip() else ""):
        r = browser_ego.finish(dict(binary=st["ego_binary"], space=st["space"]))
        print(f"tbsgen: TaskSpace closed ({r.get('status')})")
        return 0
    m = re.match(r"\s*//\s*paper:\s*(\d+)\s*\n", code)
    if not m or m.group(1) not in st["papers"]:
        print("tbsgen: the script must start with a line `// paper: N` naming one paper of the batch")
        return 2
    n = m.group(1)
    p = st["papers"][n]
    if (bad := DENY_SCRIPT.search(code)):
        print(f"tbsgen: refused: `{bad.group(0)}` is not allowed in this batch")
        return 2
    sp = bd / "papers" / f"{n}.json"
    ps = json.loads(sp.read_text()) if sp.exists() else dict(calls=0)
    if ps.get("closed"):
        print(f"tbsgen: paper {n} is closed ({ps['closed']}); move on")
        return 2
    if ps["calls"] >= int(st["max_calls_per_paper"]):
        print(f"tbsgen: paper {n}: browser call limit reached; give up on this paper")
        return 2
    led = DownloadLedger(st["ledger_path"], st["limits"], st["global_limits"], st["host_caps"])
    if "rid" not in ps:
        # first browser call for this paper: book its slot (pacing / caps / cooldowns of the publisher)
        name, keys, web_hosts = publisher_of(p["doi"])
        lkey = "chrome:" + p["doi"].split("/", 1)[0]
        aliases = tuple(k for k in keys if k != lkey)
        try:
            slot = led_next_slot(led, "browser", (lkey, *aliases))
            if slot - time.time() > float(st["gate_max_wait_s"]):
                print(f"tbsgen: publisher of paper {n} is paced; its slot opens in {slot - time.time():.0f}s. "
                      "Work on another paper and retry later.")
                return 4
            if slot > float(st["deadline"]):
                print("tbsgen: the batch deadline comes first; end now.")
                return 3
            rid = led.acquire("browser", lkey, p["doi"], related=tuple(web_hosts), group=aliases)
        except LimitReached as e:
            ps["closed"] = f"ledger: {e}"
            _atomic(sp, json.dumps(ps))
            print(f"tbsgen: paper {n} refused by the download ledger ({e}); move on")
            return 2
        ps.update(rid=rid, lkey=lkey, t0=time.time())
    ps["calls"] += 1
    _atomic(sp, json.dumps(ps))
    if (bd / "STOP").exists():
        print("tbsgen: STOP — the batch was stopped. End now.")
        return 3
    files = bd / "files" / n
    files.mkdir(parents=True, exist_ok=True)
    guard = dict(space=st["space"], allow=p["allow"], files_dir=str(files.resolve()), inputs=[])
    script = "const GUARD = " + json.dumps(guard) + ";\n" + GUARD.read_text() + "\n" + code
    env = browser_ego.child_env()
    try:
        r = subprocess.run([st["ego_binary"], "nodejs"], input=script, capture_output=True, text=True,
                           timeout=float(st["call_timeout_s"]), env=env)
        outp = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    except subprocess.TimeoutExpired:
        outp, r = "tbsgen: browser call timed out", None
    if re.search(r"tbsgen guard: (this is a verification page|never interact with a verification)", outp):
        ps = json.loads(sp.read_text())
        ps["challenge"] = True
        _atomic(sp, json.dumps(ps))
    sys.stdout.write(outp[-12000:])
    return 0 if r is not None and r.returncode == 0 else 1


def led_next_slot(led, channel: str, keys: tuple) -> float:
    """When the next request on these ledger keys may go out (read-only; mirrors DownloadLedger._next_slot)."""
    return led._next_slot(keys, float(led.limits[channel]["gap_s"]))


def _atomic(path: Path, data: str):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(data)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------- the batch
def render_prompt(template: str, **kw) -> str:
    return string.Template(template).safe_substitute(**kw)


def default_hermes_runner(argv: list[str], env: dict, timeout: float, on_event, stop: threading.Event) -> dict:
    """Run `hermes chat ... --format stream-json`; call on_event(ev) per JSON line (it returns a reason string
    to kill the run).  Returns dict(exit, killed, result_event)."""
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                         env=env, text=True, start_new_session=True)
    killed = ""
    result = None
    t_end = time.time() + timeout

    def kill(why):
        nonlocal killed
        if not killed:
            killed = why
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGTERM)

    def watch():
        while p.poll() is None:
            if stop.is_set():
                kill("stopped")
            elif time.time() > t_end:
                kill("timeout")
            time.sleep(0.5)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    assert p.stdout is not None
    for line in p.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            result = ev
        if (why := on_event(ev)):
            kill(why)
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    return dict(exit=p.returncode, killed=killed, result=result)


def run_batch(fetcher, *, wait_for_egress: bool = False, max_wait_s: float | None = None, on_success=None,
              log=print, prober: Prober | None = None, hermes_runner=None, clock=time.time,
              sleep=time.sleep, now_window_start: float | None = None) -> dict:
    """One egress-gated browser batch.  Returns stats; stats['refused'] when the egress was outside the
    allowed ranges (without --wait-for-egress)."""
    from .fulltext import paper_ident
    ft = fetcher.ft
    b = ft.get("browser") or {}
    bb = b.get("batch") or {}
    ag = b.get("agent") or {}
    stats: dict = dict(mode="browser-batch", ok=0, failed=0, script_ok=0, agent_ok=0, probes=0, deferred=0,
                       papers=[], errors=[])
    if not b.get("enabled"):
        stats["note"] = "fulltext.browser.enabled is false"
        return stats
    if not ft.get("allow_network"):
        stats["note"] = "fulltext.allow_network is false"
        return stats
    ec = ft.get("egress_check") or {}
    if not ec.get("enabled") or not ec.get("allow_cidrs"):
        stats["refused"] = "browser batches need fulltext.egress_check enabled with allow_cidrs"
        return stats
    prober = prober or Prober(ft, clock=clock, sleep=sleep)
    if not fetcher._space_held and not fetcher._try_space_lock():
        stats["note"] = "browser TaskSpace is in use by another process"
        return stats
    hint = (" " + bb["refuse_hint"]) if bb.get("refuse_hint") else ""
    # ---- 1. egress
    t_wait0 = clock()
    limit = float(max_wait_s if max_wait_s is not None else bb.get("max_wait_s", 1800))
    first = None
    while True:
        r = prober.probe()
        stats["probes"] = prober.n
        if r["ok"]:
            first = r
            break
        if not wait_for_egress:
            stats["refused"] = f"{REFUSE} (egress {r.get('ip') or 'unknown'}).{hint}"
            stats["egress"] = r
            return stats
        if clock() - t_wait0 >= limit:
            stats["waited_s"] = round(clock() - t_wait0, 1)
            stats["note"] = f"no allowed browser egress within {limit:.0f}s; nothing done"
            stats["egress"] = r
            return stats
    start = clock() if now_window_start is None else now_window_start
    deadline = start + float(bb.get("window_s", 600)) - float(bb.get("margin_s", 60))
    stats.update(egress=first, window_start=start, deadline=deadline, waited_s=round(start - t_wait0, 1))
    log(json.dumps(dict(event="egress_ok", ip=first["ip"], via=first.get("via"),
                        seconds_left=round(deadline - clock()))))
    fetcher._egress_browser = dict(ok=True, ip=first["ip"], via=first.get("via"), ts=time.time())
    # ---- 2. plan
    from .fulltext import queue_lock, queue_path, read_queue
    qp = queue_path(fetcher.cfg)
    with queue_lock(qp):
        queue = read_queue(qp)
    t_now = time.time()
    # the browser has its own backoff (browser_next_try): a failed HTTP attempt must not hide a paper from the
    # browser, and by default only papers the HTTP cascade already tried are taken (the browser quota is small)
    http_first = bool(bb.get("require_http_first", True))
    due = [e for e in queue if float(e.get("browser_next_try") or 0) <= t_now and e.get("doi")
           and fetcher.channel_applies("browser", e["doi"])
           and (not http_first or e.get("failed_channels") or e.get("tries") or e.get("last_try"))]
    due.sort(key=lambda e: float(e.get("browser_last_try") or 0))
    due = [e for e in due if not fetcher.cached(e)]          # cached papers are delivered by a normal drain
    remaining = fetcher.ledger.remaining("browser")
    use_agent = bool(ag.get("enabled"))
    plan = plan_batch(due, deadline - clock(), bb, use_agent, remaining)
    stats["planned"] = [paper_ident(e) for e in plan]
    log(json.dumps(dict(event="plan", n=len(plan), due=len(due), remaining_quota=remaining,
                        seconds_left=round(deadline - clock()))))
    if not plan:
        return stats
    results: dict[str, dict] = {}         # ident -> dict(status: ok|failed|deferred, ...)
    state = dict(proc_stop=threading.Event())
    batch_dir = _batch_dir(fetcher)

    def on_lost():
        state["proc_stop"].set()
        with contextlib.suppress(OSError):
            (batch_dir / "STOP").write_text("egress lost")
    wd = Watchdog(prober, float(bb.get("recheck_s", 30)), on_lost=on_lost)
    with wd:
        # ---- 3. script phase
        script_failed: list[dict] = []
        orig_runner = fetcher.browser_runner

        def guarded_runner(job):
            job = dict(job, cancel=state["proc_stop"])
            res = (orig_runner or browser_ego.run_one)(job)
            # accept a download only if the egress is still allowed right after it (probe now)
            if not wd.check_now():         # the egress changed while this paper was running: discard
                for k in ("path", "out_file"):
                    if res and res.get(k):
                        Path(res[k]).unlink(missing_ok=True)
                return dict(status="egress_lost", error="egress left allow_cidrs during the run",
                            out_file=(res or {}).get("out_file"))
            return res
        fetcher.browser_runner = guarded_runner
        try:
            for e in plan:
                k = paper_ident(e)
                if wd.lost.is_set() or clock() >= deadline:
                    break
                slot = led_next_slot(fetcher.ledger, "browser", _browser_keys(e["doi"]))
                if slot + float(bb.get("est_script_s", 60)) > deadline:
                    continue                                    # publisher not free in time: stays queued
                while (w := slot - clock()) > 0 and not wd.lost.is_set():
                    sleep(min(w, 5.0))
                if wd.lost.is_set():
                    break
                fetcher._egress_browser = dict(ok=True, ip=first["ip"], ts=time.time())
                t0 = clock()
                out = fetcher.fetch(e, browser=True, channels=["cache", "browser"])
                if wd.lost.is_set():
                    results[k] = dict(status="deferred", phase="script", errors=["egress lost"])
                    break
                if out.text is not None:
                    results[k] = _deliver(fetcher, e, out, on_success, stats, "script")
                    stats["script_ok"] += 1
                else:
                    blocked = any("challenge" in x or "HTTP 429" in x for x in out.rec.get("errors") or [])
                    tried = bool(out.rec.get("tried"))
                    results[k] = dict(status="failed" if tried else "deferred", phase="script",
                                      errors=out.rec.get("errors", [])[-2:], skipped=out.rec.get("skipped", [])[-2:],
                                      seconds=round(clock() - t0, 1))
                    if tried and not blocked:
                        script_failed.append(dict(e, _landing=out.rec.get("landing")))
                log(json.dumps(dict(event="script", key=k, ok=out.text is not None,
                                    status=results[k]["status"], seconds=round(clock() - t0, 1))))
        finally:
            fetcher.browser_runner = orig_runner
        # ---- 4. agent phase
        if use_agent and script_failed and not wd.lost.is_set():
            left = deadline - clock()
            n_ag = int(max(0.0, left - 20) // max(1.0, float(bb.get("est_agent_s", 120))))
            todo = script_failed[:max(0, n_ag)]
            if todo:
                ar = _agent_phase(fetcher, todo, deadline, batch_dir, wd, state, hermes_runner, log)
                stats["agent"] = {k: v for k, v in ar.items() if k != "papers"}
                for k, v in ar["papers"].items():
                    if v["status"] == "ok":
                        e = next(x for x in todo if paper_ident(x) == k)
                        results[k] = _deliver(fetcher, e, v["outcome"], on_success, stats, "agent")
                        stats["agent_ok"] += 1
                    elif v["status"] == "deferred":
                        results[k] = dict(results.get(k) or {}, status="deferred", phase="agent")
                    else:
                        results[k] = dict(status="failed", phase="agent", errors=v.get("errors", []))
    # close the TaskSpaces this batch used (script, agent, probe)
    if prober.probe_fn is None:
        for sp in {str(b.get("space", "tbsgen-s3-fulltext")), str(ag.get("space", "tbsgen-s3-agent")),
                   prober.cfg["space"]}:
            with contextlib.suppress(Exception):
                browser_ego.finish(dict(b, space=sp))
    stats["egress_lost"] = wd.why if wd.lost.is_set() else None
    stats["probes"] = prober.n
    for e in plan:
        k = paper_ident(e)
        r = results.get(k) or dict(status="deferred", phase="not started")
        stats["papers"].append(dict(key=k, doi=e.get("doi"), **{x: y for x, y in r.items() if x != "outcome"}))
        if r["status"] == "failed":
            stats["failed"] += 1
        elif r["status"] == "deferred":
            stats["deferred"] += 1
    _update_queue(fetcher, plan, results)
    with contextlib.suppress(OSError):
        if not (ft.get("browser") or {}).get("agent", {}).get("keep_batch_dir"):
            shutil.rmtree(batch_dir, ignore_errors=True)
    return stats


def _browser_keys(doi: str) -> tuple:
    from .fulltext import publisher_of
    _, keys, _ = publisher_of(doi)
    lkey = "chrome:" + doi.split("/", 1)[0]
    return (lkey, *[k for k in keys if k != lkey])


def _batch_dir(fetcher) -> Path:
    from .fulltext import queue_path
    q = queue_path(fetcher.cfg)
    d = q.with_name(q.name + ".browser") / time.strftime("batch-%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def _deliver(fetcher, e, out, on_success, stats, phase) -> dict:
    from .fulltext import _subs
    acked = []
    if on_success:
        for s in _subs(e):
            try:
                if on_success(e, out, s):
                    acked.append(s)
            except Exception as ex:  # noqa: BLE001
                stats["errors"].append(fetcher._r(f"reactivate {s[1]}: {ex}"))
    else:
        acked = _subs(e)
    stats["ok"] += 1
    return dict(status="ok", phase=phase, acked=acked, source=out.rec.get("source"))


def _update_queue(fetcher, plan, results):
    """Fetched: drop (or keep unacknowledged subscribers); failed: back off; deferred / not started: unchanged."""
    from .fulltext import _entry_key, _subs, queue_lock, queue_path, read_queue, write_queue
    qp = queue_path(fetcher.cfg)
    backoff = float(fetcher.ft.get("queue_backoff_s", 6 * 3600))
    now = time.time()
    vers = {_entry_key(e): int(e.get("ver") or 0) for e in plan}
    with queue_lock(qp):
        q = read_queue(qp)
        keep = []
        for e in q:
            k = _entry_key(e)
            r = results.get(k)
            if r and r["status"] == "ok":
                rest = [s for s in _subs(e) if s not in r.get("acked", [])]
                if not rest:
                    continue
                e["subscribers"] = rest
                e.pop("item_ids", None)
                e["next_try"] = 0
            elif r and r["status"] == "failed":
                if int(e.get("ver") or 0) == vers.get(k, 0):
                    n_try = int(e.get("browser_tries") or 0) + 1
                    e.update(browser_tries=n_try, browser_last_try=now,
                             browser_next_try=now + backoff * min(8, 2 ** (n_try - 1)))
                else:
                    e.update(browser_last_try=now, browser_next_try=0)
            keep.append(e)
        write_queue(qp, keep)


def _agent_phase(fetcher, todo: list[dict], deadline: float, bd: Path, wd: Watchdog, state: dict, hermes_runner,
                 log) -> dict:
    from .fulltext import Outcome, paper_ident, publisher_of
    ft = fetcher.ft
    b = ft["browser"]
    ag = b.get("agent") or {}
    led = fetcher.ledger
    papers = {}
    for i, e in enumerate(todo, 1):
        papers[str(i)] = dict(doi=e["doi"], title=e.get("title") or "", ident=paper_ident(e),
                              allow=allowed_domains(e["doi"], e.get("_landing"), ag.get("extra_domains") or []))
    ego = browser_ego.binary(b)
    st = dict(space=str(ag.get("space", "tbsgen-s3-agent")), papers=papers, deadline=deadline,
              ledger_path=str(led.path), limits=led.limits, global_limits=led.g, host_caps=led.host_caps,
              ego_binary=ego, gate_max_wait_s=float(ag.get("gate_max_wait_s", 150)),
              max_calls_per_paper=int(ag.get("max_calls_per_paper", 8)),
              call_timeout_s=float(ag.get("call_timeout_s", 150)))
    if deadline - time.time() - 10 < min(30.0, float(ag.get("run_budget_s", 420))):
        return dict(usage=dict(skipped="not enough time left in the egress window"),
                    papers={v["ident"]: dict(status="deferred") for v in papers.values()})
    (bd / "papers").mkdir(exist_ok=True)
    (bd / "files").mkdir(exist_ok=True)
    _atomic(bd / "batch.json", json.dumps(st))
    bin_dir = bd / "bin"
    bin_dir.mkdir(exist_ok=True)
    w = bin_dir / "ego-browser"
    w.write_text(WRAPPER.format(python=sys.executable, pkg_parent=str(Path(__file__).resolve().parent.parent),
                                batch_dir=str(bd)))
    w.chmod(0o700)
    tmpl = Path(ag["prompt_template"]).read_text() if ag.get("prompt_template") else PROMPT.read_text()
    budget = min(float(ag.get("run_budget_s", 420)), deadline - time.time() - 10)
    lines = [dict(n=int(k), doi=v["doi"], title=v["title"], allowed_domains=v["allow"],
                  save_dir=str((bd / "files" / k).resolve())) for k, v in papers.items()]
    prompt = render_prompt(tmpl, papers_json=json.dumps(lines, indent=1, ensure_ascii=False), space=st["space"],
                           minutes=f"{budget / 60:.0f}", gap_s=f"{float(led.limits['browser']['gap_s']):.0f}",
                           gap_max_s=f"{float(led.limits['browser']['gap_s']) + float(led.limits['browser']['jitter_s']):.0f}")
    (bd / "prompt.md").write_text(prompt)
    hermes = ag.get("hermes_bin") or "hermes"
    argv = [hermes, "chat", "--query-file", str(bd / "prompt.md"), "--oneshot", "--format", "stream-json",
            "--source", "tool", "--max-turns", str(int(ag.get("max_turns", 80))), "--run-budget", str(int(budget)),
            "-t", ",".join(ag.get("toolsets") or ["terminal"])]
    if ag.get("provider"):
        argv += ["--provider", str(ag["provider"])]
    if ag.get("model"):
        argv += ["-m", str(ag["model"])]
    for s in ag.get("skills") or []:
        argv += ["-s", str(s)]
    env = dict(browser_ego.child_env())
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    events = dict(tool_calls=0, violations=[])

    def on_event(ev):
        if ev.get("type") == "tool_use":
            events["tool_calls"] += 1
            inp = ev.get("input") or {}
            if ev.get("name") != "terminal":
                why = f"tool not allowed: {ev.get('name')}"
            elif inp.get("background") or inp.get("pty"):
                why = "background / pty terminal not allowed"
            else:
                why = shell_allowed(inp.get("command"))
            if why:
                events["violations"].append(why[:120])
                return f"denied command: {why[:120]}"
        if wd.lost.is_set():
            return "egress lost"
        return None
    t0 = time.time()
    log(json.dumps(dict(event="agent_start", papers=len(papers), budget_s=int(budget),
                        model=ag.get("model"), provider=ag.get("provider"))))
    runner = hermes_runner or default_hermes_runner
    rr = runner(argv, env, budget + float(ag.get("kill_grace_s", 20)), on_event, state["proc_stop"])
    res = rr.get("result") or {}
    usage = dict(tokens=res.get("tokens"), duration_ms=res.get("duration_ms"), session_id=res.get("session_id"),
                 exit=rr.get("exit"), killed=rr.get("killed"), tool_calls=events["tool_calls"],
                 violations=events["violations"], seconds=round(time.time() - t0, 1),
                 provider=ag.get("provider"), model=ag.get("model"))
    with contextlib.suppress(OSError):
        with open(bd.parent / "usage.jsonl", "a") as fh:
            fh.write(json.dumps(dict(usage, ts=time.strftime("%F %T"), papers=len(papers))) + "\n")
    log(json.dumps(dict(event="agent_end", **{k: v for k, v in usage.items() if k != "session_id"})))
    (bd / "STOP").write_text("agent run ended")       # a late wrapper call cannot book anything any more
    # ---- validate and book every paper (the agent's own report is not used)
    out: dict = dict(usage=usage, papers={})
    lost = not wd.check_now()                         # files saved while the egress was not allowed are dropped
    for k, p in papers.items():
        sp = bd / "papers" / f"{k}.json"
        ps = json.loads(sp.read_text()) if sp.exists() else {}
        c = next(e for e in todo if paper_ident(e) == p["ident"])
        rid = ps.get("rid")
        if rid is None:
            out["papers"][p["ident"]] = dict(status="deferred" if (lost or rr.get("killed")) else "failed",
                                             errors=[ps.get("closed") or "agent did not open this paper"])
            continue
        rec = dict(doi=p["doi"], key=p["ident"], tried=[f"browser:agent:{ps['lkey']}"], errors=[], skipped=[],
                   source=None)
        txt, size, why = None, 0, "no file saved"
        if lost:
            why = "egress left allow_cidrs during the agent run; files discarded"
        else:
            for f in sorted((bd / "files" / k).glob("*")) if (bd / "files" / k).exists() else []:
                if not f.is_file():
                    continue
                size = f.stat().st_size
                if size > fetcher._max_bytes():
                    why = f"too large ({size} bytes)"
                    continue
                body = f.read_bytes()
                kind = "pdf" if body[:5] == b"%PDF-" else "any"
                t, why2, tm = fetcher._validate(p["title"], kind, body)
                if t is None:
                    why = why2
                    continue
                fetcher._tl.c = c
                prov = fetcher._save(c, "browser", "agent:" + (p["allow"][0] if p["allow"] else ""), body, t, tm,
                                     "pdf" if kind == "pdf" else "html", via="agent")
                rec.update(source="browser", url=prov.get("url"), title_match=prov.get("title_match"),
                           provenance=prov)
                txt = t
                break
        _, keys, web_hosts = publisher_of(p["doi"])
        if ps.get("challenge"):
            secs = float(b.get("challenge_cooldown_s", 24 * 3600))
            for h in {ps["lkey"], *keys, *web_hosts}:
                led.block(h, "browser agent: challenge page", seconds=secs,
                          channel="browser" if h == ps["lkey"] else None)
            why = "challenge page (publisher cooled down)"
        ok = txt is not None
        led.record("browser", ps["lkey"], p["doi"], "ok" if ok else "fail", None, size,
                   "" if ok else fetcher._r(f"agent: {why}")[:300], reservation=rid)
        if ok:
            out["papers"][p["ident"]] = dict(status="ok", outcome=Outcome(txt, rec))
        else:
            out["papers"][p["ident"]] = dict(status="failed", errors=[fetcher._r(f"agent: {why}")])
    return out
