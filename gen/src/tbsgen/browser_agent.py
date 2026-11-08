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
     provider / model) for the papers the script could not get, with a prompt generated from the reviewed
     template browser_agent_prompt.md.  The agent writes no browser code: its only tool is `tbsgen-browser`
     (look / open URL / click I / save_pdf / finish), each call running the reviewed action script
     browser_agent_action.mjs, which books the paper in the ledger before its first call, only navigates to
     the paper's publisher domains, never types, never interacts with verification pages or widgets, and
     cools the publisher at once on a bot wall.  The Hermes run uses an isolated Hermes home whose
     pre_tool_call hook blocks every other command before it runs.  The agent's claims are not trusted:
     after the run, code validates every saved file (PDF / text length / title match) and books it.
  5. a watchdog re-probes the egress every `batch.recheck_s` during the batch; when it leaves allow_cidrs the
     batch stops at once (the running script / agent process is killed, a result obtained meanwhile is
     discarded) and every paper not finished stays in the queue, still due.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import string
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from typing import Callable

from . import browser_ego
from .dlledger import LimitReached

PROMPT = Path(__file__).with_name("browser_agent_prompt.md")
REFUSE = "browser egress is outside fulltext.egress_check.allow_cidrs; open the allowed egress first"


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
    last_ok: dict | None = None

    def __post_init__(self):
        b = self.ft.get("browser") or {}
        bb = b.get("batch") or {}
        ec = self.ft.get("egress_check") or {}
        self.urls = list(bb.get("watch_urls") or ec.get("urls") or [])
        self.poll_s = max(5.0, float(bb.get("poll_s", 5)))
        self.host_gap = float(bb.get("probe_host_gap_s", 15))
        self.lock = threading.Lock()
        self.cfg = dict(b, space=str(b.get("space", "tbsgen-s3-fulltext")) + "-probe")   # derived: same lock

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
            self.last_ok = dict(ok=True, ip=r["ip"], via=r["via"])
            return self.last_ok
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

    def _lose(self, r: dict):
        self.why = r.get("why") or "egress lost"
        self.lost.set()
        if self.on_lost:
            with contextlib.suppress(Exception):
                self.on_lost()

    def _run(self):
        while not self.stop_ev.wait(self.recheck_s):
            r = self.prober.probe()
            if not r["ok"]:
                self._lose(r)
                return

    def check_now(self) -> bool:
        if self.lost.is_set():
            return False
        r = self.prober.probe()
        if not r["ok"]:
            self._lose(r)
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


# ------------------------------------------------------------------------------------------- agent tools
# The agent never runs browser code of its own.  Its only browser tool is `tbsgen-browser` (a generated
# executable first on PATH): `tbsgen-browser N <action> [arg]` and `tbsgen-browser finish`.  Each call runs
# the reviewed action script browser_agent_action.mjs with one injected INPUT line.  The Hermes run itself
# uses an isolated Hermes home whose pre_tool_call hook (hook_main) allows only `tbsgen-browser ...` and
# `sleep N` terminal commands (and vision_analyze for screenshots inside the batch's files dir) *before*
# they run; anything else is blocked (and the stream watcher kills the run).
ACTION = Path(__file__).with_name("browser_agent_action.mjs")
TOOL = r"""#!{python}
# tbsgen browser batch tool (generated; removed with the batch directory)
import sys
sys.path.insert(0, {pkg_parent!r})
from tbsgen.browser_agent import {fn}
sys.exit({fn}({batch_dir!r}, sys.argv[1:]))
"""
# One-token shapes (no quotes, shell metachars or newlines ever reach a shell: the hook matches the whole
# command against these patterns and the wrapper re-parses argv).
_SAFE_TOK = r"[^\s'\"`$;&|<>\\]{1,200}"
_SAFE_TXT = r"[A-Za-z0-9 ,.\-'()_\u4e00-\u9fff]{1,60}"
_PAPER_ACTIONS = "|".join((
    r"look( full)?( @\d{1,5})?( \d{1,4})?",
    r"open https?://" + _SAFE_TOK,
    r"click \d{1,3}",
    r"click_xy \d{1,4} \d{1,4}",
    r"hover \d{1,3}",
    r"scroll (up|down)( \d{1,4})?",
    r"select \d{1,3} " + _SAFE_TXT,
    r"type \d{1,3} " + _SAFE_TXT,
    "screenshot", "tabs",
    r"switch [A-Za-z0-9_.-]{1,24}",
    "back",
    r"wait_for (url|sel) " + _SAFE_TOK + r"( \d{1,3})?",
    "save_pdf", "sso_login", "handoff_status",
))
_TOOL_CMD = re.compile(rf"^tbsgen-browser (finish|[1-9]\d? ({_PAPER_ACTIONS}))$")
_SLEEP_CMD = re.compile(r"^sleep ([1-9]\d?|[12]\d\d|300)$")
DENY_TEXT = re.compile(r"sci-?hub|libgen|z-lib|annas-archive", re.I)
# actions that can load a page of the publisher (paced per publisher and capped per paper)
NAV_ACTIONS = frozenset({"open", "click", "click_xy", "back", "sso_login"})
VISION_TOOL = "vision_analyze"


def shell_allowed(cmd) -> str | None:
    """None if the agent's terminal command is allowed, else the reason (used by the pre-execution hook and,
    as a second line, by the stream watcher)."""
    if not isinstance(cmd, str) or not cmd.strip():
        return "empty command"
    c = cmd.strip()
    if DENY_TEXT.search(c):
        return "unauthorized source"
    if "\n" in c or not (_TOOL_CMD.match(c) or _SLEEP_CMD.match(c)):
        return f"command not allowed (only `tbsgen-browser ...` and `sleep N`): {c[:60]}"
    return None


def _vision_allowed(batch_dir: str, inp: dict) -> str | None:
    """None if a vision_analyze call may run: only images the batch itself saved (screenshots)."""
    url = inp.get("image_url")
    if not isinstance(url, str) or not url:
        return "vision: image_url is required"
    try:
        root = (Path(batch_dir) / "files").resolve()
        p = Path(url).resolve()
        p.relative_to(root)
    except (OSError, ValueError):
        return "vision: only screenshots inside this batch's files directory may be viewed"
    return None


def hook_main(batch_dir: str, argv: list[str]) -> int:
    """Hermes pre_tool_call hook: stdin {tool_name, tool_input}; prints a block decision unless allowed."""
    try:
        d = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        d = {}
    name, inp = d.get("tool_name"), d.get("tool_input") or {}
    if name == VISION_TOOL:
        why = _vision_allowed(batch_dir, inp)
    elif name != "terminal":
        why = f"tool {name} is not available in this batch"
    elif inp.get("background") or inp.get("pty"):
        why = "background / pty terminals are not available"
    else:
        why = shell_allowed(inp.get("command"))
    if why:
        with contextlib.suppress(OSError):
            with open(Path(batch_dir) / "violations.jsonl", "a") as fh:
                fh.write(json.dumps(dict(ts=time.time(), tool=name, why=why[:200])) + "\n")
        print(json.dumps({"action": "block", "message": f"tbsgen: {why}"}))
    return 0


def _cool_publisher(led, ps: dict, doi: str, why: str, secs: float):
    from .fulltext import publisher_of
    _, keys, web_hosts = publisher_of(doi)
    for h in {ps["lkey"], *keys, *web_hosts}:
        led.block(h, f"browser agent: {why}"[:200], seconds=secs, channel="browser" if h == ps["lkey"] else None)


def _read_state(bd: Path) -> dict:
    """Batch-shared wrapper state: navigation pacing per publisher, one hand-off per publisher, SSO budget,
    and the TaskSpaces the batch created (one per paper, closed by `finish`)."""
    try:
        return json.loads((bd / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def _write_state(bd: Path, state: dict):
    _atomic(bd / "state.json", json.dumps(state))


def _notify_user(st: dict, msg: str):
    """A local macOS notification (and an optional local Bark command, off by default; never group chats)."""
    if not st.get("handoff_notify", True):
        return
    with contextlib.suppress(Exception):
        subprocess.run(["osascript", "-e",
                        f'display notification {json.dumps(msg[:200], ensure_ascii=False)} with title '
                        f'{json.dumps("tbsgen 浏览器批次", ensure_ascii=False)}'],
                       timeout=10, check=False)
    for cmd in st.get("bark_cmd") or []:
        with contextlib.suppress(Exception):
            subprocess.run(list(cmd) + [msg[:200]], timeout=10, check=False)


def _run_script(st: dict, inp: dict, timeout: float, bd: Path) -> tuple[dict, str]:
    """Run the reviewed action script with one injected INPUT line; returns (result, stderr-tail).

    The script (~40 KB) is passed as an argument, not on a stdin pipe: `communicate(input, timeout)` stops
    writing stdin after its first TimeoutExpired, so a child slower than that first timeout to drain the
    pipe never saw EOF and the call hung until its timeout (ego-browser also drops the output of a stdin
    script whose TaskSpace is with the user)."""
    script = "const INPUT = " + json.dumps(inp) + ";\n" + ACTION.read_text()
    proc = subprocess.Popen([st["ego_binary"], "nodejs", script], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=browser_ego.child_env())
    t_end = time.time() + max(5.0, timeout)
    so = se = ""
    while True:
        try:
            so, se = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if (bd / "STOP").exists() or time.time() > t_end:
                proc.kill()
                proc.communicate()
                return {}, "stopped"
    res: dict = {}
    for line in reversed((so + "\n" + se).splitlines()):
        if line.startswith("TBSGEN_RESULT "):
            with contextlib.suppress(ValueError):
                res = json.loads(line[len("TBSGEN_RESULT "):])
            break
    return res, (se or so)[-200:]


def _finish_spaces(bd: Path, st: dict, spaces: list[str]) -> list[str]:
    """Close the batch's TaskSpaces (the action script's `finish` takes the space back from the user first
    when a hand-off is still open, then closes it)."""
    done = []
    for sp_name in spaces:
        r, _ = _run_script(st, dict(space=sp_name, action="finish", files_dir=str(bd / "files")), 90.0, bd)
        done.append(f"{sp_name}:{r.get('status', 'error')}")
    return done


def wrapper_main(batch_dir: str, argv: list[str]) -> int:
    """`tbsgen-browser N <action> [arg...]` / `tbsgen-browser finish`."""
    from .dlledger import COOLDOWN_429_S, DownloadLedger
    from .fulltext import publisher_of
    bd = Path(batch_dir)
    st = json.loads((bd / "batch.json").read_text())

    def say(msg, rc):
        print(msg)
        return rc
    if (bd / "STOP").exists():
        return say("tbsgen: STOP - the batch was stopped (egress changed or deadline). End now.", 3)
    if argv[:1] == ["finish"]:
        state = _read_state(bd)
        spaces = sorted(set(state.get("spaces") or []) | {st["space"]})
        done = _finish_spaces(bd, st, spaces)
        return say(f"tbsgen: TaskSpaces closed ({', '.join(done)})", 0)
    usage = ("usage: tbsgen-browser N look [full] [@REF] [OFFSET]|open URL|click I|click_xy X Y|hover I|"
             "scroll up|down [PX]|select I VALUE|type I TEXT|screenshot|tabs|switch LABEL|back|"
             "wait_for url SUB|sel CSS [S]|save_pdf|sso_login|handoff_status  |  tbsgen-browser finish")
    if len(argv) < 2 or argv[0] not in st["papers"] or not _TOOL_CMD.match("tbsgen-browser " + " ".join(argv)):
        return say(usage, 2)
    n, action, arg = argv[0], argv[1], " ".join(argv[2:])
    p = st["papers"][n]
    sp = bd / "papers" / f"{n}.json"
    ps: dict = json.loads(sp.read_text()) if sp.exists() else dict(calls=0)
    if ps.get("closed"):
        return say(f"tbsgen: paper {n} is closed ({ps['closed']}); move on", 2)
    if action != "handoff_status" and ps.get("calls", 0) >= int(st["max_calls_per_paper"]):
        return say(f"tbsgen: paper {n}: call limit reached; give up on this paper", 2)
    if time.time() >= float(st["deadline"]):
        return say("tbsgen: the batch deadline has passed; end now.", 3)
    led = DownloadLedger(st["ledger_path"], st["limits"], st["global_limits"], st["host_caps"])
    _, keys, web_hosts = publisher_of(p["doi"])
    lkey = "chrome:" + p["doi"].split("/", 1)[0]
    aliases = tuple(k for k in keys if k != lkey)
    state = _read_state(bd)
    space = f"{st['space']}-p{n}"                      # one TaskSpace per paper (hand-offs pause one paper only)
    state.setdefault("spaces", [])
    if space not in state["spaces"]:
        state["spaces"].append(space)
    _write_state(bd, state)
    if "rid" not in ps:
        # first call for this paper: book its slot (pacing / caps / cooldowns of the publisher)
        try:
            slot = led_next_slot(led, "browser", (lkey, *aliases))
            if slot - time.time() > float(st["gate_max_wait_s"]):
                return say(f"tbsgen: the publisher of paper {n} is paced; its slot opens in "
                           f"{slot - time.time():.0f}s. Work on another paper, or `sleep` and retry.", 4)
            if slot > float(st["deadline"]):
                return say("tbsgen: the batch deadline comes before this publisher's next slot; skip it.", 4)
            rid = led.acquire("browser", lkey, p["doi"], related=tuple(web_hosts), group=aliases)
        except LimitReached as e:
            ps["closed"] = f"ledger: {e}"
            _atomic(sp, json.dumps(ps))
            return say(f"tbsgen: paper {n} refused by the download ledger ({e}); move on", 2)
        ps.update(rid=rid, lkey=lkey, t0=time.time())
    elif (msg := led.cooling(lkey, *aliases, *web_hosts)):
        ps["closed"] = "publisher in cooldown"
        _atomic(sp, json.dumps(ps))
        return say(f"tbsgen: paper {n}: {msg[:120]}; move on", 2)
    # a hand-off that is still open: only handoff_status (or its expiry) may act on the paper
    handoff_deadline = float(ps.get("handoff_deadline") or 0)
    if ps.get("handoff"):
        if action != "handoff_status":
            return say("tbsgen: this paper is with the user (hand-off). Poll `tbsgen-browser N handoff_status` "
                       "and work on other papers meanwhile.", 2)
        if time.time() > handoff_deadline:
            reason = str(ps.get("handoff_reason") or "")
            ps.pop("handoff", None)
            if "login" in reason:
                ps["closed"] = "login_required (hand-off timed out)"
            else:
                ps["closed"] = "challenge (hand-off timed out)"
                _cool_publisher(led, ps, p["doi"], "hand-off timed out", float(st["challenge_cooldown_s"]))
            _atomic(sp, json.dumps(ps))
            return say(f"tbsgen: paper {n}: hand-off timed out - paper closed ({ps['closed']}).", 2)
    if action != "handoff_status":
        ps["calls"] = ps.get("calls", 0) + 1
    if action == "type":
        value = " ".join(arg.split(" ")[1:])
        if not value or value not in (st.get("type_allow") or []):
            return say(f"tbsgen: paper {n}: only the configured institution names may be typed", 2)
    if action in NAV_ACTIONS:
        # pace the agent's page loads per publisher and cap them per paper
        if ps.get("navs", 0) >= int(st["max_navigations_per_paper"]):
            _atomic(sp, json.dumps(ps))
            return say(f"tbsgen: paper {n}: navigation limit reached; give up on this paper", 2)
        state = _read_state(bd)
        last = float((state.get("nav") or {}).get(lkey) or 0)
        gap = float(st["nav_gap_s"]) - (time.time() - last)
        if gap > 0:
            if gap > float(st["deadline"]) - time.time():
                _atomic(sp, json.dumps(ps))
                return say("tbsgen: the batch deadline comes before the next navigation; end now.", 3)
            time.sleep(gap)
        state.setdefault("nav", {})[lkey] = time.time()
        _write_state(bd, state)
        ps["navs"] = ps.get("navs", 0) + 1
    _atomic(sp, json.dumps(ps))
    files = (bd / "files" / n).resolve()
    files.mkdir(parents=True, exist_ok=True)
    handoff_free = (bool(st.get("handoff_enabled")) and not ps.get("handoff")
                    and not (state.get("handoff") or {}).get(lkey))
    sso_st = state.get("sso") or {}
    sso_ok = (bool(st.get("sso_login_enabled")) and (st.get("sso_domains") or [])
              and int(sso_st.get("submissions") or 0) < int(st.get("sso_max_submissions", 2))
              and not sso_st.get("failed") and not sso_st.get("logged_in"))
    inp = dict(space=space, action=action, arg=arg, allow=p["allow"], files_dir=str(files),
               look_url=ps.get("look_url") if action in ("click", "hover", "select", "type") else None,
               look_scope=ps.get("look_scope") or "", page_label=ps.get("page_label") or "",
               nav_timeout_ms=45000, download_timeout_ms=120000, download_wait_ms=6000,
               challenge_wait_ms=int(st.get("challenge_wait_ms", 40000)),
               handoff={"enabled": handoff_free}, sso={"login": sso_ok, "domains": st.get("sso_domains") or []},
               type_allow=st.get("type_allow") or [],
               handoff_state={"active": bool(ps.get("handoff")),
                              "deadline_ms": int(handoff_deadline * 1000) if ps.get("handoff") else 0})
    timeout = max(5.0, min(float(st["call_timeout_s"]), float(st["deadline"]) - time.time()))
    res, _err = _run_script(st, inp, timeout, bd)
    if not res:
        return say("tbsgen: stopped (egress changed, deadline or call timeout). End now.", 3)

    def save_ps(**kw):
        cur = json.loads(sp.read_text())
        cur.update(kw)
        _atomic(sp, json.dumps(cur))
        return cur

    if res.get("status") == "paused":
        return say("tbsgen: this paper's browser workspace is with the user (hand-off). Poll "
                   "`tbsgen-browser N handoff_status` and work on other papers meanwhile.", 2)
    if res.get("handoff") and res.get("status") == "handoff":
        # the tool handed the page to the user (verification / login): notify, mark, keep the paper open
        state = _read_state(bd)
        state.setdefault("handoff", {})[lkey] = True
        _write_state(bd, state)
        ho = res.get("handoff") or {}
        save_ps(handoff=True, handoff_used=True, handoff_reason=str(ho.get("reason") or "")[:80],
                handoff_deadline=min(time.time() + float(st["handoff_wait_s"]), float(st["deadline"])),
                handoff_note=f"handoff: {str(ho.get('reason'))[:60]} host={ho.get('host', '')}".strip())
        kind = "验证" if "challenge" in str(ho.get("reason") or "") else "登录"
        _notify_user(st, f"请在 ego lite 里完成 {ho.get('host') or '出版商'} 的{kind}（论文 {n}）")
        return say("tbsgen: handed the page to the user (the notification explains what is needed). Work on "
                   "other papers now and poll `tbsgen-browser N handoff_status`.", 0)
    if res.get("blocked"):
        # a verification page / bot wall / 429: cool the publisher now, close the paper (never interact)
        secs = COOLDOWN_429_S if res.get("http429") else float(st.get("challenge_cooldown_s", 24 * 3600))
        ps = save_ps(closed="challenge", challenge=str(res["blocked"])[:80])
        _cool_publisher(led, ps, p["doi"], str(res["blocked"]), secs)
        return say(f"tbsgen: paper {n}: {res['blocked']} - publisher cooled down; report `challenge` and move on.", 2)
    for rj in res.get("rejected") or []:              # never validated, whatever the script managed to delete
        with contextlib.suppress(OSError):
            Path(rj).unlink()
    acc = [x for x in res.get("files") or [] if Path(x).resolve().parent == files]
    if acc:
        ps = save_ps()
        ps["accepted"] = sorted(set(ps.get("accepted") or []) | set(acc))
        _atomic(sp, json.dumps(ps))
    if action == "handoff_status":
        chk = res.get("handoff_check")
        if chk == "cleared":
            ps = save_ps()
            ps.pop("handoff", None)
            _atomic(sp, json.dumps(ps))
            return say("tbsgen: the user cleared the page; continue with this paper (run `look`).", 0)
        if chk == "waiting":
            return say(f"tbsgen: still waiting for the user ({res.get('handoff_left_s')}s left); do other work "
                       "and poll again.", 0)
        if chk == "expired":
            reason = str(ps.get("handoff_reason") or "")
            ps = save_ps()
            ps.pop("handoff", None)
            if "login" in reason:
                ps["closed"] = "login_required (hand-off timed out)"
            else:
                ps["closed"] = "challenge (hand-off timed out)"
                _cool_publisher(led, ps, p["doi"], "hand-off timed out", float(st["challenge_cooldown_s"]))
            _atomic(sp, json.dumps(ps))
            return say(f"tbsgen: paper {n}: hand-off timed out - paper closed ({ps['closed']}).", 2)
        return say("tbsgen: no hand-off in progress for this paper.", 0)
    if res.get("sso") in ("ok", "fail"):
        state = _read_state(bd)
        s = state.get("sso") or {}
        s["submissions"] = int(s.get("submissions") or 0) + 1
        host = ""
        with contextlib.suppress(ValueError):
            host = urlparse((res.get("page") or {}).get("url") or "").hostname or ""
        s["failed" if res["sso"] == "fail" else "logged_in"] = True
        state["sso"] = s
        _write_state(bd, state)
        note = f"sso_login host={host} result={res['sso']}"
        if res["sso"] == "ok":
            save_ps(sso_note=note)
        else:
            save_ps(sso_note=note, closed="login_failed (sso, not retried)")
            return say("tbsgen: paper {n}: SSO login failed - no retry; report `login_required` and move on.", 2)
    if res.get("shot"):
        ps = save_ps()
        ps["shots"] = sorted(set(ps.get("shots") or []) | {res["shot"]})   # viewable, never the paper's text
        _atomic(sp, json.dumps(ps))
    if res.get("page_label"):
        ps = save_ps()
        ps["page_label"] = res["page_label"]
        _atomic(sp, json.dumps(ps))
    if res.get("look_scope") is not None:
        ps = save_ps()
        ps["look_scope"] = res["look_scope"]
        _atomic(sp, json.dumps(ps))
    if action == "look" and res.get("look_url"):
        ps = save_ps()
        ps["look_url"] = res["look_url"]
        _atomic(sp, json.dumps(ps))
    show = dict(status=res.get("status"), error=res.get("error"), files_saved=len(res.get("files") or []),
                page=res.get("page"), note=res.get("note"), left_allowed=res.get("left_allowed"),
                shot=res.get("shot"), tabs=res.get("tabs"), acted=res.get("acted"), sso=res.get("sso"),
                waited=res.get("waited"), scroll=res.get("scroll"), page_label=res.get("page_label"),
                handoff_check=res.get("handoff_check"), handoff_left_s=res.get("handoff_left_s"))
    return say(json.dumps({k: v for k, v in show.items() if v is not None}, ensure_ascii=False)[:9000],
               0 if res.get("status") == "ok" else 1)


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


STDERR_KEEP = 64 * 1024           # bytes of the agent's stderr kept in the batch directory (the tail)
STDERR_LOG_TAIL = 1500            # characters of it repeated in the agent_end log event
_SECRETISH = re.compile(r"(?i)(bearer\s+|authorization:\s*|(?:api[_-]?key|token|secret|password|passwd|pwd)"
                        r"[\"']?\s*[:=]\s*[\"']?)[^\s\"',;]+|\b(?:sk|ghp|gho|ghu|github_pat|xox[bp])[-_][A-Za-z0-9_\-]{8,}")


def redact(text: str) -> str:
    """Mask anything that looks like a credential (stderr of the agent process is logged)."""
    return _SECRETISH.sub(lambda m: (m.group(1) or "") + "[redacted]", text)


def _stderr_tail(path: Path | None, keep: int = STDERR_KEEP) -> str:
    """Truncate the stderr file to its last `keep` bytes, redact it in place; return the redacted tail."""
    if path is None:
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - keep))
            data = fh.read()
    except OSError:
        return ""
    text = redact(data.decode("utf-8", "replace"))
    with contextlib.suppress(OSError):
        _atomic(path, text)
        path.chmod(0o600)
    return text


def default_hermes_runner(argv: list[str], env: dict, timeout: float, on_event, stop: threading.Event,
                          stderr_path: Path | None = None) -> dict:
    """Run `hermes chat ... --format stream-json`; call on_event(ev) per JSON line (it returns a reason string
    to kill the run).  stderr goes to `stderr_path` (truncated to its tail and redacted afterwards).  Returns
    dict(exit, killed, result_event, stderr_tail)."""
    err = open(stderr_path, "wb") if stderr_path is not None else subprocess.DEVNULL   # noqa: SIM115
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL,
                             env=env, text=True, start_new_session=True)
    finally:
        if err is not subprocess.DEVNULL:
            err.close()                       # the child holds its own copy of the descriptor
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
    return dict(exit=p.returncode, killed=killed, result=result, stderr_tail=_stderr_tail(stderr_path))


def run_batch(fetcher, *, wait_for_egress: bool = False, max_wait_s: float | None = None, on_success=None,
              log=print, prober: Prober | None = None, hermes_runner=None, clock=time.time,
              sleep=time.sleep, now_window_start: float | None = None, dry_run: bool = False) -> dict:
    """One egress-gated browser batch.  Returns stats; stats['refused'] when the egress was outside the
    allowed ranges (without --wait-for-egress).  dry_run: plan only (no probe, no browser, no ledger)."""
    from .fulltext import paper_ident
    ft = fetcher.ft
    b = ft.get("browser") or {}
    bb = b.get("batch") or {}
    ag = b.get("agent") or {}
    stats: dict = dict(mode="browser-batch", ok=0, failed=0, script_ok=0, agent_ok=0, probes=0, deferred=0,
                       papers=[], errors=[])
    if dry_run:
        due = _due(fetcher, bb)
        plan = plan_batch(due, float(bb.get("window_s", 600)) - float(bb.get("margin_s", 60)), bb,
                          bool(ag.get("enabled")), fetcher.ledger.remaining("browser"))
        stats.update(dry_run=True, due=len(due), planned=[paper_ident(e) for e in plan])
        return stats
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
    took_lock = False
    if not fetcher._space_held:
        if not fetcher._try_space_lock():
            stats["note"] = "browser TaskSpace is in use by another process"
            return stats
        took_lock = True
    used: set[str] = set()                # TaskSpaces this batch actually used (all derived from the locked one)
    stats["_used_spaces"] = used
    try:
        return _run_batch(fetcher, stats, b, bb, ag, prober, wait_for_egress, max_wait_s, on_success, log,
                          hermes_runner, clock, sleep, now_window_start)
    finally:
        stats.pop("_used_spaces", None)
        if prober.n and prober.probe_fn is None:
            used.add(prober.cfg["space"])
        if fetcher.browser_runner is None:
            for sp in sorted(used):
                with contextlib.suppress(Exception):
                    browser_ego.finish(dict(b, space=sp))
        if took_lock:
            _release_space_lock(fetcher)


def _release_space_lock(fetcher):
    import fcntl
    fh = getattr(fetcher, "_space_fh", None)
    if fh is not None:
        with contextlib.suppress(OSError):
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
    fetcher._space_fh = None
    fetcher._space_held = False


def agent_space(b: dict) -> str:
    """The agent's TaskSpace, always derived from the (locked) script TaskSpace: two batches can only share it if
    they share the script space, and then the TaskSpace lock serializes them."""
    return f"{b.get('space', 'tbsgen-s3-fulltext')}-agent"


def _due(fetcher, bb: dict) -> list[dict]:
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
    return [e for e in due if not fetcher.cached(e)]          # cached papers are delivered by a normal drain


def _run_batch(fetcher, stats, b, bb, ag, prober, wait_for_egress, max_wait_s, on_success, log, hermes_runner,
               clock, sleep, now_window_start) -> dict:
    from .fulltext import paper_ident
    hint = (" " + bb["refuse_hint"]) if bb.get("refuse_hint") else ""
    # ---- 1. egress
    t_wait0 = clock()
    limit = float(max_wait_s if max_wait_s is not None else bb.get("max_wait_s", 1800))
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
    # ---- 2. plan
    due = _due(fetcher, bb)
    remaining = fetcher.ledger.remaining("browser")
    use_agent = bool(ag.get("enabled"))
    plan = plan_batch(due, deadline - clock(), bb, use_agent, remaining)
    stats["planned"] = [paper_ident(e) for e in plan]
    log(json.dumps(dict(event="plan", n=len(plan), due=len(due), remaining_quota=remaining,
                        seconds_left=round(deadline - clock()))))
    if not plan:
        return stats
    results: dict[str, dict] = {}         # ident -> dict(status: ok|failed|deferred, ...)
    stop = threading.Event()              # egress lost or deadline: cancels running scripts / the agent
    stop.used = stats.get("_used_spaces")  # type: ignore[attr-defined]
    batch_dir = _batch_dir(fetcher)

    def on_lost():
        stop.set()
        with contextlib.suppress(OSError):
            (batch_dir / "STOP").write_text("egress lost")
    wd = Watchdog(prober, float(bb.get("recheck_s", 30)), on_lost=on_lost)

    def deadline_guard():                 # the window ends: stop whatever runs
        while not stop.wait(1.0):
            if clock() >= deadline:
                stop.set()
                with contextlib.suppress(OSError):
                    (batch_dir / "STOP").write_text("deadline")
                return
    dg = threading.Thread(target=deadline_guard, daemon=True)
    try:
        with wd:
            dg.start()
            _script_phase(fetcher, plan, results, stats, bb, wd, stop, deadline, clock, sleep, on_success, log)
            script_failed = [dict(e, _landing=results[paper_ident(e)].get("landing")) for e in plan
                             if results.get(paper_ident(e), {}).get("retry_with_agent")]
            if use_agent and script_failed and not stop.is_set():
                n_ag = int(max(0.0, deadline - clock() - 20) // max(1.0, float(bb.get("est_agent_s", 120))))
                todo = script_failed[:max(0, n_ag)]
                if todo:
                    ar = _agent_phase(fetcher, todo, deadline, batch_dir, wd, stop, hermes_runner, log)
                    stats["agent"] = {k: v for k, v in ar.items() if k != "papers"}
                    for k, v in ar["papers"].items():
                        if v["status"] == "ok":
                            e = next(x for x in todo if paper_ident(x) == k)
                            results[k] = _deliver(fetcher, e, v["outcome"], on_success, stats, "agent")
                            stats["agent_ok"] += 1
                        else:
                            results[k] = dict(status=v["status"], phase="agent", errors=v.get("errors", []))
    finally:
        stop.set()
        wd.stop_ev.set()
        if wd.th.is_alive():
            wd.th.join(timeout=60)
        if dg.is_alive():
            dg.join(timeout=5)
        stats["egress_lost"] = wd.why if wd.lost.is_set() else None
        stats["probes"] = prober.n
        for e in plan:
            k = paper_ident(e)
            r = results.get(k) or dict(status="deferred", phase="not started")
            stats["papers"].append(dict(key=k, doi=e.get("doi"),
                                        **{x: y for x, y in r.items() if x not in ("outcome", "retry_with_agent")}))
            if r["status"] == "failed":
                stats["failed"] += 1
            elif r["status"] == "deferred":
                stats["deferred"] += 1
        _update_queue(fetcher, plan, results)
        if not ag.get("keep_batch_dir"):
            shutil.rmtree(batch_dir, ignore_errors=True)
    return stats


def _script_phase(fetcher, plan, results, stats, bb, wd, stop, deadline, clock, sleep, on_success, log):
    from .fulltext import paper_ident
    orig_runner = fetcher.browser_runner

    def guarded_runner(job):
        job = dict(job, cancel=stop)
        if job.get("mode") != "egress":
            left = deadline - clock()
            if stop.is_set() or left < float(bb.get("min_script_s", 20)):
                # the window is (almost) over: never start a script that cannot finish inside it
                stop.set()
                return dict(status="cancelled", error="batch stopped (egress changed or deadline)")
            job["cfg"] = dict(job["cfg"], script_timeout_s=min(float(job["cfg"].get("script_timeout_s", 420)), left))
            used = getattr(stop, "used", None)
            if used is not None:
                used.add(str(job["cfg"].get("space", "tbsgen-s3-fulltext")))
        res = (orig_runner or browser_ego.run_one)(job)
        if job.get("mode") == "egress":
            return res
        # accept a download only if the egress is still allowed right after it (probe now)
        if stop.is_set() or not wd.check_now():
            for k in ("path", "out_file"):
                if res and res.get(k):
                    Path(res[k]).unlink(missing_ok=True)
            return dict(status="cancelled", error="batch stopped (egress changed or deadline)",
                        out_file=(res or {}).get("out_file"))
        return res
    fetcher.browser_runner = guarded_runner
    led = fetcher.ledger
    orig_sleep = led.sleep

    def stop_aware_sleep(sec):            # a ledger wait never outlasts the batch (the runner then refuses)
        end = clock() + sec
        while not stop.is_set() and (w := end - clock()) > 0:
            orig_sleep(min(w, 1.0))
    led.sleep = stop_aware_sleep
    try:
        for e in plan:
            k = paper_ident(e)
            if stop.is_set() or clock() >= deadline:
                break
            slot = led_next_slot(fetcher.ledger, "browser", _browser_keys(e["doi"]))
            if slot + float(bb.get("est_script_s", 60)) > deadline:
                continue                                    # publisher not free in time: stays queued
            while (w := slot - clock()) > 0 and not stop.is_set():
                sleep(min(w, 2.0))
            if stop.is_set():
                break
            fetcher._egress_browser = dict(ok=True, ip=(wd.prober.last_ok or {}).get("ip"), ts=time.time())
            t0 = clock()
            out = fetcher.fetch(e, browser=True, channels=["cache", "browser"])
            if stop.is_set() and out.text is None:
                results[k] = dict(status="deferred", phase="script", errors=["batch stopped"])
                _uncount_cancelled(fetcher, e)
                break
            if out.text is not None:
                results[k] = _deliver(fetcher, e, out, on_success, stats, "script")
                stats["script_ok"] += 1
            else:
                blocked = any("challenge" in x or "HTTP 429" in x for x in out.rec.get("errors") or [])
                tried = bool(out.rec.get("tried"))
                results[k] = dict(status="failed" if tried else "deferred", phase="script",
                                  errors=out.rec.get("errors", [])[-2:], skipped=out.rec.get("skipped", [])[-2:],
                                  seconds=round(clock() - t0, 1), landing=out.rec.get("landing"),
                                  retry_with_agent=tried and not blocked)
            log(json.dumps(dict(event="script", key=k, ok=out.text is not None,
                                status=results[k]["status"], seconds=round(clock() - t0, 1))))
    finally:
        fetcher.browser_runner = orig_runner
        led.sleep = orig_sleep


def _uncount_cancelled(fetcher, e):
    """A paper cut by the batch stop is not a real failure: mark its last browser row as cancelled, so it does not
    feed the channel's consecutive-failure breaker (it still counts as an attempt)."""
    from .fulltext import norm_doi
    led = fetcher.ledger
    with led._tx() as db:
        row = db.execute("select rowid from attempts where channel='browser' and doi=? order by rowid desc limit 1",
                         (norm_doi(e.get("doi")),)).fetchone()
        if row:
            db.execute("update attempts set note='cancelled: batch stopped' where rowid=? and status='fail'", row)


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


HOOK_CONFIG = """# tbsgen browser batch: isolated Hermes home (generated; removed with the batch directory)
model:
  default: {model}
  provider: {provider}
{providers}hooks:
  pre_tool_call:
    - matcher: ".*"
      command: {hook}
      timeout: 15
      fail_closed: true
hooks_auto_accept: true
"""


PROVIDER_FIELDS = ("name", "api", "base_url", "transport", "api_mode", "default_model", "discover_models", "models",
                   "context_length")


def _user_provider(path: Path, name: str) -> dict | None:
    """The `providers.<name>` mapping of a Hermes config.yaml (PyYAML when available, else a small reader for
    the simple `key: scalar` / nested-mapping layout Hermes writes)."""
    try:
        text = path.read_text()
    except OSError:
        return None
    with contextlib.suppress(Exception):
        import yaml
        return ((yaml.safe_load(text) or {}).get("providers") or {}).get(name)
    lines = text.splitlines()
    try:
        i = lines.index("providers:")
    except ValueError:
        return None
    out: dict = {}
    stack: list[tuple[int, dict]] = []
    inside = False
    for ln in lines[i + 1:]:
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        ind = len(ln) - len(ln.lstrip())
        if ind == 0:
            break
        key, _, val = ln.strip().partition(":")
        if ind == 2:
            if inside:
                break
            inside = key == name
            stack = [(2, out)]
            continue
        if not inside:
            continue
        while stack and ind <= stack[-1][0]:
            stack.pop()
        cur = stack[-1][1] if stack else out
        val = val.strip()
        if val == "":
            cur[key] = {}
            stack.append((ind, cur[key]))
        else:
            v: object = val.strip("'\"")
            if val in ("true", "false"):
                v = val == "true"
            elif re.fullmatch(r"-?\d+", val):
                v = int(val)
            cur[key] = v
    return out or None


def _hermes_data_root(src_home: Path) -> Path | None:
    """The Hermes data root holding the dependency state (`installs/`) of the user's Hermes install.

    Hermes scopes its dependency environments per data root (`<root>/installs/<key>`), and a HERMES_HOME
    outside `~/.hermes` is a root of its own.  The agent's isolated home (inside the batch directory) would
    then have no committed environment and the launcher exits 1 at once (\"no dependency environment is
    committed for this install\")."""
    for cand in (src_home, src_home.parent.parent if src_home.parent.name == "profiles" else None,
                 Path.home() / ".hermes"):
        if cand is not None and (cand / "installs").is_dir():
            return cand
    return None


def _hermes_home(bd: Path, ag: dict, hook_cmd: str) -> Path:
    """An isolated Hermes home for the agent: only the configured provider, the requested skills and the
    pre_tool_call hook.  No user memory, rules, plugins or other skills.  The provider entry comes from
    `agent.provider_config` or, for a user-defined provider, from the user's Hermes config; its `api_key` is never
    copied (use a provider that needs no key, e.g. a local proxy, or `key_env`).  `installs/` (Hermes' dependency
    environments; no credentials) is linked from the user's data root so the launcher finds its Python packages."""
    home = bd / "hermes-home"
    (home / "skills").mkdir(parents=True, exist_ok=True)
    src_home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    if ag.get("hermes_installs_dir"):
        installs = Path(str(ag["hermes_installs_dir"])).expanduser()
    else:
        root = _hermes_data_root(src_home)
        installs = root / "installs" if root is not None else None
    if installs is not None and installs.is_dir() and not (home / "installs").exists():
        with contextlib.suppress(OSError):
            (home / "installs").symlink_to(installs.resolve(), target_is_directory=True)
    pname = ag.get("provider") or ""
    entry = ag.get("provider_config")
    if entry is None and pname:
        entry = _user_provider(src_home / "config.yaml", pname)
    prov_block = ""
    if isinstance(entry, dict):
        # only non-sensitive fields; never api keys, headers or tokens (a provider that needs a credential cannot be
        # used for the agent: its process gets no credentials)
        entry = {k: v for k, v in entry.items() if k in PROVIDER_FIELDS}
        entry["api_key"] = "none"
        prov_block = "providers: " + json.dumps({pname: entry}) + "\n"        # JSON is valid YAML
    for sk in ag.get("skills") or []:
        src = src_home / "skills" / sk
        if src.exists():
            with contextlib.suppress(OSError):
                (home / "skills" / sk).symlink_to(src.resolve())
    (home / "config.yaml").write_text(HOOK_CONFIG.format(model=json.dumps(ag.get("model") or ""),
                                                         provider=json.dumps(pname), providers=prov_block,
                                                         hook=json.dumps(shlex.quote(hook_cmd))))
    return home


def _agent_phase(fetcher, todo: list[dict], deadline: float, bd: Path, wd: Watchdog, stop: threading.Event,
                 hermes_runner, log) -> dict:
    from .fulltext import Outcome, paper_ident
    ft = fetcher.ft
    b = ft["browser"]
    ag = b.get("agent") or {}
    ho = b.get("handoff") or {}
    sso = b.get("sso") or {}
    sso_login = b.get("sso_login") or {}
    led = fetcher.ledger
    papers = {}
    sso_domains = [str(x).lower() for x in sso.get("domains") or []]
    for i, e in enumerate(todo, 1):
        papers[str(i)] = dict(doi=e["doi"], title=e.get("title") or "", ident=paper_ident(e),
                              allow=allowed_domains(e["doi"], e.get("_landing"),
                                                    [*(ag.get("extra_domains") or []), *sso_domains]))
    if deadline - time.time() - 10 < min(30.0, float(ag.get("run_budget_s", 420))):
        return dict(usage=dict(skipped="not enough time left in the egress window"),
                    papers={v["ident"]: dict(status="deferred") for v in papers.values()})
    st = dict(space=agent_space(b), papers=papers, deadline=deadline,
              ledger_path=str(led.path), limits=led.limits, global_limits=led.g, host_caps=led.host_caps,
              ego_binary=browser_ego.binary(b), gate_max_wait_s=float(ag.get("gate_max_wait_s", 150)),
              max_calls_per_paper=int(ag.get("max_calls_per_paper", 25)),
              call_timeout_s=float(ag.get("call_timeout_s", 150)),
              challenge_cooldown_s=float(b.get("challenge_cooldown_s", 24 * 3600)),
              challenge_wait_ms=int(b.get("challenge_wait_s", 40) * 1000),
              max_navigations_per_paper=int(ag.get("max_navigations_per_paper", 12)),
              nav_gap_s=float(ag.get("nav_gap_s", 2.5)),
              handoff_enabled=bool(ho.get("enabled")), handoff_wait_s=float(ho.get("wait_s", 120)),
              handoff_notify=bool(ho.get("notify", True)), bark_cmd=[list(x) for x in ho.get("bark_cmd") or []],
              sso_login_enabled=bool(sso_login.get("enabled")) and bool(sso_login.get("domains")),
              sso_domains=[str(x).lower() for x in sso_login.get("domains") or []],
              sso_max_submissions=int(sso_login.get("max_submissions", 2)),
              type_allow=[str(x) for x in ag.get("institution_names") or []])
    (bd / "papers").mkdir(exist_ok=True)
    (bd / "files").mkdir(exist_ok=True)
    _atomic(bd / "batch.json", json.dumps(st))
    bin_dir = bd / "bin"
    bin_dir.mkdir(exist_ok=True)
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    for name, fn in (("tbsgen-browser", "wrapper_main"), ("tbsgen-hook", "hook_main")):
        w = bin_dir / name
        w.write_text(TOOL.format(python=sys.executable, pkg_parent=pkg_parent, batch_dir=str(bd), fn=fn))
        w.chmod(0o700)
    home = _hermes_home(bd, ag, str(bin_dir / "tbsgen-hook"))
    tmpl = Path(ag["prompt_template"]).read_text() if ag.get("prompt_template") else PROMPT.read_text()
    budget = min(float(ag.get("run_budget_s", 420)), deadline - time.time() - 10)
    lines = [dict(n=int(k), doi=v["doi"], title=v["title"], allowed_domains=v["allow"]) for k, v in papers.items()]
    lim = led.limits["browser"]
    prompt = render_prompt(tmpl, papers_json=json.dumps(lines, indent=1, ensure_ascii=False),
                           minutes=f"{budget / 60:.0f}", gap_s=f"{float(lim['gap_s']):.0f}",
                           gap_max_s=f"{float(lim['gap_s']) + float(lim['jitter_s']):.0f}",
                           calls=str(st["max_calls_per_paper"]), navs=str(st["max_navigations_per_paper"]),
                           type_names=" / ".join(f"'{x}'" for x in st["type_allow"]) or "(none: typing is off)")
    (bd / "prompt.md").write_text(prompt)
    hermes = ag.get("hermes_bin") or "hermes"
    toolsets = "terminal,vision" if ag.get("vision", True) else "terminal"
    argv = [hermes, "chat", "--query-file", str(bd / "prompt.md"), "--oneshot", "--format", "stream-json",
            "--source", "tool", "--accept-hooks", "--ignore-rules", "--max-turns", str(int(ag.get("max_turns", 80))),
            "--run-budget", str(int(budget)), "-t", toolsets]
    if ag.get("provider"):
        argv += ["--provider", str(ag["provider"])]
    if ag.get("model"):
        argv += ["-m", str(ag["model"])]
    env = dict(browser_ego.child_env())
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["HERMES_HOME"] = str(home)
    events = dict(tool_calls=0, violations=[])

    def on_event(ev):
        if ev.get("type") == "tool_use":          # second line of defence (the hook already blocked it)
            events["tool_calls"] += 1
            inp = ev.get("input") or {}
            if ev.get("name") == VISION_TOOL:
                why = _vision_allowed(str(bd), inp)
            elif ev.get("name") != "terminal":
                why = f"tool not allowed: {ev.get('name')}"
            elif inp.get("background") or inp.get("pty"):
                why = "background / pty terminal not allowed"
            else:
                why = shell_allowed(inp.get("command"))
            if why:
                events["violations"].append(why[:120])
                if len(events["violations"]) >= int(ag.get("max_violations", 3)):
                    return f"too many denied commands: {why[:120]}"
        if stop.is_set():
            return "batch stopped (egress changed or deadline)"
        return None
    t0 = time.time()
    log(json.dumps(dict(event="agent_start", papers=len(papers), budget_s=int(budget),
                        model=ag.get("model"), provider=ag.get("provider"))))
    runner = hermes_runner or default_hermes_runner
    err_path = bd / "agent-stderr.log"
    kw = {}
    with contextlib.suppress(TypeError, ValueError):
        if "stderr_path" in inspect.signature(runner).parameters:
            kw["stderr_path"] = err_path
    rr = runner(argv, env, budget + float(ag.get("kill_grace_s", 20)), on_event, stop, **kw)
    (bd / "STOP").write_text("agent run ended")       # a late tool call cannot book anything any more
    # the agent run is over: close the batch's TaskSpaces (a hand-off still open is taken back first)
    closed_spaces = []
    if fetcher.browser_runner is None and st.get("ego_binary"):
        spaces = sorted(set(_read_state(bd).get("spaces") or []) | {agent_space(b)})
        closed_spaces = _finish_spaces(bd, st, spaces)
    res = rr.get("result") or {}
    usage = dict(tokens=res.get("tokens"), duration_ms=res.get("duration_ms"), session_id=res.get("session_id"),
                 exit=rr.get("exit"), killed=rr.get("killed"), tool_calls=events["tool_calls"],
                 violations=events["violations"], seconds=round(time.time() - t0, 1),
                 provider=ag.get("provider"), model=ag.get("model"))
    with contextlib.suppress(OSError):
        with open(bd.parent / "usage.jsonl", "a") as fh:
            fh.write(json.dumps(dict(usage, ts=time.strftime("%F %T"), papers=len(papers),
                                     closed_spaces=closed_spaces)) + "\n")
    log(json.dumps(dict(event="agent_end", **{k: v for k, v in usage.items() if k != "session_id"},
                        stderr_tail=(rr.get("stderr_tail") or "")[-STDERR_LOG_TAIL:])))
    # ---- validate and book every paper (the agent's own report is not used)
    out: dict = dict(usage=usage, papers={})
    cut = stop.is_set() and not rr.get("result")     # killed by egress loss / deadline
    lost = wd.lost.is_set() or not wd.check_now()    # files saved while the egress was not allowed are dropped
    for k, p in papers.items():
        sp = bd / "papers" / f"{k}.json"
        ps = json.loads(sp.read_text()) if sp.exists() else {}
        c = next(e for e in todo if paper_ident(e) == p["ident"])
        rid = ps.get("rid")
        if rid is None:
            out["papers"][p["ident"]] = dict(status="deferred" if (lost or cut or rr.get("killed")) else "failed",
                                             errors=[ps.get("closed") or "agent did not open this paper"])
            continue
        txt, size, why = None, 0, "no valid file saved"
        rec = dict(doi=p["doi"], key=p["ident"], tried=[f"browser:agent:{ps['lkey']}"], errors=[], skipped=[],
                   source=None)
        if lost:
            why = "egress left allow_cidrs during the agent run; files discarded"
        else:
            # only files the action script reported as saved (and the wrapper accepted) are considered
            for f in [Path(x) for x in ps.get("accepted") or []]:
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
        if ps.get("challenge"):
            why = f"challenge: {ps['challenge']} (publisher cooled down)"
        ok = txt is not None
        interrupted = not ok and (lost or cut) and not ps.get("challenge")
        # every hand-off and SSO login of the paper is noted in the ledger row (time, host, result; never
        # credentials - the action script never reads the values of credential fields)
        extras = "; ".join(x for x in (ps.get("handoff_note"), ps.get("sso_note")) if x)
        note = "" if ok else ("cancelled: batch stopped" if interrupted else fetcher._r(f"agent: {why}")[:300])
        if extras:
            note = (note + "; " if note else "") + extras
        led.record("browser", ps["lkey"], p["doi"], "ok" if ok else "fail", None, size, note, reservation=rid)
        if ok:
            out["papers"][p["ident"]] = dict(status="ok", outcome=Outcome(txt, rec))
        elif interrupted:
            out["papers"][p["ident"]] = dict(status="deferred", errors=[why])
        else:
            out["papers"][p["ident"]] = dict(status="failed", errors=[fetcher._r(f"agent: {why}")])
    return out
