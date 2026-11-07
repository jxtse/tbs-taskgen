"""Remote mode (HTTP channels on another host over ssh) and egress-gated browser batches.  Offline: ssh is an
in-process stand-in that runs `python -m tbsgen.remote` against a local directory; the browser egress is a fake
probe sequence; the Hermes CLI and ego-browser are fake executables."""
import io
import json
import os
import sqlite3
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tbsgen import browser_agent, fulltext, remote
from tbsgen.dlledger import DownloadLedger
from tbsgen.fulltext import Fetcher, Resp

from test_fulltext import BODY, ELS_XML, FAKE_PDF, SECRETS, TITLE, FakeClock, FakeNet, mkcfg

TRACE_OK = "fl=1\nip=198.51.100.7\ncolo=SHA\n"
TRACE_SG = "fl=1\nip=203.0.113.9\ncolo=SIN\n"
EC = {"enabled": True, "allow_cidrs": ["198.51.100.0/24"], "urls": ["https://echo.example/cdn-cgi/trace"]}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    for k in ("ELSEVIER_API_KEY", "ELSEVIER_INSTTOKEN", "WILEY_TDM_TOKEN"):
        monkeypatch.delenv(k, raising=False)


# ------------------------------------------------------------------------------------------- remote mode
class FakeSSH:
    """Runs the remote commands in-process against `rdir` (the 'remote host').  submit runs the worker inline
    (no fork); `drop_after_submit` simulates an ssh connection that dies after the job started."""

    def __init__(self, rdir: Path, net: FakeNet, drop_after_submit=False, worker_stop_after=None):
        self.rdir, self.net = rdir, net
        self.calls = []
        self.stdin_seen = []
        self.drop_after_submit = drop_after_submit
        self.worker_stop_after = worker_stop_after
        self.down = False

    def __call__(self, cmd: str, stdin: bytes = b"", timeout: float = 300):
        import shlex
        argv = shlex.split(cmd)
        assert argv[1:3] == ["-m", "tbsgen.remote"]
        sub, wd, job = argv[3:6]
        self.calls.append(sub)
        if self.down:
            return 255, b"", b"ssh: connect failed"
        self.stdin_seen.append(stdin)
        out = io.BytesIO()

        class W(io.TextIOWrapper):
            pass
        tout = io.TextIOWrapper(out, encoding="utf-8", write_through=True)
        old = remote._TEST_TRANSPORT
        remote._TEST_TRANSPORT = self.net
        orig_work = remote.work
        if self.worker_stop_after is not None:
            remote.work = lambda d, spec, secrets: orig_work(d, spec, secrets, stop_after=self.worker_stop_after)
        try:
            if sub == "submit":
                rc = remote.main([sub, str(self.rdir), job], stdin=io.StringIO(stdin.decode()), stdout=tout,
                                 daemon=False)
            elif sub == "export":
                rc = remote.cmd_export(str(self.rdir), job, out)
            else:
                rc = remote.main([sub, str(self.rdir), job], stdout=tout)
        finally:
            remote._TEST_TRANSPORT = old
            remote.work = orig_work
        tout.flush()
        if sub == "submit" and self.drop_after_submit:
            self.drop_after_submit = False
            return 255, b"", b"Connection reset"
        return rc, out.getvalue(), b""


def mk_remote(tmp_path, routes, **ft):
    rdir = tmp_path / "remote-host"
    cfg = mkcfg(tmp_path, egress_check=dict(EC), elsevier={"enabled": True}, wiley={"enabled": True},
                remote={"enabled": True, "host": "fake", "python": "/opt/py", "workdir": str(rdir)}, **ft)
    clock = FakeClock(time.time())
    led = DownloadLedger(cfg["fulltext"]["ledger_path"], cfg["fulltext"]["limits"], cfg["fulltext"]["global_limits"],
                         cfg["fulltext"]["host_caps"])
    net = FakeNet(dict({"echo.example": TRACE_OK.encode()}, **routes))
    local_net = FakeNet({})                          # the local side must never make a request
    f = Fetcher(cfg, ledger=led, transport=local_net, secrets=dict(SECRETS))
    return f, net, local_net, led, rdir, clock


def queue(f, *entries):
    for e in entries:
        fulltext.enqueue(fulltext.queue_path(f.cfg), fulltext.queue_entry(e, subscribers=[["", "x"]]))


ELS = dict(doi="10.1016/j.x.2020.1", title=TITLE)
WIL = dict(doi="10.1002/abc.2", title=TITLE)
OA = dict(doi="10.9999/oa.3", title=TITLE, oa_url="https://repo.example/oa3.pdf")


def drain(f, ssh, **kw):
    import tbsgen.remote as r
    orig = r.default_ssh_runner
    r.default_ssh_runner = lambda rcfg: ssh
    try:
        return fulltext.drain_queue(f, browser=False, on_success=lambda e, o, s: True, log=lambda m: None, **kw)
    finally:
        r.default_ssh_runner = orig


def test_remote_runs_http_channels_remotely_and_imports(tmp_path):
    f, net, local_net, led, rdir, _ = mk_remote(tmp_path, {
        "api.elsevier.com": ELS_XML, "api.wiley.com": FAKE_PDF, "repo.example": FAKE_PDF})
    queue(f, ELS, WIL, OA)
    ssh = FakeSSH(rdir, net)
    st = drain(f, ssh)
    assert st["ok"] == 3 and local_net.calls == []                 # nothing fetched locally
    assert ssh.calls == ["submit", "status", "export", "cleanup"]
    # texts are in the LOCAL write cache with provenance naming the remote job
    w = Path(f.cfg["fulltext"]["write_cache_dir"])
    provs = [json.loads(p.read_text()) for p in w.glob("*.json")]
    assert len(provs) == 3 and all(p.get("via", "").startswith("remote:") for p in provs)
    assert {p["channel"] for p in provs} == {"elsevier", "wiley", "oa"}
    # one authoritative ledger: the remote rows are merged locally; the lease is gone
    db = sqlite3.connect(led.path)
    rows = db.execute("select channel, status from attempts where channel != 'lease'").fetchall()
    assert ("elsevier", "ok") in rows and ("wiley", "ok") in rows and ("oa", "ok") in rows
    assert db.execute("select count(*) from attempts where channel='lease'").fetchone()[0] == 0
    assert fulltext.read_queue(fulltext.queue_path(f.cfg)) == []
    assert not (rdir / "jobs").exists() or not any((rdir / "jobs").iterdir())   # remote job dir cleaned up


def test_remote_secrets_only_on_stdin_never_on_disk_or_argv(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    queue(f, ELS)
    seen_files = []
    ssh = FakeSSH(rdir, net)
    orig_work = remote.work

    def spy(d, spec, secrets):
        assert secrets == {"ELSEVIER_API_KEY": "k-els-test", "WILEY_TDM_TOKEN": "t-wiley-test"}
        for p in d.rglob("*"):
            if p.is_file() and p.suffix != ".sqlite":
                seen_files.append(p.read_bytes())
        return orig_work(d, spec, secrets)
    remote.work = spy
    try:
        drain(f, ssh)
    finally:
        remote.work = orig_work
    sub_stdin = ssh.stdin_seen[0].decode()
    assert "k-els-test" in sub_stdin.splitlines()[1] and "k-els-test" not in sub_stdin.splitlines()[0]
    assert seen_files and not any(b"k-els-test" in b or b"t-wiley-test" in b for b in seen_files)
    assert all("k-els-test" not in c for c in ssh.calls)
    # the API key header reached the (fake) publisher API only
    assert any(h.get("X-ELS-APIKey") == "k-els-test" for u, h in net.calls if "api.elsevier.com" in u)
    assert not any("k-els-test" in json.dumps(h) for u, h in net.calls if "echo.example" in u)


def test_remote_secrets_cmd(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    f.secrets.clear()
    script = tmp_path / "sec.py"
    script.write_text("import json,sys\nnames=json.load(sys.stdin)\n"
                      "print(json.dumps({n: 'from-cmd-' + n for n in names}))\n")
    f.ft["remote"]["secrets_cmd"] = [sys.executable, str(script)]
    assert remote.load_secrets(f, ["ELSEVIER_API_KEY"]) == {"ELSEVIER_API_KEY": "from-cmd-ELSEVIER_API_KEY"}


def test_remote_refuses_without_egress_check(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    f.ft["egress_check"] = {"enabled": False, "allow_cidrs": []}
    queue(f, ELS)
    st = drain(f, FakeSSH(rdir, net))
    assert st["ok"] == 0 and st["remote"]["jobs"][0]["stage"] == "dropped"
    assert "egress_check" in st["remote"]["jobs"][0]["error"]
    assert [u for u, _ in net.calls] == []
    assert sqlite3.connect(led.path).execute("select count(*) from attempts").fetchone()[0] == 0   # lease released
    assert len(fulltext.read_queue(fulltext.queue_path(f.cfg))) == 1                             # still queued


def test_remote_worker_egress_outside_range_downloads_nothing(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    net.routes["echo.example"] = TRACE_SG.encode()
    queue(f, ELS)
    st = drain(f, FakeSSH(rdir, net))
    assert st["ok"] == 0
    assert [u for u, _ in net.calls if "echo.example" not in u] == []
    assert sqlite3.connect(led.path).execute(
        "select count(*) from attempts where channel != 'lease'").fetchone()[0] == 0


def test_remote_code_fingerprint_mismatch_is_refused(tmp_path, monkeypatch):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    queue(f, ELS)
    real = remote.code_fingerprint
    calls = {"n": 0}

    def fp():
        calls["n"] += 1
        return real() if calls["n"] == 1 else "other-revision"
    monkeypatch.setattr(remote, "code_fingerprint", fp)
    st = drain(f, FakeSSH(rdir, net))
    assert st["remote"]["jobs"][0]["stage"] == "dropped" and "fingerprint" in st["remote"]["jobs"][0]["error"]
    assert net.calls == []


def test_remote_seed_makes_remote_respect_local_usage(tmp_path):
    """The local ledger's usage today is carried over: a channel cap already used up locally stops the remote."""
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML},
                                        limits={"elsevier": {"daily_ok": 2, "gap_s": 0.0, "jitter_s": 0.0}})
    led2 = DownloadLedger(led.path, f.ft["limits"])
    for i in range(2):
        rid = led2.acquire("elsevier", "api.elsevier.com", f"10.1016/old.{i}")
        led2.record("elsevier", "api.elsevier.com", f"10.1016/old.{i}", "ok", 200, 10, "", reservation=rid)
    queue(f, ELS)
    st = drain(f, FakeSSH(rdir, net))
    assert st["ok"] == 0
    assert not any("api.elsevier.com" in u for u, _ in net.calls)


def test_remote_global_cap_is_leased(tmp_path):
    """The global daily cap is shared: the remote may use at most the quota leased to it, and the lease blocks
    the same quota for local channels while the job runs."""
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF},
                                        global_limits={"daily_ok": 1})
    queue(f, OA, dict(doi="10.9999/oa.4", title=TITLE, oa_url="https://repo.example/oa4.pdf"))
    st = drain(f, FakeSSH(rdir, net))
    assert st["ok"] == 1
    db = sqlite3.connect(led.path)
    assert db.execute("select count(*) from attempts where status='ok'").fetchone()[0] == 1
    # while a lease is live, local checks see the quota as used
    remote.lease_and_snapshot(led, "j2", 1)
    from tbsgen.dlledger import LimitReached
    with pytest.raises(LimitReached):
        led.check("browser", "chrome:10.1080")


def test_remote_cooldown_set_remotely_reaches_local_ledger(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.wiley.com": Resp(429, {}, b"slow down", "")})
    queue(f, WIL)
    drain(f, FakeSSH(rdir, net))
    assert led.cooling("api.wiley.com")


def test_remote_ssh_drop_after_submit_resumes_without_loss_or_duplicates(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML, "api.wiley.com": FAKE_PDF})
    queue(f, ELS, WIL)
    ssh = FakeSSH(rdir, net, drop_after_submit=True)
    ssh_down_after = {"n": 0}
    real_call = ssh.__call__

    class Flaky:
        """submit succeeds remotely but the connection drops; then ssh is down for this run."""
        def __call__(self, cmd, stdin=b"", timeout=300):
            if "status" in cmd and ssh_down_after["n"] == 0:
                ssh_down_after["n"] += 1
                return 255, b"", b"down"
            return real_call(cmd, stdin, timeout)
    st1 = fulltext.drain_queue(f, browser=False, log=lambda m: None, on_success=lambda e, o, s: True) \
        if False else None
    import tbsgen.remote as r
    orig = r.default_ssh_runner
    r.default_ssh_runner = lambda rcfg: Flaky()
    try:
        st1 = fulltext.drain_queue(f, browser=False, on_success=lambda e, o, s: True, log=lambda m: None)
    finally:
        r.default_ssh_runner = orig
    # first run: the job ran remotely (inline) but could not be collected in this run? it was collected after the
    # status retry; either way nothing is lost and the queue/ledger agree
    st2 = drain(f, FakeSSH(rdir, net))
    total_ok = (st1 or {}).get("ok", 0) + st2.get("ok", 0)
    assert total_ok == 2
    db = sqlite3.connect(led.path)
    assert db.execute("select count(*) from attempts where status='ok'").fetchone()[0] == 2     # not doubled
    assert db.execute("select count(*) from attempts where channel='lease'").fetchone()[0] == 0
    assert fulltext.read_queue(fulltext.queue_path(f.cfg)) == []


def test_remote_dead_worker_partial_results_are_kept_and_rest_stays_queued(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML, "api.wiley.com": FAKE_PDF})
    queue(f, ELS, WIL)
    ssh = FakeSSH(rdir, net, worker_stop_after=1)
    real = ssh.__call__

    def runner(cmd, stdin=b"", timeout=300):
        rc, so, se = real(cmd, stdin, timeout)
        if "submit" in cmd:                                  # the worker 'died': stale pid, old heartbeat
            job = cmd.split()[-1]
            sp = rdir / "jobs" / job / "status.json"
            st = json.loads(sp.read_text())
            st.update(pid=2 ** 22 + 12345, ts=time.time() - 120)
            sp.write_text(json.dumps(st))
        return rc, so, se
    st1 = drain(f, runner)
    assert st1["ok"] == 1
    q = fulltext.read_queue(fulltext.queue_path(f.cfg))
    assert len(q) == 1 and q[0]["doi"] == WIL["doi"] and not q[0].get("next_try")     # still due, no backoff
    st2 = drain(f, FakeSSH(rdir, net))
    assert st2["ok"] == 1 and fulltext.read_queue(fulltext.queue_path(f.cfg)) == []
    db = sqlite3.connect(led.path)
    assert db.execute("select count(*) from attempts where status='ok'").fetchone()[0] == 2


def test_remote_merge_is_idempotent(tmp_path):
    led = DownloadLedger(tmp_path / "l.sqlite")
    remote.lease_and_snapshot(led, "j", 2)
    rows = {"attempts": [[1, time.time(), "2026-01-01", "oa", "h", "d", "ok", 200, 5, ""],
                         [2, time.time(), "2026-01-01", "oa", "h", "d2", "pending", None, 0, "booked"]],
            "blocks": [], "meta_calls": [], "cooldown": [["h2", time.time() + 99, "x"]], "hosts": []}
    remote.merge_rows(led, "j", rows)
    remote.merge_rows(led, "j", rows)
    db = sqlite3.connect(led.path)
    assert db.execute("select status, note from attempts order by rowid").fetchall() == [
        ("ok", ""), ("fail", "cancelled: remote worker ended")]
    assert led.cooling("h2")


def test_remote_one_job_at_a_time(tmp_path):
    import fcntl
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    queue(f, ELS)
    remote.jobs_dir(f.cfg).mkdir(parents=True, exist_ok=True)
    with open(remote.jobs_dir(f.cfg) / ".lock", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        r = remote.run_job(f, [fulltext.queue_entry(ELS)], ssh_runner=FakeSSH(rdir, net))
    assert r.get("busy") and net.calls == []


def test_posix_tz_matches_local_offset():
    tz = remote.posix_tz(time.time())
    off = time.localtime().tm_gmtoff
    assert tz.startswith("TBS") and (("-" in tz) == (off >= 0))


# ------------------------------------------------------------------------------------------- browser batches
class Seq:
    """A fake egress probe: returns the next body of a sequence (the last one repeats)."""

    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        return self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]


def mk_browser(tmp_path, *, agent=False, clock=None, **bb):
    cfg = mkcfg(tmp_path, egress_check=dict(EC),
                browser={"enabled": True, "batch_only": True,
                         "batch": dict({"window_s": 600, "margin_s": 60, "poll_s": 5, "recheck_s": 3600,
                                        "max_wait_s": 60, "probe_host_gap_s": 0, "require_http_first": False},
                                       **bb),
                         "agent": {"enabled": agent}})
    clock = clock or FakeClock(time.time())
    led = DownloadLedger(cfg["fulltext"]["ledger_path"], cfg["fulltext"]["limits"], cfg["fulltext"]["global_limits"],
                         cfg["fulltext"]["host_caps"], clock=clock, sleep=clock.sleep)
    return cfg, led, clock


def bqueue(cfg, n_by_prefix):
    i = 0
    for pre, n in n_by_prefix.items():
        for _ in range(n):
            i += 1
            fulltext.enqueue(fulltext.queue_path(cfg), fulltext.queue_entry(
                dict(doi=f"{pre}/paper.{i}", title=TITLE), subscribers=[["", f"it{i}"]]))


def script_runner(tmp_path, results=None, on_call=None):
    calls = []

    def run(job):
        if job.get("mode") in ("egress", "probe", "finish"):
            return {"status": "ok", "bodies": {u: TRACE_OK for u in job.get("urls", [])}}
        calls.append(job["doi"])
        if on_call:
            on_call(job)
        r = (results or {}).get(job["doi"], "ok")
        if r == "ok":
            out = Path(job["out_dir"]) / f"dl-{len(calls)}.bin"
            out.write_bytes(FAKE_PDF)
            return {"status": "downloaded", "path": str(out), "bytes": len(FAKE_PDF), "out_file": str(out)}
        return {"status": r, "error": r}
    return run, calls


def test_wait_for_egress_starts_when_allowed_and_paces_probes(tmp_path):
    cfg, led, clock = mk_browser(tmp_path)
    bqueue(cfg, {"10.1080": 1, "10.1093": 1})
    run, calls = script_runner(tmp_path)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    seq = Seq(TRACE_SG, TRACE_SG, TRACE_SG, TRACE_OK)
    pr = browser_agent.Prober(f.ft, probe_fn=seq, clock=clock, sleep=clock.sleep)
    t0 = clock()
    st = browser_agent.run_batch(f, wait_for_egress=True, prober=pr, clock=clock, sleep=clock.sleep,
                                 log=lambda m: None)
    assert st["ok"] == 2 and sorted(calls) == ["10.1080/paper.1", "10.1093/paper.2"]
    assert st["egress"]["ip"] == "198.51.100.7"
    assert st["window_start"] - t0 >= 15                       # 3 refused probes, 5 s apart
    assert st["deadline"] == pytest.approx(st["window_start"] + 540)
    probes_ts = [s for s in clock.slept if s > 0]
    assert all(s >= 5 - 1e-6 for s in probes_ts[:3])
    # probes are not downloads: only the 2 papers are in the ledger
    assert sqlite3.connect(led.path).execute("select count(*) from attempts").fetchone()[0] == 2
    assert fulltext.read_queue(fulltext.queue_path(cfg)) == []


def test_wait_for_egress_gives_up_quietly(tmp_path):
    cfg, led, clock = mk_browser(tmp_path, max_wait_s=30)
    bqueue(cfg, {"10.1080": 1})
    run, calls = script_runner(tmp_path)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_SG), clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, wait_for_egress=True, prober=pr, clock=clock, sleep=clock.sleep,
                                 log=lambda m: None)
    assert "refused" not in st and "nothing done" in st["note"] and calls == []
    assert 6 <= st["probes"] <= 8                               # every 5 s for 30 s
    assert len(fulltext.read_queue(fulltext.queue_path(cfg))) == 1


def test_without_wait_outside_egress_is_refused_with_hint(tmp_path):
    cfg, led, clock = mk_browser(tmp_path, refuse_hint="Pause the VPN client first.")
    bqueue(cfg, {"10.1080": 1})
    run, calls = script_runner(tmp_path)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_SG), clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, log=lambda m: None)
    assert "203.0.113.9" in st["refused"] and "Pause the VPN client first." in st["refused"]
    assert calls == [] and st["probes"] == 1


def test_egress_lost_mid_batch_stops_and_requeues_rest(tmp_path):
    cfg, led, clock = mk_browser(tmp_path)
    bqueue(cfg, {"10.1080": 1, "10.1093": 1, "10.1117": 1, "10.1287": 1})
    run, calls = script_runner(tmp_path)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    # probe 1: start (ok); after paper 1: ok; after paper 2: lost
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK, TRACE_OK, TRACE_SG), clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, log=lambda m: None)
    assert len(calls) == 2 and st["ok"] == 1 and st["egress_lost"]
    q = fulltext.read_queue(fulltext.queue_path(cfg))
    assert len(q) == 3 and all(not e.get("browser_next_try") for e in q)     # all still due
    # the file obtained while the egress had already changed was discarded
    assert len(list(Path(cfg["fulltext"]["write_cache_dir"]).glob("*.txt"))) == 1


def test_batch_size_fits_the_window(tmp_path):
    cfg, led, clock = mk_browser(tmp_path, window_s=600, margin_s=60, est_script_s=60, batch_max=50,
                                 per_publisher_max=50)
    bqueue(cfg, {"10.1080": 6, "10.1093": 6, "10.1117": 6})
    run, calls = script_runner(tmp_path)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK), clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, log=lambda m: None)
    assert len(st["planned"]) == 9                             # 540 s / 60 s
    # publishers interleaved; ledger pacing (60-120 s per publisher) and the deadline cut the rest
    assert st["ok"] + st["deferred"] == 9 and st["ok"] >= 3
    assert clock() <= st["deadline"] + 60


def test_plan_respects_ledger_quota(tmp_path):
    es = [dict(doi=f"10.10{i % 3}0/x.{i}") for i in range(20)]
    assert len(browser_agent.plan_batch(es, 540, {"est_script_s": 60, "batch_max": 50, "per_publisher_max": 9},
                                        False, 4)) == 4
    p = browser_agent.plan_batch(es, 540, {"est_script_s": 60, "est_agent_s": 120, "agent_fraction": 0.5,
                                           "batch_max": 50, "per_publisher_max": 9}, True, None)
    assert len(p) == 4                                         # 540 / (60 + 60)
    pubs = [fulltext.publisher_of(e["doi"])[0] for e in p]
    assert all(a != b for a, b in zip(pubs, pubs[1:]))


def test_ledger_cap_reached_skips_browser(tmp_path):
    cfg, led, clock = mk_browser(tmp_path)
    cfg["fulltext"]["limits"]["browser"]["daily_ok"] = 0
    led.limits["browser"]["daily_ok"] = 0
    bqueue(cfg, {"10.1080": 1})
    run, calls = script_runner(tmp_path)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK), clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, log=lambda m: None)
    assert calls == [] and st["ok"] == 0 and st.get("planned") == []


def test_validation_failure_is_not_success(tmp_path, monkeypatch):
    cfg, led, clock = mk_browser(tmp_path)
    bqueue(cfg, {"10.1080": 1})
    run, calls = script_runner(tmp_path)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: "Some other paper entirely. " * 300)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK), clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, log=lambda m: None)
    assert st["ok"] == 0 and st["failed"] == 1
    assert "title mismatch" in json.dumps(st["papers"])
    q = fulltext.read_queue(fulltext.queue_path(cfg))
    assert q[0]["browser_next_try"] > time.time()               # browser backoff, HTTP backoff untouched
    assert not q[0].get("next_try")


# ---- agent phase with a fake `hermes` executable and a fake `ego-browser`
FAKE_EGO = r'''#!{py}
import json, re, sys, os
code = sys.stdin.read()
m = re.search(r"const GUARD = (\{.*?\});\n", code)
g = json.loads(m.group(1))
mode = "{ego_mode}"
if "TBSGEN_RESULT" in code or "INPUT" in code[:200]:
    print('TBSGEN_RESULT {"status": "ok", "bodies": {}}'); sys.exit(0)
if mode == "pdf":
    open(os.path.join(g["files_dir"], "paper.pdf"), "wb").write(b"%PDF-1.7 fake")
elif mode == "html":
    open(os.path.join(g["files_dir"], "paper.pdf"), "wb").write(b"<html>login</html>")
print("saved")
'''

FAKE_HERMES = r'''#!{py}
import json, os, subprocess, sys, time
argv = sys.argv[1:]
open(os.environ["FAKE_HERMES_ARGS"], "w").write(json.dumps(argv))
mode = os.environ.get("FAKE_HERMES_MODE", "work")
print(json.dumps({{"type": "system", "subtype": "init", "model": "fake"}}), flush=True)
def tool(cmd):
    print(json.dumps({{"type": "tool_use", "name": "terminal", "input": {{"command": cmd}}}}), flush=True)
if mode == "work":
    for n in (1, 2):
        tool(f"ego-browser nodejs <<'EOF' ...paper {{n}}")
        r = subprocess.run(["ego-browser", "nodejs"], input=f"// paper: {{n}}\nawait 1;\n", capture_output=True, text=True)
        print(json.dumps({{"type": "tool_result", "output": r.stdout[-200:]}}), flush=True)
elif mode == "curl":
    tool("curl -s https://sci-hub.example/10.1/x")
    time.sleep(5)
elif mode == "hang":
    time.sleep(60)
print(json.dumps({{"type": "result", "exit_code": 0, "text": "1: saved\n2: saved", "tokens": {{"input": 10, "output": 5, "total": 15}}, "duration_ms": 1234, "session_id": "s"}}), flush=True)
'''


def mk_agent(tmp_path, monkeypatch, hermes_mode="work", ego_mode="pdf", **bb):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    ego = bin_dir / "ego-real"
    ego.write_text(FAKE_EGO.replace("{py}", sys.executable).replace("{ego_mode}", ego_mode))
    ego.chmod(0o755)
    herm = bin_dir / "hermes"
    herm.write_text(FAKE_HERMES.replace("{py}", sys.executable).replace("{{", "{").replace("}}", "}"))
    herm.chmod(0o755)
    monkeypatch.setenv("FAKE_HERMES_MODE", hermes_mode)
    monkeypatch.setenv("FAKE_EGO_MODE", ego_mode)
    monkeypatch.setenv("FAKE_HERMES_ARGS", str(tmp_path / "hermes-args.json"))
    monkeypatch.setattr(browser_agent.browser_ego, "ENV_ALLOW",
                        browser_agent.browser_ego.ENV_ALLOW + ("FAKE_HERMES_MODE", "FAKE_EGO_MODE", "FAKE_HERMES_ARGS"))
    clock = FakeClock(time.time())
    clock.sleep = lambda s: None                               # real time for the agent run; no pacing waits
    cfg = mkcfg(tmp_path, egress_check=dict(EC),
                browser={"enabled": True, "batch_only": True, "binary": str(ego),
                         "batch": dict({"window_s": 600, "margin_s": 60, "poll_s": 5, "recheck_s": 3600,
                                        "probe_host_gap_s": 0, "require_http_first": False}, **bb),
                         "agent": {"enabled": True, "hermes_bin": str(herm), "provider": "prov-x", "model": "model-y",
                                   "skills": ["ego-browser"], "max_turns": 7, "run_budget_s": 30,
                                   "call_timeout_s": 20}})
    led = DownloadLedger(cfg["fulltext"]["ledger_path"], cfg["fulltext"]["limits"], cfg["fulltext"]["global_limits"],
                         cfg["fulltext"]["host_caps"])
    bqueue(cfg, {"10.1080": 1, "10.1093": 1})
    run, calls = script_runner(tmp_path, results={"10.1080/paper.1": "html", "10.1093/paper.2": "html"})
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK), sleep=lambda s: None)
    return f, led, pr, calls


def test_agent_fallback_saves_validated_files(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch)
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert len(calls) == 2                                     # the script went first
    assert st["agent_ok"] == 2 and st["ok"] == 2, st
    args = json.loads((tmp_path / "hermes-args.json").read_text())
    assert args[:2] == ["chat", "--query-file"] and "--oneshot" in args
    assert args[args.index("--provider") + 1] == "prov-x" and args[args.index("-m") + 1] == "model-y"
    assert args[args.index("--max-turns") + 1] == "7" and "-s" in args
    assert int(args[args.index("--run-budget") + 1]) <= 30
    assert st["agent"]["usage"]["tokens"]["total"] == 15
    rows = sqlite3.connect(led.path).execute("select channel, status, note from attempts order by rowid").fetchall()
    assert [r[1] for r in rows if r[0] == "browser"] == ["fail", "fail", "ok", "ok"]
    provs = [json.loads(p.read_text()) for p in Path(f.ft["write_cache_dir"]).glob("*.json")]
    assert all(p.get("via") == "agent" for p in provs) and len(provs) == 2
    assert fulltext.read_queue(fulltext.queue_path(f.cfg)) == []
    usage = (fulltext.queue_path(f.cfg).with_name(fulltext.queue_path(f.cfg).name + ".browser") / "usage.jsonl")
    assert json.loads(usage.read_text().splitlines()[-1])["model"] == "model-y"


def test_agent_claims_do_not_count_without_a_valid_file(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch, ego_mode="html")
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert st["ok"] == 0 and st["failed"] == 2                 # the agent said "saved"; code says no
    assert len(fulltext.read_queue(fulltext.queue_path(f.cfg))) == 2


def test_agent_denied_command_kills_the_run(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch, hermes_mode="curl")
    t0 = time.time()
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert time.time() - t0 < 5 and "denied command" in st["agent"]["usage"]["killed"]
    assert st["ok"] == 0


def test_agent_timeout_is_enforced(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch, hermes_mode="hang")
    f.ft["browser"]["agent"].update(run_budget_s=2, kill_grace_s=1)
    t0 = time.time()
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert time.time() - t0 < 20 and st["agent"]["usage"]["killed"] == "timeout"
    assert st["ok"] == 0 and st["deferred"] == 2               # never opened: stays due
    q = fulltext.read_queue(fulltext.queue_path(f.cfg))
    assert len(q) == 2


def test_agent_egress_lost_discards_files(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch)
    # start ok, ok after each of the 2 script papers, then lost at the post-agent check
    pr.probe_fn = Seq(TRACE_OK, TRACE_OK, TRACE_OK, TRACE_SG)
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert st["ok"] == 0 and st["egress_lost"]
    assert not list(Path(f.ft["write_cache_dir"]).glob("*.txt"))


def test_wrapper_requires_paper_line_and_refuses_denied_apis(tmp_path, monkeypatch, capsys):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch)
    bd = tmp_path / "bd"
    (bd / "papers").mkdir(parents=True)
    st = dict(space="sp", papers={"1": dict(doi="10.1080/p", title=TITLE, ident="doi:10.1080/p",
                                            allow=["tandfonline.com"])},
              deadline=time.time() + 300, ledger_path=str(led.path), limits=led.limits, global_limits=led.g,
              host_caps=led.host_caps, ego_binary="/bin/false", gate_max_wait_s=100, max_calls_per_paper=2,
              call_timeout_s=10)
    (bd / "batch.json").write_text(json.dumps(st))
    monkeypatch.setattr(sys, "stdin", io.StringIO("await page.goto('x')"))
    assert browser_agent.wrapper_main(str(bd), ["nodejs"]) == 2
    monkeypatch.setattr(sys, "stdin", io.StringIO("// paper: 1\nawait (await userPage()).goto('x')"))
    assert browser_agent.wrapper_main(str(bd), ["nodejs"]) == 2
    monkeypatch.setattr(sys, "stdin", io.StringIO("// paper: 1\nconst c = document.cookie"))
    assert browser_agent.wrapper_main(str(bd), ["nodejs"]) == 2
    assert sqlite3.connect(led.path).execute("select count(*) from attempts").fetchone()[0] == 0
    (bd / "STOP").write_text("x")
    monkeypatch.setattr(sys, "stdin", io.StringIO("// paper: 1\nawait 1"))
    assert browser_agent.wrapper_main(str(bd), ["nodejs"]) == 3
    out = capsys.readouterr().out
    assert "paper: N" in out and "userPage" in out and "STOP" in out


def test_prompt_template_carries_the_hard_rules():
    t = browser_agent.PROMPT.read_text()
    for s in ("CAPTCHA", "credentials", "allowed_domains", "Sci-Hub", "user's own tabs", "$papers_json", "// finish",
              "$gap_s"):
        assert s in t
    g = browser_agent.GUARD.read_text()
    for s in ("claimTaskSpace", "userPage", "fill", "verification", "files_dir", "allow"):
        assert s in g


def test_allowed_domains_cover_publisher_and_landing():
    d = browser_agent.allowed_domains("10.1093/abc/1", "https://academic.oup.com/x/article/1", ["doi.org"])
    assert "oup.com" in d and "doi.org" in d and "silverchair.com" in d
    assert not any("sci-hub" in x for x in d)
