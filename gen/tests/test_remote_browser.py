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
                rc = remote.main([sub, str(self.rdir), job], stdin=io.StringIO(stdin.decode()), stdout=tout)
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
    with open(str(led.path) + ".remote.lock", "a+") as fh:          # per ledger
        fcntl.flock(fh, fcntl.LOCK_EX)
        r = remote.run_job(f, [fulltext.queue_entry(ELS)], ssh_runner=FakeSSH(rdir, net))
    assert r.get("busy") and net.calls == []


def test_lease_holds_quota_past_pending_ttl(tmp_path):
    from tbsgen.dlledger import PENDING_TTL_S, LimitReached
    # base at local noon: the fake clock's +TTL never crosses midnight (a day rollover frees the quota)
    clock = FakeClock(time.mktime(time.localtime()[:3] + (12, 0, 0, 0, 0, 0)))
    led = DownloadLedger(tmp_path / "l.sqlite", global_limits={"daily_ok": 1}, clock=clock, sleep=clock.sleep)
    remote.lease_and_snapshot(led, "j", 1)
    clock.t += PENDING_TTL_S + 60
    with pytest.raises(LimitReached):
        led.check("browser", "chrome:10.1080")


def test_second_remote_job_waits_for_the_first_lease(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    remote.lease_and_snapshot(led, "other-queue-job", 1)
    r = remote.run_job(f, [fulltext.queue_entry(ELS)], ssh_runner=FakeSSH(rdir, net))
    assert "holds a lease" in r["note"] and net.calls == []


def test_worker_crash_after_save_before_result_is_recovered(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    queue(f, ELS)
    ssh = FakeSSH(rdir, net)
    real = ssh.__call__
    orig_atomic = remote._atomic

    def runner(cmd, stdin=b"", timeout=300):
        if "submit" in cmd:
            def crashy(path, data):                         # the result file of paper 0 is never written
                if path.name == "0000.json":
                    raise SystemExit("worker killed")
                return orig_atomic(path, data)
            remote._atomic = crashy
            try:
                return real(cmd, stdin, timeout)
            except SystemExit:
                return 0, b'{"ok": true}\n', b""
            finally:
                remote._atomic = orig_atomic
                job = cmd.split()[-1]
                sp = rdir / "jobs" / job / "status.json"
                st = json.loads(sp.read_text())
                st.update(state="running", ts=time.time() - 120)
                sp.write_text(json.dumps(st))
        return real(cmd, stdin, timeout)
    st = drain(f, runner)
    assert st["ok"] == 1 and fulltext.read_queue(fulltext.queue_path(f.cfg)) == []
    assert len(list(Path(f.ft["write_cache_dir"]).glob("*.txt"))) == 1
    db = sqlite3.connect(led.path)
    assert db.execute("select count(*) from attempts where status='ok'").fetchone()[0] == 1


def test_tilde_workdir_uses_one_job_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    f.ft["remote"]["workdir"] = "~/tbs-remote"
    queue(f, ELS)
    st = drain(f, FakeSSH(Path("~/tbs-remote"), net))
    assert st["ok"] == 1
    assert not (tmp_path / "home" / "tbs-remote" / "jobs").exists() or \
        not any((tmp_path / "home" / "tbs-remote" / "jobs").iterdir())


def test_ssh_and_secrets_cmd_get_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ELSEVIER_API_KEY", "k-env")
    monkeypatch.setenv("SOME_TOKEN", "t")
    seen = {}

    def fake_run(argv, **kw):
        seen.setdefault("envs", []).append(kw.get("env"))

        class P:
            returncode, stdout, stderr = 0, b"{}", b""
        return P()
    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    remote.default_ssh_runner({"host": "h"})("true")
    f, *_ = mk_remote(tmp_path, {})
    f.ft["remote"]["secrets_cmd"] = ["x"]
    remote.load_secrets(f, ["A"])
    for env in seen["envs"]:
        assert env is not None and "ELSEVIER_API_KEY" not in env and "SOME_TOKEN" not in env


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
FAKE_EGO = r"""#!{py}
import json, os, re, sys, time
# the action scripts are passed as an argument (like `ego-browser nodejs <script>`)
code = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
m = re.match(r"const INPUT = (\{.*?\});\n", code)
inp = json.loads(m.group(1)) if m else {}
mode = "{ego_mode}"
log = "{ego_log}"
if log:
    open(log, "a").write(json.dumps({"action": inp.get("action"), "arg": inp.get("arg"),
                                     "space": inp.get("space"), "ts": time.time(),
                                     "handoff": (inp.get("handoff") or {}).get("enabled"),
                                     "sso_login": (inp.get("sso") or {}).get("login")}) + "\n")
out = {"status": "ok", "files": []}
if inp.get("action") == "handoff_status":
    n_status = sum(1 for l in open(log) if '"action": "handoff_status"' in l)
    if mode == "handoff_wait":
        # the user has not returned control yet; the second poll sees the page cleared
        out = {"status": "ok", "handoff_check": "cleared" if n_status >= 2 else "waiting",
               "handoff_left_s": 99, "page": {"url": "https://pub.example/a", "title": "A"}}
    elif mode == "handoff_stuck":
        # the user cannot clear it: the poll still sees the challenge and keeps waiting
        out = {"status": "ok", "handoff_check": "waiting", "handoff_left_s": 99,
               "page": {"url": "https://pub.example/a", "title": "Just a moment"}}
    else:
        out = {"status": "ok", "handoff_check": "none"}
elif inp.get("action") == "sso_login":
    if mode == "sso_ok":
        out = {"status": "ok", "sso": "ok", "page": {"url": "https://pub.example/a"}}
    elif mode == "sso_fail":
        out = {"status": "ok", "sso": "fail", "page": {"url": "https://sso.example/login", "login_form": True}}
    else:
        out = {"status": "refused", "error": "sso_login is not enabled for this batch"}
elif inp.get("action") == "screenshot" and mode == "shot":
    f = os.path.join(inp["files_dir"], "shot-1.png")
    open(f, "wb").write(b"\x89PNG fake")
    out = {"status": "ok", "shot": f}
elif inp.get("action") == "open":
    n_status = sum(1 for l in open(log) if '"action": "handoff_status"' in l)
    if mode == "challenge":
        out["blocked"] = "verification page: Just a moment"
    elif mode in ("handoff", "handoff_wait", "handoff_stuck"):
        if mode == "handoff" and not (inp.get("handoff") or {}).get("enabled"):
            out["blocked"] = "verification page: Just a moment"   # no hand-off budget: the old blocked path
        elif mode == "handoff_stuck" and n_status >= 1:
            out["blocked"] = "verification page: Just a moment"   # the user cannot clear it
        elif n_status < 2:
            out = {"status": "handoff", "handoff": {"reason": "challenge: Just a moment", "host": "pub.example"}}
        else:                                                     # the user cleared the page
            f = os.path.join(inp["files_dir"], "dl.bin")
            open(f, "wb").write(b"%PDF-1.7 fake")
            out["files"] = [f]
    elif mode == "slow":
        time.sleep(30)
    else:
        f = os.path.join(inp["files_dir"], "dl.bin")
        open(f, "wb").write(b"%PDF-1.7 fake" if mode in ("pdf", "handoff_wait", "handoff_stuck") else b"<html>login</html>")
        out["files"] = [f]
        if mode == "rejected":                   # a refused PDF whose deletion did not happen
            open(f, "wb").write(b"%PDF-1.7 fake")
            out = {"status": "refused", "files": [], "rejected": [f]}
print("TBSGEN_RESULT " + json.dumps(out))
"""

FAKE_HERMES = r"""#!{py}
import json, os, subprocess, sys, time
argv = sys.argv[1:]
open(os.environ["FAKE_HERMES_ARGS"], "w").write(json.dumps({"argv": argv, "home": os.environ.get("HERMES_HOME"),
                                                           "env": sorted(os.environ)}))
mode = os.environ.get("FAKE_HERMES_MODE", "work")
print(json.dumps({"type": "system", "subtype": "init", "model": "fake"}), flush=True)
def tool(cmd):
    print(json.dumps({"type": "tool_use", "name": "terminal", "input": {"command": cmd}}), flush=True)
    # like Hermes: run the pre_tool_call hook first, execute only if it does not block
    home = os.environ["HERMES_HOME"]
    hook = [l.split("command:", 1)[1].strip() for l in open(os.path.join(home, "config.yaml")) if "command:" in l][0]
    r = subprocess.run([json.loads(hook)], input=json.dumps({"tool_name": "terminal", "tool_input": {"command": cmd}}),
                       capture_output=True, text=True)
    if r.stdout.strip():
        print(json.dumps({"type": "tool_result", "output": r.stdout.strip()}), flush=True)
        return
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(json.dumps({"type": "tool_result", "output": r.stdout[-300:]}), flush=True)
if mode == "work":
    for n in (1, 2):
        tool(f"tbsgen-browser {n} open https://doi.org/10.1080/x")
    tool("tbsgen-browser finish")
elif mode == "navs":                              # many navigations of one paper (nav cap / pacing)
    for _ in range(4):
        tool("tbsgen-browser 1 open https://doi.org/10.1080/x")
    tool("tbsgen-browser finish")
elif mode == "handoff":                           # a challenge is handed to the user, then cleared
    for n in (1, 2):
        tool(f"tbsgen-browser {n} open https://doi.org/10.1080/x")
        tool(f"tbsgen-browser {n} handoff_status")
        tool(f"tbsgen-browser {n} handoff_status")
        tool(f"tbsgen-browser {n} open https://doi.org/10.1080/x")
    tool("tbsgen-browser finish")
elif mode == "type":                               # typing is whitelisted institution names only
    tool("tbsgen-browser 1 type 4 Nanjing University")
    tool("tbsgen-browser 1 type 4 hunter2")
    tool("tbsgen-browser 1 open https://doi.org/10.1080/x")
    tool("tbsgen-browser finish")
elif mode == "handoff_timeout":                   # the user never finishes
    tool("tbsgen-browser 1 open https://doi.org/10.1080/x")
    time.sleep(0.3)
    tool("tbsgen-browser 1 handoff_status")
    tool("tbsgen-browser finish")
elif mode == "sso":                               # an SSO login per paper
    for n in (1, 2):
        tool(f"tbsgen-browser {n} sso_login")
    tool("tbsgen-browser finish")
elif mode == "shot":                              # a screenshot instead of a download
    tool("tbsgen-browser 1 screenshot")
    tool("tbsgen-browser finish")
elif mode == "curl":
    tool("touch " + os.environ["FAKE_MARKER"])
    tool("curl -s https://sci-hub.example/10.1/x")
    tool("env")
    time.sleep(5)
elif mode == "hang":
    time.sleep(60)
print(json.dumps({"type": "result", "exit_code": 0, "text": "1: saved\n2: saved", "tokens": {"input": 10, "output": 5, "total": 15}, "duration_ms": 1234, "session_id": "s"}), flush=True)
"""


def mk_agent(tmp_path, monkeypatch, hermes_mode="work", ego_mode="pdf", queue=None, agent=None,
             handoff=None, sso=None, sso_login=None, **bb):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    ego = bin_dir / "ego-real"
    ego.write_text(FAKE_EGO.replace("{py}", sys.executable).replace("{ego_mode}", ego_mode)
                   .replace("{ego_log}", str(tmp_path / "ego.log")))
    ego.chmod(0o755)
    herm = bin_dir / "hermes"
    herm.write_text(FAKE_HERMES.replace("{py}", sys.executable))
    herm.chmod(0o755)
    monkeypatch.setenv("FAKE_HERMES_MODE", hermes_mode)
    monkeypatch.setenv("FAKE_HERMES_ARGS", str(tmp_path / "hermes-args.json"))
    monkeypatch.setenv("FAKE_MARKER", str(tmp_path / "marker"))
    monkeypatch.setenv("FAKE_EGO_LOG", str(tmp_path / "ego.log"))
    monkeypatch.setenv("ELSEVIER_API_KEY", "k-should-not-leak")
    monkeypatch.setattr(browser_agent.browser_ego, "ENV_ALLOW", browser_agent.browser_ego.ENV_ALLOW + (
        "FAKE_HERMES_MODE", "FAKE_HERMES_ARGS", "FAKE_MARKER", "FAKE_EGO_LOG"))
    browser = {"enabled": True, "batch_only": True, "binary": str(ego),
               "batch": dict({"window_s": 600, "margin_s": 60, "poll_s": 5, "recheck_s": 3600,
                              "probe_host_gap_s": 0, "require_http_first": False}, **bb),
               "agent": dict({"enabled": True, "hermes_bin": str(herm), "provider": "prov-x", "model": "model-y",
                              "provider_config": {"api": "http://127.0.0.1:9/v1", "api_key": "SECRET-K"},
                              "max_turns": 7, "run_budget_s": 30, "call_timeout_s": 20},
                             **(agent or {}))}
    for k, v in (("handoff", handoff), ("sso", sso), ("sso_login", sso_login)):
        if v is not None:
            browser[k] = v
    cfg = mkcfg(tmp_path, egress_check=dict(EC), browser=browser)
    led = DownloadLedger(cfg["fulltext"]["ledger_path"], cfg["fulltext"]["limits"], cfg["fulltext"]["global_limits"],
                         cfg["fulltext"]["host_caps"])
    dois = []
    i = 0
    for pre, n in (queue if queue is not None else {"10.1080": 1, "10.1093": 1}).items():
        for _ in range(n):
            i += 1
            dois.append(f"{pre}/paper.{i}")
    bqueue(cfg, queue if queue is not None else {"10.1080": 1, "10.1093": 1})
    run, calls = script_runner(tmp_path, results={d: "html" for d in dois})   # the script fails them all
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK), sleep=lambda s: None)
    return f, led, pr, calls


def test_agent_fallback_saves_validated_files(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch)
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert len(calls) == 2                                     # the script went first
    assert st["agent_ok"] == 2 and st["ok"] == 2, st
    got = json.loads((tmp_path / "hermes-args.json").read_text())
    args = got["argv"]
    assert args[:2] == ["chat", "--query-file"] and "--oneshot" in args and "--accept-hooks" in args
    assert args[args.index("--provider") + 1] == "prov-x" and args[args.index("-m") + 1] == "model-y"
    assert args[args.index("--max-turns") + 1] == "7" and args[args.index("-t") + 1] == "terminal,vision"
    assert int(args[args.index("--run-budget") + 1]) <= 30
    assert "ELSEVIER_API_KEY" not in got["env"]                # the agent's process gets no credentials
    assert got["home"] and "hermes-home" in got["home"]       # isolated Hermes home with the hook
    assert st["agent"]["usage"]["tokens"]["total"] == 15
    rows = sqlite3.connect(led.path).execute("select channel, status, note from attempts order by rowid").fetchall()
    assert [r[1] for r in rows if r[0] == "browser"] == ["fail", "fail", "ok", "ok"]
    provs = [json.loads(p.read_text()) for p in Path(f.ft["write_cache_dir"]).glob("*.json")]
    assert all(p.get("via") == "agent" for p in provs) and len(provs) == 2
    assert fulltext.read_queue(fulltext.queue_path(f.cfg)) == []
    usage = (fulltext.queue_path(f.cfg).with_name(fulltext.queue_path(f.cfg).name + ".browser") / "usage.jsonl")
    assert json.loads(usage.read_text().splitlines()[-1])["model"] == "model-y"
    ego_log = [json.loads(x) for x in (tmp_path / "ego.log").read_text().splitlines()]
    base = f.ft["browser"]["space"] + "-agent"
    # one TaskSpace per paper (a hand-off pauses one paper only) + the base space closed by `finish`
    assert {e["space"] for e in ego_log} == {f"{base}-p1", f"{base}-p2", base}


def test_agent_isolated_home_has_no_api_key(tmp_path, monkeypatch):
    bd = tmp_path / "bd"
    bd.mkdir()
    home = browser_agent._hermes_home(bd, {"provider": "p", "model": "m",
                                           "provider_config": {"api": "http://x/v1", "api_key": "SECRET-K"}}, "/h")
    txt = (home / "config.yaml").read_text()
    assert "SECRET-K" not in txt and "pre_tool_call" in txt and '"/h"' in txt


def test_agent_claims_do_not_count_without_a_valid_file(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch, ego_mode="html")
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert st["ok"] == 0 and st["failed"] == 2                 # the agent said "saved"; code says no
    assert len(fulltext.read_queue(fulltext.queue_path(f.cfg))) == 2


def test_agent_challenge_cools_publisher_at_once(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch, ego_mode="challenge")
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert st["ok"] == 0 and led.cooling("chrome:10.1080") and led.cooling("www.tandfonline.com")
    assert "challenge" in json.dumps(st["papers"])


def test_agent_commands_are_blocked_before_they_run(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch, hermes_mode="curl")
    t0 = time.time()
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert not (tmp_path / "marker").exists()                 # blocked by the hook, never executed
    assert time.time() - t0 < 8 and "too many denied commands" in st["agent"]["usage"]["killed"]
    assert st["ok"] == 0 and len(st["agent"]["usage"]["violations"]) == 3


def test_agent_timeout_is_enforced(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch, hermes_mode="hang")
    f.ft["browser"]["agent"].update(run_budget_s=2, kill_grace_s=1)
    t0 = time.time()
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert time.time() - t0 < 20 and st["agent"]["usage"]["killed"] == "timeout"
    assert st["ok"] == 0 and st["deferred"] == 2               # never opened: stays due
    q = fulltext.read_queue(fulltext.queue_path(f.cfg))
    assert len(q) == 2 and not any(e.get("browser_next_try") for e in q)


def test_agent_egress_lost_discards_files_and_requeues(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch)
    # start ok, ok after each of the 2 script papers, then lost at the post-agent check
    pr.probe_fn = Seq(TRACE_OK, TRACE_OK, TRACE_OK, TRACE_SG)
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert st["ok"] == 0 and st["egress_lost"] and st["deferred"] == 2
    assert not list(Path(f.ft["write_cache_dir"]).glob("*.txt"))
    q = fulltext.read_queue(fulltext.queue_path(f.cfg))
    assert len(q) == 2 and not any(e.get("browser_next_try") for e in q)      # no backoff
    notes = [r[0] for r in sqlite3.connect(led.path).execute(
        "select note from attempts where channel='browser' and note like 'cancelled%'")]
    assert len(notes) == 2                                     # not counted towards the failure breaker


def test_wrapper_refuses_unknown_actions_and_honours_stop(tmp_path, monkeypatch, capsys):
    led = DownloadLedger(tmp_path / "l.sqlite")
    bd = tmp_path / "bd"
    (bd / "papers").mkdir(parents=True)
    st = dict(space="sp", papers={"1": dict(doi="10.1080/p", title=TITLE, ident="doi:10.1080/p",
                                            allow=["tandfonline.com"])},
              deadline=time.time() + 300, ledger_path=str(led.path), limits=led.limits, global_limits=led.g,
              host_caps=led.host_caps, ego_binary="/bin/false", gate_max_wait_s=100, max_calls_per_paper=2,
              call_timeout_s=10)
    (bd / "batch.json").write_text(json.dumps(st))
    assert browser_agent.wrapper_main(str(bd), ["1", "type", "secret"]) == 2
    assert browser_agent.wrapper_main(str(bd), ["9", "look"]) == 2
    assert sqlite3.connect(led.path).execute("select count(*) from attempts").fetchone()[0] == 0
    (bd / "STOP").write_text("x")
    assert browser_agent.wrapper_main(str(bd), ["1", "look"]) == 3
    assert "STOP" in capsys.readouterr().out


def test_hook_allowlist():
    ok = ["tbsgen-browser 1 look", "tbsgen-browser 2 open https://doi.org/10.1/x", "tbsgen-browser 3 click 12",
          "tbsgen-browser 1 save_pdf", "tbsgen-browser finish", "sleep 90"]
    bad = ["printenv", "env", "echo $ELSEVIER_API_KEY", "cat ~/.hermes/.env", "curl https://x",
           "tbsgen-browser 1 look; env", "tbsgen-browser 1 open https://x/$(id)", "tbsgen-browser 1 look\nenv",
           "ego-browser nodejs", "sleep 1 && env", "tbsgen-browser 1 open https://sci-hub.se/x", "sleep 9999"]
    assert all(browser_agent.shell_allowed(c) is None for c in ok)
    assert all(browser_agent.shell_allowed(c) for c in bad)


def test_hook_main_blocks_other_tools(tmp_path, monkeypatch, capsys):
    for payload, blocked in (({"tool_name": "terminal", "tool_input": {"command": "tbsgen-browser 1 look"}}, False),
                             ({"tool_name": "terminal", "tool_input": {"command": "env"}}, True),
                             ({"tool_name": "write_file", "tool_input": {"path": "/x"}}, True),
                             ({"tool_name": "terminal", "tool_input": {"command": "sleep 5", "background": True}}, True)):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        browser_agent.hook_main(str(tmp_path), [])
        out = capsys.readouterr().out
        assert ('"block"' in out) == blocked


def test_dry_run_plans_without_side_effects(tmp_path):
    cfg, led, clock = mk_browser(tmp_path)
    bqueue(cfg, {"10.1080": 2, "10.1093": 2})
    run, calls = script_runner(tmp_path)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    probe = Seq(TRACE_OK)
    pr = browser_agent.Prober(f.ft, probe_fn=probe, clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, dry_run=True)
    assert st["dry_run"] and len(st["planned"]) == 4 and calls == [] and probe.urls == []
    assert sqlite3.connect(led.path).execute("select count(*) from attempts").fetchone()[0] == 0


def test_running_script_is_cut_at_the_deadline(tmp_path):
    clock = FakeClock(time.time())
    cfg, led, clock = mk_browser(tmp_path, clock=clock, window_s=120, margin_s=60, est_script_s=10)
    bqueue(cfg, {"10.1080": 1})
    seen = {}

    def run(job):
        if job.get("mode") in ("egress", "probe", "finish"):
            return {"status": "ok"}
        seen["timeout"] = job["cfg"]["script_timeout_s"]
        return {"status": "timeout"}
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK), clock=clock, sleep=clock.sleep)
    browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, log=lambda m: None)
    assert seen["timeout"] <= 60                              # never past the window's deadline


def test_prompt_template_carries_the_hard_rules():
    t = browser_agent.PROMPT.read_text()
    for s in ("CAPTCHA", "credentials", "allowed_domains", "Sci-Hub", "user's own tabs", "$papers_json",
              "tbsgen-browser finish", "$gap_s"):
        assert s in t
    g = browser_agent.ACTION.read_text()
    for s in ("hostOk", "isChallenge", "login form", "verification widget", "files_dir", "finish"):
        assert s in g


def test_allowed_domains_cover_publisher_and_landing():
    d = browser_agent.allowed_domains("10.1093/abc/1", "https://academic.oup.com/x/article/1", ["doi.org"])
    assert "oup.com" in d and "doi.org" in d and "silverchair.com" in d
    assert not any("sci-hub" in x for x in d)


def test_ssl_context_falls_back_to_a_system_bundle(monkeypatch, tmp_path):
    import ssl
    bundle = tmp_path / "ca.pem"
    bundle.write_text("")
    seen = {}
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(ssl, "get_default_verify_paths",
                        lambda: ssl.DefaultVerifyPaths(None, None, "SSL_CERT_FILE", "/nonexistent", "SSL_CERT_DIR", "/x"))
    monkeypatch.setattr(fulltext, "SYSTEM_CA_BUNDLES", ("/nonexistent/a.pem", str(bundle)))
    monkeypatch.setattr(ssl, "create_default_context", lambda cafile=None: seen.setdefault("cafile", cafile))
    fulltext.ssl_context()
    assert seen["cafile"] == str(bundle)


def test_redirect_location_with_raw_spaces_is_encoded(tmp_path):
    from test_fulltext import mkfetcher
    f, net, led, clock = mkfetcher(tmp_path, {
        "repo.example/a.pdf": Resp(302, {"Location": "/b c\t.pdf?x=1 2"}, b"", ""),
        "repo.example/b%20c%09.pdf": FAKE_PDF})
    rec = dict(tried=[], errors=[], skipped=[])
    r = f._get("https://repo.example/a.pdf", {}, rec)
    assert r.status == 200 and "b%20c%09.pdf?x=1%202" in net.calls[-1][0]


def test_user_provider_reader_without_yaml(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                        lambda n, *a, **k: (_ for _ in ()).throw(ImportError()) if n == "yaml" else real_import(n, *a, **k))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  default: x\nproviders:\n  other:\n    api: http://o\n  mine:\n    name: Local (:1)\n"
                   "    api: http://127.0.0.1:1/v1\n    api_key: k\n    discover_models: false\n    models:\n"
                   "      m1:\n        context_length: 4000\n  third:\n    api: http://t\nother_top: 1\n")
    e = browser_agent._user_provider(cfg, "mine")
    assert e == {"name": "Local (:1)", "api": "http://127.0.0.1:1/v1", "api_key": "k", "discover_models": False,
                 "models": {"m1": {"context_length": 4000}}}


def test_recovered_text_is_booked_as_success(tmp_path):
    """Worker died after saving the text but before booking it: the recovered result and the ledger agree."""
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML},
                                        limits={"elsevier": {"daily_ok": 1, "gap_s": 0.0, "jitter_s": 0.0}})
    queue(f, ELS)
    ssh = FakeSSH(rdir, net)
    real = ssh.__call__
    orig_record = DownloadLedger.record

    def runner(cmd, stdin=b"", timeout=300):
        if "submit" in cmd:
            def crash(self, channel, host, doi, status, *a, **k):
                if status == "ok":
                    raise SystemExit("worker killed after saving")
                return orig_record(self, channel, host, doi, status, *a, **k)
            DownloadLedger.record = crash
            try:
                return real(cmd, stdin, timeout)
            except SystemExit:
                return 0, b'{"ok": true}\n', b""
            finally:
                DownloadLedger.record = orig_record
                job = cmd.split()[-1]
                sp = rdir / "jobs" / job / "status.json"
                st = json.loads(sp.read_text())
                st.update(state="running", ts=time.time() - 120)
                sp.write_text(json.dumps(st))
        return real(cmd, stdin, timeout)
    st = drain(f, runner)
    assert st["ok"] == 1
    db = sqlite3.connect(led.path)
    assert db.execute("select count(*) from attempts where channel='elsevier' and status='ok'").fetchone()[0] == 1
    assert led.remaining("elsevier") == 0                       # the cap is not exceeded by the recovery


def test_cooldowns_flow_both_ways_while_a_job_runs(tmp_path):
    f, net, _, led, rdir, _ = mk_remote(tmp_path, {"api.elsevier.com": ELS_XML})
    d = rdir / "jobs" / "j1"
    (d / "results").mkdir(parents=True)
    DownloadLedger(d / "ledger.sqlite").block("api.wiley.com", "remote 429", 3600)
    (d / "status.json").write_text(json.dumps(dict(state="done", ts=time.time())))
    (d / "spec.json").write_text(json.dumps(dict(entries=[], marks={})))
    led.block("www.tandfonline.com", "local challenge", 3600)
    remote.jobs_dir(f.cfg).mkdir(parents=True, exist_ok=True)
    (remote.jobs_dir(f.cfg) / "j1.json").write_text(json.dumps(dict(job="j1", stage="submitted")))
    remote.collect(f, "j1", FakeSSH(rdir, net), log=lambda m: None, poll_s=0, sleep=lambda s: None)
    assert led.cooling("api.wiley.com")                          # remote -> local
    pushed = json.loads((rdir / "jobs" / "j1" / "cooldown_push.json").read_text()) if \
        (rdir / "jobs" / "j1").exists() else None
    assert pushed is None or any(h == "www.tandfonline.com" for h, *_ in pushed)


def test_hook_config_fails_closed_and_quotes_paths(tmp_path):
    bd = tmp_path / "dir with space"
    bd.mkdir()
    home = browser_agent._hermes_home(bd, {"provider": "p", "model": "m",
                                           "provider_config": {"api": "http://x/v1", "api_key": "SECRET-K",
                                                               "extra_headers": {"Authorization": "Bearer SECRET-H"}}},
                                      str(bd / "bin" / "tbsgen-hook"))
    txt = (home / "config.yaml").read_text()
    assert "fail_closed: true" in txt and "SECRET-K" not in txt and "SECRET-H" not in txt
    import shlex
    cmd = json.loads([ln.split("command:", 1)[1] for ln in txt.splitlines() if "command:" in ln][0])
    assert shlex.split(cmd) == [str(bd / "bin" / "tbsgen-hook")]


def test_refused_batch_closes_no_taskspace_and_releases_the_lock(tmp_path, monkeypatch):
    cfg, led, clock = mk_browser(tmp_path)
    bqueue(cfg, {"10.1080": 1})
    closed = []
    monkeypatch.setattr(browser_agent.browser_ego, "finish", lambda c: closed.append(c.get("space")) or {})
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}))
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_SG), clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, log=lambda m: None)
    assert st["refused"] and closed == [] and not f._space_held
    f2 = Fetcher(cfg, ledger=led, transport=FakeNet({}))
    assert f2._try_space_lock()                                 # the lock was released


def test_no_script_starts_after_stop_or_too_close_to_deadline(tmp_path):
    clock = FakeClock(time.time())
    cfg, led, clock = mk_browser(tmp_path, clock=clock, window_s=100, margin_s=60, est_script_s=5,
                                 min_script_s=20)
    bqueue(cfg, {"10.1080": 2})                                  # same publisher: the 2nd waits 60-120 s
    run, calls = script_runner(tmp_path)
    f = Fetcher(cfg, ledger=led, transport=FakeNet({}), browser_runner=run)
    f.ft["limits"]["browser"].update(gap_s=60.0, jitter_s=0.0)
    led.limits["browser"].update(gap_s=60.0, jitter_s=0.0)
    pr = browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK), clock=clock, sleep=clock.sleep)
    st = browser_agent.run_batch(f, prober=pr, clock=clock, sleep=clock.sleep, log=lambda m: None)
    assert len(calls) == 1 and st["deferred"] >= 1               # the 2nd paper's slot is past the deadline


def test_rejected_or_unreported_files_are_never_validated(tmp_path, monkeypatch):
    f, led, pr, calls = mk_agent(tmp_path, monkeypatch, ego_mode="rejected")
    st = browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)
    assert st["ok"] == 0 and not list(Path(f.ft["write_cache_dir"]).glob("*.txt"))


def test_recovery_by_reservation_is_idempotent_and_doi_less(tmp_path):
    """Recovery books by reservation id (no DOI needed); a second export after an interrupted one still books."""
    d = tmp_path / "jobs" / "j"
    (d / "results").mkdir(parents=True)
    (d / "cache").mkdir()
    led = DownloadLedger(d / "ledger.sqlite", limits={"oa": {"gap_s": 0.0, "jitter_s": 0.0}})
    rid = led.acquire("oa", "repo.example", "title:abc")
    e = dict(title=TITLE, oa_url="https://repo.example/a.pdf")              # no DOI
    ident = fulltext.paper_ident(e)
    (d / "cache" / "k.json").write_text(json.dumps(dict(ident=ident, channel="oa", reservation=rid)))
    (d / "cache" / "k.txt").write_text(TITLE + " " + BODY)
    (d / "spec.json").write_text(json.dumps(dict(entries=[e], marks={})))
    (d / "status.json").write_text(json.dumps(dict(state="running", ts=0)))
    # first export: recovery result written, then "crash" before booking -> simulate by writing result only
    remote._recover_results(d, json.loads((d / "spec.json").read_text()))
    assert (d / "results" / "0000.json").exists()
    buf = io.BytesIO()
    remote.cmd_export(str(tmp_path), "j", buf)                               # second export still books it
    db = sqlite3.connect(d / "ledger.sqlite")
    assert db.execute("select status from attempts where rowid=?", (rid,)).fetchone()[0] == "ok"


def test_agent_space_cannot_be_overridden():
    assert browser_agent.agent_space({"space": "a", "agent": {"space": "shared"}}) == "a-agent"
