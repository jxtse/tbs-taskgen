"""Regression tests for the download queue's limit semantics, not-entitled TDM answers, `fetch --only`, remote
refusals / deployment, and the TDM purge.  Offline: HTTP, ssh and scp are faked."""
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from tbsgen import cli, fulltext, remote
from tbsgen.fulltext import Resp

from test_fulltext import (BODY, ELS_ABSTRACT_ONLY, ELS_XML, FAKE_PDF, SECRETS, TITLE, FakeClock, mkfetcher)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    for k in ("ELSEVIER_API_KEY", "ELSEVIER_INSTTOKEN", "WILEY_TDM_TOKEN"):
        monkeypatch.delenv(k, raising=False)


def q(f, *cands):
    qp = fulltext.queue_path(f.cfg)
    for c in cands:
        fulltext.enqueue(qp, fulltext.queue_entry(c, subscribers=[["r", c["doi"]]]))
    return qp


def entry(qp, doi):
    return next(e for e in fulltext.read_queue(qp) if e.get("doi") == doi)


def drain(f, clock, **kw):
    return fulltext.drain_queue(f, browser=False, on_success=lambda e, o, s: True, log=lambda s: None,
                                now=clock.t, **kw)


# ------------------------------------------------------------------ A1: a refusal is not a failure
def test_channel_cap_does_not_charge_a_try_and_is_due_when_it_lifts(tmp_path):
    """Hourly cap reached: the paper is skipped without a request; no try, no 6 h backoff; due at the reset."""
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML}, SECRETS,
                                   limits={"elsevier": {"hourly_ok": 1, "gap_s": 0, "jitter_s": 0}},
                                   oa={"enabled": False})
    qp = q(f, dict(doi="10.1016/j.a.1", title=TITLE), dict(doi="10.1016/j.a.2", title=TITLE))
    st = drain(f, clock)
    assert st["ok"] == 1 and st["limited"] == 1 and st["failed"] == 0
    e = entry(qp, "10.1016/j.a.2")
    assert not e.get("tries") and not e.get("chan_backoff")
    assert clock.t < e["next_try"] <= clock.t + 3600 + 1          # the hourly window, not queue_backoff_s (6 h)
    assert len([u for u, _ in net.calls if "elsevier" in u]) == 1   # the capped paper made no request
    clock.t = e["next_try"] + 1
    st = drain(f, clock)
    assert st["ok"] == 1 and fulltext.read_queue(qp) == []


def test_host_cooldown_skip_is_not_a_failure(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {}, SECRETS, oa={"enabled": True})
    led.block("repo.example", "test block", seconds=7200)
    qp = q(f, dict(doi="10.9999/x.1", title=TITLE, oa_url="https://repo.example/x.pdf"))
    st = drain(f, clock)
    assert st["limited"] == 1 and st["failed"] == 0 and net.calls == []
    e = entry(qp, "10.9999/x.1")
    assert not e.get("tries") and abs(e["next_try"] - (clock.t + 7200)) < 2


def test_breaker_skip_waits_until_midnight_not_backoff(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML}, SECRETS, oa={"enabled": False})
    for i in range(8):                                    # 8 consecutive failures open the elsevier breaker
        led.record("elsevier", "api.elsevier.com", f"x{i}", "fail", 500, 0, "HTTP 500")
    qp = q(f, dict(doi="10.1016/j.b.1", title=TITLE))
    st = drain(f, clock)
    assert st["limited"] == 1 and net.calls == []
    e = entry(qp, "10.1016/j.b.1")
    from tbsgen.dlledger import next_midnight
    assert not e.get("tries") and e["next_try"] == pytest.approx(next_midnight(clock.t))


def test_real_failure_still_backs_off(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": Resp(500, {}, b"err", "u")}, SECRETS,
                                   oa={"enabled": False})
    qp = q(f, dict(doi="10.1016/j.c.1", title=TITLE))
    st = drain(f, clock)
    assert st["failed"] == 1
    e = entry(qp, "10.1016/j.c.1")
    assert e["tries"] == 1 and e["next_try"] == pytest.approx(clock.t + 21600)


def test_limited_channel_does_not_reset_backoff_of_a_failed_one(tmp_path):
    """elsevier failed (backs off per channel); oa refused by a cooldown: the entry is due when oa's cooldown
    lifts, and then elsevier is skipped (still backing off) instead of being asked again."""
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": Resp(500, {}, b"err", "u")}, SECRETS,
                                   oa={"enabled": True})
    led.block("repo.example", "test", seconds=1800)
    qp = q(f, dict(doi="10.1016/j.d.1", title=TITLE, oa_url="https://repo.example/d.pdf"))
    drain(f, clock)
    e = entry(qp, "10.1016/j.d.1")
    assert not e.get("tries") and e["chan_backoff"]["elsevier"]["tries"] == 1
    assert e["next_try"] == pytest.approx(clock.t + 1800, abs=2)
    n = len(net.calls)
    clock.t += 1801
    net.routes["repo.example"] = FAKE_PDF
    st = drain(f, clock)
    assert st["ok"] == 1 and all("elsevier" not in u for u, _ in net.calls[n:])


def test_refusal_without_known_end_is_rechecked_soon(tmp_path):
    """Egress refused (no end time) next to a cap that lifts at midnight: due again after LIMIT_RECHECK_S."""
    f, net, led, clock = mkfetcher(tmp_path, {}, SECRETS, oa={"enabled": True},
                                   egress_check={"enabled": True, "allow_cidrs": ["198.51.100.0/24"],
                                                 "urls": ["https://echo.example/ip"]})
    net.routes["echo.example"] = b"203.0.113.9"
    qp = q(f, dict(doi="10.1016/j.u.1", title=TITLE, oa_url="https://repo.example/u.pdf"))
    st = drain(f, clock)
    assert st["limited"] == 1 and st["failed"] == 0
    e = entry(qp, "10.1016/j.u.1")
    assert not e.get("tries") and e["next_try"] == pytest.approx(clock.t + fulltext.LIMIT_RECHECK_S)


def test_new_link_resets_channel_backoff(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"repo.example/old": Resp(404, {}, b"", "u")}, SECRETS)
    qp = q(f, dict(doi="10.9999/n.1", title=TITLE, oa_url="https://repo.example/old.pdf"))
    drain(f, clock)
    assert entry(qp, "10.9999/n.1")["chan_backoff"]["oa"]["tries"] == 1
    fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.9999/n.1", title=TITLE, oa_url="https://repo.example/new.pdf"),
                                              subscribers=[["r", "x"]]))
    e = entry(qp, "10.9999/n.1")
    assert not e.get("chan_backoff") and e["next_try"] == 0
    net.routes["repo.example/new"] = FAKE_PDF
    assert drain(f, clock)["ok"] == 1


def test_failure_of_old_link_does_not_restore_backoff_after_new_link(tmp_path):
    """A new OA link enqueued while the old one is being fetched: the old link's failure must not put the new
    one behind the channel backoff (Codex r2)."""
    f, net, led, clock = mkfetcher(tmp_path, {}, SECRETS)
    qp = q(f, dict(doi="10.9999/m.1", title=TITLE, oa_url="https://repo.example/old.pdf"))

    def old_fails_and_new_link_arrives(url):
        fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.9999/m.1", title=TITLE,
                                                       oa_url="https://repo.example/new.pdf"), subscribers=[]))
        return Resp(404, {}, b"", url)
    net.routes["repo.example/old"] = old_fails_and_new_link_arrives
    drain(f, clock)
    e = entry(qp, "10.9999/m.1")
    assert not e.get("chan_backoff") and e["next_try"] == 0 and not e.get("tries")
    net.routes["repo.example/new"] = FAKE_PDF
    assert drain(f, clock)["ok"] == 1


def test_old_queue_format_item_ids_still_delivered(tmp_path):
    """Queue entries written before `subscribers` (run + item_ids) keep notifying their items."""
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML}, SECRETS, oa={"enabled": False})
    qp = fulltext.queue_path(f.cfg)
    qp.parent.mkdir(parents=True, exist_ok=True)
    qp.write_text(json.dumps(dict(doi="10.1016/j.l.1", title=TITLE, ident="doi:10.1016/j.l.1", run="/runs/A",
                                  item_ids=["i1", "i2"])) + "\n")
    got = []
    fulltext.drain_queue(f, browser=False, on_success=lambda e, o, s: got.append(s) or True, log=lambda s: None,
                         now=clock.t)
    assert got == [["/runs/A", "i1"], ["/runs/A", "i2"]] and fulltext.read_queue(qp) == []


# ------------------------------------------------------------------ A3: not entitled
def test_elsevier_short_text_zero_is_not_entitled_and_never_asked_again(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": ELS_ABSTRACT_ONLY}, SECRETS, oa={"enabled": False})
    qp = q(f, dict(doi="10.1016/j.e.1", title=TITLE))
    st = drain(f, clock)
    assert st["failed"] == 1
    assert led.not_entitled("elsevier", "10.1016/j.e.1")
    row = sqlite3.connect(led.path).execute("select note from attempts where channel='elsevier'").fetchone()
    assert row[0].startswith("not_entitled: short text 0")
    n = len(net.calls)
    clock.t += 30 * 86400
    fulltext.drain_queue(f, browser=False, log=lambda s: None, now=clock.t)
    assert len(net.calls) == n                                       # never asked again
    out = f.fetch(dict(doi="10.1016/j.e.1", title=TITLE))
    assert any("not entitled" in s for s in out.rec["skipped"]) and len(net.calls) == n


def test_wiley_short_text_is_not_entitled_and_ttl_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: "Abstract only. " * 20)
    f, net, led, clock = mkfetcher(tmp_path, {"api.wiley.com": FAKE_PDF}, SECRETS, oa={"enabled": False},
                                   not_entitled={"ttl_days": 10})
    out = f.fetch(dict(doi="10.1002/abc.9", title=TITLE))
    assert out.text is None and out.rec["not_entitled"] == ["wiley"] and "not_entitled: short text" in out.rec["errors"][-1]
    n = len(net.calls)
    assert f.fetch(dict(doi="10.1002/abc.9", title=TITLE)).text is None and len(net.calls) == n
    clock.t += 11 * 86400                                            # ttl_days passed: ask once more
    f.fetch(dict(doi="10.1002/abc.9", title=TITLE))
    assert len(net.calls) == n + 1
    f.fetch(dict(doi="10.1002/abc.9", title=TITLE))                  # the new answer renewed the record
    assert len(net.calls) == n + 1


def test_not_entitled_does_not_trip_the_breaker(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": ELS_ABSTRACT_ONLY}, SECRETS, oa={"enabled": False})
    for i in range(10):
        f.fetch(dict(doi=f"10.1016/j.f.{i}", title=TITLE))
    assert led.breaker("elsevier") is None


def test_not_entitled_can_be_disabled(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": ELS_ABSTRACT_ONLY}, SECRETS, oa={"enabled": False},
                                   not_entitled={"enabled": False})
    f.fetch(dict(doi="10.1016/j.g.1", title=TITLE))
    assert led.not_entitled("elsevier", "10.1016/j.g.1") is None


def test_oa_short_text_is_not_marked_not_entitled(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: "short " * 10)
    f, net, led, clock = mkfetcher(tmp_path, {"repo.example": FAKE_PDF}, SECRETS)
    out = f.fetch(dict(doi="10.9999/h.1", title=TITLE, oa_url="https://repo.example/h.pdf"))
    assert out.text is None and not out.rec.get("not_entitled")
    assert led.not_entitled("oa", "10.9999/h.1") is None


# ------------------------------------------------------------------ A4: --only
def test_fetch_only_processes_just_the_named_entries(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML}, SECRETS, oa={"enabled": False})
    qp = q(f, dict(doi="10.1016/j.o.1", title=TITLE), dict(doi="10.1016/j.o.2", title=TITLE),
           dict(doi="10.1016/j.o.3", title=TITLE, arxiv="2101.00001"))
    e2 = entry(qp, "10.1016/j.o.2")
    e2["next_try"] = clock.t + 9999                                  # not due: --only still takes it
    fulltext.write_queue(qp, [e2 if x.get("doi") == e2["doi"] else x for x in fulltext.read_queue(qp)])
    st = drain(f, clock, only=["https://doi.org/10.1016/J.O.2", "arxiv:2101.00001", "doi:10.9/missing"])
    assert st["considered"] == 2 and st["ok"] == 2 and st["only_unmatched"] == ["doi:10.9/missing"]
    assert [e["doi"] for e in fulltext.read_queue(qp)] == ["10.1016/j.o.1"]


def test_cli_fetch_only_with_browser_is_refused(tmp_path, monkeypatch, capsys):
    cfgp = tmp_path / "c.toml"
    cfgp.write_text(f'[fulltext]\nqueue_path = "{tmp_path / "q.jsonl"}"\nledger_path = "{tmp_path / "l.sqlite"}"\n')
    from tbsgen import browser_agent
    monkeypatch.setattr(browser_agent, "run_batch", lambda *a, **k: pytest.fail("batch must not run"))
    rc = cli.main(["fetch", "--run", str(tmp_path / "run"), "--config", str(cfgp), "--browser", "--only", "10.1/a"])
    assert rc == 2 and "--only is not supported with --browser" in capsys.readouterr().err


def test_cli_fetch_only_and_exit_codes(tmp_path, monkeypatch, capsys):
    cfgp = tmp_path / "c.toml"
    cfgp.write_text(f'[fulltext]\nqueue_path = "{tmp_path / "q.jsonl"}"\nledger_path = "{tmp_path / "l.sqlite"}"\n'
                    f'allow_network = false\n')
    fulltext.enqueue(tmp_path / "q.jsonl", fulltext.queue_entry(dict(doi="10.1/a", title="t"), subscribers=[]))
    seen = {}

    def fake_drain(f, **kw):
        seen.update(kw)
        return dict(considered=0, only_unmatched=["10.1/zz"])
    monkeypatch.setattr(fulltext, "drain_queue", fake_drain)
    rc = cli.main(["fetch", "--run", str(tmp_path / "run"), "--config", str(cfgp), "--only", "10.1/zz"])
    assert rc == 4 and seen["only"] == ["10.1/zz"] and seen["max_papers"] == 0
    monkeypatch.setattr(fulltext, "drain_queue", lambda f, **kw: dict(remote_error="not submitted: mismatch"))
    rc = cli.main(["fetch", "--run", str(tmp_path / "run"), "--config", str(cfgp)])
    assert rc == 3 and "remote job refused" in capsys.readouterr().err


# ------------------------------------------------------------------ A5: preflight, deploy, status
def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / remote.SRC_REL).mkdir(parents=True)
    (r / remote.SRC_REL / "x.py").write_text("x = 1\n")
    _git(r, "init", "-q")
    _git(r, "add", ".")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "init")
    return r


class FakeRun:
    def __init__(self, info=None, fail_on=None):
        self.cmds, self.info, self.fail_on = [], info, fail_on

    def __call__(self, cmd, stdin=b"", timeout=300):
        self.cmds.append(cmd)
        if self.fail_on and self.fail_on in cmd:
            return 1, b"", b"boom"
        if "tbsgen.remote info" in cmd:
            return 0, json.dumps(self.info or {}).encode(), b""
        return 0, b"", b""


def _rf(tmp_path, **rc):
    f, net, led, clock = mkfetcher(tmp_path, {}, SECRETS, remote=dict(
        dict(enabled=True, host="h", python="/opt/py", workdir="~/w", code_dir="~/code"), **rc))
    return f


def test_deploy_dry_run_plans_and_touches_nothing(tmp_path):
    f, r = _rf(tmp_path), _repo(tmp_path)
    run, copied = FakeRun(), []
    out = remote.deploy(f, dry_run=True, repo=r, ssh_runner=run, scp_runner=lambda *a: copied.append(a))
    assert out["ok"] and out["dry_run"] and out["sha"] == _git(r, "rev-parse", "HEAD")
    assert run.cmds == [] and copied == []
    assert any(f"checkout -q --detach {out['sha']}" in c for c in out["plan"])
    assert any("uv sync" in c for c in out["plan"]) and "~/code" in out["plan"][1]


def test_deploy_refuses_dirty_tree(tmp_path):
    f, r = _rf(tmp_path), _repo(tmp_path)
    (r / remote.SRC_REL / "x.py").write_text("x = 2\n")
    out = remote.deploy(f, dry_run=True, repo=r)
    assert not out["ok"] and "uncommitted" in out["error"]


def test_deploy_runs_plan_and_checks_fingerprint(tmp_path, monkeypatch):
    f, r = _rf(tmp_path), _repo(tmp_path)
    sha = _git(r, "rev-parse", "HEAD")
    copied = []
    run = FakeRun(info=dict(fingerprint=remote.code_fingerprint(), head=sha, jobs=[]))
    out = remote.deploy(f, repo=r, ssh_runner=run, scp_runner=lambda local, dst, timeout=600: (
        copied.append((Path(local).stat().st_size > 0, dst)) or (0, b"", b"")), log=lambda s: None)
    assert out["ok"], out
    assert copied == [(True, "~/code.deploy.bundle")]
    assert [c for c in run.cmds if "git fetch" in c or "uv sync" in c]
    # the remote answers with another fingerprint after the deploy: not ok
    run = FakeRun(info=dict(fingerprint="other", head=sha, jobs=[]))
    out = remote.deploy(f, repo=r, ssh_runner=run, scp_runner=lambda *a, **k: (0, b"", b""), log=lambda s: None)
    assert not out["ok"] and "differs" in out["error"]


def test_deploy_refuses_while_remote_job_outstanding(tmp_path):
    f, r = _rf(tmp_path), _repo(tmp_path)
    run = FakeRun()
    run_ls = lambda cmd, stdin=b"", timeout=300: (run.cmds.append(cmd), (0, b"job-1\n", b""))[1] \
        if cmd.startswith("ls ") else run(cmd, stdin, timeout)
    out = remote.deploy(f, repo=r, ssh_runner=run_ls, scp_runner=lambda *a, **k: (0, b"", b""))
    assert not out["ok"] and "job directories" in out["error"]


def test_preflight_refuses_dirty_tree_before_contacting_the_remote(tmp_path, monkeypatch):
    f = _rf(tmp_path, require_clean_tree=True)
    monkeypatch.setattr(remote, "local_code_state", lambda: dict(repo="/r", head="a", dirty=["gen/src/tbsgen/x.py"],
                                                                  fingerprint="fp"))
    run = FakeRun()
    why = remote.preflight(f, run)
    assert "uncommitted" in why and "tbsgen remote deploy" in why and run.cmds == []


def test_preflight_old_remote_without_info(tmp_path):
    f = _rf(tmp_path, require_clean_tree=False)
    why = remote.preflight(f, lambda cmd, stdin=b"", timeout=300: (2, b"", b"usage: python -m tbsgen.remote"))
    assert "remote `info` failed" in why and "tbsgen remote deploy" in why


def test_remote_status_reports_both_sides(tmp_path):
    f = _rf(tmp_path)
    run = FakeRun(info=dict(fingerprint=remote.code_fingerprint(), head="abc",
                            jobs=[dict(job="j1", state="running", done=1, n=3)]))
    out = remote.status(f, ssh_runner=run)
    assert out["same_code"] and out["running"] == [dict(job="j1", state="running", done=1, n=3)]


# ------------------------------------------------------------------ A6: lease counting on recall
def test_reused_fetcher_does_not_consume_earlier_remote_results(tmp_path, monkeypatch):
    """Round 1 collects a failed remote result; round 2 is refused by the preflight: the paper must not be
    charged a try from round 1's stale result."""
    f, net, led, clock = mkfetcher(tmp_path, {}, SECRETS, remote=dict(enabled=True, host="h", python="/p",
                                                                         workdir="/w"))
    qp = q(f, dict(doi="10.1016/j.s.1", title=TITLE))
    ident = "doi:10.1016/j.s.1"
    rounds = iter([dict(results={ident: dict(ok=False, text=None, rec=dict(tried=["elsevier:u"], errors=["x"]))},
                        jobs=[]),
                   dict(results={}, jobs=[], refused="code fingerprint mismatch")])
    monkeypatch.setattr(remote, "run_job", lambda *a, **k: next(rounds))
    st1 = drain(f, clock)
    assert st1["failed"] == 1 and entry(qp, "10.1016/j.s.1")["chan_backoff"]["elsevier"]["tries"] == 1
    before = entry(qp, "10.1016/j.s.1")
    clock.t += 7 * 86400
    st2 = drain(f, clock, only=["10.1016/j.s.1"])
    assert st2["remote_error"] and st2["not_tried"] == 1 and st2["failed"] == 0
    assert entry(qp, "10.1016/j.s.1") == before


def test_remote_not_entitled_survives_worker_crash(tmp_path):
    """A worker that booked not_entitled and died before writing its result: the export carries the table with the
    ledger rows and the local import merges it (Codex r2)."""
    import io
    from tbsgen.dlledger import DownloadLedger
    f, net, led, clock = mkfetcher(tmp_path, {}, SECRETS)
    d = tmp_path / "jobs" / "rjob"
    d.mkdir(parents=True)
    (d / "results").mkdir()
    DownloadLedger(d / "ledger.sqlite").mark_not_entitled("wiley", "10.1002/zz", "not_entitled: short text 5")
    (d / "spec.json").write_text(json.dumps(dict(entries=[], marks={})))
    (d / "status.json").write_text(json.dumps(dict(state="dead")))
    buf = io.BytesIO()
    assert remote.cmd_export(str(tmp_path), "rjob", buf) == 0
    res = remote.import_export(f, "rjob", buf.getvalue())
    assert res["merged"]["not_entitled"] == 1 and led.not_entitled("wiley", "10.1002/zz")


def test_lease_rows_never_count_as_attempts_or_breaker(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {}, SECRETS)
    remote.lease_and_snapshot(led, "j", 3)
    assert led.usage()["elsevier"]["attempts"] == 0 and led.breaker("elsevier") is None
    assert led.usage()["_global_ok"] == 0                            # pending leases are not successes
    remote.release_lease(led, "j")
    assert sqlite3.connect(led.path).execute("select count(*) from attempts").fetchone()[0] == 0


def test_reservation_cancelled_before_sending_uses_no_attempt_cap(tmp_path):
    """A cooldown set while a reservation waited cancels it before anything is sent: that row must not use up the
    channel's daily attempts or the host's caps (it used to, so a cooled host could exhaust a day's attempts)."""
    from tbsgen.dlledger import DownloadLedger, LimitReached
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"inst": {"daily_attempts": 2, "host_daily_attempts": 2, "gap_s": 0,
                                                          "jitter_s": 0}}, clock=clock, sleep=clock.sleep)
    orig = led.sleep

    def sleep_and_block(s):
        led.block("pub.example", "challenge", seconds=60)
        orig(s)
    for i in range(3):
        led.hosts_next = None
        led.db.execute("insert or replace into hosts values('pub.example', ?)", (clock.t + 5,))
        led.sleep = sleep_and_block
        with pytest.raises(LimitReached):
            led.acquire("inst", "pub.example", f"d{i}")
        led.sleep = orig
        clock.t += 61
    assert led.usage()["inst"]["attempts"] == 0
    rid = led.acquire("inst", "pub.example", "d9")                  # caps still free
    led.record("inst", "pub.example", "d9", "ok", 200, 1, reservation=rid)


# ------------------------------------------------------------------ B: provenance licence + purge
def _fetch_three(tmp_path):
    routes = {"api.elsevier.com": ELS_XML, "api.wiley.com": FAKE_PDF, "repo.example": FAKE_PDF}
    f, net, led, clock = mkfetcher(tmp_path, routes, SECRETS)
    for c in (dict(doi="10.1016/j.p.1", title=TITLE), dict(doi="10.1002/p.2", title=TITLE),
              dict(doi="10.9999/p.3", title=TITLE, oa_url="https://repo.example/p3.pdf")):
        assert f.fetch(c).text
    return f, led


def test_provenance_records_channel_and_licence(tmp_path):
    f, _ = _fetch_three(tmp_path)
    w = Path(f.cfg["fulltext"]["write_cache_dir"])
    lic = {json.loads(p.read_text())["channel"]: json.loads(p.read_text())["licence"] for p in w.glob("*.json")}
    assert lic == {"elsevier": "elsevier_tdm", "wiley": "wiley_tdm", "oa": "oa"}


def test_purge_tdm_channels_keeps_ledger_and_other_texts(tmp_path):
    f, led = _fetch_three(tmp_path)
    w = Path(f.cfg["fulltext"]["write_cache_dir"])
    run = tmp_path / "run"
    for name, ch, stage in (("it-els", "elsevier", "S9"), ("it-active", "wiley", "S7")):
        d = run / "items" / name
        d.mkdir(parents=True)
        (d / "fulltext.txt").write_text(BODY)
        (d / "S3_fulltext.json").write_text(json.dumps(dict(source=ch)))
        (d / "state.json").write_text(json.dumps(dict(stage=stage, status="active")))
    n_att = sqlite3.connect(led.path).execute("select count(*) from attempts").fetchone()[0]
    dry = fulltext.purge(f.cfg, ["elsevier", "wiley"], dry_run=True, runs=[run])
    assert dry["texts"] == 2 and dry["raw"] == 2 and dry["items"] == 1 and dry["items_active"] == ["it-active"]
    assert len(list(w.glob("*.txt"))) == 3                            # dry run deletes nothing
    out = fulltext.purge(f.cfg, ["elsevier", "wiley"], runs=[run])
    assert out["texts"] == 2
    left = {json.loads(p.read_text())["channel"]: p for p in w.glob("*.json")}
    assert sorted(left) == ["elsevier", "oa", "wiley"]               # provenance kept (no article text in it)
    assert json.loads(left["elsevier"].read_text())["purged_at"]
    assert [p.name for p in w.glob("*.txt")] == [left["oa"].with_suffix(".txt").name]
    assert [p.stem for p in (w / "raw").iterdir()] == [left["oa"].stem]   # only the OA raw file is left
    assert not (run / "items" / "it-els" / "fulltext.txt").exists()
    assert (run / "items" / "it-active" / "fulltext.txt").exists()
    assert sqlite3.connect(led.path).execute("select count(*) from attempts").fetchone()[0] == n_att
    # purged texts are not served from the cache any more
    assert f.cached(dict(doi="10.1016/j.p.1", title=TITLE)) is None
    # idempotent
    assert fulltext.purge(f.cfg, ["elsevier", "wiley"], runs=[run])["texts"] == 0


def test_purge_raw_match_is_exact_stem(tmp_path):
    """Purging 10.1016/a must not delete the raw file of another paper whose key starts the same (Codex r2)."""
    f, net, led, clock = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML, "repo.example": FAKE_PDF}, SECRETS)
    assert f.fetch(dict(doi="10.1016/a", title=TITLE)).text
    assert f.fetch(dict(doi="10.9999/x", title=TITLE, oa_url="https://repo.example/x.pdf")).text
    w = Path(f.cfg["fulltext"]["write_cache_dir"])
    (w / "raw" / "10.1016_a.b.pdf").write_bytes(FAKE_PDF)          # another paper's raw file, same prefix
    out = fulltext.purge(f.cfg, ["elsevier"])
    assert out["raw"] == 1 and (w / "raw" / "10.1016_a.b.pdf").exists() and not (w / "raw" / "10.1016_a.xml").exists()


def test_purge_before_date_and_bad_args(tmp_path):
    f, _ = _fetch_three(tmp_path)
    assert fulltext.purge(f.cfg, ["elsevier"], before="2000-01-01")["texts"] == 0
    assert fulltext.purge(f.cfg, ["elsevier"], before="2999-01-01", dry_run=True)["texts"] == 1
    assert "unknown channel" in fulltext.purge(f.cfg, ["springer"])["error"]
    assert "YYYY-MM-DD" in fulltext.purge(f.cfg, ["elsevier"], before="yesterday")["error"]


def test_purge_reports_but_never_writes_readonly_caches(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    (ro / "10.1016_j.r.1.txt").write_text(BODY)
    (ro / "10.1016_j.r.1.json").write_text(json.dumps(dict(doi="10.1016/j.r.1", channel="elsevier",
                                                           fetched_at="2026-10-22T03:30:38")))
    f, net, led, clock = mkfetcher(tmp_path, {}, SECRETS, cache_dirs=[str(ro)])
    out = fulltext.purge(f.cfg, ["elsevier"])
    assert out["readonly_matches"] == 1 and (ro / "10.1016_j.r.1.txt").exists()
