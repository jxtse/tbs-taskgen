"""Offline worker, admission and history tests."""
import ast
import io
import json
import os
import shlex
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tbsgen import budget, cli, fulltext, remote, stages_b
from tbsgen.config import load_config
from tbsgen.pipeline import HANDLERS, Pipeline, parse_stages
from tbsgen.report import throughput

from test_calib_s9 import _env, _finish, _runs
from test_codeagent import _agent_item
from test_fulltext import BODY, FAKE_PDF, TITLE, FakeClock
from test_remote_browser import mk_remote, queue


def pipeline(tmp_path, **cfg):
    return Pipeline(load_config(None, {"run": {"root": str(tmp_path)}, **cfg}))


@pytest.mark.parametrize("stage", ["S4c", "S9"])
def test_two_workers_launch_once(tmp_path, monkeypatch, stage):
    if stage == "S9":
        p, ids, counter, _ = _env(tmp_path, n_items=2)
    else:
        p, item, launcher = _agent_item(tmp_path, monkeypatch, {})
        ids = [item.item_id]
    other = Pipeline(p.cfg)
    for iid in ids:
        started, release = threading.Event(), threading.Event()
        handler = HANDLERS[stage]

        def hold(worker, it):
            started.set()
            assert release.wait(5)
            handler(worker, it)

        monkeypatch.setitem(HANDLERS, stage, hold)
        a, b = p.store.item(iid), other.store.item(iid)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(p.step, a)
            assert started.wait(5)
            try:
                assert pool.submit(other.step, b).result(2) == "skipped"
            finally:
                release.set()
            assert first.result(5) == "ok"
        assert other.step(b) == "skipped"  # old pass snapshot, after the outer lock was released
        monkeypatch.setitem(HANDLERS, stage, handler)
    launches = len(counter.read_text().splitlines()) if stage == "S9" else len(launcher.calls)
    assert launches == len(ids) == p.ledger.summary(None)["calls_total"]


def test_scoped_cli_passes(tmp_path, monkeypatch):
    p = pipeline(tmp_path)
    ids = p.ingest([dict(doi=f"10.1/p{i}") for i in range(3)])
    for iid, stage in zip(ids, ["S2", "S3b", "S3c"]):
        it = p.store.item(iid)
        it.state["stage"] = stage
        it.save()
    from tbsgen.state import Item
    step_lock = Item.step_lock

    def scoped_lock(it, *args):
        assert it.item_id != ids[0], "worker locked an item outside its stage scope"
        return step_lock(it, *args)

    monkeypatch.setattr(Item, "step_lock", scoped_lock)
    monkeypatch.setattr(Pipeline, "prefetch_screen", lambda *a: pytest.fail("prefetch outside intake"))
    monkeypatch.setattr("tbsgen.blocklist.reevaluate", lambda *a: pytest.fail("blocklist outside intake"))
    monkeypatch.setitem(HANDLERS, "S3b", lambda _, it: it.advance("S3b", "S4"))
    monkeypatch.setitem(HANDLERS, "S3c", lambda _, it: it.advance("S3c", "S4"))
    assert cli.main(["run", "--run", str(tmp_path), "--stages", "build,code", "--max-steps", "1"]) == 0
    assert cli.main(["run", "--run", str(tmp_path), "--stages", "S3b,S3c"]) == 0
    assert [p.store.item(i).state["stage"] for i in ids] == ["S2", "S4", "S4"]
    assert parse_stages("intake,agent,sandbox,calib,harden") >= {"S2a", "S5G", "PKG", "S9R", "harden"}
    with pytest.raises(SystemExit):
        cli.main(["run", "--run", str(tmp_path), "--stages", "unknown"])


@pytest.mark.parametrize("stop", ["file", "signal"])
def test_loop_stops_after_progress_without_idle_delay(tmp_path, stop):
    p = pipeline(tmp_path)
    calls = []

    def step(stopping):
        calls.append(1)
        if len(calls) == 2:
            if stop == "file":
                (p.store.root / "driver.stop").touch()
            else:
                signal.raise_signal(signal.SIGTERM)
        return 0, len(calls) == 1
    assert cli._worker_loop(p.store.root, 60, step) == 0
    assert len(calls) == 2


@pytest.mark.parametrize("days", [None, 7])
def test_window_caps_and_resume(tmp_path, monkeypatch, days):
    cap = dict(name="allowance", model="teacher" if days else "*", since="2026-01-02", usd=5)
    if days:
        cap["days"] = days
    p = pipeline(tmp_path, budget=dict(window_caps=[cap], daily_usd_total=6))
    day = ["2026-01-01"]
    monkeypatch.setattr(budget, "today", lambda: day[0])
    p.ledger.reserve("teacher", "S9", "earlier", True, est_usd=6)
    day[0] = "2026-01-02"
    p.ledger.reserve("teacher", "S9", "existing", True, est_usd=5)
    with pytest.raises(budget.BudgetStop, match="daily_total"):
        p.ledger.reserve("teacher", "S9", "next", True, est_usd=2)
    day[0] = "2026-01-08"
    [iid] = p.ingest([dict(doi="10.1/next")])
    monkeypatch.setitem(HANDLERS, "S1", lambda worker, it: worker.ledger.reserve(
        "teacher" if days else "student", "S1", it.item_id, True, est_usd=1))
    p.run(max_steps=1)
    assert p.store.item(iid).state["reason"] == "budget:window:allowance"
    assert p.status()["window_caps"][0]["used_usd"] == 5
    assert p.resume() == []
    day[0] = "2026-01-09"
    assert p.resume() == ([iid] if days else [])
    if days:
        p.ledger.reserve("student", "S9", "elsewhere", True, est_usd=5)
        assert p.status()["window_caps"][0]["used_usd"] == 0


def test_calib_worker_collects_and_splits_slots(tmp_path):
    p, ids, _, jobs = _env(tmp_path, n_items=2, max_concurrent_jobs_by_who=dict(glm=1, teacher=1, review=0))
    for iid in ids:
        p.run(only=[iid], stages={"S9"}, poll_calib=True)
    assert len(stages_b.running_jobs(p)) == 1
    assert "glm 1/1" in p.store.item(ids[1]).state["reason"]
    assert not p.run(only=[ids[1]], max_steps=1, stages={"S9"}, poll_calib=True)["progress"]
    _finish(jobs, _runs(p, ids[0])[0]["job_name"], reward=0, cost=1)
    p.run(only=[ids[0]], stages={"S9"}, poll_calib=True)
    assert _runs(p, ids[0])[-1]["who"] == "teacher"
    p.run(only=[ids[1]], stages={"S9"}, poll_calib=True)
    assert len(stages_b.running_jobs(p)) == 2
    assert "review 0/0" in stages_b._slots_full(p, who="review")
    p.cfg["calib"].update(max_concurrent_jobs=2, max_concurrent_jobs_by_who={"review": 1})
    assert "2/2" in stages_b._slots_full(p, who="review")


def test_throughput_history(tmp_path, capsys):
    p = pipeline(tmp_path)
    now = time.time()
    for name, history in {
        "paper": [("S3", "parked", "fulltext_unavailable", 30), ("S3", "ok", "", 20),
                  ("S4c", "paused", "budget:daily_model", 18), ("S4c", "resume", "", 12),
                  ("S4c", "waiting", "S4c:agent_running", 10), ("S4c", "ok", "", 6),
                  ("S9", "waiting", "submitted glm run 1: job", 5), ("S10", "ok", "", 2)],
        "queued": [("S9", "waiting", "calib glm run 1 queued: slots full: 1/1 jobs running", 3)]
    }.items():
        it = p.store.item(name)
        it.state.update(stage=history[-1][0], status="done" if name == "paper" else "waiting",
                        history=[dict(stage=s, result=r, reason=why, ts=now - h * 3600) for s, r, why, h in history])
        it.save()
    report = throughput(p, 1, now=now)
    assert report["waiting_item_hours"] == {"S3": {"fulltext_unavailable": 4}, "S4c": {"budget:daily_model": 6},
                                             "S9": {"queued: slots full": 3}}
    assert report["mean_concurrent_jobs"]["S4c"] == 4 / 24
    assert report["mean_concurrent_jobs"]["S9"] == 3 / 24
    assert report["releases_per_day"] == {budget.today(now - 7200): 1}
    assert report["funnel"]["S9"] == 2
    cli.main(["report", "--run", str(tmp_path), "--throughput", "--days", "1"])
    assert json.loads(capsys.readouterr().out)["funnel"] == report["funnel"]


def test_harden_group_leaves_first_version_alone(tmp_path, monkeypatch):
    p = pipeline(tmp_path)
    first, variant = p.ingest([dict(doi="10.1/base"), dict(doi="10.1/variant")])
    for iid in (first, variant):
        it = p.store.item(iid)
        it.state.update(stage="S6A_run", harden_active=iid == variant, harden_kind="variant")
        it.save()

    def finish_round(_, it):
        it.state.pop("harden_active")
        it.advance("S6A_run", "S9")
    monkeypatch.setitem(HANDLERS, "S6A_run", finish_round)
    p.run(stages=parse_stages("harden"))
    assert p.store.item(variant).state["stage"] == "S9"
    assert p.store.item(first).state["stage"] == "S6A_run"


def test_cli_fetch_loop_delivers_cached_batches(tmp_path, monkeypatch):
    p = pipeline(tmp_path, fulltext=dict(write_cache_dir=str(tmp_path / "cache"), cascade=["cache"]))
    f = p.fulltext_fetcher()
    ids = p.ingest([dict(doi=f"10.1/p{i}", title=TITLE) for i in range(2)])
    for idx, iid in enumerate(ids):
        entry = dict(doi=f"10.1/p{idx}", title=TITLE)
        f.save_imported(entry, dict(ident=fulltext.paper_ident(entry), channel="oa"), BODY, None)
        it = p.store.item(iid)
        it.state.update(stage="S3")
        it.finish("parked", "fulltext_unavailable")
        fulltext.enqueue(fulltext.queue_path(p.cfg), fulltext.queue_entry(entry, subscribers=[["", iid]]))
    original = fulltext.drain_queue
    results = []

    def drain(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result["ok"])
        if len(results) == 3:
            (p.store.root / "driver.stop").touch()
        return result
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    monkeypatch.setattr(fulltext, "drain_queue", drain)
    assert cli.main(["fetch", "--run", str(tmp_path), "--max", "1", "--loop", "0.01"]) == 0
    assert results == [1, 1, 0]
    assert [p.store.item(i).state["stage"] for i in ids] == ["S4", "S4"]


def test_contended_item_skipped_for_entire_pass(tmp_path, monkeypatch):
    p = pipeline(tmp_path, schedule=dict(order="id"))
    ids = sorted(p.ingest([dict(doi="10.1/one"), dict(doi="10.1/two")]))
    lock = p.store.item(ids[0]).step_lock()
    assert lock.__enter__()

    def finish(_, it):
        if it.item_id == ids[1]:
            lock.__exit__(None, None, None)
        it.finish("done", "finished")
    monkeypatch.setitem(HANDLERS, "S1", finish)
    p.run(stages={"S1"})
    assert [p.store.item(i).state["status"] for i in ids] == ["active", "done"]
    p.run(stages={"S1"})
    assert p.store.item(ids[0]).state["status"] == "done"


@pytest.mark.parametrize("stage", ["S9", "S9R", "S4c", "S5G"])
@pytest.mark.parametrize("boundary", ["reserved", "sent", "rollover_send", "launched"])
@pytest.mark.parametrize("rollover", [False, True], ids=["same-day", "next-day"])
def test_crash_after_reserve_commit(tmp_path, monkeypatch, stage, boundary, rollover):
    import subprocess
    from tbsgen import calib
    from test_review_s9r import _env as review_env

    if stage == "S9R":
        p, iid, jobs = review_env(tmp_path, est_usd_per_run=2)
        it = p.store.item(iid)
        it.state.update(stage="S9R", calib_label="hard", review_round=1)
        it.save()
    elif stage == "S9":
        p, [iid], _, jobs = _env(tmp_path, est_usd_per_run=2)
    else:
        p, it, launcher = _agent_item(tmp_path, monkeypatch, {}, stage=stage)
        iid = it.item_id
    it = p.store.item(iid)
    record_dir = {"S9": "calib", "S9R": "review"}.get(stage, stage)

    def record():
        if stage in ("S9", "S9R"):
            return json.loads(it.p(record_dir, "runs.json").read_text())[0]
        return json.loads(it.p(stage, "run.json").read_text())

    reserve = p.ledger.reserve
    def kill(*args, **kwargs):
        sending = kwargs["send"]
        if sending:
            rec = record()
            assert rec["launch_attempted"] and rec["tmux"] == rec["job_name"][-60:]
            if boundary == "rollover_send":
                with p.ledger._db() as db:
                    db.execute("update calls set day='2000-01-01' where id=?", (rec["ledger_call"],))
        call = reserve(*args, **kwargs)
        if (boundary in ("sent", "rollover_send") and sending) or (boundary == "reserved" and not sending):
            raise SystemExit("after commit")
        return call

    launches = []
    run = subprocess.run
    def start(cmd, **kwargs):
        if cmd[1] != "new-session":
            return run(cmd, **kwargs)
        launches.append(cmd)
        with p.ledger._db() as db:
            assert db.execute("select status from calls where id=?", (record()["ledger_call"],)).fetchone() == ("sent",)
        reply = run(cmd, **kwargs)
        if boundary == "launched":
            raise SystemExit("after commit and launch")
        return reply

    monkeypatch.setattr(calib, "session_alive", lambda *_: False)
    monkeypatch.setattr(p.ledger, "reserve", kill)
    monkeypatch.setattr(subprocess, "run", start)
    with pytest.raises(SystemExit, match="after commit"):
        p.step(it)
    assert record()["state"] == ("intent" if boundary == "reserved" else "reserved")
    with p.ledger._db() as db:
        if boundary == "rollover_send":
            old_call = record()["ledger_call"]
            assert db.execute("select status from calls where id=?", (old_call,)).fetchone() == ("cancelled",)
        [(cid, key, est)] = db.execute("select id, reservation_key, est_usd from calls where status!='cancelled'")
    assert key
    if rollover:
        with p.ledger._db() as db:
            db.execute("update calls set day='2000-01-01' where id=?", (cid,))
        p.cfg["budget"]["daily_usd_total"] = 0
    p.cfg["budget"]["window_caps"] = [dict(name="one", model="*", since=budget.today(), usd=est)]
    monkeypatch.setattr(p.ledger, "reserve", reserve)
    if boundary in ("sent", "rollover_send"):
        p.cfg["calib"]["allow_submit"] = False
        p.step(p.store.item(iid))
        assert record()["state"] == "launch_failed" and not launches
        assert record()["ledger_call"] == cid
        with p.ledger._db() as db:
            assert db.execute("select cost_kind,usd from calls where status!='cancelled'").fetchall() == [
                ("not_launched", 0)]
        return
    cancelled = rollover and boundary == "reserved"
    if cancelled:
        assert p.step(p.store.item(iid)) == "paused"
        assert p.store.item(iid).state["reason"] == "budget:daily_total"
        assert record()["state"] == "cancelled_unstarted" and record()["ledger_call"] is None
        if stage in ("S9", "S9R"):
            assert not stages_b.running_jobs(p)
        else:
            from tbsgen import stages_a
            p.cfg["code"]["max_concurrent_agent_jobs"] = 1
            assert stages_a._agent_admission(p, p.store.item("waiting-paper"), stage) is None
        with p.ledger._db() as db:
            assert db.execute("select status, usd from calls where id=?", (cid,)).fetchone() == ("cancelled", 0)
        p.cfg["budget"]["daily_usd_total"] = est
        p.resume(("budget:",))
    assert p.step(p.store.item(iid)) == "ok"
    assert len(launches) == 1
    rec = record()
    assert (rec["ledger_call"] != cid) == cancelled and rec["state"] == "launched"
    with p.ledger._db() as db:
        assert db.execute("select status from calls where id=?", (rec["ledger_call"],)).fetchone() == ("sent",)
    assert p.ledger.window_usage()[0]["used_usd"] == (0 if rollover and not cancelled else est)
    # Expired, unsent reservations are excluded from the call count.
    assert p.ledger.summary(None)["calls_total"] == 1


def test_model_retry_recovers_open_hardening_reservation(tmp_path, monkeypatch):
    from tbsgen.models import FakeClient

    p = pipeline(tmp_path)
    client = FakeClient(p.ledger, p.cfg, lambda *_: "ok")
    with budget.harden_scope(1):
        with monkeypatch.context() as m:
            def crash(*args, **kwargs):
                raise SystemExit("reserved before model call")
            m.setattr(client, "_call", crash)
            with pytest.raises(SystemExit):
                client.chat("system", "paper", stage="S6C", item="paper")
        result = client.chat("system", "paper", stage="S6C", item="paper")
        assert p.ledger.summary(None)["calls_total"] == 1
        assert client.chat("system", "paper", stage="S6C", item="paper").call_id != result.call_id


def _interrupt_remote_record(tmp_path, monkeypatch, exception):
    from tbsgen.dlledger import DownloadLedger
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, cascade=["oa"],
                                         global_limits={"daily_ok": 1})
    f.ft["remote"].update(host="a", preflight=False)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)

    def interrupt(worker, channel, host, doi, status, *args, **kwargs):
        assert status == "ok" and list((worker.path.parent / "cache").glob("*.txt"))
        raise exception

    monkeypatch.setattr(DownloadLedger, "record", interrupt)
    queue(f, OA)
    return f, ledger, wd, FakeSSH(wd, net)


def test_granted_quota_survives_worker_loss(tmp_path, monkeypatch):
    from tbsgen.dlledger import LimitReached
    from test_fulltext import FakeClock
    from test_remote_browser import OA

    f, ledger, wd, ssh = _interrupt_remote_record(tmp_path, monkeypatch, SystemExit("worker killed before record"))
    ledger.clock = clock = FakeClock(time.time())
    with pytest.raises(SystemExit, match="before record"):
        remote.run_job(f, [OA], ["oa"], ssh_runner=ssh, max_poll_s=0)
    [job] = remote.outstanding_jobs(f.cfg)
    sp = wd / "jobs" / job["job"] / "status.json"
    st = json.loads(sp.read_text())
    sp.write_text(json.dumps(dict(st, ts=time.time() - 60)))
    [dead] = remote.run_job(f, [], ["oa"], ssh_runner=ssh, max_poll_s=0)["jobs"]
    assert dead["status"]["state"] == "dead" and dead["stage"] == "submitted"
    clock.sleep(3 * 3600)
    with pytest.raises(LimitReached, match="global daily cap"):
        ledger.check("browser")
    result = remote.run_job(f, [], ["oa"], ssh_runner=ssh, max_poll_s=0)
    assert BODY in result["results"][fulltext.paper_ident(OA)]["text"]
    assert ledger.db.execute("select channel,status from attempts").fetchall() == [("oa", "ok")]


@pytest.mark.parametrize("failure", [TimeoutError("ack timeout"), ConnectionError("reply lost")])
def test_record_timeout_preserves_saved_result_and_done_export(tmp_path, monkeypatch, failure):
    from test_remote_browser import OA

    f, ledger, wd, ssh = _interrupt_remote_record(tmp_path, monkeypatch, failure)

    def transport(cmd, stdin=b"", timeout=None):
        if shlex.split(cmd)[3] == "export":
            d = wd / "jobs" / shlex.split(cmd)[-1]
            assert json.loads((d / "status.json").read_text())["state"] == "done"
            rp = d / "results/0000.json"
            saved = json.loads(rp.read_text())
            assert saved["ok"] and (d / "cache" / saved["text_file"]).exists()
            # Also collect results produced by an older worker that wrote ok=false after this timeout.
            rp.write_text(json.dumps(dict(saved, ok=False, text_file=None)))
        return ssh(cmd, stdin, timeout)

    result = remote.run_job(f, [OA], ["oa"], ssh_runner=transport)
    assert BODY in result["results"][fulltext.paper_ident(OA)]["text"]
    assert ledger.usage()["oa"]["ok"] == 1 and ledger.usage()["oa"]["attempts"] == 1


def test_missing_before_submit_and_host_lock(tmp_path, monkeypatch):
    from test_remote_browser import OA, FakeSSH

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {})
    f.ft["remote"].update(host="a", preflight=False)
    queue(f, OA)
    ssh, delayed = FakeSSH(wd, net), []

    def lost_submit(cmd, stdin=b"", timeout=None):
        if shlex.split(cmd)[3] == "submit":
            delayed.append((cmd, stdin, timeout))
            raise SystemExit("submit in flight")
        return ssh(cmd, stdin, timeout)

    def tick():
        return remote.run_job(f, [OA], ["oa"], ssh_runner=lost_submit, max_poll_s=0)

    with pytest.raises(SystemExit):
        tick()
    [receipt] = remote.outstanding_jobs(f.cfg)
    assert receipt["stage"] == "submitted" and receipt["host"] == "a" and receipt["submitted_at"]
    assert tick()["jobs"][0]["status"]["state"] == "starting"
    assert len(delayed) == 1 and ledger.db.execute("select count(*) from attempts").fetchone() == (1,)
    started, release = threading.Event(), threading.Event()

    def wait_worker(*args):
        started.set()
        assert release.wait(5)
        return 0

    monkeypatch.setattr(remote, "work", wait_worker)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(ssh, *delayed[0])
        assert started.wait(5)
        try:
            response = io.StringIO()
            payload = delayed[0][1].decode()
            rc = remote.cmd_submit(str(wd), "another", io.StringIO(payload), response, daemon=False)
            assert rc != 0 and "host lock" in json.loads(response.getvalue())["error"]
            assert not (wd / "jobs/another/status.json").exists()
        finally:
            release.set()
        assert first.result(5)[0] == 0


def test_stale_missing_requires_confirmed_empty_export(tmp_path, monkeypatch):
    from test_remote_browser import FakeSSH, OA
    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, remote=dict(preflight=False))
    ssh = FakeSSH(wd, net)
    now = [time.time()]
    ledger.clock = lambda: now[0]
    submit = []
    export_down = [True]
    queue(f, OA)

    def transport(cmd, stdin=b"", timeout=None):
        op = shlex.split(cmd)[3]
        if op == "submit":
            submit.append((cmd, stdin, timeout))
            return 255, b"", b"connection closed"
        if op == "export" and export_down[0]:
            return 255, b"", b"unreachable"
        return ssh(cmd, stdin, timeout)

    def tick(entries):
        return remote.run_job(f, entries, ["oa"], ssh_runner=transport, max_poll_s=0)["jobs"][0]

    assert tick([OA])["status"]["state"] == "starting"
    now[0] += 7199
    assert tick([])["status"]["state"] == "starting"
    now[0] += 2
    assert "export failed" in tick([])["error"]
    assert ledger.db.execute("select count(*) from attempts").fetchone() == (1,)
    export_down[0] = False
    assert tick([])["stage"] == "dropped"
    assert ledger.db.execute("select count(*) from attempts").fetchone() == (0,)
    assert ssh(*submit[0])[0] != 0  # the original delayed submit can no longer start


def test_new_lease_excludes_browser_fallback(tmp_path, monkeypatch):
    from test_remote_browser import FakeSSH, OA
    f, net, _, _, wd, _ = mk_remote(tmp_path, {}, cascade=["oa", "browser"],
                                    browser=dict(enabled=True, batch_only=False))
    f.ft["remote"].update(host="a", preflight=False)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: FakeSSH(wd, net))
    monkeypatch.setattr(remote, "work", lambda *args: 0)  # submitted, first status poll still starting
    monkeypatch.setattr(f, "_egress_ok_browser", lambda: None)
    monkeypatch.setattr(f, "ch_browser", lambda *args: pytest.fail("browser called for a leased paper"))
    queue(f, OA)
    stats = fulltext.drain_queue(f, browser=True, poll_once=True)
    assert stats["deferred"] == 1 and stats["remaining"] == 1 and remote.outstanding_jobs(f.cfg)


def test_regate_waits_at_s10_manifest_append(tmp_path, monkeypatch):
    p, [iid], _, _ = _env(tmp_path)
    it = p.store.item(iid)
    it.p("thresholds.json").write_text("{}")
    it.state.update(stage="S10", calib_label="hard")
    it.save()
    at_append, attempting_lock, release = threading.Event(), threading.Event(), threading.Event()
    append = stages_b._append_manifest

    def pause(path, record):
        if not record.get("withdrawn"):
            at_append.set()
            assert release.wait(5)
        append(path, record)

    monkeypatch.setattr(stages_b, "_append_manifest", pause)
    operator = p.store.item(iid)
    lock = operator.step_lock

    def enter(timeout=0):
        attempting_lock.set()
        return lock(timeout)

    monkeypatch.setattr(operator, "step_lock", enter)
    with ThreadPoolExecutor(2) as pool:
        worker = pool.submit(p.step, it)
        assert at_append.wait(5)
        regate = pool.submit(stages_b.regate, p, operator, "updated verifier")
        try:
            assert attempting_lock.wait(2) and not regate.done()
        finally:
            release.set()
        assert worker.result(5) == "ok"
        assert regate.result(5)["withdrawn"]
    stages_b.s10(p, it)  # a stale caller still holds the pre-regate state
    state = it.load().state
    assert state["post_regate"] and state["stage"] == "S6A_run"
    assert not stages_b._released(p, iid)


@pytest.mark.parametrize("repeat_ingest", [False, True])
def test_ingest_killed_before_initial_state(tmp_path, monkeypatch, repeat_ingest):
    from tbsgen.state import Item
    from tbsgen.intake import item_id_for
    p = pipeline(tmp_path)
    candidate = dict(doi="10.1/interrupted")
    with monkeypatch.context() as m:
        def kill(item):
            assert item.p("S1_candidate.json").exists()
            raise SystemExit("before state.json")
        m.setattr(Item, "save", kill)
        with pytest.raises(SystemExit):
            p.ingest([candidate])
    it = p.store.item(item_id_for(candidate))
    assert not it.p("state.json").exists()
    assert it.state["created"] == it.p("S1_candidate.json").stat().st_mtime
    if repeat_ingest:
        p.ingest([candidate])
        assert it.p("state.json").exists()
        it.load()
    assert p.step(it) == "ok" and it.state["stage"] != "S1"


@pytest.mark.parametrize("text", ["body", None], ids=["success", "failure"])
def test_trial_lifts_rule_before_next_s2(tmp_path, monkeypatch, text):
    from tbsgen import blocklist, stages_a
    p = pipeline(tmp_path, schedule=dict(order="id"),
                 screen=dict(blocklist=[dict(name="sample", doi_prefix="10.1234")],
                             blocklist_recheck=dict(min_trials=5, lift_rate=0.6, weekly_quota=0)))
    a, b = sorted(p.ingest([dict(doi="10.1234/" + x) for x in ("a", "b")]))
    for iid, stage in [(a, "S3"), (b, "S2")]:
        it = p.store.item(iid)
        it.state["stage"] = stage
        if iid == a:
            it.state["blocklist_trial"] = "sample"
        it.save()
    trials = [dict(outcome=result) for result in ("ok", "ok", "ok", "fail")] + [dict(item=a, outcome="pending")]
    blocklist.save_state(p.store, dict(rules=dict(sample=dict(trials=trials))))
    monkeypatch.setattr(stages_a.intake, "fetch_fulltext", lambda *args: (text, dict(source="cache")))
    monkeypatch.setattr(stages_a, "screen_paper", lambda *args: dict(keep=True, score=1, method="rules"))
    p.run(only=[a, b], max_steps=2)
    assert p.store.item(b).state["stage"] == "S3"
    assert blocklist.lifted_rules(blocklist.load_state(p.store)) == {"sample"}


def _stress_worker(cfg, jobs, ids, worker_index, barrier):
    p = Pipeline(cfg)
    barrier.wait(25)
    p.run(stages={"S9"})
    barrier.wait(25)
    if worker_index == 0:
        for iid in ids:
            [rec] = _runs(p, iid)
            _finish(jobs, rec["job_name"], reward=1, cost=0.5)
    barrier.wait(25)
    for _ in range(3):
        p.run(stages={"S9"})
        barrier.wait(25)


def test_four_processes_thirty_calibrations(tmp_path):
    import multiprocessing
    p, ids, counter, jobs = _env(tmp_path, n_items=30, quick_pass_minutes=1000, est_usd_per_run=1,
                                 max_concurrent_jobs=None, max_concurrent_jobs_by_who={})
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(4)
    workers = [ctx.Process(target=_stress_worker, args=(p.cfg, jobs, ids, i, barrier)) for i in range(4)]
    deadline = time.monotonic() + 55
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
        assert [w.exitcode for w in workers] == [0] * 4
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(1)
    assert not [e for e in p.store.events() if e["kind"] == "stage_crash"]
    assert len(counter.read_text().splitlines()) == p.ledger.summary(None)["calls_total"] == len(ids)
    calls = set()
    for iid in ids:
        [rec] = _runs(p, iid)
        state = p.store.item(iid).state
        assert rec["outcome"] == "pass" and state["status"] == "done" and state["calib_label"] == "easy"
        calls.add(rec["ledger_call"])
    assert len(calls) == len(ids)


def test_unsubmitted_lease_expires_without_ssh(tmp_path, monkeypatch):
    from test_fulltext import FakeClock
    from test_remote_browser import OA

    owner, _, _, ledger, _, _ = mk_remote(tmp_path / "one", {}, global_limits={"daily_ok": 1})
    queue(owner, OA)
    owner.ft["remote"]["preflight"] = False
    clock = FakeClock(remote._midnight_after(time.time()) - 3600)
    ledger.clock = clock

    def crash(*args, **kwargs):
        raise SystemExit("lease committed before submit")

    monkeypatch.setattr(remote, "remote_spec", crash)
    no_ssh = lambda *args: pytest.fail("unsubmitted lease contacted the worker")
    with pytest.raises(SystemExit, match="lease committed"):
        remote.run_job(owner, [OA], ["oa"], ssh_runner=no_ssh)
    [receipt] = remote.outstanding_jobs(owner.cfg)
    assert receipt["stage"] == "leasing"
    clock.sleep(599)
    remote.run_job(owner, [], ssh_runner=no_ssh)
    assert owner.ledger.remaining("oa") == 0
    clock.sleep(2)
    remote.run_job(owner, [], ssh_runner=no_ssh)
    assert owner.ledger.remaining("oa") == 1 and not remote.outstanding_jobs(owner.cfg)


def _submit_without_worker(f, entry, ssh, monkeypatch):
    # Real submit writes the job spec and ledger. Drive worker files explicitly at the crash boundary.
    with monkeypatch.context() as patch:
        patch.setattr(remote, "work", lambda *args: 0)
        [job] = remote.run_job(f, [entry], ["oa"], ssh_runner=ssh, max_poll_s=0)["jobs"]
    return job["job"]


def test_browser_batch_excludes_lease(tmp_path, monkeypatch):
    from tbsgen import browser_agent
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, browser=dict(enabled=True), remote=dict(preflight=False))
    for paper in (OA, dict(doi="10.1/free")):
        fulltext.enqueue(fulltext.queue_path(f.cfg), fulltext.queue_entry(paper, failed_channels=["oa"]))
    _submit_without_worker(f, OA, FakeSSH(wd, net), monkeypatch)
    assert ledger.remaining("browser") >= 1
    assert [e["doi"] for e in browser_agent._due(f, {})] == ["10.1/free"]


def test_single_host_timeout_is_cli_error(tmp_path, monkeypatch, capsys):
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {})
    p = Pipeline(f.cfg)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    p.fetcher = f
    f.ft["remote"]["preflight"] = False
    queue(f, OA)
    _submit_without_worker(f, OA, FakeSSH(wd, net), monkeypatch)
    limits = []
    def unreachable(cmd, stdin, timeout):
        limits.append(timeout)
        return 255, b"", b"offline"
    monkeypatch.setattr(remote, "default_ssh_runner", lambda cfg: unreachable)
    clock = FakeClock(0)
    monkeypatch.setattr(remote, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(fulltext, "time", SimpleNamespace(time=time.time, sleep=clock.sleep))
    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=clock))
    assert cli.main(["fetch", "--run", str(p.store.root), "--max-wait", "0.02"]) == 3
    assert "poll deadline reached" in capsys.readouterr().err
    assert limits and max(limits) <= 0.02 and ledger.leased_papers() == {fulltext.paper_ident(OA)}


@pytest.mark.parametrize("wait", [0.02, 300])
def test_all_fetch_ssh_calls_share_one_deadline(tmp_path, monkeypatch, wait):
    from types import SimpleNamespace
    from test_fulltext import FakeClock
    from test_remote_browser import FakeSSH, OA

    f, net, _, _, wd, _ = mk_remote(tmp_path, {}, remote=dict(max_job_s=1200))
    ssh = FakeSSH(wd, net)
    clock = FakeClock(0)
    queue(f, OA)
    ops = []
    def send(cmd, stdin, timeout):
        ops.append(shlex.split(cmd)[3])
        assert timeout == pytest.approx(min(120, wait - clock()))
        clock.sleep(0.003)
        return ssh(cmd, stdin, timeout)
    monkeypatch.setattr(remote, "time", SimpleNamespace(monotonic=clock, time=time.time, strftime=time.strftime,
                                                        localtime=time.localtime, mktime=time.mktime))
    from tbsgen import dlledger
    monkeypatch.setattr(dlledger, "time", remote.time)
    result = remote.run_job(f, [OA], ["oa"], ssh_runner=send, max_poll_s=wait)
    assert result["jobs"][0]["stage"] == "collected"
    assert ops == ["info", "submit", "status", "export", "cleanup"]


@pytest.mark.parametrize("until_midnight, wait", [(3600, 1800), (65, 605)])
def test_default_wait_ends_for_running_job(tmp_path, monkeypatch, until_midnight, wait):
    from types import SimpleNamespace
    from test_fulltext import FakeClock
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, remote=dict(max_job_s=1200, preflight=False))
    midnight = remote._midnight_after(time.time())
    ledger.clock = FakeClock(midnight - 3600)
    ssh = FakeSSH(wd, net)
    queue(f, OA)
    _submit_without_worker(f, OA, ssh, monkeypatch)
    ledger.clock = FakeClock(midnight - until_midnight)
    clock = FakeClock(0)
    monkeypatch.setattr(remote, "time", SimpleNamespace(monotonic=clock, time=time.time, localtime=time.localtime, mktime=time.mktime))
    def advance(seconds):
        clock.sleep(seconds)
        assert clock() <= wait
    result = remote.run_job(f, [], ssh_runner=ssh, sleep=advance, poll_s=1000)
    assert clock() == wait and "deadline" in result["jobs"][0]["error"]
    assert ledger.leased_papers() == {fulltext.paper_ident(OA)}


def test_watch_skips_contended_item_and_ingests_next(tmp_path, monkeypatch):
    from tbsgen.state import Item

    p, ids, _, jobs = _env(tmp_path, n_items=2, max_concurrent_jobs=2)
    busy, ready = sorted(ids)
    for iid in ids:
        p.run(only=[iid], stages={"S9"})
        _finish(jobs, _runs(p, iid)[0]["job_name"], reward=0, cost=1)
    lock = Item.step_lock
    with p.store.item(busy).step_lock():
        monkeypatch.setattr(Item, "step_lock", lambda it, timeout=0: lock(it, min(timeout, 0.01)))
        result = stages_b.watch_tick(p)
    assert [a["item"] for a in result["actions"]] == [ready]
    assert _runs(p, ready)[0]["outcome"] is not None and _runs(p, busy)[0]["outcome"] is None
    assert any(e["kind"] == "calib_watch_busy" and e["item"] == busy for e in p.store.events())


def test_overlap_lift_rechecks_next_rule(tmp_path, monkeypatch):
    from tbsgen import blocklist, stages_a

    rules = [dict(name=n, doi_prefix="10.1234") for n in ("first", "second")]
    p = pipeline(tmp_path, screen=dict(blocklist=rules, blocklist_recheck=dict(min_trials=1, weekly_quota=0)))
    [iid] = p.ingest([dict(doi="10.1234/x")])
    blocklist.save_state(p.store, dict(rules=dict(first=dict(trials=[dict(outcome="ok")]))))
    decide = blocklist.decide
    def interleave(store, cfg, candidate, rules, initial, event):
        assert initial["name"] == "first"
        assert blocklist.reevaluate(store, cfg, event) == {"first": "lifted"}
        return decide(store, cfg, candidate, rules, initial, event)
    monkeypatch.setattr(blocklist, "decide", interleave)
    stages_a.s2(p, p.store.item(iid))
    it = p.store.item(iid)
    assert it.state["reason"] == "blocklist:second"
    assert json.loads(it.p("S2_screen.json").read_text())["blocklist"]["name"] == "second"


def test_model_reservation_entry_points():
    """Review every new reserve site. Detached launchers reserve first and mark sent at launch."""
    sites = []
    for path in Path(budget.__file__).parent.rglob("*.py"):
        for call in ast.walk(ast.parse(path.read_text())):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in ("reserve", "reserve_job"):
                send = next(kw.value for kw in call.keywords if kw.arg == "send")
                sites.append((path.name, ast.unparse(send)))
    assert sorted(sites) == [("budget.py", "send"), ("calib.py", "send"), ("codeagent.py", "send"),
                             ("models.py", "self.real"), ("screen_jev.py", "True"), ("stages_b.py", "True")]


def test_browser_plan_before_remote_lease_downloads_once(tmp_path, monkeypatch):
    from tbsgen import browser_agent
    from test_remote_browser import FakeSSH, OA, Seq, TRACE_OK, script_runner

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, cascade=["oa"],
                                         global_limits={"daily_ok": 2}, browser=dict(enabled=True, batch_only=True))
    f.ft["remote"]["preflight"] = False
    paper = dict(OA, doi="10.1080/interleaved")
    fulltext.enqueue(fulltext.queue_path(f.cfg), fulltext.queue_entry(paper, failed_channels=["oa"]))
    f.browser_runner, downloads = script_runner(tmp_path)
    ssh = FakeSSH(wd, net)
    paused = []

    def pause_after_plan(message):
        if json.loads(message).get("event") == "plan":
            paused.append(_submit_without_worker(f, paper, ssh, monkeypatch))

    st = browser_agent.run_batch(f, log=pause_after_plan, prober=browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK)))
    assert st["planned"] == [fulltext.paper_ident(paper)] and st["deferred"] == 1 and not downloads
    [job] = paused
    directory = wd / "jobs" / job

    monkeypatch.setattr(remote, "_TEST_TRANSPORT", net)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    spec = json.loads((directory / "spec.json").read_text())
    spec["deadline"] = time.time() + 60
    remote.work(directory, spec, {})
    result = remote.run_job(f, [], ssh_runner=ssh, max_poll_s=0)
    assert result["results"][fulltext.paper_ident(paper)]["ok"]
    assert [url for url, _ in net.calls].count(OA["oa_url"]) == 1
    assert ledger.db.execute("select channel, status from attempts").fetchall() == [("oa", "ok")]


def test_browser_grant_blocks_remote_lease(tmp_path):
    from test_remote_browser import OA
    f, _, _, led, _, _ = mk_remote(tmp_path, {}, global_limits={"daily_ok": 2})
    queue(f, OA)
    led.acquire("browser", "chrome:10.9999", OA["doi"])
    snap = remote.lease_and_snapshot(led, "second", 1, entries=[OA], fetcher=f)
    assert not snap.get("entries") and not led.leased_papers()


def test_legacy_leased_receipt_collects_finished_job(tmp_path, monkeypatch):
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, remote=dict(preflight=False),
                                         global_limits={"daily_ok": 1})
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    ssh = FakeSSH(wd, net)

    queue(f, OA)

    def crash(cmd, stdin, timeout):
        reply = ssh(cmd, stdin, timeout)
        assert ssh.calls[-1] == "submit" and reply[0] == 0
        raise SystemExit("before submit acknowledgement")

    with pytest.raises(SystemExit):
        remote.run_job(f, [OA], ["oa"], ssh_runner=crash)
    [receipt] = remote.outstanding_jobs(f.cfg)
    job = receipt["job"]
    old = dict(job=job, stage="leased", created=time.time() - 601, channels=["oa"], idents=receipt["idents"])
    remote._job_state(f.cfg, job).write_text(json.dumps(old))
    ledger.db.execute("update attempts set doi='', ts=? where channel='lease'", (old["created"],))
    result = remote.run_job(f, [OA], ["oa"], ssh_runner=ssh, max_poll_s=0)
    assert result["results"][fulltext.paper_ident(OA)]["ok"]
    assert ssh.calls == ["submit", "status", "export", "cleanup"]
    assert [url for url, _ in net.calls].count(OA["oa_url"]) == 1 and ledger.remaining("oa") == 0


def test_one_shot_reports_submit_grace(tmp_path, monkeypatch):
    from test_remote_browser import OA

    f, _, _, ledger, _, _ = mk_remote(tmp_path, {}, remote=dict(preflight=False))
    p = Pipeline(f.cfg)
    p.fetcher = f
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    queue(f, OA)
    with monkeypatch.context() as m:
        def crash(*args):
            raise SystemExit("after lease")
        m.setattr(remote, "remote_spec", crash)
        with pytest.raises(SystemExit):
            remote.run_job(f, [OA], ["oa"], ssh_runner=lambda *a: pytest.fail())
    assert cli.main(["fetch", "--run", str(p.store.root)]) == 3
    assert ledger.leased_papers() == {fulltext.paper_ident(OA)}


def test_subscriber_timeout_after_remote_cleanup(tmp_path, monkeypatch):
    from tbsgen import state
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, cascade=["oa"],
                                         remote=dict(preflight=False))
    p = Pipeline(f.cfg)
    p.fetcher = f
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    [iid] = p.ingest([OA])
    it = p.store.item(iid)
    it.state.update(stage="S3")
    it.finish("parked", "fulltext_unavailable")
    fulltext.enqueue(fulltext.queue_path(f.cfg), fulltext.queue_entry(OA, subscribers=[["", iid]]))
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    clock = FakeClock(10)
    def wait(seconds):
        assert ssh.calls[-1] == "cleanup"
        clock.sleep(seconds)
    # The remote collector and SQLite must see the same deadline as subscriber delivery.
    # Only waiting for the held item lock advances this test's clock.
    monkeypatch.setattr(time, "monotonic", clock)
    timer = SimpleNamespace(time=time.time, monotonic=clock, sleep=wait)
    monkeypatch.setattr(state, "time", timer)
    monkeypatch.setattr(cli, "time", timer)
    with it.step_lock():
        assert cli.main(["fetch", "--run", str(p.store.root), "--max-wait", "0.15"]) == 3
    assert clock() == pytest.approx(10.15)
    assert not list((wd / "jobs").iterdir()) and f.cached(OA)
    assert fulltext.read_queue(fulltext.queue_path(f.cfg))[0]["subscribers"] == [["", iid]]
    assert cli.main(["fetch", "--run", str(p.store.root)]) == 0
    assert it.load().state["stage"] == "S4" and ledger.usage()["oa"]["ok"] == 1


def test_settlement_day_preserves_attempt_allowance(tmp_path):
    from tbsgen.dlledger import DownloadLedger, LimitReached

    clock = FakeClock(remote._midnight_after(time.time()) - 10)
    led = DownloadLedger(tmp_path / "ledger", {"oa": dict(daily_attempts=1)}, clock=clock, sleep=clock.sleep)
    day = led.today()
    remote.lease_and_snapshot(led, "j", 1)
    rows = {"attempts": [[1, clock(), day, "oa", "repo.example", "10.1/previous", "ok", 200, 10, ""]]}
    clock.sleep(20)
    remote.merge_rows(led, "j", rows)
    assert led.db.execute("select day,settled_day from attempts").fetchall() == [(day, led.today())]
    assert led.usage()["oa"]["ok"] == 1 and led.usage()["oa"]["attempts"] == 0
    led.acquire("oa", "repo.example", "10.1/next")
    with pytest.raises(LimitReached, match="daily attempt cap"):
        led.acquire("oa", "repo.example", "10.1/another")


@pytest.mark.parametrize("section", ["fulltext", "fetch"])
@pytest.mark.parametrize("hosts", [[], [{"host": "one"}]])
def test_fetch_rejects_hosts(section, hosts):
    with pytest.raises(ValueError, match=r"remote.hosts.*single remote host"):
        load_config(overrides={section: {"remote": {"hosts": hosts}}})


def test_remote_rechecks_candidate_after_browser_finishes(tmp_path, monkeypatch):
    from tbsgen import browser_agent
    from test_remote_browser import FakeSSH, OA, Seq, TRACE_OK, script_runner

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, cascade=["oa"], global_limits={"daily_ok": 2},
                                         browser=dict(enabled=True, batch_only=True))
    paper = dict(OA, doi="10.1080/finished")
    fulltext.enqueue(fulltext.queue_path(f.cfg), fulltext.queue_entry(paper, failed_channels=["oa"]))
    f.browser_runner, downloads = script_runner(tmp_path)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    ssh = FakeSSH(wd, net)

    def send(cmd, stdin, timeout):
        assert shlex.split(cmd)[3] == "info"
        st = browser_agent.run_batch(f, prober=browser_agent.Prober(f.ft, probe_fn=Seq(TRACE_OK)))
        assert st["script_ok"] == 1 and not fulltext.read_queue(fulltext.queue_path(f.cfg))
        return ssh(cmd, stdin, timeout)

    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: send)
    fulltext.drain_queue(f, poll_once=True)
    assert downloads == [paper["doi"]] and ssh.calls == ["info"] and f.cached(paper)
    assert ledger.db.execute("select channel,status from attempts").fetchall() == [("browser", "ok")]


@pytest.mark.parametrize("stale", ["cached", "removed"])
def test_browser_and_remote_grants_recheck_queue_and_cache(tmp_path, monkeypatch, stale):
    from test_remote_browser import OA

    f, _, _, ledger, _, _ = mk_remote(tmp_path, {})
    queue(f, OA)
    if stale == "cached":
        f.save_imported(OA, {"doi": OA["doi"], "channel": "oa"}, BODY, None)
    else:
        fulltext.write_queue(fulltext.queue_path(f.cfg), [])
    snap = remote.lease_and_snapshot(ledger, "j", 1, fetcher=f, entries=[OA])
    assert not snap.get("entries")
    f._space_held = True
    from tbsgen.dlledger import LimitReached
    with pytest.raises(LimitReached, match="already cached or no longer queued"):
        f.ch_browser(OA, "", OA["doi"], TITLE, {"tried": []}, queued=True)
    assert ledger.db.execute("select count(*) from attempts").fetchone() == (0,)


@pytest.mark.parametrize("cap, reason", [("daily_usd_total", "daily_total"), ("daily_calls_per_model", "daily_calls")])
def test_unsent_reservation_rechecks_new_day_caps(tmp_path, monkeypatch, cap, reason):
    p = pipeline(tmp_path, budget={"daily_usd_total": 10, "run_calls_per_model": {"m": 2}})
    monkeypatch.setattr(budget, "today", lambda: "2026-01-01")
    reserve = p.ledger.reserve
    old = reserve("m", "S9", "a", True, reservation_key="job-a", est_usd=2)
    monkeypatch.setattr(budget, "today", lambda: "2026-01-02")
    reserve("m", "S9", "b", True, reservation_key="job-b", est_usd=2, send=True)
    p.cfg["budget"][cap] = 2 if cap == "daily_usd_total" else {"m": 1}
    with pytest.raises(budget.BudgetStop, match=reason):
        reserve("m", "S9", "a", True, reservation_key="job-a", est_usd=2, send=True, reserved_id=old)
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls where id=?", (old,)).fetchone() == ("cancelled", 0)
    p.cfg["budget"][cap] = 4 if cap == "daily_usd_total" else {"m": 2}
    new = reserve("m", "S9", "a", True, reservation_key="job-a", est_usd=2, send=True, reserved_id=old)
    assert new != old
    assert reserve("m", "S9", "a", True, reservation_key="job-a", est_usd=2, send=True, reserved_id=old) == new
    with p.ledger._db() as db:
        assert db.execute("select day,status from calls where reservation_key='job-a' order by id").fetchall() == [
            ("2026-01-01", "cancelled"), ("2026-01-02", "sent")]


def test_cleanup_crash_preserves_failed_channel_backoff(tmp_path, monkeypatch):
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, cascade=["oa"], remote=dict(preflight=False))
    queue(f, OA)
    ssh = FakeSSH(wd, net)
    def send(cmd, stdin, timeout):
        result = ssh(cmd, stdin, timeout)
        if shlex.split(cmd)[3] == "cleanup":
            raise SystemExit("after cleanup")
        return result

    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: send)
    with pytest.raises(SystemExit, match="after cleanup"):
        fulltext.drain_queue(f, poll_once=True)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    fulltext.drain_queue(f, poll_once=True)
    [entry] = fulltext.read_queue(fulltext.queue_path(f.cfg))
    assert entry["tries"] == entry["chan_backoff"]["oa"]["tries"] == 1
    assert "remote_failure" not in entry and entry["next_try"] > time.time()
    assert ssh.calls.count("submit") == ledger.usage()["oa"]["attempts"] == 1


@pytest.mark.parametrize("poll_once", [True, False], ids=["loop", "one-shot"])
def test_fetch_delivers_before_submitting_next_batch(tmp_path, monkeypatch, poll_once):
    from test_remote_browser import FakeSSH, OA

    f, net, _, _, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, cascade=["oa"],
                                    remote=dict(preflight=False, batch_max=1))
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    queue(f, OA, dict(OA, doi="10.1/next"))
    with monkeypatch.context() as patch:
        patch.setattr(remote, "_collect_once", lambda f, j, *a, **kw: dict(job=j, stage="submitted"))
        fulltext.drain_queue(f, browser=False, poll_once=True)
    assert ssh.calls == ["submit"]
    delivered = []
    def acknowledge(entry, outcome, sub):
        assert ssh.calls.count("submit") == len(delivered) + 1
        delivered.append(entry["doi"])
        return True
    stats = fulltext.drain_queue(f, browser=False, poll_once=poll_once, on_success=acknowledge)
    assert delivered == [OA["doi"], "10.1/next"] and stats["progress"]
    assert ssh.calls == ["submit", "status", "export", "cleanup"] * 2
    assert stats["ok"] == 2 and stats["remaining"] == 0


def test_queue_confirmation_respects_max_wait(tmp_path, monkeypatch):
    import fcntl
    from tbsgen import state
    from test_remote_browser import OA

    f, *_ = mk_remote(tmp_path, {})
    f.ft["remote"]["enabled"] = False
    p = Pipeline(f.cfg)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    p.fetcher = f
    [iid] = p.ingest([OA])
    it = p.store.item(iid)
    it.state.update(stage="S3")
    it.finish("parked", "fulltext_unavailable")
    qp = fulltext.queue_path(f.cfg)
    fulltext.enqueue(qp, fulltext.queue_entry(OA, subscribers=[["", iid]]))
    f.save_imported(OA, {"doi": OA["doi"], "channel": "oa"}, BODY, None)
    clock = FakeClock(10)
    timer = SimpleNamespace(time=time.time, monotonic=clock, sleep=clock.sleep)
    for module in (cli, fulltext, state):
        monkeypatch.setattr(module, "time", timer)
    advance = state.Item.advance
    with open(str(qp) + ".lock", "a") as competing_writer:
        def hold_after_delivery(item, *args, **kwargs):
            advance(item, *args, **kwargs)
            fcntl.flock(competing_writer, fcntl.LOCK_EX)
        monkeypatch.setattr(state.Item, "advance", hold_after_delivery)
        assert cli.main(["fetch", "--run", str(p.store.root), "--max-wait", "0.2"]) == 3
    assert clock() == pytest.approx(10.2) and it.load().state["stage"] == "S4"
    assert cli.main(["fetch", "--run", str(p.store.root)]) == 0 and not fulltext.read_queue(qp)


def test_concurrent_one_shot_reports_pending_delivery(tmp_path, monkeypatch):
    from test_remote_browser import FakeSSH, OA

    f, net, _, _, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, cascade=["oa"], remote=dict(preflight=False))
    p = Pipeline(f.cfg)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    p.fetcher = f
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    queue(f, OA)
    def awaiting_delivery(*_):
        assert ssh.calls[-1] == "cleanup"
        assert cli.main(["fetch", "--run", str(p.store.root)]) == 3
        return False
    stats = fulltext.drain_queue(f, on_success=awaiting_delivery)
    assert stats["undelivered"] == stats["remaining"] == 1


@pytest.mark.parametrize("stage,limit", [("S5G", "daily_new_authoring"), ("S4c", "daily_new_runnability")])
@pytest.mark.parametrize("receipt", ["missing", "yesterday"])
def test_launch_intent_books_daily_paper_and_recovery_rechecks(tmp_path, monkeypatch, stage, limit, receipt):
    from tbsgen import stages_a

    p, it, launcher = _agent_item(tmp_path, monkeypatch, {}, stage=stage)
    p.cfg["code"][limit] = 1
    admissions = p.store.root / f"{stage.lower()}-admissions.jsonl"
    reserve = p.ledger.reserve
    def crash(*args, **kwargs):
        if kwargs["send"]:
            # Before even the sent marker, both durable records exist.
            [row] = [json.loads(line) for line in admissions.read_text().splitlines()]
            rec = json.loads(it.p(stage, "run.json").read_text())
            assert row["job"] == rec["job_name"] and row["day"] == stages_a._cst_day()
            assert rec["launch_attempted"] and row["item"] == it.item_id
            raise SystemExit
        return reserve(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(p.ledger, "reserve", crash)
        with pytest.raises(SystemExit):
            p.step(it)
    [row] = [json.loads(line) for line in admissions.read_text().splitlines()]
    row["day"] = "2000-01-01"
    admissions.write_text((json.dumps(row) + "\n" if receipt == "yesterday" else "") +
                          json.dumps(dict(item="other", day=stages_a._cst_day())) + "\n")
    p.step(p.store.item(it.item_id))
    assert not launcher.calls
    assert json.loads(it.p(stage, "run.json").read_text())["not_started"]
    assert [json.loads(line)["item"] for line in admissions.read_text().splitlines()] == ["other"]
    with p.ledger._db() as db:
        assert db.execute("select cost_kind,usd from calls").fetchall() == [("not_launched", 0.0)]


@pytest.mark.parametrize("receipt,state,allowed", [
    ("leased", None, True), ("leased", "done", False), ("leased", "running", False),
    ("collected", "running", False), ("collected", "missing", False), ("collected", "done", True)])
def test_release_stale_requires_remote_proof(tmp_path, monkeypatch, capsys, receipt, state, allowed):
    from test_remote_browser import FakeSSH

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {})
    p = Pipeline(f.cfg)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    p.fetcher = f
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    job = "old-job"
    path = remote._job_state(f.cfg, job)
    path.parent.mkdir()
    path.write_text(json.dumps(dict(job=job, stage=receipt)))
    remote.lease_and_snapshot(ledger, job, 1)
    job_dir = wd / "jobs" / job
    if state:
        job_dir.mkdir(parents=True)
        if state != "missing":  # a job directory without status.json is still present
            (job_dir / "status.json").write_text(json.dumps(dict(state=state, ts=time.time())))
    args = ["fetch", "--run", str(p.store.root)]
    assert cli.main(args + ["--status"]) == 0
    assert ("--release-stale old-job" in capsys.readouterr().err) == (receipt == "leased")
    assert not ssh.calls
    ssh.down = True
    assert cli.main(args + ["--release-stale", job, "--legacy-worker-stopped"]) == 3
    ssh.down = False
    assert cli.main(args + ["--release-stale", job, "--legacy-worker-stopped"]) == (0 if allowed else 3)
    stored = json.loads(path.read_text())
    if allowed:
        assert stored["release_stale"] in capsys.readouterr().out
        assert stored["stage"] == ("collected" if receipt == "collected" else "dropped")
        assert not job_dir.exists()
    else:
        assert stored == dict(job=job, stage=receipt) and job_dir.exists()
    assert ledger.db.execute("select count(*) from attempts where channel='lease'").fetchone()[0] == int(not allowed)


def test_fetch_max_wait_bounds_secrets_command(tmp_path, monkeypatch):
    import subprocess
    from test_remote_browser import OA

    f, _, _, ledger, _, _ = mk_remote(tmp_path, {}, remote=dict(preflight=False, secrets_cmd=["fake-secrets"]))
    queue(f, OA)
    def timeout(cmd, **kw):
        assert cmd == ["fake-secrets"] and 0 < kw["timeout"] <= 0.2
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(remote.subprocess, "run", timeout)
    result = remote.run_job(f, [OA], ["oa"], ssh_runner=lambda *_: pytest.fail("unexpected submit"), max_poll_s=0.2)
    assert "timed out" in result["refused"] and not result["progress"]
    assert not remote.outstanding_jobs(f.cfg) and not ledger.leased_papers()


def test_fetch_cache_import_lock_respects_max_wait(tmp_path, monkeypatch, capsys):
    import fcntl
    from test_remote_browser import OA, FakeSSH

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, remote=dict(preflight=False))
    p = Pipeline(f.cfg)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    p.fetcher = f
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    queue(f, OA)
    with monkeypatch.context() as patch:
        patch.setattr(remote, "_collect_once", lambda f, j, *a, **kw: dict(job=j, stage="submitted"))
        remote.run_job(f, [OA], ["oa"], ssh_runner=ssh, max_poll_s=0)
    cache = f.write_dir()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".write.lock").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert cli.main(["fetch", "--run", str(p.store.root), "--max-wait", "0.2"]) == 3
    assert "timed out waiting for the fetch cache" in capsys.readouterr().err
    assert ledger.leased_papers() and "cleanup" not in ssh.calls
    assert cli.main(["fetch", "--run", str(p.store.root)]) == 0
    assert f.cached(OA) and not ledger.leased_papers() and ssh.calls.count("submit") == 1


def test_remote_defers_short_midnight_window_without_progress(tmp_path, monkeypatch):
    from test_remote_browser import OA, FakeSSH

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, remote=dict(preflight=False), cascade=["oa"])
    queue(f, OA)
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    midnight = remote._midnight_after(time.time())
    ledger.clock = FakeClock(midnight - 659)
    before = fulltext.read_queue(fulltext.queue_path(f.cfg))
    for _ in range(2):
        stats = fulltext.drain_queue(f, poll_once=True, browser=False)
        assert not stats["progress"] and stats["remote"]["retry_after"] == midnight
        assert not ssh.calls and not ledger.leased_papers()
        assert before == fulltext.read_queue(fulltext.queue_path(f.cfg))
    ledger.clock.sleep(660)
    fulltext.drain_queue(f, poll_once=True, browser=False)
    assert ssh.calls.count("submit") == 1


@pytest.mark.parametrize("stage,cap", [("S4c", "daily_new_runnability"), ("S5G", "daily_new_authoring")])
@pytest.mark.parametrize("claimed", [True, False])
def test_duplicate_tmux_response_reconciles_permit(tmp_path, monkeypatch, stage, cap, claimed):
    import subprocess

    p, it, launcher = _agent_item(tmp_path, monkeypatch, {}, stage=stage)
    p.cfg["code"][cap] = 1

    def duplicate(cmd, **kwargs):
        assert cmd[1] == "new-session"
        if claimed:
            assert launcher(cmd).returncode == 0
        return subprocess.CompletedProcess(cmd, 1, "", "duplicate session")

    monkeypatch.setattr(subprocess, "run", duplicate)
    p.step(it)
    rec = json.loads(it.p(stage, "run.json").read_text())
    assert rec["state"] == ("launched" if claimed else "launch_failed")
    assert "confirm_launch_after" not in rec
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls").fetchone() == (
            ("sent", None) if claimed else ("error", 0))
    assert bool((p.store.root / f"{stage.lower()}-admissions.jsonl").read_text().strip()) == claimed


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "current"])
def test_release_stale_during_worker_startup(tmp_path, legacy):
    import fcntl
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {})
    queue(f, OA)
    job = "delayed-submit"
    receipt = dict(job=job, stage="leased") if legacy else dict(
        job=job, stage="submitted", host=f.ft["remote"]["host"], workdir=str(wd), submitted_at=ledger.clock())
    path = remote._job_state(f.cfg, job)
    path.parent.mkdir()
    path.write_text(json.dumps(receipt))
    remote.lease_and_snapshot(ledger, job, 1, fetcher=f, entries=[OA])
    ssh = FakeSSH(wd, net)
    wd.mkdir()
    with (wd / "worker.lock").open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert "error" in remote.release_stale(f, job, ssh_runner=ssh, legacy_worker_stopped=True)
    assert json.loads(path.read_text()) == receipt
    assert ledger.leased_papers() == {fulltext.paper_ident(OA)}
    if legacy:
        assert remote.release_stale(f, job, ssh_runner=ssh, legacy_worker_stopped=True)["lease"] == "released"
    else:
        assert "error" in remote.release_stale(f, job, ssh_runner=ssh)
        assert not ssh.calls
        f.ft["remote"]["stale_after_s"] = 0
        assert remote.collect(f, job, ssh, max_poll_s=0)["stage"] == "dropped"
        assert (wd / "jobs" / job / "expired").exists()
    assert not ledger.leased_papers()


@pytest.mark.parametrize("poll_once", [True, False])
def test_cleanup_pending_delivers_outside_batch(tmp_path, monkeypatch, poll_once):
    from test_remote_browser import FakeSSH, OA

    f, net, _, _, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, cascade=["oa"],
                                    remote=dict(preflight=False, batch_max=1))
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    queue(f, OA)
    set_stage = remote._set_stage

    def crash(cfg, job, stage, **kwargs):
        if stage == "collected":
            raise SystemExit
        set_stage(cfg, job, stage, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(remote, "_set_stage", crash)
        with pytest.raises(SystemExit):
            fulltext.drain_queue(f, browser=False, poll_once=True)
    assert ssh.calls[-1] == "cleanup" and f.cached(OA)
    queue(f, dict(OA, doi="10.1/new"))
    qp = fulltext.queue_path(f.cfg)
    fulltext.write_queue(qp, list(reversed(fulltext.read_queue(qp))))
    delivered = []

    def acknowledge(entry, outcome, sub):
        assert ssh.calls.count("submit") == len(delivered) + 1
        delivered.append(entry["doi"])
        return True

    stats = fulltext.drain_queue(f, max_papers=1, browser=False, poll_once=poll_once, on_success=acknowledge)
    assert delivered == [OA["doi"], "10.1/new"]
    assert stats["remaining"] == 0 and stats["ok"] == 2


def test_fetch_max_wait_includes_ledger_initialization(tmp_path, monkeypatch, capsys):
    import sqlite3

    f, _, _, ledger, _, _ = mk_remote(tmp_path, {})
    p = Pipeline(f.cfg)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    with sqlite3.connect(ledger.path) as writer:
        writer.execute("begin immediate")
        started = time.monotonic()
        assert cli.main(["fetch", "--run", str(p.store.root), "--max-wait", "0.1"]) == 3
        assert time.monotonic() - started < 0.6
        assert "download ledger" in capsys.readouterr().err
    assert cli.main(["fetch", "--run", str(p.store.root), "--max-wait", "1", "--status"]) == 0
    assert p.fetcher.ledger.db.execute("pragma busy_timeout").fetchone() == (120000,)


@pytest.mark.parametrize("delay,submitted", [(1, True), (301, False)])
def test_short_job_uses_remaining_spec_window(tmp_path, monkeypatch, delay, submitted):
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, remote=dict(preflight=False, max_job_s=600))
    clock = ledger.clock = FakeClock(remote._midnight_after(time.time()) - 43200)
    ssh = FakeSSH(wd, net)
    queue(f, OA)

    def prepare(*args, **kwargs):
        clock.sleep(delay)
        return {}

    monkeypatch.setattr(remote, "load_secrets", prepare)
    result = remote.run_job(f, [OA], ["oa"], ssh_runner=ssh, max_poll_s=0)
    assert ("submit" in ssh.calls) == submitted
    assert result["progress"] == submitted
    assert not ledger.leased_papers()


@pytest.mark.parametrize("stage,cap", [("S4c", "daily_new_runnability"), ("S5G", "daily_new_authoring")])
@pytest.mark.parametrize("claimed", [False, True], ids=["revoke-first", "claim-first"])
@pytest.mark.parametrize("rollover", [False, True], ids=["same-day", "next-day-breaker"])
@pytest.mark.parametrize("acknowledged", [False, True], ids=["worker-killed", "tmux-ack"])
def test_agent_permit_fences_late_client_at_daily_cap(tmp_path, monkeypatch, stage, cap, claimed, rollover,
                                                     acknowledged):
    import subprocess
    from tbsgen import calib, codeagent, stages_a

    p, it, launcher = _agent_item(tmp_path, monkeypatch, {}, stage=stage)
    p.cfg["code"][cap] = 1
    clock = FakeClock(1704038399)  # 23:59:59 Asia/Shanghai
    monkeypatch.setattr(time, "time", clock)
    original_day = stages_a._cst_day()
    [iid] = p.ingest([dict(doi="10.1/other", title="Other")])
    other = p.store.item(iid)
    for file in ("fulltext.txt", "S3c_code.json"):
        other.p(file).write_bytes(it.p(file).read_bytes())
    other.state.update(stage=stage)
    other.save()
    clients = []

    def submit(cmd, **kwargs):
        assert cmd[1] == "new-session"
        clients.append(cmd)
        if acknowledged:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise SystemExit("client spawned")

    with monkeypatch.context() as patch:
        patch.setattr(codeagent.subprocess, "run", submit)
        if acknowledged:
            p.step(it)
        else:
            with pytest.raises(SystemExit, match="client spawned"):
                p.step(it)
    monkeypatch.setattr(calib, "session_alive", lambda *_: claimed)
    [client] = clients
    rec = json.loads(it.p(stage, "run.json").read_text())
    permit = Path(rec["launch_permit"])
    assert permit.is_file()
    if claimed:
        # The live shell claims before creating the job directory or log.
        claim, command = client[-1].split("\n", 1)
        assert subprocess.Popen(["bash", "-c", claim]).wait() == 0
    if rollover:
        clock.sleep(2)
        p.ledger.open_breaker(rec["model"], "shared breaker open")

    p.step(p.store.item(it.item_id))
    recovered = json.loads(it.p(stage, "run.json").read_text())
    assert recovered["state"] == ("launched" if claimed else "launch_failed")
    assert Path(str(permit) + (".claimed" if claimed else ".revoked")).is_file()
    assert not Path(rec["job_dir"]).exists()
    with p.ledger._db() as db:
        assert db.execute("select day,status,usd from calls").fetchall() == [
            (original_day, "sent" if claimed else "error", None if claimed else 0)]
    if not rollover:
        p.step(other)
        assert len(launcher.calls) == int(not claimed)
        if claimed:
            assert other.load().state["reason"].startswith("budget:daily_")
    if claimed:
        # Finish the already claimed command through the fake Harbor launcher.
        launcher.finish(command)
        p.step(p.store.item(it.item_id))
        with p.ledger._db() as db:
            assert db.execute("select usd from calls where id=?", (rec["ledger_call"],)).fetchone() == (launcher.cost,)
    else:
        assert launcher(client).returncode == 97
        assert not Path(rec["job_dir"]).exists() and not Path(rec["log"]).exists()
    assert len(launcher.calls) <= 1


def test_collected_cache_reactivates_subscribers_before_running_batch(tmp_path, monkeypatch):
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, cascade=["oa"],
                                         remote=dict(preflight=False, batch_max=1))
    p = Pipeline(f.cfg, fetcher=f)
    first, second = p.ingest([OA, dict(OA, doi="10.1/second", title="Second")])
    qp = fulltext.queue_path(f.cfg)

    def park(iid):
        it = p.store.item(iid)
        it.state.update(stage="S3", status="parked", reason="fulltext_unavailable")
        it.save()
        entry = fulltext.queue_entry(json.loads(it.p("S1_candidate.json").read_text()),
                                     subscribers=[[str(p.store.root), iid]])
        fulltext.enqueue(qp, entry)

    park(first)
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    set_stage = remote._set_stage

    def crash(cfg, job, stage, **kw):
        set_stage(cfg, job, stage, **kw)
        if stage == "collected":
            raise SystemExit("before subscriber delivery")

    args = ["fetch", "--run", str(p.store.root), "--max", "1", "--no-browser"]
    with monkeypatch.context() as patch:
        patch.setattr(remote, "_set_stage", crash)
        with pytest.raises(SystemExit, match="before subscriber"):
            cli.main(args)
    park(second)
    fulltext.write_queue(qp, list(reversed(fulltext.read_queue(qp))))

    def running(cmd, stdin, timeout):
        assert p.store.item(first).state["stage"] == "S4"
        assert [e["doi"] for e in fulltext.read_queue(qp)] == ["10.1/second"]
        if shlex.split(cmd)[3] == "submit":
            return 0, b'{"ok": true}', b""
        return 0, b'{"state": "running"}', b""

    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: running)
    assert cli.main(args + ["--max-wait", "0.4"]) == 3
    assert ledger.leased_papers() == {"doi:10.1/second"}
    assert p.store.item(second).state["stage"] == "S3" and ssh.calls.count("submit") == 1


def test_fetch_max_wait_includes_run_ledger_initialization(tmp_path, capsys):
    import sqlite3

    p = pipeline(tmp_path)
    args = ["fetch", "--run", str(tmp_path), "--max-wait", "0.1"]
    with sqlite3.connect(p.ledger.path) as writer:
        writer.execute("begin immediate")
        started = time.monotonic()
        assert cli.main(args) == 3
        assert time.monotonic() - started < 0.6
        assert "run ledger" in capsys.readouterr().err
    assert cli.main(args) == 0


def test_fetch_max_wait_includes_lease_write_lock(tmp_path, monkeypatch, capsys):
    import sqlite3
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, remote=dict(preflight=False))
    p = Pipeline(f.cfg, fetcher=f)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    ssh = FakeSSH(wd, net)
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    queue(f, OA)
    args = ["fetch", "--run", str(p.store.root), "--max-wait", "0.2"]
    with sqlite3.connect(ledger.path) as writer:
        writer.execute("begin immediate")
        started = time.monotonic()
        assert cli.main(args) == 3
        assert time.monotonic() - started < 0.6
        assert "download ledger lease" in capsys.readouterr().err
    assert not ledger.leased_papers() and not ssh.calls and not remote.outstanding_jobs(f.cfg)
    assert cli.main(args[:-2]) == 0 and ssh.calls.count("submit") == 1


def test_release_stale_legacy_submit_before_job_directory(tmp_path, monkeypatch):
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, remote=dict(preflight=False))
    ssh = FakeSSH(wd, net)
    pending = []
    queue(f, OA)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)

    def receive(cmd, payload, timeout):
        pending.append((cmd, payload, timeout))
        raise SystemExit("worker has read the spec")

    with pytest.raises(SystemExit, match="read the spec"):
        remote.run_job(f, [OA], ["oa"], ssh_runner=receive)
    [receipt] = remote.outstanding_jobs(f.cfg)
    job = receipt["job"]
    receipt.pop("submitted_at")
    receipt["stage"] = "leased"
    path = remote._job_state(f.cfg, job)
    path.write_text(json.dumps(receipt))
    assert not (wd / "jobs" / job).exists()
    assert "--legacy-worker-stopped" in remote.release_stale(f, job, ssh_runner=ssh)["error"]
    assert json.loads(path.read_text()) == receipt and ledger.leased_papers() == {fulltext.paper_ident(OA)}
    assert ssh(*pending[0])[0] == 0
    assert remote.collect(f, job, ssh, max_poll_s=0)["stage"] == "collected"
    assert f.cached(OA) and ledger.usage()["oa"]["ok"] == 1 and not ledger.leased_papers()


def test_release_stale_old_legacy_receipt_requires_worker_stopped(tmp_path):
    from test_remote_browser import FakeSSH

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {}, remote=dict(max_job_s=1200))
    clock = ledger.clock = FakeClock(time.time())
    job = "old-job"
    path = remote._job_state(f.cfg, job)
    path.parent.mkdir()
    path.write_text(json.dumps(dict(job=job, stage="leased", created=clock())))
    remote.lease_and_snapshot(ledger, job, 1)
    ssh = FakeSSH(wd, net)
    clock.sleep(1800)
    assert "error" in remote.release_stale(f, job, ssh_runner=ssh)
    clock.sleep(86400)
    assert "--legacy-worker-stopped" in remote.release_stale(f, job, ssh_runner=ssh)["error"]
    assert not ssh.calls and ledger.db.execute("select count(*) from attempts where channel='lease'").fetchone() == (1,)
    assert remote.release_stale(f, job, ssh_runner=ssh, legacy_worker_stopped=True)["lease"] == "released"


@pytest.mark.parametrize("stage", ["S9", "S9R"])
@pytest.mark.parametrize("claimed", [False, True], ids=["revoke-first", "claim-first"])
def test_disabled_submit_reconciles_only_original_launch(tmp_path, monkeypatch, stage, claimed):
    import subprocess
    from test_review_s9r import _env as review_env

    if stage == "S9R":
        p, iid, _ = review_env(tmp_path)
        it = p.store.item(iid)
        it.state.update(stage=stage, calib_label="hard", review_round=1)
        it.save()
    else:
        p, [iid], _, _ = _env(tmp_path)
        it = p.store.item(iid)
    path = it.p("review" if stage == "S9R" else "calib", "runs.json")
    clients = []

    def crash(cmd, **kw):
        assert cmd[1] == "new-session"
        clients.append(cmd)
        raise SystemExit

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", crash)
        with pytest.raises(SystemExit):
            p.step(it)
    [rec] = json.loads(path.read_text())
    [client] = clients
    permit = Path(rec["launch_permit"])
    assert permit.is_file()
    shell = client[-1]
    claim, command = shell.split("\n", 1)
    if claimed:
        subprocess.run(["bash", "-c", claim], check=True)
    p.cfg["calib"]["allow_submit"] = False
    p.step(p.store.item(iid))
    recovered = json.loads(path.read_text())[0]
    assert recovered["state"] == ("launched" if claimed else "launch_failed")
    assert not Path(rec["job_dir"]).exists() and not Path(rec["log"]).exists()
    with p.ledger._db() as db:
        assert db.execute("select id,status,usd from calls").fetchall() == [
            (rec["ledger_call"], "sent" if claimed else "error", None if claimed else 0)]
    result = subprocess.run(["bash", "-c", command if claimed else shell])
    assert result.returncode == (0 if claimed else 97)
    assert Path(rec["job_dir"]).is_dir() == claimed
    assert Path(str(permit) + (".claimed" if claimed else ".revoked")).is_file()


def test_fetch_result_merge_respects_max_wait(tmp_path, monkeypatch, capsys):
    import sqlite3
    from test_remote_browser import FakeSSH, OA

    f, net, _, ledger, wd, _ = mk_remote(tmp_path, {"repo.example": FAKE_PDF}, remote=dict(preflight=False))
    p = Pipeline(f.cfg, fetcher=f)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    monkeypatch.setattr(remote, "_midnight_after", lambda now: now + 86400)
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda _: TITLE + "\n" + BODY)
    ssh = FakeSSH(wd, net)
    queue(f, OA)
    with monkeypatch.context() as patch:
        patch.setattr(remote, "_collect_once", lambda f, j, *a, **kw: dict(job=j, stage="submitted"))
        remote.run_job(f, [OA], ["oa"], ssh_runner=ssh, max_poll_s=0)

    with sqlite3.connect(ledger.path) as writer:
        def lock_after_export(cmd, payload, timeout):
            result = ssh(cmd, payload, timeout)
            if ssh.calls[-1] == "export":
                writer.execute("begin immediate")
            return result

        monkeypatch.setattr(remote, "default_ssh_runner", lambda _: lock_after_export)
        start = time.monotonic()
        assert cli.main(["fetch", "--run", str(p.store.root), "--max-wait", "0.2"]) == 3
        assert time.monotonic() - start < 0.7
        assert "download ledger" in capsys.readouterr().err
        assert ssh.calls[-1] == "export" and ledger.leased_papers()
    monkeypatch.setattr(remote, "default_ssh_runner", lambda _: ssh)
    assert cli.main(["fetch", "--run", str(p.store.root)]) == 0
    assert not ledger.leased_papers() and ledger.usage()["oa"]["ok"] == 1 and ssh.calls.count("submit") == 1


@pytest.mark.parametrize("after_timeout", ["retry", "stop-file", "sigterm"])
def test_fetch_loop_retries_startup_lock(tmp_path, monkeypatch, capsys, after_timeout):
    import sqlite3

    p = pipeline(tmp_path)
    deadline = []
    pipeline_factory = cli._pipeline
    with sqlite3.connect(p.ledger.path) as writer:
        writer.execute("begin immediate")

        def initialize(args):
            deadline.append(args.fetch_deadline)
            try:
                return pipeline_factory(args)
            except TimeoutError:
                writer.rollback()
                if after_timeout == "stop-file":
                    (tmp_path / "driver.stop").touch()
                elif after_timeout == "sigterm":
                    signal.raise_signal(signal.SIGTERM)
                raise

        drained = []
        def drain(*args, **kwargs):
            drained.append(kwargs["max_wait_s"])
            (tmp_path / "driver.stop").touch()
            return {}

        monkeypatch.setattr(cli, "_pipeline", initialize)
        monkeypatch.setattr(fulltext, "drain_queue", drain)
        assert cli.main(["fetch", "--run", str(tmp_path), "--loop", "0.01", "--max-wait", "0.15"]) == 0
    assert "run ledger" in capsys.readouterr().err
    if after_timeout == "retry":
        assert len(deadline) == 2 and deadline[1] > deadline[0] + 0.15
        assert len(drained) == 1 and 0 < drained[0] <= 0.15
    else:
        assert len(deadline) == 1 and not drained


def _interrupted_job(tmp_path, monkeypatch, stage, acknowledged, *, remote_route=False, delayed_client=None):
    """Interrupt an actual tmux submission before the launcher writes any job files."""
    import subprocess
    from test_review_s9r import _env as review_env

    if stage in ("S4c", "S5G"):
        cap = "daily_new_authoring" if stage == "S5G" else "daily_new_runnability"
        p, it, _ = _agent_item(tmp_path, monkeypatch, {}, stage=stage, **{cap: 1})
        p.cfg["code"]["max_concurrent_agent_jobs"] = 1
        if remote_route:
            p.cfg["agent"]["remote_launchers"] = None
        path = it.p(stage, "run.json")
    elif stage == "S9":
        p, [iid], _, _ = _env(tmp_path)
        it = p.store.item(iid)
        path = it.p("calib/runs.json")
        if delayed_client is not None:
            from tbsgen.state import write_json
            write_json(path, [dict(job_name="previous-student", who="glm", n=1, outcome="fail")])
    else:
        p, iid, _ = review_env(tmp_path)
        it = p.store.item(iid)
        it.state.update(stage="S9R", review_round=1, calib_label="hard")
        it.save()
        path = it.p("review/runs.json")

    if remote_route:
        p.cfg["calib"]["remote_launchers"] = None

    def claim(cmd, **kwargs):
        assert cmd[1] == "new-session"
        if delayed_client is not None:
            delayed_client.append(cmd[-1])
        else:
            preamble, _ = cmd[-1].split("\n", 1)
            assert subprocess.Popen(["sh", "-c", preamble]).wait() == 0
        if acknowledged:
            return subprocess.CompletedProcess(cmd, 0)
        raise SystemExit("controller interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", claim)
        if acknowledged:
            p.step(it)
        else:
            with pytest.raises(SystemExit, match="interrupted"):
                p.step(it)
    p.cfg["calib"]["allow_submit"] = False
    def records():
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]

    job = records()[-1]["job_name"]

    def read():
        return next(rec for rec in records() if rec["job_name"] == job)

    assert Path(read()["launch_permit"] + (".claimed" if delayed_client is None else "")).is_file()
    return p, it, path, read


@pytest.mark.parametrize("stage", ["S4c", "S5G", "S9", "S9R"])
@pytest.mark.parametrize("acknowledged", [False, True])
def test_claimed_local_job_waits_for_confirmed_session_loss(tmp_path, monkeypatch, stage, acknowledged):
    from tbsgen import calib, stages_a
    p, it, _, read = _interrupted_job(tmp_path, monkeypatch, stage, acknowledged)
    rec = read()
    assert not Path(rec["job_dir"]).exists() and not Path(rec["log"]).exists()
    monkeypatch.setattr(calib, "session_alive", lambda *_: None)
    p.step(it.load())
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls").fetchall() == [("sent", None)]
    monkeypatch.setattr(calib, "session_alive", lambda *_: False)
    p.run(only=[it.item_id], stages={stage}, max_steps=2)
    assert read()["cost_kind"] == "not_launched" and read()["cost_usd"] == 0
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls").fetchall() == [("error", 0)]
    if stage in ("S4c", "S5G"):
        assert not stages_a._admissions(p, stage) and read()["job_name"] in stages_a._nostart(it.load(), stage)


@pytest.mark.parametrize("stage", ["S4c", "S5G", "S9", "S9R"])
@pytest.mark.parametrize("marker,exit_code", [("job", "0"), ("run", "137"), ("job", None)])
@pytest.mark.parametrize("acknowledged", [False, True])
def test_claimed_remote_job_parks_until_operator_resolution(tmp_path, monkeypatch, stage, marker, exit_code, acknowledged):
    from tbsgen import calib, stages_a, stages_b
    from tbsgen.state import write_json
    from test_calib_s9 import _trial
    p, it, path, read = _interrupted_job(tmp_path, monkeypatch, stage, acknowledged)
    rec = read()
    jd = Path(rec["job_dir"])
    write_json((jd if marker == "job" else path.parent) / "tbsgen-remote.json", {})
    if exit_code is not None:
        Path(rec["log"] + ".exit").write_text(exit_code)
    monkeypatch.setattr(calib, "session_alive", lambda *_: False)
    p.run(only=[it.item_id], stages={stage})
    assert it.load().state["reason"] == f"remote_unknown: {rec['job_name']}"
    assert p.status()["remote_unknown"] == 1
    assert read()["state"] == "remote_unknown"
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls").fetchall() == [("sent", None)]
    if stage in ("S4c", "S5G"):
        assert stages_a._admissions(p, stage)[0]["job"] == rec["job_name"]
        assert "slots full" in stages_a._agent_admission(p, p.store.item("another"), stage)
    else:
        assert stages_b.running_jobs(p) == [rec["job_name"]]
    command = ["calib", "resolve-remote", "--run", str(p.store.root), "--item", it.item_id,
               "--job", rec["job_name"]]
    if stage in ("S4c", "S5G") and marker == "run":
        with pytest.raises(SystemExit, match="incomplete"):
            cli.main(command + ["--synced"])
        write_json(jd / "config.json", json.loads(Path(rec["config"]).read_text()))
        with pytest.raises(SystemExit, match="config or trial"):
            cli.main(command + ["--not-started"])
        _write_agent_outputs(_trial(jd, cost=3.5), stage)
        write_json(jd / "result.json", dict(finished_at="complete"))
        assert cli.main(command + ["--synced"]) == 0
        assert read()["state"] == "finished" and read()["cost_usd"] == 3.5
    else:
        with pytest.raises(SystemExit, match="permit already claimed: the launch may be running"):
            cli.main(command + ["--not-started"])
        assert read()["state"] == "remote_unknown"
        with p.ledger._db() as db:
            assert db.execute("select status,usd from calls").fetchall() == [("sent", None)]
        return
    assert p.status()["remote_unknown"] == 0


@pytest.mark.parametrize("stage", ["S9", "S9R"])
@pytest.mark.parametrize("permit_state", ["revoked", "claimed", "stale", "available"])
def test_printed_manual_command_checks_current_permit(tmp_path, monkeypatch, stage, permit_state):
    from tbsgen.report import daily_report
    from test_review_s9r import _env as review_env
    p, iid, jobs = review_env(tmp_path, allow_submit=False)
    it = p.store.item(iid)
    it.state.update(stage=stage, review_round=1, calib_label="hard")
    it.save()
    p.step(it)
    path = it.p("calib" if stage == "S9" else "review", "runs.json")
    [rec] = json.loads(path.read_text())
    cmd = shlex.split(rec["command"])
    assert cmd[:3] == ["tbsgen", "calib", "launch"] and rec["command"] in daily_report(p).read_text()
    permit = Path(rec["launch_permit"])
    if permit_state == "stale":
        it.goto("S10", "moved on")
    elif permit_state != "available":
        permit.rename(str(permit) + "." + permit_state)
    monkeypatch.setattr(cli, "_pipeline", lambda _: p)
    if permit_state == "available":
        assert cli.main(cmd[1:]) == 0
        assert (jobs / rec["job_name"] / "config.json").is_file()
        assert Path(str(permit) + ".claimed").is_file()
        with pytest.raises(SystemExit, match="current open attempt"):
            cli.main(cmd[1:])
        assert p.ledger.summary()["calls_total"] == 1
    else:
        with pytest.raises(SystemExit, match="current open attempt|revoked or claimed"):
            cli.main(cmd[1:])
        assert not list(jobs.iterdir()) and p.ledger.summary()["calls_total"] == 0


@pytest.mark.parametrize("credentials_returned", [False, True])
def test_pre_submit_deadline_bounds_lease_release(tmp_path, monkeypatch, credentials_returned):
    import sqlite3
    from test_remote_browser import OA
    f, _, _, ledger, _, _ = mk_remote(tmp_path, {}, remote=dict(preflight=False))
    queue(f, OA)
    with sqlite3.connect(ledger.path) as writer:
        def timeout(fetcher, names, deadline):
            writer.execute("begin immediate")
            if credentials_returned:
                time.sleep(max(0, deadline - time.monotonic()))
                return {}
            raise TimeoutError("credentials unavailable")
        monkeypatch.setattr(remote, "load_secrets", timeout)
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="download ledger lease"):
            remote.run_job(f, [OA], ["oa"], ssh_runner=lambda *_: pytest.fail("unexpected submit"),
                           max_poll_s=0.2)
        assert time.monotonic() - start < 0.8
    assert remote.outstanding_jobs(f.cfg)[0]["stage"] == "leasing" and ledger.leased_papers()


def test_manual_harbor_command_uses_original_config(tmp_path, monkeypatch):
    import subprocess
    import sys
    from tbsgen.report import daily_report
    from tbsgen.state import write_json

    run = tmp_path / "run with spaces"
    cfg = tmp_path / "manual settings.toml"
    cfg.write_text('[budget]\ndaily_usd_total = 100\n[calib]\njobs_dir = '
                   + json.dumps(str(tmp_path / "jobs")) + '\nallow_submit = false\n')
    p = Pipeline(load_config(cfg, {"run": {"root": str(run)}}))
    [iid] = p.ingest([dict(doi="10.1/manual")])
    it = p.store.item(iid)
    it.p("pkg").mkdir()
    it.goto("S9", "manual calibration")
    p.step(it)
    path = it.p("calib/runs.json")
    [rec] = json.loads(path.read_text())
    command = rec["command"]
    assert shlex.split(command)[-2:] == ["--config", str(cfg)]
    # Even a report generated without --config for a legacy raw command must print a guarded entry.
    write_json(path, [dict(rec, command="obsolete-launcher")])
    report = daily_report(Pipeline(load_config(None, {"run": {"root": str(run)}}))).read_text()
    assert command in report and "obsolete-launcher" not in report
    receipt = tmp_path / "arguments"
    harbor = tmp_path / "harbor"
    harbor.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$MANUAL_RECEIPT"\n')
    harbor.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("MANUAL_RECEIPT", str(receipt))
    argv = [sys.executable, "-m", "tbsgen.cli", *shlex.split(command)[1:]]
    assert subprocess.run(argv, check=True).returncode == 0
    assert subprocess.run(argv, capture_output=True).returncode != 0
    assert receipt.read_text().splitlines() == ["run", "--config", rec["config"], "--yes"]


@pytest.mark.parametrize("stage", ["S4c", "S5G"])
@pytest.mark.parametrize("acknowledged", [False, True])
def test_remote_agent_rsync_23_then_gone_keeps_admission(tmp_path, monkeypatch, stage, acknowledged):
    from tbsgen import calib, codeagent, stages_a
    p, it, _, read = _interrupted_job(tmp_path, monkeypatch, stage, acknowledged, remote_route=True)
    rec = read()
    assert rec["remote_route"] and not Path(rec["job_dir"]).exists()
    Path(rec["log"]).write_text("remote job: GONE\nrsync error: partial transfer (code 23)\n")
    Path(rec["log"] + ".exit").write_text("23\n")
    monkeypatch.setattr(calib, "session_alive", lambda *_: False)
    assert codeagent.poll_agent(p.cfg, it, stage, p.ledger)["state"] == "remote_unknown"
    p.run(only=[it.item_id], stages={stage})
    assert it.load().state["reason"] == f"remote_unknown: {rec['job_name']}"
    assert stages_a._admissions(p, stage)[0]["job"] == rec["job_name"]
    assert "slots full" in stages_a._agent_admission(p, p.store.item("another"), stage)
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls").fetchall() == [("sent", None)]
    cmd = ["calib", "resolve-remote", "--run", str(p.store.root), "--item", it.item_id, "--job", rec["job_name"]]
    with pytest.raises(SystemExit, match="permit already claimed"):
        cli.main(cmd + ["--not-started"])
    assert read()["state"] == "remote_unknown" and stages_a._admissions(p, stage)


@pytest.mark.parametrize("stage", ["S4c", "S5G"])
@pytest.mark.parametrize("outcome", ["--synced", "--ended-without-result", "--died", "--not-started"])
def test_agent_remote_resolution_checks(tmp_path, monkeypatch, stage, outcome):
    from test_calib_s9 import _trial
    from tbsgen import calib, stages_a
    from tbsgen.state import write_json

    p, it, _, read = _interrupted_job(tmp_path, monkeypatch, stage, acknowledged=True, remote_route=True)
    rec = read()
    jd = Path(rec["job_dir"])
    marker = dict(local_task=rec["task"], remote_task="/worker/task")
    if outcome == "--synced":
        write_json(it.p(stage, "tbsgen-remote.json"), marker)
    monkeypatch.setattr(calib, "session_alive", lambda *_: None)
    p.run(only=[it.item_id], stages={stage})
    assert read()["state"] == "remote_unknown"
    cmd = ["calib", "resolve-remote", "--run", str(p.store.root), "--item", it.item_id, "--job", rec["job_name"]]
    if outcome == "--not-started":
        with pytest.raises(SystemExit, match="force-local-gone"):
            cli.main(cmd + [outcome])
        with monkeypatch.context() as patch:
            patch.setattr(calib, "session_alive", lambda *_: True)
            with pytest.raises(SystemExit, match="launcher is alive"):
                cli.main(cmd + [outcome, "--force-local-gone"])
        with pytest.raises(SystemExit, match="permit already claimed"):
            cli.main(cmd + [outcome, "--force-local-gone"])
        # Legacy records without a permit still accept independent confirmation.
        rec = read()
        del rec["launch_permit"]
        write_json(it.p(stage, "run.json"), rec)
        assert cli.main(cmd + [outcome, "--force-local-gone"]) == 0
        assert read()["cost_kind"] == "not_launched" and not stages_a._admissions(p, stage)
        return

    conf = json.loads(Path(rec["config"]).read_text())
    conf["tasks"][0]["path"] = marker["remote_task"]
    write_json(jd / "config.json", conf)
    trial = _trial(jd, cost=2.5)
    if outcome == "--synced":
        _write_agent_outputs(trial, stage)
        assert read()["remote_marker"] == marker
        it.p(stage, "tbsgen-remote.json").unlink()
        write_json(jd / "result.json", dict(finished_at="complete"))
        with pytest.raises(SystemExit, match="complete result"):
            cli.main(cmd + ["--died"])
        assert cli.main(cmd + [outcome]) == 0
    else:
        with pytest.raises(SystemExit, match="identity mismatch"):
            cli.main(cmd + [outcome, "--remote-task", "/worker/incorrect"])
        assert read()["state"] == "remote_unknown"
        assert cli.main(cmd + [outcome, "--remote-task", marker["remote_task"]]) == 0
        assert read()["outcome"] == "no_result"
    assert json.loads((jd / "tbsgen-remote.json").read_text()) == marker
    assert read()["state"] == "finished" and read()["cost_usd"] == 2.5
    assert read()["ledger_call"] == rec["ledger_call"]
    with p.ledger._db() as db:
        assert db.execute("select status, usd from calls").fetchall() == [
            ("ok" if outcome == "--synced" else "error", 2.5)]


@pytest.mark.parametrize("stage", ["S4c", "S5G", "S9", "S9R"])
def test_not_started_revokes_before_late_tmux_client(tmp_path, monkeypatch, stage):
    import subprocess
    from tbsgen import calib, stages_a

    delayed = []
    p, it, _, read = _interrupted_job(tmp_path, monkeypatch, stage, False, remote_route=True, delayed_client=delayed)
    rec = read()
    assert rec["remote_route"]
    if stage == "S9":
        assert rec["who"] == "teacher"
    monkeypatch.setattr(calib, "session_alive", lambda *_: False)
    p.run(only=[it.item_id], stages={stage})
    assert read()["state"] == "remote_unknown"
    command = ["calib", "resolve-remote", "--run", str(p.store.root), "--item", it.item_id,
               "--job", rec["job_name"], "--not-started"]
    assert cli.main(command) == 0
    assert Path(rec["launch_permit"] + ".revoked").is_file()
    [shell_command] = delayed
    assert subprocess.Popen(["bash", "-c", shell_command]).wait() == 97
    assert not Path(rec["log"]).exists() and not Path(rec["job_dir"]).exists()
    assert read()["cost_kind"] == "not_launched"
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls").fetchall() == [("error", 0)]
    if stage in ("S4c", "S5G"):
        assert not stages_a._admissions(p, stage)


def _write_agent_outputs(trial, stage):
    from tbsgen import codeagent

    out = trial / "verifier/out"
    for name in codeagent.agent_output_files(stage):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return out


@pytest.mark.parametrize("stage", ["S4c", "S5G"])
def test_partial_agent_rsync_with_zero_exit_keeps_reservation(tmp_path, monkeypatch, stage):
    from test_calib_s9 import _trial
    from tbsgen import calib, stages_a
    from tbsgen.state import write_json

    p, it, _, read = _interrupted_job(tmp_path, monkeypatch, stage, True, remote_route=True)
    rec = read()
    jd = Path(rec["job_dir"])
    write_json(jd / "config.json", json.loads(Path(rec["config"]).read_text()))
    trial = _trial(jd, reward=1, cost=3.5)
    (trial / "verifier/metrics.jsonl").unlink()  # the agent verifier does not emit S9 metrics
    write_json(jd / "result.json", dict(finished_at="complete"))
    monkeypatch.setattr(calib, "session_alive", lambda *_: True)
    p.step(it.load())
    assert it.load().state["status"] == "waiting"
    Path(rec["log"]).write_text("rsync error: partial transfer (code 23)\n")
    Path(rec["log"] + ".exit").write_text("0\n")
    p.run(only=[it.item_id], stages={stage})
    assert it.load().state["reason"] == f"remote_unknown: {rec['job_name']}"
    assert read()["state"] == "remote_unknown" and read()["ledger_call"] == rec["ledger_call"]
    assert stages_a._admissions(p, stage)[0]["job"] == rec["job_name"]
    assert "slots full" in stages_a._agent_admission(p, p.store.item("another"), stage)

    cmd = ["calib", "resolve-remote", "--run", str(p.store.root), "--item", it.item_id,
           "--job", rec["job_name"], "--synced"]
    out = _write_agent_outputs(trial, stage)
    if stage == "S4c":
        write_json(out / "runnability.json", dict(runnable=True, reason="ok", entry="run example"))
        write_json(out / "runs/0.json", dict(rc=0, cmd="run example", log="runs/0.log"))
        missing = out / "runs/0.log"
    else:
        missing = out / "task/baselines/naive.py"
        missing.unlink()
    with pytest.raises(SystemExit, match="incomplete remote trial set"):
        cli.main(cmd)
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls").fetchall() == [("sent", None)]
    missing.write_text("complete")
    assert cli.main(cmd) == 0
    assert read()["state"] == "finished" and read()["ledger_call"] == rec["ledger_call"]
    with p.ledger._db() as db:
        assert db.execute("select status,usd from calls").fetchall() == [("ok", 3.5)]


@pytest.mark.parametrize("stage,directory", [("S9", "calib"), ("S9R", "review")])
def test_legacy_intent_command_migrates_on_load(tmp_path, stage, directory):
    from tbsgen.report import daily_report
    from tbsgen.state import write_json
    from test_review_s9r import _env as review_env

    p, iid, _ = review_env(tmp_path, allow_submit=False)
    it = p.store.item(iid)
    it.state.update(stage=stage, review_round=3, calib_label="hard")
    it.save()
    p.step(it)
    path = it.p(directory, "runs.json")
    [rec] = json.loads(path.read_text())
    guarded = rec["command"]
    raw = f"cd /worker && ./example-launch.sh run {rec['config']}"
    del rec["launch_permit"]
    write_json(path, [dict(rec, command=raw)])
    it.finish("paused", "budget:daily_model")
    loaded = Pipeline(p.cfg)
    assert json.loads(path.read_text()) == [dict(rec, command=guarded, legacy_command=raw)]
    assert loaded.ledger.summary()["calls_total"] == 0
    it.finish("waiting", "prepared for manual submission")
    before = path.stat().st_mtime_ns
    report = daily_report(Pipeline(p.cfg)).read_text()
    assert guarded in report and raw not in report and path.stat().st_mtime_ns == before
