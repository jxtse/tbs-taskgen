"""failclass: one failure classifier for S4c / S5G / S9 jobs + stage-level zero-charge / repeat-park tests."""
import json
import os
import subprocess
from pathlib import Path

from tbsgen import calib, codeagent, failclass, stages_a
from tbsgen.config import load_config
from tbsgen.pipeline import Pipeline
from tbsgen.state import read_json, write_json


def _jobdir(tmp_path, name="job", *, log=None, agent=False, exception=None, cost=None, finished=False):
    jd = tmp_path / "jobs" / name
    jd.mkdir(parents=True, exist_ok=True)
    if agent:
        tr = jd / "trial"
        (tr / "agent").mkdir(parents=True, exist_ok=True)
        (tr / "agent" / "trajectory.json").write_text("{}")
        tr.mkdir(exist_ok=True)
        body = {"agent_result": {"cost_usd": cost, "n_input_tokens": 100, "n_output_tokens": 20},
                "verifier_result": {"rewards": {"reward": 0}}}
        if exception:
            body["exception_info"] = {"exception_type": exception, "exception_message": log or ""}
        (tr / "result.json").write_text(json.dumps(body))
    if finished:
        (jd / "result.json").write_text(json.dumps({"finished_at": "2026-11-27T00:00:00Z"}))
    return jd


def test_no_model_call_egress_probe():
    """The Harbor allowlist probe dies before the agent: no agent files, no usage, no cost."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        jd = _jobdir(Path(d), log="network_mode='allowlist' is not supported")
        v = failclass.classify(jd, {"job_dir": str(jd)},
                               log_text="Traceback...\nValueError: network_mode='allowlist' is not supported\n")
    assert v.kind == failclass.NO_MODEL_CALL
    assert "not supported" in v.signature and "ValueError" in v.signature
    assert v.evidence["agent_files"] is False and v.evidence["reported_cost"] is False


def test_no_model_call_apt_429_before_agent():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        jd = _jobdir(Path(d))
        v = failclass.classify(jd, {"job_dir": str(jd)},
                               log_text="E: Failed to fetch http://deb.debian.org/... 429 Too Many Requests")
    assert v.kind == failclass.NO_MODEL_CALL
    assert "429" in failclass.normalize_message(v.message) or "too many" in v.message.lower()


def test_no_model_call_pull_access_denied():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        jd = _jobdir(Path(d))
        v = failclass.classify(jd, {"job_dir": str(jd)},
                               log_text="ERROR: failed to solve: pull access denied for tbsgen/private")
    assert v.kind == failclass.NO_MODEL_CALL
    assert "pull access denied" in v.signature


def test_infra_after_model_addtestsdirerror_keeps_cost_evidence():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        jd = _jobdir(Path(d), agent=True, exception="AddTestsDirError", cost=13.5,
                     log="AddTestsDirError: cannot create /app/tests dir")
        v = failclass.classify(jd, {"job_dir": str(jd)})
    assert v.kind == failclass.INFRA_AFTER_MODEL
    assert v.evidence["reported_cost"] is True and v.evidence["agent_files"] is True
    assert v.exception == "AddTestsDirError" and "addtestsdirerror" in v.signature.lower()


def test_infra_after_model_readerror_mid_run():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        jd = _jobdir(Path(d), agent=True, exception="ReadError", cost=2.0,
                     log="ReadError: [Errno 5] Input/output error: '/app/out/x.txt'")
        v = failclass.classify(jd, {"job_dir": str(jd)})
    assert v.kind == failclass.INFRA_AFTER_MODEL


def test_model_failure_is_not_infra():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        jd = _jobdir(Path(d), agent=True, exception="NonZeroAgentExitCodeError", cost=0.6,
                     log="NonZeroAgentExitCodeError: agent exited with code 2")
        v = failclass.classify(jd, {"job_dir": str(jd)})
    assert v.kind == failclass.MODEL_FAILED
    assert failclass.repeat_is_deterministic(v) is False


def test_signature_strips_ids_numbers_and_paths():
    a = failclass.signature("ReadError", "cannot read /tmp/run-123/item-abc/job-77/result.json after 42 s")
    b = failclass.signature("ReadError", "cannot read /tmp/run-999/item-xyz/job-13/result.json after 7 s")
    assert a == b and a[1] == b[1]


def test_legacy_record_without_job_dir_is_not_proof():
    """No job dir and no launcher exit: the job may have run somewhere else; keep the full-allowance charge."""
    v = failclass.classify(None, {"state": "finished", "job_name": "old", "cost_usd": None})
    assert v.kind == failclass.MODEL_FAILED


# ----------------------------------------------------------------------------------------------- stage level
def _cfg(tmp_path, **code):
    return load_config(None, {"run": {"root": str(tmp_path / "run")},
                              "code": dict(grounded=True, host="local", remote_root=str(tmp_path / "remote"),
                                           run_tag="t", **code),
                              "sandbox": {"backend": "ssh", "host": "bh", "remote_dir": "/w"},
                              "fulltext": {"allow_network": False, "cache_dirs": []}})


def _item(tmp_path, monkeypatch, launcher, stage="S4c", **code):
    cfg = _cfg(tmp_path, agent_harbor_root=str(tmp_path), agent_launcher="x.sh", **code)
    cfg["agent"]["remote_launchers"] = []
    cfg["budget"]["daily_usd_total"] = 100.0
    P = Pipeline(cfg)
    [a] = P.ingest([{"doi": "10.1/a", "title": "Fast smoothing"}])
    it = P.store.item(a)
    it.p("fulltext.txt").write_text("paper text")
    write_json(it.p("S3c_code.json"), {"ok": True, "pin": "abc", "remote_dir": "/r/x", "primary_url": "x",
                                       "license": "MIT"})
    it.state["stage"] = stage
    it.save()
    monkeypatch.setattr(codeagent.subprocess, "run", launcher)
    return P, it


class ProbeLauncher:
    """Launcher that writes a log + exit status and never creates a Harbor job (the egress-probe timeout shape)."""

    def __init__(self, message):
        self.message, self.calls = message, []

    def __call__(self, args, **kw):
        preamble, command = args[-1].split("\n", 1)
        rc = subprocess.Popen(["bash", "-c", preamble]).wait()
        if rc:
            return subprocess.CompletedProcess(args, rc, "", "")
        self.calls.append(command)
        log = Path(command.split(" > ")[1].split(" 2>&1")[0].strip("'"))
        log.write_text(self.message + "\n")
        exit_file = Path(str(log) + ".exit")
        exit_file.write_text("1\n")
        old = 1.0
        os.utime(exit_file, (old, old))          # the launcher ended long ago (past calib.stale_remote_grace_s)
        return subprocess.CompletedProcess(args, 0, "", "")


def test_no_model_call_never_charges_or_drops(tmp_path, monkeypatch):
    launcher = ProbeLauncher("ValueError: network_mode='allowlist' is not supported")
    P, it = _item(tmp_path, monkeypatch, launcher)
    P.run(max_steps=1, only=[it.item_id])                       # launch job 1
    P.run(max_steps=1, only=[it.item_id])                       # observe: zero-model failure, backoff armed
    st = it.load().state
    assert st["status"] == "active" and st.get("reason") is None, st
    assert float(st.get("infra_backoff_until") or 0) > 0
    assert "S4c:infra" not in st.get("attempts", {})
    assert not read_json(it.p("S4c", "spent.json"))            # nothing charged yet
    P.run(max_steps=1, only=[it.item_id])                       # backoff: wait, do not launch
    st = it.load().state
    assert st["status"] == "waiting" and st["reason"] == "S4c:infra_backoff", st
    assert len(launcher.calls) == 1
    # backoff over: the failed job is charged 0/0 and a fresh job is launched; the paper is still alive
    st["infra_backoff_until"] = 0
    it.save()
    P.run(max_steps=1, only=[it.item_id])
    spent = read_json(it.p("S4c", "spent.json"))
    assert spent and list(spent.values())[0] == {"usd": 0.0, "sec": 0.0}, spent
    assert it.load().state["status"] == "waiting" and len(launcher.calls) == 2
    assert it.load().state.get("reason") != "code_runnability_agent_failed"


def test_two_identical_no_model_signatures_park(tmp_path, monkeypatch):
    launcher = ProbeLauncher("ERROR: failed to solve: pull access denied for tbsgen/private")
    P, it = _item(tmp_path, monkeypatch, launcher)
    P.run(max_steps=1, only=[it.item_id])                       # launch job 1
    P.run(max_steps=1, only=[it.item_id])                       # observe job 1 -> retry with backoff
    st = it.load().state
    st["infra_backoff_until"] = 0
    it.save()
    P.run(max_steps=1, only=[it.item_id])                       # launch job 2
    P.run(max_steps=1, only=[it.item_id])                       # observe job 2 -> same signature -> park
    st = it.load().state
    assert st["status"] == "parked" and st["reason"].startswith("infra_repeat:"), st
    assert len(launcher.calls) == 2                            # no third job
    assert "S4c:infra" not in st.get("attempts", {})            # and never counted as an infra attempt


def _model_failure_job(tmp_path, name, exception="UnknownApiError", cost=0.6):
    jd = tmp_path / "jobs" / name
    tr = jd / "trial"
    (tr / "agent").mkdir(parents=True, exist_ok=True)
    (tr / "agent" / "trajectory.json").write_text("{}")
    (tr / "result.json").write_text(json.dumps({
        "agent_result": {"cost_usd": cost, "n_input_tokens": 10, "n_output_tokens": 5},
        "exception_info": {"exception_type": exception, "exception_message": "upstream failed"},
        "verifier_result": {"rewards": {"reward": 0}}}))
    return jd


def test_model_failure_keeps_old_infra_budget(tmp_path, monkeypatch):
    P, it = _item(tmp_path, monkeypatch, lambda *a, **k: None)
    for name in ("j1", "j2"):
        rec = dict(state="finished", job_name=name, job_dir=str(_model_failure_job(tmp_path, name)),
                   outcome="agent_error", launch_attempted=1.0, finished_ts=2.0, budget={"max_usd": 8.0},
                   failure_class=None, error_signature=None)
        write_json(it.p("S4c", "run.json"), rec)
        act, why = stages_a._retry_agent_infra(P, it.load(), "S4c", rec)
        assert act == "retry", (act, why)
    st = it.load().state
    assert st.get("attempts", {}).get("S4c:infra") == 2
    assert st["status"] == "active" and "infra_repeat" not in (st.get("reason") or "")


def test_after_model_infra_repeat_parks(tmp_path, monkeypatch):
    P, it = _item(tmp_path, monkeypatch, lambda *a, **k: None)
    for name in ("j1", "j2"):
        rec = dict(state="finished", job_name=name, job_dir=str(_model_failure_job(tmp_path, name, "AddTestsDirError", 13.5)),
                   outcome="agent_error", launch_attempted=1.0, finished_ts=2.0, budget={"max_usd": 20.0},
                   failure_class=None, error_signature=None)
        write_json(it.p("S4c", "run.json"), rec)
        act, why = stages_a._retry_agent_infra(P, it.load(), "S4c", rec)
    assert act == "park" and why.startswith("infra_repeat:"), (act, why)
    assert it.load().state["status"] == "active"                # the caller parks it; retry flag not set


def test_parked_remote_unknown_with_proven_end_is_reconciled(tmp_path, monkeypatch):
    """A parked remote_unknown item whose launcher ended and whose job dir proves nothing ran must not hold a slot."""
    P, it = _item(tmp_path, monkeypatch, lambda *a, **k: None)
    log = tmp_path / "job-x.launch.log"
    log.write_text("network_mode='allowlist' is not supported\n")
    Path(str(log) + ".exit").write_text("1\n")
    os.utime(Path(str(log) + ".exit"), (1.0, 1.0))
    write_json(it.p("S4c", "run.json"), dict(
        job_name="job-x", state="remote_unknown", job_dir=str(tmp_path / "jobs" / "absent"),
        log=str(log), remote_route=True, item=it.item_id, ledger_stage="S4c", launch_attempted=1.0))
    it.state.update(status="parked", reason="remote_unknown: job-x")
    it.save()
    P.run(max_steps=1, only=[it.item_id])
    st = it.load().state
    assert st["status"] != "parked" and not str(st.get("reason") or "").startswith("remote_unknown")
    assert float(st.get("infra_backoff_until") or 0) > 0   # zero-model job: retried after a backoff, not parked


def test_s9_next_action_skips_no_model_runs_and_parks_repeats():
    cfg = load_config(None, {})
    base = dict(who="glm", outcome="infra", result_type="infra")
    sig = {"error_signature": "AddTestsDirError:failed to add tests directory to environment",
           "error_signature_short": "abc123", "deterministic": True}
    runs = [dict(base, no_model_call=True)]
    assert calib.next_action(runs, cfg) == ("run", "glm")       # zero-model run is not an infra attempt
    runs = [dict(base, no_model_call=True)] * 5                # many zero-model runs never reach parked:infra ...
    assert calib.next_action(runs, cfg) == ("run", "glm")
    nm = dict(base, no_model_call=True, **sig)                 # ... but a repeated signature still parks them
    assert calib.next_action([nm, nm], cfg) == ("label", "parked:infra_repeat:abc123")
    transient = dict(base, error_signature="UnknownApiError:x", error_signature_short="t", deterministic=False)
    assert calib.next_action([transient, transient], cfg) == ("run", "glm")   # model failures: old budget
    runs = [dict(base, **sig), dict(base, **sig)]
    assert calib.next_action(runs, cfg) == ("label", "parked:infra_repeat:abc123")


def test_transient_signature_needs_three_repeats_to_park():
    """A rate-limit / dropped-connection signature is retried (after the backoff) once more than a deterministic one:
    two apt 429s in a row during a mirror storm must not park every item at once."""
    assert failclass.is_transient("RuntimeError:E: Failed to fetch http://deb.debian.org 429 Too Many Requests")
    assert failclass.is_transient("ReadError:")
    assert not failclass.is_transient("ValueError:network_mode='allowlist' is not supported")
    assert not failclass.is_transient("AddTestsDirError:failed to add tests directory to environment")
    code = {"infra_repeat_park": 2}
    assert failclass.repeat_limit("ReadError:", code) == 3
    assert failclass.repeat_limit("ValueError:pull access denied", code) == 2
    assert failclass.repeat_limit("ReadError:", {"infra_repeat_park": 2, "infra_repeat_park_transient": 5}) == 5
    assert failclass.repeat_limit("ReadError:", {"infra_repeat_park": 4}) == 4      # never below the base limit

    cfg = load_config(None, {})
    base = dict(who="glm", outcome="infra", result_type="infra", no_model_call=True,
                error_signature="RuntimeError:E: Failed to fetch 429 Too Many Requests",
                error_signature_short="apt429", deterministic=True)
    assert calib.next_action([base, base], cfg) == ("run", "glm")
    assert calib.next_action([base, base, base], cfg) == ("label", "parked:infra_repeat:apt429")


def test_stage_transient_repeat_parks_on_third(tmp_path, monkeypatch):
    launcher = ProbeLauncher("E: Failed to fetch http://deb.debian.org/debian/pool/x.deb  429  Too Many Requests")
    P, it = _item(tmp_path, monkeypatch, launcher)
    for n in range(3):
        P.run(max_steps=1, only=[it.item_id])                   # launch job n+1
        P.run(max_steps=1, only=[it.item_id])                   # observe it
        st = it.load().state
        if n < 2:
            assert st["status"] != "parked", (n, st)
            st["infra_backoff_until"] = 0
            it.save()
    st = it.load().state
    assert st["status"] == "parked" and st["reason"].startswith("infra_repeat:"), st
    assert len(launcher.calls) == 3


def test_pre_agent_text_counts_only_at_the_end_of_the_log(tmp_path):
    """An apt 429 that a setup retry got past, followed by a long agent run, is not proof of 'no model call' when the
    local job copy is missing: only the log's tail is searched."""
    early = "E: Failed to fetch http://deb.debian.org/x.deb  429  Too Many Requests\n"
    run = "".join(f"agent step {i}: tool call ok\n" for i in range(200)) + "harbor exit 1\n"
    v = failclass.classify(tmp_path / "absent", {"job_dir": str(tmp_path / "absent")}, log_text=early + run)
    assert v.kind == failclass.MODEL_FAILED, v
    v = failclass.classify(tmp_path / "absent", {"job_dir": str(tmp_path / "absent")}, log_text=run + early)
    assert v.kind == failclass.NO_MODEL_CALL, v
