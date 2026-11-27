"""Code-grounded S3c (fetch + pin), S4c / S5G agent stages (fake launcher), author-name leak gate, S9R checklist."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from tbsgen import codeagent, prerelease, review
from tbsgen.config import load_config
from tbsgen.pipeline import Pipeline
from tbsgen.state import read_json, write_json


def _git_repo(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "fastsmoother.py").write_text("def smooth(x):\n    return x\n")
    (d / "requirements.txt").write_text("numpy\n")
    (d / "LICENSE").write_text("MIT License\n")
    env = dict(os.environ, GIT_AUTHOR_NAME="a", GIT_AUTHOR_EMAIL="a@b", GIT_COMMITTER_NAME="a", GIT_COMMITTER_EMAIL="a@b")
    for c in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-qm", "x"]):
        subprocess.run(c, cwd=d, check=True, env=env, capture_output=True)
    return d


def _bare(d: Path) -> str:
    """file:// URL of a bare clone named <d>.git; fetch_spec appends .git to the repository URL"""
    subprocess.run(["git", "clone", "-q", "--bare", str(d), str(d) + ".git"], check=True, capture_output=True)
    return f"file://{d}"


def _cfg(tmp_path, **code):
    return load_config(None, {"run": {"root": str(tmp_path / "run")},
                              "code": dict(grounded=True, host="local", remote_root=str(tmp_path / "remote"),
                                           run_tag="t", **code),
                              "sandbox": {"backend": "ssh", "host": "bh", "remote_dir": "/w"},
                              "fulltext": {"allow_network": False, "cache_dirs": []}})


def test_s3c_fetch_local_git_pins_sha(tmp_path):
    repo = _git_repo(tmp_path / "src")
    cfg = _cfg(tmp_path)
    prim = dict(host="github", url=_bare(repo), key="github.com/a/fastsmoother")
    r = codeagent.s3c_fetch(cfg, "it1", prim)
    assert r["ok"], r
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert r["sha"] == sha and r["pin"] == sha
    assert r["languages"].get("Python") is not None and "requirements.txt" in r["dep_files"]
    assert r["license_file"].startswith("MIT")
    assert Path(r["remote_dir"], "repo", "fastsmoother.py").exists()
    assert Path(r["remote_dir"], "PIN").read_text().strip() == sha
    # a second fetch of the same pin reuses (never replaces) the stored tree
    (Path(r["remote_dir"]) / "repo" / "marker").write_text("x")
    r2 = codeagent.s3c_fetch(cfg, "it1", prim)
    assert r2["remote_dir"] == r["remote_dir"] and (Path(r["remote_dir"]) / "repo" / "marker").exists()


def test_s3c_too_large(tmp_path):
    repo = _git_repo(tmp_path / "src")
    (repo / "big.bin").write_bytes(os.urandom(2_200_000))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=a", "-c", "user.email=a@b", "commit", "-qm", "big"], cwd=repo, check=True)
    cfg = _cfg(tmp_path, max_repo_mb=1)
    r = codeagent.s3c_fetch(cfg, "it2", dict(host="github", url=_bare(repo)))
    assert not r["ok"] and r["why"] == "too_large"


def test_s3c_stage_drop_and_advance(tmp_path):
    repo = _git_repo(tmp_path / "src")
    cfg = _cfg(tmp_path)
    P = Pipeline(cfg)
    [a] = P.ingest([{"doi": "10.1/a", "title": "t"}])
    it = P.store.item(a)
    write_json(it.p("S3b_code.json"), {"keep": True, "primary": {"host": "github", "url": _bare(repo),
                                                                 "license": "MIT"}})
    it.state["stage"] = "S3c"
    it.save()
    P.run(max_steps=1, until="S3c")
    st = P.store.item(a).state
    assert st["stage"] == "S4", (st, read_json(it.p("S3c_code.json")))
    assert read_json(it.p("S3c_code.json"))["license"] == "MIT"


class FakeLauncher:
    """Stands in for tmux + the build-host launcher: writes a finished Harbor job with the given /app/out files."""

    def __init__(self, files: dict, cost=3.5):
        self.files, self.cost, self.calls = files, cost, []

    def __call__(self, args, **kw):
        preamble, command = args[-1].split("\n", 1)
        rc = subprocess.Popen(["bash", "-c", preamble]).wait()
        if rc:
            return subprocess.CompletedProcess(args, rc, "", "")
        self.finish(command)
        return subprocess.CompletedProcess(args, 0, "", "")

    def finish(self, command):
        self.calls.append(command)
        cfgp = Path(command.split(" run ")[1].split(" >")[0].strip("'"))
        conf = json.loads(cfgp.read_text())
        jd = Path(conf["jobs_dir"]) / conf["job_name"]
        tr = jd / (conf["job_name"] + "__x")
        (tr / "verifier" / "out").mkdir(parents=True)
        for rel, body in self.files.items():
            (tr / "verifier" / "out" / rel).parent.mkdir(parents=True, exist_ok=True)
            (tr / "verifier" / "out" / rel).write_text(body)
        (jd / "config.json").write_text(json.dumps(conf))
        (tr / "result.json").write_text(json.dumps({
            "agent_result": {"cost_usd": self.cost, "n_input_tokens": 10, "n_output_tokens": 5},
            "verifier_result": {"rewards": {"reward": 1}},
            "agent_execution": {"started_at": "2026-11-12T00:00:00Z", "finished_at": "2026-11-12T00:30:00Z"}}))
        (jd / "result.json").write_text(json.dumps({"finished_at": "2026-11-12T00:31:00Z"}))


def _agent_item(tmp_path, monkeypatch, files, stage="S4c", **code):
    cfg = _cfg(tmp_path, agent_harbor_root=str(tmp_path), agent_launcher="x.sh", **code)
    cfg["agent"]["remote_launchers"] = []  # FakeLauncher writes a local Harbor job.
    cfg["budget"]["daily_usd_total"] = 100.0
    P = Pipeline(cfg)
    [a] = P.ingest([{"doi": "10.1/a", "title": "Fast smoothing"}])
    it = P.store.item(a)
    it.p("fulltext.txt").write_text("paper text")
    write_json(it.p("S3c_code.json"), {"ok": True, "pin": "abc", "remote_dir": "/r/x", "primary_url": "https://github.com/a/fastsmoother",
                                       "license": "MIT"})
    it.state["stage"] = stage
    it.save()
    fl = FakeLauncher(files)
    monkeypatch.setattr(codeagent.subprocess, "run", fl)
    return P, it, fl


@pytest.mark.parametrize("stage", ["S4c", "S5G"])
@pytest.mark.parametrize("names,remote", [(None, True), (["x.sh"], True), ([], False)])
def test_agent_records_remote_route_at_launch(tmp_path, monkeypatch, stage, names, remote):
    P, it, _ = _agent_item(tmp_path, monkeypatch, {}, stage=stage)
    P.cfg["agent"]["remote_launchers"] = names
    P.run(max_steps=1, only=[it.item_id])
    assert read_json(it.p(stage, "run.json"))["remote_route"] is remote


def _agent_failure(it, stage, exc):
    rec = read_json(it.p(stage, "run.json"))
    jd = Path(rec["job_dir"])
    result = jd / (rec["job_name"] + "__x") / "result.json"
    trial = read_json(result)
    trial["exception_info"] = {"exception_type": exc}
    trial["verifier_result"]["rewards"]["reward"] = 0
    trial["agent_result"]["cost_usd"] = 0.60
    trial["agent_execution"]["finished_at"] = trial["agent_execution"]["started_at"]
    write_json(result, trial)
    Path(rec["log"] + ".exit").write_text("0\n")
    return rec


@pytest.mark.parametrize("stage,reason", [("S5G", "s5g_missing_files"), ("S4c", "code_runnability_agent_failed")])
@pytest.mark.parametrize("route", ["remote", "local"])
def test_agent_error_retries_twice_then_drops(tmp_path, monkeypatch, stage, reason, route):
    from tbsgen import stages_a

    P, it, launcher = _agent_item(tmp_path, monkeypatch, {}, stage=stage, max_concurrent_agent_jobs=1)
    P.cfg["agent"]["remote_launchers"] = ["x.sh"] if route == "remote" else []
    P.cfg["taskimage"]["enabled"] = True
    other = P.store.item("another")
    names = set()
    for attempt, exc in enumerate(("UnknownApiError", "AgentTimeoutError", "NonZeroAgentExitCodeError"), 1):
        P.step(it.load())
        rec = _agent_failure(it, stage, exc)
        assert rec["remote_route"] == (route == "remote")
        assert rec["job_name"] not in names
        names.add(rec["job_name"])
        assert "slots full" in stages_a._agent_admission(P, other, stage)
        P.step(it.load())
        finished = read_json(it.p(stage, "run.json"))
        assert finished["state"] == "finished" and finished["exception"] == exc
        assert finished["outcome"] == "agent_error" and finished["cost_usd"] == 0.60
        assert stages_a._agent_admission(P, other, stage) is None
        with P.ledger._db() as db:
            assert db.execute("select status, usd from calls").fetchall() == [("error", 0.60)] * attempt
        st = it.load().state
        assert st["stage"] == stage and st["attempts"][f"{stage}:infra"] == attempt
        if attempt < 3:
            assert st["status"] == "active" and finished["retry"]
        else:
            assert st["status"] == "dropped" and st["reason"] == reason
    assert len(launcher.calls) == 3


@pytest.mark.parametrize("stage,reason", [("S5G", "s5g_missing_files"), ("S4c", "code_runnability_agent_failed")])
def test_remote_agent_safety_refusal_drops(tmp_path, monkeypatch, stage, reason):
    P, it, _ = _agent_item(tmp_path, monkeypatch, {}, stage=stage)
    P.cfg["agent"]["remote_launchers"] = ["x.sh"]
    P.step(it.load())
    _agent_failure(it, stage, "AgentSafetyRefusalError")
    P.step(it.load())
    st = it.load().state
    assert st["status"] == "dropped" and st["reason"] == reason
    assert f"{stage}:infra" not in st["attempts"]


def test_resolve_synced_agent_error_retries(tmp_path, monkeypatch):
    from tbsgen import cli

    P, it, _ = _agent_item(tmp_path, monkeypatch, {}, stage="S5G")
    P.cfg["agent"]["remote_launchers"] = ["x.sh"]
    P.step(it.load())
    rec = _agent_failure(it, "S5G", "SyntheticApiError")
    jd = Path(rec["job_dir"])
    (jd / "result.json").unlink()
    P.step(it.load())
    assert it.load().state["reason"] == f"remote_unknown: {rec['job_name']}"
    command = ["calib", "resolve-remote", "--run", str(P.store.root), "--item", it.item_id,
               "--job", rec["job_name"], "--synced"]
    with pytest.raises(SystemExit, match="incomplete remote trial set"):
        cli.main(command)
    write_json(jd / "result.json", {"finished_at": "complete"})
    assert cli.main(command) == 0
    assert read_json(it.p("S5G/run.json"))["outcome"] == "agent_error"
    P.step(it.load())
    assert read_json(it.p("S5G/run.json"))["retry"]
    P.step(it.load())
    new = read_json(it.p("S5G/run.json"))
    assert new["job_name"] != rec["job_name"] and new["state"] == "launched"


def test_agent_exception_requires_all_configured_trials(tmp_path):
    jd = tmp_path / "job"
    rec = dict(job_dir=str(jd), ledger_stage="S5G", task=str(tmp_path / "task"))
    write_json(jd / "config.json", dict(n_attempts=2, agents=[{}], tasks=[{}]))
    write_json(jd / "result.json", dict(finished_at="complete"))
    failure = dict(exception_info=dict(exception_type="UnknownApiError"))
    write_json(jd / "trial-1/result.json", failure)
    assert not codeagent.complete_remote_agent({}, rec, final=True)
    second = jd / "trial-2/result.json"
    write_json(second, {})
    assert not codeagent.complete_remote_agent({}, rec, final=True)
    second.write_text("{")
    assert not codeagent.complete_remote_agent({}, rec, final=True)
    write_json(second, failure)
    assert codeagent.complete_remote_agent({}, rec, final=True)
    # Before the launcher's final sync, an exception trial without outputs may still be receiving files.
    assert not codeagent.complete_remote_agent({}, rec)


def test_remote_agent_error_waits_for_final_sync(tmp_path, monkeypatch):
    """Summary and exception trial synced first, launcher still running: hold the slot, do not settle."""
    from tbsgen import calib, stages_a

    P, it, _ = _agent_item(tmp_path, monkeypatch, {}, stage="S5G", max_concurrent_agent_jobs=1)
    P.cfg["agent"]["remote_launchers"] = ["x.sh"]
    P.step(it.load())
    rec = _agent_failure(it, "S5G", "UnknownApiError")
    Path(rec["log"] + ".exit").unlink()
    monkeypatch.setattr(calib, "session_alive", lambda c, r: True)
    P.step(it.load())
    run = read_json(it.p("S5G/run.json"))
    assert run["state"] == "launched" and not run.get("retry") and "S5G:infra" not in it.load().state.get("attempts", {})
    assert "slots full" in stages_a._agent_admission(P, P.store.item("another"), "S5G")
    Path(rec["log"] + ".exit").write_text("0\n")
    P.step(it.load())
    run = read_json(it.p("S5G/run.json"))
    assert run["outcome"] == "agent_error" and run["retry"]


def test_remote_agent_without_deliverables_is_final_after_sync(tmp_path, monkeypatch):
    """The agent ended normally but wrote nothing; the verifier on the build host said so. After the launcher's exit
    status, the job is final: the stage handles it (no remote_unknown park holding the slot)."""
    from tbsgen import calib, stages_a

    P, it, _ = _agent_item(tmp_path, monkeypatch, {}, stage="S5G", max_concurrent_agent_jobs=1)
    P.cfg["agent"]["remote_launchers"] = ["x.sh"]
    P.step(it.load())
    rec = read_json(it.p("S5G/run.json"))
    trial = Path(rec["job_dir"]) / (rec["job_name"] + "__x")
    result = read_json(trial / "result.json")
    result["verifier_result"]["rewards"]["reward"] = 0
    write_json(trial / "result.json", result)
    (trial / "verifier").mkdir(exist_ok=True)
    (trial / "verifier/reward.txt").write_text("0\n")
    (trial / "verifier/check.txt").write_text("missing: ['task/instruction.md']\n")
    monkeypatch.setattr(calib, "session_alive", lambda c, r: True)
    P.step(it.load())                                  # no exit status yet: still waiting, slot held
    assert read_json(it.p("S5G/run.json"))["state"] == "launched"
    assert "slots full" in stages_a._agent_admission(P, P.store.item("another"), "S5G")
    Path(rec["log"] + ".exit").write_text("0\n")
    P.step(it.load())
    st = it.load().state
    assert st["status"] != "parked", st
    assert read_json(it.p("S5G/run.json"))["state"] == "finished"
    assert stages_a._agent_admission(P, P.store.item("another"), "S5G") is None
    # A verifier receipt alone, without the launcher's final sync, never counts.
    assert not codeagent._agent_trial_complete(trial, result, "S5G", ["task/instruction.md"])


def test_multi_trial_agent_job_uses_delivered_trial(tmp_path, monkeypatch):
    """Two trials: the first lost its API connection, the second delivered a valid S4c verdict. No retry."""
    files = {"runnability.json": json.dumps({"runnable": True, "reason": "ok", "language": "Python", "seconds": 3,
                                             "entry": "python -m fastsmoother"}),
             "HOWTO.md": "pip install .", "runs/0.json": json.dumps({"cmd": "python -m fastsmoother", "rc": 0,
                                                                      "seconds": 3, "log": "runs/0.log"}),
             "runs/0.log": "ok"}
    P, it, launcher = _agent_item(tmp_path, monkeypatch, files, stage="S4c")
    P.step(it.load())
    rec = read_json(it.p("S4c/run.json"))
    jd = Path(rec["job_dir"])
    good = jd / (rec["job_name"] + "__x")
    bad = jd / (rec["job_name"] + "__a")
    (bad / "verifier").mkdir(parents=True)
    write_json(bad / "result.json", dict(read_json(good / "result.json"),
                                         exception_info={"exception_type": "UnknownApiError"},
                                         verifier_result={"rewards": {"reward": 0}}))
    conf = read_json(jd / "config.json")
    conf["n_attempts"] = 2
    write_json(jd / "config.json", conf)
    P.step(it.load())
    st = it.load().state
    assert st["stage"] == "S5G", st
    run = read_json(it.p("S4c/run.json"))
    assert run["trial"] == str(good) and run["cost_usd"] == 7.0
    assert len(launcher.calls) == 1


def test_s4c_unbacked_runnable_verdict_after_agent_error_retries(tmp_path, monkeypatch):
    """The agent wrote runnable: true and then lost its API connection before the how-to and run receipts."""
    P, it, launcher = _agent_item(tmp_path, monkeypatch, {
        "runnability.json": json.dumps({"runnable": True, "reason": "ok", "entry": "python -m fastsmoother"})},
        stage="S4c")
    P.step(it.load())
    _agent_failure(it, "S4c", "UnknownApiError")
    P.step(it.load())
    st = it.load().state
    assert st["status"] == "active" and st["stage"] == "S4c" and st["attempts"]["S4c:infra"] == 1, st
    assert read_json(it.p("S4c/run.json"))["retry"]


def test_s4c_complete_negative_verdict_after_agent_error_is_kept(tmp_path, monkeypatch):
    P, it, _ = _agent_item(tmp_path, monkeypatch, {
        "runnability.json": json.dumps({"runnable": False, "reason": "needs_gpu"}), "HOWTO.md": "x"}, stage="S4c")
    P.step(it.load())
    _agent_failure(it, "S4c", "AgentTimeoutError")
    P.step(it.load())
    st = it.load().state
    assert st["status"] == "dropped" and st["reason"] == "code_needs_gpu"


def test_s4c_runnable_then_s5g(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, {
        "runnability.json": json.dumps({"runnable": True, "reason": "ok", "language": "Python", "seconds": 3,
                                        "entry": "python -m fastsmoother"}),
        "HOWTO.md": "pip install .; python -m fastsmoother", "runs/0.json": json.dumps({"cmd": "python -m fastsmoother", "rc": 0, "seconds": 3, "log": "runs/0.log"}),
        "runs/0.log": "ok"})
    P.run(max_steps=1, until="S4c")
    assert P.store.item(it.item_id).state["status"] == "waiting"
    task = it.p("S4c", "task")
    assert read_json(task / "environment" / "work" / ".tbsgen-remote-repo") == {"repo_dir": "/r/x/repo", "pin": "abc"}
    assert "/work/repo.PIN" in (task / "environment" / "Dockerfile").read_text()
    assert not (task / "environment" / "work" / "repo").exists()      # never copied off the build host
    P.run(max_steps=2, until="S4c")
    st = P.store.item(it.item_id).state
    assert st["stage"] == "S5G", st
    assert read_json(it.p("S4c_runnability.json"))["cost_usd"] == 3.5
    with P.ledger._db() as db:
        rows = db.execute("SELECT stage, usd, status FROM calls WHERE item=?", (it.item_id,)).fetchall()
    assert rows == [("S4c", 3.5, "ok")]


def test_s4c_runnable_with_a_large_run_log(tmp_path, monkeypatch):
    """A real author run can log more than safe_read's 2 MB: the receipt log still counts as evidence (its tail is
    kept), so a finished runnable job is collected and advances (2026-11-27: a 2.05 MB log left it remote_unknown)."""
    big = "x" * 100 + "\n"
    log = big * 25_000 + "final line: ok\n"                     # ~2.5 MB
    P, it, fl = _agent_item(tmp_path, monkeypatch, {
        "runnability.json": json.dumps({"runnable": True, "reason": "ok", "language": "Python", "seconds": 3,
                                        "entry": "python -m fastsmoother"}),
        "HOWTO.md": "pip install .; python -m fastsmoother",
        "runs/0.json": json.dumps({"cmd": "python -m fastsmoother", "rc": 0, "seconds": 3, "log": "runs/0.log"}),
        "runs/0.log": log})
    P.run(max_steps=1, until="S4c")
    P.run(max_steps=2, until="S4c")
    st = P.store.item(it.item_id).state
    assert st["stage"] == "S5G", st
    kept = it.p("S4c", "run.log").read_text()
    assert kept.startswith("[log truncated:") and kept.endswith("final line: ok\n") and len(kept) < 2_100_000


def test_safe_log_tail_keeps_the_link_rule(tmp_path):
    from tbsgen import codeagent
    out = tmp_path / "out"
    (out / "runs").mkdir(parents=True)
    (out / "runs" / "small.log").write_text("ok\n")
    (out / "runs" / "big.log").write_text("a" * 50 + "TAIL")
    (tmp_path / "secret").write_text("nope")
    (out / "runs" / "link.log").symlink_to(tmp_path / "secret")
    assert codeagent.safe_log_tail(out / "runs/small.log", root=out) == "ok\n"
    tail = codeagent.safe_log_tail(out / "runs/big.log", root=out, limit=10)
    assert tail.endswith("aaaaaaTAIL") and "first 44 of 54 bytes omitted" in tail
    assert codeagent.safe_log_tail(out / "runs/link.log", root=out) is None
    assert codeagent.safe_log_tail(out / "runs/missing.log", root=out) is None


def test_s4c_not_runnable_gpu(tmp_path, monkeypatch):
    P, it, _ = _agent_item(tmp_path, monkeypatch, {
        "runnability.json": json.dumps({"runnable": False, "reason": "needs_gpu"}), "HOWTO.md": "x"})
    P.run(max_steps=3)
    st = P.store.item(it.item_id).state
    assert st["status"] == "dropped" and st["reason"] == "code_needs_gpu"


SPEC = {"generator_assumptions": [], "title": "t", "domain": "d", "field": "f", "form": "ANS", "expert_time_hours": 3, "competencies": ["x"],
        "grounding_vs_extension": "g", "metrics": [{"name": "rmse", "direction": "<=", "on": "public", "desc": "d"},
                                                   {"name": "bias", "direction": "<=", "on": "public", "desc": "d"},
                                                   {"name": "cov", "direction": ">=", "on": "public", "desc": "d"}],
        "public_files": ["obs.csv"], "hidden_files": [], "truth_files": ["truth.csv"], "why_naive_fails": "w",
        "mid_missing_step": "m", "expected_separation": {}, "method_names": ["FastSmoother", "fastsmoother"],
        "resource_metric_gating": False,
        "depgraph": {"deliverables": ["est"], "nodes": [
            {"id": "est", "kind": "deliverable", "desc": "d", "provision": "derive", "key": True, "secret_names": [],
             "carrier": None},
            {"id": "a", "kind": "compute", "desc": "d", "provision": "derive", "key": True, "secret_names": [],
             "carrier": None},
            {"id": "b", "kind": "compute", "desc": "d", "provision": "derive", "key": True, "secret_names": [],
             "carrier": None}],
            "edges": [["a", "est"], ["b", "est"]], "mid_drop": ["a"], "naive_drop": ["a", "b"],
            "difficulty_knobs": {"derive_nodes": 3, "chain_len": 2, "files_crossed": 1, "nonstd_tool": False}}}


def test_agent_error_after_complete_deliverables_is_used(tmp_path, monkeypatch):
    """An agent that wrote every S5G deliverable and then lost its API connection keeps its work."""
    files = {"_spec/spec.json": json.dumps(SPEC),
             "_spec/author_code.json": json.dumps({"package_names": ["fastsmoother"], "main_functions": ["smooth_rts"]}),
             "task/instruction.md": "Estimate the state.", "task/generator/generate.py": "print(1)",
             "task/solution/solve.py": "print(1)", "task/tests/measure.py": "print(1)",
             "task/baselines/mid.py": "print(1)", "task/baselines/naive.py": "print(1)",
             "task/environment/requirements.txt": "fastsmoother\n", "report.md": "ok"}
    P, it, launcher = _agent_item(tmp_path, monkeypatch, files, stage="S5G")
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    P.step(it.load())
    _agent_failure(it, "S5G", "UnknownApiError")
    P.step(it.load())
    st = it.load().state
    assert st["stage"] == "S6A_run", st
    assert read_json(it.p("S5G/run.json"))["outcome"] == "agent_error"
    assert "S5G:infra" not in st.get("attempts", {})
    assert len(launcher.calls) == 1


def test_s5g_writes_same_artifacts(tmp_path, monkeypatch):
    files = {"_spec/spec.json": json.dumps(SPEC),
             "_spec/author_code.json": json.dumps({"package_names": ["fastsmoother"], "main_functions": ["smooth_rts"]}),
             "task/instruction.md": "Estimate the state.", "task/generator/generate.py": "print(1)",
             "task/solution/solve.py": "print(1)", "task/tests/measure.py": "print(1)",
             "task/baselines/mid.py": "print(1)", "task/baselines/naive.py": "print(1)",
             "task/environment/requirements.txt": "fastsmoother\n", "report.md": "ok"}
    P, it, _ = _agent_item(tmp_path, monkeypatch, files, stage="S5G")
    from tbsgen import spec as specmod
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    P.run(max_steps=1, until="S5G")
    P.run(max_steps=1, until="S5G")
    st = P.store.item(it.item_id).state
    assert st["stage"] == "S6A_run", st
    assert (it.p("task", "generator", "generate.py")).exists() and (it.p("_spec", "spec.json")).exists()
    assert not it.p("task", "environment").exists()           # agent-made environment files are discarded
    ac = read_json(it.p("_spec", "author_code.json"))
    assert ac["pin"] == "abc" and ac["license"] == "MIT"
    assert read_json(it.p("_spec", "provenance.json"))["authoring"] == "agent"
    # the work dir carries the contract
    assert (it.p("S5G", "work", "contract", "S6A_GEN.txt")).exists()


def test_author_name_leak_and_readymade(tmp_path):
    task = tmp_path / "task"
    (task / "solution").mkdir(parents=True)
    (task / "instruction.md").write_text("Use smooth_rts from the toolbox.")
    (task / "solution" / "solve.py").write_text("import fastsmoother\n")
    spec = {"_author_code": {"package_names": ["fastsmoother"], "main_functions": ["smooth_rts"],
                             "repo_url": "https://github.com/alice/fastsmoother"}}
    al = prerelease.author_name_leaks(task, spec)
    assert not al["ok"] and al["findings"][0]["name"] == "smooth_rts"
    rm = prerelease.readymade_check(task, {}, spec)
    assert not rm["ok"] and rm["findings"][0]["name"] == "fastsmoother"
    (task / "instruction.md").write_text("Implement a Rauch-Tung-Striebel smoother.")
    assert prerelease.author_name_leaks(task, spec)["ok"]


def test_review_checklist_has_author_code_item(tmp_path):
    assert "i_author_code" in review.CHECKLIST
    item = tmp_path / "item"
    (item / "pkg" / "tests").mkdir(parents=True)
    (item / "_spec").mkdir()
    write_json(item / "_spec" / "author_code.json", {"package_names": ["x"]})
    review.write_bundle(item, tmp_path / "b", {"history": []})
    assert (tmp_path / "b" / "records" / "author_code.json").exists()


def test_old_config_unchanged_routes(tmp_path):
    cfg = load_config(None, {"run": {"root": str(tmp_path)}})
    assert cfg["code"]["grounded"] is False and cfg["sandbox"]["backend"] == "docker"


def test_sandbox_ssh_backend_command(tmp_path, monkeypatch):
    from tbsgen import sandbox
    cfg = load_config(None, {"sandbox": {"backend": "ssh", "host": "bh", "remote_dir": "/w/sb",
                                         "docker_config": "/w/dc"}})
    calls = []

    def fake_sh(cmd, timeout, env=None, cwd=None):
        calls.append(cmd)
        return dict(rc=0, out="ok", err="", sec=0.1)
    monkeypatch.setattr(sandbox, "sh", fake_sh)
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    r = sandbox.run(cfg, [(tmp_path / "in", "/app/data", "ro"), (tmp_path / "out", "/app/output", "rw")],
                    "python x.py", 60, "tbsgen-t")
    assert r["rc"] == 0 and r["remote"] == "bh"
    run = [c for c in calls if c[0] == "ssh" and "docker run" in c[-1] and "--network none" in c[-1]]
    assert run and "/w/dc" in run[0][-1] and ":/app/output:rw" in run[0][-1]
    back = [c for c in calls if c[0] == "rsync" and str(c[-2]).startswith("bh:")]
    assert len(back) == 1 and back[0][-1].startswith(str(tmp_path / "out"))     # only the rw mount comes back
    with pytest.raises(ValueError):
        sandbox.run(load_config(None, {"sandbox": {"backend": "ssh"}}), [], "x", 1)


def test_r1_regressions(tmp_path, monkeypatch):
    # "runnable": "false" (a string) or a missing run.log never passes
    bad_runs = {"runs/0.json": json.dumps({"cmd": "python run.py", "rc": 1, "log": "runs/0.log"}), "runs/0.log": "boom"}
    for i, files in enumerate(({"runnability.json": json.dumps({"runnable": "false"}), "HOWTO.md": "x"},
                               {"runnability.json": json.dumps({"runnable": True, "reason": "ok"}), "HOWTO.md": "x"},
                               {"runnability.json": json.dumps({"runnable": True, "reason": "crashes",
                                                                "entry": "python run.py"}), "HOWTO.md": "x", **bad_runs},
                               {"runnability.json": json.dumps({"runnable": True, "reason": "ok",
                                                                "entry": "python run.py"}), "HOWTO.md": "x", **bad_runs})):
        P, it, _ = _agent_item(tmp_path / str(i), monkeypatch, files)
        P.run(max_steps=3, until="S4c")
        st = P.store.item(it.item_id).state
        assert st["status"] == "dropped" and st["reason"] == "code_not_runnable", st
    # a symlink in the agent's deliverable is refused (never dereferenced)
    files = {"_spec/spec.json": json.dumps(SPEC), "_spec/author_code.json": "{}", "task/instruction.md": "x",
             "task/generator/generate.py": "x", "task/solution/solve.py": "x", "task/tests/measure.py": "x",
             "task/baselines/mid.py": "x", "task/baselines/naive.py": "x"}
    P, it, fl = _agent_item(tmp_path / "ln", monkeypatch, files, stage="S5G")
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    P.run(max_steps=1, until="S5G")
    tr = next(Path(it.p("S5G", "jobs")).glob("*/*__x"))
    os.symlink("/etc/hosts", tr / "verifier" / "out" / "task" / "solution" / "data.txt")
    P.run(max_steps=1, until="S5G")
    st = P.store.item(it.item_id).state
    assert st["status"] == "dropped" and st["reason"] == "s5g_unsafe_files"
    # grounded + local sandbox is refused (third-party-derived code never runs locally)
    cfg = _cfg(tmp_path)
    cfg["sandbox"]["backend"] = "docker"
    assert "ssh" in codeagent.env_ok(cfg)


def test_symlinked_deliverable_root_refused(tmp_path):
    tr = tmp_path / "trial"
    (tr / "verifier").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "x.txt").write_text("secret")
    os.symlink(tmp_path / "outside", tr / "verifier" / "out")
    assert codeagent.deliverable({"trial": str(tr)}) is None
    assert codeagent.safe_read(tr / "verifier" / "out" / "x.txt", root=tr) is None
    assert codeagent.safe_copytree(tr / "verifier" / "out", tmp_path / "dst", root=tr) == ["."]


def test_unknown_cost_counts_full_allowance(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, {"_spec/spec.json": "{}"}, stage="S5G")
    write_json(it.p("S5G", "run.json"), {"state": "finished", "job_name": "old", "retry": True,
                                         "launch_attempted": 1.0, "finished_ts": 2.0, "cost_usd": None,
                                         "budget": {"max_usd": 20.0}})
    P.run(max_steps=1, until="S5G")
    st = P.store.item(it.item_id).state
    assert st["status"] == "dropped" and st["reason"] == "s5g_budget_exhausted" and not fl.calls


def test_cumulative_authoring_budget(tmp_path, monkeypatch):
    files = {"_spec/spec.json": "{}"}
    P, it, fl = _agent_item(tmp_path, monkeypatch, files, stage="S5G", authoring_max_usd=20.0,
                            authoring_timeout_s=7200)
    fl.cost = 19.5
    P.run(max_steps=2, until="S5G")                      # job 1 finishes: missing files -> dropped
    st = P.store.item(it.item_id).state
    assert st["reason"] == "s5g_missing_files"
    # a re-author request after $19.5 spent: less than $1 left -> budget exhausted, no second job
    it2 = P.store.item(it.item_id)
    it2.state.update(stage="S5G", status="active", S5G_rerun=True)
    it2.save()
    P.run(max_steps=1, until="S5G")
    st = P.store.item(it.item_id).state
    assert st["status"] == "dropped" and st["reason"] == "s5g_budget_exhausted", st
    assert len(fl.calls) == 1


def test_launch_failed_is_retried_on_resume(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, {"runnability.json": "{}"})
    write_json(it.p("S4c", "run.json"), {"state": "launch_failed", "job_name": "old", "error": "tmux"})   # never started
    P.run(max_steps=1, until="S4c")
    assert P.store.item(it.item_id).state["status"] == "waiting" and len(fl.calls) == 1


def test_inject_script_checks_pin(tmp_path):
    sc = Path(__file__).parents[1] / "scripts" / "inject-author-repo.sh"
    pd = tmp_path / "pins" / "abc"
    (pd / "repo").mkdir(parents=True)
    (pd / "repo" / "a.py").write_text("x")
    (pd / "PIN").write_text("abc\n")
    t = tmp_path / "task"
    (t / "environment" / "work").mkdir(parents=True)
    write_json(t / "environment" / "work" / ".tbsgen-remote-repo", {"repo_dir": str(pd / "repo"), "pin": "abc"})
    root = str(tmp_path.resolve())
    pdr = (pd / "repo").resolve()
    write_json(t / "environment" / "work" / ".tbsgen-remote-repo", {"repo_dir": str(pdr), "pin": "abc"})
    r = subprocess.run(["bash", str(sc), str(t), root], capture_output=True, text=True)
    if "realpath" in r.stderr and "illegal option" in r.stderr or r.returncode == 127:
        pytest.skip("GNU realpath / cp not available")
    assert r.returncode == 0, r.stderr
    assert (t / "environment" / "work" / "repo" / "a.py").exists()
    assert (t / "environment" / "work" / "repo.PIN").read_text().strip() == "abc"
    write_json(t / "environment" / "work" / ".tbsgen-remote-repo", {"repo_dir": str(pdr), "pin": "zzz"})
    assert subprocess.run(["bash", str(sc), str(t), root], capture_output=True).returncode == 3
    # a repo dir reached through a link is refused
    os.symlink(pd, tmp_path / "lnk")
    write_json(t / "environment" / "work" / ".tbsgen-remote-repo", {"repo_dir": str(tmp_path.resolve() / "lnk" / "repo"),
                                                                     "pin": "abc"})
    assert subprocess.run(["bash", str(sc), str(t), root], capture_output=True).returncode == 3


def test_r3_regressions(tmp_path, monkeypatch):
    # a tmux failure (launch_attempted set, not_started) is not charged on resume
    P, it, fl = _agent_item(tmp_path / "a", monkeypatch, {"runnability.json": "{}"})
    write_json(it.p("S4c", "run.json"), {"state": "launch_failed", "job_name": "j0", "launch_attempted": 1.0,
                                         "not_started": True, "error": "tmux rc=1"})
    P.run(max_steps=1, until="S4c")
    assert P.store.item(it.item_id).state["status"] == "waiting" and len(fl.calls) == 1
    assert read_json(it.p("S4c", "spent.json"))["j0"] == {"usd": 0.0, "sec": 0.0}
    # charges are keyed by job: re-processing the same finished record does not double-charge or forget it
    P, it, fl = _agent_item(tmp_path / "b", monkeypatch, {"_spec/spec.json": "{}"}, stage="S5G")
    rec = {"state": "finished", "job_name": "j1", "retry": True, "launch_attempted": 1.0, "finished_ts": 61.0,
           "cost_usd": 5.0, "minutes": 1.0, "counted": True}
    write_json(it.p("S5G", "run.json"), rec)
    write_json(it.p("S5G", "spent.json"), {"j1": {"usd": 5.0, "sec": 60.0}})
    P.run(max_steps=1, until="S5G")
    assert read_json(it.p("S5G", "spent.json")) == {"j1": {"usd": 5.0, "sec": 60.0}}
    assert P.store.item(it.item_id).state["S5G_spent"] == {"usd": 5.0, "sec": 60.0}
    # an empty run log is a valid receipt
    files = {"runnability.json": json.dumps({"runnable": True, "reason": "ok", "entry": "python ex.py"}),
             "HOWTO.md": "x", "runs/0.json": json.dumps({"cmd": "python ex.py", "rc": 0, "log": "runs/0.log"}),
             "runs/0.log": ""}
    P, it, _ = _agent_item(tmp_path / "c", monkeypatch, files)
    P.run(max_steps=3, until="S4c")
    assert it.p("S4c", "run.log").exists() and P.store.item(it.item_id).state.get("reason") != "code_not_runnable"


def test_fetch_script_github_fallback_renders(tmp_path):
    """The rendered fetch script is valid bash and carries the pinned codeload fallback for GitHub."""
    import shlex
    sh = codeagent.FETCH_SH.format(root="/r", dir=shlex.quote("/r/x/it"), url=shlex.quote("https://github.com/a/b.git"),
                                   kind="git", maxmb=500, ref="''", tmo=60)
    f = tmp_path / "f.sh"
    f.write_text(sh)
    assert subprocess.run(["bash", "-n", str(f)]).returncode == 0
    assert "git ls-remote" in sh and "codeload.github.com/$GH/tar.gz/$SHA" in sh and "commits/${REF:-HEAD}" in sh


def test_fetch_codeload_fallback_end_to_end(tmp_path):
    """Fake git (probe fails) + fake curl (API SHA, codeload tarball): pinned to the SHA, links dropped, same-pin
    re-fetch keeps the published tree, a wrong SHA from the API fails closed."""
    import shlex, tarfile
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    src = tmp_path / "src" / "b-abc"
    src.mkdir(parents=True)
    (src / "m.R").write_text("f <- 1")
    os.symlink("/etc/hosts", src / "evil")
    tgz = tmp_path / "a.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(src, arcname="b-abc")
    sha = "a" * 40
    (bin_ / "git").write_text("#!/bin/bash\nexit 128\n")
    (bin_ / "curl").write_text(f"""#!/bin/bash
out=""; url=""
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift 2;; -H|--max-filesize) shift 2;; -*) shift;; *) url=$1; shift;; esac; done
case "$url" in
  *api.github.com*) echo "$(cat {tmp_path}/sha)";;
  *codeload.github.com*/{sha}) cp {tgz} "$out";;
  *) exit 22;;
esac
""")
    for f in ("git", "curl"):
        (bin_ / f).chmod(0o755)
    (tmp_path / "sha").write_text(sha)
    root = tmp_path / "remote"

    def fetch():
        sh = codeagent.FETCH_SH.format(root=shlex.quote(str(root)), dir=shlex.quote(str(root / "r" / "it")),
                                       url=shlex.quote("https://github.com/a/b.git"), kind="git", maxmb=500,
                                       ref="''", tmo=60)
        p = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}"))
        line = [x for x in p.stdout.splitlines() if x.startswith("TBSGEN_RESULT ")]
        return json.loads(line[-1][14:]) if line else {"stdout": p.stdout, "stderr": p.stderr}
    r = fetch()
    assert r.get("ok") and r["pin"] == sha and r["via"] == "codeload", r
    pd = Path(r["remote_dir"])
    assert (pd / "repo" / "m.R").exists() and not (pd / "repo" / "evil").exists()
    (pd / "repo" / "marker").write_text("x")
    assert fetch()["remote_dir"] == str(pd) and (pd / "repo" / "marker").exists()
    (tmp_path / "sha").write_text("not-a-sha")
    r = fetch()
    assert r.get("ok") is False and r.get("why") == "clone_failed"


def test_fetch_spec_quotes_zenodo_file_names():
    spec = codeagent.fetch_spec({"host": "zenodo", "key": "zenodo:5266407",
                                 "files": ["CIA-Oceanix-DinAE_4DVarNN_torch-9eef031 2.zip"]})
    assert spec["url"] == ("https://zenodo.org/records/5266407/files/CIA-Oceanix-DinAE_4DVarNN_torch-9eef031%202.zip"
                           "?download=1")
    spec = codeagent.fetch_spec({"host": "zenodo", "key": "zenodo:7024834", "files": ["SmritiChawla/Precily-v1.0.0.zip"]})
    assert "/files/SmritiChawla/Precily-v1.0.0.zip?" in spec["url"]


def test_zenodo_pick_smallest_archive_that_fits(monkeypatch):
    from tbsgen import codegate
    files = [{"key": "oil-4C.tar.gz", "size": 7_087_000_000}, {"key": "solid-3D.tar.gz", "size": 336_000_000},
             {"key": "code.zip", "size": 2_000_000}, {"key": "README.md", "size": 1000}]
    monkeypatch.setattr(codegate, "http_json", lambda url, cfg, timeout=30: (200, {"files": files}))
    spec = codeagent._zenodo_pick(dict(kind="archive", url="first", zenodo="10023590"), 500, {})
    assert spec["url"].endswith("/files/code.zip?download=1")
    big = [{"key": "Precily-v1.0.0.zip", "size": 3_472_000_000}]
    monkeypatch.setattr(codegate, "http_json", lambda url, cfg, timeout=30: (200, {"files": big}))
    spec = codeagent._zenodo_pick(dict(kind="archive", url="first", zenodo="7024834"), 500, {})
    assert spec["kind"] == "none" and "larger than 500 MB" in spec["why"]
    monkeypatch.setattr(codegate, "http_json", lambda url, cfg, timeout=30: (0, None))
    assert codeagent._zenodo_pick(dict(kind="archive", url="first", zenodo="1"), 500, {})["url"] == "first"


def test_cran_url_falls_back_to_archive(monkeypatch):
    from tbsgen import codegate
    monkeypatch.setattr(codegate, "http_head", lambda url, cfg, timeout=30: 404)
    assert codeagent._cran_url("hdmed", "hdmed_1.0.1.tar.gz", {}) == \
        "https://cran.r-project.org/src/contrib/Archive/hdmed/hdmed_1.0.1.tar.gz"
    monkeypatch.setattr(codegate, "http_head", lambda url, cfg, timeout=30: 200)
    assert codeagent._cran_url("hdmed", "hdmed_1.0.1.tar.gz", {}) == \
        "https://cran.r-project.org/src/contrib/hdmed_1.0.1.tar.gz"


def test_zenodo_pick_follows_a_superseded_record(monkeypatch):
    from tbsgen import codegate
    monkeypatch.setattr(codegate, "http_json", lambda url, cfg, timeout=30: (
        200, {"id": 10615686, "files": [{"key": "scCASE_zenodo.zip", "size": 1_000_000}]}))
    spec = codeagent._zenodo_pick(dict(kind="archive", url="first", zenodo="8382876"), 500, {})
    assert spec["url"] == "https://zenodo.org/records/10615686/files/scCASE_zenodo.zip?download=1"
