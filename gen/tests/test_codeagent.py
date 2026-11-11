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
    assert (tmp_path / "remote" / "t" / "it1" / "repo" / "fastsmoother.py").exists()


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
    """Stands in for tmux + the gu26 launcher: writes a finished Harbor job with the given /app/out files."""

    def __init__(self, files: dict, cost=3.5):
        self.files, self.cost, self.calls = files, cost, []

    def __call__(self, args, **kw):
        self.calls.append(args)
        cfgp = Path(args[-1].split(" run ")[1].split(" >")[0].strip("'"))
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
        return subprocess.CompletedProcess(args, 0, "", "")


def _agent_item(tmp_path, monkeypatch, files, stage="S4c", **code):
    cfg = _cfg(tmp_path, agent_harbor_root=str(tmp_path), agent_launcher="x.sh", **code)
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


def test_s4c_runnable_then_s5g(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, {
        "runnability.json": json.dumps({"runnable": True, "reason": "ok", "language": "Python", "seconds": 3}),
        "HOWTO.md": "pip install .; python -m fastsmoother"})
    P.run(max_steps=1, until="S4c")
    assert P.store.item(it.item_id).state["status"] == "waiting"
    task = it.p("S4c", "task")
    assert (task / "environment" / "work" / ".tbsgen-remote-repo").read_text().strip() == "/r/x/repo"
    assert not (task / "environment" / "work" / "repo").exists()      # never copied off the build host
    P.run(max_steps=2, until="S4c")
    st = P.store.item(it.item_id).state
    assert st["stage"] == "S5G", st
    assert read_json(it.p("S4c_runnability.json"))["cost_usd"] == 3.5
    with P.ledger._db() as db:
        rows = db.execute("SELECT stage, usd, status FROM calls WHERE item=?", (it.item_id,)).fetchall()
    assert rows == [("S4c", 3.5, "ok")]


def test_s4c_not_runnable_gpu(tmp_path, monkeypatch):
    P, it, _ = _agent_item(tmp_path, monkeypatch, {
        "runnability.json": json.dumps({"runnable": False, "reason": "needs_gpu"}), "HOWTO.md": "x"})
    P.run(max_steps=3)
    st = P.store.item(it.item_id).state
    assert st["status"] == "dropped" and st["reason"] == "code_needs_gpu"


SPEC = {"title": "t", "domain": "d", "field": "f", "form": "ANS", "expert_time_hours": 3, "competencies": ["x"],
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
