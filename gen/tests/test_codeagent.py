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
