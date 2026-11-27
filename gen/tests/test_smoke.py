"""`tbsgen smoke`: preflight logic against a fake environment (no network, no docker, no ssh).

Every external effect goes through `smoke.Env`; the fakes below stand in for the launcher, the Docker CLI, ssh and
the model endpoints, so these tests exercise the real checks, the real smoke ledger and the real harvest code.
"""
import contextlib
import json
import sqlite3
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tbsgen import smoke
from tbsgen.budget import Ledger
from tbsgen.config import load_config
from tbsgen.models import ModelError
from tbsgen.state import write_json


def _cfg(tmp_path, **code):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    base = dict(
        run=dict(root=str(run)),
        models=dict(teacher=dict(model="opus-test", url="https://relay.example/v1/messages", key_env=None),
                    prereview=dict(model="ds-test", url=None),
                    prices={"opus-test": {"in": 3.0, "out": 15.0}, "ds-test": {"in": 0.2, "out": 0.8}}),
        budget=dict(daily_usd_total=50.0),
        sandbox=dict(backend="docker"),
        code=dict(grounded=True, host="local", remote_root=str(tmp_path / "remote"), run_tag="t",
                  agent_harbor_root=str(tmp_path / "harbor"), agent_launcher="launch.sh",
                  agent_model="flash-test", runnability_model="flash-test",
                  runnability_env={"ANTHROPIC_BASE_URL": "https://s4c-relay.example",
                                   "ANTHROPIC_AUTH_TOKEN": "tok"}),
        calib=dict(harbor_root=str(tmp_path / "harbor"), teacher_launcher="launch.sh", allow_submit=True),
    )
    for k, v in code.items():
        base[k] = {**(base.get(k) or {}), **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    (tmp_path / "harbor").mkdir(exist_ok=True)
    (tmp_path / "harbor" / "launch.sh").write_text("#!/bin/bash\ntrue\n")
    return load_config(None, base)


class StubTeacher:
    """Books a real ledger call (so price/booking code runs) and answers with the route's own model."""

    real = True

    def __init__(self, ledger, cfg, alias, mode="ok"):
        self.ledger, self.name, self.mode = ledger, cfg["models"][alias]["model"], mode

    def chat(self, system, user, *, stage, item="", max_tokens=None, **kw):
        assert stage == smoke.STAGE and max_tokens <= 64, (stage, max_tokens)
        if self.mode == "auth":
            raise ModelError("auth", "http 401")
        got = "different-model" if self.mode == "mismatch" else self.name
        cid = self.ledger.reserve(self.name, stage, item, True, "sha", send=True)
        usage = {"input_tokens": 40, "output_tokens": 2}
        usd = self.ledger.finish(cid, usage, resp_model=got)
        return SimpleNamespace(text="pong", model=got, usage=usage, usd=usd, latency_s=0.01)


class FakeEnv(smoke.RealEnv):
    """Stands in for ssh / docker / the launcher / HTTP.  Records every argv it was asked to run."""

    def __init__(self, cfg, *, docker_ok=True, images=("tbsgen-task:abc123",), build_rc=0, build_err="",
                 egress_out="200 0.120000", egress_rc=0, df_out=None, post_fn=None, launch_files=None,
                 launch_rc=0, shell_rc=0, teacher_mode="ok"):
        """launch_files: {rel: body} for every stage job, or a callable(job_conf) -> {rel: body}."""
        super().__init__(cfg)
        self.docker_ok, self.images, self.build_rc, self.build_err = docker_ok, list(images), build_rc, build_err
        self.egress_out, self.egress_rc = egress_out, egress_rc
        self.df_out = df_out if df_out is not None else "/dev/sda1 104857600 1048576 103809024 2% /w\n"
        self.post_fn, self.posted = post_fn, []
        self.launch_files, self.launch_rc = launch_files, launch_rc
        self.shell_rc, self.teacher_mode = shell_rc, teacher_mode
        self.calls = []

    # ---- effects
    def ssh(self, cmd, timeout=60):
        self.calls.append(("ssh", cmd))
        if "df -Pk" in cmd:
            return dict(rc=0, out=self.df_out, err="")
        return dict(rc=self.egress_rc, out=self.egress_out, err="could not connect" if self.egress_rc else "")

    def docker(self, argv, timeout=120):
        self.calls.append(("docker", " ".join(argv)))
        if argv[1] == "version":
            return dict(rc=0, out="27.0.0\n", err="") if self.docker_ok else dict(rc=1, out="", err="daemon down")
        if argv[1] == "image":
            return dict(rc=0, out="\n".join(self.images) + "\n", err="")
        return dict(rc=self.build_rc, out="", err=self.build_err)

    def shell(self, argv, timeout=60):
        self.calls.append(("shell", " ".join(argv)))
        return dict(rc=self.shell_rc, out="", err="syntax error" if self.shell_rc else "")

    def context(self, cfg, src):  # type: ignore[override]
        return contextlib.nullcontext(str(src))

    def client(self, ledger, cfg, alias):  # type: ignore[override]
        return StubTeacher(ledger, cfg, alias, self.teacher_mode)

    def post(self, url, headers, body, timeout):
        self.posted.append(dict(url=url, headers=headers, body=body))
        if self.post_fn is not None:
            return self.post_fn(url, headers, body, timeout)
        return dict(status=200, body={"model": body["model"], "content": [{"type": "text", "text": "pong"}],
                                      "usage": {"input_tokens": 30, "output_tokens": 2}})

    def launch(self, root, launcher, job_json, log, timeout):
        self.calls.append(("launch", f"{root} {launcher} {job_json}"))
        conf = json.loads(Path(job_json).read_text())
        assert Path(job_json).is_relative_to(Path(self.cfg["run"]["root"]) / "smoke"), job_json
        files = self.launch_files(conf) if callable(self.launch_files) else self.launch_files
        if files is None:
            return dict(rc=self.launch_rc, out="", err="launcher failed")
        jd = Path(conf["jobs_dir"]) / conf["job_name"]
        trial = jd / "trial-1"
        (trial / "verifier" / "out").mkdir(parents=True)
        for rel, body in files.items():
            (trial / "verifier" / "out" / rel).parent.mkdir(parents=True, exist_ok=True)
            (trial / "verifier" / "out" / rel).write_text(body)
        (jd / "config.json").write_text(json.dumps(conf))
        (trial / "result.json").write_text(json.dumps({
            "agent_result": {"cost_usd": 0.02, "n_input_tokens": 10, "n_output_tokens": 5},
            "verifier_result": {"rewards": {"reward": 1}},
            "agent_execution": {"started_at": "2026-11-27T00:00:00Z", "finished_at": "2026-11-27T00:01:00Z"}}))
        (jd / "result.json").write_text(json.dumps({"finished_at": "2026-11-27T00:02:00Z"}))
        return dict(rc=0, out="", err="")


# ----------------------------------------------------------------------------------------------- config / breakers
def test_config_reports_missing_route_price_and_cap(tmp_path):
    cfg = _cfg(tmp_path, models=dict(teacher=dict(url=None, model="unpriced")), budget=dict(daily_usd_total=None))
    c = smoke.check_config(cfg)
    assert c["ok"] is False
    joined = " ".join(c["detail"]["problems"])
    assert "models.teacher.url" in joined and "models.prices" in joined and "daily_usd_total" in joined


def test_config_ok_on_a_configured_run(tmp_path):
    assert smoke.check_config(_agent_cfg(tmp_path))["ok"] is True


def test_config_flags_an_agent_route_that_could_never_run(tmp_path):
    cfg = _cfg(tmp_path, sandbox=dict(backend="docker"))          # grounded, but S4c/S5G need the ssh build host
    c = smoke.check_config(cfg)
    assert c["ok"] is False and any("S4c/S5G" in p for p in c["detail"]["problems"])


def test_open_breaker_in_the_copied_ledger_is_blocking_and_read_only(tmp_path):
    cfg = _cfg(tmp_path)
    led = Ledger(tmp_path / "run" / "ledger.sqlite", cfg)
    led.open_breaker("opus-test", "auth x1")
    before = (tmp_path / "run" / "ledger.sqlite").stat().st_mtime_ns
    c = smoke.check_breakers(cfg)
    assert c["ok"] is False and c["detail"]["open"][0]["model"] == "opus-test"
    assert "tbsgen breaker" in c["detail"]["hint"]
    assert (tmp_path / "run" / "ledger.sqlite").stat().st_mtime_ns == before, "the check must not write"
    with sqlite3.connect(tmp_path / "run" / "ledger.sqlite") as db:
        assert db.execute("SELECT open FROM breaker WHERE model='opus-test'").fetchone()[0] == 1


def test_no_ledger_is_not_a_failure(tmp_path):
    c = smoke.check_breakers(_cfg(tmp_path))
    assert c["ok"] is True and c["skipped"] is True


# ----------------------------------------------------------------------------------------------- model routes
def _models(cfg, env, cap=0.05, dry_run=False):
    path = cfg["run"]["root"] + "/smoke/ledger.sqlite"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    led = Ledger(path, smoke.ledger_cfg(cfg, cap))
    return smoke.check_models(cfg, env, led, cap, dry_run), led


def test_teacher_route_is_a_real_tiny_call_booked_under_smoke(tmp_path):
    cfg = _cfg(tmp_path)
    c, led = _models(cfg, FakeEnv(cfg))
    assert c["ok"] is True, c["detail"]
    [r] = c["detail"]["routes"]
    assert r["route"] == "teacher" and r["model_returned"] == "opus-test" and r["usd"] > 0
    with sqlite3.connect(led.path) as db:
        rows = db.execute("SELECT model, stage, item, usd, status FROM calls").fetchall()
    assert rows == [("opus-test", "SMOKE", "smoke:teacher", rows[0][3], "ok")] and rows[0][3] > 0


def test_teacher_route_mismatch_and_auth_failures_are_reported(tmp_path):
    cfg = _cfg(tmp_path)
    c, _ = _models(cfg, FakeEnv(cfg, teacher_mode="mismatch"))
    assert c["ok"] is False and "expected 'opus-test'" in c["detail"]["routes"][0]["error"]
    c, _ = _models(cfg, FakeEnv(cfg, teacher_mode="auth"))
    assert c["ok"] is False and "auth" in c["detail"]["routes"][0]["error"]


def test_agent_route_uses_the_stage_env_endpoint_and_headers(tmp_path):
    cfg = _cfg(tmp_path, sandbox=dict(backend="ssh", host="bh", remote_dir="/w"))
    env = FakeEnv(cfg)
    c, led = _models(cfg, env)
    assert c["ok"] is True, c["detail"]
    [s4c] = [r for r in c["detail"]["routes"] if r["route"] == "s4c-agent"]
    assert s4c["endpoint"] == "https://s4c-relay.example/v1/messages"
    assert s4c["endpoint_source"] == "code.runnability_env" and s4c["model"] == "flash-test"
    req = [p for p in env.posted if p["url"].startswith("https://s4c-relay")][0]
    assert req["headers"]["x-api-key"] == "tok" and req["body"]["max_tokens"] <= 64
    with sqlite3.connect(led.path) as db:
        assert db.execute("SELECT COUNT(*) FROM calls WHERE model='flash-test' AND stage='SMOKE'").fetchone()[0] == 1


def test_agent_route_model_mismatch_is_a_failure(tmp_path):
    cfg = _cfg(tmp_path, sandbox=dict(backend="ssh", host="bh", remote_dir="/w"))
    env = FakeEnv(cfg, post_fn=lambda u, h, b, t: dict(status=200, body={"model": "opus-test"}))
    c, _ = _models(cfg, env)
    [s4c] = [r for r in c["detail"]["routes"] if r["route"] == "s4c-agent"]
    assert s4c["ok"] is False and "expected 'flash-test'" in s4c["error"]


def test_agent_route_host_outside_the_allowlist_fails(tmp_path):
    cfg = _cfg(tmp_path, sandbox=dict(backend="ssh", host="bh", remote_dir="/w"),
               code=dict(agent_allowed_hosts=["relay.example"]))
    c, _ = _models(cfg, FakeEnv(cfg))
    [s4c] = [r for r in c["detail"]["routes"] if r["route"] == "s4c-agent"]
    assert s4c["ok"] is False and "agent_allowed_hosts" in s4c["error"]


def test_route_http_error_is_booked_as_an_error_call(tmp_path):
    cfg = _cfg(tmp_path, sandbox=dict(backend="ssh", host="bh", remote_dir="/w"))
    env = FakeEnv(cfg, post_fn=lambda u, h, b, t: dict(status=401, body={"error": "invalid token"}))
    c, led = _models(cfg, env)
    [s4c] = [r for r in c["detail"]["routes"] if r["route"] == "s4c-agent"]
    assert s4c["ok"] is False and s4c["status"] == 401
    with sqlite3.connect(led.path) as db:
        assert db.execute("SELECT status, error FROM calls WHERE model='flash-test'").fetchone()[0] == "error"


def test_cap_stops_further_calls(tmp_path):
    cfg = _cfg(tmp_path, models=dict(prereview=dict(url="https://ds.example/v1/messages")))
    assert [r["name"] for r in smoke.routes(cfg)] == ["teacher", "prereview"]
    c, _ = _models(cfg, FakeEnv(cfg), cap=0.0002)      # one priced call already eats 0.00015 of the cap
    assert c["ok"] is False and c["detail"]["stopped"] and c["detail"]["usd"] <= 0.0002 + 1e-9


def test_smoke_ledger_caps_are_bounded(tmp_path):
    cfg = _cfg(tmp_path)
    lc = smoke.ledger_cfg(cfg, 0.05)
    assert lc["budget"]["daily_usd_total"] == 0.05
    assert lc["budget"]["daily_usd_per_stage"] == {"SMOKE": 0.05}
    assert lc["budget"]["est_usd_per_call"] <= 0.05
    assert cfg["budget"]["daily_usd_total"] == 50.0, "the production config is not modified"


# ----------------------------------------------------------------------------------------------- docker / buildx
def test_docker_daemon_unreachable_fails(tmp_path):
    cfg = _cfg(tmp_path)
    assert smoke.check_docker(cfg, FakeEnv(cfg))["ok"] is True
    assert smoke.check_docker(cfg, FakeEnv(cfg, docker_ok=False))["ok"] is False


def test_buildx_probe_builds_from_a_local_tbsgen_task_image(tmp_path):
    cfg = _cfg(tmp_path, taskimage=dict(builder="tbsgen-builder"))
    env = FakeEnv(cfg)
    c = smoke.check_buildx(cfg, env)
    assert c["ok"] is True and c["detail"]["base_image"] == "tbsgen-task:abc123"
    argv = [a for k, a in env.calls if k == "docker" and " buildx build " in a][0]
    assert "--builder tbsgen-builder" in argv and "type=cacheonly" in argv
    assert (tmp_path / "run" / "smoke").is_dir()


def test_buildx_builder_that_cannot_see_the_image_store_fails_with_a_hint(tmp_path):
    cfg = _cfg(tmp_path, taskimage=dict(builder="container-driver"))
    env = FakeEnv(cfg, build_rc=1, build_err='ERROR: failed to solve: tbsgen-task:abc123: '
                                              'pull access denied for tbsgen-task, repository does not exist')
    c = smoke.check_buildx(cfg, env)
    assert c["ok"] is False and "default docker driver" in c["detail"]["hint"]
    assert "pull access denied" in c["detail"]["error"]


def test_buildx_without_any_local_base_image_fails(tmp_path):
    cfg = _cfg(tmp_path)
    c = smoke.check_buildx(cfg, FakeEnv(cfg, images=()))
    assert c["ok"] is False and "no local tbsgen-task" in c["detail"]["error"]
    # an explicit --base-image is used instead of the listing
    c = smoke.check_buildx(cfg, FakeEnv(cfg, images=()), base_image="tbsgen-task:pinned")
    assert c["ok"] is True and c["detail"]["base_image"] == "tbsgen-task:pinned"


# ----------------------------------------------------------------------------------------------- egress / launchers / disk
def test_egress_probe_passes_and_times_out(tmp_path):
    cfg = _cfg(tmp_path, code=dict(install_hosts=["pypi.org"]))
    env = FakeEnv(cfg)
    c = smoke.check_egress(cfg, env, timeout_s=5)
    assert c["ok"] is True and c["detail"]["probes"][0]["status"] == 200
    cmds = [a for k, a in env.calls if k == "ssh"]
    assert cmds and all("--max-time 5" in x and "curl" in x for x in cmds)
    assert any("https://pypi.org" in x for x in cmds), "every egress URL is probed"
    bad = smoke.check_egress(cfg, FakeEnv(cfg, egress_rc=28, egress_out=""), timeout_s=5)
    assert bad["ok"] is False and "could not connect" in json.dumps(bad["detail"])


def test_launchers_missing_and_syntax_errors(tmp_path):
    cfg = _cfg(tmp_path)
    good = tmp_path / "harbor" / "launch.sh"
    good.write_text("#!/bin/bash\ntrue\n")
    assert smoke.check_launchers(cfg, smoke.RealEnv(cfg))["ok"] is True
    good.write_text("#!/bin/bash\nif then\n")
    c = smoke.check_launchers(cfg, smoke.RealEnv(cfg))
    assert c["ok"] is False and "syntax" in json.dumps(c["detail"]).lower()
    good.unlink()
    c = smoke.check_launchers(cfg, smoke.RealEnv(cfg))
    assert c["ok"] is False and "missing" in json.dumps(c["detail"])


def test_disk_threshold(tmp_path):
    cfg = _cfg(tmp_path)
    c = smoke.check_disk(cfg, FakeEnv(cfg, df_out="/dev/sda1 104857600 104806400 51200 100% /w\n"), min_free_gb=10)
    assert c["ok"] is False and c["detail"]["free_gb"] < 10
    c = smoke.check_disk(cfg, FakeEnv(cfg, df_out="/dev/sda1 104857600 1048576 103809024 2% /w\n"), min_free_gb=10)
    assert c["ok"] is True and c["detail"]["free_gb"] > 90


# ----------------------------------------------------------------------------------------------- stage jobs
def _agent_cfg(tmp_path, **kw):
    return _cfg(tmp_path, sandbox=dict(backend="ssh", host="bh", remote_dir="/w"), **kw)


def test_stage_job_harvests_the_required_outputs(tmp_path):
    cfg = _agent_cfg(tmp_path)
    files = {f: "placeholder\n" for f in ["_spec/spec.json", "_spec/author_code.json", "task/instruction.md",
                                          "task/generator/generate.py", "task/solution/solve.py",
                                          "task/tests/measure.py", "task/baselines/mid.py", "task/baselines/naive.py"]}
    env = FakeEnv(cfg, launch_files=files)
    j = smoke.stage_job(cfg, env, "S5G", 0.5, 60)
    assert j["ok"] is True, j
    assert j["complete"] is True and set(files) <= set(j["harvested"])
    assert j["model"] == "flash-test"
    task = tomllib.loads((Path(j["job_dir"]).parent.parent / "task" / "task.toml").read_text())
    assert task["task"]["name"] == "tbsgen-S5G/smoke-s5g"


def test_stage_job_without_the_required_outputs_fails(tmp_path):
    cfg = _agent_cfg(tmp_path)
    j = smoke.stage_job(cfg, FakeEnv(cfg, launch_files={"HOWTO.md": "x"}), "S5G", 0.5, 60)
    assert j["ok"] is False and j["complete"] is False and "not harvested" in j["error"]


def test_stage_job_skips_when_the_route_is_not_configured(tmp_path):
    cfg = _cfg(tmp_path, sandbox=dict(backend="docker"))
    j = smoke.stage_job(cfg, FakeEnv(cfg, launch_files={}), "S4c", 0.5, 60)
    assert j["ok"] is None and j["skipped"] is True and "sandbox.backend" in j["error"]


# ----------------------------------------------------------------------------------------------- the command
S4C_FILES = {"runnability.json": '{"runnable": false, "reason": "other"}\n', "HOWTO.md": "# how\n"}
S5G_FILES = {f: "placeholder\n" for f in ["_spec/spec.json", "_spec/author_code.json", "task/instruction.md",
                                           "task/generator/generate.py", "task/solution/solve.py",
                                           "task/tests/measure.py", "task/baselines/mid.py",
                                           "task/baselines/naive.py"]}


def _env_all_ok(cfg):
    return FakeEnv(cfg, launch_files=lambda conf: S4C_FILES if "s4c" in conf["job_name"] else S5G_FILES)


def test_run_smoke_end_to_end_is_green_and_keeps_production_untouched(tmp_path):
    cfg = _agent_cfg(tmp_path)
    rep = smoke.run_smoke(cfg, env=_env_all_ok(cfg), stage_job_flag=True)
    assert rep["ok"] is True, json.dumps(rep, indent=1)
    assert {c["name"] for c in rep["checks"]} == set(smoke.CHECK_NAMES)
    assert rep["ledger_calls"] and all(r["stage"] == "SMOKE" for r in rep["ledger_calls"])
    assert Path(rep["ledger"]) == tmp_path / "run" / "smoke" / "ledger.sqlite"
    assert not (tmp_path / "run" / "ledger.sqlite").exists(), "the production ledger is not written"
    assert not (tmp_path / "run" / "items").exists()
    saved = json.loads((tmp_path / "run" / "smoke" / "last-report.json").read_text())
    assert saved["ok"] is True and saved["stage"] == "SMOKE"
    assert smoke.format_report(rep).startswith("tbsgen smoke: OK")


def test_run_smoke_green_report_is_written_even_when_a_stage_job_is_skipped(tmp_path):
    cfg = _agent_cfg(tmp_path, calib=dict(allow_submit=False))  # launcher configured but submission is manual
    rep = smoke.run_smoke(cfg, env=_env_all_ok(cfg), stage_job_flag=True)
    jobs = [j for j in rep["checks"] if j["name"] == "stage_jobs"][0]["detail"]["jobs"]
    assert {j["kind"] for j in jobs} == {"S4c", "S5G"}
    assert all(j["skipped"] and "allow_submit" in j["error"] for j in jobs)
    assert rep["ok"] is True


def test_run_smoke_fails_on_a_broken_environment(tmp_path):
    cfg = _agent_cfg(tmp_path)
    env = FakeEnv(cfg, docker_ok=False, images=(), egress_rc=28, teacher_mode="auth", shell_rc=1)
    rep = smoke.run_smoke(cfg, env=env, skip="disk")
    assert rep["ok"] is False
    failed = {c["name"] for c in rep["checks"] if c["ok"] is False}
    assert {"models", "docker", "buildx", "egress", "launchers"} <= failed
    assert [c for c in rep["checks"] if c["name"] == "disk"][0]["skipped"] is True


class StrictEnv:
    """Any external action fails the test."""

    def __getattr__(self, name):
        def boom(*a, **k):
            raise AssertionError(f"dry run performed {name}")
        return boom


def test_dry_run_performs_no_external_action(tmp_path):
    cfg = _agent_cfg(tmp_path)
    rep = smoke.run_smoke(cfg, env=StrictEnv(), stage_job_flag=True, dry_run=True)
    assert rep["dry_run"] is True
    assert [c["name"] for c in rep["checks"]][0] == "config"
    assert not (tmp_path / "run" / "smoke" / "ledger.sqlite").exists()


def test_unknown_skip_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown check"):
        smoke.run_smoke(_cfg(tmp_path), env=FakeEnv(_cfg(tmp_path)), skip="docker,nope", dry_run=True)


def test_cli_smoke_subcommand_is_registered():
    from tbsgen import cli
    seen = {}

    def fake(a):
        seen["a"] = a
        return 0
    orig = cli._smoke
    cli._smoke = fake
    try:
        assert cli.main(["smoke", "--run", "/tmp/x", "--dry-run", "--skip", "docker", "--budget-usd", "0.01"]) == 0
    finally:
        cli._smoke = orig
    assert seen["a"].dry_run is True and seen["a"].skip == "docker" and seen["a"].budget_usd == 0.01
