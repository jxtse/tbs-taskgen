"""Pre-release checks (prerelease.py) and the package's network policy (package.py): unit tests.  The Docker /
Harbor end-to-end checks are in test_integration.py."""
import json
import tomllib
from pathlib import Path

import pytest

from tbsgen import calib, package, prerelease
from tbsgen.config import load_config

CAND = dict(doi="10.1/x", title="Spike-robust decay fitting",
            code_links=["https://github.com/someone/DecayFit", "https://github.com/someone/docs"],
            data_links=["https://zenodo.org/record/1"])
SPEC = dict(title="t", method_names=["SpikeRobustFit"])


def mk_task(tmp_path, solve="import numpy as np\n", req=None, dockerfile=None, extra=None):
    t = tmp_path / "task"
    (t / "solution").mkdir(parents=True)
    (t / "baselines").mkdir()
    (t / "environment" / "data").mkdir(parents=True)
    (t / "solution" / "solve.py").write_text(solve)
    (t / "baselines" / "mid.py").write_text("import pandas\n")
    if req is not None:
        (t / "environment" / "requirements.txt").write_text(req)
    if dockerfile is not None:
        (t / "environment" / "Dockerfile").write_text(dockerfile)
    for rel, body in (extra or {}).items():
        (t / rel).parent.mkdir(parents=True, exist_ok=True)
        (t / rel).write_text(body)
    return t


# ------------------------------------------------------------------ ready-made implementation check
def test_clean_task_passes_and_records_companion_code(tmp_path):
    r = prerelease.readymade_check(mk_task(tmp_path), CAND, SPEC)
    assert r["ok"] and r["findings"] == []
    assert r["companion_code"] == ["https://github.com/someone/DecayFit", "https://github.com/someone/docs"]
    assert "decayfit" in r["signals"] and "spikerobustfit" in r["signals"]


@pytest.mark.parametrize("req", ["decayfit==1.2\n", "DecayFit\n", "decay-fit>=1\n", "git+https://github.com/someone/DecayFit\n",
                                 "pydecayfit\n", "decayfit~=1.2\n", "decayfit[extra]>=1 ; python_version>'3'\n",
                                 "-e git+https://github.com/lab/DecayFit.git#egg=decayfit\n",
                                 "-e ./vendor/DecayFit\n", "mypkg @ https://github.com/lab/DecayFit/archive/v1.zip\n"])
def test_dependency_on_the_papers_package_is_blocking(tmp_path, req):
    r = prerelease.readymade_check(mk_task(tmp_path, req=req), CAND, SPEC)
    assert not r["ok"] and r["findings"][0]["kind"] == "dependency" and "DecayFit" in r["findings"][0]["matches"]


@pytest.mark.parametrize("df", [
    "FROM python:3.11-slim\nRUN pip install --no-cache-dir numpy==2.1.3 \\\n    decayfit==0.3\n",   # continuation
    "FROM python:3.11-slim\nRUN apt-get update && python -m pip install -q decayfit && rm -rf /root/.cache\n",
    "FROM python:3.11-slim\nRUN uv pip install --system git+https://github.com/someone/DecayFit.git@v1\n",
    'FROM python:3.11-slim\nRUN pip install "decayfit==1.2"\n',
    "FROM python:3.11-slim\nRUN pip install -i https://mirror.example/simple decayfit~=1.2\n",
    "FROM python:3.11-slim\nRUN pip install -e git+https://github.com/lab/DecayFit.git#egg=decayfit\n",
])
def test_pip_line_in_a_dockerfile_is_blocking(tmp_path, df):
    """(see the parameters; quoted and ~= specifiers are covered below)"""
    r = prerelease.readymade_check(mk_task(tmp_path, dockerfile=df), CAND, SPEC)
    assert [f["kind"] for f in r["findings"]] == ["dependency"], r


def test_reference_or_baseline_import_is_blocking(tmp_path):
    t = mk_task(tmp_path, solve="import numpy as np\nfrom decayfit.core import fit\n")
    (t / "baselines" / "naive.py").write_text("import importlib\nm = importlib.import_module('spikerobustfit')\n")
    r = prerelease.readymade_check(t, CAND, SPEC)
    assert {(f["kind"], f["file"]) for f in r["findings"]} == {("import", "solution/solve.py"),
                                                                ("import", "baselines/naive.py")}


def test_vendored_copy_shipped_to_the_agent_is_blocking(tmp_path):
    t = mk_task(tmp_path, extra={"environment/data/decayfit/core.py": "def fit(): pass\n"})
    r = prerelease.readymade_check(t, CAND, SPEC)
    assert r["findings"][0]["kind"] == "public_code"


@pytest.mark.parametrize("name", ["DecayFit.tar.gz", "decayfit-1.2.0.tar.gz", "DecayFit-main.zip",
                                  "decayfit-1.2-py3-none-any.whl", "decayfit_v2/core.py"])
def test_shipped_archives_and_versioned_copies_are_blocking(tmp_path, name):
    t = mk_task(tmp_path, extra={f"environment/data/{name}": "x"})
    r = prerelease.readymade_check(t, CAND, SPEC)
    assert not r["ok"] and r["findings"][0]["kind"] == "public_code", name


@pytest.mark.parametrize("name", ["decayfit.so", "decayfit.cpython-311-x86_64-linux-gnu.so", "libdecayfit.so",
                                  "__pycache__/decayfit.cpython-311.pyc", "DecayFit.pyd"])
def test_shipped_compiled_modules_are_blocking(tmp_path, name):
    t = mk_task(tmp_path, extra={f"environment/data/{name}": "x"})
    r = prerelease.readymade_check(t, CAND, SPEC)
    assert not r["ok"] and r["findings"][0]["kind"] == "public_code", name


@pytest.mark.parametrize("req", [
    "https://files.example.org/dist/decay-fit-1.2.0.tar.gz\n",
    "https://files.example.org/w/decayfit-1.2-py3-none-any.whl#sha256=" + "a" * 64 + "\n",
    "decayfit @ https://files.example.org/w/decayfit-1.2-py3-none-any.whl#sha256=" + "b" * 64 + "\n",
    "./wheels/decayfit-1.2-py3-none-any.whl\n",
    "wheels/decayfit-1.2-py3-none-any.whl\n",
    "-e vendor/DecayFit\n",
    "https://files.example.org/dist/DecayFit_1.2.tar.gz\n",
    "py_decay_fit==1.2\n",
    "https://github.com/lab/wheels/releases/download/v1/DecayFit-1.2.tar.gz\n",
    "git+https://git.example.org/lab/DecayFit.git@v1\n",
])
def test_direct_archive_urls_resolve_to_the_distribution_name(tmp_path, req):
    r = prerelease.readymade_check(mk_task(tmp_path, req=req), CAND, SPEC)
    assert not r["ok"] and r["findings"][0]["kind"] == "dependency", (req, r)


@pytest.mark.parametrize("link", ["https://gitlab.com/group/subgroup/DecayFit",
                                  "https://gitlab.com/group/sub/DecayFit/-/tree/main",
                                  "https://cran.r-project.org/web/packages/DecayFit/index.html",
                                  "https://cran.r-project.org/package=DecayFit",
                                  "https://cran.r-project.org/?package=DecayFit",
                                  "https://CRAN.R-project.org/package=DecayFit"])
def test_gitlab_subgroups_and_cran_links_give_the_package_name(tmp_path, link):
    cand = dict(CAND, code_links=[link])
    assert [r["name"] for r in prerelease.repo_names([link])] == ["DecayFit"]
    r = prerelease.readymade_check(mk_task(tmp_path, req="decayfit==1\n"), cand, dict(title="t"))
    assert not r["ok"] and r["findings"][0]["kind"] == "dependency"


def test_data_files_named_like_the_method_are_not_code(tmp_path):
    t = mk_task(tmp_path, extra={"environment/data/decayfit_traces.csv": "t,y\n"})
    assert prerelease.readymade_check(t, CAND, SPEC)["ok"]


def test_generated_public_output_is_checked_too(tmp_path):
    """At S7 the public files are the generator's output (runs/s7-N/detA/environment), not task/environment."""
    t = mk_task(tmp_path)
    pub = tmp_path / "runs" / "s7-1" / "detA" / "environment"
    (pub / "data" / "spikerobustfit").mkdir(parents=True)
    (pub / "data" / "spikerobustfit" / "__init__.py").write_text("")
    assert prerelease.readymade_check(t, CAND, SPEC)["ok"]
    r = prerelease.readymade_check(t, CAND, SPEC, public_env=pub)
    assert not r["ok"] and r["findings"][0]["kind"] == "public_code"


def test_scheme_less_links_from_retrieval_count(tmp_path):
    cand = dict(CAND, code_links=["github.com/lab/DecayFit"], data_links=[])
    r = prerelease.readymade_check(mk_task(tmp_path, solve="import decayfit\n"), cand, {})
    assert not r["ok"] and r["companion_code"] == ["https://github.com/lab/DecayFit"]


def test_candidate_software_names_are_signals(tmp_path):
    r = prerelease.readymade_check(mk_task(tmp_path, req="pyqsm\n"), dict(doi="10.1/x", libs=["QSM"]), {})
    assert not r["ok"]


def test_s5_prompt_asks_for_method_names():
    from tbsgen import prompts
    assert '"method_names"' in prompts.S5_SPEC


def test_pyproject_dependencies_are_checked(tmp_path):
    t = mk_task(tmp_path, extra={"environment/pyproject.toml":
                                 '[project]\nname = "task"\ndependencies = ["numpy>=2", "decayfit==1.2"]\n'})
    r = prerelease.readymade_check(t, CAND, SPEC)
    assert [f["name"] for f in r["findings"]] == ["decayfit"]


def test_unrelated_dependencies_and_generic_repo_names_pass(tmp_path):
    cand = dict(CAND, code_links=["https://github.com/someone/code", "https://github.com/lab/python"])
    t = mk_task(tmp_path, req="numpy==2.1.3\nscipy\nlmfit\n", solve="import lmfit\nimport os\n")
    r = prerelease.readymade_check(t, cand, dict(SPEC, method_names=[]))
    assert r["ok"], r["findings"]


def test_no_signals_means_nothing_to_match(tmp_path):
    t = mk_task(tmp_path, req="decayfit\n", solve="import decayfit\n")
    assert prerelease.readymade_check(t, dict(doi="10.1/x"), {})["ok"]


def test_readymade_is_blocking_by_default():
    assert "readymade" in load_config()["gates"]["blocking"]


# ------------------------------------------------------------------ network policy in the package
def _pkg_toml(tmp_path, **net):
    cfg = load_config(overrides={"package": {"network": net}} if net else None)
    pol = package.network_policy(cfg)
    txt = package.TASK_TOML.format(task_id="t", desc='"d"', keywords="[]", domain='"x"', field='"x"', form='"ANS"',
                                   expert_h=1.0, verifier_sec=60.0, agent_sec=60.0, **pol)
    return tomllib.loads(txt)


def test_default_network_is_agent_allowlist_and_offline_verifier(tmp_path):
    t = _pkg_toml(tmp_path)
    assert t["environment"]["network_mode"] == "public"           # agent setup (installs the agent CLI)
    assert t["agent"]["network_mode"] == "allowlist" and t["agent"]["allowed_hosts"] == []
    assert t["verifier"]["network_mode"] == "no-network"


def test_network_modes_are_configurable(tmp_path):
    t = _pkg_toml(tmp_path, mode="agent-allowlist", agent_hosts=["api.example.com"])
    assert t["agent"]["allowed_hosts"] == ["api.example.com"]
    t = _pkg_toml(tmp_path, mode="no-network")
    assert t["environment"]["network_mode"] == "no-network" and "network_mode" not in t["agent"]
    with pytest.raises(ValueError):
        _pkg_toml(tmp_path, mode="open")


def test_harbor_accepts_the_task_toml_if_installed(tmp_path):
    """Parse with Harbor's own TaskConfig when harbor is importable (it is not a dependency of tbsgen)."""
    hc = pytest.importorskip("harbor.models.task.config")
    t = _pkg_toml(tmp_path)
    cfg = hc.TaskConfig.model_validate(t)
    assert cfg.agent.network_mode.value == "allowlist"


def test_disable_index_strips_package_indexes(tmp_path):
    cfg = load_config(overrides={"package": {"network": {"disable_index": True}}})
    df = package.dockerfile(cfg)
    assert "PIP_NO_INDEX=1" in df and "UV_OFFLINE=1" in df and df.index("pip install") < df.index("PIP_NO_INDEX")
    assert "PIP_NO_INDEX" not in package.dockerfile(load_config())


def test_calib_job_config_allows_the_model_host_in_the_agent_phase(tmp_path):
    cfg = load_config(overrides={"calib": {"agent_allowed_hosts": ["models.example.com"],
                                           "jobs_dir": str(tmp_path / "jobs")}})
    conf = calib.job_config(cfg, tmp_path, "glm", "j")
    assert conf["agents"][0]["extra_allowed_hosts"] == ["models.example.com"]


def _pkg_task(item, **net):
    """items/<id>/pkg/task.toml as package.py writes it."""
    (item / "pkg").mkdir(parents=True, exist_ok=True)
    cfg = load_config(overrides={"package": {"network": net}} if net else None)
    (item / "pkg" / "task.toml").write_text(package.TASK_TOML.format(
        task_id="t", desc='"d"', keywords="[]", domain='"x"', field='"x"', form='"ANS"', expert_h=1.0,
        verifier_sec=60.0, agent_sec=60.0, **package.network_policy(cfg)))


def test_calib_refuses_to_launch_an_allowlisted_task_without_a_model_host(tmp_path):
    item = tmp_path / "items" / "i1"
    _pkg_task(item)
    cfg = load_config(overrides={"calib": {"allow_submit": True, "harbor_root": str(tmp_path), "student_launcher": "x.sh",
                                           "jobs_dir": str(tmp_path / "jobs")}})
    rec = calib.prepare(cfg, item, "glm", 1)
    assert "agent_allowed_hosts" in rec["blocked"]
    cfg["calib"]["allow_submit"] = False
    rec = calib.prepare(cfg, item, "glm", 1)
    assert not rec.get("blocked") and "agent_allowed_hosts" in rec["warning"]
    cfg["calib"]["agent_allowed_hosts"] = ["models.example.com"]
    cfg["calib"]["allow_submit"] = True
    assert not calib.prepare(cfg, item, "glm", 1).get("blocked")


def test_calib_reads_the_network_of_the_task_it_runs_not_the_current_config(tmp_path):
    """A task packaged public stays launchable after the default changes; a task packaged offline is refused even
    if the config now says allowlist."""
    cfg = load_config(overrides={"calib": {"allow_submit": True, "harbor_root": str(tmp_path), "student_launcher": "x.sh",
                                           "jobs_dir": str(tmp_path / "jobs")}})
    pub, off = tmp_path / "items" / "pub", tmp_path / "items" / "off"
    _pkg_task(pub, mode="public")
    _pkg_task(off, mode="no-network")
    assert not calib.prepare(cfg, pub, "glm", 1).get("blocked")
    cfg["calib"]["agent_allowed_hosts"] = ["models.example.com"]
    assert "no-network" in calib.prepare(cfg, off, "glm", 1)["blocked"]
    assert calib.agent_network(pub / "pkg") == ("public", [])


def test_an_intent_saved_with_submit_off_is_rechecked_and_rewritten_at_launch(tmp_path):
    """Intent written while allow_submit=false and before the model host was configured: enabling submit later
    must not launch a job whose agent cannot reach its model; once the host is set the job config gets it."""
    item = tmp_path / "items" / "i1"
    _pkg_task(item)
    base = {"harbor_root": str(tmp_path), "student_launcher": "x.sh", "jobs_dir": str(tmp_path / "jobs"),
            "tmux_bin": "/nonexistent-tmux"}
    rec = calib.prepare(load_config(overrides={"calib": base}), item, "glm", 1)
    assert rec["state"] == "intent" and "warning" in rec
    saved = []
    cfg = load_config(overrides={"calib": dict(base, allow_submit=True)})
    out = calib.launch(cfg, dict(rec), None, saved.append)
    assert out["state"] == "intent" and "allowlist without hosts" in out["blocked"]
    cfg["calib"]["agent_allowed_hosts"] = ["models.example.com"]
    out = calib.launch(cfg, dict(rec), None, saved.append)
    conf = json.loads(Path(rec["config"]).read_text())
    assert conf["agents"][0]["extra_allowed_hosts"] == ["models.example.com"]
    assert out["state"] == "launch_failed" and "not launched" not in out["error"]   # got past the check (no tmux)


def test_resumed_intent_with_an_unreadable_config_is_blocked_not_rebuilt(tmp_path):
    item = tmp_path / "items" / "i1"
    _pkg_task(item)
    base = {"harbor_root": str(tmp_path), "student_launcher": "x.sh", "jobs_dir": str(tmp_path / "jobs"),
            "tmux_bin": "/nonexistent-tmux", "agent_allowed_hosts": ["m.example.com"]}
    rec = calib.prepare(load_config(overrides={"calib": base}), item, "glm", 1)
    Path(rec["config"]).write_text("{truncated")
    out = calib.launch(load_config(overrides={"calib": dict(base, allow_submit=True)}), dict(rec), None,
                       lambda r: None)
    assert out["state"] == "intent" and "unreadable" in out["blocked"]
    assert Path(rec["config"]).read_text() == "{truncated"


def test_resumed_intent_keeps_its_identity(tmp_path):
    """Only the allowed hosts of a saved intent's job config change on resume: model, job dir and task stay those
    of the record (ingest() checks a result against them)."""
    item = tmp_path / "items" / "i1"
    _pkg_task(item)
    base = {"harbor_root": str(tmp_path), "student_launcher": "x.sh", "jobs_dir": str(tmp_path / "jobs"),
            "tmux_bin": "/nonexistent-tmux", "agent_allowed_hosts": ["old.example.com"]}
    rec = calib.prepare(load_config(overrides={"calib": base}), item, "glm", 1)
    before = json.loads(Path(rec["config"]).read_text())
    cfg = load_config(overrides={"calib": dict(base, allow_submit=True, student_model="other-model",
                                               jobs_dir=str(tmp_path / "jobs2"),
                                               agent_allowed_hosts=["new.example.com"])})
    calib.launch(cfg, dict(rec), None, lambda r: None)
    after = json.loads(Path(rec["config"]).read_text())
    assert after["agents"][0]["extra_allowed_hosts"] == ["new.example.com"]
    after["agents"][0]["extra_allowed_hosts"] = before["agents"][0]["extra_allowed_hosts"]
    assert after == before
