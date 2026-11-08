"""Pre-release checks (prerelease.py) and the package's network policy (package.py): unit tests.  The Docker /
Harbor end-to-end checks are in test_integration.py."""
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
                                 "pydecayfit\n"])
def test_dependency_on_the_papers_package_is_blocking(tmp_path, req):
    r = prerelease.readymade_check(mk_task(tmp_path, req=req), CAND, SPEC)
    assert not r["ok"] and r["findings"][0]["kind"] == "dependency" and "DecayFit" in r["findings"][0]["matches"]


def test_pip_line_in_a_dockerfile_is_blocking(tmp_path):
    df = "FROM python:3.11-slim\nRUN pip install --no-cache-dir numpy==2.1.3 \\\\\n    decayfit==0.3\n"
    r = prerelease.readymade_check(mk_task(tmp_path, dockerfile=df), CAND, SPEC)
    assert [f["name"] for f in r["findings"]] == ["decayfit==0.3".split("==")[0]]


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


def test_calib_refuses_to_launch_an_allowlisted_task_without_a_model_host(tmp_path):
    item = tmp_path / "items" / "i1"
    (item / "pkg").mkdir(parents=True)
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
