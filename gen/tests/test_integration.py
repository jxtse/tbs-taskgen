"""Integration tests: the whole pipeline on the toy fixture with the fake model, task code executed in Docker
(tbsgen-base:py311-sci-v2).  Skipped when Docker or the image is unavailable."""
import json
from pathlib import Path

import pytest

from tbsgen import sandbox
from tbsgen.budget import Ledger
from tbsgen.config import load_config
from tbsgen.dryrun import FIX, make_toy_candidate, simulate_harbor
from tbsgen.fakes import toy_script
from tbsgen.models import FakeClient
from tbsgen.package import validate_pkg
from tbsgen.pipeline import Pipeline
from tbsgen.report import daily_report
from tbsgen.state import read_json

IMAGE = load_config()["sandbox"]["image"]
pytestmark = pytest.mark.skipif(not (sandbox.docker_available() and sandbox.image_exists(IMAGE)),
                                reason="docker / sandbox image unavailable")


def mk(tmp_path, fake_kw=None, budget=None, script=None, roles=("spec", "gen", "tests", "repair")):
    cfg = load_config(None, {"run": {"root": str(tmp_path)},
                             "fulltext": {"cache_dirs": [str(tmp_path / "fulltext_cache")], "allow_network": False},
                             "budget": budget or {}})
    led = Ledger(tmp_path / "ledger.sqlite", cfg)
    fake = FakeClient(led, cfg, script or toy_script(FIX, **(fake_kw or {})), name="fake-opus")
    return Pipeline(cfg, clients={r: fake for r in roles}), fake


def test_happy_path_to_harbor_package_and_calibration(tmp_path):
    P, fake = mk(tmp_path)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    assert it.state["stage"] == "S9" and it.state["status"] == "waiting", it.state
    gate = read_json(sorted(it.p("gates").glob("*.json"))[-1])
    assert gate["ok"], gate["gates"]
    s0 = gate["details"]["seeds"]["0"]
    assert s0["ref"]["passed"] and not s0["nop"]["passed"] and not s0["mid"]["passed"]
    assert 1 <= s0["mid"]["n_failed"] <= 2 and s0["naive"]["n_failed"] >= 2
    assert validate_pkg(it.p("pkg"))["ok"]
    assert not any(p.name in ("generate.py", "mid.py", "naive.py") for p in it.p("pkg").rglob("*"))
    runs = read_json(it.p("calib", "runs.json"))
    conf = json.loads(Path(runs[0]["config"]).read_text())
    assert conf["agents"][0]["model_name"] == "glm-5.3" and conf["agents"][0]["kwargs"]["version"] == "2.1.280"
    assert runs[0]["submitted"] is False
    # GLM pass, GLM fail -> unstable -> kept -> S10
    st = simulate_harbor(P, iid, ["pass", "fail"])
    assert st["stage"] == "S10" and st["status"] == "done" and st["calib_label"] == "unstable"
    assert (tmp_path / "release" / "manifest.jsonl").exists()
    rep = daily_report(P).read_text()
    assert "fake-opus" in rep and "日报" in rep
    # re-running a finished run does nothing and makes no model calls
    n = len(fake.log)
    P.run()
    assert len(fake.log) == n


NET_PROBE = r"""#!/bin/bash
set -uo pipefail
mkdir -p /logs/agent
{
python - <<'PY'
import urllib.request
for u in ("https://pypi.org/simple/six/", "https://github.com/"):
    try:
        urllib.request.urlopen(u, timeout=10); print("OPEN", u)
    except Exception as e:
        print("BLOCKED", u, type(e).__name__)
PY
if timeout 60 pip download --no-deps -q -d /tmp/x six >/dev/null 2>&1; then echo "PIP_OK"; else echo "PIP_FAILED"; fi
} > /logs/agent/net_probe.txt 2>&1
python /solution/solve.py
"""


def _packaged_toy(tmp_path):
    P, _ = mk(tmp_path)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    assert it.state["stage"] == "S9", it.state
    return P, it


def test_packaged_image_builds_and_reference_runs_offline(tmp_path):
    """The packaged Dockerfile builds, and the reference solution plus the verifier pass inside the image with the
    network off (docker --network none); pip install fails there."""
    import subprocess
    P, it = _packaged_toy(tmp_path)
    pkg = it.p("pkg")
    tag = f"tbsgen-pkgtest-{it.item_id}".lower()
    b = subprocess.run(["docker", "build", "-q", "-t", tag, str(pkg / "environment")], capture_output=True, text=True,
                       timeout=1200)
    assert b.returncode == 0, b.stderr[-2000:]
    try:
        cmd = ("bash /solution/solve.sh && bash /tests/test.sh && cat /logs/verifier/reward.txt && "
               "(pip install --no-cache-dir -q six >/dev/null 2>&1 && echo PIP_OK || echo PIP_FAILED)")
        r = subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{pkg / 'solution'}:/solution:ro",
                            "-v", f"{pkg / 'tests'}:/tests:ro", tag, "bash", "-c", cmd],
                           capture_output=True, text=True, timeout=1200)
        out = r.stdout.split()
        assert r.returncode == 0 and "1" in out and out[-1] == "PIP_FAILED", (r.stdout[-1000:], r.stderr[-2000:])
    finally:
        subprocess.run(["docker", "image", "rm", "-f", tag], capture_output=True)


def test_harbor_agent_phase_allowlist_blocks_internet(tmp_path):
    """Real Harbor (local Docker environment) on the packaged toy task, default network policy: during the agent
    phase pypi.org / github.com are unreachable and pip fails; --allow-agent-host opens exactly that host; the
    oracle still passes.  Needs a `harbor` binary (TBS_HARBOR_BIN or calib.harbor_root); skipped otherwise."""
    import shutil
    import subprocess
    from tbsgen import harbor_local
    harbor = harbor_local.find_harbor(load_config(overrides={"calib": {"harbor_root": __import__("os").environ.get(
        "TBS_HARBOR_ROOT")}}))
    if not harbor:
        pytest.skip("harbor binary not available")
    P, it = _packaged_toy(tmp_path)
    task = tmp_path / "h" / "tasks" / "net"
    shutil.copytree(it.p("pkg"), task)
    from tbsgen.package import DOCKERFILE_LOCAL
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE_LOCAL.format(image=IMAGE, offline=""))
    (task / "solution" / "solve.sh").write_text(NET_PROBE)
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "DOCKER_CONFIG")}

    def run(job, *extra):
        p = subprocess.run([harbor, "run", "-p", str(task), "-a", "oracle", "-o", str(tmp_path / "h" / "jobs"),
                            "--job-name", job, "-n", "1", "-q", "--yes", *extra], capture_output=True, text=True,
                           timeout=1800, env=env)
        assert p.returncode == 0, p.stderr[-2000:]
        jd = tmp_path / "h" / "jobs" / job
        probe = next(jd.rglob("net_probe.txt")).read_text()
        res = json.loads(next(x for x in jd.glob("*/result.json")).read_text())
        return probe, ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    probe, reward = run("allow")
    assert "BLOCKED https://pypi.org" in probe and "BLOCKED https://github.com" in probe and "PIP_FAILED" in probe
    assert reward == 1.0
    probe, reward = run("allowhost", "--allow-agent-host", "pypi.org")
    assert "OPEN https://pypi.org" in probe and "BLOCKED https://github.com" in probe and reward == 1.0


def test_resume_after_crash_mid_stage(tmp_path):
    """Kill-like interruption: run stops after S6A_run; a new Pipeline object continues from state.json."""
    P, fake = mk(tmp_path)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run(max_steps=7)        # S1..S6A_run
    st = P.store.item(iid).state
    assert st["stage"] == "S6B", st["stage"]
    calls_before = len(fake.log)
    P2, fake2 = mk(tmp_path)
    P2.run()
    st = P2.store.item(iid).state
    assert st["stage"] == "S9" and st["status"] == "waiting"
    assert calls_before == 2 and len(fake2.log) == 1      # S5+S6A before; only S6C after resume


def test_budget_breaker_pauses_and_resume(tmp_path):
    P, fake = mk(tmp_path, budget={"daily_calls_per_model": {"fake-opus": 2}})
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    st = P.store.item(iid).state
    assert st["status"] == "paused" and st["reason"] == "budget:daily_calls" and st["stage"] == "S6C"
    assert P.ledger.summary()["calls_total"] == 2
    # raise the cap (new config) and resume
    P2, _ = mk(tmp_path, budget={"daily_calls_per_model": {"fake-opus": 10}})
    assert P2.resume() == [iid]
    P2.run()
    assert P2.store.item(iid).state["status"] == "waiting"


def test_quota_error_opens_breaker_and_pauses(tmp_path):
    def script(stage, system, user):
        return {"error": "quota", "text": "HTTP 429 API key 额度已用完"}
    P, _ = mk(tmp_path, script=script)
    ids = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    st = P.store.item(ids[0]).state
    assert st["status"] == "paused" and st["reason"] == "model:quota"
    assert P.ledger.breaker("fake-opus")["open"]


def test_reference_crash_goes_to_repair_and_recovers(tmp_path):
    P, _ = mk(tmp_path, fake_kw={"break_reference": True})
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    assert it.state["status"] == "waiting", it.state
    assert it.state["repair_rounds"] == 1
    assert read_json(it.p("repair", "round-1", "review.json"))["ok"]


def test_injected_leak_blocked_then_repaired(tmp_path):
    P, _ = mk(tmp_path, fake_kw={"leak_in_public": True})
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    g1 = read_json(it.p("gates", "attempt-1.json"))
    assert not g1["gates"]["leak"] and "leak" in g1["blocking"], g1["gates"]
    assert "L2" in g1["details"]["leak"]["classes"]
    assert it.state["status"] == "waiting" and read_json(it.p("gates", "attempt-2.json"))["ok"]


def test_tdm_verbatim_copy_blocked_then_repaired(tmp_path):
    """A full text from Elsevier's TDM API; S6C pastes 300 chars of it into instruction.md: S7 blocks
    (tdm_verbatim), S8 removes the passage, the second S7 passes."""
    from tbsgen.dryrun import TOY_TEXT
    passage = TOY_TEXT[:300]
    P, _ = mk(tmp_path, fake_kw={"copy_source": passage})
    cand = make_toy_candidate(tmp_path)
    prov = tmp_path / "fulltext_cache" / "10.0000_tbsgen.toy.decay.json"
    prov.write_text(json.dumps(dict(doi=cand["doi"], channel="elsevier", licence="elsevier_tdm")))
    [iid] = P.ingest([cand])
    P.run()
    it = P.store.item(iid)
    assert read_json(it.p("S3_fulltext.json"))["cached_channel"] == "elsevier"
    g1 = read_json(it.p("gates", "attempt-1.json"))
    assert "tdm_verbatim" in g1["blocking"] and g1["details"]["tdm_verbatim"]["findings"][0]["chars"] >= 290
    g2 = read_json(it.p("gates", "attempt-2.json"))
    assert g2["ok"] and g2["gates"]["tdm_verbatim"], g2["gates"]
    assert passage[:80] not in it.p("pkg", "instruction.md").read_text()
    assert it.state["status"] == "waiting"


def test_unrepairable_is_dropped_after_two_rounds(tmp_path):
    P, _ = mk(tmp_path, fake_kw={"leak_in_public": True, "repair_fixes": False})
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    st = P.store.item(iid).state
    assert st["status"] == "dropped" and st["reason"].startswith("repair_limit"), st
    assert st["repair_rounds"] == 2


def test_s6a_truncated_answer_gets_one_continuation(tmp_path):
    """e2e-iter1 regression: a real S6A answer stopped at max_tokens after generator/generate.py and the item was
    dropped (s6a_missing_files).  Now: partial last block discarded, one continuation asks for the missing files."""
    base = toy_script(FIX)
    seen = {"S6A": 0}

    def script(stage, system, user):
        out = base(stage, system, user)
        if stage != "S6A":
            return out
        seen["S6A"] += 1
        if seen["S6A"] == 1:      # cut inside solution/solve.py
            cut = out.index("=== FILE: solution/solve.py ===") + 60
            return {"text": out[:cut], "stop": "max_tokens", "usage": {"output_tokens": 32000}}
        assert "Files already received" in user and "solution/solve.py" in user.split("missing files")[-1]
        return out[out.index("=== FILE: solution/solve.py ==="):]
    P, fake = mk(tmp_path, script=script)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    assert seen["S6A"] == 2
    calls = read_json(it.p("S6A_calls.json"))
    assert calls[0]["stop"] == "max_tokens" and "solution/solve.py" not in calls[0]["got"] and calls[1]["cont"]
    assert len(list(it.p("raw").glob("S6A-*.txt"))) == 2
    assert it.state["status"] == "waiting", it.state


def test_mid_not_separable_reworks_only_mid(tmp_path):
    """v2.2: S6B not_separable -> S6M rewrites baselines/mid.py around another drop node (1 call), no new spec."""
    base = toy_script(FIX)
    seen = []

    def script(stage, system, user):
        seen.append(stage)
        if stage == "S6A":   # mid == reference -> fails nothing -> not separable
            out = base(stage, system, user)
            solve = (FIX / "solution/solve.py").read_text()
            i = out.index("=== FILE: baselines/mid.py ===")
            j = out.index("=== FILE:", i + 10)
            return out[:i] + "=== FILE: baselines/mid.py ===\n" + solve + out[j:]
        if stage == "S6M":
            assert "MID_DROP" in user and "fails NONE" in user
            return "MID_DROP: C2\n=== FILE: baselines/mid.py ===\n" + (FIX / "baselines/mid.py").read_text() + \
                "=== END ===\n"
        return base(stage, system, user)
    P, fake = mk(tmp_path, script=script)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    assert seen.count("S5") == 1 and seen.count("S6A") == 1 and seen.count("S6M") == 1, seen
    assert it.state["status"] == "waiting", it.state
    assert read_json(it.p("_spec", "spec.json"))["depgraph"]["mid_drop"] == ["C2"]


def test_s6m_new_node_outside_naive_drop_keeps_inclusion(tmp_path):
    """Review #8: S6M picking a node outside naive_drop must keep naive_drop ⊇ mid_drop (spec re-validated)."""
    from tbsgen.spec import validate_spec
    base = toy_script(FIX)

    def script(stage, system, user):
        if stage == "S6A":
            out = base(stage, system, user)
            solve = (FIX / "solution/solve.py").read_text()
            i = out.index("=== FILE: baselines/mid.py ===")
            j = out.index("=== FILE:", i + 10)
            return out[:i] + "=== FILE: baselines/mid.py ===\n" + solve + out[j:]
        if stage == "S6M":
            return ("MID_DROP: P1\n=== FILE: baselines/mid.py ===\n" + (FIX / "baselines/mid.py").read_text() +
                    "=== FILE: baselines/naive.py ===\n" + (FIX / "baselines/naive.py").read_text() + "=== END ===\n")
        return base(stage, system, user)
    P, fake = mk(tmp_path, script=script)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    spec = read_json(it.p("_spec", "spec.json"))
    g = spec["depgraph"]
    if it.state["status"] == "dropped":       # P1 is a given_file premise: validate_spec must refuse it
        assert it.state["reason"].startswith("s6m_invalid_spec"), it.state
        assert g["mid_drop"] == ["C1"]
    else:
        assert set(g["mid_drop"]) <= set(g["naive_drop"]) and validate_spec(spec, P.cfg) == []


def _prog_task(root: Path, direct_exec: bool) -> tuple[Path, Path, Path]:
    task, gen = root / "task", root / "gen"
    (task / "tests").mkdir(parents=True)
    for d in ("environment/data", "tests/truth", "tests/hidden/set1"):
        (gen / d).mkdir(parents=True)
    (gen / "environment/data/x.csv").write_text("a\n1\n")
    (gen / "tests/hidden/set1/x.csv").write_text("a\n2\n")
    (gen / "tests/truth/secret.csv").write_text("truth\n42\n")
    run = ("import subprocess, sys, tempfile\n"
           "def go():\n    o = tempfile.mkdtemp()\n"
           "    p = subprocess.run([sys.executable, '/app/output/solution_cli.py', '/tests/hidden/set1', o],"
           " capture_output=True, text=True)\n"
           "    from _metrics import _record\n"
           "    _record({'name': '_solution_cli_run', 'value': 0, 'rc': p.returncode, 'stderr_tail': p.stderr,"
           " 'aux': True, 'passed': True, 'threshold': 1, 'direction': '<='})\n") if direct_exec else \
        ("from _metrics import run_solution, hidden_sets\n"
         "def go():\n    for d in hidden_sets():\n        run_solution(60, d)\n")
    (task / "tests/measure.py").write_text(run + "def compute():\n    go()\n    return {'m': 1.0}\n")
    (task / "tests/test_outputs.py").write_text("from measure import compute\nfrom _metrics import check\n"
                                                "def test_m():\n    check('m', compute()['m'], 2.0, '<=')\n")
    ref = root / "ref"
    ref.mkdir()
    (ref / "solution_cli.py").write_text("pass\n")
    return task, gen, ref


def test_verifier_sees_app_data(tmp_path):
    """v2.1 regression (e2e-iter1): verifier steps lost the /app/data mount; Harbor's verifier has it."""
    from tbsgen import runner
    cfg = load_config(None, {"run": {"root": str(tmp_path)}})
    task, gen, ref = _prog_task(tmp_path, False)
    (task / "tests/measure.py").write_text("import json\ndef compute():\n"
                                          "    return {'m': float(open('/app/data/x.csv').read().split()[1])}\n"
                                          "if __name__ == '__main__':\n    print(json.dumps(compute()))\n")
    m = runner.measure(cfg, task, gen, ref, tmp_path / "work", "ad")
    assert m["values"] == {"m": 1.0}, m


@pytest.mark.parametrize("direct_exec", [True, False])
def test_truth_isolation_probe(tmp_path, direct_exec):
    """e2e-iter1 regression: the generated measure.py ran solution_cli.py itself (as root) instead of via
    _metrics.run_solution, so /tests/truth was readable and a truth-reading CLI got Harbor reward 1."""
    from tbsgen import runner
    cfg = load_config(None, {"run": {"root": str(tmp_path)}})
    task, gen, ref = _prog_task(tmp_path, direct_exec)
    r = runner.truth_probe(cfg, task, gen, ref, tmp_path / "work", "tp")
    assert r["via_run_solution"]
    assert r["ok"] is (not direct_exec), r


def _attack_task(root: Path, cli_src: str) -> tuple[Path, Path, Path]:
    """PROG task whose verifier runs `cli_src` (the agent program) via run_solution and reports what it read."""
    task, gen, ref = _prog_task(root, False)
    (task / "tests/measure.py").write_text(
        "import json\nfrom _metrics import run_solution, hidden_sets\n"
        "def compute():\n    for d in hidden_sets():\n        run_solution(60, d)\n    return {'m': 1.0}\n"
        "if __name__ == '__main__':\n    print(json.dumps(compute()))\n")
    (ref / "solution_cli.py").write_text(cli_src)
    return task, gen, ref


ATTACK = r'''import os, sys
hits = []
for p in ("/tests/truth/secret.csv", "/src/tests/truth/secret.csv", "/app/output/leak/secret.csv",
          "/tests/measure.py", "/src/tests/measure.py"):
    try:
        open(p).read(); hits.append(p)
    except Exception:
        pass
sys.stderr.write("ATTACK_HITS " + ",".join(hits) + "\\n")
'''


def test_planted_symlink_and_staging_copy_do_not_expose_truth(tmp_path):
    """Review #1/#2 (e2e-iter1): an agent-made /app/output/leak -> /tests/truth must not get the truth chmod-ed
    open by the workspace pass, and the /src/tests staging copy must not be readable by the agent program."""
    from tbsgen import runner
    cfg = load_config(None, {"run": {"root": str(tmp_path)}})
    task, gen, ref = _attack_task(tmp_path, ATTACK)
    (ref / "leak").symlink_to("/tests/truth")
    m = runner.test(cfg, task, gen, ref, tmp_path / "work", "atk")
    runs = [x for x in m["metrics"] if x.get("name") == "_solution_cli_run"]
    assert runs and runs[0]["sandboxed"], m
    tail = runs[0]["stderr_tail"]
    assert "ATTACK_HITS" in tail and not any(p in tail for p in ("secret", "measure")), runs


def test_truth_probe_reads_concrete_paths(tmp_path):
    """Review #1: probe opens concrete target paths (no directory listing), incl. the symlink route."""
    from tbsgen import runner
    cfg = load_config(None, {"run": {"root": str(tmp_path)}})
    task, gen, ref = _prog_task(tmp_path, False)
    r = runner.truth_probe(cfg, task, gen, ref, tmp_path / "work", "tp")
    assert r["ok"] and r["n_targets"] >= 3, r


def test_intake_failure_routes(tmp_path):
    P, _ = mk(tmp_path)
    cands = [dict(doi="10.1/review", title="A review of things", abstract="This review surveys perspectives."),
             dict(doi="10.1/nofulltext", title="Robust estimator algorithm",
                  abstract="We propose an estimation algorithm with simulation benchmark; code on github in python.")]
    ids = P.ingest(cands)
    P.run()
    a, b = (P.store.item(i).state for i in ids)
    assert a["status"] == "dropped" and a["reason"] == "low_screen_score"
    assert b["status"] == "parked" and b["reason"] == "fulltext_unavailable"
    assert (P.store.item(ids[1]).p("S3_queue.json")).exists()
