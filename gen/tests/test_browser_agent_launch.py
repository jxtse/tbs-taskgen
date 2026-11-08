"""Regression tests for starting the browser agent and the action scripts.

1. The agent's isolated HERMES_HOME (inside the batch directory) is a Hermes data root of its own; Hermes
   scopes its dependency environments per data root (`<root>/installs/`), so without them the launcher
   exits 1 within 0.1 s ("no dependency environment is committed for this install").  The fake `hermes`
   below does the same; the batch links the user's `installs/` into the isolated home, and the agent's
   stderr is kept in the batch directory and shown in the `agent_end` log event.
2. Action scripts (~40 KB) used to be written to the child's stdin by the first `communicate(input,
   timeout=0.5)` call only; a child slower than that to drain the pipe never saw EOF and the call hung
   until its timeout (the intermittent failures of the hand-off / SSO tests under load).
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from tbsgen import browser_agent, browser_ego, fulltext
from test_remote_browser import BODY, TITLE, mk_agent  # noqa: F401  (helpers)

FAKE_LAUNCHER = r"""#!{py}
import json, os, subprocess, sys
home = os.environ.get("HERMES_HOME", "")
if not os.path.isdir(os.path.join(home, "installs")):
    sys.stderr.write("hermes: no dependency environment is committed for this install; run `hermes pm repair`\n")
    sys.stderr.write("debug: api_key=" + "sk-" + "THISMUSTNOTLEAK123456 token: abc123secret\n")
    sys.exit(1)
sys.stderr.write("session_id: s-1\n")
print(json.dumps({"type": "tool_use", "name": "terminal", "input": {"command": "tbsgen-browser 1 look"}}), flush=True)
hook = [l.split("command:", 1)[1].strip() for l in open(os.path.join(home, "config.yaml")) if "command:" in l][0]
r = subprocess.run([json.loads(hook)], input=json.dumps({"tool_name": "terminal",
                   "tool_input": {"command": "tbsgen-browser 1 look"}}), capture_output=True, text=True)
assert not r.stdout.strip(), r.stdout
r = subprocess.run("tbsgen-browser 1 look", shell=True, capture_output=True, text=True)
print(json.dumps({"type": "tool_result", "output": r.stdout[-300:]}), flush=True)
print(json.dumps({"type": "result", "exit_code": 0, "text": "1: skipped", "tokens": {"total": 3},
                  "duration_ms": 5, "session_id": "s-1"}), flush=True)
"""


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    for k in ("ELSEVIER_API_KEY", "ELSEVIER_INSTTOKEN", "WILEY_TDM_TOKEN"):
        monkeypatch.delenv(k, raising=False)


def _launcher(tmp_path):
    w = tmp_path / "hermes-launcher"
    w.write_text(FAKE_LAUNCHER.replace("{py}", sys.executable))
    w.chmod(0o755)
    return w


def _user_home(tmp_path, monkeypatch, with_installs=True):
    """The user's Hermes data root (HERMES_HOME of the process running the batch)."""
    uh = tmp_path / "user-hermes"
    uh.mkdir()
    if with_installs:
        (uh / "installs" / "abc").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(uh))
    return uh


def _run(f, pr, logs):
    return browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=logs.append)


def _agent_end(logs):
    return next(json.loads(m) for m in logs if '"agent_end"' in m)


def _bd(f):
    q = Path(f.cfg["fulltext"]["queue_path"])
    return next((q.parent / (q.name + ".browser")).glob("batch-*"))


def test_agent_starts_with_the_users_hermes_installs(tmp_path, monkeypatch):
    uh = _user_home(tmp_path, monkeypatch)
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, queue={"10.1080": 1},
                             agent={"hermes_bin": str(_launcher(tmp_path)), "keep_batch_dir": True})
    logs: list = []
    st = _run(f, pr, logs)
    end = _agent_end(logs)
    assert end["exit"] == 0 and end["tool_calls"] == 1, end
    home = _bd(f) / "hermes-home"
    assert (home / "installs").is_symlink() and (home / "installs").resolve() == (uh / "installs").resolve()
    # only the dependency state is shared: no user config, memories, auth or .env
    assert sorted(p.name for p in home.iterdir()) == ["config.yaml", "installs", "skills"]
    look = [json.loads(x) for x in (tmp_path / "ego.log").read_text().splitlines() if '"look"' in x]
    assert look, "the agent's tool call did not reach the action script"
    assert (_bd(f) / "agent-stderr.log").read_text().strip() == "session_id: s-1"
    assert st["agent"]["usage"]["exit"] == 0


def test_agent_start_failure_is_visible_and_redacted(tmp_path, monkeypatch):
    _user_home(tmp_path, monkeypatch, with_installs=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))   # no ~/.hermes either
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, queue={"10.1080": 1},
                             agent={"hermes_bin": str(_launcher(tmp_path)), "keep_batch_dir": True})
    logs: list = []
    _run(f, pr, logs)
    end = _agent_end(logs)
    assert end["exit"] == 1 and end["tool_calls"] == 0
    assert "no dependency environment is committed" in end["stderr_tail"]
    saved = (_bd(f) / "agent-stderr.log").read_text()
    for txt in (end["stderr_tail"], saved, json.dumps(logs)):
        assert "THISMUSTNOTLEAK" not in txt and "abc123secret" not in txt
    assert "[redacted]" in saved
    assert oct((_bd(f) / "agent-stderr.log").stat().st_mode & 0o777) == "0o600"


def test_agent_stderr_is_truncated_to_its_tail(tmp_path):
    p = tmp_path / "err.log"
    p.write_bytes(b"A" * 200_000 + b"\nlast line\n")
    tail = browser_agent._stderr_tail(p)
    assert tail.endswith("last line\n") and len(tail) <= browser_agent.STDERR_KEEP
    assert p.stat().st_size <= browser_agent.STDERR_KEEP


def test_hermes_installs_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "elsewhere"))
    inst = tmp_path / "custom-installs"
    inst.mkdir()
    bd = tmp_path / "bd"
    bd.mkdir()
    home = browser_agent._hermes_home(bd, {"hermes_installs_dir": str(inst)}, "/h")
    assert (home / "installs").resolve() == inst.resolve()


# ------------------------------------------------------------------ action scripts: no stdin pipe stall

SLOW_EGO = r"""#!{py}
import json, re, sys, time
time.sleep(1.2)                              # slower to start than the first communicate() timeout
code = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
m = re.match(r"const INPUT = (\{.*?\});\n", code)
print("TBSGEN_RESULT " + json.dumps({"status": "ok", "len": len(code), "action": json.loads(m.group(1)).get("action")}))
"""


def _slow_ego(tmp_path):
    w = tmp_path / "ego-slow"
    w.write_text(SLOW_EGO.replace("{py}", sys.executable))
    w.chmod(0o755)
    return w


def test_action_script_does_not_stall_on_a_slow_child(tmp_path):
    t0 = time.time()
    res, _ = browser_agent._run_script({"ego_binary": str(_slow_ego(tmp_path))}, dict(action="look"), 10.0, tmp_path)
    assert res.get("status") == "ok" and res["action"] == "look", res
    assert res["len"] > 30_000                    # the whole ~40 KB action script arrived
    assert time.time() - t0 < 6


def test_script_channel_does_not_stall_on_a_slow_child(tmp_path):
    t0 = time.time()
    res = browser_ego._exec({"binary": str(_slow_ego(tmp_path))}, dict(action="fetch"), 10.0)
    assert res.get("status") == "ok", res
    assert time.time() - t0 < 6
