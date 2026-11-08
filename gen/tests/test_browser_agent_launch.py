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


# ------------------------------------------------------------------ wrapper: budgets shared by concurrent calls

FAKE_EGO_SLOW_SSO = r"""#!{py}
import json, os, re, sys, time
code = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
inp = json.loads(re.match(r"const INPUT = (\{.*?\});\n", code).group(1))
log = "{log}"
open(log, "a").write(json.dumps({"action": inp.get("action"), "space": inp.get("space"),
                                 "sso_login": (inp.get("sso") or {}).get("login"),
                                 "handoff": (inp.get("handoff") or {}).get("enabled")}) + "\n")
mode = open("{mode}").read().strip() if os.path.exists("{mode}") else ""
out = {"status": "ok", "files": []}
if inp.get("action") == "sso_login":
    time.sleep(1.0)
    if not (inp.get("sso") or {}).get("login"):
        out = {"status": "refused", "error": "not enabled", "sso_submitted": False}
    elif mode == "sso_crash":
        print("crashed before the result line"); sys.exit(0)
    else:
        out = {"status": "ok", "sso": "fail", "sso_submitted": True, "sso_host": "sso.example"}
elif inp.get("action") == "open" and mode == "handoff":
    if (inp.get("handoff") or {}).get("enabled"):
        out = {"status": "handoff", "handoff": {"reason": "challenge: x", "host": "pub.example", "kind": "challenge"}}
    else:
        out = {"status": "ok", "files": [], "blocked": "verification page: x"}
print("TBSGEN_RESULT " + json.dumps(out))
"""


def _batch(tmp_path, monkeypatch, papers, **st_over):
    from tbsgen.dlledger import DownloadLedger
    from tbsgen.config import load_config
    cfg = load_config(None, {"fulltext": {"ledger_path": str(tmp_path / "led.sqlite"),
                                          "limits": {"browser": {"gap_s": 0.0, "jitter_s": 0.0}}}})
    ft = cfg["fulltext"]
    led = DownloadLedger(ft["ledger_path"], ft["limits"], ft["global_limits"], ft["host_caps"])
    ego = tmp_path / "ego"
    # the wrapper runs ego with a filtered environment: the log and mode paths are written into the script
    ego.write_text(FAKE_EGO_SLOW_SSO.replace("{py}", sys.executable).replace("{log}", str(tmp_path / "ego.log"))
                   .replace("{mode}", str(tmp_path / "ego.mode")))
    ego.chmod(0o755)
    bd = tmp_path / "batch"
    (bd / "papers").mkdir(parents=True)
    (bd / "files").mkdir()
    st = dict(space="t-sp", papers={str(i): dict(doi=d, title="T", ident=d, allow=["pub.example", "sso.example"])
                                    for i, d in enumerate(papers, 1)},
              deadline=time.time() + 600, ledger_path=str(led.path), limits=led.limits, global_limits=led.g,
              host_caps=led.host_caps, ego_binary=str(ego), gate_max_wait_s=150, max_calls_per_paper=25,
              call_timeout_s=20, challenge_cooldown_s=3600, challenge_wait_ms=1000, max_navigations_per_paper=12,
              nav_gap_s=0.0, handoff_enabled=False, handoff_wait_s=120, handoff_notify=False, bark_cmd=[],
              sso_login_enabled=True, sso_domains=["sso.example"], sso_max_submissions=2, type_allow=[])
    st.update(st_over)
    (bd / "batch.json").write_text(json.dumps(st))
    return bd, led


def _call(bd, *argv, env=None):
    import subprocess
    if env and "FAKE_EGO_MODE" in env:
        (bd.parent / "ego.mode").write_text(env["FAKE_EGO_MODE"])
    else:
        (bd.parent / "ego.mode").unlink(missing_ok=True)
    code = ("import sys; sys.path.insert(0, %r); from tbsgen.browser_agent import wrapper_main; "
            "sys.exit(wrapper_main(%r, sys.argv[1:]))" % (str(Path(browser_agent.__file__).parent.parent), str(bd)))
    return subprocess.Popen([sys.executable, "-c", code, *argv], stdout=subprocess.PIPE, text=True,
                            env=dict(os.environ, **(env or {})))


def _ego_actions(tmp_path):
    return [json.loads(x) for x in (tmp_path / "ego.log").read_text().splitlines()]


def test_concurrent_sso_logins_cannot_exceed_the_budget(tmp_path, monkeypatch):
    bd, _ = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1093/b.2", "10.1287/c.3"], sso_max_submissions=1)
    procs = [_call(bd, str(k), "sso_login") for k in ("1", "2", "3")]          # concurrent
    for p in procs:
        p.communicate(timeout=60)
    enabled = [e for e in _ego_actions(tmp_path) if e["action"] == "sso_login" and e["sso_login"]]
    assert len(enabled) == 1                                   # one reservation; the others ran disabled
    s = json.loads((bd / "state.json").read_text())["sso"]
    assert s["submissions"] == 1 and s["failed"] and "pending" not in s


def test_sso_without_a_result_still_counts_as_a_submission(tmp_path, monkeypatch):
    bd, _ = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1093/b.2"])
    p = _call(bd, "1", "sso_login", env={"FAKE_EGO_MODE": "sso_crash"})
    p.communicate(timeout=60)
    p = _call(bd, "2", "sso_login")
    p.communicate(timeout=60)
    ssos = [e for e in _ego_actions(tmp_path) if e["action"] == "sso_login"]
    # the first submission's outcome is unknown: no second automatic login in this batch
    assert ssos[0]["sso_login"] is True and ssos[1]["sso_login"] is False
    s = json.loads((bd / "state.json").read_text())["sso"]
    assert s["submissions"] == 1 and s["failed"]


def test_handoff_budget_is_per_publisher_not_per_doi_prefix(tmp_path, monkeypatch):
    # 10.1016 and 10.1006 are both Elsevier: one hand-off for the publisher per batch
    bd, _ = _batch(tmp_path, monkeypatch, ["10.1016/j.a.1", "10.1006/b.2"], handoff_enabled=True,
                   sso_login_enabled=False)
    env = {"FAKE_EGO_MODE": "handoff"}
    _call(bd, "1", "open", "https://pub.example/a", env=env).communicate(timeout=60)
    _call(bd, "2", "open", "https://pub.example/b", env=env).communicate(timeout=60)
    opens = [e for e in _ego_actions(tmp_path) if e["action"] == "open"]
    assert [e["handoff"] for e in opens] == [True, False]


def test_unpolled_handoff_is_expired_and_cools_the_publisher(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1093/b.2"], handoff_enabled=True,
                     sso_login_enabled=False, handoff_wait_s=0.5)
    _call(bd, "1", "open", "https://pub.example/a", env={"FAKE_EGO_MODE": "handoff"}).communicate(timeout=60)
    assert json.loads((bd / "papers" / "1.json").read_text())["handoff"]
    time.sleep(1.0)
    # the agent never polls paper 1; any call for another paper closes it and cools its publisher
    _call(bd, "2", "look").communicate(timeout=60)
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps["closed"].startswith("challenge") and "handoff" not in ps
    assert led.cooling("chrome:10.1080")


def test_end_of_batch_sweep_closes_open_handoffs(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1"], handoff_enabled=True, sso_login_enabled=False)
    _call(bd, "1", "open", "https://pub.example/a", env={"FAKE_EGO_MODE": "handoff"}).communicate(timeout=60)
    st = json.loads((bd / "batch.json").read_text())
    browser_agent._sweep_handoffs(bd, st, led, force=True)
    assert json.loads((bd / "papers" / "1.json").read_text())["closed"].startswith("challenge")
    assert led.cooling("chrome:10.1080")


def test_finish_is_not_cancelled_by_the_stop_file(tmp_path):
    slow = tmp_path / "ego-slow-finish"
    slow.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(1.5)\nprint('TBSGEN_RESULT {{\"status\": \"ok\"}}')\n")
    slow.chmod(0o755)
    (tmp_path / "STOP").write_text("agent run ended")
    done = browser_agent._finish_spaces(tmp_path, {"ego_binary": str(slow)}, ["sp-p1"])
    assert done == ["sp-p1:ok"]


def test_redaction_masks_whole_auth_headers_and_quoted_values():
    txt = ("Authorization: Bearer abc.def-123 TOKENTAIL\npassword = \"two words\"\n"
           "url=https://user:pw123@host/x api_key: k-1234567890 ok-text\n")
    r = browser_agent.redact(txt)
    for secret in ("abc.def-123", "TOKENTAIL", "two words", "pw123", "k-1234567890"):
        assert secret not in r, (secret, r)
    assert "ok-text" in r


def test_allowed_commands_reach_the_wrapper_unchanged_through_a_real_shell(tmp_path):
    """Every command shape the hook allows must arrive at the wrapper as the same argv (no shell comment,
    glob or quote processing)."""
    import shlex
    import subprocess
    probe = tmp_path / "tbsgen-browser"
    probe.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    probe.chmod(0o755)
    (tmp_path / "pdf").mkdir()                          # a glob would expand to this
    cmds = ["tbsgen-browser 1 wait_for sel div#pdf 20", "tbsgen-browser 1 wait_for url /pdf",
            "tbsgen-browser 1 select 2 Nanjing University", "tbsgen-browser 1 type 4 南京大学",
            "tbsgen-browser 1 open https://doi.org/10.1080/x?a=1", "tbsgen-browser 1 look @12 40"]
    for c in cmds:
        assert browser_agent.shell_allowed(c) is None, c
        r = subprocess.run(c, shell=True, capture_output=True, text=True, cwd=tmp_path,
                           env=dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"]))
        assert json.loads(r.stdout) == shlex.split(c)[1:], (c, r.stdout)
    for c in ["tbsgen-browser 1 wait_for sel #pdf", "tbsgen-browser 1 wait_for url *", "tbsgen-browser 1 type 4 a'b"]:
        assert browser_agent.shell_allowed(c), c
