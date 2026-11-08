"""Regression tests for starting the browser agent and the action scripts.

1. The agent's isolated HERMES_HOME (inside the batch directory) is a Hermes data root of its own; Hermes
   scopes its dependency environments per data root (`<root>/installs/`), so without them the launcher
   exits 1 within 0.1 s ("no dependency environment is committed for this install").  The fake `hermes`
   below does the same; the batch links the user's `installs/` into the isolated home, and the agent's
   stderr is kept in the batch directory and shown in the `agent_end` log event.
2. Action scripts (~40 KB) used to be written to the child's stdin by the first `communicate(input,
   timeout=0.5)` call only; a child slower than that to drain the pipe never saw EOF and the call hung
   until its timeout (the intermittent failures of the hand-off tests under load).
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

FAKE_EGO = r"""#!{py}
import json, os, re, sys, time
code = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
inp = json.loads(re.match(r"const INPUT = (\{.*?\});\n", code).group(1))
log = "{log}"
open(log, "a").write(json.dumps({"action": inp.get("action"), "space": inp.get("space"),
                                 "handoff": (inp.get("handoff") or {}).get("enabled")}) + "\n")
mode = open("{mode}").read().strip() if os.path.exists("{mode}") else ""
out = {"status": "ok", "files": []}
if inp.get("action") == "open" and mode == "handoff":
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
    ego.write_text(FAKE_EGO.replace("{py}", sys.executable).replace("{log}", str(tmp_path / "ego.log"))
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
              type_allow=[])
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


def test_handoff_budget_is_per_publisher_not_per_doi_prefix(tmp_path, monkeypatch):
    # 10.1016 and 10.1006 are both Elsevier: one hand-off for the publisher per batch
    bd, _ = _batch(tmp_path, monkeypatch, ["10.1016/j.a.1", "10.1006/b.2"], handoff_enabled=True)
    env = {"FAKE_EGO_MODE": "handoff"}
    _call(bd, "1", "open", "https://pub.example/a", env=env).communicate(timeout=60)
    _call(bd, "2", "open", "https://pub.example/b", env=env).communicate(timeout=60)
    opens = [e for e in _ego_actions(tmp_path) if e["action"] == "open"]
    assert [e["handoff"] for e in opens] == [True, False]


def test_unpolled_handoff_is_expired_and_cools_the_publisher(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1093/b.2"], handoff_enabled=True, handoff_wait_s=0.5)
    _call(bd, "1", "open", "https://pub.example/a", env={"FAKE_EGO_MODE": "handoff"}).communicate(timeout=60)
    assert json.loads((bd / "papers" / "1.json").read_text())["handoff"]
    time.sleep(1.0)
    # the agent never polls paper 1; any call for another paper closes it and cools its publisher
    _call(bd, "2", "look").communicate(timeout=60)
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps["closed"].startswith("challenge") and "handoff" not in ps
    assert led.cooling("chrome:10.1080")


def test_end_of_batch_sweep_closes_open_handoffs(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1"], handoff_enabled=True)
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
           "url=https://user:pw123@host/x\napi_key: k-1234567890 rest-of-line\nok-text\n")
    r = browser_agent.redact(txt)
    for secret in ("abc.def-123", "TOKENTAIL", "two words", "pw123", "k-1234567890", "rest-of-line"):
        assert secret not in r, (secret, r)
    assert "ok-text" in r                              # other lines are kept


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
            "tbsgen-browser 1 open https://doi.org/10.1080/x?a=1", "tbsgen-browser 1 look 40"]
    for c in cmds:
        assert browser_agent.shell_allowed(c) is None, c
        r = subprocess.run(c, shell=True, capture_output=True, text=True, cwd=tmp_path,
                           env=dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"]))
        assert json.loads(r.stdout) == shlex.split(c)[1:], (c, r.stdout)
    for c in ["tbsgen-browser 1 wait_for sel #pdf", "tbsgen-browser 1 wait_for url *", "tbsgen-browser 1 type 4 a'b"]:
        assert browser_agent.shell_allowed(c), c


def test_hover_is_reserved_as_a_navigation(tmp_path, monkeypatch):
    bd, _ = _batch(tmp_path, monkeypatch, ["10.1080/a.1"], max_navigations_per_paper=1)
    _call(bd, "1", "hover", "3").communicate(timeout=60)
    out, _ = _call(bd, "1", "hover", "3").communicate(timeout=60)
    assert "navigation limit" in out
    assert len([e for e in _ego_actions(tmp_path) if e["action"] == "hover"]) == 1


def test_challenge_handoff_expiry_cools_even_with_earlier_files(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1"], handoff_enabled=True)
    _call(bd, "1", "open", "https://pub.example/a", env={"FAKE_EGO_MODE": "handoff"}).communicate(timeout=60)
    sp = bd / "papers" / "1.json"
    ps = json.loads(sp.read_text())
    ps["accepted"] = [str(bd / "files" / "1" / "x.bin")]
    sp.write_text(json.dumps(ps))
    browser_agent._sweep_handoffs(bd, json.loads((bd / "batch.json").read_text()), led, force=True)
    assert led.cooling("chrome:10.1080")


def test_stop_during_the_navigation_wait_cancels_the_call(tmp_path, monkeypatch):
    bd, _ = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1080/b.2"], nav_gap_s=30.0)
    _call(bd, "1", "open", "https://pub.example/a").communicate(timeout=60)
    p = _call(bd, "2", "open", "https://pub.example/b")             # waits ~30 s for the publisher's slot
    time.sleep(1.5)
    (bd / "STOP").write_text("egress lost")
    out, _ = p.communicate(timeout=20)
    assert "STOP" in out
    assert len([e for e in _ego_actions(tmp_path) if e["action"] == "open"]) == 1


def test_redaction_of_json_headers_cookies_and_long_logs(tmp_path):
    txt = ('{"Authorization": "Basic dXNlcjpwYXNz"}\nCookie: csrf=x; sid=SIDSECRET; other=y\n'
           '"password": "with space"\n')
    r = browser_agent.redact(txt)
    for secret in ("dXNlcjpwYXNz", "SIDSECRET", "with space"):
        assert secret not in r, (secret, r)
    # a secret near the truncation point is never left without its key
    p = tmp_path / "err.log"
    head = "password=" + "Q" * 40 + "\n"
    p.write_text("x" * 10 + "\n" + head + "y" * (browser_agent.STDERR_KEEP - 20) + "\n")
    assert "QQQQ" not in browser_agent._stderr_tail(p)


def test_stderr_pipe_tail_is_redacted_line_by_line():
    import io
    data = ("noise\n" * 5000 + "token=TOPSECRET123\nlast\n").encode()
    tail = browser_agent._StderrTail(io.BytesIO(data), keep=200).text()
    assert "TOPSECRET123" not in tail and tail.endswith("last\n") and len(tail) <= 200


def test_agent_runner_crash_still_cleans_up(tmp_path, monkeypatch):
    _user_home(tmp_path, monkeypatch)
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, queue={"10.1080": 1}, agent={"keep_batch_dir": True},
                             handoff={"enabled": True})

    def boom(argv, env, timeout, on_event, stop):
        raise RuntimeError("runner crashed")
    with pytest.raises(RuntimeError):
        browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None, hermes_runner=boom)
    assert (_bd(f) / "STOP").exists()




def test_long_stderr_line_never_leaks_its_tail():
    import io
    body = b"x" * (browser_agent._StderrTail.MAX_LINE + 100) + b"password=" + b"\n"
    data = b"before\n" + body[:-1] + b"LEAKEDSECRET\nmaybe still the value\n\nnext record\n"
    tail = browser_agent._StderrTail(io.BytesIO(data), keep=4096).text()
    # an over-long line with a sensitive key: everything after it is dropped (its value's end is unknown)
    assert "LEAKEDSECRET" not in tail and "still the value" not in tail and "next record" not in tail, tail
    assert "before" in tail and "long line dropped" in tail and "rest of stderr dropped" in tail, tail
    # an over-long CONTINUATION of an open sensitive value (round 7): the record and the rest are dropped
    data = (b'token: "START\n' + b"y" * (browser_agent._StderrTail.MAX_LINE + 10) + b'\nSECRET_END"\nlater\n')
    tail = browser_agent._StderrTail(io.BytesIO(data), keep=4096).text()
    assert "START" not in tail and "SECRET_END" not in tail and "later" not in tail, tail
    # without a sensitive key the next line is kept
    data = b"x" * (browser_agent._StderrTail.MAX_LINE + 100) + b"\nnext line\n"
    assert "next line" in browser_agent._StderrTail(io.BytesIO(data), keep=4096).text()


def test_redaction_ignores_quote_tricks():
    for txt in ('{"password":"prefix\\"SECRET"}', 'Cookie: a="x\\"y"; sid=SECRET', 'token = "a" SECRET'):
        assert "SECRET" not in browser_agent.redact(txt), txt


def test_navigations_of_one_publisher_never_overlap(tmp_path, monkeypatch):
    bd, _ = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1080/b.2", "10.1080/c.3"], nav_gap_s=0.3)
    slow = tmp_path / "ego"
    slow.write_text(f"#!{sys.executable}\nimport json, time\nt0 = time.time()\ntime.sleep(0.8)\n"
                    f"open({str(tmp_path / 'spans.log')!r}, 'a').write(json.dumps([t0, time.time()]) + chr(10))\n"
                    "print('TBSGEN_RESULT ' + json.dumps({'status': 'ok', 'files': []}))\n")
    procs = [_call(bd, k, "open", f"https://pub.example/{k}") for k in ("1", "2", "3")]
    for pr in procs:
        pr.communicate(timeout=60)
    spans = sorted(json.loads(x) for x in (tmp_path / "spans.log").read_text().splitlines())
    assert len(spans) == 3
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert b0 >= a1 + 0.25, spans                  # one at a time, the gap counted from the end


def test_poll_that_ends_after_the_deadline_is_not_a_clear(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1"], handoff_enabled=True,
                     handoff_wait_s=1.5)
    _call(bd, "1", "open", "https://pub.example/a", env={"FAKE_EGO_MODE": "handoff"}).communicate(timeout=60)
    late = tmp_path / "ego"
    late.write_text(f"#!{sys.executable}\nimport json, time\ntime.sleep(2.5)\n"
                    "print('TBSGEN_RESULT ' + json.dumps({'status': 'ok', 'handoff_check': 'cleared'}))\n")
    out, _ = _call(bd, "1", "handoff_status").communicate(timeout=60)
    assert "timed out" in out
    assert json.loads((bd / "papers" / "1.json").read_text())["closed"].startswith("challenge")
    assert led.cooling("chrome:10.1080")


def test_redaction_of_nested_and_encoded_json_keys():
    for txt in ('{"body":"{\\"password\\":\\"SECRET\\"}"}', '{"a":"{\\\\\\"token\\\\\\":\\\\\\"SECRET\\\\\\"}"}',
                'q=password%22%3A%22SECRET', 'password%3DSECRET', "{'access_token': 'SECRET'}"):
        assert "SECRET" not in browser_agent.redact(txt), txt
    import io
    line = ('{"body":"{\\"password\\":\\"SECRET\\"}"}\nnext\n').encode()
    assert "SECRET" not in browser_agent._StderrTail(io.BytesIO(line), keep=4096).text()


def _pub(doi):
    from tbsgen.fulltext import publisher_of
    return publisher_of(doi)[0]


HANG_AFTER_HANDOFF_EGO = r"""#!{py}
import json, sys, time
print("TBSGEN_RESULT " + json.dumps({"status": "handoff", "handoff": {"reason": "challenge: x", "host": "pub.example",
                                     "kind": "challenge"}}), flush=True)
time.sleep(60)                                     # task.handOff() hangs: the call times out
"""

SILENT_HANG_EGO = r"""#!{py}
import time
time.sleep(60)
"""


def test_handoff_result_printed_before_a_timeout_is_still_booked(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1080/b.2"], handoff_enabled=True, call_timeout_s=3)
    (tmp_path / "ego").write_text(HANG_AFTER_HANDOFF_EGO.replace("{py}", sys.executable))
    out, _ = _call(bd, "1", "open", "https://pub.example/a").communicate(timeout=60)
    assert "handed the page to the user" in out, out
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps.get("handoff") and ps.get("handoff_kind") == "challenge"
    assert json.loads((bd / "state.json").read_text())["handoff"]["pub:" + _pub("10.1080/a.1")]


def test_silent_timeout_with_a_handoff_budget_closes_the_paper_and_cools(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1080/b.2"], handoff_enabled=True, call_timeout_s=3)
    (tmp_path / "ego").write_text(SILENT_HANG_EGO.replace("{py}", sys.executable))
    out, _ = _call(bd, "1", "open", "https://pub.example/a").communicate(timeout=60)
    assert "gave no result" in out, out
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps.get("closed") and not ps.get("handoff")
    assert json.loads((bd / "state.json").read_text())["handoff"]["pub:" + _pub("10.1080/a.1")]   # not re-opened
    assert led.cooling("chrome:10.1080")



def test_stop_during_a_call_with_a_handoff_budget_closes_the_paper_and_cools(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1080/b.2"], handoff_enabled=True, call_timeout_s=30)
    (tmp_path / "ego").write_text(SILENT_HANG_EGO.replace("{py}", sys.executable))
    pr = _call(bd, "1", "open", "https://pub.example/a")
    time.sleep(2.0)
    (bd / "STOP").write_text("x")
    out, _ = pr.communicate(timeout=60)
    assert "stopped" in out, out
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps.get("closed") and not ps.get("handoff"), ps
    assert json.loads((bd / "state.json").read_text())["handoff"]["pub:" + _pub("10.1080/a.1")]
    assert led.cooling("chrome:10.1080")


LATE_HANDOFF_EGO = r"""#!{py}
import json, time
print("TBSGEN_RESULT " + json.dumps({"status": "handoff", "handoff": {"reason": "challenge: x", "host": "pub.example",
      "kind": "challenge", "at_ms": int((time.time() - 500) * 1000)}}), flush=True)
"""


def test_handoff_window_counts_from_the_handover_not_the_return(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1"], handoff_enabled=True,
                     handoff_wait_s=120)
    (tmp_path / "ego").write_text(LATE_HANDOFF_EGO.replace("{py}", sys.executable))
    out, _ = _call(bd, "1", "open", "https://pub.example/a").communicate(timeout=60)
    assert "hand-off timed out" in out, out                  # handed over 500 s ago: window already over
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps.get("closed") and not ps.get("handoff")
    assert led.cooling("chrome:10.1080")


def test_final_sweep_books_an_interrupted_handoff_call(tmp_path, monkeypatch):
    from tbsgen.dlledger import DownloadLedger
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1"], handoff_enabled=True)
    st = json.loads((bd / "batch.json").read_text())
    pk = "pub:" + _pub("10.1080/a.1")
    rid = led.acquire("browser", "chrome:10.1080", "10.1080/a.1")
    (bd / "papers" / "1.json").write_text(json.dumps({"calls": 1, "rid": rid, "lkey": "chrome:10.1080"}))
    (bd / "state.json").write_text(json.dumps({"handoff_inflight": {pk: "1"}}))   # its wrapper was killed
    browser_agent._sweep_handoffs(bd, st, led, force=True)
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps.get("closed") and "not cleared" in ps["closed"], ps
    state = json.loads((bd / "state.json").read_text())
    assert state["handoff"][pk] and not state["handoff_inflight"]
    assert led.cooling("chrome:10.1080")



@pytest.mark.parametrize("body", ["import sys\nsys.exit(0)\n", "import sys\nprint('boom', file=sys.stderr)\nsys.exit(1)\n",
                                  # a result line without a valid status, or one that does not parse
                                  "print('TBSGEN_RESULT {\"files\": []}')\n", "print('TBSGEN_RESULT {broken')\n",
                                  "import os, sys\nsys.stdout.flush()\nos.kill(os.getpid(), 9)\n"])
def test_started_call_without_a_result_books_an_unknown_handoff(tmp_path, monkeypatch, body):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1080/b.2"], handoff_enabled=True)
    (tmp_path / "ego").write_text("#!" + sys.executable + "\n" + body)
    out, _ = _call(bd, "1", "open", "https://pub.example/a").communicate(timeout=60)
    assert "gave no result" in out, out
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps.get("closed") and not ps.get("handoff")
    state = json.loads((bd / "state.json").read_text())
    assert state["handoff"]["pub:" + _pub("10.1080/a.1")] and not (state.get("handoff_inflight") or {})
    assert led.cooling("chrome:10.1080")


def test_redaction_masks_multi_line_values():
    import io
    txt = 'password: "one\ntwo"\nnext\ntoken=\'a\nb\nc\'\nfine\n'
    for r in (browser_agent.redact(txt), browser_agent._StderrTail(io.BytesIO(txt.encode()), keep=4096).text()):
        assert "one" not in r and "two" not in r and "\nb\n" not in r and "c'" not in r, r
        assert "next" in r and "fine" in r


def test_redaction_drops_a_multi_line_record_whole():
    import io
    cases = [
        # the sensitive key on a continuation line: the whole record (its first line too) is dropped
        ("Traceback (most recent call last):\n  headers = {'Authorization': 'Bearer SECRET'}\nnext\n", "Traceback"),
        # a JSON object over several lines with a multi-line value
        ('{\n  "password": "SEC\nRET"\n}\nnext\n', "SEC"),
        # an open bracket, a backslash continuation
        ("token = [SECRET,\nSECRET2]\nnext\n", "SECRET"),
        ("secret=SECRET\\\nSECRET2\nnext\n", "SECRET"),
        # an open value is not closed by a blank line
        ('password: "SECRET\n\nSECRET2"\nnext\n', "SECRET"),
        # an empty value after the key: the value may be on the next line (round 8)

    ]
    for txt, gone in cases:
        for r in (browser_agent.redact(txt), browser_agent._StderrTail(io.BytesIO(txt.encode()), keep=4096).text()):
            assert gone not in r and "SECRET" not in r and "dropped" in r, (txt, r)
            assert r.rstrip().endswith("next"), (txt, r)
    # a one-line record keeps its key and is masked from it on; neighbouring records are untouched
    r = browser_agent.redact("before\npassword=SECRET\nafter\n")
    assert r == "before\npassword=[redacted]\nafter\n", r


def test_redaction_drop_survives_the_tail_cut():
    import io
    # a record that started before the kept tail: what is left of it is still dropped
    txt = "rec:\n" + "".join(f"  line {k}\n" for k in range(50)) + "  password=SECRET\n  more\nnext\n"
    r = browser_agent._StderrTail(io.BytesIO(txt.encode()), keep=120).text()
    assert "SECRET" not in r and "line 49" not in r and "more" not in r and r.rstrip().endswith("next"), r


def test_wrapper_exception_after_start_never_reopens_the_handoff(tmp_path, monkeypatch):
    """Round 7: a Python exception after the script started (e.g. while decoding its output) is booked as
    an unknown hand-off: the paper closes, the publisher cools, the hand-off budget stays used."""
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1080/b.2"], handoff_enabled=True)
    # output cut in the middle of a UTF-8 character, then a hang: killed at the call timeout
    (tmp_path / "ego").write_text("#!" + sys.executable + "\nimport sys, time\nsys.stdout.buffer.write(b'x\\xe4\\xb8')\n"
                                  "sys.stdout.flush()\ntime.sleep(60)\n")
    st = json.loads((bd / "batch.json").read_text())
    st["call_timeout_s"] = 2
    (bd / "batch.json").write_text(json.dumps(st))
    out, _ = _call(bd, "1", "open", "https://pub.example/a").communicate(timeout=90)
    ps = json.loads((bd / "papers" / "1.json").read_text())
    assert ps.get("closed") and not ps.get("handoff"), (out, ps)
    state = json.loads((bd / "state.json").read_text())
    assert state["handoff"]["pub:" + _pub("10.1080/a.1")] and not (state.get("handoff_inflight") or {})
    assert led.cooling("chrome:10.1080")


def test_wrapper_crash_while_holding_the_handoff_budget_books_it(tmp_path, monkeypatch):
    bd, led = _batch(tmp_path, monkeypatch, ["10.1080/a.1", "10.1080/b.2"], handoff_enabled=True)
    st = json.loads((bd / "batch.json").read_text())

    def boom(*a, **k):
        raise RuntimeError("wrapper bug")
    monkeypatch.setattr(browser_agent, "_paced_run", boom)
    with pytest.raises(RuntimeError):
        browser_agent._paper_call(bd, st, "1", "open", "https://pub.example/a", lambda m, rc: rc)
    state = json.loads((bd / "state.json").read_text())
    pk = "pub:" + _pub("10.1080/a.1")
    assert state["handoff"][pk] and not (state.get("handoff_inflight") or {}) and state["handoff_unknown"][pk] == "1"
    monkeypatch.undo()
    # the next call of the paper closes it and cools the publisher
    msgs = []
    rc = browser_agent._paper_call(bd, st, "1", "look", "", lambda m, rc: msgs.append(m) or rc)
    assert rc == 2 and "failed" in msgs[-1], msgs
    assert json.loads((bd / "papers" / "1.json").read_text())["closed"]
    assert led.cooling("chrome:10.1080")
    assert not json.loads((bd / "state.json").read_text()).get("handoff_unknown")


def test_wrapper_never_passes_a_login_mode_to_the_script(tmp_path, monkeypatch):
    """Round 9: no sso / local-test input reaches the action script, whatever the old config said."""
    bd, _ = _batch(tmp_path, monkeypatch, ["10.1080/a.1"], sso_login_enabled=True, sso_domains=["sso.example"])
    cap = tmp_path / "inp.json"
    (tmp_path / "ego").write_text("#!" + sys.executable + "\nimport json, re, sys\n"
                                  f"open({str(cap)!r}, 'w').write(re.match(r'const INPUT = (\\{{.*?\\}});\\n', sys.argv[2]).group(1))\n"
                                  "print('TBSGEN_RESULT ' + json.dumps({'status': 'ok'}))\n")
    _call(bd, "1", "look").communicate(timeout=60)
    inp = json.loads(cap.read_text())
    assert "sso" not in inp and "local_test" not in json.dumps(inp), inp
    out, _ = _call(bd, "1", "sso_login").communicate(timeout=60)
    assert "usage" in out, out                                  # not an action of the wrapper either


def test_redaction_after_an_empty_sensitive_value_drops_the_rest():
    import io
    # round 9: the value after an empty key may span any number of lines: everything after it is dropped
    for txt in ("password:\nSECRET1\nSECRET2\nnext\n", "a\ntoken =\n  SECRET1\n  SECRET2\nnext\n"):
        for r in (browser_agent.redact(txt), browser_agent._StderrTail(io.BytesIO(txt.encode()), keep=4096).text()):
            assert "SECRET" not in r and "next" not in r and "rest of stderr dropped" in r, (txt, r)
