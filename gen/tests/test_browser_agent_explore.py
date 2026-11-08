"""Offline tests for the browser agent's expanded exploration actions, hand-off and SSO login.

The action script itself (browser_agent_action.mjs) runs against a real ego-browser and local pages in
test_browser_actions_live.py; here the wrapper (browser_agent.py) is exercised with the fake `ego-browser`
and fake `hermes` from test_remote_browser.py.  The wrapper runs in child processes (the fake hermes calls
the generated `tbsgen-browser` executable), so cross-process effects are asserted through files: ego.log,
the batch directory, the ledger and a fake `osascript` first on PATH.
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from tbsgen import browser_agent, fulltext
from test_remote_browser import BODY, TITLE, mk_agent  # noqa: F401  (helpers)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    for k in ("ELSEVIER_API_KEY", "ELSEVIER_INSTTOKEN", "WILEY_TDM_TOKEN"):
        monkeypatch.delenv(k, raising=False)


def _run(f, pr):
    return browser_agent.run_batch(f, prober=pr, sleep=lambda s: None, log=lambda m: None)


def _ego_log(tmp_path):
    return [json.loads(x) for x in (tmp_path / "ego.log").read_text().splitlines()]


def _notes(led):
    rows = sqlite3.connect(led.path).execute(
        "select doi, status, note from attempts where channel='browser' order by rowid").fetchall()
    return {r[0]: (r[1], r[2] or "") for r in rows}


def _fake_osascript(tmp_path, monkeypatch):
    """A fake `osascript` first on PATH: the wrapper's notification runs in a child process."""
    log = tmp_path / "osascript.log"
    d = tmp_path / "notifybin"
    d.mkdir(exist_ok=True)
    w = d / "osascript"
    w.write_text(f"#!{sys.executable}\nimport os, sys\n"
                 f"open(os.environ['FAKE_OSASCRIPT_LOG'], 'a').write(' '.join(sys.argv[1:]) + chr(10))\n")
    w.chmod(0o755)
    monkeypatch.setenv("FAKE_OSASCRIPT_LOG", str(log))
    monkeypatch.setenv("PATH", str(d) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(browser_agent.browser_ego, "ENV_ALLOW",
                        browser_agent.browser_ego.ENV_ALLOW + ("FAKE_OSASCRIPT_LOG",))
    return log


def _bd(f):
    q = Path(f.cfg["fulltext"]["queue_path"])
    return next((q.parent / (q.name + ".browser")).glob("batch-*"))


# --------------------------------------------------------------------------- allowlist / hook

def test_shell_allowed_new_actions():
    ok = [
        "tbsgen-browser 1 look", "tbsgen-browser 2 look full", "tbsgen-browser 1 look @12",
        "tbsgen-browser 1 look full 40", "tbsgen-browser 3 look @4 8",
        "tbsgen-browser 1 open https://doi.org/10.1080/x",
        "tbsgen-browser 1 click 12", "tbsgen-browser 1 click_xy 320 240", "tbsgen-browser 1 hover 3",
        "tbsgen-browser 1 scroll down", "tbsgen-browser 1 scroll up 800",
        "tbsgen-browser 1 select 2 Nanjing University", "tbsgen-browser 1 type 4 Nanjing University",
        "tbsgen-browser 1 type 4 南京大学", "tbsgen-browser 1 type 4 hunter2",   # shape ok; the wrapper's
        # whitelist (institution names only) refuses it before the action script runs
        "tbsgen-browser 1 screenshot", "tbsgen-browser 1 tabs", "tbsgen-browser 1 switch p2",
        "tbsgen-browser 1 back", "tbsgen-browser 1 wait_for url /pdf 20",
        "tbsgen-browser 1 wait_for sel #pdf-button", "tbsgen-browser 1 save_pdf",
        "tbsgen-browser 1 sso_login", "tbsgen-browser 1 handoff_status", "tbsgen-browser finish",
        "sleep 30",
    ]
    for c in ok:
        assert browser_agent.shell_allowed(c) is None, c
    bad = [
        "tbsgen-browser 1 open https://x.example/a;rm -rf /",     # shell metachar
        "tbsgen-browser 1 open https://x.example/a && echo hi",
        "tbsgen-browser 1 open \"https://x.example/a\"",
        "tbsgen-browser 1 type 4 'x';env",
        "tbsgen-browser 1 select 1 a|b",
        "tbsgen-browser 1 click 1234",                              # 4-digit item index: out of range
        "tbsgen-browser 1 look full full",
        "tbsgen-browser 1 wait_for url $(cat /etc/passwd)",
        "tbsgen-browser 1 wait_for exec x",
        "tbsgen-browser 1 look\n touch x",
        "tbsgen-browser 1 open https://sci-hub.se/x",               # unauthorized source
        "tbsgen-browser 1 download",
        "ego-browser nodejs -e 'process.exit(1)'",
        "tbsgen-browser 1 switch p2;open x",
    ]
    for c in bad:
        assert browser_agent.shell_allowed(c), c


def _hook(batch_dir, tool, inp):
    import subprocess
    w = Path(batch_dir) / "hook"
    w.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
                 "from tbsgen.browser_agent import hook_main\nsys.exit(hook_main("
                 f"{str(batch_dir)!r}, []))")
    w.chmod(0o755)
    r = subprocess.run([str(w)], input=json.dumps({"tool_name": tool, "tool_input": inp}),
                       capture_output=True, text=True)
    return r.stdout.strip()


def test_hook_allows_vision_only_for_batch_screenshots(tmp_path):
    (tmp_path / "files" / "1").mkdir(parents=True)
    shot = str(tmp_path / "files" / "1" / "shot-1.png")
    Path(shot).write_bytes(b"png")
    assert _hook(tmp_path, "vision_analyze", {"image_url": shot, "question": "?"}) == ""
    assert "block" in _hook(tmp_path, "vision_analyze", {"image_url": "/etc/passwd", "question": "?"})
    assert "block" in _hook(tmp_path, "vision_analyze", {"image_url": "shot-1.png", "question": "?"})
    assert "block" in _hook(tmp_path, "vision_analyze", {"image_url": None, "question": "?"})
    assert "block" in _hook(tmp_path, "web_search", {"query": "x"})
    assert "block" in _hook(tmp_path, "terminal", {"command": "env"})


# --------------------------------------------------------------------------- navigation budget

def test_nav_cap_and_pacing(tmp_path, monkeypatch):
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="navs", queue={"10.1080": 1},
                             agent={"max_navigations_per_paper": 3, "nav_gap_s": 0.2})
    st = _run(f, pr)
    log = _ego_log(tmp_path)
    opens = [e for e in log if e["action"] == "open"]
    assert len(opens) == 3, st                                   # the 4th open is refused by the nav cap
    assert all(opens[i + 1]["ts"] - opens[i]["ts"] >= 0.15 for i in range(len(opens) - 1))  # paced
    assert st["papers"][0]["status"] == "ok", st["papers"]       # the first three opens saved the file


def test_type_is_whitelisted_institution_names_only(tmp_path, monkeypatch):
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="type", queue={"10.1080": 1},
                             agent={"institution_names": ["Nanjing University"]})
    st = _run(f, pr)
    log = _ego_log(tmp_path)
    types = [e for e in log if e["action"] == "type"]
    assert types and types[0]["arg"] == "4 Nanjing University"   # whitelisted: the action script ran
    assert not [e for e in log if e["action"] == "type" and "hunter2" in (e["arg"] or "")]  # refused earlier
    assert st["papers"][0]["status"] == "ok", st["papers"]


# --------------------------------------------------------------------------- hand-off

def test_handoff_flows_challenge_to_user_and_back(tmp_path, monkeypatch):
    notified = _fake_osascript(tmp_path, monkeypatch)
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="handoff", ego_mode="handoff_wait",
                             queue={"10.1080": 1}, agent={"keep_batch_dir": True},
                             handoff={"enabled": True, "wait_s": 120})
    st = _run(f, pr)
    assert notified.exists() and "ego lite" in notified.read_text() and "验证" in notified.read_text()
    log = _ego_log(tmp_path)
    # the challenge is handed over (open sees handoff enabled), polled twice, then the paper continues
    assert [e["action"] for e in log if e["space"].endswith("-p1")] == \
        ["open", "handoff_status", "handoff_status", "open", "finish"]
    assert log[0]["handoff"] is True and log[2]["handoff"] is not True   # only the challenge may hand off
    assert st["papers"][0]["status"] == "ok", st["papers"]
    notes = _notes(led)
    assert "handoff: challenge" in notes["10.1080/paper.1"][1]        # the hand-off is noted in the ledger


def test_handoff_is_once_per_publisher(tmp_path, monkeypatch):
    _fake_osascript(tmp_path, monkeypatch)
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="handoff", ego_mode="handoff",
                             queue={"10.1080": 2}, handoff={"enabled": True})
    st = _run(f, pr)
    log = _ego_log(tmp_path)
    # both papers of the publisher hit the challenge; the second one gets no hand-off (per-publisher budget)
    assert sum(1 for e in log if e.get("handoff")) == 1
    assert all(p["status"] == "failed" for p in st["papers"])


def test_handoff_timeout_closes_the_paper(tmp_path, monkeypatch):
    _fake_osascript(tmp_path, monkeypatch)
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="handoff_timeout", ego_mode="handoff_wait",
                             queue={"10.1080": 1}, handoff={"enabled": True, "wait_s": 0.05})
    st = _run(f, pr)
    # the user never finishes: the paper is closed and the publisher cooled down
    assert st["papers"][0]["status"] == "failed"
    assert led.cooling("chrome:10.1080")


def test_handoff_stuck_challenge_cools_publisher(tmp_path, monkeypatch):
    _fake_osascript(tmp_path, monkeypatch)
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="handoff", ego_mode="handoff_stuck",
                             queue={"10.1080": 1}, handoff={"enabled": True, "wait_s": 0.05})
    st = _run(f, pr)
    # the user never clears it: the poll keeps waiting until the hand-off window expires, then the
    # paper is closed and the publisher cooled down
    assert st["papers"][0]["status"] == "failed"
    assert led.cooling("chrome:10.1080")


def test_no_handoff_when_disabled(tmp_path, monkeypatch):
    notified = _fake_osascript(tmp_path, monkeypatch)
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="work", ego_mode="challenge",
                             queue={"10.1080": 1})
    _run(f, pr)
    assert not notified.exists()                    # repo default: hand-off off -> old cooldown behaviour
    assert led.cooling("chrome:10.1080")


# --------------------------------------------------------------------------- SSO login

def test_sso_login_budget_and_no_retry_after_failure(tmp_path, monkeypatch):
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="sso", ego_mode="sso_fail",
                             queue={"10.1080": 2}, agent={"keep_batch_dir": True},
                             sso_login={"enabled": True, "domains": ["sso.example"], "max_submissions": 2})
    st = _run(f, pr)
    log = _ego_log(tmp_path)
    # the first paper submits and fails; the second paper's sso_login runs with the budget closed
    ssos = [e for e in log if e["action"] == "sso_login"]
    assert len(ssos) == 2 and ssos[0]["sso_login"] is True and ssos[1]["sso_login"] is False
    assert all(p["status"] == "failed" for p in st["papers"])
    notes = _notes(led)
    assert "sso_login host=sso.example result=fail" in notes["10.1080/paper.1"][1]


def test_sso_login_success_marks_batch_logged_in(tmp_path, monkeypatch):
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="sso", ego_mode="sso_ok",
                             queue={"10.1080": 2}, agent={"keep_batch_dir": True},
                             sso_login={"enabled": True, "domains": ["sso.example"]})
    st = _run(f, pr)
    log = _ego_log(tmp_path)
    ssos = [e for e in log if e["action"] == "sso_login"]
    # one successful login signs the whole batch in: the second paper's sso_login is disabled
    assert ssos[0]["sso_login"] is True and ssos[1]["sso_login"] is False
    notes = _notes(led)
    assert "sso_login host=pub.example result=ok" in notes["10.1080/paper.1"][1]


# --------------------------------------------------------------------------- screenshots / domains

def test_screenshot_is_viewable_but_never_a_deliverable(tmp_path, monkeypatch):
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, hermes_mode="shot", ego_mode="shot",
                             queue={"10.1080": 1}, agent={"keep_batch_dir": True})
    st = _run(f, pr)
    assert st["papers"][0]["status"] == "failed"    # no validated file: a screenshot is not the paper's text
    ps = json.loads((_bd(f) / "papers" / "1.json").read_text())
    assert ps.get("shots") and Path(ps["shots"][0]).is_file()
    assert not ps.get("accepted")


def test_allowed_domains_cover_sso_and_prompt_lists_actions(tmp_path, monkeypatch):
    f, led, pr, _ = mk_agent(tmp_path, monkeypatch, agent={"keep_batch_dir": True},
                             sso={"domains": ["nju.edu.cn", "carsi.edu.cn"]},
                             handoff={"enabled": True}, sso_login={"enabled": True, "domains": ["authserver.nju.edu.cn"]})
    _run(f, pr)
    st = json.loads((_bd(f) / "batch.json").read_text())
    for d in ("nju.edu.cn", "carsi.edu.cn"):
        assert any(d in p["allow"] for p in st["papers"].values())
    assert st["handoff_enabled"] and st["sso_login_enabled"] and st["sso_domains"] == ["authserver.nju.edu.cn"]
    prompt = (_bd(f) / "prompt.md").read_text()
    for s in ("click_xy", "hover", "screenshot", "tabs", "switch LABEL", "wait_for", "sso_login",
              "handoff_status", "25 tool calls", "12 page loads"):
        assert s in prompt, s
