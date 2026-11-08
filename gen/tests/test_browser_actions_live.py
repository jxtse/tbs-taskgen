"""Live tests for the reviewed action script browser_agent_action.mjs: a real ego-browser (Ego Lite)
against LOCAL pages only (127.0.0.1 / *.localhost; no real publisher is ever contacted).

Skipped when ego-browser is not installed (e.g. CI).  Each test uses its own uniquely named TaskSpace
(`tbsgen-test-<pid>-<n>`), never one of the production spaces.  The hand-off flow is exercised with
takeOverTaskSpace standing in for the user finishing and releasing control.
"""
import http.server
import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from tbsgen import browser_agent

ACTION = browser_agent.ACTION.read_text()
EGO = shutil.which("ego-browser")
pytestmark = pytest.mark.skipif(not EGO, reason="ego-browser not installed (local-only live tests)")

# downloads / screenshots of the action script go to a temporary directory, never into the source tree
FILES = Path(tempfile.mkdtemp(prefix="tbsgen-live-"))


@pytest.fixture(scope="module", autouse=True)
def _files_cleanup():
    yield
    shutil.rmtree(FILES, ignore_errors=True)


LANDING = """<!doctype html><html><head><title>Test Landing</title></head><body>
<h1>A Test Paper</h1>
<div id="cookie" style="position:fixed;bottom:0;left:0;right:0;background:#ccf;padding:8px">
  We use cookies <button id="accept">Accept</button> <button id="reject">Reject</button>
</div>
<div class="menu"><button id="dl-btn">Download</button>
  <ul id="dl-menu" style="display:none;list-style:none;padding:0">
    <li><a id="dl-pdf" href="/pdf" download>Download PDF</a></li>
    <li><a id="dl-suppl" href="/suppl">Supplementary material</a></li>
  </ul>
</div>
<a href="/pdf" target="_blank" id="newtab">Open viewer in new tab</a>
<a href="https://denied.example/x" id="offsite">Off-site link</a>
<iframe src="/frame" width="300" height="120"></iframe>
<iframe src="http://127.0.0.1:{port2}/xframe" width="300" height="120"></iframe>
<select id="inst"><option value="">Select institution</option>
<option value="nju">Nanjing University</option><option value="pku">Peking University</option></select>
<script>
document.getElementById('dl-btn').addEventListener('mouseenter', function() {{
  document.getElementById('dl-menu').style.display = 'block';
}});
document.getElementById('accept').addEventListener('click', function() {{
  document.getElementById('cookie').remove();
}});
window.addEventListener('message', function(e) {{ if (e.data === 'x-clicked') document.title = 'X Clicked'; }});
</script>
</body></html>"""
FRAME = """<!doctype html><html><head><title>Frame</title></head><body>
<button id="frame-btn" onclick="parent.location='/other'">Frame Other Page</button>
<a href="/pdf">Frame PDF link</a></body></html>"""
XFRAME = """<!doctype html><html><head><title>XFrame</title></head><body>
<button id="xbtn" onclick="parent.postMessage('x-clicked','*')">Cross Frame Button</button></body></html>"""
PDF = b"%PDF-1.4\n%fake pdf for local action tests\n" + b"x" * 2000
OTHER = b"<html><head><title>Other Page</title></head><body><p>other page</p></body></html>"
WAYF = """<!doctype html><html><head><title>Find your institution</title></head><body>
<h1>Where are you from?</h1>
<input id="search" type="search" placeholder="Search for your institution" aria-label="Search for your institution">
<select id="instsel"><option value="">-- select --</option><option value="nju">Nanjing University</option></select>
<button id="go">Continue</button></body></html>"""
# four SSO login variants: prefilled (autofill stand-in), empty, failing (form posts back to itself),
# and prefilled with a captcha widget
SSO = """<!doctype html><html><head><title>Institution Login</title></head><body>
<h1>统一身份认证</h1>
<form method="post" action="{action}">
<input id="username" name="username" type="text" value="{user}" autocomplete="username" placeholder="用户名">
<input id="password" name="password" type="password" value="{pw}" autocomplete="current-password">
{extra}<button type="submit" id="login">登录 / Login</button>
</form></body></html>"""
JUMP = """<!doctype html><html><head><title>Jump</title></head><body>
<select id="jumpsel" onchange="location.href='https://denied.example/' + this.value">
<option value="">-- select --</option><option value="nju">Nanjing University</option></select></body></html>"""
CHALLENGE = """<!doctype html><html><head><title>Just a moment...</title></head><body>
<div class="cf-turnstile" data-sitekey="x"></div>Checking your browser before accessing.</body></html>"""


def _pages(port, port2):
    return {
        "/landing": ("text/html", LANDING.format(port2=port2).encode()),
        "/frame": ("text/html", FRAME.encode()),
        "/pdf": ("application/pdf", PDF),
        "/other": ("text/html", OTHER),
        "/wayf": ("text/html", WAYF.encode()),
        # the login form posts to the SSO host itself (as a real institution login server does)
        "/sso": ("text/html", SSO.format(action="/logged", user="u", pw="p", extra="").encode()),
        "/sso-empty": ("text/html", SSO.format(action="/logged", user="", pw="", extra="").encode()),
        # a form on the SSO host that would post the credentials to another host: never submitted
        "/sso-evil": ("text/html", SSO.format(action=f"http://127.0.0.1:{port}/logged", user="u", pw="p",
                                              extra="").encode()),
        # a WAYF select whose change handler jumps off the allowed domains
        "/jump": ("text/html", JUMP.encode()),
        # login forms that must never be submitted by sso_login: GET method, <base href> to another host
        "/sso-get": ("text/html", SSO.replace('method="post"', 'method="get"').format(action="/logged", user="u", pw="p", extra="").encode()),
        "/sso-base": ("text/html", SSO.replace("<h1>", f'<base href="http://127.0.0.1:{port}/"><h1>').format(
            action="logged", user="u", pw="p", extra="").encode()),
        # a password field associated with an outside form by form="f"; its submit button outside the form
        "/form-attr": ("text/html", b'<html><head><title>Form Attr</title></head><body><form id="f" method="post" action="/logged"></form>'
                                    b'<input type="password" form="f" id="pw"><button form="f" id="go">Continue</button></body></html>'),
        # a plain-text verification button, and a same-origin frame with a new-tab link
        "/verify-btn": ("text/html", b'<html><head><title>Article</title></head><body><button id="b1">Verify you are human</button></body></html>'),
        "/frame-blank": ("text/html", b'<html><head><title>Frame Blank</title></head><body><iframe src="/blank-inner" width="400" height="200"></iframe></body></html>'),
        "/blank-inner": ("text/html", b'<html><body><a href="/other" target="_blank" id="nb">Open Other</a></body></html>'),
        # a login form inside a same-origin iframe
        "/frame-login": ("text/html", b'<html><head><title>Frame Login</title></head><body><h1>x</h1>'
                                      b'<iframe src="/sso" width="400" height="300"></iframe></body></html>'),
        "/sso-fail": ("text/html", SSO.format(action="/sso-fail", user="u", pw="p", extra="").encode()),
        "/sso-captcha": ("text/html", SSO.format(action="/logged", user="u", pw="p",
                                                 extra='<div class="cf-turnstile" data-sitekey="x"></div>').encode()),
        "/logged": ("text/html", b"<html><head><title>Logged In</title></head><body>ok</body></html>"),
        "/challenge": ("text/html", CHALLENGE.encode()),
    }


def _serve(pages):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _x(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in pages:
                ct, body = pages[p]
                self._x(body, ct)
            else:
                self._x(b"not found", "text/html", 404)

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            self.rfile.read(n)
            p = self.path.split("?")[0]
            if p in pages and p.startswith("/sso"):        # a failed login shows the login form again
                self._x(pages[p][1], "text/html")
            else:
                self._x(b"<html><head><title>Logged In</title></head><body>ok</body></html>", "text/html")

        def log_message(self, *a):
            pass
    return H


@pytest.fixture(scope="module")
def origins():
    """Two local origins: the second one makes the iframe on the landing page cross-origin."""
    srv2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _serve({"/xframe": ("text/html", XFRAME.encode())}))
    port2 = srv2.server_address[1]
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    pages: dict = {}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _serve(pages))
    pages.update(_pages(srv.server_address[1], port2))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # *.localhost resolves to the loopback on macOS: a second hostname on the same server
    yield (f"http://127.0.0.1:{srv.server_address[1]}", f"http://127.0.0.1:{port2}",
           f"http://sso.localhost:{srv.server_address[1]}")
    srv.shutdown()
    srv2.shutdown()


def _run(space, action, arg="", *, allow=("127.0.0.1", "localhost"), sso_domains=("sso.localhost",),
         handoff=True, sso_login=True, type_allow=("Nanjing University",), handoff_state=None,
         look_url=None, look_scope="", page_label="", challenge_wait_ms=2500, **over):
    inp = dict(space=space, action=action, arg=arg, allow=list(allow), files_dir=str(FILES),
               look_url=look_url, look_scope=look_scope, page_label=page_label,
               nav_timeout_ms=20000, download_timeout_ms=30000, download_wait_ms=3000,
               challenge_wait_ms=challenge_wait_ms,
               handoff={"enabled": handoff}, sso={"login": sso_login, "domains": list(sso_domains),
                                                  "wait_ms": 5000, "local_test": True},
               type_allow=list(type_allow), handoff_state=handoff_state or {"active": False, "deadline_ms": 0})
    inp.update(over)
    # ego-browser nodejs drops the output of a stdin script whose space is with the user; a script
    # given as an argument prints even then, so always pass it as an argument
    def once():
        p = subprocess.run([EGO, "nodejs", "const INPUT = " + json.dumps(inp) + ";\n" + ACTION],
                           capture_output=True, text=True, timeout=150)
        for line in reversed((p.stdout + "\n" + p.stderr).splitlines()):
            if line.startswith("TBSGEN_RESULT "):
                return json.loads(line[len("TBSGEN_RESULT "):])
        return {"status": "no-result", "stdout": p.stdout[-300:], "stderr": p.stderr[-300:]}
    # ego drops a script's output now and then when the space's control is with the user (a
    # hand-off); a second run of the same read-only call then prints. Retry once.
    r = once()
    return r if r["status"] != "no-result" or "paused" in (r.get("stdout") + r.get("stderr")) else once()


def _take_over(space, url):
    """Stand in for the user: finish in the ego window, then release control back to the agent."""
    p = subprocess.run([EGO, "nodejs",
                        f"const t = await takeOverTaskSpace({json.dumps(space)});\n"
                        f"const pg = (await t.pages())[0];\n"
                        f"await pg.goto({json.dumps(url)}, {{timeout: 15000, waitUntil: 'domcontentloaded'}});\n"
                        f"console.log('TAKEOVER_OK');\nprocess.exit(0);"],
                       capture_output=True, text=True, timeout=60)
    return "TAKEOVER_OK" in (p.stdout + p.stderr)


def _finish(space):
    return _run(space, "finish", sso_login=False)


def _item(res, text, href=None):
    for it in (res.get("page") or {}).get("items") or []:
        if text in (it.get("text") or "") and (href is None or href in (it.get("href") or "")):
            return it
    return None


# --------------------------------------------------------------------------- pages

def test_look_cookie_banner_dropdown_iframes(origins):
    sp = "tbsgen-test-look"
    try:
        r = _run(sp, "open", origins[0] + "/landing")
        assert r["status"] == "ok", r
        r = _run(sp, "look")
        assert r["status"] == "ok" and r["page"]["items"], r
        assert _item(r, "Accept") and _item(r, "Reject")                  # cookie banner buttons
        assert _item(r, "Download PDF") and _item(r, "Supplementary material")   # hidden dropdown items
        assert _item(r, "Frame PDF link")                                 # same-origin iframe item
        x = _item(r, "Cross Frame Button")                                # cross-origin iframe (snapshot ref)
        assert x and x.get("src") == "snap", r["page"]["items"]
        off = _item(r, "Off-site link")
        assert off and off["allowed"] is False
        # click the cookie banner away
        r = _run(sp, "click", str(_item(r, "Accept")["i"]), look_url=r["page"]["url"])
        assert r["status"] == "ok", r
        r = _run(sp, "look")
        assert _item(r, "We use cookies") is None                         # the banner is gone
        # click the hidden dropdown item directly (hidden JS menus: programmatic click)
        r = _run(sp, "click", str(_item(r, "Download PDF")["i"]), look_url=r["page"]["url"])
        assert r["status"] == "ok" and r.get("files"), r
        assert Path(r["files"][0]).read_bytes()[:5] == b"%PDF-"          # the PDF was downloaded
    finally:
        _finish(sp)


def test_hover_opens_the_menu(origins):
    sp = f"tbsgen-test-hover"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        r = _run(sp, "hover", str(_item(r, "Download")["i"]), look_url=r["page"]["url"])
        assert r["status"] == "ok" and r["acted"]["kind"] == "hover", r
        # the menu is open: a fresh look sees the items as visible snapshot items
        r = _run(sp, "look")
        it = _item(r, "Download PDF")
        assert it and not it.get("hidden"), r["page"]["items"]
    finally:
        _finish(sp)


def test_cross_origin_iframe_click_by_snapshot_ref(origins):
    sp = "tbsgen-test-xframe"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        assert _run(sp, "click", str(_item(r, "Cross Frame Button")["i"]),
                    look_url=r["page"]["url"])["status"] == "ok"
        r = _run(sp, "look")
        assert r["page"]["title"] == "X Clicked", r                        # the iframe button really ran
    finally:
        _finish(sp)


def test_new_tab_switch_and_save(origins):
    sp = "tbsgen-test-tabs"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        r = _run(sp, "click", str(_item(r, "Open viewer in new tab")["i"]), look_url=r["page"]["url"])
        assert r["status"] == "ok", r
        assert r["tabs"] and r["tabs"][0]["label"] == "p2" and r["tabs"][0]["allowed"], r
        r = _run(sp, "switch", "p2", page_label="p2")
        assert r["status"] == "ok" and r["page"]["pdf"], r
        r = _run(sp, "save_pdf", page_label="p2")
        assert r["status"] == "ok" and r["files"], r
        r = _run(sp, "switch", "p1", page_label="p2")
        assert r["status"] == "ok" and "landing" in r["page"]["url"], r
        r = _run(sp, "switch", "p9", page_label="p1")
        assert r["status"] == "refused", r                                 # no such page
    finally:
        _finish(sp)


def test_scroll_screenshot_and_wait_for(origins):
    sp = "tbsgen-test-misc"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "screenshot")
        assert r["status"] == "ok" and r["shot"] and Path(r["shot"]).is_file(), r
        r = _run(sp, "scroll", "down 400")
        assert r["status"] == "ok", r
        r = _run(sp, "wait_for", "url landing")
        assert r["status"] == "ok" and r["waited"]["ok"], r
        r = _run(sp, "wait_for", "sel #dl-btn")
        assert r["status"] == "ok" and r["waited"]["ok"], r
        r = _run(sp, "wait_for", "sel #no-such-thing 1")
        assert r["status"] == "ok" and not r["waited"]["ok"], r
    finally:
        _finish(sp)


def test_wayf_type_and_select(origins):
    sp = "tbsgen-test-wayf"
    try:
        assert _run(sp, "open", origins[0] + "/wayf")["status"] == "ok"
        r = _run(sp, "look")
        assert _item(r, "Search for your institution"), r["page"]["items"]
        k_search = _item(r, "Search for your institution")["i"]
        k_sel = _item(r, "instsel")["i"]
        look_url = r["page"]["url"]
        r = _run(sp, "type", f"{k_search} Nanjing University", look_url=look_url)
        assert r["status"] == "ok", r
        r = _run(sp, "type", f"{k_search} hunter2", look_url=look_url)
        assert r["status"] == "refused" and "whitelisted" in r["error"], r   # only institution names
        r = _run(sp, "select", f"{k_sel} Nanjing University", look_url=look_url)
        assert r["status"] == "ok" and "Nanjing University" in r["acted"]["text"], r
    finally:
        _finish(sp)


def test_denied_paths(origins):
    sp = "tbsgen-test-denied"
    try:
        assert _run(sp, "open", origins[0] + "/landing", handoff=False)["status"] == "ok"
        r = _run(sp, "open", "https://denied.example/x", handoff=False)
        assert r["status"] == "refused" and "allowed domains" in r["error"], r
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Off-site link")["i"]), look_url=r["page"]["url"], handoff=False)
        assert r["status"] == "refused", r                                  # off-site link
        # login form: its fields and the submit button are never touched, and the page itself hands off
        # (login form outside the SSO domains). Use sso.localhost with domains that exclude it.
        assert _run(sp, "open", origins[2] + "/sso",
                    sso_domains=("sso.elsewhere",))["status"] == "handoff"
        assert _take_over(sp, origins[2] + "/sso")
        # after the takeover the space is agent-controlled again: `look` reports the login page (outside
        # the SSO domains it would hand off, so give the SSO domains back)
        r = _run(sp, "look")
        assert r["status"] == "ok" and r["page"]["login_form"] is True, r
        items = r["page"]["items"]
        user = next((it for it in items if it.get("ftype") == "text"), None)
        assert user and "value" not in user                    # fields list types, never values
        look_url = r["page"]["url"]
        r = _run(sp, "type", f"{user['i']} Nanjing University", look_url=look_url)
        assert r["status"] == "refused", r                                  # credential-shaped field
        r = _run(sp, "look")
        r = _run(sp, "click", str(_item(r, "Login")["i"]), look_url=r["page"]["url"])
        assert r["status"] == "refused" and "login form" in r["error"], r   # submit button
        r = _run(sp, "screenshot")
        assert r["status"] == "refused" and "credential" in r["error"], r   # no screenshots of login pages
        # sso_login outside the configured SSO domains is refused
        r = _run(sp, "sso_login", sso_domains=("sso.elsewhere",))
        assert r["status"] == "refused", r
        # hover on the (autofilled) user field never returns its value
        r = _run(sp, "look")
        user = next(it for it in r["page"]["items"] if it.get("ftype") == "text")
        r = _run(sp, "hover", str(user["i"]), look_url=r["page"]["url"])
        assert '"u"' not in json.dumps(r.get("acted")), r
        # coordinate clicks on login pages are refused
        r = _run(sp, "click_xy", "10 10")
        assert r["status"] == "refused" and "login page" in r["error"], r
    finally:
        _finish(sp)


def test_select_that_navigates_off_domain_is_reset(origins):
    sp = "tbsgen-test-jump"
    try:
        assert _run(sp, "open", origins[0] + "/jump")["status"] == "ok"
        r = _run(sp, "look")
        k = _item(r, "jumpsel")["i"]
        r = _run(sp, "select", f"{k} Nanjing University", look_url=r["page"]["url"])
        assert r["status"] == "ok" and not r["page"]["url"].startswith("https://denied"), r
        r = _run(sp, "look")
        assert "denied.example" not in r["page"]["url"], r
    finally:
        _finish(sp)


def test_screenshot_refused_with_a_login_form_in_an_iframe(origins):
    sp = "tbsgen-test-frame-login"
    try:
        assert _run(sp, "open", origins[0] + "/frame-login", handoff=False)["status"] == "ok"
        r = _run(sp, "screenshot", handoff=False)
        assert r["status"] == "refused" and "credential" in r["error"], r
    finally:
        _finish(sp)


# --------------------------------------------------------------------------- hand-off

def test_challenge_hands_off_to_the_user_and_back(origins):
    sp = "tbsgen-test-handoff"
    try:
        r = _run(sp, "open", origins[0] + "/challenge", challenge_wait_ms=1500)
        assert r["status"] == "handoff" and r["handoff"]["host"], r         # the page goes to the user
        # while the user has control the space drops every script's output (ego behaviour), so the
        # wrapper refuses all actions but handoff_status before the script even runs
        r = _run(sp, "look")
        assert r["status"] == "no-result", r
        # a poll before the user finished: takes the space back, still sees the challenge, hands it
        # back to the user and reports `waiting`
        r = _run(sp, "handoff_status", handoff_state={"active": True, "deadline_ms": int(time.time() * 1000) + 60000})
        assert r["status"] == "ok" and r["handoff_check"] == "waiting", r
        # the user finishes: the page is navigated away from the challenge (takeover stands in)
        assert _take_over(sp, origins[0] + "/landing")
        r = _run(sp, "handoff_status", handoff_state={"active": True, "deadline_ms": int(time.time() * 1000) + 60000})
        assert r["status"] == "ok" and r["handoff_check"] == "cleared" and r["page"]["url"].endswith("/landing"), r
        # the paper continues
        r = _run(sp, "look")
        assert r["status"] == "ok", r
    finally:
        _finish(sp)


def test_challenge_that_does_not_clear_fails_the_handoff(origins):
    sp = "tbsgen-test-handoff-stuck"
    try:
        r = _run(sp, "open", origins[0] + "/challenge", challenge_wait_ms=1500)
        assert r["status"] == "handoff", r
        # the user cannot clear it: every poll still sees the challenge and keeps waiting; once the
        # hand-off window expires the wrapper closes the paper and cools the publisher (offline test)
        r = _run(sp, "handoff_status", handoff_state={"active": True, "deadline_ms": int(time.time() * 1000) + 60000},
                 challenge_wait_ms=1500)
        assert r["status"] == "ok" and r["handoff_check"] == "waiting", r
    finally:
        _finish(sp)


# --------------------------------------------------------------------------- SSO login

def test_sso_login_autofill_success_and_failures(origins):
    # 1. autofilled form submits and leaves the SSO domain
    sp = "tbsgen-test-sso-ok"
    try:
        assert _run(sp, "open", origins[2] + "/sso")["status"] == "ok"
        r = _run(sp, "sso_login", sso_login=True)
        assert r["status"] == "ok" and r["sso"] == "ok" and "/logged" in r["page"]["url"], r
        assert r["sso_submitted"] is True and r["sso_host"] == "sso.localhost", r
    finally:
        _finish(sp)
    # 2. no autofill: handed to the user instead (never typed into)
    sp = "tbsgen-test-sso-empty"
    try:
        assert _run(sp, "open", origins[2] + "/sso-empty")["status"] == "ok"
        r = _run(sp, "sso_login", sso_login=True)
        assert r["status"] == "handoff" and "autofill" in r["handoff"]["reason"], r
    finally:
        _finish(sp)
    # 3. captcha on the login page: the challenge itself hands the page to the user
    sp = "tbsgen-test-sso-captcha"
    try:
        assert _run(sp, "open", origins[2] + "/sso-captcha")["status"] == "handoff"
        assert _take_over(sp, origins[2] + "/sso-captcha")
        r = _run(sp, "sso_login", sso_login=True)
        assert r["status"] == "handoff", r
    finally:
        _finish(sp)
    # 3b. no autofill and no hand-off budget: reported as login_required, nothing typed or handed over
    sp = "tbsgen-test-sso-empty-nohandoff"
    try:
        assert _run(sp, "open", origins[2] + "/sso-empty", handoff=False)["status"] == "ok"
        r = _run(sp, "sso_login", sso_login=True, handoff=False)
        assert r["status"] == "ok" and r.get("login_required") and r["sso_submitted"] is False, r
    finally:
        _finish(sp)
    # 3c. a form that posts outside the SSO login domains is never submitted
    sp = "tbsgen-test-sso-evil"
    try:
        assert _run(sp, "open", origins[2] + "/sso-evil")["status"] == "ok"
        r = _run(sp, "sso_login", sso_login=True)
        assert r["status"] == "refused" and "outside the SSO" in r["error"] and r["sso_submitted"] is False, r
    finally:
        _finish(sp)
    # 4. failed login (the form posts back to itself): fail, and the values never leak into the result
    sp = "tbsgen-test-sso-fail"
    try:
        assert _run(sp, "open", origins[2] + "/sso-fail")["status"] == "ok"
        r = _run(sp, "sso_login", sso_login=True)
        assert r["status"] == "ok" and r["sso"] == "fail", r
        assert "u" != json.dumps(r)[:60] and "p" not in (r.get("page") or {}).get("text", "")[:5]
        # 5. sso_login outside the configured SSO domains is refused
        assert _run(sp, "open", origins[2] + "/sso")["status"] == "ok"
        r = _run(sp, "sso_login", sso_login=True, sso_domains=("sso.elsewhere",))
        assert r["status"] == "refused", r
        # 6. disabled: refused
        r = _run(sp, "sso_login", sso_login=False)
        assert r["status"] == "refused", r
    finally:
        _finish(sp)


def test_look_on_login_pages_hands_off_only_outside_sso_domains(origins):
    sp = "tbsgen-test-login-look"
    try:
        # on the configured SSO domain: reported (the agent then calls sso_login), not handed off
        assert _run(sp, "open", origins[2] + "/sso")["status"] == "ok"
        r = _run(sp, "look")
        assert r["status"] == "ok" and r["page"]["login_form"] is True, r
        # outside them: the login page hands the space to the user already at `open`
        r = _run(sp, "open", origins[0] + "/sso", sso_domains=("sso.elsewhere",))
        assert r["status"] == "handoff" and "login" in r["handoff"]["reason"], r
    finally:
        _finish(sp)


def test_round2_click_and_sso_guards(origins):
    sp = "tbsgen-test-r2-guards"
    try:
        # a password field tied to a form by form="f": its outside submit button is never clicked
        assert _run(sp, "open", origins[0] + "/form-attr", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Continue")["i"]), look_url=r["page"]["url"], handoff=False)
        assert r["status"] == "refused" and "login form" in r["error"], r
        # a plain "Verify you are human" button is never clicked
        assert _run(sp, "open", origins[0] + "/verify-btn", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Verify you are human")["i"]), look_url=r["page"]["url"], handoff=False)
        assert r["status"] == "refused" and "verification" in r["error"], r
        # a new-tab link inside a same-origin frame opens its (checked) target in this page
        assert _run(sp, "open", origins[0] + "/frame-blank", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Open Other")["i"]), look_url=r["page"]["url"], handoff=False)
        assert r["status"] == "ok" and r["page"]["url"].endswith("/other"), r
    finally:
        _finish(sp)
    for path, why in (("/sso-get", "POST"), ("/sso-base", "outside the SSO")):
        sp = "tbsgen-test-r2-sso" + path.replace("/", "-")
        try:
            assert _run(sp, "open", origins[2] + path)["status"] == "ok"
            r = _run(sp, "sso_login", sso_login=True)
            assert r["status"] == "refused" and why in r["error"] and r["sso_submitted"] is False, r
            assert r["sso_stage"] == "pre", r
        finally:
            _finish(sp)
