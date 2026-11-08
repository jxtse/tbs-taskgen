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
        # round 3: a form posting off-site through a plain button; a login form in a shadow root; a page
        # that redirects off-site later; a second login step (verification code) after the password
        "/offsite-form": ("text/html", b'<html><head><title>Offsite Form</title></head><body><form method="post" '
                                       b'action="https://denied.example/x"><button id="dl">Download</button></form></body></html>'),
        "/shadow-login": ("text/html", b'<html><head><title>Shadow</title></head><body><div id="h"></div><script>'
                                       b'const r=document.getElementById("h").attachShadow({mode:"open"});'
                                       b'r.innerHTML=\'<form><input id="u" autocomplete="username"><input type="password"></form>\';'
                                       b'</script><a href="/other">Other</a></body></html>'),
        "/late-redirect": ("text/html", b'<html><head><title>Late</title></head><body>wait<script>'
                                        b'setTimeout(()=>{location.href="https://denied.example/late"},1500)</script></body></html>'),
        "/sso-2fa": ("text/html", SSO.format(action="/code-step", user="u", pw="p", extra="").encode()),
        "/code-step": ("text/html", b'<html><head><title>Verify</title></head><body><form method="post" action="/logged">'
                                    b'<input name="otp_code" autocomplete="one-time-code"> Enter the verification code'
                                    b'<button>OK</button></form></body></html>'),
        # a login form inside a same-origin iframe
        "/frame-login": ("text/html", b'<html><head><title>Frame Login</title></head><body><h1>x</h1><a href="/other">Other</a>'
                                      b'<iframe src="/sso" width="400" height="300"></iframe></body></html>'),
        "/sso-fail": ("text/html", SSO.format(action="/sso-fail", user="u", pw="p", extra="").encode()),
        "/sso-captcha": ("text/html", SSO.format(action="/logged", user="u", pw="p",
                                                 extra='<div class="cf-turnstile" data-sitekey="x"></div>').encode()),
        "/logged": ("text/html", b"<html><head><title>Logged In</title></head><body>ok</body></html>"),
        "/challenge": ("text/html", CHALLENGE.encode()),
        # round 4: a cross-origin frame on a host outside the allowed domains; a login page with a new-tab
        # link; a span inside an outside-form button posting off-site; a stand-alone verification-code page
        "/xframe-other": ("text/html", f'<html><head><title>XOther</title></head><body><iframe title="viewer" '
                                       f'src="http://localhost:{port2}/xframe" width="300" height="120"></iframe>'
                                       f'</body></html>'.encode()),
        "/login-blank": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other" target="_blank" '
                                      b'id="nb">Open Other</a><form method="post" action="/logged"><input name="username">'
                                      b'<input type="password" name="password"><button>Sign in</button></form></body></html>'),
        "/form-span": ("text/html", b'<html><head><title>Form Span</title></head><body><form id="f" method="post" '
                                    b'action="https://denied.example/x"></form><button form="f"><span role="button">'
                                    b'Get PDF</span></button></body></html>'),
        # round 5: identifier-first (e-mail) login; a span inside an off-site link; a link with a long URL
        "/email-login": ("text/html", b'<html><head><title>Sign in</title></head><body><a href="/other">Other</a>'
                                      b'<form method="post" action="/logged"><input type="email" name="email" autocomplete="email">'
                                      b'<button>Sign in</button></form></body></html>'),
        "/newsletter": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                     b'<form method="post" action="/logged"><input type="email" name="email">'
                                     b'<button>Subscribe to our newsletter</button></form></body></html>'),
        "/span-link": ("text/html", b'<html><head><title>Span Link</title></head><body><a href="https://denied.example/x">'
                                    b'<span role="button">Get PDF</span></a></body></html>'),
        "/long-link": ("text/html", ('<html><head><title>Long</title></head><body><a href="https://alice:' + "S" * 300
                                     + '@127.0.0.1/x">Long link</a></body></html>').encode()),
        # round 6: e-mail-only login form ("Continue") whose sign-in title is outside the form; a same-origin
        # frame with a hover target
        "/email-continue": ("text/html", b'<html><head><title>Example Publisher</title></head><body><h2>Sign in</h2>'
                                         b'<form method="post" action="/logged"><input type="email" name="identifier">'
                                         b'<button>Continue</button></form></body></html>'),
        # round 7: login words on a page with no login fields (title, a form button, a continue-with button,
        # a login form action, the URL); a page that redirects between look and click; a tel field; a clean
        # page with a header "Sign in" link (no field: interaction allowed)
        "/r7-title": ("text/html", b'<html><head><title>Sign in - Example</title></head><body><a href="/other">Other</a></body></html>'),
        "/r7-btn": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                 b'<form method="post" action="/x"><button>Log in</button></form></body></html>'),
        "/r7-cw": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                b'<button>Continue with Google</button></body></html>'),
        "/r7-action": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                    b'<form method="post" action="/account/login"><input type="checkbox" name="r"><button>Go</button></form></body></html>'),
        "/login/r7": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a></body></html>'),
        "/r7-tel": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                 b'<input type="tel" name="x"></body></html>'),
        "/r7-loose": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                   b'<input type="text" name="q"><button>Sign in</button></body></html>'),
        "/r7-header": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                    b'<a href="/signin-page">Sign in</a><button>Sign in</button></body></html>'),
        # the page goes to a login page 1.5 s after load (a late redirect between `look` and `click`)
        "/r7-late": ("text/html", b'<html><head><title>Late</title></head><body><a href="/other">Other</a>'
                                  b'<script>setTimeout(function(){location.href="/email-login"},1500)</script></body></html>'),
        "/frame-hover": ("text/html", b'<html><head><title>Frame Hover</title></head><body><h1>x</h1>'
                                      b'<iframe src="/frame" width="400" height="200" style="margin-left:300px"></iframe></body></html>'),
        "/wayf-form": ("text/html", b'<html><head><title>Find your institution</title></head><body><form method="post" '
                                    b'action="/logged"><input type="search" name="q" aria-label="Search for your institution">'
                                    b'<button>Find</button></form></body></html>'),
        "/code-only": ("text/html", b'<html><head><title>Check</title></head><body><a href="/other">Other</a>'
                                    b'<form method="post" action="/logged"><input name="code"><button>OK</button></form>'
                                    b'</body></html>'),
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
            if p == "/code-step":
                self._x(pages[p][1], "text/html")
            elif p in pages and p.startswith("/sso"):        # a failed login shows the login form again
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
         look_key=None, page_label="", challenge_wait_ms=2500, **over):
    inp = dict(space=space, action=action, arg=arg, allow=list(allow), files_dir=str(FILES),
               look_key=look_key, page_label=page_label,
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
        # a cross-origin iframe is ONE `frame` item (its URL); its contents are never read (no page snapshot)
        assert _item(r, "Cross Frame Button") is None, r["page"]["items"]
        x = next((it for it in r["page"]["items"] if it["kind"] == "frame"), None)
        assert x and x["href"].endswith("/xframe"), r["page"]["items"]
        off = _item(r, "Off-site link")
        assert off and off["allowed"] is False
        # click the cookie banner away
        r = _run(sp, "click", str(_item(r, "Accept")["i"]), look_key=r["look_key"])
        assert r["status"] == "ok", r
        r = _run(sp, "look")
        assert _item(r, "We use cookies") is None                         # the banner is gone
        # click the hidden dropdown item directly (hidden JS menus: programmatic click)
        r = _run(sp, "click", str(_item(r, "Download PDF")["i"]), look_key=r["look_key"])
        assert r["status"] == "ok" and r.get("files"), r
        assert Path(r["files"][0]).read_bytes()[:5] == b"%PDF-"          # the PDF was downloaded
    finally:
        _finish(sp)


def test_hover_opens_the_menu(origins):
    sp = f"tbsgen-test-hover"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        r = _run(sp, "hover", str(_item(r, "Download")["i"]), look_key=r["look_key"])
        assert r["status"] == "ok" and r["acted"]["kind"] == "hover", r
        # the menu is open: a fresh look sees the items as visible snapshot items
        r = _run(sp, "look")
        it = _item(r, "Download PDF")
        assert it and not it.get("hidden"), r["page"]["items"]
    finally:
        _finish(sp)


def test_cross_origin_iframe_opens_as_the_page(origins):
    sp = "tbsgen-test-xframe"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        fr = next(it for it in r["page"]["items"] if it["kind"] == "frame")
        # nothing is clicked inside the frame: `click` opens the frame's (checked) URL as the page itself
        r2 = _run(sp, "click", str(fr["i"]), look_key=r["look_key"])
        assert r2["status"] == "ok" and r2["page"]["url"].endswith("/xframe"), r2
        r = _run(sp, "look")
        assert _item(r, "Cross Frame Button"), r                           # now a page of its own
        # a frame on a host outside the allowed domains is never opened
        assert _run(sp, "open", origins[0] + "/xframe-other", allow=("127.0.0.1",))["status"] == "ok"
        r = _run(sp, "look", allow=("127.0.0.1",))
        fr = next(it for it in r["page"]["items"] if it["kind"] == "frame")
        assert fr["allowed"] is False
        r2 = _run(sp, "click", str(fr["i"]), look_key=r["look_key"], allow=("127.0.0.1",))
        assert r2["status"] == "refused" and "outside the allowed" in r2["error"], r2
    finally:
        _finish(sp)


def test_new_tab_link_opens_in_the_page_switch_and_save(origins):
    sp = "tbsgen-test-tabs"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        # a target=_blank link opens its checked target in this page (never a native click)
        r = _run(sp, "click", str(_item(r, "Open viewer in new tab")["i"]), look_key=r["look_key"])
        assert r["status"] == "ok" and r["page"]["pdf"], r
        r = _run(sp, "save_pdf")
        assert r["status"] == "ok" and r["files"], r
        r = _run(sp, "tabs")
        assert r["status"] == "ok" and r["tabs"][0]["label"] == "p1", r
        r = _run(sp, "switch", "p1")
        assert r["status"] == "ok" and r["page"]["pdf"], r
        r = _run(sp, "switch", "p9", page_label="p1")
        assert r["status"] == "refused", r                                 # no such page
    finally:
        _finish(sp)


def test_wait_for_and_removed_actions(origins):
    sp = "tbsgen-test-misc"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        for a, arg in (("screenshot", ""), ("click_xy", "10 10"), ("scroll", "down 400")):
            r = _run(sp, a, arg)
            assert r["status"] == "refused" and r["error"] == "unknown action", r     # removed actions
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
        look_key = r["look_key"]
        r = _run(sp, "type", f"{k_search} Nanjing University", look_key=look_key)
        assert r["status"] == "ok", r
        r = _run(sp, "type", f"{k_search} hunter2", look_key=look_key)
        assert r["status"] == "refused" and "whitelisted" in r["error"], r   # only institution names
        r = _run(sp, "select", f"{k_sel} Nanjing University", look_key=look_key)
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
        r = _run(sp, "click", str(_item(r, "Off-site link")["i"]), look_key=r["look_key"], handoff=False)
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
        look_key = r["look_key"]
        r = _run(sp, "type", f"{user['i']} Nanjing University", look_key=look_key)
        assert r["status"] == "refused", r                                  # credential-shaped field
        r = _run(sp, "look")
        r = _run(sp, "click", str(_item(r, "Login")["i"]), look_key=r["look_key"])
        assert r["status"] == "refused" and "login" in r["error"], r        # submit button
        # sso_login outside the configured SSO domains is refused
        r = _run(sp, "sso_login", sso_domains=("sso.elsewhere",))
        assert r["status"] == "refused", r
        # no hover on a login page either (and nothing returns a field value)
        r = _run(sp, "look")
        user = next(it for it in r["page"]["items"] if it.get("ftype") == "text")
        r = _run(sp, "hover", str(user["i"]), look_key=r["look_key"])
        assert r["status"] == "refused" and '"u"' not in json.dumps(r), r
    finally:
        _finish(sp)


def test_select_that_navigates_off_domain_is_reset(origins):
    sp = "tbsgen-test-jump"
    try:
        assert _run(sp, "open", origins[0] + "/jump")["status"] == "ok"
        r = _run(sp, "look")
        k = _item(r, "jumpsel")["i"]
        r = _run(sp, "select", f"{k} Nanjing University", look_key=r["look_key"])
        assert r["status"] == "ok" and not r["page"]["url"].startswith("https://denied"), r
        r = _run(sp, "look")
        assert "denied.example" not in r["page"]["url"], r
    finally:
        _finish(sp)


def test_no_interaction_with_a_login_form_in_an_iframe(origins):
    sp = "tbsgen-test-frame-login"
    try:
        assert _run(sp, "open", origins[0] + "/frame-login", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Other")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "login" in r["error"], r
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
        r = _run(sp, "click", str(_item(r, "Continue")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "login" in r["error"], r
        # a plain "Verify you are human" button is never clicked
        assert _run(sp, "open", origins[0] + "/verify-btn", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Verify you are human")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "verification" in r["error"], r
        # a new-tab link inside a same-origin frame opens its (checked) target in this page
        assert _run(sp, "open", origins[0] + "/frame-blank", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Open Other")["i"]), look_key=r["look_key"], handoff=False)
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


def test_round3_guards(origins):
    sp = "tbsgen-test-r3"
    try:
        # a plain button of a form that posts off-site is never clicked
        assert _run(sp, "open", origins[0] + "/offsite-form", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Download")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "outside the allowed" in r["error"], r
        # a login form in a shadow root: no interaction with the page
        assert _run(sp, "open", origins[0] + "/shadow-login", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Other")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "login" in r["error"], r
        # a late redirect off-site is reset before the next action reads anything
        assert _run(sp, "open", origins[0] + "/late-redirect", handoff=False)["status"] == "ok"
        time.sleep(3)
        r = _run(sp, "look", handoff=False)
        # (denied.example does not resolve: the browser shows its own error page, never content of that host)
        assert not r["page"]["url"].startswith(("http://denied", "https://denied")), r
    finally:
        _finish(sp)
    # a verification-code step after the password is handed to the user, not reported as logged in
    sp = "tbsgen-test-r3-2fa"
    try:
        assert _run(sp, "open", origins[2] + "/sso-2fa")["status"] == "ok"
        r = _run(sp, "sso_login", sso_login=True)
        assert r["status"] == "handoff" and r.get("sso") == "handoff" and r["sso_submitted"] is True, r
    finally:
        _finish(sp)



def test_round4_guards(origins):
    sp = "tbsgen-test-r4"
    try:
        # a new-tab link on a login page: refused like everything else there (no native click)
        assert _run(sp, "open", origins[0] + "/login-blank", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Open Other")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "login" in r["error"], r
        # a span inside <button form="f"> whose form posts off-site: the button's form is checked
        assert _run(sp, "open", origins[0] + "/form-span", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Get PDF")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "outside the allowed" in r["error"], r
        # a stand-alone verification-code field (name="code"): no interaction
        assert _run(sp, "open", origins[0] + "/code-only", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Other")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "login" in r["error"], r
        # token-like query parameters (any spelling) never come back; nor does a URL inside a message
        for q in ("?access_token=SECRET1", "?%61ccess_token=SECRET1", "?oauth_token=SECRET1", "#code=SECRET1"):
            r = _run(sp, "open", origins[0] + "/other" + q, handoff=False)
            assert r["status"] == "ok" and "SECRET1" not in json.dumps(r), r
            r = _run(sp, "tabs", handoff=False)
            assert "SECRET1" not in json.dumps(r), r
            r = _run(sp, "wait_for", "url SECRET1 1", handoff=False)     # the query is no guessing oracle
            assert r["waited"]["ok"] is False and "SECRET1" not in json.dumps(r), r
        r = _run(sp, "open", "https://denied.example/x?code=SECRET2", handoff=False)
        assert r["status"] == "refused" and "SECRET2" not in json.dumps(r), r
    finally:
        _finish(sp)
    # a verification-code page the user has not finished keeps the hand-off waiting
    sp = "tbsgen-test-r4-code"
    try:
        r = _run(sp, "open", origins[0] + "/challenge", challenge_wait_ms=1500)
        assert r["status"] == "handoff", r
        assert _take_over(sp, origins[0] + "/code-only")
        r = _run(sp, "handoff_status", handoff_state={"active": True, "deadline_ms": int(time.time() * 1000) + 60000})
        assert r["status"] == "ok" and r["handoff_check"] == "waiting", r
    finally:
        _finish(sp)



def test_round5_guards(origins):
    sp = "tbsgen-test-r5"
    try:
        # identifier-first e-mail login: a login page (no interaction); a newsletter e-mail field is not
        assert _run(sp, "open", origins[0] + "/email-login", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Other")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "login" in r["error"], r
        # round 7 (conservative): a rendered e-mail field anywhere - a newsletter box too - refuses all
        # interaction on the page; it is reported, but (no unmistakable login signal) not handed to the user
        r = _run(sp, "open", origins[0] + "/newsletter")
        assert r["status"] == "ok" and r["page"]["login_form"] is True, r
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Other")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "login" in r["error"], r
        # a span inside an off-site link: the link's target is checked
        assert _run(sp, "open", origins[0] + "/span-link", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        span = next(it for it in r["page"]["items"] if it["text"] == "Get PDF" and it["kind"] == "button")
        r = _run(sp, "click", str(span["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "outside the allowed" in r["error"], r
        # userinfo in a long URL never comes back, cut or not
        assert _run(sp, "open", origins[0] + "/long-link", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        assert "SSSSSSSS" not in json.dumps(r), r
    finally:
        _finish(sp)
    # a verification-code page outside the SSO domains is handed to the user at once
    sp = "tbsgen-test-r5-code"
    try:
        r = _run(sp, "open", origins[0] + "/code-only")
        assert r["status"] == "handoff" and r["handoff"]["kind"] == "mfa", r
    finally:
        _finish(sp)



def test_round6_guards(origins):
    sp = "tbsgen-test-r6"
    try:
        # an e-mail form whose "Sign in" heading is outside it: its Continue button is never clicked
        assert _run(sp, "open", origins[0] + "/email-continue", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Continue")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and ("e-mail" in r["error"] or "login" in r["error"]), r
        r = _run(sp, "look", handoff=False)
        assert "/logged" not in r["page"]["url"], r
        # a form with a free-text field is submitted only after this tool typed into every such field
        assert _run(sp, "open", origins[0] + "/wayf-form", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        k_find, k_q = _item(r, "Find")["i"], _item(r, "Search for your institution")["i"]
        r2 = _run(sp, "click", str(k_find), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "refused" and "did not fill" in r2["error"], r2
        assert _run(sp, "type", f"{k_q} Nanjing University", look_key=r["look_key"], handoff=False)["status"] == "ok"
        r2 = _run(sp, "click", str(k_find), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "ok" and r2["page"]["url"].endswith("/logged"), r2
        # no native hover inside a frame (frame-local coordinates)
        assert _run(sp, "open", origins[0] + "/frame-hover", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        it = next(x for x in r["page"]["items"] if "Frame Other Page" in (x["text"] or "") and x["frame"] > 0)
        r = _run(sp, "hover", str(it["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "frames" in r["error"], r
    finally:
        _finish(sp)


def test_round7_conservative_login_detection_and_atomic_checks(origins):
    sp = "tbsgen-test-r7"
    try:
        # every login signal refuses interaction (no hand-off budget here: reported, never handed over)
        for path in ("/r7-title", "/r7-btn", "/r7-cw", "/r7-action", "/login/r7", "/r7-tel", "/r7-loose",
                     "/email-continue"):
            r = _run(sp, "open", origins[0] + path, handoff=False)
            assert r["status"] == "ok" and r["page"]["login_form"] is True, (path, r)
            r = _run(sp, "look", handoff=False)
            it = _item(r, "Other") or _item(r, "Continue")
            for act in ("click", "hover"):
                r2 = _run(sp, act, str(it["i"]), look_key=r["look_key"], handoff=False)
                assert r2["status"] == "refused" and "login" in r2["error"], (path, act, r2)
        # a header "Sign in" entry point on an article without fields: still an article page
        r = _run(sp, "open", origins[0] + "/r7-header", handoff=False)
        assert r["status"] == "ok" and not r["page"]["login_form"], r
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Other")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "ok" and r["page"]["url"].endswith("/other"), r
        # an acting call needs a look_key (the page the agent looked at)
        r = _run(sp, "click", "0", handoff=False)
        assert r["status"] == "refused" and "look" in r["error"], r
        # a late redirect to a login page after `look`: the click is refused (re-checked where it acts)
        assert _run(sp, "open", origins[0] + "/r7-late", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        k, key = _item(r, "Other")["i"], r["look_key"]
        time.sleep(3)
        r = _run(sp, "click", str(k), look_key=key, handoff=False)
        assert r["status"] == "refused" and ("changed" in r["error"] or "login" in r["error"]), r
        r = _run(sp, "look", handoff=False)
        assert "/logged" not in r["page"]["url"] and "/other" not in r["page"]["url"], r
    finally:
        _finish(sp)


def test_round7_newsletter_field_does_not_hand_off(origins):
    sp = "tbsgen-test-r7-nl"
    try:
        # an e-mail box alone refuses interaction but is no reason to call the user (it would never clear)
        r = _run(sp, "open", origins[0] + "/newsletter")
        assert r["status"] == "ok" and r["page"]["login_form"] is True, r
        # a page with unmistakable login semantics is handed over
        r = _run(sp, "open", origins[0] + "/r7-title")
        assert r["status"] == "handoff" and r["handoff"]["kind"] == "login", r
    finally:
        _finish(sp)


def _fn_src(name):
    """The source of a top-level function of the action script (to exercise it in a page directly)."""
    a = ACTION.index("function " + name + "(")
    return ACTION[a:ACTION.index("\n}\n", a) + 2]


def test_round7_checks_run_in_the_evaluation_that_acts(origins):
    """domAct itself (the one page evaluation that acts) refuses when the page is not the looked-at URL, is off
    the allowed domains or shows any login signal - whatever the Node-side checks before it saw."""
    sp = "tbsgen-test-r7-dom"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        body = (f"const domAct = {_fn_src('domAct')};\nconst domWalk = {_fn_src('domWalk')};\n"
                f"const credScan = {_fn_src('credScan')};\nconst pageInfo = {_fn_src('pageInfo')};\n"
                f"const itemList = {_fn_src('itemList')};\nconst lookEval = {_fn_src('lookEval')};\n"
                f"const ssoSubmit = {_fn_src('ssoSubmit')};\n"
                "const t = await taskSpace(" + json.dumps(sp) + ");\nconst pg = (await t.pages())[0];\n"
                "const run = (u, allow, setup) => pg.evaluate(([a, w, s, u, allow, setup, pi]) => {\n"
                "  if (setup) document.body.insertAdjacentHTML('beforeend', setup);\n"
                "  const f = new Function('return (' + a + ')')();\n"
                "  const k = new Function('return (' + w + ')')()().findIndex((p) => (p[0].innerText || '').trim() === 'Accept');\n"
                "  return f(w, k, allow, 'hover', '', u, s, pi, 'just a moment');\n"
                "}, [domAct.toString(), domWalk.toString(), credScan.toString(), u, allow, setup || '', pageInfo.toString()]);\n"
                "const here = (await pg.info()).url;\n"
                "const res = [await run(here + '#x', ['127.0.0.1'], ''), await run(here, ['example.org'], ''),\n"
                "             await run(here, ['127.0.0.1'], '<form><input type=email name=e><button>Continue</button></form>')];\n"
                # a verification page appearing on the same URL after the Node-side checks (round 7)
                "await pg.evaluate(() => { document.title = 'Just a moment...'; document.querySelectorAll('form').forEach((f) => f.remove()); });\n"
                "res.push(await run(here, ['127.0.0.1'], ''));\n"
                # look's single evaluation: nothing is extracted from a page off the allowed domains
                "const lk = (allow) => pg.evaluate(([f, a, b, w, allow, ch]) => (new Function('return (' + f + ')')())(a, b, w, allow, ch),\n"
                "  [lookEval.toString(), pageInfo.toString(), itemList.toString(), domWalk.toString(), allow, 'just a moment']);\n"
                "const l1 = await lk(['example.org']), l2 = await lk(['127.0.0.1']);\n"
                "res.push({ foreign: l1.foreign || null, leaked: !!(l1.info || l1.items), ok: !!(l2.info && l2.items && l2.items.length) });\n"
                # sso_login's one evaluation refuses (before any click) off the SSO domains
                "res.push(await pg.evaluate(([f, s]) => (new Function('return (' + f + ')')())([['sso.elsewhere'], true, s]), [ssoSubmit.toString(), credScan.toString()]));\n"
                "console.log('R7 ' + JSON.stringify(res));\nprocess.exit(0);")
        p = subprocess.run([EGO, "nodejs", body], capture_output=True, text=True, timeout=90)
        line = next(x for x in (p.stdout + p.stderr).splitlines() if x.startswith("R7 "))
        res = json.loads(line[3:])
        assert "changed" in res[0]["refused"], res
        assert "outside the allowed" in res[1]["refused"], res
        assert "login" in res[2]["refused"], res
        assert "verification" in res[3]["refused"], res
        assert res[4] == {"foreign": "127.0.0.1", "leaked": False, "ok": True}, res
        assert res[5] == {"refused": "the page left the SSO login domain", "stage": "pre"}, res
    finally:
        _finish(sp)
