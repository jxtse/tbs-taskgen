// tbsgen browser-agent ACTION script (tbsgen.browser_agent).  The agent never writes browser code: it calls
// `tbsgen-browser N <action> [arg]`, the wrapper prepends one line `const INPUT = {...};` to this reviewed file
// and runs it with `ego-browser nodejs`.  The result is the last line starting with "TBSGEN_RESULT ";
// it is printed on stdout AND stderr (ego drops a script's stdout once the TaskSpace control is with the
// user after a hand-off, while stderr still flows).
//
// Each paper has its own TaskSpace (`<space>-p<N>`), so a hand-off to the user (verification / login page)
// pauses only that paper while the others continue.
//
// Actions (one Page - INPUT.page_label, default the first - of the paper's TaskSpace):
//   look [full] [@REF] [OFFSET]   page summary: URL, title, PDF?, challenge?, login form?, text excerpt and a
//                                 numbered list of links / buttons / fields: the page's DOM and same-origin
//                                 iframes (hidden dropdown items included), then items of cross-origin iframes
//                                 from the ego page snapshot (field texts removed); `full` = whole page,
//                                 `@REF` one snapshot subtree, OFFSET paginates
//   open URL                      navigate (URL host must be in INPUT.allow)
//   click I                       click item I of the last `look` list (DOM items in the checking evaluation,
//                                 a `_blank` link by a native click on its checked position, cross-origin
//                                 frame items through their snapshot ref)
//   click_xy X Y                  native click at viewport coordinates (the element there is checked first)
//   hover I                       move the mouse onto item I (native pointer)
//   scroll up|down [PX]           scroll the page
//   select I VALUE                choose VALUE (option value or label) of select item I
//   type I TEXT                   type TEXT into text item I (never into a password / one-time-code / login field)
//   screenshot                    PNG of the viewport under INPUT.files_dir with every field masked (refused
//                                 where fields cannot be masked: login pages, unreadable frames with fields)
//   tabs / switch LABEL           list / select the TaskSpace's pages (never the user's)
//   back                          one history step back - only to a same-origin entry on an allowed host
//   wait_for url SUBSTR|sel CSS [S]   wait until the URL contains SUBSTR / a CSS selector matches (<= 60 s)
//   save_pdf                      the current page is a PDF document: save it
//   sso_login                     on a configured institution SSO login domain: submit the browser-autofilled
//                                 login form (values never read; POST to the SSO domain only; no retry)
//   handoff_status                poll a running hand-off (the tool handed the page to the user)
//   finish                        close the TaskSpace
// Hard rules enforced here: credential VALUES never reach the output (field values are never read in the page;
// the ego snapshot is reduced in this process before use - every field's text is dropped - and screenshots
// mask all fields or are refused); no interaction with verification pages / widgets (they are handed to the
// user); no click / type / select / hover on a page with a login signal (sso_login or the hand-off only);
// navigation targets are checked before (links, forms, frames, back) and the landed URL after (a page that ends
// up outside INPUT.allow is reset to about:blank); files only under INPUT.files_dir, validated by the wrapper.
const out = { status: "error", action: INPUT.action, files: [] };
const CHALLENGE = /just a moment|verify you are human|attention required|access denied|are you a robot|security check|captcha|human verification|请稍候|请稍等|人机验证|安全验证/i;
const WIDGET = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
const COOKIE = /^(accept|reject|allow all|reject all|agree|consent|dismiss|close|ok|got it|accept all|accept all cookies|manage preferences|接受|拒绝|同意|全部接受|关闭|确定)/i;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hostOk = (u) => {
  let h;
  try { const x = new URL(u); if (!/^https?:$/.test(x.protocol)) return false; h = x.hostname.toLowerCase(); } catch (e) { return false; }
  return INPUT.allow.some((d) => h === d || h.endsWith("." + d));
};

const hostIn = (u, domains) => {
  let h, pr;
  try { const x = new URL(u); pr = x.protocol; h = x.hostname.toLowerCase(); } catch (e) { return false; }
  // HTTPS only, except loopback names when INPUT.sso.local_test (set only by the local test suite)
  const local = h === "127.0.0.1" || h === "localhost" || h.endsWith(".localhost");
  if (pr !== "https:" && !(local && INPUT.sso && INPUT.sso.local_test === true)) return false;
  return domains.some((d) => h === d || h.endsWith("." + d));
};
const ssoDomains = () => (INPUT.sso && INPUT.sso.domains) || [];

// The ego page snapshot reduced in THIS process before anything uses it: a field's snapshot text is its current
// VALUE (also in cross-origin frames), so every line below a textbox / searchbox / combobox / spinbutton /
// editable container (and below a password-like node) is dropped.  Only this form is ever kept or printed.
async function safeSnapshot(page, opts) {
  // (the raw text stays inside this function: never returned, printed or put into an error message)
  let raw = await page.snapshot(opts).then(String).catch(() => null);
  if (raw === null) return null;
  const outL = [];
  let fieldInd = -1;
  for (const ln of raw.split("\n")) {
    const ind = ln.length - ln.trimStart().length;
    if (fieldInd >= 0 && ind > fieldInd) continue;            // inside a field: its text / value
    fieldInd = -1;
    if (/^\s*(textbox|searchbox|combobox|spinbutton|listbox|slider|password)\b/.test(ln)
        || /^\s*container\b.*\[ref=/.test(ln)) fieldInd = ind;
    outL.push(ln);
  }
  raw = null;
  return outL.join("\n");
}

// Everything printed goes through emit(): every URL - a whole value or a URL inside a longer string (error
// messages) - loses its query, fragment and userinfo, whatever its host (OAuth / SAML / SSO callbacks carry
// tokens in many spellings; the agent follows links with `click I`, which uses the page's own href), and the
// raw result is never printed.  The wrapper keeps `look_url` for its own page-change check (same treatment).
function cleanUrl(u) {
  try { const x = new URL(u); return /^(https?|ftp|wss?):$/.test(x.protocol) ? x.origin + x.pathname : x.protocol; }
  catch (e) { return "[url]"; }                      // unparseable (e.g. cut off): never echoed
}
// (a URL runs to the next whitespace / double quote / angle bracket: a location.href may legally carry ' and `
// in its query or fragment, so those never end it - over-cleaning a quoted URL in a message is harmless)
const URL_IN = /\b[a-z][a-z0-9+.-]*:\/\/[^\s"<>]+/gi;
function scrub(v) {
  if (typeof v === "string") return v.replace(URL_IN, (m) => cleanUrl(m));
  if (Array.isArray(v)) return v.map(scrub);
  if (v && typeof v === "object") { const o = {}; for (const k of Object.keys(v)) o[k] = scrub(v[k]); return o; }
  return v;
}
function emit() {
  const line = "TBSGEN_RESULT " + JSON.stringify(scrub(out));
  console.log(line);
  console.error(line);
}

// a URL as reported back to the agent from login pages: no query or fragment (they can carry tokens)
const urlKey = async (u) => (await import("node:crypto")).createHash("sha256").update(String(u || "")).digest("hex").slice(0, 32);
const bareUrl = (u) => { try { const x = new URL(u); return x.origin + x.pathname; } catch (e) { return ""; } };

// ---------- page / item enumeration -----------------------------------------------------------
async function currentPage(task) {
  const pages = await task.pages();
  if (INPUT.page_label) {
    const p = pages.find((x) => x.label === INPUT.page_label);
    if (p) return p;
  }
  return pages[0] || (await task.newPage());
}

// Runs in the page (source is serialized): every clickable element and form field of the top document and
// the same-origin iframes, hidden dropdown items included, as [element, frame] pairs.  Field values are
// NEVER read here or by the callers - only types, names and labels, so credentials cannot leak.
function domWalk() {
  const SEL = "a[href], button, summary, select, [role=button], [role=link], [role=menuitem], [role=option], [role=tab],"
    + " input[type=submit], input[type=button], input[type=text], input[type=search], input:not([type])";
  const pairs = [];
  const walk = (doc, frame) => {
    if (pairs.length >= 240) return;
    for (const el of doc.querySelectorAll(SEL)) { pairs.push([el, frame]); if (pairs.length >= 240) return; }
    // same-origin frames are walked (their items act in place); a cross-origin frame is one `frame` item
    // (click opens its checked URL as the page) and its items come from the snapshot (snapItems)
    for (const f of doc.querySelectorAll("iframe")) {
      let inner = null;
      try { inner = f.contentDocument; } catch (e) { inner = null; }
      if (inner) walk(inner, frame + 1);
      else if (f.src && /^https?:/i.test(f.src) && pairs.length < 240) pairs.push([f, frame]);
    }
  };
  walk(document, 0);
  return pairs;
}

// The numbered item list of `look` (the page's own DOM only).  `look` and the acting actions must build it
// identically (the wrapper refuses actions after the page changed).
function itemList(src) {
    const walk = new Function("return (" + src + ")")();
    const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
    return walk().map(([el, frame]) => {
      const t = el.tagName.toLowerCase();
      const field = t === "input" || t === "select" || t === "textarea";
      const form = el.closest ? el.closest("form") : null;
      const r = el.getBoundingClientRect();
      // (a <button>'s value is never shown and may carry a token: only an input button's value - its visible
      // label - is used; round 12)
      const ibtn = t === "input" && ["submit", "button", "reset"].includes((el.getAttribute("type") || "").toLowerCase());
      const itext = ((field && !ibtn ? "" : (el.innerText || (ibtn ? el.value : "") || ""))
        + " " + (el.getAttribute("aria-label") || el.getAttribute("placeholder") || "")).trim().slice(0, 80);
      if (t === "iframe") {
        return { frame: frame, tag: t, kind: "frame", ftype: "", text: (el.getAttribute("title") || el.name || "iframe").slice(0, 80),
                 itext: "", href: el.src, hidden: r.width < 1 || r.height < 1, widget: W.test(el.src + " " + (el.getAttribute("title") || "")),
                 login: false, form_action: "" };
      }
      return {
        frame: frame, tag: t,
        kind: field ? "field" : (t === "a" ? "link" : "button"),
        ftype: t === "input" ? (el.getAttribute("type") || "text") : (t === "select" ? "select" : ""),
        text: (itext + " " + (el.getAttribute("name") || el.id || "")).trim().slice(0, 80),
        itext: itext,
        href: t === "a" ? el.href : undefined,
        hidden: r.width < 1 || r.height < 1,
        widget: W.test([el.id, el.className && String(el.className), el.getAttribute("name"), el.getAttribute("aria-label")].join(" ")),
        login: ((f) => !!f && ([...Object.getOwnPropertyDescriptor(HTMLFormElement.prototype, "elements").get.call(f)].some((x) => (x.getAttribute("type") || "").toLowerCase() === "password") || Element.prototype.querySelector.call(f, "input[type=password]") !== null))(el.form || form),
        form_action: form ? (form.getAttribute("action") || "") : "",
      };
    });
}
// ego snapshot lines: `  anchor [ref=4, url=...]` / `  iframe [ref=6]` / `    text "Download"`.  Only items
// below an iframe that domWalk could not enter (cross-origin) are kept; field items never carry a text.
// the page's frame tree in document order: { kids: [{ cross, kids }] } (same-origin frames are entered)
function frameTree() {
  const walk = (doc) => [...doc.querySelectorAll("iframe, frame")].map((f) => {
    let d = null; try { d = f.contentDocument; } catch (e) { d = null; }
    return d ? { cross: false, kids: walk(d) } : { cross: true, kids: [] };
  });
  return { kids: walk(document) };
}
function snapItems(snap, dom, tree) {
  const lines = String(snap || "").split("\n");
  const res = [];
  const stack = [];
  const domKeys = new Set(dom.map((x) => x.kind + "|" + (x.href || "") + "|" + (x.itext || x.text || "")));
  const rootNode = { kids: (tree && tree.kids) || [], next: 0 };
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    if (!ln.trim()) continue;
    const ind = ln.length - ln.trimStart().length;
    while (stack.length && stack[stack.length - 1].ind >= ind) stack.pop();
    if (/^\s*iframe \[ref=\d+/.test(ln)) {
      // only frames domWalk could not enter (cross-origin) give snapshot items; a same-origin frame's items are
      // DOM items (checked by domAct).  Frames are matched to the DOM's frame tree by document order; inside a
      // cross-origin frame everything is cross-origin; an unmatched frame counts as same-origin (skipped)
      const parent = [...stack].reverse().find((s) => s.iframe);
      let node;
      if (parent && parent.cross) node = { cross: true, kids: [], next: 0 };
      else {
        const pn = parent ? parent.node : rootNode;
        const k = pn && pn.kids ? pn.kids[pn.next++] : null;
        node = k ? { cross: !!k.cross, kids: k.kids || [], next: 0 } : { cross: false, kids: [], next: 0 };
      }
      stack.push({ ind: ind, iframe: true, cross: node.cross, node: node });
      continue;
    }
    const m = /^\s*(button|anchor|link|combobox|textbox|searchbox|menuitem|tab|option) (?:"[^"]*" )?\[ref=(\d+)(?:[^\]]*?url=(\S+?))?\]/.exec(ln);
    if (!m || !stack.some((s) => s.iframe) || !stack.every((s) => !s.iframe || s.cross)) continue;
    const field = ["textbox", "searchbox", "combobox"].includes(m[1]);
    const t = field ? null : /^\s*text "(.*)"$/.exec(lines[i + 1] || "");
    const nm = /^\s*\w+ "([^"]*)"/.exec(ln);
    const it = { src: "snap", kind: m[1] === "anchor" ? "link" : (field ? "field" : m[1]),
                 text: (field ? (nm ? nm[1] : "") : (t ? t[1] : (nm ? nm[1] : ""))).slice(0, 80),
                 href: (m[3] || "").replace(/,$/, ""), ref: m[2], frame: 1,
                 meta: ln.replace(/url=\S+/g, "").slice(0, 300) };
    if (domKeys.has(it.kind + "|" + it.href + "|" + it.text)) continue;
    res.push(it);
  }
  return res;
}
// a login form inside a cross-origin frame (credScan cannot read it): seen in the reduced snapshot
function snapLogin(snap) {
  return /^\s*(?:textbox|password)\b[^\n]*(?:pass|pwd|密码|one-?time|otp|验证码|动态码)/im.test(String(snap || ""))
    || /^\s*textbox\b[^\n]*type=password/im.test(String(snap || ""));
}
async function items(page, scope) {
  const dom = await page.evaluate(([a, w]) => (new Function("return (" + a + ")")())(w),
                                  [itemList.toString(), domWalk.toString()]).catch(() => []);
  // cross-origin frames only: no snapshot is taken when the page has none
  if (!dom.some((d) => d.kind === "frame") && !scope) return dom;
  const opts = scope === "full" ? { scope: "full_page" } : (scope && scope.startsWith("@") ? { scope: "subtree", root: scope } : undefined);
  const snap = (await safeSnapshot(page, opts)) || "";
  const tree = await page.evaluate((src) => (new Function("return (" + src + ")")())(), frameTree.toString()).catch(() => null);
  return dom.concat(tree ? snapItems(snap, dom, tree) : []);
}

// ---------- page state --------------------------------------------------------------------------
// in-page (serialized): the page state; `chSrc` is CHALLENGE.source - `challenge_page` applies the same test
// as isChallenge() below, inside the page (used by the evaluation that acts)
function pageInfo(chSrc) {
  // a RENDERED password field (publisher pages often carry a hidden header-login template)
  const pw = [...document.querySelectorAll("input[type=password]")].some((x) => x.getClientRects().length > 0
    && getComputedStyle(x).visibility !== "hidden");
  const r = {
    url: location.href, title: document.title || "", ctype: document.contentType || "",
    text: (document.body && document.body.innerText || "").slice(0, 1500),
    challenge: !!document.querySelector("#challenge-form, #cf-challenge-running, .cf-turnstile, #px-captcha, iframe[src*='challenges.cloudflare.com'], iframe[src*='captcha'], .g-recaptcha, .h-captcha"),
    login_form: pw,
  };
  if (chSrc) {
    const C = new RegExp(chSrc, "i");
    r.challenge_page = r.challenge || C.test(r.title) || (/^\s*$/.test(r.title) && C.test(r.text.slice(0, 300)));
  }
  return r;
}
async function info(page) {
  try {
    return await page.evaluate((src) => (new Function("return (" + src + ")")())(""), pageInfo.toString());
  } catch (e) {
    return { url: "", title: "", ctype: "", text: "", challenge: false, login_form: false };
  }
}
// ONE conservative login / MFA scan, used by every guard (interaction refusal - inside the same evaluation that
// acts -, the hand-off decision, the hand-off poll): runs in the page (source serialized).  Walks the
// top document, open shadow roots and same-origin iframes.  NARROWED (round 7): no judgement about what a form
// is "for" - any login signal makes the page a login page, and a false positive only hands the page to the user:
//   fields: a rendered input of type password / email / tel, an autocomplete token username / email /
//           current-password / new-password / one-time-code, a field in the browser's :autofill state, a
//           text-like field whose name / label looks like an account, user name or code;
//   sem:    login words ("sign in", "log in", "登录", "continue with" ...) in the page / frame title or URL, in a
//           form action, in the text of a form's button or a "continue with ..." button, or in any button's
//           text while a rendered text field outside any form exists (a script-driven login widget);
//   frames: a rendered cross-origin identity-provider frame (it cannot be inspected);
//   mfa:    a one-time-code / SMS / verification-code field, or such a prompt where a text field could take it.
// Only RENDERED fields count (publisher pages carry hidden header-login templates; the browser does not
// autofill hidden fields); a form with any editable control, rendered or not, is never submitted (domAct).  Field VALUES are never read.
function credScan() {
  const NAME = /pass|pwd|user|login|account|otp|one-time|code|sms|\bpin\b|verif|captcha|e-?mail|phone|mobile|密码|账号|帐号|用户名|学号|工号|验证码|动态码|短信|手机|邮箱/i;
  const MFA = /otp|one-time|code|sms|verif|验证码|动态码|短信/i;
  const AC = /\b(?:username|email|current-password|new-password|one-time-code|tel|webauthn)\b/i;
  const WORDS = /\b(?:sign|log)[\s_-]?(?:in|on)\b|\blogin\b|\bsignin\b|continue with|(?:choose|select|switch|pick) (?:an |your )?account|登录|登入|登陆|统一身份认证|选择账[号户]/i;
  const URLW = /log[\s_-]?in|sign[\s_-]?in|log[\s_-]?on|sign[\s_-]?on|登录/i;
  const IDP = /login|signin|sign-in|\/auth\b|oauth|\bsso\b|passport|account|\bidp\b|shibboleth|wayf|\bcas\b/i;
  // institution selection (WAYF / discovery service): the agent never chooses or types an institution - such a
  // page goes to the user like a login (round 11)
  const INST = /institution|\bwayf\b|identity provider|home organi[sz]ation|学校|机构|院校|高校/i;
  const WAYF = /\bwayf\b|where are you from|(?:select|choose|find|search for) (?:your |an )?(?:institution|organi[sz]ation|university|identity provider)|discovery service|seamless ?access|选择(?:您的|你的)?(?:学校|机构|院校|高校|单位)/i;
  const TXT = /验证码|短信|动态码|一次性|二次验证|安全验证|身份验证|one[- ]time|verification code|security code|two[- ]factor|2fa|authenticator/i;
  const shown = (el) => { try { const v = el.ownerDocument.defaultView; return el.getClientRects().length > 0
    && v.getComputedStyle(el).visibility !== "hidden"; } catch (e) { return true; } };
  const path = (u) => { try { const x = new URL(u, location.href); return x.hostname + x.pathname; } catch (e) { return String(u || ""); } };
  // the text of the labels really associated with a field (<label for>, a wrapping <label>, aria-labelledby)
  const labelText = (el) => {
    let t = "";
    try { for (const l of el.labels || []) t += " " + (l.textContent || ""); } catch (e) {}
    for (const id of (el.getAttribute("aria-labelledby") || "").split(/\s+/).filter(Boolean)) {
      const l = el.ownerDocument.getElementById(id);
      if (l) t += " " + (l.textContent || "");
    }
    return t.slice(0, 300);
  };
  const meta0 = (el) => [el.id, el.getAttribute("name"), el.getAttribute("placeholder"), el.getAttribute("aria-label"),
                         el.getAttribute("title"), labelText(el)].join(" ");
  let fields = false, strong = false, sem = false, mfa = false, text = 0, loose = 0, looseBtn = false, wayf = false;
  const walk = (doc) => {
    for (const host of doc.querySelectorAll("*")) if (host.shadowRoot) walk(host.shadowRoot);
    for (const el of doc.querySelectorAll("input, textarea")) {
      const t = (el.tagName === "TEXTAREA" ? "textarea" : (el.getAttribute("type") || "text").toLowerCase());
      if (["hidden", "submit", "button", "reset", "image", "checkbox", "radio", "file", "range", "color"].includes(t)) continue;
      if (!shown(el)) continue;
      text++;
      if (INST.test(meta0(el))) wayf = true;
      if (!el.form && !(el.closest && el.closest("form"))) loose++;     // a script-driven widget may read it
      const ac = el.getAttribute("autocomplete") || "";
      const meta = [el.id, el.getAttribute("name"), ac, el.getAttribute("placeholder"), el.getAttribute("aria-label"),
                    el.getAttribute("inputmode"), labelText(el)].join(" ");
      if (["password", "email", "tel"].includes(t) || AC.test(ac) || NAME.test(meta)) fields = true;
      // `strong`: unmistakable credential fields (these alone also decide the hand-off / its clearing)
      if (t === "password" || /\b(?:username|current-password|new-password|one-time-code)\b/i.test(ac)
          || /pass|pwd|user|login|account|otp|one-time|密码|账号|帐号|用户名|学号|工号|验证码|动态码/i.test(meta)) strong = true;
      try { if (el.matches(":autofill") || el.matches(":-webkit-autofill")) { fields = true; strong = true; } } catch (e) {}
      if (MFA.test(meta) || /one-time-code/i.test(ac)) mfa = true;
    }
    // an SSO hand-back page (SAML / WS-Federation / OIDC form_post): it carries an authentication assertion -
    // a login page, handed to the user (round 13)
    if (doc.querySelector('input[name="SAMLResponse" i], input[name="SAMLRequest" i], input[name="wresult" i], input[name="id_token" i], input[name="code" i][type="hidden" i]')) sem = true;
    // a select whose name / label is an account / user name (an account chooser) is a login field too
    for (const el of doc.querySelectorAll("select")) {
      if (!shown(el)) continue;
      if (INST.test(meta0(el))) wayf = true;
      const ac = el.getAttribute("autocomplete") || "";
      const meta = [el.id, el.getAttribute("name"), ac, el.getAttribute("aria-label"), el.getAttribute("title"),
                    labelText(el)].join(" ");
      if (AC.test(ac) || NAME.test(meta)) { fields = true; strong = true; }
    }
    for (const b of doc.querySelectorAll("button, input[type=submit], input[type=button], input[type=image], [role=button]")) {
      if (!shown(b)) continue;
      const bt = [b.innerText || "", b.tagName === "INPUT" ? (b.getAttribute("value") || "") : "",
                  b.getAttribute("aria-label") || "", b.getAttribute("title") || ""].join(" ");
      if (!WORDS.test(bt)) continue;
      // a login button of a form, or a "continue with ..." button, is login semantics; a header "Sign in"
      // button outside any form counts only together with a rendered text field outside any form (a
      // script-driven login widget) - a header entry point submits nothing (the login page it leads to is
      // checked when it loads)
      if (b.form || (b.closest && b.closest("form")) || /continue with/i.test(bt)) sem = true;
      else looseBtn = true;
    }
    // a form whose action is a login endpoint, when any control of it is rendered (publisher pages carry hidden
    // header-login templates)
    for (const f of doc.querySelectorAll("form")) {
      if (URLW.test(path(f.getAttribute("action") || "")) && [...Object.getOwnPropertyDescriptor(HTMLFormElement.prototype, "elements").get.call(f)].some((x) => (x.getAttribute("type") || "") !== "hidden" && shown(x))) sem = true;
    }
    for (const f of doc.querySelectorAll("iframe")) {
      let inner = null;
      try { inner = f.contentDocument; } catch (e) { inner = null; }
      if (inner) {
        if (WORDS.test(inner.title || "") || URLW.test(path(inner.location ? inner.location.href : ""))) sem = true;
        walk(inner);
      } else if (IDP.test((f.getAttribute("src") || "") + " " + (f.getAttribute("title") || ""))) {
        // a cross-origin identity frame cannot be inspected: a rendered one counts as login fields
        const r = f.getBoundingClientRect();
        if (r.width >= 20 && r.height >= 20 && shown(f)) { fields = true; strong = true; }
      }
    }
  };
  walk(document);
  if (looseBtn && loose > 0) sem = true;
  // a WAYF wording in the title, URL or a rendered heading: an institution selection page, with or without
  // fields (also a plain list of institution links goes to the user - the agent never picks an institution)
  const heads = [...document.querySelectorAll("h1, h2, h3, legend")].filter(shown).slice(0, 20).map((h) => h.textContent || "").join(" ");
  if (WAYF.test(document.title || "") || WAYF.test(path(location.href)) || WAYF.test(heads)) wayf = true;
  // (a WAYF page is NOT a login page: the agent may search / choose the institution there)
  if (WORDS.test(document.title || "") || URLW.test(path(location.href))) sem = true;
  // a verification-code prompt in the text counts only where a text field could take the code
  if (text > 0 && TXT.test((document.body && document.body.innerText || "").slice(0, 3000))) mfa = true;
  // cred: any signal - gates every click.  login: unmistakable credential fields or
  // login semantics - decides the hand-off and whether a handed-off page is cleared (a newsletter e-mail box
  // alone refuses interaction, but is no reason to call the user, and could never "clear")
  return { cred: fields || sem || mfa, login: strong || sem || mfa, fields: fields, sem: sem, mfa: mfa, wayf: wayf };
}
async function scan(page) {
  try {
    return await page.evaluate((src) => (new Function("return (" + src + ")")())(), credScan.toString());
  } catch (e) { return { cred: true, login: true, fields: true, sem: true, mfa: false, unknown: true }; }   // cannot be checked -> a login page
}
async function credPage(page) { return (await scan(page)).cred; }
async function loginPage(page) { return (await scan(page)).login; }
const isChallenge = (i) => i.challenge || CHALLENGE.test(i.title) || (/^\s*$/.test(i.title) && CHALLENGE.test(i.text.slice(0, 300)));

async function settle(page) {
  // a bot-wall page may clear by itself; never interact with it
  let i = await info(page);
  const t0 = Date.now();
  while (isChallenge(i) && Date.now() - t0 < INPUT.challenge_wait_ms) { await sleep(2000); i = await info(page); }
  return i;
}

async function navStatus(page) {
  try {
    return await page.evaluate(() => { const n = performance.getEntriesByType("navigation")[0]; return n && n.responseStatus ? n.responseStatus : null; });
  } catch (e) { return null; }
}

async function savedIsChallenge(file) {
  const fs = await import("node:fs/promises");
  try {
    const fh = await fs.open(file, "r");
    const buf = Buffer.alloc(20000);
    const { bytesRead } = await fh.read(buf, 0, 20000, 0);
    await fh.close();
    const head = buf.subarray(0, bytesRead).toString("latin1");
    if (head.startsWith("%PDF-")) return false;
    const t = /<title[^>]*>([^<]*)/i.exec(head);
    return (t && CHALLENGE.test(t[1])) || /cf-chl|cf_chl_opt|challenge-platform|px-captcha|\/cdn-cgi\/challenge/i.test(head);
  } catch (e) { return false; }
}

const fileName = () => `${INPUT.files_dir}/f${Date.now()}-${Math.floor(Math.random() * 1e6)}.bin`;

// arm download and popup watches around an action; report new tabs of this space (they are managed pages the
// action itself opened - never the user's tabs)
async function withDownload(task, page, act) {
  const tabsBefore = new Set((await task.tabs().catch(() => [])).map((t) => t.label));
  let done = false, end = 0, dl = null;
  let w = page.waitForEvent("download", { timeout: INPUT.download_wait_ms }).catch(() => null);
  const p = Promise.resolve().then(act).catch((e) => { out.action_error = String(e).slice(0, 200); })
    .finally(() => { done = true; end = Date.now(); });
  for (;;) {
    dl = await w;
    if (dl) break;
    const left = done ? end + INPUT.download_wait_ms - Date.now() : INPUT.download_wait_ms;
    if (done && left <= 0) break;
    w = page.waitForEvent("download", { timeout: Math.max(500, left) }).catch(() => null);
  }
  await p;
  try {
    if (dl) {
      let u = "";
      try { u = dl.url(); } catch (e) {}
      if (u && !u.startsWith("blob:") && !u.startsWith("data:") && !hostOk(u)) { out.note = "download from a host outside the allowed domains ignored"; return; }
      const f = fileName();
      await dl.saveAs(f);
      if (await savedIsChallenge(f)) { out.blocked = "downloaded bot-wall page"; (await import("node:fs/promises")).unlink(f).catch(() => {}); return; }
      out.files.push(f);
    }
  } finally {
    await closeForeignTabs(task, tabsBefore);
  }
}

// new tabs the action opened (never the user's): reported; those outside the allowed domains are closed
async function closeForeignTabs(task, tabsBefore) {
  const tabs = await task.tabs().catch(() => []);
  const newTabs = tabs.filter((t) => !tabsBefore.has(t.label));
  if (newTabs.length) {
    out.tabs = newTabs.map((t) => ({ label: t.label, url: cleanUrl(t.url || "").slice(0, 200), allowed: hostOk(t.url || "") }));
    // a new tab (opened by the action) that landed outside the allowed domains is closed at once
    const pages = await task.pages().catch(() => []);
    for (const [k, t] of newTabs.entries()) {
      if (out.tabs[k].allowed || !t.url || t.url.startsWith("about:")) continue;
      const pg = pages.find((x) => x.label === t.label);
      if (pg) { await pg.close().catch(() => {}); out.tabs[k].closed = true; }
    }
  }
}

async function after(page) {
  const i = await settle(page);
  out.http = await navStatus(page);
  if (out.http === 429) { out.blocked = "HTTP 429"; out.http429 = true; }
  if (/^https?:/i.test(i.url || "") && !hostOk(i.url)) {
    let h = ""; try { h = new URL(i.url).hostname; } catch (e) {}
    try { await page.goto("about:blank", { timeout: 5000 }); } catch (e) {}
    out.left_allowed = h;
    return await info(page);
  }
  return i;
}

// a challenge or login page: hand the paper's space to the user instead of solving it.  The result
// line is printed FIRST (ego drops this process's output once control is with the user).
// `kind`: challenge | login | mfa.  Without a hand-off budget (INPUT.handoff.enabled false) nothing is handed
// over: a challenge is reported as blocked (the wrapper cools the publisher), a login as login_required.
async function handoff(task, page, why, host, kind, title) {
  if (!(INPUT.handoff && INPUT.handoff.enabled)) {
    if (kind === "challenge") out.blocked = "verification page: " + String(why).slice(0, 60);
    else out.login_required = String(why).slice(0, 120);
    if (out.status !== "refused") out.status = "ok";
    return;
  }
  out.status = "handoff";
  out.handoff = { reason: String(why).slice(0, 120), host: host || "", kind: kind || "challenge", at_ms: Date.now(),
                  page_label: (page && page.label) || "", title: String(title || "").slice(0, 200) };
  emit();
  try { await task.handOff(); } catch (e) { /* the space may already be with the user */ }
  process.exit(0);
}

// what to do with the settled page state: nothing / report / hand off / block
async function postCheck(task, page, i, opts) {
  const h = (() => { try { return new URL(i.url).hostname; } catch (e) { return ""; } })();
  if (isChallenge(i)) {
    await handoff(task, page, "challenge: " + (i.title || "").slice(0, 60), h, "challenge", i.title);
    return i;
  }
  // the shared credential / MFA scan (password, user-name, e-mail sign-in, verification-code fields; frames,
  // shadow roots): such a page goes to the user
  // (`cred` - any login signal - is reported and refuses interaction; `login` hands the page to the user)
  const sc = i.login_form ? { cred: true, login: true, mfa: false } : await scan(page);
  if (sc.cred) i.login_form = true;
  const onSso = INPUT.sso && (INPUT.sso.enabled || INPUT.sso.login) && hostIn(i.url, ssoDomains());
  if (onSso && sc.login) { out.sso_page = true; i.login_form = true; return i; }   // the agent runs sso_login
  if (sc.login && !(opts && opts.no_login_handoff)) {
    if (INPUT.handoff && INPUT.handoff.enabled) return handoff(task, page, sc.mfa ? "verification code" : (sc.wayf ? "institution selection" : "login form"), h, sc.mfa ? "mfa" : "login", i.title);
    out.login_form = true;                       // reported; the agent books `login_required`
  }
  return i;
}

// ---------- actions -----------------------------------------------------------------------------
function parseLookArg(arg) {
  let scope = "", offset = 0;
  for (const p of String(arg || "").trim().split(/\s+/)) {
    if (p === "full") scope = "full";
    else if (/^@\d{1,5}$/.test(p)) scope = p;
    else if (/^\d{1,4}$/.test(p)) offset = parseInt(p, 10);
  }
  return { scope: scope, offset: offset };
}

const LIMIT = 120;                                   // items per `look` page

// in-page (serialized): the page state and the item list in ONE evaluation, only on an allowed host (a page
// that redirected off the domains after the entry check is never read)
function lookEval(infoSrc, listSrc, walkSrc, allow, chSrc) {
  const x = location.href;
  let ok = false;
  try { const u = new URL(x); const h = u.hostname.toLowerCase();
        ok = !/^https?:$/.test(u.protocol) ? !/^(https?|ftp|wss?|file):/i.test(x) : allow.some((d) => h === d || h.endsWith("." + d)); } catch (e) { ok = false; }
  if (!ok) { let h = ""; try { h = new URL(x).hostname; } catch (e) {} return { foreign: h || "?" }; }
  return { info: (new Function("return (" + infoSrc + ")")())(chSrc), items: (new Function("return (" + listSrc + ")")())(walkSrc) };
}

async function look(task, page) {
  const { scope, offset } = parseLookArg(INPUT.arg);
  const r = await page.evaluate(([f, a, b, w, allow, ch]) => (new Function("return (" + f + ")")())(a, b, w, allow, ch),
    [lookEval.toString(), pageInfo.toString(), itemList.toString(), domWalk.toString(), INPUT.allow, CHALLENGE.source])
    .catch((e) => ({ error: String(e).slice(0, 120) }));
  if (!r || r.error || !r.info) {
    if (r && r.foreign) {                           // nothing of it is returned; the page is reset
      await page.goto("about:blank", { timeout: 5000 }).catch(() => {});
      out.reset_foreign = r.foreign;
      out.status = "refused"; out.error = "the page left the allowed domains (reset); open an allowed URL";
      return;
    }
    out.status = "error"; out.error = "the page could not be read" + (r && r.error ? ": " + r.error : "");
    return;
  }
  const i = r.info;
  let all = r.items || [];
  if (all.some((d) => d.kind === "frame") || scope) {
    const opts = scope === "full" ? { scope: "full_page" } : (scope.startsWith("@") ? { scope: "subtree", root: scope } : undefined);
    const snap = (await safeSnapshot(page, opts)) || "";
    const tree = await page.evaluate((src) => (new Function("return (" + src + ")")())(), frameTree.toString()).catch(() => null);
    all = all.concat(tree ? snapItems(snap, all, tree) : []);
    if (snapLogin(snap)) i.login_form = true;
  }
  out.look_scope = scope;
  const pageItems = all.slice(offset, offset + LIMIT).map((it, k) => ({
    i: offset + k, kind: it.kind, text: it.text, href: it.href ? cleanUrl(it.href).slice(0, 200) : "",
    allowed: it.href ? hostOk(it.href) : undefined,
    hidden: !!it.hidden, frame: it.frame || 0, ftype: it.ftype || undefined,
    widget: !!it.widget, login: !!it.login, in_frame: it.src === "snap" || undefined,
  }));
  out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), challenge: isChallenge(i), login_form: i.login_form,
               text: i.text, items: pageItems, more: offset + LIMIT < all.length, next_offset: offset + LIMIT < all.length ? offset + LIMIT : undefined };
  out.look_key = await urlKey(i.url);   // the wrapper's page-change check: a hash, never the URL (it can carry tokens)
  out.status = "ok";
  await postCheck(task, page, i);       // a challenge or login page is handed to the user here
  out.page.login_form = !!i.login_form; // (postCheck's conservative scan, also WAYF pages)
  if (out.status === "handoff") return;
  out.status = "ok";
}

// The one evaluation that finds, checks and acts on item k of the dom list (click / hover / select / type).
// `walk` is the domWalk function itself; both run inside the page (ego's evaluate takes ONE argument).
function domAct(walkSrc, k, allow, kind, value, expectUrl, scanSrc, infoSrc, chSrc) {
  const ok = (u) => { try { const x = new URL(u, location.href); if (!/^https?:$/.test(x.protocol)) return false;
    const h = x.hostname.toLowerCase(); return allow.some((d) => h === d || h.endsWith("." + d)); } catch (e) { return false; } };
  // re-checked in the SAME evaluation that acts: still the page the agent looked at, still on an allowed host,
  // not a login / MFA page, not a verification page
  if (!ok(location.href)) return { refused: "the page is outside the allowed domains" };
  if (location.href !== expectUrl) return { refused: "the page changed since `look`; run look again" };
  if ((new Function("return (" + scanSrc + ")")())().cred) return { refused: "login / verification-code page: no interaction here (sso_login on the SSO domain, otherwise it is handed to the user)" };
  if ((new Function("return (" + infoSrc + ")")())(chSrc).challenge_page) return { refused: "verification page: do not interact" };
  const pair = new Function("return (" + walkSrc + ")")()()[k];   // domWalk: source -> function -> [el, frame]
  const el = pair && pair[0];
  if (!el) return { refused: "no such item; run look first" };
  const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
  // a form's own properties can be shadowed by its named controls (<input name=action>): native getters only
  const FP = HTMLFormElement.prototype;
  const fAction = (f) => Object.getOwnPropertyDescriptor(FP, "action").get.call(f);
  const fElems = (f) => [...Object.getOwnPropertyDescriptor(FP, "elements").get.call(f)];
  const fHasPw = (f) => Element.prototype.querySelector.call(f, "input[type=password]") !== null;
  const tag = el.tagName.toLowerCase();
  const btnInput = tag === "input" && ["submit", "button", "reset", "image"].includes((el.getAttribute("type") || "").toLowerCase());
  const isField = (tag === "input" && !btnInput) || tag === "textarea" || tag === "select";
  // a field's value is never read (it may be an autofilled credential): fields are named by their metadata
  const text = ((isField ? "" : (el.innerText || (btnInput ? el.value : "") || "")) || el.getAttribute("aria-label")
    || el.getAttribute("placeholder") || el.title || "").trim().slice(0, 80);
  if (W.test([el.id, el.className && String(el.className), el.getAttribute("name"), el.getAttribute("aria-label")].join(" "))) return { refused: "verification widget: do not interact" };
  // an item inside a (same-origin) frame that shows a verification page: never acted on (no solving)
  {
    const C = new RegExp(chSrc, "i");
    let d = el.ownerDocument, n = 0;
    while (d && d !== document && n++ < 8) {
      const body = (d.body && d.body.innerText || "").slice(0, 300);
      if (C.test(d.title || "") || (/^\s*$/.test(d.title || "") && C.test(body))
          || d.querySelector(".cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha, iframe[src*='captcha'], iframe[src*='challenges.cloudflare.com']"))
        return { refused: "verification page inside a frame: do not interact (it is handed to the user)" };
      const fe = d.defaultView && d.defaultView.frameElement;
      d = fe ? fe.ownerDocument : null;
    }
  }
  if (el.closest && el.closest(".cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha, [id*=captcha], [class*=captcha]")) return { refused: "verification widget: do not interact" };
  const lbl = el.closest && el.closest("label");
  if (lbl) {
    const c = lbl.control;
    if (c && (c.tagName === "BUTTON" || (c.tagName === "INPUT" && ["submit", "reset", "image", "button"].includes((c.getAttribute("type") || "").toLowerCase()))))
      return { refused: "a label of a button: click the button itself" };
  }
  const actv = (el.closest && el.closest("button, input")) || el;
  const form = actv.form || el.form || (el.closest ? el.closest("form") : null);
  const pwdForm = (f) => !!f && (fElems(f).some((x) => (x.getAttribute("type") || "").toLowerCase() === "password") || fHasPw(f));
  // login forms are submitted only by sso_login (checked target) or by the user
  if (pwdForm(form)) return { refused: "login form: credentials are not entered by this tool" };
  // the centre of the element in TOP-document coordinates (same-origin frames add their offsets)
  const centre = () => {
    el.scrollIntoView({ block: "center", inline: "center" });
    const r = el.getBoundingClientRect();
    let x = r.left + r.width / 2, y = r.top + r.height / 2, w = el.ownerDocument.defaultView;
    while (w && w !== window && w.frameElement) { const fr = w.frameElement.getBoundingClientRect(); x += fr.left; y += fr.top; w = w.parent; }
    return { x: Math.round(x), y: Math.round(y), vis: r.width >= 1 && r.height >= 1 };
  };
  if (tag === "iframe") {
    // a cross-origin frame as a whole: click opens its (checked) URL as this page
    if (kind !== "click") return { refused: "frame items can only be clicked (opens the frame's page); use look full for its items" };
    if (!ok(el.src)) return { refused: "frame URL is outside the allowed domains" };
    location.assign(el.src);
    return { opened: el.src, text: text };
  }
  if (kind === "click") {
    if (W.test(text)) return { refused: "verification widget: do not interact" };
    const atag = actv.tagName.toLowerCase();
    const aBtn = atag === "button" || (atag === "input" && ["submit", "image"].includes((actv.getAttribute("type") || "").toLowerCase()));
    if (form && aBtn && !ok((actv.hasAttribute("formaction") ? actv.formAction : "") || fAction(form) || location.href)) return { refused: "form target is outside the allowed domains" };
    if (form && !aBtn && !ok(fAction(form) || location.href)) return { refused: "form target is outside the allowed domains" };
    if ((el.hasAttribute("formaction") && !ok(el.formAction)) || (actv.hasAttribute("formaction") && !ok(actv.formAction))) return { refused: "form target is outside the allowed domains" };
    const link = tag === "a" ? el : (el.closest && el.closest("a[href]"));
    if (link) {
      const h = link.getAttribute("href") || "";
      // checked by its browser-resolved target (also "#frag": a <base href> can point it elsewhere)
      if (!/^\s*javascript:/i.test(h) && !ok(link.href)) return { refused: "link target is outside the allowed domains" };
      const tg = (link.getAttribute("target") || "").trim();
      if (tg && !/^_(self|top|parent)$/i.test(tg) && !/^\s*javascript:/i.test(h)) {
        // a new-tab link: a NATIVE click opens the publisher's tab (tracked; a foreign one is closed at once).
        // The point is checked to land on this link (nothing laid over it)
        const c = centre();
        if (c.vis && pair[1] === 0) {
          const hit = document.elementFromPoint(c.x, c.y);
          if (hit && (hit === link || link.contains(hit))) return { native: [c.x, c.y], newtab: true, text: text };
        }
        location.assign(link.href);                               // hidden / covered / in a frame: open here
        return { opened: link.href, text: text };
      }
    }
    el.click();
    return { clicked: text };
  }
  if (kind === "hover") {
    const c = centre();
    if (!c.vis) return { refused: "the item has no position (hidden?)" };
    return { native: [c.x, c.y], text: text };                       // moved by the caller (native pointer)
  }
  // fields: never into a password / one-time-code / login-identifier field, never a credential's own form
  const CRED = /pass|pwd|otp|one-?time|totp|verif|code|pin\b|token|secret|user|login|account|e-?mail|phone|mobile|tel\b|密码|验证码|动态码|短信|账号|帐号|用户名|学号|工号|手机|邮箱/i;
  let lab = "";
  try { for (const l of el.labels || []) lab += " " + (l.textContent || ""); } catch (e) {}
  const idn = [el.id, el.getAttribute("name"), el.getAttribute("autocomplete"), el.getAttribute("placeholder"),
               el.getAttribute("aria-label"), el.getAttribute("inputmode"), lab].join(" ");
  let af = false; try { af = el.matches(":autofill") || el.matches(":-webkit-autofill"); } catch (e) {}
  if (kind === "select") {
    if (tag !== "select") return { refused: "item is not a select" };
    if (af || CRED.test(idn)) return { refused: "this select can hold an account; it is not changed by this tool" };
    const opt = [...el.options].find((o) => o.value === value) || [...el.options].find((o) => o.label === value)
      || [...el.options].find((o) => o.label.toLowerCase() === String(value).toLowerCase());
    if (!opt) return { refused: "no such option" };
    el.value = opt.value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return { selected: opt.label.slice(0, 80) };
  }
  if (kind === "type") {
    const t = tag === "input" ? (el.getAttribute("type") || "text").toLowerCase() : (tag === "textarea" ? "textarea" : "");
    if (!["text", "search", "textarea", "url", "number"].includes(t)) return { refused: "not a plain text field" };
    if (af || CRED.test(idn)) return { refused: "this field can hold credentials (password / code / account); typing is not allowed" };
    if (form && form !== null && fElems(form).some((x) => { try { return x.matches(":autofill") || x.matches(":-webkit-autofill"); } catch (e) { return false; } }))
      return { refused: "a form with autofilled values: typing is not allowed" };
    el.focus();
    const set = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value");
    if (set && set.set) set.set.call(el, value); else el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return { typed: true };                                          // nothing of the field is read back
  }
  return { refused: "unknown action" };
}

// a snapshot item (inside a cross-origin frame): acted on through its ref
async function snapAct(task, page, kind, it, value) {
  // no action inside cross-origin frames while the page carries a verification widget frame
  const vf = await page.evaluate(() => [...document.querySelectorAll("iframe")].some((f) =>
    /captcha|turnstile|challenges\.cloudflare\.com|hcaptcha|recaptcha|perimeterx|px-cdn|arkoselabs|geetest/i.test((f.src || "") + " " + (f.title || ""))))
    .catch(() => true);
  if (vf) { out.status = "refused"; out.error = "verification widget on the page: do not interact (it is handed to the user)"; return; }
  if (it.widget || WIDGET.test(it.text || "") || WIDGET.test(it.meta || "")) { out.status = "refused"; out.error = "verification widget: do not interact"; return; }
  if (it.href && !hostOk(it.href)) { out.status = "refused"; out.error = "link target is outside the allowed domains"; return; }
  const CRED = /pass|pwd|otp|one-?time|verif|code|pin\b|token|secret|user|login|account|e-?mail|phone|mobile|密码|验证码|动态码|短信|账号|帐号|用户名|学号|工号|手机|邮箱/i;
  if ((kind === "type" || kind === "select") && (it.kind !== "field" || CRED.test(it.meta || ""))) {
    out.status = "refused"; out.error = it.kind !== "field" ? "item is not a field" : "this field can hold credentials; typing is not allowed"; return;
  }
  if (kind === "click" && !it.href) {
    // a button inside a cross-origin frame may submit a form whose target cannot be checked from here: open
    // the frame as the page (click its frame item) - there the button is a checked DOM item
    out.status = "refused"; out.error = "a button inside a cross-origin frame: click the frame item first (opens the frame as the page), then click the button"; return;
  }
  if (kind === "click" && it.href && !hostOk(it.href)) { out.status = "refused"; out.error = "link target is outside the allowed domains"; return; }
  const sel = "@" + it.ref;
  await withDownload(task, page, async () => {
    if (kind === "click") await page.click(sel, { timeout: 8000 });
    else if (kind === "hover") await page.hover(sel, { timeout: 8000 });
    else if (kind === "type") await page.fill(sel, value, { timeout: 8000 });
    else if (kind === "select") await page.selectOption(sel, value, { timeout: 8000 });
  });
  if (out.action_error) { out.status = "error"; out.error = "the item could not be used: " + out.action_error; return; }
  await sleep(kind === "hover" ? 800 : 1500);
  out.acted = { kind: kind, text: it.text || "" };
  const i = await after(page);
  if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
  out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
  if (out.status !== "handoff") out.status = "ok";
}

async function act(task, page, kind, k, value) {
  // the page must be the one the agent last looked at.  `seen` - the URL whose hash is the look_key - is
  // passed into domAct, which compares it with location.href in the same evaluation that acts
  if (!INPUT.look_key) { out.status = "refused"; out.error = "run look first"; return; }
  const ip = await info(page);
  const seen = ip.url;
  if (await urlKey(seen) !== INPUT.look_key) { out.status = "refused"; out.error = "the page changed since `look`; run look again"; return; }
  const all = await items(page, INPUT.look_scope || "");
  if (!(k >= 0) || k >= all.length) { out.status = "refused"; out.error = "no such item; run look first"; return; }
  const it = all[k];
  if (isChallenge(ip)) { out.status = "refused"; out.error = "verification page: do not interact"; await postCheck(task, page, ip); return; }
  if (ip.login_form || await credPage(page)) {
    out.status = "refused";
    out.error = "login / verification-code page: no interaction here (sso_login on the SSO domain, otherwise it is handed to the user / report login_required)";
    return;
  }
  if (it.widget || WIDGET.test(it.text || "")) { out.status = "refused"; out.error = "verification widget: do not interact"; return; }
  if (it.src === "snap") {
    if (snapLogin(await safeSnapshot(page, (INPUT.look_scope === "full") ? { scope: "full_page" } : undefined))) {
      out.status = "refused"; out.error = "login form inside a frame: no interaction here (it is handed to the user / report login_required)"; return;
    }
    return snapAct(task, page, kind, it, value);
  }
  const domRun = () => page.evaluate(([src, walk, k, allow, kind, value, u, scanSrc, infoSrc, chSrc]) => {   // ONE arg for ego
    const f = new Function("return (" + src + ")")();
    return f(walk, k, allow, kind, value, u, scanSrc, infoSrc, chSrc);
  }, [domAct.toString(), domWalk.toString(), k, INPUT.allow, kind, value || "", seen, credScan.toString(),
      pageInfo.toString(), CHALLENGE.source]);
  let r = null, failed = false;
  await withDownload(task, page, async () => {
    // (a navigation started inside the evaluation may end its context before it returns: navigated)
    r = await domRun().catch((e) => (/context|destroyed|navigat|detached/i.test(String(e)) ? { opened: "?" } : null));
    if (r && r.refused) { failed = true; out.status = "refused"; out.error = r.refused; return; }
    if (!r) { failed = true; out.status = "error"; out.error = "the page changed during the action (no result)"; return; }
    if (r.native && kind === "click") await page.mouse.click(r.native[0], r.native[1]);
    else if (r.native && kind === "hover") await page.mouse.move(r.native[0], r.native[1]);
  });
  if (failed) return;
  await sleep(kind === "hover" ? 800 : 1500);
  out.acted = { kind: kind, text: (r && (r.selected || r.text || r.clicked)) || "" };
  const i = await after(page);
  if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
  out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
  if (out.status !== "handoff") out.status = "ok";
}

function ssoSubmit(args) {
  const [domains, lt, scanSrc, infoSrc, chSrc] = args;
  const pre = (why) => ({ refused: why, stage: "pre" });
  const inDom = (u) => { try { const x = new URL(u); const hh = x.hostname.toLowerCase();
    const local = hh === "127.0.0.1" || hh === "localhost" || hh.endsWith(".localhost");
    if (x.protocol !== "https:" && !(local && lt)) return false;
    return domains.some((d) => hh === d || hh.endsWith("." + d)); } catch (e) { return false; } };
  if (!inDom(location.href)) return pre("the page left the SSO login domain");
  // the same verification-page test as everywhere else (title rules, challenge forms, captcha widgets)
  if ((new Function("return (" + infoSrc + ")")())(chSrc).challenge_page) return pre("verification page on the SSO login page");
  if ((new Function("return (" + scanSrc + ")")())().mfa
      || document.querySelector("iframe[src*='captcha'], .g-recaptcha, .h-captcha, .cf-turnstile")) return pre("MFA / captcha on the login page");
  const af = (el) => { try { return el.matches(":autofill") || el.matches(":-webkit-autofill"); } catch (e) { return false; } };
  const filled = (el) => af(el) || (lt && el.value.length > 0);
  const pwd = document.querySelector("form input[type=password]") || document.querySelector("input[type=password]");
  const form = pwd && (pwd.form || pwd.closest("form"));
  if (!form) return pre("no login form");
  const user = form.querySelector("input[autocomplete=username], input[type=text], input[type=email], input:not([type])");
  if (!user || !filled(pwd) || !filled(user)) return pre("the login form is not autofilled");
  const FP = HTMLFormElement.prototype;
  const fAction = (f) => Object.getOwnPropertyDescriptor(FP, "action").get.call(f);
  const fMethod = (f) => Object.getOwnPropertyDescriptor(FP, "method").get.call(f);
  const fElems = (f) => [...Object.getOwnPropertyDescriptor(FP, "elements").get.call(f)];
  // the button must submit THIS form (a <button form="other"> inside it would submit another one)
  const owner = (x) => { try { return Object.getOwnPropertyDescriptor(Object.getPrototypeOf(x), "form").get.call(x); } catch (e) { return x.form; } };
  const b = fElems(form).find((x) => (x.getAttribute("type") || (x.tagName === "BUTTON" ? "submit" : "")).toLowerCase() === "submit" && !x.disabled && owner(x) === form)
    || [...form.querySelectorAll("button")].find((x) => owner(x) === form && /log ?in|sign ?in|登录|登陆|submit/i.test((x.innerText || "")));
  if (!b) return pre("no submit button on the login form");
  const target = (b.hasAttribute("formaction") ? b.formAction : "") || fAction(form) || location.href;
  const method = ((b.hasAttribute("formmethod") ? b.formMethod : "") || fMethod(form) || "get").toLowerCase();
  if (method !== "post") return pre("the login form does not use POST");
  if (!inDom(target)) return pre("the login form posts outside the SSO login domains");
  const r = b.getBoundingClientRect();
  if (r.width < 1 || r.height < 1) return pre("the login button is not visible");
  b.click();
  // the page's own click / submit handlers ran synchronously inside b.click(): if they changed the form's
  // method or target, the submission (a navigation not yet started) is cancelled here, before any request.
  // (A script of the SSO page that reads the autofilled values itself is outside what any tool can prevent:
  // the configured SSO domain is trusted, as when the user logs in by hand.)
  let t2 = "", m2 = "";
  try {
    t2 = (b.hasAttribute("formaction") ? b.formAction : "") || fAction(form) || location.href;
    m2 = String((b.hasAttribute("formmethod") ? b.formMethod : "") || fMethod(form) || "get").toLowerCase();
  } catch (e) { m2 = ""; }
  if (m2 !== "post" || !inDom(t2) || !inDom(location.href)) {
    window.stop();
    return { submitted: true, aborted: "the login form changed while it was submitted; stopped" };
  }
  return { submitted: true };
}

async function ssoLogin(task, page) {
  const sso = INPUT.sso || {};
  const localTest = sso.local_test === true;          // only the local test suite sets this; never the wrapper
  // sso_stage "pre": nothing was submitted (the wrapper may return the reserved submission).  It switches to
  // "submit" BEFORE the login button is pressed, so any error or navigation from then on counts as submitted.
  out.sso_submitted = false;
  out.sso_stage = "pre";
  if (!sso.login || !(sso.domains || []).length) { out.status = "refused"; out.error = "sso_login is not enabled for this batch (or its budget is used up); report login_required"; return; }
  const i = await info(page);
  if (!hostIn(i.url, sso.domains)) { out.status = "refused"; out.error = "sso_login is only allowed on the configured institution SSO login domains (HTTPS)"; return; }
  const h = (() => { try { return new URL(i.url).hostname; } catch (e) { return ""; } })();
  out.sso_host = h;
  // Autofill is detected with the browser's own :autofill state of the two fields (no value is read).  Only
  // in the local test mode a non-empty field (a page-provided stand-in) also counts.
  const read = () => page.evaluate(([lt, scanSrc]) => {
    const af = (el) => { try { return el.matches(":autofill") || el.matches(":-webkit-autofill"); } catch (e) { return false; } };
    // MFA / verification step: the same scan as every other guard (credScan), plus captcha widgets
    const mfa = (new Function("return (" + scanSrc + ")")())().mfa
      || !!document.querySelector("iframe[src*='captcha'], .g-recaptcha, .h-captcha, .cf-turnstile");
    const pwd = document.querySelector("form input[type=password]") || document.querySelector("input[type=password]");
    if (!pwd) return { hasPwd: false, mfa: mfa };
    const form = pwd.form || pwd.closest("form") || document;
    const user = form.querySelector("input[autocomplete=username], input[type=text], input[type=email], input:not([type])");
    const filled = (el) => af(el) || (lt && el.value.length > 0);
    return { hasPwd: true, hasUser: !!user, mfa: mfa, pwFilled: filled(pwd), userFilled: !!user && filled(user) };
  }, [localTest, credScan.toString()]).catch(() => ({ hasPwd: false }));
  let st = await read();
  if (st.mfa) return handoff(task, page, "MFA / verification code on the SSO login page", h, "mfa", i.title);
  if (!st.hasPwd || !st.hasUser) return handoff(task, page, "not a recognizable SSO login form", h, "login", i.title);
  if (!st.pwFilled || !st.userFilled) {
    // the page may still be loading: look once more (no click - a click could submit), then hand over
    await sleep(2000);
    st = await read();
    if (st.mfa) return handoff(task, page, "MFA / verification code on the SSO login page", h, "mfa", i.title);
    if (!st.pwFilled || !st.userFilled) return handoff(task, page, "the browser did not autofill the login form", h, "login", i.title);
  }
  // ONE evaluation checks and submits (no later, separate click): it runs through CDP with a user gesture (the
  // browser releases autofilled values to the page on a user activation), re-checks that the page is still an
  // HTTPS configured SSO host, both fields are still autofilled, there is no MFA / captcha, the form posts
  // (POST only - never GET, which would put the credentials into a URL) to a browser-resolved target
  // (form.action / button.formAction; <base href> included) on such a host - and only then clicks the button
  // (button.click(): the page's own click / submit handlers run as for a real click).
  const expr = "(" + ssoSubmit.toString() + ")(" + JSON.stringify([sso.domains, localTest, credScan.toString(),
                                                                    pageInfo.toString(), CHALLENGE.source]) + ")";
  out.sso_stage = "submit";
  out.sso_submitted = true;                          // from here on the outcome counts as a submission
  let chk = null;
  await withDownload(task, page, async () => {
    const r = await page.cdp("Runtime.evaluate", { expression: expr, returnByValue: true, userGesture: true, awaitPromise: false },
                             { timeout: 15000 });
    chk = (r && r.result && r.result.value) || { refused: "the login form could not be checked" + (r && r.exceptionDetails ? " (exception)" : "") };
  }).catch((e) => { chk = { refused: "the login form could not be checked: " + String(e).slice(0, 80), unknown: true }; });
  if (!chk || !chk.submitted) {
    if (chk && chk.refused && !chk.unknown && chk.stage === "pre") { out.sso_stage = "pre"; out.sso_submitted = false; }
    out.status = "refused"; out.error = (chk && chk.refused) || "not submitted"; return;
  }
  if (chk.aborted) {                                 // counted as a (failed) submission: no more logins this batch
    out.sso = "fail"; out.note = chk.aborted; out.status = "ok";
    out.page = { url: bareUrl(i.url), title: i.title, login_form: true };
    return;
  }
  // wait for the login page to go away (never resubmit; a failure ends the logins of this batch)
  const t0 = Date.now();
  for (;;) {
    await sleep(1500);
    const i2 = await info(page);
    if (isChallenge(i2)) { out.sso = "handoff"; return handoff(task, page, "challenge after the SSO login", h, "challenge", i2.title); }
    const st2 = await read();
    if (st2.mfa) { out.sso = "handoff"; return handoff(task, page, "MFA / verification code after the SSO login", h, "mfa", i2.title); }
    if (!st2.hasPwd && hostIn(i2.url, sso.domains) && await loginPage(page)) {
      // still on the login server with other credential fields (a second step): the user finishes it
      out.sso = "handoff"; return handoff(task, page, "a further login step on the SSO server", h, "mfa", i2.title);
    }
    if (!st2.hasPwd) {
      const i3 = await after(page);                  // the landed page must be an allowed / SSO host
      if (out.left_allowed) { out.sso = "fail"; out.status = "ok"; return; }
      out.sso = "ok"; out.page = { url: bareUrl(i3.url), title: i3.title, pdf: /pdf/i.test(i3.ctype) }; out.status = "ok"; return;
    }
    if (Date.now() - t0 > (sso.wait_ms || 35000)) break;
  }
  out.sso = "fail";                                   // still on the login page: failed, never retried
  out.page = { url: bareUrl(i.url), title: i.title, login_form: true };
  out.status = "ok";
}

try {
  let task;
  if (INPUT.action === "handoff_status" && (INPUT.handoff_state || {}).active) {
    // a user-held space poisons the whole process's output once taskSpace() ran: for the hand-off
    // poll, go straight to the documented resume path (takeOverTaskSpace) instead
    task = await takeOverTaskSpace(INPUT.space);
  } else {
    task = await taskSpace(INPUT.space);
  }
  out.space = task.spaceId;
  if (INPUT.action === "finish") {
    try { await task.finish({ keep: [] }); }
    catch (e) {
      if (/with the user/.test(String(e))) { const t2 = await takeOverTaskSpace(INPUT.space); await t2.finish({ keep: [] }); }
      else throw e;
    }
    out.status = "ok";
  } else {
    const page = await currentPage(task);
    // whatever the action: a current page outside the allowed domains (a late redirect, a timed navigation)
    // is reset before anything is read from it or done on it
    if (!["handoff_status", "tabs", "switch"].includes(INPUT.action)) {
      const i0 = await info(page);
      // (only web pages count: about:, chrome:// new-tab and error pages are the browser's own)
      const allowedHere = !/^https?:/i.test(i0.url || "") || hostOk(i0.url);
      if (!allowedHere) {
        let hh = ""; try { hh = new URL(i0.url).hostname; } catch (e) {}
        await page.goto("about:blank", { timeout: 5000 }).catch(() => {});
        out.reset_foreign = hh;                     // reported; the action itself then runs on about:blank
      }
    }
    if (INPUT.action === "look") {
      await look(task, page);
    } else if (INPUT.action === "open") {
      const url = String(INPUT.arg || "");
      if (!hostOk(url)) { out.status = "refused"; out.error = "URL host is not in the allowed domains of this paper"; }
      else {
        await withDownload(task, page, () => page.goto(url, { timeout: INPUT.nav_timeout_ms, waitUntil: "domcontentloaded" }));
        const i = await after(page);
        if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
        out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
        if (out.status !== "handoff") out.status = "ok";
      }
    } else if (["click", "hover", "select", "type"].includes(INPUT.action)) {
      const parts = String(INPUT.arg || "").trim().split(" ");
      const k = /^\d{1,3}$/.test(parts[0]) ? parseInt(parts[0], 10) : -1;
      const value = parts.slice(1).join(" ");
      const needsValue = INPUT.action === "select" || INPUT.action === "type";
      if (k < 0 || (needsValue && !value) || (!needsValue && value)) { out.status = "refused"; out.error = `usage: ${INPUT.action} I${needsValue ? " VALUE" : ""}`; }
      else await act(task, page, INPUT.action, k, value);
    } else if (INPUT.action === "click_xy") {
      const parts = String(INPUT.arg || "").split(" ").map((x) => parseInt(x, 10));
      const i0 = await info(page);
      if (isChallenge(i0)) { out.status = "refused"; out.error = "verification page: do not interact"; await postCheck(task, page, i0); }
      else if (i0.login_form || await credPage(page)) { out.status = "refused"; out.error = "login page: coordinate clicks are not allowed (sso_login or hand-off only)"; }
      else if (!(parts[0] >= 0) || !(parts[1] >= 0)) { out.status = "refused"; out.error = "usage: click_xy X Y"; }
      else {
        // the element at the point is checked (same rules as `click`) right before the native click
        const pre = await page.evaluate(([x, y, allow, chSrc]) => {
          const ok = (u) => { try { const v = new URL(u, location.href); if (!/^https?:$/.test(v.protocol)) return false;
            const h = v.hostname.toLowerCase(); return allow.some((d) => h === d || h.endsWith("." + d)); } catch (e) { return false; } };
          const C = new RegExp(chSrc, "i");
          const isVerify = (d) => { const body = (d.body && d.body.innerText || "").slice(0, 300);
            return C.test(d.title || "") || (/^\s*$/.test(d.title || "") && C.test(body))
              || !!d.querySelector(".cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha, iframe[src*='captcha'], iframe[src*='challenges.cloudflare.com']"); };
          // the real target: through open shadow roots and SAME-ORIGIN frames (each level checked for a
          // verification page); a cross-origin frame stays the target element
          let doc = document, px = x, py = y, el = null;
          for (let n = 0; n < 8; n++) {
            if (isVerify(doc)) return { refused: "verification page: do not interact" };
            el = doc.elementFromPoint(px, py);
            if (!el) return { refused: "nothing at that point" };
            for (let m = 0; m < 8 && el.shadowRoot; m++) { const r = el.shadowRoot.elementFromPoint(px, py); if (!r || r === el) break; el = r; }
            if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
              let inner = null; try { inner = el.contentDocument; } catch (e) { inner = null; }
              if (inner) { const fr = el.getBoundingClientRect(); px -= fr.left; py -= fr.top; doc = inner; continue; }
            }
            break;
          }
          // a label forwards the click to its control: the hit element (its link) AND the control are checked
          const hit = el;
          {
            const ha = hit.closest && hit.closest("a[href]");
            if (ha) { const h = ha.getAttribute("href") || ""; if (!/^\s*javascript:/i.test(h) && !ok(ha.href)) return { refused: "link target is outside the allowed domains" }; }
          }
          const lb = el.closest && el.closest("label");
          if (lb && lb.control) el = lb.control;
          if (el.tagName === "INPUT" && (el.getAttribute("type") || "").toLowerCase() === "password") return { refused: "password field: do not interact" };
          const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
          if (el.tagName === "IFRAME" || el.tagName === "FRAME") {
            const src = el.src || "";
            if (W.test(src + " " + (el.title || "")) || /captcha|challenges\.cloudflare\.com|turnstile/i.test(src)) return { refused: "verification widget: do not interact" };
            if (src && !ok(src)) return { refused: "the frame at that point is outside the allowed domains" };
            return { ok: true, frame: true };          // a click inside a frame of an allowed host (native)
          }
          if (W.test([el.id, el.className && String(el.className), el.getAttribute("name"), el.getAttribute("aria-label")].join(" "))
            || el.closest(".cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha, iframe[src*='captcha']")) return { refused: "verification widget: do not interact" };
          if (el.tagName === "INPUT" && (el.getAttribute("type") || "").toLowerCase() === "password") return { refused: "password field: do not interact" };
          const FP = HTMLFormElement.prototype;
          const fAction = (f) => Object.getOwnPropertyDescriptor(FP, "action").get.call(f);
          const fElems = (f) => [...Object.getOwnPropertyDescriptor(FP, "elements").get.call(f)];
          const fa = el.closest("button, input");
          const own = (v) => { try { return Object.getOwnPropertyDescriptor(Object.getPrototypeOf(v), "form").get.call(v); } catch (e) { return v.form; } };
          const form = (fa && own(fa)) || (el.form !== undefined ? own(el) : null) || el.closest("form");
          if (form && (fElems(form).some((v) => (v.getAttribute("type") || "").toLowerCase() === "password") || Element.prototype.querySelector.call(form, "input[type=password]"))) return { refused: "login form: credentials are not entered by this tool" };
          const a = el.closest("a[href]");
          if (a) { const h = a.getAttribute("href") || ""; if (!/^\s*javascript:/i.test(h) && !ok(a.href)) return { refused: "link target is outside the allowed domains" }; }
          if (form && !ok((fa && fa.hasAttribute("formaction") ? fa.formAction : "") || fAction(form) || location.href)) return { refused: "form target is outside the allowed domains" };
          const vis = (v) => ((v.innerText || "") + " " + (v.tagName === "INPUT" && ["submit", "button", "reset"].includes((v.getAttribute("type") || "").toLowerCase()) ? (v.value || "") : "")
                               + " " + (v.getAttribute("aria-label") || "") + " " + (v.title || "")).slice(0, 200);
          if (W.test(vis(el)) || (hit !== el && W.test(vis(hit)))) return { refused: "verification widget: do not interact" };
          return { ok: true };
        }, [parts[0], parts[1], INPUT.allow, CHALLENGE.source]).catch(() => null);
        if (!pre) { out.status = "refused"; out.error = "the point could not be checked; use click on a look item"; }
        else if (pre.refused) { out.status = "refused"; out.error = pre.refused; }
        else {
          await withDownload(task, page, () => page.mouse.click(parts[0], parts[1]));
          await sleep(1500);
          const i = await after(page);
          if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
          out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
          if (out.status !== "handoff") out.status = "ok";
        }
      }
    } else if (INPUT.action === "scroll") {
      const a = String(INPUT.arg || "").split(/\s+/);
      const d = a[0] === "up" ? -1 : 1;
      const px = Math.min(5000, Math.max(50, parseInt(a[1] || "700", 10) || 700));
      const i0 = await info(page);
      if (isChallenge(i0)) { out.status = "refused"; out.error = "verification page: do not interact"; await postCheck(task, page, i0); }
      else {
        const sy = () => page.evaluate(() => Math.round(window.scrollY)).catch(() => 0);
        const from = await sy();
        await page.evaluate(([d, px]) => window.scrollBy(0, d * px), [d, px]).catch(() => {});
        await sleep(300);
        let to = await sy();
        if (to === from) {                                    // the window cannot scroll: wheel over the page
          await page.mouse.move(400, 400).catch(() => {});
          await page.mouse.wheel(0, d * px).catch(() => {});
          await sleep(300);
          to = await sy();
        }
        out.scroll = { from: from, to: to };
        const i = await after(page);                          // scrolling may trigger a navigation
        if (!out.blocked && !out.left_allowed && i.url !== i0.url) await postCheck(task, page, i);
        if (out.status !== "handoff") out.status = "ok";
      }
    } else if (INPUT.action === "screenshot") {
      const i = await info(page);
      if (!i.url || isChallenge(i)) { out.status = "refused"; out.error = "no screenshot of this page"; }
      else if (i.login_form || await credPage(page)) { out.status = "refused"; out.error = "the page has (or may have) credential fields; screenshots are disabled on login pages"; }
      else {
        // every field is masked (top document, open shadow roots, same-origin frames).  Fields the mask cannot
        // reach (cross-origin frames, closed shadow roots) are counted in the snapshot: if there are any, no
        // screenshot is taken
        const snap = await safeSnapshot(page, { scope: "full_page" });
        // fields per area from the (reduced) snapshot: top document vs. inside any iframe
        let topFields = 0, frameFields = 0;
        {
          const stack = [];
          for (const ln of String(snap || "").split("\n")) {
            if (!ln.trim()) continue;
            const ind = ln.length - ln.trimStart().length;
            while (stack.length && stack[stack.length - 1] >= ind) stack.pop();
            if (/^\s*iframe\b/.test(ln)) { stack.push(ind); continue; }
            if (/^\s*(textbox|searchbox|combobox|spinbutton|password|listbox|slider)\b/.test(ln)) { if (stack.length) frameFields++; else topFields++; }
          }
        }
        const MASK = "input:not([type=submit]):not([type=button]):not([type=image]):not([type=reset]),textarea,select,[contenteditable]{color:transparent!important;"
          + "-webkit-text-security:disc!important;text-shadow:none!important;background-image:none!important;caret-color:transparent!important}"
          + "input::placeholder,textarea::placeholder{color:#999!important}";
        // masks every reachable document; counts text-like fields per area and whether any frame is unreachable
        const cov = await page.evaluate((css) => {
          const TXT = "input:not([type]),input[type=text],input[type=search],input[type=email],input[type=tel],input[type=url],input[type=number],input[type=password],textarea,select,[contenteditable]";
          const res = { top: 0, frames: 0, unreachable: 0 };
          const add = (root, inFrame) => {
            const st = document.createElement("style"); st.className = "__tbsgen_mask"; st.textContent = css;
            (root.documentElement || root).appendChild(st);
            const n = [...root.querySelectorAll(TXT)].filter((x) => x.getClientRects().length > 0
              && getComputedStyle(x).visibility !== "hidden" && getComputedStyle(x).display !== "none").length;   // shown only
            if (inFrame) res.frames += n; else res.top += n;
            for (const h of root.querySelectorAll("*")) if (h.shadowRoot) add(h.shadowRoot, inFrame);
            for (const fr of root.querySelectorAll("iframe, frame")) {
              let d = null; try { d = fr.contentDocument; } catch (e) { d = null; }
              if (d) add(d, true); else res.unreachable++;
            }
          };
          try { add(document, false); } catch (e) { return null; }
          return res;
        }, MASK).catch(() => null);
        // refused unless every field the browser shows is in a masked document: no snapshot, a field in an
        // unreachable (cross-origin) frame, or more fields than the masked documents hold (closed shadow roots)
        // (a field in a cross-origin frame can never be masked: any frame field while such a frame exists refuses
        // - other frames' fields are never counted against it)
        // every field the snapshot shows must be one the mask reached: the counts per area must be EQUAL (a field
        // the mask cannot see - closed shadow root, cross-origin frame - makes them differ and refuses)
        const uncovered = !snap || !cov || (cov.unreachable > 0 && frameFields > 0)
          || topFields !== cov.top || frameFields !== cov.frames;
        const masked = uncovered ? -1 : 1, snapFields = 0;
        if (masked < 0 || snapFields > masked) {
          await page.evaluate(() => { const rm = (r) => { try { r.querySelectorAll(".__tbsgen_mask").forEach((x) => x.remove());
            for (const h of r.querySelectorAll("*")) if (h.shadowRoot) rm(h.shadowRoot);
            for (const fr of r.querySelectorAll("iframe")) { try { if (fr.contentDocument) rm(fr.contentDocument); } catch (e) {} } } catch (e) {} }; rm(document); }).catch(() => {});
          out.status = "refused"; out.error = "the page has fields that cannot be masked (cross-origin frame / closed component); use look instead";
        } else {
          const f = `${INPUT.files_dir}/shot-${Date.now()}-${Math.floor(Math.random() * 1e5)}.png`;
          await page.screenshot({ path: f });
          await page.evaluate(() => { const rm = (r) => { try { r.querySelectorAll(".__tbsgen_mask").forEach((x) => x.remove());
            for (const h of r.querySelectorAll("*")) if (h.shadowRoot) rm(h.shadowRoot);
            for (const fr of r.querySelectorAll("iframe")) { try { if (fr.contentDocument) rm(fr.contentDocument); } catch (e) {} } } catch (e) {} }; rm(document); }).catch(() => {});
          out.shot = f;
          out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype) };
          out.status = "ok";
        }
      }
    } else if (INPUT.action === "back") {
      // only to the previous entry when it is same-origin and on an allowed host (Navigation API: entries()
      // lists only same-origin entries) - checked and done in one evaluation; anything else is refused
      let r = null;
      await withDownload(task, page, async () => {
        r = await page.evaluate((allow) => {
          const ok = (u) => { try { const v = new URL(u); if (!/^https?:$/.test(v.protocol)) return false;
            const h = v.hostname.toLowerCase(); return allow.some((d) => h === d || h.endsWith("." + d)); } catch (e) { return false; } };
          if (typeof navigation === "undefined" || !navigation.currentEntry) return { refused: "back is not available on this page" };
          const cur = navigation.currentEntry.index;
          const prev = navigation.entries().find((e) => e.index === cur - 1);
          if (!prev) return { refused: "the previous page is on another site (or there is none); use open" };
          if (!ok(prev.url)) return { refused: "the previous page is outside the allowed domains" };
          navigation.back();
          return { ok: true };
        }, INPUT.allow).catch((e) => (/context|destroyed|navigat/i.test(String(e)) ? { ok: true } : { refused: "back failed" }));
      });
      if (r && r.refused) { out.status = "refused"; out.error = r.refused; }
      else {
        await sleep(1500);
        const i = await after(page);
        if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
        out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
        if (out.status !== "handoff") out.status = "ok";
      }
    } else if (INPUT.action === "sso_login") {
      await ssoLogin(task, page);
    } else if (INPUT.action === "tabs") {
      const tabs = await task.tabs();
      out.tabs = tabs.map((t) => ({ label: t.label, url: cleanUrl(t.url || "").slice(0, 200),
                                    title: hostOk(t.url || "") ? (t.title || "").slice(0, 80) : "",
                                    active: !!t.active, opened_by: t.openedBy || "", allowed: hostOk(t.url || "") }));
      out.status = "ok";
    } else if (INPUT.action === "switch") {
      const label = String(INPUT.arg || "").trim();
      const tabs = await task.tabs();
      const t = tabs.find((x) => x.label === label);
      if (!t) { out.status = "refused"; out.error = "no such page; run tabs"; }
      else if ((t.openedBy || "") === "user") { out.status = "refused"; out.error = "this page belongs to the user; it is not switched to"; }
      else if (t.url && !t.url.startsWith("about:") && !hostOk(t.url)) { out.status = "refused"; out.error = "that page is outside the allowed domains"; }
      else {
        out.page_label = label;
        const pages = await task.pages();
        const pg = pages.find((x) => x.label === label) || pages[0];
        const i = await info(pg);
        out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
        out.status = "ok";
      }
    } else if (INPUT.action === "wait_for") {
      const a = String(INPUT.arg || "").split(" ");
      const what = a[0], pat = a[1] || "";
      const secs = Math.min(60, Math.max(1, parseInt(a[2] || "15", 10) || 15));
      if (!pat || /["'`$;&|<>\\]/.test(pat)) { out.status = "refused"; out.error = "usage: wait_for url SUBSTR|sel CSS [S]"; }
      else if (what === "url") {
        const t0 = Date.now();
        // matched against origin + path only: the query / fragment (tokens) must not become a guessing oracle
        const cur = async () => cleanUrl((await page.info().catch(() => ({}))).url || "");
        let u = await cur();
        while (!u.includes(pat) && Date.now() - t0 < secs * 1000) { await sleep(1000); u = await cur(); }
        out.waited = { what: "url", ok: u.includes(pat), url: u.slice(0, 200) };
        out.status = "ok";
        const iw = await after(page);
        if (!out.blocked && !out.left_allowed) await postCheck(task, page, iw);
        if (out.left_allowed) out.waited.url = "about:blank";
        if (out.status !== "handoff") out.status = "ok";
      } else if (what === "sel") {
        const r = await page.waitForSelector(pat, { timeout: secs * 1000 }).then(() => true).catch(() => false);
        out.waited = { what: "sel", ok: r };
        out.status = "ok";
        const iw = await after(page);
        if (!out.blocked && !out.left_allowed) await postCheck(task, page, iw);
        if (out.status !== "handoff") out.status = "ok";
      } else { out.status = "refused"; out.error = "usage: wait_for url SUBSTR|sel CSS [S]"; }
    } else if (INPUT.action === "save_pdf") {
      const i = await info(page);
      if (!/pdf/i.test(i.ctype)) { out.status = "refused"; out.error = "the current page is not a PDF document"; }
      else if (!hostOk(i.url)) { out.status = "refused"; out.error = "PDF host is not in the allowed domains"; }
      else {
        const f = fileName();
        const r = await page.fetch(i.url, { saveAs: f, timeout: INPUT.download_timeout_ms });
        out.http = r.status;
        if (r.url && !hostOk(r.url)) {
          await (await import("node:fs/promises")).unlink(f).catch(() => {});
          out.rejected = [f];                       // the wrapper deletes it too and never validates it
          out.status = "refused"; out.error = "PDF redirected outside the allowed domains";
          emit(); process.exit(0);
        }
        if (r.status === 429 || await savedIsChallenge(f)) { out.blocked = r.status === 429 ? "HTTP 429" : "bot-wall page"; out.http429 = r.status === 429; }
        else if (r.ok) out.files.push(f);
        out.status = "ok";
      }
    } else if (INPUT.action === "handoff_status") {
      const hs = INPUT.handoff_state || {};
      if (!hs.active) { out.handoff_check = "none"; out.status = "ok"; }
      else {
        const left = hs.deadline_ms - Date.now();
        if (left <= 0) { out.handoff_check = "expired"; out.status = "ok"; }
        else {
          // the poll took the space back (see above); read the page: if the user finished, the
          // challenge/login is gone; if not, the page still shows it and the hand-off is re-armed
          const pages = await task.pages();
          const page2 = pages.find((x) => x.label === hs.page_label) || pages[0];   // the page that was handed over
          if (!page2) { out.handoff_check = "cleared"; out.status = "ok"; }
          else {
            const i = await settle(page2);
            out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
            // cleared only when no challenge, no login / MFA form is left on that page
            if (isChallenge(i) || i.login_form || await loginPage(page2)) {
              // the user has not finished: hand the space back and keep waiting.  A verification page seen
              // now is reported (the wrapper upgrades a login hand-off to a challenge: cooled on timeout)
              if (isChallenge(i)) out.handoff_kind_now = "challenge";
              await task.handOff();
              out.handoff_check = "waiting";
              out.handoff_left_s = Math.round(left / 1000);
            } else {
              out.handoff_check = "cleared";
              if (i.url && !i.url.startsWith("about:") && !hostOk(i.url)) {   // never resume on a foreign page
                await page2.goto("about:blank", { timeout: 5000 }).catch(() => {});
                out.left_allowed = (() => { try { return new URL(i.url).hostname; } catch (e) { return ""; } })();
                out.page = { url: "about:blank", title: "" };
              }
            }
            out.status = "ok";
          }
        }
      }
    } else {
      out.status = "refused"; out.error = "unknown action";
    }
  }
} catch (e) {
  const msg = String(e && e.message || e);
  if (/with the user/.test(msg)) { out.status = "paused"; out.error = "this paper's browser workspace is with the user (handoff); run handoff_status"; }
  else { out.status = "error"; out.error = msg.slice(0, 300); }
}
emit();
process.exit(0);
