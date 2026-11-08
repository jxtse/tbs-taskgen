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
//   look [OFFSET]                 page summary: URL, title, PDF?, challenge?, login form?, text excerpt and a
//                                 numbered list of links / buttons / fields of the page's own DOM (hidden
//                                 dropdown items included; every iframe is one `frame` item - clicking it
//                                 opens the frame's checked URL as the page);
//                                 OFFSET paginates the item list
//   open URL                      navigate (URL host must be in INPUT.allow); downloads and new tabs are saved /
//                                 reported
//   click I                       click item I of the last `look` list (same numbering; run `look` first)
//   tabs                          list the TaskSpace's pages (label, URL, allowed)
//   switch LABEL                  make page LABEL the current page (never a page opened by the user)
//   wait_for url SUBSTR|sel CSS [S]   wait until the URL contains SUBSTR / a CSS selector matches (<= 60 s)
//   save_pdf                      the current page is a PDF document: save it
//   handoff_status                poll a running hand-off (the tool handed the page to the user)
//   finish                        close the TaskSpace
// NARROWED (round 11): the agent only reads pages, navigates and clicks - there is no hover, select or type,
// and no form field is ever changed by this tool (institution selection / WAYF pages go to the user).
// Rules enforced here: no form field is changed (fields cannot be clicked; a form with any editable control is
// never submitted), no interaction with verification pages / widgets, no reading of input values, storage or
// the browser's page snapshot (it carries field values), no screenshots, no native mouse / wheel input at all,
// no click at all on a page with ANY login signal (credScan, checked in the same page evaluation that acts,
// together with the looked-at URL and the allowed domains), login forms are never submitted (every login page - the institution's SSO login too - is handed to the user, whose
// browser autofills it), no action inside any iframe (a frame is one item whose `click` opens its checked URL
// as the page), navigation targets are checked before and the
// landed URL is checked after (a page that ends up outside INPUT.allow is reset to about:blank), files are
// only written under INPUT.files_dir, and verification / login pages are handed to the user (task.handOff)
// instead of being solved.
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
    // NARROWED (round 9): every iframe - same-origin too - is ONE `frame` item (its URL); nothing inside a
    // frame is listed or acted on (`click` on the item opens the frame's checked URL as the page, where every
    // check runs on the top document)
    for (const f of doc.querySelectorAll("iframe")) {
      if (f.src && /^https?:/i.test(f.src) && pairs.length < 240) pairs.push([f, frame]);
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
async function items(page) {
  return await page.evaluate(([a, w]) => (new Function("return (" + a + ")")())(w),
                             [itemList.toString(), domWalk.toString()]).catch(() => []);
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
  if (wayf) sem = true;
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
  if (sc.login && !(opts && opts.no_login_handoff)) {
    if (INPUT.handoff && INPUT.handoff.enabled) return handoff(task, page, sc.mfa ? "verification code" : (sc.wayf ? "institution selection" : "login form"), h, sc.mfa ? "mfa" : "login", i.title);
    out.login_form = true;                       // reported; the agent books `login_required`
  }
  return i;
}

// ---------- actions -----------------------------------------------------------------------------
function parseLookArg(arg) {
  const x = String(arg || "").trim();
  return { offset: /^\d{1,4}$/.test(x) ? parseInt(x, 10) : 0 };
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
  const { offset } = parseLookArg(INPUT.arg);
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
  const all = r.items || [];
  const pageItems = all.slice(offset, offset + LIMIT).map((it, k) => ({
    i: offset + k, kind: it.kind, text: it.text, href: it.href ? cleanUrl(it.href).slice(0, 200) : "",
    allowed: it.href ? hostOk(it.href) : undefined,
    hidden: !!it.hidden, frame: it.frame || 0, ftype: it.ftype || undefined,
    widget: !!it.widget, login: !!it.login,
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

// The one evaluation that finds, checks and clicks item k of the dom list.  `walk` is the
// domWalk function itself; both run inside the page (ego's evaluate takes ONE argument, passed as an array).
function domAct(walkSrc, k, allow, kind, expectUrl, scanSrc, infoSrc, chSrc) {
  const ok = (u) => { try { const x = new URL(u, location.href); if (!/^https?:$/.test(x.protocol)) return false;
    const h = x.hostname.toLowerCase(); return allow.some((d) => h === d || h.endsWith("." + d)); } catch (e) { return false; } };
  // re-checked in the SAME evaluation that acts (the page may have changed since the Node-side checks): still
  // the page the agent looked at, still on an allowed host, still no credential / verification-code fields
  if (kind !== "click") return { refused: "unknown action" };
  if (location.href !== expectUrl) return { refused: "the page changed since `look`; run look again" };
  if (!ok(location.href)) return { refused: "the page is outside the allowed domains" };
  if ((new Function("return (" + scanSrc + ")")())().cred) return { refused: "login / verification-code page: no click here" };
  if ((new Function("return (" + infoSrc + ")")())(chSrc).challenge_page) return { refused: "verification page: do not interact" };
  const pair = new Function("return (" + walkSrc + ")")()()[k];   // domWalk: source -> function -> [el, frame]
  const el = pair && pair[0];
  if (!el) return { refused: "no such item; run look first" };
  const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
  const COOKIE = /^(accept|reject|allow all|reject all|agree|consent|dismiss|close|ok|got it|accept all|manage preferences|接受|拒绝|同意|全部接受|关闭|确定)/i;
  // a form's own properties can be shadowed by its named controls (<input name=action>, name=elements ...):
  // read them through the native getters only (round 11)
  const FP = HTMLFormElement.prototype;
  const fAction = (f) => Object.getOwnPropertyDescriptor(FP, "action").get.call(f);
  const fElems = (f) => [...Object.getOwnPropertyDescriptor(FP, "elements").get.call(f)];
  const fHasPw = (f) => Element.prototype.querySelector.call(f, "input[type=password]") !== null;
  const tag0 = el.tagName.toLowerCase();
  const btnInput = tag0 === "input" && ["submit", "button", "reset", "image"].includes((el.getAttribute("type") || "").toLowerCase());
  const isField = (tag0 === "input" && !btnInput) || tag0 === "textarea" || tag0 === "select";
  // a field's value is never read (it may be an autofilled credential): fields are named by their metadata
  const text = ((isField ? "" : (el.innerText || (btnInput ? el.value : "") || "")) || el.getAttribute("aria-label")
    || el.getAttribute("placeholder") || el.title || "").trim().slice(0, 80);
  if (W.test([el.id, el.className && String(el.className), el.getAttribute("name"), el.getAttribute("aria-label")].join(" "))) return { refused: "verification widget: do not interact" };
  // the control a click really activates: the element itself or its enclosing button / input (a span with
  // role=button inside <button form="f"> submits form f)
  const actv = (el.closest && el.closest("button, input")) || el;
  const form = actv.form || el.form || (el.closest ? el.closest("form") : null);
  const pwdForm = ((f) => !!f && (fElems(f).some((x) => (x.getAttribute("type") || "").toLowerCase() === "password") || fHasPw(f)));
  // nothing inside (or submitting) a form with a password field (also one associated by form="..."): login
  // forms are only ever submitted by the user
  if (pwdForm(form)) return { refused: "login form: credentials are not entered by this tool" };
  // NARROWED (round 11): this tool never changes a form field.  A field item itself is never clicked (a click
  // could focus / toggle / open it), and a form with ANY editable control - free text, select, checkbox,
  // radio, textarea, file ..., rendered or not - is never submitted or clicked inside: it may hold an
  // autofilled identifier or need a choice only the user makes (WAYF, account choosers).  Only forms made of
  // buttons and hidden inputs (e.g. a "Download PDF" form) may be submitted.
  if (isField) return { refused: "form fields are not changed by this tool (an institution selection page is handed to the user; otherwise report login_required)" };
  // a label activates its control: a click on (anything inside) a label of a field is a click on the field
  // (round 12: every click inside any <label> is refused - a label can also activate a submit / reset button
  // of a form elsewhere, around every form check)
  if (el.closest && el.closest("label")) return { refused: "labels are not clicked by this tool (they activate form controls)" };
  const editable = (f) => !!f && fElems(f).some((x) => {
    const xt = x.tagName;
    if (xt === "TEXTAREA" || xt === "SELECT") return true;
    if (xt !== "INPUT") return false;
    return !["hidden", "submit", "button", "reset", "image"].includes((x.getAttribute("type") || "text").toLowerCase());
  });
  if (editable(form)) return { refused: "form with an input field: not submitted by this tool (report login_required)" };
  if (el.closest && el.closest(".cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha, [id*=captcha], [class*=captcha]")) return { refused: "verification widget: do not interact" };
  if (W.test(text)) return { refused: "verification widget: do not interact" };
  const tag = el.tagName.toLowerCase();
  if (tag === "iframe") {
    // a frame (any origin): never acted on inside; `click` opens its (checked) URL as this page
    if (!ok(el.src)) return { refused: "frame URL is outside the allowed domains" };
    location.assign(el.src);                                   // navigated here, in the checking evaluation
    return { opened: el.src, text: text };
  }
  {
    // the browser-resolved targets (respects <base href> and form= / formaction)
    // (button.formAction is the document URL when the button has no formaction attribute: use form.action then)
    const atag = actv.tagName.toLowerCase();
    const aBtn = atag === "button" || (atag === "input" && ["submit", "image"].includes((actv.getAttribute("type") || "").toLowerCase()));
    if (form && aBtn && !ok((actv.hasAttribute("formaction") ? actv.formAction : "") || fAction(form) || location.href)) return { refused: "form target is outside the allowed domains" };
    if (form && !aBtn && !ok(fAction(form) || location.href)) return { refused: "form target is outside the allowed domains" };
    // the link a click really follows: the element or its enclosing <a href> (a span inside a link)
    const link = tag === "a" ? el : (el.closest && el.closest("a[href]"));
    if (link) {
      const h = link.getAttribute("href") || "";
      // every link but javascript: is checked by its browser-resolved target - also "#frag" (a <base href> can
      // point it at another host)
      if (!/^\s*javascript:/i.test(h) && !ok(link.href)) return { refused: "link target is outside the allowed domains" };
      // a new-tab link (top document or a frame): open its checked target in this page (no native click -
      // a native click lands on whatever is on top at that point)
      // (navigated here, in the checking evaluation - no later step)
      if ((link.getAttribute("target") || "").toLowerCase() === "_blank" && !h.startsWith("#") && !/^javascript:/i.test(h)) {
        location.assign(link.href);
        return { opened: link.href, text: text };
      }
      // a named target window ("viewer"): also opened here (checked) instead of by a click
      const tg = (link.getAttribute("target") || "").trim();
      if (tg && !/^_(self|top|parent)$/i.test(tg) && !h.startsWith("#") && !/^javascript:/i.test(h)) { location.assign(link.href); return { opened: link.href, text: text }; }
    }
    if ((el.hasAttribute("formaction") && !ok(el.formAction)) || (actv.hasAttribute("formaction") && !ok(actv.formAction))) return { refused: "form target is outside the allowed domains" };
    el.click();
    return { clicked: text };
  }
  return { refused: "unknown action" };
}

async function act(task, page, k) {
  const kind = "click";
  // the page must be the one the agent last looked at.  `seen` - the URL whose hash is the look_key - is
  // passed into domAct, which compares it with location.href in the same evaluation that acts
  if (!INPUT.look_key) { out.status = "refused"; out.error = "run look first"; return; }
  const ip = await info(page);
  const seen = ip.url;
  if (await urlKey(seen) !== INPUT.look_key) { out.status = "refused"; out.error = "the page changed since `look`; run look again"; return; }
  const all = await items(page);
  if (!(k >= 0) || k >= all.length) { out.status = "refused"; out.error = "no such item; run look first"; return; }
  const it = all[k];
  if (isChallenge(ip)) { out.status = "refused"; out.error = "verification page: do not interact"; await postCheck(task, page, ip); return; }
  // a page with credential / MFA fields (anywhere: frames, shadow roots, identity-provider frames) is never
  // interacted with: the hand-off only
  if (ip.login_form || await credPage(page)) {
    out.status = "refused";
    out.error = "login / verification-code page: no click here (it is handed to the user, otherwise report login_required)";
    return;
  }
  if (it.widget || WIDGET.test(it.text || "")) { out.status = "refused"; out.error = "verification widget: do not interact"; return; }
  if (it.login) { out.status = "refused"; out.error = "login form: credentials are not entered by this tool (report login_required)"; return; }
  const domRun = () => page.evaluate(([src, walk, k, allow, kind, u, scanSrc, infoSrc, chSrc]) => {   // ONE arg for ego
    const f = new Function("return (" + src + ")")();        // domAct; walk is domWalk, both plain functions
    return f(walk, k, allow, kind, u, scanSrc, infoSrc, chSrc);
  }, [domAct.toString(), domWalk.toString(), k, INPUT.allow, kind, seen, credScan.toString(),
      pageInfo.toString(), CHALLENGE.source]);
  {
    {
      await withDownload(task, page, async () => {
        // (a navigation started inside the evaluation may end its context before it returns: navigated)
        const r = await domRun().catch((e) => (/context|destroyed|navigat|detached/i.test(String(e)) ? { opened: "?" } : null));
        if (r && r.refused) { out.status = "refused"; out.error = r.refused; }
        else if (!r) { out.status = "error"; out.error = "the page changed during the click (no result)"; }
      });
      if (out.status === "refused") return;
      await sleep(1500);
      const i = await after(page);
      if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
      out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
      if (out.status !== "handoff") out.status = "ok";
      return;
    }
  }
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
    } else if (INPUT.action === "click") {
      const arg = String(INPUT.arg || "").trim();
      if (!/^\d{1,3}$/.test(arg)) { out.status = "refused"; out.error = "usage: click I"; }
      else await act(task, page, parseInt(arg, 10));
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
