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
//                                 dropdown items and same-origin iframes included; a cross-origin iframe is
//                                 one `frame` item - clicking it opens the frame's checked URL as the page);
//                                 OFFSET paginates the item list
//   open URL                      navigate (URL host must be in INPUT.allow); downloads and new tabs are saved /
//                                 reported
//   click I                       click item I of the last `look` list (same numbering; run `look` first)
//   hover I                       move the mouse onto item I (opens CSS / JS dropdown menus)
//   scroll up|down [PX]           scroll the page
//   select I VALUE                choose VALUE (option value or label) of select item I
//   type I TEXT                   type TEXT into text item I - only the whitelisted institution names in
//                                 INPUT.type_allow, never into anything that can hold credentials
//   tabs                          list the TaskSpace's pages (label, URL, allowed)
//   switch LABEL                  make page LABEL the current page (never a page opened by the user)
//   back                          go back one history step
//   wait_for url SUBSTR|sel CSS [S]   wait until the URL contains SUBSTR / a CSS selector matches (<= 60 s)
//   save_pdf                      the current page is a PDF document: save it
//   sso_login                     on a configured institution SSO login domain (INPUT.sso.domains): submit the
//                                 browser-autofilled login form.  The values of credential fields are never
//                                 read, never returned and never logged; no retry after a failure; any
//                 captcha / slider / SMS / MFA code or missing autofill hands the page to the user instead.
//   handoff_status                poll a running hand-off (the tool handed the page to the user)
//   finish                        close the TaskSpace
// Rules enforced here: nothing is typed except whitelisted institution names into checked plain text fields,
// no interaction with verification pages / widgets, no reading of input values, storage or the browser's
// page snapshot (it carries field values), no screenshots, no native mouse clicks except sso_login's checked
// Login button, no click / hover / select / type at all on a page with credential or MFA fields, login forms
// are never submitted except by `sso_login` with browser autofill, navigation targets are checked before and the
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
const hostIn = (u, domains) => {
  let h, pr;
  try { const x = new URL(u); pr = x.protocol; h = x.hostname.toLowerCase(); } catch (e) { return false; }
  // HTTPS only, except loopback hosts and *.localhost (local tests); real SSO login domains are never local
  // HTTPS only.  INPUT.sso.local_test (set only by the local test suite, never by the batch wrapper) admits
  // exact loopback names; "127.0.0.1.example.org" is a public host either way
  const local = h === "127.0.0.1" || h === "localhost" || h.endsWith(".localhost");
  if (pr !== "https:" && !(local && INPUT.sso && INPUT.sso.local_test === true)) return false;
  return domains.some((d) => h === d || h.endsWith("." + d));
};

// Everything printed goes through emit(): every URL - a whole value or a URL inside a longer string (error
// messages) - loses its query, fragment and userinfo, whatever its host (OAuth / SAML / SSO callbacks carry
// tokens in many spellings; the agent follows links with `click I`, which uses the page's own href), and the
// raw result is never printed.  The wrapper keeps `look_url` for its own page-change check (same treatment).
function cleanUrl(u) {
  try { const x = new URL(u); return /^(https?|ftp|wss?):$/.test(x.protocol) ? x.origin + x.pathname : x.protocol; }
  catch (e) { return String(u).split(/[?#]/)[0]; }
}
const URL_IN = /\b[a-z][a-z0-9+.-]*:\/\/[^\s"'<>`]+/gi;
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
    for (const f of doc.querySelectorAll("iframe")) {
      let inner = null;
      try { inner = f.contentDocument; } catch (e) { inner = null; }
      if (inner) walk(inner, frame + 1);
      else if (f.src && pairs.length < 240) pairs.push([f, frame]);   // cross-origin: one `frame` item (its URL)
    }
  };
  walk(document, 0);
  return pairs;
}

// The numbered item list of `look` (the page's own DOM only).  `look` and the acting actions must build it
// identically (the wrapper refuses actions after the page changed).
async function items(page) {
  const dom = await page.evaluate((src) => {
    const walk = new Function("return (" + src + ")")();
    const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
    return walk().map(([el, frame]) => {
      const t = el.tagName.toLowerCase();
      const field = t === "input" || t === "select" || t === "textarea";
      const form = el.closest ? el.closest("form") : null;
      const r = el.getBoundingClientRect();
      const itext = ((field ? "" : (el.innerText || el.value || ""))
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
        login: ((f) => !!f && ([...(f.elements || [])].some((x) => (x.type || "").toLowerCase() === "password") || !!f.querySelector("input[type=password]")))(el.form || form),
        form_action: form ? (form.getAttribute("action") || "") : "",
      };
    });
  }, domWalk.toString()).catch(() => []);
  return dom;
}

// ---------- page state --------------------------------------------------------------------------
async function info(page) {
  try {
    return await page.evaluate(() => {
      // a RENDERED password field (publisher pages often carry a hidden header-login template)
      const pw = [...document.querySelectorAll("input[type=password]")].some((x) => x.getClientRects().length > 0
        && getComputedStyle(x).visibility !== "hidden");
      return {
        url: location.href, title: document.title || "", ctype: document.contentType || "",
        text: (document.body && document.body.innerText || "").slice(0, 1500),
        challenge: !!document.querySelector("#challenge-form, #cf-challenge-running, .cf-turnstile, #px-captcha, iframe[src*='challenges.cloudflare.com'], iframe[src*='captcha'], .g-recaptcha, .h-captcha"),
        login_form: pw,
      };
    });
  } catch (e) {
    return { url: "", title: "", ctype: "", text: "", challenge: false, login_form: false };
  }
}
// ONE credential / MFA scan, used by every guard (interaction refusal, sso_login, the hand-off poll): runs in
// the page (source serialized).  Walks the top document, open shadow roots and same-origin iframes.  `cred`:
// a password field, or a text-like field whose metadata looks like a user name / account / one-time code; or a
// cross-origin iframe of an identity / captcha provider (it cannot be inspected).  `mfa`: a one-time-code /
// SMS / verification-code field or such a prompt in the page text.  Field VALUES are never read.
function credScan() {
  // (a plain e-mail field - newsletter sign-up - is not a credential; an identifier-first login field carries
  // autocomplete=username, which matches `user`)
  const CRED = /pass|pwd|user|login|account|otp|one-time|code|sms|\bpin\b|verif|captcha|密码|账号|帐号|用户名|学号|工号|验证码|动态码|短信/i;
  const MFA = /otp|one-time|code|sms|verif|验证码|动态码|短信/i;
  // (captcha widgets are not credentials: an invisible reCAPTCHA badge sits on many article pages; challenge
  // pages are caught by the challenge detection)
  const IDP = /login|signin|sign-in|\/auth\b|oauth|\bsso\b|passport|account|\bidp\b|shibboleth|wayf|\bcas\b/i;
  const TXT = /验证码|短信|动态码|一次性|二次验证|安全验证|身份验证|one[- ]time|verification code|security code|two[- ]factor|2fa|authenticator/i;
  // only fields that are rendered count (a publisher page often carries a hidden header-login template; the
  // browser does not autofill hidden fields); a field inside a frame counts when the frame is rendered too
  const shown = (el) => { try { const v = el.ownerDocument.defaultView; return el.getClientRects().length > 0
    && v.getComputedStyle(el).visibility !== "hidden"; } catch (e) { return true; } };
  let cred = false, mfa = false, fields = 0;
  const walk = (doc) => {
    for (const host of doc.querySelectorAll("*")) if (host.shadowRoot) walk(host.shadowRoot);
    for (const el of doc.querySelectorAll("input, textarea")) {
      if (!shown(el)) continue;
      const t = (el.getAttribute("type") || "text").toLowerCase();
      if (t === "password") { cred = true; continue; }
      if (!["text", "email", "tel", "number", "search", "textarea", ""].includes(el.tagName === "TEXTAREA" ? "textarea" : t)) continue;
      fields++;
      const meta = [el.id, el.getAttribute("name"), el.getAttribute("autocomplete"), el.getAttribute("placeholder"),
                    el.getAttribute("aria-label"), el.getAttribute("inputmode")].join(" ");
      if (CRED.test(meta)) cred = true;
      if (MFA.test(meta) || /one-time-code/i.test(el.getAttribute("autocomplete") || "")) mfa = true;
    }
    for (const f of doc.querySelectorAll("iframe")) {
      let inner = null;
      try { inner = f.contentDocument; } catch (e) { inner = null; }
      if (inner) walk(inner);
      else if (IDP.test((f.getAttribute("src") || "") + " " + (f.getAttribute("title") || ""))) {
        // a cross-origin identity / captcha frame cannot be inspected: a rendered one counts as credentials
        const r = f.getBoundingClientRect();
        if (r.width >= 20 && r.height >= 20 && shown(f)) cred = true;
      }
    }
  };
  walk(document);
  // a verification-code prompt in the text counts only where a text field could take the code
  if (fields > 0 && TXT.test((document.body && document.body.innerText || "").slice(0, 3000))) mfa = true;
  return { cred: cred || mfa, mfa: mfa };
}
async function scan(page) {
  try {
    return await page.evaluate((src) => (new Function("return (" + src + ")")())(), credScan.toString());
  } catch (e) { return { cred: true, mfa: false, unknown: true }; }   // cannot be checked -> treated as a login page
}
async function credPage(page) { return (await scan(page)).cred; }
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
    out.tabs = newTabs.map((t) => ({ label: t.label, url: (t.url || "").slice(0, 200), allowed: hostOk(t.url || "") }));
    // a new tab (opened by the action) that landed outside the allowed domains is closed at once
    const pages = await task.pages().catch(() => []);
    for (const t of out.tabs) {
      if (t.allowed || !t.url || t.url.startsWith("about:")) continue;
      const pg = pages.find((x) => x.label === t.label);
      if (pg) { await pg.close().catch(() => {}); t.closed = true; }
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

// a challenge or (non-SSO) login page: hand the paper's space to the user instead of solving it.  The result
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
  out.handoff = { reason: String(why).slice(0, 120), host: host || "", kind: kind || "challenge",
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
  if (i.login_form && !hostIn(i.url, (INPUT.sso && INPUT.sso.domains) || []) && !(opts && opts.no_login_handoff)) {
    if (INPUT.handoff && INPUT.handoff.enabled) return handoff(task, page, "login form", h, "login", i.title);
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

async function look(task, page) {
  const { offset } = parseLookArg(INPUT.arg);
  const i = await info(page);
  const all = await items(page);
  const pageItems = all.slice(offset, offset + LIMIT).map((it, k) => ({
    i: offset + k, kind: it.kind, text: it.text, href: (it.href || "").slice(0, 200),
    allowed: it.href ? hostOk(it.href) : undefined,
    hidden: !!it.hidden, frame: it.frame || 0, ftype: it.ftype || undefined,
    widget: !!it.widget, login: !!it.login,
  }));
  out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), challenge: isChallenge(i), login_form: i.login_form,
               text: i.text, items: pageItems, more: offset + LIMIT < all.length, next_offset: offset + LIMIT < all.length ? offset + LIMIT : undefined };
  out.look_key = await urlKey(i.url);   // the wrapper's page-change check: a hash, never the URL (it can carry tokens)
  out.status = "ok";
  await postCheck(task, page, i);       // a challenge or (non-SSO) login page is handed to the user here
  if (out.status === "handoff") return;
  out.status = "ok";
}

// The one evaluation that finds, checks and clicks / hovers / fills item k of the dom list.  `walk` is the
// domWalk function itself; both run inside the page (ego's evaluate takes ONE argument, passed as an array).
function domAct(walkSrc, k, allow, kind, value) {
  const ok = (u) => { try { const x = new URL(u, location.href); if (!/^https?:$/.test(x.protocol)) return false;
    const h = x.hostname.toLowerCase(); return allow.some((d) => h === d || h.endsWith("." + d)); } catch (e) { return false; } };
  const pair = new Function("return (" + walkSrc + ")")()()[k];   // domWalk: source -> function -> [el, frame]
  const el = pair && pair[0];
  if (!el) return { refused: "no such item; run look first" };
  const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
  const COOKIE = /^(accept|reject|allow all|reject all|agree|consent|dismiss|close|ok|got it|accept all|manage preferences|接受|拒绝|同意|全部接受|关闭|确定)/i;
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
  const pwdForm = ((f) => !!f && ([...(f.elements || [])].some((x) => (x.type || "").toLowerCase() === "password") || !!f.querySelector("input[type=password]")));
  // nothing inside (or submitting) a form with a password field (also one associated by form="..."): only
  // sso_login submits login forms
  if (pwdForm(form)) return { refused: "login form: credentials are not entered by this tool" };
  if (el.closest && el.closest(".cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha, [id*=captcha], [class*=captcha]")) return { refused: "verification widget: do not interact" };
  if (kind === "click" && W.test(text)) return { refused: "verification widget: do not interact" };
  const tag = el.tagName.toLowerCase();
  if (tag === "iframe") {
    // a cross-origin frame: never acted on inside; `click` opens its (checked) URL as this page
    if (kind !== "click") return { refused: "frame items can only be clicked (opens the frame's page)" };
    if (!ok(el.src)) return { refused: "frame URL is outside the allowed domains" };
    return { open: el.src, text: text };
  }
  if (kind === "click") {
    if (tag === "input" && (el.getAttribute("type") || "").toLowerCase() === "password") return { refused: "password field: do not interact" };
    // the browser-resolved targets (respects <base href> and form= / formaction)
    // (button.formAction is the document URL when the button has no formaction attribute: use form.action then)
    const atag = actv.tagName.toLowerCase();
    const aBtn = atag === "button" || (atag === "input" && ["submit", "image"].includes((actv.getAttribute("type") || "").toLowerCase()));
    if (form && aBtn && !ok((actv.hasAttribute("formaction") ? actv.formAction : "") || form.action || location.href)) return { refused: "form target is outside the allowed domains" };
    if (form && !aBtn && !ok(form.action || location.href)) return { refused: "form target is outside the allowed domains" };
    if (tag === "a") {
      const h = el.getAttribute("href") || "";
      if (!h.startsWith("#") && !/^javascript:/i.test(h) && !ok(el.href)) return { refused: "link target is outside the allowed domains" };
      // a new-tab link (top document or a frame): open its checked target in this page (no native click -
      // a native click lands on whatever is on top at that point)
      if ((el.getAttribute("target") || "").toLowerCase() === "_blank" && !h.startsWith("#") && !/^javascript:/i.test(h)) {
        return { open: el.href, text: text };
      }
    }
    if ((el.hasAttribute("formaction") && !ok(el.formAction)) || (actv.hasAttribute("formaction") && !ok(actv.formAction))) return { refused: "form target is outside the allowed domains" };
    el.click();
    return { clicked: text };
  }
  if (kind === "hover") {
    const r = el.getBoundingClientRect();
    return { center: [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)], text: text };
  }
  if (kind === "select") {
    if (tag !== "select") return { refused: "item is not a select" };
    let opt = [...el.options].find((o) => o.value === value) || [...el.options].find((o) => o.label === value)
      || [...el.options].find((o) => o.label.toLowerCase() === String(value).toLowerCase());
    if (!opt) return { refused: "no such option" };
    el.value = opt.value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return { selected: opt.label.slice(0, 80) };
  }
  if (kind === "type") {
    const t = tag === "input" ? (el.getAttribute("type") || "text") : tag === "textarea" ? "textarea" : "";
    if (!["text", "search", "textarea", ""].includes(t)) return { refused: "not a plain text field" };
    const idn = [el.id, el.getAttribute("name"), el.getAttribute("autocomplete"), el.getAttribute("placeholder"), el.getAttribute("aria-label")].join(" ");
    if (/pass|pwd|user|account|login|email|otp|verif|code|验证码|密码|账号|用户名/i.test(idn)) return { refused: "this field can hold credentials; typing is not allowed" };
    if (pwdForm(el.form || el.closest("form"))) return { refused: "login form: typing is not allowed" };
    const set = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value");
    if (set && set.set) set.set.call(el, value); else el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return { typed: true, len: el.value.length };                    // the length only - never the value
  }
  return { refused: "unknown action" };
}

async function act(task, page, kind, k, value) {
  if (INPUT.look_key) {
    const i0 = await info(page);                       // the page must be the one the agent last looked at
    if (await urlKey(i0.url) !== INPUT.look_key) { out.status = "refused"; out.error = "the page changed since `look`; run look again"; return; }
  }
  const all = await items(page);
  if (!(k >= 0) || k >= all.length) { out.status = "refused"; out.error = "no such item; run look first"; return; }
  const it = all[k];
  const ip = await info(page);
  if (isChallenge(ip)) { out.status = "refused"; out.error = "verification page: do not interact"; await postCheck(task, page, ip); return; }
  // a page with credential / MFA fields (anywhere: frames, shadow roots, identity-provider frames) is never
  // interacted with: sso_login (configured SSO domains) or the hand-off only
  if (ip.login_form || await credPage(page)) {
    out.status = "refused";
    out.error = "login / verification-code page: no click, hover, select or type here (sso_login on the SSO domain, otherwise report login_required)";
    return;
  }
  if (it.widget || WIDGET.test(it.text || "")) { out.status = "refused"; out.error = "verification widget: do not interact"; return; }
  if (kind === "click" && it.login) { out.status = "refused"; out.error = "login form: credentials are not entered by this tool (report login_required)"; return; }
  const domRun = () => page.evaluate(([src, walk, k, allow, kind, value]) => {   // ego's evaluate takes ONE arg
    const f = new Function("return (" + src + ")")();        // domAct; walk is domWalk, both plain functions
    return f(walk, k, allow, kind, value);
  }, [domAct.toString(), domWalk.toString(), k, INPUT.allow, kind, value || ""]);
  {
    if (kind === "click") {
      await withDownload(task, page, async () => {
        const r = await domRun();
        if (r && r.refused) { out.status = "refused"; out.error = r.refused; }
        else if (r && r.open) {
          await page.goto(r.open, { timeout: INPUT.nav_timeout_ms, waitUntil: "domcontentloaded" });
        }
      });
      if (out.status === "refused") return;
      await sleep(1500);
      const i = await after(page);
      if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
      out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
      if (out.status !== "handoff") out.status = "ok";
      return;
    }
    if (kind === "hover") {
      const r = await domRun();
      if (r && r.refused) { out.status = "refused"; out.error = r.refused; return; }
      if (!r || !r.center) { out.status = "error"; out.error = "the item has no position (hidden?)"; return; }
      const u0 = (await info(page)).url;
      await page.mouse.move(r.center[0], r.center[1]).catch(() => {});
      await sleep(800);
      out.acted = { kind: "hover", text: r.text || "" };
      const i = await info(page);
      if (i.url !== u0) {                                // a hover that navigated: checked like a click
        out.navigated = true;
        const i2 = await after(page);
        if (!out.blocked && !out.left_allowed) await postCheck(task, page, i2);
        if (out.status === "handoff") return;
      }
      out.status = "ok";
      return;
    }
    // select / type: their change handlers may navigate - checked like a click (the wrapper counts and paces
    // them as page loads)
    let r = null;
    await withDownload(task, page, async () => { r = await domRun(); });
    if (r && r.refused) { out.status = "refused"; out.error = r.refused; return; }
    out.acted = { kind: kind, text: (r && r.selected) || "" };
    await sleep(1000);
    const i = await after(page);
    if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
    out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
    if (out.status !== "handoff") out.status = "ok";
  }
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
  // Check, in one page evaluation: the page is still on an HTTPS configured SSO host, the form posts (method
  // POST - never GET, which would put the credentials into a URL) to the browser-resolved target (form.action /
  // button.formAction; <base href> included) on such a host.  Then press the button with a native click (the
  // browser releases autofilled values to the page on a real user gesture only).
  const chk = await page.evaluate(([domains, lt]) => {
    const inDom = (u) => { try { const x = new URL(u); const hh = x.hostname.toLowerCase();
      const local = hh === "127.0.0.1" || hh === "localhost" || hh.endsWith(".localhost");
      if (x.protocol !== "https:" && !(local && lt)) return false;
      return domains.some((d) => hh === d || hh.endsWith("." + d)); } catch (e) { return false; } };
    if (!inDom(location.href)) return { refused: "the page left the SSO login domain" };
    const pwd = document.querySelector("form input[type=password]") || document.querySelector("input[type=password]");
    const form = pwd && (pwd.form || pwd.closest("form"));
    if (!form) return { refused: "no login form" };
    const b = [...form.elements].find((x) => (x.type || "").toLowerCase() === "submit" && !x.disabled)
      || [...form.querySelectorAll("button")].find((x) => /log ?in|sign ?in|登录|登陆|submit/i.test((x.innerText || "")));
    if (!b) return { refused: "no submit button on the login form" };
    const target = (b.hasAttribute("formaction") ? b.formAction : "") || form.action || location.href;
    const method = ((b.hasAttribute("formmethod") ? b.formMethod : "") || form.method || "get").toLowerCase();
    if (method !== "post") return { refused: "the login form does not use POST" };
    if (!inDom(target)) return { refused: "the login form posts outside the SSO login domains" };
    b.scrollIntoView({ block: "center" });
    const r = b.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return { refused: "the login button is not visible" };
    const cx = Math.round(r.left + r.width / 2), cy = Math.round(r.top + r.height / 2);
    if (document.elementFromPoint(cx, cy) !== b && !b.contains(document.elementFromPoint(cx, cy))) return { refused: "the login button is covered" };
    return { center: [cx, cy] };
  }, [sso.domains, localTest]).catch((e) => ({ refused: "the login form could not be checked: " + String(e).slice(0, 80) }));
  if (!chk || !chk.center) { out.status = "refused"; out.error = (chk && chk.refused) || "not submitted"; return; }
  out.sso_stage = "submit";
  out.sso_submitted = true;                          // from here on the outcome counts as a submission
  await withDownload(task, page, () => page.mouse.click(chk.center[0], chk.center[1]));
  // wait for the login page to go away (never resubmit; a failure ends the logins of this batch)
  const t0 = Date.now();
  for (;;) {
    await sleep(1500);
    const i2 = await info(page);
    if (isChallenge(i2)) { out.sso = "handoff"; return handoff(task, page, "challenge after the SSO login", h, "challenge", i2.title); }
    const st2 = await read();
    if (st2.mfa) { out.sso = "handoff"; return handoff(task, page, "MFA / verification code after the SSO login", h, "mfa", i2.title); }
    if (!st2.hasPwd && hostIn(i2.url, sso.domains) && await credPage(page)) {
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
      const allowedHere = !/^https?:/i.test(i0.url || "") || hostOk(i0.url)
        || hostIn(i0.url, (INPUT.sso && INPUT.sso.domains) || []);
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
    } else if (INPUT.action === "click" || INPUT.action === "hover" || INPUT.action === "select" || INPUT.action === "type") {
      const parts = String(INPUT.arg || "").split(" ");
      const k = parseInt(parts[0], 10);
      const value = parts.slice(1).join(" ");
      if (INPUT.action === "type" && !(INPUT.type_allow || []).includes(value)) {
        out.status = "refused"; out.error = "only the whitelisted institution names may be typed";
      } else {
        await act(task, page, INPUT.action, k, value);
      }
    } else if (INPUT.action === "scroll") {
      const a = String(INPUT.arg || "").split(/\s+/);
      const d = a[0] === "up" ? -1 : 1;
      const px = Math.min(5000, Math.max(50, parseInt(a[1] || "700", 10) || 700));
      const before = await page.info().catch(() => ({}));
      await page.evaluate(([d, px]) => window.scrollBy(0, d * px), [d, px]).catch(() => {});
      await sleep(300);
      let afterI = await page.info().catch(() => ({}));
      if ((afterI.sy || 0) === (before.sy || 0)) {                       // the window cannot scroll: wheel over the page
        await page.mouse.move(400, 400).catch(() => {});
        await page.mouse.wheel(0, d * px).catch(() => {});
        await sleep(300);
        afterI = await page.info().catch(() => ({}));
      }
      out.scroll = { from: before.sy || 0, to: afterI.sy || 0 };
      out.status = "ok";
      if ((afterI.url || "") && before.url && afterI.url !== before.url) {   // scrolling triggered a navigation
        out.navigated = true;
        const i = await after(page);
        if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
        if (out.status !== "handoff") out.status = "ok";
      }
    } else if (INPUT.action === "tabs") {
      const tabs = await task.tabs();
      out.tabs = tabs.map((t) => ({ label: t.label, url: (t.url || "").slice(0, 200), title: (t.title || "").slice(0, 80),
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
    } else if (INPUT.action === "back") {
      await withDownload(task, page, () => page.evaluate(() => history.back()));
      await sleep(1500);
      const i = await after(page);
      if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
      out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
      if (out.status !== "handoff") out.status = "ok";
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
    } else if (INPUT.action === "sso_login") {
      await ssoLogin(task, page);
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
            if (isChallenge(i) || i.login_form || await credPage(page2)) {
              // the user has not finished: hand the space back and keep waiting
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
