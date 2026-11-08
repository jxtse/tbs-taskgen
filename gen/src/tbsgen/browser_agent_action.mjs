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
//                                 numbered list of links / buttons / fields (hidden dropdown items included,
//                                 same-origin iframes included, cross-origin iframe items included via the
//                                 page snapshot); `full` covers the whole scrollable page, `@REF` one snapshot
//                                 subtree, OFFSET paginates the item list
//   open URL                      navigate (URL host must be in INPUT.allow); downloads and new tabs are saved /
//                                 reported
//   click I                       click item I of the last `look` list (same numbering; run `look` first)
//   click_xy X Y                  click viewport coordinates (refused on verification pages, on login forms and
//                                 when the element at the point is a verification widget or a credential field)
//   hover I                       move the mouse onto item I (opens CSS / JS dropdown menus)
//   scroll up|down [PX]           scroll the page
//   select I VALUE                choose VALUE (option value or label) of select item I
//   type I TEXT                   type TEXT into text item I - only the whitelisted institution names in
//                                 INPUT.type_allow, never into anything that can hold credentials
//   screenshot                    save a PNG of the page under INPUT.files_dir (refused on pages with
//                                 credential fields; the agent views it with its vision tool)
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
// no interaction with verification pages / widgets, no reading of input values or storage, login forms are
// never submitted except by `sso_login` with browser autofill, navigation targets are checked before and the
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

// a URL as reported back to the agent from login pages: no query or fragment (they can carry tokens)
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
      try { if (f.contentDocument) walk(f.contentDocument, frame + 1); } catch (e) { /* cross-origin */ }
    }
  };
  walk(document, 0);
  return pairs;
}

// ego page snapshot lines: `  anchor [ref=4, url=...]` / `  iframe [ref=6]` / `    text "Download"`.
// Only items under an iframe subtree that domWalk could not reach (cross-origin iframes) are kept.
function snapItems(snap, dom) {
  const lines = String(snap).split("\n");
  const items = [];
  const stack = [];
  // the same same-origin element found by domWalk is not listed twice
  const domKeys = new Set(dom.map((x) => x.kind + "|" + (x.href || "") + "|" + (x.itext || x.text || "")));
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    if (!ln.trim()) continue;
    const ind = ln.length - ln.trimStart().length;
    while (stack.length && stack[stack.length - 1].ind >= ind) stack.pop();
    const fm = /^\s*iframe \[ref=(\d+)/.exec(ln);
    if (fm) { stack.push({ ind: ind, iframe: true }); continue; }
    const m = /^\s*(button|anchor|link|combobox|textbox|menuitem|tab|option) \[ref=(\d+)(?:[^\]]*?url=(\S+?))?\]/.exec(ln);
    if (!m || !stack.some((s) => s.iframe)) continue;
    const t = /^\s*text "(.*)"$/m.exec(lines[i + 1] || "");
    // a field's snapshot text may be its current value (e.g. an autofilled user name): never shown for fields
    const field = m[1] === "textbox" || m[1] === "combobox";
    const it = { src: "snap", kind: m[1] === "anchor" ? "link" : m[1], text: (t && !field ? t[1] : "").slice(0, 80),
                 href: m[3] || "", ref: m[2] };
    if (domKeys.has(it.kind + "|" + (it.href || "") + "|" + (it.text || ""))) continue;   // same-origin: already listed
    items.push(it);
  }
  return items;
}

// The numbered item list of `look` (dom items first, cross-origin snapshot items after).  `look` and the
// acting actions must build it identically (the wrapper refuses actions after the page changed).
async function items(page, scope) {
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
  let snap = "";
  if (dom.some((d) => d.login)) return dom;           // login page: no raw snapshot (it could hold field texts)
  if (scope === "full") snap = await page.snapshot({ scope: "full_page" }).then(String).catch(() => "");
  else if (scope && scope.startsWith("@")) snap = await page.snapshot({ scope: "subtree", root: scope }).then(String).catch(() => "");
  else snap = await page.snapshot().then(String).catch(() => "");
  return dom.concat(snapItems(snap, dom));
}

// ---------- page state --------------------------------------------------------------------------
async function info(page) {
  try {
    return await page.evaluate(() => {
      const pw = !!document.querySelector("input[type=password]");
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
// Credential fields anywhere on the page (top document and same-origin iframes), or a cross-origin iframe that
// looks like a login / identity provider frame.  Field VALUES are never read here.  Unknown -> treated as a
// login page (screenshots and clicks are refused when this cannot be checked).
async function credPage(page) {
  try {
    return await page.evaluate(() => {
      const CRED = /pass|pwd|user|login|account|otp|one-time|verif|密码|账号|用户名|学号|工号/i;
      const IDP = /login|signin|sign-in|auth|sso|passport|account|idp|shibboleth|wayf|cas\b/i;
      let found = false;
      const walk = (doc) => {
        for (const el of doc.querySelectorAll("input")) {
          const t = (el.getAttribute("type") || "text").toLowerCase();
          if (t === "password") { found = true; return; }
          if (["text", "email", "tel", "number", ""].includes(t)
              && CRED.test([el.id, el.getAttribute("name"), el.getAttribute("autocomplete"), el.getAttribute("placeholder"),
                            el.getAttribute("aria-label")].join(" "))) { found = true; return; }
        }
        for (const f of doc.querySelectorAll("iframe")) {
          let inner = null;
          try { inner = f.contentDocument; } catch (e) { inner = null; }
          if (inner) walk(inner);
          else if (IDP.test(f.getAttribute("src") || "")) found = true;
          if (found) return;
        }
      };
      walk(document);
      return found;
    }) || await crossFrameFields(page);
  } catch (e) { return true; }
}

// Do cross-origin iframes (unreadable from the page) contain form fields?  Read from the ego snapshot (it
// reaches them); no snapshot -> unknown -> true.  Field texts of the snapshot are never returned.
async function crossFrameFields(page) {
  const snap = await page.snapshot({ scope: "full_page" }).then(String).catch(() => null);
  if (snap === null) return true;
  const lines = snap.split("\n");
  const stack = [];
  for (const ln of lines) {
    if (!ln.trim()) continue;
    const ind = ln.length - ln.trimStart().length;
    while (stack.length && stack[stack.length - 1] >= ind) stack.pop();
    if (/^\s*iframe \[ref=/.test(ln)) { stack.push(ind); continue; }
    if (stack.length && /^\s*(textbox|combobox|searchbox)\b/.test(ln)) return true;
  }
  return false;
}
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
  if (i.url && !i.url.startsWith("about:") && !i.url.startsWith("chrome-error:") && !hostOk(i.url)) {
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
  console.log("TBSGEN_RESULT " + JSON.stringify(out));
  console.error("TBSGEN_RESULT " + JSON.stringify(out));
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
  const a = String(arg || "").split(/\s+/).filter(Boolean);
  const r = { scope: "", offset: 0 };
  for (const x of a) {
    if (x === "full") r.scope = "full";
    else if (x.startsWith("@")) r.scope = x;
    else if (/^\d{1,4}$/.test(x)) r.offset = parseInt(x, 10);
  }
  return r;
}

const LIMIT = 120;                                   // items per `look` page

async function look(task, page) {
  const { scope, offset } = parseLookArg(INPUT.arg);
  const i = await info(page);
  const all = await items(page, scope);
  const pageItems = all.slice(offset, offset + LIMIT).map((it, k) => ({
    i: offset + k, kind: it.kind, text: it.text, href: (it.href || "").slice(0, 200),
    allowed: it.href ? hostOk(it.href) : undefined,
    hidden: !!it.hidden, frame: it.frame || 0, ftype: it.ftype || undefined, src: it.src || "dom",
    widget: !!it.widget, login: !!it.login,
  }));
  out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), challenge: isChallenge(i), login_form: i.login_form,
               text: i.text, items: pageItems, more: offset + LIMIT < all.length, next_offset: offset + LIMIT < all.length ? offset + LIMIT : undefined };
  out.look_url = i.url;
  out.look_scope = scope;
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
  const form = el.form || (el.closest ? el.closest("form") : null);
  const pwdForm = ((f) => !!f && ([...(f.elements || [])].some((x) => (x.type || "").toLowerCase() === "password") || !!f.querySelector("input[type=password]")));
  // nothing inside (or submitting) a form with a password field (also one associated by form="..."): only
  // sso_login submits login forms
  if (pwdForm(form)) return { refused: "login form: credentials are not entered by this tool" };
  if (el.closest && el.closest(".cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha, [id*=captcha], [class*=captcha]")) return { refused: "verification widget: do not interact" };
  if (kind === "click" && W.test(text)) return { refused: "verification widget: do not interact" };
  const tag = el.tagName.toLowerCase();
  if (kind === "click") {
    if (tag === "input" && (el.getAttribute("type") || "").toLowerCase() === "password") return { refused: "password field: do not interact" };
    // the browser-resolved targets (respects <base href> and form= / formaction)
    if (form && (tag === "button" || btnInput) && !ok(el.formAction || form.action || location.href)) return { refused: "form target is outside the allowed domains" };
    if (form && !(tag === "button" || btnInput) && !ok(form.action || location.href)) return { refused: "form target is outside the allowed domains" };
    if (tag === "a") {
      const h = el.getAttribute("href") || "";
      if (!h.startsWith("#") && !/^javascript:/i.test(h) && !ok(el.href)) return { refused: "link target is outside the allowed domains" };
      if (el.getAttribute("target") === "_blank" && pair[1] > 0) {
        // a new-tab link inside a frame: frame-local coordinates are useless at the top level; open its
        // (checked) target in this page instead
        return { open: el.href, text: text };
      }
      if (el.getAttribute("target") === "_blank") {
        // a synthetic el.click() does NOT open target=_blank in a new tab - the native mouse click does.
        // Return the coordinates: the caller clicks with native input (the publisher's new tab is tracked).
        const r = el.getBoundingClientRect();
        if (r.width >= 1 && r.height >= 1) {
          return { center: [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)], text: text, newtab: true };
        }
        el.removeAttribute("target");                     // hidden new-tab link: open it in this Page
      }
    }
    if (el.getAttribute("formaction") && !ok(el.formAction || el.getAttribute("formaction"))) return { refused: "form target is outside the allowed domains" };
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

// page-level check for snapshot-ref actions: a page with a visible password field only accepts
// cookie-banner-like controls
async function snapClickCheck(page, item) {
  const i = await info(page);
  if (isChallenge(i)) return "verification page: do not interact";
  if (WIDGET.test(item.text || "")) return "verification widget: do not interact";
  if (item.kind === "link" && item.href && !hostOk(item.href)) return "link target is outside the allowed domains";
  if (i.login_form || await credPage(page)) return "login page / form in a frame: items inside frames are not clicked there";
  if (item.kind !== "link") {
    // a control inside a cross-origin frame: its own target cannot be read, so every cross-origin frame of the
    // page must come from an allowed host
    const srcs = await page.evaluate(() => [...document.querySelectorAll("iframe")].map((f) => {
      try { if (f.contentDocument) return null; } catch (e) {} return f.src || ""; }).filter((x) => x !== null)).catch(() => null);
    if (!srcs || srcs.some((u) => !u || !hostOk(u))) return "a control inside a frame from a host outside the allowed domains";
  }
  return null;
}

async function act(task, page, kind, k, value) {
  if (INPUT.look_url) {
    const i0 = await info(page);                       // the page must be the one the agent last looked at
    if (i0.url && i0.url !== INPUT.look_url) { out.status = "refused"; out.error = "the page changed since `look`; run look again"; return; }
  }
  const all = await items(page, INPUT.look_scope || "");
  if (!(k >= 0) || k >= all.length) { out.status = "refused"; out.error = "no such item; run look first"; return; }
  const it = all[k];
  const ip = await info(page);
  if (isChallenge(ip)) { out.status = "refused"; out.error = "verification page: do not interact"; await postCheck(task, page, ip); return; }
  if (it.widget || WIDGET.test(it.text || "")) { out.status = "refused"; out.error = "verification widget: do not interact"; return; }
  if (kind === "click" && it.login) { out.status = "refused"; out.error = "login form: credentials are not entered by this tool (report login_required)"; return; }
  if (kind === "type" && (it.src || "dom") !== "dom") { out.status = "refused"; out.error = "typing is only allowed into fields of the page itself"; return; }
  const isDom = (it.src || "dom") === "dom";        // domWalk items carry no src field
  const domRun = () => page.evaluate(([src, walk, k, allow, kind, value]) => {   // ego's evaluate takes ONE arg
    const f = new Function("return (" + src + ")")();        // domAct; walk is domWalk, both plain functions
    return f(walk, k, allow, kind, value);
  }, [domAct.toString(), domWalk.toString(), k, INPUT.allow, kind, value || ""]);
  if (isDom) {
    if (kind === "click") {
      await withDownload(task, page, async () => {
        const r = await domRun();
        if (r && r.refused) { out.status = "refused"; out.error = r.refused; }
        else if (r && r.open) {
          await page.goto(r.open, { timeout: INPUT.nav_timeout_ms, waitUntil: "domcontentloaded" });
        }
        else if (r && r.newtab && r.center) {
          // a target=_blank link: click with native input so the publisher's new tab really opens
          await page.mouse.click(r.center[0], r.center[1]).catch(() => {});
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
    return;
  }
  // cross-origin iframe item: act through the ego snapshot ref
  const why = await snapClickCheck(page, it);
  if (why) { out.status = "refused"; out.error = why; return; }
  if (kind !== "click") { out.status = "refused"; out.error = "only clicks are supported for cross-origin iframe items"; return; }
  await withDownload(task, page, () => page.click("@" + it.ref, { timeout: 8000 }));
  await sleep(1500);
  const i = await after(page);
  if (!out.blocked && !out.left_allowed) await postCheck(task, page, i);
  out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
  if (out.status !== "handoff") out.status = "ok";
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
  const read = () => page.evaluate((lt) => {
    const af = (el) => { try { return el.matches(":autofill") || el.matches(":-webkit-autofill"); } catch (e) { return false; } };
    const pwd = document.querySelector("form input[type=password]") || document.querySelector("input[type=password]");
    if (!pwd) return { hasPwd: false };
    const form = pwd.form || pwd.closest("form") || document;
    const user = form.querySelector("input[autocomplete=username], input[type=text], input[type=email], input:not([type])");
    const mfa = !!form.querySelector("input[autocomplete='one-time-code'], input[name*=otp], input[name*='sms'], iframe[src*='captcha'], .g-recaptcha, .h-captcha, .cf-turnstile")
      || /验证码|短信|动态码|一次性|one[- ]time|verification code/i.test((document.body && document.body.innerText || "").slice(0, 2000));
    const filled = (el) => af(el) || (lt && el.value.length > 0);
    return { hasPwd: true, hasUser: !!user, mfa: mfa, pwFilled: filled(pwd), userFilled: !!user && filled(user) };
  }, localTest).catch(() => ({ hasPwd: false }));
  let st = await read();
  if (st.mfa) return handoff(task, page, "MFA / verification code on the SSO login page", h, "mfa", i.title);
  if (!st.hasPwd || !st.hasUser) return handoff(task, page, "not a recognizable SSO login form", h, "login", i.title);
  if (!st.pwFilled || !st.userFilled) {
    // some Chromium versions autofill only after the user field was clicked once (a click, never typing)
    const c = await page.evaluate(() => {
      const pwd = document.querySelector("form input[type=password]") || document.querySelector("input[type=password]");
      const form = (pwd && (pwd.form || pwd.closest("form"))) || document;
      const user = form.querySelector("input[autocomplete=username], input[type=text], input[type=email], input:not([type])");
      if (!user) return null;
      const r = user.getBoundingClientRect();
      return r.width >= 1 ? [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)] : null;
    }).catch(() => null);
    if (c) await page.mouse.click(c[0], c[1]).catch(() => {});
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
    } else if (INPUT.action === "click_xy") {
      const parts = String(INPUT.arg || "").split(" ").map((x) => parseInt(x, 10));
      const i0 = await info(page);
      if (isChallenge(i0)) { out.status = "refused"; out.error = "verification page: do not interact"; out.blocked = "verification page: " + i0.title.slice(0, 60); }
      else if (i0.login_form || await credPage(page)) { out.status = "refused"; out.error = "login page: coordinate clicks are not allowed (sso_login or hand-off only)"; }
      else if (!(parts[0] >= 0) || !(parts[1] >= 0)) { out.status = "refused"; out.error = "usage: click_xy X Y"; }
      else {
        const pre = await page.evaluate(([x, y, allow]) => {
          const el = document.elementFromPoint(x, y);
          if (!el) return null;
          const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
          if (W.test([el.id, el.className && String(el.className), el.getAttribute("name"), el.getAttribute("aria-label")].join(" "))
            || el.closest(".cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha, iframe[src*='captcha']")) return { refused: "verification widget: do not interact" };
          if (el.tagName === "INPUT" && (el.getAttribute("type") || "").toLowerCase() === "password") return { refused: "password field: do not interact" };
          if (el.tagName === "IFRAME" || el.tagName === "FRAME") return { refused: "a point inside a frame cannot be checked; use click on a look item" };
          const form = el.form || el.closest("form");
          if (((f) => !!f && ([...(f.elements || [])].some((x) => (x.type || "").toLowerCase() === "password") || !!f.querySelector("input[type=password]")))(form)) return { refused: "login form: credentials are not entered by this tool" };
          const ok = (u) => { try { const x = new URL(u, location.href); if (!/^https?:$/.test(x.protocol)) return false;
            const h = x.hostname.toLowerCase(); return allow.some((d) => h === d || h.endsWith("." + d)); } catch (e) { return false; } };
          const a = el.closest("a[href]");
          if (a) { const h = a.getAttribute("href") || ""; if (!h.startsWith("#") && !/^javascript:/i.test(h) && !ok(a.href)) return { refused: "link target is outside the allowed domains" }; }
          if (form && !ok(form.action || location.href)) return { refused: "form target is outside the allowed domains" };
          const fa = el.closest("button, input");
          if (fa && fa.formAction && fa.form && !ok(fa.formAction)) return { refused: "form target is outside the allowed domains" };
          if (W.test(((el.innerText || "") + " " + (el.getAttribute("aria-label") || "")).slice(0, 200))) return { refused: "verification widget: do not interact" };
          return { ok: true };
        }, [parts[0], parts[1], INPUT.allow]).catch(() => null);
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
    } else if (INPUT.action === "screenshot") {
      const i = await info(page);
      if (!i.url || i.login_form || await credPage(page)) { out.status = "refused"; out.error = "the page has (or may have) credential fields; screenshots are disabled on login pages"; }
      else {
        const f = `${INPUT.files_dir}/shot-${Date.now()}-${Math.floor(Math.random() * 1e5)}.png`;
        // whatever the fields hold is masked in the picture too (same-origin documents)
        const MASK = "input:not([type=submit]):not([type=button]),textarea{color:transparent!important;"
          + "-webkit-text-security:disc!important;text-shadow:none!important}";
        await page.evaluate((css) => {
          const add = (doc) => { try { const st = doc.createElement("style"); st.id = "__tbsgen_mask"; st.textContent = css;
            doc.documentElement.appendChild(st); for (const fr of doc.querySelectorAll("iframe")) { try { if (fr.contentDocument) add(fr.contentDocument); } catch (e) {} } } catch (e) {} };
          add(document);
        }, MASK).catch(() => {});
        await page.screenshot({ path: f });
        await page.evaluate(() => {
          const rm = (doc) => { try { doc.querySelectorAll("#__tbsgen_mask").forEach((x) => x.remove());
            for (const fr of doc.querySelectorAll("iframe")) { try { if (fr.contentDocument) rm(fr.contentDocument); } catch (e) {} } } catch (e) {} };
          rm(document);
        }).catch(() => {});
        out.shot = f;
        out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype) };
        out.status = "ok";
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
        let u = (await page.info().catch(() => ({}))).url || "";
        while (!u.includes(pat) && Date.now() - t0 < secs * 1000) { await sleep(1000); u = (await page.info().catch(() => ({}))).url || ""; }
        out.waited = { what: "url", ok: u.includes(pat), url: u.slice(0, 200) };
        out.status = "ok";
      } else if (what === "sel") {
        const r = await page.waitForSelector(pat, { timeout: secs * 1000 }).then(() => true).catch(() => false);
        out.waited = { what: "sel", ok: r };
        out.status = "ok";
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
          console.log("TBSGEN_RESULT " + JSON.stringify(out));
          console.error("TBSGEN_RESULT " + JSON.stringify(out)); process.exit(0);
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
console.log("TBSGEN_RESULT " + JSON.stringify(out));
console.error("TBSGEN_RESULT " + JSON.stringify(out));
process.exit(0);
