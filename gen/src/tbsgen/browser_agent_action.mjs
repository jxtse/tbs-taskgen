// tbsgen browser-agent ACTION script (tbsgen.browser_agent).  The agent never writes browser code: it calls
// `tbsgen-browser N <action> [arg]`, the wrapper prepends one line `const INPUT = {...};` to this reviewed file
// and runs it with `ego-browser nodejs`.  The last stdout line starting with "TBSGEN_RESULT " is the result.
//
// Actions (one Page of the batch's own TaskSpace):
//   look        page summary: URL, title, PDF?, challenge?, login form?, text excerpt, numbered links / buttons
//   open URL    navigate (URL host must be in INPUT.allow); a download it triggers is saved
//   click I     click item I of the last `look` list (same order); refused on a verification page, on a
//               verification widget, on a login form submit, or on a link to a host outside INPUT.allow
//   save_pdf    the current page is a PDF document: save it (same-origin page.fetch)
//   finish      close the TaskSpace
// Rules enforced here: no typing anywhere, no interaction with verification pages / widgets, no reading of
// input values or storage, navigation targets checked before and the landed URL checked after (a page that
// ends up outside INPUT.allow is reset to about:blank), files only under INPUT.files_dir.
const out = { status: "error", action: INPUT.action, files: [] };
const CHALLENGE = /just a moment|verify you are human|attention required|access denied|are you a robot|security check|captcha|human verification|请稍候|请稍等|人机验证|安全验证/i;
const WIDGET = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hostOk = (u) => {
  let h;
  try { const x = new URL(u); if (!/^https?:$/.test(x.protocol)) return false; h = x.hostname.toLowerCase(); } catch (e) { return false; }
  return INPUT.allow.some((d) => h === d || h.endsWith("." + d));
};

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

async function withDownload(page, act) {
  // arm a download wait before the action, keep it armed while the action runs and download_wait_ms after
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
  if (dl) {
    let u = "";
    try { u = dl.url(); } catch (e) {}
    if (u && !u.startsWith("blob:") && !u.startsWith("data:") && !hostOk(u)) { out.note = "download from a host outside the allowed domains ignored"; return; }
    const f = fileName();
    await dl.saveAs(f);
    if (await savedIsChallenge(f)) { out.blocked = "downloaded bot-wall page"; (await import("node:fs/promises")).unlink(f).catch(() => {}); return; }
    out.files.push(f);
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
  if (isChallenge(i)) out.blocked = out.blocked || ("verification page: " + (i.title || "").slice(0, 60));
  return i;
}

function clickables() {
  // same order for `look` and `click`: visible links and buttons, verification widgets excluded
  const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
  const els = [];
  for (const el of document.querySelectorAll("a[href], button, input[type=submit], input[type=button], [role=button]")) {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    const attrs = [el.id, el.className && String(el.className), el.getAttribute("name"), el.getAttribute("aria-label")].join(" ");
    if (W.test(attrs) || el.closest("iframe, .cf-turnstile, .g-recaptcha, .h-captcha, #challenge-form, #px-captcha")) continue;
    els.push(el);
    if (els.length >= 60) break;
  }
  return els;
}

async function look(page) {
  const i = await info(page);
  const items = await page.evaluate((src) => {
    const f = new Function("return (" + src + ")")();
    return f().map((el, k) => {
      const form = el.closest("form");
      return { i: k, kind: el.tagName.toLowerCase() === "a" ? "link" : "button",
               text: ((el.innerText || el.value || el.getAttribute("aria-label") || el.title || "").trim()).slice(0, 80),
               href: el.tagName.toLowerCase() === "a" ? el.href : undefined,
               login_form: !!(form && form.querySelector("input[type=password]")) };
    });
  }, clickables.toString()).catch(() => []);
  for (const it of items) if (it.href) it.allowed = hostOk(it.href);
  return { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), challenge: isChallenge(i), login_form: i.login_form,
           text: i.text, items };
}

try {
  const task = await taskSpace(INPUT.space);
  out.space = task.spaceId;
  if (INPUT.action === "finish") {
    await task.finish({ keep: [] });
    out.status = "ok";
  } else {
    const pages = await task.pages();
    const page = pages.length ? pages[0] : await task.newPage();
    if (INPUT.action === "look") {
      out.page = await look(page);
      out.look_url = out.page.url;
      if (out.page.challenge) out.blocked = "verification page: " + out.page.title.slice(0, 60);
      out.status = "ok";
    } else if (INPUT.action === "open") {
      const url = String(INPUT.arg || "");
      if (!hostOk(url)) { out.status = "refused"; out.error = "URL host is not in the allowed domains of this paper"; }
      else {
        await withDownload(page, () => page.goto(url, { timeout: INPUT.nav_timeout_ms, waitUntil: "domcontentloaded" }));
        const i = await after(page);
        out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
        out.status = "ok";
      }
    } else if (INPUT.action === "click") {
      const k = parseInt(INPUT.arg, 10);
      const i0 = await info(page);
      if (isChallenge(i0)) { out.status = "refused"; out.error = "verification page: do not interact"; out.blocked = "verification page: " + i0.title.slice(0, 60); }
      else {
        // check and click in ONE page evaluation, on the element found now, with every condition re-checked:
        // allowed link target / form action, no verification widget or text, no login form, same page as `look`
        await withDownload(page, async () => {
          const r = await page.evaluate(([src, k, allow, lookUrl]) => {
            const ok = (u) => { try { const x = new URL(u, location.href); if (!/^https?:$/.test(x.protocol)) return false;
              const h = x.hostname.toLowerCase(); return allow.some((d) => h === d || h.endsWith("." + d)); } catch (e) { return false; } };
            if (lookUrl && location.href !== lookUrl) return { refused: "the page changed since `look`; run look again" };
            const f = new Function("return (" + src + ")")();
            const el = f()[k];
            if (!el) return { refused: "no such item; run look first" };
            const W = /captcha|turnstile|challenge|recaptcha|hcaptcha|verify|i am human|not a robot|cf-|px-|验证/i;
            const text = ((el.innerText || el.value || el.getAttribute("aria-label") || el.title || "").trim()).slice(0, 80);
            if (W.test(text)) return { refused: "verification widget: do not interact" };
            const form = el.closest("form") || (el.form || null);
            if (form && form.querySelector("input[type=password]")) return { refused: "login form: credentials are not entered by this tool (report login_required)" };
            if (form && !ok(form.getAttribute("action") || location.href)) return { refused: "form target is outside the allowed domains" };
            const tag = el.tagName.toLowerCase();
            if (tag === "a") {
              const h = el.getAttribute("href") || "";
              if (!h.startsWith("#") && !/^javascript:/i.test(h) && !ok(el.href)) return { refused: "link target is outside the allowed domains" };
              el.removeAttribute("target");                       // stay in this Page
            }
            if (el.getAttribute("formaction") && !ok(el.getAttribute("formaction"))) return { refused: "form target is outside the allowed domains" };
            el.click();
            return { clicked: text };
          }, [clickables.toString(), k, INPUT.allow, INPUT.look_url || ""]);
          if (r && r.refused) { out.status = "refused"; out.error = r.refused; }
        });
        if (out.status !== "refused") {
          await sleep(1500);
          const i = await after(page);
          out.page = { url: i.url, title: i.title, pdf: /pdf/i.test(i.ctype), login_form: i.login_form };
          out.status = "ok";
        }
      }
    } else if (INPUT.action === "save_pdf") {
      const i = await info(page);
      if (!/pdf/i.test(i.ctype)) { out.status = "refused"; out.error = "the current page is not a PDF document"; }
      else if (!hostOk(i.url)) { out.status = "refused"; out.error = "PDF host is not in the allowed domains"; }
      else {
        const f = fileName();
        const r = await page.fetch(i.url, { saveAs: f, timeout: INPUT.download_timeout_ms });
        out.http = r.status;
        if (r.url && !hostOk(r.url)) { (await import("node:fs/promises")).unlink(f).catch(() => {}); out.status = "refused"; out.error = "PDF redirected outside the allowed domains"; console.log("TBSGEN_RESULT " + JSON.stringify(out)); process.exit(0); }
        if (r.status === 429 || await savedIsChallenge(f)) { out.blocked = r.status === 429 ? "HTTP 429" : "bot-wall page"; out.http429 = r.status === 429; }
        else if (r.ok) out.files.push(f);
        out.status = "ok";
      }
    } else {
      out.status = "refused"; out.error = "unknown action";
    }
  }
} catch (e) {
  out.status = "error";
  out.error = String(e && e.message || e).slice(0, 300);
}
console.log("TBSGEN_RESULT " + JSON.stringify(out));
process.exit(0);
