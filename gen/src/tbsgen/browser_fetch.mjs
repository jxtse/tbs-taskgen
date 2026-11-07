// ego-browser script template for the S3 browser channel (tbsgen.browser_ego).
// Runs inside `ego-browser nodejs` (Node ESM, ego-browser SDK globals).  tbsgen prepends one line
//   const INPUT = {...};
// (absolute paths, no secrets) and reads the last stdout line starting with "TBSGEN_RESULT ".
//
// Rules (same as the earlier pipeline's browser channel):
//  - the user's logged-in browser profile; only Agent-managed Pages of one fixed TaskSpace are used,
//    user tabs are never touched;
//  - nothing is clicked through: a challenge page ("Just a moment...", captcha, ...) is given
//    INPUT.challenge_wait_ms to clear by itself, otherwise the result is status "challenge";
//  - PDFs are taken from a download event or from the PDF document itself (same-origin page.fetch),
//    not from cross-origin in-page fetches.
const out = { status: "error", doi: INPUT.doi || "", tried: [], http: null, bytes: 0 };
const CHALLENGE = /just a moment|verify you are human|attention required|access denied|are you a robot|security check|captcha|请稍候|请稍等/i;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function pageFor(task, slot) {
  // slot k -> k-th managed Page (tbsgen creates them sequentially with mode "ensure_pages" before parallel use)
  let pages = await task.pages();
  while (pages.length <= slot) {
    await task.newPage();
    pages = await task.pages();
  }
  return pages[slot];
}

async function info(page) {
  try {
    return await page.evaluate(() => ({
      url: location.href,
      title: document.title || "",
      ctype: document.contentType || "",
      text: (document.body && document.body.innerText || "").slice(0, 400),
      challenge: !!document.querySelector("#challenge-form, #cf-challenge-running, .cf-turnstile, #px-captcha, iframe[src*='challenges.cloudflare.com']"),
    }));
  } catch (e) {
    return { url: "", title: "", ctype: "", text: "", challenge: false, error: String(e).slice(0, 200) };
  }
}

function isChallenge(i) {
  return i.challenge || CHALLENGE.test(i.title) || (/^\s*$/.test(i.title) && CHALLENGE.test(i.text));
}

async function waitChallenge(page) {
  // wait for a bot-wall page to clear by itself; never interact with it
  let i = await info(page);
  const t0 = Date.now();
  while (isChallenge(i) && Date.now() - t0 < INPUT.challenge_wait_ms) {
    await sleep(2000);
    i = await info(page);
  }
  return i;
}

async function grabPdfDocument(page, i, file) {
  // the Page shows a PDF (Chrome viewer): re-read the same URL same-origin with the Page's cookies
  const r = await page.fetch(i.url, { saveAs: file, timeout: INPUT.download_timeout_ms });
  return { ok: r.ok, status: r.status, ctype: (r.headers && (r.headers["content-type"] || r.headers["Content-Type"])) || "" };
}

async function navStatus(page) {
  try {
    return await page.evaluate(() => {
      const n = performance.getEntriesByType("navigation")[0];
      return n && n.responseStatus ? n.responseStatus : null;
    });
  } catch (e) { return null; }
}

async function tryUrl(page, url, file) {
  out.tried.push(url.slice(0, 200));
  // Download waits cover the whole navigation: short waits are re-armed back to back while goto is pending,
  // plus one more after it settles.  Each wait settles before the next is armed (a Page allows one at a time),
  // so nothing is left pending when the next navigation starts.
  let navDone = false;
  const gotoP = page.goto(url, { timeout: INPUT.nav_timeout_ms, waitUntil: "domcontentloaded" })
    .catch(() => null)                 // a navigation that turns into a download aborts the document
    .finally(() => { navDone = true; });
  let dl = null;
  for (;;) {
    dl = await page.waitForEvent("download", { timeout: INPUT.download_wait_ms }).catch(() => null);
    if (dl || navDone) break;
  }
  await gotoP;
  if (dl) {
    await dl.saveAs(file);
    return { kind: "download", url: dl.url() };
  }
  out.http = await navStatus(page);
  if (out.http === 429) return { kind: "challenge", title: "HTTP 429" };
  let i = await waitChallenge(page);
  if (isChallenge(i)) return { kind: "challenge", title: i.title || i.text.slice(0, 80) };
  if (/pdf/i.test(i.ctype)) {
    const g = await grabPdfDocument(page, i, file);
    out.http = g.status;
    return g.ok ? { kind: "pdfdoc", url: i.url } : { kind: "http", status: g.status };
  }
  // HTML wrapper pages
  if (/sciencedirect\.com/.test(i.url) && /\/pdf/.test(url)) {
    // ScienceDirect pdfft: JS redirect to pdf.sciencedirectassets.com
    try {
      await page.waitForURL(/sciencedirectassets\.com/, { timeout: 20000 });
      i = await info(page);
      if (/pdf/i.test(i.ctype)) {
        const g = await grabPdfDocument(page, i, file);
        if (g.ok) return { kind: "pdfdoc", url: i.url };
      }
    } catch (e) { /* fall through */ }
  }
  const inner = await page.evaluate(() => {
    const pick = [];
    for (const el of document.querySelectorAll("iframe[src], embed[src], object[data]")) {
      const s = el.getAttribute("src") || el.getAttribute("data") || "";
      if (/pdf|getPDF|stampPDF/i.test(s)) pick.push(new URL(s, location.href).href);
    }
    const meta = document.querySelector("meta[http-equiv='refresh' i]");
    if (meta) {
      const m = /url=(.*)$/i.exec(meta.getAttribute("content") || "");
      if (m) pick.push(new URL(m[1].replace(/['"]/g, ""), location.href).href);
    }
    return pick;
  }).catch(() => []);
  if (inner.length) return { kind: "follow", urls: inner };
  return { kind: "html", url: i.url, title: i.title };
}

async function landingCandidates(page) {
  return await page.evaluate(() => {
    const c = [];
    const add = (u) => { try { if (u) c.push(new URL(u, location.href).href); } catch (e) {} };
    for (const m of document.querySelectorAll("meta[name='citation_pdf_url']")) add(m.getAttribute("content"));
    for (const a of document.querySelectorAll("a[href]")) {
      const h = a.getAttribute("href") || "";
      const t = (a.textContent || "").trim();
      if (/\/pdfft\b|\/doi\/pdf(direct)?\/|\/stamp\/stamp\.jsp|\/article-pdf\/|\.pdf(\?|$)/i.test(h) ||
          /^(download )?(full[- ]text )?pdf$|^download pdf|^view pdf/i.test(t)) add(h);
    }
    return [...new Set(c)].slice(0, 6);
  }).catch(() => []);
}

try {
  const task = await taskSpace(INPUT.space);
  out.space = task.spaceId;
  if (INPUT.mode === "ensure_pages") {
    for (let k = 0; k < INPUT.n; k++) await pageFor(task, k);
    out.status = "ok";
  } else if (INPUT.mode === "finish") {
    await task.finish({ keep: [] });
    out.status = "ok";
  } else if (INPUT.mode === "egress") {
    const page = await task.newPage();               // own Page: never disturbs a download slot
    out.bodies = {};
    for (const u of INPUT.urls) {
      try {
        await page.goto(u, { timeout: 20000 });
        out.bodies[u] = (await page.evaluate(() => document.body ? document.body.innerText : "")).slice(0, 2000);
      } catch (e) { out.bodies[u] = ""; }
    }
    try { await page.close(); } catch (e) {}
    out.status = "ok";
  } else {
    const page = await pageFor(task, INPUT.slot || 0);
    const file = INPUT.out_file;
    let queue = [...(INPUT.pdf_urls || [])];
    let landed = false;
    let last = null;
    for (let step = 0; step < INPUT.max_navigations && (queue.length || !landed); step++) {
      let url = queue.shift();
      if (!url) {
        // no (more) direct PDF URL: open the DOI landing page and read its PDF links
        landed = true;
        const landingUrl = (INPUT.landing_base || "https://doi.org/") + INPUT.doi;
        out.tried.push(landingUrl);
        try {
          await page.goto(landingUrl, { timeout: INPUT.nav_timeout_ms, waitUntil: "domcontentloaded" });
        } catch (e) { last = { kind: "nav", error: String(e).slice(0, 200) }; continue; }
        out.http = await navStatus(page);
        if (out.http === 429) { last = { kind: "challenge", title: "HTTP 429" }; break; }
        const i = await waitChallenge(page);
        if (isChallenge(i)) { last = { kind: "challenge", title: i.title }; break; }
        out.landing = i.url;
        const cands = await landingCandidates(page);
        queue.push(...cands.filter((u) => !out.tried.includes(u.slice(0, 200))));
        if (!queue.length) { last = { kind: "html", url: i.url, title: i.title, note: "no PDF link on landing page" }; break; }
        continue;
      }
      last = await tryUrl(page, url, file);
      if (last.kind === "download" || last.kind === "pdfdoc") break;
      if (last.kind === "challenge") break;
      if (last.kind === "follow") queue.unshift(...last.urls.filter((u) => !out.tried.includes(u.slice(0, 200))));
    }
    const fs = await import("node:fs/promises");
    let st = await fs.stat(file).catch(() => null);
    if (st && st.size > INPUT.max_bytes) { await fs.unlink(file).catch(() => {}); out.error = "too large"; st = null; }
    out.bytes = st ? st.size : 0;
    if (last && last.kind === "challenge") { out.status = "challenge"; out.title = last.title || ""; }
    else if (st && st.size > 0) { out.status = "downloaded"; out.path = file; out.final_url = last && last.url || ""; }
    else { out.status = last ? last.kind : "nothing"; out.error = JSON.stringify(last || {}).slice(0, 300); }
    try { await page.goto("about:blank", { timeout: 5000 }); } catch (e) {}
  }
} catch (e) {
  out.status = "error";
  out.error = String(e && e.stack || e).slice(0, 400);
}
console.log("TBSGEN_RESULT " + JSON.stringify(out));
process.exit(0);   // do not keep the script alive for timers of settled/abandoned waits
