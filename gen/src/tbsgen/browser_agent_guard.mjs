// tbsgen browser-agent guard.  The ego-browser wrapper of a browser batch (tbsgen.browser_agent) prepends this
// file, with one injected line `const GUARD = {...};` above it, to every script the agent runs.  It enforces the
// batch rules in code where the SDK allows it:
//   - only the batch's own TaskSpace (GUARD.space); no claiming / listing / taking over other spaces, no user tabs
//     (userPage, tabs, adopt, release, handOff) and no raw CDP;
//   - navigation and page.fetch only to GUARD.allow domains (the queued DOIs' publishers); a page that ends up
//     elsewhere is reset to about:blank and the call fails;
//   - no typing (fill only with GUARD.inputs values, which default to none; no keyboard / paste / insertText),
//     no mouse or drag (slider / checkbox challenges), no click / hover on a challenge page or on a target whose
//     selector names a captcha / verification widget;
//   - no reading cookies / storage / password fields from page JavaScript;
//   - files may only be saved under GUARD.files_dir.
// The wrapper additionally rejects scripts that mention the denied APIs at all.  This is a guard against
// mistakes, not a sandbox: the batch code validates every saved file and books every attempt itself.
{
  const G = GUARD;
  const realTaskSpace = globalThis.taskSpace;
  const deny = (n) => () => { throw new Error(`tbsgen guard: ${n} is not allowed in this batch`); };
  for (const n of ["claimTaskSpace", "takeOverTaskSpace", "listTaskSpaces"]) globalThis[n] = deny(n);
  const CHALLENGE_TITLE = /just a moment|verify you are human|attention required|are you a robot|security check|captcha|human verification|请稍候|人机验证|安全验证/i;
  const CHALLENGE_SEL = /captcha|turnstile|challenge|robot|verify|human|cf-|验证/i;
  const EVAL_DENY = /cookie|localStorage|sessionStorage|indexedDB|password|credential|\.value\s*=|location\s*=|location\.(href\s*=|assign|replace)|window\.open|\bfetch\s*\(|XMLHttpRequest|sendBeacon|WebSocket/i;
  const KEYS_OK = new Set(["Enter", "Escape", "Tab", "ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", "Space"]);
  const hostOk = (u) => {
    let h;
    try { h = new URL(u).hostname.toLowerCase(); } catch (e) { return false; }
    return G.allow.some((d) => h === d || h.endsWith("." + d));
  };
  const urlOk = (u) => !u || u === "about:blank" || u.startsWith("chrome-error:") || hostOk(u);
  const inFiles = (p) => typeof p === "string" && p.startsWith(G.files_dir + "/") && !p.includes("/../");
  async function after(page) {
    let u = "";
    try { u = await page.url(); } catch (e) { return; }
    if (!urlOk(u)) {
      try { await page.goto("about:blank", { timeout: 5000 }); } catch (e) {}
      let h = "";
      try { h = new URL(u).hostname; } catch (e) {}
      throw new Error(`tbsgen guard: page left the allowed domains (${h}); it was reset to about:blank`);
    }
  }
  const challengeProbe = () => {
    const t = document.title || "";
    const sel = "#challenge-form, #cf-challenge-running, .cf-turnstile, #px-captcha, iframe[src*='challenges.cloudflare.com']";
    return { t, c: !!document.querySelector(sel) };
  };
  async function isChallenge(page) {
    let r = { t: "", c: false };
    try { r = await page.evaluate(challengeProbe); } catch (e) {}
    return r.c || CHALLENGE_TITLE.test(r.t);
  }
  const BLOCK = new Set(["cdp", "keyboard", "mouse", "type", "insertText", "paste", "setInputFiles", "waitForFileChooser", "dragAndDrop"]);
  const ACT = new Set(["click", "dblclick", "hover"]);
  function wrapDownload(d) {
    return new Proxy(d, {
      get(t, p) {
        if (p === "saveAs") return async (path) => {
          if (!inFiles(path)) throw new Error("tbsgen guard: save downloads only under " + G.files_dir);
          return t.saveAs(path);
        };
        const v = t[p];
        return typeof v === "function" ? v.bind(t) : v;
      },
    });
  }
  function wrapPage(pg) {
    if (!pg || typeof pg !== "object") return pg;
    return new Proxy(pg, {
      get(t, p) {
        if (BLOCK.has(p)) return deny(`page.${String(p)}`);
        if (p === "goto") return async (url, o) => {
          if (!urlOk(url)) throw new Error(`tbsgen guard: ${url} is outside the allowed domains`);
          const r = await t.goto(url, o);
          await after(t);
          return r;
        };
        if (p === "fetch") return async (url, o = {}) => {
          let abs = url;
          try { abs = new URL(url, await t.url()).href; } catch (e) {}
          if (!hostOk(abs)) throw new Error(`tbsgen guard: fetch of ${abs} is outside the allowed domains`);
          if (o.method && !/^(GET|HEAD)$/i.test(o.method)) throw new Error("tbsgen guard: only GET/HEAD fetches");
          if (o.headers && Object.keys(o.headers).some((k) => /authorization|cookie/i.test(k))) throw new Error("tbsgen guard: no auth headers");
          if (o.saveAs !== undefined && !inFiles(o.saveAs)) throw new Error("tbsgen guard: saveAs only under " + G.files_dir);
          return t.fetch(url, o);
        };
        if (p === "fill") return async (sel, value, o) => {
          if (!G.inputs.includes(value)) throw new Error("tbsgen guard: typing is not allowed (no credentials, no free text)");
          const r = await t.fill(sel, value, o);
          await after(t);
          return r;
        };
        if (p === "press") return async (sel, key, o) => {
          if (!KEYS_OK.has(key)) throw new Error(`tbsgen guard: key ${key} is not allowed`);
          const r = await t.press(sel, key, o);
          await after(t);
          return r;
        };
        if (p === "evaluate") return async (fn, arg) => {
          if (EVAL_DENY.test(String(fn))) throw new Error("tbsgen guard: this page script is not allowed (storage, credentials, navigation, requests)");
          const r = await t.evaluate(fn, arg);
          await after(t);
          return r;
        };
        if (p === "acceptDialog") return async (txt) => {
          if (txt !== undefined) throw new Error("tbsgen guard: no prompt text");
          return t.acceptDialog();
        };
        if (p === "screenshot") return async (o = {}) => {
          if (o.path !== undefined && !inFiles(o.path)) throw new Error("tbsgen guard: screenshots only under " + G.files_dir);
          return t.screenshot(o);
        };
        if (ACT.has(p)) return async (sel, o) => {
          if (CHALLENGE_SEL.test(String(sel)) || (o && o.label && CHALLENGE_SEL.test(String(o.label))))
            throw new Error("tbsgen guard: never interact with a verification widget");
          if (await isChallenge(t)) throw new Error("tbsgen guard: this is a verification page; do not interact, report it as challenge");
          const r = await t[p](sel, o);
          await after(t);
          return r;
        };
        if (p === "waitForEvent") return async (ev, o) => {
          const r = await t.waitForEvent(ev, o);
          if (ev === "download") return wrapDownload(r);
          if (ev === "popup") { await after(r); return wrapPage(r); }
          return r;
        };
        if (["reload", "waitForURL", "waitForLoadState", "waitForSelector", "waitForFunction", "selectOption"].includes(p))
          return async (...a) => { const r = await t[p](...a); await after(t); return r; };
        const v = t[p];
        return typeof v === "function" ? v.bind(t) : v;
      },
    });
  }
  function wrapTask(task) {
    return new Proxy(task, {
      get(t, p) {
        if (["userPage", "tabs", "adopt", "release", "handOff", "cdp", "waitForControl"].includes(p)) return deny(`task.${String(p)}`);
        if (p === "page") return (label) => wrapPage(t.page(label));
        if (p === "newPage") return async (...a) => wrapPage(await t.newPage(...a));
        if (p === "pages") return async () => (await t.pages()).map(wrapPage);
        const v = t[p];
        return typeof v === "function" ? v.bind(t) : v;
      },
    });
  }
  let spaceId = null;
  globalThis.taskSpace = async (nameOrId) => {
    if (nameOrId !== G.space && !(spaceId !== null && String(nameOrId) === String(spaceId)) && nameOrId !== undefined) {
      throw new Error(`tbsgen guard: use only the batch TaskSpace "${G.space}"`);
    }
    const task = await realTaskSpace(G.space);
    spaceId = task.spaceId;
    return wrapTask(task);
  };
}
