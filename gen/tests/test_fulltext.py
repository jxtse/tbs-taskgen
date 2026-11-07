"""S3 full-text cascade, download ledger, browser channel and queue.  Offline: every HTTP response and every
browser run is faked; no network, no browser."""
import json
import sqlite3
import time
from pathlib import Path

import pytest

from tbsgen import browser_ego, fulltext
from tbsgen.config import load_config
from tbsgen.dlledger import DownloadLedger, LimitReached, block_signal, challenge_page
from tbsgen.fulltext import Fetcher, Resp

@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch):
    """Tests must not see credentials of the machine they run on."""
    for k in ("ELSEVIER_API_KEY", "ELSEVIER_INSTTOKEN", "WILEY_TDM_TOKEN", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
              "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)


TITLE = "Robust estimation of drift in noisy sensor arrays"
BODY = ("Introduction. " + "We study robust estimation of drift in noisy sensor arrays with a least-squares "
        "method. " * 120 + " Methods. Results. Discussion. Conclusions. References.")
ELS_XML = (f"<full-text-retrieval-response><coredata><dc:title>{TITLE}</dc:title></coredata><originalText>"
           f"<ce:sections><ce:section><ce:section-title>Introduction</ce:section-title><ce:para>{BODY}</ce:para>"
           "</ce:section></ce:sections></originalText></full-text-retrieval-response>").encode()
ELS_ABSTRACT_ONLY = (f"<full-text-retrieval-response><coredata><dc:title>{TITLE}</dc:title>"
                     "<dc:description>abstract</dc:description></coredata></full-text-retrieval-response>").encode()
JATS = (f"<article><front><article-title>{TITLE}</article-title></front><body><sec><title>Intro</title>"
        f"<p>{BODY}</p></sec></body></article>").encode()
FAKE_PDF = b"%PDF-1.7 fake"
CF = b"<html><head><title>Just a moment...</title></head><body>cf-chl</body></html>"


class FakeClock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class FakeNet:
    """url-substring -> Resp | callable; records every request (url, headers)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, headers, timeout, max_bytes):
        self.calls.append((url, dict(headers)))
        for k, v in self.routes.items():
            if k in url:
                v = v(url) if callable(v) else v
                return v if isinstance(v, Resp) else Resp(200, {}, v, url)
        return Resp(404, {}, b"not found", url)

    def hosts(self):
        return [u for u, _ in self.calls]


def mkcfg(tmp_path, **ft):
    base = {"cache_dirs": [], "write_cache_dir": str(tmp_path / "wc"), "ledger_path": str(tmp_path / "dl.sqlite"),
            "queue_path": str(tmp_path / "q.jsonl"), "allow_network": True,
            "oa": {"enabled": True, "europepmc": False},
            "elsevier": {"enabled": True}, "wiley": {"enabled": True},
            "browser": {"space": "t-" + str(abs(hash(str(tmp_path))))},
            "limits": {ch: {"gap_s": 0.0, "jitter_s": 0.0} for ch in ("oa", "elsevier", "wiley", "inst", "browser")}}
    for k, v in ft.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return load_config(None, {"run": {"root": str(tmp_path / "run")}, "fulltext": base})


def mkfetcher(tmp_path, routes, secrets=None, clock=None, browser_runner=None, **ft):
    cfg = mkcfg(tmp_path, **ft)
    clock = clock or FakeClock()
    led = DownloadLedger(cfg["fulltext"]["ledger_path"], cfg["fulltext"]["limits"], cfg["fulltext"]["global_limits"],
                         cfg["fulltext"]["host_caps"], clock=clock, sleep=clock.sleep)
    net = FakeNet(routes)
    f = Fetcher(cfg, ledger=led, transport=net, secrets=secrets or {}, browser_runner=browser_runner)
    return f, net, led, clock


SECRETS = {"ELSEVIER_API_KEY": "k-els-test", "WILEY_TDM_TOKEN": "t-wiley-test"}


# ------------------------------------------------------------------ cascade order / routing
def test_cache_first_no_network(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    (ro / f"{fulltext.doi_key('10.1016/j.x.2020.1')}.txt").write_text(BODY)
    f, net, _, _ = mkfetcher(tmp_path, {}, SECRETS, cache_dirs=[str(ro)])
    out = f.fetch(dict(doi="10.1016/J.X.2020.1", title=TITLE))
    assert out.text and out.rec["source"] == "cache" and net.calls == []
    assert list(ro.iterdir()) == [ro / "10.1016_j.x.2020.1.txt"]          # read-only cache untouched


def test_elsevier_prefix_routing_and_headers(tmp_path):
    f, net, led, _ = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML}, SECRETS)
    out = f.fetch(dict(doi="10.1016/j.x.2020.1", title=TITLE))
    assert out.rec["source"] == "elsevier", out.rec
    url, hdr = net.calls[-1]
    assert url.startswith("https://api.elsevier.com/content/article/doi/10.1016/j.x.2020.1")
    assert hdr["X-ELS-APIKey"] == "k-els-test" and hdr["Accept"] == "text/xml"
    assert not any("api.wiley.com" in u for u in net.hosts())
    # provenance + write cache, no secret anywhere on disk
    wc = tmp_path / "wc"
    prov = json.loads((wc / "10.1016_j.x.2020.1.json").read_text())
    assert prov["channel"] == "elsevier" and prov["bytes"] == len(ELS_XML) and prov["title_match"] >= 0.6
    for p in wc.rglob("*"):
        if p.is_file():
            assert b"k-els-test" not in p.read_bytes()
    assert b"k-els-test" not in Path(tmp_path / "dl.sqlite").read_bytes()
    assert led.usage()["elsevier"]["ok"] == 1


def test_wiley_prefixes_and_non_matching_doi_skips_tdm(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    f, net, _, _ = mkfetcher(tmp_path, {"api.wiley.com": FAKE_PDF}, SECRETS)
    for doi in ("10.1002/abc.1", "10.1111/xyz.2"):
        out = f.fetch(dict(doi=doi, title=TITLE))
        assert out.rec["source"] == "wiley"
        assert net.calls[-1][1]["Wiley-TDM-Client-Token"] == "t-wiley-test"
    n = len(net.calls)
    out = f.fetch(dict(doi="10.1234/other", title=TITLE))
    assert out.text is None and len(net.calls) == n          # no TDM request for other publishers


def test_cascade_order_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    routes = {"api.elsevier.com": ELS_XML, "oa.example.org": FAKE_PDF}
    c = dict(doi="10.1016/j.x.2020.1", title=TITLE, oa_url="https://oa.example.org/a.pdf")
    f, net, _, _ = mkfetcher(tmp_path / "a", routes, SECRETS)
    assert f.fetch(c).rec["source"] == "oa"                                 # default: cache, oa, elsevier
    f, net, _, _ = mkfetcher(tmp_path / "b", routes, SECRETS, cascade=["cache", "elsevier", "oa"])
    assert f.fetch(c).rec["source"] == "elsevier"
    assert "oa.example.org" not in " ".join(net.hosts())


def test_missing_secret_skips_channel(tmp_path):
    f, net, _, _ = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML}, {})
    out = f.fetch(dict(doi="10.1016/j.x.2020.1", title=TITLE))
    assert out.text is None and net.calls == []
    assert any("ELSEVIER_API_KEY not set" in s for s in out.rec["skipped"])


def test_disabled_channels_and_allow_network(tmp_path):
    f, net, _, _ = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML}, SECRETS, elsevier={"enabled": False})
    assert f.fetch(dict(doi="10.1016/j.x.1", title=TITLE)).text is None and net.calls == []
    f, net, _, _ = mkfetcher(tmp_path / "n", {"api.elsevier.com": ELS_XML}, SECRETS, allow_network=False)
    out = f.fetch(dict(doi="10.1016/j.x.1", title=TITLE, oa_url="https://x.org/a.pdf"))
    assert out.text is None and net.calls == []


def test_elsevier_abstract_only_is_a_failure(tmp_path):
    f, _, led, _ = mkfetcher(tmp_path, {"api.elsevier.com": ELS_ABSTRACT_ONLY}, SECRETS)
    out = f.fetch(dict(doi="10.1016/j.x.1", title=TITLE))
    assert out.text is None and "short text" in out.rec["errors"][-1]
    assert led.usage()["elsevier"] == dict(led.usage()["elsevier"], ok=0, attempts=1)


def test_inst_routes_from_config_only(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    routes = {"link.springer.com": FAKE_PDF}
    f, net, _, _ = mkfetcher(tmp_path / "a", routes, SECRETS, inst={"enabled": True})
    assert f.fetch(dict(doi="10.1007/s1-2", title=TITLE)).text is None and net.calls == []   # default: no routes
    f, net, _, _ = mkfetcher(tmp_path / "b", routes, SECRETS, inst={
        "enabled": True, "routes": [{"prefix": "10.1007/", "url": "https://link.springer.com/content/pdf/{doi}.pdf"}]})
    out = f.fetch(dict(doi="10.1007/s1-2", title=TITLE))
    assert out.rec["source"] == "inst" and net.calls[0][0] == "https://link.springer.com/content/pdf/10.1007/s1-2.pdf"


def test_europepmc_jats(tmp_path):
    routes = {"rest/search": json.dumps({"resultList": {"result": [
        {"pmcid": "PMC123", "inEPMC": "Y", "isOpenAccess": "Y"}]}}).encode(),
        "PMC123/fullTextXML": JATS}
    f, net, led, _ = mkfetcher(tmp_path, routes, SECRETS, oa={"europepmc": True})
    out = f.fetch(dict(doi="10.9999/x.1", title=TITLE))
    assert out.rec["source"] == "oa" and out.rec["provenance"]["kind"] == "jats"
    u = led.usage()
    assert u["oa"]["ok"] == 1 and u["_meta_calls"] == 1 and u["_global_ok"] == 1   # search not a download


# ------------------------------------------------------------------ validation
def test_title_mismatch_rejected(tmp_path):
    f, _, _, _ = mkfetcher(tmp_path, {"api.elsevier.com": ELS_XML}, SECRETS)
    out = f.fetch(dict(doi="10.1016/j.x.1", title="Completely different paper about volcanic ash plumes"))
    assert out.text is None and "title mismatch" in out.rec["errors"][-1]
    assert not (tmp_path / "wc" / "10.1016_j.x.1.txt").exists()


def test_html_wrapper_instead_of_pdf_rejected(tmp_path):
    wrapper = b"<html><head><title>Article</title></head><body><iframe src='/stamp/getPDF'></iframe></body></html>"
    f, _, led, _ = mkfetcher(tmp_path, {"pub.example.org": wrapper}, SECRETS, inst={
        "enabled": True, "routes": [{"prefix": "10.5555/", "url": "https://pub.example.org/{suffix}.pdf"}]})
    out = f.fetch(dict(doi="10.5555/a1", title=TITLE))
    assert out.text is None and "not a PDF" in out.rec["errors"][-1]
    assert led.cooling("pub.example.org") is None                          # a wrapper page is not a block


def test_challenge_detection():
    assert challenge_page(CF) and block_signal(200, CF) == 24 * 3600
    assert block_signal(429) == 6 * 3600
    assert block_signal(403, b"<html><title>Purchase this article</title></html>") is None   # paywall 403
    assert block_signal(403, b"<html><head><title>Access Denied</title></head></html>") == 24 * 3600
    page_with_recaptcha_widget = (b"<html><head><title>Paper title</title></head><body><div class='g-recaptcha'>"
                                  b"</div> please solve the captcha to comment</body></html>")
    assert not challenge_page(page_with_recaptcha_widget)
    assert block_signal(200, page_with_recaptcha_widget) is None


# ------------------------------------------------------------------ ledger: caps, pacing, cooldown, breaker
def test_cloudflare_cools_host_and_blocks_later_requests(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"pub.example.org": Resp(403, {}, CF, "https://pub.example.org/x")},
                                   SECRETS, inst={"enabled": True, "routes": [
                                       {"prefix": "10.5555/", "url": "https://pub.example.org/{suffix}.pdf"}]})
    f.fetch(dict(doi="10.5555/a1", title=TITLE))
    assert led.cooling("pub.example.org")
    n = len(net.calls)
    out = f.fetch(dict(doi="10.5555/a2", title=TITLE))
    assert len(net.calls) == n and any("cooldown" in s for s in out.rec["skipped"])
    clock.t += 24 * 3600 + 1
    assert led.cooling("pub.example.org") is None


def test_daily_cap_and_hourly_cap(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"elsevier": {"daily_ok": 3, "hourly_ok": 2, "gap_s": 0, "jitter_s": 0}},
                         clock=clock, sleep=clock.sleep)
    for i in range(2):
        rid = led.acquire("elsevier", "api.elsevier.com", f"d{i}")
        led.record("elsevier", "api.elsevier.com", f"d{i}", "ok", 200, 10, reservation=rid)
    with pytest.raises(LimitReached, match="hourly"):
        led.acquire("elsevier", "api.elsevier.com", "d2")
    clock.t += 3601
    rid = led.acquire("elsevier", "api.elsevier.com", "d2")
    led.record("elsevier", "api.elsevier.com", "d2", "ok", 200, 10, reservation=rid)
    clock.t += 3601
    with pytest.raises(LimitReached, match="daily success cap 3"):
        led.acquire("elsevier", "api.elsevier.com", "d3")


def test_pending_reservations_count_against_caps(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"browser": {"daily_ok": 1, "gap_s": 0, "jitter_s": 0}},
                         clock=clock, sleep=clock.sleep)
    led.acquire("browser", "chrome:10.1021", "a")                   # not yet recorded: still holds the slot
    with pytest.raises(LimitReached):
        led.acquire("browser", "chrome:10.1093", "b")


def test_per_host_gap_and_global_cap(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"browser": {"gap_s": 60, "jitter_s": 60}},
                         global_limits={"daily_ok": 2}, clock=clock, sleep=clock.sleep)
    t0 = clock.t
    r = led.acquire("browser", "chrome:10.1021", "a")
    led.record("browser", "chrome:10.1021", "a", "ok", 200, 1, reservation=r)
    r = led.acquire("browser", "chrome:10.1021", "b")                 # same publisher: waits 60-120 s
    assert 60 <= clock.t - t0 <= 120
    led.record("browser", "chrome:10.1021", "b", "ok", 200, 1, reservation=r)
    t1 = clock.t
    with pytest.raises(LimitReached, match="global"):
        led.acquire("browser", "chrome:10.1093", "c")
    assert clock.t == t1


def test_other_publisher_not_delayed(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"browser": {"gap_s": 60, "jitter_s": 60}}, clock=clock,
                         sleep=clock.sleep)
    t0 = clock.t
    led.acquire("browser", "chrome:10.1021", "a")
    led.acquire("browser", "chrome:10.1093", "b")
    assert clock.t == t0


def test_breaker_on_repeated_blocks_and_failures(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"wiley": {"gap_s": 0, "jitter_s": 0}},
                         global_limits={"breaker_blocks_per_day": 2, "breaker_consecutive_fail": 3},
                         clock=clock, sleep=clock.sleep)
    for i in range(3):
        r = led.acquire("wiley", "api.wiley.com", f"d{i}")
        led.record("wiley", "api.wiley.com", f"d{i}", "fail", 401, 0, reservation=r)
    with pytest.raises(LimitReached) as e:
        led.acquire("wiley", "api.wiley.com", "d9")
    assert e.value.kind == "breaker"
    led2 = DownloadLedger(tmp_path / "l2.sqlite", global_limits={"breaker_blocks_per_day": 2}, clock=clock,
                          sleep=clock.sleep)
    led2.block("chrome:10.1021", "cf", channel="browser")
    led2.block("chrome:10.1093", "cf", channel="browser")
    with pytest.raises(LimitReached, match="breaker"):
        led2.acquire("browser", "chrome:10.1117", "x")


def test_oa_breaker_ignores_failure_runs(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"oa": {"gap_s": 0, "jitter_s": 0}}, clock=clock, sleep=clock.sleep)
    for i in range(20):
        r = led.acquire("oa", f"repo{i}.example.org", f"d{i}")
        led.record("oa", f"repo{i}.example.org", f"d{i}", "fail", 404, 0, reservation=r)
    led.acquire("oa", "arxiv.org", "x")                              # still allowed


def test_ledger_persists_across_restarts(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"oa": {"daily_ok": 1, "gap_s": 0, "jitter_s": 0}}, clock=clock,
                         sleep=clock.sleep)
    r = led.acquire("oa", "arxiv.org", "a")
    led.record("oa", "arxiv.org", "a", "ok", 200, 1, reservation=r)
    led.block("www.mdpi.com", "akamai")
    led = DownloadLedger(tmp_path / "l.sqlite", {"oa": {"daily_ok": 1}}, clock=clock, sleep=clock.sleep)
    with pytest.raises(LimitReached):
        led.acquire("oa", "arxiv.org", "b")
    assert led.cooling("www.mdpi.com")


def test_shares_legacy_ledger_schema(tmp_path):
    """An earlier pipeline's ledger file (attempts/cooldown/hosts) is used as is; its rows count."""
    p = tmp_path / "legacy.sqlite"
    db = sqlite3.connect(p)
    db.executescript("""
        CREATE TABLE attempts(ts real, day text, channel text, host text, doi text,
                              status text, http int, bytes int, note text);
        CREATE TABLE cooldown(host text primary key, until real, reason text);
        CREATE INDEX a_day on attempts(day, channel);
        CREATE TABLE hosts(host text primary key, next_ts real);""")
    clock = FakeClock()
    day = time.strftime("%Y-%m-%d", time.localtime(clock.t))
    db.executemany("insert into attempts values(?,?,?,?,?,?,?,?,?)",
                   [(clock.t - 7200, day, "wiley", "api.wiley.com", f"d{i}", "ok", 200, 1, "") for i in range(2)])
    db.execute("insert into cooldown values('chrome:10.1093', ?, 'challenge')", (clock.t + 3600,))
    db.commit()
    db.close()
    led = DownloadLedger(p, {"wiley": {"daily_ok": 2}}, clock=clock, sleep=clock.sleep)
    with pytest.raises(LimitReached, match="wiley daily success cap"):
        led.acquire("wiley", "api.wiley.com", "x")
    assert led.cooling("chrome:10.1093")
    cols = [r[1] for r in sqlite3.connect(p).execute("pragma table_info(attempts)")]
    assert cols == ["ts", "day", "channel", "host", "doi", "status", "http", "bytes", "note"]


# ------------------------------------------------------------------ network hygiene
def test_default_transport_ignores_env_proxy(monkeypatch):
    import urllib.request
    seen = {}
    real = urllib.request.build_opener

    def spy(*handlers):
        seen["handlers"] = handlers
        return real(*handlers)
    monkeypatch.setattr(urllib.request, "build_opener", spy)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    fulltext.make_transport(False)
    ph = [h for h in seen["handlers"] if isinstance(h, urllib.request.ProxyHandler)]
    assert ph and ph[0].proxies == {}
    fulltext.make_transport(True)
    assert not any(isinstance(h, urllib.request.ProxyHandler) for h in seen["handlers"])


def test_egress_check_refuses_outside_allowed_ranges(tmp_path):
    ec = {"enabled": True, "allow_cidrs": ["198.51.100.0/24"], "urls": ["https://ip.example/"]}
    f, net, _, _ = mkfetcher(tmp_path / "a", {"ip.example": b"203.0.113.7\n", "api.elsevier.com": ELS_XML},
                             SECRETS, egress_check=ec)
    out = f.fetch(dict(doi="10.1016/j.x.1", title=TITLE))
    assert out.text is None and "egress" in out.rec["egress_refused"]
    assert not any("api.elsevier.com" in u for u in net.hosts())
    f, net, _, _ = mkfetcher(tmp_path / "b", {"ip.example": b"198.51.100.20", "api.elsevier.com": ELS_XML},
                             SECRETS, egress_check=ec)
    assert f.fetch(dict(doi="10.1016/j.x.1", title=TITLE)).rec["source"] == "elsevier"
    assert len([u for u in net.hosts() if "ip.example" in u]) == 1          # cached for ttl_s


def test_signed_query_strings_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    f, _, _, _ = mkfetcher(tmp_path, {"cdn.example.org": FAKE_PDF}, SECRETS)
    out = f.fetch(dict(doi="10.9/x", title=TITLE, oa_url="https://cdn.example.org/a.pdf?token=SECRET123&e=1"))
    assert out.rec["source"] == "oa" and "SECRET123" not in json.dumps(out.rec)
    assert b"SECRET123" not in (tmp_path / "wc" / "10.9_x.json").read_bytes()


# ------------------------------------------------------------------ browser channel + queue
def fake_browser(results):
    calls = []

    def run(job):
        calls.append(job)
        r = dict(results(job) if callable(results) else results)
        if r.get("status") == "downloaded":
            p = Path(job["out_dir"]) / f"dl-{len(calls)}.bin"
            p.write_bytes(r.pop("_body"))
            r = dict(r, path=str(p), bytes=p.stat().st_size)
        return r
    return run, calls


def test_browser_queue_only_by_default(tmp_path):
    run, calls = fake_browser({"status": "downloaded", "_body": FAKE_PDF})
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    out = f.fetch(dict(doi="10.1021/acs.1", title=TITLE))
    assert out.text is None and calls == [] and any("queued" in s for s in out.rec["skipped"])


def test_browser_challenge_cools_publisher(tmp_path):
    run, calls = fake_browser({"status": "challenge", "title": "Just a moment..."})
    f, _, led, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    out = f.fetch(dict(doi="10.1021/acs.1", title=TITLE), browser=True)
    assert out.text is None and led.cooling("chrome:10.1021")
    out = f.fetch(dict(doi="10.1021/acs.2", title=TITLE), browser=True)
    assert len(calls) == 1 and any("cooldown" in s for s in out.rec["skipped"])


def test_browser_html_instead_of_pdf_and_success(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    wrapper = b"<html><head><title>ScienceDirect</title></head><body>Loading PDF...</body></html>"
    run, _ = fake_browser(lambda job: {"status": "downloaded", "_body": wrapper if "1016" in job["doi"] else FAKE_PDF,
                                       "final_url": "https://pubs.example/x.pdf?sig=abc"})
    f, _, led, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True},
                             elsevier={"enabled": False})
    bad = f.fetch(dict(doi="10.1016/j.x.1", title=TITLE), browser=True)
    assert bad.text is None and "not a PDF or full-text article page" in bad.rec["errors"][-1]
    good = f.fetch(dict(doi="10.1021/acs.1", title=TITLE), browser=True)
    assert good.rec["source"] == "browser" and "sig=abc" not in good.rec["url"]
    assert led.usage()["browser"]["ok"] == 1
    assert not list((tmp_path / "wc" / "_browser_tmp").iterdir())          # temp download removed


def test_queue_idempotent_and_drain(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    run, calls = fake_browser(lambda job: {"status": "downloaded", "_body": FAKE_PDF} if "10.1021" in job["doi"]
                              else {"status": "html", "error": "no PDF link"})
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run,
                           browser={"enabled": True, "concurrency": 2})
    qp = fulltext.queue_path(f.cfg)
    assert fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.1021/acs.1", title=TITLE), subscribers=[["r", "i1"]]))
    assert not fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.1021/ACS.1", title=TITLE),
                                                         subscribers=[["r", "i2"]]))
    assert fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.1093/x.2", title=TITLE), subscribers=[["r", "i3"]]))
    q = fulltext.read_queue(qp)
    assert len(q) == 2 and q[0]["subscribers"] == [["r", "i1"], ["r", "i2"]]
    done = []

    def ack(e, o, sub):
        done.append(sub[1])
        return True
    st = fulltext.drain_queue(f, max_papers=10, on_success=ack, log=lambda s: None, now=1000.0)
    assert st["ok"] == 1 and st["failed"] == 1 and st["remaining"] == 1 and sorted(done) == ["i1", "i2"]
    assert {c["slot"] for c in calls} <= {0, 1}
    left = fulltext.read_queue(qp)
    assert left[0]["doi"] == "10.1093/x.2" and left[0]["tries"] == 1 and left[0]["next_try"] > 1000.0
    # backed off: not due right away, due after the backoff
    assert fulltext.drain_queue(f, max_papers=10, log=lambda s: None, now=1001.0)["considered"] == 0
    assert fulltext.drain_queue(f, max_papers=10, log=lambda s: None, now=1000.0 + 7 * 3600)["considered"] == 1


def test_drain_keeps_entry_when_reactivation_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    run, _ = fake_browser({"status": "downloaded", "_body": FAKE_PDF})
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    qp = fulltext.queue_path(f.cfg)
    fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.1021/acs.1", title=TITLE), subscribers=[["r", "i1"]]))

    def boom(e, o, sub):
        raise OSError("disk full")
    st = fulltext.drain_queue(f, on_success=boom, log=lambda s: None)
    assert st["ok"] == 1 and any("disk full" in str(x) for x in st["errors"])
    assert [e["doi"] for e in fulltext.read_queue(qp)] == ["10.1021/acs.1"]     # kept; text is cached now
    st = fulltext.drain_queue(f, on_success=lambda e, o, s: True, log=lambda s: None)
    assert st["by_channel"] == {"cache": 1} and fulltext.read_queue(qp) == []


def test_subscriber_added_during_drain_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    qp_holder = {}

    def run_and_enqueue(job):
        fulltext.enqueue(qp_holder["qp"], fulltext.queue_entry(dict(doi="10.1021/acs.1", title=TITLE),
                                                               subscribers=[["r", "late"]]))
        p = Path(job["out_dir"]) / "x.bin"
        p.write_bytes(FAKE_PDF)
        return {"status": "downloaded", "path": str(p), "bytes": len(FAKE_PDF)}
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run_and_enqueue, browser={"enabled": True})
    qp_holder["qp"] = qp = fulltext.queue_path(f.cfg)
    fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.1021/acs.1", title=TITLE), subscribers=[["r", "i1"]]))
    fulltext.drain_queue(f, on_success=lambda e, o, s: True, log=lambda s: None)
    q = fulltext.read_queue(qp)
    assert len(q) == 1 and q[0]["subscribers"] == [["r", "late"]]


def test_queue_concurrent_enqueue_is_not_lost(tmp_path):
    import threading
    qp = tmp_path / "q.jsonl"
    ths = [threading.Thread(target=fulltext.enqueue, args=(qp, fulltext.queue_entry(
        dict(doi=f"10.1/x{i}", title="t"), subscribers=[["r", f"i{i}"]]))) for i in range(20)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert len(fulltext.read_queue(qp)) == 20


def test_queue_keeps_retrieval_fields_and_doi_less_papers(tmp_path):
    qp = tmp_path / "q.jsonl"
    c = dict(title="No DOI paper", openAccessPdf={"url": "https://repo.example/p.pdf"}, pmcid="PMC1", arxiv="2101.1")
    e = fulltext.queue_entry(c, subscribers=[["r", "i"]])
    assert fulltext.enqueue(qp, e)
    [got] = fulltext.read_queue(qp)
    assert got["openAccessPdf"]["url"] == "https://repo.example/p.pdf" and got["pmcid"] == "PMC1"
    assert got["ident"] == "arxiv:2101.1"


def test_s3_parks_and_queues_then_fetch_reactivates(tmp_path, monkeypatch):
    """Pipeline level: S3 with nothing reachable parks + queues; a later fetch success re-runs S3 from cache."""
    from tbsgen import cli
    from tbsgen.pipeline import Pipeline
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    cfg = mkcfg(tmp_path, oa={"enabled": False}, browser={"enabled": True})
    P = Pipeline(cfg)
    P.fetcher = Fetcher(cfg, transport=FakeNet({}), secrets={})
    [iid] = P.ingest([dict(doi="10.1021/acs.9", title=TITLE,
                           abstract="We propose an estimation algorithm with simulation benchmark; code on github.")])
    P.run(max_steps=3)
    it = P.store.item(iid)
    assert it.state["status"] == "parked" and it.state["reason"] == "fulltext_unavailable"
    q = fulltext.read_queue(fulltext.queue_path(cfg))
    assert [e["doi"] for e in q] == ["10.1021/acs.9"] and q[0]["subscribers"] == [[str(P.store.root), iid]]
    P.run(max_steps=3)                                                  # parked: not retried, not re-queued twice
    assert len(fulltext.read_queue(fulltext.queue_path(cfg))) == 1
    run, _ = fake_browser({"status": "downloaded", "_body": FAKE_PDF})
    P.fetcher.browser_runner = run

    class A:
        status = finish_browser = dry_run = no_browser = False
        max, channels = 5, ""
    monkeypatch.setattr(sys_mod(), "stderr", open("/dev/null", "w"))
    cli._fetch(P, A())
    it = P.store.item(iid)
    assert it.state["status"] == "active" and it.state["stage"] == "S4"
    assert it.p("fulltext.txt").read_text().startswith(TITLE) and it.state["history"][-1]["source"] == "browser"
    assert fulltext.read_queue(fulltext.queue_path(cfg)) == []


def sys_mod():
    import sys
    return sys


def test_browser_script_renders_input_without_env(tmp_path):
    js = browser_ego.render({"space": "s", "doi": "10.1/x", "out_file": "/abs/x.bin"})
    assert js.startswith("const INPUT = {") and "TBSGEN_RESULT" in js and "process.env" not in js
    assert browser_ego.publisher_key("10.1016/j.x") == "chrome:10.1016"
    assert browser_ego.pdf_url_hint("10.1021/acs.jcim.5b00654")[0].startswith("https://pubs.acs.org/doi/pdf/")
    assert browser_ego.binary({"binary": "/nonexistent/ego-browser"}) is None


# ------------------------------------------------------------------ review round 1 regressions
def test_redirect_strips_credentials_and_refuses_downgrade(tmp_path):
    routes = {"api.elsevier.com": Resp(302, {"Location": "https://cdn.other.example/x.xml?sig=1"}, b"",
                                       "https://api.elsevier.com/x"),
              "cdn.other.example": ELS_XML}
    f, net, _, _ = mkfetcher(tmp_path, routes, SECRETS)
    out = f.fetch(dict(doi="10.1016/j.x.1", title=TITLE))
    assert out.rec["source"] == "elsevier"
    (u1, h1), (u2, h2) = net.calls
    assert "X-ELS-APIKey" in h1 and "X-ELS-APIKey" not in h2 and u2.startswith("https://cdn.other.example/")
    assert out.rec["url"] == "https://cdn.other.example/x.xml"                 # signed query dropped
    routes = {"api.elsevier.com": Resp(302, {"Location": "http://plain.example/x"}, b"", "https://api.elsevier.com/x")}
    f, net, _, _ = mkfetcher(tmp_path / "b", routes, SECRETS)
    out = f.fetch(dict(doi="10.1016/j.x.1", title=TITLE))
    assert out.text is None and len(net.calls) == 1 and "https -> http" in out.rec["errors"][-1]


def test_redirect_target_in_cooldown_is_not_requested(tmp_path):
    routes = {"oa.example.org": Resp(302, {"Location": "https://blocked.example/p.pdf"}, b"", "x")}
    f, net, led, _ = mkfetcher(tmp_path, routes, SECRETS)
    led.block("blocked.example", "cf")
    f.fetch(dict(doi="10.9/x", title=TITLE, oa_url="https://oa.example.org/p"))
    assert not any("blocked.example" in u for u in net.hosts())


def test_browser_child_env_allowlist(monkeypatch):
    monkeypatch.setenv("ELSEVIER_INSTTOKEN", "tok-should-not-leak")
    monkeypatch.setenv("EXA_API_KEY_SECONDARY", "k2")
    monkeypatch.setenv("MY_CUSTOM_CREDENTIAL", "c")
    env = browser_ego.child_env()
    assert "PATH" in env and not ({"ELSEVIER_INSTTOKEN", "EXA_API_KEY_SECONDARY", "MY_CUSTOM_CREDENTIAL"} & set(env))
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)

        class P:
            stdout = 'TBSGEN_RESULT {"status": "ok"}'
            stderr = ""
        return P()
    monkeypatch.setattr(browser_ego.subprocess, "run", fake_run)
    monkeypatch.setattr(browser_ego, "binary", lambda cfg: "/bin/true")
    assert browser_ego.finish({})["status"] == "ok"
    assert "ELSEVIER_INSTTOKEN" not in seen["env"]


def test_errors_and_ledger_notes_are_redacted(tmp_path):
    run, _ = fake_browser({"status": "html", "error": "landed on https://pub.example/x?token=SECRETVALUE123 for "
                                                      "k-els-test"})
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    out = f.fetch(dict(doi="10.1021/acs.1", title=TITLE), browser=True)
    blob = json.dumps(out.rec)
    assert "SECRETVALUE123" not in blob and "k-els-test" not in blob
    notes = " ".join(r[0] for r in sqlite3.connect(tmp_path / "dl.sqlite").execute("select note from attempts"))
    assert "SECRETVALUE123" not in notes and "k-els-test" not in notes
    assert fulltext.safe_url("https://user:pw@h.example/p#frag") == "https://h.example/p"


def test_cooldown_set_while_waiting_cancels_reservation(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"browser": {"gap_s": 60, "jitter_s": 0}}, clock=clock)
    r = led.acquire("browser", "chrome:10.1021", "a")
    led.record("browser", "chrome:10.1021", "a", "ok", 200, 1, reservation=r)

    def sleep_and_get_blocked(s):
        clock.t += s
        led.block("chrome:10.1021", "challenge seen by another worker")
    led.sleep = sleep_and_get_blocked
    with pytest.raises(LimitReached, match="cooldown"):
        led.acquire("browser", "chrome:10.1021", "b")
    rows = sqlite3.connect(tmp_path / "l.sqlite").execute("select status, note from attempts where doi='b'").fetchall()
    assert rows[0][0] == "fail" and rows[0][1].startswith("cancelled")


def test_caps_count_on_the_day_of_the_slot(tmp_path):
    import datetime as dt
    t = dt.datetime(2026, 11, 7, 23, 59, 30).timestamp()
    clock = FakeClock(t)
    led = DownloadLedger(tmp_path / "l.sqlite", {"browser": {"daily_ok": 1, "gap_s": 60, "jitter_s": 0}},
                         clock=clock, sleep=clock.sleep)
    led.acquire("browser", "chrome:10.1021", "a")                  # today, still pending
    led.acquire("browser", "chrome:10.1021", "b")                  # slot after midnight: tomorrow's quota
    days = [r[0] for r in sqlite3.connect(tmp_path / "l.sqlite").execute("select day from attempts order by ts")]
    assert days == ["2026-11-07", "2026-11-08"]
    with pytest.raises(LimitReached, match="daily success cap"):
        led.acquire("browser", "chrome:10.1021", "c")              # tomorrow's single slot is taken


def test_pending_reservation_holds_quota_for_long_downloads(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"browser": {"daily_ok": 1, "gap_s": 0, "jitter_s": 0}},
                         clock=clock, sleep=clock.sleep)
    led.acquire("browser", "chrome:10.1021", "a")
    clock.t += 3600 + 60                                            # a slow download, still running
    with pytest.raises(LimitReached):
        led.acquire("browser", "chrome:10.1093", "b")


def test_publisher_aliases_share_cooldown_across_channels(tmp_path):
    run, calls = fake_browser({"status": "downloaded", "_body": FAKE_PDF})
    f, _, led, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    led.block("onlinelibrary.wiley.com", "akamai on oa")                # set by an HTTP channel
    out = f.fetch(dict(doi="10.1111/x.1", title=TITLE), browser=True, channels=["browser"])
    assert calls == [] and any("cooldown" in s for s in out.rec["skipped"])
    led2_run, calls2 = fake_browser({"status": "challenge", "title": "Just a moment..."})
    f2, _, led2, _ = mkfetcher(tmp_path / "b", {}, SECRETS, browser_runner=led2_run, browser={"enabled": True})
    f2.fetch(dict(doi="10.1002/a.1", title=TITLE), browser=True, channels=["browser"])
    assert led2.cooling("chrome:10.1111") and led2.cooling("onlinelibrary.wiley.com")
    assert fulltext.publisher_of("10.1029/x")[0] == fulltext.publisher_of("10.1002/y")[0] == "wiley"


def test_browser_429_and_downloaded_challenge_page_cool_down(tmp_path):
    run, _ = fake_browser({"status": "html", "http": 429})
    f, _, led, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    f.fetch(dict(doi="10.1021/a", title=TITLE), browser=True)
    row = sqlite3.connect(tmp_path / "dl.sqlite").execute("select until from cooldown where host='chrome:10.1021'").fetchone()
    assert row and abs(row[0] - f.ledger.clock() - 6 * 3600) < 5
    run, _ = fake_browser({"status": "downloaded", "_body": CF})
    f, _, led, _ = mkfetcher(tmp_path / "b", {}, SECRETS, browser_runner=run, browser={"enabled": True})
    f.fetch(dict(doi="10.1093/a", title=TITLE), browser=True)
    assert led.cooling("chrome:10.1093")


def test_html_paywall_page_is_not_full_text():
    page = ("<html><head><title>" + TITLE + "</title></head><body><h1>" + TITLE + "</h1><h2>Abstract</h2><p>"
            + BODY + "</p><h2>Introduction</h2><h2>Methods</h2><h2>Results</h2><p>Purchase this article to read "
            "the full text.</p></body></html>").encode()
    assert fulltext.html_fulltext(page) is None
    article = ("<html><head><title>" + TITLE + "</title></head><body><h1>" + TITLE + "</h1><h2>1. Introduction</h2>"
               "<p>" + BODY + "</p><h2>2. Methods</h2><p>x</p><h2>3. Results</h2><p>y</p><h2>References</h2>"
               "</body></html>").encode()
    assert fulltext.html_fulltext(article)


def test_oversize_responses_are_rejected(tmp_path):
    big = Resp(200, {}, b"%PDF-" + b"x" * 100, "https://oa.example.org/a.pdf", truncated=True)
    f, _, _, _ = mkfetcher(tmp_path, {"oa.example.org": big}, SECRETS, max_pdf_mb=0.00001)
    out = f.fetch(dict(doi="10.9/x", title=TITLE, oa_url="https://oa.example.org/a.pdf"))
    assert out.text is None and "too large" in out.rec["errors"][-1]


def test_corrupt_provenance_does_not_break_cache(tmp_path):
    wc = tmp_path / "wc"
    wc.mkdir()
    (wc / "10.9_x.txt").write_text(BODY)
    (wc / "10.9_x.json").write_text('{"channel": "oa", "url": ')
    f, net, _, _ = mkfetcher(tmp_path, {}, SECRETS)
    out = f.fetch(dict(doi="10.9/x", title=TITLE))
    assert out.rec["source"] == "cache" and net.calls == []


def test_host_cap_on_one_oa_source_does_not_stop_others(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    f, net, led, _ = mkfetcher(tmp_path, {"good.example.org": FAKE_PDF}, SECRETS,
                               oa={"europepmc": True, "max_urls": 3}, host_caps={"www.ebi.ac.uk": 0})
    led.meta["daily_calls"] = 0                                      # metadata host refused
    out = f.fetch(dict(doi="10.9/x", title=TITLE, oa_url="https://good.example.org/a.pdf"))
    assert out.rec["source"] == "oa" and any("metadata cap" in s for s in out.rec["skipped"])


def test_egress_check_reads_cloudflare_trace():
    trace = b"fl=12f\nh=pub.example\nip=198.51.100.9\nts=1.0\nloc=CN\n"
    tr = lambda u, h, t, m: Resp(200, {}, trace, u)                  # noqa: E731
    assert fulltext.check_egress({"allow_cidrs": ["198.51.100.0/24"], "urls": ["https://pub.example/cdn-cgi/trace"]},
                                 tr)["ip"] == "198.51.100.9"



# ------------------------------------------------------------------ review round 2 regressions
def test_redirect_hop_is_booked_and_capped(tmp_path):
    routes = {"oa.example.org": Resp(302, {"Location": "https://cdn.example.net/p.pdf"}, b"", "x"),
              "cdn.example.net": FAKE_PDF}
    f, net, led, _ = mkfetcher(tmp_path, routes, SECRETS, host_caps={"cdn.example.net": 0})
    out = f.fetch(dict(doi="10.9/x", title=TITLE, oa_url="https://oa.example.org/p"))
    assert out.text is None and not any("cdn.example.net" in u for u in net.hosts())
    assert "redirect target refused" in out.rec["errors"][-1]
    f, net, led, _ = mkfetcher(tmp_path / "b", routes, SECRETS)
    f.fetch(dict(doi="10.9/x", title=TITLE, oa_url="https://oa.example.org/p"))
    hops = sqlite3.connect(tmp_path / "b" / "dl.sqlite").execute(
        "select host, status from attempts where status='hop'").fetchall()
    assert hops == [("cdn.example.net", "hop")]


def test_publisher_prefixes_share_pacing_and_caps(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"browser": {"gap_s": 60, "jitter_s": 0, "host_daily_attempts": 2}},
                         clock=clock, sleep=clock.sleep)
    wiley = ("chrome:10.1002", "chrome:10.1111", "chrome:10.1029")
    t0 = clock.t
    for i, k in enumerate(wiley[:2]):
        r = led.acquire("browser", k, f"d{i}", group=tuple(x for x in wiley if x != k))
        led.record("browser", k, f"d{i}", "ok", 200, 1, reservation=r)
    assert clock.t - t0 >= 60                                       # second prefix waited for the first
    with pytest.raises(LimitReached, match="daily attempt cap 2"):
        led.acquire("browser", "chrome:10.1029", "d3", group=wiley[:2])


def test_late_wakeup_is_repaced(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"oa": {"gap_s": 10, "jitter_s": 0}}, clock=clock, sleep=lambda s: None)
    r1 = led.acquire("oa", "h.example", "a")        # sends now
    clock.t += 30                                  # r1 still running; a second booking was made long ago...
    sent = []

    def sleep(s):
        clock.t += s
        sent.append(clock.t)
    led.sleep = sleep
    led.record("oa", "h.example", "a", "ok", 200, 1, reservation=r1)
    db = sqlite3.connect(tmp_path / "l.sqlite")
    db.execute("update attempts set ts=? where rowid=?", (clock.t, r1))      # r1 actually sent late
    db.commit()
    t_before = clock.t
    led.acquire("oa", "h.example", "b")
    assert clock.t - t_before >= 10


def test_wakeup_after_midnight_counts_on_new_day(tmp_path):
    import datetime as dt
    clock = FakeClock(dt.datetime(2026, 11, 7, 23, 59, 50).timestamp())
    led = DownloadLedger(tmp_path / "l.sqlite", {"browser": {"daily_ok": 1, "gap_s": 0, "jitter_s": 0}},
                         clock=clock, sleep=clock.sleep)
    db = sqlite3.connect(tmp_path / "l.sqlite")
    tomorrow = (dt.datetime(2026, 11, 8, 0, 0, 5)).timestamp()
    db.execute("insert into attempts values(?,?,?,?,?,?,?,?,?)",
               (tomorrow, "2026-11-08", "browser", "chrome:10.1093", "x", "ok", 200, 1, ""))
    db.execute("insert into attempts values(?,?,?,?,?,?,?,?,?)",          # forces a wait before the next send
               (clock.t, "2026-11-07", "browser", "chrome:10.1021", "y", "fail", 404, 0, ""))
    db.commit()
    real_sleep = clock.sleep
    led.sleep = lambda s: real_sleep(s + 30)        # wakes after midnight
    led.limits["browser"]["gap_s"] = 5
    with pytest.raises(LimitReached, match="daily success cap"):
        led.acquire("browser", "chrome:10.1021", "a")
    row = db.execute("select status, note from attempts where doi='a'").fetchone()
    assert row[0] == "fail" and row[1].startswith("cancelled")


def test_cooldown_is_never_shortened(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", clock=clock, sleep=clock.sleep)
    led.block("h.example", "challenge", seconds=24 * 3600)
    led.block("h.example", "429", seconds=6 * 3600)
    clock.t += 7 * 3600
    assert led.cooling("h.example")


def test_tdm_api_and_website_cooldowns_are_independent(tmp_path):
    """Declined review item: API throttling and website bot walls are separate signals (see PUBLISHERS)."""
    run, calls = fake_browser({"status": "downloaded", "_body": FAKE_PDF})
    f, _, led, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    led.block("api.wiley.com", "429 on api")
    f.fetch(dict(doi="10.1002/x", title=TITLE), browser=True, channels=["browser"])
    assert len(calls) == 1


def test_block_reasons_and_egress_errors_are_redacted(tmp_path):
    run, _ = fake_browser({"status": "challenge", "title": "Just a moment k-els-test"})
    f, _, led, _ = mkfetcher(tmp_path, {"ip.example": b"203.0.113.7"}, SECRETS, browser_runner=run,
                             browser={"enabled": True})
    f.fetch(dict(doi="10.1021/a", title=TITLE), browser=True)
    blob = " ".join(str(r) for r in sqlite3.connect(tmp_path / "dl.sqlite").execute("select * from cooldown"))
    blob += " ".join(str(r) for r in sqlite3.connect(tmp_path / "dl.sqlite").execute("select * from blocks"))
    assert "k-els-test" not in blob
    f2, _, _, _ = mkfetcher(tmp_path / "e", {"ip.example": b"203.0.113.7"}, SECRETS, egress_check={
        "enabled": True, "allow_cidrs": ["198.51.100.0/24"], "urls": ["https://ip.example/?key=k-els-test"]})
    out = f2.fetch(dict(doi="10.1016/j.x", title=TITLE))
    assert "k-els-test" not in json.dumps(out.rec)


def test_reactivation_delivers_text_across_runs(tmp_path, monkeypatch):
    from tbsgen import cli
    from tbsgen.pipeline import Pipeline
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    shared_q = str(tmp_path / "shared_q.jsonl")
    Ps = []
    for name in ("A", "B"):
        cfg = mkcfg(tmp_path / name, oa={"enabled": False}, browser={"enabled": True}, queue_path=shared_q,
                    ledger_path=str(tmp_path / "shared.sqlite"))
        P = Pipeline(cfg)
        P.fetcher = Fetcher(cfg, transport=FakeNet({}), secrets={})
        [iid] = P.ingest([dict(doi="10.1021/acs.9", title=TITLE,
                               abstract="We propose an estimation algorithm with simulation benchmark; code on github.")])
        P.run(max_steps=3)
        Ps.append((P, iid))
    assert len(fulltext.read_queue(Path(shared_q))[0]["subscribers"]) == 2
    run, _ = fake_browser({"status": "downloaded", "_body": FAKE_PDF})
    Ps[0][0].fetcher.browser_runner = run

    class A:
        status = finish_browser = dry_run = no_browser = False
        max, channels = 5, ""
    monkeypatch.setattr(sys_mod(), "stderr", open("/dev/null", "w"))
    cli._fetch(Ps[0][0], A())
    for P, iid in Ps:
        it = P.store.item(iid)
        assert it.state["stage"] == "S4" and it.p("fulltext.txt").exists()


def test_reconcile_requeues_parked_items(tmp_path):
    from tbsgen.pipeline import Pipeline
    cfg = mkcfg(tmp_path, oa={"enabled": False}, browser={"enabled": True})
    P = Pipeline(cfg)
    P.fetcher = Fetcher(cfg, transport=FakeNet({}), secrets={})
    [iid] = P.ingest([dict(doi="10.1021/acs.9", title=TITLE,
                           abstract="We propose an estimation algorithm with simulation benchmark; code on github.")])
    P.run(max_steps=3)
    qp = fulltext.queue_path(cfg)
    qp.unlink()                                                     # simulate a crash before the enqueue
    assert fulltext.reconcile(P.store, qp) == 1 and fulltext.read_queue(qp)[0]["subscribers"][0][1] == iid
    assert fulltext.reconcile(P.store, qp) == 0


def test_doi_key_collision_does_not_mix_papers(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    f, _, _, _ = mkfetcher(tmp_path, {"oa.example.org": FAKE_PDF}, SECRETS)
    a = dict(doi="10.1234/a/b", title=TITLE, oa_url="https://oa.example.org/1.pdf")
    b = dict(doi="10.1234/a_b", title=TITLE, oa_url="https://oa.example.org/2.pdf")
    assert f.fetch(a).rec["source"] == "oa"
    out_b = f.fetch(b)
    assert out_b.rec["source"] == "oa"                             # not served a's cache file
    assert f.fetch(a).rec["source"] == "cache" and f.fetch(b).rec["source"] == "cache"
    qp = tmp_path / "q.jsonl"
    fulltext.enqueue(qp, fulltext.queue_entry(a, subscribers=[["r", "1"]]))
    fulltext.enqueue(qp, fulltext.queue_entry(b, subscribers=[["r", "2"]]))
    assert len(fulltext.read_queue(qp)) == 2


def test_enqueue_merges_new_retrieval_fields(tmp_path):
    qp = tmp_path / "q.jsonl"
    fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.1/x", title="t"), subscribers=[["r", "1"]]))
    with fulltext.queue_lock(qp):
        q = fulltext.read_queue(qp)
        q[0]["next_try"] = 9e12
        fulltext.write_queue(qp, q)
    fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.1/x", title="t", oa_url="https://oa/x.pdf"),
                                              subscribers=[["r", "1"]]))
    [e] = fulltext.read_queue(qp)
    assert e["oa_url"] == "https://oa/x.pdf" and e["next_try"] == 0


def test_metadata_cooldown_keeps_oa_links(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "pdf_to_text", lambda b: TITLE + "\n" + BODY)
    f, _, led, _ = mkfetcher(tmp_path, {"good.example.org": FAKE_PDF}, SECRETS, oa={"europepmc": True})
    led.block("www.ebi.ac.uk", "429")
    out = f.fetch(dict(doi="10.9/x", title=TITLE, oa_url="https://good.example.org/a.pdf"))
    assert out.rec["source"] == "oa"


def test_html_references_do_not_count(tmp_path):
    refs = "".join(f"<li>Ref {i}: {TITLE} et al. Journal {i}</li>" for i in range(400))
    page = ("<html><head><title>" + TITLE + "</title></head><body><h1>" + TITLE + "</h1><p>Short abstract.</p>"
            "<h2>Introduction</h2><h2>Methods</h2><h2>Results</h2><h2>References</h2><ol>" + refs + "</ol></body></html>")
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS)
    txt, why, _ = f._validate(TITLE, "any", page.encode())
    assert txt is None and "short text" in why


def test_concurrent_drains_do_not_overlap(tmp_path):
    import fcntl
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS)
    qp = fulltext.queue_path(f.cfg)
    qp.parent.mkdir(parents=True, exist_ok=True)
    with open(qp.with_name(qp.name + ".drain.lock"), "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert fulltext.drain_queue(f, log=lambda s: None).get("busy")



# ------------------------------------------------------------------ review round 3 regressions
def test_hop_rechecks_cooldown_and_gap_after_waiting(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"oa": {"gap_s": 10, "jitter_s": 0}}, clock=clock)
    r = led.acquire("oa", "cdn.example", "a")
    led.record("oa", "cdn.example", "a", "ok", 200, 1, reservation=r)

    def sleep_then_block(s):
        clock.t += s
        led.block("cdn.example", "429 seen elsewhere", seconds=6 * 3600)
    led.sleep = sleep_then_block
    with pytest.raises(LimitReached, match="cooldown"):
        led.acquire_hop("oa", "cdn.example", "b")
    assert not sqlite3.connect(tmp_path / "l.sqlite").execute(
        "select count(*) from attempts where status='hop'").fetchone()[0]


def test_hops_do_not_use_channel_attempts(tmp_path):
    clock = FakeClock()
    led = DownloadLedger(tmp_path / "l.sqlite", {"oa": {"gap_s": 0, "jitter_s": 0, "daily_attempts": 1}},
                         clock=clock, sleep=clock.sleep)
    r = led.acquire("oa", "a.example", "x")
    led.acquire_hop("oa", "b.example", "x")
    led.record("oa", "a.example", "x", "ok", 200, 1, reservation=r)
    assert led.usage()["oa"]["attempts"] == 1


def test_cooldown_on_any_publisher_prefix_blocks_the_others(tmp_path):
    run, calls = fake_browser({"status": "downloaded", "_body": FAKE_PDF})
    f, _, led, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    led.block("chrome:10.1002", "challenge (written by the old pipeline)")
    out = f.fetch(dict(doi="10.1111/x.1", title=TITLE), browser=True, channels=["browser"])
    assert calls == [] and any("cooldown" in s for s in out.rec["skipped"])


def test_blocked_flag_from_script_wins_over_later_success(tmp_path):
    run, _ = fake_browser({"status": "downloaded", "_body": FAKE_PDF, "blocked": True})
    f, _, led, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    out = f.fetch(dict(doi="10.1021/a", title=TITLE), browser=True)
    assert out.text is None and led.cooling("chrome:10.1021")


def test_browser_temp_file_removed_on_failure(tmp_path):
    def run(job):
        p = Path(job["out_dir"]) / "leftover.bin"
        p.write_bytes(CF)
        return {"status": "challenge", "title": "Just a moment...", "out_file": str(p)}
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True})
    f.fetch(dict(doi="10.1021/a", title=TITLE), browser=True)
    assert not list((tmp_path / "wc" / "_browser_tmp").iterdir())


def test_pdftotext_gets_no_credentials(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)

        class P:
            stdout = b"text"
        return P()
    monkeypatch.setenv("ELSEVIER_API_KEY", "should-not-leak")
    monkeypatch.setattr(fulltext.subprocess, "run", fake_run)
    fulltext.pdf_to_text(b"%PDF-1.4")
    assert "ELSEVIER_API_KEY" not in seen["env"] and "PATH" in seen["env"]


def test_requeued_entry_stays_due_after_older_failed_attempt(tmp_path):
    hold = {}

    def run_and_update(job):
        fulltext.enqueue(hold["qp"], fulltext.queue_entry(dict(doi="10.1021/a", title=TITLE,
                                                               oa_url="https://oa.example/new.pdf"),
                                                          subscribers=[["r", "1"]]))
        return {"status": "html", "error": "no PDF link"}
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run_and_update, browser={"enabled": True},
                           oa={"enabled": False})
    hold["qp"] = qp = fulltext.queue_path(f.cfg)
    fulltext.enqueue(qp, fulltext.queue_entry(dict(doi="10.1021/a", title=TITLE), subscribers=[["r", "1"]]))
    fulltext.drain_queue(f, log=lambda s: None, now=1000.0)
    [e] = fulltext.read_queue(qp)
    assert e["next_try"] == 0 and e["oa_url"] == "https://oa.example/new.pdf"


def test_inline_browser_respects_taskspace_lock(tmp_path):
    import fcntl
    run, calls = fake_browser({"status": "downloaded", "_body": FAKE_PDF})
    f, _, _, _ = mkfetcher(tmp_path, {}, SECRETS, browser_runner=run, browser={"enabled": True, "inline": True,
                                                                               "space": f"t-{tmp_path.name}"})
    with open(f.space_lock_path(), "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        out = f.fetch(dict(doi="10.1021/a", title=TITLE))
    assert calls == [] and any("in use" in s for s in out.rec["skipped"])
