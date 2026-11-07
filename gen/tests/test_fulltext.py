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
    page_with_recaptcha_widget = b"<html><head><title>Paper title</title></head><body>g-recaptcha</body></html>"
    assert not challenge_page(page_with_recaptcha_widget)


# ------------------------------------------------------------------ ledger: caps, pacing, cooldown, breaker
def test_cloudflare_cools_host_and_blocks_later_requests(tmp_path):
    f, net, led, clock = mkfetcher(tmp_path, {"pub.example.org": Resp(403, {}, CF, "https://pub.example.org/x")},
                                   SECRETS, inst={"enabled": True, "routes": [
                                       {"prefix": "10.5555/", "url": "https://pub.example.org/{suffix}.pdf"}]})
    f.fetch(dict(doi="10.5555/a1", title=TITLE))
    assert led.cooling("pub.example.org")
    n = len(net.calls)
    out = f.fetch(dict(doi="10.5555/a2", title=TITLE))
    assert len(net.calls) == n and any("cooling" in s for s in out.rec["skipped"])
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
        r = results(job) if callable(results) else results
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
    assert bad.text is None and "short text" in bad.rec["errors"][-1]
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
    assert fulltext.enqueue(qp, dict(doi="10.1021/acs.1", title=TITLE, item_ids=["i1"]))
    assert not fulltext.enqueue(qp, dict(doi="10.1021/ACS.1", title=TITLE, item_ids=["i2"]))
    assert fulltext.enqueue(qp, dict(doi="10.1093/x.2", title=TITLE, item_ids=["i3"]))
    q = fulltext.read_queue(qp)
    assert len(q) == 2 and q[0]["item_ids"] == ["i1", "i2"]
    done = []
    st = fulltext.drain_queue(f, max_papers=10, on_success=lambda e, o: done.append(e["doi"]), log=lambda s: None)
    assert st["ok"] == 1 and st["failed"] == 1 and st["remaining"] == 1 and done == ["10.1021/acs.1"]
    assert {c["slot"] for c in calls} <= {0, 1}
    # second drain: cache hit for nothing left; the failed one is tried again, the fetched one is gone
    st = fulltext.drain_queue(f, max_papers=10, log=lambda s: None)
    assert st["considered"] == 1


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
    assert [e["doi"] for e in q] == ["10.1021/acs.9"] and q[0]["item_ids"] == [iid]
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
    assert it.state["status"] == "active" and it.state["stage"] == "S3"
    from tbsgen import stages_a
    stages_a.s3(P, it)
    it = P.store.item(iid)
    assert it.state["stage"] == "S4" and it.state["history"][-1]["source"] == "cache"


def sys_mod():
    import sys
    return sys


def test_browser_script_renders_input_without_env(tmp_path):
    js = browser_ego.render({"space": "s", "doi": "10.1/x", "out_file": "/abs/x.bin"})
    assert js.startswith("const INPUT = {") and "TBSGEN_RESULT" in js and "process.env" not in js
    assert browser_ego.publisher_key("10.1016/j.x") == "chrome:10.1016"
    assert browser_ego.pdf_url_hint("10.1021/acs.jcim.5b00654")[0].startswith("https://pubs.acs.org/doi/pdf/")
    assert browser_ego.binary({"binary": "/nonexistent/ego-browser"}) is None
