"""`tbsgen smoke` — preflight the configured run's environment without touching a production item.

What it answers, in the order a host migration breaks things (2026-11-26):

  1. `config`    the run root exists, the teacher route has a URL + a price entry, the daily cap is set
  2. `breakers`  no circuit breaker is open in the run's own ledger (a ledger copied from the old host carries
                 its breakers with it: `tbsgen breaker --model M --reset`)
  3. `models`    every configured model route answers a tiny real request through the same endpoint and headers
                 its stage uses: the teacher client (`models.teacher`) and the S4c / S5G agent routes (Claude
                 Code's `ANTHROPIC_BASE_URL` / auth token from `code.runnability_env`, else the teacher URL).
                 Calls are booked under stage `SMOKE` in a smoke-only ledger (`<run>/smoke/ledger.sqlite`) with a
                 hard total cap (default $0.05).  The production ledger is NOT written: a failed smoke probe must
                 never open a breaker on the model a live run is using.
  4. `docker`    the Docker daemon the launcher would use is reachable
  5. `buildx`    the builder the launcher would pick (`taskimage.builder`) can see a local `tbsgen-task:` image:
                 `docker buildx build --output type=cacheonly` of a two-line `FROM <local image>` Dockerfile.  A
                 container-driver builder that cannot see the host image store fails here with `pull access
                 denied` instead of failing the first real build.
  6. `egress`    an HTTP probe from the build host reaches the model endpoint and the install hosts within the
                 configured timeout (Harbor's egress probe timeout, seen after the migration)
  7. `launchers` every configured launcher script exists and passes `bash -n`
  8. `disk`      free space at the build host's remote root (task images are large)
  9. `stage_job` (--stage-job only) ONE real minimal agent job per agent route (S4c, S5G) on a built-in toy task
                 with `--max-budget-usd`, harvesting the stage's required outputs

Everything external goes through `Env`, so the unit tests run the whole command against a fake (no network, no
docker, no ssh).  `--dry-run` performs no external action and reports what each check *would* do.

  tbsgen smoke --run runs/R [--config C] [--json-out report.json] [--stage-job] [--budget-usd 0.05]
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import shlex
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import codeagent, taskimage
from .budget import BudgetStop, Ledger
from .models import ModelError, TeacherClient
from .state import read_json, write_json

STAGE = "SMOKE"
DEFAULT_CAP_USD = 0.05
DEFAULT_DISK_MIN_GB = 10.0
DEFAULT_EGRESS_TIMEOUT_S = 30.0
DEFAULT_STAGE_JOB_USD = 0.50
DEFAULT_STAGE_JOB_TIMEOUT_S = 1800.0
CHECK_NAMES = ("config", "breakers", "models", "docker", "buildx", "egress", "launchers", "disk", "stage_jobs")
TINY = "Reply with exactly one word: pong"


def settings(cfg: dict) -> dict:
    return cfg.get("smoke") or {}


def cap_usd(cfg: dict, override: float | None = None) -> float:
    v = override if override is not None else settings(cfg).get("max_usd", DEFAULT_CAP_USD)
    return float(v)


def smoke_dir(cfg: dict) -> Path:
    """Scratch directory of the smoke command: `<run>/smoke`, never a production item directory."""
    root = Path(str((cfg.get("run") or {}).get("root") or "."))
    d = root / "smoke"
    if "items" in d.parts:
        raise ValueError(f"smoke must not run inside the items tree: {d}")
    return d


def ledger_cfg(cfg: dict, cap: float) -> dict:
    """The config a smoke ledger runs with: the same routes and prices, budget caps that bound the whole smoke run
    (stage SMOKE + day total) and a per-call reservation that fits under the cap."""
    c = copy.deepcopy(cfg)
    b = c.setdefault("budget", {})
    n = max(1, len(routes(cfg)))
    b["daily_usd_total"] = cap
    b["window_caps"] = []
    b["daily_usd_per_stage"] = {STAGE: cap}
    b["daily_usd_per_model"] = {}
    b["run_usd_total"] = None
    b["item_usd_cap"] = None
    b["max_calls_per_item"] = n + 1
    b["est_usd_per_call"] = min(float(b.get("est_usd_per_call") or 0.0) or cap, cap / n)
    return c


# ----------------------------------------------------------------------------------------------- routes
def _base_url(url: str) -> str:
    u = str(url).rstrip("/")
    return u.rsplit("/v1/messages", 1)[0] if u.endswith("/v1/messages") else u


def _agent_endpoint(cfg: dict, kind: str) -> dict:
    """Endpoint + headers of an S4c / S5G agent job, resolved the way `codeagent.agent_job_config` resolves them:
    code.runnability_env overrides the launcher's own env for S4c when it has its own model."""
    cc = cfg.get("code") or {}
    own = kind == "S4c" and bool(cc.get("runnability_model"))
    over = {str(k): str(v) for k, v in (cc.get("runnability_env") or {}).items()} if own else {}
    base, source = over.get("ANTHROPIC_BASE_URL"), "code.runnability_env"
    if not base:
        base = _base_url(cfg["models"]["teacher"].get("url") or "")
        source = "models.teacher.url"
    token = over.get("ANTHROPIC_AUTH_TOKEN") or over.get("ANTHROPIC_API_KEY")
    token_src = "code.runnability_env" if token else "models.teacher.key_env"
    if not token:
        key_env = cfg["models"]["teacher"].get("key_env")
        token = os.environ.get(key_env) if key_env else None
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if token:
        headers["x-api-key"] = token
        headers["authorization"] = "Bearer " + token
    return dict(base=base or None, url=(base.rstrip("/") + "/v1/messages") if base else None, source=source,
                token_source=token_src, has_token=bool(token), headers=headers)


def routes(cfg: dict) -> list[dict]:
    """Every model route a stage would use in this run."""
    out: list[dict] = [dict(name="teacher", kind="teacher", alias="teacher", stages="S5/S6A/S6C/S8/S9R",
                model=cfg["models"]["teacher"].get("model"))]
    pre = cfg["models"].get("prereview") or {}
    if pre.get("url"):
        out.append(dict(name="prereview", kind="teacher", alias="prereview", stages="pre-review",
                        model=pre.get("model") or cfg["models"]["teacher"].get("model")))
    cc = cfg.get("code") or {}
    if (cc.get("grounded") or cc.get("agent_launcher") or cc.get("host")) and codeagent.env_ok(cfg, agent=True) is None:
        for kind in ("S4c", "S5G"):
            ep = _agent_endpoint(cfg, kind)
            out.append(dict(name=f"{kind.lower()}-agent", kind="agent", stage=kind, stages=kind,
                            model=codeagent.stage_model(cfg, kind), endpoint=ep))
    return out


# ----------------------------------------------------------------------------------------------- env
class RealEnv:
    """Every external effect of the command.  The tests replace this whole object."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def ssh(self, cmd: str, timeout: float = 60) -> dict:
        return codeagent.ssh_run(self.cfg, cmd, timeout=timeout)

    def docker(self, argv: list[str], timeout: float = 120) -> dict:
        return taskimage._exec(self.cfg, argv, timeout=int(timeout), check=False)

    def shell(self, argv: list[str], timeout: float = 60) -> dict:
        try:
            p = subprocess.run(argv, capture_output=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as e:
            return dict(rc=-1, out="", err=f"{type(e).__name__}: {e}")
        return dict(rc=p.returncode, out=p.stdout.decode("utf8", "replace"), err=p.stderr.decode("utf8", "replace"))

    def context(self, cfg: dict, src: Path):
        """The build context the builder will read (rsynced to the build host when it is remote)."""
        return taskimage._context(cfg, src)

    def client(self, ledger, cfg: dict, alias: str):
        return TeacherClient(ledger, cfg, alias)

    def post(self, url: str, headers: dict, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return dict(status=r.status, body=json.loads(r.read().decode("utf8", "replace") or "{}"))
        except urllib.error.HTTPError as e:
            return dict(status=e.code, body=_json_or_text(e.read()))
        except (urllib.error.URLError, OSError, ValueError) as e:
            return dict(status=0, body={}, error=f"{type(e).__name__}: {e}")

    def launch(self, root: Path, launcher: str, job_json: Path, log: Path, timeout: float) -> dict:
        cmd = (f"cd {shlex.quote(str(root))} && env -u PYTHONPATH -u VIRTUAL_ENV "
               f"./{shlex.quote(launcher)} run {shlex.quote(str(job_json))} > {shlex.quote(str(log))} 2>&1")
        try:
            p = subprocess.run(["bash", "-lc", cmd], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return dict(rc=-9, out="", err=f"launcher did not finish within {timeout:.0f}s")
        except OSError as e:
            return dict(rc=-1, out="", err=f"{type(e).__name__}: {e}")
        return dict(rc=p.returncode, out="", err=p.stderr.decode("utf8", "replace")[-2000:])


def _json_or_text(raw: bytes) -> dict:
    try:
        v = json.loads(raw.decode("utf8", "replace") or "{}")
        return v if isinstance(v, dict) else {"raw": raw.decode("utf8", "replace")[:500]}
    except ValueError:
        return {"raw": raw.decode("utf8", "replace")[:500]}


# ----------------------------------------------------------------------------------------------- checks
def _check(name: str, ok=None, detail=None, skipped: bool = False, blocking: bool = True) -> dict:
    return dict(name=name, ok=ok, detail=detail, skipped=skipped, blocking=blocking and not skipped)


def check_config(cfg: dict) -> dict:
    root = Path(str((cfg.get("run") or {}).get("root") or ""))
    t = cfg["models"]["teacher"]
    problems = []
    if not root or not root.is_dir():
        problems.append(f"run root {root or '(unset)'} does not exist")
    if not str(t.get("url") or "").strip():
        problems.append("models.teacher.url is not set (or TBS_TEACHER_URL)")
    if not str(t.get("model") or "").strip():
        problems.append("models.teacher.model is not set")
    elif not (cfg["models"].get("prices") or {}).get(t["model"]):
        problems.append(f"models.prices has no entry for {t['model']}: smoke cannot price the calls")
    if cfg["budget"].get("daily_usd_total") is None:
        problems.append("budget.daily_usd_total is not set (real model calls refuse to start)")
    cc = cfg.get("code") or {}
    if cc.get("grounded") or cc.get("agent_launcher"):
        if (why := codeagent.env_ok(cfg, agent=True)) is not None:
            problems.append(f"S4c/S5G cannot run: {why}")
    return _check("config", ok=not problems, detail=dict(run=str(root), config=cfg.get("config_path"),
                                                         teacher=t.get("model"), problems=problems))


def check_breakers(cfg: dict) -> dict:
    """Read-only look at the run's production ledger: a breaker copied with the ledger from the old host stops every
    stage of that model.  Read-only on purpose: opening a breaker here would be a production side effect."""
    path = Path(str((cfg.get("run") or {}).get("root") or ".")) / "ledger.sqlite"
    if not path.is_file():
        return _check("breakers", ok=True, detail=dict(ledger=str(path), open=[]), skipped=True)
    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT model, reason, ts FROM breaker WHERE open=1").fetchall()
    except sqlite3.Error as e:
        return _check("breakers", ok=False, detail=dict(ledger=str(path), error=f"{type(e).__name__}: {e}"))
    now = time.time()
    opened = [dict(model=m, reason=(r or "")[:200], age_h=round((now - float(ts or 0)) / 3600, 1)) for m, r, ts in rows]
    hint = "reset with: tbsgen breaker --run <run> --model M --reset" if opened else None
    return _check("breakers", ok=not opened, detail=dict(ledger=str(path), open=opened, hint=hint))


def _host_allowed(cfg: dict, base: str) -> bool | None:
    hosts = [str(h) for h in (cfg.get("code") or {}).get("agent_allowed_hosts") or []]
    if not hosts or not base:
        return None
    try:
        host = urllib.parse.urlparse(base).hostname
    except ValueError:
        return None
    return bool(host) and any(host == h.split(":")[0] or host.endswith("." + h.split(":")[0].lstrip(".")) for h in hosts)


def call_route(cfg: dict, env, ledger: Ledger, route: dict, est_usd: float, timeout: float = 120) -> dict:
    """One tiny real request through the route's own endpoint and headers, booked in the smoke ledger."""
    item = f"smoke:{route['name']}"
    t0 = time.time()
    if route["kind"] == "teacher":
        client = env.client(ledger, cfg, route["alias"])
        try:
            r = client.chat("You are a preflight check.", TINY, stage=STAGE, item=item, max_tokens=24)
        except (ModelError, BudgetStop) as e:
            cls = getattr(e, "cls", type(e).__name__)
            return dict(route=route["name"], model=route["model"], ok=False, error=f"{cls}: {e}"[:300],
                        latency_s=round(time.time() - t0, 2))
        got = str(r.model or "")
        ok = got.startswith(str(route["model"]).split(":")[0][:16])
        return dict(route=route["name"], model=route["model"], ok=ok, model_returned=got,
                    text=str(getattr(r, "text", "") or "").strip()[:20], usage=r.usage, usd=round(r.usd, 6),
                    latency_s=round(time.time() - t0, 2),
                    **({} if ok else {"error": f"route answered with model {got!r}, expected {route['model']!r}"}))
    ep = route["endpoint"]
    body = {"model": route["model"], "max_tokens": 24,
            "messages": [{"role": "user", "content": "Say: pong"}]}
    out = dict(route=route["name"], model=route["model"], endpoint=ep.get("url"), endpoint_source=ep.get("source"))
    if not ep.get("url") or not ep.get("has_token"):
        return dict(out, ok=None, skipped=True,
                    error="no endpoint/token in config (the launcher's process env supplies them)")
    if (allowed := _host_allowed(cfg, ep.get("base"))) is False:
        return dict(out, ok=False, error="endpoint host is not in code.agent_allowed_hosts: the agent phase cannot "
                                         "reach its model")
    cid = ledger.reserve(route["model"], STAGE, item, True, "", est_usd, send=True)
    r = env.post(ep["url"], ep["headers"], body, timeout)
    status = int(r.get("status") or 0)
    if status != 200:
        cls = "auth" if status in (401, 403) else ("quota" if status == 429 else "http")
        ledger.finish(cid, error_class=cls, error=f"http {status} {json.dumps(r.get('body') or r.get('error'))[:200]}",
                      latency=time.time() - t0)
        return dict(out, ok=False, status=status, error=f"http {status}",
                    body=(r.get("body") or r.get("error")))
    got = (r.get("body") or {}).get("model")
    usage = (r.get("body") or {}).get("usage") or {}
    usd = ledger.finish(cid, usage=usage, latency=time.time() - t0, resp_model=got)
    text = "".join(str(c.get("text") or "") for c in (r.get("body") or {}).get("content") or []
                   if isinstance(c, dict)).strip()[:20]
    ok = bool(got) and str(got).startswith(str(route["model"]).split(":")[0][:16])
    return dict(out, ok=ok, model_returned=got, text=text, usage=usage, usd=round(usd, 6),
                latency_s=round(time.time() - t0, 2),
                **({} if ok else {"error": f"route answered with model {got!r}, expected {route['model']!r}"}))


def check_models(cfg: dict, env, ledger: "Ledger | None", cap: float, dry_run: bool = False,
                 timeout: float = 120) -> dict:
    rs = routes(cfg)
    if ledger is None and not dry_run:
        raise ValueError("the models check needs a smoke ledger")
    est = cap / max(1, len(rs))
    results, spent, stopped = [], 0.0, None
    if dry_run:
        return _check("models", ok=None, skipped=True,
                      detail=dict(dry_run=True, cap_usd=cap,
                                  routes=[dict(route=r["name"], model=r["model"], skipped=True) for r in rs]))
    assert ledger is not None
    for r in rs:
        if spent + est > cap + 1e-12:
            stopped = f"cap ${cap:.4f} reached before {r['name']}"
            break
        res = call_route(cfg, env, ledger, r, est, timeout)
        results.append(res)
        priced = bool((cfg["models"].get("prices") or {}).get(res["model"]))
        spent += float(res.get("usd") or 0.0) if priced else est
        # an unpriced model books $0 in the ledger: charge the reservation here so the cap still bounds the run
    failed = [r for r in results if r.get("ok") is False]
    unconfigured = [r["route"] for r in results if r.get("skipped") and not r.get("dry_run")]
    ok = not failed and not stopped
    return _check("models", ok=ok, detail=dict(routes=results, usd=round(spent, 6), cap_usd=cap, stopped=stopped,
                                               unconfigured=unconfigured))


def check_docker(cfg: dict, env, dry_run: bool = False) -> dict:
    if dry_run:
        return _check("docker", ok=None, skipped=True, detail=dict(dry_run=True))
    r = env.docker(["docker", "version", "--format", "{{.Server.Version}}"], timeout=30)
    ok = r.get("rc") == 0 and bool(str(r.get("out") or "").strip())
    return _check("docker", ok=ok, detail=dict(host=(cfg.get("code") or {}).get("host") or "local",
                                               server=str(r.get("out") or "").strip()[:60],
                                               error=None if ok else (r.get("err") or r.get("out") or "")[-600:]))


# LABEL, not RUN: the probe is about the builder seeing the base image; a RUN step also needs a build network, which
# the host daemon's default builder does not have on h20x ("network bridge not found") and real task images never use
SMOKE_DOCKERFILE = "FROM {image}\nLABEL tbsgen.smoke=1\n"
# Builds FROM a daemon-local tbsgen-task: image (teacher / review / code-agent tasks) go through the host daemon's
# `default` builder: the launcher (h20-harbor.sh) switches to it because a docker-container builder such as
# taskimage.builder cannot see the daemon image store.  [smoke] buildx_builder overrides.
SMOKE_BUILDX_BUILDER = "default"


def check_buildx(cfg: dict, env, cap_timeout: float = 600, base_image: str | None = None,
                 dry_run: bool = False) -> dict:
    """The builder the launcher picks must see the local `tbsgen-task:` image store (a container-driver builder
    does not: `pull access denied` on the first real build)."""
    if dry_run:
        return _check("buildx", ok=None, skipped=True, detail=dict(dry_run=True, builder=cfg["taskimage"].get("builder")))
    listing = env.docker(["docker", "image", "ls", "tbsgen-task", "--format", "{{.Repository}}:{{.Tag}}"], timeout=60)
    if listing.get("rc"):
        return _check("buildx", ok=False, detail=dict(error=(listing.get("err") or listing.get("out") or "")[-600:]))
    images = sorted({ln.strip() for ln in str(listing.get("out") or "").splitlines() if ln.strip()})
    image = base_image or (images[-1] if images else None)
    detail = dict(builder=cfg["taskimage"].get("builder"), base_image=image, local_images=len(images))
    if not image:
        return _check("buildx", ok=False, detail=dict(detail, error="no local tbsgen-task: image on the build host "
                                                                    "(build one, or pass --base-image)"))
    src = smoke_dir(cfg) / f"buildx-probe-{os.getpid()}"
    src.mkdir(parents=True, exist_ok=True)
    (src / "Dockerfile").write_text(SMOKE_DOCKERFILE.format(image=image))
    builder = str(settings(cfg).get("buildx_builder") or SMOKE_BUILDX_BUILDER)
    detail["builder"] = builder
    bx = taskimage._buildx({**cfg, "taskimage": {**cfg["taskimage"], "builder": builder}})
    argv = bx + ["--progress=plain", "--output", "type=cacheonly", "-f", "Dockerfile", "."]
    with env.context(cfg, src) as ctx:
        if str(ctx) != str(src):
            argv = bx + ["--progress=plain", "--output", "type=cacheonly", "-f", f"{ctx}/Dockerfile", str(ctx)]
        r = env.docker(argv, timeout=cap_timeout)
    err = (r.get("err") or "") + (r.get("out") or "")
    ok = r.get("rc") == 0
    hint = None
    if not ok:
        low = err.lower()
        if "pull access denied" in low or "failed to resolve source metadata" in low:
            hint = ("the builder cannot see the host image store: set taskimage.builder to the default docker "
                    "driver (or load the image into the builder)")
    return _check("buildx", ok=ok, detail=dict(detail, hint=hint, error=None if ok else err[-1200:]))


def egress_urls(cfg: dict) -> list[str]:
    s = (settings(cfg).get("egress") or {})
    urls = [str(u) for u in (s.get("urls") or [])]
    if not urls:
        t = str(cfg["models"]["teacher"].get("url") or "")
        if t:
            urls.append(t)
        hosts = [str(h) for h in ((cfg.get("code") or {}).get("install_hosts") or [])][:3]
        urls += [f"https://{h}" for h in hosts]
    return urls


def check_egress(cfg: dict, env, timeout_s: float | None = None, dry_run: bool = False) -> dict:
    """HTTP reachability from the build host, within the configured timeout (Harbor's egress probe)."""
    s = (settings(cfg).get("egress") or {})
    tmo = float(timeout_s if timeout_s is not None else s.get("timeout_s", DEFAULT_EGRESS_TIMEOUT_S))
    urls = egress_urls(cfg)
    if dry_run:
        return _check("egress", ok=None, skipped=True, detail=dict(dry_run=True, urls=urls, timeout_s=tmo))
    if not urls:
        return _check("egress", ok=None, skipped=True, detail=dict(error="no probe URL configured"))
    probes = []
    for u in urls:
        cmd = (f"curl -sS -o /dev/null -w '%{{http_code}} %{{time_total}}' --max-time {tmo:.0f} {shlex.quote(u)}")
        r = env.ssh(cmd, timeout=tmo + 20)
        parts = str(r.get("out") or "").strip().split()
        code = int(parts[0]) if parts and parts[0].isdigit() else 0
        probes.append(dict(url=u, status=code, seconds=float(parts[1]) if len(parts) > 1 else None,
                           ok=r.get("rc") == 0 and code > 0,
                           error=None if r.get("rc") == 0 else ((r.get("err") or r.get("out") or "")[-300:])))
    bad = [p for p in probes if not p["ok"]]
    return _check("egress", ok=not bad, detail=dict(timeout_s=tmo, probes=probes,
                                                    host=(cfg.get("code") or {}).get("host") or "local"))


def launchers(cfg: dict) -> list[dict]:
    c = cfg["calib"]
    cc = cfg.get("code") or {}
    out = []
    # the same roots the launch paths use: the teacher (and review) launcher lives in calib.teacher_harbor_root when
    # that is set (calib.role), the code agent in code.agent_harbor_root, else the teacher root
    from .calib import role
    teacher_root = role(c, "teacher", "harbor_root")
    for label, root, script in (("calib.teacher", teacher_root, c.get("teacher_launcher")),
                                ("calib.student", c.get("harbor_root"), c.get("student_launcher")),
                                ("code.agent", cc.get("agent_harbor_root") or c.get("teacher_harbor_root"),
                                 cc.get("agent_launcher") or c.get("teacher_launcher"))):
        if script and root:
            out.append(dict(label=label, root=str(root), script=str(script), path=str(Path(str(root)) / str(script))))
    return out


def check_launchers(cfg: dict, env, dry_run: bool = False) -> dict:
    items = launchers(cfg)
    if not items:
        return _check("launchers", ok=None, skipped=True,
                      detail=dict(error="no launcher configured (calib.harbor_root / code.agent_harbor_root)"))
    results = []
    for it in items:
        p = Path(it["path"])
        res = dict(it)
        if not p.is_file():
            res.update(ok=False, error=f"missing: {p}")
        elif dry_run:
            res.update(ok=None, dry_run=True)
        else:
            r = env.shell(["bash", "-n", str(p)], timeout=30)
            res.update(ok=r.get("rc") == 0, error=None if r.get("rc") == 0 else (r.get("err") or "")[-400:])
        results.append(res)
    return _check("launchers", ok=all(x["ok"] is not False for x in results), detail=dict(items=results,
                                                                                         allow_submit=cfg["calib"].get("allow_submit")))


def check_disk(cfg: dict, env, min_free_gb: float | None = None, dry_run: bool = False) -> dict:
    s = settings(cfg).get("disk") or {}
    need = float(min_free_gb if min_free_gb is not None else s.get("min_free_gb", DEFAULT_DISK_MIN_GB))
    cc = cfg.get("code") or {}
    path = str(cc.get("remote_root") or (cfg.get("sandbox") or {}).get("remote_dir")
               or (cfg.get("run") or {}).get("root") or ".")
    if dry_run:
        return _check("disk", ok=None, skipped=True, detail=dict(dry_run=True, path=path, min_free_gb=need))
    r = env.ssh(f"df -Pk {shlex.quote(path)} | tail -1", timeout=30)
    cols = str(r.get("out") or "").split()
    if r.get("rc") != 0 or len(cols) < 4 or not cols[3].isdigit():
        return _check("disk", ok=False, detail=dict(path=path, error=(r.get("err") or r.get("out") or "")[-300:]))
    free_gb, total_gb = int(cols[3]) / 1024 / 1024, int(cols[1]) / 1024 / 1024
    return _check("disk", ok=free_gb >= need, detail=dict(path=path, free_gb=round(free_gb, 1),
                                                          total_gb=round(total_gb, 1), min_free_gb=need))


# ----------------------------------------------------------------------------------------------- stage jobs
STAGE_JOB_INSTRUCTION = """\
# tbsgen smoke check ({kind})

This is an automated preflight of the {kind} model route. Do not install packages and do not use the network.

Write exactly these files under `/app/out` (placeholder content is fine; the verifier only checks that each exists):

{files}

Then stop.
"""

TOY_S4C_RUNNABILITY = json.dumps(dict(runnable=False, reason="other", entry="", notes="smoke toy task"))


def _toy_work(kind: str, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    (work / "paper.txt").write_text("Smoke-check paper. The authors report the mean of a table of numbers.\n")
    (work / "HOWTO.md").write_text("# How to run\n\n`python run.py`\n")
    (work / "runnability.json").write_text(TOY_S4C_RUNNABILITY + "\n")
    (work / "repo").mkdir(exist_ok=True)
    (work / "repo" / "README.md").write_text("authors code placeholder\n")
    return work


def stage_job(cfg: dict, env, kind: str, cap: float, timeout_s: float, dry_run: bool = False) -> dict:
    """ONE real minimal agent job on the built-in toy task, in `<run>/smoke` (never the items directory)."""
    out = dict(kind=kind, model=codeagent.stage_model(cfg, kind), route=(cfg.get("code") or {}).get("agent_launcher")
               or cfg["calib"].get("teacher_launcher"))
    if (why := codeagent.env_ok(cfg, agent=True)) is not None:
        return dict(out, ok=None, skipped=True, error=why)
    if not launchers(cfg):
        return dict(out, ok=None, skipped=True, error="no agent launcher configured")
    if dry_run:
        return dict(out, ok=None, skipped=True, dry_run=True)
    base = smoke_dir(cfg) / "stage-job" / kind.lower()
    task = codeagent.build_agent_task(cfg, kind, f"smoke-{kind.lower()}", _toy_work(kind, base / "work"),
                                      base / "task", STAGE_JOB_INSTRUCTION.format(
                                          kind=kind, files="\n".join(f"- `{f}`" for f in
                                                                     codeagent.agent_output_files(kind))),
                                      codeagent.agent_output_files(kind), timeout_s)
    job_name = f"smoke-{kind.lower()}-{time.strftime('%Y%m%d-%H%M%S')}"
    conf = codeagent.agent_job_config(cfg, task, job_name, base / "jobs", cap, kind=kind)
    job_json = base / f"{job_name}.json"
    write_json(job_json, conf)
    log = base / f"{job_name}.launch.log"
    root = (cfg.get("code") or {}).get("agent_harbor_root") or cfg["calib"].get("harbor_root")
    script = Path(str(out["route"] or "")).name
    if cfg["calib"].get("allow_submit") is False or not root or not script:
        return dict(out, ok=None, skipped=True, job=job_name, job_json=str(job_json),
                    error="calib.allow_submit=false or no agent launcher: run the printed command yourself")
    r = env.launch(Path(str(root)), script, job_json, log, timeout_s)
    jd = Path(conf["jobs_dir"]) / job_name
    rec = dict(job_name=job_name, job_dir=str(jd), task=str(task), config=str(job_json), ledger_stage=kind)
    complete = False
    try:
        complete = bool(codeagent.complete_remote_agent(cfg, rec, final=True))
    except (OSError, ValueError) as e:
        out["error"] = f"harvest: {type(e).__name__}: {e}"
    harvested = []
    for trial in sorted(x for x in jd.glob("*") if x.is_dir()):
        vend = codeagent.deliverable(dict(trial=trial))
        if vend is not None:
            harvested += sorted(x.relative_to(vend).as_posix() for x in vend.rglob("*") if x.is_file())
    ok = r.get("rc") == 0 and complete
    return dict(out, ok=ok, job=job_name, job_dir=str(jd), log=str(log), launcher_rc=r.get("rc"),
                harvested=sorted(set(harvested)),
                required=codeagent.agent_output_files(kind), complete=complete,
                error=None if ok else ((r.get("err") or "")[-600:] or
                                       ("launcher ok but the required outputs were not harvested" if r.get("rc") == 0
                                        else "launcher failed")))


# ----------------------------------------------------------------------------------------------- run
def run_smoke(cfg: dict, *, env=None, stage_job_flag: bool = False, max_usd: float | None = None,
              ledger_path: str | Path | None = None, skip: str | list[str] | None = None,
              dry_run: bool = False, base_image: str | None = None, timeout: float | None = None,
              stage_job_timeout: float | None = None) -> dict:
    env = env or RealEnv(cfg)
    cap = cap_usd(cfg, max_usd)
    s = settings(cfg)
    tmo = float(timeout if timeout is not None else s.get("timeout_s", 120))
    skipped = {x.strip() for x in ((skip.split(",") if isinstance(skip, str) else skip) or []) if x.strip()}
    unknown = skipped - set(CHECK_NAMES)
    if unknown:
        raise ValueError(f"unknown check(s) to skip: {sorted(unknown)}")
    path = Path(ledger_path) if ledger_path else smoke_dir(cfg) / "ledger.sqlite"
    lcfg = ledger_cfg(cfg, cap)
    ledger = None if dry_run else Ledger(path, lcfg)
    checks, jobs = [], []
    plan = [("config", lambda: check_config(cfg)),
            ("breakers", lambda: check_breakers(cfg)),
            ("models", lambda: check_models(cfg, env, ledger, cap, dry_run, tmo)),
            ("docker", lambda: check_docker(cfg, env, dry_run)),
            ("buildx", lambda: check_buildx(cfg, env, dry_run=dry_run, base_image=base_image)),
            ("egress", lambda: check_egress(cfg, env, dry_run=dry_run)),
            ("launchers", lambda: check_launchers(cfg, env, dry_run)),
            ("disk", lambda: check_disk(cfg, env, dry_run=dry_run))]
    for name, fn in plan:
        if name in skipped:
            checks.append(_check(name, ok=None, skipped=True, detail=dict(skipped_by_operator=True)))
        else:
            checks.append(fn())
    if stage_job_flag:
        st_tmo = float(stage_job_timeout if stage_job_timeout is not None
                       else s.get("stage_job_timeout_s", DEFAULT_STAGE_JOB_TIMEOUT_S))
        job_cap = float(s.get("stage_job_max_usd", DEFAULT_STAGE_JOB_USD))
        for kind in ("S4c", "S5G"):
            if not any(r.get("kind") == "agent" and r.get("stage") == kind for r in routes(cfg)):
                jobs.append(dict(kind=kind, ok=None, skipped=True, error="agent route not configured"))
                continue
            jobs.append(stage_job(cfg, env, kind, job_cap, st_tmo, dry_run))
        failed = [j for j in jobs if j.get("ok") is False]
        checks.append(_check("stage_jobs", ok=not failed, detail=dict(jobs=jobs, max_usd_per_job=job_cap,
                                                                      timeout_s=st_tmo)))
    report = dict(ok=all(c["ok"] is not False for c in checks), dry_run=dry_run, stage=STAGE,
                  run=str((cfg.get("run") or {}).get("root")), config=cfg.get("config_path"),
                  cap_usd=cap, ledger=str(path), checks=checks,
                  usd=round(sum(float((c.get("detail") or {}).get("usd") or 0) for c in checks
                                if c["name"] == "models"), 6))
    if ledger is not None:
        with contextlib.closing(ledger._db()) as db:  # noqa: SLF001 - read-only report of this smoke ledger
            report["ledger_calls"] = [dict(zip(("model", "stage", "item", "usd", "status"), row)) for row in
                                      db.execute("SELECT model, stage, item, usd, status FROM calls "
                                                 "WHERE stage=? ORDER BY id", (STAGE,)).fetchall()]
    if not dry_run:
        smoke_dir(cfg).mkdir(parents=True, exist_ok=True)
        write_json(smoke_dir(cfg) / "last-report.json", report)
    return report


def format_report(rep: dict) -> str:
    width = max(len(c["name"]) for c in rep["checks"])
    lines = [f"tbsgen smoke: {'OK' if rep['ok'] else 'FAILED'}  (stage {STAGE}, cap ${rep['cap_usd']:.2f}, "
             f"spent ${rep['usd']:.4f}){', dry-run' if rep['dry_run'] else ''}"]
    for c in rep["checks"]:
        state = "skip" if c["skipped"] else ("ok" if c["ok"] else ("FAIL" if c["ok"] is False else "-"))
        detail = json.dumps(c["detail"], ensure_ascii=False, default=str)
        lines.append(f"  {c['name']:<{width}}  {state:<4}  {detail[:300]}")
    return "\n".join(lines)
