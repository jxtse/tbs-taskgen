"""Task-image contract and lifecycle tests. All build host and registry operations are simulated."""
import hashlib
import json
import shlex
from pathlib import Path

import pytest

from tbsgen import calib, codeagent, gates, package, prerelease, runner, sandbox, stages_a, taskimage as ti
from tbsgen.config import load_config
from tbsgen.prompts_code import CODE_PROMPT_VERSION
from tbsgen.spec import validate_spec
from tbsgen.state import read_json, write_json
from tbsgen.taskimage_probe import installed
from test_codeagent import _agent_item
from test_unit import SPEC

BASE = "ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"
DF = "FROM " + BASE + "\n"
UV = "ghcr.io/astral-sh/uv:0.11.1@sha256:fc93e9ecd7218e9ec8fba117af89348eef8fd2463c50c13347478769aaedd0ce"
MAMBA = "condaforge/miniforge3:25.3.0-3@sha256:f1f537ff80646e3aaf84e64abf6249cdc3f20889aa0a8d41535e62925c959f64"
IMAGE = dict(base=BASE, packages=["h5py"], notes="HDF5 reader")
INVENTORY = installed("pip", '[{"name":"h5py","version":"3.11.0"}]')


@pytest.mark.parametrize("text", [
    "FROM " + MAMBA + "\nRUN mamba install -y -c conda-forge h5py=3.11.0",
    DF + "RUN apt-get update && apt-get install --no-install-recommends ca-certificates=20240203",
    "FROM " + UV + " AS uv\n" + DF + "COPY --from=uv /uv /uvx /usr/local/bin/\nRUN uv pip install --system h5py==3.11.0",
    DF.replace("ubuntu:24.04", "rocker/r-ver:4.3.0") + "USER nobody\nUSER root",
    DF + "RUN python -m pip install --no-cache-dir \\\n h5py==3.11.0",
    DF + "RUN curl -o data.tgz https://example.org/data && echo '" + "a" * 64 + "  data.tgz' | sha256sum -c -",
])
def test_good_dockerfiles(text):
    assert ti.lint(text) == []


@pytest.mark.parametrize("text", [
    "FROM rocker/r-ver:4.3.0", DF.replace("24.04", "latest"), DF + "FROM ubuntu:24.04",
    DF + "RUN pip install h5py", DF + "RUN uv pip install --system h5py>=3",
    DF + "RUN apt-get update && apt-get install -y hdf5-tools",
    DF + "RUN micromamba install -n base h5py", DF + "RUN wget https://example.org/a",
    DF + "ADD https://example.org/a /a", DF + "COPY data /app/data", DF + "COPY . /app",
    DF + "RUN cat /tests/file", DF + "ENTRYPOINT [\"python\"]", DF + "CMD bash",
    DF + "EXPOSE 80", DF + "VOLUME /app", DF + "USER nobody",
])
def test_bad_dockerfiles(text):
    assert ti.lint(text)


@pytest.mark.parametrize(("name", "command", "valid", "bad"), [
    ("requirements.txt", "pip install -r", "h5py==3.11.0", "h5py"),
    ("environment.yml", "mamba env create -f", "channels:\n  - conda-forge\ndependencies:\n  - python=3.11.1",
     "dependencies:\n  - h5py"),
    ("env.lock", "conda install --file", "@EXPLICIT\nhttps://example.org/x.conda#" + "a"*64, "@EXPLICIT\nunpinned"),
])
def test_context_locks(tmp_path, name, command, valid, bad):
    df = DF + f"COPY {name} /deps/{name}\nRUN {command} /deps/{name}"
    assert ti.lint(df, tmp_path)
    (tmp_path / name).write_text(valid)
    assert not ti.lint(df, tmp_path)
    (tmp_path / name).write_text(bad)
    assert ti.lint(df, tmp_path)


def setup_agent(tmp_path, monkeypatch, env, *, base=BASE, **code):
    spec = dict(SPEC, generator_assumptions=[], method_names=["FastSmoother"], image=dict(IMAGE, base=base), data_formats=["h5"])
    files = dict.fromkeys(codeagent.agent_output_files("S5G"), "fixture")
    files.update({"_spec/spec.json": json.dumps(spec), "_spec/author_code.json": '{"package_names":["FastSmoother"]}',
                  "report.md": "fixture", **{"task/environment/"+k: v for k, v in env.items()}})
    P, it, launcher = _agent_item(tmp_path, monkeypatch, files, stage="S5G", **code)
    P.cfg["taskimage"]["enabled"] = True
    P.run(max_steps=2, until="S5G")
    return P, it.load(), launcher


def test_legacy_fallback(tmp_path, monkeypatch):
    P, it, _ = setup_agent(tmp_path, monkeypatch, {})
    assert it.state["stage"] == "S6A_run" and it.state["image_kind"] == "legacy"
    assert not it.p("task/environment").exists()
    assert ti.image_for(P.cfg, it.dir) == P.cfg["sandbox"]["image"]


@pytest.mark.parametrize("extra", [{"payload.bin": "x"}, {"data/x.csv": "x"}, {"requirements.txt": "x"*1_000_001}])
def test_s5g_rejects_unapproved_context(tmp_path, monkeypatch, extra):
    _, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF, **extra})
    assert it.state["stage"] == "S5G" and it.state["image_retry_used"]
    assert not it.p("task/environment").exists()
    assert read_json(it.p("S6I_image.json"))["failed_gate"] == "lint"


def fake_host(monkeypatch, packages, fail_build=False, imagetools=None):
    calls = []
    ids = {}
    def execute(cfg, argv, timeout=60, *, check=True, full=False):
        calls.append(argv)
        if argv[:4] == ["docker", "buildx", "imagetools", "inspect"] and imagetools:
            assert full
            return dict(rc=0, out=imagetools(argv), err="")
        if argv[:3] == ["docker", "image", "inspect"]:
            if argv[-2] == "{{.Id}}":
                return dict(rc=int(argv[-1] not in ids), out=ids.get(argv[-1], ""), err="No such image")
            return dict(out="linux/amd64 12345")
        if argv[:2] == ["docker", "build"] or argv[:3] == ["docker", "buildx", "build"]:
            ctx = Path(argv[-1])
            assert list((ctx / "data").iterdir()) == []
            assert (ctx / "Dockerfile").read_text().endswith(ti.REQUIRED_TAIL)
            if fail_build:
                raise ValueError('#3 0.1 connection refused\nERROR: process "/bin/sh -c fixture-install" '
                                 'did not complete successfully: exit code: 1')
            ids[argv[-2]] = "sha256:" + "c"*64
        elif argv[:2] == ["docker", "tag"]:
            ids[argv[-1]] = argv[-2]
        elif argv[:2] == ["docker", "run"]:
            assert argv[argv.index("--network") + 1] == "none"
            assert full and "h5py" in argv[-1] and "test.sh" in argv[-1]
            helper = json.loads(shlex.split(argv[-1])[-1])["helper"]
            assert '_open_workspace("/app/output")' in helper and argv[argv.index("bash") - 1] in ids.values()
            return dict(out=json.dumps(dict(python="3.11.1", floor_errors=[], import_errors=[], inventory=packages)))
        else:
            pytest.fail(f"unexpected call {argv}")
        return dict(rc=0, out="", err="")
    monkeypatch.setattr(ti, "_exec", execute)
    monkeypatch.setattr(ti, "_export_base", lambda cfg, item, digest: "sha256:" + "e"*64)
    return calls


def test_s6i_through_packaging(tmp_path, monkeypatch):
    df = DF + "COPY requirements.txt /deps/requirements.txt\nRUN pip install -r /deps/requirements.txt\n"
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": df, "requirements.txt": "h5py==3.11.0"})
    assert it.state["stage"] == "S6I", it.state
    assert it.p("task/environment/requirements.txt").read_text() == "h5py==3.11.0"
    P.cfg["sandbox"]["backend"] = "docker"
    calls = fake_host(monkeypatch, INVENTORY)
    from tbsgen.pipeline import parse_stages
    build = parse_stages("build")
    assert build == {"S6I"}
    with it.step_lock() as held:
        assert held
        P.run(max_steps=1, stages=build)
        assert not calls and it.load().state["stage"] == "S6I"
    P.run(max_steps=1, stages=build)
    st = it.load().state
    assert st["stage"] == "S6A_run", st
    assert all(read_json(it.p("S6I_image.json"))["gates"].values())
    assert ti.image_for(P.cfg, it.dir) == st["image_digest"] and st["inventory"] == INVENTORY
    assert ti.build(P.cfg, it.dir)["image_tag"] == st["image_tag"]
    assert len([c for c in calls if c[:2] == ["docker", "build"]]) == 1
    assert ti.base_catalog(ti.catalog_path(P.cfg))[0]["tasks"] == [it.item_id]
    gen = it.p("gen0")
    for d in ["environment/data", "tests/truth", "tests/hidden"]:
        (gen / d).mkdir(parents=True)
    (gen / "environment/data/test.h5").touch()
    it.p("task/tests/test_outputs.py").touch()
    P.cfg["taskimage"]["enabled"] = False
    pkg = package.package(P.cfg, it.dir, gen, read_json(it.p("_spec/spec.json")), {})
    assert (pkg / "environment/Dockerfile").read_text() == (
        f"FROM {ti.harbor_tag(it.dir, st['image_digest'])}\nCOPY data /app/data\n")
    assert not (pkg / "environment/requirements.txt").exists()
    prov = read_json(pkg / "provenance.json")
    assert all(prov[k] == st[k] for k in ("image_tag", "image_digest", "inventory_hash"))
    assert prov["base"] == BASE
    assert gates.realism({}, pkg / "environment/data", {}, INVENTORY)["data_realism"][1]["readers"] == {".h5": ["h5py"]}


@pytest.mark.parametrize("manifest_type", ["index", "single", "descriptor"])
def test_s6i_pins_base_and_reuses_it(tmp_path, monkeypatch, manifest_type):
    tag, digest = BASE.split("@")
    df = f"# FROM {tag}\nFROM --platform=linux/amd64 {tag} AS deps\nFROM {tag}\nRUN echo {tag}\n"
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": df}, base=tag)
    assert it.state["stage"] == "S6I"
    P.cfg["sandbox"]["backend"] = "docker"
    P.cfg["taskimage"]["builder"] = "test-builder"
    platform = dict(os="linux", architecture="amd64")
    def inspect(argv):
        assert argv[4:6] == ["--builder", "test-builder"]
        if "--format" not in argv:
            assert manifest_type == "single" and argv[-1] == tag
            return f"Name: {tag}\nMediaType: application/vnd.oci.image.manifest.v1+json\nDigest: {digest}\n"
        if argv[-2] == "{{json .Image}}":
            assert argv[-1] == BASE
            return json.dumps(platform)
        assert argv[-2:] == ["{{json .Manifest}}", tag]
        if manifest_type == "index":
            return json.dumps(dict(manifests=[
                dict(digest="sha256:" + "f"*64, platform=dict(os="linux", architecture="arm64")),
                dict(digest=digest, platform=platform)]))
        if manifest_type == "single":
            return json.dumps(dict(schemaVersion=2, config=dict(digest="sha256:" + "a"*64), layers=[]))
        return json.dumps(dict(digest=digest))
    calls = fake_host(monkeypatch, INVENTORY, imagetools=inspect)
    lint = ti.lint
    def verify_pin_before_lint(text, context):
        assert read_json(it.p("state.json"))["base"] == BASE
        assert read_json(it.p("_spec/spec.json"))["image"]["base_resolved_from"] == tag
        assert text == it.p("task/environment/Dockerfile").read_text()
        return lint(text, context)
    monkeypatch.setattr(ti, "lint", verify_pin_before_lint)
    P.run(max_steps=1, until="S6I")
    state = it.load().state
    assert state["stage"] == "S6A_run"
    for record in (state, read_json(it.p("_spec/spec.json"))["image"], read_json(it.p("S6I_image.json"))):
        assert record["base_resolved_from"] == tag and record["base"] == BASE
    expected = f"# FROM {tag}\nFROM --platform=linux/amd64 {BASE} AS deps\nFROM {BASE}\nRUN echo {tag}\n"
    assert it.p("task/environment/Dockerfile").read_text() == expected
    assert ti.build(P.cfg, it.dir)["image_tag"] == state["image_tag"]
    assert sum("{{json .Manifest}}" in cmd for cmd in calls) == 1


@pytest.mark.parametrize("single", [True, False])
def test_resolve_rejects_wrong_platform(monkeypatch, single):
    arm = dict(digest="sha256:" + "a" * 64, platform=dict(os="linux", architecture="arm64"))
    manifest = arm if single else dict(manifests=[arm])
    monkeypatch.setattr(ti, "_exec", lambda *args, **kwargs: dict(out=json.dumps(manifest)))
    with pytest.raises(ValueError, match="linux/amd64"):
        ti._resolve_tag(load_config(), "python:3.11")


def test_s6i_resolution_failure_reaches_reauthor(tmp_path, monkeypatch):
    tag = "example.org/image:missing"
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": f"FROM {tag}\n"}, base=tag)
    P.cfg["taskimage"].update(host="build-host", builder="test-builder", docker_config="$HOME/docker-config")
    def fail(argv, timeout, output_limit=None):
        assert argv[-2] == "build-host" and output_limit is None
        assert 'DOCKER_CONFIG="$HOME"/docker-config' in argv[-1]
        assert "imagetools inspect --builder test-builder" in argv[-1] and tag in argv[-1]
        return dict(rc=1, out="", err="manifest unknown for " + tag)
    monkeypatch.setattr(sandbox, "sh", fail)
    P.run(max_steps=1, until="S6I")
    assert it.load().state["stage"] == "S5G"
    assert it.p("task/environment/Dockerfile").read_text() == f"FROM {tag}\n"
    result = read_json(it.p("S6I_image.json"))
    assert result["failed_gate"] == "resolve_base" and result["gates"]["built"] is None
    P.run(max_steps=1, until="S5G")
    assert "manifest unknown for " + tag in read_json(it.p("S5G/work/previous_attempt.json"))["S6_failure.json"]["finding"]


CACHE_500 = ("failed to copy: httpReadSeeker: failed open: unexpected status from GET request to "
             "http://cache.example:5000/v2/library/sample/manifests/sha256:abc?ns=upstream.example: "
             "500 Internal Server Error")
# The registry client's real output: the transport line, then a generic trailer.
CACHE_500_RAW = "ERROR: " + CACHE_500 + "\nunknown: unknown error\n"


@pytest.mark.parametrize(("gate", "error", "infra"), [
    ("resolve_base", CACHE_500, True),
    ("resolve_base", CACHE_500_RAW, True),
    ("built", CACHE_500_RAW, True),
    ("resolve_base", "manifest unknown: cache.example/sample:missing\nunknown: unknown error", False),
    ("built", CACHE_500, True),
    ("resolve_base", "HTTP/2 503", True),
    ("resolve_base", "unexpected status code 429", True),
    ("resolve_base", "toomanyrequests: retry later", True),
    ("built", 'failed to resolve source metadata: Head "https://cache.example": unexpected EOF', True),
    ("built", "dial tcp 192.0.2.1:443: connect: connection refused", True),
    ("resolve_base", "read: connection reset by peer", True),
    ("resolve_base", "dial tcp: i/o timeout", True),
    ("resolve_base", "TIMEOUT after 120s", True),
    ("resolve_base", "net/http: TLS handshake timeout", True),
    ("resolve_base", "lookup cache.example: no such host", True),
    ("built", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock", True),
    ("built", "failed to connect to the docker API at unix:///var/run/docker.sock: no such file or directory", True),
    ("resolve_base", "ssh: Could not resolve hostname build.example: Name or service not known", True),
    ("built", "ssh: connect to host build.example port 22: Connection timed out", True),
    ("built", "ssh exited 255", True),
    ("resolve_base", "manifest unknown: cache.example/sample:missing", False),
    ("resolve_base", "HTTP 404 Not Found", False),
    ("built", "TIMEOUT after 2400s", False),
    ("built", "unknown package fixture-lib", False),
    ("built", "curl: (7) connection refused", False),
    ("built", "unexpected EOF", True),
    ("resolve_base", "unexpected build-context entry: unexpected EOF", False),
    ("built", CACHE_500 + '\nERROR: process "/bin/sh -c sample" did not complete successfully: exit code: 1', False),
    ("lint", "unexpected EOF", False),
    ("offline_imports", CACHE_500, False),
    ("verifier_floor", CACHE_500, False),
])
def test_image_infra_classifier(gate, error, infra):
    assert ti.infra_failure(dict(failed_gate=gate, error=error)) is infra


def test_image_infra_backoff_and_park(tmp_path, monkeypatch):
    from tbsgen.pipeline import parse_stages
    tag = "sample.example/base:1"
    P, it, launcher = setup_agent(tmp_path, monkeypatch, {"Dockerfile": f"FROM {tag}\n"},
                                  base=tag, authoring_max_jobs=1)
    P.cfg["taskimage"]["infra_retries"] = 4
    it.state["harden_active"] = True  # Infrastructure parking must survive the automatic hardening-revert check.
    it.save()
    costs = P.item_cost(it.item_id)
    calls = []
    def failing_inspect(argv):
        calls.append(argv)
        raise ValueError(CACHE_500)
    fake_host(monkeypatch, INVENTORY, imagetools=failing_inspect)
    clock = 100_000
    monkeypatch.setattr(stages_a.time, "time", lambda: clock)
    build = parse_stages("build")
    for n, delay in enumerate([600, 1800, 3600, None], 1):
        P.run(max_steps=3, stages=build if n % 2 else None)
        st = it.load().state
        assert st["stage"] == "S6I"
        assert st["attempts"]["S6I:infra"] == len(calls) == n
        assert st["attempts"]["S5G:job"] == len(launcher.calls) == 1
        assert P.item_cost(it.item_id) == costs
        assert "image_retry_used" not in st and not it.p("S6_failure.json").exists()
        history = read_json(it.p("S6I_image.json"))["history"]
        assert [(h["gate"], h["error"]) for h in history] == [("resolve_base", CACHE_500)] * n
        if delay:
            assert (st["status"], st["reason"]) == ("waiting", "S6I:infra:resolve_base")
            assert st["history"][-1]["retry_at"] == clock + delay
            clock += delay - 1
            P.run(stages=build)
            assert len(calls) == n
            clock += 1
    assert (st["status"], st["reason"]) == ("parked", "parked:infra")
    clock += 86400
    P.run()
    assert len(calls) == 4 and P.requeue_image(it.item_id)
    P.cfg["sandbox"]["backend"] = "docker"
    manifest = dict(manifests=[dict(digest="sha256:" + "f"*64, platform=dict(os="linux", architecture="amd64"))])
    fake_host(monkeypatch, INVENTORY, imagetools=lambda argv: json.dumps(manifest))
    P.run(stages=build)
    assert it.load().state["stage"] == "S6A_run"
    receipt = read_json(it.p("S6I_image.json"))
    assert receipt["ok"] and receipt["history"] == history


@pytest.mark.parametrize(("reason", "error", "allowed"), [
    ("s5g_jobs_exhausted:image_build_failed:resolve_base", CACHE_500, True),
    ("s5g_jobs_exhausted:image_build_failed:resolve_base", "manifest unknown", False),
    ("s5g_jobs_exhausted:image_build_failed:built", CACHE_500, False),
    ("image_build_failed:resolve_base", CACHE_500, False),
])
def test_requeue_legacy_image_drop(tmp_path, monkeypatch, reason, error, allowed):
    from tbsgen import cli
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF})
    it.state["image_retry_used"] = True
    it.finish("dropped", reason)
    write_json(it.p("S6I_image.json"), dict(ok=False, failed_gate="resolve_base", error=error))
    write_json(it.p("S6_failure.json"), dict(why="image_build_failed:resolve_base", finding=error))
    args = ["requeue-image", "--run", str(P.store.root), "--item", it.item_id]
    before = it.state.copy()
    if allowed:
        assert cli.main(args) == 0
        assert it.load().state["stage"] == "S6I" and it.state["status"] == "active"
        assert "image_retry_used" not in it.state and not it.p("S6_failure.json").exists()
        assert it.state["attempts"] == before["attempts"]
    else:
        with pytest.raises(SystemExit) as refused:
            cli.main(args)
        assert refused.value.code == 2
        assert it.load().state == before and it.p("S6_failure.json").exists()


@pytest.mark.parametrize("version", ["cg-v7", "cg-v10"])
def test_image_spec_version_boundary(version):
    cfg = load_config()
    legacy = dict(SPEC, generator_assumptions=[])
    assert not validate_spec(legacy, cfg, "cg-v6")
    errors = validate_spec(legacy, cfg, version)
    assert any("data_formats" in e for e in errors) and any("image" in e for e in errors)
    for extensions in ([], "bam", ["file name"], ["a/b"]):
        assert validate_spec(dict(legacy, image=IMAGE, data_formats=extensions), cfg, version)
    for ref in ("registry.example:5000/ns/image:1.2", BASE, BASE.replace(":24.04", "")):
        assert not validate_spec(dict(legacy, image=dict(IMAGE, base=ref), data_formats=[".bam", "nii.gz"]), cfg, version)


@pytest.mark.parametrize("base", ["python", "localhost:5000/pkg", "python:latest", BASE.replace("24.04", "latest"),
                                  "python:TODO", "python:REPLACE_ME", "python:<version>", "python:3@sha256:REPLACE"])
def test_spec_rejects_invalid_image_base(base):
    spec = dict(SPEC, generator_assumptions=[], data_formats=["csv"], image=dict(IMAGE, base=base))
    assert any("base" in e for e in validate_spec(spec, load_config(), CODE_PROMPT_VERSION))


def test_s6a_warns_on_public_format_mismatch(tmp_path, monkeypatch):
    P, it, _ = setup_agent(tmp_path, monkeypatch, {}, base=BASE)
    spec = read_json(it.p("_spec/spec.json"))
    spec["data_formats"] = [".BAM", "nii.gz"]
    write_json(it.p("_spec/spec.json"), spec)
    P.cfg["gen"]["seeds"] = [0, 1]
    def generate(cfg, task, seed, gen):
        public = gen / "environment/data/subdir"
        public.mkdir(parents=True)
        (public / "volume.nii.gz").touch()
        (public / ("data.bam" if seed == 0 else "data.tsv")).touch()
        (gen / "tests/hidden").mkdir(parents=True)
        (gen / "tests/hidden/private.h5").touch()
        return dict(rc=0, sec=0, err="", public_mb=0)
    monkeypatch.setattr(runner, "generate", generate)
    monkeypatch.setattr(runner, "speed_bench", lambda *args: 1)
    monkeypatch.setattr(runner, "solve", lambda *args: dict(rc=0, sec=0, err="", outdir=str(tmp_path)))
    monkeypatch.setattr(runner, "measure", lambda *args: dict(rc=0, err="", values={}))
    stages_a.s6a_run(P, it)
    assert it.load().state["stage"] == "S6B"
    runs = read_json(it.p("S6A_measure.json"))["runs"]
    assert "data_formats_warning" not in runs["gen0"]
    assert runs["gen1"]["data_formats_warning"] == dict(missing=["bam"], undeclared=["tsv"])


@pytest.mark.parametrize(("hit", "reason"), [(True, "readymade_image"), (False, "image_build_failed:built")])
def test_retry_uses_s5g_budget_then_drops(tmp_path, monkeypatch, hit, reason):
    P, it, launcher = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF + "RUN fixture-install\n"},
                                  authoring_timeout_s=10000)
    packages = installed("conda", '[{"name":"FastSmoother","version":"1"}]')
    # Simulate the remote build layer only. The S5G launcher continues to enforce its normal SSH contract.
    monkeypatch.setattr(ti, "_host", lambda cfg: None)
    fake_host(monkeypatch, packages, fail_build=not hit)
    P.run(max_steps=1, until="S6I")
    assert it.load().state["stage"] == "S5G"
    P.run(max_steps=3, until="S6I")
    assert it.load().state["reason"] == reason
    assert len(launcher.calls) == 2 and it.state["attempts"]["S5G:job"] == 2
    feedback = read_json(it.p("S5G/work/previous_attempt.json"))["S6_failure.json"]
    assert ("FastSmoother" if hit else "fixture-install") in feedback["finding"]


def test_inventory_and_pins(tmp_path):
    rows = installed("dpkg", "fastsmoother\t1.0\nsamtools\t1.2\n")
    write_json(tmp_path / "state.json", dict(image_kind="authored", inventory=rows))
    check = prerelease.readymade_check(tmp_path / "task", {}, dict(method_names=["FastSmoother"]))
    assert [(p["name"], p["file"]) for p in check["findings"]] == [("fastsmoother", "dpkg")]
    spec = dict(SPEC, generator_assumptions=[], data_formats=["h5"], prompt_version=CODE_PROMPT_VERSION)
    assert validate_spec(spec, load_config())
    assert not validate_spec(dict(spec, image=IMAGE), load_config())
    for bad in [None, [], dict(IMAGE, base=5), dict(IMAGE, packages="h5py"), dict(IMAGE, notes=[])]:
        assert validate_spec(dict(spec, image=bad), load_config())


def test_runner_uses_tag_and_cleanup_image(tmp_path, monkeypatch):
    cfg = load_config(overrides=dict(taskimage={"enabled": True}, sandbox=dict(
        backend="ssh", host="test", remote_dir="/tmp/sb", cleanup_image="helper:test")))
    write_json(tmp_path / "state.json", dict(image_kind="authored", image_tag="tbsgen-task:local",
                                               image_digest="sha256:" + "c"*64))
    (tmp_path / "task/generator").mkdir(parents=True)
    calls = []
    def command(cmd, timeout):
        calls.append(cmd)
        return dict(rc=0, out="", err="")
    monkeypatch.setattr(sandbox, "sh", command)
    assert runner.generate(cfg, tmp_path / "task", 0, tmp_path / "gen")["rc"] == 0
    ssh = [c[-1] for c in calls if c[0] == "ssh"]
    assert any("sha256:" + "c"*64 + " bash -lc" in line for line in ssh)
    assert any("helper:test chmod" in line for line in ssh)


def test_prepare_receives_layout(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    state = dict(image_kind="task", image_tag="tbsgen-task:test", image_digest="sha256:" + "c"*64,
                 image_manifest="sha256:" + "a"*64)
    write_json(tmp_path / "state.json", state)
    write_json(pkg / "provenance.json", state)
    (pkg / "task.toml").write_text("[verifier]\ntimeout_sec = 900\n")
    cfg = load_config(overrides=dict(taskimage={"enabled": False}, calib=dict(
        prepared_tasks_dir=str(tmp_path / "prepared"), prepare_cwd=str(tmp_path),
        prepare_command='mkdir -p {out} && cp {pkg}/task.toml {out}/ && '
                        'echo "$TBSGEN_IMAGE_OCI $TBSGEN_IMAGE_TAG" > {out}/image-layout')))
    calls = []
    def publish(cfg, item):
        calls.append(item)
        digest = "sha256:" + "f"*64
        write_json(item / "calib/image-push.json", dict(digest=digest, pkg_sha=calib.pkg_sha(item / "pkg")))
        return dict(TBSGEN_IMAGE_OCI="builder:/layout", TBSGEN_IMAGE_TAG="acr:tag", TBSGEN_IMAGE_DIGEST=digest)
    monkeypatch.setattr(ti, "publish", publish)
    rec = calib.ensure_prepared(cfg, tmp_path)
    assert rec["ok"] and (Path(rec["out"]) / "image-layout").read_text().strip() == "builder:/layout acr:tag"
    assert calib.ensure_prepared(cfg, tmp_path)["reused"] and calls == [tmp_path]


def test_gc_selection():
    states = dict(a=dict(status="done", stage="S10", acr_push=True), b=dict(status="dropped"),
                  c=dict(status="paused", updated=0), d=dict(status="active"), e=dict(status="done", stage="S10"))
    keys = [item + "-" + "a"*20 for item in "abcdef"]
    tags = ["tbsgen-task:" + key for key in keys]
    result = ti.gc_selection(states, tags + ["ubuntu:latest", "tbsgen-task:other"], keys + ["../../a"], 7, 10*86400)
    assert result == dict(images=tags[:3], layouts=keys[:3])


def test_publish_full_oci_reuse_and_change(tmp_path, monkeypatch):
    import subprocess
    cfg = load_config(overrides=dict(run={"root": str(tmp_path)}, taskimage={"enabled": True, "builder": "tbs"}))
    item = tmp_path / "items/test"
    env = item / "pkg/environment"
    (env / "data").mkdir(parents=True)
    digest = "sha256:" + "c"*64
    manifest = "sha256:" + "e"*64
    state = dict(image_tag="tbsgen-task:test-" + "a"*20, image_digest=digest, image_manifest=manifest)
    write_json(item / "state.json", state)
    write_json(item / "pkg/provenance.json", state)
    (env / "Dockerfile").write_text(ti.data_recipe(ti.harbor_tag(item, digest)))
    def verify(cfg, action, path, expected_id, expected_manifest, *args):
        assert action == "verify" and expected_id == digest and expected_manifest == manifest
        return ("sha256:" + "b"*64) if args else manifest
    monkeypatch.setattr(ti, "_oci", verify)
    monkeypatch.setattr(ti, "_export_base", lambda cfg, item, expected: manifest if expected == digest else None)
    builds = []
    def execute(cfg, argv, timeout=60, *, check=True, full=False):
        if argv[0] == "docker":
            assert argv[:7] == ["docker", "buildx", "build", "--builder", "tbs", "--platform", "linux/amd64"]
            ctx = Path(argv[-1])
            assert (ctx / "Dockerfile").read_text() == "FROM taskbase\nCOPY data /app/data\n"
            assert argv[argv.index("--build-context") + 1].endswith("@" + manifest)
            assert "--provenance=false" in argv
            dest = Path(argv[-2].split("dest=")[1])
            write_json(dest / "index.json", {"manifests": []})
            (dest / "blobs").mkdir()
            (dest / "blobs/blob").write_bytes(b"1234")
            builds.append((ctx / "data/input").read_text())
            return dict(rc=0, out="", err="")
        if argv[0] == "realpath":
            return dict(rc=0, out=argv[-1], err="")
        assert argv[0] in ("test", "mkdir", "python3", "rm")
        res = subprocess.run(argv, check=check, text=True, capture_output=True)
        return dict(rc=res.returncode, out=res.stdout, err=res.stderr)
    monkeypatch.setattr(ti, "_exec", execute)
    (env / "data/input").write_text("seed A")
    with pytest.raises(ValueError, match="taskimage.registry_prefix"):
        ti.publish(cfg, item)
    assert not builds
    cfg["taskimage"]["registry_prefix"] = "example.org/ns/repo:tbsgen-img"
    a = ti.publish(cfg, item)
    assert ti.publish(cfg, item) == a
    (env / "data/input").write_text("seed B")
    b = ti.publish(cfg, item)
    assert a["TBSGEN_IMAGE_TAG"] != b["TBSGEN_IMAGE_TAG"] and builds == ["seed A", "seed B"]
    cfg["taskimage"]["max_gb"] = 1e-9
    with pytest.raises(ValueError, match="exceeds"):
        ti.publish(cfg, item)
    (env / "Dockerfile").write_text(DF)
    with pytest.raises(ValueError, match="differs"):
        ti.publish(cfg, item)


def test_prepare_script_records_verified_digest(tmp_path):
    import os
    import subprocess
    script = Path(__file__).resolve().parents[1] / "scripts/prep-task-image.sh"
    pkg = tmp_path / "run/items/example/pkg"
    pkg.mkdir(parents=True)
    (pkg / "task.toml").write_text("fixture")
    out = tmp_path / "run/sandbox/tasks/example"
    tools = tmp_path / "tools"
    tools.mkdir()
    digest = "sha256:" + "e"*64
    write_json(tmp_path / "oci/index.json", dict(manifests=[dict(digest=digest)]))
    (tools / "crane").write_text(f'#!/bin/bash\ncase "$1" in push) exit 0;; digest) echo "{digest}";; *) exit 2;; esac\n')
    (tools / "uv").write_text("""#!/bin/bash
set -eu
cmd=$6
state=$7
if [[ "$cmd" == init ]]; then
    test -f "$state/image-overrides.json"
elif [[ "$cmd" == run ]]; then
    mkdir -p "$TBSGEN_PREPARED_DIR"
    cp "$TBSGEN_PKG/task.toml" "$TBSGEN_PREPARED_DIR/task.toml"
    echo '{"done":1,"waiting":0,"failures":[]}' > "$state/register-summary.json"
else
    exit 2
fi
""")
    for f in ("crane", "uv"):
        (tools / f).chmod(0o755)
    (tools / "direct-env.sh").touch()
    tag = "example.org/bench:tbsgen-img-example-b"
    env = dict(os.environ, TBSGEN_PKG=str(pkg), TBSGEN_ITEM_ID="example", TBSGEN_PREPARED_DIR=str(out),
               TBSGEN_PKG_SHA="pkg-sha", TBSGEN_IMAGE_DIGEST=digest, TBSGEN_IMAGE_OCI=str(tmp_path / "oci"), TBSGEN_IMAGE_TAG=tag,
               SANDBOX_TEMPLATES=str(tools / "templates.py"), DOCKER_CONFIG=str(tools / "config"),
               CRANE=str(tools / "crane"), TBS_ALIYUN_ROOT=str(tools), PATH=str(tools)+":"+os.environ["PATH"])
    for required in ("TBS_ALIYUN_ROOT", "CRANE", "DOCKER_CONFIG", "SANDBOX_TEMPLATES"):
        result = subprocess.run(["bash", script, "example"], env=dict(env, **{required: ""}), capture_output=True)
        assert result.returncode and required.encode() in result.stderr
    legacy = dict(env, TBSGEN_IMAGE_OCI="", TBSGEN_LEGACY_PREP="")
    refused = subprocess.run(["bash", script, "example"], env=legacy, capture_output=True)
    assert refused.returncode and b"TBSGEN_LEGACY_PREP" in refused.stderr
    (tools / "legacy.sh").write_text('test "$1" = example\n')
    subprocess.run(["bash", script, "example"], env=dict(legacy, TBSGEN_LEGACY_PREP=str(tools / "legacy.sh")), check=True)
    overrides = tmp_path / "run/sandbox/state/image-overrides.json"
    write_json(overrides, {"previous/agent": "other"})
    subprocess.run(["bash", script, "example"], env=env, check=True)
    assert read_json(overrides) == {"example/agent": "example.org/bench@"+digest, "previous/agent": "other"}
    assert read_json(pkg.parent / "calib/image-push.json") == dict(tag=tag, digest=digest, pkg_sha="pkg-sha")
    assert (out / "task.toml").read_text() == "fixture"


@pytest.mark.parametrize(("fields", "gate"), [
    ({"floor_errors": ["missing nobody"]}, "verifier_floor"),
    ({"import_errors": ["h5py"]}, "offline_imports"),
    ({"arch": "linux/arm64"}, "arch"),
    ({"size": 5_000_000_001}, "size"),
])
def test_build_gates(tmp_path, monkeypatch, fields, gate):
    cfg = load_config(overrides=dict(run={"root": str(tmp_path)}, taskimage={"enabled": True}))
    item = tmp_path / "items/example"
    (item / "task/environment").mkdir(parents=True)
    (item / "task/environment/Dockerfile").write_text(DF)
    write_json(item / "_spec/spec.json", {"image": IMAGE})
    monkeypatch.setattr(ti, "_load_image", lambda *args: "digest")
    probe = dict(floor_errors=[], import_errors=[], inventory=INVENTORY, python="3.11", arch="linux/amd64", size=1)
    monkeypatch.setattr(ti, "inspect", lambda *args: dict(probe, **fields))
    result = ti.build(cfg, item)
    assert result["failed_gate"] == gate and result["gates"][gate] is False


def test_gc_removes_only_owned_assets(tmp_path, monkeypatch):
    root = tmp_path / "run"
    cfg = load_config(overrides=dict(run={"root": str(root)}, taskimage={"enabled": True}))
    key = "sample-" + "1"*20
    tag = "tbsgen-task:" + key
    write_json(root / "items/sample/state.json", dict(item_id="sample", stage="S6I", status="dropped",
                                                       image_tag=tag, updated=100))
    actions = []
    def execute(cfg, cmd, *args, **kwargs):
        if cmd[:3] == ["docker", "image", "ls"]:
            return dict(out=tag + "\ntbsgen-task:sample-" + "2"*20 + "\nother:tag\n")
        if cmd[0] == "test":
            return dict(rc=0)
        if cmd[0] == "ls":
            return dict(out=key+"\nunrelated\n")
        actions.append(cmd)
        return dict(rc=0)
    monkeypatch.setattr(ti, "_exec", execute)
    expected = dict(images=[tag], layouts=[key])
    assert ti.gc(cfg, dry_run=True) == expected and not actions
    assert ti.gc(cfg) == expected and len(actions) == 2
    assert actions == [["docker", "image", "rm", tag], ["rm", "-rf", "--", ti._namespace(cfg) + "/oci/" + key]]
    assert read_json(root / "items/sample/taskimage-evicted.json") == {"image_tag": tag}


def test_public_data_cap(tmp_path, monkeypatch):
    cfg = load_config(overrides={"gen": {"max_public_mb": 0.001}})
    (tmp_path / "task/generator").mkdir(parents=True)
    write_json(tmp_path / "state.json", dict(image_kind="task", image_digest="sha256:" + "c"*64))
    def fill(cfg, mounts, *args, **kwargs):
        (mounts[0][0] / "environment/data/raw.fits").write_bytes(b"x"*1001)
        return dict(rc=0, out="", err="")
    monkeypatch.setattr(sandbox, "run", fill)
    result = runner.generate(cfg, tmp_path / "task", 0, tmp_path / "gen")
    assert result["rc"] and result["hash_env"] is None and "max_public_mb" in result["err"]


def test_build_host_command_and_context_cleanup(tmp_path, monkeypatch):
    cfg = load_config(overrides=dict(run={"root": str(tmp_path)}, sandbox=dict(backend="ssh", host="build-host", remote_dir="/runs"),
                                     taskimage=dict(enabled=True, builder="tbs", docker_config="$HOME/registry",
                                                    build_timeout_s=2400)))
    calls = []
    def command(argv, timeout, **kwargs):
        calls.append((argv, timeout))
        return dict(rc=0, out="", err="")
    monkeypatch.setattr(sandbox, "sh", command)
    with ti._context(cfg, tmp_path) as path:
        ti._exec(cfg, ti._buildx(cfg) + ["--load", "-t", "tbsgen-task:demo", path], 2400)
    assert calls[0][0][-1] == "mkdir -p " + path
    assert calls[1][0][0] == "rsync" and calls[1][0][-1] == "build-host:" + path + "/"
    argv, timeout = calls[2]
    assert argv[:2] == ["ssh", "-o"] and argv[-2] == "build-host" and timeout > 2400
    assert 'DOCKER_CONFIG="$HOME"/registry timeout -k 30 2400 docker buildx build --builder tbs --platform linux/amd64 --load' in argv[-1]
    assert calls[3][0][-1] == "rm -rf -- " + path


def test_build_args_clear_builder_proxy():
    cfg = load_config(overrides=dict(taskimage=dict(enabled=True, builder="tbs",
                                                    build_args={"HTTPS_PROXY": "", "HTTP_PROXY": ""})))
    argv = ti._buildx(cfg)
    assert argv[:6] == ["docker", "buildx", "build", "--builder", "tbs", "--platform"]
    assert argv[-4:] == ["--build-arg", "HTTPS_PROXY=", "--build-arg", "HTTP_PROXY="] or \
        argv[-4:] == ["--build-arg", "HTTP_PROXY=", "--build-arg", "HTTPS_PROXY="]
    assert "--build-arg" not in ti._buildx(load_config())


def test_runtime_permissions_after_prepare(tmp_path, monkeypatch):
    import pwd
    import subprocess
    from tbsgen import taskimage_probe, templates_metrics
    app = tmp_path / "app"
    python = app / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch(mode=0o755)
    site = app / ".venv/lib/site-packages"
    site.mkdir(parents=True)
    out = app / "output"
    out.mkdir()
    assert '_open_workspace("/app/output")' in runner.metrics_template(True)
    templates_metrics._open_workspace(str(out))
    paths = taskimage_probe.runtime_paths(str(python), [str(site)], [str(python.parent)])
    assert {str(python), str(python.parent), str(site), str(app)} <= set(paths)
    assert all(not p.stat().st_mode & 2 for p in (python, python.parent, site, app))
    assert out.stat().st_mode & 2
    nobody = pwd.getpwnam("nobody")
    check_output = subprocess.check_output
    def run_check(argv, **kwargs):
        assert kwargs.pop("user") == nobody.pw_uid and kwargs.pop("group") == nobody.pw_gid
        assert kwargs.pop("extra_groups") == []
        # This host cannot change uid. Run the actual checker as the owner of the fixture runtime.
        return check_output(argv, **kwargs)
    monkeypatch.setattr(subprocess, "check_output", run_check)
    assert str(python) in taskimage_probe.writable_runtime(paths, nobody)
    assert '_open_workspace("/app")' in runner.metrics_template(False)
    templates_metrics._open_workspace(str(app))
    assert app.stat().st_mode & 2


@pytest.mark.parametrize("mount", ["target=/mnt", "type=bind,target=/mnt", "type=cache,from=ctx,target=/mnt"])
def test_context_mounts(mount):
    assert ti.lint(DF + f"RUN --mount={mount} true")


def test_downloads_checked_individually():
    line = "RUN curl -o a https://example.org/a && wget -O b https://example.org/b"
    check = " && echo '" + "a" * 64 + "  {}' | sha256sum -c -"
    assert ti.lint(DF + line + check.format("a"))
    assert not ti.lint(DF + line + check.format("a") + check.format("b"))


def test_build_args_in_local_tag(tmp_path, monkeypatch):
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF})
    P.cfg["sandbox"]["backend"] = "docker"
    calls = fake_host(monkeypatch, INVENTORY)
    P.cfg["taskimage"]["build_args"] = {"CHANNEL": "one"}
    a = ti.build(P.cfg, it.dir)
    P.cfg["taskimage"]["build_args"]["CHANNEL"] = "two"
    b = ti.build(P.cfg, it.dir)
    assert a["image_tag"] != b["image_tag"]
    assert any("CHANNEL=one" in cmd and "--build-arg" in cmd for cmd in calls)


@pytest.mark.parametrize("kind", ["authored", "task"])
def test_flag_off_keeps_verified_image(tmp_path, monkeypatch, kind):
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF})
    P.cfg["sandbox"]["backend"] = "docker"
    fake_host(monkeypatch, INVENTORY)
    P.run(max_steps=1, until="S6I")
    it.load()
    it.state["image_kind"] = kind
    it.save()
    P.cfg["taskimage"]["enabled"] = False
    assert ti.image_for(P.cfg, it.dir) == it.state["image_digest"]


@pytest.mark.parametrize("stage", ["S6A_run", "S7", "S9", "S9R"])
def test_shared_tag_eviction_regates(tmp_path, monkeypatch, stage):
    from tbsgen.pipeline import HANDLERS
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF})
    tag = "tbsgen-task:shared-" + "0"*20
    it.state.update(stage=stage, image_tag=tag, image_digest="sha256:" + "0"*64)
    it.save()
    sibling = tmp_path / "sibling/items" / it.item_id
    write_json(sibling / "taskimage-evicted.json", {"image_tag": tag})
    monkeypatch.setattr(ti, "_image_id", lambda cfg, ref: None)
    monkeypatch.setitem(HANDLERS, stage, lambda *args: pytest.fail("used a missing image"))
    P.step(it)
    assert it.load().state["stage"] == "S6I" and it.state["image_resume_stage"] == stage


def test_legacy_public_data_60mb(tmp_path, monkeypatch):
    cfg = load_config(overrides={"gen": {"max_public_mb": 50}})
    (tmp_path / "task/generator").mkdir(parents=True)
    def generate(cfg, mounts, *args):
        with (mounts[0][0] / "environment/data/raw").open("wb") as f:
            f.truncate(60_000_000)
        return dict(rc=0, out="", err="")
    monkeypatch.setattr(sandbox, "run", generate)
    result = runner.generate(cfg, tmp_path / "task", 0, tmp_path / "gen")
    assert result["rc"] == 0 and result["public_mb"] == 60


def test_harbor_local_task_recipe(tmp_path, monkeypatch):
    import subprocess
    from tbsgen import harbor_local
    item = tmp_path / "items/example"
    tag = "tbsgen-task:example-" + "a"*20
    digest = "sha256:" + "b"*64
    state = dict(image_kind="task", image_tag=tag, image_digest=digest, image_manifest="sha256:" + "e"*64)
    write_json(item / "state.json", state)
    write_json(item / "pkg/provenance.json", state)
    df = item / "pkg/environment/Dockerfile"
    df.parent.mkdir()
    recipe = f"FROM tbsgen-task:example-{digest[7:]}\nCOPY data /app/data\n"
    df.write_text(recipe)
    monkeypatch.setattr(ti, "_image_id", lambda cfg, ref: digest)
    def harbor(cmd, **kwargs):
        task = Path(cmd[cmd.index("-p") + 1])
        assert (task / "environment/Dockerfile").read_text() == recipe
        job = Path(cmd[cmd.index("-o") + 1]) / cmd[cmd.index("--job-name") + 1]
        reward = int(cmd[cmd.index("-a") + 1] == "oracle")
        write_json(job / "trial/result.json", dict(verifier_result=dict(rewards=dict(reward=reward))))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", harbor)
    assert harbor_local.check(load_config(), tmp_path, item, harbor="fixture")["ok"]


def test_private_review_context_uses_verified_base(tmp_path, monkeypatch):
    from tbsgen import review
    item = tmp_path / "items/example"
    tag = "tbsgen-task:example-" + "a"*20
    state = dict(image_kind="task", image_tag=tag, image_digest="sha256:" + "b"*64, image_manifest="sha256:" + "e"*64)
    write_json(item / "state.json", state)
    write_json(item / "pkg/provenance.json", state)
    env = item / "pkg/environment"
    (env / "data").mkdir(parents=True)
    (env / "Dockerfile").write_text(DF + "RUN echo author-step\n")
    (item / "pkg/task.toml").write_text("[environment]\n")
    monkeypatch.setattr(ti, "_image_id", lambda cfg, ref: state["image_digest"])
    monkeypatch.setattr(review, "instruction", lambda *args: "review")
    task = review.build_task(load_config(), item, state, 1)
    recipe = (task / "environment/Dockerfile").read_text()
    assert recipe.startswith("FROM tbsgen-task:example-" + "b"*64 + "\n") and "author-step" not in recipe
    assert "COPY review /review" in recipe


@pytest.mark.parametrize("compressed", [False, True])
def test_export_daemon_image_and_verify_ancestry(tmp_path, monkeypatch, compressed):
    import gzip
    import hashlib
    import io
    import subprocess
    import tarfile
    from tbsgen import taskimage_oci as oci

    item = tmp_path / "items/test"
    cfg = load_config(overrides=dict(run={"root": str(tmp_path)}))
    def sha(blob):
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    def archive(path, contents):
        layers = [f"layer{i}.tar" for i in range(len(contents))]
        config = json.dumps(dict(rootfs=dict(type="layers", diff_ids=list(map(sha, contents))))).encode()
        manifest = json.dumps([dict(Config="config.json", Layers=layers)]).encode()
        blobs = dict(zip(layers, map(gzip.compress, contents) if compressed else contents))
        with tarfile.open(path, "w") as tar:
            for name, data in dict(blobs, **{"config.json": config, "manifest.json": manifest}).items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return sha(config)

    layers = [b"fixture dependency layer"]
    digest = archive(tmp_path / "base.tar", layers)
    saves = []
    def execute(cfg, cmd, timeout=60, *, check=True, full=False):
        if cmd[:3] == ["docker", "image", "save"]:
            assert cmd[-1] == digest
            Path(cmd[-2]).write_bytes((tmp_path / "base.tar").read_bytes())
            saves.append(cmd)
            return dict(rc=0, out="", err="")
        result = subprocess.run(cmd, check=check, text=True, capture_output=True)
        return dict(rc=result.returncode, out=result.stdout, err=result.stderr)
    monkeypatch.setattr(ti, "_exec", execute)
    manifest = ti._export_base(cfg, item, digest)
    assert ti._export_base(cfg, item, digest) == manifest and len(saves) == 1
    base = Path(ti._base_layout(cfg, item, digest))
    restored = tmp_path / "restore.tar"
    oci.archive_for_load(base, digest, manifest, restored)
    oci.export(restored, tmp_path / "restored", digest)
    assert oci.verify(tmp_path / "restored", digest) == manifest
    for name, deps in [("good", layers), ("wrong", [b"other dependencies"])]:
        final = tmp_path / name
        image_id = archive(tmp_path / "full.tar", deps + [b"task data"])
        oci.export(tmp_path / "full.tar", final, image_id)
        if name == "good":
            assert oci.verify(base, digest, manifest, final) == oci.identity(final)[0]
        else:
            with pytest.raises(ValueError, match="dependency layers"):
                oci.verify(base, digest, manifest, final)
    with pytest.raises(ValueError, match="S6I digest"):
        oci.verify(base, sha(b"other config"), manifest)
    with pytest.raises(ValueError, match="S6I digest"):
        oci.verify(base, digest, sha(b"other manifest"))


def test_gc_pending_review_and_sibling_regate(tmp_path, monkeypatch):
    from tbsgen.pipeline import Pipeline
    from tbsgen import stages_b
    a, b = tmp_path / "run-a", tmp_path / "run-b"
    tag = "tbsgen-task:shared-" + "a"*20
    digest = "sha256:" + "b"*64
    state = dict(item_id="shared", stage="S10", status="done", updated=1, image_kind="task", image_tag=tag,
                 image_digest=digest, image_manifest="sha256:" + "e"*64)
    cfg = load_config(overrides=dict(run={"root": str(a)}))
    P = Pipeline(cfg)
    for root in (a, b):
        item = root / "items/shared"
        write_json(item / "state.json", state if root == a else dict(state, stage="S9R", status="waiting"))
        write_json(item / "calib/image-push.json", dict(digest=digest, pkg_sha="pkg"))
        write_json(item / "calib/prepare.json", dict(ok=True, pkg_sha="pkg"))
    present = {tag: digest}
    def execute(cfg, cmd, *args, **kwargs):
        if cmd[:3] == ["docker", "image", "ls"]:
            return dict(out="\n".join(present))
        if cmd[0] == "test":
            return dict(rc=1)
        assert cmd == ["docker", "image", "rm", tag]
        del present[tag]
        return dict(rc=0)
    monkeypatch.setattr(ti, "_exec", execute)
    cfg_b = dict(cfg, run={"root": str(b)})
    assert not ti.gc(cfg)["images"] and not ti.gc(cfg_b)["images"]
    write_json(b / "items/shared/state.json", state)
    assert ti.gc(cfg_b)["images"] == [tag]
    it = P.store.item("shared")
    assert not it.p("taskimage-evicted.json").exists()
    it.p("pkg").mkdir()
    write_json(it.p("thresholds.json"), {})
    stages_b.regate(P, it, "verify")
    monkeypatch.setattr(ti, "_image_id", lambda cfg, key: present.get(key))
    P.step(it)
    assert it.load().state["stage"] == "S6I" and it.state["image_resume_stage"] == "S6A_run"


@pytest.mark.parametrize("problem", ["missing host", "wrong ID"])
def test_resume_pauses_on_config_or_identity_error(tmp_path, monkeypatch, problem):
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF})
    it.state.update(stage="S7", image_kind="task", image_tag="tbsgen-task:test", image_digest="sha256:" + "b"*64)
    it.save()
    P.cfg["taskimage"]["enabled"] = False
    if problem == "missing host":
        P.cfg["sandbox"]["host"] = None
    monkeypatch.setattr(ti, "_image_id", lambda *args: "sha256:" + "a"*64)
    assert P.step(it) == "paused"
    assert ("sandbox.host" if problem == "missing host" else "expected S6I ID") in it.load().state["reason"]


def test_verified_image_resume_from_pre_oci_package(tmp_path, monkeypatch):
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF})
    P.cfg["sandbox"]["backend"] = "docker"
    fake_host(monkeypatch, INVENTORY)
    P.run(max_steps=1, until="S6I")
    it.load()
    it.state.update(stage="S9")
    it.save()
    write_json(it.p("pkg/provenance.json"), {k: it.state[k] for k in ("image_tag", "image_digest")})
    write_json(it.p("thresholds.json"), {})
    write_json(it.p("calib/runs.json"), [dict(outcome="pass")])
    P.cfg["taskimage"]["enabled"] = False
    P.step(it)
    assert it.load().state["stage"] == "S6A_run" and it.state["thresholds_locked"]
    assert it.state["calib_epoch"] == 1 and it.p("calib/epoch-0/runs.json").is_file()


def test_missing_image_on_separate_execution_host(tmp_path, monkeypatch):
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF})
    P.cfg["taskimage"]["host"] = "builder"
    it.state.update(stage="S7", image_digest="sha256:" + "a"*64, image_tag="tbsgen-task:test",
                    image_manifest="sha256:" + "e"*64)
    it.save()
    hosts = []
    def lookup(cfg, tag):
        hosts.append(ti._host(cfg))
        return it.state["image_digest"] if hosts[-1] == "builder" else None
    monkeypatch.setattr(ti, "_image_id", lookup)
    P.step(it)
    assert hosts == ["builder", "bh"] and it.load().state["stage"] == "S6I"


def test_default_authoring_contract(tmp_path, monkeypatch):
    from tbsgen.prompts_code import LEGACY_AUTHOR
    P, it, _ = _agent_item(tmp_path, monkeypatch, {}, stage="S5G")
    assert not P.cfg["taskimage"]["enabled"]
    assert P.cfg["gen"]["max_public_mb"] == 50
    assert load_config(overrides={"taskimage": {"enabled": True}})["gen"]["max_public_mb"] == 500
    P.step(it)
    rec = read_json(it.p("S5G/run.json"))
    assert rec["prompt_version"] == "cg-v6-2026-11-25" and "image_outputs" not in rec
    staged = Path(rec["task"])
    expected = LEGACY_AUTHOR.format(item=it.item_id, minutes=int(P.cfg["code"]["authoring_timeout_s"] // 60),
                                   forms=P.cfg["spec"]["allowed_forms"], hours_lo=2, hours_hi=6)
    assert (staged / "instruction.md").read_text() == expected
    # Fingerprints of the submitted cg-v6 contract, including whitespace.
    assert hashlib.sha256(LEGACY_AUTHOR.encode()).hexdigest() == "d6f90dabd0797a884cb57c121d4cf60403ec9987b932233fef5ab2dc330051f5"
    contract = staged / "environment/work/contract"
    for name, digest in {
        "S5_SPEC.txt": "23cd1d846b0ad6bbf2ffeab88a5d54cfeef0182c10c2ed45ec5ba7a4021dbb62",
        "S6A_GEN.txt": "fc743443f09344b2cf53b2f4973b169f566e32dee8ca3bc962c9a2c81972ba26",
    }.items():
        assert hashlib.sha256((contract / name).read_bytes()).hexdigest() == digest
    assert (contract / "_metrics_template.py").read_bytes() == runner.METRICS_HELPER.read_bytes()


def test_remote_image_authoring_requires_complete_context(tmp_path, monkeypatch):
    P, it, _ = setup_agent(tmp_path, monkeypatch, {"Dockerfile": DF})
    rec = dict(read_json(it.p("S5G/run.json")), ledger_stage="S5G")
    assert rec["image_outputs"]
    # Recovery uses the submitted contract even after an operator turns the feature off.
    P.cfg["taskimage"]["enabled"] = False
    trial = Path(rec["trial"])
    assert not calib._remote_results_ready(P.cfg, rec)
    receipt = trial / "verifier/image-files.json"
    env = trial / "verifier/out/task/environment"
    write_json(receipt, ["Dockerfile", "requirements.txt"])
    assert not calib._remote_results_ready(P.cfg, rec)
    (env / "requirements.txt").write_text("h5py==3.11.0")
    assert calib._remote_results_ready(P.cfg, rec)
    (trial / "verifier/out/report.md").unlink()
    assert not calib._remote_results_ready(P.cfg, rec)

@pytest.mark.parametrize("stage", ["S9", "S9R"])
def test_claimed_job_reconciles_before_image_restoration(tmp_path, monkeypatch, stage):
    from test_concurrency import _interrupted_job
    P, it, _, read = _interrupted_job(tmp_path, monkeypatch, stage, False, remote_route=True)
    original = read()
    it.load().state.update(image_kind="task")
    it.save()
    monkeypatch.setattr(calib, "session_alive", lambda *_: False)
    monkeypatch.setattr(ti, "needs_restore", lambda *args: pytest.fail("interrupted submission bypassed"))
    P.step(it)
    assert read()["state"] == "remote_unknown"
    assert it.load().state["reason"] == f"remote_unknown: {original['job_name']}"
    assert read()["ledger_call"] == original["ledger_call"]
