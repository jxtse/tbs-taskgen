"""Author-owned dependencies, pipeline-owned data layer, and build-host image lifecycle."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import posixpath
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from . import prerelease, sandbox
from .state import read_json, write_json

REQUIRED_TAIL = """WORKDIR /app
COPY data /app/data
RUN mkdir -p /app/output
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
"""
DIGEST = r"sha256:[0-9a-f]{64}"
BASE_REF = re.compile(
    r"(?:(?:[a-z0-9][a-z0-9.-]*|\[[a-f0-9:]+\])(?::[0-9]+)?/)?"
    rf"[a-z0-9]+(?:(?:[./]|__?|-+)[a-z0-9]+)*(?::(?P<tag>[\w][\w.-]{{0,127}}))?(?:@(?P<digest>{DIGEST}))?", re.ASCII)
FROM_ARGS = re.compile(r"(?P<platform>--platform=linux/amd64\s+)?(?P<ref>\S+)(?P<alias>\s+AS\s+[\w.-]+)?", re.I)
IMPORT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
CONTEXT_FILE = re.compile(r"(?:requirements(?:-[\w-]+)?\.txt|environment\.ya?ml|conda-lock\.ya?ml|[\w-]+\.lock)")
PIP_PIN = re.compile(r"[\w.-]+(?:\[[\w,.-]+\])?==[\w][\w.+!-]*")
CONDA_PIN = re.compile(r"(?:[\w.-]+::)?[\w.-]+=[\w.+-]+(?:=[\w.+-]+)?")
APT_PIN = re.compile(r"[\w.+-]+(?::[\w-]+)?=[\w:.+~-]+")
READERS = {".fits": {"astropy"}, ".h5": {"h5py"}, ".bam": {"pysam", "samtools"},
           ".nii": {"nibabel"}, ".tif": {"tifffile", "rasterio"}, ".nc": {"netcdf4", "xarray"},
           ".cif": {"gemmi", "biopython", "pymatgen"}, ".pdb": {"gemmi", "biopython", "pymatgen"},
           ".mzml": {"pyteomics"}}


STATE_FIELDS = ("base", "base_resolved_from", "image_kind", "image_tag", "image_digest", "image_manifest",
                "inventory_hash", "inventory")
GATES = ("resolve_base", "lint", "built", "verifier_floor", "offline_imports", "size", "arch", "readymade_image")


def lint_failure(errors):
    gates = dict.fromkeys(GATES)
    gates["lint"] = False
    return dict(ok=False, gates=gates, failed_gate="lint", error="; ".join(errors))


def enabled(cfg):
    return bool(cfg.get("taskimage", {}).get("enabled"))


def valid_base(ref):
    match = BASE_REF.fullmatch(ref) if isinstance(ref, str) else None
    return bool(match and (match["tag"] or match["digest"]) and (match["tag"] or "").lower() != "latest"
                and not re.search(r"(?<![a-z0-9])(?:replace|todo|placeholder)(?![a-z0-9])", ref, re.I))


def _logical_lines(lines):
    # Docker drops blank lines and full-line comments, including within continued instructions.
    start = None
    parts = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if start is None:
            start = i
        if line.endswith("\\"):
            parts.append(line[:-1])
            continue
        yield start, i + 1, " ".join([*parts, line])
        start = None
        parts.clear()
    if start is not None:
        yield start, len(lines), " ".join(parts) + "\\"


def _instructions(text):
    for _, _, line in _logical_lines(text.splitlines()):
        parts = line.split(None, 1)
        yield parts[0].upper(), parts[1] if len(parts) > 1 else ""


def bases(text):
    return [m["ref"] for op, args in _instructions(text) if op == "FROM" and (m := FROM_ARGS.fullmatch(args))]


def _copy(args):
    if args.startswith("["):
        words = json.loads(args)
        if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
            raise ValueError("invalid COPY")
        return words
    return shlex.split(args)


def _context_sources(text):
    sources = {}
    for op, args in _instructions(text):
        if op == "COPY" and not args.startswith("--from="):
            words = _copy(args)
            if len(words) < 2:
                raise ValueError("COPY needs a source and destination")
            for source in words[:-1]:
                name = source.removeprefix("./")
                if not CONTEXT_FILE.fullmatch(name):
                    raise ValueError(f"COPY {source}: only dependency manifests may enter the build context")
                dest = words[-1]
                sources[name] = dest + name if dest.endswith("/") else name if dest == "." else dest
    return sources


def context_errors(env):
    if env.is_symlink() or not env.is_dir():
        return ["invalid environment directory"]
    files = list(env.iterdir())
    errors = [f"unexpected build-context entry: {p.name}" for p in files
              if p.is_symlink() or not p.is_file() or not (p.name == "Dockerfile" or CONTEXT_FILE.fullmatch(p.name))]
    if errors:
        return errors
    sizes = {p.name: p.stat().st_size for p in files}
    errors = [f"{name} exceeds 1 MB" for name, size in sizes.items() if size > 1_000_000]
    if sum(sizes.values()) > 5_000_000:
        errors.append("environment exceeds 5 MB")
    if not errors:
        try:
            sources = set(_context_sources((env / "Dockerfile").read_text()))
            errors.extend(f"unused or missing build-context file: {name}"
                          for name in sorted(sources ^ (set(sizes) - {"Dockerfile"})))
        except (OSError, ValueError) as e:
            errors.append(str(e))
    return errors


def _pinned_environment(text):
    """Recognize conda's plain environment.yml list syntax. Leave complex YAML to explicit locks."""
    section = None
    seen = set()
    deps = 0
    pip_indent = None
    for raw in text.splitlines():
        line = raw.partition(" #")[0].rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith((" ", "-")):
            key, colon, value = line.partition(":")
            if key not in ("name", "channels", "dependencies") or not colon or key in seen:
                return False
            if key == "name" and not re.fullmatch(r"[\w.-]+", value.strip()):
                return False
            if key != "name" and value.strip():
                return False
            section = key
            seen.add(key)
            pip_indent = None
        elif stripped.startswith("- "):
            value = stripped[2:].strip()
            indent = len(line) - len(line.lstrip(" "))
            if section == "channels" and re.fullmatch(r"[\w./:-]+", value):
                continue
            if section != "dependencies":
                return False
            if pip_indent is not None and indent > pip_indent:
                if not PIP_PIN.fullmatch(value):
                    return False
            elif value == "pip:":
                pip_indent = indent
                continue
            else:
                pip_indent = None
                if not CONDA_PIN.fullmatch(value):
                    return False
            deps += 1
        else:
            return False
    return deps > 0


def _manifest_ok(path, manager):
    if path is None or not path.is_file() or path.is_symlink() or path.stat().st_size > 1_000_000:
        return False
    text = path.read_text()
    lines = [s.strip() for s in text.splitlines() if s.strip() and not s.lstrip().startswith("#")]
    if manager == "pip":
        return bool(lines) and all(PIP_PIN.fullmatch(s.split(" #")[0]) for s in lines)
    if path.name in ("environment.yml", "environment.yaml") and lines[:1] != ["@EXPLICIT"]:
        return _pinned_environment(text)
    # Conda's native @EXPLICIT format, with immutable package URLs and md5/sha256 hashes.
    return len(lines) > 1 and lines[0] == "@EXPLICIT" and all(
        re.fullmatch(r"https://\S+\.(?:conda|tar\.bz2)#(?:[\da-f]{32}|[\da-f]{64})", s) for s in lines[1:])


INSTALLER = re.compile(r"(?:^|/)(pip(?:3(?:\.\d+)?)?|conda|mamba|micromamba|apt-get|apt)$")


def _install_errors(words, manifests, workdir):
    # Work on a tokenized shell segment, including python -m pip and uv pip. Options can precede install.
    errors = []
    for i, word in enumerate(words):
        tool_match = INSTALLER.search(word)
        if not tool_match:
            continue
        tool = tool_match[1]
        verb = next((j for j in range(i + 1, len(words)) if words[j] in ("install", "create")), None)
        if verb is None:
            continue
        is_pip = tool.startswith("pip")
        is_apt = tool.startswith("apt")
        pin = PIP_PIN if is_pip else APT_PIN if is_apt else CONDA_PIN
        value_flags = {"--index-url", "-i"} if is_pip else set() if is_apt else {
            "-n", "--name", "-p", "--prefix", "-c", "--channel"}
        file_flags = ("-r", "--requirement") if is_pip else () if is_apt else ("-f", "--file")
        args = iter(words[verb + 1:])
        for arg in args:
            opt, eq, val = arg.partition("=")
            if opt in file_flags or is_pip and arg.startswith("-r"):
                if is_pip and arg.startswith("-r") and len(arg) > 2:
                    filename = arg[2:]
                else:
                    filename = val if eq else next(args, "")
                path = posixpath.normpath(posixpath.join(workdir, filename))
                if not _manifest_ok(manifests.get(path), "pip" if is_pip else tool):
                    errors.append(f"{tool}: missing or unpinned dependency file {filename!r}")
            elif opt in value_flags:
                if not (val if eq else next(args, "")):
                    errors.append(f"{tool}: missing value for {opt}")
            elif arg in ("-y", "-q", "-qq", "-qy", "-yq", "--yes", "--system", "--no-cache-dir", "--no-deps",
                         "--no-install-recommends", "--no-install-suggests", "--override-channels", "--quiet",
                         "--upgrade", "--break-system-packages"):
                continue
            elif not pin.fullmatch(arg):
                errors.append(f"{tool}: expected an exact version pin, got {arg!r}")
        break
    return errors


def _download_target(words):
    tool = Path(words[0]).name
    output_flags = {"curl": ("-o", "--output"), "wget": ("-O", "--output-document")}
    if tool not in output_flags:
        return None
    urls, targets = [], []
    args = iter(words[1:])
    for arg in args:
        if arg in output_flags[tool]:
            targets.append(next(args, ""))
        elif re.fullmatch(r"https?://[A-Za-z0-9._~:/?&=%+#-]+", arg):
            urls.append(arg)
        elif not (tool == "curl" and re.fullmatch(r"-[fsSL]+", arg) or tool == "wget" and arg == "-q"):
            return None
    if len(urls) == len(targets) == 1 and re.fullmatch(r"[\w./-]+", targets[0]):
        return targets[0]
    return None


def _run_errors(args, manifests, workdir):
    separators = []
    if args.startswith("["):
        words = json.loads(args)
        if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
            raise ValueError("invalid RUN argument list")
        if len(words) >= 3 and Path(words[0]).name in ("bash", "sh") and words[1] in ("-c", "-lc"):
            return _run_errors(words[2], manifests, workdir)
        segments = [words]
    else:
        lexer = shlex.shlex(args, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        segments = [[]]
        for word in lexer:
            if word and set(word) <= set(";&|"):
                separators.append(word)
                segments.append([])
            else:
                segments[-1].append(word)
    errors = []
    pending = set()
    downloads = False
    checksum_pipes = set()
    for i, words in enumerate(segments):
        if not words:
            continue
        if any(Path(w).name in ("curl", "wget") for w in words):
            downloads = True
            target = _download_target(words)
            if target is None:
                errors.append("curl/wget needs one literal URL and an explicit output filename")
            else:
                target = posixpath.normpath(posixpath.join(workdir, target))
                if target in pending:
                    errors.append(f"download overwrites unchecked {target}")
                pending.add(target)
        if (words in (["sha256sum", "-c", "-"], ["sha256sum", "--check", "-"]) and
                i and separators[i-1] == "|" and segments[i-1][:1] == ["echo"]):
            check = re.fullmatch(r"([0-9a-fA-F]{64}) [ *]([\w./-]+)", " ".join(segments[i-1][1:]))
            if check:
                checksum_pipes.add(i-1)
                pending.discard(posixpath.normpath(posixpath.join(workdir, check[2])))
        if words[0] == "cd" and len(words) == 2:
            workdir = posixpath.normpath(posixpath.join(workdir, words[1]))
        if words[0] in ("sh", "bash") and len(words) == 3 and words[1] in ("-c", "-lc"):
            errors.extend(_run_errors(words[2], manifests, workdir))
        else:
            errors.extend(_install_errors(words, manifests, workdir))
    if pending:
        errors.append("missing SHA-256 check for " + ", ".join(sorted(pending)))
    if downloads and any(sep != "&&" and i not in checksum_pipes for i, sep in enumerate(separators)):
        errors.append("download and checksum commands must succeed in an && chain")
    return errors


def lint(dockerfile_text: str, context: Path | None = None) -> list[str]:
    """Strict apt rule: every explicit apt install operand needs pkg=version, even on snapshots."""
    errors = []
    if dockerfile_text.rstrip().endswith("\\"):
        errors.append("unfinished Dockerfile line continuation before the required tail")
    refs = bases(dockerfile_text)
    if not refs:
        errors.append("missing digest-pinned FROM")
    user = "root"
    stages = set()
    workdir = "."
    manifests = {}
    if re.search(r"(?im)^\s*#\s*(?:syntax\s*=\s*(?!docker/dockerfile:)|escape\s*=\s*`)", dockerfile_text):
        errors.append("custom Dockerfile frontends/escapes are unsupported")
    for op, args in _instructions(dockerfile_text):
        if re.search(r"(?:^|[^\w.-])(?:tests|solution|generator)(?:/|\b)", args, re.I):
            errors.append(f"{op}: references private task paths")
        if op == "FROM":
            user = "root"
            workdir = "."
            manifests = {}
            parts = args.split()
            if parts and parts[0] == "--platform=linux/amd64":
                parts.pop(0)
            if not parts or not valid_base(parts[0]) or "@" not in parts[0] or not (
                    len(parts) == 1 or len(parts) == 3 and parts[1].lower() == "as"):
                errors.append(f"FROM must pin a non-latest base by digest: {args}")
            if len(parts) == 3:
                stages.add(parts[2])
        elif op == "WORKDIR":
            workdir = posixpath.normpath(posixpath.join(workdir, args))
        elif op == "USER":
            user = args
        elif op in {"ENTRYPOINT", "CMD", "EXPOSE", "VOLUME", "ONBUILD", "SHELL"}:
            errors.append(f"{op} is incompatible with the task harness")
        elif op == "ADD":
            if not re.fullmatch(r"--checksum=sha256:[\da-fA-F]{64}\s+https?://\S+\s+\S+", args):
                errors.append("ADD requires a remote URL and SHA256 --checksum")
        elif op == "COPY":
            if re.search(r"""(?:^|[\s/"'])data(?:[/\s"']|$)""", args, re.I):
                errors.append("COPY data is pipeline-owned")
            try:
                if args.startswith("--from="):
                    if args.split()[0][7:] not in stages:
                        errors.append("COPY --from must reference a digest-pinned build stage")
                else:
                    for name, dest in _context_sources(f"COPY {args}").items():
                        if context is not None:
                            path = posixpath.normpath(posixpath.join(workdir, dest))
                            manifests[path] = Path(context) / name
            except ValueError as e:
                errors.append(str(e))
        elif op == "RUN":
            try:
                for opt in shlex.split(args) if args.strip() else []:
                    if opt.lower().startswith("--mount"):
                        fields = opt.lower().partition("=")[2].split(",")
                        types = [f for f in fields if f.startswith("type=")]
                        if types not in (["type=cache"], ["type=tmpfs"]) or any(
                                f.partition("=")[0] in ("from", "source", "src") for f in fields):
                            errors.append("RUN mount may expose build context")
                errors.extend(_run_errors(args, manifests, workdir))
            except ValueError as e:
                errors.append(f"RUN: {e}")
    if user not in {"root", "0", "0:0", "root:root"}:
        errors.append(f"final USER must be root (got {user})")
    return errors


def _resolve_tag(cfg, ref):
    builder = cfg["taskimage"].get("builder")
    cmd = ["docker", "buildx", "imagetools", "inspect"]
    if builder:
        cmd.extend(["--builder", builder])

    def inspect_json(target, field):
        output = _exec(cfg, [*cmd, "--format", "{{json ." + field + "}}", target], full=True)["out"]
        value = json.loads(output)
        if not isinstance(value, dict):
            raise ValueError(f"invalid {field} returned for {target}")
        return value

    manifest = inspect_json(ref, "Manifest")
    if "manifests" in manifest:
        entries = manifest["manifests"]
        if not isinstance(entries, list):
            raise ValueError(f"invalid manifest list for {ref}")
        digest = next((entry.get("digest") for entry in entries if isinstance(entry, dict)
                       and isinstance(platform := entry.get("platform"), dict)
                       and (platform.get("os"), platform.get("architecture")) == ("linux", "amd64")), None)
    else:
        # A single-image manifest's config.digest identifies its config blob, not the manifest.
        digest = manifest.get("digest")
        if not digest and "layers" in manifest:
            header = _exec(cfg, [*cmd, ref], full=True)["out"]
            match = re.search(rf"^Digest:\s+({DIGEST})\s*$", header, re.M)
            digest = match[1] if match else None
        if isinstance(digest, str) and re.fullmatch(DIGEST, digest):
            platform = manifest.get("platform") or inspect_json(ref + "@" + digest, "Image")
            if not isinstance(platform, dict) or (platform.get("os"), platform.get("architecture")) != ("linux", "amd64"):
                raise ValueError(f"single manifest for {ref} is not linux/amd64")
    if not isinstance(digest, str) or not re.fullmatch(DIGEST, digest):
        raise ValueError(f"no linux/amd64 manifest digest for {ref}")
    return ref + "@" + digest


def _pin_bases(cfg, item_dir, spec):
    env = item_dir / "task/environment"
    errors = context_errors(env)
    if errors:
        raise ValueError("; ".join(errors))
    path = env / "Dockerfile"
    lines = path.read_text().splitlines(keepends=True)
    stages = []
    for start, end, line in _logical_lines(lines):
        parts = line.split(None, 1)
        if parts[0].upper() == "FROM":
            match = FROM_ARGS.fullmatch(parts[1]) if len(parts) == 2 else None
            if not match or not valid_base(match["ref"]):
                raise ValueError(f"FROM needs a specific tag or digest: {line}")
            stages.append((start, end, match))
    image = spec["image"]
    if not stages:
        raise ValueError("missing FROM")
    final = stages[-1][2]["ref"]
    pinned = image["base"]
    source = image.get("base_resolved_from")
    resolved = {}
    if source and pinned.startswith(source + "@") and valid_base(pinned):
        # A prior attempt may have committed the spec immediately before its Dockerfile rename.
        resolved[source] = pinned
    if resolved.get(final, final) != pinned:
        raise ValueError(f"spec.image.base must match the final FROM {final}")
    changed = False
    for start, end, match in reversed(stages):
        ref = match["ref"]
        if "@" in ref:
            continue
        if ref not in resolved:
            resolved[ref] = _resolve_tag(cfg, ref)
        lines[start:end] = [f"FROM {match['platform'] or ''}{resolved[ref]}{match['alias'] or ''}\n"]
        changed = True
    if changed:
        if final in resolved:
            image.update(base=resolved[final], base_resolved_from=final)
            write_json(item_dir / "_spec/spec.json", spec)
        # Keep the temporary sibling outside the strict build context in case of interruption.
        tmp = item_dir / ".Dockerfile.pinned.tmp"
        tmp.write_text("".join(lines))
        tmp.replace(path)


def assemble(cfg, item_dir, dest):
    from .package import OFFLINE_INDEX
    env = item_dir / "task/environment"
    errors = context_errors(env)
    if errors:
        raise ValueError("; ".join(errors))
    text = (env / "Dockerfile").read_text()
    errors = lint(text, env)
    if errors:
        raise ValueError("; ".join(errors))
    dest.mkdir(parents=True, exist_ok=True)
    for f in env.iterdir():
        if f.name != "Dockerfile":
            shutil.copyfile(f, dest / f.name)
    offline = (cfg["package"].get("network") or {}).get("disable_index")
    (dest / "Dockerfile").write_text(text.rstrip() + "\n" + REQUIRED_TAIL + (OFFLINE_INDEX if offline else ""))
    (dest / "data").mkdir(exist_ok=True)
    return bases(text)[-1]


def context_hash(root, *, empty_data=False, build_args=None):
    h = hashlib.sha256()
    if build_args:
        h.update(json.dumps(build_args, sort_keys=True, separators=(",", ":")).encode() + b"\0")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if empty_data and rel.parts[0] == "data":
            continue
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            raise ValueError(f"invalid build-context entry {rel}")
        h.update(str(rel).encode() + b"\0")
        h.update(str(p.stat().st_mode & 0o7777).encode() + b"\0")
        if p.is_file():
            with p.open("rb") as f:
                h.update(hashlib.file_digest(f, "sha256").digest())
    return h.hexdigest()[:20]


def catalog_path(cfg):
    return Path(cfg["taskimage"].get("catalog") or Path(cfg["run"]["root"]).resolve().parent / "taskimage-catalog.jsonl")


def base_catalog(path):
    path = Path(path)
    entries = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
    return sorted(entries, key=lambda row: (-len(row["tasks"]), row["base"]))


def record_base(path, base, item_id, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = {row["base"]: row for row in base_catalog(path)}
        tasks = set(entries.get(base, {}).get("tasks", [])) | {item_id}
        entries[base] = dict(base=base, tasks=sorted(tasks), size=result["size"], python=result["python"],
                             inventory_hash=result["inventory_hash"], measured_task=item_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(row) + "\n" for row in entries.values()))
        tmp.replace(path)


def is_task(record):
    return record.get("image_kind") in ("task", "authored") or bool(record.get("image_tag"))


def author_images(cfg, state):
    if state.get("image_kind"):
        return is_task(state)
    return enabled(cfg)


def image_for(cfg, item_dir):
    state = read_json(Path(item_dir) / "state.json", {})
    if is_task(state):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", state.get("image_digest") or ""):
            raise ValueError("task image has no verified S6I image ID")
        return state["image_digest"]
    return cfg["sandbox"]["image"]


def run(cfg, item_dir, *args, **kwargs):
    image = image_for(cfg, item_dir)
    if image != cfg["sandbox"]["image"]:
        kwargs["image"] = image
    return sandbox.run(cfg, *args, **kwargs)


def _host(cfg):
    if cfg["sandbox"]["backend"] == "docker":
        return None
    host = cfg["taskimage"].get("host") or cfg["sandbox"].get("host")
    if not host:
        raise ValueError("missing taskimage.host / sandbox.host")
    return host


def _exec(cfg, argv, timeout=60, *, check=True, full=False):
    host = _host(cfg)
    if argv[0] == "docker":
        if host and argv[1] in ("build", "buildx", "run"):
            # Bound the job on the build host even if the SSH connection drops.
            argv = ["timeout", "-k", "30", str(int(timeout)), *argv]
            timeout += 60
        dc = cfg["taskimage"].get("docker_config") or cfg["sandbox"].get("docker_config")
        if dc:
            if host and str(dc).startswith(("$HOME/", "~/")):
                suffix = str(dc).split("/", 1)[1]
                # Expand HOME on the build host, with no evaluation of configured path contents.
                line = 'DOCKER_CONFIG="$HOME"/' + shlex.quote(suffix) + " " + shlex.join(argv)
                argv = sandbox._ssh(host) + [line]
                host = None
            else:
                if not host and str(dc).startswith(("$HOME/", "~/")):
                    dc = Path.home() / str(dc).split("/", 1)[1]
                argv = ["env", "DOCKER_CONFIG=" + str(dc), *argv]
    if host:
        argv = sandbox._ssh(host) + [shlex.join(argv)]
    result = sandbox.sh(argv, timeout, output_limit=None) if full else sandbox.sh(argv, timeout)
    if check and result["rc"]:
        raise ValueError(result["err"] or result["out"] or f"{argv[0]} exited {result['rc']}")
    return result


def _namespace(cfg):
    # Layouts and scratch contexts belong to this run on a shared build host.
    run_id = hashlib.sha256(str(Path(cfg["run"]["root"]).resolve()).encode()).hexdigest()[:16]
    parent = cfg["sandbox"].get("remote_dir") if _host(cfg) else cfg["run"]["root"]
    return str(Path(parent or ".") / "taskimages" / run_id)


@contextlib.contextmanager
def _context(cfg, src):
    host = _host(cfg)
    if not host:
        yield str(src.resolve())
        return
    rd = _namespace(cfg) + "/ctx-" + uuid.uuid4().hex
    _exec(cfg, ["mkdir", "-p", rd])
    try:
        result = sandbox.sh(["rsync", "-a", "-e", shlex.join(sandbox._ssh(host)[:-1]),
                             str(src) + "/", host + ":" + shlex.quote(rd + "/")], cfg["taskimage"]["build_timeout_s"])
        if result["rc"]:
            raise ValueError("rsync: " + result["err"])
        yield rd
    finally:
        _exec(cfg, ["rm", "-rf", "--", rd], check=False)


def _build_args(cfg):
    # Build-time env overrides, e.g. {"HTTPS_PROXY": ""} when the builder's predefined proxy is down but the
    # build host has direct egress. Predefined proxy args are not part of BuildKit's cache key.
    return [a for k, v in sorted((cfg["taskimage"].get("build_args") or {}).items())
            for a in ("--build-arg", f"{k}={v}")]


def _buildx(cfg):
    builder = cfg["taskimage"].get("builder")
    return ["docker", "buildx", "build", *(["--builder", builder] if builder else []), "--platform", "linux/amd64",
            *_build_args(cfg)]


def _image_id(cfg, tag):
    try:
        result = _exec(cfg, ["docker", "image", "inspect", "--format", "{{.Id}}", tag], check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise ImageUnavailable(f"could not inspect {tag}: {e}") from e
    if result["rc"] == 0:
        digest = result["out"].strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ImageUnavailable(f"invalid image ID returned for {tag}")
        return digest
    error = (result.get("err") or result.get("out") or "").strip()
    if "no such image" in error.lower() or "no such object" in error.lower():
        return None
    raise ImageUnavailable(f"could not inspect {tag}: {error or result['rc']}")


class ImageUnavailable(ValueError):
    """An execution/configuration issue; retrying author code cannot fix it."""


def _sandbox_host_cfg(cfg):
    sb = cfg["sandbox"]
    if sb["backend"] == "docker":
        return cfg
    if sb["backend"] != "ssh" or not sb.get("host") or not sb.get("remote_dir"):
        raise ImageUnavailable("task image requires sandbox.backend=docker or ssh with sandbox.host and remote_dir")
    return dict(cfg, taskimage=dict(cfg["taskimage"], host=sb["host"], docker_config=sb.get("docker_config")))


def harbor_tag(item_dir, digest):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest or ""):
        raise ImageUnavailable("missing verified S6I image ID")
    return "tbsgen-task:" + _key(item_dir, digest[7:])


def check_tag(cfg, tag, expected):
    actual = _image_id(cfg, tag)
    if actual and actual != expected:
        raise ImageUnavailable(f"{tag}: expected S6I ID {expected}, found {actual}")
    return bool(actual)


def needs_restore(cfg, item_dir, state):
    sandbox_cfg = _sandbox_host_cfg(cfg)
    digest, tag = state.get("image_digest"), state.get("image_tag")
    if not tag or not digest:
        return True
    if not check_tag(cfg, tag, digest):
        return True
    if _host(cfg) != _host(sandbox_cfg) and not check_tag(sandbox_cfg, tag, digest):
        return True
    if not state.get("image_manifest"):   # pre-OCI S6I receipt needs the new runtime floor + export
        return True
    return not check_tag(cfg, harbor_tag(item_dir, digest), digest)


def package_identity(item_dir):
    state = read_json(item_dir / "state.json", {})
    prov = read_json(item_dir / "pkg/provenance.json", {})
    if not (is_task(state) or is_task(prov)):
        return None
    for k in ("image_digest", "image_tag", "image_manifest"):
        if not state.get(k) or state[k] != prov.get(k):
            raise ImageUnavailable(f"pkg {k} differs from S6I; repackage the verified item")
    return prov


def package_matches(item_dir):
    prov = package_identity(item_dir)
    return bool(prov and (item_dir / "pkg/environment/Dockerfile").read_text() ==
                data_recipe(harbor_tag(item_dir, prov["image_digest"])))


def data_recipe(base):
    return f"FROM {base}\nCOPY data /app/data\n"


def _oci(cfg, action, path, digest, *args):
    script = Path(__file__).with_name("taskimage_oci.py").read_text()
    return _exec(cfg, ["python3", "-c", script, action, path, digest, *args],
                 cfg["taskimage"]["build_timeout_s"])["out"].strip()


def _install_layout(cfg, tmp, dest):
    result = _exec(cfg, ["python3", "-c", "import os,sys; os.rename(*sys.argv[1:])", tmp, dest], check=False)
    if result["rc"]:  # Concurrent build of the same key; the caller verifies the winning layout.
        _exec(cfg, ["test", "-s", dest + "/index.json"])


def _base_layout(cfg, item_dir, digest):
    return _namespace(cfg) + "/oci/" + _key(item_dir, digest.removeprefix("sha256:"))


def _export_base(cfg, item_dir, digest):
    dest = _base_layout(cfg, item_dir, digest)
    if _exec(cfg, ["test", "-s", dest + "/index.json"], check=False)["rc"]:
        tmp = dest + ".tmp-" + uuid.uuid4().hex
        _exec(cfg, ["mkdir", "-p", tmp])
        try:
            archive = tmp + "/image.tar"
            _exec(cfg, ["docker", "image", "save", "--output", archive, digest], cfg["taskimage"]["build_timeout_s"])
            _oci(cfg, "export", tmp + "/layout", digest, archive)
            _install_layout(cfg, tmp + "/layout", dest)
        finally:
            _exec(cfg, ["rm", "-rf", "--", tmp], check=False)
    return _oci(cfg, "verify", dest, digest)


def _key(item_dir, digest):
    key = item_dir.name + "-" + digest
    if not re.fullmatch(r"[a-zA-Z0-9_][\w.-]{0,110}", key, re.ASCII):
        raise ValueError(f"invalid image tag for item {item_dir.name!r}")
    return key


def restore_tag(cfg, item_dir, state):
    """Recover the exact prior build from the daemon or the run's OCI export before trying a rebuild."""
    digest, tag = state.get("image_digest"), state.get("image_tag")
    if not digest or not tag or check_tag(cfg, tag, digest):
        return
    if not check_tag(cfg, digest, digest):
        base = _base_layout(cfg, item_dir, digest)
        if not state.get("image_manifest") or _exec(cfg, ["test", "-s", base + "/index.json"], check=False)["rc"]:
            return
        archive = base + ".restore-" + uuid.uuid4().hex + ".tar"
        try:
            _oci(cfg, "archive", base, digest, state["image_manifest"], archive)
            _exec(cfg, ["docker", "image", "load", "--input", archive], cfg["taskimage"]["build_timeout_s"])
        finally:
            _exec(cfg, ["rm", "-f", "--", archive], check=False)
        if not check_tag(cfg, digest, digest):
            raise ImageUnavailable(f"restored OCI layout did not load verified ID {digest}")
    _exec(cfg, ["docker", "tag", digest, tag])


def _load_image(cfg, ctx, tag):
    digest = _image_id(cfg, tag)
    if not digest:
        with _context(cfg, ctx) as path:
            if _host(cfg) or cfg["taskimage"].get("builder"):
                cmd = _buildx(cfg) + ["--load"]
            else:
                cmd = ["docker", "build", "--platform", "linux/amd64", *_build_args(cfg)]
            _exec(cfg, cmd + ["-t", tag, path], cfg["taskimage"]["build_timeout_s"])
        digest = _image_id(cfg, tag)
    if not digest:
        raise ValueError("built image has no local image ID")
    return digest


def _sandbox_copy(cfg, tag, digest):
    host = _host(cfg)
    target = cfg["sandbox"].get("host")
    if not host or host == target:
        return
    if not target:
        raise ValueError("sandbox.host is required for remote execution")
    dest = dict(cfg, taskimage=dict(cfg["taskimage"], host=target, docker_config=cfg["sandbox"].get("docker_config")))
    if _image_id(dest, tag) != digest:
        # SSH endpoints need not trust each other. Stream the image between their daemons via the controller.
        sender = sandbox._ssh(host) + [shlex.join(["docker", "save", tag])]
        receiver = sandbox._ssh(target) + ["docker load"]
        r = sandbox.sh(["bash", "-o", "pipefail", "-c", shlex.join(sender) + " | " + shlex.join(receiver)],
                       cfg["taskimage"]["build_timeout_s"])
        if r["rc"] or _image_id(dest, tag) != digest:
            raise ValueError(f"failed to load {tag} on {target}: {r['err']}")


def inspect(cfg, tag, imports=()):
    """Probe the actual image with its inherited entrypoint, with the verifier's network policy."""
    from .runner import metrics_helper_text, TEST_SH
    meta = _exec(cfg, ["docker", "image", "inspect", "--format", "{{.Os}}/{{.Architecture}} {{.Size}}", tag])
    arch, size = meta["out"].split()
    probe = Path(__file__).with_name("taskimage_probe.py").read_text()
    payload = json.dumps(dict(imports=imports, helper=metrics_helper_text(cfg, {"image_kind": "task"}), script=TEST_SH))
    cmd = "python -c " + shlex.quote(probe) + " " + shlex.quote(payload)
    name = "tbsgen-inspect-" + uuid.uuid4().hex
    try:
        res = _exec(cfg, ["docker", "run", "--rm", "--name", name, "--pull=never", "--network", "none",
                          "--pids-limit", "256", "--memory", cfg["sandbox"]["memory"],
                          "--cpus", str(cfg["sandbox"]["cpus"]), tag, "bash", "-lc", cmd], 300, full=True)
    except (OSError, ValueError):
        _exec(cfg, ["docker", "rm", "-f", name], check=False)
        raise
    info = json.loads(res["out"])
    return dict(info, arch=arch, size=int(size), probe_log=res.get("err", "")[-2500:])


def build(cfg, item_dir, *, state=None):
    gates = dict.fromkeys(GATES)
    result = dict(ok=False, gates=gates)
    gate = "resolve_base"
    _sandbox_host_cfg(cfg)
    if state is None:
        state = read_json(item_dir / "state.json", {})
    try:
        spec = read_json(item_dir / "_spec/spec.json", {})
        _pin_bases(cfg, item_dir, spec)
        base_info = {k: spec["image"][k] for k in ("base", "base_resolved_from") if k in spec["image"]}
        result.update(base_info)
        if state and any(state.get(key) != value for key, value in base_info.items()):
            state.update(base_info)
            write_json(item_dir / "state.json", state)
        gates[gate] = True
        gate = "lint"
        with tempfile.TemporaryDirectory() as td:
            ctx = Path(td)
            assemble(cfg, item_dir, ctx)
            gates[gate] = True
            gate = "built"
            inputs = context_hash(ctx, empty_data=True, build_args=cfg["taskimage"].get("build_args"))
            tag = "tbsgen-task:" + _key(item_dir, inputs)
            if state.get("image_resume_stage"):
                restore_tag(cfg, item_dir, state)
            if state.get("image_resume_stage") and state.get("image_tag") and state["image_tag"] != tag:
                # Older context hashes excluded build args. Reuse their exact verified ID if still available.
                if check_tag(cfg, state["image_tag"], state["image_digest"]):
                    tag = state["image_tag"]
                else:
                    raise ImageUnavailable("S6I build inputs changed and the original image is missing")
            # Include failed and superseded builds in this run's GC inventory.
            claims = item_dir / "taskimage-assets"
            claims.mkdir(exist_ok=True)
            (claims / tag.split(":")[1]).touch()
            result.update(image_tag=tag)
            result["image_digest"] = _load_image(cfg, ctx, tag)
            if (state.get("image_digest") and state.get("image_resume_stage")
                    and state["image_digest"] != result["image_digest"]):
                raise ImageUnavailable("restored S6I image ID differs from the verified item")
            gates[gate] = True
        gate = "verifier_floor"
        info = inspect(cfg, result["image_digest"], spec["image"]["packages"])
        result.update(info)
        gates.update(verifier_floor=not info["floor_errors"], offline_imports=not info["import_errors"],
                     arch=info["arch"] == "linux/amd64", size=info["size"] <= cfg["taskimage"]["max_gb"] * 1e9)
        signals = prerelease.signals(read_json(item_dir / "S1_candidate.json", {}),
                                     prerelease.with_author_code(item_dir, spec))
        hits = prerelease.inventory_hits(info["inventory"], signals)
        gates["readymade_image"] = not hits
        result["inventory_hash"] = hashlib.sha256(json.dumps(info["inventory"], sort_keys=True).encode()).hexdigest()
        for gate, details in (("readymade_image", hits), ("verifier_floor", info["floor_errors"]),
                              ("offline_imports", info["import_errors"]), ("arch", info["arch"]), ("size", info["size"])):
            if not gates[gate]:
                raise ValueError(str(details))
        gate = "built"
        digest = result["image_digest"]
        result["image_manifest"] = _export_base(cfg, item_dir, digest)
        alias = harbor_tag(item_dir, digest)
        (claims / alias.split(":")[1]).touch()
        if not check_tag(cfg, alias, digest):
            _exec(cfg, ["docker", "tag", digest, alias])
        if not check_tag(cfg, alias, digest) or not check_tag(cfg, tag, digest):
            raise ImageUnavailable("verified image disappeared during S6I")
        _sandbox_copy(cfg, tag, digest)
        result["ok"] = True
    except ImageUnavailable:
        raise
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        gates[gate] = False
        details = str(e)[-3500:]
        if gate in ("offline_imports", "verifier_floor") and result.get("probe_log"):
            details += "\n" + result["probe_log"]
        result.update(failed_gate=gate, error=details)
    return result


def publish(cfg, item_dir):
    """Append only the packaged data to the exact OCI manifest exported at S6I."""
    prefix = cfg["taskimage"].get("registry_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("set taskimage.registry_prefix before publishing a task image")
    ctx = item_dir / "pkg/environment"
    prov = package_identity(item_dir)
    if not prov:
        raise ValueError("publish requires an S6I image")
    digest, manifest = prov["image_digest"], prov["image_manifest"]
    if (ctx / "Dockerfile").read_text() != data_recipe(harbor_tag(item_dir, digest)):
        raise ValueError("packaged Dockerfile differs from the verified data recipe")
    base = _base_layout(cfg, item_dir, digest)
    if _export_base(cfg, item_dir, digest) != manifest:
        raise ImageUnavailable("OCI base differs from verified S6I manifest")
    key = _key(item_dir, context_hash(ctx, build_args=cfg["taskimage"].get("build_args")))
    parent = _namespace(cfg) + "/oci"
    dest = parent + "/" + key
    if _exec(cfg, ["test", "-s", dest + "/index.json"], check=False)["rc"]:
        _exec(cfg, ["mkdir", "-p", parent])
        tmp = dest + ".tmp-" + uuid.uuid4().hex
        try:
            with tempfile.TemporaryDirectory() as td:
                clean = Path(td)
                shutil.copytree(ctx / "data", clean / "data")
                (clean / "Dockerfile").write_text(data_recipe("taskbase"))
                absolute = _exec(cfg, ["realpath", base])["out"].strip()
                with _context(cfg, clean) as context:
                    _exec(cfg, _buildx(cfg) + ["--build-context", f"taskbase=oci-layout://{absolute}@{manifest}",
                                              "--provenance=false", "--sbom=false", "--output",
                                              "type=oci,tar=false,dest=" + tmp, context],
                          cfg["taskimage"]["build_timeout_s"])
            _oci(cfg, "verify", base, digest, manifest, tmp)
            _install_layout(cfg, tmp, dest)
        finally:
            _exec(cfg, ["rm", "-rf", "--", tmp], check=False)
    published = _oci(cfg, "verify", base, digest, manifest, dest)
    # OCI blobs include the compressed task data layer, which was absent at S6I.
    size_probe = ("from pathlib import Path; import sys; "
                  "print(sum(p.stat().st_size for p in Path(sys.argv[1], 'blobs').rglob('*') if p.is_file()))")
    size = int(_exec(cfg, ["python3", "-c", size_probe, dest])["out"])
    if size > cfg["taskimage"]["max_gb"] * 1e9:
        raise ValueError(f"full OCI layout exceeds taskimage.max_gb: {size} bytes")
    host = _host(cfg)
    return dict(TBSGEN_IMAGE_OCI=(host + ":" if host else "") + dest,
                TBSGEN_IMAGE_DIGEST=published, TBSGEN_IMAGE_TAG=prefix + "-" + key)


def gc_selection(states, images, layouts, keep_days, now):
    """Use exact item IDs and content-hash names. Ignore unknown runs, tags and layout directories."""
    eligible = {item for item, st in states.items()
                if st.get("status") == "dropped" or
                st.get("status") == "done" and st.get("stage") == "S10" and st.get("acr_push") or
                (st.get("image_resume_stage") or st.get("stage")) not in ("S9", "S9R")
                and now - st.get("updated", now) > keep_days * 86400}

    def owned(name):
        item, _, suffix = name.rpartition("-")
        return item in eligible and bool(re.fullmatch(r"(?:[a-f0-9]{20}|[a-f0-9]{64})", suffix))

    return dict(images=sorted(tag for tag in images if tag.startswith("tbsgen-task:") and owned(tag.removeprefix("tbsgen-task:"))),
                layouts=sorted(name for name in layouts if owned(name)))


def _claimed_tags(item_dir, state):
    tags = {"tbsgen-task:" + p.name for p in (item_dir / "taskimage-assets").glob("*")}
    if state.get("image_tag"):
        tags.add(state["image_tag"])
    return tags


def gc(cfg, dry_run=False, item_id=None, *, step_locked=False):
    root = Path(cfg["run"]["root"])
    states = {}
    claims = set()
    for path in (root / "items").glob("*/state.json"):
        if item_id is None or path.parent.name == item_id:
            state = read_json(path, {})
            receipt = read_json(path.parent / "calib/image-push.json", {})
            prepared = read_json(path.parent / "calib/prepare.json", {})
            state["acr_push"] = (receipt.get("digest") and prepared.get("ok") and
                                 prepared.get("pkg_sha") == receipt.get("pkg_sha"))
            claims.update(_claimed_tags(path.parent, state))
            states[path.parent.name] = state
    daemons = [cfg]
    if _host(cfg) and _host(cfg) != cfg["sandbox"].get("host"):
        daemons.append(dict(cfg, taskimage=dict(cfg["taskimage"], host=cfg["sandbox"]["host"],
                                                docker_config=cfg["sandbox"].get("docker_config"))))
    images = []
    for daemon in daemons:
        listing = _exec(daemon, ["docker", "image", "ls", "tbsgen-task", "--format", "{{.Repository}}:{{.Tag}}"],
                        full=True)
        images.append(set(listing["out"].splitlines()))
    parent = _namespace(cfg) + "/oci"
    layouts = []
    if not _exec(cfg, ["test", "-d", parent], check=False)["rc"]:
        layouts = _exec(cfg, ["ls", "-1", parent], full=True)["out"].splitlines()
    candidates = claims.intersection(set.union(*images))
    selected = gc_selection(states, candidates, layouts, cfg["taskimage"]["keep_days"], time.time())
    # Sibling runs can share a content-addressed image. A referenced tag stays live until both runs release it.
    if selected["images"]:
        for path in root.resolve().parent.glob("*/items/*/state.json"):
            if path.parents[2] == root.resolve():
                continue
            other = read_json(path, {})
            if other.get("status") not in ("done", "dropped"):
                protected = _claimed_tags(path.parent, other)
                selected["images"] = [tag for tag in selected["images"] if tag not in protected]
    if dry_run:
        return selected
    from .stages_b import item_lock
    from .state import Item
    removed = dict(images=[], layouts=[])
    for kind, entries in selected.items():
        for entry in entries:
            item = entry.removeprefix("tbsgen-task:").rpartition("-")[0]
            it = Item(root, item)
            # S10 owns the step lock for its item. A standalone collector skips busy workers.
            lock = contextlib.nullcontext(True) if step_locked and item == item_id else it.step_lock()
            with lock as held:
                if not held:
                    continue
                with item_lock(it):
                    if it.load().state.get("updated") != states[item].get("updated"):
                        continue  # An admission, re-author or resume raced the listing.
                    if kind == "layouts":
                        _exec(cfg, ["rm", "-rf", "--", parent + "/" + entry])
                    else:
                        if entry == it.state.get("image_tag"):
                            write_json(it.p("taskimage-evicted.json"), {"image_tag": entry})
                        for daemon, present in zip(daemons, images):
                            if entry in present:
                                _exec(daemon, ["docker", "image", "rm", entry])
                    removed[kind].append(entry)
    return removed
