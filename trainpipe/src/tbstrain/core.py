"""Local content-addressed stage execution and durable decision ledgers."""

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import tomllib
from contextlib import contextmanager
from pathlib import Path


def code_version():
    """Hash of the installed tbstrain sources (rubrics included): a code change invalidates every stage cache.
    The judge's per-request cache is separate and keyed by prompt + model + rubric, so judge calls are not repeated."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        root = Path(__file__).parent
        h = hashlib.sha256()
        for f in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".md", ".json", ".toml")):
            h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes() + b"\0")
        _CODE_VERSION = h.hexdigest()[:16]
    return _CODE_VERSION


_CODE_VERSION = None


def dumps(value, *, sort_keys=True):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=sort_keys, separators=(",", ":"))


def loads(data):
    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    return json.loads(data, parse_constant=invalid, parse_float=finite)


def loads_source(data):
    """Pipeline / Harbor inputs: verifier metrics legitimately record a failed value as Infinity or NaN.
    Non-finite numbers become None (JSON null) so nothing downstream can write them back out."""
    return json.loads(data, parse_constant=lambda value: None,
                      parse_float=lambda value: number if math.isfinite(number := float(value)) else None)


def read(path, default=None):
    return loads_source(path.read_text(encoding="utf-8")) if path.is_file() else default


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield loads_source(line)


def write_lines(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(dumps(record) + "\n")


def atomic(path, value, *, sort_keys=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(dumps(value, sort_keys=sort_keys) + "\n")
    os.replace(tmp.name, path)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hash(path):
    if path.is_file():
        return file_hash(path)
    files = {}
    for child in sorted(path.rglob("*")):
        relative = str(child.relative_to(path))
        if child.is_symlink():
            files[relative] = ["link", os.readlink(child)]
        elif child.is_dir():
            files[relative] = ["directory", child.stat().st_mode & 0o777]
        elif child.is_file():
            files[relative] = ["file", child.stat().st_mode & 0o777, file_hash(child)]
    return digest(files)


class Context:
    def __init__(self, config):
        self.base = config.absolute().parent if config else Path.cwd()
        self.cfg = tomllib.loads(config.read_text("utf-8")) if config else {}
        self.out = self.path(self.cfg.get("out", "state/trainpipe"))
        self.parents = {}
        self.hashes = {}
        self.transport = None
        self.work = None

    def path(self, value):
        return (self.base / Path(value).expanduser()).absolute()

    def content_hash(self, path):
        if not path.is_file():
            return None
        stat = path.stat()
        key = (path, stat.st_ino, stat.st_size, stat.st_ctime_ns)
        if key not in self.hashes:
            self.hashes[key] = file_hash(path)
        return self.hashes[key]

    def check_sources(self, table):
        for row in self.rows(table):
            for path, expected in row.get("source_hashes", {}).items():
                if self.content_hash(Path(path)) != expected:
                    raise ValueError(f"input changed since ingest: {path}. Run ingest again.")

    def rows(self, name):
        return lines(self.out / (name + ".jsonl"))

    def emit(self, name, records):
        write_lines(self.work / (name + ".jsonl"), records)

    @contextmanager
    def lock(self):
        self.out.mkdir(parents=True, exist_ok=True)
        # Custom output paths may also live inside a public checkout.
        try:
            with (self.out / ".gitignore").open("x", encoding="utf-8") as guard:
                guard.write("*\n")
        except FileExistsError:
            pass
        with (self.out / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def stage(self, name, inputs, action):
        parents, self.parents = self.parents, {}
        key = digest([1, code_version(), name, inputs, parents])
        root = self.out / ".stages" / name
        root.mkdir(parents=True, exist_ok=True)
        dest = root / key
        receipt = read(dest / ".receipt")
        skipped = receipt is not None
        if not skipped:
            self.work = Path(tempfile.mkdtemp(dir=root))
            try:
                stats = action()
                ledger = self.work / (name + ".ledger.jsonl")
                if not ledger.is_file():
                    raise RuntimeError(f"missing decision ledger for {name}")
                receipt = dict(stage=name, version=1, code=code_version(), key=key, parents=parents, **stats)
                atomic(self.work / ".receipt", receipt)
                self.work.rename(dest)
            except Exception as exc:
                error = dict(stage=name, key=key, error=str(exc))
                failed_ledger = self.work / (name + ".ledger.jsonl")
                with failed_ledger.open("a", encoding="utf-8") as failure:
                    failure.write(dumps({**error, "reason": "stage_failed"}) + "\n")
                (self.out / "reports").mkdir(exist_ok=True)
                os.replace(failed_ledger, self.out / "reports" / (name + ".failed.jsonl"))
                atomic(self.out / "reports" / f"{name}.json", error)
                raise
            finally:
                if self.work.exists():
                    shutil.rmtree(self.work)
                self.work = None
        # All public names resolve through one generation pointer. Publishing two tables
        # separately would leave a mixed generation if the process stopped in between.
        pointer = root / "current"
        for artifact in dest.iterdir():
            if artifact.name == ".receipt":
                continue
            link = self.out / artifact.name
            target = os.path.relpath(pointer / artifact.name, self.out)
            if link.is_symlink() and os.readlink(link) == target:
                continue
            if link.exists() and not link.is_symlink():
                raise ValueError(f"output collision: {link}")
            tmp = root / (".link-" + artifact.name)
            tmp.unlink(missing_ok=True)
            tmp.symlink_to(target)
            os.replace(tmp, link)
        next_pointer = root / ".next"
        next_pointer.unlink(missing_ok=True)
        next_pointer.symlink_to(key)
        os.replace(next_pointer, pointer)
        atomic(self.out / "reports" / f"{name}.json", receipt)
        print(f"{name} {'cached' if skipped else 'done'} {key[:12]} {receipt.get('count', 0)} records")
        return receipt

    def dependency(self, stage):
        receipt = read(self.out / "reports" / f"{stage}.json")
        if not receipt or "error" in receipt:
            raise ValueError(f"run {stage} first")
        pointer = self.out / ".stages" / stage / "current"
        if pointer.is_symlink() and os.readlink(pointer) != receipt["key"]:
            raise ValueError(f"interrupted publication: rerun {stage}")
        for parent, expected in receipt.get("parents", {}).items():
            if self.dependency(parent) != expected:
                raise ValueError(f"run {stage} again: {parent} changed")
        self.parents[stage] = receipt["key"]
        return receipt["key"]
