"""Private wire profiles supplied by the operator."""

import hashlib
import json
from pathlib import Path
import re
import tempfile

from .core import file_hash, loads


class ProfileError(ValueError):
    """Configuration or evidence is insufficient to reconstruct the offered context."""


def required_profile(profile, harness, version=None):
    if profile is None:
        raise ProfileError(f"wire profile required for {harness}")
    if version is not None or harness == "claude-code":
        if not version or profile["version"] != version:
            raise ProfileError(f"{harness} wire profile version mismatch (trajectory {version!r})")
    return profile


def system_text(blocks):
    if not isinstance(blocks, list) or not blocks or not all(
        isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        for b in blocks
    ):
        raise ProfileError("system.json must contain text blocks")
    result = "\n\n".join(b["text"] for b in blocks)
    if not result.strip():
        raise ProfileError("empty system prompt")
    return result


def load_profile(directory, harness):
    manifest = loads((directory / "profile.json").read_bytes())
    if manifest.get("harness", harness) != harness or not manifest.get("version"):
        raise ProfileError("invalid profile harness or version")
    fields = {}
    for name in ("system", "tools"):
        path = directory / (name + ".json")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest.get("sha256", {}).get(path.name):
            raise ProfileError(f"profile checksum mismatch for {path.name}")
        fields[name] = loads(data)
    system_text(fields["system"])
    if not isinstance(fields["tools"], list):
        raise ProfileError("tools.json must contain a list")
    instruction = directory / "compaction.txt"
    return dict(**fields, version=manifest["version"],
                instruction=instruction.read_text("utf-8") if instruction.exists() else None)


def configured_profiles(ctx):
    export = ctx.cfg["export"]
    profiles = {name: load_profile(ctx.path(entry["dir"]), name)
                for name, entry in export.get("wire", {}).items()}
    cc = profiles.get("claude-code")
    if cc and not cc["instruction"] and export.get("compaction_instruction"):
        cc["instruction"] = ctx.path(export["compaction_instruction"]).read_text("utf-8")
    return profiles


def raw_fields(document):
    """Locate member value spans without reserializing capture strings or schemas."""
    decoder = json.JSONDecoder()
    pos = document.index("{") + 1
    while True:
        while document[pos].isspace():
            pos += 1
        if document[pos] == "}":
            return
        key, end = decoder.raw_decode(document, pos)
        pos = document.index(":", end) + 1
        while document[pos].isspace():
            pos += 1
        _, end = decoder.raw_decode(document, pos)
        yield key, document[pos:end]
        pos = end
        while document[pos].isspace():
            pos += 1
        if document[pos] == "}":
            return
        if document[pos] != ",":
            raise ProfileError("invalid capture object")
        pos += 1


def capture_profile(path, explicit_version=None):
    raw = path.read_bytes().decode("utf-8")
    capture = loads(raw)
    headers = {k.lower(): v for k, v in capture.get("headers", {}).items()}
    agent = re.search(r"\bclaude-cli/([^\s()]+)", headers.get("user-agent", ""))
    version = agent[1] if agent else capture.get("version") or explicit_version
    if not version or (explicit_version and version != explicit_version):
        raise ProfileError("missing or conflicting Claude Code version (use --version for bare bodies)")
    body = dict(raw_fields(raw)).get("body", raw)
    fields = {k: v.encode("utf-8") for k, v in raw_fields(body) if k in {"system", "tools"}}
    if set(fields) != {"system", "tools"}:
        raise ProfileError("missing capture system or tools")
    system_text(loads(fields["system"]))
    tools = loads(fields["tools"])
    if not isinstance(tools, list) or not all(
        isinstance(t, dict) and t.get("name") and isinstance(t.get("input_schema"), dict)
        and t["input_schema"] for t in tools
    ):
        raise ProfileError("invalid captured Anthropic schemas")
    return version, fields


def extract(paths, directory, version=None):
    baseline = capture_profile(paths[0], version)
    for path in paths[1:]:
        if capture_profile(path, version) != baseline:
            raise ProfileError(f"capture profile mismatch at {path.name}")
    directory = Path(directory)
    if directory.exists():
        raise ProfileError("profile output already exists")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as tmp:
        staging = Path(tmp) / "profile"
        staging.mkdir(mode=0o700)
        (staging / ".gitignore").write_text("*\n")
        hashes = {}
        for name, data in baseline[1].items():
            path = staging / (name + ".json")
            path.write_bytes(data)
            hashes[path.name] = file_hash(path)
        manifest = dict(harness="claude-code", version=baseline[0], sha256=hashes)
        (staging / "profile.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(directory)
    return manifest
