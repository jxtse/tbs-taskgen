"""Standalone build-host helper. Export a Docker save archive as OCI without resolving or running its image."""
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path

MEDIA = "application/vnd.oci.image."


def blob_path(root, digest):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"invalid OCI digest {digest!r}")
    return root / "blobs/sha256" / digest[7:]


def write_blob(root, stream, media):
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
            tmp.write(chunk)
        size = tmp.tell()
    digest = "sha256:" + h.hexdigest()
    os.replace(tmp.name, blob_path(root, digest))
    return dict(mediaType=MEDIA + media, size=size, digest=digest)


def export(archive, root, image_id):
    (root / "blobs/sha256").mkdir(parents=True)
    with tarfile.open(archive) as tar:
        entry, = json.load(tar.extractfile("manifest.json"))
        config = write_blob(root, tar.extractfile(entry["Config"]), "config.v1+json")
        if config["digest"] != image_id:
            raise ValueError("Docker archive config differs from S6I image ID")
        diffs = json.loads(blob_path(root, image_id).read_bytes())["rootfs"]["diff_ids"]
        if len(diffs) != len(entry["Layers"]):
            raise ValueError("Docker archive layer count differs from image config")
        layers = []
        for name, expected in zip(entry["Layers"], diffs):
            stream = tar.extractfile(name)
            compressed = stream.peek(2)[:2] == b"\x1f\x8b"
            layer = write_blob(root, stream, "layer.v1.tar+gzip" if compressed else "layer.v1.tar")
            diff = layer["digest"]
            if compressed:
                h = hashlib.sha256()
                with gzip.open(blob_path(root, diff)) as raw:
                    for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                        h.update(chunk)
                diff = "sha256:" + h.hexdigest()
            if diff != expected:
                raise ValueError("Docker archive layer differs from verified image config")
            layers.append(layer)
    manifest = dict(schemaVersion=2, mediaType=MEDIA + "manifest.v1+json", config=config, layers=layers)
    desc = write_blob(root, io.BytesIO(json.dumps(manifest).encode()), "manifest.v1+json")
    (root / "index.json").write_text(json.dumps(dict(schemaVersion=2, manifests=[desc])))
    (root / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')


def read_blob(root, digest):
    data = blob_path(root, digest).read_bytes()
    if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("OCI blob differs from its descriptor")
    return json.loads(data)


def identity(root):
    desc, = json.loads((root / "index.json").read_text())["manifests"]
    manifest = read_blob(root, desc["digest"])
    config_id = manifest["config"]["digest"]
    config = read_blob(root, config_id)
    return desc["digest"], config_id, config["rootfs"]["diff_ids"]


def verify(base, image_id, expected_manifest=None, published=None):
    manifest, config_id, diff_ids = identity(base)
    if config_id != image_id or expected_manifest and manifest != expected_manifest:
        raise ValueError("OCI base differs from verified S6I digest")
    if published:
        final, _, layers = identity(published)
        if layers[:len(diff_ids)] != diff_ids or len(layers) > len(diff_ids) + 1:
            raise ValueError("published dependency layers differ from S6I")
        return final
    return manifest


def archive_for_load(root, image_id, manifest, archive):
    verify(root, image_id, manifest)
    layers = read_blob(root, manifest)["layers"]
    config = blob_path(root, image_id)
    names = [blob_path(root, layer["digest"]).relative_to(root).as_posix() for layer in layers]
    entry = dict(Config=config.relative_to(root).as_posix(), RepoTags=[], Layers=names)
    data = json.dumps([entry]).encode()
    with tarfile.open(archive, "x") as tar:
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for name in dict.fromkeys([entry["Config"], *names]):
            tar.add(root / name, arcname=name, recursive=False)


if __name__ == "__main__":
    action, path, image_id, *rest = sys.argv[1:]
    root = Path(path)
    if action == "export":
        export(rest[0], root, image_id)
    elif action == "archive":
        archive_for_load(root, image_id, *rest)
    else:
        print(verify(root, image_id, rest[0] if rest else None, Path(rest[1]) if len(rest) > 1 else None))
