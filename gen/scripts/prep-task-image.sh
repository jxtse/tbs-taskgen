#!/usr/bin/env bash
# calib.prepare_command = "bash /absolute/path/to/gen/scripts/prep-task-image.sh {item_id}"
set -euo pipefail
id=${TBSGEN_ITEM_ID:?}
[[ "$id" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ && "$id" == "${1:-$id}" ]]
pkg=${TBSGEN_PKG:?}
item=$(dirname "$pkg")
run=$(dirname "$(dirname "$item")")
if [[ -z ${TBSGEN_IMAGE_OCI:-} ]]; then
    legacy=${TBSGEN_LEGACY_PREP:?set TBSGEN_LEGACY_PREP to the legacy preparation script}
    # Both prep paths modify the same template plan and overrides.
    mkdir -p "$run/sandbox/state"
    exec python3 - "$run/sandbox/state/.prep-task-image.lock" "$legacy" "$id" <<'PYTHON'
import fcntl, subprocess, sys
with open(sys.argv[1], "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    sys.exit(subprocess.call(["bash", *sys.argv[2:]]))
PYTHON
fi
aliyun=${TBS_ALIYUN_ROOT:?set TBS_ALIYUN_ROOT to the preparation project directory}
crane=${CRANE:?set CRANE to the crane executable}
export DOCKER_CONFIG=${DOCKER_CONFIG:?set DOCKER_CONFIG to the registry configuration directory}
tag=${TBSGEN_IMAGE_TAG:?}
registry=${tag%:*}
out=${TBSGEN_PREPARED_DIR:?}
templates=${SANDBOX_TEMPLATES:?set SANDBOX_TEMPLATES to the template tool path}
sandbox_dir=$run/sandbox
state=$sandbox_dir/state
dataset=$sandbox_dir/dataset
mkdir -p "$state" "$dataset" "$(dirname "$out")" "$item/calib"
work=$(mktemp -d "$sandbox_dir/.prep-image.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy
export NO_PROXY='*' no_proxy='*'
rsync -a --partial "$TBSGEN_IMAGE_OCI/" "$work/oci/"
python3 - "$work/oci/index.json" "${TBSGEN_IMAGE_DIGEST:?}" <<'PYTHON'
import json, sys
if json.load(open(sys.argv[1]))["manifests"][0]["digest"] != sys.argv[2]:
    sys.exit("transferred OCI manifest differs from verified publish digest")
PYTHON
"$crane" push "$work/oci" "$tag"
digest=$("$crane" digest "$tag")
python3 - "$work/oci/index.json" "$digest" <<'PY'
import json, re, sys
index = json.load(open(sys.argv[1]))
if not re.fullmatch(r"sha256:[0-9a-f]{64}", sys.argv[2]) or sys.argv[2] != index["manifests"][0]["digest"]:
    sys.exit("ACR digest differs from built OCI manifest")
PY

cd "$aliyun"
source direct-env.sh
# The template tool updates shared state. Serialize its init/run and the override merge across preparations.
python3 - "$state" "$dataset" "$id" "$out" "$pkg" "$templates" "$registry" "$tag" "$digest" "$work" <<'PY'
import fcntl, json, os, shutil, subprocess, sys
from pathlib import Path

state, dataset, item_id, output, pkg, templates, registry, tag, digest, work = sys.argv[1:]
state, dataset, output, pkg, work = map(Path, (state, dataset, output, pkg, work))
env = os.environ.copy()
env.pop("PYTHONPATH", None)
ref = registry + "@" + digest
with (state / ".prep-task-image.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    overrides = state / "image-overrides.json"
    entries = json.loads(overrides.read_text()) if overrides.exists() else {}
    entries[item_id + "/agent"] = ref
    temp = overrides.with_suffix(".tmp")
    temp.write_text(json.dumps(entries, indent=1, sort_keys=True))
    temp.replace(overrides)
    receipt = pkg.parent / "calib/image-push.json"
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(tag=tag, digest=digest, pkg_sha=os.environ["TBSGEN_PKG_SHA"])))
    temp.replace(receipt)
    dst = dataset / item_id
    shutil.copytree(pkg, work / "dataset")
    if dst.exists():
        dst.rename(work / "previous-dataset")
    (work / "dataset").rename(dst)
    if output.exists():
        output.rename(work / "previous-output")
    cmd = ["uv", "run", "--project", ".", "python", templates]
    try:
        subprocess.run(cmd + ["init", state, "--dataset", dataset, "--output", output.parent, "--tag", "tbsprod",
                              "--registry", registry, "--env-file", ".env"], check=True, env=env)
        subprocess.run(cmd + ["run", state, "--task", item_id, "--workers", "3"], check=True, env=env)
        summary = json.loads((state / "register-summary.json").read_text())
        if summary["done"] != 1 or summary["waiting"] or summary["failures"]:
            raise ValueError("template registration did not finish")
        if not (output / "task.toml").is_file():
            raise ValueError("template preparation produced no task.toml")
    except BaseException:
        if output.exists():
            output.rename(work / "failed-output")
        if (work / "previous-output").exists():
            (work / "previous-output").rename(output)
        raise
PY
