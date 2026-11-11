#!/bin/bash
# usage (on the build host, called by the agent-job launcher after syncing the task, before the image is built):
#   inject-author-repo.sh <synced-task-dir> <code.remote_root>
# Code-grounded S4c / S5G jobs: if <task>/environment/work/.tbsgen-remote-repo exists (JSON {repo_dir, pin}), copy
# repo_dir into <task>/environment/work/repo after checking that repo_dir is a real directory under remote_root (no
# link at any level) and that <repo_dir>/../PIN equals pin (the commit / archive pinned by S3c).  Links inside the
# copy are removed.  The job's Dockerfile refuses to build without the injected repo and matching pin.
# Exit 0 = nothing to do or injected; 3 = refused (do not start the job).
set -euo pipefail
T=$1; ROOT=$(realpath -e "$2")
M="$T/environment/work/.tbsgen-remote-repo"
[ -f "$M" ] || exit 0
read -r RD PIN < <(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["repo_dir"],d.get("pin") or "")' "$M")
[ -n "$PIN" ] || { echo "no pin" >&2; exit 3; }
# every component from / down must be a real directory (no links), and the path must lie under ROOT
P=""; for c in $(echo "$RD" | tr '/' ' '); do P="$P/$c"; [ -L "$P" ] && { echo "link in path: $P" >&2; exit 3; }; done
[ -d "$RD" ] || { echo "authors' repo $RD missing" >&2; exit 3; }
R=$(realpath -e "$RD"); case "$R" in "$ROOT"/*) ;; *) echo "$RD outside $ROOT" >&2; exit 3 ;; esac
[ "$R" = "$RD" ] || { echo "non-canonical path $RD" >&2; exit 3; }
GOT=$(cat "$(dirname "$RD")/PIN" 2>/dev/null || true)
[ "$GOT" = "$PIN" ] || { echo "pin mismatch: want $PIN got $GOT" >&2; exit 3; }
W="$T/environment/work"
[ -L "$W" ] || [ -L "$T/environment" ] && { echo "link in task dir" >&2; exit 3; }
rm -rf "$W/repo"
cp -a --no-dereference "$RD" "$W/repo"
find "$W/repo" -type l -delete                      # links could point anywhere on the host
[ -d "$W/repo/.git" ] && [ ! -L "$W/repo/.git" ] && rm -rf "$W/repo/.git"
echo "$PIN" > "$W/repo.PIN"
echo "authors' repo $RD ($PIN) injected"
