#!/bin/bash
# usage (on the build host, called by the agent-job launcher after syncing the task):
#   inject-author-repo.sh <synced-task-dir>
# Code-grounded S4c / S5G jobs: if <task>/environment/work/.tbsgen-remote-repo exists (JSON {repo_dir, pin}), copy
# repo_dir into <task>/environment/work/repo after checking that <repo_dir>/../PIN equals pin (the commit / archive
# pinned by S3c).  The authors' code is copied only on the build host; it never travels to the orchestrator.
# Exit 0 = nothing to do or copied; 3 = pin mismatch / missing repo (do not start the job).
set -euo pipefail
T=$1
M="$T/environment/work/.tbsgen-remote-repo"
[ -f "$M" ] || exit 0
read -r RD PIN < <(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["repo_dir"],d.get("pin") or "")' "$M")
[ -d "$RD" ] || { echo "authors' repo $RD missing" >&2; exit 3; }
GOT=$(cat "$(dirname "$RD")/PIN" 2>/dev/null || true)
[ -n "$PIN" ] && [ "$GOT" = "$PIN" ] || { echo "pin mismatch: want $PIN got $GOT" >&2; exit 3; }
rm -rf "$T/environment/work/repo"
cp -a "$RD" "$T/environment/work/repo"
rm -rf "$T/environment/work/repo/.git"
find "$T/environment/work/repo" -type l -delete        # links could point outside the tree
echo "authors' repo $RD ($PIN) injected"
