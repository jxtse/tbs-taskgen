"""Copied from tbs-datapipe-20261015/pipeline/tbsdata/templates_metrics.py (v0.3); v2 change: ENTRY can be
overridden by TBSGEN_ENTRY (local test runner only).

Injected into every generated task as tests/_metrics.py (dense sub-scores for closeness analysis).

v0.3: run_solution() runs the agent program as `nobody` with tests/truth unreadable (see _sandbox_prepare).
v0.2 adds run_solution(): the verifier executes the agent's delivered program on hidden, distribution-shifted
evaluation inputs (/tests/hidden), so tuning only on the public data does not suffice (anti circular validation).
"""
import json, math, os, subprocess, sys, tempfile, time

_PATH = os.environ.get("TBSGEN_METRICS", "/logs/verifier/metrics.jsonl")
ENTRY = os.environ.get("TBSGEN_ENTRY", "/app/output/solution_cli.py")
HIDDEN = os.environ.get("TBSGEN_HIDDEN", "/tests/hidden")
_CACHE = {}


def _record(rec):
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    with open(_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def check(name, value, threshold, direction):
    value = float(value)
    passed = (value >= threshold) if direction == ">=" else (value <= threshold)
    if math.isnan(value):
        passed = False
    _record({"name": name, "value": value, "threshold": float(threshold), "direction": direction,
             "passed": bool(passed)})
    assert passed, f"{name}={value:.6g} fails {direction} {threshold}"


def _cpu_quota():
    """CPUs granted by the container's cgroup quota (docker --cpus); BLAS otherwise sees every host core."""
    try:  # cgroup v2
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            return max(1, int(round(int(q) / int(per))))
    except Exception:
        pass
    try:  # cgroup v1 (some self-hosted servers)
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, int(round(q / per)))
    except Exception:
        pass
    return None


def _thread_env(env):
    n = _cpu_quota()
    if n:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.setdefault(k, str(n))
    return env


def _open_workspace(app):
    """Make the agent workspace usable for `nobody` (caches, relative writes) without ever following a symlink or
    touching anything whose real path is outside `app`, and without touching hard-linked files."""
    if not os.path.isdir(app) or os.path.islink(app):
        return
    real_app = os.path.realpath(app)
    for root, dirs, files in os.walk(app, followlinks=False):
        for name in dirs + files:
            p = os.path.join(root, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if os.path.islink(p) or not os.path.realpath(p).startswith(real_app + os.sep):
                continue
            if os.path.isdir(p):
                os.chmod(p, 0o777)
            elif os.path.isfile(p) and st.st_nlink == 1:
                os.chmod(p, 0o777 if os.access(p, os.X_OK) else 0o666)
    os.chmod(app, 0o777)


def _lock_tree(path):
    """root-only, recursively; symlinks are left alone (their targets are locked where they live)."""
    if os.path.islink(path):
        return
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path, followlinks=False):
            os.chmod(root, 0o700)
            for f in files:
                p = os.path.join(root, f)
                if not os.path.islink(p):
                    os.chmod(p, 0o600)
    else:
        os.chmod(path, 0o600)


def _unlocked(tests_root, hidden_root):
    """paths under tests_root (outside the hidden inputs) that group/other can still read or traverse"""
    import stat
    paths = []
    for name in os.listdir(tests_root):
        top = os.path.join(tests_root, name)
        if os.path.abspath(top) == hidden_root or os.path.islink(top):
            continue
        if os.path.isdir(top):
            for root, dirs, files in os.walk(top, followlinks=False):
                paths += [root] + [os.path.join(root, f) for f in files]
        else:
            paths.append(top)
    return [p for p in paths if not os.path.islink(p) and os.lstat(p).st_mode & (stat.S_IRWXG | stat.S_IRWXO)]


def _sandbox_prepare(input_dir, out):
    """v0.3: when the verifier runs as root (Harbor), run the agent program as `nobody` and make the
    answer-bearing tests/truth unreadable to it, so solution_cli.py cannot read ground truth.
    Returns (user, group, env) for subprocess.run, or None when not root (self-hosted validator runs our own
    reference/baselines as an unprivileged uid; nothing untrusted runs there)."""
    if os.geteuid() != 0:
        return None
    import pwd
    try:
        pw = pwd.getpwnam("nobody")
    except KeyError:
        return None
    # v2.1: lock everything under the tests root (parent of HIDDEN, i.e. /tests) except the hidden-input tree.
    # v2.3 (review of e2e-iter1): the agent workspace pass must not follow symlinks -- an agent-made
    # /app/leak -> /tests/truth used to be chmod-ed 0777 through the link -- and the lock runs AFTER it and is
    # verified; on any lock failure the agent program is not run at all.
    hidden_root = os.path.abspath(HIDDEN)
    inp = os.path.abspath(input_dir)
    if not (inp == hidden_root or inp.startswith(hidden_root + os.sep)):
        hidden_root = inp
    tests_root = os.path.dirname(hidden_root)
    _open_workspace("/app")
    for name in os.listdir(tests_root):  # everything in /tests except the hidden inputs becomes root-only
        pth = os.path.join(tests_root, name)
        if os.path.abspath(pth) != hidden_root:
            _lock_tree(pth)
    os.chmod(tests_root, 0o711)
    for root, dirs, files in os.walk(hidden_root):
        os.chmod(root, 0o755)
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                os.chmod(p, 0o644)
    bad = _unlocked(tests_root, hidden_root)
    if bad:
        raise PermissionError(f"truth lock failed, agent program not run: {bad[:3]}")
    os.chown(out, pw.pw_uid, pw.pw_gid)
    home = tempfile.mkdtemp(prefix="solhome-")
    os.chown(home, pw.pw_uid, pw.pw_gid)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TBSGEN")}
    env.update(HOME=home, TMPDIR=home, MPLCONFIGDIR=home)
    return pw.pw_uid, pw.pw_gid, env


def hidden_sets():
    """Hidden input sets: subdirectories of HIDDEN if it has any (several shifted regimes), else [HIDDEN]."""
    subs = sorted(os.path.join(HIDDEN, d) for d in os.listdir(HIDDEN) if os.path.isdir(os.path.join(HIDDEN, d))) \
        if os.path.isdir(HIDDEN) else []
    return subs or [HIDDEN]


def _no_new_privs():
    """preexec: PR_SET_NO_NEW_PRIVS, so a setuid-root binary the (root) agent planted earlier cannot lift the
    unprivileged program back to root (review of e2e-iter1)."""
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0)
    except Exception:
        pass


def run_solution(timeout_sec, input_dir=HIDDEN, entry=ENTRY):
    """Run `python <entry> <input_dir> <out_dir>` once per session; returns (out_dir, wall_seconds, returncode).
    Several hidden sets: call once per set, e.g. `for d in hidden_sets(): out, sec, rc = run_solution(t, d)`.
    This is the ONLY sanctioned way to execute agent code in the verifier (it drops to `nobody` and locks truth).
    The wall time is also recorded (not a pass/fail check by itself; tests decide whether to check() it)."""
    key = (entry, input_dir)
    if key in _CACHE:
        return _CACHE[key]
    out = tempfile.mkdtemp(prefix="sol-")
    try:
        sb = _sandbox_prepare(input_dir, out)
    except Exception as e:  # never run untrusted code with truth possibly readable
        _record({"name": "_solution_cli_run", "value": 0.0, "rc": -1, "stderr_tail": f"not run: {e}"[-500:],
                 "threshold": float(timeout_sec), "direction": "<=", "passed": False, "aux": True,
                 "sandboxed": False, "lock_failed": True})
        _CACHE[key] = (out, 0.0, -1)
        return _CACHE[key]
    kw = dict(env=_thread_env(dict(os.environ))) if sb is None else \
        dict(user=sb[0], group=sb[1], extra_groups=[], env=_thread_env(sb[2]), cwd=sb[2]["HOME"],
             preexec_fn=_no_new_privs)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, entry, input_dir, out], capture_output=True, text=True,
                           timeout=timeout_sec, **kw)
        rc, err = p.returncode, p.stderr[-2000:]
    except subprocess.TimeoutExpired:
        rc, err = -9, f"timeout after {timeout_sec}s"
    sec = time.time() - t0
    _record({"name": "_solution_cli_run", "value": sec, "rc": rc, "stderr_tail": err[-500:],
             "threshold": float(timeout_sec), "direction": "<=", "passed": rc == 0, "aux": True,
             "sandboxed": sb is not None})
    _CACHE[key] = (out, sec, rc)
    return _CACHE[key]
