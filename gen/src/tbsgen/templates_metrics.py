"""Verifier helper (v0.3, carried over from the previous pipeline); ENTRY can be overridden by TBSGEN_ENTRY
(local test runner only).

Injected into every generated task as tests/_metrics.py (dense sub-scores for closeness analysis).

v0.4: run_solution() starts the agent program in its own session and, after every call (and on timeout), kills
its process group and every remaining process of the sandbox uid, so it cannot leave a detached process behind
(S9R review of e2e-prod-20261109: a daemon left by the CLI rewrote /logs/verifier/reward.txt to 1 after pytest).
/logs/verifier is made unwritable for the sandbox uid before the first run; test.sh writes the reward last via
`python _metrics.py --finalize-reward RC`, after the sandbox uid has no process left.  Wall time = median of
TIMING_REPEATS runs (outputs and return code of the first run; the repeats only time the program).
v0.3: run_solution() runs the agent program as `nobody` with tests/truth unreadable (see _sandbox_prepare).
v0.2 adds run_solution(): the verifier executes the agent's delivered program on hidden, distribution-shifted
evaluation inputs (/tests/hidden), so tuning only on the public data does not suffice (anti circular validation).
"""
import json, math, os, re, signal, subprocess, sys, tempfile, time

_PATH = os.environ.get("TBSGEN_METRICS", "/logs/verifier/metrics.jsonl")
ENTRY = os.environ.get("TBSGEN_ENTRY", "/app/output/solution_cli.py")
HIDDEN = os.environ.get("TBSGEN_HIDDEN", "/tests/hidden")
LOGS = os.path.dirname(_PATH)
# number of timed runs per hidden set (wall time = median); written by the pipeline when it copies this file
# (config thresholds.timing_repeats for tasks with a resource metric, else 1)
TIMING_REPEATS = 3  # tbsgen:timing_repeats
# machine-speed reference (thresholds.speed_bench_sec, measured on the host that set the thresholds); None = no
# scaling (first measurement: no thresholds exist yet).  Written by the pipeline when it copies this file.
SPEED_BASE = None  # tbsgen:speed_base
# resource-metric policy (spec.resource_metric_gating): false = wall time / memory are recorded but never decide
# a run; true = they are asserted at the machine-scaled threshold
RESOURCE_GATING = False  # tbsgen:resource_gating
# the task's resource metrics {name: "time" | "memory"} (spec: name-based tbsgen.spec.RESOURCE_METRIC or
# kind="resource"); written by the pipeline.  None (a standalone copy) -> the same name rule as tbsgen.spec.
# Time-like ones scale with this machine's speed_bench() / SPEED_BASE (clamped); memory-like ones (peak RSS) do not
# (they do not track CPU speed).  Any other metric is checked exactly as before.
RESOURCE_METRICS = None  # tbsgen:resource_metrics
_RESOURCE_NAME = re.compile(r"(?:^|_)(?:runtime|wall_?time|walltime|cpu_time|peak_memory|peak_rss|max_rss|peak_mem)"
                            r"(?:_|$)", re.I)
_MEMORY_NAME = re.compile(r"(?:^|_)(?:peak_memory|peak_rss|max_rss|peak_mem|memory|rss|mem|mb|mib|bytes)(?:_|$)", re.I)


def resource_kind(name):
    """"time" | "memory" for a resource metric, None for any other metric."""
    if RESOURCE_METRICS is not None:
        return RESOURCE_METRICS.get(name)
    if not _RESOURCE_NAME.search(name):
        return None
    if _MEMORY_NAME.search(name):
        return "memory"
    # only an unmistakable time name is speed-scaled; anything else is treated like memory (never scaled)
    return "time" if re.search(r"runtime|wall_?time|walltime|cpu_?time", name, re.I) else "memory"
SPEED_MIN, SPEED_MAX = 0.25, 8.0
# |ratio - 1| <= deadband -> ratio 1: same-host noise never moves a threshold (config thresholds.speed_deadband)
SPEED_DEADBAND = 0.1  # tbsgen:speed_deadband
_CACHE = {}


# the fixed standard computation every machine runs (single-threaded env as the solutions get): the machine-speed
# reference for the time-threshold scaling.  Deterministic, pure CPU, a few hundred ms.
_BENCH_CODE = r'''
import json, time
try:
    import numpy as np          # imported before the clock starts: interpreter / import I/O is not CPU speed
except Exception:
    np = None
t0 = time.perf_counter()
x = 0.0
for i in range(400000):
    x += (i % 7) * 0.5 - (i % 3)
try:
    a = np.linspace(0.0, 1.0, 65536)
    for _ in range(48):
        a = (np.fft.irfft(np.fft.rfft(a * 0.5))[: a.size] * 0.5) + 0.1
    m = np.arange(192 * 192, dtype=float).reshape(192, 192)
    for _ in range(50):
        m = np.roll(m, 1, axis=0) * 0.99 + m * 0.01
except Exception:
    pass
print(json.dumps({"sec": time.perf_counter() - t0, "x": round(x, 3)}))
'''


def speed_bench(repeats: int = 5):
    """This machine's time on the fixed standard computation: median of `repeats` runs in a fresh interpreter with
    the same thread environment the solutions get.  Cached per process; recorded (aux) in metrics.jsonl."""
    if _CACHE.get("speed_bench") is not None:
        return _CACHE["speed_bench"]
    times = []
    for _ in range(max(1, int(repeats))):
        try:
            p = subprocess.run([sys.executable, "-c", _BENCH_CODE], capture_output=True, text=True,
                               env=_thread_env(dict(os.environ)), timeout=120)
            d = json.loads(p.stdout.strip().splitlines()[-1])
            t = float(d.get("sec") or 0.0)
            if t > 0:
                times.append(t)
        except Exception:
            pass
    med = _median(times) if times else None
    _CACHE["speed_bench"] = med
    if med is not None:
        _record({"name": "_speed_bench", "value": round(med, 6), "aux": True, "unit": "sec"})
    return med


def _record(rec):
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    with open(_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def check(name, value, threshold, direction):
    """One metric check.  Resource metrics (wall time / memory) follow the resource-metric policy written by the
    pipeline: RESOURCE_GATING False -> recorded (value, raw and effective threshold, speed ratio) but never
    deciding a run; True -> asserted at the machine-scaled threshold.  Time-like resource thresholds scale with
    this machine's speed (speed_bench() / SPEED_BASE, measured where the thresholds were set, clamped to
    [SPEED_MIN, SPEED_MAX]); memory-like ones do not.  The recorded "threshold" stays the locked literal — the
    pipeline's lock checks compare it (runner / review.runtime_thresholds); the effective one is recorded next to
    it.  Non-resource metrics are asserted at the literal threshold as before."""
    value = float(value)
    kind = resource_kind(name)
    if kind:
        ratio = 1.0
        if kind == "time" and SPEED_BASE:
            b = speed_bench()
            if b and b > 0:
                ratio = b / float(SPEED_BASE)
                if abs(ratio - 1.0) <= SPEED_DEADBAND:
                    ratio = 1.0                      # same host: noise never moves a threshold
                ratio = min(SPEED_MAX, max(SPEED_MIN, ratio))
        eff = float(threshold) * ratio if direction == "<=" else float(threshold) / ratio
        rec = {"name": name, "value": value, "threshold": float(threshold), "direction": direction,
               "threshold_effective": eff, "speed_ratio": round(ratio, 4)}
        if not RESOURCE_GATING:
            rec["gated"] = False
            rec["passed_raw"] = bool((value >= threshold) if direction == ">=" else (value <= threshold))
            _record(rec)
            return
        passed = (value >= eff) if direction == ">=" else (value <= eff)
        if math.isnan(value):
            passed = False
        rec["passed"] = bool(passed)
        _record(rec)
        assert passed, f"{name}={value:.6g} fails {direction} {eff:.6g} (threshold {threshold:.6g} x speed ratio {ratio:.3f})"
        return
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
    _lock_logs(pw.pw_uid)
    os.chown(out, pw.pw_uid, pw.pw_gid)
    home, _hp, _hi = _private_dir("solhome-", (pw.pw_uid, pw.pw_gid))
    env = {k: v for k, v in os.environ.items() if not k.startswith("TBSGEN")}
    env.update(HOME=home, TMPDIR=home, MPLCONFIGDIR=home, PYTHONDONTWRITEBYTECODE="1")
    return pw.pw_uid, pw.pw_gid, env


def _lock_logs(uid):
    """v0.4: the verifier log dir (reward.txt, metrics.jsonl) must not be writable by the sandbox uid.  Harbor mounts
    it world-writable (0777, owned by the host user): drop group/other write, keep the owner (the host user must still
    be able to clean the job dir up); a dir owned by the sandbox uid itself goes to root.  Raises when the sandbox
    uid could still write there, so the agent program is not run."""
    os.makedirs(LOGS, exist_ok=True)
    st = os.lstat(LOGS)
    if st.st_uid == uid:
        os.chown(LOGS, 0, 0)
    os.chmod(LOGS, st.st_mode & 0o7755 & ~0o022)
    for name in os.listdir(LOGS):
        q = os.path.join(LOGS, name)
        qs = os.lstat(q)
        if qs.st_uid == 0 and not os.path.islink(q) and os.path.isfile(q) and qs.st_nlink == 1:
            os.chmod(q, qs.st_mode & 0o7755 & ~0o022)        # root's own files (metrics.jsonl): no group/other write
        elif os.path.isdir(q) and not os.path.islink(q):  # anything else (non-root, links, dirs) is removed
            import shutil
            shutil.rmtree(q)
        else:
            os.unlink(q)
    st = os.lstat(LOGS)
    if st.st_uid == uid or st.st_mode & 0o022:
        raise PermissionError(f"verifier log dir {LOGS} still writable by the sandbox uid (mode {oct(st.st_mode)})")


def _uid_pids(uid):
    """pids of every process whose real, effective, saved or fs uid is `uid` (/proc scan)"""
    pids = []
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(f"/proc/{d}/status") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        if str(uid) in line.split()[1:5]:
                            pids.append(int(d))
                        break
        except OSError:
            continue
    return pids


def _reap_uid(uid, rounds=50):
    """SIGKILL every process of `uid` (a forked helper drops to `uid` and kill(-1), which also reaches processes
    that left the session / process group, then the /proc scan confirms).  Returns the pids still alive (zombies
    of our own children aside, they cannot run).  Root only."""
    if os.geteuid() != 0 or uid == 0:
        return []
    left = []
    for _ in range(rounds):
        left = [p for p in _uid_pids(uid) if not _zombie(p)]
        if not left:
            return []
        pid = os.fork()
        if pid == 0:                                 # helper: become `uid`, kill everything `uid` may signal
            try:
                os.setgroups([])
                os.setgid(os.getgid() if uid == 0 else _gid_of(uid))
                os.setuid(uid)
                os.kill(-1, signal.SIGKILL)
            except Exception:
                pass
            os._exit(0)
        os.waitpid(pid, 0)
        for p in left:                               # and directly, as root
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.05)
    return [p for p in _uid_pids(uid) if not _zombie(p)]


def _nonroot_uids():
    uids = set()
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/status") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        uids.update(int(x) for x in line.split()[1:5] if x != "0")
                        break
        except OSError:
            continue
    return uids


def _reap_nonroot():
    """SIGKILL every process running under any non-root uid (the sandbox uid, and e.g. a process the root agent
    left behind as the uid that owns the Harbor log mount, which keeps write access to /logs/verifier).
    Returns the pids still alive."""
    if os.geteuid() != 0:
        return []
    left = []
    for _ in range(3):
        uids = _nonroot_uids()
        if not uids:
            return []
        left = [q for u in sorted(uids) for q in _reap_uid(u)]
    return left


def _zombie(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] in ("Z", "X")
    except OSError:
        return True


def _gid_of(uid):
    import pwd
    try:
        return pwd.getpwuid(uid).pw_gid
    except KeyError:
        return uid


def _sandbox_uid():
    if os.geteuid() != 0:
        return None
    import pwd
    try:
        return pwd.getpwnam("nobody").pw_uid
    except KeyError:
        return None


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


TIMING_BUDGET_SEC = 300.0  # tbsgen:timing_budget
# Extra wall time the timed repeats may use per verifier session, snapshots included.  The pipeline adds the same
# amount to the task's verifier timeout, so the scored (first) runs keep the whole original timeout.
_TIMING_SPENT = [0.0]
# Total wall time all scored (first) runs of one verifier session may use, whatever timeout tests/measure.py
# passes: each call's timeout is clamped to what is left, and a call with nothing left fails like a timeout
# (rc -9, wall time inf).  Written by the pipeline: package.verifier_timeout - package.verifier_overhead_sec, so
# scored runs + TIMING_BUDGET_SEC always end before the verifier timeout (e2e-prod review: run_solution(1800)
# under a 1200 s verifier timeout).  None = no cap (a standalone copy).
SCORED_BUDGET_SEC = None  # tbsgen:scored_budget
_SCORED_SPENT = [0.0]
_SNAP_SKIP_TOP = ("/proc", "/sys", "/dev")
_SNAP_LIMIT = 400000


def _skipped(q, skip):
    return any(q == k or q.startswith(k + os.sep) for k in skip)


def _xattrs(q):
    try:
        return tuple(sorted((n, os.getxattr(q, n, follow_symlinks=False))
                            for n in os.listxattr(q, follow_symlinks=False)))
    except OSError:
        return ()


def _ident(q, st, tmp):
    """Identity of one entry: (ino, ctime, size, mode, uid) -- ctime moves on any content / mode / owner / xattr
    change and, for a directory, on any entry added or removed.  The verifier itself (root) creates and deletes
    temp files only in its temp dir: there ctime is replaced by the xattrs (a sticky world-writable dir: the
    sandbox uid can add entries -- caught as new paths -- but not set xattrs on it)."""
    if q == tmp:
        return (st.st_ino, None, None, st.st_mode, st.st_uid, _xattrs(q))
    return (st.st_ino, st.st_ctime_ns, st.st_size if not os.path.isdir(q) else None, st.st_mode, st.st_uid)


def _state_snapshot(uid, skip, deadline=None):
    """Whole-filesystem state the agent program could carry from one timed run to the next: {path: identity} for
    every entry outside /proc, /sys, /dev (but /dev/shm) and the `skip` trees, the SysV IPC objects and kernel
    key usage of `uid`.  None when the tree is too big or the deadline passes (-> no repeats)."""
    snap, n = {}, 0
    tmp = os.path.realpath(tempfile.gettempdir())
    tops = ["/dev/shm"] if os.path.isdir("/dev/shm") else []
    tops += [os.path.join("/", d) for d in sorted(os.listdir("/")) if os.path.join("/", d) not in _SNAP_SKIP_TOP]
    for top in tops:
        if _skipped(top, skip):
            continue
        try:
            st = os.lstat(top)
        except OSError:
            continue
        snap[top] = _ident(top, st, tmp)
        if os.path.islink(top) or not os.path.isdir(top):
            continue
        for root, dirs, files in os.walk(top, followlinks=False):
            dirs[:] = [d for d in dirs if not _skipped(os.path.join(root, d), skip)]
            for name in dirs + files:
                q = os.path.join(root, name)
                if _skipped(q, skip):
                    continue
                try:
                    st = os.lstat(q)
                except OSError:
                    continue
                snap[q] = _ident(q, st, tmp)
                n += 1
                if n > _SNAP_LIMIT or (deadline is not None and n % 2000 == 0 and time.time() > deadline):
                    return None
    for kind in ("shm", "msg", "sem"):
        try:
            lines = open(f"/proc/sysvipc/{kind}").read().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            f = ln.split()
            if len(f) > 8 and str(uid) in f[4:9]:
                snap[f"sysvipc:{kind}:{f[1]}"] = (0, 0, 0, uid)
    try:
        for ln in open("/proc/key-users").read().splitlines():
            if ln.split(":")[0].strip() == str(uid):
                snap["key-users"] = ln.strip()
    except OSError:
        pass
    if deadline is not None and time.time() > deadline:
        return None
    return snap


def _state_changes(before, after):
    """Entries added, removed or changed between two snapshots (None snapshot -> 'unavailable')."""
    if before is None or after is None:
        return ["state snapshot unavailable"]
    return sorted([q for q in after if q not in before] + [q for q in before if q not in after] +
                  [q for q, v in after.items() if q in before and before[q] != v])


def _private_dir(prefix, sb):
    """dir owned by the sandbox uid inside a root-owned 0711 parent: the program can use it but cannot rename it
    or swap it for a symlink (that needs write access to the parent).  Returns (dir, parent, inode)."""
    parent = tempfile.mkdtemp(prefix=prefix)
    os.chmod(parent, 0o711)
    d = os.path.join(parent, "d")
    os.mkdir(d, 0o755)
    if sb is not None:
        os.chown(d, sb[0], sb[1])
    return d, parent, os.lstat(d).st_ino


def _seal_tree(d, ino=None):
    """After the scored run: its output dir becomes root-owned and private, so a timed repeat can neither read
    (copy) nor change it.  Returns a reason when that cannot be guaranteed (-> no repeats): the dir itself is not
    the real directory created for the run, or the tree holds symlinks / hard links / special files (they could
    point the scored output at a file a repeat may write)."""
    st = os.lstat(d)
    if os.path.islink(d) or not os.path.isdir(d) or (ino is not None and st.st_ino != ino):
        return f"{d} was replaced"
    bad = []
    for root, dirs, files in os.walk(d, followlinks=False):
        for name in dirs + files:
            q = os.path.join(root, name)
            qs = os.lstat(q)
            if os.path.islink(q) or not (os.path.isdir(q) or os.path.isfile(q)) or \
                    (os.path.isfile(q) and qs.st_nlink != 1):
                bad.append(q)
                continue
            os.lchown(q, 0, 0)
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if os.fstat(fd).st_ino != st.st_ino:
            return f"{d} was replaced"
        os.fchown(fd, 0, 0)
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)
    return f"output holds links / special files: {bad[:3]}" if bad else None


def _run_once(entry, input_dir, out, timeout_sec, sb, home=None):
    """One run of the agent program in its own session.  stdout/stderr go to files (a detached child holding a
    pipe cannot stall the verifier); afterwards the whole process group and every process of the sandbox uid are
    killed.  Returns (rc, stderr_tail, wall_seconds, leftover_pids)."""
    if sb is None:
        kw = dict(env=_thread_env(dict(os.environ)))
    else:
        env = dict(sb[2])
        if home:
            env.update(HOME=home, TMPDIR=home, MPLCONFIGDIR=home)
        kw = dict(user=sb[0], group=sb[1], extra_groups=[], env=_thread_env(env), cwd=env["HOME"],
                  preexec_fn=_no_new_privs)
    with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
        t0 = time.time()
        p = subprocess.Popen([sys.executable, entry, input_dir, out], stdin=subprocess.DEVNULL, stdout=fo, stderr=fe,
                             start_new_session=True, **kw)
        try:
            rc = p.wait(timeout=timeout_sec)
            err_extra = ""
        except subprocess.TimeoutExpired:
            rc, err_extra = -9, f"timeout after {timeout_sec}s"
        sec = time.time() - t0
        try:
            os.killpg(p.pid, signal.SIGKILL)         # the session / process group it started in
        except OSError:
            pass
        left = _reap_uid(sb[0]) if sb is not None else []
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        fe.seek(0)
        err = fe.read()[-4000:].decode("utf-8", "replace") + err_extra
    return rc, err, sec, left


def _tree_digest(d):
    import hashlib
    h = hashlib.sha256()
    for root, dirs, files in os.walk(d, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            q = os.path.join(root, name)
            h.update(os.path.relpath(q, d).encode() + b"\0")
            try:
                if os.path.islink(q):
                    h.update(b"L" + os.readlink(q).encode())
                else:
                    with open(q, "rb") as f:
                        for b in iter(lambda: f.read(1 << 20), b""):
                            h.update(b)
            except OSError:
                h.update(b"?")
    return h.hexdigest()


def _median(xs):
    xs = sorted(xs)
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else 0.5 * (xs[k - 1] + xs[k])


MAX_SPEEDUP = 1.5   # the repeats may lower the reported time to at most first-run time / MAX_SPEEDUP


def run_solution(timeout_sec, input_dir=HIDDEN, entry=ENTRY):
    """Run `python <entry> <input_dir> <out_dir>` once per session; returns (out_dir, wall_seconds, returncode).
    Several hidden sets: call once per set, e.g. `for d in hidden_sets(): out, sec, rc = run_solution(t, d)`.
    This is the ONLY sanctioned way to execute agent code in the verifier (it drops to `nobody`, locks truth and
    the verifier logs, and kills everything the program left running).
    out_dir and returncode are those of the first (scored) run.  wall_seconds is the median over TIMING_REPEATS
    runs (a failed / timed-out repeat counts as +inf), never below first-run time / MAX_SPEEDUP, when: the first
    run succeeded; the repeats and their bookkeeping fit the extra TIMING_BUDGET_SEC (absolute: snapshots stop
    and each repeat's timeout is the budget left; the pipeline adds the budget to the verifier timeout); the
    scored output dir and HOME (each in a root-owned 0711 parent, so they cannot be swapped) hold only plain files
    and become root-only 0700 (a repeat cannot copy or change them); and no run changes anything else on the
    filesystem (whole-FS snapshot outside /proc, /sys, /dev but /dev/shm, by ctime / xattrs; SysV IPC; kernel
    keys) or writes outputs different from the scored run's.  Otherwise the first run's time alone, with the
    reason in the record (`timing_repeats_skipped`).  The wall time is also recorded (not a pass/fail check by
    itself; tests decide whether to check() it)."""
    key = (entry, input_dir)
    if key in _CACHE:
        return _CACHE[key]
    out, out_parent, out_ino = _private_dir("sol-", None)
    entry_exists = os.path.isfile(entry)              # calibration tells "no deliverable" from "deliverable failed"
    try:
        sb = _sandbox_prepare(input_dir, out)
    except Exception as e:  # never run untrusted code with truth possibly readable
        _record({"name": "_solution_cli_run", "value": 0.0, "rc": -1, "stderr_tail": f"not run: {e}"[-500:],
                 "threshold": float(timeout_sec), "direction": "<=", "passed": False, "aux": True,
                 "sandboxed": False, "lock_failed": True, "entry": entry, "entry_exists": entry_exists})
        _CACHE[key] = (out, 0.0, -1)
        return _CACHE[key]
    import shutil

    def left_budget():
        return TIMING_BUDGET_SEC - _TIMING_SPENT[0]

    def timed(fn, *a):
        t = time.time()
        try:
            return fn(*a)
        finally:
            _TIMING_SPENT[0] += time.time() - t

    t_first = float(timeout_sec)
    if t_first != t_first:                            # NaN: never a timeout that cannot expire
        t_first = float("inf")
    if SCORED_BUDGET_SEC is not None:
        t_first = min(t_first, float(SCORED_BUDGET_SEC) - _SCORED_SPENT[0])
    if t_first <= 0:                                  # verifier time for the scored runs used up: a timeout
        _record({"name": "_solution_cli_run", "value": float("inf"), "rc": -9,
                 "stderr_tail": f"not run: the {SCORED_BUDGET_SEC:.0f}s verifier budget of the scored runs is used up",
                 "threshold": float(timeout_sec), "timeout_effective": 0.0, "direction": "<=", "passed": False,
                 "aux": True, "sandboxed": sb is not None, "entry": entry, "entry_exists": entry_exists,
                 "budget_exhausted": True})
        _CACHE[key] = (out, float("inf"), -9)
        return _CACHE[key]
    n_rep = max(1, int(TIMING_REPEATS))
    home = sb[2]["HOME"] if sb is not None else None
    keep = [out_parent, LOGS] + ([os.path.dirname(home)] if home else [])
    want = n_rep > 1 and sb is not None and left_budget() > 0
    before = timed(_state_snapshot, sb[0], keep, time.time() + left_budget()) if want else None
    rc, err, sec, left = _run_once(entry, input_dir, out, t_first if math.isfinite(t_first) else None, sb)
    _SCORED_SPENT[0] += sec
    times, why = [sec], None
    if n_rep > 1 and rc == 0 and not left:
        if sb is None:
            why = "not sandboxed (no root): single run"
        elif before is None or left_budget() < sec * (n_rep - 1) * 1.2:
            why = f"first run {sec:.1f}s: repeats would exceed the {TIMING_BUDGET_SEC:.0f}s timing budget"
        else:
            why = timed(_seal_tree, out, out_ino) or timed(_seal_tree, home)
            if why is None:
                ch = _state_changes(before, timed(_state_snapshot, sb[0], keep, time.time() + left_budget()))
                if ch:
                    why = f"first run changed state a repeat could reuse: {ch[:3]}"
        if why is None:
            ref_digest = timed(_tree_digest, out)
            for _ in range(n_rep - 1):
                o2, op2, oi2 = _private_dir("sol-", sb)
                h2, hp2, _hi2 = _private_dir("solhome-", sb)
                b2 = timed(_state_snapshot, sb[0], keep + [op2, hp2], time.time() + left_budget())
                t_rep = left_budget()                                  # absolute cap: never past the budget
                if b2 is None or t_rep < sec * 1.2:
                    times, why = [sec], "timing budget exhausted"
                    shutil.rmtree(op2, ignore_errors=True)
                    shutil.rmtree(hp2, ignore_errors=True)
                    break
                r2, _e2, s2, left = _run_once(entry, input_dir, o2, min(t_first, t_rep), sb, home=h2)
                _TIMING_SPENT[0] += s2
                same = r2 == 0 and not left and timed(_tree_digest, o2) == ref_digest
                ch = ["left processes"] if left else \
                    _state_changes(b2, timed(_state_snapshot, sb[0], keep + [op2, hp2], time.time() + left_budget()))
                shutil.rmtree(op2, ignore_errors=True)
                shutil.rmtree(hp2, ignore_errors=True)
                if left:
                    break
                if ch:
                    times, why = [sec], f"a timed repeat changed state: {ch[:3]}"
                    break
                if r2 == 0 and not same:
                    # a repeat that wrote other outputs was not timing the scored computation: first run only
                    times, why = [sec], "a timed repeat produced outputs different from the scored run"
                    break
                times.append(s2 if r2 == 0 else float("inf"))
    if left:                                          # could not kill what it left running: no score
        rc, err = -10, (err + f"\nsandbox processes survived SIGKILL: {left[:5]}")[-2000:]
    med = max(_median(times), sec / MAX_SPEEDUP)
    _record({"name": "_solution_cli_run", "value": med, "rc": rc, "stderr_tail": err[-500:],
             "threshold": float(timeout_sec), "timeout_effective": round(t_first, 3) if math.isfinite(t_first) else None, "direction": "<=",
             "passed": rc == 0, "aux": True, "sandboxed": sb is not None, "entry": entry, "entry_exists": entry_exists,
             "wall_times": [round(t, 6) if math.isfinite(t) else None for t in times], "timing_repeats": len(times),
             "timing_repeats_skipped": why})
    _CACHE[key] = (out, med, rc)
    return _CACHE[key]


def finalize_reward(pytest_rc, path=None):
    """Called by test.sh after pytest: kill every non-root process (sandbox uid included), make the log dir
    unwritable for the sandbox uid,
    remove any reward file written before, then write the reward (1 iff pytest passed and nothing survived).
    The file is created with O_EXCL|O_NOFOLLOW (never through a planted link)."""
    path = path or os.path.join(LOGS, "reward.txt")
    reward = 1 if int(pytest_rc) == 0 else 0
    uid = _sandbox_uid()
    if os.geteuid() == 0:
        if _reap_nonroot():
            reward = 0
        if uid is not None:
            try:
                _lock_logs(uid)
            except Exception:
                reward = 0
        if _reap_nonroot():
            reward = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for name in ("reward.txt", "reward.json"):
        q = os.path.join(os.path.dirname(path), name)
        if os.path.lexists(q):
            if os.path.isdir(q) and not os.path.islink(q):
                import shutil
                shutil.rmtree(q)
            else:
                os.unlink(q)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(f"{reward}\n")
    return reward


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--finalize-reward":
        print("reward", finalize_reward(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "--reap":
        print("left", _reap_nonroot())
    elif len(sys.argv) == 2 and sys.argv[1] == "--speed-bench":
        print(json.dumps({"speed_bench_sec": speed_bench()}))
