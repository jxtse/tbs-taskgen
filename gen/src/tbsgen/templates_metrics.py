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
import json, math, os, signal, subprocess, sys, tempfile, time

_PATH = os.environ.get("TBSGEN_METRICS", "/logs/verifier/metrics.jsonl")
ENTRY = os.environ.get("TBSGEN_ENTRY", "/app/output/solution_cli.py")
HIDDEN = os.environ.get("TBSGEN_HIDDEN", "/tests/hidden")
LOGS = os.path.dirname(_PATH)
# number of timed runs per hidden set (wall time = median); written by the pipeline when it copies this file
# (config thresholds.timing_repeats for tasks with a resource metric, else 1)
TIMING_REPEATS = 3  # tbsgen:timing_repeats
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
    _lock_logs(pw.pw_uid)
    os.chown(out, pw.pw_uid, pw.pw_gid)
    home = tempfile.mkdtemp(prefix="solhome-")
    os.chown(home, pw.pw_uid, pw.pw_gid)
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
_SNAP_SKIP_TOP = ("/proc", "/sys", "/dev")
_SNAP_LIMIT = 400000


def _skipped(q, skip):
    return any(q == k or q.startswith(k + os.sep) for k in skip)


def _state_snapshot(uid, skip):
    """Whole-filesystem state the agent program could carry from one timed run to the next: {path: identity} for
    every entry outside /proc, /sys, /dev (but /dev/shm) and the `skip` trees -- regular files by (ino, ctime,
    size), directories by (ino, mode, owner) (their own times move when root creates temp files), plus the SysV
    IPC objects of `uid`.  None when the tree is too big to compare (-> no repeats)."""
    snap, n = {}, 0
    tops = ["/dev/shm"] if os.path.isdir("/dev/shm") else []
    tops += [os.path.join("/", d) for d in sorted(os.listdir("/")) if os.path.join("/", d) not in _SNAP_SKIP_TOP]
    for top in tops:
        if _skipped(top, skip) or os.path.islink(top):
            continue
        try:
            st = os.lstat(top)
        except OSError:
            continue
        snap[top] = (st.st_ino, None, st.st_mode, st.st_uid) if os.path.isdir(top) else \
            (st.st_ino, st.st_ctime_ns, st.st_size, st.st_uid)
        if not os.path.isdir(top):
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
                snap[q] = (st.st_ino, None, st.st_mode, st.st_uid) if os.path.isdir(q) and not os.path.islink(q) \
                    else (st.st_ino, st.st_ctime_ns, st.st_size, st.st_uid)
                n += 1
                if n > _SNAP_LIMIT:
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
    return snap


def _state_changes(before, after):
    """Entries added, removed or changed between two snapshots (None snapshot -> 'unavailable')."""
    if before is None or after is None:
        return ["state snapshot unavailable"]
    return sorted([q for q in after if q not in before] + [q for q in before if q not in after] +
                  [q for q, v in after.items() if q in before and before[q] != v])


def _seal_tree(d):
    """After the scored run: its output dir becomes root-owned and private, so a timed repeat can neither read
    (copy) nor change it.  Returns a reason when the tree holds anything but plain files / dirs (a symlink or a
    hard link could point the scored output at a file a repeat may write) -> no repeats."""
    bad = []
    for root, dirs, files in os.walk(d, followlinks=False):
        for name in dirs + files:
            q = os.path.join(root, name)
            st = os.lstat(q)
            if os.path.islink(q) or not (os.path.isdir(q) or os.path.isfile(q)) or \
                    (os.path.isfile(q) and st.st_nlink != 1):
                bad.append(q)
                continue
            os.lchown(q, 0, 0)
    os.chown(d, 0, 0)
    os.chmod(d, 0o700)
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


def run_solution(timeout_sec, input_dir=HIDDEN, entry=ENTRY):
    """Run `python <entry> <input_dir> <out_dir>` once per session; returns (out_dir, wall_seconds, returncode).
    Several hidden sets: call once per set, e.g. `for d in hidden_sets(): out, sec, rc = run_solution(t, d)`.
    This is the ONLY sanctioned way to execute agent code in the verifier (it drops to `nobody`, locks truth and
    the verifier logs, and kills everything the program left running).
    out_dir and returncode are those of the first (scored) run.  wall_seconds is the median over TIMING_REPEATS
    runs (a failed / timed-out repeat counts as +inf) when: the first run succeeded; the repeats fit the extra
    TIMING_BUDGET_SEC (absolute: each repeat's timeout is the budget left; the pipeline adds the budget to the
    verifier timeout); the scored output and HOME hold only plain files and become root-only 0700 (a repeat cannot
    copy or change them); and no run changes anything else on the filesystem (whole-FS snapshot outside /proc,
    /sys, /dev but /dev/shm; SysV IPC) or writes outputs different from the scored run's.  Otherwise the first
    run's time alone, with the reason in the record (`timing_repeats_skipped`).  The
    wall time is also recorded (not a pass/fail check by itself; tests decide whether to check() it)."""
    key = (entry, input_dir)
    if key in _CACHE:
        return _CACHE[key]
    out = tempfile.mkdtemp(prefix="sol-")
    entry_exists = os.path.isfile(entry)              # calibration tells "no deliverable" from "deliverable failed"
    try:
        sb = _sandbox_prepare(input_dir, out)
    except Exception as e:  # never run untrusted code with truth possibly readable
        _record({"name": "_solution_cli_run", "value": 0.0, "rc": -1, "stderr_tail": f"not run: {e}"[-500:],
                 "threshold": float(timeout_sec), "direction": "<=", "passed": False, "aux": True,
                 "sandboxed": False, "lock_failed": True, "entry": entry, "entry_exists": entry_exists})
        _CACHE[key] = (out, 0.0, -1)
        return _CACHE[key]
    n_rep = max(1, int(TIMING_REPEATS))
    t_snap = time.time()
    keep = ([out, sb[2]["HOME"]] if sb is not None else [out]) + [LOGS]   # LOGS: locked for the sandbox uid
    before = _state_snapshot(sb[0], keep) if sb is not None and n_rep > 1 else None
    _TIMING_SPENT[0] += time.time() - t_snap
    rc, err, sec, left = _run_once(entry, input_dir, out, timeout_sec, sb)
    times, why = [sec], None
    if n_rep > 1 and rc == 0 and not left:
        if sb is None:
            why = "not sandboxed (no root): single run"
        elif _TIMING_SPENT[0] + sec * (n_rep - 1) * 1.2 > TIMING_BUDGET_SEC:
            why = f"first run {sec:.1f}s: repeats would exceed the {TIMING_BUDGET_SEC:.0f}s timing budget"
        else:
            t_snap = time.time()
            why = _seal_tree(out) or _seal_tree(sb[2]["HOME"])
            if why is None:
                ch = _state_changes(before, _state_snapshot(sb[0], keep))
                if ch:
                    why = f"first run changed state a repeat could reuse: {ch[:3]}"
            _TIMING_SPENT[0] += time.time() - t_snap
        if why is None:
            import shutil
            ref_digest = _tree_digest(out)
            for _ in range(n_rep - 1):
                o2 = tempfile.mkdtemp(prefix="sol-")
                h2 = tempfile.mkdtemp(prefix="solhome-")
                os.chown(o2, sb[0], sb[1])
                os.chown(h2, sb[0], sb[1])
                t_snap = time.time()
                b2 = _state_snapshot(sb[0], keep + [o2, h2])
                _TIMING_SPENT[0] += time.time() - t_snap
                t_rep = TIMING_BUDGET_SEC - _TIMING_SPENT[0]          # absolute cap: never past the budget
                if t_rep < sec * 1.2:
                    times, why = [sec], "timing budget exhausted"
                    shutil.rmtree(o2, ignore_errors=True)
                    shutil.rmtree(h2, ignore_errors=True)
                    break
                r2, _e2, s2, left = _run_once(entry, input_dir, o2, min(float(timeout_sec), t_rep), sb, home=h2)
                _TIMING_SPENT[0] += s2
                t_snap = time.time()
                same = r2 == 0 and not left and _tree_digest(o2) == ref_digest
                shutil.rmtree(o2, ignore_errors=True)
                shutil.rmtree(h2, ignore_errors=True)
                ch = _state_changes(b2, _state_snapshot(sb[0], keep)) if not left else ["left processes"]
                _TIMING_SPENT[0] += time.time() - t_snap
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
    med = _median(times)
    _record({"name": "_solution_cli_run", "value": med, "rc": rc, "stderr_tail": err[-500:],
             "threshold": float(timeout_sec), "direction": "<=", "passed": rc == 0, "aux": True,
             "sandboxed": sb is not None, "entry": entry, "entry_exists": entry_exists,
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
