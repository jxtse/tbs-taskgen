"""Remote proof for legacy fetch receipts. Also runs via stdin on the old worker."""
import fcntl
import io
import json
import sys
from pathlib import Path

from tbsgen.remote import cmd_cleanup, cmd_status


def check(workdir, job, stage):
    root = Path(workdir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "jobs" / job
    with (root / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(error="another worker holds the host lock")
        cleaned = False
        if directory.exists():
            if stage != "collected":
                return dict(error="remote job directory exists")
            # cmd_status reads the holder's job name. This check now holds the lock.
            lock.truncate(0)
            buf = io.StringIO()
            cmd_status(workdir, job, buf)
            if json.loads(buf.getvalue()).get("state") not in ("done", "failed", "dead"):
                return dict(error="remote job is not finished")
            # Older workers only take a job-local lock.
            with (directory / "worker.lock").open("a+") as local:
                try:
                    fcntl.flock(local, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return dict(error="another worker holds the job lock")
                cmd_cleanup(workdir, job, io.StringIO())
                cleaned = True
        try:
            directory.lstat()
        except FileNotFoundError:
            return dict(job=job, absent=True, cleaned=cleaned)
        return dict(error="remote job directory still exists")


if __name__ == "__main__":
    print(json.dumps(check(*sys.argv[1:])))
