"""Standard-library-only probe executed inside the offline task image by taskimage.inspect."""
import json
import os
import platform
import pwd
import re
import shutil
import subprocess
import sys
from pathlib import Path


def installed(source, output):
    if source in ("dpkg", "R"):
        pairs = [line.split("\t") for line in output.splitlines()]
    else:
        pairs = [(p["name"], p["version"]) for p in json.loads(output)]
    return [dict(name=name, version=version, source=source) for name, version in pairs]


def output(*argv):
    return subprocess.check_output(argv, text=True)


def inventory():
    records = []
    seen = set()
    for directory in os.get_exec_path():
        for path in sorted(Path(directory).glob("python*")):
            if not re.fullmatch(r"python(?:[23](?:\.\d+)?)?", path.name) or not os.access(path, os.X_OK):
                continue
            # Aliases in a single bin/ are common. A venv can symlink to the system python.
            identity = (path.parent.resolve(), path.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            try:
                packages = output(str(path), "-m", "pip", "list", "--format", "json")
            except subprocess.CalledProcessError:
                # Some distribution pythons omit pip. Query that interpreter's installed metadata directly.
                script = ("import json,importlib.metadata as m; "
                          "print(json.dumps([dict(name=d.metadata['Name'], version=d.version) for d in m.distributions()]))")
                packages = output(str(path), "-c", script)
            records.extend(installed(str(path), packages))
    for manager in ("conda", "mamba", "micromamba"):
        if shutil.which(manager):
            envs = json.loads(output(manager, "env", "list", "--json"))["envs"]
            for env in envs:
                records.extend(installed(f"{manager}:{env}", output(manager, "list", "--json", "--prefix", env)))
            break
    if shutil.which("dpkg-query"):
        lines = output("dpkg-query", "-W", "-f=${db:Status-Status}\t${Package}\t${Version}\n")
        records.extend(installed("dpkg", "\n".join(s[10:] for s in lines.splitlines() if s.startswith("installed\t"))))
    if shutil.which("R"):
        expr = 'p <- installed.packages(); write.table(p[,c("Package","Version")],row.names=FALSE,col.names=FALSE,quote=FALSE,sep="\\t")'
        records.extend(installed("R", output("R", "--vanilla", "--slave", "-e", expr)))
    return sorted(records, key=lambda p: (p["source"], p["name"], p["version"]))


def runtime_paths(interpreter, sys_path, path):
    paths = set()
    for entry in (interpreter, *sys_path, *path):
        p = Path(entry or ".").absolute()
        pending = [p]
        while pending:
            candidate = pending.pop()
            for ancestor in (candidate, *candidate.parents):
                if str(ancestor) in paths:
                    continue
                paths.add(str(ancestor))
                if ancestor.is_symlink():
                    pending.append(ancestor.parent / ancestor.readlink())
            real = candidate.resolve()
            if str(real) not in paths:
                pending.append(real)
    return sorted(paths)


def writable_runtime(paths, nobody):
    # Drop supplementary groups too, exactly as the verifier does for solution_cli.py.
    script = ("import json,os,sys; print(json.dumps([p for p in json.loads(sys.argv[1]) "
              "if os.access(p, os.W_OK) or (os.path.exists(p) and os.stat(p).st_uid == os.getuid())]))")
    return json.loads(subprocess.check_output(
        [sys.executable, "-I", "-S", "-c", script, json.dumps(paths)], text=True,
        user=nobody.pw_uid, group=nobody.pw_gid, extra_groups=[]))


def main(payload):
    floor = []
    if sys.version_info < (3, 10) or os.getcwd() != "/app" or os.geteuid() != 0:
        floor.append("requires python >=3.10, WORKDIR /app and root")
    for path in (Path("/app/data"), Path("/app/output")):
        if not path.is_dir() or any(path.iterdir()):
            floor.append(f"expected empty directory {path}")
    try:
        import pytest  # noqa: F401
        if pwd.getpwnam("nobody").pw_uid == 0 or not output("id", "nobody"):
            raise ValueError("invalid nobody user")
        tests = Path("/tests")
        tests.mkdir(exist_ok=True)
        (tests / "_metrics.py").write_text(payload["helper"])
        (tests / "test.sh").write_text(payload["script"])
        (tests / "hidden").mkdir(exist_ok=True)
        Path("/app/output/solution_cli.py").write_text(
            "import os,pwd\nassert os.getuid() == pwd.getpwnam('nobody').pw_uid\n")
        (tests / "test_outputs.py").write_text(
            "import json,os,sys,_metrics\ndef test_verifier():\n    _metrics.TIMING_REPEATS = 1\n"
            "    with open('/tests/runtime.json', 'w') as f:\n"
            "        modules = [m.__file__ for m in sys.modules.copy().values() if getattr(m, '__file__', None)]\n"
            "        paths = (sys.executable, *sys.path, *os.get_exec_path(), *modules)\n"
            "        json.dump([os.path.abspath(p) for p in paths], f)\n"
            "    assert _metrics.run_solution(10)[2] == 0\n    _metrics.check('smoke', 1, 1, '>=')\n")
        subprocess.run(["bash", "/tests/test.sh"], check=True, stdout=sys.stderr)
        if Path("/logs/verifier/reward.txt").read_text().strip() != "1":
            raise ValueError("test.sh did not produce a passing reward")
    except Exception as e:
        floor.append(str(e))
    if not floor:
        try:
            pytest_paths = json.loads(Path("/tests/runtime.json").read_text())
            paths = runtime_paths(sys.executable, [*sys.path, *pytest_paths], os.get_exec_path())
            writable = writable_runtime(paths, pwd.getpwnam("nobody"))
            if writable:
                floor.append(f"nobody can write verifier runtime: {writable}")
        except Exception as e:
            floor.append(f"cannot check verifier runtime: {e}")
    imports = []
    for module in payload["imports"]:
        try:
            # An import can write to stdout. Keep the probe's JSON channel separate.
            output(sys.executable, "-c", "import importlib,sys; importlib.import_module(sys.argv[1])", module)
        except subprocess.CalledProcessError:
            imports.append(module)
    print(json.dumps(dict(python=platform.python_version(), floor_errors=floor, import_errors=imports,
                          inventory=inventory())))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
