"""L3 introspection probe.  This file is copied into a throwaway container and executed there
(`python /probe/introspect_probe.py /data /out.json`) with --network none, read-only data mount.

It imports every *.py under the public data dir, instantiates each public class with the argument patterns
public task code uses (data_dir, problem_id, log_dir) for every problem id found in public JSON/file names,
and walks the resulting objects: instance __dict__ (private included), class attributes, closures, module
globals, numpy arrays, nested dicts/lists.  Numbers are emitted as named leaves; the host side runs the L2
numeric scan on them against tests/truth.  Stdlib + numpy only.
"""
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
import types

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

LEAVES = []
MAX = 20000


def emit(name, v):
    if len(LEAVES) >= MAX:
        return
    try:
        if np is not None and isinstance(v, np.ndarray):
            if v.dtype.kind in "fiub" and v.size:
                LEAVES.append([name, v.astype(float).ravel()[:200000].tolist()])
            return
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            LEAVES.append([name, [float(v)]])
    except Exception:
        pass


def walk(name, v, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if depth > 5 or id(v) in seen or len(LEAVES) >= MAX:
        return
    seen.add(id(v))
    if np is not None and isinstance(v, np.ndarray) or isinstance(v, (int, float)):
        emit(name, v)
        return
    if isinstance(v, (str, bytes, type(None))):
        return
    if isinstance(v, dict):
        for k, x in list(v.items())[:2000]:
            walk(f"{name}.{k}", x, depth + 1, seen)
        return
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            LEAVES.append([name, [float(x) for x in v]])
        else:
            for i, x in enumerate(v[:2000]):
                walk(f"{name}[{i}]", x, depth + 1, seen)
        return
    if isinstance(v, types.FunctionType):
        for i, c in enumerate(v.__closure__ or ()):
            try:
                walk(f"{name}.<closure{i}>", c.cell_contents, depth + 1, seen)
            except ValueError:
                pass
        return
    if isinstance(v, (types.ModuleType, type)):
        return
    d = getattr(v, "__dict__", None)
    if isinstance(d, dict):
        walk(name, d, depth + 1, seen)
    for slot in getattr(type(v), "__slots__", ()) or ():
        try:
            walk(f"{name}.{slot}", getattr(v, slot), depth + 1, seen)
        except Exception:
            pass


def problem_ids(data):
    ids = set()
    for root, _, files in os.walk(data):
        for f in files:
            stem = f.split(".")[0]
            if re.fullmatch(r"[A-Za-z]{0,3}\d{1,4}", stem):
                ids.add(stem)
            if f.endswith(".json"):
                try:
                    o = json.load(open(os.path.join(root, f)))
                except Exception:
                    continue
                stack = [o]
                while stack:
                    x = stack.pop()
                    if isinstance(x, dict):
                        for k in ("problem_id", "id", "pid", "name"):
                            if isinstance(x.get(k), str) and len(x[k]) <= 16:
                                ids.add(x[k])
                        if all(isinstance(k, str) and re.fullmatch(r"[A-Za-z]{0,3}\d{1,4}", k) for k in x) and x:
                            ids.update(x.keys())
                        stack.extend(x.values())
                    elif isinstance(x, list):
                        stack.extend(x)
    return sorted(ids)[:64]


def main(data, out):
    sys.path.insert(0, data)
    os.chdir(tempfile.mkdtemp())
    log = tempfile.mkdtemp()
    pids = problem_ids(data)
    report = dict(modules=[], instances=0, errors=[], problem_ids=pids)
    for root, _, files in os.walk(data):
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            mname = "pub_" + re.sub(r"\W", "_", os.path.relpath(path, data)[:-3])
            try:
                spec = importlib.util.spec_from_file_location(mname, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mname] = mod
                spec.loader.exec_module(mod)
            except BaseException as e:  # noqa: BLE001
                report["errors"].append(f"import {f}: {type(e).__name__}: {e}"[:300])
                continue
            report["modules"].append(f)
            for k, v in vars(mod).items():
                if not isinstance(v, (types.ModuleType, type)) and not callable(v):
                    walk(f"{f}:{k}", v)
            for cname, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mname or issubclass(cls, BaseException):
                    continue
                walk(f"{f}:{cname}.<class>", {k: v for k, v in vars(cls).items() if not k.startswith("__")})
                for pid in pids:
                    obj = None
                    for args in ((data, pid, log), (data, pid), (pid, data, log), (pid, data), (pid,)):
                        try:
                            obj = cls(*args)
                            break
                        except BaseException:  # noqa: BLE001
                            continue
                    if obj is not None:
                        report["instances"] += 1
                        walk(f"{f}:{cname}[{pid}]", obj)
    report["leaves"] = LEAVES
    json.dump(report, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
