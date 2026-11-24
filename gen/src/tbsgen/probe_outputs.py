"""Data mutations for S7. Also copied as a standalone script into the verifier sandbox."""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


class Unsupported(ValueError):
    """An output this probe cannot transform: the probe reports n/a (S9R sees it) instead of blocking the task."""


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _mode(values):
    counts = Counter(v for v in values if isinstance(v, str) and _number(v) is None)
    return counts.most_common(1)[0][0] if counts else ""


def _garbage_rows(rows):
    modes = [_mode(row[i] for row in rows if i < len(row)) for i in range(max(map(len, rows), default=0))]
    return [["0" if _number(cell) is not None else modes[i] for i, cell in enumerate(row)] for row in rows]


def _table(path, arm, delimiter):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))
    if not rows:
        return False
    header, data = rows[0], rows[1:]
    if arm == "one_confident":
        data = [row for row in data if row]
        if len(data) <= 5:
            return False
        score_names = {"score", "confidence", "prob", "probability", "p"}
        names = [s.strip().lower() for s in header]
        column = next((i for i, name in enumerate(names) if name in score_names), None)
        if column is None:
            column = next((i for i, name in enumerate(names) if score_names.intersection(re.split(r"[^a-z]+", name))),
                          None)

        def confidence(row):
            val = _number(row[column]) if column is not None and column < len(row) else None
            return val if val is not None and math.isfinite(val) else -math.inf
        data = [max(data, key=confidence)]
    else:
        data = _garbage_rows(data)
    with path.open("w", newline="") as stream:
        csv.writer(stream, delimiter=delimiter).writerows([header, *data])
    return True


def _json_garbage(value):
    # Group leaves by key path (ignoring array indices), so labels in each column use their own mode.
    columns = {}

    def visit(v, key, replace):
        if isinstance(v, dict):
            return {k: visit(x, key + (k,), replace) for k, x in v.items()}
        if isinstance(v, list):
            if v and all(isinstance(row, list) for row in v):
                return [[visit(cell, key + (i,), replace) for i, cell in enumerate(row)] for row in v]
            return [visit(x, key, replace) for x in v]
        if _number(v) is not None:
            return "0" if isinstance(v, str) else 0
        if isinstance(v, str):
            if replace:
                return columns[key].most_common(1)[0][0]
            columns.setdefault(key, Counter())[v] += 1
        return v
    visit(value, (), False)
    return visit(value, (), True)


def _array_garbage(a):
    import numpy as np
    out = np.zeros_like(a)
    if a.dtype.names:
        for name in a.dtype.names:
            out[name] = _array_garbage(a[name])
    elif a.dtype.kind in "SU" and a.size:
        for index in np.ndindex(a.shape[1:]):
            col = (slice(None),) + index if a.ndim else ()
            values = np.atleast_1d(a[col]).astype(str)
            mode = _mode(values)
            cells = ["0" if _number(value) is not None else mode for value in values]
            out[col] = cells if a.ndim else cells[0]
    return out


def mutate(root, arm):
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("linked probe output")
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext == ".csv" or (arm == "hidden_garbage" and ext == ".tsv"):
            count += _table(path, arm, "," if ext == ".csv" else "\t")
        elif arm == "one_confident":
            continue
        elif ext == ".json":
            path.write_text(json.dumps(_json_garbage(json.loads(path.read_text()))))
            count += 1
        elif ext in (".npy", ".npz"):
            import numpy as np
            if ext == ".npy":
                np.save(path, _array_garbage(np.load(path, allow_pickle=False)))
            else:
                with np.load(path, allow_pickle=False) as data:
                    arrays = {key: _array_garbage(data[key]) for key in data.files}
                np.savez(path, **arrays)
            count += 1
        elif ext == ".txt":
            lines = path.read_text().splitlines()
            # Whitespace-delimited numeric tables, optionally preceded by column names / comments.
            start = next((i for i, line in enumerate(lines) if not line.startswith("#") and
                          any(_number(v) is not None for v in line.split())), None)
            if start is None:
                raise Unsupported(f"cannot identify table in {path.name}")
            rows = [line.split() for line in lines[start:]]
            replaced = [" ".join(row) for row in _garbage_rows(rows)]
            path.write_text("\n".join(lines[:start] + replaced) + "\n")
            count += 1
        else:
            raise Unsupported(f"unsupported hidden output format: {path.name}")
    return count


def hidden_main(arm):
    # Only this unprivileged CLI executes the reference. The verifier supplies a different input for each call.
    input_dir, dest = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(Path(__file__).with_name("_probe_reference.py")), input_dir, tmp],
                       check=True)
        try:
            count = mutate(Path(tmp), arm)
        except Unsupported as e:
            print("TBSGEN_PROBE_NA " + json.dumps([Path(input_dir).name, str(e)]), file=sys.stderr)
            return
        shutil.copytree(tmp, dest, dirs_exist_ok=True)
    print("TBSGEN_PROBE " + json.dumps([Path(input_dir).name, count]), file=sys.stderr)
