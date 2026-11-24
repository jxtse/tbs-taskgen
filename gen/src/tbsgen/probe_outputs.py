"""Output mutations for the S7 shortcut probes.  Also copied as a standalone script into the verifier sandbox, where
it wraps the reference's solution_cli.py (renamed _probe_reference.py).

Each probe verifier run targets ONE scope: the public deliverables, or a single hidden set.  Every other scope gets
the reference's untouched output, so a rejection can only come from the targeted scope (a verifier that stops at
its first failed check is fine, and a verifier that scores set A but silently skips set B is caught on B).

hidden_garbage: tables with >= 3 data rows keep their first (key) column and have every other column rotated by
  half the rows -- ids and value distributions stay valid, the row-to-answer association is destroyed; shorter
  tables, JSON and arrays get numbers zeroed and labels replaced by the column mode.
one_confident: CSV tables with > 5 data rows keep only the most confident row (score/confidence/prob column, else
  the first row).
Files the probe cannot transform (images, FITS, HDF5, ...) are left untouched; a scope where nothing was
transformed reports count 0 (= no evidence, n/a for that scope)."""
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
    """An output this probe cannot transform."""


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


def _rotate_rows(rows):
    """Keep column 0, rotate column i (i >= 1) by its own offset (half the rows + i - 1): id-like columns stay valid
    permutations, every row-to-answer association is destroyed whichever column holds the key."""
    n = len(rows)
    width = max(map(len, rows), default=0)
    cols = [[row[i] if i < len(row) else "" for row in rows] for i in range(width)]
    for i in range(1 if width > 1 else 0, width):
        k = (n // 2 + max(i - 1, 0)) % n or 1
        cols[i] = cols[i][k:] + cols[i][:k]
    return [[cols[i][r] for i in range(width)] for r in range(n)]


def _table(path, arm, delimiter):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))
    if not rows:
        return False
    header, data = rows[0], [row for row in rows[1:] if row]
    if arm == "one_confident":
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
        if not data:
            return False
        data = _rotate_rows(data) if len(data) >= 3 else _garbage_rows(data)
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


def _mutate_file(path, arm):
    ext = path.suffix.lower()
    if ext == ".csv" or (arm == "hidden_garbage" and ext == ".tsv"):
        return int(_table(path, arm, "," if ext == ".csv" else "\t"))
    if arm == "one_confident":
        return 0
    if ext == ".json":
        path.write_text(json.dumps(_json_garbage(json.loads(path.read_text()))))
        return 1
    if ext in (".npy", ".npz"):
        import numpy as np
        if ext == ".npy":
            np.save(path, _array_garbage(np.load(path, allow_pickle=False)))
        else:
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: _array_garbage(data[key]) for key in data.files}
            np.savez(path, **arrays)
        return 1
    if ext == ".txt":
        lines = path.read_text().splitlines()
        # Whitespace-delimited numeric tables, optionally preceded by column names / comments.
        start = next((i for i, line in enumerate(lines) if not line.startswith("#") and
                      any(_number(v) is not None for v in line.split())), None)
        if start is None:
            raise Unsupported(f"cannot identify table in {path.name}")
        rows = [line.split() for line in lines[start:]]
        replaced = [" ".join(row) for row in _garbage_rows(rows)]
        path.write_text("\n".join(lines[:start] + replaced) + "\n")
        return 1
    raise Unsupported(f"unsupported output format: {path.name}")


def mutate(root, arm, skipped=None):
    """Transform every supported file under root; unsupported files stay as they are (listed in `skipped`).
    Returns the number of files transformed."""
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("linked probe output")
        if not path.is_file():
            continue
        try:
            count += _mutate_file(path, arm)
        except Unsupported as e:
            if skipped is not None:
                skipped.append(str(e))
    return count


def hidden_main(arm, target):
    """Wrapper CLI: run the reference into a temp dir; transform its output only when this call's input set is
    `target` (pass the reference output through otherwise).  Only this unprivileged CLI executes the reference."""
    input_dir, dest = sys.argv[1:]
    name = Path(input_dir).name
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(Path(__file__).with_name("_probe_reference.py")), input_dir, tmp],
                       check=True)
        count, skipped = 0, []
        if name == target:
            count = mutate(Path(tmp), arm, skipped)
        shutil.copytree(tmp, dest, dirs_exist_ok=True)
    print("TBSGEN_PROBE " + json.dumps([name, count, name == target, skipped[:3]]), file=sys.stderr)
