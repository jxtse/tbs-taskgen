"""Output mutations for the S7 shortcut probes.  Also copied as a standalone script into the verifier sandbox, where
it wraps the reference's solution_cli.py (renamed _probe_reference.py).

Each probe verifier run targets ONE scope: the public deliverables, or a single hidden set.  Every other scope gets
the reference's untouched output, so a rejection can only come from the targeted scope (a verifier that stops at
its first failed check is fine, and a verifier that scores set A but silently skips set B is caught on B).

hidden_garbage: rotate table answers across rows, preserving ID/key columns (by convention the first column
  in a multi-column table without named keys). JSON and arrays get numbers zeroed and labels replaced by mode.
one_confident: CSV tables with > 5 data rows keep only the most confident row (score/confidence/prob column, else
  the first row).
Files the probe cannot transform (images, FITS, HDF5, ...) are left untouched; a scope where nothing was
transformed reports count 0 (= no evidence, n/a for that scope)."""
import csv
import json
import math
import os
import pwd
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


def _is_key(name):
    return bool(re.search(r"(?:^|[_ -])(?:ids?|keys?|identifiers?|index|idx)(?:$|[_ -])", name.strip(), re.I))


def _rotate_rows(rows, header=()):
    if not rows:
        return rows
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise Unsupported("ragged output table")
    keys = {i for i, name in enumerate(header) if _is_key(name)}
    if not keys and width > 1:
        keys.add(0)
    out = [row.copy() for row in rows]
    if len(rows) == 1:
        for i, cell in enumerate(rows[0]):
            if i not in keys and _number(cell) is not None:
                out[0][i] = "0"
        return out
    for i in range(width):
        if i not in keys:
            offset = (len(rows) // 2 + max(i - 1, 0)) % len(rows) or 1
            for r, row in enumerate(out):
                row[i] = rows[(r + offset) % len(rows)][i]
    return out


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
        changed = _rotate_rows(data, header)
        if changed == data:
            return False
        data = changed
    with path.open("w", newline="") as stream:
        csv.writer(stream, delimiter=delimiter).writerows([header, *data])
    return True


def _json_garbage(value):
    # Group leaves by key path (ignoring array indices), so labels in each column use their own mode.
    columns = {}

    def visit(v, key, replace):
        if isinstance(v, dict):
            return {k: x if _is_key(k) else visit(x, key + (k,), replace) for k, x in v.items()}
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
            out[name] = a[name] if _is_key(name) else _array_garbage(a[name])
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
        original = json.loads(path.read_text())
        changed = _json_garbage(original)
        if original == changed:
            return 0
        path.write_text(json.dumps(changed))
        return 1
    if ext in (".npy", ".npz"):
        import numpy as np
        if ext == ".npy":
            a = np.load(path, allow_pickle=False)
            changed = _array_garbage(a)
            if a.tobytes() == changed.tobytes():
                return 0
            with path.open("wb") as stream:
                np.save(stream, changed)
        else:
            with np.load(path, allow_pickle=False) as data:
                arrays = {}
                changed = False
                for key in data.files:
                    a = data[key]
                    arrays[key] = a if _is_key(key) else _array_garbage(a)
                    changed |= a.tobytes() != arrays[key].tobytes()
            if not changed:
                return 0
            with path.open("wb") as stream:
                np.savez(stream, **arrays)
        return 1
    if ext == ".txt":
        lines = path.read_text().splitlines()
        # Whitespace-delimited numeric tables, optionally preceded by column names / comments.
        start = next((i for i, line in enumerate(lines) if not line.startswith("#") and
                      any(_number(v) is not None for v in line.split())), None)
        if start is None:
            raise Unsupported(f"cannot identify table in {path.name}")
        rows = [line.split() for line in lines[start:]]
        header = lines[start - 1].lstrip("#").split() if start else []
        changed = _rotate_rows(rows, header)
        if changed == rows:
            return 0
        replaced = [" ".join(row) for row in changed]
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
    account = pwd.getpwnam("nobody")
    if os.geteuid() == 0 or (os.geteuid(), os.getegid()) != (account.pw_uid, account.pw_gid):
        raise PermissionError("probe reference requires the sandbox nobody account")
    input_dir, dest = sys.argv[1:]
    name = Path(input_dir).name
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(Path(__file__).with_name("_probe_reference.py")), input_dir, tmp],
                       check=True)
        count, skipped = 0, []
        if name == target and arm != "identity":
            count = mutate(Path(tmp), arm, skipped)
        shutil.copytree(tmp, dest, dirs_exist_ok=True)
    print("TBSGEN_PROBE " + json.dumps([name, count, name == target, skipped[:3]]), file=sys.stderr)
