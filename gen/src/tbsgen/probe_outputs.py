"""Output mutations for the S7 shortcut probes.  Also copied as a standalone script into the verifier sandbox, where
it wraps the reference's solution_cli.py (renamed _probe_reference.py).

Each probe verifier run targets ONE scope: the public deliverables, or a single hidden set.  Every other scope keeps
the reference's answer values, so a rejection can only come from the targeted scope (a verifier that stops at
its first failed check is fine, and a verifier that scores set A but silently skips set B is caught on B).

hidden_garbage: rotate table answers across rows, preserving unique, named row identifiers. JSON and arrays get numbers zeroed and labels replaced by mode.
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


_KEY_NAME = re.compile(r"(?:(?:sample|item|row|cell|case|record|subject|obs|site|trial)_)?"
                       r"(?:ids?|keys?|identifiers?|index|idx)", re.I)


def _is_key(name, values):
    if not _KEY_NAME.fullmatch(name.strip()):
        return False
    seen = set()
    for value in values:
        if value in seen:
            return False
        seen.add(value)
    return bool(seen)


def _same_values(a, b):
    # Containers are produced by the same shape-preserving transform. Compare numeric leaves as scored.
    if isinstance(a, (list, tuple)):
        return all(_same_values(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return all(_same_values(x, b[k]) for k, x in a.items())
    if a == b:
        return True
    x, y = _number(a), _number(b)
    return x is not None and y is not None and (x == y or (math.isnan(x) and math.isnan(y)))


def _rotate_rows(rows, header=()):
    if not rows:
        return rows
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise Unsupported("ragged output table")
    keys = {i for i, name in enumerate(header) if i < width and _is_key(name, (r[i] for r in rows))}
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


def _table(path, arm, delimiter, identity):
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
        changed = [max(data, key=confidence)]
    else:
        changed = _rotate_rows(data, header)
        if _same_values(data, changed):
            return False
    with path.open("w", newline="") as stream:
        csv.writer(stream, delimiter=delimiter).writerows([header, *(data if identity else changed)])
    return True


def _json_garbage(value):
    # Group leaves by key path (ignoring array indices), so labels in each column use their own mode.
    columns = {}
    key_columns = {}

    def visit(v, key, replace):
        if isinstance(v, dict):
            return {k: visit(x, key + (k,), replace) for k, x in v.items()}
        if isinstance(v, list):
            if v and all(isinstance(row, list) for row in v):
                return [[visit(cell, key + (i,), replace) for i, cell in enumerate(row)] for row in v]
            return [visit(x, key, replace) for x in v]
        if key and isinstance(key[-1], str) and _KEY_NAME.fullmatch(key[-1].strip()):
            if not replace:
                key_columns.setdefault(key, []).append(v)
            elif key in keys:
                return v
        if _number(v) is not None:
            return "0" if isinstance(v, str) else 0
        if isinstance(v, str):
            if replace:
                return columns[key].most_common(1)[0][0]
            columns.setdefault(key, Counter())[v] += 1
        return v
    visit(value, (), False)
    keys = {k for k, vs in key_columns.items() if _is_key(k[-1], vs)}
    return visit(value, (), True)


def _array_key(name, a):
    return a.ndim == 1 and not a.dtype.names and _is_key(name, a)


def _same_array(a, b):
    import numpy as np
    if a.dtype.names:
        return all(_same_array(a[name], b[name]) for name in a.dtype.names)
    if a.dtype.kind in "SU":
        return all(_same_values(x, y) for x, y in zip(a.flat, b.flat))
    return np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")


def _array_garbage(a):
    import numpy as np
    out = np.zeros_like(a)
    if a.dtype.names:
        for name in a.dtype.names:
            out[name] = a[name] if _array_key(name, a[name]) else _array_garbage(a[name])
    elif a.dtype.kind in "SU" and a.size:
        for index in np.ndindex(a.shape[1:]):
            col = (slice(None),) + index if a.ndim else ()
            values = np.atleast_1d(a[col]).astype(str)
            mode = _mode(values)
            cells = ["0" if _number(value) is not None else mode for value in values]
            out[col] = cells if a.ndim else cells[0]
    return out


def _mutate_file(path, arm, identity):
    ext = path.suffix.lower()
    if ext == ".csv" or (arm == "hidden_garbage" and ext == ".tsv"):
        return int(_table(path, arm, "," if ext == ".csv" else "\t", identity))
    if arm == "one_confident":
        return 0
    if ext == ".json":
        original = json.loads(path.read_text())
        changed = _json_garbage(original)
        if _same_values(original, changed):
            return 0
        path.write_text(json.dumps(original if identity else changed))
        return 1
    if ext in (".npy", ".npz"):
        import numpy as np
        if ext == ".npy":
            a = np.load(path, allow_pickle=False)
            changed = _array_garbage(a)
            if _same_array(a, changed):
                return 0
            with path.open("wb") as stream:
                np.save(stream, a if identity else changed)
        else:
            with np.load(path, allow_pickle=False) as data:
                arrays = {}
                changed = False
                for key in data.files:
                    a = data[key]
                    candidate = a if _array_key(key, a) else _array_garbage(a)
                    changed |= not _same_array(a, candidate)
                    arrays[key] = a if identity else candidate
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
        if _same_values(rows, changed):
            return 0
        replaced = [" ".join(row) for row in (rows if identity else changed)]
        path.write_text("\n".join(lines[:start] + replaced) + "\n")
        return 1
    raise Unsupported(f"unsupported output format: {path.name}")


def mutate(root, arm, skipped=None, *, identity=False):
    """Transform every supported file under root; unsupported files stay as they are (listed in `skipped`).
    An identity control computes the same candidate and serializes the original values for each applicable file.
    Returns the number of applicable files."""
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("linked probe output")
        if not path.is_file():
            continue
        try:
            count += _mutate_file(path, arm, identity)
        except Unsupported as e:
            if skipped is not None:
                skipped.append(str(e))
    return count


def hidden_main(arm, target, identity=False):
    """Run the unprivileged reference into a temp directory. Mutate only the trial target, or exercise
    the identity transform on every set in a control run."""
    account = pwd.getpwnam("nobody")
    if os.geteuid() == 0 or (os.geteuid(), os.getegid()) != (account.pw_uid, account.pw_gid):
        raise PermissionError("probe reference requires the sandbox nobody account")
    input_dir, dest = sys.argv[1:]
    name = Path(input_dir).name
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(Path(__file__).with_name("_probe_reference.py")), input_dir, tmp],
                       check=True)
        skipped = []
        # The timer includes this wrapper on every set. Do the same work on non-target sets as the shared control.
        count = mutate(Path(tmp), arm, skipped, identity=identity or name != target)
        shutil.copytree(tmp, dest, dirs_exist_ok=True)
    print("TBSGEN_PROBE " + json.dumps([name, count, identity or name == target, skipped[:3]]), file=sys.stderr)
