"""Output mutations for the S7 shortcut probes.  Also copied as a standalone script into the verifier sandbox, where
it wraps the reference's solution_cli.py (renamed _probe_reference.py).

Each probe verifier run targets ONE scope: the public deliverables, or a single hidden set.  Every other scope keeps
the reference's answer values, so a rejection can only come from the targeted scope (a verifier that stops at
its first failed check is fine, and a verifier that scores set A but silently skips set B is caught on B).

hidden_garbage: rotate table answers across rows, preserving the detected row identifier. JSON and arrays get numbers zeroed and labels replaced by mode.
one_confident: CSV tables with > 5 data rows keep only the most confident row (score/confidence/prob column, else
  the first row).
Files the probe cannot transform (images, FITS, HDF5, ...) are left untouched; a scope where nothing was
transformed reports count 0. Uncertain table keys are reported separately."""
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


class UncertainTable(ValueError):
    """The probe cannot establish a unique row-to-answer mapping."""


def _number(value):
    try:
        if isinstance(value, int) or (isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip())):
            return int(value)
        if isinstance(value, bytes):
            return _number(value.decode("ascii"))
        return float(value)
    except (ValueError, TypeError):
        return None


def _mode(values):
    counts = Counter(v for v in values if isinstance(v, str) and _number(v) is None)
    return counts.most_common(1)[0][0] if counts else ""


_KEY_NAME = re.compile(r"(?:.*_)?(?:ids?|keys?|identifiers?|index|idx)|name|sample", re.I)


def _unique(values):
    seen = set()
    for value in values:
        if value is None or isinstance(value, (dict, list)) or isinstance(value, str) and not value.strip():
            return False
        if value != value or value in seen:
            return False
        seen.add(value)
    return bool(seen)


def _key_column(columns):
    candidates = [i for i, (name, values) in enumerate(columns)
                  if (i == 0 or _KEY_NAME.fullmatch(name.strip())) and _unique(values)]
    if len(candidates) != 1:
        raise UncertainTable("ambiguous keys" if candidates else "no unique key")
    return candidates[0]


def _same_values(a, b):
    # Containers are produced by the same shape-preserving transform. Compare numeric leaves as scored.
    if isinstance(a, (list, tuple)):
        return all(_same_values(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return all(_same_values(x, b[k]) for k, x in a.items())
    if a == b:
        return True
    x, y = _number(a), _number(b)
    return x is not None and y is not None and (x == y or (x != x and y != y))


def _row_key(rows, header):
    width = len(rows[0])
    if not width or (header and len(header) != width) or any(len(row) != width for row in rows):
        raise UncertainTable("ragged output table")
    return _key_column((header[i] if header else "", (row[i] for row in rows)) for i in range(width))


def _mapping_changed(original, changed, key):
    before = {row[key]: row for row in original}
    after = {row[key]: row for row in changed}
    return before.keys() != after.keys() or any(not _same_values(row, after[k]) for k, row in before.items())


def _rotate_rows(rows, key):
    width = len(rows[0])
    out = [row.copy() for row in rows]
    if len(rows) == 1:
        for i, cell in enumerate(rows[0]):
            if i != key and _number(cell) is not None:
                out[0][i] = "0"
        return out
    for i in range(width):
        if i != key:
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
    if not data or (arm == "one_confident" and len(data) <= 5):
        return False
    key = _row_key(data, header)
    if arm == "one_confident":
        score_names = {"score", "confidence", "prob", "probability", "p"}
        names = [s.strip().lower() for s in header]
        column = next((i for i, name in enumerate(names) if name in score_names), None)
        if column is None:
            column = next((i for i, name in enumerate(names) if score_names.intersection(re.split(r"[^a-z]+", name))),
                          None)

        def confidence(row):
            val = _number(row[column]) if column is not None and column < len(row) else None
            return val if isinstance(val, int) or (val is not None and math.isfinite(val)) else -math.inf
        changed = [max(data, key=confidence)]
    else:
        changed = _rotate_rows(data, key)
    if not _mapping_changed(data, changed, key):
        return False
    with path.open("w", newline="") as stream:
        csv.writer(stream, delimiter=delimiter).writerows([header, *(data if identity else changed)])
    return True


def _json_garbage(value):
    # Reduce labels within each answer column. Preserve row order and the selected key verbatim.
    def column(values):
        mode = _mode(values)
        return [transform(v, mode) for v in values]

    def records(rows, names):
        key = _row_key(rows, names)
        columns = [vs if i == key else column(vs) for i, vs in enumerate(zip(*rows))]
        return list(zip(*columns))

    def transform(v, mode=None):
        if isinstance(v, dict):
            if len(v) > 1 and all(isinstance(xs, list) and xs for xs in v.values()):
                if len({len(xs) for xs in v.values()}) == 1:
                    rows = records(list(zip(*v.values())), list(v))
                    return dict(zip(v, map(list, zip(*rows))))
            return {k: x if _KEY_NAME.fullmatch(k) else transform(x) for k, x in v.items()}
        if isinstance(v, list):
            if v and all(isinstance(row, dict) for row in v):
                names = list(v[0])
                if any(row.keys() != v[0].keys() for row in v):
                    raise UncertainTable("ragged output table")
                rows = records([[row[k] for k in names] for row in v], names)
                return [dict(zip(names, row)) for row in rows]
            if v and all(isinstance(row, list) for row in v):
                return list(map(list, records(v, [])))
            return column(v)
        if _number(v) is not None:
            return "0" if isinstance(v, str) else 0
        return mode if isinstance(v, str) and mode is not None else v

    return transform(value)


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
        if a.ndim != 1:
            raise UncertainTable("expected one-dimensional record array")
        key = _key_column((name, a[name] if a[name].ndim == 1 else ()) for name in a.dtype.names)
        for i, name in enumerate(a.dtype.names):
            out[name] = a[name] if i == key else _array_garbage(a[name])
    elif a.ndim == 2 and a.size:
        key = _key_column(("", col) for col in a.T)
        for i in range(a.shape[1]):
            out[:, i] = a[:, i] if i == key else _array_garbage(a[:, i])
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
                original = {name: data[name] for name in data.files}
                key_column = None
                vectors = {name: a for name, a in original.items() if a.ndim == 1 and not a.dtype.names}
                if len(original) > 1 and vectors and all(a.ndim for a in original.values()):
                    if len({len(a) for a in original.values()}) == 1:
                        key_column = _key_column((name, vectors.get(name, ())) for name in original)
                arrays = {}
                changed = False
                for i, (key, a) in enumerate(original.items()):
                    named_key = key in vectors and _KEY_NAME.fullmatch(key) and _unique(a)
                    candidate = a if i == key_column or named_key else _array_garbage(a)
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
        key = _row_key(rows, header)
        changed = _rotate_rows(rows, key)
        if not _mapping_changed(rows, changed, key):
            return 0
        replaced = [" ".join(row) for row in (rows if identity else changed)]
        path.write_text("\n".join(lines[:start] + replaced) + "\n")
        return 1
    raise Unsupported(f"unsupported output format: {path.name}")


def mutate(root, arm, skipped=None, *, identity=False, uncertain=None):
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
        except UncertainTable as e:
            if uncertain is not None:
                uncertain.append(f"{path.name}: {e}")
        except Unsupported as e:
            if skipped is not None:
                skipped.append(str(e))
    return count


def mutation_report(root, arm, identity=False):
    uncertain = []
    count = mutate(root, arm, identity=identity, uncertain=uncertain)
    return count, [reason[:120] for reason in uncertain[:2]]


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
        # The timer includes this wrapper on every set. Do the same work on non-target sets as the shared control.
        count, uncertain = mutation_report(Path(tmp), arm, identity=identity or name != target)
        shutil.copytree(tmp, dest, dirs_exist_ok=True)
    print("TBSGEN_PROBE " + json.dumps([name, count, identity or name == target, uncertain]), file=sys.stderr)
