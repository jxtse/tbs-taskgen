"""Collect the grader and its local Python helpers without importing task code."""

import ast
from pathlib import Path


def _imports(path, root):
    try:
        tree = ast.parse(path.read_text("utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            base = root
        elif isinstance(node, ast.ImportFrom):
            base = path.parent if node.level else root
            for _ in range(1, node.level):
                base = base.parent
            modules = [node.module] if node.module else []
            modules.extend(".".join(filter(None, (node.module, alias.name))) for alias in node.names)
        else:
            continue
        for module in modules:
            parts = module.split(".")
            if base == root and parts[0] == "tests":
                parts = parts[1:]
            directory = base
            for part in parts:
                directory /= part
                yield directory / "__init__.py"
            if parts:
                yield directory.with_suffix(".py")


def grader_files(pkg):
    root = Path(pkg) / "tests"
    resolved_root = root.resolve()
    paths = [root / name for name in ("test_outputs.py", "measure.py", "test.sh") if (root / name).is_file()]
    if not paths:
        return []
    metrics = root / "_metrics.py"
    if metrics.is_file() and metrics.resolve().is_relative_to(resolved_root):
        paths.append(metrics)
    seen = set(paths)
    for path in paths:
        if path.suffix == ".py":
            for helper in _imports(path, root):
                if helper not in seen and helper.is_file() and helper.resolve().is_relative_to(resolved_root):
                    seen.add(helper)
                    paths.append(helper)
    return paths
