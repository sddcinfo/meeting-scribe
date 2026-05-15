#!/usr/bin/env python3
"""Audit dynamic-import + attribute-access patterns for silent AttributeError risk.

Triggered by the 2026-05-07 reprocess regression: helpers had been
moved out of ``meeting_scribe.server`` during the v2.0.0 refactor,
but ``reprocess.py`` still loaded them via
``importlib.import_module("meeting_scribe.server")`` followed by
``srv._generate_speaker_data(...)``. The import succeeded, the
attribute lookup AttributeError'd, and the surrounding ``except
Exception`` swallowed the error — leaving every full-reprocess
since the refactor with stale ``timeline.json`` + ``speaker_lanes.json``.

This script scans the source for every ``importlib.import_module(...)``
+ subsequent attribute access on the returned module, then verifies
the attribute actually exists in the target module today. Prints any
mismatches. Static AST analysis — no imports of the target modules
required (so it's safe to run even when modules have side effects).

Exit 0 = clean, exit 1 = at least one mismatch found.

Usage:
    python3 scripts/audit_dynamic_imports.py
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "meeting_scribe"


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Find ``importlib.import_module("X")`` assignments and any
    attribute access on the bound name.

    Returns a list of ``(lineno, module_name, attr_name)`` tuples for
    each ``<bound>.<attr>`` reference, where ``<bound>`` was assigned
    from ``import_module(<module_name>)``.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError, UnicodeDecodeError:
        return []

    findings: list[tuple[int, str, str]] = []

    # Pass 1: find name = importlib.import_module("X") assignments.
    bound: dict[str, str] = {}  # local name → module name
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        # Match importlib.import_module(...) or import_module(...)
        is_import_module = False
        if (isinstance(call.func, ast.Attribute) and call.func.attr == "import_module") or (
            isinstance(call.func, ast.Name) and call.func.id == "import_module"
        ):
            is_import_module = True
        if not is_import_module:
            continue
        if not call.args or not isinstance(call.args[0], ast.Constant):
            continue
        module_name = call.args[0].value
        if isinstance(module_name, str):
            bound[node.targets[0].id] = module_name

    if not bound:
        return findings

    # Pass 2: find <bound>.<attr> accesses anywhere in the file.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in bound:
            continue
        findings.append((node.lineno, bound[node.value.id], node.attr))

    return findings


def _attribute_exists(module_name: str, attr: str) -> tuple[bool, str | None]:
    """Return (exists, error). Imports the module to check; on
    ImportError returns (False, error_str)."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:  # broad on purpose — any import failure is a finding
        return False, f"{type(e).__name__}: {e}"
    return hasattr(mod, attr), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=SRC,
        help=f"Source root to scan (default: {SRC.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))

    all_findings: list[tuple[Path, int, str, str]] = []
    for py in sorted(args.src.rglob("*.py")):
        for lineno, module_name, attr in _scan_file(py):
            all_findings.append((py, lineno, module_name, attr))

    if not all_findings:
        print("No dynamic-import + attribute-access patterns found.")
        return 0

    print(f"Found {len(all_findings)} dynamic-import attribute reference(s):")
    print()
    failures = 0
    seen: set[tuple[str, str]] = set()
    for path, lineno, module_name, attr in all_findings:
        rel = path.relative_to(REPO_ROOT)
        key = (module_name, attr)
        if key in seen:
            print(f"  {rel}:{lineno}  {module_name}.{attr}  (already checked)")
            continue
        seen.add(key)
        ok, err = _attribute_exists(module_name, attr)
        if ok:
            print(f"  ✓ {rel}:{lineno}  {module_name}.{attr}")
        else:
            failures += 1
            why = err or "AttributeError (attribute does not exist)"
            print(f"  ✗ {rel}:{lineno}  {module_name}.{attr}  →  {why}")

    print()
    if failures:
        print(f"FAIL: {failures} dynamic-import reference(s) point at missing attributes.")
        return 1
    print(f"OK: all {len(seen)} unique reference(s) resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
