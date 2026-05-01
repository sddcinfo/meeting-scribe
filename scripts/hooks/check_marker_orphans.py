#!/usr/bin/env python3
"""Check that every pytest marker used in tests/ is in the routing table.

Why this exists: a test marked @pytest.mark.foo where `foo` isn't in any CI
lane silently never runs. CI passes, the test exists, the user thinks they
have coverage — they don't. This hook fails the commit if any test uses a
marker that isn't in the policy table below.

The table mirrors `pyproject.toml` `[tool.pytest.ini_options].markers`
and the lane wiring in `.github/workflows/tests.yml`. Add a marker here,
to pyproject.toml, and to a CI lane in the same change set.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Markers known to the lane policy. Every marker that appears in any test
# file under tests/ MUST be in this set, or the hook aborts.
KNOWN_MARKERS: frozenset[str] = frozenset({
    # Core lifecycle markers — wired into pyproject.toml + tests.yml
    "integration",  # nightly + CI nightly cron
    "system",       # full-stack tests, requires autosre start (manual lane)
    "gb10",         # local-only on GB10 dev machine (no remote runner)
    "slow",         # nightly only
    "browser",      # smoke (subset 1.A) + nightly (full Playwright)
    "manual",       # never auto-run; surfaced via runbook 3.L

    # v2.0 fresh-GB10 install markers — local-only on the dev box
    # (require pip-tools / qemu / real GB10 hardware respectively).
    # Excluded from the default selection; not wired into a remote
    # runner lane because the harness lives outside CI.
    "lockfile_sync",  # plan §1.6a — pip-compile byte-equality
    "demo_smoke",     # plan §4.1 — real e2e demo gate
    "qemu_kvm",       # plan §4.2 — real ISO smoke-test, ~45 min
    "qemu_tcg",       # plan §4.2 — cross-arch ISO smoke-test, ~2 hr
    "real_gb10",      # plan §4.3 — real GB10 host on the LAN

    # Pytest builtin / framework markers — always allowed
    "asyncio",
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
})

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"


def collect_markers(path: Path) -> set[str]:
    """Parse a test file and return every pytest.mark.<name> referenced."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()

    markers: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
            # Match: pytest.mark.<name>
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "mark"
                and isinstance(value.value, ast.Name)
                and value.value.id == "pytest"
            ):
                markers.add(node.attr)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            # Match: pytest.mark.<name>(...) — captured via Attribute above.
            self.generic_visit(node)

    _Visitor().visit(tree)
    return markers


def main() -> int:
    if not TESTS_DIR.exists():
        return 0

    offenders: dict[Path, set[str]] = {}
    for path in TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        markers = collect_markers(path)
        unknown = markers - KNOWN_MARKERS
        if unknown:
            offenders[path] = unknown

    if not offenders:
        return 0

    print("✗ marker-orphan: unknown pytest markers found", file=sys.stderr)
    print(file=sys.stderr)
    for path, unknown in sorted(offenders.items()):
        rel = path.relative_to(REPO_ROOT)
        for m in sorted(unknown):
            print(f"  {rel}: @pytest.mark.{m}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        "Add the marker to scripts/hooks/check_marker_orphans.py KNOWN_MARKERS,",
        file=sys.stderr,
    )
    print(
        "to pyproject.toml [tool.pytest.ini_options].markers, AND wire it",
        file=sys.stderr,
    )
    print("into a lane in .github/workflows/tests.yml.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
