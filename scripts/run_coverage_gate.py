#!/usr/bin/env python3
"""Run the unit coverage gate.

Runs the cheap unit slice (no integration, no browser, no GPU) under
``pytest-cov`` and fails if the resulting percentage drops below the
baseline pinned in ``.ci/coverage-floor``. The gate is the single
source of truth for the coverage threshold — neither pyproject.toml
nor the CI workflow hard-code a number.

Usage:

    python scripts/run_coverage_gate.py            # run the gate
    python scripts/run_coverage_gate.py --record   # update the floor

The ``--record`` mode is for the operator to bump the floor manually
when a phase materially raises coverage; the gate refuses to record a
floor lower than the current one (the floor only ratchets up).

Floor source-of-truth: ``.ci/coverage-floor`` reflects CI's reachable
percentage, not local's. A dev box with vLLM/diarize/TTS backends
running will runtime-pass tests that CI runtime-skips (the
``num_requests_running`` / health-probe tests in particular); the gap
between local and CI is typically 1–2 percentage points. If the floor
is recorded from a hot local box, CI will fail at the lower number.
Always re-record from a CI run output or from a checkout with no
project backends running.

Exit codes:

    0   coverage at or above the floor
    1   coverage below the floor (a regression)
    2   pytest run failed for any reason other than coverage
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_FLOOR = REPO_ROOT / ".ci" / "coverage-floor"
# Selection matches the existing pytest default addopts (skip
# integration + browser) plus a narrow set of "needs hardware /
# external service / interactive" markers that cannot run in CI or
# under coverage. Tests that auto-skip when their dependencies are
# missing (lockfile_sync without pip-tools, slow on a constrained
# runner) stay IN — keeping them reachable lets coverage credit them
# wherever the dependency happens to be present.
_DEFAULT_PYTEST_ARGS = [
    "-m",
    "not integration and not browser and not gb10 and not real_gb10 "
    "and not qemu_kvm and not qemu_tcg and not demo_smoke and not manual",
    "--no-header",
]


def _read_floor() -> float:
    if not COVERAGE_FLOOR.exists():
        # First run: no floor yet → effectively 0% so the gate
        # establishes a baseline. The operator runs --record after.
        return 0.0
    raw = COVERAGE_FLOOR.read_text().strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"invalid floor in {COVERAGE_FLOOR}: {raw!r}") from exc


def _write_floor(value: float) -> None:
    COVERAGE_FLOOR.parent.mkdir(parents=True, exist_ok=True)
    # Two decimals is enough resolution; whole numbers print clean.
    COVERAGE_FLOOR.write_text(f"{value:.2f}\n")


def _run_pytest_with_coverage(extra_args: list[str]) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--cov=src/meeting_scribe",
        "--cov-report=term",
        *_DEFAULT_PYTEST_ARGS,
        *extra_args,
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    # Mirror scripts/ci_local.py's smoke lane: PYTHONPATH=src lets
    # mise-global Python import the in-repo ``meeting_scribe`` package
    # without an editable install having happened in this interpreter.
    import os

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src{os.pathsep}{existing}" if existing else "src"
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


_TOTAL_RE = re.compile(r"^TOTAL\s+\S+\s+\S+\s+(?:\S+\s+\S+\s+)?(\d+(?:\.\d+)?)%", re.MULTILINE)


def _parse_total_percent(report: str) -> float | None:
    """Pull the TOTAL row's percentage out of the term report.

    Without branch coverage: ``TOTAL  120  10  92%``
    With branch coverage:    ``TOTAL  120  10  40  4   92%``

    The regex tolerates both shapes.
    """
    match = _TOTAL_RE.search(report)
    if not match:
        return None
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Update .ci/coverage-floor to the measured percentage. "
        "Refuses to lower the floor — only ratchets up.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pytest after a literal --",
    )
    ns = parser.parse_args()

    extra: list[str] = []
    if ns.extra and ns.extra[0] == "--":
        extra = ns.extra[1:]

    rc, output = _run_pytest_with_coverage(extra)
    print(output, end="")
    if rc not in (0, 1):  # pytest rc 1 = failures only, rc 5 = no tests
        print(f"  pytest exited with rc={rc} (not a coverage failure)", file=sys.stderr)
        return 2

    pct = _parse_total_percent(output)
    if pct is None:
        print("  could not parse TOTAL coverage from pytest output", file=sys.stderr)
        return 2

    floor = _read_floor()
    print(f"  coverage: {pct:.2f}% · floor: {floor:.2f}%")

    if ns.record:
        if pct + 0.01 < floor:
            print(
                f"  refusing to record {pct:.2f}% — floor is {floor:.2f}% "
                f"and ratchets up only. Investigate the regression first.",
                file=sys.stderr,
            )
            return 1
        _write_floor(pct)
        print(f"  recorded new floor: {pct:.2f}% in {COVERAGE_FLOOR.relative_to(REPO_ROOT)}")
        return 0

    if rc == 1:
        # Test failures take precedence over the coverage gate — the
        # operator should fix the failing tests before judging coverage.
        print("  pytest reported test failures (rc=1); fix tests before checking coverage")
        return 1

    if pct + 0.01 < floor:
        print(
            f"  FAIL — coverage {pct:.2f}% is below floor {floor:.2f}%. "
            f"Either fix the regression or, if the drop is intentional, "
            f"explain it in the PR and re-run with --record.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
