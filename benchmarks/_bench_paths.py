"""Shared --out path validator for the 2026-Q2 model-challenger bench.

Hard-fails any harness invocation whose ``--out`` (or related output
path) resolves inside the repo.  All raw bench artifacts must land
under ``/data/meeting-scribe-fixtures/bench-runs/``; only
``decision_gate.md`` files are allowed under ``reports/2026-Q2-bench/``
(and they are written by the reducer, not the harnesses themselves).

The ``.gitignore`` covers casual leaks but cannot stop ``git add -f``
or a deliberately-malformed path; this validator is the second wall.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_BENCH_ROOT = Path("/data/meeting-scribe-fixtures/bench-runs")


def assert_offline_path(path: str | Path) -> Path:
    """Resolve ``path`` and refuse if it lands inside the repo.

    Returns the resolved path on success.  Raises ``SystemExit`` on
    failure so the harness aborts before any IO.
    """
    p = Path(path).expanduser().resolve()
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        return p
    raise SystemExit(
        f"--out path {p} is inside the repo ({REPO_ROOT}). "
        f"Bench outputs must land under {OFFLINE_BENCH_ROOT}/<track>/. "
        f"Refusing to write."
    )
