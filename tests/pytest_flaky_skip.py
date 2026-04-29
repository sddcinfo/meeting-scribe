"""Pytest plugin: skip / xfail tests listed in ``tests/.flaky.toml``.

Wired in via ``conftest.py``:

    from tests import pytest_flaky_skip
    pytest_collection_modifyitems = pytest_flaky_skip.pytest_collection_modifyitems

Lane detection via ``CI_LANE`` env var:
  - ``CI_LANE=smoke``  → quarantined tests are skipped (don't break PR)
  - ``CI_LANE=nightly``→ quarantined tests xfail(strict=False) — still
                          runs, surfaces failures in a separate JUnit
                          report, but doesn't break the lane.
  - unset (local dev) → behaves like nightly (run + xfail).
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import pytest


def _load_quarantine() -> dict:
    """Load the .flaky.toml registry, if present."""
    path = Path(__file__).resolve().parent / ".flaky.toml"
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        print(f"warning: tests/.flaky.toml parse error: {e}", file=sys.stderr)
        return {}
    return data.get("entries", {}) or {}


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    quarantine = _load_quarantine()
    if not quarantine:
        return
    lane = os.environ.get("CI_LANE", "").lower()
    for item in items:
        nodeid = item.nodeid
        # Match by exact nodeid OR by file::testname prefix.
        for entry_id, meta in quarantine.items():
            if nodeid == entry_id or nodeid.startswith(entry_id + "["):
                reason = (meta or {}).get("reason", "quarantined as flaky")
                if lane == "smoke":
                    item.add_marker(pytest.mark.skip(reason=f"flaky-quarantine: {reason}"))
                else:
                    item.add_marker(pytest.mark.xfail(reason=f"flaky-quarantine: {reason}", strict=False))
                break
