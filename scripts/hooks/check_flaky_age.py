#!/usr/bin/env python3
"""Pre-push deadline enforcer for tests/.flaky.toml entries.

Each entry has an ``added_at`` date. Policy:

  - 14 days old  → warn on every push
  - 21 days old  → abort the push (forces fix-or-delete)

Single-person repo: there's no team to escalate to. The forcing
function is the developer's own friction.
"""

from __future__ import annotations

import datetime as dt
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FLAKY_TOML = REPO_ROOT / "tests" / ".flaky.toml"
WARN_DAYS = 14
ABORT_DAYS = 21


def main() -> int:
    if not FLAKY_TOML.exists():
        return 0
    data = tomllib.loads(FLAKY_TOML.read_text())
    entries = data.get("entries") or {}
    if not entries:
        return 0
    today = dt.date.today()
    warn: list[tuple[str, int]] = []
    abort: list[tuple[str, int]] = []
    for nodeid, meta in entries.items():
        added = (meta or {}).get("added_at")
        if not added:
            warn.append((nodeid, -1))
            continue
        try:
            added_d = dt.date.fromisoformat(added)
        except ValueError:
            warn.append((nodeid, -1))
            continue
        age = (today - added_d).days
        if age >= ABORT_DAYS:
            abort.append((nodeid, age))
        elif age >= WARN_DAYS:
            warn.append((nodeid, age))

    if warn:
        print("⚠ flaky-quarantine entries past 14d:", file=sys.stderr)
        for nodeid, age in warn:
            print(f"    {nodeid}  ({age}d)", file=sys.stderr)
    if abort:
        print("✗ flaky-quarantine entries past 21d — fix or delete:", file=sys.stderr)
        for nodeid, age in abort:
            print(f"    {nodeid}  ({age}d)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
