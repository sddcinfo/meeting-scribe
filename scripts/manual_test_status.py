#!/usr/bin/env python3
"""Manual-test-status checker + stamp bumper.

Reads ``tests/manual/.last_verified.json`` and reports staleness for
each item in the runbook (``tests/manual/README.md``). Warns at 30
days; intended to be invoked as a release-gate check on tagged commits.

Usage:
  scripts/manual_test_status.py                # report status
  scripts/manual_test_status.py --bump <slug>  # mark <slug> verified now
  scripts/manual_test_status.py --strict       # exit 1 if anything > 30 days
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAMPS_PATH = REPO_ROOT / "tests" / "manual" / ".last_verified.json"

WARN_DAYS = 30


def _load() -> dict:
    if not STAMPS_PATH.exists():
        return {}
    return json.loads(STAMPS_PATH.read_text())


def _save(data: dict) -> None:
    STAMPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAMPS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def _age_days(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        ts = dt.datetime.fromisoformat(iso)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.UTC)
    now = dt.datetime.now(dt.UTC)
    return (now - ts).days


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump", help="mark <slug> as verified at the current time")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any item is stale (>30 days)")
    args = parser.parse_args()

    stamps = _load()
    if args.bump:
        if args.bump not in stamps:
            print(f"unknown slug {args.bump!r}; known: {sorted(stamps.keys())}", file=sys.stderr)
            return 2
        stamps[args.bump] = dt.datetime.now(dt.UTC).isoformat()
        _save(stamps)
        print(f"✓ stamped {args.bump} @ {stamps[args.bump]}")
        return 0

    print("Manual-test runbook status:")
    fail = False
    for slug, iso in sorted(stamps.items()):
        age = _age_days(iso)
        if age is None:
            print(f"  ✗ {slug:36s}  never verified")
            fail = True
        elif age >= WARN_DAYS:
            print(f"  ✗ {slug:36s}  {age:4d}d old (>{WARN_DAYS})")
            fail = True
        else:
            print(f"  ✓ {slug:36s}  {age:4d}d old")
    if fail and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
