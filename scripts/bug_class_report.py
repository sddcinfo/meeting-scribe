#!/usr/bin/env python3
"""Bug-class dashboard — computed on-demand from `git log` trailers.

The pre-push hook (``scripts/hooks/classify_push.py``) refuses to push
a bugfix without a ``Bug-class:`` trailer (allowed slugs in
``ALLOWED_BUG_CLASSES``). This script walks history and aggregates the
distribution.

Single-person repo + rebase-merge (or fast-forward) preserves trailers
in ``main``. No persisted log file, no bot PR, no CI workflow — the
git history IS the source of truth.

Usage:
  scripts/bug_class_report.py                  # last 30 days
  scripts/bug_class_report.py --since 90       # last 90 days
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter

ALLOWED_BUG_CLASSES = (
    "cross-window-sync",
    "ws-lifecycle",
    "event-dedup",
    "async-render",
    "platform-quirk",
    "data-shape",
    "backend-lifecycle",
)


def _git_log_trailers(since_days: int) -> list[str]:
    """Return the list of `Bug-class:` trailer values in the last
    ``since_days`` days."""
    proc = subprocess.run(
        [
            "git", "log",
            f"--since={since_days}.days.ago",
            "--pretty=format:%(trailers:key=Bug-class,valueonly)",
        ],
        capture_output=True, text=True, check=False,
    )
    out = proc.stdout
    return [line.strip() for line in out.split("\n") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=int, default=30, help="lookback in days (default: 30)")
    parser.add_argument("--unknown", action="store_true", help="also list unrecognized slugs")
    args = parser.parse_args()

    trailers = _git_log_trailers(args.since)
    counts = Counter(trailers)

    print(f"Bug-class distribution (last {args.since} days):")
    print(f"  total tagged commits: {sum(counts.values())}")
    print()
    for slug in ALLOWED_BUG_CLASSES:
        n = counts.get(slug, 0)
        bar = "█" * n
        print(f"  {slug:22s} {n:3d}  {bar}")

    unknown = {k: v for k, v in counts.items() if k not in ALLOWED_BUG_CLASSES}
    if unknown:
        print()
        print(f"  unrecognized slugs: {sum(unknown.values())} commits")
        if args.unknown:
            for slug, n in sorted(unknown.items(), key=lambda x: -x[1]):
                print(f"    {slug:22s} {n:3d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
