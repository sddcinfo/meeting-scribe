#!/usr/bin/env python3
"""Suggest new entries for tests/.flaky.toml from local test history.

Walks recent JUnit XML output (default: ``test-results/*.xml``) and
counts per-test pass/fail rates. Tests with > 5% failure rate over
the last N runs that aren't already in ``tests/.flaky.toml`` are
printed for the developer to review and commit.

Usage:
  scripts/check_flaky.py                       # report
  scripts/check_flaky.py --runs 50             # consider last 50 XMLs
  scripts/check_flaky.py --threshold 0.05      # 5% default

Single-person repo: no bot PR machinery. The developer eyeballs the
output, decides if an entry should be added, edits .flaky.toml.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLAKY_TOML = REPO_ROOT / "tests" / ".flaky.toml"
DEFAULT_RESULTS_DIR = REPO_ROOT / "test-results"


def _existing_quarantine() -> set[str]:
    if not FLAKY_TOML.exists():
        return set()
    data = tomllib.loads(FLAKY_TOML.read_text())
    return set((data.get("entries") or {}).keys())


def _walk_junit(results_dir: Path, limit: int) -> tuple[Counter, Counter]:
    """Return (runs_per_test, failures_per_test)."""
    runs: Counter = Counter()
    fails: Counter = Counter()
    files = sorted(results_dir.rglob("*.xml"))[-limit:]
    for path in files:
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        for case in tree.iter("testcase"):
            cls = case.get("classname", "")
            name = case.get("name", "")
            if not name:
                continue
            nodeid = f"{cls}::{name}" if cls else name
            runs[nodeid] += 1
            if case.find("failure") is not None or case.find("error") is not None:
                fails[nodeid] += 1
    return runs, fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=50, help="number of XMLs to consider")
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="failure-rate threshold to suggest quarantine (default: 0.05)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="dir to walk for JUnit XML",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"no results dir at {args.results_dir.relative_to(REPO_ROOT)} — "
              f"have you run tests with --junit-xml=test-results/...?",
              file=sys.stderr)
        return 0

    runs, fails = _walk_junit(args.results_dir, args.runs)
    existing = _existing_quarantine()
    suggestions: list[tuple[str, float, int, int]] = []
    for nodeid, n_runs in runs.items():
        if n_runs < 5:  # not enough data
            continue
        n_fails = fails.get(nodeid, 0)
        rate = n_fails / n_runs
        if rate >= args.threshold and nodeid not in existing:
            suggestions.append((nodeid, rate, n_fails, n_runs))

    if not suggestions:
        print(f"No new flake suggestions (threshold ≥ {args.threshold:.0%}, runs ≥ {args.runs}).")
        return 0
    print(f"Suggested new entries for tests/.flaky.toml:")
    for nodeid, rate, f, n in sorted(suggestions, key=lambda x: -x[1]):
        print(f"  {nodeid}  ({rate:.0%}, {f}/{n} runs)")
    print()
    print("Add to tests/.flaky.toml as:")
    print("  [entries.\"<nodeid>\"]")
    print("  added_at = \"YYYY-MM-DD\"")
    print("  assignee = \"brad\"")
    print("  reason = \"<one-line>\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
