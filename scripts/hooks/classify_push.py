#!/usr/bin/env python3
"""Pre-push regression-test gate, powered by local Opus 4.7.

Workflow:

  1. Before pushing, compute the diff of the commits about to be pushed
     (HEAD vs the upstream tracking branch).
  2. Hand the diff + last 3 commit messages to ``claude -p`` and ask it
     to classify the change: is this a bugfix? is a regression test
     added? what bug class is it?
  3. If the classifier says "bugfix" + "no test added", abort the push.
     Operator either adds a test and re-pushes, or runs ``git push
     --no-verify`` AFTER recording a one-line waiver in
     ``.git-waivers/<sha>.txt`` (the waiver file is committed and
     forms an audit trail).
  4. If the classifier says "bugfix" + "has test", and the most recent
     commit lacks a ``Bug-class:`` trailer, suggest one based on the
     classifier's output and abort with instructions to amend.

Why local-only: this exploits the GB10 + Claude Code (Opus 4.7) that
the user has on their dev machine. No GitHub CI workflow is needed —
the gate sits where it actually catches regressions before they leave
the repo. CI is a backstop (tests still run there).

Bypass: ``git push --no-verify`` skips the hook. The accompanying
post-push audit script (``scripts/check_no_verify_audit.py``) flags
any push whose head commit lacks a regression test AND lacks a waiver
file — so even a bypass leaves a trail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WAIVERS_DIR = REPO_ROOT / ".git-waivers"

# Allowed Bug-class trailer slugs — must mirror the plan's bug-class
# taxonomy (CONTRIBUTING.md Phase 0.4 + scripts/bug_class_report.py).
ALLOWED_BUG_CLASSES = frozenset({
    "cross-window-sync",
    "ws-lifecycle",
    "event-dedup",
    "async-render",
    "platform-quirk",
    "data-shape",
    "backend-lifecycle",
})


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


def _staged_or_pushed_diff() -> tuple[str, list[str]]:
    """Return (diff_text, last_3_subjects). Falls back gracefully if no
    upstream is set (returns the diff against origin/main)."""
    # Resolve "what am I about to push?"
    upstream = _run(["git", "rev-parse", "--abbrev-ref", "@{u}"]).stdout.strip()
    if not upstream:
        upstream = "origin/main"
    diff = _run(["git", "diff", f"{upstream}...HEAD"]).stdout
    log = _run([
        "git", "log", f"{upstream}..HEAD", "--pretty=format:%s", "-n", "3",
    ]).stdout
    subjects = [s for s in log.split("\n") if s.strip()]
    return diff, subjects


def _has_test_changes(diff: str) -> bool:
    """Quick textual check: does the diff add or modify any test file?"""
    for line in diff.split("\n"):
        if not line.startswith(("+++ b/", "--- a/")):
            continue
        path = line[6:]
        if "/tests/" in path or path.startswith("tests/"):
            if "test_" in path or path.endswith(".test.mjs") or path.endswith(".test.ts"):
                return True
    return False


def _commit_trailer(commit: str = "HEAD") -> str | None:
    """Return the ``Bug-class:`` trailer value on ``commit``, or None."""
    out = _run(["git", "log", commit, "-n", "1", "--pretty=%(trailers:key=Bug-class,valueonly)"]).stdout.strip()
    return out or None


def _has_waiver_for_head() -> bool:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if not head:
        return False
    return (WAIVERS_DIR / f"{head}.txt").exists()


def _classify_with_claude(diff: str, subjects: list[str]) -> dict | None:
    """Ask local Opus 4.7 to classify the diff.

    Returns ``{"is_bugfix": bool, "has_test": bool, "bug_class":
    str|null, "reasoning": str}`` or None if ``claude`` isn't
    available / the call failed (in which case the hook falls back to
    a regex heuristic).
    """
    from shutil import which

    if not which("claude"):
        return None

    # Cap diff size — Opus has plenty of context but most useful signal
    # is in the first ~10000 chars of a focused diff.
    if len(diff) > 20000:
        diff = diff[:20000] + "\n[...truncated...]"

    prompt = (
        "You are a strict pre-push code reviewer. "
        "Classify the diff below.\n\n"
        f"Last commit subjects:\n{chr(10).join('  - ' + s for s in subjects)}\n\n"
        f"Diff:\n```\n{diff}\n```\n\n"
        "Respond with EXACTLY one JSON object on a single line, no prose, "
        "no code fence:\n"
        '{"is_bugfix": bool, "has_test": bool, "bug_class": '
        '"cross-window-sync"|"ws-lifecycle"|"event-dedup"|"async-render"|'
        '"platform-quirk"|"data-shape"|"backend-lifecycle"|null, '
        '"reasoning": "<one short sentence>"}\n\n'
        "Rules:\n"
        "- is_bugfix=true if the diff fixes user-observable broken behavior, "
        "even if the commit message says 'chore' or 'refactor'.\n"
        "- has_test=true ONLY if the diff adds or meaningfully modifies a "
        "file under tests/ that exercises the changed code path.\n"
        "- bug_class is required when is_bugfix=true; null otherwise.\n"
    )
    proc = _run(
        ["claude", "-p", "--output-format", "text", "--max-turns", "1", prompt],
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    # Extract the JSON object (in case Opus wrapped with prose despite
    # the instruction).
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed


def _fallback_heuristic(subjects: list[str], diff: str) -> dict:
    """Regex-only fallback when claude isn't available."""
    is_bugfix = any(
        s.startswith(("fix:", "fix(")) or "FIX " in s.upper()
        for s in subjects
    )
    has_test = _has_test_changes(diff)
    return {
        "is_bugfix": is_bugfix,
        "has_test": has_test,
        "bug_class": None,
        "reasoning": "fallback-heuristic (claude CLI unavailable)",
    }


def main() -> int:
    diff, subjects = _staged_or_pushed_diff()
    if not diff.strip():
        # Nothing to push — let git handle it.
        return 0

    classifier_result = _classify_with_claude(diff, subjects)
    if classifier_result is None:
        classifier_result = _fallback_heuristic(subjects, diff)

    is_bugfix = bool(classifier_result.get("is_bugfix"))
    has_test = bool(classifier_result.get("has_test"))
    bug_class = classifier_result.get("bug_class")
    reasoning = classifier_result.get("reasoning", "")

    # ── Gate 1: bugfix MUST have a regression test ──────────────────
    if is_bugfix and not has_test:
        if _has_waiver_for_head():
            print("⚠ pre-push: bugfix without test — waiver present, allowing push.", file=sys.stderr)
        else:
            head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
            print("✗ pre-push regression-test gate FAILED", file=sys.stderr)
            print(f"  Classifier: {classifier_result}", file=sys.stderr)
            print(file=sys.stderr)
            print("  This change looks like a bugfix that doesn't add a", file=sys.stderr)
            print("  regression test. Please add a failing-before / passing-after", file=sys.stderr)
            print(f"  test under tests/ before pushing.", file=sys.stderr)
            print(file=sys.stderr)
            print("  To bypass (audit-trailed):", file=sys.stderr)
            print(f"    mkdir -p .git-waivers && \\", file=sys.stderr)
            print(f"      echo 'reason here' > .git-waivers/{head}.txt && \\", file=sys.stderr)
            print(f"      git add .git-waivers/{head}.txt && \\", file=sys.stderr)
            print(f"      git commit --amend --no-edit && \\", file=sys.stderr)
            print(f"      git push --no-verify", file=sys.stderr)
            return 1

    # ── Gate 2: bugfix SHOULD have a Bug-class trailer ──────────────
    if is_bugfix and not _commit_trailer():
        suggested = bug_class or "<choose-from-enum>"
        print(f"⚠ pre-push: bugfix commit is missing a `Bug-class:` trailer.", file=sys.stderr)
        print(f"  Classifier suggests: {suggested}", file=sys.stderr)
        print(f"  Allowed slugs: {sorted(ALLOWED_BUG_CLASSES)}", file=sys.stderr)
        print(file=sys.stderr)
        print("  Amend with:", file=sys.stderr)
        print(f"    git commit --amend --trailer 'Bug-class: {suggested}'", file=sys.stderr)
        if os.environ.get("PRE_PUSH_TRAILER_STRICT") == "1":
            return 1

    print(f"✓ pre-push: {reasoning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
