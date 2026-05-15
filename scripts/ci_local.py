#!/usr/bin/env python3
"""Local CI parity runner — mirror the GitHub Actions checks pre-push.

CI is the fail-safe, not the first line of detection. Every step here lines up
with a step in ``.github/workflows/tests.yml`` so a green local run predicts a
green PR. If you add a CI step, add it here too.

Default ("light") scope — the lanes that block PR merges and finish in
under two minutes:
  * lint (ruff check/format, secret scan, UI/how-it-works validators)
  * smoke (marker-orphan check, pytest unit, node --test JS)
  * security (pip-audit, npm audit overlay)

Heavy lanes — opt in via ``--full`` or ``CI_LOCAL_FULL=1`` env:
  * browser-smoke (Playwright Chromium suite, ~50 s + chromium download)
  * quality (saved-meeting live ASR/translation/TTS regression, ~50 s, needs
    the live stack warm)

The pre-push hook runs in light mode by default so committing/pushing isn't
gated on browser tests every time. Run ``--full`` (or ``CI_LOCAL_FULL=1
git push``) before a release / after a frontend or audio-pipeline refactor
to validate the heavy lanes too.

CodeQL has no practical local equivalent — run it via ``gh`` after push if you
need it.

Usage:
  python3 scripts/ci_local.py             # light pipeline (default)
  python3 scripts/ci_local.py --full      # everything, incl. browser + saved-meeting
  python3 scripts/ci_local.py --only lint # single lane (any of the lanes above)
  CI_LOCAL_FULL=1 git push                # push with the full pipeline
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Mirror the pip-audit ignores from .github/workflows/tests.yml.
# - CVE-2026-3219, CVE-2026-6357: pip's own advisories (pip 26.0.1).
#   pip is the package manager, not a runtime dep we ship; the CI
#   runner's pre-installed pip is whatever ``setup-python@v6`` ships.
# - CVE-2026-44405 (paramiko 4.0.0), CVE-2026-44432 + CVE-2026-44431
#   (urllib3 2.6.3): not project dependencies — neither appears in
#   pyproject.toml or requirements.lock. They're pulled in by the
#   local mise python's pre-installed tooling (the venv inherits
#   site-packages) and aren't shipped with the appliance. The
#   ``actions/setup-python@v6`` CI runner has its own clean
#   environment without these packages.
PIP_AUDIT_IGNORES = (
    "CVE-2026-3219",
    "CVE-2026-6357",
    "CVE-2026-44405",
    "CVE-2026-44432",
    "CVE-2026-44431",
)
LIVE_STACK_MEETING_FILE = ROOT / ".local" / "live_stack_meetings.txt"


class Step:
    __slots__ = ("argv", "cwd", "env_extra", "lane", "name")

    def __init__(
        self,
        lane: str,
        name: str,
        argv: list[str],
        cwd: Path | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> None:
        self.lane = lane
        self.name = name
        self.argv = argv
        self.cwd = cwd or ROOT
        self.env_extra = env_extra or {}


def lint_steps() -> list[Step]:
    return [
        Step("lint", "ruff check", ["ruff", "check", "src/", "tests/"]),
        Step("lint", "ruff format --check", ["ruff", "format", "--check", "src/", "tests/"]),
        Step("lint", "secret scan", [sys.executable, "scripts/check_secrets.py"]),
        # Whole-repo validators that used to live in ``.githooks/pre-commit``.
        # Moved here when we split the hooks into fast pre-commit + heavy
        # pre-push so a one-file commit doesn't get blocked on unrelated
        # pre-existing tech debt.
        Step("lint", "ui-style validator", [sys.executable, "scripts/check_ui_style.py"]),
        Step(
            "lint",
            "how-it-works validator",
            [sys.executable, "scripts/validate_how_it_works.py"],
        ),
        # Cross-source claim validator for the hardware-scaling
        # editorial page. Self-contained (stdlib-only), reads its own
        # claims JSON manifest, and cross-checks the rendered prose
        # against repo source files.
        Step(
            "lint",
            "hardware-scaling validator",
            [sys.executable, "scripts/validate_hardware_scaling.py"],
        ),
        # Tailwind v4 build freshness gate. Trips iff the committed CSS in
        # static/css/dist/ (or any HTML cache-bust ?v= stamp) diverges from
        # what a fresh build would produce. `--mode check` writes to a temp
        # dir and byte-compares — never mutates the worktree. Cheap (<1s
        # for the current entry count), so it earns its place in lint.
        Step(
            "lint",
            "css build freshness check",
            [sys.executable, "scripts/build_css.py", "--mode", "check"],
        ),
    ]


def smoke_steps() -> list[Step]:
    return [
        Step(
            "smoke",
            "marker-orphan check",
            [sys.executable, "scripts/hooks/check_marker_orphans.py"],
        ),
        Step(
            "smoke",
            # Single command runs the unit suite AND the coverage gate;
            # `scripts/run_coverage_gate.py` executes pytest with --cov,
            # parses the TOTAL row, and exits non-zero if the percentage
            # drops below the floor pinned in `.ci/coverage-floor`.
            # Replaces the prior bare ``pytest tests/`` invocation so PRs
            # never silently lose test coverage.
            "pytest unit suite + coverage gate",
            [sys.executable, "scripts/run_coverage_gate.py"],
        ),
        Step(
            "smoke",
            "node --test JS suite",
            ["node", "--test"] + sorted(str(p) for p in (ROOT / "tests/js").glob("*.test.mjs")),
        ),
        Step(
            "smoke",
            # Phase 6 — run the demo gate with hardware checks skipped
            # (CI doesn't have the SP325 / nmcli stack). This exercises
            # the entire preflight pipeline + the wiring of the demo
            # checks; the hardware-bound branches no-op via
            # ``--skip-hardware``. The corresponding GB10 invocation is
            # ``meeting-scribe demo-smoke`` (which runs without
            # ``--skip-hardware``).
            "preflight --mode=demo --skip-hardware",
            [
                sys.executable,
                "-c",
                "from meeting_scribe.cli import cli; cli()",
                "preflight",
                "--mode=demo",
                "--skip-hardware",
                "--wait",
                "30",
            ],
        ),
    ]


def quality_steps() -> list[Step]:
    meeting_ids = os.environ.get("LIVE_STACK_MEETING_IDS")
    if meeting_ids:
        ids = meeting_ids.split()
    elif LIVE_STACK_MEETING_FILE.exists():
        ids = [
            line.strip()
            for line in LIVE_STACK_MEETING_FILE.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        ids = []
    if not ids:
        return [
            Step(
                "quality",
                "saved-meeting live ASR/translation/TTS regression",
                [
                    sys.executable,
                    "-c",
                    (
                        "print('SKIP: set LIVE_STACK_MEETING_IDS or create "
                        ".local/live_stack_meetings.txt to run the private "
                        "saved-meeting live-stack regression')"
                    ),
                ],
            )
        ]
    out = os.environ.get(
        "LIVE_STACK_REGRESSION_OUT",
        str(Path(tempfile.gettempdir()) / "meeting_scribe_live_stack_regression_hook.json"),
    )
    return [
        Step(
            "quality",
            "saved-meeting live ASR/translation/TTS regression",
            [
                sys.executable,
                "scripts/bench/live_stack_regression.py",
                "--meeting-ids",
                *ids,
                "--out",
                out,
            ],
            env_extra={"PYTHONPATH": "src"},
        )
    ]


def security_steps() -> list[Step]:
    overlay = ROOT / "overlay"
    pip_argv = [
        sys.executable,
        "-m",
        "pip_audit",
        "--strict",
        "--vulnerability-service",
        "osv",
    ]
    for cve in PIP_AUDIT_IGNORES:
        pip_argv.append(f"--ignore-vuln={cve}")
    return [
        Step("security", "pip-audit", pip_argv),
        Step(
            "security",
            "npm audit overlay",
            # Mirror the CI workflow: generate package-lock.json on the fly if
            # it's missing, then audit. The overlay tree commits no lockfile
            # because Electron is the only direct dep and we don't want a
            # 60k-line file in review.
            [
                "bash",
                "-lc",
                "if [ -f package-lock.json ]; then "
                "  npm audit --audit-level=moderate --omit=dev; "
                "else "
                "  npm install --package-lock-only && "
                "  npm audit --audit-level=moderate --omit=dev; "
                "fi",
            ],
            cwd=overlay,
        ),
    ]


def browser_steps() -> list[Step]:
    return [
        Step(
            "browser-smoke",
            "playwright browser smoke",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/browser/",
                "-v",
                "--tb=short",
                "-m",
                "browser",
            ],
            env_extra={"PYTHONPATH": "src"},
        ),
    ]


def run_step(step: Step) -> tuple[bool, float]:
    label = f"[{step.lane}] {step.name}"
    print(f"\n── {label} ─────────────────────────────────────────────")
    env = os.environ.copy()
    env.update(step.env_extra)
    started = time.monotonic()
    try:
        proc = subprocess.run(step.argv, cwd=step.cwd, env=env, check=False)
        ok = proc.returncode == 0
    except FileNotFoundError as exc:
        print(f"missing binary: {exc}", file=sys.stderr)
        ok = False
    elapsed = time.monotonic() - started
    status = "PASS" if ok else "FAIL"
    print(f"── {label}: {status} in {elapsed:.1f}s")
    return ok, elapsed


def precheck(lanes: set[str]) -> None:
    """Bail before running anything if a binary needed by ``lanes`` is missing."""
    needs: dict[str, str] = {}
    if "lint" in lanes:
        needs["ruff"] = "ruff"
    if "smoke" in lanes:
        needs["node"] = "node"
    if "security" in lanes:
        needs["npm"] = "npm"
    if "browser-smoke" in lanes:
        needs["playwright"] = "playwright (pip install -e '.[dev]' && playwright install chromium)"

    missing = [pkg for binary, pkg in needs.items() if shutil.which(binary) is None]
    if "security" in lanes:
        try:
            import pip_audit  # noqa: F401
        except ImportError:
            missing.append("pip-audit (pip install pip-audit)")
    if missing:
        print(
            "ci-local needs these on PATH first:\n  "
            + "\n  ".join(missing)
            + "\nInstall via: pip install -e '.[dev]' pip-audit  &&  apt install nodejs npm",
            file=sys.stderr,
        )
        sys.exit(127)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("lint", "smoke", "quality", "security", "browser"),
        help="Run a single lane.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Include the heavy lanes (browser-smoke + saved-meeting quality "
            "regression). Equivalent to ``CI_LOCAL_FULL=1``. Default is the "
            "light pipeline (lint + smoke + security)."
        ),
    )
    args = parser.parse_args()

    full = args.full or os.environ.get("CI_LOCAL_FULL", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )

    if args.only == "lint":
        steps = lint_steps()
    elif args.only == "smoke":
        steps = smoke_steps()
    elif args.only == "quality":
        steps = quality_steps()
    elif args.only == "security":
        steps = security_steps()
    elif args.only == "browser":
        steps = browser_steps()
    else:
        steps = lint_steps() + smoke_steps()
        if full:
            steps += browser_steps() + quality_steps()
        steps += security_steps()

    mode = "full" if full else "light"
    print(f"ci-local: mode={mode} (set CI_LOCAL_FULL=1 or pass --full for heavy lanes)")
    precheck({step.lane for step in steps})

    failures: list[tuple[Step, float]] = []
    total_started = time.monotonic()
    for step in steps:
        ok, elapsed = run_step(step)
        if not ok:
            failures.append((step, elapsed))

    total = time.monotonic() - total_started
    print("\n══════════════════════════════════════════════════════════════")
    if failures:
        print(f"FAIL ({mode}): {len(failures)} step(s) failed in {total:.1f}s total")
        for step, elapsed in failures:
            print(f"  - [{step.lane}] {step.name} ({elapsed:.1f}s)")
        return 1
    print(f"OK ({mode}): {len(steps)} step(s) green in {total:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
