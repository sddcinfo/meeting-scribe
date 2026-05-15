#!/usr/bin/env python3
"""Self-heal the meeting-scribe venv.

Two failure modes are recovered:

1. **Truncated console-script wrappers.** A pip-installed entry point
   normally has a ``#!.../.venv/bin/python`` shebang and imports the
   package's ``main`` function. Twice now (2026-05-07 and 2026-05-12)
   ``.venv/bin/meeting-scribe`` has been rewritten to a 2-line
   ``#!/bin/sh\\n`` stub by an out-of-venv pip run, causing the CLI to
   silently exit 0 with no output and breaking ``meeting-scribe setup``
   in bootstrap.sh.

2. **Python version drift.** ``mise.toml`` pins a specific Python
   (e.g. 3.14.4). The venv is created once at bootstrap time. If mise
   later bumps the pin, the venv's interpreter goes stale and any pip
   run from the new (mise-current) Python may corrupt the venv on
   write. We detect drift and recreate the venv from scratch.

Usage:
    scripts/repair_venv.py            # detect + repair (idempotent)
    scripts/repair_venv.py --check    # exit 1 if repair needed, no writes
    scripts/repair_venv.py --force    # rebuild venv unconditionally
    scripts/repair_venv.py --dev      # also install [dev] extras (pytest, ruff, ...)

Called from bootstrap.sh after the mise/venv stanza so every bootstrap
self-heals before the rest of the install runs. Safe to invoke
manually after observing the CLI exit 0 with no output.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"
VENV_BIN = VENV_DIR / "bin"
VENV_PY = VENV_BIN / "python"
MISE_TOML = REPO_ROOT / "mise.toml"

# pyproject.toml [project.scripts] keys. Truncated wrappers fail
# silently with exit 0 — we explicitly probe each one.
EXPECTED_SCRIPTS = ("meeting-scribe",)


def _pinned_python_version() -> tuple[int, int, int] | None:
    if not MISE_TOML.exists():
        return None
    try:
        data = tomllib.loads(MISE_TOML.read_text())
    except OSError, tomllib.TOMLDecodeError:
        return None
    pinned = data.get("tools", {}).get("python")
    if not pinned or not isinstance(pinned, str):
        return None
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", pinned)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _venv_python_version() -> tuple[int, int, int] | None:
    if not VENV_PY.exists():
        return None
    try:
        out = subprocess.check_output(
            [
                str(VENV_PY),
                "-c",
                "import sys;print('.'.join(str(x) for x in sys.version_info[:3]))",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError, OSError:
        return None
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", out)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _wrapper_is_healthy(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        head = path.read_text(errors="replace")
    except OSError:
        return False
    lines = head.splitlines()
    if not lines or not lines[0].startswith("#!"):
        return False
    # A healthy pip-generated wrapper points at the venv python AND
    # imports the package. The pathological stub is `#!/bin/sh\n`
    # with no body, or a shebang that doesn't include `python`.
    if "python" not in lines[0]:
        return False
    return "from meeting_scribe" in head


def _broken_wrappers() -> list[Path]:
    return [
        VENV_BIN / name for name in EXPECTED_SCRIPTS if not _wrapper_is_healthy(VENV_BIN / name)
    ]


def _mise_python_binary() -> Path | None:
    try:
        out = subprocess.check_output(
            ["mise", "where", "python"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError, OSError, FileNotFoundError:
        return None
    if not out:
        return None
    candidate = Path(out) / "bin" / "python"
    return candidate if candidate.exists() else None


def _rebuild_venv() -> None:
    if VENV_DIR.exists():
        print(f"[repair_venv] removing stale .venv at {VENV_DIR}", flush=True)
        shutil.rmtree(VENV_DIR)
    py = _mise_python_binary() or Path(sys.executable)
    print(f"[repair_venv] creating .venv with {py}", flush=True)
    subprocess.check_call([str(py), "-m", "venv", str(VENV_DIR)])
    subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    _install_package(allow_lockfile=True)


def _reinstall_wrappers() -> None:
    print("[repair_venv] reinstalling editable package to regenerate wrappers", flush=True)
    subprocess.check_call(
        [
            str(VENV_PY),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            "-e",
            str(REPO_ROOT),
            "--quiet",
        ]
    )


def _install_package(*, allow_lockfile: bool) -> None:
    """Mirror bootstrap.sh's install order: lockfile first, plain editable as fallback."""
    lock = REPO_ROOT / "requirements.lock"
    if allow_lockfile and lock.exists():
        rc = subprocess.call([str(VENV_PY), "-m", "pip", "install", "-r", str(lock), "--quiet"])
        if rc == 0:
            subprocess.check_call(
                [
                    str(VENV_PY),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "-e",
                    str(REPO_ROOT),
                    "--quiet",
                ]
            )
            return
        print(
            "[repair_venv] lockfile install failed (likely architecture mismatch) — falling back to unlocked editable install",
            flush=True,
        )
    subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-e", str(REPO_ROOT), "--quiet"])


def _install_dev_extras() -> None:
    """Install [dev] extras (pytest, ruff, …) into the existing venv.

    bootstrap.sh's lockfile install is the customer/production set and
    doesn't bring in the dev tooling, so a freshly-bootstrapped venv
    can't run ``pytest``. This helper bridges the gap without forcing
    every customer install to pull the dev tree.
    """
    print("[repair_venv] installing [dev] extras", flush=True)
    subprocess.check_call(
        [str(VENV_PY), "-m", "pip", "install", "-e", f"{REPO_ROOT}[dev]", "--quiet"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect and repair a corrupted/drifted meeting-scribe venv.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report only; exit 1 if repair needed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild venv unconditionally",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="also install [dev] extras (pytest, ruff, pip-tools, ...)",
    )
    args = parser.parse_args(argv)

    if args.force:
        _rebuild_venv()
        if args.dev:
            _install_dev_extras()
        print("[repair_venv] forced rebuild complete", flush=True)
        return 0

    issues: list[str] = []
    pinned = _pinned_python_version()
    current = _venv_python_version()
    if pinned and current and pinned != current:
        issues.append(
            f"python version drift: mise.toml pins {'.'.join(map(str, pinned))}, "
            f".venv has {'.'.join(map(str, current))}"
        )
    if pinned and current is None and VENV_DIR.exists():
        issues.append(".venv exists but its python is unrunnable")

    bad = _broken_wrappers()
    for p in bad:
        issues.append(f"console script unhealthy: {p}")

    if not issues:
        if args.dev:
            _install_dev_extras()
        return 0

    for msg in issues:
        print(f"[repair_venv] issue: {msg}", flush=True)

    if args.check:
        return 1

    needs_rebuild = bool(
        (pinned and current and pinned != current)
        or (pinned and current is None and VENV_DIR.exists())
    )
    if needs_rebuild:
        _rebuild_venv()
    else:
        _reinstall_wrappers()
    if args.dev:
        _install_dev_extras()
    print("[repair_venv] repair complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
