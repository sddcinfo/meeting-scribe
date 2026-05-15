#!/usr/bin/env python3
"""Install the Tailwind v4 standalone CLI for this repo, offline-deployable.

The Tailwind build runs entirely offline once the binary is present. The repo
pins a per-platform sha256 in ``scripts/tailwind_versions.json``; this script
downloads the matching binary from upstream once, verifies the hash against
the repo-committed value (NOT against an upstream-fetched sha256sums.txt —
that would defeat the supply-chain control), and drops it at ``.tools/tailwindcss``.

Idempotent. If the on-disk binary already matches the pinned sha256 the
script skips the download. To force a refresh, delete ``.tools/tailwindcss``.

Platform detection:
  - Linux x86_64 / aarch64 (glibc by default; set MEETING_SCRIBE_TAILWIND_MUSL=1
    on Alpine or other musl distros to pick the -musl asset).
  - macOS arm64 / x86_64.
  - Windows x64.

If your platform isn't supported, the script exits non-zero with a clear
message. The committed ``static/css/dist/*.css`` artifacts mean you can still
work on the repo as a verifier without rebuilding; you just can't author
new CSS until you're on a supported platform.

Usage:
  python3 scripts/install_tailwind.py            # idempotent install
  python3 scripts/install_tailwind.py --verify   # exit 0 if already installed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "scripts" / "tailwind_versions.json"
INSTALL_DIR = REPO_ROOT / ".tools"
BINARY_PATH = INSTALL_DIR / "tailwindcss"


def _platform_asset_name() -> str:
    """Map host platform to the Tailwind release asset name.

    Raises SystemExit on unsupported hosts so callers get a clear error and
    the rest of the build pipeline doesn't pretend it found a binary.
    """
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Linux":
        # Default to glibc; explicit opt-in for musl (Alpine etc.).
        suffix = "-musl" if os.environ.get("MEETING_SCRIBE_TAILWIND_MUSL") == "1" else ""
        if machine in ("aarch64", "arm64"):
            return f"tailwindcss-linux-arm64{suffix}"
        if machine in ("x86_64", "amd64"):
            return f"tailwindcss-linux-x64{suffix}"
        sys.exit(f"Unsupported Linux machine '{machine}'. See scripts/tailwind_versions.json.")

    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return "tailwindcss-macos-arm64"
        if machine in ("x86_64", "amd64"):
            return "tailwindcss-macos-x64"
        sys.exit(f"Unsupported macOS machine '{machine}'.")

    if system == "Windows":
        return "tailwindcss-windows-x64.exe"

    sys.exit(
        f"Unsupported platform: system='{system}', machine='{machine}'. "
        "The repo ships pre-built CSS at static/css/dist/, so you can still "
        "develop as a verifier — you just can't rebuild Tailwind output."
    )


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    with MANIFEST.open() as fh:
        return json.load(fh)


def _download(url: str, dest: Path) -> None:
    print(f"  → downloading {url}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Exit 0 if the installed binary matches the pinned sha256; exit 1 otherwise. No download.",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    asset = _platform_asset_name()
    expected_sha = manifest["sha256"].get(asset)
    if not expected_sha:
        sys.exit(f"Asset '{asset}' is not in tailwind_versions.json. Add its sha256 to upgrade.")

    base_url = manifest["base_url"]
    version = manifest["version"]
    url = f"{base_url}/{asset}"

    print(
        f"Tailwind v{version} → asset '{asset}' (sha256 prefix {expected_sha[:12]}…)",
        file=sys.stderr,
    )

    if BINARY_PATH.exists():
        observed = _sha256_of(BINARY_PATH)
        if observed == expected_sha:
            print(f"  ✓ already installed at {BINARY_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)
            return 0
        if args.verify:
            print(
                f"  ✗ sha256 mismatch: have {observed[:12]}… expected {expected_sha[:12]}…",
                file=sys.stderr,
            )
            return 1
        print("  · existing binary has different sha256, replacing", file=sys.stderr)

    if args.verify:
        print(f"  ✗ binary not present at {BINARY_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    _download(url, BINARY_PATH)

    observed = _sha256_of(BINARY_PATH)
    if observed != expected_sha:
        BINARY_PATH.unlink()
        sys.exit(
            f"FATAL: sha256 mismatch after download.\n"
            f"  expected: {expected_sha}\n"
            f"  got:      {observed}\n"
            f"This means the upstream release was modified, the network path was tampered with, "
            f"OR scripts/tailwind_versions.json is stale. Investigate before re-running."
        )

    BINARY_PATH.chmod(BINARY_PATH.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  ✓ installed at {BINARY_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
