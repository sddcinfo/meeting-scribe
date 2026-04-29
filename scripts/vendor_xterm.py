#!/usr/bin/env python3
"""Vendor xterm.js + first-party addons into static/vendor/xterm/.

Pure stdlib — no npm, no Node.js. Fetches tarballs directly from the
npm registry, extracts the UMD bundles + CSS, and writes them to a flat
directory. Idempotent: re-running produces the same files as long as
upstream versions match :data:`PACKAGES`.

Usage:
    python scripts/vendor_xterm.py                  # fetch + write
    python scripts/vendor_xterm.py --integrity-check # verify sha256s

The terminal panel loads these assets directly via ``<script>`` tags;
there is no bundler step in meeting-scribe's static pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# ─── Packages to vendor ──────────────────────────────────────────
# Each entry: (npm name, version, local basename).
# Basenames are lowercase, hyphenated — terminal-panel.js loads them
# by these exact filenames.
@dataclass(frozen=True)
class Pkg:
    npm: str
    version: str
    basename: str
    asset: str  # lib/<asset>.js OR css/xterm.css


PACKAGES: Final[list[Pkg]] = [
    # Core — must be loaded first. xterm.css is served separately.
    Pkg("@xterm/xterm", "6.0.0", "xterm", "lib/xterm.js"),
    # Addons — UMD builds under lib/. Loaded after the core.
    Pkg("@xterm/addon-fit", "0.11.0", "addon-fit", "lib/addon-fit.js"),
    Pkg("@xterm/addon-webgl", "0.19.0", "addon-webgl", "lib/addon-webgl.js"),
    Pkg("@xterm/addon-clipboard", "0.2.0", "addon-clipboard", "lib/addon-clipboard.js"),
    Pkg("@xterm/addon-web-links", "0.12.0", "addon-web-links", "lib/addon-web-links.js"),
    Pkg("@xterm/addon-search", "0.16.0", "addon-search", "lib/addon-search.js"),
    Pkg(
        "@xterm/addon-unicode-graphemes",
        "0.4.0",
        "addon-unicode-graphemes",
        "lib/addon-unicode-graphemes.js",
    ),
    Pkg("@xterm/addon-serialize", "0.14.0", "addon-serialize", "lib/addon-serialize.js"),
]

# The core CSS also needs to be vendored — UMD JS alone doesn't include styles.
CSS_PKG = Pkg("@xterm/xterm", "6.0.0", "xterm.css", "css/xterm.css")


VENDOR_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "static" / "vendor" / "xterm"
VERSIONS_FILE: Final[str] = "VERSIONS.json"


def _registry_url(pkg: Pkg) -> str:
    # Scoped packages: https://registry.npmjs.org/@xterm/xterm/-/xterm-5.5.0.tgz
    # Tarball filename is the unscoped name + version.
    unscoped = pkg.npm.split("/", 1)[1] if "/" in pkg.npm else pkg.npm
    return f"https://registry.npmjs.org/{pkg.npm}/-/{unscoped}-{pkg.version}.tgz"


def _fetch(url: str) -> bytes:
    print(f"  fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def _extract_member(tar_bytes: bytes, member_relpath: str) -> bytes:
    """Extract ``package/<member_relpath>`` from a tarball."""
    buf = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        target = f"package/{member_relpath}"
        try:
            member = tf.getmember(target)
        except KeyError as e:
            names = [
                m.name
                for m in tf.getmembers()
                if m.name.startswith("package/lib") or m.name.startswith("package/css")
            ]
            raise RuntimeError(
                f"member {target!r} not found in tarball; saw lib/css entries: {names[:20]}"
            ) from e
        f = tf.extractfile(member)
        if f is None:
            raise RuntimeError(f"tarball member {target!r} has no data")
        return f.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(target: Path, data: bytes) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)


def _load_versions() -> dict:
    path = VENDOR_DIR / VERSIONS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_versions(data: dict) -> None:
    path = VENDOR_DIR / VERSIONS_FILE
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def vendor() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    versions: dict[str, dict[str, str]] = {}
    all_pkgs = list(PACKAGES) + [CSS_PKG]
    for pkg in all_pkgs:
        print(f"▶ {pkg.npm}@{pkg.version} ({pkg.asset} → {pkg.basename})", file=sys.stderr)
        tar_bytes = _fetch(_registry_url(pkg))
        member_bytes = _extract_member(tar_bytes, pkg.asset)
        target = VENDOR_DIR / (pkg.basename + (".js" if pkg.asset.endswith(".js") else ""))
        # If basename already has an extension, don't double-append.
        if pkg.basename.endswith(".css") or pkg.basename.endswith(".js"):
            target = VENDOR_DIR / pkg.basename
        _write_atomic(target, member_bytes)
        versions[pkg.basename] = {
            "npm": pkg.npm,
            "version": pkg.version,
            "asset": pkg.asset,
            "sha256": _sha256(member_bytes),
            "size": str(len(member_bytes)),
        }
    _save_versions(versions)
    print(f"✓ wrote {len(versions)} files to {VENDOR_DIR}", file=sys.stderr)


def integrity_check() -> int:
    versions = _load_versions()
    if not versions:
        print(
            "no VERSIONS.json in vendor dir — run `python scripts/vendor_xterm.py` first",
            file=sys.stderr,
        )
        return 2
    failed = 0
    for basename, meta in versions.items():
        target = VENDOR_DIR / basename
        if not (basename.endswith(".css") or basename.endswith(".js")):
            target = VENDOR_DIR / (basename + ".js")
        if not target.exists():
            print(f"  ✗ {basename}: missing", file=sys.stderr)
            failed += 1
            continue
        got = _sha256(target.read_bytes())
        want = meta.get("sha256")
        if got != want:
            print(
                f"  ✗ {basename}: sha256 mismatch\n      got  {got}\n      want {want}",
                file=sys.stderr,
            )
            failed += 1
        else:
            print(f"  ✓ {basename}", file=sys.stderr)
    if failed:
        print(f"✗ {failed} file(s) failed integrity check", file=sys.stderr)
        return 1
    print(f"✓ all {len(versions)} files match recorded sha256s", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--integrity-check",
        action="store_true",
        help="Verify on-disk files against recorded sha256s; exit nonzero on mismatch",
    )
    args = p.parse_args()
    if args.integrity_check:
        return integrity_check()
    vendor()
    return 0


if __name__ == "__main__":
    sys.exit(main())
