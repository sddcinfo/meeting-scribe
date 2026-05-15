#!/usr/bin/env python3
"""Build the Tailwind v4 CSS bundles for meeting-scribe — offline-only.

The Tailwind v4 standalone CLI runs entirely offline once
``scripts/install_tailwind.py`` has dropped the platform-matching binary at
``.tools/tailwindcss``. This script orchestrates a multi-entry build: every
``*.css`` file directly under ``static/css/src/`` (NOT recursive — partials
in ``static/css/src/components/`` are imported by entries, not built
themselves) becomes an output at ``static/css/dist/<same-name>.css``.

Two modes:

  --mode build (default)
      Author workflow. Runs tailwindcss for every entry, writes
      ``static/css/dist/*.css``, writes ``static/css/dist/manifest.json``
      (basename → short content hash), and rewrites the ``?v=…`` query
      strings in every HTML file under ``static/**/*.html`` that links a
      managed stylesheet. May mutate the working tree.

  --mode check
      Pre-push hook / CI verifier. Runs the same build into
      ``.build-cache/css-check/`` (gitignored) and byte-compares each output
      against the committed file via ``filecmp.cmp(shallow=False)``. Also
      walks every HTML file and asserts that its ``?v=`` query strings
      match the candidate manifest. Does NOT mutate the working tree.
      Exits 1 with a precise list of stale paths on any drift; exits 0
      otherwise.

The deployment story:
  * ``static/css/dist/*.css`` is COMMITTED to the repo. The appliance ships
    pre-built CSS — offline installs never run this script.
  * ``--mode check`` is the gate. It trips iff the committed artifacts
    diverge from what a fresh build would produce.

Usage:
  python3 scripts/build_css.py                # build (default)
  python3 scripts/build_css.py --mode check   # verify only

Implementation notes:
  * Cache-bust strategy: only stylesheet links whose basename is in the
    manifest get their ``?v=`` rewritten. Legacy ``static/css/tokens.css``,
    ``style.css`` etc. (deleted in Phase 5) keep their manual stamps.
  * The build is reproducible: same source → same minified output → same
    content hash → same ``?v=``. Determinism matters because the diff is
    reviewed.
  * Empty ``static/css/src/`` (Phase 0 state) is fine: the script does
    nothing and exits 0.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAILWIND_BIN = REPO_ROOT / ".tools" / "tailwindcss"
SRC_DIR = REPO_ROOT / "static" / "css" / "src"
DIST_DIR = REPO_ROOT / "static" / "css" / "dist"
CACHE_DIR = REPO_ROOT / ".build-cache" / "css-check"
HTML_ROOT = REPO_ROOT / "static"
MANIFEST_BASENAME = "manifest.json"

LINK_RE = re.compile(
    r"""<link\b[^>]*?\brel\s*=\s*["']stylesheet["'][^>]*?\bhref\s*=\s*(["'])([^"']+?)\1""",
    re.IGNORECASE,
)


def _short_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _discover_entries() -> list[Path]:
    """Entry CSS = direct children of static/css/src/ ending in .css."""
    if not SRC_DIR.is_dir():
        return []
    return sorted(p for p in SRC_DIR.iterdir() if p.is_file() and p.suffix == ".css")


def _run_tailwind(input_path: Path, output_path: Path) -> None:
    """Invoke the standalone CLI for one entry. Minified, offline."""
    if not TAILWIND_BIN.exists():
        sys.exit(
            f"Tailwind binary not found at {TAILWIND_BIN.relative_to(REPO_ROOT)}. "
            "Run `python3 scripts/install_tailwind.py` first."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(TAILWIND_BIN), "-i", str(input_path), "-o", str(output_path), "--minify"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(f"tailwindcss failed for {input_path.relative_to(REPO_ROOT)}")


def _build_to(out_dir: Path, entries: list[Path]) -> dict[str, str]:
    """Build every entry into out_dir/<entry>. Return manifest dict (basename → hash)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for entry in entries:
        output_path = out_dir / entry.name
        _run_tailwind(entry, output_path)
        manifest[entry.name] = _short_hash(output_path)
    return manifest


def _write_manifest(out_dir: Path, manifest: dict[str, str]) -> None:
    """Write manifest.json deterministically (sorted keys, trailing newline)."""
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out_dir / MANIFEST_BASENAME).write_text(payload, encoding="utf-8")


def _iter_html_files() -> list[Path]:
    """Every HTML file under static/, recursively. Sorted for determinism."""
    return sorted(HTML_ROOT.rglob("*.html"))


def _rewrite_html_inline(html_text: str, manifest: dict[str, str]) -> tuple[str, list[str]]:
    """Rewrite <link rel="stylesheet" href="…?v=…"> for managed dist files.

    Only touches links whose path part contains ``/css/dist/``. This is the
    namespace fence between legacy hand-authored stylesheets at
    ``static/css/<name>.css`` (which keep their hand-managed ``?v=…``) and
    the Tailwind build output at ``static/css/dist/<name>.css`` (managed by
    this script). Some basenames collide across the two (tokens.css,
    portal.css, how-it-works.css); the path-prefix check is what keeps the
    rewriter from stamping legacy links with build hashes.

    Returns (new_text, list_of_changed_links). Untouched links are unchanged.
    """
    changes: list[str] = []

    out_parts: list[str] = []
    pos = 0
    tag_re = re.compile(r"<link\b[^>]*?>", re.IGNORECASE | re.DOTALL)
    for m in tag_re.finditer(html_text):
        out_parts.append(html_text[pos : m.start()])
        tag = m.group(0)
        href_m = re.search(r"""href\s*=\s*(["'])([^"']+?)\1""", tag, re.IGNORECASE)
        rel_m = re.search(r"""rel\s*=\s*(["'])([^"']+?)\1""", tag, re.IGNORECASE)
        if not href_m or not rel_m or "stylesheet" not in rel_m.group(2).lower():
            out_parts.append(tag)
            pos = m.end()
            continue
        quote, href = href_m.group(1), href_m.group(2)
        # Strip query/fragment, then require the path to point at the
        # managed dist directory. Legacy stylesheets keep their own stamps.
        # Matches both absolute (``/static/css/dist/…``) and relative
        # (``css/dist/…``) hrefs — how-it-works.html uses relative paths
        # so the same source can be published statically (GitHub Pages)
        # without rewriting URLs at publish time.
        path_part = href.split("?", 1)[0].split("#", 1)[0]
        if "/css/dist/" not in path_part and not path_part.startswith("css/dist/"):
            out_parts.append(tag)
            pos = m.end()
            continue
        basename = path_part.rsplit("/", 1)[-1]
        if basename not in manifest:
            out_parts.append(tag)
            pos = m.end()
            continue
        new_href = f"{path_part}?v={manifest[basename]}"
        if new_href == href:
            out_parts.append(tag)
            pos = m.end()
            continue
        new_tag = tag[: href_m.start()] + f"href={quote}{new_href}{quote}" + tag[href_m.end() :]
        out_parts.append(new_tag)
        changes.append(f"{basename}: {href} → {new_href}")
        pos = m.end()
    out_parts.append(html_text[pos:])
    return "".join(out_parts), changes


def _build_mode(entries: list[Path]) -> int:
    print(f"build_css.py --mode build: {len(entries)} entry(ies)", file=sys.stderr)
    if not entries:
        # Phase 0 state: no src/ yet. Wipe stale dist + manifest if they exist.
        if DIST_DIR.exists():
            # Don't actually wipe — Phase 5 cleanup deletes legacy; for now,
            # just refuse to overwrite a non-empty dist when src is empty so
            # we don't silently nuke checked-in CSS.
            print(
                "  · no src/ entries, dist/ untouched. "
                "(This is expected in Phase 0 before any src/*.css is authored.)",
                file=sys.stderr,
            )
        return 0
    manifest = _build_to(DIST_DIR, entries)
    _write_manifest(DIST_DIR, manifest)
    print(f"  ✓ built {len(manifest)} entries → static/css/dist/", file=sys.stderr)

    # Rewrite cache-bust stamps across every HTML file.
    touched = 0
    for html_path in _iter_html_files():
        original = html_path.read_text(encoding="utf-8")
        rewritten, changes = _rewrite_html_inline(original, manifest)
        if rewritten != original:
            html_path.write_text(rewritten, encoding="utf-8")
            touched += 1
            for change in changes:
                print(f"  ~ {html_path.relative_to(REPO_ROOT)}: {change}", file=sys.stderr)
    print(f"  ✓ rewrote ?v= in {touched} HTML file(s)", file=sys.stderr)
    return 0


def _check_mode(entries: list[Path]) -> int:
    print(f"build_css.py --mode check: {len(entries)} entry(ies)", file=sys.stderr)
    if not entries:
        # Nothing to check. Same Phase 0 graceful path.
        return 0

    # Build into a temp dir; never touch the worktree.
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    candidate_manifest = _build_to(CACHE_DIR, entries)

    drift: list[str] = []

    # Compare each CSS output byte-for-byte against the committed file.
    for basename in candidate_manifest:
        candidate = CACHE_DIR / basename
        tracked = DIST_DIR / basename
        if not tracked.exists():
            drift.append(f"missing committed file: static/css/dist/{basename}")
            continue
        if not filecmp.cmp(str(candidate), str(tracked), shallow=False):
            drift.append(f"stale: static/css/dist/{basename}")

    # Compare manifest.json.
    candidate_manifest_path = CACHE_DIR / MANIFEST_BASENAME
    _write_manifest(CACHE_DIR, candidate_manifest)
    tracked_manifest = DIST_DIR / MANIFEST_BASENAME
    if not tracked_manifest.exists():
        drift.append(f"missing committed file: static/css/dist/{MANIFEST_BASENAME}")
    elif not filecmp.cmp(str(candidate_manifest_path), str(tracked_manifest), shallow=False):
        drift.append(f"stale: static/css/dist/{MANIFEST_BASENAME}")

    # Walk every HTML file and verify ?v= matches the candidate manifest.
    for html_path in _iter_html_files():
        original = html_path.read_text(encoding="utf-8")
        rewritten, changes = _rewrite_html_inline(original, candidate_manifest)
        if rewritten != original:
            for change in changes:
                drift.append(f"stale ?v= in {html_path.relative_to(REPO_ROOT)} — {change}")

    if drift:
        sys.stderr.write("✗ committed CSS/HTML is stale relative to source:\n")
        for item in drift:
            sys.stderr.write(f"    · {item}\n")
        sys.stderr.write("\nRun `python3 scripts/build_css.py` and commit the result.\n")
        return 1

    print("  ✓ committed artifacts match a fresh build", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["build", "check"],
        default="build",
        help="build = author workflow (may mutate worktree); check = verifier (read-only).",
    )
    args = parser.parse_args()

    entries = _discover_entries()
    if args.mode == "build":
        return _build_mode(entries)
    return _check_mode(entries)


if __name__ == "__main__":
    sys.exit(main())
