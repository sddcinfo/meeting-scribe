#!/usr/bin/env python3
"""Backfill slide_notes.json for every deck on disk that's missing it.

Runs ``meeting_scribe.slides.convert.extract_notes_from_pptx`` against
each deck's ``_upload.pptx`` (the original uploaded file kept for
re-rendering). Saves to ``slide_notes.json`` next to ``meta.json``.

The 2026-05-07 fix wired notes extraction into the prod worker
pipeline (slides/worker.py), so all NEW uploads get notes
automatically. Decks uploaded before that fix don't have the file —
this script catches them up without forcing the operator to
re-upload.

Idempotent: skips decks that already have ``slide_notes.json``
unless ``--force`` is set.

Usage:
    PYTHONPATH=src python3 scripts/backfill_slide_notes.py
    PYTHONPATH=src python3 scripts/backfill_slide_notes.py --force
    PYTHONPATH=src python3 scripts/backfill_slide_notes.py --root meetings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "meetings",
        help="Meetings root (default: repo's meetings dir)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if slide_notes.json already exists",
    )
    args = parser.parse_args()

    from meeting_scribe.slides.convert import extract_notes_from_pptx

    if not args.root.is_dir():
        print(f"FAIL: meetings root {args.root} doesn't exist")
        return 2

    pptx_paths = list(args.root.glob("*/slides/*/_upload.pptx"))
    if not pptx_paths:
        print(f"No decks found under {args.root}")
        return 0

    print(f"Found {len(pptx_paths)} deck(s).")
    written = 0
    skipped = 0
    failed = 0
    for pptx in sorted(pptx_paths):
        deck_dir = pptx.parent
        notes_path = deck_dir / "slide_notes.json"
        if notes_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            notes = extract_notes_from_pptx(pptx)
            notes_path.write_text(json.dumps({"notes": notes}, ensure_ascii=False, indent=2))
            non_empty = sum(1 for n in notes if n.strip())
            written += 1
            print(
                f"  ✓ {deck_dir.relative_to(args.root)}: {non_empty}/{len(notes)} slides have notes"
            )
        except Exception as exc:
            failed += 1
            print(f"  ✗ {deck_dir.relative_to(args.root)}: {exc}")

    print(f"\nDone. wrote: {written}  skipped: {skipped}  failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
