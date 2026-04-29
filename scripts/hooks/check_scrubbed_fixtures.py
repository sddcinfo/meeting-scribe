#!/usr/bin/env python3
"""Shape-only validator for ``tests/fixtures/journals/*.jsonl``.

Runs as a pre-commit hook AND in CI. **Does NOT reference any data
derived from real ``meetings/`` content** — only the fixture file and
static schema/enum data are inspected. Content-aware leak detection
is the scrubber's job (``scripts/scrub_journal.py``) and runs only on
the developer's machine where the raw journals already exist.

Invariants enforced:

  1. Every record's keys are in the allowlist.
  2. The record passes a minimal Pydantic-like shape check.
  3. Enum-typed fields are values from the corresponding registry.
  4. Free-form text fields match a generic synthetic-shape heuristic
     (no URL-like substrings, no email-like substrings, no contiguous
     digit runs > 6 chars). Shape-only — no real-content reference.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "journals"

_TOP_ALLOWED = {
    "segment_id", "revision", "is_final", "start_ms", "end_ms",
    "language", "target_language", "text", "translation",
    "furigana_html", "speakers",
}
_TR_ALLOWED = {"status", "text", "target_language", "completed_at", "furigana_html"}
_SPK_ALLOWED = {"cluster_id", "source"}
_LANGS = {"en", "ja", "de", "fr", "es", "pt", "zh", "ko", "auto"}
_TR_STATUS = {"pending", "in_progress", "done", "failed", "skipped"}
_SPK_SOURCE = {
    "diarization", "diarize", "enrolled", "orphan_reassigned",
    "self_intro", "manual", "time_proximity",
}

_RE_URL = re.compile(r"https?://|www\.[a-z]")
_RE_EMAIL = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
_RE_LONG_DIGITS = re.compile(r"\d{7,}")


def _err(path: Path, idx: int, msg: str) -> str:
    return f"{path.relative_to(REPO_ROOT)}:{idx + 1}: {msg}"


def _shape_check_text(s: str) -> str | None:
    """Return an error string if ``s`` looks like real-world content
    rather than synthetic placeholder text."""
    if not isinstance(s, str):
        return f"non-string text: {type(s).__name__}"
    if _RE_URL.search(s):
        return f"URL-like substring in text: {s[:60]!r}"
    if _RE_EMAIL.search(s):
        return f"email-like substring in text: {s[:60]!r}"
    if _RE_LONG_DIGITS.search(s):
        return f"long digit run in text (looks like an ID): {s[:60]!r}"
    return None


def _validate_record(record: dict, path: Path, idx: int) -> list[str]:
    errs: list[str] = []
    for k in record:
        if k not in _TOP_ALLOWED:
            errs.append(_err(path, idx, f"unknown top-level key {k!r}"))
    if not isinstance(record.get("segment_id"), str) or not record["segment_id"]:
        errs.append(_err(path, idx, "missing/invalid segment_id"))
    if record.get("language") not in _LANGS:
        errs.append(_err(path, idx, f"invalid language {record.get('language')!r}"))
    if "text" in record:
        sh = _shape_check_text(record["text"])
        if sh:
            errs.append(_err(path, idx, sh))
    tr = record.get("translation")
    if tr is not None:
        if not isinstance(tr, dict):
            errs.append(_err(path, idx, "translation must be a dict"))
        else:
            for k in tr:
                if k not in _TR_ALLOWED:
                    errs.append(_err(path, idx, f"unknown translation key {k!r}"))
            if tr.get("status") not in _TR_STATUS:
                errs.append(_err(path, idx, f"invalid translation.status {tr.get('status')!r}"))
            tlang = tr.get("target_language")
            if tlang is not None and tlang not in _LANGS:
                errs.append(_err(path, idx, f"invalid translation.target_language {tlang!r}"))
            if "text" in tr:
                sh = _shape_check_text(tr["text"])
                if sh:
                    errs.append(_err(path, idx, f"translation.{sh}"))
    speakers = record.get("speakers")
    if speakers is not None:
        if not isinstance(speakers, list):
            errs.append(_err(path, idx, "speakers must be a list"))
        else:
            for s in speakers:
                if not isinstance(s, dict):
                    errs.append(_err(path, idx, "speaker must be a dict"))
                    continue
                for k in s:
                    if k not in _SPK_ALLOWED:
                        errs.append(_err(path, idx, f"unknown speaker key {k!r}"))
                src = s.get("source")
                if src is not None and src not in _SPK_SOURCE:
                    errs.append(_err(path, idx, f"invalid speaker.source {src!r}"))
    return errs


def main() -> int:
    if not FIXTURES_DIR.exists():
        print(f"no fixtures dir at {FIXTURES_DIR.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 0
    all_errs: list[str] = []
    files = list(FIXTURES_DIR.glob("*.jsonl"))
    if not files:
        return 0
    for path in files:
        for idx, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                all_errs.append(_err(path, idx, f"invalid JSON: {e}"))
                continue
            all_errs.extend(_validate_record(record, path, idx))
    if all_errs:
        print("✗ scrubbed-fixture validation failed:", file=sys.stderr)
        for e in all_errs[:30]:
            print(f"  {e}", file=sys.stderr)
        if len(all_errs) > 30:
            print(f"  ...and {len(all_errs) - 30} more", file=sys.stderr)
        return 1
    print(f"  ✓ scrubbed-fixture validator ({len(files)} files OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
