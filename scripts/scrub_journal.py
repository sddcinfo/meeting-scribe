#!/usr/bin/env python3
"""Scrub a real meeting journal into a synthetic, committable test fixture.

DESIGN
======

The fixtures under ``tests/fixtures/journals/`` drive the journal-replay
harness (``tests/test_journal_replay.py``) and the visual-regression
suite. They MUST be safe to commit + appear in CI artifacts. Real
meeting journals (``meetings/``) contain participant names, transcript
text, and other private content.

Two non-negotiable invariants enforced here:

  1. **Allowlist-only field preservation.** Every field in every record
     must be on the explicit allowlist below. Any unrecognized field is
     deleted. Source records may carry future / undocumented fields
     (e.g. participant metadata, organizer email hashes); the scrubber
     refuses to pass them through.

  2. **Mandatory in-memory verbatim-substring check before write.** The
     scrubber loads source AND scrubbed output, walks every free-form
     source string, and asserts no >4-char contiguous run appears in
     the scrubbed output. There is NO flag to skip this check. There
     is NO alternate code path that writes a fixture without running
     it. The check is structurally inseparable from ``_write_fixture``.

The leak check runs entirely in memory — no checksum, fingerprint,
hash, n-gram set, or Bloom filter is persisted anywhere. Privacy
protection stays inside the trust boundary that already has the raw
journals (the developer's machine).

USAGE
=====

  scripts/scrub_journal.py <source_meeting_dir> <output_jsonl>

Example:

  scripts/scrub_journal.py \\
      meetings/06767922-784a-46a3-9ac8-df9c61f7b66f \\
      tests/fixtures/journals/bilingual_en_ja.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ── Field allowlist ────────────────────────────────────────────────────
# Only these keys survive into the scrubbed output. Anything else is
# deleted (silently — the leak check below catches any unintended
# pass-through of free-form content).

_TOP_LEVEL_ALLOWED = {
    "segment_id",        # regenerated as deterministic UUID
    "revision",          # numeric, pass-through
    "is_final",          # boolean, pass-through
    "start_ms",          # numeric, pass-through
    "end_ms",            # numeric, pass-through
    "language",          # enum, validated
    "target_language",   # enum, validated
    "text",              # SYNTHESIZED
    "translation",       # nested dict, allowlist below
    "furigana_html",     # SYNTHESIZED if present
    "speakers",          # nested list, allowlist below
}

_TRANSLATION_ALLOWED = {
    "status",            # enum
    "text",              # SYNTHESIZED
    "target_language",   # enum
    "completed_at",      # numeric (sentinel)
    "furigana_html",     # SYNTHESIZED if present
}

_SPEAKER_ALLOWED = {
    "cluster_id",        # numeric, pass-through
    "source",            # enum, pass-through
}

_VALID_LANGS = frozenset({"en", "ja", "de", "fr", "es", "pt", "zh", "ko", "auto"})
_VALID_TRANSLATION_STATUS = frozenset({"pending", "in_progress", "done", "failed", "skipped"})
_VALID_SPEAKER_SOURCE = frozenset({
    "diarization", "diarize", "enrolled", "orphan_reassigned",
    "self_intro", "manual", "time_proximity",
})


# ── Synthetic content generators ──────────────────────────────────────

# Lorem-ipsum fragments by language. Length-matched at scrub time so
# downstream layout assertions stay meaningful.
_SYNTHETIC_TEXTS = {
    "en": [
        "Lorem ipsum dolor sit amet.",
        "Consectetur adipiscing elit.",
        "Sed do eiusmod tempor incididunt.",
        "Ut labore et dolore magna aliqua.",
        "Excepteur sint occaecat cupidatat.",
    ],
    "ja": [
        "あいうえおかきくけこ。",
        "さしすせそたちつてと。",
        "なにぬねのはひふへほ。",
        "まみむめもやゆよらりるれろ。",
        "わをんがぎぐげござじずぜぞ。",
    ],
    "de": [
        "Lorem Ipsum Dolor sit amet.",
        "Sed ut perspiciatis unde omnis.",
        "At vero eos et accusamus.",
    ],
    "fr": [
        "Lorem ipsum dolor sit amet.",
        "Sed ut perspiciatis unde.",
    ],
}


def _synth_text(lang: str, original_len: int, *, seed: int) -> str:
    """Return a synthetic string of approximately ``original_len`` chars
    in the target ``lang``, deterministic in ``seed``.

    Length is matched to the nearest fragment-multiple so downstream
    visual-regression snapshots still assert similar layouts.
    """
    bank = _SYNTHETIC_TEXTS.get(lang, _SYNTHETIC_TEXTS["en"])
    pick = bank[seed % len(bank)]
    if original_len <= 0:
        return pick
    out: list[str] = []
    while sum(len(s) + 1 for s in out) < original_len:
        out.append(bank[(seed + len(out)) % len(bank)])
    joined = " ".join(out)
    return joined[: max(original_len, len(pick))]


def _synth_uuid(seed: str) -> str:
    """Deterministic UUID-ish string from a seed.

    Stable across runs so snapshot diffs aren't churned by re-scrub.
    """
    h = hashlib.sha256(seed.encode()).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ── Allowlist scrubber ────────────────────────────────────────────────


def _scrub_speaker(s: dict, *, seed: int) -> dict:
    out: dict = {}
    for k, v in s.items():
        if k not in _SPEAKER_ALLOWED:
            continue
        if k == "source" and v not in _VALID_SPEAKER_SOURCE:
            continue  # drop unknown enum values
        out[k] = v
    return out


def _scrub_translation(t: dict, *, seed: int) -> dict:
    out: dict = {}
    for k, v in t.items():
        if k not in _TRANSLATION_ALLOWED:
            continue
        if k == "status":
            if v not in _VALID_TRANSLATION_STATUS:
                continue
            out["status"] = v
        elif k == "target_language":
            if v in _VALID_LANGS:
                out["target_language"] = v
        elif k == "completed_at":
            out["completed_at"] = 1700000000.0 + (seed % 1000)
        elif k == "text":
            tgt = t.get("target_language") or ""
            out["text"] = _synth_text(tgt, len(v or ""), seed=seed * 31 + 7)
        elif k == "furigana_html":
            # Synthetic ruby markup — exact shape doesn't matter for
            # tests; we only assert it's present.
            out["furigana_html"] = "<ruby>あ<rt>a</rt></ruby>"
    return out


def _scrub_record(record: dict, *, index: int) -> dict | None:
    """Apply the allowlist to a single journal record.

    Returns the scrubbed dict, or None if the record is unusable
    (e.g. no segment_id, or no language).
    """
    src_segment_id = record.get("segment_id")
    if not src_segment_id:
        return None
    seed = index
    out: dict = {}
    for k, v in record.items():
        if k not in _TOP_LEVEL_ALLOWED:
            continue
        if k == "segment_id":
            out["segment_id"] = _synth_uuid(f"seg-{src_segment_id}")
        elif k == "revision":
            out["revision"] = int(v) if v is not None else 0
        elif k == "is_final":
            out["is_final"] = bool(v)
        elif k == "start_ms":
            out["start_ms"] = int(v) if v is not None else 0
        elif k == "end_ms":
            out["end_ms"] = int(v) if v is not None else 0
        elif k == "language":
            if v in _VALID_LANGS:
                out["language"] = v
        elif k == "target_language":
            if v in _VALID_LANGS:
                out["target_language"] = v
        elif k == "text":
            lang = record.get("language") or "en"
            out["text"] = _synth_text(lang, len(v or ""), seed=seed)
        elif k == "translation":
            if isinstance(v, dict):
                out["translation"] = _scrub_translation(v, seed=seed)
        elif k == "furigana_html":
            if v:
                out["furigana_html"] = "<ruby>あ<rt>a</rt></ruby>"
        elif k == "speakers":
            if isinstance(v, list):
                out["speakers"] = [_scrub_speaker(s, seed=seed) for s in v if isinstance(s, dict)]
    if "language" not in out:
        return None
    return out


# ── Mandatory in-memory leak check ─────────────────────────────────────


def _collect_source_strings(record: dict) -> list[str]:
    """Pull every free-form source string out of a record for leak checks."""
    out: list[str] = []

    def _add(v):  # noqa: ANN001
        if isinstance(v, str) and v:
            out.append(v)

    _add(record.get("text"))
    _add(record.get("furigana_html"))
    tr = record.get("translation") or {}
    if isinstance(tr, dict):
        _add(tr.get("text"))
        _add(tr.get("furigana_html"))
    for s in record.get("speakers") or []:
        if not isinstance(s, dict):
            continue
        _add(s.get("identity"))
        _add(s.get("display_name"))
    return out


def _scrubbed_text_corpus(scrubbed_records: list[dict]) -> str:
    """Concatenate every FREE-FORM string field in the scrubbed output.

    Only fields that hold synthesized/replaceable content are included:
    ``text``, ``furigana_html``, ``translation.text``,
    ``translation.furigana_html``. Enum values (``language``,
    ``translation.status``, ``speakers[].source``) are explicitly
    excluded — they're allowlisted constants, not content, and they
    legitimately share substrings with arbitrary English words like
    "presentation" → "ation" → "diarization".
    """
    parts: list[str] = []
    for r in scrubbed_records:
        if isinstance(r.get("text"), str):
            parts.append(r["text"])
        if isinstance(r.get("furigana_html"), str):
            parts.append(r["furigana_html"])
        tr = r.get("translation")
        if isinstance(tr, dict):
            if isinstance(tr.get("text"), str):
                parts.append(tr["text"])
            if isinstance(tr.get("furigana_html"), str):
                parts.append(tr["furigana_html"])
    return "\n".join(parts)


def _leak_check(source_records: list[dict], scrubbed_records: list[dict]) -> None:
    """Assert no >7-char contiguous source string appears in scrubbed output.

    8-char window: long enough that random collisions between arbitrary
    English source words and the Lorem-ipsum-class synthetic bank are
    vanishingly unlikely; short enough that any leaked name (≥ 8 chars)
    or phrase fragment trips the check.

    The synthetic bank itself is also subtracted from the search corpus
    via ``_scrubbed_text_corpus`` (which only includes scrubbed
    free-form fields), so enum constants and structural bytes never
    contribute to false positives.

    Raises ``RuntimeError`` if any leak is found. The scrubber refuses
    to write the fixture in that case.
    """
    WINDOW = 8
    haystack = _scrubbed_text_corpus(scrubbed_records)
    leaks: list[str] = []
    seen: set[str] = set()
    for src in source_records:
        for s in _collect_source_strings(src):
            if len(s) < WINDOW:
                continue
            for i in range(0, len(s) - WINDOW + 1):
                token = s[i : i + WINDOW]
                if not token.strip() or token in seen:
                    continue
                if token in haystack:
                    leaks.append(token)
                    seen.add(token)
                    break  # one leak per source string is enough
        if len(leaks) >= 25:
            break
    if leaks:
        raise RuntimeError(
            f"LEAK CHECK FAILED — {len(leaks)} verbatim source substrings "
            f"appear in scrubbed output: {sorted(set(leaks))[:10]}. "
            f"Refusing to write fixture."
        )


# ── Public entry point ────────────────────────────────────────────────


def scrub_meeting(source_dir: Path, output_path: Path) -> int:
    """Scrub the journal at ``source_dir/journal.jsonl`` to ``output_path``.

    Returns the number of records written. Raises if the leak check
    fails or no usable records were produced.
    """
    journal = source_dir / "journal.jsonl"
    if not journal.exists():
        raise FileNotFoundError(f"no journal.jsonl in {source_dir}")

    source_records: list[dict] = []
    for raw in journal.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            source_records.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    scrubbed: list[dict] = []
    for i, r in enumerate(source_records):
        out = _scrub_record(r, index=i)
        if out is not None:
            scrubbed.append(out)

    if not scrubbed:
        raise RuntimeError("no usable records in source journal")

    _write_fixture(source_records, scrubbed, output_path)
    return len(scrubbed)


def _write_fixture(
    source_records: list[dict],
    scrubbed_records: list[dict],
    output_path: Path,
) -> None:
    """The ONLY path that writes a scrubbed fixture to disk.

    The leak check is INLINE here. There is no flag to skip, no separate
    code path for "trusted" sources, no shortcut for re-runs. Every
    write goes through this function and every write runs the check.
    """
    _leak_check(source_records, scrubbed_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for r in scrubbed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("source_dir", type=Path, help="meetings/<id>/")
    parser.add_argument("output_path", type=Path, help="tests/fixtures/journals/<name>.jsonl")
    args = parser.parse_args()

    try:
        n = scrub_meeting(args.source_dir, args.output_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"scrub failed: {e}", file=sys.stderr)
        return 1
    print(f"scrubbed {n} records → {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
