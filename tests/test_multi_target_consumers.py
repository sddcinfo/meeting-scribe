"""Consumer-side multi-target fan-out tests: journal dedup + export merge.

These guard the boundary between the demand-driven translation queue
(fan-out-aware) and the downstream consumers (journal replay, text
export). Pre-fan-out journals MUST continue to work; fan-out journals
MUST collapse to one row per segment with all target languages kept.
"""

from __future__ import annotations

import json
from pathlib import Path

from meeting_scribe.export import _load_events_with_corrections, transcript_to_text


def _write_journal(path: Path, events: list[dict]) -> None:
    lines = [json.dumps(e) for e in events]
    path.write_text("\n".join(lines) + "\n")


def _asr_final(sid: str, text: str, lang: str = "en", revision: int = 0) -> dict:
    return {
        "segment_id": sid,
        "revision": revision,
        "is_final": True,
        "start_ms": 0,
        "end_ms": 1000,
        "language": lang,
        "text": text,
        "speakers": [],
        "translation": None,
    }


def _translation(sid: str, text: str, source: str, target: str, revision: int = 0) -> dict:
    base = _asr_final(sid, "orig", source, revision)
    base["translation"] = {
        "status": "done",
        "text": text,
        "target_language": target,
    }
    return base


def test_export_loader_single_target_legacy(tmp_path: Path) -> None:
    """Pre-fan-out journal: ASR + single translation still loads cleanly."""
    journal = tmp_path / "journal.jsonl"
    _write_journal(
        journal,
        [
            _asr_final("s1", "hello"),
            _translation("s1", "こんにちは", "en", "ja"),
        ],
    )
    events = _load_events_with_corrections(journal)
    assert len(events) == 1
    ev = events[0]
    assert ev["text"] == "orig"  # last-seen ASR text (the translation entry)
    assert ev["translation"]["text"] == "こんにちは"
    assert ev["translations"] == {"ja": {"status": "done", "text": "こんにちは", "target_language": "ja"}}


def test_export_loader_multi_target_fanout(tmp_path: Path) -> None:
    """Fan-out journal: multiple target langs collapse to ONE event w/ translations map."""
    journal = tmp_path / "journal.jsonl"
    _write_journal(
        journal,
        [
            _asr_final("s1", "hello"),
            _translation("s1", "こんにちは", "en", "ja"),
            _translation("s1", "bonjour", "en", "fr"),
        ],
    )
    events = _load_events_with_corrections(journal)
    assert len(events) == 1
    ev = events[0]
    assert set(ev["translations"]) == {"ja", "fr"}
    assert ev["translations"]["ja"]["text"] == "こんにちは"
    assert ev["translations"]["fr"]["text"] == "bonjour"


def test_export_transcript_lang_filter_picks_target(tmp_path: Path) -> None:
    """lang='fr' should pick the French translation, skipping non-matching events."""
    journal = tmp_path / "journal.jsonl"
    _write_journal(
        journal,
        [
            _asr_final("s1", "hello"),
            _translation("s1", "こんにちは", "en", "ja"),
            _translation("s1", "bonjour", "en", "fr"),
        ],
    )
    out = transcript_to_text(journal, lang="fr")
    assert "bonjour" in out
    assert "こんにちは" not in out


def test_export_transcript_unfiltered_inlines_every_target(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    _write_journal(
        journal,
        [
            _asr_final("s1", "hello"),
            _translation("s1", "こんにちは", "en", "ja"),
            _translation("s1", "bonjour", "en", "fr"),
        ],
    )
    out = transcript_to_text(journal, lang=None)
    # Source text appears exactly once, both translations inlined.
    assert out.count("orig") == 1
    assert "[fr: bonjour]" in out
    assert "[ja: こんにちは]" in out


def test_read_journal_raw_dedup_by_segment_and_target(tmp_path: Path) -> None:
    """Journal replay: one line per (segment_id, target_language) bucket."""
    from meeting_scribe.config import ServerConfig
    from meeting_scribe.storage import MeetingStorage

    storage_dir = tmp_path / "meetings"
    storage_dir.mkdir()
    mid = "m1"
    meeting_dir = storage_dir / mid
    meeting_dir.mkdir()
    journal = meeting_dir / "journal.jsonl"
    _write_journal(
        journal,
        [
            _asr_final("s1", "hello"),
            _translation("s1", "こんにちは(wip)", "en", "ja"),
            _translation("s1", "こんにちは", "en", "ja"),  # same bucket, later wins
            _translation("s1", "bonjour", "en", "fr"),
        ],
    )
    storage = MeetingStorage(ServerConfig(meetings_dir=storage_dir))
    lines = storage.read_journal_raw(mid)
    # 2 buckets: ("s1", "ja"), ("s1", "fr"). The ASR-only bucket is
    # dropped because translation entries already carry the source text.
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    targets = {
        (p.get("translation") or {}).get("target_language", "") for p in parsed
    }
    assert targets == {"ja", "fr"}
    # Within the ja bucket, the later "こんにちは" wins over the wip version.
    ja_line = next(
        p for p in parsed if (p.get("translation") or {}).get("target_language") == "ja"
    )
    assert ja_line["translation"]["text"] == "こんにちは"
