"""Translation-preserving dedup contract — ``prefer_event`` + the
in-place dedup sites that the post-finalize journal rewrite depends on.

Background: ``models.TranscriptEvent.with_translation`` reuses the source
event's revision number when stamping a translation, so the same
segment_id can hold two final journal lines at the same rev — one
without translation, one with. A naive ``new.rev > existing.rev`` dedup
silently drops whichever rev-tie line was written second. The
2026-05-14 audit of meeting ``7ae7a0f2`` found 13 of 107 segments lost
this way.

These tests pin the helper's contract (highest rev wins as base,
translation carries forward) AND assert that
``_generate_speaker_data`` round-trips a translation that landed under
a rev-tie.
"""

from __future__ import annotations

import json

from meeting_scribe.server_support.meeting_artifacts import (
    _generate_speaker_data,
    prefer_event,
)


def _ev(
    sid: str,
    rev: int,
    *,
    text: str = "hi",
    translation: str | None = None,
    is_final: bool = True,
    cluster_id: int = 1,
    start_ms: int = 0,
    end_ms: int = 1000,
) -> dict:
    """Compact factory for the journal event shape used by every dedup site."""
    out: dict = {
        "segment_id": sid,
        "revision": rev,
        "is_final": is_final,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "language": "en",
        "text": text,
        "speakers": [{"cluster_id": cluster_id, "display_name": None, "identity": None}],
        "translation": None,
    }
    if translation is not None:
        out["translation"] = {
            "status": "done",
            "text": translation,
            "target_language": "ja",
        }
    return out


class TestPreferEvent:
    """The pure helper that drives every dedup site."""

    def test_translation_wins_on_rev_tie(self):
        ex = _ev("a", rev=2, translation=None)
        new = _ev("a", rev=2, translation="こんにちは")
        result = prefer_event(ex, new)
        assert (result.get("translation") or {}).get("text") == "こんにちは"

    def test_higher_rev_no_translation_carries_existing_translation_forward(self):
        # ASR refinement landed AFTER the translation under a lower rev.
        ex = _ev("a", rev=2, translation="こんにちは", text="hi")
        new = _ev("a", rev=3, translation=None, text="hi there")
        result = prefer_event(ex, new)
        assert result["text"] == "hi there"
        assert result["revision"] == 3
        assert (result.get("translation") or {}).get("text") == "こんにちは"

    def test_existing_higher_rev_keeps_existing_translation(self):
        ex = _ev("a", rev=3, translation="こんにちは", text="hi there")
        new = _ev("a", rev=2, translation=None, text="hi")
        result = prefer_event(ex, new)
        assert result["text"] == "hi there"
        assert result["revision"] == 3
        assert (result.get("translation") or {}).get("text") == "こんにちは"

    def test_lower_rev_no_translation_loses(self):
        ex = _ev("a", rev=3, translation=None)
        new = _ev("a", rev=2, translation=None)
        result = prefer_event(ex, new)
        assert result["revision"] == 3

    def test_both_translated_higher_rev_wins(self):
        ex = _ev("a", rev=3, translation="A")
        new = _ev("a", rev=2, translation="B")
        result = prefer_event(ex, new)
        assert result["revision"] == 3
        assert (result.get("translation") or {}).get("text") == "A"

    def test_rev_tie_both_translated_takes_later_write(self):
        # Append-only journals: a second translation row at the same rev
        # always reflects more-recent state than the first.
        ex = _ev("a", rev=2, translation="first")
        new = _ev("a", rev=2, translation="second")
        result = prefer_event(ex, new)
        assert (result.get("translation") or {}).get("text") == "second"

    def test_lower_rev_with_translation_overrides_higher_rev_without(self):
        # The translation should land regardless of revision asymmetry,
        # because losing a translation is worse than reverting to a
        # slightly older ASR refinement of the same text. (In practice
        # the higher-rev branch above carries the translation forward,
        # but when the rev-tie path is taken we still must not drop it.)
        ex = _ev("a", rev=4, translation=None, text="hi there friend")
        new = _ev("a", rev=2, translation="やあ友達", text="hi friend")
        result = prefer_event(ex, new)
        # Higher rev wins as base, translation carried forward.
        assert result["revision"] == 4
        assert result["text"] == "hi there friend"
        assert (result.get("translation") or {}).get("text") == "やあ友達"


class TestGenerateSpeakerDataPreservesTranslations:
    """End-to-end: a journal with ASR-final + same-rev translation rows
    must come out with the translation intact after
    ``_generate_speaker_data`` rewrites it."""

    def _write_journal(self, meeting_dir, events: list[dict]) -> None:
        path = meeting_dir / "journal.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_translation_survives_rev_tie_rewrite(self, tmp_path):
        meeting_dir = tmp_path / "m1"
        meeting_dir.mkdir()
        # Two segments, each written twice at rev=2:
        #   line 1: ASR-final, no translation
        #   line 2: with translation
        # Pre-fix, the rev>rev dedup dropped line 2 silently.
        events = [
            _ev("seg-1", rev=2, text="Hello.", translation=None),
            _ev("seg-1", rev=2, text="Hello.", translation="こんにちは。"),
            _ev(
                "seg-2",
                rev=2,
                text="World.",
                translation=None,
                cluster_id=2,
                start_ms=1000,
                end_ms=2000,
            ),
            _ev(
                "seg-2",
                rev=2,
                text="World.",
                translation="世界。",
                cluster_id=2,
                start_ms=1000,
                end_ms=2000,
            ),
        ]
        self._write_journal(meeting_dir, events)

        _generate_speaker_data(meeting_dir, meeting_dir / "journal.jsonl", json)

        rewritten = [
            json.loads(line)
            for line in (meeting_dir / "journal.jsonl").read_text().splitlines()
            if line.strip()
        ]
        translations = {
            e["segment_id"]: (e.get("translation") or {}).get("text")
            for e in rewritten
            if e.get("segment_id")
        }
        assert translations["seg-1"] == "こんにちは。", (
            "Expected translation to survive rev-tie rewrite"
        )
        assert translations["seg-2"] == "世界。"

    def test_higher_rev_no_translation_does_not_clobber_lower_rev_translation(self, tmp_path):
        meeting_dir = tmp_path / "m2"
        meeting_dir.mkdir()
        # Speaker correction at rev=3 (no translation) lands AFTER a
        # rev=2 translation. Rewrite must keep both the higher-rev text
        # AND the translation.
        events = [
            _ev("seg-1", rev=2, text="Hello.", translation="こんにちは。"),
            _ev("seg-1", rev=3, text="Hello.", translation=None),
        ]
        self._write_journal(meeting_dir, events)

        _generate_speaker_data(meeting_dir, meeting_dir / "journal.jsonl", json)

        rewritten = [
            json.loads(line)
            for line in (meeting_dir / "journal.jsonl").read_text().splitlines()
            if line.strip()
        ]
        seg1 = next(e for e in rewritten if e.get("segment_id") == "seg-1")
        assert (seg1.get("translation") or {}).get("text") == "こんにちは。"


class TestSpeakerStatsConsistentWithDetectedSpeakers:
    """``summary.json`` ``speaker_stats`` must match the segment counts
    that ``detected_speakers.json`` records for the same meeting.

    The bug: ``_parallel_summary_task`` captures the journal BEFORE
    ``_generate_speaker_data`` rewrites cluster_ids to seq_index, so the
    summary's stats reference pre-rewrite IDs and end up inverted vs
    the speakers list the rest of the UI joins on. Fix: recompute
    ``speaker_stats`` against the rewritten journal after finalize.
    This test pins the invariant.
    """

    def test_recomputed_stats_match_detected_speakers_counts(self, tmp_path):
        from meeting_scribe.summary import (
            _calculate_speaker_stats,
            build_transcript_text,
        )

        meeting_dir = tmp_path / "m3"
        meeting_dir.mkdir()
        # Two raw clusters with imbalanced segment counts. cluster 7 has
        # 3 segments, cluster 4 has 1. ``_generate_speaker_data`` assigns
        # seq_index by segment_count descending, so cluster 7 → seq 1,
        # cluster 4 → seq 2. Post-rewrite journal carries seq_index in
        # the cluster_id slot.
        events = [
            _ev("a", rev=1, text="seg a", cluster_id=7, start_ms=0, end_ms=1000),
            _ev("b", rev=1, text="seg b", cluster_id=4, start_ms=1000, end_ms=2000),
            _ev("c", rev=1, text="seg c", cluster_id=7, start_ms=2000, end_ms=3000),
            _ev("d", rev=1, text="seg d", cluster_id=7, start_ms=3000, end_ms=4000),
        ]
        path = meeting_dir / "journal.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        _generate_speaker_data(meeting_dir, path, json)

        # Detected speakers list — written by _generate_speaker_data.
        speakers = json.loads((meeting_dir / "detected_speakers.json").read_text())
        # Re-read journal + recompute stats the same way the finalize
        # path's post-rewrite recompute does.
        events_after, _ = build_transcript_text(meeting_dir)
        stats = _calculate_speaker_stats(events_after, speakers)

        # Each speaker's stats segment_count must equal the
        # detected_speakers segment_count under the same display_name.
        by_name_stats = {s["name"]: s["segments"] for s in stats}
        by_name_detected = {s["display_name"]: s["segment_count"] for s in speakers}
        assert set(by_name_stats) == set(by_name_detected)
        for name, segs in by_name_stats.items():
            assert segs == by_name_detected[name], (
                f"Speaker {name}: summary stats say {segs}, "
                f"detected_speakers says {by_name_detected[name]}"
            )
