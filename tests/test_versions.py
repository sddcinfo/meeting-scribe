"""Tests for the reprocess-version snapshot + diff tooling in
``meeting_scribe.versions``.

The module is mostly pure functions over a meetings/{id}/versions/
on-disk layout, so most of these tests construct a tmp meeting dir,
seed it with the artifacts the production code writes, and verify
the outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from meeting_scribe import versions

# ──────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────


class TestSlugify:
    def test_simple_label(self):
        assert versions._slugify("post-dutch-fix") == "post-dutch-fix"

    def test_strips_unsafe_chars(self):
        assert versions._slugify("hello world / 123!") == "hello-world-123"

    def test_collapses_runs_of_dashes(self):
        # Underscores are kept; only runs of dashes collapse.
        assert versions._slugify("a---b___c") == "a-b___c"

    def test_strips_leading_trailing_dashes(self):
        assert versions._slugify("---trim me---") == "trim-me"

    def test_truncates_at_48_chars(self):
        long = "x" * 80
        assert len(versions._slugify(long)) == 48

    def test_empty_falls_back_to_v(self):
        assert versions._slugify("") == "v"
        assert versions._slugify("---") == "v"


class TestVerdict:
    def test_within_threshold_is_same(self):
        assert versions._verdict(0.04, higher_better=True) == "same"
        assert versions._verdict(-0.04, higher_better=True) == "same"

    def test_higher_better_positive_delta_is_better(self):
        assert versions._verdict(0.10, higher_better=True) == "better"

    def test_higher_better_negative_delta_is_worse(self):
        assert versions._verdict(-0.10, higher_better=True) == "worse"

    def test_lower_better_negative_delta_is_better(self):
        # speakers.count uses higher_better=False — fewer is better
        assert versions._verdict(-0.20, higher_better=False) == "better"

    def test_lower_better_positive_delta_is_worse(self):
        assert versions._verdict(0.20, higher_better=False) == "worse"

    def test_custom_threshold(self):
        assert versions._verdict(0.06, higher_better=True, threshold=0.10) == "same"


class TestRel:
    def test_both_zero(self):
        assert versions._rel(0, 0) == 0.0

    def test_zero_baseline_positive_compare(self):
        assert versions._rel(0, 5) == 1.0

    def test_zero_baseline_negative_compare(self):
        assert versions._rel(0, -5) == -1.0

    def test_increase(self):
        assert versions._rel(10.0, 12.0) == 0.2

    def test_decrease(self):
        assert versions._rel(10.0, 8.0) == -0.2


# ──────────────────────────────────────────────────────────────────
# Snapshot + list
# ──────────────────────────────────────────────────────────────────


def _seed_meeting(tmp: Path) -> Path:
    """Build a minimal meeting dir with the files snapshot_meeting copies."""
    m = tmp / "meeting-001"
    m.mkdir()
    (m / "journal.jsonl").write_text(
        '{"type":"transcript","is_final":true,"segment_id":"s1","revision":1,'
        '"text":"hello","language":"en","start_ms":0,"end_ms":1000,'
        '"speakers":[{"cluster_id":1}],'
        '"translation":{"text":"こんにちは"}}\n'
    )
    (m / "summary.json").write_text(
        json.dumps(
            {
                "executive_summary": "short",
                "key_insights": ["a", "b"],
                "action_items": ["x"],
                "topics": [],
                "decisions": [],
                "key_quotes": [],
                "schema_version": 3,
            }
        )
    )
    (m / "timeline.json").write_text(
        json.dumps(
            {
                "duration_ms": 1000,
                "segments": [
                    {"speaker_id": "a", "start_ms": 0, "end_ms": 1000},
                    {"speaker_id": "b", "start_ms": 1000, "end_ms": 2000},
                ],
            }
        )
    )
    (m / "detected_speakers.json").write_text(
        json.dumps([{"display_name": "Alice", "segment_count": 5}])
    )
    return m


class TestSnapshotMeeting:
    def test_returns_none_without_journal(self, tmp_path):
        m = tmp_path / "empty"
        m.mkdir()
        assert versions.snapshot_meeting(m, "label") is None

    def test_copies_only_existing_snapshot_files(self, tmp_path):
        m = _seed_meeting(tmp_path)
        snap = versions.snapshot_meeting(m, "post-fix")
        assert snap is not None
        assert snap.parent == m / "versions"
        assert snap.name.endswith("__post-fix")
        # All seeded files copied
        for name in ("journal.jsonl", "summary.json", "timeline.json", "detected_speakers.json"):
            assert (snap / name).exists()
        # Non-seeded snapshot files don't error
        assert not (snap / "speaker_lanes.json").exists()

    def test_writes_manifest(self, tmp_path):
        m = _seed_meeting(tmp_path)
        snap = versions.snapshot_meeting(m, "post-fix", inputs={"asr_url": "http://x"})
        assert snap is not None
        manifest = json.loads((snap / "manifest.json").read_text())
        assert manifest["label"] == "post-fix"
        assert manifest["source"] == "auto:reprocess"
        assert manifest["inputs"] == {"asr_url": "http://x"}
        assert "journal.jsonl" in manifest["files"]

    def test_no_label_no_suffix(self, tmp_path):
        m = _seed_meeting(tmp_path)
        snap = versions.snapshot_meeting(m, None)
        assert snap is not None
        assert "__" not in snap.name


class TestListVersions:
    def test_empty_when_no_versions_dir(self, tmp_path):
        m = tmp_path / "m"
        m.mkdir()
        assert versions.list_versions(m) == []

    def test_returns_newest_first(self, tmp_path):
        m = _seed_meeting(tmp_path)
        # Two snapshots — names are timestamped, second sorts after
        snap1 = versions.snapshot_meeting(m, "first")
        # bump the snap1 name so we can guarantee sort order independent of clock
        renamed = snap1.parent / ("0001-01-01T00-00-00__first")
        snap1.rename(renamed)
        snap2 = versions.snapshot_meeting(m, "second")
        listing = versions.list_versions(m)
        assert len(listing) == 2
        assert listing[0]["name"] == snap2.name  # newest first
        assert listing[1]["name"] == "0001-01-01T00-00-00__first"

    def test_corrupt_manifest_does_not_crash(self, tmp_path):
        m = _seed_meeting(tmp_path)
        snap = versions.snapshot_meeting(m, "x")
        (snap / "manifest.json").write_text("{not json")
        listing = versions.list_versions(m)
        assert listing[0]["manifest"]["_parse_error"] is True


# ──────────────────────────────────────────────────────────────────
# Per-artifact summarizers
# ──────────────────────────────────────────────────────────────────


class TestSummarizeJournal:
    def test_missing_returns_present_false(self, tmp_path):
        out = versions._summarize_journal(tmp_path / "missing.jsonl")
        assert out == {"present": False}

    def test_basic_counts(self, tmp_path):
        p = tmp_path / "j.jsonl"
        p.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "transcript",
                            "is_final": True,
                            "segment_id": "s1",
                            "revision": 1,
                            "text": "hi",
                            "language": "en",
                            "start_ms": 0,
                            "end_ms": 500,
                            "speakers": [{"cluster_id": 1}],
                            "translation": {"text": "hola"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "transcript",
                            "is_final": True,
                            "segment_id": "s2",
                            "revision": 1,
                            "text": "yo",
                            "language": "en",
                            "start_ms": 500,
                            "end_ms": 800,
                            "speakers": [{"cluster_id": 2}],
                            "translation": {"text": "hola"},
                        }
                    ),
                    "",  # blank line
                    "{not json",  # malformed line — skipped
                    json.dumps(
                        {  # speaker_correction is filtered
                            "type": "speaker_correction",
                            "segment_id": "s3",
                            "is_final": True,
                        }
                    ),
                ]
            )
        )
        out = versions._summarize_journal(p)
        assert out["present"] is True
        assert out["segment_count"] == 2
        assert out["language_counts"] == {"en": 2}
        assert out["translated_segments"] == 2
        assert out["translation_coverage"] == 1.0
        assert out["total_text_chars"] == 4  # "hi" + "yo"
        assert out["unique_clusters_in_journal"] == 2
        assert out["total_speech_ms"] == 800  # 500 + 300

    def test_higher_revision_overrides_same_segment(self, tmp_path):
        p = tmp_path / "j.jsonl"
        p.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "transcript",
                            "is_final": True,
                            "segment_id": "s1",
                            "revision": 1,
                            "text": "draft",
                            "language": "en",
                            "start_ms": 0,
                            "end_ms": 1000,
                        }
                    ),
                    json.dumps(
                        {
                            "type": "transcript",
                            "is_final": True,
                            "segment_id": "s1",
                            "revision": 5,
                            "text": "final-text",
                            "language": "en",
                            "start_ms": 0,
                            "end_ms": 1000,
                        }
                    ),
                ]
            )
        )
        out = versions._summarize_journal(p)
        assert out["segment_count"] == 1
        assert out["total_text_chars"] == len("final-text")


class TestSummarizeSpeakers:
    def test_missing(self, tmp_path):
        assert versions._summarize_speakers(tmp_path / "x.json") == {"present": False}

    def test_basic_list(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(
            json.dumps(
                [
                    {"display_name": "A", "segment_count": 3},
                    {"display_name": "B", "segment_count": 7},
                ]
            )
        )
        out = versions._summarize_speakers(p)
        assert out["count"] == 2
        assert out["labels"] == ["A", "B"]
        assert out["segment_counts_per_speaker"] == [3, 7]

    def test_invalid_shape(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"not": "a list"}))
        out = versions._summarize_speakers(p)
        assert out == {"present": True, "count": 0}

    def test_parse_error(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text("garbage")
        assert versions._summarize_speakers(p)["_parse_error"] is True


class TestSummarizeSummaryDoc:
    def test_missing(self, tmp_path):
        assert versions._summarize_summary_doc(tmp_path / "x") == {"present": False}

    def test_full_doc(self, tmp_path):
        p = tmp_path / "summary.json"
        p.write_text(
            json.dumps(
                {
                    "executive_summary": "abc",
                    "key_insights": [1, 2, 3],
                    "action_items": [{"text": "x"}],
                    "topics": [],
                    "decisions": [{}, {}],
                    "key_quotes": [],
                    "schema_version": 3,
                }
            )
        )
        out = versions._summarize_summary_doc(p)
        assert out["executive_summary_chars"] == 3
        assert out["key_insights_count"] == 3
        assert out["action_items_count"] == 1
        assert out["decisions_count"] == 2
        assert out["schema_version"] == 3

    def test_invalid_shape(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps([1, 2, 3]))
        out = versions._summarize_summary_doc(p)
        assert out == {"present": True, "_invalid_shape": True}


class TestSummarizeTimeline:
    def test_missing(self, tmp_path):
        assert versions._summarize_timeline(tmp_path / "x") == {"present": False}

    def test_unique_speakers_and_count(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(
            json.dumps(
                {
                    "duration_ms": 5000,
                    "segments": [
                        {"speaker_id": "alice"},
                        {"speaker_id": "bob"},
                        {"speaker_id": "alice"},
                        {},  # missing speaker_id — filtered
                    ],
                }
            )
        )
        out = versions._summarize_timeline(p)
        assert out["duration_ms"] == 5000
        assert out["segment_count"] == 4
        assert out["unique_speaker_ids"] == ["alice", "bob"]


# ──────────────────────────────────────────────────────────────────
# Composite metrics + diff
# ──────────────────────────────────────────────────────────────────


class TestMetricsFor:
    def test_metrics_for_current_pulls_top_level_files(self, tmp_path):
        m = _seed_meeting(tmp_path)
        out = versions.metrics_for_current(m)
        assert out["journal"]["segment_count"] == 1
        assert out["speakers"]["count"] == 1
        assert out["summary"]["key_insights_count"] == 2
        assert out["timeline"]["duration_ms"] == 1000

    def test_metrics_for_version_reads_snapshot(self, tmp_path):
        m = _seed_meeting(tmp_path)
        snap = versions.snapshot_meeting(m, "x")
        out = versions.metrics_for_version(m, snap.name)
        assert out["journal"]["segment_count"] == 1


class TestDiffVersions:
    def _metrics(self, segments=10, chars=100, speakers=2):
        return {
            "journal": {
                "segment_count": segments,
                "total_text_chars": chars,
                "translation_coverage": 1.0,
                "total_speech_ms": 1000,
                "language_counts": {"en": segments},
            },
            "speakers": {"count": speakers},
            "summary": {
                "key_insights_count": 5,
                "action_items_count": 2,
                "executive_summary_chars": 200,
            },
        }

    def test_no_change_marks_all_same(self):
        out = versions.diff_versions(self._metrics(), self._metrics())
        assert out["totals"]["same"] >= 6  # most dimensions
        assert out["totals"]["better"] == 0
        assert out["totals"]["worse"] == 0

    def test_more_text_chars_is_better(self):
        out = versions.diff_versions(self._metrics(chars=100), self._metrics(chars=200))
        assert out["dimensions"]["transcript.total_text_chars"]["verdict"] == "better"

    def test_more_speakers_is_worse(self):
        # speakers.count has higher_better=False
        out = versions.diff_versions(self._metrics(speakers=2), self._metrics(speakers=10))
        assert out["dimensions"]["speakers.count"]["verdict"] == "worse"

    def test_emits_language_distribution_info(self):
        a = self._metrics(segments=10)
        b = self._metrics(segments=20)
        out = versions.diff_versions(a, b)
        assert out["language_distribution"]["baseline"] == {"en": 10}
        assert out["language_distribution"]["compare"] == {"en": 20}
