"""Unit tests for benchmarks/validate_meeting.py.

Uses synthetic meeting directories to confirm each check PASSES on a
well-formed meeting and FAILS on meetings that exhibit the specific
failure modes we've been hunting (orphan lanes, non-contiguous seqs,
empty speakers, journal/detected mismatch, drift, missing coverage,
lost-name preservation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from validate_meeting import validate_meeting


def _make_meeting(tmp_path: Path, *, audio_ms: int, detected: list, lanes: dict,
                  journal: list, timeline: dict, backup: list | None = None) -> Path:
    md = tmp_path / "m-test"
    md.mkdir()
    (md / "audio").mkdir()
    # Write PCM of the right length (zero-filled is fine — we just check byte size)
    pcm = md / "audio" / "recording.pcm"
    pcm.write_bytes(b"\x00\x00" * int(audio_ms * 16))  # 16 kHz × 2 bytes × ms/1000
    (md / "meta.json").write_text(json.dumps({"meeting_id": "m-test", "state": "complete"}))
    (md / "detected_speakers.json").write_text(json.dumps(detected))
    (md / "speaker_lanes.json").write_text(json.dumps(lanes))
    (md / "timeline.json").write_text(json.dumps(timeline))
    with (md / "journal.jsonl").open("w") as f:
        for e in journal:
            f.write(json.dumps(e) + "\n")
    if backup is not None:
        with (md / "journal.jsonl.bak").open("w") as f:
            for e in backup:
                f.write(json.dumps(e) + "\n")
    return md


def _final(sid: str, start: int, end: int, text: str, cid: int | None, identity: str | None = None):
    sp = []
    if cid is not None:
        sp.append({"cluster_id": cid, "identity": identity, "display_name": identity})
    return {
        "segment_id": sid, "revision": 1, "is_final": True,
        "start_ms": start, "end_ms": end, "text": text, "speakers": sp,
    }


def _speaker(cluster_id: int, seq_index: int, name: str, count: int = 1):
    return {
        "cluster_id": cluster_id, "seq_index": seq_index, "display_name": name,
        "segment_count": count, "total_speaking_ms": count * 1000,
        "first_seen_ms": 0, "last_seen_ms": count * 1000,
    }


class TestHealthyMeeting:
    def test_all_checks_pass(self, tmp_path):
        md = _make_meeting(
            tmp_path,
            audio_ms=120_000,  # 2 min
            detected=[
                _speaker(1, 1, "Mark", count=3),
                _speaker(2, 2, "Brad", count=2),
            ],
            lanes={
                "1": [{"start_ms": 0, "end_ms": 1000, "segment_id": "s1"}],
                "2": [{"start_ms": 1000, "end_ms": 2000, "segment_id": "s2"}],
            },
            timeline={
                "duration_ms": 120_000,
                "segments": [
                    {"segment_id": f"s{i}", "start_ms": i * 30_000, "end_ms": i * 30_000 + 4000,
                     "text": f"seg {i}", "language": "en", "speaker_id": 1}
                    for i in range(4)  # one per 30s bucket
                ],
            },
            journal=[
                _final("s1", 0, 1000, "a", 1),
                _final("s2", 30_000, 31_000, "b", 2),
                _final("s3", 60_000, 61_000, "c", 1),
                _final("s4", 90_000, 91_000, "d", 2),
            ],
        )
        result = validate_meeting(md)
        assert result.all_passed, [
            f"{c.name}={c.passed} ({c.detail})" for c in result.checks if not c.passed
        ]


class TestAlignmentCheck:
    def test_timeline_shorter_than_audio_fails(self, tmp_path):
        md = _make_meeting(
            tmp_path, audio_ms=120_000,
            detected=[_speaker(1, 1, "A")],
            lanes={"1": []},
            timeline={"duration_ms": 60_000, "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 60_000, "text": "x", "language": "en"},
            ]},
            journal=[_final("s1", 0, 60_000, "x", 1)],
        )
        result = validate_meeting(md)
        align = next(c for c in result.checks if c.name == "alignment")
        assert not align.passed
        assert align.metrics["duration_drift_ms"] == -60_000


class TestSeqDensity:
    def test_non_contiguous_seq_fails(self, tmp_path):
        # Simulates the old bug: cluster_id=0 ate seq_index=1, leaving 2,3,4
        md = _make_meeting(
            tmp_path, audio_ms=60_000,
            detected=[
                _speaker(1, 2, "A"),
                _speaker(2, 3, "B"),
                _speaker(3, 4, "C"),
            ],
            lanes={"2": [], "3": [], "4": []},
            timeline={"duration_ms": 60_000, "segments": [
                {"segment_id": f"s{i}", "start_ms": i * 30_000, "end_ms": i * 30_000 + 1000,
                 "text": "x", "language": "en"} for i in range(2)
            ]},
            journal=[
                _final("s1", 0, 1000, "a", 2),
                _final("s2", 30_000, 31_000, "b", 3),
            ],
        )
        result = validate_meeting(md)
        seq = next(c for c in result.checks if c.name == "seq_density")
        assert not seq.passed
        assert seq.metrics["seq_indices"] == [2, 3, 4]


class TestOrphanLane:
    def test_zero_lane_fails(self, tmp_path):
        md = _make_meeting(
            tmp_path, audio_ms=60_000,
            detected=[_speaker(1, 1, "A")],
            lanes={"0": [{"start_ms": 0, "end_ms": 1000, "segment_id": "s0"}], "1": []},
            timeline={"duration_ms": 60_000, "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 1000, "text": "x", "language": "en"},
                {"segment_id": "s2", "start_ms": 30_000, "end_ms": 31_000, "text": "y", "language": "en"},
            ]},
            journal=[_final("s1", 0, 1000, "a", 1)],
        )
        result = validate_meeting(md)
        lane = next(c for c in result.checks if c.name == "no_orphan_lane")
        assert not lane.passed


class TestJournalDetectedAgreement:
    def test_journal_cluster_not_in_detected_fails(self, tmp_path):
        md = _make_meeting(
            tmp_path, audio_ms=60_000,
            detected=[_speaker(1, 1, "A")],
            lanes={"1": []},
            timeline={"duration_ms": 60_000, "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 1000, "text": "x", "language": "en"},
                {"segment_id": "s2", "start_ms": 30_000, "end_ms": 31_000, "text": "y", "language": "en"},
            ]},
            journal=[
                _final("s1", 0, 1000, "a", 1),
                _final("s99", 30_000, 31_000, "dangling", 99),  # cluster 99 isn't in detected
            ],
        )
        result = validate_meeting(md)
        agree = next(c for c in result.checks if c.name == "journal_detected_agreement")
        assert not agree.passed
        assert 99 in agree.metrics["unknown_cluster_ids"]


class TestEmptySpeakers:
    def test_empty_speakers_fails(self, tmp_path):
        md = _make_meeting(
            tmp_path, audio_ms=60_000,
            detected=[_speaker(1, 1, "A")],
            lanes={"1": []},
            timeline={"duration_ms": 60_000, "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 1000, "text": "x", "language": "en"},
                {"segment_id": "s2", "start_ms": 30_000, "end_ms": 31_000, "text": "y", "language": "en"},
            ]},
            journal=[
                _final("s1", 0, 1000, "a", 1),
                _final("s2", 30_000, 31_000, "unattributed", None),  # empty speakers
            ],
        )
        result = validate_meeting(md)
        empty = next(c for c in result.checks if c.name == "no_empty_speakers")
        assert not empty.passed
        assert empty.metrics["empty_count"] == 1


class TestCoverage:
    def test_gap_in_middle_fails(self, tmp_path):
        # 2-minute meeting, segments only in the first 30s → last 3 buckets empty
        md = _make_meeting(
            tmp_path, audio_ms=120_000,
            detected=[_speaker(1, 1, "A")],
            lanes={"1": []},
            timeline={"duration_ms": 120_000, "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 1000, "text": "x", "language": "en"},
            ]},
            journal=[_final("s1", 0, 1000, "a", 1)],
        )
        result = validate_meeting(md)
        cov = next(c for c in result.checks if c.name == "coverage")
        assert not cov.passed
        # 120s / 30s = 4 buckets, only bucket 0 populated → 3 empty
        assert cov.metrics["empty_bucket_count"] >= 2


class TestNamePreservation:
    def test_missing_names_fails(self, tmp_path):
        # Backup has 3 named speakers, new detected only has 1 — preservation failed
        bak = [
            {"type": "speaker_correction", "segment_id": "o1", "speaker_name": "Joel"},
            {"type": "speaker_correction", "segment_id": "o2", "speaker_name": "Nikul"},
            {"type": "speaker_correction", "segment_id": "o3", "speaker_name": "Mark"},
            _final("o1", 0, 1000, "a", 5),
            _final("o2", 10_000, 11_000, "b", 6),
            _final("o3", 20_000, 21_000, "c", 7),
        ]
        md = _make_meeting(
            tmp_path, audio_ms=60_000,
            detected=[_speaker(1, 1, "Mark")],
            lanes={"1": []},
            timeline={"duration_ms": 60_000, "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 1000, "text": "x", "language": "en"},
                {"segment_id": "s2", "start_ms": 30_000, "end_ms": 31_000, "text": "y", "language": "en"},
            ]},
            journal=[_final("s1", 0, 1000, "a", 1)],
            backup=bak,
        )
        result = validate_meeting(md)
        name = next(c for c in result.checks if c.name == "name_preservation")
        assert not name.passed
        # Joel + Nikul missing, Mark preserved → 33% coverage
        assert set(name.metrics["missing_names"]) == {"Joel", "Nikul"}

    def test_skipped_without_backup(self, tmp_path):
        md = _make_meeting(
            tmp_path, audio_ms=60_000,
            detected=[_speaker(1, 1, "A")],
            lanes={"1": []},
            timeline={"duration_ms": 60_000, "segments": [
                {"segment_id": "s1", "start_ms": 0, "end_ms": 1000, "text": "x", "language": "en"},
                {"segment_id": "s2", "start_ms": 30_000, "end_ms": 31_000, "text": "y", "language": "en"},
            ]},
            journal=[_final("s1", 0, 1000, "a", 1)],
        )
        result = validate_meeting(md)
        name = next(c for c in result.checks if c.name == "name_preservation")
        assert name.passed
        assert name.metrics.get("skipped") is True
