"""Regression tests for `_generate_speaker_data` seq_index allocation.

Bug fixed 2026-04-13: orphan events with cluster_id=0 (diarize couldn't
attribute) were being added to `speaker_stats` and claiming seq_index=1,
then filtered out at the end — leaving real speakers as Speaker 2/3/4
with no Speaker 1 in the participants list. Triggered for any meeting
where the opening minute had no diarized voice (e.g. host opening
remarks before the diarize buffer was full).

Also covers: orphan reassignment falling back to the nearest real
speaker with NO time-window cap when the 5s window can't reach one.
"""

from __future__ import annotations

import json
from pathlib import Path


def _make_meeting(tmp_path, journal_lines: list[dict]) -> Path:
    md = tmp_path / "test-meeting"
    md.mkdir()
    j = md / "journal.jsonl"
    j.write_text("\n".join(json.dumps(e) for e in journal_lines) + "\n")
    return md


def _final(seg_id: str, start_ms: int, end_ms: int, text: str, cluster_id: int):
    return {
        "segment_id": seg_id,
        "revision": 1,
        "is_final": True,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "language": "ja",
        "text": text,
        "speakers": [{"cluster_id": cluster_id}] if cluster_id is not None else [],
    }


def _run(meeting_dir: Path) -> tuple[list[dict], dict]:
    """Invoke _generate_speaker_data and return (detected_speakers, lanes)."""
    from meeting_scribe.server_support.meeting_artifacts import _generate_speaker_data

    journal_path = meeting_dir / "journal.jsonl"
    _generate_speaker_data(meeting_dir, journal_path, json)

    detected = json.loads((meeting_dir / "detected_speakers.json").read_text())
    lanes = json.loads((meeting_dir / "speaker_lanes.json").read_text())
    return detected, lanes


class TestSeqIndexAllocation:
    def test_orphans_do_not_eat_speaker_1(self, tmp_path):
        """Old bug: cluster_id=0 segments at meeting start would claim
        seq_index=1, then get filtered out, leaving Speaker 2/3/4."""
        md = _make_meeting(
            tmp_path,
            [
                # 30 s of orphans at the start
                _final("s1", 0, 1500, "a", 0),
                _final("s2", 1500, 3000, "b", 0),
                _final("s3", 3000, 4500, "c", 0),
                # Then real speakers — but each more than 5 s after the orphans
                _final("s4", 60_000, 61_500, "d", 7),
                _final("s5", 61_500, 63_000, "e", 7),
                _final("s6", 90_000, 91_500, "f", 9),
            ],
        )
        detected, _lanes = _run(md)
        seq_indices = sorted(s["seq_index"] for s in detected)
        # Critical: Speaker 1 must exist
        assert 1 in seq_indices
        # Real speakers get seq 1 and 2 (no gaps)
        assert seq_indices == [1, 2]

    def test_orphan_reassignment_handles_distant_speakers(self, tmp_path):
        """If the only real speaker is more than 5 s away, the orphan
        still gets attributed to them rather than left as cluster=0."""
        md = _make_meeting(
            tmp_path,
            [
                _final("orphan", 0, 1500, "intro", 0),
                _final("real", 60_000, 61_500, "actual speech", 7),
            ],
        )
        detected, lanes = _run(md)
        # One real speaker → both segments belong to them
        assert len(detected) == 1
        assert detected[0]["seq_index"] == 1
        # The orphan was attributed: its segment_id appears in lane "1"
        all_segment_ids = {entry["segment_id"] for entry in lanes["1"]}
        assert "orphan" in all_segment_ids
        assert "real" in all_segment_ids

    def test_no_lane_for_cluster_zero(self, tmp_path):
        """speaker_lanes.json must never contain a "0" key."""
        md = _make_meeting(
            tmp_path,
            [
                _final("o1", 0, 1500, "x", 0),
                _final("r1", 5_000, 6_500, "y", 3),
            ],
        )
        _, lanes = _run(md)
        assert "0" not in lanes

    def test_seq_index_dense_no_gaps(self, tmp_path):
        """Three real speakers → seq_index 1, 2, 3 (not 2, 3, 4)."""
        md = _make_meeting(
            tmp_path,
            [
                _final("s1", 1000, 2000, "a", 5),
                _final("s2", 3000, 4000, "b", 7),
                _final("s3", 5000, 6000, "c", 9),
            ],
        )
        detected, _ = _run(md)
        assert sorted(s["seq_index"] for s in detected) == [1, 2, 3]

    def test_journal_rewrite_cluster_ids_match_detected(self, tmp_path):
        """The rewritten journal's cluster_ids must equal the seq_index
        the UI reads from detected_speakers.json. Without this match,
        clicking a speaker in the participant list shows zero segments."""
        md = _make_meeting(
            tmp_path,
            [
                _final("s1", 1000, 2000, "a", 7),
                _final("s2", 3000, 4000, "b", 9),
                _final("s3", 5000, 6000, "c", 7),
            ],
        )
        detected, _ = _run(md)
        cluster_to_seq = {s["cluster_id"]: s["seq_index"] for s in detected}
        # Original raw IDs: 7 and 9. Both must be in the map.
        assert 7 in cluster_to_seq
        assert 9 in cluster_to_seq
        # Read the rewritten journal — every event's cluster_id must be
        # a seq_index that exists in detected.
        valid_seqs = {s["seq_index"] for s in detected}
        with (md / "journal.jsonl").open() as f:
            for line in f:
                ev = json.loads(line)
                if ev.get("speakers"):
                    cid = ev["speakers"][0].get("cluster_id")
                    assert cid in valid_seqs, f"journal cluster_id {cid} not in detected"

    def test_cluster_display_name_uses_dominant_identity_vote(self, tmp_path):
        """Cluster display_name should reflect the MAJORITY identity on
        its events, not just the first time-ordered one. This is the
        f38d5807 case: pyannote clustered Joel + Danny together, and
        the first event in the cluster is Joel but the dominant is
        Danny. Participant list should show Danny."""
        md = _make_meeting(
            tmp_path,
            [
                # Inject speaker_corrections that set identities on specific events
                {"type": "speaker_correction", "segment_id": "s1", "speaker_name": "Joel"},
                {"type": "speaker_correction", "segment_id": "s2", "speaker_name": "Danny"},
                {"type": "speaker_correction", "segment_id": "s3", "speaker_name": "Danny"},
                {"type": "speaker_correction", "segment_id": "s4", "speaker_name": "Danny"},
                _final("s1", 1000, 2000, "joel line", 7),  # First event — Joel
                _final("s2", 3000, 4000, "danny line a", 7),
                _final("s3", 5000, 6000, "danny line b", 7),
                _final("s4", 7000, 8000, "danny line c", 7),
            ],
        )
        detected, _ = _run(md)
        # Only one real cluster → cluster shows DOMINANT name "Danny"
        # (4 Danny events > 1 Joel event), not "Joel" (first-encountered)
        assert len(detected) == 1
        assert detected[0]["display_name"] == "Danny"

        # Journal must still carry the per-event identity (Joel stays Joel)
        joel_preserved = False
        for line in (md / "journal.jsonl").open():
            ev = json.loads(line)
            if ev.get("type") == "speaker_correction":
                continue
            if ev.get("segment_id") == "s1":
                sp = ev.get("speakers") or []
                if sp and sp[0].get("identity") == "Joel":
                    joel_preserved = True
        assert joel_preserved, "per-event identity (Joel) must survive in journal"

    def test_journal_rewrite_carries_orphan_reassignment(self, tmp_path):
        """The rewritten journal must include the orphan-reassigned
        speakers (not just the remapped real ones). Otherwise the live
        UI shows the first 60-140 s of a meeting as 'no speaker'
        despite detected_speakers.json reporting 3 speakers — the
        original bug from meeting 0a96."""
        md = _make_meeting(
            tmp_path,
            [
                # First minute: orphans (cluster_id=0 — diarize hadn't caught up)
                _final("o1", 0, 4000, "intro 1", 0),
                _final("o2", 4000, 8000, "intro 2", 0),
                _final("o3", 60_000, 64_000, "still intro", 0),
                # Then real speakers far away
                _final("r1", 140_000, 144_000, "first real", 5),
                _final("r2", 200_000, 204_000, "another voice", 7),
            ],
        )
        _run(md)
        # Re-read the rewritten journal — every final must now have a
        # non-empty speakers list with a positive cluster_id.
        unattributed = []
        with (md / "journal.jsonl").open() as f:
            for line in f:
                ev = json.loads(line)
                if ev.get("type") == "speaker_correction":
                    continue
                sp = ev.get("speakers") or []
                if not sp or (sp[0].get("cluster_id") or 0) <= 0:
                    unattributed.append(ev["segment_id"])
        assert not unattributed, (
            f"orphan-reassigned events lost their speaker in journal rewrite: {unattributed}"
        )
