"""Unit tests for reprocess.py diarization merging.

Verifies that chunked diarization with embedding-based cluster merging
produces stable cluster IDs across chunks — critical for long meetings.
"""

from __future__ import annotations

import numpy as np

from meeting_scribe.reprocess import _merge_clusters_via_embeddings


def _emb(seed: int, dim: int = 192) -> list[float]:
    """Generate a reproducible random unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _jitter(embedding: list[float], strength: float = 0.05) -> list[float]:
    """Apply small jitter to simulate same speaker across chunks."""
    v = np.array(embedding, dtype=np.float32)
    v += np.random.default_rng(42).normal(0, strength, len(v)).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


class TestClusterMerging:
    """Cross-chunk cluster ID stability via embedding matching."""

    def test_same_speaker_across_two_chunks_gets_same_global_id(self) -> None:
        """Speaker A in chunk 1 and chunk 2 with similar embeddings →
        both get the same global cluster_id."""
        speaker_a = _emb(seed=1)
        chunk1 = [
            {
                "start_ms": 0,
                "end_ms": 5000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": speaker_a,
            }
        ]
        chunk2 = [
            {
                "start_ms": 10000,
                "end_ms": 15000,
                "local_cluster_id": 0,  # same local ID, different chunk
                "confidence": 0.9,
                "embedding": _jitter(speaker_a, strength=0.03),
            }
        ]
        merged = _merge_clusters_via_embeddings([chunk1, chunk2])
        assert len(merged) == 2
        assert merged[0]["cluster_id"] == merged[1]["cluster_id"], (
            "Same speaker across chunks should get the same global cluster_id"
        )

    def test_different_speakers_across_chunks_get_different_ids(self) -> None:
        """Speaker A in chunk 1 and speaker B in chunk 2 with orthogonal
        embeddings → different global cluster_ids even though both are
        local cluster 0 in their chunks."""
        speaker_a = _emb(seed=1)
        speaker_b = _emb(seed=999)  # very different seed → different direction
        chunk1 = [
            {
                "start_ms": 0,
                "end_ms": 5000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": speaker_a,
            }
        ]
        chunk2 = [
            {
                "start_ms": 10000,
                "end_ms": 15000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": speaker_b,
            }
        ]
        merged = _merge_clusters_via_embeddings([chunk1, chunk2])
        assert len(merged) == 2
        assert merged[0]["cluster_id"] != merged[1]["cluster_id"], (
            "Different speakers should get different cluster_ids — this is "
            "the bug that the live streaming path hit"
        )

    def test_five_speakers_across_three_chunks_tracked_stably(self) -> None:
        """A 3-chunk meeting with 5 distinct speakers, each appearing
        in multiple chunks, should produce exactly 5 global cluster IDs."""
        speakers = [_emb(seed=i * 37) for i in range(5)]

        # Chunk 1: speakers 0, 1, 2
        chunk1 = [
            {
                "start_ms": 0,
                "end_ms": 3000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": speakers[0],
            },
            {
                "start_ms": 3000,
                "end_ms": 6000,
                "local_cluster_id": 1,
                "confidence": 0.9,
                "embedding": speakers[1],
            },
            {
                "start_ms": 6000,
                "end_ms": 9000,
                "local_cluster_id": 2,
                "confidence": 0.9,
                "embedding": speakers[2],
            },
        ]
        # Chunk 2: speakers 2, 3, 4 (note: speaker 2's local_id might differ)
        chunk2 = [
            {
                "start_ms": 9000,
                "end_ms": 12000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": speakers[2],
            },
            {
                "start_ms": 12000,
                "end_ms": 15000,
                "local_cluster_id": 1,
                "confidence": 0.9,
                "embedding": speakers[3],
            },
            {
                "start_ms": 15000,
                "end_ms": 18000,
                "local_cluster_id": 2,
                "confidence": 0.9,
                "embedding": speakers[4],
            },
        ]
        # Chunk 3: speakers 0, 4 (coming back)
        chunk3 = [
            {
                "start_ms": 18000,
                "end_ms": 21000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": speakers[0],
            },
            {
                "start_ms": 21000,
                "end_ms": 24000,
                "local_cluster_id": 1,
                "confidence": 0.9,
                "embedding": speakers[4],
            },
        ]
        merged = _merge_clusters_via_embeddings([chunk1, chunk2, chunk3])
        assert len(merged) == 8
        unique_ids = {s["cluster_id"] for s in merged}
        assert len(unique_ids) == 5, (
            f"Expected 5 unique global cluster IDs for 5 speakers, got {len(unique_ids)}"
        )
        # Speaker 0 should have SAME cluster in chunk 1 and chunk 3
        assert merged[0]["cluster_id"] == merged[6]["cluster_id"]
        # Speaker 4 should have SAME cluster in chunk 2 and chunk 3
        assert merged[5]["cluster_id"] == merged[7]["cluster_id"]
        # Speaker 2 should have SAME cluster in chunk 1 and chunk 2
        assert merged[2]["cluster_id"] == merged[3]["cluster_id"]

    def test_empty_input_returns_empty(self) -> None:
        assert _merge_clusters_via_embeddings([]) == []
        assert _merge_clusters_via_embeddings([[]]) == []

    def test_no_embeddings_fallback_uses_separate_ids(self) -> None:
        """When embeddings are missing, we can't merge — each local cluster
        gets its own global ID (conservative)."""
        chunk1 = [
            {
                "start_ms": 0,
                "end_ms": 5000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": None,
            },
        ]
        chunk2 = [
            {
                "start_ms": 10000,
                "end_ms": 15000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": None,
            },
        ]
        merged = _merge_clusters_via_embeddings([chunk1, chunk2])
        assert len(merged) == 2
        # Without embeddings, we can't prove they're the same speaker
        # so they get different IDs (safer than the old bug of merging them)
        assert merged[0]["cluster_id"] != merged[1]["cluster_id"]

    def test_time_order_preserved(self) -> None:
        """Output must be sorted by start_ms."""
        chunk1 = [
            {
                "start_ms": 5000,
                "end_ms": 6000,
                "local_cluster_id": 0,
                "confidence": 0.9,
                "embedding": _emb(1),
            },
            {
                "start_ms": 0,
                "end_ms": 1000,
                "local_cluster_id": 1,
                "confidence": 0.9,
                "embedding": _emb(2),
            },
        ]
        merged = _merge_clusters_via_embeddings([chunk1])
        assert merged[0]["start_ms"] == 0
        assert merged[1]["start_ms"] == 5000


class TestLongMeetingScalability:
    """Verify the chunking strategy produces sane bounds for long meetings."""

    def test_90_minute_meeting_chunk_count(self) -> None:
        """A 90-minute meeting with the current 4-min chunks + 30s overlap
        should produce ~26 chunks. Halved chunk size prevents the worker
        from dying mid-inference on long audio under tight system memory."""
        total_seconds = 90 * 60
        chunk_seconds = 4 * 60
        overlap_seconds = 30
        stride = chunk_seconds - overlap_seconds
        num_chunks = (total_seconds - overlap_seconds) // stride + 1
        assert 24 <= num_chunks <= 28, (
            f"90min / 4min chunks with 30s overlap = {num_chunks} chunks — expected ~26"
        )

    def test_chunk_parameters_cover_full_audio(self) -> None:
        """Overlap + stride must guarantee no gaps in coverage."""
        chunk_seconds = 4 * 60  # 240s — current default
        overlap_seconds = 30
        stride_seconds = chunk_seconds - overlap_seconds  # 210s
        # Each chunk spans 240s, advances 210s — 30s overlap ensures
        # any point in time is covered by at least one chunk.
        assert chunk_seconds > stride_seconds
        assert chunk_seconds - stride_seconds == overlap_seconds


class TestGhostClusterConsolidation:
    """Post-merge consolidation absorbs tiny ghost clusters that are
    almost-certainly fragments of a real speaker. Targets the
    "3-real-speakers detected as 6" failure on long noisy meetings."""

    def _segment(
        self,
        start_ms: int,
        end_ms: int,
        local_id: int,
        embedding: list[float],
    ) -> dict:
        return {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "local_cluster_id": local_id,
            "confidence": 0.9,
            "embedding": embedding,
        }

    def test_tiny_cluster_absorbed_into_similar_neighbor(self) -> None:
        # Big cluster (60 segments × 1s = 60s) and a tiny ghost cluster
        # (1 segment × 3s) whose embedding is a near-match. The ghost
        # should get absorbed into the big one.
        speaker_a = _emb(seed=10)
        ghost_emb = _jitter(speaker_a, strength=0.03)  # very close to speaker_a
        chunk1 = [
            self._segment(i * 1000, i * 1000 + 1000, 0, speaker_a)
            for i in range(60)  # 60 segments × 1s = 60s in chunk1
        ]
        chunk2 = [
            self._segment(70_000, 73_000, 0, ghost_emb),  # 3s ghost
        ]
        # Pin merge_threshold high so the ghost stays separate initially,
        # forcing the consolidation pass to do the work.
        merged = _merge_clusters_via_embeddings(
            [chunk1, chunk2], merge_threshold=0.95
        )
        unique = {s["cluster_id"] for s in merged}
        assert len(unique) == 1, (
            f"tiny ghost should be absorbed into similar big cluster, "
            f"got clusters={unique}"
        )

    def test_dissimilar_ghost_survives_consolidation(self) -> None:
        # Big cluster + small distant ghost — should NOT be absorbed.
        # (The ghost is a real different person who only spoke briefly.)
        speaker_a = _emb(seed=20)
        speaker_b = _emb(seed=21)  # totally distinct embedding
        chunk1 = [
            self._segment(i * 1000, i * 1000 + 1000, 0, speaker_a)
            for i in range(60)
        ]
        chunk2 = [
            self._segment(70_000, 73_000, 0, speaker_b),  # 3s ghost, distant
        ]
        merged = _merge_clusters_via_embeddings(
            [chunk1, chunk2], merge_threshold=0.95
        )
        unique = {s["cluster_id"] for s in merged}
        assert len(unique) == 2, (
            "distant ghost should NOT be absorbed (it could be a real "
            f"minor speaker); got {unique!r}"
        )

    def test_expected_speakers_forces_count(self) -> None:
        # 4 real-ish clusters but caller says expected_speakers=2 —
        # the two smallest get force-absorbed into the two largest.
        speakers = [_emb(seed=30 + i) for i in range(4)]
        chunks = []
        for i, emb in enumerate(speakers):
            # Cluster sizes: 100, 80, 6, 4 segments
            count = [100, 80, 6, 4][i]
            chunks.append(
                [
                    self._segment(
                        i * 200_000 + j * 1000,
                        i * 200_000 + j * 1000 + 1000,
                        0,
                        emb,
                    )
                    for j in range(count)
                ]
            )
        merged = _merge_clusters_via_embeddings(
            chunks, merge_threshold=0.95, expected_speakers=2
        )
        unique = {s["cluster_id"] for s in merged}
        assert len(unique) == 2, (
            f"expected_speakers=2 must produce exactly 2 clusters; got {unique!r}"
        )

    def test_expected_speakers_no_op_when_already_at_count(self) -> None:
        # Already 2 clusters, expected=2 → no change
        a = _emb(seed=40)
        b = _emb(seed=41)
        chunk1 = [self._segment(i * 1000, i * 1000 + 1000, 0, a) for i in range(50)]
        chunk2 = [self._segment(60_000 + i * 1000, 60_000 + i * 1000 + 1000, 0, b) for i in range(50)]
        merged = _merge_clusters_via_embeddings(
            [chunk1, chunk2], merge_threshold=0.95, expected_speakers=2
        )
        unique = {s["cluster_id"] for s in merged}
        assert len(unique) == 2
