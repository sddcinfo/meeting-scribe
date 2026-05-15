"""Tests for speaker identity pipeline — enrollment, verification, cosine similarity."""

from __future__ import annotations

import numpy as np
import pytest

from meeting_scribe.speaker.enrollment import EnrolledSpeaker, SpeakerEnrollmentStore
from meeting_scribe.speaker.verification import (
    SpeakerVerifier,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == 0.0

    def test_similar_vectors_high_score(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        assert cosine_similarity(a, b) > 0.99


class TestEnrolledSpeaker:
    def test_to_dict_roundtrip(self):
        emb = np.random.randn(192).astype(np.float32)
        speaker = EnrolledSpeaker(
            name="Brad", embedding=emb, enrollment_id="e1", audio_duration_seconds=5.0
        )
        d = speaker.to_dict()
        restored = EnrolledSpeaker.from_dict(d)
        assert restored.name == "Brad"
        assert restored.enrollment_id == "e1"
        assert np.allclose(restored.embedding, emb)

    def test_from_dict_minimal(self):
        d = {"name": "Test", "embedding": [0.1, 0.2, 0.3]}
        s = EnrolledSpeaker.from_dict(d)
        assert s.name == "Test"
        assert len(s.embedding) == 3


class TestEnrollmentStore:
    def _make_speaker(self, eid, name):
        return EnrolledSpeaker(
            name=name, embedding=np.random.randn(192).astype(np.float32), enrollment_id=eid
        )

    def test_add_and_get(self):
        store = SpeakerEnrollmentStore()
        store.add(self._make_speaker("e1", "Brad"))
        assert "e1" in store.speakers
        assert store.speakers["e1"].name == "Brad"

    def test_remove(self):
        store = SpeakerEnrollmentStore()
        store.add(self._make_speaker("e1", "Brad"))
        assert store.remove("e1") is True
        assert "e1" not in store.speakers

    def test_remove_nonexistent(self):
        store = SpeakerEnrollmentStore()
        assert store.remove("nope") is False

    def test_get_all_embeddings(self):
        store = SpeakerEnrollmentStore()
        store.add(self._make_speaker("e1", "Brad"))
        store.add(self._make_speaker("e2", "Alice"))
        embs = store.get_all_embeddings()
        assert len(embs) == 2
        assert embs[0][1] in ("Brad", "Alice")  # (eid, name, embedding)

    def test_persist_and_load(self, tmp_path):
        path = tmp_path / "speakers.json"
        store = SpeakerEnrollmentStore(storage_path=path)
        store.add(self._make_speaker("e1", "Brad"))

        store2 = SpeakerEnrollmentStore(storage_path=path)
        store2.load()
        assert "e1" in store2.speakers
        assert store2.speakers["e1"].name == "Brad"

    def test_empty_store(self):
        store = SpeakerEnrollmentStore()
        assert store.speakers == {}
        assert store.get_all_embeddings() == []


class TestSpeakerVerifier:
    def _make_store_with_speakers(self):
        store = SpeakerEnrollmentStore()
        emb_brad = np.zeros(192, dtype=np.float32)
        emb_brad[:96] = 1.0
        emb_alice = np.zeros(192, dtype=np.float32)
        emb_alice[96:] = 1.0
        store.add(EnrolledSpeaker(name="Brad", embedding=emb_brad, enrollment_id="e1"))
        store.add(EnrolledSpeaker(name="Alice", embedding=emb_alice, enrollment_id="e2"))
        return store

    def test_verify_returns_attribution(self):
        store = self._make_store_with_speakers()
        verifier = SpeakerVerifier(store)
        emb = np.zeros(192, dtype=np.float32)
        emb[:96] = 0.9
        result = verifier.verify(cluster_id=0, embedding=emb)
        assert result is not None
        assert hasattr(result, "cluster_id")

    def test_verify_empty_store(self):
        store = SpeakerEnrollmentStore()
        verifier = SpeakerVerifier(store)
        result = verifier.verify(cluster_id=0, embedding=np.random.randn(192).astype(np.float32))
        # Should return an attribution even with no enrolled speakers
        assert result is not None

    def test_cluster_state_tracked(self):
        store = self._make_store_with_speakers()
        verifier = SpeakerVerifier(store)
        emb = np.zeros(192, dtype=np.float32)
        emb[:96] = 1.0
        for _ in range(5):
            verifier.verify(cluster_id=0, embedding=emb)
        assert 0 in verifier._clusters
        assert verifier._clusters[0].segment_count == 5
