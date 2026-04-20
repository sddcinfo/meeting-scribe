"""Speaker verification — map diarization cluster_id to enrolled identities.

Separate from diarization: diarization assigns cluster_id per segment,
then this layer verifies against enrolled embeddings to assign identity.

Flow:
    Audio → DiarizeBackend → cluster_id per segment
                                  ↓
                       Speaker Verification Layer
                       (cosine similarity vs enrolled embeddings)
                                  ↓
                       identity + confidence, or source="unknown"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from meeting_scribe.models import SpeakerAttribution
from meeting_scribe.speaker.enrollment import SpeakerEnrollmentStore

logger = logging.getLogger(__name__)

# Confidence threshold for identity assignment
IDENTITY_THRESHOLD = 0.6
# How often to re-verify as cluster centroids update (every N segments)
REVERIFY_INTERVAL = 10


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class ClusterState:
    """Tracked state for a diarization cluster."""

    cluster_id: int
    centroid: np.ndarray | None = None
    segment_count: int = 0
    matched_identity: str | None = None
    matched_enrollment_id: str | None = None
    match_confidence: float = 0.0
    last_verified_at: int = 0  # segment_count at last verification


class SpeakerVerifier:
    """Maps diarization cluster IDs to enrolled speaker identities.

    Re-verifies periodically as cluster centroids update (drift handling).
    Supports overlap: returns multiple SpeakerAttributions for overlapping speech.
    """

    def __init__(self, enrollment_store: SpeakerEnrollmentStore) -> None:
        self._store = enrollment_store
        self._clusters: dict[int, ClusterState] = {}

    def verify(
        self,
        cluster_id: int,
        embedding: np.ndarray | None = None,
    ) -> SpeakerAttribution:
        """Verify a cluster against enrolled speakers.

        Args:
            cluster_id: Diarization cluster ID.
            embedding: Current segment embedding (optional, for centroid update).

        Returns:
            SpeakerAttribution with identity if confident, else "unknown".
        """
        state = self._clusters.get(cluster_id)
        if state is None:
            state = ClusterState(cluster_id=cluster_id)
            self._clusters[cluster_id] = state

        # Update centroid with new embedding
        if embedding is not None:
            if state.centroid is None:
                state.centroid = embedding.copy()
            else:
                # Running average
                state.centroid = (state.centroid * state.segment_count + embedding) / (
                    state.segment_count + 1
                )
            state.segment_count += 1

        # Re-verify if needed
        should_verify = (
            state.segment_count - state.last_verified_at >= REVERIFY_INTERVAL
            or state.matched_identity is None
        )

        if should_verify and state.centroid is not None:
            self._match_to_enrolled(state)

        # Build attribution
        if state.matched_identity and state.match_confidence >= IDENTITY_THRESHOLD:
            return SpeakerAttribution(
                cluster_id=cluster_id,
                identity=state.matched_identity,
                identity_confidence=state.match_confidence,
                source="enrolled",
            )

        return SpeakerAttribution(
            cluster_id=cluster_id,
            identity=None,
            identity_confidence=state.match_confidence,
            source="cluster_only" if state.segment_count > 0 else "unknown",
        )

    def verify_multiple(
        self,
        cluster_ids: list[int],
        embeddings: list[np.ndarray | None] | None = None,
    ) -> list[SpeakerAttribution]:
        """Verify multiple clusters (for overlapping speech segments)."""
        if embeddings is None:
            embeddings = [None] * len(cluster_ids)
        return [self.verify(cid, emb) for cid, emb in zip(cluster_ids, embeddings, strict=False)]

    def _match_to_enrolled(self, state: ClusterState) -> None:
        """Match a cluster centroid against all enrolled embeddings."""
        enrolled = self._store.get_all_embeddings()
        if not enrolled or state.centroid is None:
            return

        best_score = 0.0
        best_name: str | None = None
        best_eid: str | None = None

        for eid, name, embedding in enrolled:
            score = cosine_similarity(state.centroid, embedding)
            if score > best_score:
                best_score = score
                best_name = name
                best_eid = eid

        state.matched_identity = best_name
        state.matched_enrollment_id = best_eid
        state.match_confidence = best_score
        state.last_verified_at = state.segment_count

        if best_score >= IDENTITY_THRESHOLD:
            logger.info(
                "Cluster %d → %s (confidence=%.2f)",
                state.cluster_id,
                best_name,
                best_score,
            )
        else:
            logger.debug(
                "Cluster %d → no match (best=%.2f for %s)",
                state.cluster_id,
                best_score,
                best_name,
            )

    def reset(self) -> None:
        """Reset all cluster state (e.g., for a new meeting)."""
        self._clusters.clear()
