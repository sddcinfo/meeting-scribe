"""Tests for SortformerBackend's annealing + consolidation passes.

These target the "51 raw cluster_ids but only 6 real speakers" symptom
seen on the 2026-04-13 meetings. The fix makes two moves:

1. Anneal ``_cluster_merge_threshold`` from 0.55 → 0.48 over 120 s so
   later-arriving fragments re-merge into their speaker's centroid.
2. Periodic ``_consolidate_centroids`` pass merges any centroid pair
   with cosine ≥ 0.85.
"""

from __future__ import annotations

import numpy as np
import pytest

from meeting_scribe.backends.diarize_sortformer import SortformerBackend


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / max(float(np.linalg.norm(vec)), 1e-9)


def test_consolidation_merges_near_duplicates():
    be = SortformerBackend()
    # Three centroids, A and B are very close (cosine ≈ 0.97), C is distant.
    rng = np.random.default_rng(0)
    base_a = _unit(rng.standard_normal(128).astype(np.float32))
    # Small perturbation → drift stays cosine-close (~0.97) to base_a.
    perturb = 0.02 * rng.standard_normal(128).astype(np.float32)
    drift = _unit(base_a + perturb)
    assert float(np.dot(base_a, drift)) > 0.95
    distant = _unit(rng.standard_normal(128).astype(np.float32))

    be._global_centroids = {1: base_a, 2: drift, 3: distant}
    be._global_centroid_counts = {1: 10, 2: 3, 3: 8}
    be._global_centroid_created_at = {1: 0.0, 2: 10.0, 3: 5.0}

    be._consolidate_centroids()

    # 2 should be folded into 1 (higher count survives).
    assert set(be._global_centroids.keys()) == {1, 3}
    # Rename recorded so external consumers can rewrite stale refs.
    assert be._cluster_rename.get(2) == 1
    # Distant speaker untouched.
    assert be._global_centroid_counts[3] == 8
    # Merged count is sum of the two.
    assert be._global_centroid_counts[1] == 13


def test_consolidation_no_op_when_all_distinct():
    be = SortformerBackend()
    rng = np.random.default_rng(42)
    # Three truly orthogonal-ish vectors.
    cs = [_unit(rng.standard_normal(128).astype(np.float32)) for _ in range(3)]
    be._global_centroids = {1: cs[0], 2: cs[1], 3: cs[2]}
    be._global_centroid_counts = {1: 5, 2: 5, 3: 5}

    be._consolidation_cos_threshold = 0.85  # default
    be._consolidate_centroids()

    assert set(be._global_centroids.keys()) == {1, 2, 3}
    assert be._cluster_rename == {}


def test_annealing_relaxes_after_window():
    be = SortformerBackend()
    be._anneal_window_s = 10.0
    be._session_start_ts = None  # fresh

    rng = np.random.default_rng(1)
    a = _unit(rng.standard_normal(128).astype(np.float32))

    # First call: anneal starts at strict threshold.
    be._assign_global_cluster(local_id=0, embedding=a)
    first_threshold = be._cluster_merge_threshold

    # Simulate time passing past the window.
    import time as _time

    be._session_start_ts = _time.monotonic() - 20.0

    b = _unit(rng.standard_normal(128).astype(np.float32))
    be._assign_global_cluster(local_id=1, embedding=b)
    relaxed_threshold = be._cluster_merge_threshold

    assert first_threshold > relaxed_threshold
    assert relaxed_threshold == pytest.approx(be._cluster_merge_threshold_relaxed, abs=0.01)
