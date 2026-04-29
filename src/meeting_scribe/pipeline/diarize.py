"""Chunked / single-call diarization + cross-chunk cluster merging.

Production pipeline: ``pyannote/speaker-diarization-community-1``
(pyannote.audio 4.0.4 — promoted 2026-04-28 after Track C of
plans/stateful-marinating-whistle.md cleared every gate).

The diarize HTTP server returns two parallel arrays per chunk:

* ``segments`` — standard pyannote segmentation; carries embeddings
  and may have ≥ 2 speakers active in the same window when there's
  cross-talk.  Used here for cross-chunk cluster stitching.
* ``exclusive_segments`` — community-1 single-speaker-per-frame
  timeline (every frame is assigned to ONE speaker).  Used by
  ``speaker_attach._attach_speakers_to_events`` as the source of truth
  for primary-speaker assignment on each ASR event window.  This is the
  feature that fixes overlap-conflict reconciliation in the refinement
  worker — community-1 resolves 100 % of 3.1's overlapping seconds to
  a single-speaker assignment (Track C measurement, 40-min 4-speaker
  production meeting).

Three entry points:

* ``_diarize_single_call`` — one HTTP POST to the pyannote container
  for a single PCM chunk; returns ``{segments, exclusive_segments}``
  with chunk-local cluster IDs.
* ``_merge_clusters_via_embeddings`` — stitches per-chunk cluster IDs
  into a global numbering using cosine similarity on speaker
  embeddings (from ``segments``), then applies the same merge map to
  ``exclusive_segments``.  Absorbs ghost clusters and (optionally)
  forces the cluster count down to a caller-supplied speaker count.
* ``_diarize_full_audio`` — orchestrates the above for a whole-meeting
  recording, choosing single-call vs chunked-with-overlap based on
  duration.  Returns ``DiarizeResult(segments, exclusive_segments)``.
  Includes container-restart retry for the latent CUDA-wedge bug in
  the pyannote sortformer container.

Both the batch ``reprocess_meeting`` flow and the live "stop meeting"
finalize path in ``routes.meeting_lifecycle`` consume these.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

import httpx
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


@dataclass
class DiarizeResult:
    """Output of ``_diarize_full_audio``.

    Both arrays use the same global cluster ids (the cross-chunk merge
    map is applied to both).  Callers should prefer ``exclusive_segments``
    for primary-speaker assignment (each frame has exactly one speaker,
    by construction) and use ``segments`` only for cross-talk detection
    (where two speakers' standard turns overlap).
    """

    segments: list[dict] = field(default_factory=list)
    exclusive_segments: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.segments) or bool(self.exclusive_segments)

    def __len__(self) -> int:
        return len(self.segments)


# Cross-chunk merge threshold: cosine similarity above this counts as the
# same speaker. Production pyannote pipelines on WeSpeaker ResNet34
# embeddings sit in 0.70–0.80; we keep the initial pass strict to avoid
# false-positive merges of distinct speakers, then rely on the ghost
# consolidation below to fold in tiny fragments that obviously belong
# somewhere. Override with MEETING_SCRIBE_DIARIZE_MERGE_THRESHOLD.
try:
    _DEFAULT_MERGE_THRESHOLD = float(
        os.environ.get("MEETING_SCRIBE_DIARIZE_MERGE_THRESHOLD", "0.70")
    )
except ValueError:
    _DEFAULT_MERGE_THRESHOLD = 0.70

# Ghost-cluster heuristics. Mirrors pyannote's internal min_cluster_size=15
# rule but applied to our cross-chunk merged output. A cluster is a "ghost"
# if it has very little speech AND few segments. Ghosts get absorbed into
# their nearest neighbor if the embedding is a reasonable match — looser
# than the initial threshold since we already trust the fragment belongs
# *somewhere*. If no neighbor matches, the ghost survives as-is.
_GHOST_MAX_DURATION_MS = 15_000  # 15 s — matches pyannote min_cluster_size×~1s avg
_GHOST_MAX_SEGMENTS = 10
_GHOST_ABSORB_MIN_SIM = 0.55


async def _diarize_single_call(
    pcm_data: bytes,
    diarize_url: str,
    max_speakers: int,
    timeout: float,
    time_offset_ms: int = 0,
    min_speakers: int = 2,
) -> tuple[list[dict], list[dict]]:
    """POST a chunk of PCM to the diarization container.

    Returns ``(standard_segments, exclusive_segments)`` for this chunk;
    cluster ids are chunk-local and need to be stitched across chunks
    by ``_merge_clusters_via_embeddings`` for any meeting > 10 minutes.
    The two arrays share cluster ids by construction (server-side
    speaker_map is reused), so the merge map applies to both.

    min_speakers=2 is critical — without this hint, pyannote collapses
    ambiguous clustering into 1 speaker almost every time.

    time_offset_ms is added to returned start/end so caller can map
    chunk-local times back into the meeting timeline.
    """
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{diarize_url}/v1/diarize",
            content=pcm_data,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Max-Speakers": str(max_speakers),
                "X-Min-Speakers": str(min_speakers),
            },
        )
        r.raise_for_status()
        data = r.json()

    def _parse(arr: list[dict], *, with_embedding: bool) -> list[dict]:
        out = []
        for seg in arr:
            row = {
                "start_ms": int(seg.get("start", 0) * 1000) + time_offset_ms,
                "end_ms": int(seg.get("end", 0) * 1000) + time_offset_ms,
                "local_cluster_id": int(seg.get("speaker_id", 0)),
                "confidence": float(seg.get("confidence", 1.0)),
            }
            if with_embedding:
                row["embedding"] = seg.get("embedding")  # list of floats, may be None
            out.append(row)
        return out

    standard = _parse(data.get("segments", []), with_embedding=True)
    exclusive = _parse(data.get("exclusive_segments", []), with_embedding=False)
    return standard, exclusive


def _merge_clusters_via_embeddings(
    chunk_segments_list: list[list[dict]],
    merge_threshold: float | None = None,
    expected_speakers: int | None = None,
    chunk_exclusive_list: list[list[dict]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Merge diarization clusters across chunks using embedding similarity.

    Each chunk's diarization run uses its own local cluster IDs. To track
    speakers across chunks, we compare embeddings (carried only on
    standard segments — the exclusive output ships without embeddings):
    standard segments from different chunks with cosine similarity ≥
    threshold get the same global ID.

    After cross-chunk merging, runs a consolidation pass that absorbs
    ghost clusters (very little speech, small segment count) into their
    nearest neighbor when the embedding match is reasonable. This catches
    the common over-clustering pattern where pyannote splits one real
    speaker across multiple noisy sub-clusters in different chunks.

    If ``expected_speakers`` is given AND the consolidated cluster count
    is still higher, runs an additional forced-absorption pass: the
    smallest clusters get folded into their nearest neighbor (regardless
    of the absorb-similarity floor) until the count matches.

    The merge map (chunk-local → global cluster id, plus any subsequent
    ghost / forced absorptions) is applied to ``chunk_exclusive_list``
    too when provided, so both standard and exclusive segments share a
    consistent global cluster numbering.

    Returns ``(merged_segments, merged_exclusive_segments)``.  Segments
    without embeddings get their own cluster IDs (best effort).
    """
    if merge_threshold is None:
        merge_threshold = _DEFAULT_MERGE_THRESHOLD

    global_centroids: dict[int, np.ndarray] = {}
    global_counts: dict[int, int] = {}
    next_global_id = 1

    # Per-chunk merge maps so we can apply the same (local_cluster_id →
    # global_id) transformation to the exclusive segments after the
    # standard segments have been merged + consolidated.
    per_chunk_local_to_global: list[dict[int, int]] = [
        {} for _ in chunk_segments_list
    ]

    flat = []
    for chunk_idx, segs in enumerate(chunk_segments_list):
        # Within a single chunk, local cluster IDs are already consistent.
        # Build a chunk-local → global map once per local ID per chunk.
        local_to_global: dict[int, int] = per_chunk_local_to_global[chunk_idx]
        for seg in segs:
            local_id = seg["local_cluster_id"]
            if local_id in local_to_global:
                # Reuse the global ID we already assigned for this local ID
                # (consistent within the chunk)
                gid = local_to_global[local_id]
                if seg.get("embedding") is not None and gid in global_centroids:
                    # Update centroid with new sample
                    emb = np.array(seg["embedding"], dtype=np.float32)
                    count = global_counts[gid]
                    old = global_centroids[gid]
                    global_centroids[gid] = (old * count + emb) / (count + 1)
                    global_counts[gid] = count + 1
            else:
                # First time seeing this local ID — decide if it maps to an
                # existing global cluster or a new one
                emb_raw = seg.get("embedding")
                if emb_raw is None:
                    # No embedding → can't merge, allocate new global ID
                    gid = next_global_id
                    next_global_id += 1
                else:
                    emb_arr = np.array(emb_raw, dtype=np.float32)
                    # Find best matching global centroid
                    best_gid = None
                    best_score = -1.0
                    for g, centroid in global_centroids.items():
                        num = float(np.dot(emb_arr, centroid))
                        den = float(np.linalg.norm(emb_arr) * np.linalg.norm(centroid))
                        if den <= 0:
                            continue
                        score = num / den
                        if score > best_score:
                            best_score = score
                            best_gid = g
                    if best_gid is not None and best_score >= merge_threshold:
                        gid = best_gid
                        # Update centroid
                        count = global_counts[gid]
                        old = global_centroids[gid]
                        global_centroids[gid] = (old * count + emb_arr) / (count + 1)
                        global_counts[gid] = count + 1
                    else:
                        gid = next_global_id
                        next_global_id += 1
                        global_centroids[gid] = emb_arr.copy()
                        global_counts[gid] = 1
                local_to_global[local_id] = gid
                # (note: already assigned above for first-time case)

            flat.append(
                {
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                    "cluster_id": local_to_global[seg["local_cluster_id"]],
                    "confidence": seg["confidence"],
                }
            )

    flat.sort(key=lambda s: s["start_ms"])
    pre_consolidate_count = len(global_centroids)

    # ── Ghost-cluster consolidation ───────────────────────────
    # Real-world failure: a 3-speaker meeting comes back with 6 clusters
    # because pyannote split one speaker across noisy chunks. The post-
    # merge embedding centroids tell us which small clusters are likely
    # fragments of which big ones — absorb them.
    if len(global_centroids) > 1 and len(flat) > 0:
        # Aggregate per-cluster duration + segment count
        per_cluster_segments: dict[int, int] = {}
        per_cluster_duration_ms: dict[int, int] = {}
        for seg in flat:
            cid = seg["cluster_id"]
            per_cluster_segments[cid] = per_cluster_segments.get(cid, 0) + 1
            per_cluster_duration_ms[cid] = per_cluster_duration_ms.get(cid, 0) + max(
                0, seg["end_ms"] - seg["start_ms"]
            )

        ghost_candidates = sorted(
            (
                gid
                for gid, dur in per_cluster_duration_ms.items()
                if dur < _GHOST_MAX_DURATION_MS
                and per_cluster_segments.get(gid, 0) <= _GHOST_MAX_SEGMENTS
                and gid in global_centroids
            ),
            key=lambda g: per_cluster_duration_ms.get(g, 0),  # smallest first
        )

        absorbed: dict[int, int] = {}  # ghost_gid → host_gid
        for ghost in ghost_candidates:
            ghost_emb = global_centroids[ghost]
            ghost_norm = float(np.linalg.norm(ghost_emb))
            if ghost_norm <= 0:
                continue
            best_host = None
            best_score = -1.0
            for host_gid, host_emb in global_centroids.items():
                if host_gid == ghost or host_gid in absorbed:
                    continue
                # Don't absorb into another ghost — only into "real" clusters
                if per_cluster_duration_ms.get(host_gid, 0) < _GHOST_MAX_DURATION_MS:
                    continue
                host_norm = float(np.linalg.norm(host_emb))
                if host_norm <= 0:
                    continue
                score = float(np.dot(ghost_emb, host_emb)) / (ghost_norm * host_norm)
                if score > best_score:
                    best_score = score
                    best_host = host_gid
            if best_host is not None and best_score >= _GHOST_ABSORB_MIN_SIM:
                absorbed[ghost] = best_host
                logger.info(
                    "Diarize consolidation: absorbing ghost cluster %d "
                    "(%d segs / %.1fs) → cluster %d (cosine=%.2f)",
                    ghost,
                    per_cluster_segments.get(ghost, 0),
                    per_cluster_duration_ms.get(ghost, 0) / 1000.0,
                    best_host,
                    best_score,
                )

        if absorbed:
            for seg in flat:
                cid = seg["cluster_id"]
                if cid in absorbed:
                    seg["cluster_id"] = absorbed[cid]
            # Recompute per-cluster duration after ghost absorption so the
            # forced-count pass below sees the merged sizes.
            per_cluster_duration_ms = {}
            per_cluster_segments = {}
            for seg in flat:
                cid = seg["cluster_id"]
                per_cluster_segments[cid] = per_cluster_segments.get(cid, 0) + 1
                per_cluster_duration_ms[cid] = per_cluster_duration_ms.get(cid, 0) + max(
                    0, seg["end_ms"] - seg["start_ms"]
                )

    # ── Forced absorption to expected_speakers (caller hint) ──
    # When the user says "this meeting had N speakers", trust them: keep
    # the N largest clusters and fold the rest into their nearest neighbor
    # by embedding similarity, regardless of the absorb-floor. This is
    # the "I know better than the model" lever — it's wrong if the user
    # is wrong, but the user usually isn't.
    if expected_speakers is not None and expected_speakers > 0 and flat:
        live_clusters = list({s["cluster_id"] for s in flat})
        if len(live_clusters) > expected_speakers:
            sizes = sorted(
                live_clusters,
                key=lambda g: per_cluster_duration_ms.get(g, 0),
                reverse=True,
            )
            keep = set(sizes[:expected_speakers])
            drop = sizes[expected_speakers:]
            forced: dict[int, int] = {}
            for ghost in drop:
                if ghost not in global_centroids:
                    # Without an embedding, just attach to the largest cluster
                    forced[ghost] = sizes[0]
                    continue
                ghost_emb = global_centroids[ghost]
                ghost_norm = float(np.linalg.norm(ghost_emb))
                best_host = sizes[0]
                best_score = -1.0
                for host_gid in keep:
                    host_emb_opt = global_centroids.get(host_gid)
                    if host_emb_opt is None:
                        continue
                    host_emb = host_emb_opt
                    host_norm = float(np.linalg.norm(host_emb))
                    if host_norm <= 0 or ghost_norm <= 0:
                        continue
                    score = float(np.dot(ghost_emb, host_emb)) / (ghost_norm * host_norm)
                    if score > best_score:
                        best_score = score
                        best_host = host_gid
                forced[ghost] = best_host
                logger.info(
                    "Diarize forced-absorb to expected_speakers=%d: "
                    "cluster %d (%d segs / %.1fs) → cluster %d (cosine=%.2f)",
                    expected_speakers,
                    ghost,
                    per_cluster_segments.get(ghost, 0),
                    per_cluster_duration_ms.get(ghost, 0) / 1000.0,
                    best_host,
                    best_score,
                )
            for seg in flat:
                if seg["cluster_id"] in forced:
                    seg["cluster_id"] = forced[seg["cluster_id"]]

    final_count = len({s["cluster_id"] for s in flat}) if flat else 0
    logger.info(
        "Merged %d chunks into %d global speakers (merge threshold=%.2f, consolidated %d → %d)",
        len(chunk_segments_list),
        final_count,
        merge_threshold,
        pre_consolidate_count,
        final_count,
    )

    # Project the same merge map onto the exclusive segments so both
    # arrays share a global cluster numbering.  Ghost / forced
    # absorptions tracked above are folded in as a chained mapping.
    exclusive_flat: list[dict] = []
    if chunk_exclusive_list:
        # Build a single chained map: local (chunk_idx, local_id) →
        # global → (after-ghost-absorb) → (after-forced-absorb).
        ghost_absorb = locals().get("absorbed", {}) or {}
        forced_absorb = locals().get("forced", {}) or {}

        def _final_global(global_id: int) -> int:
            gid = ghost_absorb.get(global_id, global_id)
            gid = forced_absorb.get(gid, gid)
            return gid

        for chunk_idx, excl_segs in enumerate(chunk_exclusive_list):
            local_to_global = per_chunk_local_to_global[chunk_idx] if chunk_idx < len(per_chunk_local_to_global) else {}
            for seg in excl_segs:
                local_id = seg["local_cluster_id"]
                global_id = local_to_global.get(local_id)
                if global_id is None:
                    # The exclusive output saw a speaker the standard
                    # output never assigned — allocate a fresh global
                    # id so the cluster doesn't collide with merged ones.
                    global_id = next_global_id
                    next_global_id += 1
                exclusive_flat.append(
                    {
                        "start_ms": seg["start_ms"],
                        "end_ms": seg["end_ms"],
                        "cluster_id": _final_global(global_id),
                        "confidence": seg["confidence"],
                    }
                )
        exclusive_flat.sort(key=lambda s: s["start_ms"])

    return flat, exclusive_flat


async def _diarize_full_audio(
    pcm_data: bytes,
    diarize_url: str,
    max_speakers: int = 10,
    min_speakers: int = 2,
    expected_speakers: int | None = None,
) -> DiarizeResult:
    """Run diarization on the ENTIRE meeting audio.

    Strategy:
      - ≤10 minutes: single call (fastest, best clustering).
      - >10 minutes: split into 8-minute chunks with 45s overlap and
        merge cluster IDs across chunks via embedding similarity.
        Scales to any meeting length (tested mental model: 90 min = 12 chunks).

    Returns a ``DiarizeResult`` carrying ``segments`` (standard
    pyannote 4.x output, possibly overlapping when there's cross-talk)
    and ``exclusive_segments`` (community-1 single-speaker timeline,
    every frame assigned to ≤ 1 speaker).  Both arrays share global
    cluster ids and are sorted by ``start_ms``.
    """
    total_samples = len(pcm_data) // BYTES_PER_SAMPLE
    total_ms = int(total_samples / SAMPLE_RATE * 1000)

    # Small meetings → single call. Pyannote already clustered the whole
    # file together, so we trust its cluster IDs directly — no cross-call
    # merging needed. Re-merging with cosine similarity can collapse
    # distinct speakers back into one when their embeddings are noisy.
    if total_ms <= 10 * 60 * 1000:
        try:
            standard, exclusive = await _diarize_single_call(
                pcm_data,
                diarize_url,
                max_speakers,
                timeout=600.0,
                min_speakers=min_speakers,
            )

            def _local_to_final(rows: list[dict]) -> list[dict]:
                return [
                    {
                        "start_ms": s["start_ms"],
                        "end_ms": s["end_ms"],
                        "cluster_id": s["local_cluster_id"],
                        "confidence": s["confidence"],
                    }
                    for s in rows
                ]

            return DiarizeResult(
                segments=_local_to_final(standard),
                exclusive_segments=_local_to_final(exclusive),
            )
        except Exception as e:
            logger.warning("Single-call diarization failed: %s", e)
            return DiarizeResult()

    # Long meetings → chunked with overlap.
    #
    # Why 4 minutes (was 8): real-world failure mode is the diarize worker
    # dying mid-inference on big chunks under tight system memory. A
    # 109MB / 60-min meeting was producing 9 × 480s chunks, each ~30MB of
    # int16 PCM that pyannote expanded to float32 + intermediate tensors,
    # and the worker disconnected partway through. Halving the chunk drops
    # working set ~2x at the cost of ~2x cluster-merge stitching, which is
    # cheap compared to losing a chunk entirely.
    #
    # Override via MEETING_SCRIBE_DIARIZE_CHUNK_SECONDS if a deployment has
    # the headroom and wants the original behavior back.
    try:
        CHUNK_SECONDS = int(os.environ.get("MEETING_SCRIBE_DIARIZE_CHUNK_SECONDS", "240"))
    except ValueError:
        CHUNK_SECONDS = 240
    OVERLAP_SECONDS = 30  # overlap window for cluster stitching
    chunk_bytes = CHUNK_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE
    stride_bytes = (CHUNK_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE * BYTES_PER_SAMPLE

    tasks = []
    for offset in range(0, len(pcm_data), stride_bytes):
        chunk = pcm_data[offset : offset + chunk_bytes]
        if len(chunk) < SAMPLE_RATE * BYTES_PER_SAMPLE * 5:
            # <5s — not enough to diarize, skip
            continue
        offset_ms = int(offset / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000)
        tasks.append((offset, offset_ms, chunk))

    logger.info(
        "Chunked diarization: %.0fmin audio → %d chunks of %ds (%ds overlap)",
        total_ms / 60000,
        len(tasks),
        CHUNK_SECONDS,
        OVERLAP_SECONDS,
    )

    # Process chunks SERIALLY with automatic container recovery.
    #
    # Pyannote sortformer has a latent CUDA bug: after ~6-8 consecutive
    # calls its context enters an unrecoverable 'unknown error' state
    # and every subsequent call returns 500 in ~20 ms. On f38d5807
    # (88 min audio = 13 chunks) this used to truncate the transcript
    # at the 44-min mark and leave half the meeting un-diarized.
    #
    # Recovery strategy:
    #
    #   1. SERIAL calls only (parallel calls wedge CUDA even faster).
    #   2. On N consecutive failures, assume the container is wedged
    #      and restart it in-place via ``compose restart
    #      pyannote-diarize``. A single container restart takes ~5-8 s
    #      once fastsafetensors is enabled — far cheaper than losing
    #      44 minutes of transcript to a circuit breaker.
    #   3. Wait for the health endpoint to return 200 with
    #      ``diarization_model: true``, then retry the failed chunk.
    #   4. Cap the restart attempts per reprocess at
    #      MAX_CONTAINER_RESTARTS so a hard-broken container (not
    #      just wedged) doesn't loop forever.
    MAX_CONSECUTIVE_FAILURES = 2
    MAX_CONTAINER_RESTARTS = 3
    all_chunk_segs: list[list[dict]] = []
    all_chunk_exclusive: list[list[dict]] = []
    consecutive_failures = 0
    container_restarts = 0

    async def _wait_for_diarize_health(timeout_s: float = 30.0) -> bool:
        """Poll the pyannote container health endpoint until ready.

        Returns True when the container reports it has both the
        diarization and embedding models loaded.
        """
        import time as _t

        deadline = _t.monotonic() + timeout_s
        async with httpx.AsyncClient(timeout=3.0) as c:
            while _t.monotonic() < deadline:
                try:
                    r = await c.get(f"{diarize_url}/health")
                    if r.status_code == 200:
                        j = r.json()
                        if j.get("diarization_model") and j.get("embedding_model"):
                            return True
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        return False

    async def _restart_diarize_container() -> bool:
        """Restart the pyannote container via docker compose and wait
        for its health endpoint to come back. Returns True on success."""
        nonlocal container_restarts
        if container_restarts >= MAX_CONTAINER_RESTARTS:
            logger.error(
                "Diarize container restart budget exhausted (%d) — giving up on remaining chunks",
                container_restarts,
            )
            return False
        container_restarts += 1
        logger.warning(
            "Restarting pyannote-diarize container (attempt %d/%d)...",
            container_restarts,
            MAX_CONTAINER_RESTARTS,
        )
        try:
            from meeting_scribe.infra.compose import compose_restart

            await asyncio.to_thread(compose_restart, "pyannote-diarize", 30)
        except Exception as e:
            logger.error("compose restart pyannote-diarize failed: %s", e)
            return False
        if await _wait_for_diarize_health(timeout_s=30.0):
            logger.info("pyannote-diarize recovered")
            return True
        logger.error("pyannote-diarize did not come back healthy after restart")
        return False

    async def _diarize_with_retry(
        chunk: bytes, offset_ms: int
    ) -> tuple[list[dict], list[dict]] | None:
        """Try a single diarize call; on consecutive failure, restart
        the container and try again. Returns ``(standard, exclusive)``
        or None if permanently broken."""
        nonlocal consecutive_failures
        try:
            result = await _diarize_single_call(
                chunk,
                diarize_url,
                max_speakers,
                timeout=240.0,
                time_offset_ms=offset_ms,
                min_speakers=min_speakers,
            )
            consecutive_failures = 0
            return result
        except Exception as e:
            consecutive_failures += 1
            logger.warning("diarize call failed (%d consecutive): %s", consecutive_failures, e)
            if consecutive_failures < MAX_CONSECUTIVE_FAILURES:
                return None  # transient — let caller decide
            # Circuit-breaker tripped. Restart the container and retry.
            if not await _restart_diarize_container():
                return None
            consecutive_failures = 0
            try:
                result = await _diarize_single_call(
                    chunk,
                    diarize_url,
                    max_speakers,
                    timeout=240.0,
                    time_offset_ms=offset_ms,
                    min_speakers=min_speakers,
                )
                return result
            except Exception as e2:
                logger.error("diarize failed again after restart: %s", e2)
                return None

    for i, (_offset, offset_ms, chunk) in enumerate(tasks):
        result_maybe = await _diarize_with_retry(chunk, offset_ms)
        if result_maybe is None:
            logger.warning(
                "Skipping diarize chunk %d/%d — unable to recover",
                i + 1,
                len(tasks),
            )
            continue
        standard, exclusive = result_maybe
        logger.info(
            "Diarize chunk %d/%d: %d standard / %d exclusive segments at offset %ds",
            i + 1,
            len(tasks),
            len(standard),
            len(exclusive),
            offset_ms // 1000,
        )
        all_chunk_segs.append(standard)
        all_chunk_exclusive.append(exclusive)

    if not all_chunk_segs:
        return DiarizeResult()

    # Merge via embedding similarity for cross-chunk cluster stability;
    # the same merge map is then applied to the exclusive segments so
    # both arrays share global cluster ids.
    merged, merged_exclusive = _merge_clusters_via_embeddings(
        all_chunk_segs,
        expected_speakers=expected_speakers,
        chunk_exclusive_list=all_chunk_exclusive,
    )
    return DiarizeResult(segments=merged, exclusive_segments=merged_exclusive)
