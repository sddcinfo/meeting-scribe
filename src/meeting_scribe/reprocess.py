"""Full meeting reprocessing — re-run ASR + translation on original audio.

Reads the raw PCM recording, transcribes with Qwen3-ASR, translates,
and regenerates all meeting artifacts (journal, timeline, speakers, summary).

Parallelized: ASR and translation run with concurrent workers to leverage
vLLM's batching (4+ concurrent requests = ~3x throughput).

Usage:
    from meeting_scribe.reprocess import reprocess_meeting
    result = await reprocess_meeting(meeting_dir, asr_url, translate_url)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import shutil
import time
import uuid
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SECONDS = 4.0
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
CONCURRENCY = 10  # vLLM does continuous batching; 4 was underfeeding
# the ASR + translate instances. Reprocess runs at default priority
# (no header), so live ASR/TTS/translate (priority -20/-10) always
# preempt it — a meeting in progress is never delayed by a reprocess
# pass, regardless of how wide we set concurrency here.


def _encode_wav(pcm_chunk: bytes) -> str:
    """Encode s16le PCM bytes to base64 WAV."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_chunk)
    return base64.b64encode(buf.getvalue()).decode()


async def _transcribe_chunk(
    client: httpx.AsyncClient,
    asr_url: str,
    asr_model: str,
    chunk: bytes,
    start_ms: int,
    end_ms: int,
    language_pair: list[str] | tuple[str, ...] | None = None,
) -> dict | None:
    """Transcribe a single audio chunk. Returns event dict or None.

    ``language_pair`` enables the lingua post-correction (same as the
    live ASR path). Accepts 1 or 2 codes. Without it, Qwen3-ASR's
    English bias on Germanic speech (Dutch/German/etc.) goes
    uncorrected.
    """
    from meeting_scribe.backends.asr_filters import (
        _is_hallucination,
        _parse_qwen3_asr_response,
    )

    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    if float(np.sqrt(np.mean(audio**2))) < 0.005:
        return None  # Silence

    wav_b64 = _encode_wav(chunk)

    # NOTE: Tried adding a system prompt naming the meeting languages here
    # (matching the live ASR backend) and it caused a 56% segment-count
    # regression on the e5b376b2 dataset. The diarize-window-sized chunks
    # used in reprocess are shorter and noisier than the live backend's
    # 3.5s buffered ones — Qwen3-ASR with the prompt rejects more of
    # them. Caught by `meeting-scribe versions diff` after run #2.
    # The lingua post-correction below remains since it's free (~0.6ms)
    # and per-segment, so it works even without the prompt.
    try:
        resp = await client.post(
            f"{asr_url}/v1/chat/completions",
            json={
                "model": asr_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": wav_b64, "format": "wav"},
                            }
                        ],
                    }
                ],
                "max_tokens": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug("ASR chunk at %dms failed: %s", start_ms, e)
        return None

    # Parse Qwen3-ASR response: "language English<asr_text>actual text"
    text, detected_lang = _parse_qwen3_asr_response(text)

    if not text or len(text) < 2 or _is_hallucination(text):
        return None

    # Second-opinion the language tag against lingua, constrained to the
    # meeting's pair. Same correction step the live ASR backend runs.
    if language_pair:
        try:
            from meeting_scribe.language_correction import correct_segment_language

            detected_lang = correct_segment_language(text, detected_lang, language_pair)
        except Exception:
            logger.debug("lingua correction failed in reprocess", exc_info=True)

    return {
        "segment_id": str(uuid.uuid4()),
        "revision": 0,
        "is_final": True,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "language": detected_lang,
        "text": text,
        "speakers": [],
        "translation": None,
    }


async def _translate_event(
    client: httpx.AsyncClient,
    translate_url: str,
    trans_model: str,
    event: dict,
    language_pair: list[str] | tuple[str, ...],
) -> None:
    """Translate a single event in-place. No-op for monolingual meetings
    (``get_translation_target`` returns ``None``)."""
    from meeting_scribe.languages import get_translation_prompt, get_translation_target

    target = get_translation_target(event["language"], language_pair)
    if not target:
        return

    prompt = get_translation_prompt(event["language"], target)
    try:
        resp = await client.post(
            f"{translate_url}/v1/chat/completions",
            json={
                "model": trans_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": event["text"]},
                ],
                "max_tokens": 200,
                "temperature": 0.1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        event["translation"] = {"status": "done", "text": raw, "target_language": target}
    except Exception as e:
        logger.debug("Translation failed for %s: %s", event["segment_id"], e)


async def _diarize_single_call(
    pcm_data: bytes,
    diarize_url: str,
    max_speakers: int,
    timeout: float,
    time_offset_ms: int = 0,
    min_speakers: int = 2,
) -> list[dict]:
    """POST a chunk of PCM to the diarization container and parse segments.

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

    segments = []
    for seg in data.get("segments", []):
        segments.append(
            {
                "start_ms": int(seg.get("start", 0) * 1000) + time_offset_ms,
                "end_ms": int(seg.get("end", 0) * 1000) + time_offset_ms,
                "local_cluster_id": int(seg.get("speaker_id", 0)),
                "confidence": float(seg.get("confidence", 1.0)),
                "embedding": seg.get("embedding"),  # list of floats, may be None
            }
        )
    return segments


# Cross-chunk merge threshold: cosine similarity above this counts as the
# same speaker. Production pyannote pipelines on WeSpeaker ResNet34
# embeddings sit in 0.70–0.80; we keep the initial pass strict to avoid
# false-positive merges of distinct speakers, then rely on the ghost
# consolidation below to fold in tiny fragments that obviously belong
# somewhere. Override with MEETING_SCRIBE_DIARIZE_MERGE_THRESHOLD.
import os as _os

try:
    _DEFAULT_MERGE_THRESHOLD = float(
        _os.environ.get("MEETING_SCRIBE_DIARIZE_MERGE_THRESHOLD", "0.70")
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


def _merge_clusters_via_embeddings(
    chunk_segments_list: list[list[dict]],
    merge_threshold: float | None = None,
    expected_speakers: int | None = None,
) -> list[dict]:
    """Merge diarization clusters across chunks using embedding similarity.

    Each chunk's diarization run uses its own local cluster IDs. To track
    speakers across chunks, we compare embeddings: segments from different
    chunks with cosine similarity >= threshold get the same global ID.

    After cross-chunk merging, runs a consolidation pass that absorbs
    ghost clusters (very little speech, small segment count) into their
    nearest neighbor when the embedding match is reasonable. This catches
    the common over-clustering pattern where pyannote splits one real
    speaker across multiple noisy sub-clusters in different chunks.

    If ``expected_speakers`` is given AND the consolidated cluster count
    is still higher, runs an additional forced-absorption pass: the
    smallest clusters get folded into their nearest neighbor (regardless
    of the absorb-similarity floor) until the count matches.

    Returns a flat list of segments with stable `cluster_id` across chunks.
    Segments without embeddings get their own cluster IDs (best effort).
    """
    if merge_threshold is None:
        merge_threshold = _DEFAULT_MERGE_THRESHOLD

    global_centroids: dict[int, np.ndarray] = {}
    global_counts: dict[int, int] = {}
    next_global_id = 1

    flat = []
    for chunk_idx, segs in enumerate(chunk_segments_list):
        # Within a single chunk, local cluster IDs are already consistent.
        # Build a chunk-local → global map once per local ID per chunk.
        local_to_global: dict[int, int] = {}
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
                gid for gid, dur in per_cluster_duration_ms.items()
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
        "Merged %d chunks into %d global speakers (merge threshold=%.2f, "
        "consolidated %d → %d)",
        len(chunk_segments_list),
        final_count,
        merge_threshold,
        pre_consolidate_count,
        final_count,
    )
    return flat


def _audio_quality_report(pcm_data: bytes) -> dict:
    """Detect if a recording has been corrupted by the zero-gap writer bug.

    Returns {zero_fill_pct, longest_zero_run_ms, usable}. Zero-fill >40%
    means diarization is fundamentally broken for this meeting.
    """
    if len(pcm_data) < 4:
        return {"zero_fill_pct": 0, "longest_zero_run_ms": 0, "usable": True}
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    zero_mask = samples == 0
    total_zero = int(zero_mask.sum())
    pct = total_zero / len(samples) * 100

    # Find longest contiguous zero run
    longest = 0
    current = 0
    for z in zero_mask:
        if z:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    longest_ms = int(longest / SAMPLE_RATE * 1000)

    return {
        "zero_fill_pct": round(pct, 1),
        "longest_zero_run_ms": longest_ms,
        "usable": pct < 40,
    }


async def _diarize_full_audio(
    pcm_data: bytes,
    diarize_url: str,
    max_speakers: int = 10,
    min_speakers: int = 2,
    expected_speakers: int | None = None,
) -> list[dict]:
    """Run diarization on the ENTIRE meeting audio.

    Strategy:
      - ≤10 minutes: single call (fastest, best clustering).
      - >10 minutes: split into 8-minute chunks with 45s overlap and
        merge cluster IDs across chunks via embedding similarity.
        Scales to any meeting length (tested mental model: 90 min = 12 chunks).

    Returns a list of {start_ms, end_ms, cluster_id, confidence} with
    stable cluster_ids across the whole meeting, sorted by start_ms.
    """
    total_samples = len(pcm_data) // BYTES_PER_SAMPLE
    total_ms = int(total_samples / SAMPLE_RATE * 1000)

    # Small meetings → single call. Pyannote already clustered the whole
    # file together, so we trust its cluster IDs directly — no cross-call
    # merging needed. Re-merging with cosine similarity can collapse
    # distinct speakers back into one when their embeddings are noisy.
    if total_ms <= 10 * 60 * 1000:
        try:
            segs = await _diarize_single_call(
                pcm_data,
                diarize_url,
                max_speakers,
                timeout=600.0,
                min_speakers=min_speakers,
            )
            return [
                {
                    "start_ms": s["start_ms"],
                    "end_ms": s["end_ms"],
                    "cluster_id": s["local_cluster_id"],
                    "confidence": s["confidence"],
                }
                for s in segs
            ]
        except Exception as e:
            logger.warning("Single-call diarization failed: %s", e)
            return []

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
    import os as _os
    try:
        CHUNK_SECONDS = int(_os.environ.get("MEETING_SCRIBE_DIARIZE_CHUNK_SECONDS", "240"))
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
                "Diarize container restart budget exhausted (%d) — "
                "giving up on remaining chunks", container_restarts,
            )
            return False
        container_restarts += 1
        logger.warning(
            "Restarting pyannote-diarize container (attempt %d/%d)...",
            container_restarts, MAX_CONTAINER_RESTARTS,
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

    async def _diarize_with_retry(chunk: bytes, offset_ms: int) -> list[dict] | None:
        """Try a single diarize call; on consecutive failure, restart
        the container and try again. Returns None if permanently broken."""
        nonlocal consecutive_failures
        try:
            segs = await _diarize_single_call(
                chunk, diarize_url, max_speakers,
                timeout=240.0, time_offset_ms=offset_ms, min_speakers=min_speakers,
            )
            consecutive_failures = 0
            return segs
        except Exception as e:
            consecutive_failures += 1
            logger.warning("diarize call failed (%d consecutive): %s",
                           consecutive_failures, e)
            if consecutive_failures < MAX_CONSECUTIVE_FAILURES:
                return None  # transient — let caller decide
            # Circuit-breaker tripped. Restart the container and retry.
            if not await _restart_diarize_container():
                return None
            consecutive_failures = 0
            try:
                segs = await _diarize_single_call(
                    chunk, diarize_url, max_speakers,
                    timeout=240.0, time_offset_ms=offset_ms, min_speakers=min_speakers,
                )
                return segs
            except Exception as e2:
                logger.error("diarize failed again after restart: %s", e2)
                return None

    for i, (_offset, offset_ms, chunk) in enumerate(tasks):
        segs_maybe: list[dict] | None = await _diarize_with_retry(chunk, offset_ms)
        if segs_maybe is None:
            logger.warning(
                "Skipping diarize chunk %d/%d — unable to recover",
                i + 1, len(tasks),
            )
            continue
        segs = segs_maybe
        logger.info(
            "Diarize chunk %d/%d: %d segments at offset %ds",
            i + 1, len(tasks), len(segs), offset_ms // 1000,
        )
        all_chunk_segs.append(segs)

    if not all_chunk_segs:
        return []

    # Merge via embedding similarity for cross-chunk cluster stability
    return _merge_clusters_via_embeddings(
        all_chunk_segs, expected_speakers=expected_speakers
    )


def _attach_speakers_to_events(
    events: list[dict],
    diarize_segments: list[dict],
) -> None:
    """Attach diarization clusters to ASR events, preserving overlap.

    pyannote emits one diarization segment per (speaker, turn) — so an
    ASR final that spans ~4 s of cross-talk gets multiple overlapping
    diarize segments with different cluster_ids. Previously we took the
    single best-overlapping one and dropped the rest, losing all signal
    that two (or more) speakers were active. Now we keep every speaker
    whose overlap is meaningful, ordered primary-first so downstream
    code that reads `speakers[0]` still gets the dominant voice.

    Inclusion rules (keep both thresholds narrow enough that a 200 ms
    "mhm" interjection doesn't get tagged as co-speech):
      * secondary speaker's overlap ≥ 30 % of the event's duration
      * secondary speaker's overlap ≥ 50 % of the primary's overlap
    """
    if not diarize_segments:
        # No diarization — leave events with empty speakers (fallback to
        # time-proximity clustering done elsewhere)
        return

    _MIN_SECONDARY_FRAC_OF_EVENT = 0.30
    _MIN_SECONDARY_FRAC_OF_PRIMARY = 0.50

    # First pass: standard max-overlap primary + secondary assignment.
    for event in events:
        s, e = event.get("start_ms", 0), event.get("end_ms", 0)
        ev_dur = max(1, e - s)
        # Collect per-cluster overlap (pyannote may return multiple
        # diarize segments for the same cluster across the event).
        overlap_by_cluster: dict[int, float] = {}
        conf_by_cluster: dict[int, float] = {}
        for seg in diarize_segments:
            overlap = max(0, min(e, seg["end_ms"]) - max(s, seg["start_ms"]))
            if overlap <= 0:
                continue
            cid = seg["cluster_id"]
            overlap_by_cluster[cid] = overlap_by_cluster.get(cid, 0) + overlap
            # Keep the best confidence if multiple sub-segments exist.
            conf_by_cluster[cid] = max(conf_by_cluster.get(cid, 0.0), seg.get("confidence", 1.0))
        if not overlap_by_cluster:
            continue

        ranked = sorted(overlap_by_cluster.items(), key=lambda kv: -kv[1])
        primary_cid, primary_overlap = ranked[0]
        speakers_list = [
            {
                "cluster_id": primary_cid,
                "identity": None,
                "identity_confidence": conf_by_cluster[primary_cid],
                "source": "diarization",
                "display_name": None,
            }
        ]
        for cid, ov in ranked[1:]:
            if ov / ev_dur < _MIN_SECONDARY_FRAC_OF_EVENT:
                continue
            if ov / primary_overlap < _MIN_SECONDARY_FRAC_OF_PRIMARY:
                continue
            speakers_list.append(
                {
                    "cluster_id": cid,
                    "identity": None,
                    "identity_confidence": conf_by_cluster[cid],
                    "source": "diarization_overlap",
                    "display_name": None,
                }
            )
        event["speakers"] = speakers_list
        if len(speakers_list) > 1:
            event["overlapping_speakers"] = True

    # Second pass: minority-speaker recovery. Some speakers (especially
    # the ones who only said "yes" or "mmm-hmm") have diarize segments
    # that never WIN the primary overlap on any ASR event — so they'd
    # disappear from detected_speakers even though pyannote correctly
    # identified them as distinct voices. Catch this by cross-referencing
    # the set of clusters pyannote actually found vs the set of clusters
    # that ended up as primary on at least one event. Any cluster that
    # got attributed to NO events gets rescued here: find the event
    # with the highest total overlap on that cluster and promote the
    # cluster to primary on THAT event (the previous primary becomes a
    # secondary). Preserves "minority voice has ≥1 primary row" without
    # fighting pyannote or lowering the overlap thresholds globally.
    diarize_clusters = {seg["cluster_id"] for seg in diarize_segments}
    primary_clusters_seen: set[int] = set()
    for ev in events:
        sp = (ev.get("speakers") or [])
        if sp:
            primary_clusters_seen.add(sp[0].get("cluster_id"))
    missing = diarize_clusters - primary_clusters_seen

    for miss_cid in missing:
        # Find the event with the most total overlap on this cluster.
        best_ev = None
        best_ov = 0
        for ev in events:
            s, e = ev.get("start_ms", 0), ev.get("end_ms", 0)
            ov = 0
            for seg in diarize_segments:
                if seg["cluster_id"] != miss_cid:
                    continue
                ov += max(0, min(e, seg["end_ms"]) - max(s, seg["start_ms"]))
            if ov > best_ov:
                best_ov = ov
                best_ev = ev
        if best_ev is None or best_ov <= 0:
            # Cluster found by diarize but lives in an ASR gap. This
            # happens when finalize re-diarizes a meeting whose ASR
            # chunks were shaped by a PREVIOUS (different) diarize
            # pass — the new cluster's time ranges don't coincide
            # with any ASR event. Only fixable by running full
            # /reprocess which re-shapes ASR chunks to match.
            cluster_ranges = [
                (seg["start_ms"], seg["end_ms"])
                for seg in diarize_segments
                if seg["cluster_id"] == miss_cid
            ]
            total_s = sum(e - s for s, e in cluster_ranges) / 1000 if cluster_ranges else 0
            logger.warning(
                "diarization_minority_rescue: cluster %s was found by "
                "diarize (%d segments, %.1fs total) but has ZERO overlap "
                "with any ASR event — this speaker will be invisible "
                "unless you run /reprocess to re-chunk ASR.",
                miss_cid, len(cluster_ranges), total_s,
            )
            continue
        # Promote miss_cid to primary on best_ev. Existing primary (if
        # any) moves to secondary position.
        existing = best_ev.get("speakers") or []
        existing_primary = existing[0] if existing else None
        new_sp = [
            {
                "cluster_id": miss_cid,
                "identity": None,
                "identity_confidence": 1.0,
                "source": "diarization_minority_rescue",
                "display_name": None,
            }
        ]
        if existing_primary is not None:
            existing_primary = dict(existing_primary)
            existing_primary["source"] = "diarization_demoted"
            new_sp.append(existing_primary)
        # Preserve other secondaries from the original assignment.
        for sp_entry in existing[1:]:
            if sp_entry.get("cluster_id") not in (miss_cid,):
                new_sp.append(sp_entry)
        best_ev["speakers"] = new_sp
        if len(new_sp) > 1:
            best_ev["overlapping_speakers"] = True
        logger.info(
            "diarization_minority_rescue: cluster %s had no primary — "
            "promoted on event at %d-%dms (overlap=%dms)",
            miss_cid, best_ev.get("start_ms", 0), best_ev.get("end_ms", 0), best_ov,
        )


async def reprocess_meeting(
    meeting_dir: Path,
    asr_url: str = "http://localhost:8003",
    translate_url: str = "http://localhost:8010",
    diarize_url: str = "http://localhost:8001",
    language_pair: list[str] | tuple[str, ...] = ("en", "ja"),
    on_progress: Callable[..., Any] | None = None,
    expected_speakers: int | None = None,
) -> dict:
    """Fully reprocess a meeting from its raw audio recording.

    Parallelized pipeline:
    1. Back up journal, set state to "reprocessing"
    2. ASR: 4 concurrent workers process audio chunks
    3. Diarization: single call on the full audio (perfect cluster stability)
    4. Translation: 4 concurrent workers translate segments
    5. Attach speakers to events via time-range overlap
    6. Generate timeline, speaker data, summary
    7. Set state to "complete"
    """
    pcm_path = meeting_dir / "audio" / "recording.pcm"
    if not pcm_path.exists():
        return {"error": "No recording.pcm found"}

    journal_path = meeting_dir / "journal.jsonl"
    meta_path = meeting_dir / "meta.json"

    def progress(step, total, msg):
        logger.info("Reprocess [%d/%d]: %s", step, total, msg)
        if on_progress:
            on_progress(step, total, msg)

    # Structured phase-timing markers so the bench harness and log
    # parsers can extract wall time per pipeline phase. Format:
    #   reprocess_phase meeting=<id> phase=<name> wall_ms=<int>
    # Meeting id is emitted with every marker so concurrent reprocesses
    # (if they ever happen) can be disambiguated.
    phase_t0: dict[str, float] = {}

    def phase_start(name: str) -> None:
        phase_t0[name] = time.monotonic()

    def phase_end(name: str, extra: str = "") -> None:
        wall_ms = int((time.monotonic() - phase_t0.get(name, time.monotonic())) * 1000)
        tail = (" " + extra) if extra else ""
        logger.info(
            "reprocess_phase meeting=%s phase=%s wall_ms=%d%s",
            meeting_dir.name, name, wall_ms, tail,
        )

    total_start = time.monotonic()

    # 0. Set state + backup
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["state"] = "reprocessing"
        meta_path.write_text(json.dumps(meta, indent=2))

    if journal_path.exists():
        shutil.copy2(journal_path, meeting_dir / "journal.jsonl.bak")

    # Versioned snapshot of all derived artifacts (journal, summary,
    # timeline, speakers) BEFORE the new run overwrites them. The label
    # captures the inputs that drove the PRIOR run as best we can — for
    # the current run, the manifest stored separately on the new
    # snapshot will reflect the new inputs.
    try:
        from meeting_scribe.versions import snapshot_meeting

        snapshot_meeting(
            meeting_dir,
            label="pre-reprocess",
            inputs={
                "trigger": "reprocess_meeting",
                "asr_url": asr_url,
                "translate_url": translate_url,
                "diarize_url": diarize_url,
                "language_pair": list(language_pair),
                "expected_speakers": expected_speakers,
            },
        )
    except Exception:
        # Snapshot failure must never block reprocess — log and continue.
        logger.exception("Pre-reprocess snapshot failed; continuing anyway")

    # 1. Load audio
    phase_start("load_audio")
    pcm_data = pcm_path.read_bytes()
    total_samples = len(pcm_data) // BYTES_PER_SAMPLE
    duration_ms = int(total_samples / SAMPLE_RATE * 1000)
    phase_end("load_audio", f"audio_ms={duration_ms}")
    progress(1, 7, f"Audio: {duration_ms / 1000:.0f}s")

    # 2. Diarize FIRST — produces (start_ms, end_ms, cluster_id) tuples
    # that become the ASR chunk boundaries. This is the alignment fix:
    # transcript rows now snap to actual speaker-turn boundaries, so
    # clicking "7:42" in the timeline plays the audio the row is about.
    # Fixed-window chunking (the previous approach) had up to CHUNK_SECONDS
    # of misalignment because a sentence straddling a window boundary got
    # stamped with the window's [0, 4000] instead of the utterance's
    # real [3500, 5200].
    progress(2, 7, "Diarizing full audio...")
    phase_start("diarize")
    try:
        # When the caller pinned expected_speakers, hint pyannote at that
        # count AND tell the cross-chunk merger to force-absorb extras.
        if expected_speakers is not None and 1 <= expected_speakers <= 12:
            diar_min = expected_speakers
            diar_max = expected_speakers
        else:
            diar_min = 2
            diar_max = 10
        diarize_segments = await _diarize_full_audio(
            pcm_data, diarize_url,
            max_speakers=diar_max, min_speakers=diar_min,
            expected_speakers=expected_speakers,
        )
        logger.info("Full-audio diarization: %d segments", len(diarize_segments))
    except Exception as e:
        logger.warning("Diarization failed: %s", e)
        diarize_segments = []
    phase_end("diarize", f"segments={len(diarize_segments)}")

    # If diarize failed, fall back to fixed-window chunking so we still
    # produce SOME transcript. Better no-speaker-ids + rough alignment
    # than zero transcript.
    if not diarize_segments:
        logger.warning("No diarize segments — falling back to fixed-window chunks")
        asr_chunks: list[tuple[bytes, int, int, int | None]] = []
        for i in range(0, len(pcm_data), CHUNK_BYTES):
            chunk = pcm_data[i : i + CHUNK_BYTES]
            if len(chunk) < BYTES_PER_SAMPLE * 1000:
                continue
            start_ms = int(i // BYTES_PER_SAMPLE / SAMPLE_RATE * 1000)
            end_ms = int((i + len(chunk)) // BYTES_PER_SAMPLE / SAMPLE_RATE * 1000)
            asr_chunks.append((chunk, start_ms, end_ms, None))
    else:
        # Convert diarize segments → ASR chunks. Constraints:
        #   - Long segments (>15 s) get split into ≤10 s pieces so
        #     Qwen3-ASR stays within its buffer limits.
        #   - Tiny segments (<400 ms) are skipped — they're usually
        #     a "mhm" or mic pop that ASR will hallucinate on.
        #   - GAPS between diarize segments longer than 1 s get
        #     their own blind ASR chunks (cluster_id=None) so the
        #     timeline stays fully covered. Without this, any audio
        #     pyannote classified as "no speaker" disappears from
        #     the transcript even though the user can click the
        #     timeline there and hear content. The blind chunks'
        #     speakers are attributed by the orphan-reassignment
        #     step in _generate_speaker_data (nearest-in-time real
        #     speaker).
        MAX_CHUNK_MS = 10_000
        MIN_CHUNK_MS = 400
        GAP_FILL_THRESHOLD_MS = 1_000  # gaps larger than this get blind ASR
        asr_chunks = []

        def _slice_bytes(start_ms: int, end_ms: int) -> bytes:
            byte_s = int(start_ms / 1000 * SAMPLE_RATE) * BYTES_PER_SAMPLE
            byte_e = int(end_ms / 1000 * SAMPLE_RATE) * BYTES_PER_SAMPLE
            return pcm_data[byte_s:byte_e]

        def _append_pieces(start_ms: int, end_ms: int, cid: int | None) -> None:
            piece_start = start_ms
            while piece_start < end_ms:
                piece_end = min(piece_start + MAX_CHUNK_MS, end_ms)
                pcm_slice = _slice_bytes(piece_start, piece_end)
                if len(pcm_slice) >= BYTES_PER_SAMPLE * int(MIN_CHUNK_MS * SAMPLE_RATE / 1000):
                    asr_chunks.append((pcm_slice, piece_start, piece_end, cid))
                piece_start = piece_end

        # Sort diarize segments by start_ms so we can walk gaps between
        # consecutive segments. pyannote output order isn't guaranteed.
        sorted_diarize = sorted(diarize_segments, key=lambda s: s["start_ms"])

        # Head gap: audio before the first diarize segment.
        if sorted_diarize:
            first_start = sorted_diarize[0]["start_ms"]
            if first_start > GAP_FILL_THRESHOLD_MS:
                _append_pieces(0, first_start, None)
        else:
            _append_pieces(0, duration_ms, None)

        prev_end = 0
        for seg in sorted_diarize:
            seg_s = seg["start_ms"]
            seg_e = seg["end_ms"]
            # Gap between previous segment and this one
            if seg_s - prev_end >= GAP_FILL_THRESHOLD_MS and prev_end > 0:
                _append_pieces(prev_end, seg_s, None)
            # The diarize segment itself (split into pieces)
            if seg_e - seg_s >= MIN_CHUNK_MS:
                _append_pieces(seg_s, seg_e, seg.get("cluster_id"))
            prev_end = max(prev_end, seg_e)

        # Tail gap: audio after the last diarize segment.
        if sorted_diarize and duration_ms - prev_end >= GAP_FILL_THRESHOLD_MS:
            _append_pieces(prev_end, duration_ms, None)

        # Re-sort by time for deterministic processing order
        asr_chunks.sort(key=lambda c: c[1])
        gap_chunks = sum(1 for c in asr_chunks if c[3] is None)
        logger.info(
            "Built %d ASR chunks (%d diarize-aligned, %d gap-fill with cluster_id=None)",
            len(asr_chunks), len(asr_chunks) - gap_chunks, gap_chunks,
        )

    progress(3, 7, f"ASR: {len(asr_chunks)} diarize-aligned chunks ({CONCURRENCY} workers)")
    phase_start("asr")

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.get(f"{asr_url}/v1/models")
            asr_model = r.json()["data"][0]["id"]
        except Exception:
            asr_model = "Qwen/Qwen3-ASR-1.7B"

        semaphore = asyncio.Semaphore(CONCURRENCY)
        completed = 0

        async def process_chunk(chunk_data):
            nonlocal completed
            pcm, start_ms, end_ms, cluster_id = chunk_data
            async with semaphore:
                result = await _transcribe_chunk(
                    client, asr_url, asr_model, pcm, start_ms, end_ms,
                    language_pair=language_pair,
                )
                completed += 1
                if completed % 20 == 0:
                    progress(3, 7, f"ASR: {completed}/{len(asr_chunks)} chunks")
                if result is not None and cluster_id is not None:
                    # Stamp the diarize cluster directly on the event.
                    # Per-event identity / dominant-cluster naming is
                    # applied later by the correction preservation pass.
                    result["speakers"] = [
                        {
                            "cluster_id": cluster_id,
                            "identity": None,
                            "display_name": None,
                            "source": "diarization",
                            "identity_confidence": 1.0,
                        }
                    ]
                return result

        results = await asyncio.gather(*[process_chunk(c) for c in asr_chunks])
        events = [r for r in results if r is not None]
        events.sort(key=lambda e: e["start_ms"])

    events_with_speakers = sum(1 for e in events if e.get("speakers"))
    logger.info(
        "Diarize-aligned ASR: %d events (%d with speakers, %d clusters)",
        len(events),
        events_with_speakers,
        len({s["cluster_id"] for s in diarize_segments}) if diarize_segments else 0,
    )
    phase_end("asr", f"events={len(events)} speakers={events_with_speakers}")

    # Monolingual meetings skip the translation phase end-to-end: no
    # worker pool, no per-event translation call, no translation
    # entries written to the journal. The rest of the reprocess
    # pipeline (speaker alignment, summary, export) tolerates events
    # without a ``translation`` field.
    if len(language_pair) == 1:
        progress(4, 8, "Translating: skipped (monolingual meeting)")
        phase_start("translate")
        phase_end("translate", "events=0 skipped=monolingual")
    else:
        progress(4, 8, f"Translating ({CONCURRENCY} workers)...")
        phase_start("translate")

        # 3. Translation — parallel workers
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                r = await client.get(f"{translate_url}/v1/models")
                trans_model = r.json()["data"][0]["id"]
            except Exception:
                trans_model = "default"

            semaphore = asyncio.Semaphore(CONCURRENCY)
            completed = 0

            async def translate_one(event):
                nonlocal completed
                async with semaphore:
                    await _translate_event(
                        client, translate_url, trans_model, event, language_pair
                    )
                    completed += 1
                    if completed % 20 == 0:
                        progress(3, 7, f"Translated {completed}/{len(events)}")

            await asyncio.gather(*[translate_one(e) for e in events])
        phase_end("translate", f"events={len(events)}")

    progress(4, 7, f"Writing {len(events)} events...")
    phase_start("preserve_corrections")

    # 3.5 Preserve user speaker_correction entries from the OLD journal.
    #
    # Reprocess regenerates segment_ids, so corrections keyed on old
    # segment_ids would be orphaned and the user's name mappings would
    # silently disappear. Recovery strategy:
    #
    #   1. For each NEW event, find the OLD named range with the
    #      tightest time overlap. Each event gets its own "best name"
    #      (or None if nothing overlaps strongly enough).
    #   2. Per-cluster, compute the DOMINANT best-name by event count.
    #      That's what detected_speakers.json shows in the participant
    #      list (e.g. cluster 3 → "Danny" because 800 of its events
    #      matched Danny best).
    #   3. Emit ONE cluster-level correction per cluster, targeting the
    #      first event in the cluster whose best-name equals the
    #      dominant. This pins the cluster display_name in the sidebar.
    #   4. Emit a PER-EVENT correction for every other event whose
    #      best-name differs from its cluster's dominant. These are
    #      individual segments where the user labeled a minority speaker
    #      who happens to share a diarization cluster with the dominant
    #      voice — e.g. Joel inside cluster 3 (Danny). The transcript
    #      row for that event will display "Joel", while the cluster
    #      color and participant list stay as Danny. Correct per-row,
    #      correct per-cluster, no voice fighting pyannote.
    preserved_corrections: list[dict] = []
    bak_path = meeting_dir / "journal.jsonl.bak"
    if bak_path.exists():
        old_segments_by_id: dict[str, dict] = {}
        old_corrections: list[dict] = []
        with bak_path.open() as bf:
            for line in bf:
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                if evt.get("type") == "speaker_correction":
                    old_corrections.append(evt)
                elif evt.get("is_final") and evt.get("text"):
                    sid = evt.get("segment_id")
                    if sid:
                        old_segments_by_id[sid] = evt

        # Build a (start_ms, end_ms, name) list from old corrections.
        old_named_ranges: list[tuple[int, int, str]] = []
        for corr in old_corrections:
            old_sid = corr.get("segment_id") or ""
            old_seg = old_segments_by_id.get(old_sid)
            if not old_seg:
                continue
            name = corr.get("speaker_name") or ""
            if not name:
                continue
            old_named_ranges.append(
                (old_seg.get("start_ms", 0), old_seg.get("end_ms", 0), name)
            )

        # Step 1 — find best-name per new event.
        from collections import Counter
        event_best_name: dict[str, str] = {}  # segment_id → name
        event_cluster: dict[str, int] = {}    # segment_id → cluster_id
        for new_e in events:
            new_start = new_e.get("start_ms", 0)
            new_end = new_e.get("end_ms", new_start)
            new_dur = max(1, new_end - new_start)
            sid = new_e.get("segment_id")
            if not sid:
                continue
            sp_list = new_e.get("speakers") or []
            if not sp_list:
                continue
            cid = sp_list[0].get("cluster_id")
            if cid is None:
                continue
            event_cluster[sid] = cid

            # Match by best overlap. Old thresholds (≥200 ms AND ≥50 %
            # of new_dur) were tuned for the 4 s fixed-chunk pipeline
            # where new events were always ~4 s. Diarize-aligned chunks
            # can span a 10 s speaker turn, and an old 4 s correction
            # only covers 40 % of that — the old 50 % threshold silently
            # dropped the match. Now: require ≥200 ms AND (old covers
            # ≥30 % of new OR new covers ≥30 % of old). This accepts
            # partial overlaps in BOTH directions, which is correct when
            # the two chunk granularities differ.
            best_name = None
            best_overlap = 0
            for old_start, old_end, name in old_named_ranges:
                overlap = max(0, min(new_end, old_end) - max(new_start, old_start))
                if overlap < 200:
                    continue
                old_dur = max(1, old_end - old_start)
                if overlap < 0.3 * new_dur and overlap < 0.3 * old_dur:
                    continue
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = name
            if best_name is not None:
                event_best_name[sid] = best_name

        # Step 2 — dominant name per cluster by event count.
        cluster_votes: dict[int, Counter] = {}
        for sid, name in event_best_name.items():
            cid = event_cluster.get(sid)
            if cid is None:
                continue
            cluster_votes.setdefault(cid, Counter())[name] += 1
        dominant_by_cluster: dict[int, str] = {
            cid: votes.most_common(1)[0][0]
            for cid, votes in cluster_votes.items()
            if votes
        }

        # Step 3 — pin cluster display_name. Target the first event
        # (time-sorted) whose best-name equals the dominant, so we never
        # collide with a per-event override on that segment.
        cluster_anchor: dict[int, str] = {}
        for evt in sorted(events, key=lambda x: x.get("start_ms", 0)):
            sid = evt.get("segment_id")
            if not sid:
                continue
            cid = event_cluster.get(sid)
            if cid not in dominant_by_cluster:
                continue
            if cid in cluster_anchor:
                continue
            if event_best_name.get(sid) == dominant_by_cluster[cid]:
                cluster_anchor[cid] = sid
        # Fallback anchor: first event in the cluster (any best-name)
        # so clusters without a matching event still get their name.
        for evt in sorted(events, key=lambda x: x.get("start_ms", 0)):
            sid = evt.get("segment_id")
            if not sid:
                continue
            cid = event_cluster.get(sid)
            if cid not in dominant_by_cluster or cid in cluster_anchor:
                continue
            cluster_anchor[cid] = sid

        for cid, name in dominant_by_cluster.items():
            sid = cluster_anchor.get(cid)
            if not sid:
                continue
            preserved_corrections.append({
                "type": "speaker_correction",
                "segment_id": sid,
                "speaker_name": name,
                "preserved_from_reprocess": True,
                "scope": "cluster",
            })

        # Step 4 — per-event override for events whose best-name differs
        # from the cluster's dominant. Applies to events where the user
        # manually labeled a minority speaker who shares a cluster with
        # the dominant voice.
        per_event_count = 0
        for sid, name in event_best_name.items():
            cid = event_cluster.get(sid)
            if cid is None:
                continue
            dominant = dominant_by_cluster.get(cid)
            if name == dominant:
                continue  # matches cluster default, no override needed
            # Don't collide with the cluster-anchor correction on the
            # same segment (shouldn't happen because cluster anchor is
            # chosen to match dominant, but be safe).
            if cluster_anchor.get(cid) == sid:
                continue
            preserved_corrections.append({
                "type": "speaker_correction",
                "segment_id": sid,
                "speaker_name": name,
                "preserved_from_reprocess": True,
                "scope": "event",
            })
            per_event_count += 1

        unique_names = {c["speaker_name"] for c in preserved_corrections}
        logger.info(
            "Reprocess preserved names: %d unique across %d cluster(s) + "
            "%d per-event overrides (from %d original corrections)",
            len(unique_names),
            len(dominant_by_cluster),
            per_event_count,
            len(old_corrections),
        )

    # 4. Write journal — events first, then preserved corrections so
    # _generate_speaker_data picks them up on the next pass.
    with open(journal_path, "w") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        for corr in preserved_corrections:
            f.write(json.dumps(corr, ensure_ascii=False) + "\n")

    phase_end("preserve_corrections", f"events={len(events)} corrections={len(preserved_corrections)}")

    # 5. Generate timeline + speaker data via the SAME path server.py's
    # live finalize uses. This preserves seq_index assignment, orphan
    # reassignment, and journal remapping → every finalize entry point
    # (live stop, refinalize, full-reprocess) produces identical shape
    # under `detected_speakers.json` / `speaker_lanes.json` /
    # `timeline.json`, so the UI renders consistently regardless of
    # which path got the meeting to complete state.
    progress(5, 7, "Generating timeline and speaker data...")
    phase_start("speaker_data")
    import importlib
    import json as _json
    srv = importlib.import_module("meeting_scribe.server")
    try:
        srv._generate_speaker_data(
            meeting_dir, journal_path, _json,
            expected_speakers=expected_speakers,
        )
        srv._generate_timeline(meeting_dir.name, meeting_dir=meeting_dir)
    except Exception as e:
        logger.warning("reprocess: _generate_speaker_data/_generate_timeline failed: %s", e)
    phase_end("speaker_data")

    # speaker_stats for the return summary (re-load after generation)
    try:
        speaker_stats = {
            s["cluster_id"]: s
            for s in _json.loads((meeting_dir / "detected_speakers.json").read_text())
        }
    except Exception:
        speaker_stats = {}

    # 6. Summary (LLM call)
    progress(6, 7, "Generating summary...")
    phase_start("summary")
    summary = {}
    try:
        from meeting_scribe.summary import generate_summary

        summary = await generate_summary(meeting_dir, vllm_url=translate_url)
    except Exception as e:
        logger.warning("Summary failed: %s", e)
        summary = {"error": str(e)}
    phase_end("summary", f"ok={'error' not in summary}")

    # 7. Finalize state
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["state"] = "complete"
        meta_path.write_text(json.dumps(meta, indent=2))

    progress(7, 7, "Complete")

    total_wall_ms = int((time.monotonic() - total_start) * 1000)
    logger.info(
        "reprocess_phase meeting=%s phase=TOTAL wall_ms=%d audio_ms=%d events=%d speakers=%d",
        meeting_dir.name, total_wall_ms, duration_ms, len(events), len(speaker_stats),
    )

    return {
        "segments": len(events),
        "duration_ms": duration_ms,
        "translated": sum(1 for e in events if (e.get("translation") or {}).get("text")),
        "speakers": len(speaker_stats),
        "has_summary": "error" not in summary,
    }
