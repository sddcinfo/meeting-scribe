"""Neural speaker diarization backend (GB10 production).

Talks to a diarization container (pyannote.audio) via HTTP for real
speaker diarization. Returns cluster IDs and embeddings; the server-owned
SpeakerVerifier maps clusters to enrolled identities.

Architecture:
    Audio chunks → DiarizeBackend → POST to pyannote container (/v1/diarize)
    Container returns: speaker segments with cluster_id + embedding
    Server passes (cluster_id, embedding) to SpeakerVerifier → identity

Time alignment:
    Diarization results are cached by time range. The server queries
    by TranscriptEvent.start_ms / end_ms to align ASR ↔ diarization,
    since diarization has independent latency from ASR.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING

import httpx
import numpy as np

from meeting_scribe.backends.base import DiarizeBackend
from meeting_scribe.models import SpeakerAttribution
from meeting_scribe.runtime import state

if TYPE_CHECKING:
    from meeting_scribe.speaker.verification import SpeakerVerifier

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Cache diarization results for this many seconds before eviction
# Cache TTL: how long diarization results stay available for the catch-up
# loop in server.py to retroactively attribute speakers to ASR events that
# arrived before the diarization batch finished. 120s gives the catch-up
# loop plenty of time to walk pending events; GB10 has tons of RAM so the
# memory cost is trivial.
_CACHE_TTL_SECONDS = 120.0
# Maximum cached results (prevent unbounded memory)
_CACHE_MAX_SIZE = 500


class DiarizationResult:
    """A single diarization result from the diarization container."""

    __slots__ = ("cluster_id", "confidence", "embedding", "end_ms", "start_ms", "timestamp")

    def __init__(
        self,
        start_ms: int,
        end_ms: int,
        cluster_id: int,
        embedding: np.ndarray | None = None,
        confidence: float = 1.0,
    ) -> None:
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.cluster_id = cluster_id
        self.embedding = embedding
        self.confidence = confidence
        self.timestamp = time.monotonic()


class SortformerBackend(DiarizeBackend):
    """Neural speaker diarization via pyannote.audio.

    Runs against a diarization container for unsupervised
    speaker clustering. The backend accepts injected SpeakerVerifier
    for identity resolution (not self-created).

    Usage:
        verifier = SpeakerVerifier(enrollment_store)
        backend = SortformerBackend(url="http://gb10:8001", verifier=verifier)
        await backend.start()

        # During meeting:
        attributions = await backend.process_audio(audio_chunk, sample_offset)
        # attributions have cluster_id, verifier resolves identity

        # Query by time range (for ASR event alignment):
        results = backend.get_results_for_range(start_ms, end_ms)
    """

    def __init__(
        self,
        url: str = "http://localhost:8001",
        verifier: SpeakerVerifier | None = None,
        flush_interval_seconds: float = 4.0,
        window_seconds: float = 16.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._verifier = verifier
        self._client: httpx.AsyncClient | None = None
        self._max_speakers = 4

        # Rolling-window diarization.
        #
        # Why rolling: pyannote needs ~15s+ of multi-speaker context to
        # actually separate speakers. On 4s chunks it always returns ONE
        # cluster (whoever is talking most), so the live path used to
        # collapse every utterance into global cluster #1. See the
        # "Diar merge: local 0 → global #1" log runs from 2026-04-10.
        #
        # We now keep a rolling buffer of the last `window_seconds` of
        # audio, and every `flush_interval_seconds` we diarize the ENTIRE
        # window (not just the newest slice). That gives pyannote enough
        # context to find multiple clusters in one pass. To avoid
        # re-emitting the same segments, we only cache segments whose
        # end time is past the previously emitted horizon.
        self._rolling_audio: list[np.ndarray] = []
        self._rolling_samples = 0  # total samples currently in rolling buffer
        self._rolling_start_sample = 0  # absolute sample index where buffer starts
        self._samples_since_flush = 0
        self._window_max_samples = int(window_seconds * SAMPLE_RATE)
        self._flush_interval_samples = int(flush_interval_seconds * SAMPLE_RATE)
        # Minimum buffer size before we bother calling pyannote. Below this,
        # pyannote will only ever see a single speaker, which is worse than
        # skipping the call and letting the catch-up loop retroactively
        # attribute once enough context has accumulated.
        self._min_diarize_samples = int(6.0 * SAMPLE_RATE)
        # Meeting-absolute end time (ms) of the last segment we emitted.
        # New segments are clipped to [_last_emitted_end_ms, window_end_ms]
        # so each moment of the meeting is attributed exactly once.
        self._last_emitted_end_ms = 0

        # Time-aligned result cache (ordered by insertion time)
        self._result_cache: OrderedDict[str, DiarizationResult] = OrderedDict()

        # Global cluster identity tracking — pyannote returns cluster IDs
        # that are only meaningful WITHIN a single inference call. "Cluster 0"
        # in buffer 1 is NOT necessarily the same person as "cluster 0" in
        # buffer 2. We maintain a centroid per global cluster ID, and on
        # every new diarization result we map the local cluster to the
        # closest global centroid (or create a new one). This gives us
        # stable speaker identity across the whole meeting.
        self._global_centroids: dict[int, np.ndarray] = {}  # global_id → normalized embedding
        self._global_centroid_counts: dict[int, int] = {}  # weight for running avg
        self._next_global_id: int = 1  # start at 1 so 0 = "unknown"
        # Centroid-allocation timestamps, used by the periodic consolidation
        # pass + threshold annealing so we can distinguish "just created,
        # centroid still noisy" from "old, stable".
        self._global_centroid_created_at: dict[int, float] = {}
        # Starting cosine similarity threshold. After ~2 min of meeting
        # audio the embeddings per speaker have enough observations that we
        # can relax the threshold without mis-merging. See `_current_threshold`.
        self._cluster_merge_threshold_initial: float = 0.55
        self._cluster_merge_threshold_relaxed: float = 0.45
        self._cluster_merge_threshold: float = self._cluster_merge_threshold_initial
        # Required gap between best match and second-best match. Was 0.08 —
        # in live meetings with 3+ speakers present, the embedding space
        # gets crowded and top-2 can easily be within 0.05 of each other,
        # which was causing same-speaker fragments to spawn new clusters
        # (observed on 2026-04-13 meeting 18984813 where #4 and #5 had
        # best_score 0.65-0.67 but still allocated new ids). 0.03 keeps
        # the ambiguity guard without being pathologically strict.
        self._cluster_margin: float = 0.03
        # Score floor above which we trust best-match even if margin is
        # tight. Was 0.70 — lowered to 0.60 so a 0.65 best-match with a
        # 0.62 second-match (pathological but seen in practice) still
        # merges instead of fragmenting.
        self._cluster_score_bypass: float = 0.60
        # Consolidation pass state — when was the last time we walked the
        # centroid list to merge any that have drifted close together.
        self._last_consolidation_ts: float = 0.0
        self._consolidation_interval_s: float = 30.0
        # 0.85 was too tight to fold back fragments in practice — observed
        # same-speaker fragments sitting at 0.65-0.75 cosine. 0.72 catches
        # the typical "same voice, different phrase / mic angle" drift
        # without mis-merging genuinely distinct voices (pyannote
        # between-speaker scores on meeting audio rarely exceed 0.55).
        self._consolidation_cos_threshold: float = 0.72
        # `_cluster_rename` records id remaps so external consumers (e.g.
        # server.py's catch-up loop) can rewrite stale references on the
        # next pass. Keyed by the deleted id → the surviving id.
        self._cluster_rename: dict[int, int] = {}
        # Timestamp of the very first _assign_global_cluster call — used for
        # the annealing window.
        self._session_start_ts: float | None = None
        self._anneal_window_s: float = 120.0
        # Debug tracking: last call's local→global mapping for logging
        self._last_mapping: dict[int, int] = {}

        # Failure tracking — after N consecutive 500s the backend reports
        # as degraded so /api/status surfaces the truth.
        self._consecutive_failures = 0
        self._last_error: str | None = None
        self._degraded = False

    def _assign_global_cluster(
        self,
        local_id: int,
        embedding: np.ndarray | None,
    ) -> int:
        """Map a pyannote per-call cluster ID to a stable global cluster ID.

        Strategy:
        1. L2-normalize the embedding (cosine similarity = dot product)
        2. Score against every known centroid
        3. If best score >= threshold AND there's a clear margin over 2nd place,
           merge. Otherwise allocate a new global ID.
        4. Centroid is the running average of ALL samples assigned to it,
           re-normalized to stay on the unit sphere.

        This design re-identifies returning speakers even after long gaps,
        because the centroid is built from all their past utterances.
        """
        if embedding is None:
            return int(local_id) if local_id is not None else 0

        # L2-normalize the input embedding once
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-9:
            return 0  # all-zero embedding, garbage
        emb_unit = embedding / norm

        # Anneal merge threshold over the session. For the first
        # `_anneal_window_s` we keep the strict initial threshold so we
        # don't collapse two legitimately-different speakers when we've
        # only seen 1-2 utterances from each. After that window we relax
        # to let legit same-speaker drift (mic motion, emotion, different
        # phrases) re-merge without allocating a new global id every time.
        now_ts = time.monotonic()
        if self._session_start_ts is None:
            self._session_start_ts = now_ts
        age = now_ts - self._session_start_ts
        if age >= self._anneal_window_s:
            self._cluster_merge_threshold = self._cluster_merge_threshold_relaxed
        else:
            # Linear interpolation over the window so the change is smooth.
            frac = max(0.0, min(1.0, age / self._anneal_window_s))
            self._cluster_merge_threshold = (
                self._cluster_merge_threshold_initial * (1.0 - frac)
                + self._cluster_merge_threshold_relaxed * frac
            )

        # Periodic centroid consolidation: merge any pair of existing
        # global centroids whose cosine ≥ `_consolidation_cos_threshold`.
        # This fixes the common "two speakers got split into N clusters
        # early when data was sparse, now they've drifted close enough to
        # deserve merging" case without waiting for the session to end.
        if now_ts - self._last_consolidation_ts > self._consolidation_interval_s:
            self._last_consolidation_ts = now_ts
            self._consolidate_centroids()

        # Score against every known centroid (they're already normalized)
        scores: list[tuple[int, float]] = []
        for gid, centroid in self._global_centroids.items():
            score = float(np.dot(emb_unit, centroid))
            scores.append((gid, score))
        scores.sort(key=lambda x: -x[1])

        best_global_id = scores[0][0] if scores else None
        best_score = scores[0][1] if scores else -1.0
        second_score = scores[1][1] if len(scores) >= 2 else -1.0
        margin = best_score - second_score

        if (
            best_global_id is not None
            and best_score >= self._cluster_merge_threshold
            # If we have ambiguity between top-2, require a stricter score
            and (margin >= self._cluster_margin or best_score >= self._cluster_score_bypass)
        ):
            # Merge into existing global cluster — update centroid via
            # weighted running average, then re-normalize.
            count = self._global_centroid_counts[best_global_id]
            old = self._global_centroids[best_global_id]
            # Cap weight at 20 samples so the centroid stays adaptable to
            # gradual voice drift (different phrases, mic position, etc.)
            effective_count = min(count, 20)
            new_centroid = (old * effective_count + emb_unit) / (effective_count + 1)
            new_centroid /= max(float(np.linalg.norm(new_centroid)), 1e-9)
            self._global_centroids[best_global_id] = new_centroid
            self._global_centroid_counts[best_global_id] = count + 1
            self._last_mapping[local_id] = best_global_id
            logger.warning(
                "Diar merge: local %d → global #%d (score=%.2f margin=%.2f, #obs=%d)",
                local_id,
                best_global_id,
                best_score,
                margin,
                self._global_centroid_counts[best_global_id],
            )
            return best_global_id

        # No match — allocate a new global ID
        gid = self._next_global_id
        self._next_global_id += 1
        self._global_centroids[gid] = emb_unit.copy()
        self._global_centroid_counts[gid] = 1
        self._global_centroid_created_at[gid] = now_ts
        self._last_mapping[local_id] = gid
        logger.warning(
            "Diar NEW: global speaker #%d (best existing match=%.2f threshold=%.2f, #known=%d)",
            gid,
            best_score,
            self._cluster_merge_threshold,
            len(self._global_centroids),
        )
        return gid

    def _consolidate_centroids(self) -> None:
        """Merge any pair of global centroids whose cosine >= threshold.

        Runs every ``_consolidation_interval_s`` seconds from inside
        ``_assign_global_cluster``. Greedy: sort pairs by descending
        similarity, merge the highest-scoring pair that still clears the
        threshold, repeat until no pair does. The surviving id is the one
        with more observations (higher count). Any renames are recorded
        in ``_cluster_rename`` so ``_result_cache`` + external consumers
        can rewrite stale references on the next pass.
        """
        if len(self._global_centroids) < 2:
            return
        renamed = 0
        # One sweep: find the closest pair above threshold, merge, repeat.
        while True:
            gids = list(self._global_centroids.keys())
            best_pair: tuple[int, int] | None = None
            best_score = -1.0
            for i in range(len(gids)):
                for j in range(i + 1, len(gids)):
                    a = self._global_centroids[gids[i]]
                    b = self._global_centroids[gids[j]]
                    score = float(np.dot(a, b))
                    if score > best_score:
                        best_score = score
                        best_pair = (gids[i], gids[j])
            if not best_pair or best_score < self._consolidation_cos_threshold:
                break
            a_id, b_id = best_pair
            # Keep the centroid with more observations as the survivor so
            # its history dominates the merged centroid.
            if self._global_centroid_counts[a_id] >= self._global_centroid_counts[b_id]:
                survivor, retired = a_id, b_id
            else:
                survivor, retired = b_id, a_id
            count_s = self._global_centroid_counts[survivor]
            count_r = self._global_centroid_counts[retired]
            c_s = self._global_centroids[survivor]
            c_r = self._global_centroids[retired]
            merged = (c_s * count_s + c_r * count_r) / max(float(count_s + count_r), 1.0)
            merged /= max(float(np.linalg.norm(merged)), 1e-9)
            self._global_centroids[survivor] = merged
            self._global_centroid_counts[survivor] = count_s + count_r
            # Retire the loser.
            del self._global_centroids[retired]
            self._global_centroid_counts.pop(retired, None)
            self._global_centroid_created_at.pop(retired, None)
            # Transitive rename: if something previously mapped to `retired`,
            # now map it to `survivor`.
            for k, v in list(self._cluster_rename.items()):
                if v == retired:
                    self._cluster_rename[k] = survivor
            self._cluster_rename[retired] = survivor
            # Update last-mapping too so the next incoming local id uses
            # the surviving global id.
            for lk, lv in list(self._last_mapping.items()):
                if lv == retired:
                    self._last_mapping[lk] = survivor
            renamed += 1
            logger.warning(
                "Diar consolidate: merged global #%d → #%d (cos=%.2f, obs %d+%d)",
                retired,
                survivor,
                best_score,
                count_s,
                count_r,
            )
        if renamed:
            logger.info(
                "Diar consolidation pass merged %d centroid(s); now tracking %d speakers",
                renamed,
                len(self._global_centroids),
            )

    # Match the Qwen3TTSBackend interface used by server.py health reporting
    MAX_CONSECUTIVE_FAILURES = 3

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def available(self) -> bool:
        return self._client is not None and not self._degraded

    async def check_health(self) -> bool:
        """Deep health probe — re-queries /health and clears degraded if OK.

        Called by the retry loop in server.py after a container restart.
        """
        if self._client is None:
            return False
        try:
            r = await self._client.get(f"{self._url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                if self._degraded:
                    logger.info("Diarization recovered from degraded state")
                self._degraded = False
                self._consecutive_failures = 0
                self._last_error = None
                return True
        except Exception as e:
            self._last_error = str(e)
        return False

    @property
    def verifier(self) -> SpeakerVerifier | None:
        return self._verifier

    @verifier.setter
    def verifier(self, v: SpeakerVerifier) -> None:
        self._verifier = v

    async def start(self, max_speakers: int = 4) -> None:
        """Connect to diarization container and verify health."""
        self._max_speakers = max_speakers
        self._client = httpx.AsyncClient(timeout=30.0)

        # Health check with retry
        for attempt in range(3):
            try:
                resp = await self._client.get(f"{self._url}/health")
                if resp.status_code == 200:
                    logger.info(
                        "Sortformer connected at %s (max_speakers=%d)",
                        self._url,
                        max_speakers,
                    )
                    return
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.warning("Sortformer health check attempt %d: %s", attempt + 1, e)
                if attempt < 2:
                    import asyncio

                    await asyncio.sleep(2)

        msg = f"Sortformer container not responding at {self._url}"
        raise ConnectionError(msg)

    async def stop(self) -> None:
        """Release resources and reset global cluster state."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._rolling_audio.clear()
        self._rolling_samples = 0
        self._rolling_start_sample = 0
        self._samples_since_flush = 0
        self._last_emitted_end_ms = 0
        self._result_cache.clear()
        # Reset global cluster tracking — each meeting should start fresh.
        self._global_centroids.clear()
        self._global_centroid_counts.clear()
        self._next_global_id = 1
        self._last_mapping.clear()

    async def process_audio(
        self,
        audio: np.ndarray,
        sample_offset: int,
        sample_rate: int = 16000,
    ) -> list[SpeakerAttribution]:
        """Append audio to the rolling buffer; flush and diarize periodically.

        Each flush sends the ENTIRE rolling window (up to ``window_seconds``)
        to pyannote so it has enough multi-speaker context to actually
        separate voices. Only segments past the previously-emitted horizon
        are cached and returned, so each moment is attributed exactly once.
        """
        if self._client is None:
            return [SpeakerAttribution(cluster_id=0, source="unknown")]

        # Append new audio and trim the front of the buffer so it never
        # exceeds the configured window.
        self._rolling_audio.append(audio)
        self._rolling_samples += len(audio)
        self._samples_since_flush += len(audio)
        while self._rolling_samples > self._window_max_samples:
            front = self._rolling_audio[0]
            excess = self._rolling_samples - self._window_max_samples
            if len(front) <= excess:
                self._rolling_audio.pop(0)
                self._rolling_samples -= len(front)
                self._rolling_start_sample += len(front)
            else:
                self._rolling_audio[0] = front[excess:]
                self._rolling_samples -= excess
                self._rolling_start_sample += excess

        # Flush conditions: enough NEW audio since last call AND enough
        # total context for pyannote to separate speakers.
        if self._samples_since_flush < self._flush_interval_samples:
            return []
        if self._rolling_samples < self._min_diarize_samples:
            return []

        self._samples_since_flush = 0
        combined = np.concatenate(self._rolling_audio)
        window_start_ms = int(self._rolling_start_sample / sample_rate * 1000)
        window_end_ms = window_start_ms + int(len(combined) / sample_rate * 1000)
        window_duration_s = len(combined) / sample_rate

        # The pyannote container auto-disables the min_speakers hint for
        # windows < 15s. Once we're past that, min_speakers=2 stops it
        # collapsing ambiguous clustering into a single speaker — which is
        # exactly the bug that used to plague the live path.
        min_speakers_hdr = "2" if window_duration_s >= 15.0 else "0"

        _rtt_t0 = time.monotonic()
        try:
            pcm_s16 = (combined * 32768).astype(np.int16).tobytes()
            resp = await self._client.post(
                f"{self._url}/v1/diarize",
                content=pcm_s16,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Sample-Rate": str(sample_rate),
                    "X-Max-Speakers": str(self._max_speakers),
                    "X-Min-Speakers": min_speakers_hdr,
                },
            )
            resp.raise_for_status()
            # W5: backend RTT histogram (successful requests only).
            try:
                state.metrics.diarize_request_rtt_ms.append((time.monotonic() - _rtt_t0) * 1000)
            except AttributeError:
                pass  # state.metrics not yet initialised at warmup time
            result = resp.json()
            if self._consecutive_failures > 0:
                logger.info(
                    "Sortformer diarization recovered after %d failures",
                    self._consecutive_failures,
                )
            self._consecutive_failures = 0
            self._last_error = None
        except (httpx.HTTPError, ValueError) as e:
            self._consecutive_failures += 1
            self._last_error = str(e)
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                self._degraded = True
                logger.error(
                    "Sortformer diarization marked degraded after %d failures: %s",
                    self._consecutive_failures,
                    e,
                )
            else:
                logger.warning(
                    "Sortformer diarization failed (%d/%d): %s",
                    self._consecutive_failures,
                    self.MAX_CONSECUTIVE_FAILURES,
                    e,
                )
            return [SpeakerAttribution(cluster_id=0, source="unknown")]

        # Parse segments, translate their window-local times into
        # meeting-absolute times, and keep only the NEW portion past the
        # last-emitted horizon. Clipping (rather than skipping) lets a
        # long utterance that straddled two windows still get attributed
        # for its newly-visible tail.
        segments = result.get("segments", [])
        attributions: list[SpeakerAttribution] = []
        new_segment_count = 0

        for seg in segments:
            local_cluster_id = int(seg.get("speaker_id", 0))
            embedding = None
            if "embedding" in seg:
                embedding = np.array(seg["embedding"], dtype=np.float32)

            seg_start_ms = window_start_ms + int(seg.get("start", 0) * 1000)
            seg_end_ms = window_start_ms + int(seg.get("end", 0) * 1000)

            # Clip to the un-emitted portion
            if seg_end_ms <= self._last_emitted_end_ms:
                continue
            if seg_start_ms < self._last_emitted_end_ms:
                seg_start_ms = self._last_emitted_end_ms
            if seg_end_ms <= seg_start_ms:
                continue

            # Map pyannote's per-call cluster id to a stable global id
            # via embedding centroid matching. Now that the window is long
            # enough for pyannote to return multiple clusters at once, this
            # actually produces distinct global ids for distinct speakers.
            global_cluster_id = self._assign_global_cluster(
                local_cluster_id,
                embedding,
            )

            cache_key = f"{seg_start_ms}-{seg_end_ms}-{global_cluster_id}"
            dr = DiarizationResult(
                start_ms=seg_start_ms,
                end_ms=seg_end_ms,
                cluster_id=global_cluster_id,
                embedding=embedding,
                confidence=seg.get("confidence", 1.0),
            )
            self._result_cache[cache_key] = dr
            new_segment_count += 1

            if self._verifier and embedding is not None:
                attr = self._verifier.verify(global_cluster_id, embedding)
            else:
                attr = SpeakerAttribution(
                    cluster_id=global_cluster_id,
                    identity_confidence=seg.get("confidence", 1.0),
                    source="cluster_only",
                )
            attributions.append(attr)

        # Advance the emission horizon past this window.
        if window_end_ms > self._last_emitted_end_ms:
            self._last_emitted_end_ms = window_end_ms

        if segments:
            logger.info(
                "Diarized rolling window %.1fs [%.1fs→%.1fs] → %d raw segs, "
                "%d new, %d global speakers (min_speakers=%s)",
                window_duration_s,
                window_start_ms / 1000.0,
                window_end_ms / 1000.0,
                len(segments),
                new_segment_count,
                len(self._global_centroids),
                min_speakers_hdr,
            )

        self._evict_cache()

        return (
            attributions if attributions else [SpeakerAttribution(cluster_id=0, source="unknown")]
        )

    async def enroll_speaker(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """Enroll a speaker via the diarization container's embedding extraction.

        Delegates ECAPA-TDNN embedding extraction to the GPU-side container.
        """
        if self._client is None:
            msg = "Sortformer backend not started"
            raise RuntimeError(msg)

        pcm_s16 = (audio * 32768).astype(np.int16).tobytes()

        try:
            resp = await self._client.post(
                f"{self._url}/v1/embed",
                content=pcm_s16,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Sample-Rate": str(sample_rate),
                    "X-Speaker-Name": name,
                },
            )
            resp.raise_for_status()
            result = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            msg = f"Speaker enrollment failed: {e}"
            raise RuntimeError(msg) from e

        embedding = np.array(result["embedding"], dtype=np.float32)
        enrollment_id = result.get("enrollment_id", str(uuid.uuid4()))

        # Register with verifier's enrollment store
        if self._verifier:
            from meeting_scribe.speaker.enrollment import EnrolledSpeaker

            speaker = EnrolledSpeaker(
                name=name,
                embedding=embedding,
                enrollment_id=enrollment_id,
                audio_duration_seconds=len(audio) / sample_rate,
            )
            self._verifier._store.add(speaker)

        logger.info("Enrolled speaker via Sortformer: %s (id=%s)", name, enrollment_id)
        return enrollment_id

    def get_results_for_range(
        self,
        start_ms: int,
        end_ms: int,
    ) -> list[DiarizationResult]:
        """Look up cached diarization results overlapping a time range.

        Used by server.py to align ASR events with diarization results.
        Sortformer and ASR run independently; this bridges the timing gap.
        """
        results = []
        for dr in self._result_cache.values():
            # Check for time overlap
            if dr.start_ms < end_ms and dr.end_ms > start_ms:
                results.append(dr)
        return results

    def _evict_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.monotonic()
        while self._result_cache:
            key, dr = next(iter(self._result_cache.items()))
            if now - dr.timestamp > _CACHE_TTL_SECONDS:
                del self._result_cache[key]
            else:
                break

        # Hard limit
        while len(self._result_cache) > _CACHE_MAX_SIZE:
            self._result_cache.popitem(last=False)
