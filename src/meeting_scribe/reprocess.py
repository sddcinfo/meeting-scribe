"""Full meeting reprocessing — re-run ASR + translation on original audio.

Reads the raw PCM recording, transcribes with Qwen3-ASR, translates,
and regenerates all meeting artifacts (journal, timeline, speakers, summary).

Parallelized: ASR and translation run with concurrent workers to leverage
vLLM's batching (4+ concurrent requests = ~3x throughput).

Usage:
    from meeting_scribe.reprocess import reprocess_meeting
    result = await reprocess_meeting(meeting_dir, asr_url, translate_url)

ASR / translate seam — DELIBERATELY NOT shared with the live path
-----------------------------------------------------------------
``_transcribe_chunk`` and ``_translate_event`` look like duplicates of
``backends.asr_vllm`` and ``backends.translate_vllm`` but they are not
unifiable. The two paths solve different problems:

* **Live ASR** (``backends/asr_vllm.py``) is stateful and incremental.
  It buffers ~3.5 s of audio between ASR calls, builds a system prompt
  naming the meeting's expected languages (kills Germanic→English bias
  on the live buffer length), and runs at vLLM ``priority=-20`` so it
  preempts every other request on the box. It also tracks
  ``segment_id`` / ``revision`` for partial-result updates and drops
  out-of-pair-language segments aggressively because the user hears the
  TTS within ~600 ms.

* **Reprocess ASR** (``_transcribe_chunk``) is stateless and one-shot.
  Chunks are diarize-window-sized (4 s) — strictly larger than the live
  buffers — and **the language-naming system prompt regresses
  segment counts by 56 %** on those chunks (caught by ``meeting-scribe
  versions diff`` after run #2 against e5b376b2). So reprocess deliberately
  does NOT send the prompt, keeps temperature unconstrained, and runs
  at default priority (live ASR/TTS/translate at -20 / -10 always
  preempt it — a meeting in progress is never delayed).

* **Live translate** uses ``translation.queue.TranslationQueue`` +
  ``backends.translate_vllm.TranslateVllmBackend``: bounded queue with
  per-segment cancellation, multi-worker concurrency, dedup against the
  cache, and a per-replica retry/quarantine policy. Reprocess does not
  need any of that — there are no users waiting and segments arrive in
  one batch — so ``_translate_event`` is a thin one-shot HTTP wrapper.

Pulling these into a shared seam was tried and rejected: the live path's
state-keeping and SLA discipline drag every shared call site into
ceremony reprocess does not need, and the prompt difference would have to
be an else-branch on a parameter that is always set to the opposite
value in the two callers — i.e. zero shared logic, just a different name
for two different things. The shared parts that DO unify cleanly already
live in ``meeting_scribe.pipeline.{diarize,quality,speaker_attach}``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import time
import uuid
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from meeting_scribe.models import MeetingState
from meeting_scribe.pipeline.diarize import (
    _diarize_full_audio,
)
from meeting_scribe.storage import MeetingStorage

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


async def reprocess_meeting(
    meeting_dir: Path,
    storage: MeetingStorage,
    asr_url: str = "http://localhost:8003",
    translate_url: str = "http://localhost:8010",
    diarize_url: str = "http://localhost:8001",
    language_pair: list[str] | tuple[str, ...] = ("en", "ja"),
    on_progress: Callable[..., Any] | None = None,
    expected_speakers: int | None = None,
) -> dict:
    """Fully reprocess a meeting from its raw audio recording.

    Parallelized pipeline:
    1. transition_state(COMPLETE -> REPROCESSING), backup journal
    2. ASR: 4 concurrent workers process audio chunks
    3. Diarization: single call on the full audio (perfect cluster stability)
    4. Translation: 4 concurrent workers translate segments
    5. Attach speakers to events via time-range overlap
    6. Generate timeline, speaker data, summary
    7. transition_state(REPROCESSING -> COMPLETE)

    The state flips go through ``MeetingStorage.transition_state`` so
    the state-machine invariants (allowed source states, journal
    handling, logging) stay uniform with the live recording path.
    ``storage`` is a required parameter — both the server request
    handler and the ``meeting-scribe reprocess`` CLI instantiate or
    re-use a ``MeetingStorage`` rooted at the same ``meetings_dir``.
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
            meeting_dir.name,
            name,
            wall_ms,
            tail,
        )

    total_start = time.monotonic()

    # 0. Set state + backup. State flip goes through transition_state()
    # so the source-state guard (only COMPLETE -> REPROCESSING is legal)
    # fires if we get called on a still-recording meeting.
    #
    # Also write a PID+epoch lock file at ``meeting_dir/.reprocess.lock``
    # BEFORE the state flip. This is the anti-race contract with
    # ``MeetingStorage.recover_interrupted``: the recovery branch that
    # flips REPROCESSING -> COMPLETE only fires when this lock is
    # missing or stale. Without it, a live scribe restart in the
    # middle of a CLI reprocess would see REPROCESSING and "recover"
    # it to COMPLETE — then step 7's COMPLETE -> COMPLETE transition
    # crashes the whole reprocess. 2026-04-21 mass-reprocess incident.
    lock_path = meeting_dir / ".reprocess.lock"
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started_epoch": int(time.time())}))
    if meta_path.exists():
        storage.transition_state(meeting_dir.name, MeetingState.REPROCESSING)

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

    # 1. Load audio. Trim any trailing partial sample — a recording
    # terminated mid-write (disk full, crash, SIGKILL) can leave an
    # odd byte count that np.frombuffer(dtype=int16) rejects with
    # "buffer size must be a multiple of element size". Dropping the
    # last byte loses 31 µs of audio and unblocks the whole pipeline.
    phase_start("load_audio")
    pcm_data = pcm_path.read_bytes()
    trailing_bytes = len(pcm_data) % BYTES_PER_SAMPLE
    if trailing_bytes:
        logger.warning(
            "recording.pcm for %s has %d trailing byte(s) (size=%d, "
            "not a multiple of %d) — truncating to the last whole sample",
            meeting_dir.name,
            trailing_bytes,
            len(pcm_data),
            BYTES_PER_SAMPLE,
        )
        pcm_data = pcm_data[:-trailing_bytes]
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
        diarize_result = await _diarize_full_audio(
            pcm_data,
            diarize_url,
            max_speakers=diar_max,
            min_speakers=diar_min,
            expected_speakers=expected_speakers,
        )
        # reprocess uses the standard segmentation to *shape* ASR chunks
        # (each chunk inherits its overlapping diarize segment's
        # cluster_id), so we keep ``diarize_segments`` aliased to the
        # standard array for the rest of this function.  The exclusive
        # array is preserved for downstream tooling that wants the
        # single-speaker timeline (timeline.json + UI).
        diarize_segments = diarize_result.segments
        diarize_exclusive = diarize_result.exclusive_segments
        logger.info(
            "Full-audio diarization: %d standard / %d exclusive segments",
            len(diarize_segments),
            len(diarize_exclusive),
        )
        # Persist frame-level exclusive timeline as a sidecar artifact.
        # Additive — UI ignores unknown files.  See
        # plans/2026-Q3-followups.md Phase C2.
        if diarize_exclusive:
            from meeting_scribe.routes.meeting_lifecycle import _persist_exclusive_segments

            _persist_exclusive_segments(meeting_dir, diarize_exclusive)
    except Exception as e:
        logger.warning("Diarization failed: %s", e)
        diarize_segments = []
        diarize_exclusive = []
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
            len(asr_chunks),
            len(asr_chunks) - gap_chunks,
            gap_chunks,
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
                    client,
                    asr_url,
                    asr_model,
                    pcm,
                    start_ms,
                    end_ms,
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
                    await _translate_event(client, translate_url, trans_model, event, language_pair)
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
            old_named_ranges.append((old_seg.get("start_ms", 0), old_seg.get("end_ms", 0), name))

        # Step 1 — find best-name per new event.
        from collections import Counter

        event_best_name: dict[str, str] = {}  # segment_id → name
        event_cluster: dict[str, int] = {}  # segment_id → cluster_id
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
            cid: votes.most_common(1)[0][0] for cid, votes in cluster_votes.items() if votes
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
            preserved_corrections.append(
                {
                    "type": "speaker_correction",
                    "segment_id": sid,
                    "speaker_name": name,
                    "preserved_from_reprocess": True,
                    "scope": "cluster",
                }
            )

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
            preserved_corrections.append(
                {
                    "type": "speaker_correction",
                    "segment_id": sid,
                    "speaker_name": name,
                    "preserved_from_reprocess": True,
                    "scope": "event",
                }
            )
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

    phase_end(
        "preserve_corrections", f"events={len(events)} corrections={len(preserved_corrections)}"
    )

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
            meeting_dir,
            journal_path,
            _json,
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

    # 7. Finalize state + drop the lock. The lock goes away AFTER the
    # state transitions back to COMPLETE so no observer can see
    # "COMPLETE with a stale lock" in between.
    if meta_path.exists():
        storage.transition_state(meeting_dir.name, MeetingState.COMPLETE)
    lock_path.unlink(missing_ok=True)

    progress(7, 7, "Complete")

    total_wall_ms = int((time.monotonic() - total_start) * 1000)
    logger.info(
        "reprocess_phase meeting=%s phase=TOTAL wall_ms=%d audio_ms=%d events=%d speakers=%d",
        meeting_dir.name,
        total_wall_ms,
        duration_ms,
        len(events),
        len(speaker_stats),
    )

    return {
        "segments": len(events),
        "duration_ms": duration_ms,
        "translated": sum(1 for e in events if (e.get("translation") or {}).get("text")),
        "speakers": len(speaker_stats),
        "has_summary": "error" not in summary,
    }
