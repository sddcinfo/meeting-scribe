"""Near-realtime refinement worker for meeting audio.

Trails 45s behind the live recording, re-transcribes audio in batch mode
at higher quality, extracts speaker embeddings, and translates. By meeting
end, most audio is already refined — post-meeting delay is <60s.

Results are stored as polished.json alongside journal.jsonl.

Context window
--------------
The refinement worker optionally folds a rolling window of recent
already-refined ``(source_text, translation)`` tuples into each batch
translate prompt.  This is the "follow-shortly-after quality uplift"
path for JA→EN in particular — the live path is stateless because of
its ~400 ms latency SLO, so fragmented utterances (very common in
Japanese, where subjects drop and sentences trail off in particles)
get translated without anchoring.  The refinement path has 45 s of
headroom and can afford the extra prompt tokens, so we use it to give
the model a running view of the meeting's topic, speakers, and
proper-nouns.  Controlled by
``ServerConfig.refinement_context_window_segments`` (default 0 = OFF).
Runtime-flippable via the ``refinement_context_window_segments`` knob
in runtime_config so operators can sweep window sizes against a single
meeting without restarting scribe.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from meeting_scribe import runtime_config
from meeting_scribe.backends.asr_filters import (
    _detect_language_from_text,
    _is_hallucination,
    _parse_qwen3_asr_response,
)
from meeting_scribe.backends.translate_vllm import _log_translation

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2  # s16le

# Separate JSONL for refinement ASR rows. The translate aggregator
# strict-filters on ``kind == "translate"``, so mixing ASR rows into the
# translate file would either pollute the aggregation or force every
# reader to schema-discriminate. Keep them in their own file with their
# own schema.
_REFINEMENT_ASR_LOG_PATH = Path.home() / ".local" / "share" / "autosre" / "scribe-asr.jsonl"
_REFINEMENT_ASR_SCHEMA_VERSION = 1


def _log_refinement_asr(
    *,
    meeting_id: str,
    model: str,
    start_ms: int,
    end_ms: int,
    elapsed_ms: float,
    input_tokens: int,
    output_tokens: int,
    language: str,
    text_prefix: str,
) -> None:
    """Fire-and-forget JSONL logger for refinement ASR calls. Never raises."""
    try:
        _REFINEMENT_ASR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps(
            {
                "ts": time.time(),
                "schema_version": _REFINEMENT_ASR_SCHEMA_VERSION,
                "kind": "asr",
                "source": "refinement",
                "meeting_id": meeting_id,
                "model": model,
                "audio_start_ms": start_ms,
                "audio_end_ms": end_ms,
                "elapsed_ms": round(elapsed_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "language": language,
                "text_prefix": text_prefix[:80],
            }
        )
        with _REFINEMENT_ASR_LOG_PATH.open("a") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # Never let logging break refinement


class RefinementWorker:
    """Background worker that trails live transcription and re-processes audio at higher quality."""

    def __init__(
        self,
        *,
        meeting_id: str,
        meeting_dir: Path,
        asr_url: str = "http://localhost:8003",
        translate_url: str = "http://localhost:8010",
        language_pair: list[str] | tuple[str, ...] = ("en", "ja"),
        trail_seconds: float = 45.0,
        chunk_seconds: float = 10.0,
        context_window_segments: int = 0,
    ) -> None:
        self._meeting_id = meeting_id
        self._meeting_dir = meeting_dir
        self._asr_url = asr_url.rstrip("/")
        self._translate_url = translate_url.rstrip("/")
        # 1 or 2 codes; monolingual meetings short-circuit the translation
        # branch via ``get_translation_target`` returning ``None``.
        self._language_pair: list[str] = list(language_pair)
        self._trail_seconds = trail_seconds
        self._chunk_seconds = chunk_seconds
        # Static constructor default; per-batch reads fall through to
        # runtime_config first so operators can flip the sweep value
        # without restarting.  0 disables the context-window uplift.
        self._context_window_segments_default = max(0, int(context_window_segments))

        self._pcm_path = meeting_dir / "audio" / "recording.pcm"
        self._processed_offset = 0  # bytes already processed
        self._results: list[dict] = []
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._paused = False

        self._asr_client: httpx.AsyncClient | None = None
        self._translate_client: httpx.AsyncClient | None = None
        self._asr_model: str | None = None
        self._translate_model: str | None = None

        # Load counters — incremented on every ASR/translate call the
        # worker actually issues. Snapshotted into the drain registry at
        # meeting stop (see server._DrainEntry) so validation can read
        # the final numbers after refinement_worker has been nulled.
        self.translate_call_count: int = 0
        self.asr_call_count: int = 0
        self.last_error_count: int = 0

    async def start(self) -> None:
        """Start the refinement background task.

        May be wrapped in ``asyncio.create_task`` by the server so the
        HTTP start response returns without waiting for ASR/translate
        model-id probes. If the enclosing meeting ends during that
        window, the server-side cancellation in ``_drain_refinement``
        takes precedence; this coroutine additionally re-checks its own
        ``_stop_event`` just before arming ``_run_loop`` so a late
        completion never attaches to a dead meeting.
        """
        self._asr_client = httpx.AsyncClient(base_url=self._asr_url, timeout=60)
        self._translate_client = httpx.AsyncClient(base_url=self._translate_url, timeout=60)

        # Auto-detect model IDs
        try:
            resp = await self._asr_client.get("/v1/models")
            self._asr_model = resp.json()["data"][0]["id"]
        except Exception as e:
            logger.warning("Refinement: ASR model detection failed: %s", e)
            self._asr_model = "Qwen/Qwen3-ASR-1.7B"

        try:
            resp = await self._translate_client.get("/v1/models")
            self._translate_model = resp.json()["data"][0]["id"]
        except Exception as e:
            logger.warning("Refinement: Translation model detection failed: %s", e)

        # Late-arrival guard: if stop() was called while we were awaiting
        # the model-id probes above, do NOT arm the run loop.
        if self._stop_event.is_set():
            logger.info(
                "Refinement worker start aborted post-probe for meeting %s (stop signaled)",
                self._meeting_id,
            )
            return

        self._task = asyncio.create_task(self._run_loop(), name="refinement-worker")
        logger.info("Refinement worker started for meeting %s", self._meeting_id)

    async def stop(self) -> None:
        """Stop the worker, process remaining audio, save results.

        Safe to call even if ``start()`` never armed ``_run_loop`` — e.g.
        when the server cancelled the backgrounded start task before it
        could complete. In that case ``_task`` is None and
        ``_process_remaining`` is skipped because the worker never
        ingested anything.
        """
        self._stop_event.set()
        if self._task:
            await self._task
            # Process remaining audio (no trail — process everything)
            await self._process_remaining()
            self._save_polished()

        if self._asr_client:
            await self._asr_client.aclose()
        if self._translate_client:
            await self._translate_client.aclose()

        logger.info(
            "Refinement worker stopped: %d segments polished for meeting %s",
            len(self._results),
            self._meeting_id,
        )

    def pause(self) -> None:
        """Pause refinement (e.g., when VRAM is high)."""
        self._paused = True
        logger.info("Refinement worker paused")

    def resume(self) -> None:
        """Resume refinement."""
        self._paused = False
        logger.info("Refinement worker resumed")

    def _apply_vram_gate(self) -> bool:
        """Consult gpu_monitor and mutate ``self._paused`` per thresholds.

        Pause when VRAM pct > 95, resume when < 80.  Returns ``True`` if
        the worker should skip this iteration of the main loop
        (currently paused or just transitioned to paused).  Extracted
        from ``_run_loop`` so unit tests can exercise the gate without
        driving the 10s wait loop.
        """
        try:
            from meeting_scribe.gpu_monitor import get_vram_usage

            vram = get_vram_usage()
        except Exception:
            return self._paused  # gpu_monitor unavailable → preserve state

        if vram and vram.pct > 95:
            if not self._paused:
                logger.warning("VRAM >95%%, pausing refinement")
                self._paused = True
            return True
        if self._paused and vram and vram.pct < 80:
            self._paused = False
            logger.info("VRAM <80%%, resuming refinement")
        return self._paused

    async def _run_loop(self) -> None:
        """Main loop: wake every 10s, process available audio."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=10)
                break  # stop_event was set
            except TimeoutError:
                pass  # 10s elapsed, check for work

            if self._apply_vram_gate():
                continue

            try:
                await self._process_available()
            except Exception:
                logger.exception("Refinement worker error")

    async def _process_available(self) -> None:
        """Read new audio from recording.pcm and process in batch."""
        if not self._pcm_path.exists():
            return

        file_size = await asyncio.to_thread(lambda: self._pcm_path.stat().st_size)
        trail_bytes = int(self._trail_seconds * BYTES_PER_SEC)
        available = file_size - trail_bytes

        if available <= self._processed_offset:
            return

        # Verify file size is stable (avoid partial writes)
        await asyncio.sleep(0.1)
        file_size2 = await asyncio.to_thread(lambda: self._pcm_path.stat().st_size)
        if file_size2 != file_size:
            return  # File is being written to, wait

        chunk_bytes = int(self._chunk_seconds * BYTES_PER_SEC)
        end = min(self._processed_offset + chunk_bytes, available)

        # Read PCM chunk
        pcm = await asyncio.to_thread(self._read_pcm, self._processed_offset, end)
        if not pcm:
            return

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        start_ms = int(self._processed_offset / BYTES_PER_SEC * 1000)
        end_ms = int(end / BYTES_PER_SEC * 1000)

        # Batch ASR
        result = await self._batch_transcribe(audio, start_ms, end_ms)
        if result:
            self._results.append(result)

        # Back up by 1 second for overlapping chunks (better accuracy at boundaries)
        overlap_bytes = int(1.0 * BYTES_PER_SEC)
        self._processed_offset = max(end - overlap_bytes, self._processed_offset)

        # Throttle to avoid saturating vLLM
        await asyncio.sleep(0.5)

    async def _process_remaining(self) -> None:
        """Process any unprocessed audio after meeting stop."""
        if not self._pcm_path.exists():
            return

        file_size = await asyncio.to_thread(lambda: self._pcm_path.stat().st_size)

        while self._processed_offset < file_size:
            chunk_bytes = int(self._chunk_seconds * BYTES_PER_SEC)
            end = min(self._processed_offset + chunk_bytes, file_size)

            pcm = await asyncio.to_thread(self._read_pcm, self._processed_offset, end)
            if not pcm:
                break

            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            start_ms = int(self._processed_offset / BYTES_PER_SEC * 1000)
            end_ms = int(end / BYTES_PER_SEC * 1000)

            result = await self._batch_transcribe(audio, start_ms, end_ms)
            if result:
                self._results.append(result)

            self._processed_offset = end
            await asyncio.sleep(0.1)

    def _read_pcm(self, start: int, end: int) -> bytes | None:
        """Read a slice of the PCM file (runs in thread)."""
        try:
            with open(self._pcm_path, "rb") as f:
                f.seek(start)
                return f.read(end - start)
        except Exception as e:
            logger.warning("Refinement: PCM read error: %s", e)
            return None

    async def _batch_transcribe(self, audio: np.ndarray, start_ms: int, end_ms: int) -> dict | None:
        """Transcribe an audio chunk via vLLM Qwen3-ASR (batch mode)."""
        if len(audio) < SAMPLE_RATE * 0.5:  # Skip very short chunks
            return None
        if self._asr_client is None:
            logger.warning("Refinement: ASR client not started")
            return None

        # Encode as WAV
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        asr_t0 = time.monotonic()
        try:
            resp = await self._asr_client.post(
                "/v1/chat/completions",
                json={
                    "model": self._asr_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": audio_b64, "format": "wav"},
                                }
                            ],
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    # Refinement sits at the same tier as live
                    # translation (-10). The old -8 left it between
                    # summary (-5) and translation (-10), so summary
                    # could not preempt a backlogged refinement queue.
                    # Live translation running during a meeting still
                    # wins via FCFS within the same tier (refinement
                    # trails 45 s behind live, so it's only scheduled
                    # after the live request completes).
                    "priority": -10,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            raw = body["choices"][0]["message"]["content"]
            asr_usage = body.get("usage", {}) or {}
        except Exception as e:
            self.last_error_count += 1
            logger.warning("Refinement: ASR failed for chunk %d-%dms: %s", start_ms, end_ms, e)
            return None

        asr_elapsed_ms = (time.monotonic() - asr_t0) * 1000
        self.asr_call_count += 1

        text, lang = _parse_qwen3_asr_response(raw)
        if not text or _is_hallucination(text):
            # Still count the call — the worker did issue it — but
            # record only enough of the prefix for debugging.
            _log_refinement_asr(
                meeting_id=self._meeting_id,
                model=self._asr_model or "unknown",
                start_ms=start_ms,
                end_ms=end_ms,
                elapsed_ms=asr_elapsed_ms,
                input_tokens=int(asr_usage.get("prompt_tokens", 0)),
                output_tokens=int(asr_usage.get("completion_tokens", 0)),
                language="hallucination" if text else "empty",
                text_prefix=text or "",
            )
            return None

        if lang == "unknown":
            lang = _detect_language_from_text(text)

        _log_refinement_asr(
            meeting_id=self._meeting_id,
            model=self._asr_model or "unknown",
            start_ms=start_ms,
            end_ms=end_ms,
            elapsed_ms=asr_elapsed_ms,
            input_tokens=int(asr_usage.get("prompt_tokens", 0)),
            output_tokens=int(asr_usage.get("completion_tokens", 0)),
            language=lang,
            text_prefix=text,
        )

        result = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "language": lang,
            "source": "refinement",
        }

        # Translate if we have a translation model
        from meeting_scribe.languages import get_translation_target

        target = (
            get_translation_target(lang, self._language_pair) if self._translate_model else None
        )
        if target:
            translation = await self._batch_translate(text, lang, target)
            if translation:
                result["translation"] = {"text": translation, "target_language": target}

        return result

    def _collect_prior_context(
        self, source_lang: str, target_lang: str, window: int
    ) -> list[tuple[str, str]]:
        """Return up to *window* most-recent ``(source, translation)`` pairs
        from already-refined results for the same direction.

        Only same-direction pairs are returned — mixing directions would
        muddle the prompt (the LLM would see JA→EN exemplars while asked
        to translate EN→JA).  Order is oldest → newest so the closing
        instruction in :func:`get_translation_prompt` still reads as
        "translate the *next* utterance".
        """
        if window <= 0:
            return []
        collected: list[tuple[str, str]] = []
        # Walk newest → oldest, filter, then reverse so caller sees
        # oldest first.
        for prior in reversed(self._results):
            if prior.get("language") != source_lang:
                continue
            translation = prior.get("translation")
            if not isinstance(translation, dict):
                continue
            tgt_text = translation.get("text") or ""
            if translation.get("target_language") != target_lang:
                continue
            src_text = prior.get("text") or ""
            if src_text and tgt_text:
                collected.append((src_text, tgt_text))
            if len(collected) >= window:
                break
        collected.reverse()
        return collected

    async def _batch_translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """Translate text via vLLM, optionally with rolling meeting context.

        Reads the context-window size from ``runtime_config`` per batch
        so operators can run a sweep (0, 2, 4, 6, 8, ...) against a
        single meeting without restarting scribe.  Fallback order is
        runtime-config → constructor default → 0.
        """
        from meeting_scribe.languages import get_translation_prompt

        if self._translate_client is None:
            logger.warning("Refinement: translate client not started")
            return None

        window = int(
            runtime_config.get(
                "refinement_context_window_segments",
                self._context_window_segments_default,
            )
            or 0
        )
        prior_context = self._collect_prior_context(source_lang, target_lang, window)
        system = get_translation_prompt(
            source_lang, target_lang, prior_context=prior_context or None
        )
        prompt = text

        tx_t0 = time.monotonic()
        try:
            resp = await self._translate_client.post(
                "/v1/chat/completions",
                json={
                    "model": self._translate_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 512,
                    # Refinement sits at the same tier as live
                    # translation (-10). The old -8 left it between
                    # summary (-5) and translation (-10), so summary
                    # could not preempt a backlogged refinement queue.
                    # Live translation running during a meeting still
                    # wins via FCFS within the same tier (refinement
                    # trails 45 s behind live, so it's only scheduled
                    # after the live request completes).
                    "priority": -10,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            translated = body["choices"][0]["message"]["content"].strip()
            usage = body.get("usage", {}) or {}
        except Exception as e:
            self.last_error_count += 1
            logger.warning("Refinement: Translation failed: %s", e)
            return None

        tx_elapsed_ms = (time.monotonic() - tx_t0) * 1000
        self.translate_call_count += 1

        # Log to the same translate JSONL the live path writes to — the
        # ``source="refinement"`` tag is what separates the two paths
        # when ``translate_stats.py`` aggregates by ``(meeting_id, source)``.
        _log_translation(
            model=self._translate_model or "unknown",
            source_lang=source_lang,
            target_lang=target_lang,
            text=text,
            translated=translated,
            elapsed_ms=tx_elapsed_ms,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            kind="translate",
            source="refinement",
            meeting_id=self._meeting_id,
        )
        return translated

    def _save_polished(self) -> None:
        """Write polished results to polished.json."""
        path = self._meeting_dir / "polished.json"
        data = {
            "meeting_id": self._meeting_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "segments": self._results,
            "total_segments": len(self._results),
            "audio_duration_ms": int(self._processed_offset / BYTES_PER_SEC * 1000),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        logger.info("Polished transcript saved: %d segments → %s", len(self._results), path)
