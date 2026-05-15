"""Qwen3-TTS voice cloning server for meeting-scribe.

OpenAI-compatible /v1/audio/speech endpoint with dynamic voice cloning.
Voice references are passed inline as base64-encoded WAV in the 'voice' field.

Uses faster-qwen3-tts CUDA graph optimization when available,
falls back to qwen-tts baseline otherwise.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import struct
import threading
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("tts-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Meeting Scribe TTS")

# Global model reference
_model = None
_model_name: str = ""
_backend: str = "unknown"  # "faster" or "baseline"
_sample_rate: int = 24000
_synth_lock: asyncio.Lock | None = None
_synth_started_at: float | None = None
_synth_request: str = ""
_fatal_error: str | None = None
_SERVER_SYNTH_TIMEOUT_S = float(os.environ.get("TTS_SERVER_SYNTH_TIMEOUT_S", "12.0"))
_MAX_AUDIO_S = float(os.environ.get("TTS_MAX_AUDIO_S", "12.0"))
_MAX_AUDIO_BASE_S = float(os.environ.get("TTS_MAX_AUDIO_BASE_S", "1.5"))
_MAX_AUDIO_PER_CHAR_S = float(os.environ.get("TTS_MAX_AUDIO_PER_CHAR_S", "0.22"))
_EXIT_ON_FATAL = os.environ.get("TTS_EXIT_ON_FATAL", "1") != "0"
_FASTER_XVEC_ONLY = os.environ.get("TTS_FASTER_XVEC_ONLY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
_FASTER_NON_STREAMING_MODE = os.environ.get(
    "TTS_FASTER_NON_STREAMING_MODE", "0"
).strip().lower() not in {
    "0",
    "false",
    "no",
}


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request."""

    model: str = "qwen3-tts"
    input: str  # Text to synthesize
    # Voice selection:
    #   "default"                    — bundled reference WAV
    #   "<named_speaker>"            — Qwen3-TTS CustomVoice timbre
    #                                  (Aiden, Vivian, Ono_Anna, Sohee, …);
    #                                  only works on CustomVoice model variants
    #   base64 WAV (>100 chars)      — inline voice cloning reference
    voice: str = "default"
    language: str = "English"  # Full language name
    response_format: str = "wav"
    speed: float = 1.0


def _get_synth_lock() -> asyncio.Lock:
    """Return the per-process synthesis lock for the active event loop."""
    global _synth_lock
    if _synth_lock is None:
        _synth_lock = asyncio.Lock()
    return _synth_lock


def _schedule_fatal_exit(reason: str) -> None:
    """Exit after returning the current HTTP response so Docker restarts us."""
    if not _EXIT_ON_FATAL:
        return

    def _exit() -> None:
        time.sleep(0.25)
        logger.error("Exiting TTS process after fatal synthesis failure: %s", reason)
        os._exit(70)

    threading.Thread(target=_exit, name="tts-fatal-exit", daemon=True).start()


def _mark_fatal(reason: str) -> None:
    """Mark synthesis unhealthy and schedule a process restart."""
    global _fatal_error
    _fatal_error = reason
    _schedule_fatal_exit(reason)


def _load_model() -> None:
    """Load Qwen3-TTS model at startup."""
    global _model, _model_name, _backend, _sample_rate

    model_id = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    requested_backend = os.environ.get("TTS_BACKEND", "auto").strip().lower()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # In offline mode, resolve the Hub model ID to its local cache snapshot
    # path. from_pretrained() with a Hub ID hits the Hub API even when the
    # model is fully cached; a local path avoids this entirely.
    #
    # HF cache layout: <cache_dir>/hub/models--<org>--<name>/refs/main → SHA
    #                   <cache_dir>/hub/models--<org>--<name>/snapshots/<SHA>/
    if os.environ.get("HF_HUB_OFFLINE") == "1" and "/" in model_id:
        from pathlib import Path

        cache_dir = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
        model_dir_name = "models--" + model_id.replace("/", "--")
        refs_main = cache_dir / "hub" / model_dir_name / "refs" / "main"
        if refs_main.exists():
            sha = refs_main.read_text().strip()
            snapshot_path = cache_dir / "hub" / model_dir_name / "snapshots" / sha
            if snapshot_path.is_dir():
                logger.info("Offline mode: resolved %s -> %s", model_id, snapshot_path)
                model_id = str(snapshot_path)

    logger.info("Loading TTS model: %s on %s (%s)", model_id, device, dtype)
    t0 = time.monotonic()

    if requested_backend not in {"auto", "faster", "baseline"}:
        raise ValueError(f"Unsupported TTS_BACKEND={requested_backend!r}")

    # Try faster-qwen3-tts first (CUDA graph optimization), unless disabled.
    # The GB10 production image pins faster-qwen3-tts 0.2.6 plus PyTorch 2.11.
    # Keep the baseline backend available as an emergency rollback path, but
    # the saved-meeting regression gate promotes the faster runtime only.
    if requested_backend != "baseline":
        try:
            from faster_qwen3_tts import FasterQwen3TTS

            _model = FasterQwen3TTS.from_pretrained(
                model_id,
                device=device,
                dtype=dtype,
            )
            _backend = "faster"
            _model_name = model_id
            _sample_rate = 24000
            logger.info(
                "TTS loaded with faster-qwen3-tts backend in %.1fs",
                time.monotonic() - t0,
            )
            return
        except ImportError:
            if requested_backend == "faster":
                raise
            logger.info("faster-qwen3-tts not available, trying baseline qwen-tts")
        except Exception as e:
            if requested_backend == "faster":
                raise
            logger.warning("faster-qwen3-tts failed: %s, trying baseline", e)

    # Fallback to baseline qwen-tts
    try:
        from qwen_tts import Qwen3TTSModel

        _model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
        )
        _backend = "baseline"
        _model_name = model_id
        _sample_rate = 24000
        logger.info(
            "TTS loaded with baseline qwen-tts backend in %.1fs",
            time.monotonic() - t0,
        )
    except Exception as e:
        logger.error("Failed to load TTS model: %s", e)
        raise


def _decode_voice_ref(voice_b64: str) -> tuple[np.ndarray, int]:
    """Decode a base64-encoded WAV into numpy array + sample rate."""
    wav_bytes = base64.b64decode(voice_b64)
    buf = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Mono
    return audio, sr


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 audio to WAV bytes."""
    buf = io.BytesIO()
    # Clip and convert to int16
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# Default voice references — bundled WAV files. X-vector embeddings are
# pre-computed at startup so synthesis doesn't re-extract per request.
_DEFAULT_VOICE_DIR = Path(__file__).parent
_DEFAULT_VOICES = {
    "male": _DEFAULT_VOICE_DIR / "voice_male.wav",
    "female": _DEFAULT_VOICE_DIR / "voice_female.wav",
}


def _default_ref_path() -> str:
    """Return path to the default voice reference (real meeting audio)."""
    return str(_DEFAULT_VOICES["male"])


# Qwen3-TTS CustomVoice named speakers — valid only when the loaded model
# is the CustomVoice variant (``*-CustomVoice``). On Base models this list
# is ignored and we fall back to the default reference clip.
_CUSTOM_VOICE_SPEAKERS: frozenset[str] = frozenset(
    {
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ryan",
        "Aiden",
        "Ono_Anna",
        "Sohee",
    }
)


def _is_custom_voice_model() -> bool:
    """Return True when the loaded model supports ``speaker=`` named timbres."""
    return "CustomVoice" in (_model_name or "")


def _synthesize_custom_voice(text: str, language: str, speaker: str) -> np.ndarray:
    """Synthesize with a Qwen3-TTS named speaker (CustomVoice variant only)."""
    fn = getattr(_model, "generate_custom_voice", None)
    if fn is None:
        # Base-model image — cannot honour named speakers; let the caller
        # fall back to reference-audio synthesis.
        return np.zeros(0, dtype=np.float32)
    wavs = fn(text=text, language=language, speaker=speaker)
    if isinstance(wavs, tuple):
        wavs = wavs[0]
    if isinstance(wavs, list):
        wavs = wavs[0]
    if isinstance(wavs, torch.Tensor):
        wavs = wavs.cpu().numpy()
    return np.asarray(wavs, dtype=np.float32)


def _synthesize_faster(
    text: str, language: str, ref_audio: np.ndarray | None, ref_sr: int
) -> np.ndarray:
    """Synthesize using faster-qwen3-tts backend."""
    if ref_audio is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, ref_audio, ref_sr)
        try:
            audio = _model.generate_voice_clone(
                text=text,
                ref_audio=tmp_path,
                language=language,
                xvec_only=_FASTER_XVEC_ONLY,
                non_streaming_mode=_FASTER_NON_STREAMING_MODE,
            )
        finally:
            import os

            os.unlink(tmp_path)
    else:
        audio = _model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=_default_ref_path(),
            xvec_only=_FASTER_XVEC_ONLY,
            non_streaming_mode=_FASTER_NON_STREAMING_MODE,
        )

    if isinstance(audio, tuple):
        audio = audio[0]
    if isinstance(audio, list):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    return audio.astype(np.float32)


def _synthesize_baseline(
    text: str, language: str, ref_audio: np.ndarray | None, ref_sr: int
) -> np.ndarray:
    """Synthesize using baseline qwen-tts backend."""
    if ref_audio is not None:
        wavs, _sr = _model.generate_voice_clone(
            text=text,
            ref_audio=(ref_audio, ref_sr),
            language=language,
            x_vector_only_mode=True,
            non_streaming_mode=True,
        )
    else:
        ref, ref_sr2 = sf.read(str(_DEFAULT_VOICES["male"]))
        wavs, _sr = _model.generate_voice_clone(
            text=text,
            ref_audio=(ref, ref_sr2),
            language=language,
            x_vector_only_mode=True,
            non_streaming_mode=True,
        )
    audio = wavs[0] if isinstance(wavs, list) else wavs
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    return audio.astype(np.float32)


@app.on_event("startup")
async def startup():
    _load_model()


@app.get("/health")
async def health():
    if _model is None:
        return {"status": "loading", "model": _model_name, "backend": _backend, "device": "cpu"}

    now = time.monotonic()
    if _fatal_error:
        return JSONResponse(
            {
                "status": "fatal",
                "model": _model_name,
                "backend": _backend,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "error": _fatal_error,
            },
            status_code=503,
        )
    if _synth_started_at is not None:
        busy_for = now - _synth_started_at
        if busy_for > _SERVER_SYNTH_TIMEOUT_S:
            return JSONResponse(
                {
                    "status": "stalled",
                    "model": _model_name,
                    "backend": _backend,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "busy_for_s": round(busy_for, 2),
                    "request": _synth_request,
                },
                status_code=503,
            )

    # Real CUDA health check — catches GPU context corruption that leaves
    # the model object alive but every inference call failing with CUDA errors
    cuda_ok = True
    cuda_error = None
    if torch.cuda.is_available():
        try:
            # Minimal CUDA op: allocate a tiny tensor and sync
            t = torch.zeros(1, device="cuda:0")
            del t
            torch.cuda.synchronize()
        except Exception as e:
            cuda_ok = False
            cuda_error = str(e)

    status = "healthy" if cuda_ok else "cuda_error"
    resp = {
        "status": status,
        "model": _model_name,
        "backend": _backend,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        resp["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        resp["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
    if cuda_error:
        resp["error"] = cuda_error
    if not cuda_ok:
        return JSONResponse(resp, status_code=503)
    return resp


def _synthesize_request_sync(
    req: SpeechRequest,
    *,
    language: str,
    ref_audio: np.ndarray | None,
    ref_sr: int,
    named_speaker: str | None,
) -> np.ndarray:
    """Run the blocking model call in a worker thread.

    The baseline qwen-tts path can occasionally wedge inside
    ``generate_voice_clone``. Running it off the FastAPI event loop keeps
    /health responsive so the parent app can detect the stall and restart
    this container instead of waiting on a shallow 200 OK.
    """
    if named_speaker and _is_custom_voice_model():
        audio = _synthesize_custom_voice(req.input, language, named_speaker)
        if len(audio) > 0:
            return audio
        return (
            _synthesize_faster(req.input, language, ref_audio, ref_sr)
            if _backend == "faster"
            else _synthesize_baseline(req.input, language, ref_audio, ref_sr)
        )
    if _backend == "faster":
        return _synthesize_faster(req.input, language, ref_audio, ref_sr)
    return _synthesize_baseline(req.input, language, ref_audio, ref_sr)


def _audio_budget_s(text: str) -> float:
    return max(1.2, min(_MAX_AUDIO_S, _MAX_AUDIO_BASE_S + _MAX_AUDIO_PER_CHAR_S * len(text)))


def _enforce_audio_budget(audio: np.ndarray, request: str, text: str) -> np.ndarray:
    """Cap pathological TTS output without restarting a healthy worker.

    Qwen3-TTS can over-speak very short fragments, especially after punctuation
    or disfluency-heavy input. That is a content-quality issue, not evidence
    that CUDA or the model process is corrupt. Truncate the audio so listeners
    are protected, but reserve fatal restarts for real synthesis failures and
    hard timeouts.
    """
    budget_s = _audio_budget_s(text)
    max_samples = max(1, int(budget_s * _sample_rate))
    if len(audio) <= max_samples:
        return audio
    reason = (
        f"synthesis exceeded audio budget: {len(audio) / max(_sample_rate, 1):.1f}s "
        f"> {budget_s:.1f}s ({request})"
    )
    logger.warning(reason)
    return audio[:max_samples]


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    """OpenAI-compatible TTS endpoint with voice cloning support."""
    global _fatal_error, _synth_request, _synth_started_at

    if _model is None:
        raise HTTPException(503, "Model not loaded")
    if _fatal_error:
        raise HTTPException(503, f"TTS worker unhealthy: {_fatal_error}")

    if not req.input or not req.input.strip():
        raise HTTPException(400, "Empty input text")

    t0 = time.monotonic()

    # Decode voice reference if provided
    ref_audio = None
    ref_sr = 16000
    MAX_VOICE_REF_SIZE = 1_000_000  # 1MB base64 limit (~10s of 16kHz audio)
    named_speaker: str | None = None
    if req.voice and req.voice != "default" and len(req.voice) > 100:
        if len(req.voice) > MAX_VOICE_REF_SIZE:
            raise HTTPException(
                400, f"Voice reference too large ({len(req.voice)} bytes, max {MAX_VOICE_REF_SIZE})"
            )
        try:
            ref_audio, ref_sr = _decode_voice_ref(req.voice)
            logger.info(
                "Voice cloning: %.1fs reference at %dHz",
                len(ref_audio) / ref_sr,
                ref_sr,
            )
        except Exception as e:
            logger.warning("Failed to decode voice reference: %s", e)
    elif req.voice in _CUSTOM_VOICE_SPEAKERS:
        named_speaker = req.voice

    # Map language codes to full names
    lang_map = {
        "en": "English",
        "ja": "Japanese",
        "zh": "Chinese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
    }
    language = lang_map.get(req.language, req.language)

    # Synthesize
    lock = _get_synth_lock()
    if lock.locked():
        raise HTTPException(429, "TTS synthesis already in progress")
    try:
        async with lock:
            _synth_started_at = time.monotonic()
            _synth_request = (
                f"chars={len(req.input)} lang={language} voice={req.voice[:40]} backend={_backend}"
            )
            try:
                audio = await asyncio.wait_for(
                    asyncio.to_thread(
                        _synthesize_request_sync,
                        req,
                        language=language,
                        ref_audio=ref_audio,
                        ref_sr=ref_sr,
                        named_speaker=named_speaker,
                    ),
                    timeout=_SERVER_SYNTH_TIMEOUT_S,
                )
                audio = _enforce_audio_budget(audio, _synth_request, req.input)
            except TimeoutError as e:
                reason = (
                    f"synthesis timeout after {_SERVER_SYNTH_TIMEOUT_S:.1f}s ({_synth_request})"
                )
                logger.error("Synthesis timed out: %s", reason)
                _mark_fatal(reason)
                raise HTTPException(504, reason) from e
            finally:
                _synth_started_at = None
                _synth_request = ""
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        reason = f"synthesis failed: {type(e).__name__}: {e}"
        logger.exception("Synthesis failed; marking worker fatal")
        _mark_fatal(reason)
        raise HTTPException(500, reason) from e

    elapsed_ms = (time.monotonic() - t0) * 1000
    duration_ms = len(audio) / _sample_rate * 1000
    logger.info(
        "Synthesized %.1fs audio in %.0fms (RTF=%.2fx) backend=%s",
        duration_ms / 1000,
        elapsed_ms,
        duration_ms / max(elapsed_ms, 1),
        _backend,
    )

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    if req.response_format.lower() == "pcm":
        return Response(content=pcm, media_type="application/octet-stream")
    wav_bytes = _encode_wav(audio, _sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


def _synthesize_faster_streaming(
    text: str, language: str, ref_audio: np.ndarray | None, ref_sr: int, chunk_size: int
):
    """Yield (audio_chunk_np_float32, sample_rate) pairs from faster-qwen3-tts.

    Falls back to a single non-streaming chunk if the backend doesn't expose
    ``generate_voice_clone_streaming`` — keeps the endpoint functional on
    older images where streaming isn't yet available.
    """
    streaming_fn = getattr(_model, "generate_voice_clone_streaming", None)
    if streaming_fn is None:
        audio = _synthesize_faster(text, language, ref_audio, ref_sr)
        yield audio, _sample_rate
        return

    import tempfile as _tempfile

    ref_path = None
    tmp_path = None
    if ref_audio is not None:
        with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, ref_audio, ref_sr)
        ref_path = tmp_path
    else:
        ref_path = _default_ref_path()

    try:
        stream_kwargs = {
            "text": text,
            "ref_audio": ref_path,
            "language": language,
            "chunk_size": chunk_size,
            "xvec_only": _FASTER_XVEC_ONLY,
            "non_streaming_mode": _FASTER_NON_STREAMING_MODE,
        }
        iterator = streaming_fn(**stream_kwargs)
        for item in iterator:
            # The upstream signature varies by release — accept either
            # (audio, sr) or (audio, sr, timing). Normalise to (np, sr).
            if isinstance(item, tuple):
                audio = item[0]
                sr = item[1] if len(item) > 1 else _sample_rate
            else:
                audio = item
                sr = _sample_rate
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            yield np.asarray(audio, dtype=np.float32), int(sr)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/v1/audio/speech/stream")
async def speech_stream(req: SpeechRequest):
    """Streaming variant of /v1/audio/speech.

    Body framing on the wire: a sequence of frames, each
    ``[4 bytes big-endian uint32 length][length bytes int16 PCM @ sample_rate]``.
    A zero-length frame marks end-of-stream. Clients read one frame at a
    time and can forward each chunk to listeners without waiting for the
    full synthesis. Drops the time-to-first-audio from the full RTF*duration
    to ~1 chunk (~667 ms at chunk_size=8).
    """
    global _fatal_error, _synth_request, _synth_started_at

    if _model is None:
        raise HTTPException(503, "Model not loaded")
    if _fatal_error:
        raise HTTPException(503, f"TTS worker unhealthy: {_fatal_error}")
    if not req.input or not req.input.strip():
        raise HTTPException(400, "Empty input text")

    t0 = time.monotonic()

    # Decode voice reference (same logic as the non-streaming endpoint).
    ref_audio = None
    ref_sr = 16000
    named_speaker: str | None = None
    MAX_VOICE_REF_SIZE = 1_000_000
    if req.voice and req.voice != "default" and len(req.voice) > 100:
        if len(req.voice) > MAX_VOICE_REF_SIZE:
            raise HTTPException(
                400,
                f"Voice reference too large ({len(req.voice)} bytes, max {MAX_VOICE_REF_SIZE})",
            )
        try:
            ref_audio, ref_sr = _decode_voice_ref(req.voice)
        except Exception as e:
            logger.warning("Failed to decode voice reference: %s", e)
    elif req.voice in _CUSTOM_VOICE_SPEAKERS:
        named_speaker = req.voice

    lang_map = {
        "en": "English",
        "ja": "Japanese",
        "zh": "Chinese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
    }
    language = lang_map.get(req.language, req.language)
    chunk_size = int(os.environ.get("TTS_STREAM_CHUNK_SIZE", "8"))

    if _backend != "faster" or (named_speaker and _is_custom_voice_model()):
        lock = _get_synth_lock()
        if lock.locked():
            raise HTTPException(429, "TTS synthesis already in progress")
        try:
            async with lock:
                _synth_started_at = time.monotonic()
                _synth_request = (
                    f"stream chars={len(req.input)} lang={language} "
                    f"voice={req.voice[:40]} backend={_backend}"
                )
                try:
                    audio = await asyncio.wait_for(
                        asyncio.to_thread(
                            _synthesize_request_sync,
                            req,
                            language=language,
                            ref_audio=ref_audio,
                            ref_sr=ref_sr,
                            named_speaker=named_speaker,
                        ),
                        timeout=_SERVER_SYNTH_TIMEOUT_S,
                    )
                    audio = _enforce_audio_budget(audio, _synth_request, req.input)
                except TimeoutError as e:
                    reason = (
                        f"synthesis timeout after {_SERVER_SYNTH_TIMEOUT_S:.1f}s ({_synth_request})"
                    )
                    logger.error("Streaming synthesis timed out: %s", reason)
                    _mark_fatal(reason)
                    raise HTTPException(504, reason) from e
                finally:
                    _synth_started_at = None
                    _synth_request = ""
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            reason = f"streaming synthesis failed: {type(e).__name__}: {e}"
            logger.exception("Streaming synthesis failed; marking worker fatal")
            _mark_fatal(reason)
            raise HTTPException(500, reason) from e

        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        data = pcm.tobytes()
        logger.info(
            "Synthesized framed %.1fs audio in %.0fms (RTF=%.2fx) backend=%s",
            len(pcm) / max(_sample_rate, 1),
            (time.monotonic() - t0) * 1000,
            (len(pcm) / max(_sample_rate, 1)) / max(time.monotonic() - t0, 0.001),
            _backend,
        )
        return Response(
            content=struct.pack(">I", len(data)) + data + struct.pack(">I", 0),
            media_type="application/octet-stream",
        )

    def _iter():
        global _fatal_error, _synth_request, _synth_started_at

        first_yield_t = None
        total_samples = 0
        budget_s = _audio_budget_s(req.input)
        max_samples = max(1, int(budget_s * _sample_rate))
        _synth_started_at = time.monotonic()
        _synth_request = (
            f"stream chars={len(req.input)} lang={language} "
            f"voice={req.voice[:40]} backend={_backend}"
        )
        try:
            if named_speaker and _is_custom_voice_model():
                # Named speakers don't have a documented streaming API yet;
                # synthesise the full clip and yield it as a single chunk
                # so the wire format stays consistent. Still wins over the
                # non-streaming endpoint by reusing the same client path.
                full = _synthesize_custom_voice(req.input, language, named_speaker)
                iterator = [(full, _sample_rate)]
            elif _backend == "faster":
                iterator = _synthesize_faster_streaming(
                    req.input, language, ref_audio, ref_sr, chunk_size
                )
            else:
                # Baseline has no streaming API — emit as one chunk so the
                # wire format is consistent.
                audio = _synthesize_baseline(req.input, language, ref_audio, ref_sr)
                iterator = [(audio, _sample_rate)]

            for audio_chunk, sr in iterator:
                if first_yield_t is None:
                    first_yield_t = time.monotonic()
                    logger.info(
                        "TTFA %.0fms (chunk_size=%d, lang=%s, chars=%d)",
                        (first_yield_t - t0) * 1000,
                        chunk_size,
                        language,
                        len(req.input),
                    )
                pcm = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                if total_samples + len(pcm) > max_samples:
                    allowed = max(0, max_samples - total_samples)
                    if allowed > 0:
                        pcm = pcm[:allowed]
                        data = pcm.tobytes()
                        total_samples += len(pcm)
                        yield struct.pack(">I", len(data)) + data
                    reason = (
                        f"streaming synthesis exceeded audio budget: "
                        f"{(total_samples + len(audio_chunk)) / max(_sample_rate, 1):.1f}s "
                        f"> {budget_s:.1f}s ({_synth_request})"
                    )
                    logger.warning(reason)
                    yield struct.pack(">I", 0)
                    return
                data = pcm.tobytes()
                total_samples += len(pcm)
                yield struct.pack(">I", len(data)) + data
            # End-of-stream sentinel
            yield struct.pack(">I", 0)
        except Exception as e:
            _mark_fatal(f"streaming synthesis failed: {type(e).__name__}: {e}")
            logger.exception("Streaming synthesis failed; marking worker fatal")
            # Best-effort EOS so clients blocked on frame reads unblock. The
            # next health check returns 503 and lets the parent restart us.
            yield struct.pack(">I", 0)
            return
        finally:
            _synth_started_at = None
            _synth_request = ""
        elapsed = time.monotonic() - t0
        duration_s = total_samples / max(_sample_rate, 1)
        logger.info(
            "Synthesized streaming %.1fs audio in %.0fms (RTF=%.2fx) backend=%s",
            duration_s,
            elapsed * 1000,
            duration_s / max(elapsed, 0.001),
            _backend,
        )

    return StreamingResponse(_iter(), media_type="application/octet-stream")


if __name__ == "__main__":
    port = int(os.environ.get("TTS_PORT", "8002"))
    logger.info("Starting TTS server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
