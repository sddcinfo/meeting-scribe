"""pyannote.audio speaker diarization server.

Wraps pyannote's pretrained pipeline in a FastAPI server that matches
the same HTTP API as the NeMo Sortformer container (/health, /v1/diarize).
This allows meeting-scribe's SortformerBackend to work unchanged.

Runs on NVIDIA GB10 (aarch64, Blackwell) with SM_121 patches applied.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pyannote.core import Annotation

# Process-level lock — only ONE diarize call can hold the GPU at a
# time. Sortformer's internal CUDA streams are not thread-safe and
# concurrent calls cause CUDA 'unknown error' wedges. This lock
# serializes HTTP requests at the app layer so the model never sees
# overlapping calls even if the client fires multiple requests.
_pipeline_lock = asyncio.Lock()

logger = logging.getLogger("pyannote-server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="pyannote Speaker Diarization")

# Global pipeline (loaded once at startup)
pipeline = None

SAMPLE_RATE = 16000
MAX_SPEAKERS = int(os.environ.get("DIARIZE_MAX_SPEAKERS", "4"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def _apply_blackwell_patches():
    """Apply SM_121 (Blackwell) compatibility patches.

    Blackwell GPUs (SM_121) are binary-compatible with Hopper (SM_90)
    but some PyTorch code paths check the arch and fail. We spoof
    Hopper to bypass these checks.

    KNOWN GAP (see ../../../../UPGRADE-NOTES-2026-04.md): this spoof
    only covers Python-level `torch.cuda.get_device_capability` checks.
    pyannote 4.x's `powerset.to_multilabel` calls into a C++/CUDA
    `torch.nn.functional.one_hot` kernel that checks the real arch and
    raises `torch.AcceleratorError: CUDA error: unknown error` under
    certain shared-GPU-contention conditions. Symptom: /health stays
    200 (pipeline object is alive) but /v1/diarize returns 500.
    Mitigation: `docker compose up -d --force-recreate pyannote-diarize`
    clears the CUDA state and inference resumes. No code fix available
    until pyannote upstream gets SM_121 support or NGC ships a newer
    base image that works with the GB10 driver.
    """
    original_capability = torch.cuda.get_device_capability

    def patched_capability(device=None):
        real = original_capability(device)
        # SM_121 (Blackwell) → report as SM_90 (Hopper)
        if real[0] >= 12:
            return (9, 0)
        return real

    torch.cuda.get_device_capability = patched_capability
    logger.info("Applied Blackwell SM_121 → SM_90 compatibility patch")

    # CUDA `one_hot` kernel does its own arch check that bypasses our
    # Python-level spoof and raises `CUDA error: unknown error` under
    # shared-GPU contention (e.g. when the 35B vLLM is also running
    # on the same GPU). Pyannote 4.x's `powerset.to_multilabel` calls
    # one_hot on tiny tensors (shape (T, N) where N≤10 speakers), so
    # forcing it to CPU costs ~microseconds but eliminates the wedge
    # entirely. We patch `torch.nn.functional.one_hot` to always run
    # on CPU and move the result back to the input's device.
    _orig_one_hot = torch.nn.functional.one_hot

    def _one_hot_via_cpu(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
        if tensor.is_cuda:
            result = _orig_one_hot(tensor.cpu(), num_classes=num_classes)
            return result.to(tensor.device)
        return _orig_one_hot(tensor, num_classes=num_classes)

    torch.nn.functional.one_hot = _one_hot_via_cpu
    logger.info(
        "Patched torch.nn.functional.one_hot → CPU fallback "
        "(prevents SM_121 pyannote wedge under GPU contention)"
    )


@app.on_event("startup")
async def startup():
    global pipeline, embedding_model

    if torch.cuda.is_available():
        _apply_blackwell_patches()
        device = "cuda"
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("Running on CPU (no CUDA)")

    # Load pyannote speaker diarization pipeline
    from pyannote.audio import Pipeline

    logger.info("Loading pyannote speaker-diarization-3.1...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN or None,
    )
    pipeline.to(torch.device(device))
    logger.info("Pipeline loaded on %s", device)


@app.get("/health")
async def health():
    if pipeline is None:
        return JSONResponse({"status": "loading"}, status_code=503)

    cuda_ok = True
    cuda_error = None
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda:0")
            del t
            torch.cuda.synchronize()
        except Exception as e:
            cuda_ok = False
            cuda_error = str(e)

    resp = {
        "status": "ok" if cuda_ok else "cuda_error",
        "diarization_model": True,
        "embedding_model": True,
    }
    if torch.cuda.is_available():
        resp["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        resp["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
    if cuda_error:
        resp["error"] = cuda_error
    if not cuda_ok:
        return JSONResponse(resp, status_code=503)
    return JSONResponse(resp)


@app.post("/v1/diarize")
async def diarize(request: Request):
    """Diarize audio. Accepts raw s16le PCM or WAV.

    Headers:
        X-Sample-Rate: sample rate (default 16000)
        X-Max-Speakers: max speakers to detect (default 4)
        X-Min-Speakers: minimum speakers hint (default 2, 0 to disable)
        X-Num-Speakers: exact speaker count, overrides min/max (default none)

    Returns:
        {segments: [{speaker_id, start, end, confidence, embedding?}]}
    """
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not loaded"}, status_code=503)

    body = await request.body()
    sample_rate = int(request.headers.get("X-Sample-Rate", str(SAMPLE_RATE)))
    max_speakers = int(request.headers.get("X-Max-Speakers", str(MAX_SPEAKERS)))
    # min_speakers hint — without it, pyannote collapses ambiguous clustering
    # into 1 speaker. But it REJECTS requests where audio is too short to
    # contain N speakers, so we must auto-disable for short audio.
    # Clients can pass X-Min-Speakers: 0 to disable entirely.
    min_speakers_hdr = request.headers.get("X-Min-Speakers")
    min_speakers = int(min_speakers_hdr) if min_speakers_hdr is not None else 2
    num_speakers_hdr = request.headers.get("X-Num-Speakers")
    num_speakers = int(num_speakers_hdr) if num_speakers_hdr is not None else 0

    # Convert s16le PCM to float32
    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    audio_duration_s = len(audio) / sample_rate

    # pyannote 4.x strict input validation rejects waveform with
    # shape[0] > shape[1] (see pyannote/audio/core/io.py:174). Empty
    # audio ((1, 0) tensor) hits this because 1 > 0. Live WebSocket
    # clients sometimes send empty chunks during connection churn —
    # v3 was lenient here, v4 raises ValueError. Return empty result
    # early instead of hitting the pipeline.
    if len(audio) == 0:
        return JSONResponse({
            "segments": [],
            "num_speakers": 0,
            "audio_duration_s": 0.0,
            "processing_ms": 0,
        })

    # Auto-disable min_speakers for short audio (live streaming chunks) —
    # pyannote needs ~15s+ to reliably detect 2 speakers. Below that it
    # errors with "audio file too short to contain 2 or more speakers".
    if audio_duration_s < 15.0 and min_speakers > 1:
        logger.debug(
            "Audio too short (%.1fs) for min_speakers=%d — disabling hint",
            audio_duration_s, min_speakers,
        )
        min_speakers = 0

    # pyannote expects a dict with "waveform" and "sample_rate"
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)

    # Build pyannote kwargs based on what the client asked for
    pipeline_kwargs: dict = {}
    if num_speakers > 0:
        # Exact count — overrides any min/max
        pipeline_kwargs["num_speakers"] = num_speakers
    else:
        pipeline_kwargs["max_speakers"] = max_speakers
        if min_speakers > 0 and min_speakers <= max_speakers:
            pipeline_kwargs["min_speakers"] = min_speakers

    # Run the synchronous pipeline call in a threadpool so uvicorn's
    # event loop stays responsive, and serialize with a per-process
    # lock so concurrent clients never race the GPU. ``inference_mode``
    # suppresses autograd graph buildup (pyannote has no gradients).
    # Bind `waveform`/`sample_rate` into the closure at definition time.
    # Ruff flagged them as undefined via F821 when referenced implicitly
    # inside the nested `_run_pipeline` body.  Passing them as defaults
    # also makes the capture explicit and guarantees the closure
    # doesn't re-read a possibly-reassigned outer name.
    def _run_pipeline(waveform=waveform, sample_rate=sample_rate):
        with torch.inference_mode():
            out = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                **pipeline_kwargs,
            )
        # Flush any in-flight kernels BEFORE returning so the next
        # call can't step on half-finished work.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out

    t0 = time.monotonic()
    async with _pipeline_lock:
        try:
            raw_output = await run_in_threadpool(_run_pipeline)
        finally:
            # Drop the input tensor reference + empty CUDA cache so
            # pyannote's cached activations don't accumulate across
            # calls. This is the single biggest fix for the "wedges
            # after N calls" bug.
            del waveform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elapsed_ms = (time.monotonic() - t0) * 1000

    # pyannote 4.x returns DiarizeOutput; 3.x returns Annotation directly
    speaker_embeddings = None
    if isinstance(raw_output, Annotation):
        annotation = raw_output
    else:
        annotation = raw_output.speaker_diarization
        speaker_embeddings = raw_output.speaker_embeddings

    # Convert pyannote Annotation to our API format
    segments = []
    speaker_map = {}
    labels = annotation.labels()

    # Build per-speaker embedding lookup from pipeline output
    emb_by_label = {}
    if speaker_embeddings is not None and len(labels) > 0:
        for i, label in enumerate(labels):
            if i < len(speaker_embeddings):
                emb = speaker_embeddings[i]
                if not np.any(np.isnan(emb)):
                    emb_by_label[label] = emb.tolist()

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = len(speaker_map)

        seg = {
            "speaker_id": speaker_map[speaker],
            "start": turn.start,
            "end": turn.end,
            "confidence": 1.0,
        }

        # Attach per-speaker embedding from pipeline output
        if speaker in emb_by_label:
            seg["embedding"] = emb_by_label[speaker]

        segments.append(seg)

    logger.info(
        "Diarized %.1fs audio in %.0fms → %d segments, %d speakers",
        len(audio) / sample_rate,
        elapsed_ms,
        len(segments),
        len(speaker_map),
    )

    return JSONResponse({
        "segments": segments,
        "num_speakers": len(speaker_map),
        "audio_duration_s": len(audio) / sample_rate,
        "processing_ms": round(elapsed_ms),
    })


@app.post("/v1/embed")
async def embed(request: Request):
    """Extract speaker embedding from audio via diarization pipeline.

    Runs the full pipeline on a short clip and returns the dominant
    speaker's embedding. For best results, send 2-10s of single-speaker audio.

    Returns: {embedding: [float...]}
    """
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not loaded"}, status_code=503)

    body = await request.body()
    sample_rate = int(request.headers.get("X-Sample-Rate", str(SAMPLE_RATE)))

    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio).unsqueeze(0)

    # See _run_pipeline above for the explicit default-capture rationale.
    def _run_embed(waveform=waveform, sample_rate=sample_rate):
        with torch.inference_mode():
            out = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                max_speakers=1,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out

    async with _pipeline_lock:
        try:
            raw_output = await run_in_threadpool(_run_embed)
        finally:
            del waveform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if isinstance(raw_output, Annotation):
        return JSONResponse({"error": "Pipeline too old for embeddings"}, status_code=501)

    if raw_output.speaker_embeddings is None or len(raw_output.speaker_embeddings) == 0:
        return JSONResponse({"error": "No speaker detected"}, status_code=422)

    emb = raw_output.speaker_embeddings[0]
    if np.any(np.isnan(emb)):
        return JSONResponse({"error": "Embedding contains NaN"}, status_code=422)

    return JSONResponse({
        "embedding": emb.tolist(),
    })


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DIARIZE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
