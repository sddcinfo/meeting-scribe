"""Minimal FastAPI wrapper for NeMo Sortformer diarization.

Exposes:
    GET  /health              — health check
    POST /v1/diarize          — diarize audio chunk, return speaker segments
    POST /v1/embed            — extract speaker embedding from audio

Audio format: raw s16le PCM at the sample rate specified in X-Sample-Rate header.
"""

from __future__ import annotations

import logging
import os
import uuid

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Sortformer Diarization")

# Global model state
_diarization_model = None
_embedding_model = None
_max_speakers = int(os.environ.get("SORTFORMER_MAX_SPEAKERS", "4"))


@app.on_event("startup")
async def load_models() -> None:
    """Load Sortformer and ECAPA-TDNN models on startup."""
    global _diarization_model, _embedding_model

    logger.info("Loading Sortformer diarization model...")
    try:
        import nemo.collections.asr as nemo_asr

        # Load pre-trained Sortformer for online diarization
        _diarization_model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(
            "nvidia/sortformer_diar_4spk_v1"
        )
        _diarization_model.eval()
        if torch.cuda.is_available():
            _diarization_model = _diarization_model.cuda()
        logger.info("Sortformer loaded (GPU=%s)", torch.cuda.is_available())
    except Exception as e:
        logger.error("Failed to load Sortformer: %s", e)

    logger.info("Loading ECAPA-TDNN embedding model...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        _embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        logger.info("ECAPA-TDNN loaded")
    except Exception as e:
        logger.error("Failed to load ECAPA-TDNN: %s", e)


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "ok" if _diarization_model is not None else "degraded",
            "diarization_model": _diarization_model is not None,
            "embedding_model": _embedding_model is not None,
        }
    )


@app.post("/v1/diarize")
async def diarize(request: Request) -> JSONResponse:
    """Diarize an audio chunk.

    Accepts raw s16le PCM in the request body.
    Returns speaker segments with cluster_id and optional embedding.
    """
    if _diarization_model is None:
        return JSONResponse({"error": "Diarization model not loaded"}, status_code=503)

    body = await request.body()
    sample_rate = int(request.headers.get("X-Sample-Rate", "16000"))
    max_speakers = int(request.headers.get("X-Max-Speakers", str(_max_speakers)))

    # Decode s16le PCM
    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    if torch.cuda.is_available():
        audio_tensor = audio_tensor.cuda()

    # Run Sortformer inference
    try:
        with torch.no_grad():
            # Sortformer returns speaker labels per frame
            preds = _diarization_model.forward(
                input_signal=audio_tensor,
                input_signal_length=torch.tensor([len(audio)]),
            )

        # Parse predictions into segments
        # Sortformer output shape: (batch, num_frames, max_speakers)
        segments = _parse_sortformer_output(
            preds, audio, sample_rate, max_speakers
        )

        # Optionally extract embeddings per segment
        if _embedding_model is not None:
            for seg in segments:
                start_sample = int(seg["start"] * sample_rate)
                end_sample = int(seg["end"] * sample_rate)
                if end_sample > start_sample + sample_rate // 4:  # Min 250ms
                    seg_audio = audio[start_sample:end_sample]
                    seg_tensor = torch.from_numpy(seg_audio).unsqueeze(0).float()
                    emb = _embedding_model.encode_batch(seg_tensor)
                    seg["embedding"] = emb.squeeze().cpu().numpy().tolist()

    except Exception as e:
        logger.error("Diarization inference failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"segments": segments})


@app.post("/v1/embed")
async def embed(request: Request) -> JSONResponse:
    """Extract speaker embedding from audio.

    Used for speaker enrollment — extracts ECAPA-TDNN embedding
    from reference audio for later verification.
    """
    if _embedding_model is None:
        return JSONResponse({"error": "Embedding model not loaded"}, status_code=503)

    body = await request.body()
    speaker_name = request.headers.get("X-Speaker-Name", "Unknown")

    # Decode s16le PCM
    audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()

    try:
        emb = _embedding_model.encode_batch(audio_tensor)
        embedding = emb.squeeze().cpu().numpy().tolist()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    enrollment_id = str(uuid.uuid4())
    logger.info("Extracted embedding for speaker: %s (id=%s)", speaker_name, enrollment_id)

    return JSONResponse({
        "embedding": embedding,
        "enrollment_id": enrollment_id,
        "speaker_name": speaker_name,
    })


def _parse_sortformer_output(
    preds: torch.Tensor,
    audio: np.ndarray,
    sample_rate: int,
    max_speakers: int,
) -> list[dict]:
    """Parse Sortformer frame-level predictions into speaker segments.

    Sortformer outputs per-frame speaker probabilities. We threshold
    and merge consecutive frames into segments.
    """
    # preds shape: (batch, num_frames, max_speakers) — sigmoid probabilities
    if isinstance(preds, tuple):
        preds = preds[0]

    probs = torch.sigmoid(preds[0]).cpu().numpy()  # (num_frames, max_speakers)
    num_frames = probs.shape[0]
    audio_duration = len(audio) / sample_rate
    frame_duration = audio_duration / num_frames if num_frames > 0 else 0

    threshold = 0.5
    segments: list[dict] = []

    for speaker_id in range(min(max_speakers, probs.shape[1])):
        active = probs[:, speaker_id] > threshold
        if not active.any():
            continue

        # Find contiguous active regions
        changes = np.diff(active.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if active[0]:
            starts = np.concatenate([[0], starts])
        if active[-1]:
            ends = np.concatenate([ends, [num_frames]])

        for start_frame, end_frame in zip(starts, ends, strict=False):
            start_sec = start_frame * frame_duration
            end_sec = end_frame * frame_duration
            confidence = float(probs[start_frame:end_frame, speaker_id].mean())

            segments.append({
                "speaker_id": speaker_id,
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "confidence": round(confidence, 3),
            })

    # Sort by start time
    segments.sort(key=lambda s: s["start"])
    return segments
