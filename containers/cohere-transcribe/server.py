"""Cohere Transcribe (cohere-transcribe-03-2026) FastAPI server.

Exposes the same ``POST /v1/chat/completions`` shape that
``benchmarks/asr_accuracy_latency.py:73`` already calls.  This lets
Track A reuse the existing ASR harness with zero code change — just
point ``--url`` at this container's port.

Request shape (subset we honor):

    {
      "model": "auto",
      "messages": [
        {"role": "system", "content": "Transcribe in the spoken language."},
        {"role": "user", "content": [
          {"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}
        ]}
      ],
      ...
    }

Response shape:

    {"choices": [{"message": {"content": "<transcript>"}}]}

The actual model load uses ``transformers`` rather than vLLM (Cohere
Transcribe ships transformers-only at the time of writing), pinned
via the ``COHERE_REVISION`` env var per the S3 reproducibility rule.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("cohere-transcribe")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.environ.get("COHERE_MODEL_ID", "CohereLabs/cohere-transcribe-03-2026")
MODEL_REVISION = os.environ.get("COHERE_REVISION", "") or None
HF_TOKEN = os.environ.get("HF_TOKEN", "") or None

app = FastAPI(title="Cohere Transcribe (bench)")
_model = None
_processor = None


@app.on_event("startup")
async def _load() -> None:
    global _model, _processor
    # Cohere ships a custom ``CohereAsrForConditionalGeneration`` class
    # in transformers; the AutoModelForSpeechSeq2Seq path does not work
    # for this checkpoint.  Confirmed via the model card on HF:
    # https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration  # type: ignore

    rev_label = f" @ {MODEL_REVISION[:8]}" if MODEL_REVISION else ""
    logger.info("Loading %s%s ...", MODEL_ID, rev_label)
    kwargs: dict = {"token": HF_TOKEN}
    if MODEL_REVISION:
        kwargs["revision"] = MODEL_REVISION
    _processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
    _model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        **kwargs,
    )
    _model.eval()
    logger.info("Cohere Transcribe ready.")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok" if _model is not None else "loading"})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return await health()


def _decode_wav(b64: str) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(b64)
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


@app.post("/v1/chat/completions")
async def transcribe(request: Request) -> JSONResponse:
    if _model is None or _processor is None:
        raise HTTPException(503, "model loading")

    body = await request.json()
    messages = body.get("messages") or []
    audio_b64: str | None = None
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    audio_b64 = part["input_audio"]["data"]
                    break
        if audio_b64:
            break
    if not audio_b64:
        raise HTTPException(400, "no input_audio in messages")

    audio, sr = _decode_wav(audio_b64)
    target_sr = getattr(_processor, "sampling_rate", 16000)
    if sr != target_sr:
        # Lazy import — librosa pulls a slow load.
        import librosa  # type: ignore

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Cohere processor takes a `language` hint per the model card.
    # We accept an optional `language` field in the request body
    # (defaults to None → auto-detect).
    language = body.get("language")

    t0 = time.monotonic()
    proc_kwargs: dict = {"sampling_rate": target_sr, "return_tensors": "pt"}
    if language:
        proc_kwargs["language"] = language
    inputs = _processor(audio, **proc_kwargs)
    inputs = inputs.to(_model.device, dtype=_model.dtype)
    with torch.inference_mode():
        gen = _model.generate(**inputs, max_new_tokens=int(body.get("max_tokens", 256)))
    # The Cohere processor's ``decode`` returns a list-of-strings (one
    # per batch element), not a single string.  Take the first element
    # since we only ever pass a single audio per request.
    decoded = _processor.decode(gen, skip_special_tokens=True)
    text = decoded[0].strip() if isinstance(decoded, list) else decoded.strip()
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    logger.info(
        "Transcribed %.1fs audio in %.0fms (%d chars)",
        len(audio) / target_sr,
        elapsed_ms,
        len(text),
    )
    return JSONResponse(
        {
            "model": MODEL_ID,
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"audio_seconds": len(audio) / target_sr, "elapsed_ms": elapsed_ms},
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ASR_PORT", "8013"))
    uvicorn.run(app, host="0.0.0.0", port=port)
