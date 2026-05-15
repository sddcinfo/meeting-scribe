"""Speaker-audio helpers shared by enrollment and attribution paths.

These wrap the vLLM ASR endpoint and SpeechBrain ECAPA-TDNN embedding
in a small, reusable surface that both ``/api/room/enroll/*`` and the
runtime speaker-attribution pipeline call. The ECAPA classifier is
cached at module scope — SpeechBrain's ``EncoderClassifier.from_hparams``
loads ~40 MB of weights and moves them to CUDA on each call, so we
pay that cost once.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from meeting_scribe.runtime import state
from meeting_scribe.speaker.name_extraction import extract_name as _extract_name_from_text

logger = logging.getLogger(__name__)

_ECAPA_CLASSIFIER: Any = None


async def _asr_transcribe(audio: np.ndarray) -> str:
    """Run ASR on a PCM audio buffer and return plain text via the
    already-running vLLM ASR endpoint."""
    import base64
    import io

    import httpx
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{state.config.asr_vllm_url}/v1/chat/completions",
            json={
                "model": state.config.asr_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Transcribe the audio in the original spoken language. Do not translate. Preserve the original script (e.g. Japanese in kanji/hiragana, Korean in hangul).",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": "wav"},
                            }
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 64,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    if "<asr_text>" in raw:
        _, _, text = raw.partition("<asr_text>")
        return text.strip()
    return raw.strip()


async def _transcribe_name(audio: np.ndarray) -> str:
    """Transcribe enrollment audio to extract the speaker's name.

    Used by the legacy fixed-duration flow. Runs ASR, then delegates
    name extraction to ``extract_name``, falling back to a trimmed
    version of the raw transcription when no pattern matches so the
    old flow still produces *some* name.
    """
    text = await _asr_transcribe(audio)
    logger.info("Enrollment transcription: '%s'", text)
    if not text:
        return ""
    name = _extract_name_from_text(text)
    if name:
        return name
    # Fallback: use the raw transcription (trimmed) so the legacy flow
    # keeps producing something rather than dropping silently.
    lower = text.lower()
    for filler in ("hello", "hi", "hey", "um", "uh", "so", "okay", "well"):
        if lower.startswith(filler):
            text = text[len(filler) :].strip().lstrip(",").lstrip(".").strip()
            break
    return text[:20].strip().rstrip(".")


async def _extract_embedding(audio: np.ndarray) -> np.ndarray:
    """Extract speaker embedding using SpeechBrain ECAPA-TDNN.

    Caches the classifier at module scope: SpeechBrain's
    ``EncoderClassifier.from_hparams`` loads ~40 MB of weights and moves
    them to CUDA on each call, which otherwise cold-pays ~1.5 s per
    enrollment and racks up GPU memory churn when a user enrolls
    several people in a row.
    """
    global _ECAPA_CLASSIFIER
    try:
        import torch
        from speechbrain.inference.speaker import (  # type: ignore[import-not-found]
            EncoderClassifier,
        )

        if _ECAPA_CLASSIFIER is None:
            _ECAPA_CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/speechbrain-ecapa",
            )
        classifier = _ECAPA_CLASSIFIER
        # Feed the audio to whichever device the classifier landed on
        # (CUDA when SpeechBrain picks a GPU, CPU otherwise). Moving
        # the input to the correct device up front avoids a silent
        # mixed-device path inside encode_batch.
        device = (
            next(classifier.mods.parameters()).device
            if hasattr(classifier, "mods")
            else torch.device("cpu")
        )
        waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
        # encode_batch returns a tensor on ``device``. Move back to CPU
        # before .numpy(); calling .numpy() directly on a CUDA tensor
        # raises "can't convert cuda:0 device type tensor to numpy".
        # Prior to this fix every enrollment since the SpeechBrain
        # integration silently fell through to the MFCC-based
        # _simple_embedding fallback, which made speaker matching at
        # meeting time unreliable (the "first capture worked, second
        # didn't" regression on 2026-04-21).
        embedding = classifier.encode_batch(waveform).squeeze().detach().cpu().numpy()
        return embedding
    except ImportError:
        logger.warning("SpeechBrain not installed — using simple embedding")
        return _simple_embedding(audio)


def _simple_embedding(audio: np.ndarray) -> np.ndarray:
    """Fallback: extract simple audio features as a pseudo-embedding.

    Uses MFCCs as a basic voice fingerprint. Not as good as ECAPA-TDNN
    but works without SpeechBrain.
    """
    # Compute basic spectral features
    n_fft = 512
    # Simple energy + zero-crossing based features
    frame_size = 16000  # 1s frames
    n_frames = max(1, len(audio) // frame_size)
    features = []
    for i in range(n_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        rms = float(np.sqrt(np.mean(frame**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))))
        # Simple FFT features (16 bins)
        fft = np.abs(np.fft.rfft(frame, n=n_fft))[: n_fft // 2]
        spectral = np.array([np.mean(fft[j::16]) for j in range(16)])
        features.append(np.concatenate([[rms, zcr], spectral]))

    embedding = np.mean(features, axis=0).astype(np.float32)
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding
