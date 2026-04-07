"""Qwen3-TTS voice cloning backend — frictionless voice synthesis.

Automatically captures voice characteristics from the first speech segments.
No enrollment or separate recording needed — the 4s audio chunks used for ASR
ARE the voice reference.

Platform support:
  - macOS (MLX): Qwen3-TTS-0.6B via mlx-audio (~1-2s latency)
  - GB10 (CUDA): Qwen3-TTS via vLLM-Omni (97ms streaming)
  - Fallback: disabled if neither available

Setup (macOS):
  1. Accept Qwen3-TTS license on HuggingFace
  2. Set HF_TOKEN environment variable
  3. pip install mlx-audio
"""

from __future__ import annotations

import logging
import tempfile
import time

import numpy as np
import soundfile as sf

from meeting_scribe.backends.base import TTSBackend

logger = logging.getLogger(__name__)


class Qwen3TTSBackend(TTSBackend):
    """Qwen3-TTS with zero-shot voice cloning.

    Auto-detects platform: MLX on macOS, vLLM endpoint on GB10.
    Voice reference extracted from conversation audio — no enrollment needed.
    """

    def __init__(self, vllm_url: str | None = None) -> None:
        self._mode: str = "disabled"  # "mlx" | "vllm" | "disabled"
        self._model = None
        self._processor = None
        self._vllm_url = vllm_url
        # Cache voice references per speaker
        self._voice_cache: dict[str, np.ndarray] = {}

    async def start(self) -> None:
        """Initialize TTS — try MLX first, then vLLM, then disable."""
        # Try MLX (macOS Apple Silicon)
        try:
            from mlx_audio.tts import load_model

            t0 = time.monotonic()
            result = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16")
            # load_model may return (model, processor) or (model, processor, tokenizer)
            if isinstance(result, tuple):
                self._model = result[0]
                self._processor = result[1] if len(result) > 1 else None
            else:
                self._model = result
            load_ms = (time.monotonic() - t0) * 1000
            self._mode = "mlx"
            logger.info("TTS: Qwen3-TTS-0.6B (MLX) loaded in %.0fms", load_ms)
            return
        except ImportError:
            logger.debug("MLX not available for TTS")
        except Exception as e:
            logger.warning("MLX TTS failed: %s", e)

        # Try vLLM endpoint (GB10)
        if self._vllm_url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5) as c:
                    r = await c.get(f"{self._vllm_url}/health")
                    if r.status_code == 200:
                        self._mode = "vllm"
                        logger.info("TTS: vLLM-Omni at %s", self._vllm_url)
                        return
            except Exception as e:
                logger.warning("vLLM TTS endpoint not available: %s", e)

        self._mode = "disabled"
        logger.info("TTS: disabled (no MLX or vLLM available)")

    async def stop(self) -> None:
        self._model = None
        self._processor = None
        self._voice_cache.clear()

    def cache_voice(self, speaker_id: str, audio_chunk: np.ndarray) -> None:
        """Cache a voice reference for a speaker. Called automatically from conversation audio."""
        if speaker_id not in self._voice_cache and len(audio_chunk) >= 16000:
            self._voice_cache[speaker_id] = audio_chunk[: 16000 * 4]  # max 4s
            logger.info(
                "Cached voice reference for speaker '%s' (%.1fs)",
                speaker_id,
                len(audio_chunk) / 16000,
            )

    def has_voice(self, speaker_id: str) -> bool:
        return speaker_id in self._voice_cache

    @property
    def available(self) -> bool:
        return self._mode != "disabled"

    async def synthesize(
        self,
        text: str,
        language: str,
        voice_reference: np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Synthesize speech from text, optionally cloning a voice.

        Args:
            text: Text to speak.
            language: "ja" or "en".
            voice_reference: Optional reference audio for voice cloning.
            sample_rate: Output sample rate.

        Returns:
            Float32 audio samples at the requested sample rate.
        """
        if self._mode == "mlx":
            return await self._synthesize_mlx(text, language, voice_reference)
        elif self._mode == "vllm":
            return await self._synthesize_vllm(text, language, voice_reference)
        else:
            return np.zeros(0, dtype=np.float32)

    async def _synthesize_mlx(
        self, text: str, language: str, voice_ref: np.ndarray | None
    ) -> np.ndarray:
        """Synthesize via MLX on Apple Silicon."""
        try:
            from mlx_audio.tts import generate

            # Prepare voice reference as temp wav if provided
            ref_path = None
            if voice_ref is not None and len(voice_ref) > 0:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, voice_ref, 16000)
                    ref_path = f.name

            t0 = time.monotonic()
            result = generate(
                text=text,
                model_id_or_path="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
                voice=ref_path,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Result may be tuple (audio, sr) or just audio
            if isinstance(result, tuple):
                audio, _sr = result
            else:
                audio = result

            # Convert to numpy if needed
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32).flatten()

            logger.debug("TTS synthesized %d samples in %.0fms", len(audio), elapsed_ms)
            return audio

        except Exception as e:
            logger.warning("MLX TTS synthesis failed: %s", e)
            return np.zeros(0, dtype=np.float32)

    async def _synthesize_vllm(
        self, text: str, language: str, voice_ref: np.ndarray | None
    ) -> np.ndarray:
        """Synthesize via vLLM-Omni endpoint on GB10."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30) as c:
                body = {
                    "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
                    "input": text,
                    "voice": "default",
                }
                r = await c.post(f"{self._vllm_url}/v1/audio/speech", json=body)
                r.raise_for_status()

                # Response is audio bytes
                audio = np.frombuffer(r.content, dtype=np.float32)
                return audio

        except Exception as e:
            logger.warning("vLLM TTS synthesis failed: %s", e)
            return np.zeros(0, dtype=np.float32)
