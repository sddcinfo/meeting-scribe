"""Configuration for meeting-scribe server.

Backend selection, quotas, and runtime settings. All configurable
via environment variables with sensible defaults for POC.

Platform modularity:
    macOS (POC):  asr_backend_type=mlx-whisper, translate via Ollama
    GB10 (Prod):  asr_backend_type=faster-whisper, translate via vLLM
    Remote:       asr_backend_type=openai-api (vLLM OpenAI-compat endpoint)
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


def _detect_asr_backend() -> str:
    """Auto-detect the best ASR backend for the current platform."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx-whisper"
    return "faster-whisper"


@dataclass
class ServerConfig:
    """Main server configuration."""

    # Network
    host: str = "127.0.0.1"
    port: int = 8080

    # Backend selection: "poc" or "prod"
    backend_mode: str = "poc"

    # ASR — WhisperLiveKit
    asr_backend_type: str = "auto"  # "mlx-whisper" | "faster-whisper" | "openai-api" | "auto"
    asr_model: str = (
        "medium"  # "medium" (fast+good JA, RTF=0.04) | "tiny" (fastest) | "large-v3-turbo"
    )
    asr_language: str = "auto"  # auto-detect; set to "ja" or "en" to force
    supported_languages: str = "ja,en"  # comma-separated; only these are accepted for translation
    asr_streaming_policy: str = "localagreement"  # "localagreement" (robust, no stale buffer) | "simulstreaming" (low latency)

    # Diarization (via WhisperLiveKit Sortformer)
    diarize_enabled: bool = False  # Sortformer needs pyannote models; disabled by default
    diarize_backend: str = "sortformer"  # "sortformer" | "diart"

    # Translation
    translate_enabled: bool = True
    translate_backend: str = "ct2"  # "ct2" (NLLB, CPU) | "vllm" (LLM, GPU) | "ollama" (fallback)
    translate_vllm_url: str = "http://localhost:8000"  # vLLM endpoint
    translate_vllm_model: str = ""  # auto-detect from vLLM if empty
    translate_model: str = "qwen3.5:27b-nvfp4"  # Ollama fallback model
    translate_ollama_url: str = "http://localhost:11434"

    # Storage
    meetings_dir: Path = Path(__file__).parent.parent.parent / "meetings"
    journal_fsync_seconds: int = 5
    retention_days: int = 30

    # Translation queue
    translate_queue_maxsize: int = 50
    translate_queue_concurrency: int = 2
    translate_timeout_seconds: int = 30  # NLLB is fast, reduce from 60s

    @property
    def resolved_asr_backend(self) -> str:
        """Resolve 'auto' to the platform-appropriate backend."""
        if self.asr_backend_type == "auto":
            return _detect_asr_backend()
        return self.asr_backend_type

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Load configuration from environment variables."""
        return cls(
            host=os.environ.get("SCRIBE_HOST", cls.host),
            port=int(os.environ.get("SCRIBE_PORT", str(cls.port))),
            backend_mode=os.environ.get("SCRIBE_BACKEND", cls.backend_mode),
            asr_backend_type=os.environ.get("SCRIBE_ASR_BACKEND", cls.asr_backend_type),
            asr_model=os.environ.get("SCRIBE_ASR_MODEL", cls.asr_model),
            asr_language=os.environ.get("SCRIBE_ASR_LANGUAGE", cls.asr_language),
            asr_streaming_policy=os.environ.get("SCRIBE_ASR_POLICY", cls.asr_streaming_policy),
            diarize_enabled=os.environ.get("SCRIBE_DIARIZE", "false").lower() == "true",
            translate_enabled=os.environ.get("SCRIBE_TRANSLATE", "true").lower() == "true",
            translate_backend=os.environ.get("SCRIBE_TRANSLATE_BACKEND", cls.translate_backend),
            translate_vllm_url=os.environ.get("SCRIBE_TRANSLATE_VLLM_URL", cls.translate_vllm_url),
            translate_vllm_model=os.environ.get(
                "SCRIBE_TRANSLATE_VLLM_MODEL", cls.translate_vllm_model
            ),
            translate_model=os.environ.get("SCRIBE_TRANSLATE_MODEL", cls.translate_model),
            translate_ollama_url=os.environ.get("SCRIBE_OLLAMA_URL", cls.translate_ollama_url),
            meetings_dir=Path(os.environ.get("SCRIBE_MEETINGS_DIR", str(cls.meetings_dir))),
        )
