"""Configuration for meeting-scribe server (GB10-only).

Backend selection, quotas, and runtime settings. All configurable
via environment variables with sensible defaults for GB10 production.

Backends:
    ASR:        Qwen3-ASR-1.7B via vLLM
    Translate:  vLLM (Qwen3.6-35B-A3B-FP8)
    Diarize:    pyannote.audio
    TTS:        Qwen3-TTS via faster-qwen3-tts

Profiles:
    SCRIBE_PROFILE=gb10 sets all backends to GB10 production mode.
    Individual env vars override profile defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_captive_http_port(default_value: int) -> int:
    """Resolve ``captive_http_port`` from env. ``SCRIBE_CAPTIVE_HTTP_PORT``
    wins. ``SCRIBE_GUEST_PORT`` is the deprecated fallback, kept for
    one release with a one-shot deprecation warning so operators
    pinned to the old name don't lose traffic mid-upgrade. Invalid
    values fall back to ``default_value`` rather than crashing
    startup."""
    new_val = os.environ.get("SCRIBE_CAPTIVE_HTTP_PORT", "").strip()
    legacy_val = os.environ.get("SCRIBE_GUEST_PORT", "").strip()

    if legacy_val:
        logger.warning(
            "SCRIBE_GUEST_PORT is deprecated; use SCRIBE_CAPTIVE_HTTP_PORT. "
            "The legacy name will be dropped in the next major version."
        )

    candidate = new_val or legacy_val
    if not candidate:
        return default_value
    try:
        return int(candidate)
    except ValueError:
        logger.warning(
            "Invalid captive HTTP port %r; falling back to default %d",
            candidate,
            default_value,
        )
        return default_value


# GB10 profile defaults — highest-end settings
_GB10_DEFAULTS: dict[str, object] = {
    "backend_mode": "prod",
    "host": "0.0.0.0",
    "asr_model": "Qwen/Qwen3-ASR-1.7B",
    "asr_vllm_url": "http://localhost:8003",
    "diarize_enabled": True,
    "diarize_backend": "sortformer",
    "diarize_url": "http://localhost:8001",
    "translate_backend": "vllm",
    "translate_vllm_url": "http://localhost:8010",  # Shared with autosre coding agent
    # TTS endpoint. The backend still accepts a comma-separated pool for
    # experiments, but GB10 reliability mode defaults to one resident
    # faster-qwen3-tts container. 2026-05-09 live testing showed two
    # resident TTS models can push the full ASR/translation/diarize stack
    # into CUDA-OOM territory.
    "tts_vllm_url": "http://localhost:8002",
    "name_extraction_backend": "auto",
    "translate_queue_concurrency": 4,
}


@dataclass
class ServerConfig:
    """Main server configuration."""

    # Network
    #
    # v1.0 unified single-listener model:
    #
    #   Main app:      HTTPS bound to ``0.0.0.0:443`` so admin
    #                  reaches the appliance from the AP
    #                  (``10.42.0.1``) AND the LAN at the same
    #                  port (``IP_FREEBIND`` so the socket pre-binds
    #                  before NetworkManager activates the AP IP).
    #   Captive sub-app: HTTP bound to the AP IP on
    #                  ``captive_http_port`` (default 80). Issues 308
    #                  redirects to the canonical HTTPS URL plus the
    #                  OS captive-portal probe paths. Stays AP-IP-
    #                  scoped because LAN clients have no reason to
    #                  hit the captive surface.
    #
    # ``host`` is retained for tests that spin up uvicorn through
    # paths other than ``server.main()`` (which uses its own bind).
    host: str = "0.0.0.0"
    port: int = 443
    # Was ``guest_port`` (env: SCRIBE_GUEST_PORT) under the dead
    # dual-listener model. Renamed to ``captive_http_port`` (env:
    # SCRIBE_CAPTIVE_HTTP_PORT) so the field name reflects what the
    # listener actually does in v1.0. The legacy env var is still
    # honoured with a deprecation warning — see
    # ``ServerConfig.from_env``.
    captive_http_port: int = 80

    # Backend selection: "prod"
    backend_mode: str = "prod"

    # ASR — Qwen3-ASR via vLLM
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    asr_vllm_url: str = "http://localhost:8003"  # vLLM Qwen3-ASR endpoint
    asr_language: str = "auto"  # auto-detect; supports all languages in LANGUAGE_REGISTRY
    # Default meeting languages. Accepts "en" (monolingual) or "en,ja"
    # (bilingual pair). Configurable per meeting via SCRIBE_LANGUAGE_PAIR.
    # Invalid values log a loud WARNING at startup and fall back to the
    # built-in default — see ``parse_languages`` in languages.py.
    default_language_pair: str = "en,ja"

    # Diarization
    diarize_enabled: bool = True  # pyannote.audio on GB10
    diarize_backend: str = "sortformer"  # "sortformer" (pyannote server uses same API)
    diarize_url: str = "http://localhost:8001"  # Diarization container endpoint

    # Translation
    translate_enabled: bool = True
    translate_backend: str = "vllm"  # "vllm" (LLM, GPU)
    translate_vllm_url: str = "http://localhost:8010"  # Shared vLLM endpoint (autosre coding agent)
    translate_vllm_model: str = ""  # auto-detect from vLLM if empty
    translate_realtime_vllm_url: str = (
        ""  # Optional: smaller model for live translation (falls back to translate_vllm_url)
    )

    # Rolling context window size for selected live translate directions.
    # Default 0 = OFF (stateless per-utterance prompt, current
    # behaviour).  Meeting-scoped history is bound / cleared by
    # ``TranslationQueue.bind_meeting`` to prevent cross-meeting
    # utterance leakage.  ``live_translate_context_directions`` is a
    # comma-separated list like ``ja:en,en:ja``. The default keeps the
    # measured 2026-04-19 win for JA→EN without silently changing EN→JA.
    live_translate_context_window_ja_en: int = 0
    live_translate_context_directions: str = "ja:en"

    # Fragment-gated context: when True, only fragmentary JA
    # utterances receive prior_context; short affirmatives ("はい",
    # "OK", "そうですね") stay stateless and cache-eligible.  Preserves
    # the 15-20 % LRU cache hit rate the backend earns on repeated
    # short phrases (see ``backends/translate_vllm.py:99-107``) while
    # still anchoring the fragmentary utterances that are the actual
    # hallucination target.  Only has an effect when
    # ``live_translate_context_window_ja_en`` > 0; independently
    # toggleable from the window-size knob.
    live_translate_context_fragment_gated: bool = False

    # Slide render parallelism — number of concurrent LibreOffice
    # invocations the express-batch render path is allowed to run.
    # Serial renders (=1) waste the GB10's other 19 cores while one LO
    # process serializes. 2026-04-20 bench on a 50-slide deck measured:
    #   parallelism 1 → 14.4s total (20 slides)
    #   parallelism 2 → 10.5s (1.37x)
    #   parallelism 4 →  7.0s (2.06x)  ← sweet spot
    #   parallelism 8 →  6.9s (diminishing returns; font cache / disk IO
    #                           contention caps the gain)
    # Default 4 drops translated-slide-visible wall clock ~3x with no
    # effect on first-paint latency. Bench script lives at
    # ``scripts/slide_batch_bench/bench_parallel.py`` — rerun after any
    # LibreOffice upgrade or GB10 hardware change.
    # Override via SCRIBE_SLIDE_RENDER_PARALLELISM. Clamped to [1, 16]
    # at load time to avoid a fork bomb from a bad env value.
    slide_render_parallelism: int = 4

    # TTS — single endpoint by default; comma-separated pools are still
    # supported explicitly for experiments.
    tts_vllm_url: str = "http://localhost:8002"

    # Speaker name extraction
    name_extraction_backend: str = "auto"  # "regex" | "llm" | "auto"

    # GB10 infrastructure
    gb10_host: str = ""  # GB10 IP address for remote management

    # WiFi hotspot
    # Regulatory domain — 2-letter ISO 3166 country code (JP, US, DE, ...).
    # Set before every AP rotation; kernel caps 5 GHz TX power under the
    # default world domain (00), which prevents phones from associating.
    # Overridable via SCRIBE_WIFI_REGDOMAIN env var or PUT /api/admin/settings.
    wifi_regdomain: str = "JP"

    # Display timezone — IANA name (Asia/Tokyo, America/Los_Angeles, ...).
    # Empty string means "use the server's local time". Set via
    # SCRIBE_TIMEZONE env var or PUT /api/admin/settings.
    timezone: str = ""

    # Storage
    meetings_dir: Path = Path(__file__).parent.parent.parent / "meetings"
    journal_fsync_seconds: int = 5
    retention_days: int = 30

    # Translation queue
    translate_queue_maxsize: int = 50
    translate_queue_concurrency: int = 4
    translate_timeout_seconds: int = 30  # vLLM translation is fast

    @classmethod
    def from_profile(cls, profile: str) -> ServerConfig:
        """Create config from a named profile.

        Profiles set all backends to sensible defaults for a target platform.
        Individual env vars override profile defaults.

        Args:
            profile: "gb10" for GB10 production defaults.
        """
        cfg = cls(**_GB10_DEFAULTS) if profile == "gb10" else cls()  # type: ignore[arg-type]

        # Apply any env var overrides on top of profile
        return cls._apply_env(cfg)

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Load configuration from environment variables.

        If SCRIBE_PROFILE is set, starts from that profile's defaults.
        """
        profile = os.environ.get("SCRIBE_PROFILE", "")
        if profile:
            return cls.from_profile(profile)

        return cls._apply_env(cls())

    @classmethod
    def _apply_env(cls, cfg: ServerConfig) -> ServerConfig:
        """Apply environment variable overrides to an existing config."""

        def _env(key: str, default: str) -> str:
            return os.environ.get(key, default)

        def _env_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key)
            if val is None:
                return default
            return val.lower() == "true"

        return cls(
            host=_env("SCRIBE_HOST", cfg.host),
            port=int(_env("SCRIBE_PORT", str(cfg.port))),
            captive_http_port=_resolve_captive_http_port(cfg.captive_http_port),
            backend_mode=_env("SCRIBE_BACKEND", cfg.backend_mode),
            asr_model=_env("SCRIBE_ASR_MODEL", cfg.asr_model),
            asr_vllm_url=_env("SCRIBE_ASR_VLLM_URL", cfg.asr_vllm_url),
            asr_language=_env("SCRIBE_ASR_LANGUAGE", cfg.asr_language),
            default_language_pair=_env("SCRIBE_LANGUAGE_PAIR", cfg.default_language_pair),
            diarize_enabled=_env_bool("SCRIBE_DIARIZE", cfg.diarize_enabled),
            diarize_backend=_env("SCRIBE_DIARIZE_BACKEND", cfg.diarize_backend),
            diarize_url=_env("SCRIBE_DIARIZE_URL", cfg.diarize_url),
            translate_enabled=_env_bool("SCRIBE_TRANSLATE", cfg.translate_enabled),
            translate_backend=_env("SCRIBE_TRANSLATE_BACKEND", cfg.translate_backend),
            translate_vllm_url=_env("SCRIBE_TRANSLATE_VLLM_URL", cfg.translate_vllm_url),
            translate_vllm_model=_env("SCRIBE_TRANSLATE_VLLM_MODEL", cfg.translate_vllm_model),
            translate_realtime_vllm_url=_env(
                "SCRIBE_TRANSLATE_REALTIME_VLLM_URL", cfg.translate_realtime_vllm_url
            ),
            tts_vllm_url=_env("SCRIBE_TTS_VLLM_URL", cfg.tts_vllm_url),
            name_extraction_backend=_env("SCRIBE_NAME_EXTRACTION", cfg.name_extraction_backend),
            gb10_host=_env("SCRIBE_GB10_HOST", cfg.gb10_host),
            wifi_regdomain=_env("SCRIBE_WIFI_REGDOMAIN", cfg.wifi_regdomain),
            timezone=_env("SCRIBE_TIMEZONE", cfg.timezone),
            meetings_dir=Path(_env("SCRIBE_MEETINGS_DIR", str(cfg.meetings_dir))),
            translate_queue_concurrency=int(
                _env("SCRIBE_TRANSLATE_CONCURRENCY", str(cfg.translate_queue_concurrency))
            ),
            translate_queue_maxsize=int(
                _env("SCRIBE_TRANSLATE_QUEUE_MAXSIZE", str(cfg.translate_queue_maxsize))
            ),
            translate_timeout_seconds=int(
                _env("SCRIBE_TRANSLATE_TIMEOUT_SECONDS", str(cfg.translate_timeout_seconds))
            ),
            live_translate_context_window_ja_en=int(
                _env(
                    "SCRIBE_LIVE_TRANSLATE_CONTEXT_WINDOW_JA_EN",
                    str(cfg.live_translate_context_window_ja_en),
                )
            ),
            live_translate_context_directions=_env(
                "SCRIBE_LIVE_TRANSLATE_CONTEXT_DIRECTIONS",
                cfg.live_translate_context_directions,
            ),
            live_translate_context_fragment_gated=_env_bool(
                "SCRIBE_LIVE_TRANSLATE_CONTEXT_FRAGMENT_GATED",
                cfg.live_translate_context_fragment_gated,
            ),
            slide_render_parallelism=max(
                1,
                min(
                    16,
                    int(
                        _env(
                            "SCRIBE_SLIDE_RENDER_PARALLELISM",
                            str(cfg.slide_render_parallelism),
                        )
                    ),
                ),
            ),
        )
