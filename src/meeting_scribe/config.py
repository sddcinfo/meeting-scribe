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

import os
from dataclasses import dataclass
from pathlib import Path

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
    # Comma-separated TTS replica pool. The Qwen3TTSBackend round-robins
    # synthesis requests across this list. Default deploys both faster-
    # qwen3-tts containers from docker-compose.gb10.yml so the GIL-bound
    # 100 % CPU ceiling on a single replica stops capping throughput.
    # Override with SCRIBE_TTS_VLLM_URL=http://localhost:8002 for single-
    # replica setups.
    "tts_vllm_url": "http://localhost:8002,http://localhost:8012",
    "name_extraction_backend": "auto",
    "translate_queue_concurrency": 4,
}


@dataclass
class ServerConfig:
    """Main server configuration."""

    # Network
    #
    # The server runs two uvicorn listeners against the same FastAPI app so
    # all in-process globals (current meeting, TTS queue, audio-out client
    # set) are shared between them:
    #
    #   Admin: HTTPS on ``port`` bound to the auto-detected management IP
    #          (``ip route get`` src address — not 0.0.0.0). This makes the
    #          admin socket physically unreachable from the hotspot
    #          interface even if firewall rules regress.
    #   Guest: HTTP on ``guest_port`` bound to the hotspot AP IP
    #          (``10.42.0.1``) via ``IP_FREEBIND`` so the socket can be
    #          pre-bound before ``nmcli`` assigns the address to ``wlan0``.
    #
    # ``host`` is retained as a back-compat default for unit tests that
    # spin up uvicorn without the split-listener entry point.
    host: str = "0.0.0.0"
    port: int = 8080
    guest_port: int = 80

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
    translate_offline_vllm_url: str = (
        ""  # Optional: larger model for refinement (falls back to translate_vllm_url)
    )

    # Near-realtime refinement worker. Gated OFF by default — the worker
    # adds continuous translate load during meetings (see the
    # 2026-04-19 sweep at `reports/context_window_sweep/2026-04-19/
    # summary.md`).  Toggle via SCRIBE_ENABLE_REFINEMENT=1 when the
    # operator has run the refinement-validation harness and accepts
    # the measured extra load.  When enabled the worker consults
    # ``refinement_context_window_segments`` (below) for its rolling
    # meeting-context window size.
    refinement_enabled: bool = False

    # Rolling context window size for the live JA→EN translate path.
    # Default 0 = OFF (stateless per-utterance prompt, current
    # behaviour).  Phase B1 of the refinement rollout flips this to 2
    # after the sweep's on/off arm shows p50 Δ ≤ +20 ms on the shared
    # GPU.  Direction-gated: the knob intentionally affects JA→EN only
    # because that is where the sweep measured the quality gain (83 % of
    # utterances change vs 62 % for EN→JA — see summary.md:37-42).
    # Meeting-scoped history is bound / cleared by
    # ``TranslationQueue.bind_meeting`` to prevent cross-meeting
    # utterance leakage.
    live_translate_context_window_ja_en: int = 0

    # Phase B2 — when True, only fragmentary JA utterances receive
    # prior_context; short affirmatives ("はい", "OK", "そうですね")
    # stay stateless and cache-eligible.  Preserves the 15-20 % LRU
    # cache hit rate the backend earns on repeated short phrases
    # (see ``backends/translate_vllm.py:99-107``) while still
    # anchoring the fragmentary utterances that are the actual
    # hallucination target.  Only has an effect when
    # ``live_translate_context_window_ja_en`` > 0; independently
    # toggleable so the B1→B2 rollout can test with and without.
    live_translate_context_fragment_gated: bool = False

    # Rolling meeting-context window size for the near-realtime
    # RefinementWorker (follow-shortly-after path).  The last N already-
    # refined (source, translation) tuples are folded into the system
    # prompt for each batch translate call so the LLM can anchor on
    # running topic/speakers/proper-nouns rather than hallucinate full
    # sentences from fragmented ASR chunks.  Most useful for JA→EN where
    # Japanese utterances routinely drop subjects ("その辺をクリアしないと、
    # 非常に" — completed by context from the prior turn).
    #
    # Default 4 is the production value picked by the 2026-04-19 sweep
    # (`reports/context_window_sweep/2026-04-19/summary.md`): p50
    # latency +49 ms vs baseline (still inside refinement's 45 s trailing
    # headroom), +113 prompt tokens, JA→EN changes on 83% of fragments
    # with qualitative wins on honorifics + fragment preservation.
    # Quality plateaus at ≥4; anything above 8 is wasted prompt budget.
    #
    # Set ``0`` to disable (pre-2026-04-19 stateless behaviour).
    # Override via SCRIBE_REFINEMENT_CONTEXT_WINDOW_SEGMENTS or the
    # ``refinement_context_window_segments`` runtime_config knob
    # (hot-reloadable for re-running the sweep without restarting).
    # The live translate path deliberately does NOT use this —
    # see project_scribe_reasoning_path_split in auto-memory for why
    # the live path keeps its stateless prompt budget.
    refinement_context_window_segments: int = 4

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

    # TTS — comma-separated replica pool for round-robin (see DEFAULTS).
    tts_vllm_url: str = "http://localhost:8002,http://localhost:8012"

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
            guest_port=int(_env("SCRIBE_GUEST_PORT", str(cfg.guest_port))),
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
            translate_offline_vllm_url=_env(
                "SCRIBE_TRANSLATE_OFFLINE_VLLM_URL", cfg.translate_offline_vllm_url
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
            refinement_context_window_segments=int(
                _env(
                    "SCRIBE_REFINEMENT_CONTEXT_WINDOW_SEGMENTS",
                    str(cfg.refinement_context_window_segments),
                )
            ),
            refinement_enabled=_env_bool("SCRIBE_ENABLE_REFINEMENT", cfg.refinement_enabled),
            live_translate_context_window_ja_en=int(
                _env(
                    "SCRIBE_LIVE_TRANSLATE_CONTEXT_WINDOW_JA_EN",
                    str(cfg.live_translate_context_window_ja_en),
                )
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
