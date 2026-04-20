"""Tests for ServerConfig profiles and environment variable loading."""

from __future__ import annotations

from meeting_scribe.config import ServerConfig


class TestConfigDefaults:
    """GB10 production defaults."""

    def test_default_backend_mode(self):
        cfg = ServerConfig()
        assert cfg.backend_mode == "prod"

    def test_default_translate_backend(self):
        cfg = ServerConfig()
        assert cfg.translate_backend == "vllm"

    def test_default_diarize_disabled(self):
        cfg = ServerConfig()
        assert cfg.diarize_enabled is True

    def test_default_name_extraction(self):
        cfg = ServerConfig()
        assert cfg.name_extraction_backend == "auto"


class TestGB10Profile:
    """GB10 production profile."""

    def test_gb10_profile_sets_prod_mode(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.backend_mode == "prod"

    def test_gb10_profile_enables_diarization(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.diarize_enabled is True
        assert cfg.diarize_backend == "sortformer"

    def test_gb10_profile_uses_vllm(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.translate_backend == "vllm"
        assert cfg.translate_vllm_url == "http://localhost:8010"

    def test_gb10_profile_uses_qwen3_asr(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.asr_model == "Qwen/Qwen3-ASR-1.7B"
        assert cfg.asr_vllm_url == "http://localhost:8003"

    def test_gb10_profile_tts_url(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.tts_vllm_url == "http://localhost:8002,http://localhost:8012"

    def test_gb10_profile_diarize_url(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.diarize_url == "http://localhost:8001"

    def test_gb10_profile_auto_name_extraction(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.name_extraction_backend == "auto"

    def test_gb10_profile_binds_all_interfaces(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.host == "0.0.0.0"

    def test_gb10_higher_translation_concurrency(self):
        cfg = ServerConfig.from_profile("gb10")
        assert cfg.translate_queue_concurrency == 4


class TestEnvOverrides:
    """Environment variable overrides."""

    def test_profile_env_var(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_PROFILE", "gb10")
        cfg = ServerConfig.from_env()
        assert cfg.backend_mode == "prod"

    def test_env_overrides_profile(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_PROFILE", "gb10")
        monkeypatch.setenv("SCRIBE_TRANSLATE_BACKEND", "vllm")
        cfg = ServerConfig.from_env()
        # Profile sets vllm, env override also vllm
        assert cfg.translate_backend == "vllm"

    def test_gb10_host_env(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_GB10_HOST", "192.168.1.100")
        cfg = ServerConfig.from_env()
        assert cfg.gb10_host == "192.168.1.100"

    def test_diarize_env(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_DIARIZE", "true")
        cfg = ServerConfig.from_env()
        assert cfg.diarize_enabled is True

    def test_tts_vllm_url_env(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_TTS_VLLM_URL", "http://gb10:8002")
        cfg = ServerConfig.from_env()
        assert cfg.tts_vllm_url == "http://gb10:8002"

    def test_name_extraction_env(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_NAME_EXTRACTION", "llm")
        cfg = ServerConfig.from_env()
        assert cfg.name_extraction_backend == "llm"

    def test_slide_render_parallelism_default(self):
        cfg = ServerConfig.from_env()
        assert cfg.slide_render_parallelism == 4

    def test_slide_render_parallelism_env(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_SLIDE_RENDER_PARALLELISM", "8")
        cfg = ServerConfig.from_env()
        assert cfg.slide_render_parallelism == 8

    def test_slide_render_parallelism_clamped_low(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_SLIDE_RENDER_PARALLELISM", "0")
        cfg = ServerConfig.from_env()
        assert cfg.slide_render_parallelism == 1

    def test_slide_render_parallelism_clamped_high(self, monkeypatch):
        monkeypatch.setenv("SCRIBE_SLIDE_RENDER_PARALLELISM", "99")
        cfg = ServerConfig.from_env()
        assert cfg.slide_render_parallelism == 16
