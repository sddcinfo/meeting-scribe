from __future__ import annotations

from types import SimpleNamespace

from meeting_scribe.runtime import init, state


def test_tts_restart_containers_defaults_to_single_replica(monkeypatch):
    monkeypatch.delenv("SCRIBE_TTS_RESTART_CONTAINERS", raising=False)
    monkeypatch.setattr(state, "tts_backend", SimpleNamespace(_urls=["http://localhost:8002"]))
    monkeypatch.setattr(state, "config", SimpleNamespace(tts_vllm_url="http://localhost:8002"))

    assert init._tts_restart_containers() == ("scribe-tts",)


def test_tts_restart_containers_keeps_managed_single_container_with_external_pool(monkeypatch):
    monkeypatch.delenv("SCRIBE_TTS_RESTART_CONTAINERS", raising=False)
    monkeypatch.setattr(state, "tts_backend", None)
    monkeypatch.setattr(
        state,
        "config",
        SimpleNamespace(tts_vllm_url="http://localhost:8002,http://localhost:8012"),
    )

    assert init._tts_restart_containers() == ("scribe-tts",)


def test_tts_restart_containers_allows_env_override(monkeypatch):
    monkeypatch.setenv("SCRIBE_TTS_RESTART_CONTAINERS", "custom-a, custom-b")
    monkeypatch.setattr(state, "tts_backend", None)
    monkeypatch.setattr(state, "config", SimpleNamespace(tts_vllm_url="http://localhost:8002"))

    assert init._tts_restart_containers() == ("custom-a", "custom-b")
