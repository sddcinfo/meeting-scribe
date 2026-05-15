"""Container-level TTS response contract helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tts_server_module():
    path = Path(__file__).resolve().parents[1] / "containers" / "tts" / "server.py"
    spec = importlib.util.spec_from_file_location("tts_container_server", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pcm_response_format_does_not_include_wav_header(monkeypatch):
    mod = _load_tts_server_module()
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)

    monkeypatch.setattr(mod, "_model", object())
    monkeypatch.setattr(mod, "_backend", "baseline")
    monkeypatch.setattr(mod, "_synthesize_baseline", lambda *args, **kwargs: audio)

    response = __import__("asyncio").run(
        mod.speech(
            mod.SpeechRequest(
                input="hello",
                language="en",
                voice="default",
                response_format="pcm",
            )
        )
    )

    assert response.media_type == "application/octet-stream"
    assert not bytes(response.body).startswith(b"RIFF")
    assert len(response.body) == len(audio) * 2
