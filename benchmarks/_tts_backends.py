"""TTS request-body builder per backend, for the 2026-Q2 bench.

Production talks to ``Qwen3TTSBackend`` (``src/meeting_scribe/backends/tts_qwen3.py``)
which sends Qwen3-specific fields (``priority``, ``seed``, ``temperature``,
``ref_audio``, ``voice``) on top of the OpenAI ``/v1/audio/speech``
shape.  Fun-CosyVoice3.5's OpenAI shim is more conservative — it
accepts the canonical ``{model, input, voice, stream, response_format}``
fields and rejects unknown keys.

The bench harnesses (``tts_quality_mos.py``, ``tts_concurrent_load.py``)
use ``build_body(backend, ...)`` to produce a backend-correct request
body.  Default backend is ``qwen3`` so existing callers (and the
production TTS replicas on 8002 / 8012) see no change.

A real production-side ``FunCosyVoiceTTSBackend`` subclass is a
Promote-time follow-up, not part of this plan.
"""

from __future__ import annotations

from typing import Literal

Backend = Literal["qwen3", "funcosyvoice"]
DEFAULT_BACKEND: Backend = "qwen3"

# Default ``model`` field per backend.  Either backend ignores the
# value at request time (the loaded model is fixed by the container),
# but the field is required by the OpenAI schema.
DEFAULT_MODEL: dict[Backend, str] = {
    "qwen3": "qwen3-tts",
    "funcosyvoice": "fun-cosyvoice-3.5",
}


def build_body(
    backend: Backend,
    *,
    text: str,
    voice: str,
    model: str | None = None,
    ref_audio_uri: str | None = None,
    priority: int = -10,
) -> dict:
    """Return a JSON body for ``POST /v1/audio/speech`` on the given backend.

    ``text`` and ``voice`` are required.  ``ref_audio_uri`` (a
    ``data:audio/wav;base64,...`` URI) toggles the cloned profile.
    ``priority`` is a Qwen3-only knob — silently ignored on Cosy.
    """
    body: dict = {
        "model": model or DEFAULT_MODEL[backend],
        "input": text,
        "voice": voice,
        "stream": True,
        "response_format": "pcm",
    }
    if backend == "qwen3":
        body["priority"] = priority
    if ref_audio_uri is not None:
        body["ref_audio"] = ref_audio_uri
    return body


def add_backend_arg(p) -> None:
    """Add ``--backend {qwen3,funcosyvoice}`` to an argparse parser."""
    p.add_argument(
        "--backend",
        choices=["qwen3", "funcosyvoice"],
        default=DEFAULT_BACKEND,
        help="Which TTS backend's request schema to use (default: qwen3).",
    )
