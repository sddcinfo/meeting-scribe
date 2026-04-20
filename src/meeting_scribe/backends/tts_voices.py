"""Studio voice registry for Qwen3-TTS.

Qwen3-TTS ships 9 premium multilingual speakers under the model's Apache 2.0
license. Using a named speaker skips the reference-audio encode step (~0.8 s
per request under load) while still yielding studio-booth quality, and
licenses cleanly for commercial use.

The mapping prefers each language's native speaker where one exists; the
remaining languages fall back to a multilingual default. Speakers can still
voice any supported language — the Qwen team just recommends native for best
quality.
"""

from __future__ import annotations

# All speakers (from QwenLM/Qwen3-TTS):
#   Vivian   — Chinese, bright/edgy young female
#   Serena   — Chinese, warm/gentle young female
#   Uncle_Fu — Chinese, seasoned male, low mellow
#   Dylan    — Chinese (Beijing dialect), youthful male
#   Eric     — Chinese (Sichuan dialect), lively male
#   Ryan     — English, dynamic male, strong rhythm
#   Aiden    — English, sunny American male, clear midrange
#   Ono_Anna — Japanese, playful young female
#   Sohee    — Korean, warm female, rich emotion

# Fallback for any Qwen3-TTS language without a dedicated native speaker.
# Serena is multilingual and the warmest all-rounder.
_STUDIO_FALLBACK = "serena"

# Language code (ISO 639-1) → Qwen3-TTS speaker name.
# Covers the 10 languages Qwen3-TTS supports natively. Languages with
# tts_native=False in LANGUAGE_REGISTRY are gated out in synthesize_stream
# before reaching voice resolution.
_STUDIO_VOICES: dict[str, str] = {
    "en": "aiden",
    "zh": "vivian",
    "ja": "ono_anna",
    "ko": "sohee",
    "de": _STUDIO_FALLBACK,
    "fr": _STUDIO_FALLBACK,
    "es": "aiden",
    "it": "aiden",
    "pt": _STUDIO_FALLBACK,
    "ru": "uncle_fu",
}


def studio_voice_for(language: str) -> str:
    """Return the studio speaker name to use for ``language``.

    Unknown / unsupported codes fall back to the multilingual default so the
    synth call never fails closed on a missing mapping.
    """
    return _STUDIO_VOICES.get((language or "").lower(), _STUDIO_FALLBACK)


def all_studio_voices() -> dict[str, str]:
    """Read-only copy of the language → speaker registry."""
    return dict(_STUDIO_VOICES)
