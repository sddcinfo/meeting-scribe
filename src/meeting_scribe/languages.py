"""Language registry and utilities for multi-language support.

Centralizes all language definitions, translation prompt generation,
and language pair validation. Replaces hardcoded EN/JA references
throughout the codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Language:
    """A supported language."""

    code: str  # ISO 639-1 (e.g., "en", "ja")
    name: str  # English name (e.g., "English", "Japanese")
    native_name: str  # Native name (e.g., "日本語")
    css_font_class: str = ""  # CSS class for font rendering (e.g., "ja" triggers Noto Sans JP)
    tts_native: bool = True  # Qwen3-TTS can synthesize this language natively


# Selectable languages for a meeting. Two tiers:
#   - TTS-native (tts_native=True): full round-trip ASR → translate → TTS.
#     These 10 languages match backends/tts_voices.py::_STUDIO_VOICES.
#   - ASR+translate only (tts_native=False): transcription and translation work,
#     but TTS synthesis is skipped (gated in synthesize_stream). The guest portal
#     hides the listen selector for these languages.
LANGUAGE_REGISTRY: dict[str, Language] = {
    # ── TTS-native languages (Qwen3-TTS) ──
    "en": Language("en", "English", "English"),
    "zh": Language("zh", "Chinese", "中文", "zh"),
    "ja": Language("ja", "Japanese", "日本語", "ja"),
    "ko": Language("ko", "Korean", "한국어", "ko"),
    "fr": Language("fr", "French", "Français"),
    "de": Language("de", "German", "Deutsch"),
    "es": Language("es", "Spanish", "Español"),
    "it": Language("it", "Italian", "Italiano"),
    "pt": Language("pt", "Portuguese", "Português"),
    "ru": Language("ru", "Russian", "Русский"),
    # ── ASR + translation only (no TTS) ──
    "nl": Language("nl", "Dutch", "Nederlands", tts_native=False),
    "ar": Language("ar", "Arabic", "العربية", tts_native=False),
    "th": Language("th", "Thai", "ไทย", tts_native=False),
    "vi": Language("vi", "Vietnamese", "Tiếng Việt", tts_native=False),
    "id": Language("id", "Indonesian", "Bahasa Indonesia", tts_native=False),
    "ms": Language("ms", "Malay", "Bahasa Melayu", tts_native=False),
    "hi": Language("hi", "Hindi", "हिन्दी", tts_native=False),
    "tr": Language("tr", "Turkish", "Türkçe", tts_native=False),
    "pl": Language("pl", "Polish", "Polski", tts_native=False),
    "uk": Language("uk", "Ukrainian", "Українська", tts_native=False),
}

# Qwen3-ASR language name → code mapping. Covers all 20 languages in
# LANGUAGE_REGISTRY. Anything outside this set normalizes to "unknown" and is
# handled as a no-translation fall-through.
_ASR_NAME_TO_CODE: dict[str, str] = {
    # English names (from Qwen3-ASR output)
    "english": "en",
    "chinese": "zh",
    "mandarin": "zh",
    "cantonese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "dutch": "nl",
    "arabic": "ar",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    "hindi": "hi",
    "turkish": "tr",
    "polish": "pl",
    "ukrainian": "uk",
    # ISO 639-1 and 639-2 codes (including bibliographic variants)
    "en": "en",
    "eng": "en",
    "zh": "zh",
    "zho": "zh",
    "cmn": "zh",
    "yue": "zh",
    "ja": "ja",
    "jpn": "ja",
    "ko": "ko",
    "kor": "ko",
    "fr": "fr",
    "fra": "fr",
    "de": "de",
    "deu": "de",
    "es": "es",
    "spa": "es",
    "it": "it",
    "ita": "it",
    "pt": "pt",
    "por": "pt",
    "ru": "ru",
    "rus": "ru",
    "nl": "nl",
    "nld": "nl",
    "dut": "nl",
    "ar": "ar",
    "ara": "ar",
    "th": "th",
    "tha": "th",
    "vi": "vi",
    "vie": "vi",
    "id": "id",
    "ind": "id",
    "ms": "ms",
    "msa": "ms",
    "may": "ms",
    "hi": "hi",
    "hin": "hi",
    "tr": "tr",
    "tur": "tr",
    "pl": "pl",
    "pol": "pl",
    "uk": "uk",
    "ukr": "uk",
}


def normalize_language(lang: str) -> str:
    """Normalize any language name/code to a 2-letter ISO 639-1 code.

    Returns the code if recognized, "unknown" otherwise.
    """
    return _ASR_NAME_TO_CODE.get((lang or "").lower().strip(), "unknown")


def is_supported(code: str) -> bool:
    """Check if a language code is in the registry."""
    return code in LANGUAGE_REGISTRY


def is_tts_native(code: str) -> bool:
    """Check if a language has native Qwen3-TTS support.

    Returns False for unknown codes (fail-closed).
    """
    lang = LANGUAGE_REGISTRY.get(code)
    return bool(lang and lang.tts_native)


def get_language(code: str) -> Language | None:
    """Get a Language by code, or None if not found."""
    return LANGUAGE_REGISTRY.get(code)


def get_language_name(code: str) -> str:
    """Get the English name for a language code."""
    lang = LANGUAGE_REGISTRY.get(code)
    return lang.name if lang else code.upper()


def get_language_native(code: str) -> str:
    """Get the native name for a language code."""
    lang = LANGUAGE_REGISTRY.get(code)
    return lang.native_name if lang else code.upper()


DEFAULT_LANGUAGE_PAIR: tuple[str, str] = ("en", "ja")


def is_valid_language_pair(a: str, b: str) -> bool:
    """Two codes form a valid meeting pair iff both are in the registry and distinct."""
    return a != b and a in LANGUAGE_REGISTRY and b in LANGUAGE_REGISTRY


def is_valid_languages(langs: list[str]) -> bool:
    """A meeting's languages are valid iff length is 1 or 2, every code is in the
    registry, and length-2 pairs are distinct. This is the authoritative shape
    check used by ``MeetingMeta``'s field validator — every other call site
    should reach the model through construction rather than revalidating here.
    """
    if not isinstance(langs, list) or len(langs) not in (1, 2):
        return False
    if any(code not in LANGUAGE_REGISTRY for code in langs):
        return False
    return not (len(langs) == 2 and langs[0] == langs[1])


def parse_languages_strict(s: str) -> list[str] | None:
    """Parse ``"en"`` or ``"ja,en"`` into a validated list of 1 or 2 codes.

    Returns ``None`` on any invalid input (unknown code, duplicate pair, >2
    parts, empty). Used for untrusted input (e.g. the meeting-create endpoint)
    where a bad value must fail loudly with a 400 rather than silently running
    the meeting in the wrong language.
    """
    if not isinstance(s, str):
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not is_valid_languages(parts):
        return None
    return parts


def parse_languages(s: str) -> list[str]:
    """Lenient config-only parser. Accepts ``"en"`` / ``"ja,en"`` and falls back
    to ``DEFAULT_LANGUAGE_PAIR`` on malformed or unsupported input so startup
    never crashes — but emits a loud WARNING log so a typo in a deployment
    surfaces instead of silently running every meeting in the wrong language.
    """
    parsed = parse_languages_strict(s)
    if parsed is not None:
        return parsed
    default = list(DEFAULT_LANGUAGE_PAIR)
    logger.warning(
        "Invalid SCRIBE_LANGUAGE_PAIR value %r — falling back to default %s. "
        "Check the env var if this was unintentional.",
        s,
        default,
    )
    return default


def parse_language_pair(pair_str: str) -> tuple[str, str]:
    """Parse a 'ja,en' format string into a tuple of language codes.

    Falls back to the default pair for malformed or unsupported input so legacy
    config paths never crash. Retained for callers that *must* have two distinct
    languages (e.g. the ASR bilingual prompt); monolingual-aware call sites
    should use :func:`parse_languages` instead.
    """
    parts = [p.strip() for p in pair_str.split(",")]
    if len(parts) != 2:
        return DEFAULT_LANGUAGE_PAIR
    a, b = parts[0], parts[1]
    if not is_valid_language_pair(a, b):
        return DEFAULT_LANGUAGE_PAIR
    return (a, b)


def get_translation_target(
    detected_lang: str, languages: list[str] | tuple[str, ...]
) -> str | None:
    """Given a detected language and the meeting's languages, return the
    translation target — or ``None`` when no translation should happen.

    ``languages`` may be length 1 (monolingual meeting — always returns
    ``None``) or length 2 (bilingual pair). For the length-2 case, returns the
    *other* code when ``detected_lang`` matches one side of the pair; otherwise
    ``None`` (detected language outside the meeting's pair).
    """
    if len(languages) == 1:
        return None
    if detected_lang == languages[0]:
        return languages[1]
    if detected_lang == languages[1]:
        return languages[0]
    return None


# Meeting-scoped few-shot exemplars used to anchor translation style on
# the pair where we have validated BLEU (EN↔JA — scribe's primary live
# direction).  See ``reports/phase3/prompt_ablation_2026-04-18.md`` for
# the sweep that picked this prompt: ``fewshot_meeting`` beat the
# generic "professional translator" baseline by +6.88 BLEU EN→JA and
# +8.01 BLEU JA→EN on the 72-pair meeting corpus with no measurable
# latency regression (p50 414 ms vs baseline 409 ms).
#
# NOTE: reasoning mode (``enable_thinking=True``) is explicitly NOT
# used on the live path — see ``project_scribe_reasoning_path_split``
# auto-memory and the SLO test in
# ``tests/test_translate_vllm_latency_slo.py``.
_FEWSHOT_PAIRS: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("en", "ja"): [
        (
            "Let's schedule a follow-up meeting to review the proposal.",
            "提案を検討するためのフォローアップ会議を設定しましょう。",
        ),
        (
            "The API response time has improved.",
            "APIのレスポンスタイムが改善されました。",
        ),
    ],
    ("ja", "en"): [
        (
            "提案を検討するためのフォローアップ会議を設定しましょう。",
            "Let's schedule a follow-up meeting to review the proposal.",
        ),
        (
            "APIのレスポンスタイムが改善されました。",
            "The API response time has improved.",
        ),
    ],
}


def get_translation_prompt(
    source_lang: str,
    target_lang: str,
    prior_context: list[tuple[str, str]] | None = None,
) -> str:
    """Generate a translation system prompt for any language pair.

    For EN↔JA (scribe's primary live direction, validated by the Phase 3
    prompt-ablation sweep) this returns a few-shot prompt with two
    meeting-scoped exemplars.  For every other pair we fall back to the
    generic "professional translator" prompt — we don't have validated
    exemplars for those pairs and a bad exemplar can hurt more than the
    generic prompt.

    When ``prior_context`` is provided, an "Earlier in this meeting"
    block is appended between the few-shot exemplars and the current
    instruction.  Each tuple is ``(source_text, translation)`` from
    already-processed utterances — usually the tail of the refinement
    worker's own ``self._results``.  This anchors the model on the
    running topic/speakers/proper-nouns so fragmented ASR output (very
    common in JP→EN where utterances drop subjects and trail off with
    particles) is translated in continuation rather than hallucinated
    into a free-standing sentence.  Order matters: oldest → newest, so
    the closing instruction still reads as "translate the *next*
    utterance".  Pass ``None`` (the default) to preserve the stateless
    prompt for the live path.
    """
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    exemplars = _FEWSHOT_PAIRS.get((source_lang, target_lang))
    context_block = ""
    if prior_context:
        ctx_body = "\n".join(
            f"{source_name}: {src}\n{target_name}: {tgt}"
            for src, tgt in prior_context
            if src and tgt
        )
        if ctx_body:
            context_block = (
                "Earlier in this meeting (for reference only, do not "
                f"re-translate):\n\n{ctx_body}\n\n"
            )
    if exemplars:
        body = "\n".join(f"{source_name}: {src}\n{target_name}: {tgt}" for src, tgt in exemplars)
        return (
            f"You are translating live bilingual meeting utterances.  "
            f"Here are two translated examples:\n\n{body}\n\n"
            f"{context_block}"
            f"Now translate the next {source_name} utterance into "
            f"{target_name}.  Use the same natural meeting register.  "
            f"Return only the translation."
        )
    return (
        f"You are a professional {source_name}-to-{target_name} translator. "
        f"Translate the following {source_name} text into natural, fluent {target_name}. "
        f"Preserve the meaning, tone, and context. "
        f"{context_block}"
        f"Return only the translation, no explanation or commentary."
    )


def to_api_response() -> dict:
    """Return the selectable language list + default pair for the UI.

    The UI renders two independent language pickers over this list. Every
    combination (a, b) with ``a != b`` is a valid meeting pair, so there is no
    separate "popular pairs" concept — the client composes the pair itself and
    the server validates it at meeting-create time via ``is_valid_language_pair``.
    Each entry carries both the English name and the native name so labels can
    read e.g. ``"German — Deutsch"`` and avoid the Deutsch/Dutch false cognate.
    """
    return {
        "languages": [
            {
                "code": lang.code,
                "name": lang.name,
                "native_name": lang.native_name,
                "css_font_class": lang.css_font_class,
                "tts_supported": lang.tts_native,
            }
            for lang in LANGUAGE_REGISTRY.values()
        ],
        "default_pair": list(DEFAULT_LANGUAGE_PAIR),
    }
