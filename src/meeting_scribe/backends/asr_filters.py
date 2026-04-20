"""Shared ASR filtering and normalization utilities.

Used by the vLLM Qwen3-ASR backend for hallucination detection,
language normalization, and response parsing.
"""

from __future__ import annotations

import re
from collections import Counter

# Known hallucination patterns produced by Whisper-family models
_HALLUCINATIONS = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "ご視聴ありがとうございました",
    "チャンネル登録",
    "字幕",
    "subtitles",
}


def _is_hallucination(text: str) -> bool:
    """Detect common ASR hallucination patterns."""
    lower = text.lower().strip()
    for p in _HALLUCINATIONS:
        if p in lower:
            return True
    words = lower.split()
    if len(words) >= 3 and words[0] == words[1] == words[2]:
        return True
    stripped = lower.replace(" ", "").replace(",", "").replace(".", "")
    if len(stripped) >= 4:
        _most, count = Counter(stripped).most_common(1)[0]
        if count / len(stripped) > 0.5:
            return True
    return any(len(word) > 60 for word in text.split())


def _normalize_language(lang: str) -> str:
    """Normalize any ASR language name/code to an ISO 639-1 code.

    Uses the centralized language registry. Returns "unknown" for
    unrecognized languages.
    """
    from meeting_scribe.languages import normalize_language

    return normalize_language(lang)


def _detect_language_from_text(text: str) -> str:
    """Heuristic language detection from text content.

    Uses script analysis (CJK, Hangul, Kana, Latin, etc.) to guess
    the language when ASR doesn't provide it.
    """
    if not text or len(text.strip()) < 2:
        return "unknown"

    total = len(text.replace(" ", ""))
    if total == 0:
        return "unknown"

    cjk = len(re.findall(r"[\u3000-\u9fff\uff00-\uffef]", text))
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    hangul = len(re.findall(r"[\uac00-\ud7af\u1100-\u11ff]", text))
    cyrillic = len(re.findall(r"[\u0400-\u04ff]", text))
    arabic = len(re.findall(r"[\u0600-\u06ff]", text))
    thai = len(re.findall(r"[\u0e00-\u0e7f]", text))
    devanagari = len(re.findall(r"[\u0900-\u097f]", text))

    ratio_cjk = cjk / total

    # Japanese: has kana (hiragana/katakana)
    if kana > 0 and ratio_cjk > 0.3:
        return "ja"
    # Korean: has hangul
    if hangul > 0 and hangul / total > 0.3:
        return "ko"
    # Chinese: CJK without kana or hangul
    if ratio_cjk > 0.5 and kana == 0 and hangul == 0:
        return "zh"
    # Russian/Ukrainian: Cyrillic
    if cyrillic / total > 0.3:
        return "ru"
    # Arabic
    if arabic / total > 0.3:
        return "ar"
    # Thai
    if thai / total > 0.3:
        return "th"
    # Hindi/Devanagari
    if devanagari / total > 0.3:
        return "hi"
    # Latin script — default to "en" (most common), but could be fr/de/es/etc.
    if ratio_cjk < 0.1:
        return "en"

    return "unknown"


def _parse_qwen3_asr_response(raw: str) -> tuple[str, str]:
    """Parse Qwen3-ASR response format.

    Qwen3-ASR returns: "language English<asr_text>actual transcribed text"
    Returns: (text, language_code)
    """
    if not raw:
        return "", "unknown"

    # Extract language and text from "language X<asr_text>Y" format
    if "<asr_text>" in raw:
        prefix, _, text = raw.partition("<asr_text>")
        text = text.strip()

        # Parse language from prefix: "language English" or "language Japanese"
        lang_raw = prefix.replace("language", "").strip()
        lang = _normalize_language(lang_raw)

        return text, lang

    # Fallback: no special tokens, just text
    return raw.strip(), _detect_language_from_text(raw)
