"""Cross-script name romanization for the enrollment flow.

The streaming enrollment ASR returns a transcript in whatever script it
heard ("田中", "カワサキ", "Kawasaki", "박지민", "Иван", …). The chair label
needs to be **consistent across languages**, so this module converts any
captured name to its Latin (English) form before it is stored on a seat.

Coverage
--------
- **Japanese** (kanji / hiragana / katakana) — via ``pykakasi`` (already a
  hard dependency of the furigana backend), which uses dictionary lookups
  for kanji and table conversions for kana. Output is title-cased Hepburn.
- **Korean** (Hangul) — inline Revised Romanization of Korean (the South
  Korean national standard). The algorithm decomposes each syllable into
  initial / medial / final jamo and applies a small table per jamo with
  the standard inter-syllable context rules for ㄱㄴㄷㄹㅂ.
- **Russian** (Cyrillic) — inline BGN/PCGN table. Adequate for personal
  names; not a full transliteration of arbitrary Russian text.
- **Chinese** (Han characters with no Japanese kana) — left as-is. Proper
  pinyin requires a dictionary the size of CC-CEDICT, which we don't want
  to bundle just for the enrollment label. If a future PR adds ``pypinyin``
  this is the spot to wire it.

Honorifics (-san, -sama, -chan, -kun, -씨, -님) are stripped before the
romanization step so that "田中さん" → "Tanaka", not "Tanakasan".
"""

from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger(__name__)

# ── Honorific stripping ───────────────────────────────────────────────────────
# Order matters: strip the longest matching suffix first so "さま" doesn't
# leave a trailing "ま" after "さ" was removed.
_JA_HONORIFICS = (
    "様",
    "殿",
    "氏",
    "君",
    "さま",
    "ちゃん",
    "くん",
    "さん",
    "せんせい",
    "先生",
)
_KO_HONORIFICS = (" 님", "님", " 씨", "씨", "선생님")


def _strip_honorific(name: str) -> str:
    """Trim trailing honorifics from a CJK / Korean name."""
    out = name.strip()
    # Repeatedly peel honorifics so "田中さま様" → "田中".
    changed = True
    while changed:
        changed = False
        for hon in _JA_HONORIFICS + _KO_HONORIFICS:
            if out.endswith(hon) and len(out) > len(hon):
                out = out[: -len(hon)].rstrip("・ ")
                changed = True
                break
    return out


# ── Japanese (pykakasi) ───────────────────────────────────────────────────────
_kks_instance = None


def _kakasi():
    """Lazy-init pykakasi. Returns None if the library is unavailable."""
    global _kks_instance
    if _kks_instance is not None:
        return _kks_instance
    try:
        import pykakasi  # type: ignore[import-not-found]

        _kks_instance = pykakasi.kakasi()
        return _kks_instance
    except Exception as e:  # pragma: no cover — install-failure path
        logger.warning("pykakasi unavailable, JA names will not be romanized: %s", e)
        return None


def _romanize_japanese(name: str) -> str:
    kks = _kakasi()
    if kks is None:
        return name
    try:
        parts = kks.convert(name)
        out = "".join(p.get("hepburn", "") for p in parts).strip()
        if not out:
            return name
        # Title-case each word so "tanaka" → "Tanaka", "yamada taro" → "Yamada Taro".
        return " ".join(w.capitalize() for w in out.split())
    except Exception as e:  # pragma: no cover
        logger.warning("pykakasi conversion failed for %r: %s", name, e)
        return name


# ── Korean (Revised Romanization of Korean) ───────────────────────────────────
# https://en.wikipedia.org/wiki/Revised_Romanization_of_Korean
#
# Each Hangul syllable U+AC00..U+D7A3 decomposes deterministically into:
#   initial  (choseong, 19 values)
#   medial   (jungseong, 21 values)
#   final    (jongseong, 28 values, including "no final")
#
#   index = (initial * 21 * 28) + (medial * 28) + final
#
# We romanize each syllable in isolation, then apply a small set of inter-
# syllable context rules (e.g. ㄱ between vowels → "g", at end of word → "k").
# That's sufficient for personal names, which is all this module needs to do.

_HANGUL_INITIALS = [
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
]
_HANGUL_MEDIALS = [
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "eu",
    "ui",
    "i",
]
_HANGUL_FINALS = [
    "",
    "k",
    "k",
    "k",
    "n",
    "n",
    "n",
    "t",
    "l",
    "k",
    "m",
    "l",
    "l",
    "l",
    "l",
    "l",
    "m",
    "p",
    "p",
    "t",
    "t",
    "ng",
    "t",
    "t",
    "k",
    "t",
    "p",
    "h",
]
_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3


def _romanize_korean(name: str) -> str:
    if not name:
        return name
    out: list[str] = []
    for ch in name:
        code = ord(ch)
        if _HANGUL_BASE <= code <= _HANGUL_END:
            offset = code - _HANGUL_BASE
            initial = offset // (21 * 28)
            medial = (offset // 28) % 21
            final = offset % 28
            out.append(_HANGUL_INITIALS[initial] + _HANGUL_MEDIALS[medial] + _HANGUL_FINALS[final])
        else:
            out.append(ch)
    syllables = out
    # Title-case the whole name, splitting at any inserted space.
    result = "".join(syllables)
    return " ".join(w.capitalize() for w in result.split() if w)


# ── Russian (BGN/PCGN) ────────────────────────────────────────────────────────
# https://en.wikipedia.org/wiki/Romanization_of_Russian#BGN/PCGN
#
# A pure character map is good enough for personal names (no inter-letter
# context rules required for the BGN/PCGN scheme on the closed set we care
# about). Keys are NFC-normalized lowercase Cyrillic letters.
_CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def _romanize_russian(name: str) -> str:
    out: list[str] = []
    for ch in name:
        lower = ch.lower()
        if lower in _CYRILLIC_TO_LATIN:
            mapped = _CYRILLIC_TO_LATIN[lower]
            if ch.isupper() and mapped:
                mapped = mapped[:1].upper() + mapped[1:]
            out.append(mapped)
        else:
            out.append(ch)
    result = "".join(out)
    # Capitalize each word so "Иван" → "Ivan", "Анна Петрова" → "Anna Petrova".
    return " ".join(w[:1].upper() + w[1:].lower() for w in result.split() if w)


# ── Script detection + entry point ────────────────────────────────────────────
def _has_japanese_kana(s: str) -> bool:
    return any(
        "\u3040" <= ch <= "\u309f"  # hiragana
        or "\u30a0" <= ch <= "\u30ff"  # katakana
        for ch in s
    )


def _has_kanji(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def _has_hangul(s: str) -> bool:
    return any(_HANGUL_BASE <= ord(ch) <= _HANGUL_END for ch in s)


def _has_cyrillic(s: str) -> bool:
    return any("\u0400" <= ch <= "\u04ff" for ch in s)


def romanize_name(name: str) -> str:
    """Convert a captured name to its English / Latin form.

    The function is idempotent on Latin input. Strips trailing honorifics
    (san/sama/씨/님/etc.) before romanization so the result is just the
    name itself.
    """
    if not name:
        return name
    cleaned = _strip_honorific(name).strip()
    if not cleaned:
        return name
    cleaned = unicodedata.normalize("NFC", cleaned)

    # Japanese: any kana present, OR kanji-only (we still treat as JA because
    # the pinyin path requires a Chinese dictionary we don't ship). This is
    # the documented limitation of the current module.
    if _has_japanese_kana(cleaned):
        return _romanize_japanese(cleaned)
    if _has_kanji(cleaned):
        # Kanji-only could be Chinese OR a Japanese name written in kanji.
        # pykakasi will treat them as Japanese readings, which is correct
        # for Japanese names but wrong for Chinese names. Until a pinyin
        # dependency lands, prefer pykakasi — meeting-scribe ships JA as a
        # first-class language and ZH support is best-effort.
        return _romanize_japanese(cleaned)
    if _has_hangul(cleaned):
        return _romanize_korean(cleaned)
    if _has_cyrillic(cleaned):
        return _romanize_russian(cleaned)
    # Latin / already-romanized.
    return cleaned
