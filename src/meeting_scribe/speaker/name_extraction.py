"""Multilingual self-introduction → name extractor.

Used by the streaming enrollment flow: the admin UI records continuously
and POSTs the accumulated buffer to /api/room/enroll/detect-name, which
runs ASR and feeds the transcription into ``extract_name``. As soon as a
high-confidence self-introduction fires, the chair captures the name and
the embedding-extraction call follows.

Designed to be conservative: ambiguous transcripts return ``None`` and the
client keeps listening.

Patterns are organized by the natural way each of the 10 supported meeting
languages says "my name is X", drawn from common phrasebook references
(Rosetta Stone, Practice Portuguese, Migaku, JapanesePod101,
ChineseClass101, master-russian) — see PR notes for sources.
"""

from __future__ import annotations

import re

from meeting_scribe.speaker.romanization import romanize_name

# ── Stop-words ────────────────────────────────────────────────────────────────
# Latin tokens that ASR sometimes capitalizes mid-sentence and that we never
# want to treat as a person's name. Mostly English filler + the closed-class
# pronouns / copulas of the supported Romance / Germanic languages.
_NAME_STOPWORDS_LATIN: set[str] = {
    "the", "a", "an", "it", "so", "just", "very", "really", "here", "there",
    "not", "going", "ready", "happy", "sorry", "sure", "glad", "excited",
    "able", "trying", "looking", "working", "coming", "leaving", "done",
    "fine", "good", "great", "well", "okay", "back", "still", "also", "about",
    "from", "with", "your", "this", "that", "what", "when", "all", "new",
    "old", "big", "now", "like", "only", "yes", "no", "hi", "hello",
    # Romance / Germanic copulas + pronouns
    "soy", "es", "yo", "tu", "el", "la",
    "je", "il", "elle", "moi", "toi", "suis", "appelle", "nom",
    "ich", "bin", "mein", "name", "ist", "heisse", "heiße",
    "io", "sono", "mi", "chiamo", "nome",
    "eu", "sou", "meu", "chamo",
    "ya", "menya", "zovut",
    # Japanese romanized particles / copulas (ASR may romanize instead of using kana)
    "desu", "watashi", "watashiwa", "boku", "ore", "namae",
    "moushimasu", "iimasu", "kochira",
}

# ── Capture classes ───────────────────────────────────────────────────────────
# Latin: capital first letter, optional internal apostrophes/hyphens
# (O'Brien, Anne-Marie). Cap length to filter out runaway captures.
_LATIN_NAME = r"([A-Z][\w'\-]{1,14})"
# Hangul (Korean): 2-4 syllable blocks.
_HANGUL_NAME = r"([\uac00-\ud7af]{2,4})"
# Han (Chinese surnames + given names): 2-4 characters.
_HAN_NAME = r"([\u4e00-\u9fff]{2,4})"
# Japanese: 1-6 chars across kanji / hiragana / katakana — covers
# "田中", "ブラッド", "さくら".
_JA_NAME = r"([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]{1,6})"
# Cyrillic: capital first letter, 2-15 letters total.
_CYR_NAME = r"([\u0410-\u042f][\u0430-\u044f]{1,14})"

# ── Self-introduction patterns ────────────────────────────────────────────────
# Order matters: more specific multi-word phrases come first so they win
# over loose "I'm X" / "Я X" style fallbacks.
_INTRO_PATTERNS: list[tuple[str, int]] = [
    # ── English ──────────────────────────────────────────────────────────────
    (rf"my name'?s?\s+(?:is\s+)?{_LATIN_NAME}", re.IGNORECASE),
    (rf"(?:hi|hello|hey),?\s*(?:i'?m|i am)\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bi'?m\s+{_LATIN_NAME}\b", re.IGNORECASE),
    (rf"\bi am\s+{_LATIN_NAME}\b", re.IGNORECASE),
    (rf"\bcall me\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bthis is\s+{_LATIN_NAME}(?:\s+speaking)?", re.IGNORECASE),
    (rf"^{_LATIN_NAME}\s+here\b", re.IGNORECASE),
    (rf"^it'?s\s+{_LATIN_NAME}\b", re.IGNORECASE),
    # ── Spanish ──────────────────────────────────────────────────────────────
    (rf"\bme llamo\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bmi nombre es\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bsoy\s+{_LATIN_NAME}", re.IGNORECASE),
    # ── French ───────────────────────────────────────────────────────────────
    (rf"\bje m'?appelle\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bmon nom est\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bmoi,?\s*c'?est\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bje suis\s+{_LATIN_NAME}", re.IGNORECASE),
    # ── German ───────────────────────────────────────────────────────────────
    (rf"\bich hei[sß]+e\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bmein name ist\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bich bin\s+{_LATIN_NAME}", re.IGNORECASE),
    # ── Italian ──────────────────────────────────────────────────────────────
    (rf"\bmi chiamo\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bil mio nome [eè]\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bsono\s+{_LATIN_NAME}", re.IGNORECASE),
    # ── Portuguese ───────────────────────────────────────────────────────────
    (rf"\b(?:eu\s+)?me chamo\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\bchamo[- ]me\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\b(?:o\s+)?meu nome [eé]\s+{_LATIN_NAME}", re.IGNORECASE),
    (rf"\b(?:eu\s+)?sou\s+(?:o\s+|a\s+)?{_LATIN_NAME}", re.IGNORECASE),
    # ── Russian (Cyrillic) ───────────────────────────────────────────────────
    (rf"меня зовут\s+{_CYR_NAME}", re.IGNORECASE),
    (rf"моё имя\s+{_CYR_NAME}", re.IGNORECASE),
    (rf"мое имя\s+{_CYR_NAME}", re.IGNORECASE),
    (rf"^я\s+{_CYR_NAME}", re.IGNORECASE),
    # ── Russian (romanized fallback for ASR mishaps) ─────────────────────────
    (rf"\bmenya zovut\s+{_LATIN_NAME}", re.IGNORECASE),
    # ── Japanese ─────────────────────────────────────────────────────────────
    # Formal / standard patterns
    (rf"私の名前は{_JA_NAME}です", 0),
    (rf"わたしの名前は{_JA_NAME}です", 0),
    (rf"私は{_JA_NAME}です", 0),
    (rf"わたしは{_JA_NAME}です", 0),
    (rf"僕は{_JA_NAME}です", 0),
    (rf"俺は{_JA_NAME}です", 0),
    (rf"{_JA_NAME}と申します", 0),
    (rf"{_JA_NAME}と言います", 0),
    (rf"^{_JA_NAME}です[。\.]?$", 0),
    # Without trailing です (casual speech / ASR truncation)
    (rf"私の名前は{_JA_NAME}[。\.]?$", 0),
    (rf"わたしの名前は{_JA_NAME}[。\.]?$", 0),
    (rf"私は{_JA_NAME}[。\.]?$", 0),
    (rf"わたしは{_JA_NAME}[。\.]?$", 0),
    # Honorific variants (ASR may include -さん, -くん)
    (rf"私は{_JA_NAME}(?:さん|くん|ちゃん)?です", 0),
    # Third-person/presenter style ("これは田中です" = "this is Tanaka")
    (rf"こちらは{_JA_NAME}です", 0),
    (rf"こちら{_JA_NAME}です", 0),
    # ASR may romanize Japanese names — catch "Watashi wa Tanaka desu" style
    (rf"\bwatashi\s*(?:wa|ha)\s+{_LATIN_NAME}\s*desu\b", re.IGNORECASE),
    (rf"\bwatashi\s*(?:wa|ha)\s+{_LATIN_NAME}\b", re.IGNORECASE),
    (rf"\b{_LATIN_NAME}\s*desu\b", re.IGNORECASE),
    (rf"\b{_LATIN_NAME}\s+to\s+moushimasu\b", re.IGNORECASE),
    # ── Korean ───────────────────────────────────────────────────────────────
    (rf"제 이름은\s*{_HANGUL_NAME}\s*(?:입니다|이에요|예요)", 0),
    (rf"내 이름은\s*{_HANGUL_NAME}\s*(?:이에요|예요)", 0),
    (rf"저는\s*{_HANGUL_NAME}\s*(?:입니다|이에요|예요)", 0),
    (rf"^{_HANGUL_NAME}\s*(?:입니다|이에요|예요)", 0),
    # ── Chinese ──────────────────────────────────────────────────────────────
    (rf"我的名字(?:是|叫)\s*{_HAN_NAME}", 0),
    (rf"我叫\s*{_HAN_NAME}", 0),
    (rf"我是\s*{_HAN_NAME}", 0),
    (rf"我姓\s*{_HAN_NAME}", 0),
]

# Pre-compile for hot-path use (called once per ASR probe during enrollment).
_COMPILED_INTRO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, flags) for p, flags in _INTRO_PATTERNS
]

_BARE_TOKEN_RE = re.compile(
    r"[A-Za-z\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u0410-\u044f\uac00-\ud7af]+"
)

_JA_PARTICLES = ("の", "に", "を", "は", "が", "で", "と", "も",
                 "から", "まで", "より", "です", "ます")


def _clean_name(raw: str) -> str | None:
    """Validate a captured name and normalize its casing.

    Rejects stop-words, mis-cased Latin tokens, and obvious garbage. Returns
    ``None`` if the candidate isn't plausibly a person's name.
    """
    name = raw.strip().strip(".,!?。、")
    if not name:
        return None
    first = name[0]
    # CJK / Hangul: trust the regex's character class — script alone is signal.
    if "\u3040" <= first <= "\u9fff" or "\uac00" <= first <= "\ud7af":
        return name if 1 <= len(name) <= 8 else None
    # Cyrillic.
    if "\u0400" <= first <= "\u04ff":
        return name if 2 <= len(name) <= 20 else None
    # Latin: must look like a proper noun (capitalized, not a stop-word).
    if not first.isupper():
        return None
    if name.lower() in _NAME_STOPWORDS_LATIN:
        return None
    if len(name) < 2 or len(name) > 20:
        return None
    return name[:1].upper() + name[1:].lower()


def extract_name(text: str) -> str | None:
    """Detect a self-introduction in ``text`` and return the speaker's name.

    Tries the natural self-introduction patterns of each of the 10 supported
    meeting languages, then falls back to a strict bare-name path for users
    who just say their name into the mic without a phrase.

    The returned name is **always romanized to Latin** (e.g. "田中" → "Tanaka",
    "박지민" → "Bakjimin", "Иван" → "Ivan") so chair labels stay consistent
    regardless of which script the ASR happened to emit. Trailing honorifics
    (san/sama/씨/님/…) are stripped as part of romanization.
    """
    if not text:
        return None
    text = text.strip()

    for compiled in _COMPILED_INTRO_PATTERNS:
        m = compiled.search(text)
        if not m:
            continue
        name = _clean_name(m.group(1))
        if name:
            return romanize_name(name)

    # Bare-name path: only fires when the entire transcript collapses to ≤4
    # instances of the same short token (e.g. "Brad", "Brad. Brad.",
    # "田中。田中。", "박지민"). Strict on purpose — we don't want to pluck a
    # random capitalized word out of running speech.
    bare_tokens = _BARE_TOKEN_RE.findall(text)
    if 1 <= len(bare_tokens) <= 4 and len({t.lower() for t in bare_tokens}) == 1:
        tok = bare_tokens[0]
        first = tok[0]
        if "\u3040" <= first <= "\u9fff":
            if 1 <= len(tok) <= 6 and not any(p in tok for p in _JA_PARTICLES):
                return romanize_name(tok)
        elif "\uac00" <= first <= "\ud7af":
            if 2 <= len(tok) <= 4:
                return romanize_name(tok)
        elif "\u0410" <= first <= "\u044f":
            if 2 <= len(tok) <= 15 and tok[0].isupper():
                return romanize_name(tok)
        else:
            cleaned = _clean_name(tok)
            if cleaned:
                return romanize_name(cleaned)

    return None
