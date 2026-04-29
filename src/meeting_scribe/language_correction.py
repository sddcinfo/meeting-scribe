"""Post-correction of ASR-tagged language using lingua.

Qwen3-ASR has a strong English bias on Germanic/Romance speech: Dutch,
German, Norwegian, Swedish, sometimes Spanish/Italian get confidently
mislabeled as English. The system-prompt fix (naming the meeting's
languages explicitly) helps but doesn't eliminate the bias — we still
saw 53/64 Dutch segments tagged ``en``.

This module gives every segment a second opinion from
``lingua-language-detector``, **constrained to the meeting's language
pair** (so it can only flip between e.g. nl ↔ en, never to a random
third language). Lingua reports per-language confidences, so we only
override the ASR tag when lingua is confident. Short or ambiguous text
keeps the ASR label.

Falls back to a no-op if lingua isn't installed.

Instrumentation: every call updates module-level counters
(``correction_stats``) so callers can measure latency and override
rates per language pair. The stats are intentionally process-global —
the live ASR backend and the reprocess pipeline both feed into the
same counters so a benchmark script can read them via ``snapshot()``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from threading import Lock

logger = logging.getLogger(__name__)


# ── Instrumentation ─────────────────────────────────────────────
class _CorrectionStats:
    """Thread-safe rolling counters for the lingua post-corrector.

    Tracked per (asr_lang, corrected_lang) pair so a benchmark / status
    endpoint can show "Dutch meetings: 45% of en-tagged segments
    corrected to nl, mean latency 2.1ms".
    """

    __slots__ = (
        "_lock",
        "_started_at",
        "calls",
        "kept",
        "max_latency_ms",
        "overridden",
        "override_pairs",
        "skipped_no_detector",
        "skipped_short",
        "total_latency_ms",
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self.calls = 0
        self.skipped_short = 0
        self.skipped_no_detector = 0
        self.kept = 0
        self.overridden = 0
        # ``"asr_lang→corrected_lang"`` → count
        self.override_pairs: dict[str, int] = {}
        self.total_latency_ms = 0.0
        self.max_latency_ms = 0.0
        self._started_at = time.time()

    def record(
        self,
        *,
        outcome: str,  # one of: "skipped_short", "skipped_no_detector", "kept", "overridden"
        asr_lang: str = "",
        corrected: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        with self._lock:
            self.calls += 1
            self.total_latency_ms += latency_ms
            if latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms
            if outcome == "skipped_short":
                self.skipped_short += 1
            elif outcome == "skipped_no_detector":
                self.skipped_no_detector += 1
            elif outcome == "kept":
                self.kept += 1
            elif outcome == "overridden":
                self.overridden += 1
                k = f"{asr_lang}→{corrected}"
                self.override_pairs[k] = self.override_pairs.get(k, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            mean = (self.total_latency_ms / self.calls) if self.calls else 0.0
            graded = (self.overridden + self.kept) or 1
            return {
                "started_at": self._started_at,
                "uptime_s": round(time.time() - self._started_at, 1),
                "calls": self.calls,
                "skipped_short": self.skipped_short,
                "skipped_no_detector": self.skipped_no_detector,
                "kept": self.kept,
                "overridden": self.overridden,
                "override_rate": round(self.overridden / graded, 4),
                "mean_latency_ms": round(mean, 3),
                "max_latency_ms": round(self.max_latency_ms, 3),
                "override_pairs": dict(self.override_pairs),
            }

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.skipped_short = 0
            self.skipped_no_detector = 0
            self.kept = 0
            self.overridden = 0
            self.override_pairs = {}
            self.total_latency_ms = 0.0
            self.max_latency_ms = 0.0
            self._started_at = time.time()


correction_stats = _CorrectionStats()

# Per-language-pair detector cache. Lingua loads its models lazily, so the
# first detection in a pair is slower (~1s); subsequent calls are fast.
_DETECTOR_CACHE: dict[frozenset[str], object] = {}
_CACHE_LOCK = Lock()

# ISO-639-1 → lingua Language enum mapping. Limited to the languages the
# meeting registry supports + a few common neighbors that the ASR might
# misclassify into. New entries here just need: (a) a Language member
# in lingua, (b) an ISO code we already use elsewhere.
# Lingua splits some languages by orthography (Norwegian → BOKMAL/NYNORSK,
# Chinese has SIMPLIFIED/TRADITIONAL etc.). We pick the dominant variant
# for ISO codes that don't carry that distinction. ``no`` is intentionally
# omitted — lingua has no umbrella ``NORWEGIAN`` member, and we don't
# currently support Norwegian as a meeting language.
_ISO_TO_LINGUA = {
    "en": "ENGLISH",
    "ja": "JAPANESE",
    "zh": "CHINESE",
    "ko": "KOREAN",
    "nl": "DUTCH",
    "de": "GERMAN",
    "fr": "FRENCH",
    "es": "SPANISH",
    "it": "ITALIAN",
    "pt": "PORTUGUESE",
    "ru": "RUSSIAN",
    "sv": "SWEDISH",
    "da": "DANISH",
    "fi": "FINNISH",
    "pl": "POLISH",
    "tr": "TURKISH",
    "ar": "ARABIC",
    "vi": "VIETNAMESE",
    "th": "THAI",
    "id": "INDONESIAN",
}

# Override only when lingua is THIS confident the ASR was wrong.
# Tuned against the Dutch test case (e5b376b2): a value of 0.65 catches
# the obvious mislabels while leaving short/ambiguous segments alone.
_OVERRIDE_MIN_CONFIDENCE = 0.65

# Skip very short text — lingua needs ~3 words to be reliable. Below
# this we just trust ASR.
_MIN_TEXT_CHARS = 8


def _is_lingua_available() -> bool:
    try:
        import lingua  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception:
        return False


def _build_detector(pair: frozenset[str]):
    """Build (and cache) a lingua detector constrained to the given codes."""
    try:
        from lingua import Language, LanguageDetectorBuilder
    except Exception:
        return None

    langs = []
    for code in pair:
        name = _ISO_TO_LINGUA.get(code)
        if name and hasattr(Language, name):
            langs.append(getattr(Language, name))
    if len(langs) < 2:
        # Lingua requires at least 2 languages to disambiguate.
        return None

    return (
        LanguageDetectorBuilder.from_languages(*langs)
        .with_low_accuracy_mode()  # faster, fine for ≥2-word text
        .build()
    )


def _get_detector(meeting_pair: Iterable[str]):
    """Return a (cached) detector constrained to ``meeting_pair`` codes."""
    pair = frozenset(c.lower() for c in meeting_pair if c)
    if not pair:
        return None
    with _CACHE_LOCK:
        det = _DETECTOR_CACHE.get(pair)
        if det is None:
            det = _build_detector(pair)
            if det is not None:
                _DETECTOR_CACHE[pair] = det
    return det


def correct_segment_language(
    text: str,
    asr_lang: str,
    meeting_pair: Iterable[str],
) -> str:
    """Second-opinion an ASR language tag against lingua.

    Returns either the original ``asr_lang`` (no change) or a corrected
    code from ``meeting_pair`` if lingua is confidently disagreeing.

    Conservative by design — we only override when:
      * lingua is installed
      * meeting_pair has at least 2 codes lingua knows
      * text is long enough to be reliable
      * lingua's confidence on its top pick is ≥ the threshold AND
        the top pick differs from the ASR tag

    Otherwise we trust the ASR. Always cheap on the no-op path.

    Every call updates ``correction_stats`` so a benchmark / status
    endpoint can show override rates and per-call latency.
    """
    t0 = time.perf_counter()
    if not text or len(text.strip()) < _MIN_TEXT_CHARS:
        correction_stats.record(
            outcome="skipped_short",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return asr_lang

    pair = [c.lower() for c in meeting_pair if c]
    if len(pair) < 2:
        correction_stats.record(
            outcome="skipped_no_detector",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return asr_lang

    asr_lc = (asr_lang or "").lower()

    detector = _get_detector(pair)
    if detector is None:
        correction_stats.record(
            outcome="skipped_no_detector",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return asr_lang

    try:
        confs = detector.compute_language_confidence_values(text)
    except Exception:
        logger.debug("lingua detection raised", exc_info=True)
        correction_stats.record(
            outcome="kept",
            asr_lang=asr_lc,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return asr_lang
    if not confs:
        correction_stats.record(
            outcome="kept",
            asr_lang=asr_lc,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return asr_lang

    top = confs[0]
    top_score = float(getattr(top, "value", 0.0))
    top_name = getattr(top.language, "name", "")
    top_iso = next(
        (iso for iso, name in _ISO_TO_LINGUA.items() if name == top_name),
        None,
    )
    if not top_iso:
        correction_stats.record(
            outcome="kept",
            asr_lang=asr_lc,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return asr_lang

    # Lingua agrees with ASR — leave it.
    if top_iso == asr_lc:
        correction_stats.record(
            outcome="kept",
            asr_lang=asr_lc,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return asr_lang

    # Disagreement — only override if lingua is confident.
    if top_score >= _OVERRIDE_MIN_CONFIDENCE:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Lingua corrected ASR tag: %r → %r (conf=%.2f, %.1fms) on %r",
            asr_lang,
            top_iso,
            top_score,
            latency_ms,
            text[:40],
        )
        correction_stats.record(
            outcome="overridden",
            asr_lang=asr_lc,
            corrected=top_iso,
            latency_ms=latency_ms,
        )
        return top_iso

    correction_stats.record(
        outcome="kept",
        asr_lang=asr_lc,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return asr_lang
