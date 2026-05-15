"""Furigana (ruby text) generator for Japanese transcript segments.

Uses `pykakasi` — a pure-Python dictionary-based converter. Correct,
deterministic, ~0.1 ms per call after warm-up. Keeps the async surface
so server.py can fire it as a background task with no special casing,
but the call is cheap enough that we could inline it if we wanted.

Generates HTML of the form `<ruby>漢字<rt>かんじ</rt></ruby>` with every
kanji run wrapped. Text outside kanji (hiragana, katakana, ASCII,
punctuation) is HTML-escaped and passed through unchanged.
"""

from __future__ import annotations

import asyncio
import html as _html
import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


def _has_kanji(s: str) -> bool:
    return any(0x4E00 <= ord(c) <= 0x9FFF for c in s)


class FuriganaBackend:
    """Sync-safe pykakasi wrapper with an async facade + LRU cache."""

    def __init__(self) -> None:
        # pykakasi.kakasi object, populated by start().  Typed Any to
        # avoid a hard import-time dependency on pykakasi when mypy
        # runs without the package installed (the package is a runtime
        # requirement, not a type-checking one).
        self._kks: Any = None
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_max = 1024
        self._cache_hits = 0
        self._cache_misses = 0
        self._latency_samples: list[float] = []
        self._errors = 0

    async def start(self) -> None:
        # pykakasi is a REQUIRED dependency (pyproject.toml). If the import
        # fails we fail loudly — silent fallback made furigana invisibly
        # broken across an entire meeting when the env was missing the
        # package. Health gate catches this before users can start a meeting.
        import pykakasi

        self._kks = pykakasi.kakasi()
        self._kks.convert("初期化")  # warm the dictionary
        logger.info("Furigana backend ready (pykakasi)")

    async def stop(self) -> None:
        self._kks = None

    def _render(self, text: str) -> str | None:
        if self._kks is None:
            return None
        try:
            parts = self._kks.convert(text)
        except Exception as e:
            self._errors += 1
            logger.debug("pykakasi convert failed: %s", e)
            return None
        out: list[str] = []
        for r in parts:
            orig = r.get("orig", "")
            hira = r.get("hira", "")
            if orig and hira and _has_kanji(orig) and hira != orig:
                out.append(f"<ruby>{_html.escape(orig)}<rt>{_html.escape(hira)}</rt></ruby>")
            else:
                out.append(_html.escape(orig))
        return "".join(out)

    async def annotate(self, text: str) -> str | None:
        """Return HTML-annotated form of `text` or None.

        Skips early if no kanji are present. Caches identical phrases so
        repeated fillers cost zero.
        """
        if not text or self._kks is None:
            return None
        stripped = text.strip()
        if not stripped or not _has_kanji(stripped):
            return None

        cached = self._cache.get(stripped)
        if cached is not None:
            self._cache.move_to_end(stripped)
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        # pykakasi is sync + fast — run inline on the event loop.
        # ~0.1 ms per call means it never materially blocks.
        t0 = time.monotonic()
        try:
            html = await asyncio.to_thread(self._render, stripped)
        except Exception as e:
            self._errors += 1
            logger.debug("furigana annotate failed: %s", e)
            return None
        self._latency_samples.append((time.monotonic() - t0) * 1000.0)
        if len(self._latency_samples) > 256:
            self._latency_samples = self._latency_samples[-256:]
        if not html:
            return None

        self._cache[stripped] = html
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return html

    def stats(self) -> dict:
        samples = sorted(self._latency_samples)

        def pct(p):
            return (
                round(samples[min(len(samples) - 1, int(len(samples) * p))], 2) if samples else None
            )

        return {
            "backend": "pykakasi",
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "errors": self._errors,
            "latency_ms": {
                "p50": pct(0.5),
                "p95": pct(0.95),
                "p99": pct(0.99),
                "n": len(samples),
            },
        }
