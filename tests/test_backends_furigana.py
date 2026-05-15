"""Unit tests for FuriganaBackend (pykakasi wrapper + LRU cache)."""

from __future__ import annotations

import pytest

from meeting_scribe.backends.furigana import FuriganaBackend, _has_kanji


class TestHasKanji:
    def test_kanji_present(self):
        assert _has_kanji("会議")
        assert _has_kanji("今日は会議です")

    def test_hiragana_only(self):
        assert not _has_kanji("こんにちは")

    def test_katakana_only(self):
        assert not _has_kanji("コンピュータ")

    def test_ascii_only(self):
        assert not _has_kanji("hello world")

    def test_empty(self):
        assert not _has_kanji("")

    def test_mixed_kanji_kana(self):
        assert _has_kanji("お会議")


@pytest.fixture
async def backend():
    b = FuriganaBackend()
    await b.start()
    yield b
    await b.stop()


class TestAnnotate:
    async def test_kanji_phrase_produces_ruby(self, backend):
        html = await backend.annotate("会議")
        assert html is not None
        assert "<ruby>" in html
        assert "<rt>" in html
        assert "かいぎ" in html

    async def test_hiragana_only_returns_none(self, backend):
        # No kanji → early return
        assert await backend.annotate("こんにちは") is None

    async def test_empty_string_returns_none(self, backend):
        assert await backend.annotate("") is None
        assert await backend.annotate("   ") is None

    async def test_ascii_returns_none(self, backend):
        assert await backend.annotate("hello world") is None

    async def test_mixed_kanji_and_kana(self, backend):
        html = await backend.annotate("今日は会議です")
        assert html is not None
        assert "<ruby>" in html
        # The hiragana parts must be escaped-passthrough, not ruby-wrapped
        assert "です" in html

    async def test_html_escapes_special_chars(self, backend):
        # pykakasi passes through non-kanji; the renderer must html-escape
        html = await backend.annotate("会議<script>")
        assert html is not None
        assert "<script>" not in html  # escaped
        assert "&lt;script&gt;" in html


class TestCache:
    async def test_cache_hit_on_second_call(self, backend):
        await backend.annotate("会議")
        assert backend._cache_misses == 1
        assert backend._cache_hits == 0

        await backend.annotate("会議")
        assert backend._cache_misses == 1
        assert backend._cache_hits == 1

    async def test_cache_normalizes_whitespace(self, backend):
        # annotate() strips — both forms should hit the same cache entry
        await backend.annotate("会議")
        await backend.annotate("  会議  ")
        assert backend._cache_hits == 1

    async def test_cache_eviction_lru(self, backend):
        backend._cache_max = 3
        phrases = ["会議", "質問", "重要", "明日"]
        for p in phrases:
            await backend.annotate(p)
        # Cache capped at 3; oldest ("会議") evicted
        assert len(backend._cache) == 3
        assert "会議" not in backend._cache
        assert "明日" in backend._cache

    async def test_cache_move_to_end_on_hit(self, backend):
        backend._cache_max = 2
        await backend.annotate("会議")
        await backend.annotate("質問")
        await backend.annotate("会議")  # bump 会議 to the front
        await backend.annotate("重要")  # evict 質問, not 会議
        assert "会議" in backend._cache
        assert "質問" not in backend._cache
        assert "重要" in backend._cache


class TestStats:
    async def test_stats_shape(self, backend):
        await backend.annotate("会議")
        s = backend.stats()
        assert s["backend"] == "pykakasi"
        assert s["cache_size"] == 1
        assert s["cache_misses"] == 1
        assert s["errors"] == 0
        assert "latency_ms" in s
        assert s["latency_ms"]["n"] == 1

    async def test_stats_empty_before_calls(self, backend):
        s = backend.stats()
        assert s["cache_size"] == 0
        assert s["cache_misses"] == 0
        assert s["latency_ms"]["p50"] is None


class TestErrorPath:
    async def test_annotate_when_not_started_returns_none(self):
        b = FuriganaBackend()
        # Never called start()
        assert await b.annotate("会議") is None

    async def test_render_returns_none_when_pykakasi_raises(self, backend, monkeypatch):
        def boom(_text):
            raise RuntimeError("pykakasi exploded")

        monkeypatch.setattr(backend._kks, "convert", boom)
        # _render swallows and increments errors
        assert backend._render("会議") is None
        assert backend._errors == 1
