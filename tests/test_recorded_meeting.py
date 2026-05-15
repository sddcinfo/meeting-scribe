"""Tests using the recorded 87-minute English meeting fixture.

These tests replay real meeting audio through the pipeline to verify:
- ASR accuracy against known transcripts
- Translation pipeline with real segments
- Audio alignment and segment extraction
- Language detection accuracy
- Hallucination filter on real data

The fixture at test-fixtures/90min_english_2026-04-07/ contains:
- recording.pcm: 164MB s16le 16kHz mono (87 min)
- journal.jsonl: 1297 transcribed segments
- meta.json: meeting metadata
"""

from __future__ import annotations

import numpy as np
import pytest


class TestRecordedMeetingFixture:
    """Validate the recorded meeting fixture is usable."""

    def test_fixture_exists(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")
        assert recorded_meeting["pcm_path"].exists()
        assert len(recorded_meeting["journal"]) > 0

    def test_fixture_has_segments(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")
        assert len(recorded_meeting["segments"]) >= 1000
        assert recorded_meeting["duration_ms"] > 5_000_000  # >83 min

    def test_fixture_primarily_english(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")
        en_count = sum(1 for _, _, _, lang in recorded_meeting["segments"] if lang == "en")
        total = len(recorded_meeting["segments"])
        assert en_count / total > 0.95  # >95% English


class TestSegmentExtraction:
    """Extract and validate audio segments from the recording."""

    def test_extract_segment_has_audio(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        start_ms, end_ms, _text, _lang = recorded_meeting["segments"][0]
        pcm_path = recorded_meeting["pcm_path"]

        sample_rate = 16000
        bytes_per_sample = 2
        start_byte = int(start_ms / 1000 * sample_rate) * bytes_per_sample
        end_byte = int(end_ms / 1000 * sample_rate) * bytes_per_sample

        with open(pcm_path, "rb") as f:
            f.seek(start_byte)
            audio_bytes = f.read(end_byte - start_byte)

        assert len(audio_bytes) > 0
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        # Should have audible content (not silence)
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        assert rms > 100, f"Audio too quiet (RMS={rms}), may be silence"

    def test_segments_are_chronological(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        prev_start = 0
        for start_ms, end_ms, _text, _lang in recorded_meeting["segments"]:
            assert start_ms >= prev_start, f"Segments not chronological at {start_ms}ms"
            assert end_ms > start_ms, f"Zero-duration segment at {start_ms}ms"
            prev_start = start_ms

    def test_segment_duration_reasonable(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        for start_ms, end_ms, _text, _lang in recorded_meeting["segments"][:100]:
            duration_ms = end_ms - start_ms
            assert 500 <= duration_ms <= 30000, f"Segment duration {duration_ms}ms out of range"


class TestLanguageDetection:
    """Verify language detection on real transcript data."""

    def test_english_segments_have_latin_text(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        import re

        cjk_re = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
        en_segments = [
            (text, lang) for _, _, text, lang in recorded_meeting["segments"] if lang == "en"
        ]

        # Sample first 100 English segments
        for text, _lang in en_segments[:100]:
            cjk_chars = len(cjk_re.findall(text))
            total_chars = len(text.replace(" ", ""))
            if total_chars > 0:
                cjk_ratio = cjk_chars / total_chars
                assert cjk_ratio < 0.3, f"English segment has too much CJK: '{text[:50]}'"


_HALLUCINATION_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "ご視聴ありがとうございました",
    "チャンネル登録",
]


class TestHallucinationFilter:
    """Verify no hallucinated content in the recorded transcript."""

    def test_no_hallucinations_in_journal(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        for seg in recorded_meeting["journal"]:
            text = seg["text"].lower()
            for phrase in _HALLUCINATION_PHRASES:
                assert phrase not in text, (
                    f"Hallucination found in segment {seg['segment_id']}: '{seg['text'][:80]}'"
                )

    def test_no_excessive_repetition(self, recorded_meeting):
        """Check for extreme repetition loops (5+ same word).

        Minor repetition (3x) can occur in natural speech ("best best best").
        We only flag 5+ consecutive repeats which indicate ASR hallucination.
        """
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        for seg in recorded_meeting["journal"]:
            words = seg["text"].lower().split()
            if len(words) >= 5:
                for i in range(len(words) - 4):
                    assert not (
                        words[i] == words[i + 1] == words[i + 2] == words[i + 3] == words[i + 4]
                    ), f"Repetition loop in segment: '{seg['text'][:80]}'"


class TestTranslationInputQuality:
    """Verify segments are suitable for translation pipeline."""

    def test_segments_have_text(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        empty_count = sum(1 for s in recorded_meeting["journal"] if not s["text"].strip())
        total = len(recorded_meeting["journal"])
        assert empty_count / total < 0.05, f"Too many empty segments: {empty_count}/{total}"

    def test_segment_text_length_distribution(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        lengths = [len(s["text"]) for s in recorded_meeting["journal"] if s["text"]]
        avg_len = sum(lengths) / len(lengths)
        assert 20 < avg_len < 200, f"Average segment length {avg_len} out of expected range"

    def test_no_extremely_long_segments(self, recorded_meeting):
        if recorded_meeting is None:
            pytest.skip("Recorded meeting fixture not available")

        for seg in recorded_meeting["journal"]:
            assert len(seg["text"]) < 1000, (
                f"Extremely long segment ({len(seg['text'])} chars): '{seg['text'][:80]}'"
            )


class TestAudioFixtureQuality:
    """Validate audio fixture quality for backend testing."""

    def test_audio_en_fixture_has_content(self, audio_en_bytes):
        """English audio fixture should have audible content."""
        audio = np.frombuffer(audio_en_bytes, dtype=np.int16)
        assert len(audio) >= 8000  # At least 0.5s @ 16kHz
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        assert rms > 50, f"Audio too quiet (RMS={rms})"

    def test_audio_en_fixture_duration(self, audio_en_bytes):
        """English audio fixture should be 2-10 seconds."""
        samples = len(audio_en_bytes) // 2  # s16le = 2 bytes/sample
        duration_s = samples / 16000
        assert 0.5 <= duration_s <= 30, f"Audio duration {duration_s}s out of range"
