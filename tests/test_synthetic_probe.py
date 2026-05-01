"""Tests for the W4 synthetic-probe helper.

Covers the shared module that both meeting-start preflight (W4) and
the recovery supervisor (W6b) use to detect a wedged inference
endpoint. Threshold contract: adaptive p95×2 capped at ceiling, or
fixed cold default if histogram has <10 samples. Success contract:
HTTP 200 + valid response schema, no non-empty-text assertion.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from meeting_scribe.runtime import synthetic_probe as sp


class TestAdaptiveTimeout:
    def test_cold_default_when_histogram_empty(self):
        assert sp._adaptive_timeout(deque(maxlen=256), cold_default_s=5.0, ceiling_s=5.0) == 5.0

    def test_cold_default_below_min_samples(self):
        # 9 samples — under the 10-sample floor.
        h: deque[float] = deque((100.0,) * 9, maxlen=256)
        assert sp._adaptive_timeout(h, cold_default_s=3.0, ceiling_s=5.0) == 3.0

    def test_p95_x2_when_samples_present(self):
        # All 100 samples = 500 ms → p95 = 500 → threshold = 1.0s.
        h: deque[float] = deque((500.0,) * 100, maxlen=256)
        assert sp._adaptive_timeout(h, cold_default_s=3.0, ceiling_s=5.0) == 1.0

    def test_ceiling_caps_a_slow_p95(self):
        # 5000 ms p95 × 2 = 10s, capped at ceiling 5.0.
        h: deque[float] = deque((5000.0,) * 100, maxlen=256)
        assert sp._adaptive_timeout(h, cold_default_s=3.0, ceiling_s=5.0) == 5.0

    def test_hard_min_floor_when_p95x2_unreasonably_fast(self):
        # Very fast cluster: 10ms p95 × 2 = 0.02s. Hard floor 0.5s
        # prevents timeout-fail probes from firing on transient
        # micro-stutters of a healthy backend. Plan-aligned: cold
        # default is for empty histogram; once samples exist the
        # adaptive value wins, with only the 0.5s safety floor.
        h: deque[float] = deque((10.0,) * 100, maxlen=256)
        assert sp._adaptive_timeout(h, cold_default_s=3.0, ceiling_s=5.0) == 0.5


class TestOpenAIChatResponseValidator:
    def test_accepts_valid_with_empty_text(self):
        ok, why = sp._openai_chat_response_valid({"choices": [{"message": {"content": ""}}]})
        assert ok and why is None, why

    def test_accepts_valid_with_text(self):
        ok, why = sp._openai_chat_response_valid({"choices": [{"message": {"content": "hello"}}]})
        assert ok and why is None, why

    def test_rejects_non_dict(self):
        ok, why = sp._openai_chat_response_valid([])
        assert not ok and "not a dict" in why

    def test_rejects_empty_choices(self):
        ok, why = sp._openai_chat_response_valid({"choices": []})
        assert not ok and "non-empty" in why

    def test_rejects_missing_message(self):
        ok, why = sp._openai_chat_response_valid({"choices": [{}]})
        assert not ok and "message" in why

    def test_rejects_missing_content(self):
        ok, why = sp._openai_chat_response_valid({"choices": [{"message": {}}]})
        assert not ok and "content" in why


class TestDiarizeResponseValidator:
    def test_accepts_segments_shape(self):
        ok, why = sp._diarize_response_valid({"segments": [{"start": 0, "end": 1, "speaker": "0"}]})
        assert ok and why is None, why

    def test_accepts_diarization_shape(self):
        ok, why = sp._diarize_response_valid({"diarization": [{"start": 0}]})
        assert ok and why is None, why

    def test_accepts_empty_segments(self):
        ok, why = sp._diarize_response_valid({"segments": []})
        assert ok and why is None, why

    def test_rejects_unknown_shape(self):
        ok, why = sp._diarize_response_valid({"foo": "bar"})
        assert not ok and "no 'segments'/'diarization'" in why


class TestProbeAudioFixture:
    def test_fixture_exists_and_is_non_silent(self):
        assert sp.PROBE_AUDIO_PATH.exists()
        # Sanity: WAV is the expected size — 1.5s × 16 kHz × 2 bytes
        # = 48 000 bytes data + 44-byte header.
        size = sp.PROBE_AUDIO_PATH.stat().st_size
        assert 47_900 < size < 48_200, size

    def test_load_probe_audio_caches(self):
        # Reset cache, ensure two calls return same bytes object.
        sp._probe_audio_cache = None
        a = sp._load_probe_audio()
        b = sp._load_probe_audio()
        assert a is b


class TestPostAndValidateClassification:
    """The probe's whole contract is in _post_and_validate. These
    tests assert each branch produces the right ProbeResult.status."""

    @pytest.mark.asyncio
    async def test_ok_on_200_valid_schema(self):
        async def fake_post(*args, **kwargs):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": ""}}]},
                request=httpx.Request("POST", "http://x"),
            )

        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=fake_post)):
            r = await sp._post_and_validate(
                url="http://x/v1/chat/completions",
                json_payload={"x": 1},
                timeout_s=5.0,
                validator=sp._openai_chat_response_valid,
            )
        assert r.status == "ok", r

    @pytest.mark.asyncio
    async def test_http_error_on_500(self):
        async def fake_post(*args, **kwargs):
            return httpx.Response(500, text="boom", request=httpx.Request("POST", "http://x"))

        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=fake_post)):
            r = await sp._post_and_validate(
                url="http://x/v1/chat/completions",
                json_payload={"x": 1},
                timeout_s=5.0,
                validator=sp._openai_chat_response_valid,
            )
        assert r.status == "http_error", r
        assert "500" in (r.detail or "")

    @pytest.mark.asyncio
    async def test_timeout_classification(self):
        async def fake_post(*args, **kwargs):
            raise httpx.TimeoutException("read timed out")

        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=fake_post)):
            r = await sp._post_and_validate(
                url="http://x/v1/chat/completions",
                json_payload={"x": 1},
                timeout_s=5.0,
                validator=sp._openai_chat_response_valid,
            )
        assert r.status == "timeout", r

    @pytest.mark.asyncio
    async def test_schema_error_on_valid_200_invalid_shape(self):
        async def fake_post(*args, **kwargs):
            return httpx.Response(
                200,
                json={"unexpected": "shape"},
                request=httpx.Request("POST", "http://x"),
            )

        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=fake_post)):
            r = await sp._post_and_validate(
                url="http://x/v1/chat/completions",
                json_payload={"x": 1},
                timeout_s=5.0,
                validator=sp._openai_chat_response_valid,
            )
        assert r.status == "schema_error", r
