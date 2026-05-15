"""Schema + attribution invariants for the translate JSONL log.

Every translate row is required to carry ``kind == "translate"``,
``source == "live"`` (the only call site since the refinement worker
was removed), and a ``meeting_id`` so post-meeting tooling can group
calls by meeting. These tests pin the contract at the call site.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from meeting_scribe.backends.translate_vllm import (
    _TRANSLATE_LOG_SCHEMA_VERSION,
    VllmTranslateBackend,
    _log_translation,
)


def _fake_ok_response(text: str = "Hello!", prompt_tokens: int = 10, completion_tokens: int = 3):
    resp = AsyncMock()
    resp.raise_for_status = lambda: None
    resp.status_code = 200
    resp.json = lambda: {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestLogTranslationSchema:
    def test_every_row_carries_attribution_fields(self, tmp_path, monkeypatch):
        log_path = tmp_path / "scribe-translations.jsonl"
        monkeypatch.setattr("meeting_scribe.backends.translate_vllm._TRANS_LOG_PATH", log_path)

        _log_translation(
            model="test-model",
            source_lang="ja",
            target_lang="en",
            text="こんにちは",
            translated="Hello",
            elapsed_ms=42.0,
            input_tokens=10,
            output_tokens=3,
            kind="translate",
            source="live",
            meeting_id="mtg-abc",
        )

        rows = _read_rows(log_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "translate"
        assert row["source"] == "live"
        assert row["meeting_id"] == "mtg-abc"
        assert row["schema_version"] == _TRANSLATE_LOG_SCHEMA_VERSION

    def test_default_source_is_live_empty_meeting_id(self, tmp_path, monkeypatch):
        """Callers that omit ``source`` / ``meeting_id`` still produce a
        parseable row — the aggregator's strict filter will exclude them
        from grouping, but they must not crash the logger."""
        log_path = tmp_path / "scribe-translations.jsonl"
        monkeypatch.setattr("meeting_scribe.backends.translate_vllm._TRANS_LOG_PATH", log_path)

        _log_translation(
            model="test",
            source_lang="en",
            target_lang="ja",
            text="hi",
            translated="やあ",
            elapsed_ms=10.0,
        )

        rows = _read_rows(log_path)
        assert rows[0]["source"] == "live"
        assert rows[0]["meeting_id"] == ""


class TestLiveBackendRoundTripsMeetingId:
    @pytest.mark.asyncio
    async def test_translate_threads_meeting_id_to_log(self, tmp_path, monkeypatch):
        log_path = tmp_path / "scribe-translations.jsonl"
        monkeypatch.setattr("meeting_scribe.backends.translate_vllm._TRANS_LOG_PATH", log_path)

        backend = VllmTranslateBackend(base_url="http://localhost:8010", model="test")
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=_fake_ok_response("Hi"))

        await backend.translate(
            "こんにちは",
            source_language="ja",
            target_language="en",
            meeting_id="mtg-xyz",
        )

        rows = _read_rows(log_path)
        assert len(rows) == 1
        assert rows[0]["source"] == "live"
        assert rows[0]["meeting_id"] == "mtg-xyz"
        assert rows[0]["kind"] == "translate"
