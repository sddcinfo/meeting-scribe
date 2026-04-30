"""Tests for the W4 meeting-start preflight gate.

The preflight runs synthetic-inference probes against ASR + translate
+ diarize before transitioning a meeting to RECORDING. The contract:

* ASR / translate are REQUIRED — wait-with-deadline admission. Probes
  retry on a fixed cadence up to ``SCRIBE_PREFLIGHT_BUDGET_S`` (default
  30 s) before returning 503. A normal cold-start warmup completes
  inside the budget; a wedged backend exhausts it and the 503 carries
  the LAST attempt's failures in the ``not_ready`` shape the existing
  frontend modal renders.
* Diarize is WARNING-ONLY — never blocks meeting start.
* Probe success = HTTP 200 + valid response schema; non-empty
  transcribed text is NOT asserted (the fixture is a tone, not speech).

Tests stub the synthetic_probe module's helpers so they don't hit
real backends; that integration is exercised by the live drill in
the plan's Tier-2 verification, not by this unit suite.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, patch

import pytest

from meeting_scribe.routes import meeting_lifecycle as ml
from meeting_scribe.runtime import state as runtime_state
from meeting_scribe.runtime.metrics import Metrics
from meeting_scribe.runtime.synthetic_probe import ProbeResult


@pytest.fixture
def state(monkeypatch) -> None:
    """Wire fresh metrics + a minimal config stub into runtime.state."""
    m = Metrics()
    monkeypatch.setattr(runtime_state, "metrics", m, raising=False)

    class _CfgStub:
        asr_vllm_url = "http://localhost:8003"
        asr_model = "Qwen/Qwen3-ASR-1.7B"
        translate_vllm_url = "http://localhost:8010"
        translate_vllm_model = ""  # auto-detect path
        diarize_url = "http://localhost:8001"

    monkeypatch.setattr(runtime_state, "config", _CfgStub(), raising=False)


def _ok(latency_ms: float = 50.0) -> ProbeResult:
    return ProbeResult(status="ok", latency_ms=latency_ms, detail=None)


def _fail(status: str, detail: str = "stub") -> ProbeResult:
    return ProbeResult(status=status, latency_ms=999.0, detail=detail)


@pytest.mark.asyncio
async def test_all_probes_ok_returns_none(state):
    """Happy path: all three probes succeed → preflight returns None
    so caller proceeds to the existing deep_backend_health gate."""
    with (
        patch.object(ml, "_meeting_start_preflight", wraps=ml._meeting_start_preflight),
        patch(
            "meeting_scribe.runtime.synthetic_probe.asr_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.translate_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.diarize_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
    ):
        result = await ml._meeting_start_preflight()
    assert result is None


@pytest.mark.asyncio
async def test_asr_failure_returns_503(state, monkeypatch):
    """ASR is REQUIRED — sustained failure exhausts the budget and
    returns 503. Response shape uses ``not_ready`` so the existing
    frontend modal renders correctly."""
    monkeypatch.setenv("SCRIBE_PREFLIGHT_BUDGET_S", "0")
    monkeypatch.setenv("SCRIBE_PREFLIGHT_RETRY_INTERVAL_S", "0")
    with (
        patch(
            "meeting_scribe.runtime.synthetic_probe.asr_synthetic_probe",
            new=AsyncMock(return_value=_fail("timeout", "wedged")),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.translate_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.diarize_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
    ):
        result = await ml._meeting_start_preflight()
    assert result is not None
    assert result.status_code == 503
    import json as _json
    body = _json.loads(result.body)
    assert body["error"] == "Backends not ready"
    assert "not_ready" in body, body
    assert any(f["backend"] == "asr" for f in body["not_ready"])
    # detail string carries the probe status + latency for triage.
    asr_detail = next(f["detail"] for f in body["not_ready"] if f["backend"] == "asr")
    assert "timeout" in asr_detail and "wedged" in asr_detail


@pytest.mark.asyncio
async def test_translate_failure_returns_503(state, monkeypatch):
    """Translate is REQUIRED — sustained failure returns 503."""
    monkeypatch.setenv("SCRIBE_PREFLIGHT_BUDGET_S", "0")
    monkeypatch.setenv("SCRIBE_PREFLIGHT_RETRY_INTERVAL_S", "0")
    with (
        patch(
            "meeting_scribe.runtime.synthetic_probe.asr_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.translate_synthetic_probe",
            new=AsyncMock(return_value=_fail("http_error", "status 500")),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.diarize_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
    ):
        result = await ml._meeting_start_preflight()
    assert result is not None
    assert result.status_code == 503
    import json as _json
    body = _json.loads(result.body)
    assert body["error"] == "Backends not ready"
    assert any(f["backend"] == "translate" for f in body["not_ready"])


@pytest.mark.asyncio
async def test_diarize_failure_is_warning_only(state, caplog):
    """Diarize is WARNING-ONLY — failure logs a warning but the
    preflight still returns None (meeting start proceeds)."""
    import logging as _logging

    with (
        patch(
            "meeting_scribe.runtime.synthetic_probe.asr_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.translate_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.diarize_synthetic_probe",
            new=AsyncMock(return_value=_fail("timeout", "diarize wedged")),
        ),
        caplog.at_level(_logging.WARNING, logger=ml.logger.name),
    ):
        result = await ml._meeting_start_preflight()
    assert result is None  # diarize did NOT block the start
    assert any(
        "diarize_degraded" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_both_required_failures_in_one_response(state, monkeypatch):
    """If ASR AND translate both fail, the 503 includes both."""
    monkeypatch.setenv("SCRIBE_PREFLIGHT_BUDGET_S", "0")
    monkeypatch.setenv("SCRIBE_PREFLIGHT_RETRY_INTERVAL_S", "0")
    with (
        patch(
            "meeting_scribe.runtime.synthetic_probe.asr_synthetic_probe",
            new=AsyncMock(return_value=_fail("schema_error", "bad json")),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.translate_synthetic_probe",
            new=AsyncMock(return_value=_fail("timeout", "translate wedged")),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.diarize_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
    ):
        result = await ml._meeting_start_preflight()
    assert result is not None
    assert result.status_code == 503
    import json as _json
    body = _json.loads(result.body)
    backends = {f["backend"] for f in body["not_ready"]}
    assert backends == {"asr", "translate"}


@pytest.mark.asyncio
async def test_wait_with_deadline_succeeds_on_retry(state, monkeypatch):
    """The signature cold-start case: ASR fails initially but warms up
    on the second attempt. Preflight should succeed without a 503."""
    monkeypatch.setenv("SCRIBE_PREFLIGHT_BUDGET_S", "5")
    monkeypatch.setenv("SCRIBE_PREFLIGHT_RETRY_INTERVAL_S", "0")

    asr_results = iter([_fail("timeout", "cold"), _ok(latency_ms=120.0)])

    async def _fake_asr(*_args, **_kw):
        return next(asr_results)

    with (
        patch(
            "meeting_scribe.runtime.synthetic_probe.asr_synthetic_probe",
            new=AsyncMock(side_effect=_fake_asr),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.translate_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.diarize_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
    ):
        result = await ml._meeting_start_preflight()
    assert result is None  # warmup succeeded inside budget


@pytest.mark.asyncio
async def test_wait_with_deadline_returns_last_failure_on_exhaustion(state, monkeypatch):
    """Sustained failure across multiple retries → 503 with the LAST
    attempt's failure shape, not the first."""
    monkeypatch.setenv("SCRIBE_PREFLIGHT_BUDGET_S", "0.05")
    monkeypatch.setenv("SCRIBE_PREFLIGHT_RETRY_INTERVAL_S", "0")

    failures = [
        _fail("timeout", "first"),
        _fail("timeout", "second"),
        _fail("schema_error", "third"),
    ]
    fail_iter = iter(failures)

    async def _fake_asr(*_args, **_kw):
        try:
            return next(fail_iter)
        except StopIteration:
            return failures[-1]

    with (
        patch(
            "meeting_scribe.runtime.synthetic_probe.asr_synthetic_probe",
            new=AsyncMock(side_effect=_fake_asr),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.translate_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.diarize_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
    ):
        result = await ml._meeting_start_preflight()
    assert result is not None
    assert result.status_code == 503
    import json as _json
    body = _json.loads(result.body)
    asr_detail = next(f["detail"] for f in body["not_ready"] if f["backend"] == "asr")
    # Latest probe was schema_error/third. First/second must NOT be the surfaced state.
    assert "schema_error" in asr_detail and "third" in asr_detail


@pytest.mark.asyncio
async def test_translate_model_falls_back_to_recipe_default(state, monkeypatch):
    """When config.translate_vllm_model is empty, preflight uses the
    canonical recipe default (Qwen3.6-35B-A3B-FP8)."""
    captured: dict = {}

    async def _capture_translate(base_url, model_id, histogram):
        captured["model_id"] = model_id
        return _ok()

    with (
        patch(
            "meeting_scribe.runtime.synthetic_probe.asr_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.translate_synthetic_probe",
            new=AsyncMock(side_effect=_capture_translate),
        ),
        patch(
            "meeting_scribe.runtime.synthetic_probe.diarize_synthetic_probe",
            new=AsyncMock(return_value=_ok()),
        ),
    ):
        await ml._meeting_start_preflight()
    assert captured["model_id"] == "Qwen/Qwen3.6-35B-A3B-FP8"
