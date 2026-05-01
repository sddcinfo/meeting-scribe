"""Tests for meeting_scribe.hf_preflight.

Branch coverage for all 5 HfStatus values plus the
`ValidationReport.ok` / `has_only_network_failures` predicate
contract that prevents the customer-side blocking probe from
silently degrading on network failures.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from meeting_scribe.hf_preflight import (
    HfModelResult,
    HfStatus,
    ValidationReport,
    validate_hf_access,
)


def _ok_model_info(sha: str = "abc123def"):
    """Stub HfApi.model_info return: a SimpleNamespace-like with `.sha`."""
    info = MagicMock()
    info.sha = sha
    return info


def _mock_response(status: int = 500) -> MagicMock:
    """Build an httpx.Response-shaped Mock acceptable to huggingface_hub
    1.13+ error constructors (which require a response keyword)."""
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    response.request = MagicMock()
    return response


def _http_error(status: int) -> HfHubHTTPError:
    """Construct an HfHubHTTPError with a status code we can read back."""
    return HfHubHTTPError(f"HTTP {status}", response=_mock_response(status))


def _gated_error() -> GatedRepoError:
    return GatedRepoError("EULA not accepted", response=_mock_response(403))


def _not_found_error() -> RepositoryNotFoundError:
    return RepositoryNotFoundError("not found", response=_mock_response(404))


# ── ValidationReport contract ─────────────────────────────────


def test_ok_true_when_every_result_is_OK() -> None:
    report = ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[
            HfModelResult("a", HfStatus.OK),
            HfModelResult("b", HfStatus.OK),
        ],
    )
    assert report.ok is True
    assert report.has_only_network_failures is False


def test_ok_false_when_any_network_error() -> None:
    """The customer-side probe is the sole authority for
    RuntimeManifest.model_revisions — a NETWORK_ERROR must block."""
    report = ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[
            HfModelResult("a", HfStatus.OK),
            HfModelResult("b", HfStatus.NETWORK_ERROR),
        ],
    )
    assert report.ok is False


def test_has_only_network_failures_true_when_all_non_OK_are_network() -> None:
    report = ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[
            HfModelResult("a", HfStatus.OK),
            HfModelResult("b", HfStatus.NETWORK_ERROR),
        ],
    )
    assert report.has_only_network_failures is True


def test_has_only_network_failures_false_when_any_token_or_eula_failure() -> None:
    """Mixed network+EULA failure is NOT 'network only' — token/EULA
    failures must always block."""
    report = ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[
            HfModelResult("a", HfStatus.NETWORK_ERROR),
            HfModelResult("b", HfStatus.GATED_NOT_ACCEPTED),
        ],
    )
    assert report.has_only_network_failures is False


def test_has_only_network_failures_false_when_no_network_failures() -> None:
    """All-OK and all-non-network-failure both must return False."""
    report = ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[HfModelResult("a", HfStatus.OK)],
    )
    assert report.has_only_network_failures is False
    report2 = ValidationReport(
        token_prefix="hf_xxxx",
        whoami=None,
        results=[HfModelResult("a", HfStatus.BAD_TOKEN)],
    )
    assert report2.has_only_network_failures is False


def test_url_auto_populates_from_model_id() -> None:
    r = HfModelResult("Qwen/Qwen3.6-35B-A3B-FP8", HfStatus.OK)
    assert r.url == "https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8"


def test_render_only_lists_failing_rows() -> None:
    report = ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[
            HfModelResult("ok-model", HfStatus.OK),
            HfModelResult(
                "Qwen/Qwen3.6-35B-A3B-FP8",
                HfStatus.GATED_NOT_ACCEPTED,
            ),
        ],
    )
    out = report.render()
    assert "ok-model" not in out
    assert "Qwen/Qwen3.6-35B-A3B-FP8" in out
    assert "Agree and access repository" in out
    assert "https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8" in out


# ── validate_hf_access branch coverage ────────────────────────


@patch("meeting_scribe.hf_preflight.HfApi")
def test_validates_ok_when_all_models_accessible(mock_api_cls) -> None:
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    api.model_info.side_effect = lambda mid, **kw: _ok_model_info(sha=f"sha-of-{mid}")
    mock_api_cls.return_value = api

    report = validate_hf_access("hf_xxxxxxxx", ["a", "b", "c"])
    assert report.ok is True
    assert report.whoami == "alice"
    assert {r.model_id for r in report.results} == {"a", "b", "c"}
    assert {r.revision for r in report.results} == {"sha-of-a", "sha-of-b", "sha-of-c"}


@patch("meeting_scribe.hf_preflight.HfApi")
def test_detects_bad_token_via_whoami(mock_api_cls) -> None:
    api = MagicMock()
    api.whoami.side_effect = _http_error(401)
    mock_api_cls.return_value = api

    report = validate_hf_access("hf_xxxx", ["a", "b"])
    assert report.whoami is None
    assert all(r.status is HfStatus.BAD_TOKEN for r in report.results)
    out = report.render()
    assert "settings/tokens" in out


@patch("meeting_scribe.hf_preflight.HfApi")
def test_detects_gated_not_accepted(mock_api_cls) -> None:
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}

    def _model_info(mid, **kw):
        if mid == "Qwen/Qwen3.6-35B-A3B-FP8":
            raise _gated_error()
        return _ok_model_info()

    api.model_info.side_effect = _model_info
    mock_api_cls.return_value = api

    report = validate_hf_access(
        "hf_xxxx",
        ["Qwen/Qwen3.6-35B-A3B-FP8", "Qwen/Qwen3-ASR-1.7B"],
    )
    assert report.ok is False
    statuses = {r.model_id: r.status for r in report.results}
    assert statuses["Qwen/Qwen3.6-35B-A3B-FP8"] is HfStatus.GATED_NOT_ACCEPTED
    assert statuses["Qwen/Qwen3-ASR-1.7B"] is HfStatus.OK
    out = report.render()
    assert "https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8" in out
    assert "Agree and access repository" in out


@patch("meeting_scribe.hf_preflight.HfApi")
def test_detects_not_found_for_typo(mock_api_cls) -> None:
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    api.model_info.side_effect = _not_found_error()
    mock_api_cls.return_value = api

    report = validate_hf_access("hf_xxxx", ["nonexistent/model"])
    assert report.ok is False
    assert report.results[0].status is HfStatus.NOT_FOUND


@patch("meeting_scribe.hf_preflight.time.sleep")
@patch("meeting_scribe.hf_preflight.HfApi")
def test_network_error_retries_then_marks_non_ok(mock_api_cls, mock_sleep) -> None:
    """Network errors must retry (per network_retries) and then mark
    NETWORK_ERROR. report.ok MUST be False (the strict contract from
    plan §1.1 fix)."""
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    api.model_info.side_effect = ConnectionError("dns failure")
    mock_api_cls.return_value = api

    report = validate_hf_access("hf_xxxx", ["a"], network_retries=3, network_backoff_s=0)
    assert report.ok is False
    assert report.has_only_network_failures is True
    assert report.results[0].status is HfStatus.NETWORK_ERROR
    # Exactly network_retries attempts before giving up:
    assert api.model_info.call_count == 3


@patch("meeting_scribe.hf_preflight.time.sleep")
@patch("meeting_scribe.hf_preflight.HfApi")
def test_network_then_recover(mock_api_cls, mock_sleep) -> None:
    """Verify the retry loop succeeds on attempt N when the first N-1 fail."""
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    info = _ok_model_info()
    api.model_info.side_effect = [ConnectionError("dns"), info]
    mock_api_cls.return_value = api

    report = validate_hf_access("hf_xxxx", ["a"], network_retries=3, network_backoff_s=0)
    assert report.ok is True
    assert report.results[0].status is HfStatus.OK
    assert api.model_info.call_count == 2


@patch("meeting_scribe.hf_preflight.HfApi")
def test_403_classified_as_gated(mock_api_cls) -> None:
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    api.model_info.side_effect = _http_error(403)
    mock_api_cls.return_value = api

    report = validate_hf_access("hf_xxxx", ["a"])
    assert report.results[0].status is HfStatus.GATED_NOT_ACCEPTED


def test_render_truncates_long_network_detail() -> None:
    long_detail = "x" * 500
    report = ValidationReport(
        token_prefix="hf_xx",
        whoami="alice",
        results=[
            HfModelResult("a", HfStatus.NETWORK_ERROR, detail=long_detail),
        ],
    )
    out = report.render()
    # Truncated to 120 chars + ellipsis
    assert "x" * 120 in out
    assert "x" * 200 not in out


# ── recipe-driven model id integration ────────────────────────


def test_uses_recipe_model_ids() -> None:
    """The validation function takes an arbitrary list, but in real use
    the orchestrator passes the recipe-derived list. Sanity-check that
    `all_model_ids(include_shared=True)` returns the four gated models."""
    from meeting_scribe.recipes import all_model_ids

    ids = set(all_model_ids(include_shared=True))
    expected = {
        "Qwen/Qwen3.6-35B-A3B-FP8",
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "pyannote/speaker-diarization-community-1",
    }
    assert ids == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
