"""Phase 6 — ``preflight --mode=demo`` + ``--skip-hardware`` tests.

The demo gate is what ``meeting-scribe demo-smoke`` runs on the GB10
before the E2E flow, and what CI runs (with ``--skip-hardware``) so
the same gate exercises CI without an attached SP325 / wifi radio.

Tests focus on the gate's *aggregation* + the ``--skip-hardware``
short-circuit; the individual probe contracts (mic liveness, SP325
compliance, AP profile validity, hotspot up) are covered by their
dedicated probe-level tests elsewhere.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from meeting_scribe import preflight as pf


def _mk_ctx(tmp_path):
    return pf.CheckContext(
        repo_root=tmp_path,
        compose_file=tmp_path / "compose.yml",
        env_file=tmp_path / ".env",
        deadline=0.0,  # zero deadline = no retries in run_demo_checks
    )


@pytest.mark.asyncio
async def test_skip_hardware_short_circuits_all_demo_checks(tmp_path):
    """Every hardware_required check must return passed=True with
    ``detail`` mentioning the skip reason. Verifies the contract that
    CI runs of the gate don't shell out to nmcli / pw-record / Dell
    Peripheral Manager."""
    ctx = _mk_ctx(tmp_path)
    results = await pf.run_demo_checks(ctx, skip_hardware=True)
    assert len(results) == len(pf.DEMO_CHECKS)
    for r in results:
        assert r.passed is True, f"{r.name}: skip-hardware should pass — got {r.detail}"
        assert r.hardware_required is True
        assert "skipped" in r.detail.lower()


@pytest.mark.asyncio
async def test_skip_hardware_false_runs_each_check(tmp_path, monkeypatch):
    """When skip_hardware=False the actual check functions run.
    Mock each to return a known result and verify they're all invoked."""
    ctx = _mk_ctx(tmp_path)
    calls: list[str] = []

    async def _stub(check_name: str):
        async def _inner(_ctx):
            calls.append(check_name)
            return pf.CheckResult(
                name=check_name,
                passed=True,
                detail="mocked",
                phase=2,
                duration_ms=1,
                hardware_required=True,
            )

        return _inner

    monkeypatch.setattr(
        pf,
        "DEMO_CHECKS",
        (
            await _stub("mic_bound_and_live"),
            await _stub("sp325_compliance_ok"),
            await _stub("ap_profile_valid"),
            await _stub("hotspot_up"),
        ),
    )
    results = await pf.run_demo_checks(ctx, skip_hardware=False)
    assert [r.name for r in results] == [
        "mic_bound_and_live",
        "sp325_compliance_ok",
        "ap_profile_valid",
        "hotspot_up",
    ]
    assert calls == [
        "mic_bound_and_live",
        "sp325_compliance_ok",
        "ap_profile_valid",
        "hotspot_up",
    ]


@pytest.mark.asyncio
async def test_mic_check_failure_when_server_not_reachable(tmp_path, monkeypatch):
    """The demo gate's mic check queries /api/status (server's live
    state, not the CLI subprocess's empty state). When /api/status is
    unreachable the check must surface passed=False with an actionable
    detail — not silently treat the empty CLI state as authoritative."""
    from meeting_scribe.cli import _common as cli_common

    monkeypatch.setattr(cli_common, "_api_request", lambda *_a, **_kw: None)
    ctx = _mk_ctx(tmp_path)
    result = await pf._check_mic_bound_and_live(ctx)
    assert result.passed is False
    assert "could not reach" in result.detail.lower()
    assert result.hardware_required is True


@pytest.mark.asyncio
async def test_mic_check_failure_when_audio_route_status_not_ok(tmp_path, monkeypatch):
    """When the running server reports a failure-state audio route
    (Phase 1's ambiguous/unresolved/capture_failed), the demo gate
    must refuse so the operator dismisses the banner first."""
    from meeting_scribe.cli import _common as cli_common

    monkeypatch.setattr(
        cli_common,
        "_api_request",
        lambda *_a, **_kw: {
            "audio_route_status": "ambiguous",
            "server_mic_active_live": False,
            "metrics": {"audio_chunks": 0},
        },
    )
    ctx = _mk_ctx(tmp_path)
    result = await pf._check_mic_bound_and_live(ctx)
    assert result.passed is False
    assert "ambiguous" in result.detail


@pytest.mark.asyncio
async def test_mic_check_passes_when_audio_chunks_advance(tmp_path, monkeypatch):
    """Two consecutive /api/status reads with advancing audio_chunks
    proves the capture pipeline is delivering frames — the demo gate
    accepts this as evidence the mic is live."""
    from meeting_scribe.cli import _common as cli_common

    responses = iter(
        [
            {
                "audio_route_status": "ok",
                "server_mic_active_live": True,
                "metrics": {"audio_chunks": 100},
            },
            {
                "audio_route_status": "ok",
                "server_mic_active_live": True,
                "metrics": {"audio_chunks": 110},
            },
        ]
    )
    monkeypatch.setattr(cli_common, "_api_request", lambda *_a, **_kw: next(responses))
    ctx = _mk_ctx(tmp_path)
    result = await pf._check_mic_bound_and_live(ctx)
    assert result.passed is True
    assert "advanced" in result.detail
    assert result.hardware_required is True


@pytest.mark.asyncio
async def test_mic_check_fails_when_audio_chunks_stuck(tmp_path, monkeypatch):
    """Counter frozen across both reads → demo gate fails (the actual
    pw-record-hung failure mode caught on the GB10 2026-05-14)."""
    from meeting_scribe.cli import _common as cli_common

    responses = iter(
        [
            {
                "audio_route_status": "ok",
                "server_mic_active_live": True,
                "metrics": {"audio_chunks": 990},
            },
            {
                "audio_route_status": "ok",
                "server_mic_active_live": True,
                "metrics": {"audio_chunks": 990},
            },
        ]
    )
    monkeypatch.setattr(cli_common, "_api_request", lambda *_a, **_kw: next(responses))
    ctx = _mk_ctx(tmp_path)
    result = await pf._check_mic_bound_and_live(ctx)
    assert result.passed is False
    assert "stuck" in result.detail.lower()
    assert "990" in result.detail


@pytest.mark.asyncio
async def test_ap_profile_check_passes_when_key_mgmt_set(tmp_path, monkeypatch):
    """nmcli output with non-empty key-mgmt → passed=True."""
    fake_proc = MagicMock(stdout="802-11-wireless-security.key-mgmt:sae\n", returncode=0)
    monkeypatch.setattr("meeting_scribe.preflight.subprocess.run", lambda *a, **k: fake_proc)
    result = await pf._check_ap_profile_valid(_mk_ctx(tmp_path))
    assert result.passed is True
    assert "sae" in result.detail


@pytest.mark.asyncio
async def test_ap_profile_check_fails_when_key_mgmt_missing(tmp_path, monkeypatch):
    """nmcli output with empty key-mgmt → passed=False (the
    2026-05-14 demo failure pattern)."""
    fake_proc = MagicMock(stdout="802-11-wireless-security.key-mgmt:\n", returncode=0)
    monkeypatch.setattr("meeting_scribe.preflight.subprocess.run", lambda *a, **k: fake_proc)
    result = await pf._check_ap_profile_valid(_mk_ctx(tmp_path))
    assert result.passed is False
    assert "unset" in result.detail.lower()


@pytest.mark.asyncio
async def test_hotspot_check_uses_rotation_snapshot_when_available(tmp_path, monkeypatch):
    """If a recent rotation has happened, the hotspot check reads the
    cached rotation_ok flag instead of shelling out to nmcli."""
    from meeting_scribe.hotspot import ap_control

    monkeypatch.setattr(ap_control, "_LAST_ROTATION_AT", 1000.0)
    monkeypatch.setattr(ap_control, "_LAST_ROTATION_OK", True)
    monkeypatch.setattr(ap_control, "_LAST_ROTATION_TARGET_SSID", "Dell-AP-1")
    monkeypatch.setattr(ap_control, "_LAST_ROTATION_LIVE_SSID", "Dell-AP-1")
    # Set a subprocess fail so we know it wasn't called as the fall-through.
    bad_run = MagicMock(side_effect=AssertionError("must not shell out when snapshot present"))
    monkeypatch.setattr("meeting_scribe.preflight.subprocess.run", bad_run)
    result = await pf._check_hotspot_up(_mk_ctx(tmp_path))
    assert result.passed is True


@pytest.mark.asyncio
async def test_hotspot_check_fails_when_rotation_mismatch_recorded(tmp_path, monkeypatch):
    from meeting_scribe.hotspot import ap_control

    monkeypatch.setattr(ap_control, "_LAST_ROTATION_AT", 1000.0)
    monkeypatch.setattr(ap_control, "_LAST_ROTATION_OK", False)
    monkeypatch.setattr(ap_control, "_LAST_ROTATION_TARGET_SSID", "Dell-AP-NEW")
    monkeypatch.setattr(ap_control, "_LAST_ROTATION_LIVE_SSID", "Dell-AP-OLD")
    result = await pf._check_hotspot_up(_mk_ctx(tmp_path))
    assert result.passed is False
    assert "Dell-AP-NEW" in result.detail
    assert "Dell-AP-OLD" in result.detail


def test_check_result_carries_hardware_required_flag():
    """Phase 6 added the hardware_required field; the dataclass must
    accept it and default to False (backward compatibility with the
    existing Phase 0/1/2 checks)."""
    r1 = pf.CheckResult(name="legacy", passed=True, detail="ok", phase=0, duration_ms=1)
    assert r1.hardware_required is False
    r2 = pf.CheckResult(
        name="demo-check",
        passed=True,
        detail="ok",
        phase=2,
        duration_ms=1,
        hardware_required=True,
    )
    assert r2.hardware_required is True


def test_classify_exit_ok_when_all_passed_with_skip_hardware():
    """The skip-hardware short-circuit must produce a green exit code."""
    results = [
        pf.CheckResult(
            name=f"check{i}",
            passed=True,
            detail="skipped",
            phase=2,
            duration_ms=0,
            hardware_required=True,
        )
        for i in range(4)
    ]
    assert pf.classify_exit(results) == pf.EXIT_OK
