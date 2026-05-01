"""Tests for the memory-pressure canary added 2026-05-01.

Covers parsing, severity bucketing, the latest-snapshot cache that
feeds the /metrics endpoint, and the monitor loop's hysteresis logic
(no spam at the same severity level, fresh log on escalation).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from meeting_scribe.runtime import health_monitors as hm

_OK = (
    "some avg10=0.01 avg60=0.42 avg300=0.23 total=36750861\n"
    "full avg10=0.01 avg60=0.42 avg300=0.23 total=36733220\n"
)
_WARN = (
    "some avg10=15.00 avg60=8.00 avg300=4.00 total=99999999\n"
    "full avg10=0.50 avg60=0.20 avg300=0.10 total=88888888\n"
)
_CRIT_SOME = (
    "some avg10=40.00 avg60=20.00 avg300=10.00 total=99999999\n"
    "full avg10=1.00 avg60=0.50 avg300=0.20 total=88888888\n"
)
_CRIT_FULL = (
    "some avg10=20.00 avg60=10.00 avg300=5.00 total=99999999\n"
    "full avg10=8.00 avg60=4.00 avg300=2.00 total=88888888\n"
)


def test_parse_returns_none_for_garbage() -> None:
    assert hm.parse_pressure_memory("not psi") is None
    assert hm.parse_pressure_memory("") is None


def test_parse_extracts_both_lines() -> None:
    snap = hm.parse_pressure_memory(_OK)
    assert snap is not None
    assert snap.some_avg10 == 0.01
    assert snap.some_avg60 == 0.42
    assert snap.some_total_us == 36750861
    assert snap.full_avg300 == 0.23


def test_severity_ok_for_idle() -> None:
    snap = hm.parse_pressure_memory(_OK)
    assert snap is not None
    assert snap.severity() == "ok"


def test_severity_warn_when_some_above_warn_below_crit() -> None:
    snap = hm.parse_pressure_memory(_WARN)
    assert snap is not None
    assert snap.severity() == "warn"


def test_severity_crit_when_some_avg10_high() -> None:
    snap = hm.parse_pressure_memory(_CRIT_SOME)
    assert snap is not None
    assert snap.severity() == "crit"


def test_severity_crit_when_full_avg10_sustained() -> None:
    """``full`` stalls (every task blocked) escalate to CRIT even if
    the ``some`` line is only at WARN level — that's the kernel
    actively starving and one allocation away from OOM."""
    snap = hm.parse_pressure_memory(_CRIT_FULL)
    assert snap is not None
    assert snap.severity() == "crit"


def test_read_snapshot_falls_back_to_none_when_path_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(hm, "PRESSURE_PATH", tmp_path / "nope")
    assert hm.read_pressure_snapshot() is None


def test_read_snapshot_returns_parsed_when_path_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "memory"
    fake.write_text(_WARN)
    monkeypatch.setattr(hm, "PRESSURE_PATH", fake)
    snap = hm.read_pressure_snapshot()
    assert snap is not None
    assert snap.severity() == "warn"


@pytest.mark.asyncio
async def test_monitor_skips_when_psi_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog
) -> None:
    """Hosts without PSI must exit cleanly, not crash-loop the
    background task."""
    monkeypatch.setattr(hm, "PRESSURE_PATH", tmp_path / "nope")
    caplog.set_level(logging.INFO, logger=hm.logger.name)
    await hm.mem_pressure_monitor()
    assert any("not available" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_monitor_logs_warn_then_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog
) -> None:
    """Walks the loop through OK → WARN → OK and checks each
    transition emits exactly one log message at the expected level.
    Hysteresis means the second OK does not double-log."""
    fake = tmp_path / "memory"
    fake.write_text(_OK)
    monkeypatch.setattr(hm, "PRESSURE_PATH", fake)
    monkeypatch.setattr(hm, "_PRESSURE_TICK_S", 0.0)

    # Drive the loop ourselves: cancel after 3 ticks via a sentinel
    # that flips the file content between calls.
    call_count = {"n": 0}
    real_read = hm.read_pressure_snapshot

    def _staged_read():
        n = call_count["n"]
        call_count["n"] += 1
        if n == 0:
            fake.write_text(_OK)
        elif n == 1:
            fake.write_text(_WARN)
        elif n == 2:
            fake.write_text(_OK)
        else:
            raise asyncio.CancelledError()
        return real_read()

    monkeypatch.setattr(hm, "read_pressure_snapshot", _staged_read)
    caplog.set_level(logging.INFO, logger=hm.logger.name)

    with pytest.raises(asyncio.CancelledError):
        await hm.mem_pressure_monitor()

    levels = [(rec.levelno, rec.message) for rec in caplog.records]
    # First sample: OK after default ok → no log.
    # Second: WARN → warning log. Third: back to OK → info log.
    warn_logs = [m for lvl, m in levels if lvl == logging.WARNING]
    info_recovered = [m for lvl, m in levels if lvl == logging.INFO and "recovered" in m]
    assert len(warn_logs) == 1, levels
    assert "WARN" in warn_logs[0]
    assert len(info_recovered) == 1, levels


@pytest.mark.asyncio
async def test_monitor_caches_latest_snapshot_for_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The /metrics endpoint reads ``latest_pressure()`` — the monitor
    must populate that cache on every successful sample, not only on
    severity transitions."""
    fake = tmp_path / "memory"
    fake.write_text(_OK)
    monkeypatch.setattr(hm, "PRESSURE_PATH", fake)
    monkeypatch.setattr(hm, "_PRESSURE_TICK_S", 0.0)
    monkeypatch.setattr(hm, "_LATEST_PRESSURE", None)

    calls = {"n": 0}
    real_read = hm.read_pressure_snapshot

    def _once_then_cancel():
        calls["n"] += 1
        if calls["n"] >= 2:
            raise asyncio.CancelledError()
        return real_read()

    monkeypatch.setattr(hm, "read_pressure_snapshot", _once_then_cancel)
    with pytest.raises(asyncio.CancelledError):
        await hm.mem_pressure_monitor()

    cached = hm.latest_pressure()
    assert cached is not None
    assert cached.severity() == "ok"
