"""Tests for the child-node settle loop in the daemon's discovery path.

Codex P1 from plan review: hot-plug must not race the child-node
population. We test the helper that waits for the event*/hidraw*
children to appear after a USB-level add event.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_scribe.speakerphone import daemon as sp_daemon


class _FakeScan:
    """Returns ``None, None`` for the first ``misses`` calls then a real pair."""

    def __init__(self, misses: int, target: tuple[Path, Path]) -> None:
        self._misses = misses
        self._target = target
        self.calls = 0

    def __call__(self, vid: int, pid: int) -> tuple[Path | None, Path | None]:
        self.calls += 1
        if self.calls <= self._misses:
            return None, None
        return self._target


@pytest.mark.asyncio
async def test_settle_returns_when_children_arrive_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (Path("/dev/input/event42"), Path("/dev/hidraw9"))
    fake = _FakeScan(misses=2, target=target)
    monkeypatch.setattr(sp_daemon, "_scan_for_device", fake)
    found = await sp_daemon._wait_for_child_nodes(
        0x413C,
        0x8223,
        timeout_s=2.0,
        poll_ms=50,
    )
    assert found == target
    assert fake.calls >= 3


@pytest.mark.asyncio
async def test_settle_times_out_when_children_never_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_miss(vid: int, pid: int) -> tuple[Path | None, Path | None]:
        return None, None

    monkeypatch.setattr(sp_daemon, "_scan_for_device", always_miss)
    found = await sp_daemon._wait_for_child_nodes(
        0x413C,
        0x8223,
        timeout_s=0.3,
        poll_ms=50,
    )
    assert found is None


@pytest.mark.asyncio
async def test_settle_succeeds_on_first_call_when_children_already_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (Path("/dev/input/event11"), Path("/dev/hidraw0"))
    fake = _FakeScan(misses=0, target=target)
    monkeypatch.setattr(sp_daemon, "_scan_for_device", fake)
    found = await sp_daemon._wait_for_child_nodes(
        0x413C,
        0x8223,
        timeout_s=2.0,
        poll_ms=50,
    )
    assert found == target
    assert fake.calls == 1
