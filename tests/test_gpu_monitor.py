"""Tests for GPU VRAM monitor."""

from __future__ import annotations

import pytest

from meeting_scribe.gpu_monitor import VRAMUsage, get_vram_usage


class TestVRAMUsage:
    def test_percentage_calculation(self):
        v = VRAMUsage(used_mb=50000, total_mb=128000)
        assert v.pct == pytest.approx(39.06, abs=0.1)

    def test_free_mb(self):
        v = VRAMUsage(used_mb=50000, total_mb=128000)
        assert v.free_mb == 78000

    def test_zero_total_no_crash(self):
        v = VRAMUsage(used_mb=0, total_mb=0)
        assert v.pct == 0.0

    def test_high_usage(self):
        v = VRAMUsage(used_mb=120000, total_mb=128000)
        assert v.pct > 90


class TestGetVRAMUsage:
    def test_returns_usage_or_none(self):
        """get_vram_usage should return VRAMUsage on GB10, None if no GPU."""
        result = get_vram_usage()
        if result is not None:
            assert isinstance(result, VRAMUsage)
            assert result.total_mb > 0
            assert result.used_mb >= 0
            assert result.used_mb <= result.total_mb
