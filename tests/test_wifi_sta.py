"""Tests for wifi_sta.py — iw scan parser, channel preflight, layered
egress probe, owned-keyfile scope, duplicate UUID detection, posture
markers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from meeting_scribe import wifi_sta as sta


_IW_SCAN_FIXTURE = dedent(
    """
    BSS aa:bb:cc:dd:ee:ff(on wlan_sta)
        TSF: 12345 usec
        freq: 5180
        beacon interval: 100 TUs
        capability: ESS Privacy (0x0411)
        signal: -42.00 dBm
        last seen: 0 ms ago
        SSID: yunomotocho
        DS Parameter set: channel 36
        RSN:    * Version: 1
                * Group cipher: CCMP
    BSS 11:22:33:44:55:66(on wlan_sta)
        freq: 2412
        signal: -67.00 dBm
        SSID: cafe-net
        DS Parameter set: channel 1
    BSS aa:bb:cc:dd:ee:00(on wlan_sta)
        freq: 5260
        signal: -52.00 dBm
        SSID: yunomotocho
        DS Parameter set: channel 52
        RSN:    * Version: 1
    """
).strip()


def test_parse_iw_scan_extracts_three_bss() -> None:
    entries = sta.parse_iw_scan(_IW_SCAN_FIXTURE)
    assert len(entries) == 3
    assert entries[0].ssid == "yunomotocho"
    assert entries[0].channel == 36
    assert entries[0].rsn_present is True
    assert entries[1].ssid == "cafe-net"
    assert entries[1].channel == 1


def test_channel_preflight_accepts_matching_channel() -> None:
    scan = sta.parse_iw_scan(_IW_SCAN_FIXTURE)
    ok, reason = sta.channel_preflight(
        target_ssid="yunomotocho",
        target_bssid=None,
        ap_channel=36,
        scan=scan,
    )
    assert ok
    assert reason is None


def test_channel_preflight_refuses_cross_channel() -> None:
    """When the target SSID is visible only on a non-AP channel, refuse
    — single-radio dual-mode pins both AP + STA to the AP's channel."""
    scan = sta.parse_iw_scan(_IW_SCAN_FIXTURE)
    ok, reason = sta.channel_preflight(
        target_ssid="yunomotocho",
        target_bssid="aa:bb:cc:dd:ee:00",  # the channel-52 BSS
        ap_channel=36,
        scan=scan,
    )
    assert not ok
    assert reason == "cross_channel"


def test_channel_preflight_refuses_unknown_ssid() -> None:
    scan = sta.parse_iw_scan(_IW_SCAN_FIXTURE)
    ok, reason = sta.channel_preflight(
        target_ssid="not-on-the-air",
        target_bssid=None,
        ap_channel=36,
        scan=scan,
    )
    assert not ok
    assert reason == "no_match"


# ── Layered egress probe ────────────────────────────────────────


def test_egress_probe_short_circuits_on_first_failure() -> None:
    """If link is down, DHCP/DNS/HTTP probes are NEVER called — the
    failure cascade is unambiguous."""
    calls: list[str] = []

    async def link() -> bool:
        calls.append("link")
        return False

    async def dhcp() -> bool:
        calls.append("dhcp")
        return True

    async def dns() -> bool:
        calls.append("dns")
        return True

    async def http() -> bool:
        calls.append("http")
        return True

    async def scenario() -> sta.EgressProbe:
        return await sta.layered_egress_check(
            link_probe=link, dhcp_probe=dhcp, dns_probe=dns, http_probe=http
        )

    result = asyncio.run(scenario())
    assert result.ok is False
    assert result.first_failure == "link"
    assert calls == ["link"]


def test_egress_probe_runs_through_when_each_layer_passes() -> None:
    async def yes() -> bool:
        return True

    async def scenario() -> sta.EgressProbe:
        return await sta.layered_egress_check(
            link_probe=yes, dhcp_probe=yes, dns_probe=yes, http_probe=yes
        )

    result = asyncio.run(scenario())
    assert result.ok is True
    assert result.first_failure is None


def test_egress_probe_reports_dhcp_when_link_ok_but_no_lease() -> None:
    async def yes() -> bool:
        return True

    async def no() -> bool:
        return False

    async def scenario() -> sta.EgressProbe:
        return await sta.layered_egress_check(
            link_probe=yes, dhcp_probe=no, dns_probe=yes, http_probe=yes
        )

    result = asyncio.run(scenario())
    assert result.ok is False
    assert result.first_failure == "dhcp"


# ── Owned-keyfile scope ────────────────────────────────────────


def test_is_owned_keyfile_path_only(tmp_path: Path) -> None:
    """Plan 2 §15: ownership decided by filename, never by content."""
    owned = tmp_path / "meeting-scribe-sta-yunomotocho"
    owned.write_text("not even valid INI")
    assert sta.is_owned_keyfile(owned) is True

    foreign = tmp_path / "Wired connection 1"
    foreign.write_text("[connection]\nuuid=meeting-scribe-sta-fakeout\n")
    assert sta.is_owned_keyfile(foreign) is False


def test_scan_owned_keyfiles_filters_by_prefix(tmp_path: Path) -> None:
    (tmp_path / "meeting-scribe-sta-a").write_text("")
    (tmp_path / "meeting-scribe-sta-b").write_text("")
    (tmp_path / "Wired connection 1").write_text("")
    (tmp_path / "subdir").mkdir()  # directories ignored
    out = sta.scan_owned_keyfiles(tmp_path)
    assert {p.name for p in out} == {"meeting-scribe-sta-a", "meeting-scribe-sta-b"}


def test_scan_owned_keyfiles_handles_missing_dir(tmp_path: Path) -> None:
    assert sta.scan_owned_keyfiles(tmp_path / "absent") == []


# ── Duplicate UUIDs ────────────────────────────────────────────


def test_find_duplicate_uuids_flags_collision(tmp_path: Path) -> None:
    """Plan 2 §16: duplicate UUIDs across keyfiles is a HARD failure."""
    a = tmp_path / "meeting-scribe-sta-a"
    b = tmp_path / "meeting-scribe-sta-b"
    a.write_text("[connection]\nuuid=collision-uuid\n")
    b.write_text("[connection]\nuuid=collision-uuid\n")
    dups = sta.find_duplicate_uuids([a, b])
    assert dups == ["collision-uuid"]


def test_find_duplicate_uuids_clean_set(tmp_path: Path) -> None:
    a = tmp_path / "meeting-scribe-sta-a"
    b = tmp_path / "meeting-scribe-sta-b"
    a.write_text("[connection]\nuuid=alpha\n")
    b.write_text("[connection]\nuuid=beta\n")
    assert sta.find_duplicate_uuids([a, b]) == []


# ── Posture markers ────────────────────────────────────────────


def test_write_and_clear_sta_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "STA-DEGRADED"
    monkeypatch.setattr(sta, "STA_DEGRADED_MARKER", marker)
    sta.write_sta_degraded("test reason")
    assert marker.exists()
    text = marker.read_text(encoding="utf-8")
    assert "test reason" in text
    sta.clear_sta_degraded()
    assert not marker.exists()


def test_boot_should_skip_returns_reason_when_marker_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "STA-DEGRADED"
    repair = tmp_path / "REPAIR-NEEDED"
    monkeypatch.setattr(sta, "STA_DEGRADED_MARKER", marker)
    monkeypatch.setattr(sta, "REPAIR_NEEDED_MARKER", repair)

    skip, reason = sta.boot_should_skip_sta()
    assert skip is False
    assert reason is None

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("blocked")
    skip, reason = sta.boot_should_skip_sta()
    assert skip is True
    assert reason is not None
    assert "STA-DEGRADED" in reason

    marker.unlink()
    repair.write_text("blocked")
    skip, reason = sta.boot_should_skip_sta()
    assert skip is True
    assert reason is not None
    assert "REPAIR-NEEDED" in reason
