"""Regression tests for the per-device native capture-rate detector.

History: the Poly Sync 20-M was originally captured at 48 kHz with a
software resampler downstream (it natively supports 48 kHz, so quality
held up). The Dell SP325 advertises **both 16 kHz and 48 kHz** for
capture; the on-device DSP is tuned for 16 kHz so forcing 48 kHz makes
ASR audio audibly worse. ``native_rate.pick_rate`` must always prefer
16 kHz for the SP325 fixture.

The /proc/asound fixture below is the exact text dumped from the
SP325 on 2026-05-13 (``cat /proc/asound/card1/stream0``).
"""

from __future__ import annotations

import pytest

from meeting_scribe.audio import native_rate

SP325_STREAM_FIXTURE = """\
Dell Inc. Dell SP325 Speakerphone at usb-NVDA8000:00-1, full speed : USB Audio

Playback:
  Status: Stop
  Interface 1
    Altset 1
    Format: S16_LE
    Channels: 2
    Endpoint: 0x01 (1 OUT) (ADAPTIVE)
    Rates: 48000
    Bits: 16
    Channel map: FL FR

Capture:
  Status: Stop
  Interface 2
    Altset 1
    Format: S16_LE
    Channels: 1
    Endpoint: 0x81 (1 IN) (ADAPTIVE)
    Rates: 16000, 48000
    Bits: 16
    Channel map: MONO
"""

POLY_STREAM_FIXTURE = """\
Plantronics Poly Sync 20-M : USB Audio

Capture:
  Status: Stop
  Interface 3
    Altset 1
    Format: S16_LE
    Channels: 1
    Endpoint: 0x83 (3 IN) (ADAPTIVE)
    Rates: 48000
    Bits: 16
    Channel map: MONO
"""


# ── _parse_rates_from_stream_file ──────────────────────────────────────


def test_sp325_fixture_exposes_both_rates() -> None:
    rates = native_rate._parse_rates_from_stream_file(SP325_STREAM_FIXTURE)
    assert rates == [16000, 48000]


def test_poly_fixture_exposes_only_48k() -> None:
    rates = native_rate._parse_rates_from_stream_file(POLY_STREAM_FIXTURE)
    assert rates == [48000]


def test_capture_only_lines_ignored_for_playback_section() -> None:
    """A Playback "Rates: 96000" line must not leak into capture rates."""
    fixture = """\
Some Device : USB Audio

Playback:
  Interface 1
    Rates: 96000

Capture:
  Interface 2
    Rates: 16000, 48000
"""
    rates = native_rate._parse_rates_from_stream_file(fixture)
    assert rates == [16000, 48000]


def test_empty_fixture_returns_empty_list() -> None:
    assert native_rate._parse_rates_from_stream_file("") == []


# ── pick_rate ──────────────────────────────────────────────────────────


def test_sp325_picks_16000() -> None:
    # The whole point of this fix — the SP325 must NOT default to 48 kHz.
    picked = native_rate.pick_rate([16000, 48000])
    assert picked == 16000


def test_poly_picks_48000_when_only_48k_supported() -> None:
    # The Poly only exposes 48 kHz, so falling through to 48000 is correct.
    picked = native_rate.pick_rate([48000])
    assert picked == 48000


def test_empty_supported_uses_default() -> None:
    picked = native_rate.pick_rate([], default=22050)
    assert picked == 22050


def test_picks_smallest_eligible_rate_when_16k_absent() -> None:
    # A device that only does 22050 and 44100 picks 22050 (the smallest
    # ≥ 16000 ≤ 48000).
    picked = native_rate.pick_rate([22050, 44100])
    assert picked == 22050


def test_skips_rates_above_max() -> None:
    # A bizarre device exposing 96 kHz only should still degrade to
    # default rather than pull 96 kHz audio.
    picked = native_rate.pick_rate([96000], default=48000)
    assert picked == 48000


def test_skips_rates_below_asr_target() -> None:
    # 8 kHz alone is below the ASR target → fall through to default.
    picked = native_rate.pick_rate([8000], default=48000)
    assert picked == 48000


# ── detect_capture_rate end-to-end (mocked) ────────────────────────────


def test_detect_capture_rate_returns_default_for_none_node() -> None:
    assert native_rate.detect_capture_rate(None, default=48000) == 48000
    assert native_rate.detect_capture_rate("", default=48000) == 48000


def test_detect_capture_rate_returns_default_when_wpctl_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_rate, "_wpctl_alsa_card_dev", lambda _n: None)
    assert native_rate.detect_capture_rate("nonexistent_node", default=44100) == 44100


def test_detect_capture_rate_picks_16k_for_sp325(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """End-to-end: SP325 wpctl resolution + /proc/asound parse → 16000."""
    monkeypatch.setattr(
        native_rate,
        "_wpctl_alsa_card_dev",
        lambda _n: (1, 0),
    )
    monkeypatch.setattr(
        native_rate,
        "supported_capture_rates",
        lambda _c: [16000, 48000],
    )
    picked = native_rate.detect_capture_rate(
        "alsa_input.usb-Dell_Inc._Dell_SP325_..._pro-input-0",
    )
    assert picked == 16000


def test_detect_capture_rate_picks_48k_for_poly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_rate,
        "_wpctl_alsa_card_dev",
        lambda _n: (2, 0),
    )
    monkeypatch.setattr(
        native_rate,
        "supported_capture_rates",
        lambda _c: [48000],
    )
    picked = native_rate.detect_capture_rate("alsa_input.usb-Plantronics_Poly_Sync_20-M...")
    assert picked == 48000


# ── ServerMicCapture frame-size derivation ─────────────────────────────


def test_frame_bytes_for_scales_with_rate() -> None:
    from meeting_scribe.audio.server_mic import _frame_bytes_for, _rate_header_for

    # 16 kHz mono int16, 50 ms → 16000 * 0.05 * 2 = 1600 bytes.
    assert _frame_bytes_for(16000) == 1600
    # 48 kHz mono int16, 50 ms → 48000 * 0.05 * 2 = 4800 bytes.
    assert _frame_bytes_for(48000) == 4800
    # Rate header is a 4-byte little-endian uint32.
    assert _rate_header_for(16000) == b"\x80\x3e\x00\x00"
    assert _rate_header_for(48000) == b"\x80\xbb\x00\x00"
