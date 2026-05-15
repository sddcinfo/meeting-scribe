"""Empirical compliance check for the SP325 wideband audio config.

Why behavioral, not configurational
-----------------------------------

The SP325's USB descriptor reports a fixed ``bcdDevice = 1.00`` regardless
of firmware version. Dell's actual firmware version string
("1.3.6.0" as of 2026-05) lives in the device's NVRAM and is only readable
via the HID vendor protocol Dell Peripheral Manager uses — which isn't
publicly documented. The same is true for the AI noise-cancellation
toggle and the EQ preset: they live in the device, controllable from
Windows DPM, opaque from Linux.

So instead of querying configuration, we **measure outcome**. A 5-second
capture from the configured mic node, an FFT, and the share of energy
above 4 kHz tells us whether the device is delivering its full 100 Hz –
14 kHz spec (the datasheet number) or has reverted to telephony-band
mode (≤1 kHz). Empirically, on a verified-good SP325 with the user's
working config (firmware 1.3.6.0, both NR toggles OFF, EQ preset
Default), the noise-floor ``high_band_pct`` measures ≥ 3 %; under load
(actual speech) it climbs to 10–15 %. A bad config drops it to <1 %.

The check covers every way the device could regress:

* Firmware downgrade or fresh-out-of-box install
* User toggled NR back on in DPM
* Factory reset
* Replaced unit with one running older firmware
* Cable/USB-hub change that triggered re-enumeration with stale settings

All of those manifest as the same observable: high-band energy collapses
to the telephony band. One check covers them all.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Minimum share of total power above 4 kHz, measured on a captured
# WAV with at least some ambient noise. On a wideband SP325 the
# noise floor alone clears this; a telephony-bandlimited stream
# bottoms out at ~0.5 %. Set well below the lowest "good" observation
# (3.29 %) but above the highest "bad" one (0.66 %), with margin.
MIN_HIGH_BAND_PCT: float = 1.5

# Same intent for "above the telephony 3.4 kHz wall". Helps catch
# devices that have some residual 4–6 kHz leakage but everything
# else is still telephony-band-locked.
MIN_ROLLOFF_3400HZ_PCT: float = 1.5

# Capture window — long enough to average out short transients,
# short enough to keep startup snappy.
CAPTURE_SECONDS: float = 5.0


@dataclass(frozen=True)
class ComplianceResult:
    """Outcome of a single compliance probe."""

    status: str  # "pass" | "warn" | "fail"
    high_band_pct: float
    rolloff_3400hz_pct: float
    rms: float
    reason: str
    bands_pct: dict[str, float]

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "high_band_pct": self.high_band_pct,
            "rolloff_3400hz_pct": self.rolloff_3400hz_pct,
            "rms": self.rms,
            "reason": self.reason,
            "bands_pct": self.bands_pct,
        }


_BANDS_HZ: list[tuple[int, int]] = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 24000),
]


def _analyze_wav(path: Path) -> tuple[float, dict[str, float], float, float]:
    """Compute (rms, bands_pct, high_band_pct, rolloff_3400_pct).

    Pure-Python wrapper around numpy.fft.rfft so this stays a single
    module without pulling scipy.
    """
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise RuntimeError(
                f"{path}: expected mono int16; got ch={wf.getnchannels()} sw={wf.getsampwidth()}",
            )
        frames = wf.readframes(wf.getnframes())

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return 0.0, {}, 0.0, 0.0

    rms = float(np.sqrt(np.mean(samples * samples)))
    win = np.hanning(samples.size)
    spec = np.abs(np.fft.rfft(samples * win))
    power = spec * spec
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / rate)
    total = float(power.sum()) or 1.0

    bands_pct: dict[str, float] = {}
    for lo, hi in _BANDS_HZ:
        hi_eff = min(hi, rate / 2)
        if hi_eff <= lo:
            bands_pct[f"{lo}-{hi}Hz"] = 0.0
            continue
        mask = (freqs >= lo) & (freqs < hi_eff)
        bands_pct[f"{lo}-{hi}Hz"] = round(100.0 * float(power[mask].sum()) / total, 2)

    high_band = bands_pct.get("4000-6000Hz", 0.0) + bands_pct.get("6000-24000Hz", 0.0)
    above_3400 = float(power[freqs >= 3400].sum())
    rolloff_pct = round(100.0 * above_3400 / total, 2)
    return rms, bands_pct, round(high_band, 2), rolloff_pct


def _capture_window(target_node: str, seconds: float, out_path: Path) -> bool:
    """Spawn pw-record for ``seconds`` and return success.

    Returns ``False`` if pw-record can't be reached or produces an
    empty file — caller turns that into a ``fail`` ComplianceResult
    so the daemon's structured warning is uniform regardless of the
    failure mode.
    """
    try:
        proc = subprocess.Popen(
            [
                "pw-record",
                f"--target={target_node}",
                "--format=s16",
                "--rate=16000",
                "--channels=1",
                str(out_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.warning("compliance: pw-record not on PATH; skipping")
        return False

    try:
        proc.wait(timeout=seconds + 1.5)
        # pw-record runs until killed; the wait above will time out
        # in normal cases. If it returned before the window elapsed
        # the device is in a bad state.
        if proc.returncode is not None:
            logger.warning(
                "compliance: pw-record exited early (rc=%s) for %s",
                proc.returncode,
                target_node,
            )
            return out_path.exists() and out_path.stat().st_size > 44
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    return out_path.exists() and out_path.stat().st_size > 44


def probe_device(
    target_node: str,
    *,
    capture_seconds: float = CAPTURE_SECONDS,
    min_high_band_pct: float = MIN_HIGH_BAND_PCT,
    min_rolloff_pct: float = MIN_ROLLOFF_3400HZ_PCT,
) -> ComplianceResult:
    """Capture from ``target_node`` and grade the spectral signature.

    Status grades:

    * ``pass`` — high_band_pct ≥ threshold AND rolloff_3400 ≥ threshold.
      Device is delivering wideband audio.
    * ``warn`` — rolloff_3400 passes but high_band_pct just under. May
      be a device with a slight HF rolloff but mostly intact wideband.
    * ``fail`` — both below threshold, **or** capture failed entirely.
      For SP325: NR is probably back on, EQ may be wrong, or firmware
      reverted. Tell the operator.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        ok = _capture_window(target_node, capture_seconds, wav_path)
        if not ok:
            return ComplianceResult(
                status="fail",
                high_band_pct=0.0,
                rolloff_3400hz_pct=0.0,
                rms=0.0,
                reason=(
                    f"pw-record could not capture from {target_node} — "
                    "device may be disconnected, busy, or misconfigured."
                ),
                bands_pct={},
            )

        rms, bands_pct, hbp, rolloff = _analyze_wav(wav_path)

        if hbp >= min_high_band_pct and rolloff >= min_rolloff_pct:
            status = "pass"
            reason = (
                f"wideband mode active: high_band_pct={hbp}% rolloff_3400={rolloff}% (≥ thresholds)"
            )
        elif rolloff >= min_rolloff_pct:
            status = "warn"
            reason = (
                f"partial wideband: high_band_pct={hbp}% below "
                f"{min_high_band_pct}% but rolloff_3400={rolloff}% ok"
            )
        else:
            status = "fail"
            reason = (
                f"narrowband/telephony cutoff detected: high_band_pct={hbp}% "
                f"rolloff_3400={rolloff}% (need ≥ {min_high_band_pct}% / "
                f"{min_rolloff_pct}%). Likely causes: SP325 firmware "
                "<1.3.6.0, AI Noise Cancellation toggled back on (Outgoing "
                "or Incoming), or EQ preset not 'Default'. Fix via Dell "
                "Peripheral Manager on Windows; settings persist in device "
                "NVRAM across replugs."
            )

        return ComplianceResult(
            status=status,
            high_band_pct=hbp,
            rolloff_3400hz_pct=rolloff,
            rms=round(rms, 1),
            reason=reason,
            bands_pct=bands_pct,
        )
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass


# Per-device expected behavior. Add new entries here when adding
# devices to the catalog; the speakerphone daemon picks the right
# minima per VID:PID.
EXPECTED_BY_DEVICE: dict[str, dict[str, float]] = {
    # Dell SP325 — measured 2026-05 against firmware 1.3.6.0 + NR off
    # + EQ Default. Noise floor was 3.29 %, speech-laden was 11.12 %.
    # Threshold of 1.5 % is a comfortable floor that still catches the
    # ~0.5 % bad-config baseline.
    "413c:8223": {
        "min_high_band_pct": 1.5,
        "min_rolloff_pct": 1.5,
    },
    # SP3022 not yet measured under the same harness. Same thresholds
    # apply as a conservative default — refine after first capture.
    "413c:8222": {
        "min_high_band_pct": 1.5,
        "min_rolloff_pct": 1.5,
    },
}


def expected_thresholds(device_key: str) -> tuple[float, float]:
    """Return (min_high_band_pct, min_rolloff_pct) for a known device."""
    entry = EXPECTED_BY_DEVICE.get(device_key, {})
    return (
        float(entry.get("min_high_band_pct", MIN_HIGH_BAND_PCT)),
        float(entry.get("min_rolloff_pct", MIN_ROLLOFF_3400HZ_PCT)),
    )
