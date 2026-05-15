"""Spectral + level audit of a meeting's recording.pcm.

The on-disk recording.pcm is mono PCM16 little-endian at the meeting's
configured ASR rate (typically 16 kHz). At 16 kHz the Nyquist limit is
8 kHz, so any wideband loss above 8 kHz is invisible here — but the
3.4 kHz telephony rolloff that the SP325 narrowband mode produces is
fully observable. We surface RMS, peak, dBFS RMS, clipping count, plus
a 0-500 / 500-1k / 1-2k / 2-4k / 4-7k / 7-8k Hz energy distribution.

The compliance.py thresholds (high_band_pct >= 1.5%, rolloff_3400 >= 1.5%)
use the 24 kHz live capture and are not 1:1 comparable to the 16-kHz
post-capture file, but the same telephony-cutoff signature shows here as
a sharp dropoff between the 2-4 kHz and 4-7 kHz bands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def audit(pcm_path: Path, sample_rate: int = 16_000) -> dict:
    raw = np.fromfile(pcm_path, dtype=np.int16)
    if raw.size == 0:
        return {"error": "empty"}
    floats = raw.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(floats**2)))
    peak = float(np.max(np.abs(floats)))
    clipped = int(np.sum(np.abs(raw) >= 32700))
    dbfs_rms = 20 * np.log10(rms + 1e-12)

    n = floats.size
    win = np.hanning(min(n, sample_rate * 30))  # up to 30s window
    if win.size < n:
        floats = floats[: win.size]
    spec = np.fft.rfft(floats * (np.hanning(floats.size)))
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(floats.size, 1.0 / sample_rate)

    def band_pct(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(power[mask].sum() / power.sum() * 100.0) if power.sum() > 0 else 0.0

    bands = {
        "0-500Hz": band_pct(0, 500),
        "500-1000Hz": band_pct(500, 1000),
        "1000-2000Hz": band_pct(1000, 2000),
        "2000-3400Hz": band_pct(2000, 3400),
        "3400-4000Hz": band_pct(3400, 4000),
        "4000-7000Hz": band_pct(4000, 7000),
        "7000-8000Hz": band_pct(7000, 8000),
    }
    # Telephony cutoff signature: > 95% of energy below 3.4 kHz, < 2%
    # above 4 kHz. The SP325 wideband-good signature should have
    # at least ~5% above 4 kHz on speech material.
    below_3400 = (
        bands["0-500Hz"] + bands["500-1000Hz"] + bands["1000-2000Hz"] + bands["2000-3400Hz"]
    )
    above_4000 = bands["4000-7000Hz"] + bands["7000-8000Hz"]
    telephony_signature = below_3400 > 95 and above_4000 < 2

    duration_s = raw.size / sample_rate
    return {
        "file": str(pcm_path),
        "sample_rate_hz": sample_rate,
        "duration_s": round(duration_s, 1),
        "samples": int(raw.size),
        "rms": round(rms, 4),
        "peak": round(peak, 3),
        "dbfs_rms": round(float(dbfs_rms), 1),
        "clipped_samples": clipped,
        "bands_pct": {k: round(v, 2) for k, v in bands.items()},
        "energy_below_3400Hz_pct": round(below_3400, 2),
        "energy_above_4000Hz_pct": round(above_4000, 2),
        "telephony_signature": telephony_signature,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _audio_quality_audit.py <recording.pcm> [<recording.pcm> ...]")
        return 2
    out = [audit(Path(p)) for p in sys.argv[1:]]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
