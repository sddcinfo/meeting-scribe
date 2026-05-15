#!/usr/bin/env python3
"""Spectral analyzer for WAV captures — diagnose telephony-band cutoffs.

Why this matters: Qwen3-ASR distinguishes Japanese phonemes from
their English/romaji counterparts using high-frequency consonant
energy (し/ち/つ sibilants live in 4–8 kHz). A telephony-bandlimited
stream (drops sharply >3.4 kHz) strips that information and the ASR
falls back to its English prior, producing romaji transcripts of
Japanese audio — exactly the failure mode seen on the Dell SP325
that the Plantronics Poly Sync 20-M never exhibited.

Reports per file:
  * energy share in 6 frequency bands (0–500, 500–1k, 1–2k, 2–4k,
    4–6k, 6k+)
  * "high_band_pct" — sum of 4–6 kHz and 6 kHz+ shares. <5 % == the
    device or capture pipeline has bandlimited the signal.
  * "rolloff_3400hz" — how much energy persists above 3.4 kHz
    relative to total. Telephony codecs (G.711/G.722) cut here.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

BANDS_HZ: list[tuple[int, int]] = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 24000),
]


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        if sw != 2 or n_ch != 1:
            raise SystemExit(f"{path}: expected mono int16, got ch={n_ch} sw={sw}")
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    return rate, samples


def analyze(path: Path) -> dict[str, object]:
    rate, samples = _read_wav(path)
    n = len(samples)
    if n == 0:
        return {"path": str(path), "error": "empty"}

    duration_s = n / rate
    # Hann window for less spectral leakage.
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(samples * win))
    power = spec * spec  # |X(f)|² ∝ energy density
    freqs = np.fft.rfftfreq(n, d=1.0 / rate)

    total = float(power.sum()) or 1.0
    band_pct = {}
    for lo, hi in BANDS_HZ:
        hi_eff = min(hi, rate / 2)
        if hi_eff <= lo:
            band_pct[f"{lo}-{hi}Hz"] = 0.0
            continue
        mask = (freqs >= lo) & (freqs < hi_eff)
        band_pct[f"{lo}-{hi}Hz"] = round(100.0 * float(power[mask].sum()) / total, 2)

    high_band_pct = band_pct.get("4000-6000Hz", 0.0) + band_pct.get("6000-24000Hz", 0.0)
    above_3400 = float(power[freqs >= 3400].sum())
    rolloff_3400hz_pct = round(100.0 * above_3400 / total, 2)

    rms_raw = float(np.sqrt(np.mean(samples * samples)))
    return {
        "path": str(path),
        "rate": rate,
        "duration_s": round(duration_s, 3),
        "rms": round(rms_raw, 1),
        "bands_pct": band_pct,
        "high_band_pct": round(high_band_pct, 2),
        "rolloff_3400hz_pct": rolloff_3400hz_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wavs", nargs="+", type=Path, help="WAV files to analyze")
    args = parser.parse_args()
    results = [analyze(p) for p in args.wavs]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
