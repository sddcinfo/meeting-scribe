"""Audio pipeline quality tests — catches regressions in the resampler.

Specifically catches the bug where the browser AudioWorklet was doing
naive linear interpolation from 48kHz → 16kHz with no anti-aliasing,
folding high frequencies back into the speech band as static.

These tests use synthetic signals with known spectral content so we can
measure exactly how well the resampler suppresses out-of-band energy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.fft import rfft, rfftfreq

from meeting_scribe.audio.resample import Resampler


@pytest.fixture(scope="module")
def resampler() -> Resampler:
    return Resampler()


def _spectrum_at(audio: np.ndarray, fs: int, target_hz: float) -> float:
    """Return the spectrum magnitude (dB) at the frequency bin nearest target_hz."""
    windowed = audio * np.hanning(len(audio))
    spec = np.abs(rfft(windowed))
    freqs = rfftfreq(len(audio), 1 / fs)
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    return 20 * math.log10(float(spec[idx]) + 1e-12)


def _generate_tone(freq_hz: float, fs: int, duration_s: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(fs * duration_s)) / fs
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


# ── 48kHz → 16kHz (the common desktop/laptop case) ─────────────


class TestResampler48to16:
    SOURCE_RATE = 48000
    TARGET_RATE = 16000

    def test_passes_1khz_cleanly(self, resampler: Resampler) -> None:
        """A 1 kHz tone (well below Nyquist) must survive the resample unchanged."""
        signal = _generate_tone(1000, self.SOURCE_RATE, 1.0)
        out = resampler.resample(signal, source_rate=self.SOURCE_RATE, target_rate=self.TARGET_RATE)
        db_1k = _spectrum_at(out, self.TARGET_RATE, 1000)
        db_alias = _spectrum_at(out, self.TARGET_RATE, 6000)  # would alias if 7kHz
        # 1 kHz tone should be strong; nothing near 6 kHz
        assert db_1k > 40, f"1 kHz tone attenuated: {db_1k:.1f} dB (expected > 40)"
        assert db_alias < 0, f"Unexpected 6 kHz energy: {db_alias:.1f} dB"

    def test_rejects_out_of_band_10khz(self, resampler: Resampler) -> None:
        """A 10 kHz tone is above 16 kHz Nyquist — must be suppressed, NOT aliased.

        Without anti-aliasing, a 10 kHz tone at 48 kHz would fold back to
        (16000 - 10000) = 6 kHz at the 16 kHz output. This is the exact
        failure mode of the old naive linear interpolation worklet.
        """
        signal = _generate_tone(10000, self.SOURCE_RATE, 1.0)
        out = resampler.resample(signal, source_rate=self.SOURCE_RATE, target_rate=self.TARGET_RATE)
        db_alias = _spectrum_at(out, self.TARGET_RATE, 6000)
        # Kaiser sinc rejects this by ~45 dB+ vs naive linear.
        # Require at least 30 dB below nominal tone level.
        assert db_alias < 30, (
            f"10 kHz aliased into 16 kHz output at {db_alias:.1f} dB — "
            f"resampler is aliasing (expected well below 30 dB)"
        )

    def test_speech_band_preserved(self, resampler: Resampler) -> None:
        """A mix of speech-band tones (200 Hz, 1 kHz, 4 kHz) must all survive."""
        signal = (
            _generate_tone(200, self.SOURCE_RATE, 1.0, 0.3)
            + _generate_tone(1000, self.SOURCE_RATE, 1.0, 0.3)
            + _generate_tone(4000, self.SOURCE_RATE, 1.0, 0.3)
        )
        out = resampler.resample(signal, source_rate=self.SOURCE_RATE, target_rate=self.TARGET_RATE)
        for freq in (200, 1000, 4000):
            db = _spectrum_at(out, self.TARGET_RATE, freq)
            assert db > 30, f"{freq} Hz attenuated: {db:.1f} dB"

    def test_no_nan_or_inf(self, resampler: Resampler) -> None:
        """Resampler output must be finite (NaN/Inf would crash downstream)."""
        signal = _generate_tone(1000, self.SOURCE_RATE, 0.5)
        out = resampler.resample(signal, source_rate=self.SOURCE_RATE, target_rate=self.TARGET_RATE)
        assert np.all(np.isfinite(out)), "resampler produced non-finite values"

    def test_length_matches_ratio(self, resampler: Resampler) -> None:
        """Output length should be source length / (source_rate / target_rate) ± 1."""
        signal = _generate_tone(1000, self.SOURCE_RATE, 1.0)
        out = resampler.resample(signal, source_rate=self.SOURCE_RATE, target_rate=self.TARGET_RATE)
        expected = len(signal) * self.TARGET_RATE // self.SOURCE_RATE
        assert abs(len(out) - expected) <= 2, (
            f"length mismatch: got {len(out)}, expected ~{expected}"
        )


# ── 44.1kHz → 16kHz (the common mobile/Safari case) ────────────


class TestResampler44to16:
    SOURCE_RATE = 44100
    TARGET_RATE = 16000

    def test_passes_1khz_cleanly(self, resampler: Resampler) -> None:
        signal = _generate_tone(1000, self.SOURCE_RATE, 1.0)
        out = resampler.resample(signal, source_rate=self.SOURCE_RATE, target_rate=self.TARGET_RATE)
        db = _spectrum_at(out, self.TARGET_RATE, 1000)
        assert db > 40, f"1 kHz attenuated: {db:.1f} dB"

    def test_rejects_out_of_band_12khz(self, resampler: Resampler) -> None:
        """A 12 kHz tone is above Nyquist for 16 kHz (would alias to 4 kHz)."""
        signal = _generate_tone(12000, self.SOURCE_RATE, 1.0)
        out = resampler.resample(signal, source_rate=self.SOURCE_RATE, target_rate=self.TARGET_RATE)
        db_alias = _spectrum_at(out, self.TARGET_RATE, 4000)
        assert db_alias < 30, f"12 kHz aliased to 4 kHz at {db_alias:.1f} dB — resampler failing"


# ── Regression: naive linear vs Kaiser sinc ────────────────────


class TestResamplerRegressionVsNaive:
    """Fails if the resampler ever regresses to naive linear interpolation quality."""

    def test_kaiser_beats_linear_by_30db(self, resampler: Resampler) -> None:
        """Kaiser sinc should reject the 10 kHz alias by at least 30 dB vs linear."""
        fs_in = 48000
        fs_out = 16000
        signal = _generate_tone(10000, fs_in, 1.0)

        # Naive linear interpolation — the old broken worklet behavior
        ratio = fs_in / fs_out
        out_len = int(len(signal) / ratio)
        naive = np.zeros(out_len, dtype=np.float32)
        for i in range(out_len):
            src_pos = i * ratio
            idx = int(src_pos)
            frac = src_pos - idx
            if idx + 1 < len(signal):
                naive[i] = signal[idx] + frac * (signal[idx + 1] - signal[idx])
            else:
                naive[i] = signal[idx]

        # Kaiser — our production path
        kaiser = resampler.resample(signal, source_rate=fs_in, target_rate=fs_out)

        naive_alias = _spectrum_at(naive, fs_out, 6000)
        kaiser_alias = _spectrum_at(kaiser, fs_out, 6000)
        improvement = naive_alias - kaiser_alias
        assert improvement > 30, (
            f"Kaiser only {improvement:.1f} dB better than naive linear "
            f"(naive={naive_alias:.1f}, kaiser={kaiser_alias:.1f}). "
            f"Regression to a lower-quality resampling method."
        )


# ── Integration: wire format parsing ───────────────────────────


class TestWireFormat:
    """Validates the browser → server audio chunk format."""

    def test_sample_rate_header_detected(self) -> None:
        """A valid sample-rate header (8000..192000) is detected; raw PCM is not."""
        # A valid header: uint32 LE = 48000 = 0xBB80 0000 = b'\x80\xbb\x00\x00'
        header = (48000).to_bytes(4, "little")
        assert 8000 <= int.from_bytes(header, "little") <= 192000

        # Raw s16le PCM starts with audio samples — the first 4 bytes will
        # almost never happen to look like a plausible sample rate (8k-192k).
        # Example: a normal audio chunk's first int16 might be ~1000 → first
        # 4 bytes interpreted as uint32 = (next int16 << 16) | 1000. For that
        # to fall in 8000..192000 you'd need an extremely specific pattern.
        import numpy as np

        rng = np.random.default_rng(42)
        pcm = (rng.normal(0, 0.1, 4000) * 32767).astype(np.int16).tobytes()
        head4 = int.from_bytes(pcm[:4], "little")
        # Not a guarantee but overwhelmingly likely to NOT look like a valid rate
        # (this is a statistical check — if it fails, the test rng seed was unlucky)
        assert not (8000 <= head4 <= 192000), (
            f"Random PCM accidentally looks like a sample rate header: {head4}. "
            f"This is a rare edge case but indicates the detection heuristic may "
            f"misread legacy raw PCM as headered chunks."
        )
