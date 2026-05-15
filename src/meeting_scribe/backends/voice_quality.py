"""Heuristic quality score for a TTS voice-reference clip.

The goal is to pick a good 3–8 s chunk of speaker audio to use as a cloning
reference for Qwen3-TTS. The scoring is deliberately cheap (no model loads,
just numpy) so it can run on every live segment without slowing the TTS
hot path.

Score components (all 0..1, then weighted-averaged):

* ``duration``  — 3–8 s is the sweet spot per Qwen3-TTS guidance; clips
  outside get penalised smoothly rather than dropped outright so we still
  have *something* to clone with if the speaker only ever says short bursts.
* ``level``     — peak RMS should land in a healthy -24..-6 dBFS band.
  Too quiet ⇒ noisy-ish clone; too loud ⇒ probable clipping.
* ``clipping``  — fraction of samples pinned at ±1.0. Anything over ~0.2 %
  audibly distorts the clone.
* ``snr``       — estimated SNR in dB. Noise floor = 10th-percentile
  frame energy (assumed to be silence / breath), speech level = 90th.
  >25 dB is effectively studio, <10 dB is unusable.
* ``voiced``    — fraction of 20 ms frames above the noise floor. Under
  ~0.6 means the clip is mostly silence with a word or two, which
  confuses the voice encoder.

The final score is a weighted mean; a clip that fails hard on *any one*
dimension (e.g. heavy clipping) is floored so one bad component can't be
masked by four good ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ─── Tunables ────────────────────────────────────────────────────────────
_IDEAL_DURATION_RANGE = (3.0, 8.0)  # seconds
_MIN_DURATION = 1.0  # below this we don't even score — caller should skip
_DBFS_IDEAL = (-24.0, -6.0)  # peak-normalised RMS band we want to land in
_CLIP_BUDGET = 0.002  # 0.2 % clipped samples → score=0 on clipping axis
_SNR_FLOOR_DB = 5.0  # below this → 0
_SNR_CEIL_DB = 25.0  # at/above this → 1
_VOICED_FLOOR = 0.3  # below this → 0
_VOICED_CEIL = 0.8  # at/above this → 1
_FRAME_MS = 20  # VAD/SNR frame size
_WEIGHTS = {
    "duration": 0.15,
    "level": 0.15,
    "clipping": 0.15,
    "snr": 0.35,
    "voiced": 0.20,
}


@dataclass(frozen=True)
class QualityScore:
    total: float  # 0..1, bigger is better
    duration_s: float
    dbfs: float  # peak RMS in dBFS
    clip_pct: float
    snr_db: float
    voiced_ratio: float
    components: dict[str, float]  # per-axis 0..1 so we can explain decisions

    def is_usable(self) -> bool:
        return self.duration_s >= _MIN_DURATION and self.total > 0.0


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * np.log10(rms)


def _score_duration(seconds: float) -> float:
    lo, hi = _IDEAL_DURATION_RANGE
    if lo <= seconds <= hi:
        return 1.0
    if seconds < lo:
        # Linear from _MIN_DURATION..lo → 0..1
        if seconds <= _MIN_DURATION:
            return 0.0
        return _clamp01((seconds - _MIN_DURATION) / (lo - _MIN_DURATION))
    # seconds > hi — mild decay; still useful to ~15 s
    return _clamp01(1.0 - (seconds - hi) / (hi * 2.0))


def _score_level(dbfs: float) -> float:
    lo, hi = _DBFS_IDEAL
    if lo <= dbfs <= hi:
        return 1.0
    # Symmetric triangular falloff in dB
    if dbfs < lo:
        return _clamp01(1.0 - (lo - dbfs) / 18.0)
    return _clamp01(1.0 - (dbfs - hi) / 6.0)


def _score_clipping(pct: float) -> float:
    if pct <= 0.0:
        return 1.0
    if pct >= _CLIP_BUDGET:
        return 0.0
    return 1.0 - (pct / _CLIP_BUDGET)


def _frame_energy(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return per-frame RMS (float32)."""
    hop = max(1, int(sr * _FRAME_MS / 1000))
    if len(audio) < hop:
        return np.array([float(np.sqrt(np.mean(audio**2) + 1e-12))], dtype=np.float32)
    n_frames = len(audio) // hop
    trimmed = audio[: n_frames * hop].reshape(n_frames, hop).astype(np.float64)
    rms = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)
    return rms.astype(np.float32)


def _score_snr_and_voiced(frames: np.ndarray) -> tuple[float, float, float, float]:
    """Return (snr_score, snr_db, voiced_score, voiced_ratio)."""
    if frames.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    # 10th percentile ≈ quiet/silence floor, 90th ≈ speech peaks.
    noise = float(np.percentile(frames, 10))
    speech = float(np.percentile(frames, 90))
    if noise <= 1e-9:
        # Basically digital silence noise floor — treat as high SNR.
        snr_db = 60.0
    else:
        snr_db = 20.0 * np.log10(max(speech, 1e-9) / noise)
    snr_score = _clamp01((snr_db - _SNR_FLOOR_DB) / (_SNR_CEIL_DB - _SNR_FLOOR_DB))

    # Voiced ratio: frames above 3× noise floor
    threshold = max(noise * 3.0, 1e-4)
    voiced_ratio = float(np.mean(frames > threshold))
    voiced_score = _clamp01((voiced_ratio - _VOICED_FLOOR) / (_VOICED_CEIL - _VOICED_FLOOR))
    return snr_score, snr_db, voiced_score, voiced_ratio


def score_reference(audio: np.ndarray, sr: int = 16000) -> QualityScore:
    """Score a mono float32 waveform in [-1, 1] as a TTS voice reference.

    Returns a ``QualityScore`` whose ``total`` is safe to compare across
    candidates: higher means better reference, 0 means unusable.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)

    duration = len(audio) / float(sr)
    if duration < _MIN_DURATION:
        return QualityScore(
            total=0.0,
            duration_s=duration,
            dbfs=-120.0,
            clip_pct=0.0,
            snr_db=0.0,
            voiced_ratio=0.0,
            components={k: 0.0 for k in _WEIGHTS},
        )

    dbfs = _rms_dbfs(audio)
    clip_pct = float(np.mean(np.abs(audio) >= 0.999))
    frames = _frame_energy(audio, sr)
    snr_score, snr_db, voiced_score, voiced_ratio = _score_snr_and_voiced(frames)

    components = {
        "duration": _score_duration(duration),
        "level": _score_level(dbfs),
        "clipping": _score_clipping(clip_pct),
        "snr": snr_score,
        "voiced": voiced_score,
    }
    # Weighted mean, but gate by the hard axes: if clipping or SNR scores
    # zero we treat the whole clip as unusable rather than averaging it up.
    if components["clipping"] <= 0.0 or components["snr"] <= 0.0:
        total = 0.0
    else:
        total = sum(components[k] * w for k, w in _WEIGHTS.items())

    return QualityScore(
        total=float(total),
        duration_s=duration,
        dbfs=dbfs,
        clip_pct=clip_pct,
        snr_db=snr_db,
        voiced_ratio=voiced_ratio,
        components=components,
    )
