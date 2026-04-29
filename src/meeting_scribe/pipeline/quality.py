"""Audio-quality probe shared by the reprocess + finalize paths.

Detects the zero-gap writer bug: meetings where the audio writer
silently filled stretches of the recording with zeros instead of real
samples. These meetings still play back, but diarization is broken
because pyannote sees the zero region as a distinct (silent) "speaker"
and over-clusters the rest of the audio around it.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def _audio_quality_report(pcm_data: bytes) -> dict:
    """Detect if a recording has been corrupted by the zero-gap writer bug.

    Returns ``{zero_fill_pct, longest_zero_run_ms, usable}``. Zero-fill
    >40% means diarization is fundamentally broken for this meeting.
    """
    if len(pcm_data) < 4:
        return {"zero_fill_pct": 0, "longest_zero_run_ms": 0, "usable": True}
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    zero_mask = samples == 0
    total_zero = int(zero_mask.sum())
    pct = total_zero / len(samples) * 100

    # Find longest contiguous zero run
    longest = 0
    current = 0
    for z in zero_mask:
        if z:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    longest_ms = int(longest / SAMPLE_RATE * 1000)

    return {
        "zero_fill_pct": round(pct, 1),
        "longest_zero_run_ms": longest_ms,
        "usable": pct < 40,
    }
