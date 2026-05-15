"""Generate a 3-second 16 kHz 220 Hz sine PCM blob for OOM-tune diarize probing."""

import sys

import numpy as np

if __name__ == "__main__":
    sr = 16_000
    seconds = 3.0
    freq = 220
    a = (
        0.3
        * np.sin(2 * np.pi * freq * np.linspace(0, seconds, int(sr * seconds), endpoint=False))
        * 32767
    ).astype(np.int16)
    sys.stdout.buffer.write(a.tobytes())
