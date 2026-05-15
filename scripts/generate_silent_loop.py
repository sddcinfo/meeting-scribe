#!/usr/bin/env python3
"""Generate the silent-MP3 loop used by the guest audio session keeper.

Writes ``static/silent-loop.mp3`` — a 0.5 second silent MP3 at 48 kHz
that the guest page plays in a hidden ``<audio loop>`` element to
promote Safari's audio session category from "ambient" (silenced by
the ringer switch) to "playback" (routes to the speaker at full
volume, ignoring the ringer). The same trick is used by Google Meet
and Zoom web clients.

Run once. The resulting MP3 is committed to the repo as a static
asset served at ``/static/silent-loop.mp3``. No need to regenerate
unless you're changing the keeper's duration or sample rate.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import av
import numpy as np

SAMPLE_RATE = 48000
DURATION_S = 0.5
CHANNELS = 1
BITRATE = 32000  # very low; we're only encoding silence

OUT_PATH = Path(__file__).resolve().parent.parent / "static" / "silent-loop.mp3"


def main() -> int:
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp3")
    stream = container.add_stream("mp3", rate=SAMPLE_RATE)
    stream.bit_rate = BITRATE
    stream.layout = "mono"

    # One AudioFrame of silence. MP3 encoders like frames of 1152 samples
    # per layer III packet; feeding ~24000 samples (0.5 s) produces a
    # clean MP3 stream.
    num_samples = int(SAMPLE_RATE * DURATION_S)
    silence = np.zeros(num_samples, dtype=np.float32)
    frame = av.AudioFrame(format="fltp", layout="mono", samples=num_samples)
    frame.sample_rate = SAMPLE_RATE
    frame.pts = 0
    frame.planes[0].update(silence.tobytes())

    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    data = buf.getvalue()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(data)
    print(f"wrote {OUT_PATH} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
