#!/usr/bin/env python3
"""Generate the synthetic-probe WAV fixture for the meeting-start
preflight gate (W4 of the 2026-04-30 reliability plan).

The fixture is consumed by `meeting_scribe.runtime.synthetic_probe`
at server startup. It must:

1. Be NON-SILENT — the production VAD (`asr_vllm.VAD_ENERGY_THRESHOLD`)
   drops low-RMS buffers BEFORE building the OpenAI chat-completions
   audio request, so a silence probe exercises a path real traffic
   never hits. A 200 Hz sine at 0.3 amplitude clears the threshold.

2. Pass through the production request-builder unchanged — the
   probe's whole point is to confirm the live ASR pipeline isn't
   wedged. We don't assert non-empty transcribed text (a pure tone
   may legitimately produce empty text); we assert HTTP 200 + valid
   OpenAI response schema.

3. Be small (~50 KB) so the fixture can be loaded once at startup
   and held in memory.

Regenerate via:
    python scripts/gen_probe_audio_fixture.py
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "tests" / "fixtures" / "probe_audio_1500ms.wav"

SAMPLE_RATE = 16_000
DURATION_S = 1.5
FREQ_HZ = 200.0
AMPLITUDE = 0.3


def main() -> None:
    n_samples = int(DURATION_S * SAMPLE_RATE)
    samples = [
        int(AMPLITUDE * 32767 * math.sin(2 * math.pi * FREQ_HZ * i / SAMPLE_RATE))
        for i in range(n_samples)
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(SAMPLE_RATE)
        w.writeframes(struct.pack(f"<{n_samples}h", *samples))
    size = OUT.stat().st_size
    print(f"wrote {OUT} ({size} bytes, {n_samples} samples @ {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
