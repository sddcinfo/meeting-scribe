#!/usr/bin/env python3
"""Spectrum analysis on a raw int16 PCM tail (no WAV header).

For poking at ``recording.pcm`` files written by ``audio_writer`` — they
have no RIFF header, just int16 mono samples at the recorded rate. Pass
the file path, the rate, and (optionally) how many seconds from the tail
to analyze.

Calls into ``audio_spectrum.analyze`` after wrapping the tail in a
synthetic WAV so the existing analyzer stays the source of truth.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcm_path", type=Path)
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="How many seconds from the tail to analyze.",
    )
    parser.add_argument(
        "--offset-from-end",
        type=float,
        default=0.0,
        help="Skip this many seconds *before* the tail (cut off trailing silence).",
    )
    args = parser.parse_args()

    size = args.pcm_path.stat().st_size
    bytes_per_s = args.rate * 2  # int16 mono
    tail_bytes = int(args.seconds * bytes_per_s)
    skip_bytes = int(args.offset_from_end * bytes_per_s)
    offset = max(0, size - tail_bytes - skip_bytes)
    with args.pcm_path.open("rb") as f:
        f.seek(offset)
        data = f.read(tail_bytes)

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as tmp:
        wav_path = Path(tmp.name)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(args.rate)
        w.writeframes(data)

    # Defer to the existing analyzer. ``scripts/`` isn't a Python
    # package, so we import by file path.
    import importlib.util
    import sys as _sys

    spec_path = Path(__file__).parent / "audio_spectrum.py"
    spec = importlib.util.spec_from_file_location("audio_spectrum", spec_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {spec_path}")
    audio_spectrum = importlib.util.module_from_spec(spec)
    _sys.modules["audio_spectrum"] = audio_spectrum
    spec.loader.exec_module(audio_spectrum)

    print(json.dumps(audio_spectrum.analyze(wav_path), indent=2))


if __name__ == "__main__":
    main()
