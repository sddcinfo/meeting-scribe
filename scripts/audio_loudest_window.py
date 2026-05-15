#!/usr/bin/env python3
"""Find the loudest N-second window in a raw int16 PCM file.

Spectrum analysis on a mostly-silent capture mis-attributes HVAC hum
to "low-band dominance" — useless. This helper finds the section of
``recording.pcm`` with the highest RMS (real speech, not silence) so
the spectrum analyzer can be pointed at a useful slice.

Strategy: walk the file in non-overlapping ``window`` chunks, compute
RMS, return the offset (in seconds from start) of the highest one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcm_path", type=Path)
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Window size in seconds (default 10).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of loudest windows to report (default 5).",
    )
    args = parser.parse_args()

    win_bytes = int(args.window * args.rate * 2)
    size = args.pcm_path.stat().st_size
    n_windows = size // win_bytes
    if n_windows == 0:
        raise SystemExit("file too short for chosen window")

    # Streaming RMS — avoids holding whole file in memory.
    import struct

    results: list[tuple[float, float]] = []  # (offset_s, rms)
    with args.pcm_path.open("rb") as f:
        for i in range(n_windows):
            chunk = f.read(win_bytes)
            if len(chunk) < win_bytes:
                break
            samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
            sq = sum(s * s for s in samples) / len(samples)
            rms = math.sqrt(sq)
            offset_s = (i * win_bytes) / (args.rate * 2)
            results.append((round(offset_s, 1), round(rms, 1)))

    results.sort(key=lambda x: x[1], reverse=True)
    out = {
        "file": str(args.pcm_path),
        "total_seconds": round(size / (args.rate * 2), 1),
        "window_s": args.window,
        "loudest": [{"offset_s": o, "rms": r} for o, r in results[: args.top]],
        "quietest_rms": min(r for _, r in results),
        "median_rms": sorted(r for _, r in results)[len(results) // 2],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
