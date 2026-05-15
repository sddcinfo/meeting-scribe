#!/usr/bin/env python3
"""Validate a USB-speakerphone capture path for ASR quality regressions.

History: when the Dell SP325 replaced the Plantronics Poly Sync 20-M,
ASR audio quality dropped. Hypothesis: the SP325 advertises both 16 kHz
and 48 kHz capture, and its on-device DSP (AEC/NR/AGC) is tuned for
16 kHz; running it at 48 kHz forces firmware-side resampling that hurts
quality. The Poly only advertises 48 kHz, so it's unaffected.

This script captures the same audio from a PipeWire source at multiple
rates and prints a side-by-side quality report — duration, RMS, peak,
silence ratio, dropout count — so we can prove or disprove the
regression empirically. It does not need root; it runs alongside any
existing pw-record consumer (PipeWire supports multiple readers).

Usage:

    scripts/audio_capture_validate.py \\
        --node alsa_input.usb-Dell_Inc._Dell_SP325_..._pro-input-0 \\
        --rates 16000,48000 \\
        --duration 5

Output is a JSON object per rate suitable for archival in
``reports/speakerphone-quality/<host>-<timestamp>.json``.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import contextlib
import json
import math
import shutil
import struct
import sys
import time
import wave
from pathlib import Path


def _rms_int16(frames: bytes) -> int:
    """Compute the RMS of a buffer of little-endian int16 samples.

    Replaces ``audioop.rms`` which was removed in Python 3.13. We keep
    this script vendored so the validation tool works on the
    GB10's Python 3.14 venv without pulling audioop-lts as a dep.
    """
    if not frames:
        return 0
    a = array.array("h")
    a.frombytes(frames)
    if not a:
        return 0
    sq = 0
    for s in a:
        sq += s * s
    return int(math.sqrt(sq / len(a)))


async def _capture_one(
    node: str,
    rate: int,
    duration_s: float,
    out_path: Path,
) -> dict[str, object]:
    """Spawn pw-record for ``duration_s`` seconds and analyze the result."""
    pw_record = shutil.which("pw-record")
    if pw_record is None:
        raise RuntimeError("pw-record not on PATH")

    argv = [
        pw_record,
        f"--target={node}",
        "--format=s16",
        f"--rate={rate}",
        "--channels=1",
        str(out_path),
    ]
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(duration_s + 1.0):
            await asyncio.sleep(duration_s)
    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            async with asyncio.timeout(2.0):
                await proc.wait()
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    elapsed = time.monotonic() - t0

    if not out_path.exists() or out_path.stat().st_size <= 44:
        return {
            "rate_requested": rate,
            "out_path": str(out_path),
            "elapsed_wall_s": round(elapsed, 3),
            "error": "no audio captured",
        }

    return {
        "rate_requested": rate,
        "out_path": str(out_path),
        "elapsed_wall_s": round(elapsed, 3),
        **_analyze_wav(out_path),
    }


def _analyze_wav(path: Path) -> dict[str, object]:
    """Compute RMS / peak / silence ratio / dropout count.

    Silence ratio: fraction of 50 ms windows whose RMS is below
    -55 dBFS (a generous floor that still catches consumer-mic noise
    floors). Dropout count: number of consecutive 20 ms windows that
    contain only zero samples — pro-audio profile mishaps would show
    up as bursts of zeros mid-stream.
    """
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if channels != 1 or sampwidth != 2:
        return {"error": f"unexpected wav fmt ch={channels} bytes={sampwidth}"}

    total_frames = len(frames) // 2
    duration_s = total_frames / rate

    rms_overall = _rms_int16(frames)
    peak = max(abs(s) for s in struct.unpack(f"<{total_frames}h", frames)) if total_frames else 0

    # 50 ms windows for silence ratio.
    win_frames = int(rate * 0.05)
    silent_windows = 0
    total_windows = 0
    if win_frames > 0:
        for offset in range(0, total_frames - win_frames + 1, win_frames):
            chunk = frames[offset * 2 : (offset + win_frames) * 2]
            chunk_rms = _rms_int16(chunk)
            total_windows += 1
            if chunk_rms < 60:  # ~ -55 dBFS
                silent_windows += 1

    # 20 ms zero-only windows for dropout count.
    drop_win = int(rate * 0.020)
    dropouts = 0
    if drop_win > 0:
        for offset in range(0, total_frames - drop_win + 1, drop_win):
            chunk = frames[offset * 2 : (offset + drop_win) * 2]
            if not any(b != 0 for b in chunk):
                dropouts += 1

    def db(v: float) -> float:
        return 20 * math.log10(v / 32768.0) if v else -float("inf")

    return {
        "rate_actual": rate,
        "duration_s": round(duration_s, 3),
        "frames": total_frames,
        "rms": rms_overall,
        "rms_dbfs": round(db(rms_overall), 2),
        "peak": peak,
        "peak_dbfs": round(db(peak), 2),
        "silence_ratio": round(silent_windows / total_windows, 3) if total_windows else 0.0,
        "silent_windows": silent_windows,
        "total_windows": total_windows,
        "dropouts_20ms": dropouts,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, help="PipeWire source node name")
    parser.add_argument(
        "--rates",
        default="16000,48000",
        help="Comma-separated capture rates to test (default 16000,48000)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds per capture (default 5)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory to write the WAV samples + JSON report",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="JSON report filename (default: speakerphone-quality-<ts>.json)",
    )
    args = parser.parse_args()

    rates = [int(r) for r in args.rates.split(",") if r.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")

    results: list[dict[str, object]] = []
    for rate in rates:
        out = args.out_dir / f"sp-capture-{rate}-{ts}.wav"
        print(f"capturing {rate} Hz → {out}", file=sys.stderr)
        result = await _capture_one(args.node, rate, args.duration, out)
        results.append(result)

    report = {
        "node": args.node,
        "duration_s": args.duration,
        "captured_at": ts,
        "results": results,
    }
    report_path = args.out_dir / (args.report_name or f"speakerphone-quality-{ts}.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nreport written to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
