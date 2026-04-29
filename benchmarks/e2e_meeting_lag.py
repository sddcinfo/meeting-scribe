"""End-to-end meeting lag harness (B7 phase implementation).

Stands up a transient meeting against the live local server, streams a
fixture WAV through the audio-in WebSocket, and measures the time from
each utterance-end to its translation completion as observed via the
view broadcast WS. Reports p50/p95/p99 in ms.

Designed to be invoked from ``meeting-scribe validate --e2e``:

    python benchmarks/e2e_meeting_lag.py --threshold-ms 8000 --json

Exit code: 0 if p95 ≤ ``--threshold-ms``, 1 otherwise. ``--json``
emits one JSON object on stdout: ``{p50_lag_ms, p95_lag_ms, p99_lag_ms,
sample_count, threshold_ms, passed}``.

Caveats:
  * Requires the admin server to be reachable at ``--scribe-url``
    (default https://127.0.0.1:8080).
  * Starts AND stops a meeting — perturbs the running stack. Don't run
    during a real recording.
  * The fixture WAV is the same one used by ``meeting-scribe validate``
    (``tests/fixtures/validate/audio_en_short.wav``); short enough that
    a full run completes in < 90 s on the GB10.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import ssl
import sys
import time
import wave
from pathlib import Path

import httpx

try:
    from websockets.asyncio.client import connect as _ws_connect
except ImportError:  # pragma: no cover — fallback for older websockets
    from websockets.client import connect as _ws_connect


_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "validate"
    / "audio_en_short.wav"
)


def _read_pcm16_mono(wav_path: Path) -> tuple[bytes, int]:
    """Return (raw int16 PCM bytes, sample_rate)."""
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise SystemExit(f"fixture {wav_path} must be mono")
        if wf.getsampwidth() != 2:
            raise SystemExit(f"fixture {wav_path} must be 16-bit PCM")
        return wf.readframes(wf.getnframes()), wf.getframerate()


async def _stream_audio(scribe_ws_url: str, pcm: bytes, sample_rate: int) -> None:
    """Stream PCM as ~20 ms chunks over the audio-in WS, prefixed with
    the sample-rate header the server expects."""
    chunk_samples = sample_rate // 50  # 20 ms
    chunk_bytes = chunk_samples * 2

    ssl_ctx: ssl.SSLContext | None = None
    if scribe_ws_url.startswith("wss://"):
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    async with _ws_connect(scribe_ws_url, ssl=ssl_ctx, max_size=2**24) as ws:
        sr_header = sample_rate.to_bytes(4, "little")
        for i in range(0, len(pcm), chunk_bytes):
            chunk = pcm[i : i + chunk_bytes]
            if len(chunk) < 2:
                break
            await ws.send(sr_header + chunk)
            await asyncio.sleep(0.02)
        # Allow the ASR backend to flush any tail buffer before we close
        await asyncio.sleep(2.0)


async def _watch_view_ws(
    scribe_ws_url: str,
    sample_window_s: float,
    samples_out: list[float],
) -> None:
    """Listen on the view broadcast WS; for every translation event we
    receive, compute (now - utterance_end_at) in ms and append to
    ``samples_out``."""
    ssl_ctx: ssl.SSLContext | None = None
    if scribe_ws_url.startswith("wss://"):
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    async with _ws_connect(scribe_ws_url, ssl=ssl_ctx, max_size=2**24) as ws:
        deadline = time.monotonic() + sample_window_s
        while time.monotonic() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except TimeoutError:
                continue
            try:
                ev = json.loads(msg) if isinstance(msg, str) else None
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            translation = ev.get("translation") or {}
            if not translation.get("text"):
                continue
            utt_end = ev.get("utterance_end_at")
            completed = translation.get("completed_at") or time.monotonic()
            if utt_end is None:
                continue
            try:
                lag_ms = max(0.0, (float(completed) - float(utt_end)) * 1000.0)
            except (TypeError, ValueError):
                continue
            samples_out.append(lag_ms)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


async def run(scribe_url: str, threshold_ms: float, json_only: bool) -> dict:
    """Drive the full e2e probe and return a result dict.

    Caller (CLI / `meeting-scribe validate`) is responsible for setting
    exit code based on ``passed``.
    """
    if not _FIXTURE.exists():
        raise SystemExit(f"fixture missing: {_FIXTURE}")

    pcm, sample_rate = _read_pcm16_mono(_FIXTURE)
    audio_seconds = len(pcm) / 2 / sample_rate
    sample_window_s = max(audio_seconds + 30.0, 60.0)

    # Build WS URLs from the admin URL.
    if scribe_url.startswith("https://"):
        ws_scheme = "wss://"
        host = scribe_url[len("https://") :]
    elif scribe_url.startswith("http://"):
        ws_scheme = "ws://"
        host = scribe_url[len("http://") :]
    else:
        raise SystemExit(f"scribe_url must be http(s)://...; got {scribe_url}")
    audio_ws = f"{ws_scheme}{host}/api/ws/audio"
    view_ws = f"{ws_scheme}{host}/api/ws/view"

    verify = False  # self-signed admin TLS; safe local-only

    # Start a meeting
    async with httpx.AsyncClient(verify=verify, timeout=15.0) as c:
        r = await c.post(f"{scribe_url}/api/meeting/start", json={"language_pair": ["en", "ja"]})
        if r.status_code not in (200, 201):
            raise SystemExit(f"meeting start failed: HTTP {r.status_code} body={r.text[:200]}")
        meeting_id = r.json().get("meeting_id")

    samples: list[float] = []
    try:
        watcher = asyncio.create_task(_watch_view_ws(view_ws, sample_window_s, samples))
        # Small grace period so the view WS is connected before audio starts
        await asyncio.sleep(0.5)
        await _stream_audio(audio_ws, pcm, sample_rate)
        # Let the watcher run a bit more to catch trailing translations
        await asyncio.sleep(min(15.0, sample_window_s - audio_seconds))
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
    finally:
        # Stop the meeting (best effort)
        async with httpx.AsyncClient(verify=verify, timeout=60.0) as c:
            try:
                await c.post(f"{scribe_url}/api/meeting/stop")
            except Exception:
                pass

    p50 = _percentile(samples, 0.5)
    p95 = _percentile(samples, 0.95)
    p99 = _percentile(samples, 0.99)
    passed = bool(samples) and p95 <= threshold_ms

    result = {
        "meeting_id": meeting_id,
        "sample_count": len(samples),
        "p50_lag_ms": p50,
        "p95_lag_ms": p95,
        "p99_lag_ms": p99,
        "threshold_ms": threshold_ms,
        "passed": passed,
        "audio_seconds": audio_seconds,
    }

    if json_only:
        sys.stdout.write(json.dumps(result) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scribe-url",
        default=os.environ.get("SCRIBE_URL", "https://127.0.0.1:8080"),
        help="admin server URL (https; default: https://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=8000.0,
        help="p95 lag threshold; exit 1 if exceeded (default: 8000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a single JSON line to stdout (machine-readable)",
    )
    args = parser.parse_args()
    result = asyncio.run(run(args.scribe_url, args.threshold_ms, args.json))
    if not args.json:
        print(
            f"e2e_lag p50={result['p50_lag_ms']:.0f}ms "
            f"p95={result['p95_lag_ms']:.0f}ms "
            f"p99={result['p99_lag_ms']:.0f}ms "
            f"samples={result['sample_count']} "
            f"threshold={result['threshold_ms']:.0f}ms "
            f"=> {'PASS' if result['passed'] else 'FAIL'}"
        )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
