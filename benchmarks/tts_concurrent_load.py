"""Phase 1b load-capacity gate — concurrent /v1/audio/speech streams.

Compares the single vllm-omni instance against the 2-replica baseline.
Both profiles (studio + cloned) must meet pass criteria before Phase 6
deletion.

Usage:
    python benchmarks/tts_concurrent_load.py --url http://localhost:8002 \
        --profile studio --concurrency 8 --out results/single_studio.json
    python benchmarks/tts_concurrent_load.py --url http://localhost:8002 \
        --profile cloned --concurrency 8 --out results/single_cloned.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
import wave
from pathlib import Path

import httpx
import numpy as np
from benchmarks._bench_paths import assert_offline_path
from benchmarks._consent_check import enforce_consent
from benchmarks._tts_backends import Backend, add_backend_arg, build_body

_STUDIO_VOICES = ["aiden", "vivian", "ono_anna", "sohee", "uncle_fu"]
_SHORT = "Hello, this is a short utterance."
_LONG = (
    "This is the longer benchmark utterance used to measure continuous-batching "
    "throughput under concurrent load. It spans roughly forty words so that "
    "streamed synthesis stays resident long enough for multiple requests to "
    "overlap on the single vllm-omni instance we are validating today."
)


def _make_ref_audio_uri(seconds: float = 6.0, sample_rate: int = 24_000) -> str:
    """Boundary-size ref_audio at _REF_AUDIO_MAX_SECONDS to stress the cloned path."""
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


async def _one_stream(client: httpx.AsyncClient, url: str, body: dict, request_id: int) -> dict:
    t0 = time.monotonic()
    ttfa = None
    bytes_total = 0
    err = None
    try:
        async with client.stream("POST", f"{url}/v1/audio/speech", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk and ttfa is None:
                    ttfa = (time.monotonic() - t0) * 1000.0
                bytes_total += len(chunk)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "request_id": request_id,
        "ttfa_ms": ttfa,
        "total_ms": (time.monotonic() - t0) * 1000.0,
        "bytes": bytes_total,
        "error": err,
    }


async def run(
    url: str, profile: str, concurrency: int, total: int, model: str, backend: Backend
) -> dict:
    ref_uri = _make_ref_audio_uri() if profile == "cloned" else None
    bodies = []
    for i in range(total):
        voice = _STUDIO_VOICES[i % len(_STUDIO_VOICES)]
        text = _SHORT if (i % 2 == 0) else _LONG
        bodies.append(
            build_body(
                backend,
                text=text,
                voice=voice,
                model=model,
                ref_audio_uri=ref_uri if profile == "cloned" else None,
            )
        )

    sem = asyncio.Semaphore(concurrency)

    async def bound(i: int, body: dict, client: httpx.AsyncClient) -> dict:
        async with sem:
            return await _one_stream(client, url, body, i)

    async with httpx.AsyncClient(timeout=60) as client:
        results = await asyncio.gather(*(bound(i, b, client) for i, b in enumerate(bodies)))

    ttfas = [r["ttfa_ms"] for r in results if r["ttfa_ms"] is not None]
    totals = [r["total_ms"] for r in results if r["error"] is None]
    errors = [r for r in results if r["error"]]
    return {
        "url": url,
        "backend": backend,
        "profile": profile,
        "concurrency": concurrency,
        "total_requests": total,
        "success": len(totals),
        "errors": len(errors),
        "p50_ttfa_ms": statistics.median(ttfas) if ttfas else None,
        "p95_ttfa_ms": statistics.quantiles(ttfas, n=20)[18] if len(ttfas) >= 20 else None,
        "p50_total_ms": statistics.median(totals) if totals else None,
        "p95_total_ms": statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else None,
        "per_request": results,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="TTS base URL, e.g. http://localhost:8002")
    p.add_argument("--profile", choices=["studio", "cloned"], required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--total", type=int, default=40)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=None, help="model field sent in /v1/audio/speech body")
    add_backend_arg(p)
    p.add_argument(
        "--allow-in-repo-out",
        action="store_true",
        help="Bypass the offline-path validator (test-only).",
    )
    args = p.parse_args()
    if not args.allow_in_repo_out:
        assert_offline_path(args.out)
    if args.profile == "cloned":
        enforce_consent()

    result = asyncio.run(
        run(args.url, args.profile, args.concurrency, args.total, args.model, args.backend)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_request"}, indent=2))


if __name__ == "__main__":
    main()
