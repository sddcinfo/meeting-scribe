"""Synthetic performance baseline for the live meeting-scribe stack.

Probes each backend with synthetic audio / text, captures p50/p95/p99
latency under: (a) sequential single-request load and (b) concurrent
load that mimics a live meeting's real concurrency. Also grabs
scribe-main's rolling loop-lag + system stats from /api/status.

Usage:
    python benchmarks/perf_baseline.py --out benchmarks/results/baseline_<date>.json

Re-run after any change (e.g. furigana generation) to see the delta.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import ssl
import statistics
import time
import wave
from pathlib import Path

import httpx
import numpy as np

# ─── Synthetic inputs ─────────────────────────────────────────────────────


def _synthetic_wav_bytes(seconds: float = 1.5, sr: int = 16000, freq: int = 220) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    a = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(a.tobytes())
    return buf.getvalue()


_ASR_B64 = base64.b64encode(_synthetic_wav_bytes()).decode()
_DIARIZE_PCM = (
    (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 3.0, 48000, endpoint=False)) * 32767)
    .astype(np.int16)
    .tobytes()
)

_JA_PHRASES = [
    "今日はいい天気ですね。",
    "会議の開始時間を教えてください。",
    "次のスライドをお願いします。",
    "ちょっと休憩しましょう。",
    "来週の予定はどうですか。",
    "この案件の進捗を確認したい。",
    "申し訳ありませんが、少し遅れます。",
    "資料を共有していただけますか。",
]
_EN_PHRASES = [
    "What's on the agenda today?",
    "Let me pull up the latest numbers.",
    "I think we should revisit this next week.",
    "Can everyone hear me okay?",
    "Great, thanks for catching that.",
    "I'll follow up with them this afternoon.",
    "Could you share your screen?",
    "Let's take a short break.",
]


# ─── Probes ───────────────────────────────────────────────────────────────


async def probe_asr(client: httpx.AsyncClient) -> float:
    """POST /v1/chat/completions with a synthetic audio clip. Returns ms."""
    body = {
        "model": "Qwen/Qwen3-ASR-1.7B",
        "messages": [
            {"role": "system", "content": "Transcribe."},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": _ASR_B64, "format": "wav"}}
                ],
            },
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "priority": -20,
    }
    t0 = time.monotonic()
    r = await client.post("http://localhost:8003/v1/chat/completions", json=body, timeout=30)
    r.raise_for_status()
    return (time.monotonic() - t0) * 1000.0


_TRANSLATE_MODEL_CACHE: str | None = None


async def _translate_model_id(client: httpx.AsyncClient) -> str:
    """Auto-discover the served translation model via /v1/models.

    Hard-coding the model name used to couple this bench to a
    specific quantisation (Intel Qwen3.5-INT4 AutoRound). After the
    2026-04-19 switch to Qwen3.6-FP8 the hardcoded ID started
    returning 404. Discover at runtime so swapping backends doesn't
    require editing every benchmark.
    """
    global _TRANSLATE_MODEL_CACHE
    if _TRANSLATE_MODEL_CACHE is not None:
        return _TRANSLATE_MODEL_CACHE
    r = await client.get("http://localhost:8010/v1/models", timeout=10)
    r.raise_for_status()
    data = r.json().get("data") or []
    if not data:
        raise RuntimeError(
            "vLLM /v1/models returned an empty list — is the translation backend loaded?"
        )
    _TRANSLATE_MODEL_CACHE = data[0]["id"]
    return _TRANSLATE_MODEL_CACHE


async def probe_translate(client: httpx.AsyncClient, text: str, src: str, tgt: str) -> float:
    body = {
        "model": await _translate_model_id(client),
        "messages": [
            {"role": "system", "content": f"Translate from {src} to {tgt}."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "priority": -10,
    }
    t0 = time.monotonic()
    r = await client.post("http://localhost:8010/v1/chat/completions", json=body, timeout=30)
    r.raise_for_status()
    return (time.monotonic() - t0) * 1000.0


async def probe_diarize(client: httpx.AsyncClient) -> float:
    t0 = time.monotonic()
    r = await client.post(
        "http://localhost:8001/v1/diarize",
        content=_DIARIZE_PCM,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": "16000",
            "X-Max-Speakers": "4",
            "X-Min-Speakers": "0",
        },
        timeout=30,
    )
    r.raise_for_status()
    return (time.monotonic() - t0) * 1000.0


_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


async def probe_scribe_status(_client: httpx.AsyncClient) -> dict:
    # Use a per-call client because the verify= is set at client construction
    # in httpx, not per-request. Cheap for a once-per-run probe.
    async with httpx.AsyncClient(verify=_SSL_CTX, timeout=5) as c:
        r = await c.get("https://localhost:8080/api/status")
        return r.json()


# ─── Harness ──────────────────────────────────────────────────────────────


def _percentiles(samples: list[float]) -> dict:
    if not samples:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    s = sorted(samples)

    def pct(p):
        return s[min(len(s) - 1, int(len(s) * p))]

    return {
        "n": len(s),
        "min": round(s[0], 1),
        "p50": round(pct(0.50), 1),
        "p95": round(pct(0.95), 1),
        "p99": round(pct(0.99), 1),
        "max": round(s[-1], 1),
        "mean": round(statistics.fmean(s), 1),
    }


async def sequential_probe(name: str, coroutine_factory, iterations: int) -> dict:
    samples: list[float] = []
    async with httpx.AsyncClient() as c:
        for _ in range(iterations):
            ms = await coroutine_factory(c)
            samples.append(ms)
    return {"probe": name, "mode": "sequential", "latency_ms": _percentiles(samples)}


async def concurrent_probe(name: str, coroutine_factory, concurrency: int, total: int) -> dict:
    samples: list[float] = []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as c:

        async def one():
            async with sem:
                samples.append(await coroutine_factory(c))

        await asyncio.gather(*(one() for _ in range(total)))
    return {
        "probe": name,
        "mode": f"concurrent(c={concurrency},n={total})",
        "latency_ms": _percentiles(samples),
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sequential-iters", type=int, default=25)
    p.add_argument("--concurrent-iters", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--label", default="baseline")
    args = p.parse_args()

    results: dict = {
        "label": args.label,
        "timestamp": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "sequential_iters": args.sequential_iters,
            "concurrent_iters": args.concurrent_iters,
            "concurrency": args.concurrency,
        },
    }

    # 1. Scribe-main status snapshot (before probing)
    async with httpx.AsyncClient() as c:
        results["scribe_status_before"] = await probe_scribe_status(c)

    # 2. Sequential probes (warm cache state, single-stream latency)
    print("== sequential probes ==")
    results["asr_sequential"] = await sequential_probe("asr", probe_asr, args.sequential_iters)
    print(f"  asr: {results['asr_sequential']['latency_ms']}")

    # Translate: rotate through JA and EN phrases so the LRU cache doesn't
    # dominate the sample. We're measuring cold model latency + throughput.
    async def _translate_ja_en(c):
        import random

        t = random.choice(_JA_PHRASES)
        return await probe_translate(c, t, "ja", "en")

    async def _translate_en_ja(c):
        import random

        t = random.choice(_EN_PHRASES)
        return await probe_translate(c, t, "en", "ja")

    results["translate_ja_en_sequential"] = await sequential_probe(
        "translate_ja_en", _translate_ja_en, args.sequential_iters
    )
    print(f"  translate ja→en: {results['translate_ja_en_sequential']['latency_ms']}")

    results["translate_en_ja_sequential"] = await sequential_probe(
        "translate_en_ja", _translate_en_ja, args.sequential_iters
    )
    print(f"  translate en→ja: {results['translate_en_ja_sequential']['latency_ms']}")

    results["diarize_sequential"] = await sequential_probe(
        "diarize", probe_diarize, args.sequential_iters
    )
    print(f"  diarize: {results['diarize_sequential']['latency_ms']}")

    # 3. Concurrent probes — mix everything at once to measure how the
    # stack behaves when a meeting is spewing audio, translations, and
    # diarize calls simultaneously.
    print("== concurrent probes ==")
    results["asr_concurrent"] = await concurrent_probe(
        "asr", probe_asr, args.concurrency, args.concurrent_iters
    )
    print(f"  asr: {results['asr_concurrent']['latency_ms']}")

    results["translate_concurrent"] = await concurrent_probe(
        "translate_ja_en", _translate_ja_en, args.concurrency, args.concurrent_iters
    )
    print(f"  translate ja→en: {results['translate_concurrent']['latency_ms']}")

    results["diarize_concurrent"] = await concurrent_probe(
        "diarize", probe_diarize, args.concurrency, args.concurrent_iters
    )
    print(f"  diarize: {results['diarize_concurrent']['latency_ms']}")

    # 4. Mixed pipeline stress — simulates a live meeting's actual load:
    # ASR + diarize + translate all running in parallel.
    print("== mixed pipeline stress (simulates live meeting) ==")
    mixed_samples_asr: list[float] = []
    mixed_samples_tr: list[float] = []
    mixed_samples_diar: list[float] = []

    async def asr_worker(c):
        for _ in range(args.concurrent_iters // 3):
            mixed_samples_asr.append(await probe_asr(c))

    async def tr_worker(c):
        for _ in range(args.concurrent_iters // 3):
            mixed_samples_tr.append(await _translate_ja_en(c))

    async def diar_worker(c):
        for _ in range(args.concurrent_iters // 3):
            mixed_samples_diar.append(await probe_diarize(c))

    t_mixed_start = time.monotonic()
    async with httpx.AsyncClient() as c:
        await asyncio.gather(asr_worker(c), tr_worker(c), diar_worker(c))
    mixed_wall_ms = (time.monotonic() - t_mixed_start) * 1000.0

    results["mixed_stress"] = {
        "wall_ms": round(mixed_wall_ms, 1),
        "asr": _percentiles(mixed_samples_asr),
        "translate": _percentiles(mixed_samples_tr),
        "diarize": _percentiles(mixed_samples_diar),
    }
    print(f"  mixed total wall: {mixed_wall_ms:.0f}ms")
    print(f"  asr: {results['mixed_stress']['asr']}")
    print(f"  translate: {results['mixed_stress']['translate']}")
    print(f"  diarize: {results['mixed_stress']['diarize']}")

    # 5. Scribe-main post-probe state
    async with httpx.AsyncClient() as c:
        results["scribe_status_after"] = await probe_scribe_status(c)

    # 6. Surface key top-level deltas
    before = results["scribe_status_before"].get("metrics", {}).get("loop_lag_ms", {}) or {}
    after = results["scribe_status_after"].get("metrics", {}).get("loop_lag_ms", {}) or {}
    results["loop_lag_delta"] = {
        "p50_before": before.get("p50"),
        "p50_after": after.get("p50"),
        "p95_before": before.get("p95"),
        "p95_after": after.get("p95"),
        "p99_before": before.get("p99"),
        "p99_after": after.get("p99"),
    }

    gpu_before = results["scribe_status_before"].get("gpu", {})
    gpu_after = results["scribe_status_after"].get("gpu", {})
    results["gpu_vram_pct"] = {
        "before": gpu_before.get("vram_pct"),
        "after": gpu_after.get("vram_pct"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    # Short-form summary
    print(f"\n== {args.label} SUMMARY ==")
    for key in (
        "asr_sequential",
        "translate_ja_en_sequential",
        "translate_en_ja_sequential",
        "diarize_sequential",
    ):
        r = results[key]
        lm = r["latency_ms"]
        print(
            f"  {r['probe']:22s} p50={lm['p50']}ms p95={lm['p95']}ms p99={lm['p99']}ms n={lm['n']}"
        )
    for key in ("asr_concurrent", "translate_concurrent", "diarize_concurrent"):
        r = results[key]
        lm = r["latency_ms"]
        print(f"  {r['probe']:22s} {r['mode']:30s} p50={lm['p50']}ms p95={lm['p95']}ms")
    print(
        f"  mixed wall={results['mixed_stress']['wall_ms']}ms (ASR+translate+diarize concurrently)"
    )
    print(
        f"  loop_lag p99 before={results['loop_lag_delta']['p99_before']} after={results['loop_lag_delta']['p99_after']}"
    )
    print(f"  gpu_pct {gpu_before.get('vram_pct')} → {gpu_after.get('vram_pct')}")


if __name__ == "__main__":
    asyncio.run(main())
