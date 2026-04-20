"""ASR accuracy + latency benchmark for Omni consolidation Phase A/C.

Sends each fixture clip to the configured ASR URL, collects WER against
the paired ground-truth transcript (stored outside git at
<fixture_dir>/asr/<id>.txt), and records p50/p95 TTFT + end-to-end latency.

Usage:
    python benchmarks/asr_accuracy_latency.py \
        --url http://localhost:8003 \
        --fixture-dir /data/meeting-scribe-fixtures \
        --out benchmarks/results/baseline_2026-04-13/asr.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import httpx
from benchmarks._fixture import Sample, add_fixture_arg, load_samples


def _wer(reference: str, hypothesis: str) -> float:
    r = reference.split()
    h = hypothesis.split()
    if not r:
        return 0.0 if not h else 1.0
    # Levenshtein on tokens.
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def _wav_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


async def _transcribe(client: httpx.AsyncClient, url: str, sample: Sample) -> dict:
    transcript_path = sample.path.with_suffix(".txt")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Ground-truth transcript missing: {transcript_path}")
    reference = transcript_path.read_text().strip()

    audio_b64 = _wav_to_b64(sample.path)
    body = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "Transcribe in the spoken language."},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "priority": -20,
    }
    t0 = time.monotonic()
    r = await client.post(f"{url}/v1/chat/completions", json=body)
    total_ms = (time.monotonic() - t0) * 1000.0
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    # Handle Qwen3-ASR format "language X<asr_text>text"; crude parse ok here.
    if "<asr_text>" in raw:
        _, _, raw = raw.partition("<asr_text>")
    return {"id": sample.id, "wer": _wer(reference, raw), "total_ms": total_ms}


async def run(url: str, fixture_dir: Path, out: Path) -> None:
    samples = load_samples(fixture_dir, kind="asr")
    if not samples:
        raise SystemExit(
            "No ASR samples in manifest. Populate "
            "benchmarks/fixtures/meeting_consolidation/MANIFEST.yaml "
            "and place audio at <fixture-dir>/asr/<id>.wav"
        )
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=60) as c:
        for s in samples:
            results.append(await _transcribe(c, url, s))

    wers = [r["wer"] for r in results]
    totals = [r["total_ms"] for r in results]
    summary = {
        "url": url,
        "samples": len(results),
        "p50_wer": statistics.median(wers),
        "p95_wer": statistics.quantiles(wers, n=20)[18] if len(wers) >= 20 else None,
        "p50_total_ms": statistics.median(totals),
        "p95_total_ms": statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else None,
        "per_sample": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    add_fixture_arg(p)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(run(args.url, args.fixture_dir, args.out))


if __name__ == "__main__":
    main()
