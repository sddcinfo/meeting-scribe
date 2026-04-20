#!/usr/bin/env python3
"""vLLM optimization benchmark suite for GB10.

Tests different vLLM configurations to find optimal settings for
meeting-scribe translation workload on the GB10 (128GB unified memory).

Optimization levers tested:
- gpu-memory-utilization: 0.70 → 0.92 (KV cache size)
- max-num-seqs: 10 → 64 (concurrent batch size)
- max-num-batched-tokens: 4096 → 16384 (prefill budget)
- enforce-eager vs CUDAGraph (compilation overhead vs throughput)
- kv-cache-dtype: auto / fp8 / turboquant (memory vs quality)
- enable-prefix-caching: on/off (repeated prompt optimization)
- chunked-prefill: on/off (long context efficiency)
- attention-backend: default / flashinfer / triton

Usage:
    # Benchmark current config
    python benchmarks/vllm_optimization.py --url http://localhost:8010

    # Benchmark with concurrent requests (simulates real meeting load)
    python benchmarks/vllm_optimization.py --url http://localhost:8010 --concurrency 1,4,8,16,32

    # Compare two configs
    python benchmarks/vllm_optimization.py --compare results/config_a.json results/config_b.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

# Translation workload that matches real meeting-scribe usage:
# Short segments (1-3 sentences), mixed JA/EN, system prompt reused
SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English translator. "
    "Translate the following Japanese text into natural, fluent English. "
    "Preserve the meaning, tone, and context. "
    "Return only the translation, no explanation or commentary."
)

# Real meeting-style segments (short, conversational)
SEGMENTS = [
    "今日の会議の議題について確認しましょう。",
    "この提案について何かご質問はありますか？",
    "次のステップを決めましょう。",
    "四半期の売上目標を達成しました。",
    "新しいプロジェクトの予算を承認する必要があります。",
    "来週のデモに向けて準備を進めています。",
    "APIのレスポンスタイムが改善されました。",
    "データベースのマイグレーションは明日実行します。",
    "本番環境にデプロイする前にテストを完了してください。",
    "すみません、もう一度言っていただけますか？",
    "その点については同意します。",
    "詳しく説明していただけますか？",
    "お客様からのフィードバックに基づいて、UIを改善する計画です。",
    "この問題を解決するためには、チーム全体で協力する必要があると考えています。",
    "スケジュールの遅延を最小限に抑えるため、優先順位を再検討しましょう。",
]


async def _translate_one(client: httpx.AsyncClient, url: str, model: str, text: str) -> tuple[float, str]:
    """Translate a single segment, return (latency_ms, translated_text)."""
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "max_tokens": 150,
                "temperature": 0.1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        latency = (time.monotonic() - t0) * 1000
        return latency, raw
    except Exception as e:
        return (time.monotonic() - t0) * 1000, f"ERROR: {e}"


async def benchmark_concurrency(
    url: str,
    concurrency: int,
    num_requests: int = 30,
) -> dict:
    """Benchmark throughput at a given concurrency level."""
    async with httpx.AsyncClient(timeout=60) as client:
        # Auto-detect model
        try:
            r = await client.get(f"{url}/v1/models")
            model = r.json()["data"][0]["id"]
        except Exception:
            model = "default"

        # Warmup
        await _translate_one(client, url, model, SEGMENTS[0])

        # Run concurrent requests
        semaphore = asyncio.Semaphore(concurrency)
        latencies = []
        errors = 0

        async def bounded_translate(text):
            async with semaphore:
                lat, result = await _translate_one(client, url, model, text)
                if result.startswith("ERROR"):
                    nonlocal errors
                    errors += 1
                return lat

        t0 = time.monotonic()
        tasks = [bounded_translate(SEGMENTS[i % len(SEGMENTS)]) for i in range(num_requests)]
        latencies = await asyncio.gather(*tasks)
        wall_time = time.monotonic() - t0

        valid = [lat for lat in latencies if lat < 30000]  # Filter timeouts

        return {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "wall_time_s": round(wall_time, 2),
            "throughput_rps": round(num_requests / wall_time, 2),
            "latency_p50_ms": round(statistics.median(valid), 1) if valid else 0,
            "latency_p90_ms": round(sorted(valid)[int(len(valid) * 0.9)], 1) if valid else 0,
            "latency_p99_ms": round(sorted(valid)[int(len(valid) * 0.99)], 1) if valid else 0,
            "latency_mean_ms": round(statistics.mean(valid), 1) if valid else 0,
            "errors": errors,
        }


async def run_benchmark(url: str, concurrency_levels: list[int], label: str) -> dict:
    """Run full benchmark across concurrency levels."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"URL: {url}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"{'='*60}\n")

    # Get server info and vLLM config
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{url}/v1/models")
            model = r.json()["data"][0]["id"]
            print(f"Model: {model}")
        except Exception:
            model = "unknown"

    # Capture vLLM container config for reproducibility
    import subprocess
    vllm_config = {}
    try:
        cmd_out = subprocess.run(
            ["docker", "inspect", "autosre-vllm-local", "--format", "{{join .Config.Cmd \" \"}}"],
            capture_output=True, text=True, timeout=5,
        )
        if cmd_out.returncode == 0:
            vllm_config["command"] = cmd_out.stdout.strip()
        driver_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if driver_out.returncode == 0:
            vllm_config["driver"] = driver_out.stdout.strip()
        vllm_config["gpu"] = "NVIDIA GB10"
    except Exception:
        pass

    results = []
    for conc in concurrency_levels:
        print(f"\n--- Concurrency {conc} ---")
        result = await benchmark_concurrency(url, conc)
        results.append(result)
        print(f"  Throughput: {result['throughput_rps']} req/s")
        print(f"  Latency p50/p90/p99: {result['latency_p50_ms']}/{result['latency_p90_ms']}/{result['latency_p99_ms']}ms")
        print(f"  Wall time: {result['wall_time_s']}s, Errors: {result['errors']}")

    output = {
        "label": label,
        "url": url,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vllm_config": vllm_config,
        "results": results,
    }

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Conc':>5} {'RPS':>8} {'p50ms':>8} {'p90ms':>8} {'p99ms':>8} {'Errors':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['concurrency']:>5} {r['throughput_rps']:>8.1f} {r['latency_p50_ms']:>8.1f} {r['latency_p90_ms']:>8.1f} {r['latency_p99_ms']:>8.1f} {r['errors']:>6}")

    return output


def compare_results(file_a: str, file_b: str):
    """Compare two benchmark results."""
    a = json.loads(Path(file_a).read_text())
    b = json.loads(Path(file_b).read_text())

    print(f"\n{'='*70}")
    print(f"A: {a['label']} | B: {b['label']}")
    print(f"{'='*70}")
    print(f"{'Conc':>5} | {'A RPS':>8} {'A p50':>8} | {'B RPS':>8} {'B p50':>8} | {'Δ RPS':>8}")
    print("-" * 70)

    a_by_conc = {r["concurrency"]: r for r in a["results"]}
    b_by_conc = {r["concurrency"]: r for r in b["results"]}

    for conc in sorted(set(list(a_by_conc.keys()) + list(b_by_conc.keys()))):
        ar = a_by_conc.get(conc, {})
        br = b_by_conc.get(conc, {})
        a_rps = ar.get("throughput_rps", 0)
        b_rps = br.get("throughput_rps", 0)
        delta = b_rps - a_rps
        sign = "+" if delta > 0 else ""
        print(f"{conc:>5} | {a_rps:>8.1f} {ar.get('latency_p50_ms',0):>7.0f}ms | {b_rps:>8.1f} {br.get('latency_p50_ms',0):>7.0f}ms | {sign}{delta:>7.1f}")


async def main():
    parser = argparse.ArgumentParser(description="vLLM optimization benchmark for GB10")
    parser.add_argument("--url", default="http://localhost:8010")
    parser.add_argument("--label", default="current")
    parser.add_argument("--concurrency", default="1,4,8,16", help="Comma-separated concurrency levels")
    parser.add_argument("--requests", type=int, default=30, help="Requests per concurrency level")
    parser.add_argument("--compare", nargs=2, metavar="FILE")
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    levels = [int(x) for x in args.concurrency.split(",")]
    output = await run_benchmark(args.url, levels, args.label)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.label.replace(' ', '_')}_concurrency.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
