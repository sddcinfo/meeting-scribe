"""GPU profiling wrapper for Phase A/C.

Runs ``nvidia-smi dmon -s umtpc`` (util / mem / temp / power / clocks)
alongside a workload invocation, captures the stream to a CSV, scrapes
each managed vLLM `/metrics` endpoint once at start and once at end.

Usage:
    python benchmarks/gpu_profile.py --duration 120 \
        --vllm-urls http://localhost:8003 http://localhost:8002 \
        --out benchmarks/results/baseline_2026-04-13/gpu_profile.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

import httpx


async def scrape_metrics(urls: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=5) as c:
        for u in urls:
            try:
                r = await c.get(f"{u}/metrics")
                out[u] = r.text if r.status_code == 200 else f"ERR {r.status_code}"
            except Exception as exc:
                out[u] = f"ERR {type(exc).__name__}: {exc}"
    return out


async def run(duration: int, vllm_urls: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dmon_csv = out.with_suffix(".dmon.csv")
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "umtpc", "-c", str(duration), "-f", str(dmon_csv)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    t0 = time.monotonic()
    start = await scrape_metrics(vllm_urls)
    # Let dmon run for the full duration.
    while dmon.poll() is None and (time.monotonic() - t0) < duration + 5:
        await asyncio.sleep(1.0)
    if dmon.poll() is None:
        dmon.terminate()
    end = await scrape_metrics(vllm_urls)
    summary = {
        "duration_s": duration,
        "dmon_csv": str(dmon_csv),
        "vllm_metrics_start": start,
        "vllm_metrics_end": end,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"dmon_csv": str(dmon_csv), "duration_s": duration}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=120)
    p.add_argument("--vllm-urls", nargs="+", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(run(args.duration, args.vllm_urls, args.out))


if __name__ == "__main__":
    main()
