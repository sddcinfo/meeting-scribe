"""Translate accuracy (BLEU) + latency for Omni consolidation Phase A/C.

Fixture: <fixture-dir>/translate/<id>.json with keys
  {"source": str, "source_lang": "ja"|"en", "target_lang": "ja"|"en", "reference": str}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
from benchmarks._fixture import add_fixture_arg, load_samples


def _bleu(reference: str, hypothesis: str) -> float:
    """Tiny token-level BLEU-1/2 harmonic proxy — good enough for gating.

    For the real plan we swap this for sacrebleu; this skeleton just
    keeps dependencies minimal until the fixture populates.
    """
    from collections import Counter

    def _ngrams(s: str, n: int) -> list[tuple[str, ...]]:
        toks = s.split()
        return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]

    scores: list[float] = []
    for n in (1, 2):
        ref = Counter(_ngrams(reference, n))
        hyp = Counter(_ngrams(hypothesis, n))
        if not hyp:
            scores.append(0.0)
            continue
        overlap = sum((ref & hyp).values())
        scores.append(overlap / max(sum(hyp.values()), 1))
    return statistics.fmean(scores)


async def _translate(client: httpx.AsyncClient, url: str, sample_path: Path) -> dict:
    meta = json.loads(sample_path.with_suffix(".json").read_text())
    body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": f"Translate from {meta['source_lang']} to {meta['target_lang']}.",
            },
            {"role": "user", "content": meta["source"]},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
        "priority": -10,
    }
    t0 = time.monotonic()
    r = await client.post(f"{url}/v1/chat/completions", json=body)
    total_ms = (time.monotonic() - t0) * 1000.0
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"].strip()
    return {
        "id": sample_path.stem,
        "bleu_proxy": _bleu(meta["reference"], out),
        "total_ms": total_ms,
    }


async def run(url: str, fixture_dir: Path, out: Path) -> None:
    samples = load_samples(fixture_dir, kind="translate")
    if not samples:
        raise SystemExit(
            "No translate samples in manifest. Add entries and place "
            "<id>.json files at <fixture-dir>/translate/"
        )
    results = []
    async with httpx.AsyncClient(timeout=60) as c:
        for s in samples:
            results.append(await _translate(c, url, s.path))

    bleus = [r["bleu_proxy"] for r in results]
    totals = [r["total_ms"] for r in results]
    summary = {
        "url": url,
        "samples": len(results),
        "p50_bleu_proxy": statistics.median(bleus),
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
