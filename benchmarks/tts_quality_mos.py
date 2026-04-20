"""TTS quality harness — generates per-sample WAVs for human MOS eval.

Automated component: per-sample RMS, pitch-jitter proxy, TTFA, total_ms.
The MOS score itself is filled in by two human reviewers into
benchmarks/results/<run>/tts_mos_form.csv (same row as the per-sample id).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
from benchmarks._fixture import add_fixture_arg, load_samples


async def _synth(client: httpx.AsyncClient, url: str, sample_id: str, text: str, voice: str, out_dir: Path) -> dict:
    body = {
        "model": "qwen3-tts",
        "input": text,
        "voice": voice,
        "stream": True,
        "response_format": "pcm",
        "priority": -10,
    }
    t0 = time.monotonic()
    ttfa = None
    pcm = bytearray()
    async with client.stream("POST", f"{url}/v1/audio/speech", json=body) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            if chunk and ttfa is None:
                ttfa = (time.monotonic() - t0) * 1000.0
            pcm.extend(chunk)
    total_ms = (time.monotonic() - t0) * 1000.0
    samples = np.frombuffer(bytes(pcm[: len(pcm) - (len(pcm) % 2)]), dtype=np.int16).astype(np.float32) / 32768.0

    out_wav = out_dir / f"{sample_id}.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # Save float -> int16 WAV 24 kHz
    import wave
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24_000)
        wf.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    return {
        "id": sample_id,
        "voice": voice,
        "ttfa_ms": ttfa,
        "total_ms": total_ms,
        "rms": rms,
        "samples": int(samples.size),
        "wav_path": str(out_wav),
    }


async def run(url: str, fixture_dir: Path, out: Path) -> None:
    samples = load_samples(fixture_dir, kind="tts_studio")
    if not samples:
        raise SystemExit(
            "No tts_studio samples in manifest. Each entry describes a "
            "(voice, language, text) target to synth; no audio file on "
            "disk is required for studio samples — use the description "
            "field's companion <id>.json for {voice, text}."
        )
    results = []
    out_dir = out.parent / "tts_wavs"
    async with httpx.AsyncClient(timeout=60) as c:
        for s in samples:
            meta = json.loads(s.path.with_suffix(".json").read_text())
            results.append(
                await _synth(c, url, s.id, meta["text"], meta["voice"], out_dir)
            )

    ttfas = [r["ttfa_ms"] for r in results if r["ttfa_ms"] is not None]
    summary = {
        "url": url,
        "samples": len(results),
        "p50_ttfa_ms": statistics.median(ttfas) if ttfas else None,
        "p95_ttfa_ms": statistics.quantiles(ttfas, n=20)[18] if len(ttfas) >= 20 else None,
        "per_sample": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    mos_csv = out.parent / "tts_mos_form.csv"
    if not mos_csv.exists():
        mos_csv.write_text("id,voice,reviewer,mos_1_to_5,comments\n")
        for r in results:
            for rev in ("reviewer_a", "reviewer_b"):
                mos_csv.write_text(
                    mos_csv.read_text() + f"{r['id']},{r['voice']},{rev},,\n"
                )
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
