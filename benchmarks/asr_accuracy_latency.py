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


import re
import unicodedata

# Languages that don't use whitespace word boundaries — score with
# character-level edit distance (CER) instead of word-level (WER).
# `cmn` is the Fleurs label for Mandarin (zh).
_CHAR_LEVEL_LANGUAGES = frozenset({"zh", "cmn", "ja", "th"})

# Punctuation/symbol Unicode general categories: P (punctuation), S (symbols).
_PUNCT_CATEGORIES = frozenset({"P", "S"})


def _normalize(text: str) -> str:
    """Lowercase + strip punctuation — fair WER across languages.

    Fleurs references contain locale-specific punctuation (e.g. CJK ，。）
    that ASR output lacks, and German/Russian references rely on case the
    model frequently lowercases. Stripping both is the standard practice
    for fleurs scoring (e.g. NIST sclite normalization).
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] not in _PUNCT_CATEGORIES)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    d = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(reference)][len(hypothesis)] / len(reference)


def _score(reference: str, hypothesis: str, language: str) -> tuple[float, str]:
    """Return (error_rate, metric_name) — CER for unspaced scripts, else WER."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if language in _CHAR_LEVEL_LANGUAGES:
        # Fleurs CJK references space every character (`它 的 长`); the
        # model emits contiguous text. Strip all whitespace for char-level
        # scoring so that artifact doesn't dominate the edit distance.
        ref = re.sub(r"\s+", "", ref)
        hyp = re.sub(r"\s+", "", hyp)
        return _edit_distance(list(ref), list(hyp)), "cer"
    return _edit_distance(ref.split(), hyp.split()), "wer"


# ISO 639-1 → Qwen3-ASR English-name hint (mirrors backends/asr_vllm).
_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "cmn": "Chinese",  # Fleurs label for Mandarin
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "nl": "Dutch",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "uk": "Ukrainian",
}


def _wav_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


async def _transcribe(
    client: httpx.AsyncClient, url: str, model: str, sample: Sample
) -> dict:
    transcript_path = sample.path.with_suffix(".txt")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Ground-truth transcript missing: {transcript_path}")
    reference = transcript_path.read_text().strip()

    audio_b64 = _wav_to_b64(sample.path)
    # Qwen3-ASR needs an explicit language hint or it refuses to emit text;
    # use the manifest language to mirror what the live path does in
    # backends/asr_vllm._build_system_prompt.
    lang_name = _LANGUAGE_NAMES.get(sample.language, sample.language.upper())
    system_prompt = f"Transcribe the {lang_name} audio. Do not translate."

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
    err, metric = _score(reference, raw, sample.language)
    return {
        "id": sample.id,
        "language": sample.language,
        "metric": metric,
        "error_rate": err,
        "total_ms": total_ms,
    }


async def run(
    url: str,
    model: str,
    fixture_dir: Path,
    out: Path,
    *,
    limit: int | None = None,
    language: str | None = None,
) -> None:
    samples = load_samples(fixture_dir, kind="asr", language=language)
    if not samples:
        raise SystemExit(
            "No ASR samples in manifest. Populate "
            "benchmarks/fixtures/meeting_consolidation/MANIFEST.yaml "
            "and place audio at <fixture-dir>/asr/<id>.wav"
        )
    if limit is not None:
        samples = samples[:limit]
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=60) as c:
        for s in samples:
            results.append(await _transcribe(c, url, model, s))

    def _agg(rs: list[dict]) -> dict:
        if not rs:
            return {}
        errs = [r["error_rate"] for r in rs]
        totals = [r["total_ms"] for r in rs]
        return {
            "samples": len(rs),
            "metric": rs[0]["metric"],
            "p50_error": statistics.median(errs),
            "p95_error": (
                statistics.quantiles(errs, n=20)[18] if len(errs) >= 20 else None
            ),
            "p50_total_ms": statistics.median(totals),
            "p95_total_ms": (
                statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else None
            ),
        }

    by_language: dict[str, list[dict]] = {}
    for r in results:
        by_language.setdefault(r["language"], []).append(r)

    summary = {
        "url": url,
        "model": model,
        "overall": _agg(results),
        "per_language": {lang: _agg(rs) for lang, rs in sorted(by_language.items())},
        "per_sample": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model name to send in the OpenAI-compatible request body.",
    )
    add_fixture_arg(p)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N samples.")
    p.add_argument(
        "--language",
        default=None,
        help="ISO-639-1 language filter (e.g. en, ja). Default: all.",
    )
    args = p.parse_args()
    asyncio.run(
        run(
            args.url,
            args.model,
            args.fixture_dir,
            args.out,
            limit=args.limit,
            language=args.language,
        )
    )


if __name__ == "__main__":
    main()
