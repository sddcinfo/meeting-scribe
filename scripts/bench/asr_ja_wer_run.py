"""Track A: synthesize JA audio from en_ja_meeting_v2.jsonl, send each
clip to two ASR backends, score character-level WER.

Why TTS-synthesized references rather than CommonVoice/Fleurs:

* Public-domain JA corpora are gated (CommonVoice 17 needs accept) or
  awkward to slice without the `datasets` library.
* The 72-pair `en_ja_meeting_v2` corpus is the same JA reference text
  we already trust for translation BLEU; using it as ASR ground truth
  keeps both axes scored against one curated reference.
* Audio is synthesized fresh at run time and lives offline; nothing
  PII-touched is involved.
* WER is computed at the character level (JA has no word delimiters).

Outputs land at:
    /data/meeting-scribe-fixtures/bench-runs/2026-Q2/asr_cohere/
        ja_synth_wavs/<corpus_id>.wav
        qwen3_asr.json
        cohere_transcribe.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import statistics
import sys
import time
import unicodedata
import wave
from pathlib import Path

import httpx

CORPUS = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "corpus"
    / "en_ja_meeting_v2.jsonl"
)


# ---------------------------------------------------------------------------
# Audio synthesis (Qwen3-TTS on prod 8002)
# ---------------------------------------------------------------------------


async def _synth_ja(client: httpx.AsyncClient, tts_url: str, text: str) -> bytes:
    body = {
        "model": "qwen3-tts",
        "input": text,
        "voice": "ono_anna",  # JA voice in our prod TTS pool
        "stream": True,
        "response_format": "pcm",
        "priority": -10,
    }
    pcm = bytearray()
    async with client.stream("POST", f"{tts_url}/v1/audio/speech", json=body) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            pcm.extend(chunk)
    return bytes(pcm)


def _pcm_to_wav(pcm: bytes, path: Path, sample_rate: int = 24_000) -> None:
    """Write s16le PCM as WAV, downsampled to 16 kHz.

    Qwen3-ASR's vLLM container ships without ``vllm[audio]`` so it
    cannot resample on the fly; both ASR backends accept 16 kHz mono
    natively, so we downsample once at synthesis time.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    clean = pcm[: len(pcm) - (len(pcm) % 2)]
    samples = np.frombuffer(clean, dtype=np.int16)
    if sample_rate != 16_000:
        # Simple linear-interpolation resample is fine for ASR input.
        n_target = round(len(samples) * 16_000 / sample_rate)
        x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_target, endpoint=False)
        resampled = np.interp(x_new, x_old, samples.astype(np.float32))
        samples = np.clip(resampled, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# ASR clients (OpenAI shape)
# ---------------------------------------------------------------------------


async def _asr_request(
    client: httpx.AsyncClient,
    url: str,
    wav_bytes: bytes,
    *,
    model: str,
    language: str | None,
) -> tuple[str, float]:
    audio_b64 = base64.b64encode(wav_bytes).decode()
    body: dict = {
        "model": model,
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
    }
    if language:
        body["language"] = language

    t0 = time.monotonic()
    r = await client.post(f"{url}/v1/chat/completions", json=body, timeout=120)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    # Qwen3-ASR returns "language X<asr_text>...".  Strip if present.
    if "<asr_text>" in raw:
        _, _, raw = raw.partition("<asr_text>")
    return raw.strip(), elapsed_ms


# ---------------------------------------------------------------------------
# JA character-level WER
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[、。「」『』！？!?,.\s]+")


def _normalize_ja(text: str) -> str:
    """NFKC + strip whitespace + drop punctuation for fair WER scoring."""
    nfkc = unicodedata.normalize("NFKC", text)
    return _PUNCT_RE.sub("", nfkc)


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def char_wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Returns (WER fraction, edit_count, ref_token_count)."""
    ref = list(_normalize_ja(reference))
    hyp = list(_normalize_ja(hypothesis))
    if not ref:
        return (0.0 if not hyp else 1.0, len(hyp), 0)
    edits = _levenshtein(ref, hyp)
    return edits / len(ref), edits, len(ref)


# ---------------------------------------------------------------------------
# Bench driver
# ---------------------------------------------------------------------------


async def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "ja_synth_wavs"

    # Load JA references — two corpus sources supported:
    #
    #   tts_synth     — synthesize JA via Qwen3-TTS from the
    #                    `en_ja_meeting_v2.jsonl` corpus (the original
    #                    Track A path).  Audio is clean to the point of
    #                    ceiling-bound WER on most samples; CI tends to
    #                    straddle zero on small subsets.
    #   public_domain — read pre-pulled Fleurs JA WAVs from the manifest
    #                    (`benchmarks/fixtures/meeting_consolidation/MANIFEST.yaml`).
    #                    Real recordings, no synthetic ceiling.
    pairs: list[tuple[str, str]] = []
    if args.corpus_source == "tts_synth":
        for line in CORPUS.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            # The corpus has both directions:
            #   ej_*  EN source -> JA reference (use reference_text — JA)
            #   je_*  JA source -> EN reference (use source_text — JA)
            # Either way we want the JA-language utterance for the ASR bench.
            if rec.get("source_lang") == "ja":
                ja_text = rec["source_text"]
            elif rec.get("target_lang") == "ja":
                ja_text = rec["reference_text"]
            else:
                continue
            pairs.append((rec["corpus_id"], ja_text))
    elif args.corpus_source == "public_domain":
        # Pre-recorded clips on disk; no synthesis step.  Manifest entries
        # under `kind: asr, language: ja` carry the wav path + reference
        # transcript at `<asr_dir>/<id>.{wav,txt}`.
        import yaml

        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "fixtures"
            / "meeting_consolidation"
            / "MANIFEST.yaml"
        )
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        asr_dir = Path(args.fixture_dir) / "asr"
        for entry in manifest.get("samples", []) or []:
            if entry.get("kind") != "asr":
                continue
            if entry.get("language") not in ("ja",) and not args.public_include_en:
                continue
            wav = asr_dir / f"{entry['id']}.wav"
            txt = asr_dir / f"{entry['id']}.txt"
            if not wav.exists() or not txt.exists():
                print(f"skip {entry['id']}: file missing under {asr_dir}", flush=True)
                continue
            pairs.append((entry["id"], txt.read_text(encoding="utf-8").strip()))
    else:
        raise SystemExit(f"unknown corpus_source: {args.corpus_source}")

    if args.limit:
        pairs = pairs[: args.limit]
    print(f"corpus pairs: {len(pairs)} (source={args.corpus_source})", flush=True)

    qwen_results: list[dict] = []
    cohere_results: list[dict] = []

    async with httpx.AsyncClient(timeout=120) as c:
        for i, (cid, ja_text) in enumerate(pairs, 1):
            if args.corpus_source == "tts_synth":
                wav_path = wav_dir / f"{cid}.wav"
                if not wav_path.exists():
                    pcm = await _synth_ja(c, args.tts_url, ja_text)
                    _pcm_to_wav(pcm, wav_path)
            else:  # public_domain — pre-recorded clip
                wav_path = (
                    Path(args.fixture_dir) / "asr" / f"{cid}.wav"
                )
            wav_bytes = wav_path.read_bytes()

            # Qwen3-ASR
            try:
                qhyp, q_ms = await _asr_request(
                    c, args.qwen_url, wav_bytes, model=args.qwen_model, language=None
                )
            except Exception as e:
                print(f"qwen FAIL {cid}: {e}", flush=True)
                continue
            q_wer, q_edits, ref_tokens = char_wer(ja_text, qhyp)
            qwen_results.append({
                "id": cid,
                "language": "ja",
                "reference_tokens": ref_tokens,
                "wer": q_wer,
                "edits": q_edits,
                "total_ms": q_ms,
                "hypothesis": qhyp,
            })

            # Cohere Transcribe
            try:
                chyp, c_ms = await _asr_request(
                    c, args.cohere_url, wav_bytes, model="auto", language="ja"
                )
            except Exception as e:
                print(f"cohere FAIL {cid}: {e}", flush=True)
                continue
            c_wer, c_edits, _ = char_wer(ja_text, chyp)
            cohere_results.append({
                "id": cid,
                "language": "ja",
                "reference_tokens": ref_tokens,
                "wer": c_wer,
                "edits": c_edits,
                "total_ms": c_ms,
                "hypothesis": chyp,
            })

            if i % 10 == 0 or i == len(pairs):
                print(
                    f"[{i}/{len(pairs)}] qwen wer={q_wer:.3f} {q_ms:.0f}ms  "
                    f"cohere wer={c_wer:.3f} {c_ms:.0f}ms",
                    flush=True,
                )

    def summarize(results: list[dict]) -> dict:
        wers = [r["wer"] for r in results]
        totals = [r["total_ms"] for r in results]
        sum_edits = sum(r["edits"] for r in results)
        sum_tokens = sum(r["reference_tokens"] for r in results)
        return {
            "samples": len(results),
            "corpus_wer": (sum_edits / sum_tokens) if sum_tokens else None,
            "p50_wer": statistics.median(wers) if wers else None,
            "p95_wer": (
                statistics.quantiles(wers, n=20)[18] if len(wers) >= 20 else None
            ),
            "p50_total_ms": statistics.median(totals) if totals else None,
            "p95_total_ms": (
                statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else None
            ),
            "per_sample": results,
        }

    qwen_summary = summarize(qwen_results)
    cohere_summary = summarize(cohere_results)
    (out_dir / "qwen3_asr.json").write_text(json.dumps(qwen_summary, indent=2))
    (out_dir / "cohere_transcribe.json").write_text(json.dumps(cohere_summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps({
        "qwen": {k: v for k, v in qwen_summary.items() if k != "per_sample"},
        "cohere": {k: v for k, v in cohere_summary.items() if k != "per_sample"},
    }, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-url", default="http://localhost:8003")
    p.add_argument("--cohere-url", default="http://localhost:8013")
    p.add_argument("--tts-url", default="http://localhost:8002")
    p.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all corpus rows")
    p.add_argument(
        "--corpus-source",
        choices=("tts_synth", "public_domain"),
        default="tts_synth",
        help=(
            "tts_synth: synthesize JA via Qwen3-TTS from en_ja_meeting_v2 (Track A original). "
            "public_domain: read pre-pulled Fleurs JA WAVs from MANIFEST.yaml (Phase B1)."
        ),
    )
    p.add_argument(
        "--fixture-dir",
        default="/data/meeting-scribe-fixtures",
        help="Offline fixture root (used by --corpus-source public_domain).",
    )
    p.add_argument(
        "--public-include-en",
        action="store_true",
        help="Also include EN entries (Fleurs en_us) when --corpus-source public_domain.",
    )
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks._bench_paths import assert_offline_path

    assert_offline_path(args.out_dir)

    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
