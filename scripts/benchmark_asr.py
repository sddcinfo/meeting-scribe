#!/usr/bin/env python3
"""ASR Model Benchmarking Suite.

Compares ASR models on English and Japanese audio samples, measuring:
- Word Error Rate (WER) / Character Error Rate (CER)
- Latency (time to transcribe)
- Language detection accuracy

Designed to be rerun when new models become available (e.g., Qwen3.5-Omni-Plus).

Usage:
    python scripts/benchmark_asr.py                    # Run all benchmarks
    python scripts/benchmark_asr.py --model qwen3-asr  # Single model
    python scripts/benchmark_asr.py --generate-samples  # Create test samples from meetings

Requirements:
    - faster-whisper (pip install faster-whisper)
    - httpx (pip install httpx)
    - Qwen3-ASR vLLM running at localhost:8003
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks"
SAMPLES_DIR = BENCHMARK_DIR / "samples"
RESULTS_DIR = BENCHMARK_DIR / "results"

# ── Test Samples ────────────────────────────────────────────


@dataclass
class Sample:
    """A test audio sample with ground truth."""

    name: str
    language: str  # "en" or "ja"
    audio_path: Path
    reference: str  # ground truth transcript
    duration_s: float = 0.0

    def load_audio(self) -> np.ndarray:
        """Load audio as float32 numpy array at 16kHz."""
        if self.audio_path.suffix == ".pcm":
            raw = np.frombuffer(self.audio_path.read_bytes(), dtype=np.int16)
            return raw.astype(np.float32) / 32768.0
        audio, sr = sf.read(self.audio_path)
        if sr != SAMPLE_RATE:
            import torchaudio

            audio_t = __import__("torch").from_numpy(audio).float()
            if audio_t.dim() == 1:
                audio_t = audio_t.unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, SAMPLE_RATE)
            audio = audio_t.squeeze().numpy()
        self.duration_s = len(audio) / SAMPLE_RATE
        return audio


def load_builtin_samples() -> list[Sample]:
    """Load built-in benchmark samples from benchmarks/samples/."""
    samples = []
    manifest = SAMPLES_DIR / "manifest.json"
    if not manifest.exists():
        return samples
    for entry in json.loads(manifest.read_text()):
        samples.append(
            Sample(
                name=entry["name"],
                language=entry["language"],
                audio_path=SAMPLES_DIR / entry["file"],
                reference=entry["reference"],
            )
        )
    return samples


# ── WER/CER Calculation ─────────────────────────────────────


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[m]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) for English text."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) for Japanese/CJK text."""
    # Remove spaces for CJK comparison
    ref_chars = list(re.sub(r"\s+", "", reference))
    hyp_chars = list(re.sub(r"\s+", "", hypothesis))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


# ── ASR Model Backends ──────────────────────────────────────


@dataclass
class TranscriptionResult:
    text: str
    language: str
    latency_ms: float
    model_name: str


class ASRModel:
    """Abstract base for an ASR model to benchmark."""

    name: str

    def transcribe(self, audio: np.ndarray, language_hint: str = "") -> TranscriptionResult:
        raise NotImplementedError


class FasterWhisperModel(ASRModel):
    """faster-whisper (CTranslate2) — Whisper large-v3-turbo."""

    def __init__(self, model_size: str = "large-v3-turbo"):
        self.name = f"faster-whisper-{model_size}"
        self._model_size = model_size
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            print(f"  Loading {self.name}...")
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray, language_hint: str = "") -> TranscriptionResult:
        self._ensure_loaded()
        kwargs = {}
        if language_hint:
            kwargs["language"] = language_hint
        t0 = time.monotonic()
        segments, info = self._model.transcribe(audio, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        latency = (time.monotonic() - t0) * 1000
        return TranscriptionResult(
            text=text,
            language=info.language,
            latency_ms=latency,
            model_name=self.name,
        )


class Qwen3ASRModel(ASRModel):
    """Qwen3-ASR-1.7B via vLLM OpenAI-compatible endpoint."""

    def __init__(self, base_url: str = "http://localhost:8003"):
        self.name = "qwen3-asr-1.7b-vllm"
        self._base_url = base_url.rstrip("/")
        self._model_id: str | None = None

    def _get_model_id(self) -> str:
        if self._model_id is None:
            resp = httpx.get(f"{self._base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            self._model_id = resp.json()["data"][0]["id"]
        return self._model_id

    def transcribe(self, audio: np.ndarray, language_hint: str = "") -> TranscriptionResult:
        # Encode audio as base64 WAV
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        model_id = self._get_model_id()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            }
        ]

        t0 = time.monotonic()
        resp = httpx.post(
            f"{self._base_url}/v1/chat/completions",
            json={"model": model_id, "messages": messages, "temperature": 0.0, "max_tokens": 1024},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        latency = (time.monotonic() - t0) * 1000

        # Parse Qwen3-ASR response format: "language English<asr_text>text"
        text, lang = "", "unknown"
        if "<asr_text>" in raw:
            prefix, _, text = raw.partition("<asr_text>")
            text = text.strip()
            lang_raw = prefix.replace("language", "").strip().lower()
            lang = {"japanese": "ja", "english": "en"}.get(lang_raw, lang_raw)
        else:
            text = raw.strip()

        return TranscriptionResult(
            text=text,
            language=lang,
            latency_ms=latency,
            model_name=self.name,
        )


# ── Benchmark Runner ─────────────────────────────────────────


@dataclass
class BenchmarkResult:
    model: str
    sample: str
    language: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    latency_ms: float
    detected_language: str
    audio_duration_s: float
    rtf: float  # Real-Time Factor (latency / audio_duration)


def run_benchmark(
    models: list[ASRModel],
    samples: list[Sample],
) -> list[BenchmarkResult]:
    """Run all models against all samples."""
    results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model.name}")
        print(f"{'='*60}")

        for sample in samples:
            print(f"\n  Sample: {sample.name} ({sample.language})")
            audio = sample.load_audio()
            duration_s = len(audio) / SAMPLE_RATE

            try:
                result = model.transcribe(audio, language_hint=sample.language)

                wer = word_error_rate(sample.reference, result.text)
                cer = character_error_rate(sample.reference, result.text)
                rtf = result.latency_ms / (duration_s * 1000)

                print(f"    Reference:  {sample.reference[:80]}...")
                print(f"    Hypothesis: {result.text[:80]}...")
                print(f"    WER: {wer:.1%}  CER: {cer:.1%}  Latency: {result.latency_ms:.0f}ms  RTF: {rtf:.2f}")
                print(f"    Lang detect: {result.language} (expected: {sample.language})")

                results.append(
                    BenchmarkResult(
                        model=model.name,
                        sample=sample.name,
                        language=sample.language,
                        reference=sample.reference,
                        hypothesis=result.text,
                        wer=wer,
                        cer=cer,
                        latency_ms=result.latency_ms,
                        detected_language=result.language,
                        audio_duration_s=duration_s,
                        rtf=rtf,
                    )
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(
                    BenchmarkResult(
                        model=model.name,
                        sample=sample.name,
                        language=sample.language,
                        reference=sample.reference,
                        hypothesis=f"ERROR: {e}",
                        wer=1.0,
                        cer=1.0,
                        latency_ms=0,
                        detected_language="error",
                        audio_duration_s=duration_s,
                        rtf=0,
                    )
                )

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary comparison table."""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    # Group by model
    models = sorted(set(r.model for r in results))
    languages = sorted(set(r.language for r in results))

    for lang in languages:
        lang_label = {"en": "English", "ja": "Japanese"}.get(lang, lang)
        print(f"\n  {lang_label}:")
        print(f"  {'Model':<30} {'Avg WER':>8} {'Avg CER':>8} {'Avg Latency':>12} {'Avg RTF':>8} {'Lang Acc':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*8}")

        for model in models:
            model_lang = [r for r in results if r.model == model and r.language == lang]
            if not model_lang:
                continue
            avg_wer = sum(r.wer for r in model_lang) / len(model_lang)
            avg_cer = sum(r.cer for r in model_lang) / len(model_lang)
            avg_lat = sum(r.latency_ms for r in model_lang) / len(model_lang)
            avg_rtf = sum(r.rtf for r in model_lang) / len(model_lang)
            lang_acc = sum(1 for r in model_lang if r.detected_language == lang) / len(model_lang)

            print(
                f"  {model:<30} {avg_wer:>7.1%} {avg_cer:>7.1%} {avg_lat:>10.0f}ms {avg_rtf:>7.2f} {lang_acc:>7.0%}"
            )


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [
            {
                "model": r.model,
                "sample": r.sample,
                "language": r.language,
                "reference": r.reference,
                "hypothesis": r.hypothesis,
                "wer": round(r.wer, 4),
                "cer": round(r.cer, 4),
                "latency_ms": round(r.latency_ms, 1),
                "detected_language": r.detected_language,
                "audio_duration_s": round(r.audio_duration_s, 2),
                "rtf": round(r.rtf, 4),
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {output_path}")


# ── Sample Generation from Meetings ─────────────────────────


def generate_samples_from_meetings(meetings_dir: Path, max_samples: int = 5) -> None:
    """Extract short audio clips from past meetings to use as benchmark samples.

    Reads journal.jsonl to find segments with known text, extracts the
    corresponding audio slice from recording.pcm.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for meeting_dir in sorted(meetings_dir.iterdir()):
        if not meeting_dir.is_dir():
            continue
        journal = meeting_dir / "journal.jsonl"
        pcm = meeting_dir / "audio" / "recording.pcm"
        if not journal.exists() or not pcm.exists():
            continue

        # Load full audio
        raw = np.frombuffer(pcm.read_bytes(), dtype=np.int16)
        audio = raw.astype(np.float32) / 32768.0
        # Parse journal for final segments with text
        for line in journal.read_text().splitlines():
            if len(manifest) >= max_samples:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not event.get("is_final") or not event.get("text"):
                continue

            text = event["text"].strip()
            lang = event.get("language", "unknown")
            if lang not in ("en", "ja"):
                continue
            if len(text) < 10:
                continue

            start_ms = event.get("start_ms", 0)
            end_ms = event.get("end_ms", start_ms + 5000)
            start_sample = int(start_ms * SAMPLE_RATE / 1000)
            end_sample = min(int(end_ms * SAMPLE_RATE / 1000), len(audio))

            if end_sample - start_sample < SAMPLE_RATE:  # at least 1s
                continue

            segment_audio = audio[start_sample:end_sample]
            mid = meeting_dir.name
            name = f"{mid}_{lang}_{start_ms}"
            filename = f"{name}.wav"
            filepath = SAMPLES_DIR / filename

            sf.write(str(filepath), segment_audio, SAMPLE_RATE)
            manifest.append(
                {
                    "name": name,
                    "language": lang,
                    "file": filename,
                    "reference": text,
                    "meeting_id": meeting_dir.name,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            )
            print(f"  Extracted: {name} ({lang}, {(end_ms-start_ms)/1000:.1f}s): {text[:60]}...")

        if len(manifest) >= max_samples:
            break

    manifest_path = SAMPLES_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nGenerated {len(manifest)} samples → {manifest_path}")


# ── Main ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ASR Model Benchmark Suite")
    parser.add_argument("--model", choices=["qwen3-asr", "faster-whisper", "all"], default="all")
    parser.add_argument("--generate-samples", action="store_true", help="Generate samples from meetings")
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples to generate")
    parser.add_argument("--asr-url", default="http://localhost:8003", help="Qwen3-ASR vLLM URL")
    parser.add_argument("--whisper-model", default="large-v3-turbo", help="Whisper model size")
    args = parser.parse_args()

    meetings_dir = Path(__file__).parent.parent / "meetings"

    if args.generate_samples:
        print("Generating benchmark samples from past meetings...")
        generate_samples_from_meetings(meetings_dir, max_samples=args.max_samples)
        return

    # Load samples
    samples = load_builtin_samples()
    if not samples:
        print("No benchmark samples found. Run with --generate-samples first:")
        print(f"  python {__file__} --generate-samples")
        return

    print(f"Loaded {len(samples)} benchmark samples")

    # Select models
    models: list[ASRModel] = []
    if args.model in ("qwen3-asr", "all"):
        models.append(Qwen3ASRModel(base_url=args.asr_url))
    if args.model in ("faster-whisper", "all"):
        models.append(FasterWhisperModel(model_size=args.whisper_model))

    if not models:
        print("No models selected")
        return

    # Run benchmark
    results = run_benchmark(models, samples)

    # Summary
    print_summary(results)

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
