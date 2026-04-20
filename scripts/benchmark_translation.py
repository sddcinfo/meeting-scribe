#!/usr/bin/env python3
"""Translation Model Benchmarking Suite.

Compares translation models served by vLLM, measuring:
- Per-segment latency
- Total throughput (segments/sec, chars/sec)
- Optionally compares two endpoints side-by-side

Loads real segments from meeting journals as test data.

Usage:
    python scripts/benchmark_translation.py
    python scripts/benchmark_translation.py --url http://localhost:8010
    python scripts/benchmark_translation.py --url http://localhost:8010 --url2 http://localhost:8001
    python scripts/benchmark_translation.py --max-samples 10

Requirements:
    - httpx (pip install httpx)
    - meeting_scribe package (for get_translation_prompt)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from meeting_scribe.languages import get_translation_prompt

# ── Constants ──────────────────────────────────────────────

MEETINGS_DIR = Path(__file__).parent.parent / "meetings"
RESULTS_DIR = Path(__file__).parent.parent / "benchmarks" / "results"

# Meeting IDs for sourcing test segments
EN_MEETING = "f38d5807-bbdf-4c5c-96fb-cb8267e55ed0"
JA_MEETING = "c98490ec-6aff-45a0-88f5-bf020ec35016"

# Translation directions
EN_TO_JA = ("en", "ja")
JA_TO_EN = ("ja", "en")


# ── Data Structures ────────────────────────────────────────


@dataclass
class Segment:
    """A text segment from a meeting journal."""

    segment_id: str
    language: str
    text: str
    meeting_id: str


@dataclass
class TranslationResult:
    """Result of a single translation request."""

    segment: Segment
    source_lang: str
    target_lang: str
    translated_text: str
    latency_ms: float
    input_chars: int
    output_chars: int
    error: str | None = None


@dataclass
class EndpointReport:
    """Aggregated stats for one endpoint."""

    url: str
    model_name: str
    results: list[TranslationResult] = field(default_factory=list)

    @property
    def successful(self) -> list[TranslationResult]:
        return [r for r in self.results if r.error is None]

    @property
    def errors(self) -> list[TranslationResult]:
        return [r for r in self.results if r.error is not None]

    def stats_for(self, direction: str) -> dict:
        """Compute stats for a given direction like 'en->ja'."""
        src, tgt = direction.split("->")
        subset = [r for r in self.successful if r.source_lang == src and r.target_lang == tgt]
        if not subset:
            return {}
        latencies = [r.latency_ms for r in subset]
        input_chars = sum(r.input_chars for r in subset)
        output_chars = sum(r.output_chars for r in subset)
        total_time_s = sum(latencies) / 1000
        return {
            "count": len(subset),
            "latency_mean_ms": statistics.mean(latencies),
            "latency_median_ms": statistics.median(latencies),
            "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[-1],
            "latency_min_ms": min(latencies),
            "latency_max_ms": max(latencies),
            "throughput_seg_per_s": len(subset) / total_time_s if total_time_s > 0 else 0,
            "throughput_input_chars_per_s": input_chars / total_time_s if total_time_s > 0 else 0,
            "throughput_output_chars_per_s": output_chars / total_time_s if total_time_s > 0 else 0,
            "total_input_chars": input_chars,
            "total_output_chars": output_chars,
            "total_time_s": total_time_s,
        }


# ── Segment Loading ────────────────────────────────────────


def load_segments(meeting_id: str, language: str, max_samples: int) -> list[Segment]:
    """Load segments from a meeting journal, filtering by language.

    Skips very short segments (< 10 chars) to focus on meaningful text.
    """
    journal_path = MEETINGS_DIR / meeting_id / "journal.jsonl"
    if not journal_path.exists():
        print(f"  WARNING: Journal not found: {journal_path}")
        return []

    segments = []
    for line in journal_path.read_text().splitlines():
        if len(segments) >= max_samples:
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not event.get("is_final"):
            continue
        text = (event.get("text") or "").strip()
        lang = event.get("language", "")
        if lang != language or len(text) < 10:
            continue

        segments.append(
            Segment(
                segment_id=event.get("segment_id", ""),
                language=lang,
                text=text,
                meeting_id=meeting_id,
            )
        )

    return segments


def load_test_segments(max_samples: int) -> list[Segment]:
    """Load EN and JA segments from the designated meetings."""
    print(f"Loading segments (max {max_samples} per language)...")

    en_segments = load_segments(EN_MEETING, "en", max_samples)
    print(f"  EN segments: {len(en_segments)} from meeting {EN_MEETING}")

    ja_segments = load_segments(JA_MEETING, "ja", max_samples)
    print(f"  JA segments: {len(ja_segments)} from meeting {JA_MEETING}")

    return en_segments + ja_segments


# ── vLLM Client ────────────────────────────────────────────


def detect_model(base_url: str) -> str:
    """Auto-detect the model served by vLLM."""
    resp = httpx.get(f"{base_url}/v1/models", timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    if not models:
        raise RuntimeError(f"No models available at {base_url}")
    return models[0]["id"]


def translate_segment(
    client: httpx.Client,
    base_url: str,
    model: str,
    segment: Segment,
    source_lang: str,
    target_lang: str,
) -> TranslationResult:
    """Send a single translation request and measure latency."""
    system_prompt = get_translation_prompt(source_lang, target_lang)

    t0 = time.monotonic()
    try:
        resp = client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": segment.text},
                ],
                "temperature": 0.3,
                "max_tokens": 1024,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        translated = data["choices"][0]["message"]["content"].strip()
        latency_ms = (time.monotonic() - t0) * 1000

        return TranslationResult(
            segment=segment,
            source_lang=source_lang,
            target_lang=target_lang,
            translated_text=translated,
            latency_ms=latency_ms,
            input_chars=len(segment.text),
            output_chars=len(translated),
        )

    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return TranslationResult(
            segment=segment,
            source_lang=source_lang,
            target_lang=target_lang,
            translated_text="",
            latency_ms=latency_ms,
            input_chars=len(segment.text),
            output_chars=0,
            error=str(e),
        )


# ── Benchmark Runner ───────────────────────────────────────


def run_endpoint(base_url: str, segments: list[Segment]) -> EndpointReport:
    """Run all segments through a single vLLM endpoint."""
    base_url = base_url.rstrip("/")

    print(f"\nConnecting to {base_url}...")
    model_name = detect_model(base_url)
    print(f"  Model: {model_name}")

    report = EndpointReport(url=base_url, model_name=model_name)

    with httpx.Client(timeout=60.0) as client:
        total = len(segments)
        for i, seg in enumerate(segments, 1):
            # Determine translation direction
            if seg.language == "en":
                src, tgt = EN_TO_JA
            else:
                src, tgt = JA_TO_EN

            result = translate_segment(client, base_url, model_name, seg, src, tgt)
            report.results.append(result)

            # Progress output
            status = "OK" if result.error is None else f"ERR: {result.error[:40]}"
            preview = seg.text[:40].replace("\n", " ")
            print(
                f"  [{i:3d}/{total}] {src}->{tgt} {result.latency_ms:7.0f}ms "
                f"({result.input_chars:3d}c->{result.output_chars:3d}c) "
                f"{status}  \"{preview}...\""
            )

    return report


# ── Reporting ──────────────────────────────────────────────


def print_report(report: EndpointReport, label: str = "") -> None:
    """Print stats for a single endpoint."""
    header = f"{label}: {report.model_name}" if label else report.model_name
    print(f"\n{'='*70}")
    print(f"  {header}")
    print(f"  URL: {report.url}")
    print(f"  Total: {len(report.results)} segments, "
          f"{len(report.successful)} OK, {len(report.errors)} errors")
    print(f"{'='*70}")

    for direction in ("en->ja", "ja->en"):
        stats = report.stats_for(direction)
        if not stats:
            continue
        print(f"\n  {direction.upper()} ({stats['count']} segments):")
        print(f"    Latency  mean:   {stats['latency_mean_ms']:8.0f} ms")
        print(f"    Latency  median: {stats['latency_median_ms']:8.0f} ms")
        print(f"    Latency  p95:    {stats['latency_p95_ms']:8.0f} ms")
        print(f"    Latency  min:    {stats['latency_min_ms']:8.0f} ms")
        print(f"    Latency  max:    {stats['latency_max_ms']:8.0f} ms")
        print(f"    Throughput:      {stats['throughput_seg_per_s']:8.2f} seg/s")
        print(f"    Input chars/s:   {stats['throughput_input_chars_per_s']:8.0f}")
        print(f"    Output chars/s:  {stats['throughput_output_chars_per_s']:8.0f}")
        print(f"    Total time:      {stats['total_time_s']:8.1f} s")


def print_comparison(report_a: EndpointReport, report_b: EndpointReport) -> None:
    """Print a side-by-side comparison of two endpoints."""
    print(f"\n{'='*70}")
    print("  COMPARISON")
    print(f"{'='*70}")
    print(f"  A: {report_a.model_name} ({report_a.url})")
    print(f"  B: {report_b.model_name} ({report_b.url})")

    for direction in ("en->ja", "ja->en"):
        stats_a = report_a.stats_for(direction)
        stats_b = report_b.stats_for(direction)
        if not stats_a or not stats_b:
            continue

        print(f"\n  {direction.upper()}:")
        print(f"    {'Metric':<25} {'A':>12} {'B':>12} {'Delta':>12}")
        print(f"    {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

        comparisons = [
            ("Latency mean (ms)", "latency_mean_ms", True),
            ("Latency median (ms)", "latency_median_ms", True),
            ("Latency p95 (ms)", "latency_p95_ms", True),
            ("Throughput (seg/s)", "throughput_seg_per_s", False),
            ("Input chars/s", "throughput_input_chars_per_s", False),
            ("Output chars/s", "throughput_output_chars_per_s", False),
        ]

        for label, key, lower_is_better in comparisons:
            va = stats_a[key]
            vb = stats_b[key]
            pct = ((vb - va) / va) * 100 if va > 0 else 0.0

            # Determine which is better
            if lower_is_better:
                winner = "A" if va < vb else "B" if vb < va else "="
            else:
                winner = "A" if va > vb else "B" if vb > va else "="

            delta_str = f"{pct:+.1f}% ({winner})"
            print(f"    {label:<25} {va:>12.1f} {vb:>12.1f} {delta_str:>12}")

    # Sample translations side by side
    print("\n  SAMPLE TRANSLATIONS (first 3 per direction):")
    for direction in ("en->ja", "ja->en"):
        src, tgt = direction.split("->")
        results_a = [r for r in report_a.successful if r.source_lang == src and r.target_lang == tgt]
        results_b = [r for r in report_b.successful if r.source_lang == src and r.target_lang == tgt]

        print(f"\n    {direction.upper()}:")
        for i in range(min(3, len(results_a), len(results_b))):
            ra, rb = results_a[i], results_b[i]
            print(f"      Source:  {ra.segment.text[:70]}")
            print(f"      A ({ra.latency_ms:.0f}ms): {ra.translated_text[:70]}")
            print(f"      B ({rb.latency_ms:.0f}ms): {rb.translated_text[:70]}")
            print()


def save_results(reports: list[EndpointReport], output_path: Path) -> None:
    """Save benchmark results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoints": [],
    }
    for report in reports:
        endpoint_data = {
            "url": report.url,
            "model_name": report.model_name,
            "stats": {},
            "results": [],
        }
        for direction in ("en->ja", "ja->en"):
            stats = report.stats_for(direction)
            if stats:
                endpoint_data["stats"][direction] = {
                    k: round(v, 2) if isinstance(v, float) else v
                    for k, v in stats.items()
                }
        for r in report.results:
            endpoint_data["results"].append({
                "segment_id": r.segment.segment_id,
                "source_lang": r.source_lang,
                "target_lang": r.target_lang,
                "source_text": r.segment.text,
                "translated_text": r.translated_text,
                "latency_ms": round(r.latency_ms, 1),
                "input_chars": r.input_chars,
                "output_chars": r.output_chars,
                "error": r.error,
            })
        data["endpoints"].append(endpoint_data)

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {output_path}")


# ── Main ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Translation Model Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                   # Single endpoint benchmark
  %(prog)s --url http://localhost:8010        # Explicit URL
  %(prog)s --url http://gpu1:8000 --url2 http://gpu1:8001  # Compare two models
  %(prog)s --max-samples 10                  # Quick run with fewer samples
        """,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8010",
        help="vLLM translation endpoint URL (default: http://localhost:8010)",
    )
    parser.add_argument(
        "--url2",
        default=None,
        help="Optional second vLLM URL for side-by-side comparison",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Max samples per language (default: 25)",
    )
    args = parser.parse_args()

    # Load test segments
    segments = load_test_segments(args.max_samples)
    if not segments:
        print("ERROR: No segments loaded. Check that meeting directories exist:")
        print(f"  EN: {MEETINGS_DIR / EN_MEETING}")
        print(f"  JA: {MEETINGS_DIR / JA_MEETING}")
        return

    print(f"\nTotal segments: {len(segments)}")

    # Run benchmarks
    reports: list[EndpointReport] = []

    report_a = run_endpoint(args.url, segments)
    reports.append(report_a)
    print_report(report_a, label="Endpoint A" if args.url2 else "")

    if args.url2:
        report_b = run_endpoint(args.url2, segments)
        reports.append(report_b)
        print_report(report_b, label="Endpoint B")
        print_comparison(report_a, report_b)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"translation_benchmark_{timestamp}.json"
    save_results(reports, output_path)


if __name__ == "__main__":
    main()
