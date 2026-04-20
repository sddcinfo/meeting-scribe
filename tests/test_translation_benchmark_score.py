"""Offline scoring test for benchmarks/translation_benchmark.py.

Phase 1 prereq #4 of the Qwen3.6-FP8 migration plan needs a
``--score-only`` mode that reads two raw JSONL dumps (from the
``--no-score`` pass) and emits an aggregate Markdown report without
leaking source/target strings.  This test pins that contract end-to-end
without needing a live vLLM endpoint or the optional sacreBLEU/COMET
deps — we use deliberately trivial inputs + assert on the aggregate
output structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from translation_benchmark import (
    _latency_percentiles,
    _load_corpus,
    _score_pair,
    score_runs,
)


def _write_runs(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")


class TestLatencyPercentiles:
    """Trivial unit — but the gate depends on these being right."""

    def test_empty_list_returns_zeros(self):
        assert _latency_percentiles([]) == {"p50": 0.0, "p90": 0.0, "p99": 0.0}

    def test_single_value_gives_same_everywhere(self):
        out = _latency_percentiles([42.0])
        assert out == {"p50": 42.0, "p90": 42.0, "p99": 42.0}

    def test_ordering_is_stable(self):
        out = _latency_percentiles([10.0, 100.0, 200.0, 500.0, 1000.0])
        # Sorted: [10, 100, 200, 500, 1000]; indices 50%/90%/99% of 5.
        assert out["p50"] == 200.0
        assert out["p90"] >= out["p50"]
        assert out["p99"] >= out["p90"]


class TestScorePairNoOptionalDeps:
    """Aggregate generation must not require sacreBLEU/COMET installed."""

    def test_latency_and_counts_without_optional_deps(self):
        runs = [
            {
                "corpus_id": "r1",
                "source_lang": "ja",
                "target_lang": "en",
                "source_text": "テスト1",
                "reference_text": "test one",
                "translated": "test one",
                "latency_ms": 120.0,
            },
            {
                "corpus_id": "r2",
                "source_lang": "ja",
                "target_lang": "en",
                "source_text": "テスト2",
                "reference_text": "test two",
                "translated": "test two",
                "latency_ms": 200.0,
            },
            {
                "corpus_id": "r3",
                "source_lang": "en",
                "target_lang": "fr",
                "source_text": "hello",
                # No reference — unreferenced pair.
                "translated": "bonjour",
                "latency_ms": 150.0,
            },
        ]
        scored = _score_pair(runs)
        assert scored["n_referenced"] == 2
        assert scored["n_unreferenced"] == 1
        # Latency percentiles land regardless of optional deps.
        assert scored["latency_ms"]["p50"] in (120.0, 150.0, 200.0)
        # bleu/comet/backtranslation dicts are empty (deps absent on CI) —
        # the gate is "doesn't crash", not "produces specific numbers".
        assert isinstance(scored["bleu_by_target"], dict)
        assert isinstance(scored["comet_by_target"], dict)
        assert isinstance(scored["backtranslation_bleu_by_source"], dict)


class TestScoreRunsReportContract:
    """The committed report has NO source/target strings — privacy gate."""

    def test_report_contains_no_raw_text(self, tmp_path: Path):
        runs_a = [
            {
                "corpus_id": "priv1",
                "source_lang": "ja",
                "target_lang": "en",
                "source_text": "SECRET_JAPANESE_TEXT_ABC",
                "reference_text": "SECRET_ENGLISH_REFERENCE_XYZ",
                "translated": "SECRET_OUTPUT_A_123",
                "latency_ms": 100.0,
            },
        ]
        runs_b = [
            {
                "corpus_id": "priv1",
                "source_lang": "ja",
                "target_lang": "en",
                "source_text": "SECRET_JAPANESE_TEXT_ABC",
                "reference_text": "SECRET_ENGLISH_REFERENCE_XYZ",
                "translated": "SECRET_OUTPUT_B_456",
                "latency_ms": 110.0,
            },
        ]
        a_path = tmp_path / "a.jsonl"
        b_path = tmp_path / "b.jsonl"
        report_path = tmp_path / "report.md"
        _write_runs(a_path, runs_a)
        _write_runs(b_path, runs_b)

        score_runs(str(a_path), str(b_path), str(report_path), label_a="3.5", label_b="3.6")

        body = report_path.read_text()
        # None of the privacy-sensitive strings must leak into the report —
        # Phase 3 shadow-replay privacy guardrails require aggregates only.
        assert "SECRET_JAPANESE_TEXT_ABC" not in body
        assert "SECRET_ENGLISH_REFERENCE_XYZ" not in body
        assert "SECRET_OUTPUT_A_123" not in body
        assert "SECRET_OUTPUT_B_456" not in body
        assert "priv1" not in body  # corpus_id isn't leaked either

        # Structural checks — the report is still useful without optional deps.
        assert "Translation A/B" in body
        assert "3.5" in body and "3.6" in body
        assert "## Latency" in body

    def test_multilingual_corpus_loads_cleanly(self):
        """Corpus file is valid JSONL with the expected schema."""
        corpus_path = BENCH_DIR / "corpus" / "multilingual_v1.jsonl"
        records = _load_corpus(str(corpus_path), direction="ja_to_en")
        # Spot-check the loader + corpus contents.
        assert len(records) >= 25  # at least 5 pairs × 5 directions
        for rec in records:
            assert rec["source_lang"] in {"en", "zh", "ko", "fr", "es", "de", "ja"}
            assert rec["target_lang"] in {"en", "zh", "ko", "fr", "es", "de", "ja"}
            assert rec["source_text"]  # non-empty
            assert rec["source_lang"] != rec["target_lang"]
        # The multilingual corpus has NO reference_text by design —
        # those pairs use back-translation agreement as their gate.
        assert all("reference_text" not in r for r in records)


class TestLoadCorpus:
    """Corpus loader picks the right source + flips direction correctly."""

    def test_legacy_corpus_ja_to_en(self):
        records = _load_corpus(None, "ja_to_en")
        assert all(r["source_lang"] == "ja" for r in records)
        assert all(r["target_lang"] == "en" for r in records)
        assert all(r["source_text"] for r in records)
        # Legacy corpus has hardcoded references for sacreBLEU.
        assert all(r["reference_text"] for r in records)

    def test_legacy_corpus_en_to_ja(self):
        records = _load_corpus(None, "en_to_ja")
        assert all(r["source_lang"] == "en" for r in records)
        assert all(r["target_lang"] == "ja" for r in records)

    def test_external_corpus_wins_over_direction(self, tmp_path: Path):
        corpus = tmp_path / "c.jsonl"
        corpus.write_text(
            '{"source_lang": "fr", "target_lang": "de", "source_text": "Bonjour"}\n'
            '# comment line\n'
            '{"source_lang": "de", "target_lang": "fr", "source_text": "Hallo"}\n'
        )
        records = _load_corpus(str(corpus), direction="ja_to_en")
        assert len(records) == 2  # comment line skipped
        assert records[0]["source_lang"] == "fr"
        assert records[1]["source_lang"] == "de"
