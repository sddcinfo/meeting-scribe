#!/usr/bin/env python3
"""Translation model benchmarking — compare quality and latency across models.

Supports three modes so the Qwen3.6-FP8 migration (Phase 3 of
~/.claude/plans/sprightly-prancing-valiant.md) can run single-resident
sequential evals: dump raw outputs from each model while it's loaded,
then score offline once both dumps exist.

Modes:

  --run (default)
    Runs the corpus through the given ``--url`` and writes a single
    JSON result file with per-pair latency + char-similarity, matching
    the legacy behavior.

  --no-score --output PATH
    Runs the corpus through ``--url`` but writes raw JSONL (one record
    per pair: ``{corpus_id, source_lang, target_lang, source_text,
    reference_text, translated, latency_ms, model_id}``).  No aggregates,
    no scoring — meant to be stored for later ``--score-only``
    comparison.

  --score-only --runs-a A.jsonl --runs-b B.jsonl --report REPORT.md
    Reads two ``--no-score`` JSONL dumps, computes sacreBLEU + COMET
    on referenced pairs, back-translation agreement on unreferenced
    pairs (via ``--back-a`` / ``--back-b``), and writes an
    aggregate-only Markdown report with NO source/target strings.

  --back-translate --runs-in IN.jsonl --via-url URL --runs-out OUT.jsonl
    Second pass for unreferenced pairs: reads a ``--no-score`` dump
    and re-translates each candidate output *back* through the given
    production URL.  The resulting JSONL has the round-trip in the
    ``backtranslation`` field; ``--score-only`` compares that against
    the original source with sacreBLEU.

  --compare FILE FILE
    Legacy side-by-side of two old ``--run`` result files (latency +
    char-similarity only).  Retained for callers who don't need the
    BLEU/COMET rigor.

Corpus selection:
  --direction ja_to_en | en_to_ja   → legacy JA↔EN corpus (default)
  --corpus PATH                      → external JSONL corpus; each line
                                       is ``{source_lang, target_lang,
                                       source_text, reference_text?}``.
                                       Used for the multilingual eval
                                       (EN↔ZH/KO/FR/ES/DE) where human
                                       references don't exist.

Optional deps:
  sacrebleu + unbabel-comet for the referenced-pair metrics.  Install
  with ``pip install -e '.[bench]'`` (see pyproject).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

# Test corpus: parallel EN↔JA sentences for quality evaluation
TEST_CORPUS = [
    # Simple greetings
    {
        "ja": "今日の会議の議題について確認しましょう。",
        "en": "Let's confirm the agenda for today's meeting.",
    },
    {
        "ja": "この提案について何かご質問はありますか？",
        "en": "Do you have any questions about this proposal?",
    },
    {"ja": "次のステップを決めましょう。", "en": "Let's decide on the next steps."},
    # Business terminology
    {"ja": "四半期の売上目標を達成しました。", "en": "We achieved our quarterly sales target."},
    {
        "ja": "新しいプロジェクトの予算を承認する必要があります。",
        "en": "We need to approve the budget for the new project.",
    },
    {
        "ja": "来週のデモに向けて準備を進めています。",
        "en": "We are preparing for next week's demo.",
    },
    # Technical content
    {"ja": "APIのレスポンスタイムが改善されました。", "en": "The API response time has improved."},
    {
        "ja": "データベースのマイグレーションは明日実行します。",
        "en": "We will run the database migration tomorrow.",
    },
    {
        "ja": "本番環境にデプロイする前にテストを完了してください。",
        "en": "Please complete testing before deploying to production.",
    },
    # Conversational
    {
        "ja": "すみません、もう一度言っていただけますか？",
        "en": "Excuse me, could you say that again?",
    },
    {"ja": "その点については同意します。", "en": "I agree with that point."},
    {"ja": "詳しく説明していただけますか？", "en": "Could you explain in more detail?"},
    # Complex sentences
    {
        "ja": "この問題を解決するためには、チーム全体で協力する必要があると考えています。",
        "en": "I believe we need the whole team to cooperate to solve this problem.",
    },
    {
        "ja": "スケジュールの遅延を最小限に抑えるため、優先順位を再検討しましょう。",
        "en": "Let's re-examine priorities to minimize schedule delays.",
    },
    {
        "ja": "お客様からのフィードバックに基づいて、UIを改善する計画です。",
        "en": "We plan to improve the UI based on customer feedback.",
    },
]


def _char_similarity(reference: str, hypothesis: str) -> float:
    """Character-level Jaccard similarity — useful for CJK where word boundaries differ."""
    ref_chars = set(reference)
    hyp_chars = set(hypothesis)
    if not ref_chars and not hyp_chars:
        return 1.0
    intersection = ref_chars & hyp_chars
    union = ref_chars | hyp_chars
    return len(intersection) / len(union) if union else 0.0


async def benchmark_model(
    url: str,
    model_name: str,
    direction: str = "ja_to_en",
    output_dir: str = "benchmarks/results",
) -> dict:
    """Run the benchmark against a translation endpoint."""
    from meeting_scribe.languages import get_translation_prompt

    source_lang = "ja" if direction == "ja_to_en" else "en"
    target_lang = "en" if direction == "ja_to_en" else "ja"
    system_prompt = get_translation_prompt(source_lang, target_lang)

    latencies = []
    results = []

    # Auto-detect model name from vLLM
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            models_resp = await client.get(f"{url}/v1/models")
            model_id = models_resp.json()["data"][0]["id"]
        except Exception:
            model_id = "default"
        print(f"Using model: {model_id}\n")

        for item in TEST_CORPUS:
            source_text = item[source_lang]
            reference = item[target_lang]

            t0 = time.monotonic()
            try:
                resp = await client.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": source_text},
                        ],
                        "max_tokens": 200,
                        "temperature": 0.1,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
                # Strip thinking tags if present
                if "</think>" in raw:
                    raw = raw.split("</think>")[-1].strip()
                translated = raw
            except Exception as e:
                translated = f"ERROR: {e}"

            latency_ms = (time.monotonic() - t0) * 1000
            latencies.append(latency_ms)

            similarity = _char_similarity(reference, translated)
            results.append(
                {
                    "source": source_text,
                    "reference": reference,
                    "translated": translated,
                    "latency_ms": round(latency_ms, 1),
                    "char_similarity": round(similarity, 3),
                }
            )

            print(
                f"  [{latency_ms:.0f}ms] sim={similarity:.2f} {source_text[:30]}... → {translated[:30]}..."
            )

    # Aggregate stats
    stats = {
        "model_name": model_name,
        "url": url,
        "direction": direction,
        "corpus_size": len(TEST_CORPUS),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p90_ms": round(sorted(latencies)[int(len(latencies) * 0.9)], 1),
        "latency_p99_ms": round(max(latencies), 1),
        "latency_mean_ms": round(statistics.mean(latencies), 1),
        "avg_char_similarity": round(statistics.mean(r["char_similarity"] for r in results), 3),
        "total_time_s": round(sum(latencies) / 1000, 2),
        "throughput_sps": round(len(TEST_CORPUS) / (sum(latencies) / 1000), 2),
        "results": results,
    }

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name.replace(' ', '_')}_{direction}.json"
    out_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_file}")

    return stats


def compare_results(file_a: str, file_b: str) -> None:
    """Compare two benchmark result files side by side."""
    a = json.loads(Path(file_a).read_text())
    b = json.loads(Path(file_b).read_text())

    print(f"\n{'Metric':<25} {'Model A':>15} {'Model B':>15} {'Delta':>10}")
    print("-" * 70)

    for key in (
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_mean_ms",
        "avg_char_similarity",
        "throughput_sps",
    ):
        va = a.get(key, 0)
        vb = b.get(key, 0)
        delta = vb - va
        sign = "+" if delta > 0 else ""
        print(f"{key:<25} {va:>15.1f} {vb:>15.1f} {sign}{delta:>9.1f}")

    print(f"\n  Model A: {a['model_name']}")
    print(f"  Model B: {b['model_name']}")


# ── Shared corpus + translate primitives ────────────────────────────


def _load_corpus(corpus_path: str | None, direction: str) -> list[dict]:
    """Return the corpus as a list of ``{source_lang, target_lang,
    source_text, reference_text?}`` records.

    If ``corpus_path`` is given, loads a JSONL where each line already
    has those fields (used for the multilingual v1 corpus).  Otherwise
    falls back to the hardcoded JA↔EN ``TEST_CORPUS`` and flips it to
    match ``--direction``.
    """
    if corpus_path:
        records = []
        for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
        return records

    source_lang = "ja" if direction == "ja_to_en" else "en"
    target_lang = "en" if direction == "ja_to_en" else "ja"
    return [
        {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_text": item[source_lang],
            "reference_text": item[target_lang],
        }
        for item in TEST_CORPUS
    ]


async def _translate_one(
    client: httpx.AsyncClient,
    url: str,
    model_id: str,
    system_prompt: str,
    source_text: str,
) -> tuple[str, float]:
    """Translate one sentence; return (translated_text, latency_ms).

    Errors surface as ``ERROR: <reason>`` in the translated field so a
    bad endpoint doesn't kill the whole run.  sacreBLEU/COMET treat
    those strings as poor translations and the scored report flags the
    model instead of crashing.
    """
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": source_text},
                ],
                "max_tokens": 200,
                "temperature": 0.1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        translated = raw
    except Exception as e:
        translated = f"ERROR: {e}"
    return translated, (time.monotonic() - t0) * 1000


async def _detect_model_id(client: httpx.AsyncClient, url: str) -> str:
    try:
        resp = await client.get(f"{url}/v1/models")
        return resp.json()["data"][0]["id"]
    except Exception:
        return "default"


# ── --no-score: dump raw runs for later offline scoring ─────────────


async def dump_runs(
    url: str,
    corpus: list[dict],
    output: str,
) -> None:
    """Run the corpus and write one JSON line per pair — no aggregates.

    Each line has every field sacreBLEU/COMET/back-translation need to
    score offline: ``{corpus_id, source_lang, target_lang, source_text,
    reference_text?, translated, latency_ms, model_id}``.  Reference
    text is passed through verbatim from the corpus so ``--score-only``
    can join on ``corpus_id`` without re-loading the corpus.
    """
    from meeting_scribe.languages import get_translation_prompt

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30) as client:
        model_id = await _detect_model_id(client, url)
        print(f"Dumping runs from {url} (model={model_id}) → {out_path}")

        with out_path.open("w", encoding="utf-8") as fout:
            for i, rec in enumerate(corpus):
                system_prompt = get_translation_prompt(rec["source_lang"], rec["target_lang"])
                translated, latency_ms = await _translate_one(
                    client, url, model_id, system_prompt, rec["source_text"]
                )
                line = {
                    "corpus_id": rec.get("corpus_id", f"i{i}"),
                    "source_lang": rec["source_lang"],
                    "target_lang": rec["target_lang"],
                    "source_text": rec["source_text"],
                    "reference_text": rec.get("reference_text"),
                    "translated": translated,
                    "latency_ms": round(latency_ms, 1),
                    "model_id": model_id,
                    "url": url,
                }
                fout.write(json.dumps(line, ensure_ascii=False) + "\n")
                print(f"  [{latency_ms:.0f}ms] {rec['source_text'][:30]}... → {translated[:30]}...")

    print(f"Wrote {out_path} ({len(corpus)} records)")


# ── --back-translate: reverse pass for unreferenced pairs ───────────


async def back_translate_runs(
    runs_in: str,
    via_url: str,
    runs_out: str,
) -> None:
    """Read a raw runs JSONL, reverse-translate each candidate output
    through ``via_url`` (typically production 3.5), and write a new
    JSONL adding a ``backtranslation`` field.

    For referenced pairs the backtranslation is still captured (cheap
    extra signal), but ``--score-only`` only USES it for unreferenced
    pairs where sacreBLEU against ``reference_text`` isn't available.
    """
    from meeting_scribe.languages import get_translation_prompt

    in_records = [
        json.loads(line)
        for line in Path(runs_in).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_path = Path(runs_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30) as client:
        model_id = await _detect_model_id(client, via_url)
        print(f"Back-translating via {via_url} (model={model_id}) → {out_path}")

        with out_path.open("w", encoding="utf-8") as fout:
            for rec in in_records:
                # Reverse direction: tgt → src.
                prompt = get_translation_prompt(rec["target_lang"], rec["source_lang"])
                back, latency_ms = await _translate_one(
                    client, via_url, model_id, prompt, rec["translated"]
                )
                rec["backtranslation"] = back
                rec["backtranslation_latency_ms"] = round(latency_ms, 1)
                rec["backtranslation_model_id"] = model_id
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(in_records)} records)")


# ── --score-only: offline scoring from saved JSONL dumps ────────────


def _load_runs(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latency_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    s = sorted(values)
    n = len(s)
    return {
        "p50": round(s[int(n * 0.50)], 1),
        "p90": round(s[min(n - 1, int(n * 0.90))], 1),
        "p99": round(s[min(n - 1, int(n * 0.99))], 1),
    }


def _sacrebleu_corpus(hyp: list[str], ref: list[str], target_lang: str) -> float | None:
    try:
        import sacrebleu
    except ImportError:
        return None
    # JA/ZH/KO need character tokenization for meaningful BLEU.
    tokenize = None
    if target_lang == "ja":
        tokenize = "ja-mecab"  # falls back to "13a" if mecab isn't present
    elif target_lang == "zh":
        tokenize = "zh"
    elif target_lang == "ko":
        tokenize = "char"
    try:
        bleu = (
            sacrebleu.corpus_bleu(hyp, [ref], tokenize=tokenize)
            if tokenize
            else sacrebleu.corpus_bleu(hyp, [ref])
        )
        return round(bleu.score, 2)
    except Exception as exc:
        print(f"sacreBLEU failed for target_lang={target_lang}: {exc}")
        return None


def _comet_score(
    sources: list[str],
    hyps: list[str],
    refs: list[str],
) -> float | None:
    """Mean COMET (wmt22-comet-da) across sentences; None if COMET unavailable.

    Loaded lazily — COMET downloads ~500MB on first use.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        return None
    try:
        ckpt = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(ckpt)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs, strict=False)]
        # gpus=0 keeps the score from blocking the GPU the live backend holds.
        out = model.predict(data, batch_size=8, gpus=0)
        return round(float(out["system_score"]), 4)
    except Exception as exc:
        print(f"COMET scoring failed: {exc}")
        return None


def _score_pair(runs: list[dict]) -> dict:
    """Per-language-pair aggregate over one model's runs."""
    referenced = [r for r in runs if r.get("reference_text")]
    unreferenced = [r for r in runs if not r.get("reference_text")]
    latencies = [r["latency_ms"] for r in runs if "latency_ms" in r]

    # Bucket referenced by target_lang so sacreBLEU uses the right tokenizer.
    from collections import defaultdict

    bleu_scores: dict[str, float] = {}
    comet_scores: dict[str, float] = {}
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in referenced:
        by_lang[r["target_lang"]].append(r)
    for tgt, pairs in by_lang.items():
        hyp = [p["translated"] for p in pairs]
        ref = [p["reference_text"] for p in pairs]
        src = [p["source_text"] for p in pairs]
        bleu = _sacrebleu_corpus(hyp, ref, tgt)
        if bleu is not None:
            bleu_scores[tgt] = bleu
        comet = _comet_score(src, hyp, ref)
        if comet is not None:
            comet_scores[tgt] = comet

    # Back-translation agreement for unreferenced pairs.
    # Score: sacreBLEU of backtranslation against source_text, per source_lang.
    bt_scores: dict[str, float] = {}
    bt_by_src: dict[str, list[dict]] = defaultdict(list)
    for r in unreferenced:
        if r.get("backtranslation"):
            bt_by_src[r["source_lang"]].append(r)
    for src_lang, pairs in bt_by_src.items():
        back = [p["backtranslation"] for p in pairs]
        src = [p["source_text"] for p in pairs]
        score = _sacrebleu_corpus(back, src, src_lang)
        if score is not None:
            bt_scores[src_lang] = score

    return {
        "n_referenced": len(referenced),
        "n_unreferenced": len(unreferenced),
        "bleu_by_target": bleu_scores,
        "comet_by_target": comet_scores,
        "backtranslation_bleu_by_source": bt_scores,
        "latency_ms": _latency_percentiles(latencies),
    }


def score_runs(
    runs_a: str,
    runs_b: str,
    report_path: str,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Offline compare two dump files and write an aggregate Markdown report.

    Only aggregate numbers land in the report — no source/target
    strings, no IDs.  This is the privacy contract in Phase 3's
    shadow-replay guardrails.
    """
    records_a = _load_runs(runs_a)
    records_b = _load_runs(runs_b)

    scored_a = _score_pair(records_a)
    scored_b = _score_pair(records_b)

    def _delta(bv: float | None, av: float | None) -> str:
        if bv is None or av is None:
            return "—"
        d = bv - av
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f}"

    lines: list[str] = []
    lines.append(f"# Translation A/B — {label_a} vs {label_b}")
    lines.append("")
    lines.append(
        f"- {label_a}: {len(records_a)} records "
        f"({scored_a['n_referenced']} referenced, {scored_a['n_unreferenced']} unreferenced)"
    )
    lines.append(
        f"- {label_b}: {len(records_b)} records "
        f"({scored_b['n_referenced']} referenced, {scored_b['n_unreferenced']} unreferenced)"
    )
    lines.append("")
    lines.append("## Latency")
    lines.append("| percentile | " + label_a + " ms | " + label_b + " ms | delta ms |")
    lines.append("|---|---:|---:|---:|")
    for p in ("p50", "p90", "p99"):
        lines.append(
            f"| {p} | {scored_a['latency_ms'][p]} | {scored_b['latency_ms'][p]} "
            f"| {_delta(scored_b['latency_ms'][p], scored_a['latency_ms'][p])} |"
        )
    lines.append("")

    langs = sorted(set(scored_a["bleu_by_target"]) | set(scored_b["bleu_by_target"]))
    if langs:
        lines.append("## sacreBLEU (referenced pairs)")
        lines.append("| target | " + label_a + " BLEU | " + label_b + " BLEU | delta |")
        lines.append("|---|---:|---:|---:|")
        for lang in langs:
            lines.append(
                f"| {lang} | {scored_a['bleu_by_target'].get(lang, '—')} "
                f"| {scored_b['bleu_by_target'].get(lang, '—')} "
                f"| {_delta(scored_b['bleu_by_target'].get(lang), scored_a['bleu_by_target'].get(lang))} |"
            )
        lines.append("")

    langs_c = sorted(set(scored_a["comet_by_target"]) | set(scored_b["comet_by_target"]))
    if langs_c:
        lines.append("## COMET (wmt22-comet-da, referenced pairs)")
        lines.append("| target | " + label_a + " COMET | " + label_b + " COMET | delta |")
        lines.append("|---|---:|---:|---:|")
        for lang in langs_c:
            lines.append(
                f"| {lang} | {scored_a['comet_by_target'].get(lang, '—')} "
                f"| {scored_b['comet_by_target'].get(lang, '—')} "
                f"| {_delta(scored_b['comet_by_target'].get(lang), scored_a['comet_by_target'].get(lang))} |"
            )
        lines.append("")

    bt_langs = sorted(
        set(scored_a["backtranslation_bleu_by_source"])
        | set(scored_b["backtranslation_bleu_by_source"])
    )
    if bt_langs:
        lines.append("## Back-translation agreement (unreferenced pairs)")
        lines.append(
            "Round-trip src→tgt (candidate) → tgt→src (production); "
            "sacreBLEU against the original source.  Higher = candidate "
            "preserved meaning better."
        )
        lines.append("")
        lines.append("| source | " + label_a + " BLEU | " + label_b + " BLEU | delta |")
        lines.append("|---|---:|---:|---:|")
        for lang in bt_langs:
            lines.append(
                f"| {lang} | {scored_a['backtranslation_bleu_by_source'].get(lang, '—')} "
                f"| {scored_b['backtranslation_bleu_by_source'].get(lang, '—')} "
                f"| {_delta(scored_b['backtranslation_bleu_by_source'].get(lang), scored_a['backtranslation_bleu_by_source'].get(lang))} |"
            )
        lines.append("")

    if not any(
        s
        for s in (
            scored_a["bleu_by_target"],
            scored_b["bleu_by_target"],
            scored_a["comet_by_target"],
            scored_b["comet_by_target"],
        )
    ):
        lines.append(
            "> NOTE: sacreBLEU/COMET are unavailable — install `sacrebleu` + `unbabel-comet`"
        )
        lines.append("> via ``pip install -e '.[bench]'`` to enable them.")
        lines.append("")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Translation model benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", default="http://localhost:8010", help="vLLM endpoint URL")
    parser.add_argument(
        "--model-name", default="default", help="Label for this model run (legacy --run mode)"
    )
    parser.add_argument("--direction", default="ja_to_en", choices=["ja_to_en", "en_to_ja"])
    parser.add_argument("--corpus", help="External JSONL corpus path (overrides --direction)")
    parser.add_argument(
        "--compare", nargs=2, metavar="FILE", help="Compare two legacy result files"
    )

    # Phase 3 split-mode A/B:
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Dump raw runs without scoring (for later --score-only)",
    )
    parser.add_argument("--output", help="Output path for --no-score JSONL dump")

    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score two existing --no-score dumps offline; requires --runs-a --runs-b --report",
    )
    parser.add_argument("--runs-a", help="First model's --no-score dump")
    parser.add_argument("--runs-b", help="Second model's --no-score dump")
    parser.add_argument("--label-a", default="A", help="Label for the first model in the report")
    parser.add_argument("--label-b", default="B", help="Label for the second model in the report")
    parser.add_argument("--report", help="Output markdown report path for --score-only")

    parser.add_argument(
        "--back-translate",
        action="store_true",
        help="Reverse-translate a --no-score dump through --via-url; requires --runs-in --runs-out",
    )
    parser.add_argument("--runs-in", help="Source dump for --back-translate")
    parser.add_argument("--via-url", help="Reverse-translation endpoint (production URL)")
    parser.add_argument("--runs-out", help="Output dump with backtranslation field")

    args = parser.parse_args()

    # ── Mode dispatch — mutually exclusive ─────────────────────

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    if args.score_only:
        if not (args.runs_a and args.runs_b and args.report):
            parser.error("--score-only requires --runs-a, --runs-b, and --report")
        score_runs(args.runs_a, args.runs_b, args.report, args.label_a, args.label_b)
        return

    if args.back_translate:
        if not (args.runs_in and args.via_url and args.runs_out):
            parser.error("--back-translate requires --runs-in, --via-url, and --runs-out")
        await back_translate_runs(args.runs_in, args.via_url, args.runs_out)
        return

    if args.no_score:
        if not args.output:
            parser.error("--no-score requires --output")
        corpus = _load_corpus(args.corpus, args.direction)
        await dump_runs(args.url, corpus, args.output)
        return

    # Legacy --run path: one-shot benchmark + scored JSON result.
    print(f"Benchmarking {args.model_name} at {args.url} ({args.direction})")
    print(f"Corpus: {len(TEST_CORPUS)} sentence pairs\n")

    stats = await benchmark_model(args.url, args.model_name, args.direction)

    print(f"\n{'=' * 50}")
    print(f"Model:        {stats['model_name']}")
    print(f"Latency p50:  {stats['latency_p50_ms']}ms")
    print(f"Latency p90:  {stats['latency_p90_ms']}ms")
    print(f"Similarity:   {stats['avg_char_similarity']:.1%}")
    print(f"Throughput:    {stats['throughput_sps']:.1f} sentences/s")


if __name__ == "__main__":
    asyncio.run(main())
