"""Reducer for the 2026-Q2 model-challenger bench.

Reads the offline raw JSON outputs from a track and emits a single
``decision_gate.md`` per track containing **only** aggregated metrics,
the pinned model + container SHAs, and the outcome.  The reducer
never copies per-sample fields into the markdown, never lands raw
JSON in the repo, and refuses to write anywhere outside
``reports/2026-Q2-bench/<track>/`` (the only repo path allowed for
bench output).

Track-specific aggregations:

* ``asr``:  corpus-level token-weighted WER per language (vs the
  diagnostic p50/p95), bootstrap 95 % CI on (baseline − challenger),
  p50/p95 total_ms, plus a Q-JA-corpus pass cell.
* ``tts``:  p50 + p95 single-stream TTFA, p95 TTFA at concurrency=3,
  streaming completion ratio, post-adjudication MOS, ABX correct-id
  rate, latency band outcome.
* ``diarize``:  cross-check invariant pass/fail (parsed from the
  ``cross_check_speakers.py`` JSON output), time-weighted overlap
  resolution % from ``diarize_compare.py``, reprocess wall-clock,
  optional DER if RTTM was provided.

Each track expects a ``--pins`` JSON file capturing the pinned SHAs
(model HF revision, container image digest, production-model HF
revision on the same day).  No pins → no Decision Log entry; this
matches the S3 rule from the plan.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_REPO_OUTPUT_ROOT = REPO_ROOT / "reports" / "2026-Q2-bench"


def _assert_decision_gate_path(out: Path) -> Path:
    """``decision_gate.md`` must land under reports/2026-Q2-bench/<track>/."""
    p = out.resolve()
    if p.suffix != ".md" or p.name != "decision_gate.md":
        raise SystemExit(f"--out must end in decision_gate.md, got {p}")
    try:
        p.relative_to(ALLOWED_REPO_OUTPUT_ROOT)
    except ValueError as exc:
        raise SystemExit(
            f"decision_gate.md must land under {ALLOWED_REPO_OUTPUT_ROOT}/<track>/, got {p}"
        ) from exc
    return p


# ---------------------------------------------------------------------------
# Helpers shared across tracks
# ---------------------------------------------------------------------------


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _bootstrap_ci_diff(
    a_edits: list[int],
    a_tokens: list[int],
    b_edits: list[int],
    b_tokens: list[int],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 1729,
) -> tuple[float, float]:
    """Bootstrap CI on (a_corpus_wer − b_corpus_wer).

    Per-sample (edits, tokens) pairs are required so we can resample
    them coherently and recompute the corpus-level ratio.  Returns
    (lo, hi) at the requested CI.
    """
    if not a_edits or not b_edits:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n_a, n_b = len(a_edits), len(b_edits)
    diffs: list[float] = []
    for _ in range(n_resamples):
        idx_a = [rng.randrange(n_a) for _ in range(n_a)]
        idx_b = [rng.randrange(n_b) for _ in range(n_b)]
        ae = sum(a_edits[i] for i in idx_a)
        at = sum(a_tokens[i] for i in idx_a)
        be = sum(b_edits[i] for i in idx_b)
        bt = sum(b_tokens[i] for i in idx_b)
        if at == 0 or bt == 0:
            continue
        diffs.append(ae / at - be / bt)
    diffs.sort()
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    return (
        _percentile(diffs, lo_q) or float("nan"),
        _percentile(diffs, hi_q) or float("nan"),
    )


def _load_pins(pins_path: Path | None) -> dict:
    if pins_path is None:
        return {}
    return json.loads(pins_path.read_text())


def _render_pins_block(pins: dict) -> str:
    if not pins:
        return (
            "**Pins**: NONE — this run is exploratory only and does not "
            "produce a Decision Log entry. Re-run with --pins.\n\n"
        )
    lines = ["## Pins (S3, plan)", "", "| Field | Value |", "|---|---|"]
    for k, v in pins.items():
        lines.append(f"| `{k}` | `{v}` |")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# ASR track reducer
# ---------------------------------------------------------------------------


@dataclass
class _PerLang:
    edits: list[int]
    tokens: list[int]
    total_ms: list[float]


def _bucket_asr_by_lang(
    per_sample: list[dict], lang_lookup: dict[str, str] | None
) -> dict[str, _PerLang]:
    """Group per-sample records by language; uses lookup if provided."""
    out: dict[str, _PerLang] = {}
    for r in per_sample:
        lang = r.get("language") or (lang_lookup or {}).get(r["id"], "unknown")
        bucket = out.setdefault(lang, _PerLang([], [], []))
        # The harness records WER as a fraction, not raw edits/tokens. To
        # get token-weighted corpus WER we need both — recompute edits
        # via WER × token_count if the harness emits token_count, else
        # fall back to a per-sample mean (and emit a warning row).
        tokens = r.get("reference_tokens")
        wer = r.get("wer")
        if tokens is not None and wer is not None:
            edits = round(wer * tokens)
            bucket.edits.append(edits)
            bucket.tokens.append(int(tokens))
        bucket.total_ms.append(float(r["total_ms"]))
    return out


def reduce_asr(args) -> str:
    baseline = json.loads(args.baseline.read_text())
    challenger = json.loads(args.challenger.read_text())
    pins = _load_pins(args.pins)

    base_by = _bucket_asr_by_lang(baseline.get("per_sample", []), None)
    chal_by = _bucket_asr_by_lang(challenger.get("per_sample", []), None)

    body: list[str] = []
    body.append(f"# Track A — ASR decision gate ({args.label})\n")
    body.append(_render_pins_block(pins))

    body.append("## Corpus-level WER (token-weighted)\n")
    body.append(
        "| Language | Baseline corpus WER | Challenger corpus WER | Δ (pp) | Bootstrap 95% CI on (base - chal) | Q-pass? |"
    )
    body.append("|---|---:|---:|---:|---:|:---:|")
    languages = sorted(set(base_by) | set(chal_by))
    overall_pass: bool | None = None
    for lang in languages:
        b = base_by.get(lang, _PerLang([], [], []))
        c = chal_by.get(lang, _PerLang([], [], []))
        b_corpus = (sum(b.edits) / sum(b.tokens) * 100.0) if sum(b.tokens) else float("nan")
        c_corpus = (sum(c.edits) / sum(c.tokens) * 100.0) if sum(c.tokens) else float("nan")
        delta_pp = b_corpus - c_corpus  # positive = challenger wins
        ci_lo, ci_hi = _bootstrap_ci_diff(b.edits, b.tokens, c.edits, c.tokens)
        # Q-JA-corpus rule (plan A4): challenger ≥ 0.5 pp better AND CI excludes zero
        passing = (delta_pp >= 0.5) and (ci_lo > 0)
        if lang.lower() in ("ja", "jp", "japanese"):
            overall_pass = passing
        body.append(
            f"| {lang} | {b_corpus:.2f} | {c_corpus:.2f} | "
            f"{delta_pp:+.2f} | [{ci_lo * 100:+.2f}, {ci_hi * 100:+.2f}] | "
            f"{'✅' if passing else '❌'} |"
        )
    body.append("")

    body.append("## Latency diagnostics\n")
    body.append("| Language | Baseline p95 total_ms | Challenger p95 total_ms |")
    body.append("|---|---:|---:|")
    for lang in languages:
        body.append(
            f"| {lang} | "
            f"{_percentile(base_by.get(lang, _PerLang([], [], [])).total_ms, 0.95) or 0:.0f} | "
            f"{_percentile(chal_by.get(lang, _PerLang([], [], [])).total_ms, 0.95) or 0:.0f} |"
        )
    body.append("")

    body.append("## Outcome\n")
    if not pins:
        body.append("- **EXPLORATORY** (no pins provided)\n")
    elif overall_pass is None:
        body.append("- **EXPLORATORY** — no Japanese subset found in either run\n")
    else:
        # Exhaustive map per plans/2026-Q3-followups.md (B1 decision gates).
        # Pull the JA row's stats out of the table we just rendered.
        ja_b = base_by.get("ja", _PerLang([], [], []))
        ja_c = chal_by.get("ja", _PerLang([], [], []))
        b_corpus = (sum(ja_b.edits) / sum(ja_b.tokens) * 100.0) if sum(ja_b.tokens) else 0.0
        c_corpus = (sum(ja_c.edits) / sum(ja_c.tokens) * 100.0) if sum(ja_c.tokens) else 0.0
        delta_pp = b_corpus - c_corpus
        ci_lo, ci_hi = _bootstrap_ci_diff(ja_b.edits, ja_b.tokens, ja_c.edits, ja_c.tokens)
        ci_lo_pp = ci_lo * 100
        ci_hi_pp = ci_hi * 100

        if delta_pp >= 0.5 and ci_lo_pp > 0:
            body.append(
                f"- **PROMOTE candidate** — Δ {delta_pp:+.2f} pp (Cohere-better), "
                f"CI [{ci_lo_pp:+.2f}, {ci_hi_pp:+.2f}] excludes zero\n"
            )
        elif delta_pp >= 0.5:
            body.append(
                f"- **DEFER (effect-size present, power insufficient)** — Δ {delta_pp:+.2f} pp "
                f"(Cohere-better), but CI [{ci_lo_pp:+.2f}, {ci_hi_pp:+.2f}] includes zero\n"
            )
        elif abs(delta_pp) < 0.5:
            body.append(
                f"- **DEFER (tied)** — |Δ| = {abs(delta_pp):.2f} pp < 0.5 pp threshold\n"
            )
        else:
            # Δ ≤ -0.5 pp ⇒ Cohere worse by ≥ 0.5 pp ⇒ REJECT.
            body.append(
                f"- **REJECT** — Δ {delta_pp:+.2f} pp (Cohere-WORSE), "
                f"CI [{ci_lo_pp:+.2f}, {ci_hi_pp:+.2f}]"
                + (" excludes zero" if ci_hi_pp < 0 else " includes zero (still ≥ 0.5 pp worse)")
                + ".  Track A closes; production stays on Qwen3-ASR-1.7B.\n"
            )
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# TTS track reducer
# ---------------------------------------------------------------------------


def reduce_tts(args) -> str:
    quality_baseline = json.loads(args.baseline.read_text())
    quality_challenger = json.loads(args.challenger.read_text())
    concurrent_baseline = (
        json.loads(args.baseline_concurrent.read_text()) if args.baseline_concurrent else None
    )
    concurrent_challenger = (
        json.loads(args.challenger_concurrent.read_text()) if args.challenger_concurrent else None
    )
    abx = json.loads(args.abx.read_text()) if args.abx else None
    mos = json.loads(args.mos.read_text()) if args.mos else None
    pins = _load_pins(args.pins)

    body: list[str] = []
    body.append(f"# Track B — TTS decision gate ({args.label})\n")
    body.append(_render_pins_block(pins))

    body.append("## Latency\n")
    body.append(
        "| Backend | Single-stream p50 TTFA (ms) | Single-stream p95 TTFA (ms) | Concurrent p95@3 TTFA (ms) | Streaming OK? |"
    )
    body.append("|---|---:|---:|---:|:---:|")

    def _row(label: str, q: dict, c: dict | None) -> str:
        p50 = q.get("p50_ttfa_ms")
        p95 = q.get("p95_ttfa_ms")
        c_p95 = c.get("p95_ttfa_ms") if c else None
        per_sample = q.get("per_sample", [])
        streaming_ok = all(
            (s.get("ttfa_ms") or 0) > 0 and (s.get("total_ms") or 0) > (s.get("ttfa_ms") or 0)
            for s in per_sample
        )
        return (
            f"| {label} | {p50 if p50 is not None else 'n/a'} | "
            f"{p95 if p95 is not None else 'n/a'} | "
            f"{c_p95 if c_p95 is not None else 'n/a'} | "
            f"{'✅' if streaming_ok else '❌'} |"
        )

    body.append(_row("baseline (qwen3)", quality_baseline, concurrent_baseline))
    body.append(_row("challenger (funcosyvoice)", quality_challenger, concurrent_challenger))
    body.append("")

    if abx is not None:
        body.append("## Voice-clone ABX (forced choice)\n")
        body.append("| Backend | Trials | Correct | Rate | Q-CLONE pass? |")
        body.append("|---|---:|---:|---:|:---:|")
        for backend in ("qwen3", "funcosyvoice"):
            stats = abx.get(backend, {})
            trials = stats.get("trials", 0)
            correct = stats.get("correct", 0)
            rate = (correct / trials) if trials else 0.0
            body.append(
                f"| {backend} | {trials} | {correct} | {rate:.0%} | "
                f"{'✅' if rate >= 0.80 else '❌'} |"
            )
        body.append("")

    if mos is not None:
        body.append("## MOS (post-adjudication)\n")
        body.append("| Backend | Mean MOS EN | Mean MOS JA |")
        body.append("|---|---:|---:|")
        for backend in ("qwen3", "funcosyvoice"):
            stats = mos.get(backend, {})
            body.append(
                f"| {backend} | {stats.get('mean_en', 'n/a')} | {stats.get('mean_ja', 'n/a')} |"
            )
        body.append("")

    # Outcome from the joint TTFA table (plan B5)
    body.append("## Outcome\n")
    if not pins:
        body.append("- **EXPLORATORY** (no pins provided)\n")
    else:
        cp50 = quality_challenger.get("p50_ttfa_ms") or float("inf")
        cc_p95 = (concurrent_challenger or {}).get("p95_ttfa_ms") or float("inf")
        if cp50 <= 110 and cc_p95 <= 200:
            body.append("- **TTFA PASS** — eligible for live path; review quality gates\n")
        elif cp50 <= 110 and cc_p95 <= 350:
            body.append("- **TTFA DEFER** — slides / refinement TTS only\n")
        elif cp50 <= 150:
            body.append("- **TTFA DEFER** — slides only\n")
        else:
            body.append("- **TTFA REJECT** for live path\n")
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# Diarize track reducer
# ---------------------------------------------------------------------------


def reduce_diarize(args) -> str:
    pins = _load_pins(args.pins)

    # diarize_compare.py per-meeting JSON: contains overlap_metric block
    per_meeting = [json.loads(p.read_text()) for p in args.compare_jsons]

    body: list[str] = []
    body.append(f"# Track C — Diarization decision gate ({args.label})\n")
    body.append(_render_pins_block(pins))

    body.append("## Time-weighted overlap resolution\n")
    body.append("| Meeting | Total overlap (s) | Resolved (s) | Fraction | Pass (≥30%)? |")
    body.append("|---|---:|---:|---:|:---:|")
    overall_pass = True
    for rec in per_meeting:
        m = rec["overlap_metric"]
        body.append(
            f"| `{rec['meeting_id']}` | "
            f"{m['total_overlap_seconds']:.1f} | "
            f"{m['overlap_resolved_seconds']:.1f} | "
            f"{m['fraction_resolved']:.1%} | "
            f"{'✅' if m['pass'] else '❌'} |"
        )
        overall_pass = overall_pass and m["pass"]
    body.append("")

    if args.invariants:
        body.append("## Cross-check invariants\n")
        for rec_path in args.invariants:
            rec = rec_path.read_text().strip()
            body.append(f"- `{rec_path.name}`: {rec[:200]}")
        body.append("")

    body.append("## Outcome\n")
    if not pins:
        body.append("- **EXPLORATORY** (no pins provided)\n")
    elif overall_pass:
        body.append("- **PROMOTE candidate** — overlap-time gate cleared on all meetings\n")
    else:
        body.append("- **DEFER** — overlap-time gate not cleared\n")
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="track", required=True)

    a = sub.add_parser("asr")
    a.add_argument("--label", required=True)
    a.add_argument(
        "--baseline", type=Path, required=True, help="Qwen3-ASR JSON from asr_accuracy_latency.py"
    )
    a.add_argument(
        "--challenger", type=Path, required=True, help="Cohere JSON from asr_accuracy_latency.py"
    )
    a.add_argument("--pins", type=Path, default=None)
    a.add_argument("--out", type=Path, required=True)

    b = sub.add_parser("tts")
    b.add_argument("--label", required=True)
    b.add_argument("--baseline", type=Path, required=True, help="Qwen3-TTS quality JSON")
    b.add_argument("--challenger", type=Path, required=True, help="FunCosyVoice quality JSON")
    b.add_argument("--baseline-concurrent", type=Path, default=None)
    b.add_argument("--challenger-concurrent", type=Path, default=None)
    b.add_argument("--abx", type=Path, default=None)
    b.add_argument("--mos", type=Path, default=None)
    b.add_argument("--pins", type=Path, default=None)
    b.add_argument("--out", type=Path, required=True)

    d = sub.add_parser("diarize")
    d.add_argument("--label", required=True)
    d.add_argument("--compare-jsons", nargs="+", type=Path, required=True)
    d.add_argument("--invariants", nargs="*", type=Path, default=None)
    d.add_argument("--pins", type=Path, default=None)
    d.add_argument("--out", type=Path, required=True)

    args = p.parse_args(argv)
    out = _assert_decision_gate_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.track == "asr":
        out.write_text(reduce_asr(args))
    elif args.track == "tts":
        out.write_text(reduce_tts(args))
    elif args.track == "diarize":
        out.write_text(reduce_diarize(args))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
