"""Deep real-world ASR bench — Cohere vs Qwen3-ASR on actual meeting audio.

Why this exists.  The Fleurs B1 bench told us Cohere is significantly
worse on read-aloud parliamentary EN.  That's not what production
meeting audio looks like — meetings have disfluencies, code-switching,
accents, technical vocabulary, multi-speaker overlap, and silence.
This harness exercises both backends on the audio we actually have on
disk and characterizes WHERE they differ, not just by how much on
average.

Methodology
-----------

For each meeting in ``--meeting-ids``:
1. Slice ``meetings/<id>/audio/recording.pcm`` (s16le, 16 kHz mono)
   into 30-second chunks with 2 s overlap.
2. For each chunk, compute RMS to flag near-silent windows.
3. Send each chunk to Qwen3-ASR (port 8003) and Cohere (port 8013) in
   parallel; capture transcript, latency, error.
4. Compute character-level Levenshtein between the two transcripts
   (disagreement metric, no ground truth needed).
5. Optionally compare each transcript against the meeting's journal
   for that time window (= "what did the production pipeline see at
   ASR-time?").  Journal events with overlap ≥ 50 % of the chunk
   duration count.

Outputs
-------

Per-meeting JSON (offline, under ``--out-dir``):
* ``<meeting_id>.json`` — per-chunk records with text/lat/disagreement.

Aggregated report (Markdown, written to the path passed via
``--report``).  Suitable to land under
``reports/2026-Q2-bench/asr_realworld/`` (the only repo path allowed
by the offline-bench-paths rule when the file ends in `.md`).

Run::

    MEETING_SCRIBE_BENCH_WINDOW=1 python3 scripts/bench/asr_meeting_realworld.py \\
        --meeting-ids fe77b412-... 73d3fbbd-... 4cee0e9b-... d34ca2f0-... f38d5807-... \\
        --out-dir /data/meeting-scribe-fixtures/bench-runs/2026-Q3/asr_realworld \\
        --report reports/2026-Q2-bench/asr_realworld/decision_gate.md
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import statistics
import sys
import time
import unicodedata
import wave
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks._bench_paths import assert_offline_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MEETINGS_DIR = REPO_ROOT / "meetings"

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
CHUNK_SECONDS = 20
OVERLAP_SECONDS = 2

QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"

# Cohere supports 14 languages on paper but in practice some codes
# trigger a CUDA device-side assert that wedges the GPU context.  We
# restrict to en/ja — the only two we actually need for this bench
# (real meetings are predominantly EN+JA with occasional code-switch).
# Any other Qwen-detected language falls back to "en".
#
# Once the CUDA context is corrupt, every subsequent Cohere call 500s
# until the container is restarted; restricting to the two
# known-stable codes is the cheapest insurance.
COHERE_SAFE_LANGS = {"en", "ja"}


# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------


def _slice_pcm_to_wavs(pcm: bytes, chunk_s: int, overlap_s: int):
    """Yield ``(start_ms, end_ms, wav_bytes, rms)`` chunks."""
    chunk_bytes = chunk_s * SAMPLE_RATE * BYTES_PER_SAMPLE
    stride = (chunk_s - overlap_s) * SAMPLE_RATE * BYTES_PER_SAMPLE
    for off in range(0, len(pcm), stride):
        chunk = pcm[off : off + chunk_bytes]
        if len(chunk) < SAMPLE_RATE * BYTES_PER_SAMPLE * 2:
            # < 2 s — skip
            continue
        # WAV header
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(chunk)
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        start_ms = int(off / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000)
        end_ms = int((off + len(chunk)) / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000)
        yield start_ms, end_ms, buf.getvalue(), rms


# ---------------------------------------------------------------------------
# ASR clients
# ---------------------------------------------------------------------------


_QWEN_LANG_MAP = {
    "english": "en",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
}


async def _ask_asr(
    c: httpx.AsyncClient, url: str, wav: bytes, model: str, language: str | None = None
) -> tuple[str, float, str | None, str | None]:
    """Returns (text, elapsed_ms, error, language_iso)."""
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Transcribe in the spoken language."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"},
                    }
                ],
            },
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    if language:
        body["language"] = language
    t0 = time.monotonic()
    try:
        r = await c.post(f"{url}/v1/chat/completions", json=body, timeout=120)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        r.raise_for_status()
    except Exception as exc:
        return "", (time.monotonic() - t0) * 1000.0, f"{type(exc).__name__}: {exc}", None
    raw = r.json()["choices"][0]["message"]["content"].strip()
    detected: str | None = None
    if "<asr_text>" in raw:
        prefix, _, raw = raw.partition("<asr_text>")
        # prefix is "language English" or similar
        m = re.search(r"language\s+([A-Za-z]+)", prefix, re.IGNORECASE)
        if m:
            detected = _QWEN_LANG_MAP.get(m.group(1).lower())
    return raw.strip(), elapsed_ms, None, detected


# ---------------------------------------------------------------------------
# Disagreement metric
# ---------------------------------------------------------------------------


_PUNCT = re.compile(r"[、。「」『』！？!?,.\s]+")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text))


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


def _char_disagreement(a: str, b: str) -> tuple[float, int]:
    """Levenshtein on chars; returns (fraction_of_longer, edits)."""
    na, nb = _normalize(a), _normalize(b)
    if not na and not nb:
        return 0.0, 0
    edits = _levenshtein(list(na), list(nb))
    longer = max(len(na), len(nb), 1)
    return edits / longer, edits


def _detect_lang(text: str) -> str:
    """Crude script-based language hint. JA dominates if any kana/kanji."""
    if not text:
        return "?"
    ja = sum(
        1
        for c in text
        if "぀" <= c <= "ヿ" or "一" <= c <= "鿿" or "ㇰ" <= c <= "ㇿ"
    )
    if ja >= 1:
        return "ja"
    return "en"


# ---------------------------------------------------------------------------
# Journal lookup (production ASR ground reference)
# ---------------------------------------------------------------------------


def _load_journal_segments(meeting_id: str) -> list[dict]:
    journal = MEETINGS_DIR / meeting_id / "journal.jsonl"
    if not journal.exists():
        return []
    out: list[dict] = []
    for line in journal.open():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not ev.get("is_final"):
            continue
        if not ev.get("text"):
            continue
        out.append(
            {
                "start_ms": ev.get("start_ms", 0),
                "end_ms": ev.get("end_ms", 0),
                "text": ev["text"],
                "language": ev.get("language", "?"),
            }
        )
    out.sort(key=lambda e: e["start_ms"])
    return out


def _journal_text_for_window(
    journal: list[dict], start_ms: int, end_ms: int, min_overlap_pct: float = 0.5
) -> str:
    chunk_dur = max(1, end_ms - start_ms)
    parts: list[str] = []
    for ev in journal:
        ov = max(0, min(end_ms, ev["end_ms"]) - max(start_ms, ev["start_ms"]))
        if ov / chunk_dur >= min_overlap_pct:
            parts.append(ev["text"])
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Per-meeting bench driver
# ---------------------------------------------------------------------------


async def _bench_one_meeting(
    c: httpx.AsyncClient,
    meeting_id: str,
    qwen_url: str,
    cohere_url: str,
    out_dir: Path,
) -> dict:
    pcm_path = MEETINGS_DIR / meeting_id / "audio" / "recording.pcm"
    if not pcm_path.exists():
        raise FileNotFoundError(pcm_path)
    pcm = pcm_path.read_bytes()
    journal = _load_journal_segments(meeting_id)

    chunks = list(_slice_pcm_to_wavs(pcm, CHUNK_SECONDS, OVERLAP_SECONDS))
    print(f"[{meeting_id}] {len(chunks)} chunks, journal events={len(journal)}", flush=True)

    rows: list[dict] = []
    for i, (start_ms, end_ms, wav, rms) in enumerate(chunks):
        # Two-stage call: Qwen3-ASR first (it auto-detects + emits the
        # detected language as a prefix); then call Cohere with that
        # language hint, which Cohere's processor REQUIRES.
        qhyp, q_ms, q_err, q_lang = await _ask_asr(c, qwen_url, wav, model=QWEN_MODEL)
        # Default Cohere's language hint to the Qwen-detected one;
        # fall back to "en" if Qwen didn't emit a parseable prefix or
        # if the detected language is outside Cohere's supported set
        # (mainly affects code-switched Chinese on JA meetings).
        cohere_lang = q_lang if q_lang in COHERE_SAFE_LANGS else "en"
        chyp, c_ms, c_err, _ = await _ask_asr(
            c, cohere_url, wav, model="auto", language=cohere_lang
        )

        disagree_frac, disagree_edits = _char_disagreement(qhyp, chyp)
        ref_text = _journal_text_for_window(journal, start_ms, end_ms)
        q_vs_ref_frac, _ = _char_disagreement(qhyp, ref_text) if ref_text else (None, None)
        c_vs_ref_frac, _ = _char_disagreement(chyp, ref_text) if ref_text else (None, None)
        lang_q = _detect_lang(qhyp)
        lang_c = _detect_lang(chyp)
        # Joint language: ja if either side detects ja, else en, else ?.
        if "ja" in (lang_q, lang_c):
            joint_lang = "ja"
        elif "en" in (lang_q, lang_c):
            joint_lang = "en"
        else:
            joint_lang = "?"

        rows.append(
            {
                "chunk_idx": i,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "rms": round(rms, 4),
                "lang": joint_lang,
                "qwen": {
                    "text": qhyp,
                    "ms": round(q_ms, 1),
                    "lang_detected": lang_q,
                    "lang_qwen_prefix": q_lang,
                    "error": q_err,
                },
                "cohere": {
                    "text": chyp,
                    "ms": round(c_ms, 1),
                    "lang_detected": lang_c,
                    "language_hint_used": cohere_lang,
                    "error": c_err,
                },
                "char_disagreement_frac": round(disagree_frac, 4),
                "char_disagreement_edits": disagree_edits,
                "qwen_vs_journal_frac": (
                    None if q_vs_ref_frac is None else round(q_vs_ref_frac, 4)
                ),
                "cohere_vs_journal_frac": (
                    None if c_vs_ref_frac is None else round(c_vs_ref_frac, 4)
                ),
                "journal_text_present": bool(ref_text),
            }
        )
        if (i + 1) % 10 == 0:
            print(
                f"[{meeting_id}] {i + 1}/{len(chunks)} disagree={disagree_frac:.2%} "
                f"q={q_ms:.0f}ms c={c_ms:.0f}ms",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{meeting_id}.json").write_text(json.dumps({"meeting_id": meeting_id, "rows": rows}, indent=2))
    return {"meeting_id": meeting_id, "rows": rows}


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    import math

    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summarize(rows: list[dict]) -> dict:
    """Bucketed summary statistics for one or many meetings of rows."""
    speech = [r for r in rows if r["rms"] >= 0.005]  # filter near-silence
    by_lang: dict[str, list[dict]] = {}
    for r in speech:
        by_lang.setdefault(r["lang"], []).append(r)

    def _bucket(rows: list[dict], key: str) -> dict:
        vals = [r[key] for r in rows if r[key] is not None]
        return {
            "n": len(vals),
            "p50": _percentile(vals, 0.5),
            "p95": _percentile(vals, 0.95),
            "mean": (sum(vals) / len(vals)) if vals else None,
        }

    out: dict = {
        "total_chunks": len(rows),
        "speech_chunks": len(speech),
        "silent_chunks": len(rows) - len(speech),
        "qwen_errors": sum(1 for r in rows if r["qwen"].get("error")),
        "cohere_errors": sum(1 for r in rows if r["cohere"].get("error")),
    }
    out["disagreement"] = {
        lang: _bucket(rs, "char_disagreement_frac") for lang, rs in by_lang.items()
    }
    out["qwen_vs_journal"] = {
        lang: _bucket(
            [r for r in rs if r["journal_text_present"]],
            "qwen_vs_journal_frac",
        )
        for lang, rs in by_lang.items()
    }
    out["cohere_vs_journal"] = {
        lang: _bucket(
            [r for r in rs if r["journal_text_present"]],
            "cohere_vs_journal_frac",
        )
        for lang, rs in by_lang.items()
    }
    # Latency uses nested r["qwen"]["ms"] / r["cohere"]["ms"], so build
    # the buckets inline rather than via the flat-key _bucket helper.
    out["latency_ms"] = {
        "qwen": {
            "p50": _percentile([r["qwen"]["ms"] for r in speech if not r["qwen"].get("error")], 0.5),
            "p95": _percentile([r["qwen"]["ms"] for r in speech if not r["qwen"].get("error")], 0.95),
            "n": sum(1 for r in speech if not r["qwen"].get("error")),
        },
        "cohere": {
            "p50": _percentile([r["cohere"]["ms"] for r in speech if not r["cohere"].get("error")], 0.5),
            "p95": _percentile([r["cohere"]["ms"] for r in speech if not r["cohere"].get("error")], 0.95),
            "n": sum(1 for r in speech if not r["cohere"].get("error")),
        },
    }
    return out


def _render_report(per_meeting: list[dict], overall: dict, top_disagreements: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Real-world ASR bench — Cohere vs Qwen3-ASR on production meetings")
    lines.append("")
    lines.append(
        f"Aggregate: **{overall['total_chunks']}** chunks across "
        f"**{len(per_meeting)}** meetings · **{overall['speech_chunks']}** with speech, "
        f"**{overall['silent_chunks']}** near-silent (skipped from disagreement stats)."
    )
    lines.append(
        f"Errors: Qwen3-ASR **{overall['qwen_errors']}**, Cohere **{overall['cohere_errors']}**."
    )
    lines.append("")

    lines.append("## Char-level disagreement between backends (per language)")
    lines.append("")
    lines.append("| Language | n | p50 | p95 | mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for lang, stats in sorted(overall["disagreement"].items()):
        if not stats["n"]:
            continue
        lines.append(
            f"| {lang} | {stats['n']} | "
            f"{stats['p50']:.1%} | {stats['p95']:.1%} | {stats['mean']:.1%} |"
        )
    lines.append("")
    lines.append(
        "_This is char-level Levenshtein between Qwen3-ASR's and Cohere's transcripts "
        "on the same chunk.  No ground truth needed.  Higher = backends disagree more "
        "on what was said._"
    )
    lines.append("")

    lines.append("## Each backend vs the meeting journal (production-ASR-at-record-time)")
    lines.append("")
    lines.append(
        "Each chunk's transcripts compared against the journal events that overlap "
        "the chunk window by ≥ 50 %.  This is _not_ ground truth — the journal IS "
        "Qwen3-ASR's output at record time — but it lets us see how much each "
        "backend would diverge from what the production pipeline produced."
    )
    lines.append("")
    lines.append("### Qwen3-ASR vs journal (consistency check on its own past output)")
    lines.append("")
    lines.append("| Language | n | p50 | p95 | mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for lang, stats in sorted(overall["qwen_vs_journal"].items()):
        if not stats["n"]:
            continue
        lines.append(
            f"| {lang} | {stats['n']} | "
            f"{stats['p50']:.1%} | {stats['p95']:.1%} | {stats['mean']:.1%} |"
        )
    lines.append("")
    lines.append("### Cohere vs journal (would-have-been-this-instead)")
    lines.append("")
    lines.append("| Language | n | p50 | p95 | mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for lang, stats in sorted(overall["cohere_vs_journal"].items()):
        if not stats["n"]:
            continue
        lines.append(
            f"| {lang} | {stats['n']} | "
            f"{stats['p50']:.1%} | {stats['p95']:.1%} | {stats['mean']:.1%} |"
        )
    lines.append("")

    lines.append("## Latency (speech chunks only)")
    lines.append("")
    lat = overall["latency_ms"]
    lines.append("| Backend | p50 (ms) | p95 (ms) | n |")
    lines.append("|---|---:|---:|---:|")

    def _fmt(x):
        return f"{x:.0f}" if x is not None else "n/a"

    lines.append(
        f"| Qwen3-ASR-1.7B | {_fmt(lat['qwen']['p50'])} | {_fmt(lat['qwen']['p95'])} | {lat['qwen']['n']} |"
    )
    lines.append(
        f"| Cohere Transcribe | {_fmt(lat['cohere']['p50'])} | {_fmt(lat['cohere']['p95'])} | {lat['cohere']['n']} |"
    )
    lines.append("")

    lines.append("## Per-meeting summary")
    lines.append("")
    lines.append("| Meeting | Chunks | Speech | Q errors | C errors | Disagreement p50 | Disagreement p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in per_meeting:
        s = _summarize(m["rows"])
        # Fold all langs together for the per-meeting one-liner.
        all_disagree = [r["char_disagreement_frac"] for r in m["rows"] if r["rms"] >= 0.005]
        p50 = _percentile(all_disagree, 0.5)
        p95 = _percentile(all_disagree, 0.95)
        lines.append(
            f"| `{m['meeting_id'][:8]}` | {s['total_chunks']} | {s['speech_chunks']} | "
            f"{s['qwen_errors']} | {s['cohere_errors']} | "
            f"{p50:.1%} | {p95:.1%} |"
        )
    lines.append("")

    if top_disagreements:
        lines.append("## Top-30 highest-disagreement chunks (where the backends saw very different things)")
        lines.append("")
        lines.append("| Meeting | Window | RMS | Lang | Disagreement | Qwen text (truncated) | Cohere text (truncated) |")
        lines.append("|---|---|---:|---|---:|---|---|")
        for r in top_disagreements:
            qt = (r["qwen"]["text"] or "(empty)").replace("|", "/")[:60]
            ct = (r["cohere"]["text"] or "(empty)").replace("|", "/")[:60]
            lines.append(
                f"| `{r['meeting_id'][:8]}` | "
                f"{r['start_ms'] // 1000}s–{r['end_ms'] // 1000}s | "
                f"{r['rms']:.3f} | {r['lang']} | "
                f"{r['char_disagreement_frac']:.1%} | {qt} | {ct} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Char-level disagreement is a structural similarity metric — it does not "
        "tell you which transcript is correct.  It DOES tell you where the two "
        "backends would have written different transcripts on the same audio.  "
        "Use it to characterize the failure modes (long utterances, code-switching, "
        "rare vocabulary, near-silence) and decide whether one backend's strengths "
        "are worth the other's weaknesses on real production audio._"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def run(args) -> None:
    out_dir = assert_offline_path(args.out_dir)
    per_meeting: list[dict] = []
    async with httpx.AsyncClient(timeout=180) as c:
        for mid in args.meeting_ids:
            result = await _bench_one_meeting(c, mid, args.qwen_url, args.cohere_url, out_dir)
            per_meeting.append(result)

    all_rows = [r for m in per_meeting for r in m["rows"]]
    for r in all_rows:
        r["meeting_id"] = r.get("meeting_id") or next(
            m["meeting_id"] for m in per_meeting if any(rr is r for rr in m["rows"])
        )
    # Annotate each row with its meeting id so the top-disagreements table can cite it.
    for m in per_meeting:
        for r in m["rows"]:
            r["meeting_id"] = m["meeting_id"]

    overall = _summarize(all_rows)
    speech_rows = [r for r in all_rows if r["rms"] >= 0.005]
    top_disagreements = sorted(
        speech_rows, key=lambda r: r["char_disagreement_frac"], reverse=True
    )[:30]

    report = _render_report(per_meeting, overall, top_disagreements)
    if args.report:
        # Reports under reports/2026-Q2-bench/ are .md only (gitignore covers others).
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"\nWrote report to {args.report}", flush=True)
    else:
        print(report)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meeting-ids", nargs="+", required=True)
    p.add_argument("--qwen-url", default="http://localhost:8003")
    p.add_argument("--cohere-url", default="http://localhost:8013")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
