#!/usr/bin/env python3
"""Saved-meeting regression bench for the live ASR -> translate -> TTS stack.

This is intentionally built from production meeting artifacts, not toy phrases.
It replays real recorded audio slices against the current ASR service, sends
real journal utterances through the current translation service, and synthesizes
real translated meeting text through the current TTS service. The pass/fail
gates target the failures that are expensive to discover live:

* ASR service stalls/timeouts/errors on real Poly-captured audio.
* ASR output diverges sharply from the journal generated for the same audio.
* TTS requests time out or return pathological audio durations.
* Translation fails or returns obvious wrong-script output.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import re
import statistics
import struct
import time
import unicodedata
import wave
from pathlib import Path

import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MEETINGS_DIR = REPO_ROOT / "meetings"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2

PUNCT_RE = re.compile(r"[、。「」『』！？!?,.\s]+")  # noqa: RUF001


def _normalize(text: str) -> str:
    return PUNCT_RE.sub("", unicodedata.normalize("NFKC", text or "").lower())


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


def _char_distance(a: str, b: str) -> float:
    na = _normalize(a)
    nb = _normalize(b)
    if not na and not nb:
        return 0.0
    return _levenshtein(list(na), list(nb)) / max(len(na), len(nb), 1)


def _script_lang(text: str) -> str:
    if any("぀" <= c <= "ヿ" for c in text or ""):
        return "ja"
    if any("一" <= c <= "鿿" for c in text or ""):
        return "ja"
    if any(("a" <= c.lower() <= "z") for c in text or ""):
        return "en"
    return "unknown"


def _pcm_slice_to_wav(pcm: bytes, start_ms: int, end_ms: int) -> tuple[bytes, float]:
    start = max(0, int(start_ms * SAMPLE_RATE / 1000) * BYTES_PER_SAMPLE)
    end = min(len(pcm), int(end_ms * SAMPLE_RATE / 1000) * BYTES_PER_SAMPLE)
    chunk = pcm[start:end]
    samples = np.frombuffer(chunk[: len(chunk) - len(chunk) % 2], dtype=np.int16)
    rms = (
        float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        if samples.size
        else 0.0
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(chunk)
    return buf.getvalue(), rms


def _load_journal(meeting_id: str) -> list[dict]:
    path = MEETINGS_DIR / meeting_id / "journal.jsonl"
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("is_final") and rec.get("text"):
            rows.append(rec)
    rows.sort(key=lambda r: int(r.get("start_ms") or 0))
    return rows


def _pick_segments(meeting_id: str, limit: int) -> list[dict]:
    rows = _load_journal(meeting_id)
    candidates = [
        r
        for r in rows
        if len(_normalize(r.get("text", ""))) >= 5
        and int(r.get("end_ms") or 0) > int(r.get("start_ms") or 0)
    ]
    if len(candidates) <= limit:
        return candidates
    step = len(candidates) / limit
    return [candidates[min(math.floor(i * step), len(candidates) - 1)] for i in range(limit)]


def _pick_tts_texts(meeting_id: str, limit: int) -> list[dict]:
    rows = _load_journal(meeting_id)
    candidates: list[dict] = []
    for row in rows:
        tr = row.get("translation") or {}
        text = (tr.get("text") or "").strip()
        lang = tr.get("target_language")
        if lang != "en" or len(text) < 8:
            continue
        candidates.append(
            {
                "meeting_id": meeting_id,
                "segment_id": row.get("segment_id"),
                "text": text,
                "language": "en",
            }
        )
    if len(candidates) <= limit:
        return candidates
    step = len(candidates) / limit
    return [candidates[min(math.floor(i * step), len(candidates) - 1)] for i in range(limit)]


async def _asr_request(
    client: httpx.AsyncClient, url: str, wav: bytes
) -> tuple[str, float, str | None]:
    body = {
        "model": "Qwen/Qwen3-ASR-1.7B",
        "messages": [
            {"role": "system", "content": "Transcribe in the spoken language."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(wav).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        resp = await client.post(f"{url}/v1/chat/completions", json=body, timeout=45)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if "<asr_text>" in raw:
            _, _, raw = raw.partition("<asr_text>")
        return raw.strip(), elapsed_ms, None
    except Exception as exc:
        return "", (time.monotonic() - t0) * 1000.0, f"{type(exc).__name__}: {exc}"


async def _translate_request(
    client: httpx.AsyncClient, url: str, text: str, source: str, target: str
) -> tuple[str, float, str | None]:
    model = await _discover_model(client, url)
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Translate {source} to {target}. Return only the translation, "
                    "with no explanation or notes."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "priority": -10,
    }
    t0 = time.monotonic()
    try:
        resp = await client.post(f"{url}/v1/chat/completions", json=body, timeout=45)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip(), elapsed_ms, None
    except Exception as exc:
        return "", (time.monotonic() - t0) * 1000.0, f"{type(exc).__name__}: {exc}"


_MODEL_CACHE: dict[str, str] = {}


async def _discover_model(client: httpx.AsyncClient, url: str) -> str:
    cached = _MODEL_CACHE.get(url)
    if cached:
        return cached
    resp = await client.get(f"{url}/v1/models", timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data") or []
    if not models:
        raise RuntimeError(f"No models reported by {url}")
    model = models[0]["id"]
    _MODEL_CACHE[url] = model
    return model


async def _tts_request(client: httpx.AsyncClient, url: str, text: str, language: str) -> dict:
    body = {
        "model": "qwen3-tts",
        "input": text,
        "voice": "ryan" if language == "en" else "ono_anna",
        "language": language,
        "response_format": "pcm",
    }
    started = time.monotonic()
    samples = 0
    chunks = 0
    err: str | None = None
    for attempt in range(2):
        samples = 0
        chunks = 0
        try:
            async with client.stream(
                "POST", f"{url}/v1/audio/speech/stream", json=body, timeout=90
            ) as resp:
                resp.raise_for_status()
                pending = bytearray()
                async for part in resp.aiter_bytes():
                    pending.extend(part)
                    while len(pending) >= 4:
                        size = struct.unpack(">I", pending[:4])[0]
                        if size == 0:
                            pending = pending[4:]
                            break
                        if len(pending) < 4 + size:
                            break
                        frame = pending[4 : 4 + size]
                        samples += len(frame) // 2
                        chunks += 1
                        pending = pending[4 + size :]
            err = None
            break
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if attempt == 1:
                break
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                try:
                    health = await client.get(f"{url}/health", timeout=3)
                    if health.status_code == 200:
                        break
                except Exception:
                    pass  # backend still cycling after the 5xx; keep polling until deadline
                await asyncio.sleep(1)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    return {
        "text": text,
        "language": language,
        "chars": len(text),
        "chunks": chunks,
        "audio_s": samples / 24000.0,
        "elapsed_ms": elapsed_ms,
        "error": err,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (k - lo)


def _summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


async def run(args: argparse.Namespace) -> dict:
    asr_rows: list[dict] = []
    translate_rows: list[dict] = []
    tts_rows: list[dict] = []

    async with httpx.AsyncClient(timeout=90) as client:
        for meeting_id in args.meeting_ids:
            pcm = (MEETINGS_DIR / meeting_id / "audio" / "recording.pcm").read_bytes()
            for seg in _pick_segments(meeting_id, args.asr_segments_per_meeting):
                start = max(0, int(seg["start_ms"]) - args.audio_pad_ms)
                end = int(seg["end_ms"]) + args.audio_pad_ms
                wav, rms = _pcm_slice_to_wav(pcm, start, end)
                hyp, elapsed, err = await _asr_request(client, args.asr_url, wav)
                ref = seg.get("text", "")
                asr_rows.append(
                    {
                        "meeting_id": meeting_id,
                        "segment_id": seg.get("segment_id"),
                        "start_ms": seg.get("start_ms"),
                        "end_ms": seg.get("end_ms"),
                        "rms": rms,
                        "reference": ref,
                        "hypothesis": hyp,
                        "distance": _char_distance(ref, hyp) if not err else None,
                        "ref_lang": seg.get("language"),
                        "hyp_lang": _script_lang(hyp),
                        "elapsed_ms": elapsed,
                        "error": err,
                    }
                )

            for seg in _pick_segments(meeting_id, args.text_segments_per_meeting):
                source = seg.get("language") or _script_lang(seg.get("text", ""))
                target = "en" if source == "ja" else "ja"
                translated, elapsed, err = await _translate_request(
                    client, args.translate_url, seg.get("text", ""), source, target
                )
                translate_rows.append(
                    {
                        "meeting_id": meeting_id,
                        "segment_id": seg.get("segment_id"),
                        "source": source,
                        "target": target,
                        "source_text": seg.get("text", ""),
                        "translation": translated,
                        "translation_lang": _script_lang(translated),
                        "elapsed_ms": elapsed,
                        "error": err,
                    }
                )
                if target == "en" and translated and not err:
                    tts_rows.append(await _tts_request(client, args.tts_url, translated, target))

            for item in _pick_tts_texts(meeting_id, args.tts_segments_per_meeting):
                result = await _tts_request(client, args.tts_url, item["text"], item["language"])
                result["meeting_id"] = meeting_id
                result["segment_id"] = item["segment_id"]
                result["source"] = "saved_journal_translation"
                tts_rows.append(result)

    asr_errors = [r for r in asr_rows if r["error"]]
    asr_consistency_rows = [
        r
        for r in asr_rows
        if r["distance"] is not None
        and r["ref_lang"] == r["hyp_lang"]
        and r["ref_lang"] in {"ja", "en"}
    ]
    asr_distances = [r["distance"] for r in asr_consistency_rows]
    asr_latencies = [r["elapsed_ms"] for r in asr_rows if not r["error"]]
    tts_errors = [r for r in tts_rows if r["error"]]
    tts_audio = [r["audio_s"] for r in tts_rows if not r["error"]]
    tts_elapsed = [r["elapsed_ms"] for r in tts_rows if not r["error"]]
    tts_pathological = [
        r
        for r in tts_rows
        if not r["error"]
        and (r["audio_s"] > args.max_tts_audio_s or r["audio_s"] > 1.5 + 0.22 * r["chars"])
    ]
    translate_errors = [r for r in translate_rows if r["error"]]
    wrong_script = [
        r
        for r in translate_rows
        if not r["error"]
        and r["target"] in {"en", "ja"}
        and r["translation_lang"] != "unknown"
        and r["translation_lang"] != r["target"]
    ]

    gates = {
        "asr_error_rate": len(asr_errors) / max(len(asr_rows), 1),
        "asr_distance_p95": _percentile(asr_distances, 0.95),
        "asr_consistency_count": len(asr_consistency_rows),
        "asr_lang_mismatch_count": sum(
            1
            for r in asr_rows
            if not r["error"] and r["ref_lang"] in {"ja", "en"} and r["ref_lang"] != r["hyp_lang"]
        ),
        "asr_latency_p95_ms": _percentile(asr_latencies, 0.95),
        "translate_error_rate": len(translate_errors) / max(len(translate_rows), 1),
        "translate_wrong_script": len(wrong_script),
        "tts_error_rate": len(tts_errors) / max(len(tts_rows), 1),
        "tts_audio_max_s": max(tts_audio) if tts_audio else None,
        "tts_elapsed_p95_ms": _percentile(tts_elapsed, 0.95),
        "tts_pathological_count": len(tts_pathological),
    }
    passed = (
        gates["asr_error_rate"] <= args.max_asr_error_rate
        and (
            gates["asr_distance_p95"] is None
            or gates["asr_distance_p95"] <= args.max_asr_distance_p95
        )
        and (
            gates["asr_latency_p95_ms"] is None
            or gates["asr_latency_p95_ms"] <= args.max_asr_latency_p95_ms
        )
        and gates["translate_error_rate"] == 0
        and gates["translate_wrong_script"] == 0
        and gates["tts_error_rate"] == 0
        and gates["tts_pathological_count"] == 0
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "meeting_ids": args.meeting_ids,
        "passed": passed,
        "gates": gates,
        "summary": {
            "asr": {
                "count": len(asr_rows),
                "errors": len(asr_errors),
                "consistency_count": len(asr_consistency_rows),
                "distance": _summary(asr_distances),
                "latency_ms": _summary(asr_latencies),
            },
            "translation": {
                "count": len(translate_rows),
                "errors": len(translate_errors),
                "wrong_script": len(wrong_script),
            },
            "tts": {
                "count": len(tts_rows),
                "errors": len(tts_errors),
                "audio_s": _summary(tts_audio),
                "elapsed_ms": _summary(tts_elapsed),
                "pathological": len(tts_pathological),
            },
        },
        "samples": {
            "asr_worst": sorted(
                [r for r in asr_rows if r["distance"] is not None],
                key=lambda r: r["distance"],
                reverse=True,
            )[:10],
            "tts_pathological": tts_pathological[:10],
            "translation_wrong_script": wrong_script[:10],
            "errors": {
                "asr": asr_errors[:10],
                "translation": translate_errors[:10],
                "tts": tts_errors[:10],
            },
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-ids", nargs="+", required=True)
    parser.add_argument("--asr-url", default="http://localhost:8003")
    parser.add_argument("--translate-url", default="http://localhost:8010")
    parser.add_argument("--tts-url", default="http://localhost:8002")
    parser.add_argument("--asr-segments-per-meeting", type=int, default=8)
    parser.add_argument("--text-segments-per-meeting", type=int, default=6)
    parser.add_argument("--tts-segments-per-meeting", type=int, default=6)
    parser.add_argument("--audio-pad-ms", type=int, default=350)
    parser.add_argument("--max-asr-error-rate", type=float, default=0.0)
    parser.add_argument("--max-asr-distance-p95", type=float, default=0.90)
    parser.add_argument("--max-asr-latency-p95-ms", type=float, default=15000)
    parser.add_argument("--max-tts-audio-s", type=float, default=12.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = asyncio.run(run(args))
    out = args.out
    if out is None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        out = RESULTS_DIR / f"live_stack_regression_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: report[k] for k in ("passed", "gates", "summary")}, ensure_ascii=False, indent=2
        )
    )
    print(f"report saved: {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
