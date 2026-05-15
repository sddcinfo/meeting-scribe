#!/usr/bin/env python3
"""Production-path replay quality benchmark for saved meetings.

This bench is intentionally local/private-data aware:

* Meeting ids are CLI/env input, never hard-coded.
* Reports default to aggregates only.
* The default output path should be under /tmp or another ignored location.

It compares two paths on the same saved recording audio:

* ``legacy_endpoint``: fixed 3.5s ASR chunks sent directly to the ASR vLLM
  endpoint with the older generic production prompt and no post-filtering.
* ``current_production``: the real ``VllmASRBackend`` path, including the
  meeting language pair, prompt, script correction, VAD, trimming, and filters.

The current ASR output is then sent through the configured translate and TTS
services to measure end-to-end request volume and latency.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import statistics
import struct
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from scripts.bench.live_stack_regression import (
    _char_distance,
    _script_lang,
    _translate_request,
)

from meeting_scribe.backends.asr_filters import (
    _detect_language_from_text,
    _is_low_value_english_fragment,
    _parse_qwen3_asr_response,
)
from meeting_scribe.backends.asr_vllm import SAMPLE_RATE, VAD_ENERGY_THRESHOLD, VllmASRBackend
from meeting_scribe.translation.queue import _translation_matches_target_language

REPO_ROOT = Path(__file__).resolve().parents[2]
MEETINGS_DIR = REPO_ROOT / "meetings"
BYTES_PER_SAMPLE = 2


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (k - lo)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": _pct(values, 0.50),
        "p95": _pct(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def _old_prompt(language_pair: list[str]) -> str:
    if not language_pair:
        return "Transcribe the audio in the original spoken language. Do not translate."

    from meeting_scribe.languages import LANGUAGE_REGISTRY

    names: list[str] = []
    for code in sorted(language_pair):
        lang = LANGUAGE_REGISTRY.get(code)
        names.append(f"{lang.name} ({code})" if lang is not None else code.upper())
    joined = " or ".join(names) if len(names) <= 2 else ", ".join(names)
    return (
        "Transcribe the audio in the original spoken language. "
        f"The speaker is using {joined}. "
        "Do NOT translate. If the speech is in one of these languages, "
        "output it verbatim in that language."
    )


def _load_pcm(meeting_id: str) -> np.ndarray:
    pcm_path = MEETINGS_DIR / meeting_id / "audio" / "recording.pcm"
    pcm = pcm_path.read_bytes()
    return (
        np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16).astype(np.float32) / 32768.0
    )


def _load_reference_rows(meeting_id: str) -> list[dict[str, Any]]:
    path = MEETINGS_DIR / meeting_id / "journal.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("is_final") and rec.get("text"):
            rows.append(rec)
    return rows


def _reference_lang(rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> str:
    scores: dict[str, int] = {}
    for row in rows:
        rs = int(row.get("start_ms") or 0)
        re = int(row.get("end_ms") or 0)
        overlap = max(0, min(end_ms, re) - max(start_ms, rs))
        if overlap <= 0:
            continue
        lang = row.get("language") or _script_lang(row.get("text", ""))
        scores[lang] = scores.get(lang, 0) + overlap
    if not scores:
        return "unknown"
    return max(scores.items(), key=lambda item: item[1])[0]


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def _pcm24k_to_wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24_000)
        wf.writeframes(pcm)
    return buf.getvalue()


def _write_pcm24k_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_pcm24k_to_wav_bytes(pcm))


async def _tts_request_capture(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    language: str,
    *,
    capture_dir: Path | None = None,
    capture_stem: str = "tts",
) -> dict[str, Any]:
    body = {
        "model": "qwen3-tts",
        "input": text,
        "voice": "ryan" if language == "en" else "ono_anna",
        "language": language,
        "response_format": "pcm",
    }
    started = time.monotonic()
    pcm_out = bytearray()
    chunks = 0
    err: str | None = None
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
                    pcm_out.extend(frame)
                    chunks += 1
                    pending = pending[4 + size :]
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"

    elapsed_ms = (time.monotonic() - started) * 1000.0
    capture_path: str | None = None
    if pcm_out and capture_dir is not None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        path = capture_dir / f"{capture_stem}.{language}.{digest}.wav"
        _write_pcm24k_wav(path, bytes(pcm_out))
        capture_path = str(path)
    return {
        "language": language,
        "chars": len(text),
        "chunks": chunks,
        "audio_s": len(pcm_out) / 2 / 24_000,
        "elapsed_ms": elapsed_ms,
        "error": err,
        "capture_path": capture_path,
        "_pcm": bytes(pcm_out),
        "_text": text,
    }


async def _asr_generated_tts(
    client: httpx.AsyncClient,
    asr_url: str,
    text: str,
    language: str,
    pcm24k: bytes,
) -> tuple[float | None, str | None]:
    if not pcm24k:
        return None, None
    body = {
        "model": await _discover_asr_model(client, asr_url),
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Transcribe this generated {language} speech exactly. Do not translate."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(_pcm24k_to_wav_bytes(pcm24k)).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "priority": -20,
    }
    try:
        resp = await client.post(f"{asr_url}/v1/chat/completions", json=body, timeout=45)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        hyp, _lang = _parse_qwen3_asr_response(raw)
        return _char_distance(text, hyp), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def _translate_with_script_retry(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    source: str,
    target: str,
) -> tuple[str, float, str | None, int]:
    translated, elapsed_ms, error = await _translate_request(client, url, text, source, target)
    attempts = 1
    if error or _translation_matches_target_language(translated, target):
        return translated, elapsed_ms, error, attempts

    retry_text, retry_ms, retry_error = await _translate_request(client, url, text, source, target)
    attempts += 1
    elapsed_ms += retry_ms
    if retry_error is None:
        translated = retry_text
    error = retry_error
    return translated, elapsed_ms, error, attempts


async def _legacy_asr_request(
    client: httpx.AsyncClient,
    asr_url: str,
    model: str,
    prompt: str,
    audio: np.ndarray,
) -> tuple[str, str, float, str | None]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(_wav_bytes(audio)).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "priority": -20,
    }
    started = time.monotonic()
    try:
        resp = await client.post(f"{asr_url}/v1/chat/completions", json=body, timeout=45)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        text, lang = _parse_qwen3_asr_response(raw)
        return text, lang, elapsed_ms, None
    except Exception as exc:
        return "", "unknown", (time.monotonic() - started) * 1000.0, f"{type(exc).__name__}: {exc}"


async def _discover_asr_model(client: httpx.AsyncClient, asr_url: str) -> str:
    resp = await client.get(f"{asr_url}/v1/models", timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data") or []
    if not models:
        raise RuntimeError(f"No ASR models reported by {asr_url}")
    return models[0]["id"]


async def _run_legacy_endpoint(
    *,
    meeting_id: str,
    audio: np.ndarray,
    ref_rows: list[dict[str, Any]],
    asr_url: str,
    language_pair: list[str],
    chunk_seconds: float,
) -> dict[str, Any]:
    prompt = _old_prompt(language_pair)
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=60) as client:
        model = await _discover_asr_model(client, asr_url)
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) < SAMPLE_RATE // 2:
                continue
            rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
            if rms < VAD_ENERGY_THRESHOLD:
                continue
            start_ms = int(start / SAMPLE_RATE * 1000)
            end_ms = int((start + len(chunk)) / SAMPLE_RATE * 1000)
            text, lang, elapsed_ms, error = await _legacy_asr_request(
                client, asr_url, model, prompt, chunk
            )
            if not text and not error:
                continue
            rows.append(
                {
                    "meeting_id": meeting_id,
                    "segment_id": str(uuid.uuid4()),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "language": lang,
                    "text": text,
                    "script_lang": _detect_language_from_text(text) if text else "unknown",
                    "reference_lang": _reference_lang(ref_rows, start_ms, end_ms),
                    "elapsed_ms": elapsed_ms,
                    "error": error,
                }
            )
    return {"rows": rows, "wall_ms": (time.monotonic() - started) * 1000.0}


async def _run_current_production(
    *,
    meeting_id: str,
    audio: np.ndarray,
    ref_rows: list[dict[str, Any]],
    asr_url: str,
    language_pair: list[str],
    chunk_seconds: float,
) -> dict[str, Any]:
    backend = VllmASRBackend(base_url=asr_url, languages=language_pair)
    backend._async_live_submit = False
    backend.audio_wall_at_start = time.monotonic()
    rows: list[dict[str, Any]] = []
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    started = time.monotonic()
    await backend.start()
    try:
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if not len(chunk):
                continue
            async for event in backend.process_audio(chunk, sample_offset=start, _is_replay=True):
                rows.append(
                    {
                        "meeting_id": meeting_id,
                        "segment_id": event.segment_id,
                        "start_ms": event.start_ms,
                        "end_ms": event.end_ms,
                        "language": event.language,
                        "text": event.text,
                        "script_lang": _detect_language_from_text(event.text),
                        "reference_lang": _reference_lang(ref_rows, event.start_ms, event.end_ms),
                        "elapsed_ms": None,
                        "error": None,
                    }
                )
        async for event in backend.flush():
            rows.append(
                {
                    "meeting_id": meeting_id,
                    "segment_id": event.segment_id,
                    "start_ms": event.start_ms,
                    "end_ms": event.end_ms,
                    "language": event.language,
                    "text": event.text,
                    "script_lang": _detect_language_from_text(event.text),
                    "reference_lang": _reference_lang(ref_rows, event.start_ms, event.end_ms),
                    "elapsed_ms": None,
                    "error": None,
                }
            )
    finally:
        await backend.stop()
    request_ms = list(
        getattr(
            __import__("meeting_scribe.runtime.state").runtime.state.metrics,
            "asr_request_rtt_ms",
            [],
        )
    )
    return {
        "rows": rows,
        "wall_ms": (time.monotonic() - started) * 1000.0,
        "request_ms": request_ms,
    }


def _asr_quality(rows: list[dict[str, Any]], audio_s: float, wall_ms: float) -> dict[str, Any]:
    usable = [r for r in rows if r.get("text") and not r.get("error")]
    errors = [r for r in rows if r.get("error")]
    lang_counts: dict[str, int] = {}
    for row in usable:
        lang = row.get("language") or "unknown"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    ref_lang_mismatch = [
        r
        for r in usable
        if r.get("reference_lang") in {"en", "ja"}
        and r.get("language") in {"en", "ja"}
        and r.get("reference_lang") != r.get("language")
    ]
    script_tag_mismatch = [
        r
        for r in usable
        if r.get("script_lang") in {"en", "ja"}
        and r.get("language") in {"en", "ja"}
        and r.get("script_lang") != r.get("language")
    ]
    low_value_en = [
        r
        for r in usable
        if r.get("language") == "en" and _is_low_value_english_fragment(str(r.get("text") or ""))
    ]
    tts_candidate_chars = sum(len(str(r.get("text") or "")) for r in usable)
    return {
        "events": len(usable),
        "errors": len(errors),
        "lang_counts": lang_counts,
        "reference_lang_mismatch": len(ref_lang_mismatch),
        "script_tag_mismatch": len(script_tag_mismatch),
        "low_value_english_fragments": len(low_value_en),
        "tts_candidate_chars": tts_candidate_chars,
        "audio_s": audio_s,
        "wall_ms": wall_ms,
        "realtime_factor": (wall_ms / 1000.0) / audio_s if audio_s else None,
    }


async def _run_downstream(
    rows: list[dict[str, Any]],
    *,
    asr_url: str,
    translate_url: str,
    tts_url: str,
    language_pair: list[str],
    max_segments: int,
    capture_dir: Path | None,
    label: str,
    validate_tts_asr: bool,
) -> dict[str, Any]:
    source_rows = [r for r in rows if r.get("text") and r.get("language") in language_pair]
    source_rows = source_rows[:max_segments]
    translate_rows: list[dict[str, Any]] = []
    tts_rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=90) as client:
        for row in source_rows:
            source = str(row["language"])
            targets = [lang for lang in language_pair if lang != source]
            if not targets:
                continue
            target = targets[0]
            translated, elapsed_ms, error, attempts = await _translate_with_script_retry(
                client, translate_url, str(row["text"]), source, target
            )
            translate_rows.append(
                {
                    "source": source,
                    "target": target,
                    "translation_lang": _script_lang(translated),
                    "translation_text": translated,
                    "attempts": attempts,
                    "elapsed_ms": elapsed_ms,
                    "error": error,
                }
            )
            if translated and not error and target in {"en", "ja"}:
                (
                    roundtrip,
                    roundtrip_ms,
                    roundtrip_error,
                    roundtrip_attempts,
                ) = await _translate_with_script_retry(
                    client,
                    translate_url,
                    translated,
                    target,
                    source,
                )
                translate_rows[-1].update(
                    {
                        "roundtrip_attempts": roundtrip_attempts,
                        "roundtrip_elapsed_ms": roundtrip_ms,
                        "roundtrip_error": roundtrip_error,
                        "roundtrip_lang": _script_lang(roundtrip),
                        "roundtrip_text": roundtrip,
                        "roundtrip_distance": (
                            _char_distance(str(row["text"]), roundtrip)
                            if not roundtrip_error
                            else None
                        ),
                    }
                )
                idx = len(tts_rows)
                result = await _tts_request_capture(
                    client,
                    tts_url,
                    translated,
                    target,
                    capture_dir=capture_dir,
                    capture_stem=f"{label}.{idx:04d}",
                )
                if validate_tts_asr and not result["error"]:
                    distance, asr_error = await _asr_generated_tts(
                        client,
                        asr_url,
                        translated,
                        target,
                        result.get("_pcm") or b"",
                    )
                    result["asr_distance"] = distance
                    result["asr_error"] = asr_error
                result.pop("_pcm", None)
                result.pop("_text", None)
                tts_rows.append(result)

    translate_errors = [r for r in translate_rows if r.get("error")]
    roundtrip_errors = [r for r in translate_rows if r.get("roundtrip_error")]
    translate_retries = sum(max(0, int(r.get("attempts") or 1) - 1) for r in translate_rows)
    roundtrip_retries = sum(
        max(0, int(r.get("roundtrip_attempts") or 1) - 1)
        for r in translate_rows
        if r.get("roundtrip_attempts") is not None
    )
    roundtrip_wrong_script = [
        r
        for r in translate_rows
        if not r.get("roundtrip_error")
        and not _translation_matches_target_language(
            str(r.get("roundtrip_text") or ""),
            str(r.get("source") or ""),
        )
    ]
    roundtrip_distances = [
        float(r["roundtrip_distance"])
        for r in translate_rows
        if r.get("roundtrip_distance") is not None and not r.get("roundtrip_error")
    ]
    roundtrip_en_source_distances = [
        float(r["roundtrip_distance"])
        for r in translate_rows
        if r.get("source") == "en"
        and r.get("roundtrip_distance") is not None
        and not r.get("roundtrip_error")
    ]
    roundtrip_ja_source_distances = [
        float(r["roundtrip_distance"])
        for r in translate_rows
        if r.get("source") == "ja"
        and r.get("roundtrip_distance") is not None
        and not r.get("roundtrip_error")
    ]
    wrong_script = [
        r
        for r in translate_rows
        if not r.get("error")
        and not _translation_matches_target_language(
            str(r.get("translation_text") or ""),
            str(r.get("target") or ""),
        )
    ]
    tts_errors = [r for r in tts_rows if r.get("error")]
    tts_asr_errors = [r for r in tts_rows if r.get("asr_error")]
    tts_asr_distances = [
        float(r["asr_distance"])
        for r in tts_rows
        if r.get("asr_distance") is not None and not r.get("asr_error")
    ]
    return {
        "translation": {
            "count": len(translate_rows),
            "errors": len(translate_errors),
            "wrong_script": len(wrong_script),
            "retries": translate_retries,
            "roundtrip": {
                "count": len(roundtrip_distances),
                "errors": len(roundtrip_errors),
                "wrong_script": len(roundtrip_wrong_script),
                "retries": roundtrip_retries,
                "distance": _summary(roundtrip_distances),
                "distance_en_source": _summary(roundtrip_en_source_distances),
                "distance_ja_source": _summary(roundtrip_ja_source_distances),
            },
            "elapsed_ms": _summary(
                [float(r["elapsed_ms"]) for r in translate_rows if not r.get("error")]
            ),
        },
        "tts": {
            "count": len(tts_rows),
            "errors": len(tts_errors),
            "asr_validation_errors": len(tts_asr_errors),
            "asr_distance": _summary(tts_asr_distances),
            "audio_s": _summary([float(r["audio_s"]) for r in tts_rows if not r.get("error")]),
            "elapsed_ms": _summary(
                [float(r["elapsed_ms"]) for r in tts_rows if not r.get("error")]
            ),
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    language_pair = [part.strip().lower() for part in args.language_pair.split(",") if part.strip()]
    legacy_rows: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    legacy_wall_ms = 0.0
    current_wall_ms = 0.0
    audio_s = 0.0

    for meeting_id in args.meeting_ids:
        audio = _load_pcm(meeting_id)
        ref_rows = _load_reference_rows(meeting_id)
        audio_s += len(audio) / SAMPLE_RATE
        legacy = await _run_legacy_endpoint(
            meeting_id=meeting_id,
            audio=audio,
            ref_rows=ref_rows,
            asr_url=args.asr_url,
            language_pair=language_pair,
            chunk_seconds=args.asr_buffer_seconds,
        )
        current = await _run_current_production(
            meeting_id=meeting_id,
            audio=audio,
            ref_rows=ref_rows,
            asr_url=args.asr_url,
            language_pair=language_pair,
            chunk_seconds=args.input_chunk_seconds,
        )
        legacy_rows.extend(legacy["rows"])
        current_rows.extend(current["rows"])
        legacy_wall_ms += float(legacy["wall_ms"])
        current_wall_ms += float(current["wall_ms"])

    capture_dir = args.capture_tts_dir
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)

    downstream_legacy = await _run_downstream(
        legacy_rows,
        asr_url=args.asr_url,
        translate_url=args.translate_url,
        tts_url=args.tts_url,
        language_pair=language_pair,
        max_segments=args.downstream_segments,
        capture_dir=capture_dir / "legacy" if capture_dir else None,
        label="legacy",
        validate_tts_asr=args.validate_tts_asr,
    )
    downstream_current = await _run_downstream(
        current_rows,
        asr_url=args.asr_url,
        translate_url=args.translate_url,
        tts_url=args.tts_url,
        language_pair=language_pair,
        max_segments=args.downstream_segments,
        capture_dir=capture_dir / "current" if capture_dir else None,
        label="current",
        validate_tts_asr=args.validate_tts_asr,
    )
    legacy_quality = _asr_quality(legacy_rows, audio_s, legacy_wall_ms)
    current_quality = _asr_quality(current_rows, audio_s, current_wall_ms)
    delta = {
        "events": current_quality["events"] - legacy_quality["events"],
        "reference_lang_mismatch": current_quality["reference_lang_mismatch"]
        - legacy_quality["reference_lang_mismatch"],
        "script_tag_mismatch": current_quality["script_tag_mismatch"]
        - legacy_quality["script_tag_mismatch"],
        "low_value_english_fragments": current_quality["low_value_english_fragments"]
        - legacy_quality["low_value_english_fragments"],
        "tts_candidate_chars": current_quality["tts_candidate_chars"]
        - legacy_quality["tts_candidate_chars"],
        "realtime_factor": (
            current_quality["realtime_factor"] - legacy_quality["realtime_factor"]
            if current_quality["realtime_factor"] is not None
            and legacy_quality["realtime_factor"] is not None
            else None
        ),
    }
    passed = (
        current_quality["reference_lang_mismatch"] <= legacy_quality["reference_lang_mismatch"]
        and current_quality["script_tag_mismatch"] <= legacy_quality["script_tag_mismatch"]
        and current_quality["low_value_english_fragments"]
        <= legacy_quality["low_value_english_fragments"]
        and downstream_current["translation"]["errors"] == 0
        and downstream_current["translation"]["wrong_script"] == 0
        and downstream_current["tts"]["errors"] == 0
    )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "meeting_count": len(args.meeting_ids),
        "language_pair": language_pair,
        "passed": passed,
        "asr": {
            "legacy_endpoint": legacy_quality,
            "current_production": current_quality,
            "delta_current_minus_legacy": delta,
        },
        "downstream": {
            "legacy_endpoint": downstream_legacy,
            "current_production": downstream_current,
            "delta_current_minus_legacy": {
                "translation_count": downstream_current["translation"]["count"]
                - downstream_legacy["translation"]["count"],
                "translation_errors": downstream_current["translation"]["errors"]
                - downstream_legacy["translation"]["errors"],
                "translation_wrong_script": downstream_current["translation"]["wrong_script"]
                - downstream_legacy["translation"]["wrong_script"],
                "tts_count": downstream_current["tts"]["count"] - downstream_legacy["tts"]["count"],
                "tts_errors": downstream_current["tts"]["errors"]
                - downstream_legacy["tts"]["errors"],
                "tts_asr_distance_p95": (
                    downstream_current["tts"]["asr_distance"]["p95"]
                    - downstream_legacy["tts"]["asr_distance"]["p95"]
                    if downstream_current["tts"]["asr_distance"]["p95"] is not None
                    and downstream_legacy["tts"]["asr_distance"]["p95"] is not None
                    else None
                ),
                "tts_elapsed_p95_ms": (
                    downstream_current["tts"]["elapsed_ms"]["p95"]
                    - downstream_legacy["tts"]["elapsed_ms"]["p95"]
                    if downstream_current["tts"]["elapsed_ms"]["p95"] is not None
                    and downstream_legacy["tts"]["elapsed_ms"]["p95"] is not None
                    else None
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-ids", nargs="+", required=True)
    parser.add_argument("--language-pair", default="en,ja")
    parser.add_argument("--asr-url", default="http://localhost:8003")
    parser.add_argument("--translate-url", default="http://localhost:8010")
    parser.add_argument("--tts-url", default="http://localhost:8002")
    parser.add_argument("--asr-buffer-seconds", type=float, default=3.5)
    parser.add_argument("--input-chunk-seconds", type=float, default=0.25)
    parser.add_argument("--downstream-segments", type=int, default=16)
    parser.add_argument("--capture-tts-dir", type=Path, default=None)
    parser.add_argument("--validate-tts-asr", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/meeting_scribe_production_replay_quality.json")
    )
    args = parser.parse_args()

    report = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report saved: {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
