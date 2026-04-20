"""Bootstrap the Omni-consolidation fixture from a captured meeting.

The Omni consolidation plan (Phase A) requires consented audio + ground
truth transcripts under /data/meeting-scribe-fixtures/ so ASR / translate
/ TTS / diarize harnesses can run against real numbers. This script does
the mechanical work:

1. Slices the meeting's recording.pcm into fixture-ready WAV clips aligned
   to journal finals (one clip per final, start_ms → end_ms).
2. Writes a ground-truth transcript `<id>.txt` next to each clip from the
   highest-revision `text` in the journal.
3. Computes sha256 for every clip + transcript.
4. Appends manifest entries to benchmarks/fixtures/meeting_consolidation/MANIFEST.yaml.

Raw audio + transcripts are NEVER written under the repo — they go to
the fixture root the plan specifies, default `/data/meeting-scribe-fixtures/`.
The manifest in-repo only holds opaque IDs + checksums + durations.

Usage:
    python benchmarks/bootstrap_fixture_from_meeting.py \\
        --meeting-dir meetings/99c4f3f8-29c0-4067-83e8-e9ed8e0d41a9 \\
        --fixture-dir /data/meeting-scribe-fixtures \\
        --kind asr \\
        --limit 40
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
import wave
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml required: pip install pyyaml") from exc

import numpy as np

REPO_MANIFEST = (
    Path(__file__).parent / "fixtures" / "meeting_consolidation" / "MANIFEST.yaml"
)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_pcm(pcm_path: Path, sample_rate: int) -> np.ndarray:
    """Load raw s16le mono PCM into a float32 array at `sample_rate`."""
    raw = pcm_path.read_bytes()
    return np.frombuffer(raw, dtype=np.int16)


def _extract_final_segments(journal_path: Path) -> list[dict]:
    """Return the highest-revision is_final entry per segment_id, sorted by start."""
    best: dict[str, dict] = {}
    for line in journal_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if not e.get("is_final") or not e.get("text"):
            continue
        sid = e.get("segment_id")
        if not sid:
            continue
        rev = e.get("revision", 0)
        if sid not in best or rev > best[sid].get("revision", 0):
            best[sid] = e
    return sorted(best.values(), key=lambda e: e.get("start_ms", 0))


def _write_wav(samples: np.ndarray, sample_rate: int, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


def bootstrap(
    meeting_dir: Path,
    fixture_dir: Path,
    kind: str,
    limit: int,
) -> list[dict]:
    pcm_path = meeting_dir / "audio" / "recording.pcm"
    journal_path = meeting_dir / "journal.jsonl"
    meta_path = meeting_dir / "meta.json"
    if not pcm_path.exists():
        raise SystemExit(f"recording.pcm missing under {meeting_dir}")
    if not journal_path.exists():
        raise SystemExit(f"journal.jsonl missing under {meeting_dir}")

    meta = json.loads(meta_path.read_text())
    sample_rate = int(meta.get("audio_sample_rate", 16000))

    all_samples = _load_pcm(pcm_path, sample_rate)
    finals = _extract_final_segments(journal_path)

    # Keep a mix of short + long clips.
    finals_sorted = sorted(
        finals, key=lambda e: e.get("end_ms", 0) - e.get("start_ms", 0)
    )
    shorts = [e for e in finals_sorted if (e["end_ms"] - e["start_ms"]) < 5000]
    longs = [e for e in finals_sorted if (e["end_ms"] - e["start_ms"]) >= 5000]
    picks: list[dict] = []
    picks.extend(shorts[: limit // 2])
    picks.extend(longs[-(limit - len(picks)) :] if longs else [])
    # Fallback: if we're still short, top up with anything we haven't picked.
    picked_ids = {e["segment_id"] for e in picks}
    for e in finals_sorted:
        if len(picks) >= limit:
            break
        if e["segment_id"] not in picked_ids:
            picks.append(e)
            picked_ids.add(e["segment_id"])

    manifest_entries: list[dict] = []
    for e in picks[:limit]:
        start = int(e["start_ms"] * sample_rate / 1000)
        end = int(e["end_ms"] * sample_rate / 1000)
        if end <= start:
            continue
        clip = all_samples[start:end]
        if clip.size < sample_rate // 2:
            continue  # skip sub-0.5s clips — not useful for WER

        opaque_id = uuid.uuid4().hex
        wav_path = fixture_dir / kind / f"{opaque_id}.wav"
        txt_path = wav_path.with_suffix(".txt")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(clip.tobytes())
        wav_bytes = buf.getvalue()

        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(wav_bytes)
        txt_path.write_text((e.get("text") or "").strip() + "\n", encoding="utf-8")

        manifest_entries.append(
            {
                "id": opaque_id,
                "kind": kind,
                "language": e.get("language", "unknown"),
                "duration_seconds": round((e["end_ms"] - e["start_ms"]) / 1000.0, 2),
                "sha256": _sha256_bytes(wav_bytes),
                "description": (
                    f"meeting={meeting_dir.name} segment={e['segment_id']} "
                    f"lang={e.get('language','?')}"
                ),
            }
        )
    return manifest_entries


def append_to_manifest(entries: list[dict]) -> None:
    existing = yaml.safe_load(REPO_MANIFEST.read_text()) or {}
    existing.setdefault("samples", []).extend(entries)
    existing.setdefault("version", 1)
    existing.setdefault("fixture_root_on_disk", "/data/meeting-scribe-fixtures/")
    REPO_MANIFEST.write_text(yaml.safe_dump(existing, sort_keys=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meeting-dir", type=Path, required=True)
    p.add_argument("--fixture-dir", type=Path, default=Path("/data/meeting-scribe-fixtures/"))
    p.add_argument("--kind", choices=["asr", "translate", "tts_cloned_ref", "meeting_e2e"], default="asr")
    p.add_argument("--limit", type=int, default=40)
    args = p.parse_args()

    entries = bootstrap(args.meeting_dir, args.fixture_dir, args.kind, args.limit)
    append_to_manifest(entries)
    print(f"Wrote {len(entries)} fixture samples to {args.fixture_dir / args.kind}")
    print(f"Appended {len(entries)} manifest entries to {REPO_MANIFEST}")


if __name__ == "__main__":
    main()
