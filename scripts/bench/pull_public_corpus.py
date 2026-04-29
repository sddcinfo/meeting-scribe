"""Phase A1 — Pull a curated public-domain JA / EN ASR corpus on disk.

Triggers downloads via ``huggingface_hub.snapshot_download`` (the
``datasets`` Python package is intentionally not used: pulling specific
files via the bare hub API keeps the dep surface in the meeting-scribe
venv flat and gives us reproducibility through pinned revisions).

Targets:
    google/fleurs (Apache-2.0, ungated) — ja_jp + en_us test splits
        * data/<lang>/test.tar.gz   raw audio
        * data/<lang>/test.tsv      audio_id<TAB>raw_transcription<TAB>...

Output layout (under ``--target-dir`` which must resolve outside the
repo, per the offline-bench-paths rule from the 2026-Q2 plan):

    <target-dir>/asr/
        <id>.wav                  # 16 kHz mono s16le WAV
        <id>.txt                  # reference transcription (NFKC-normalised)

Per fixture, append a row to ``benchmarks/fixtures/meeting_consolidation/MANIFEST.yaml``::

    - id: fleurs_ja_jp_<idx>
      kind: asr
      language: ja
      source_corpus: google/fleurs:ja_jp:test@<revision_sha>
      duration_seconds: <float>
      sha256: <hex>
      description: "Fleurs JA test split, utterance <idx>"

Usage::

    python3 scripts/bench/pull_public_corpus.py \\
        --target-dir /data/meeting-scribe-fixtures \\
        --languages ja_jp en_us \\
        --per-language 100
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import tarfile
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# benchmarks/_bench_paths.assert_offline_path() — refuse in-repo --target-dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks._bench_paths import assert_offline_path  # noqa: E402

logger = logging.getLogger("pull_public_corpus")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FLEURS_REPO = "google/fleurs"
TARGET_SAMPLE_RATE = 16_000


@dataclass
class FleursClip:
    audio_id: str  # e.g. "1.wav"
    transcription: str
    audio_bytes: bytes  # raw bytes from the tar (any sample rate)


def _resolve_revision(repo_id: str, repo_type: str = "dataset") -> str:
    """Return the current main-branch commit SHA for traceability."""
    from huggingface_hub import HfApi

    api = HfApi()
    refs = api.list_repo_refs(repo_id, repo_type=repo_type)
    for branch in refs.branches:
        if branch.name == "main":
            return branch.target_commit
    raise RuntimeError(f"could not resolve main branch for {repo_id}")


def _download_split(
    repo_id: str, split_lang: str, revision: str, cache_dir: Path
) -> tuple[Path, Path]:
    """Pull just `data/<split_lang>/test.tsv` + `test.tar.gz` for one language."""
    from huggingface_hub import snapshot_download

    pattern_root = f"data/{split_lang}"
    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[
            f"{pattern_root}/test.tsv",
            f"{pattern_root}/audio/test.tar.gz",
        ],
        cache_dir=str(cache_dir),
    )
    tsv = Path(local) / pattern_root / "test.tsv"
    tar = Path(local) / pattern_root / "audio" / "test.tar.gz"
    if not tsv.exists() or not tar.exists():
        raise RuntimeError(f"expected fleurs {split_lang} files missing under {local}")
    return tsv, tar


def _parse_fleurs_tsv(tsv: Path, limit: int) -> dict[str, str]:
    """Return ``{audio_id: transcription}`` for up to ``limit`` rows."""
    out: dict[str, str] = {}
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        # Fleurs rows: id, audio_id, raw_transcription, transcription, num_samples, gender
        audio_id = parts[1]
        # Prefer the cleaned transcription (col 3); fall back to raw if absent.
        text = parts[3] if len(parts) > 3 else parts[2]
        text = unicodedata.normalize("NFKC", text).strip()
        if not text:
            continue
        out[audio_id] = text
        if len(out) >= limit:
            break
    return out


def _resample_to_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    """Linear-resample to 16 kHz mono s16le.  Same approach as
    ``scripts/bench/asr_ja_wer_run.py:_pcm_to_wav`` so the JA WER bench
    sees identical pre-processing on synth and real corpora."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr == TARGET_SAMPLE_RATE:
        # Nothing to do — keep as float and quantize at write time.
        return audio.astype(np.float32)
    n_target = round(len(audio) * TARGET_SAMPLE_RATE / sr)
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_target, endpoint=False)
    return np.interp(x_new, x_old, audio.astype(np.float32))


def _write_wav(path: Path, audio_f32: np.ndarray) -> tuple[float, str]:
    """Write a 16 kHz s16le WAV.  Returns (duration_seconds, sha256_hex)."""
    samples = np.clip(audio_f32, -1.0, 1.0)
    samples_i16 = (samples * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(samples_i16.tobytes())
    duration = len(samples_i16) / TARGET_SAMPLE_RATE
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return duration, sha


def _iter_fleurs_clips(
    tar_path: Path, transcriptions: dict[str, str]
) -> "list[FleursClip]":
    """Walk the tar archive and yield clips matching the transcription
    table.  Audio inside Fleurs tars is named like ``<idx>.wav``."""
    out: list[FleursClip] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            base = Path(member.name).name
            if base not in transcriptions:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            out.append(
                FleursClip(
                    audio_id=base,
                    transcription=transcriptions[base],
                    audio_bytes=f.read(),
                )
            )
            if len(out) >= len(transcriptions):
                break
    return out


def _build_manifest_rows(
    clips: "list[FleursClip]",
    *,
    asr_dir: Path,
    split_lang: str,
    revision: str,
    iso_lang: str,
) -> list[dict]:
    rows: list[dict] = []
    for i, clip in enumerate(clips):
        # Decode the in-memory audio (any sample rate / format soundfile knows).
        audio, sr = sf.read(io.BytesIO(clip.audio_bytes), dtype="float32")
        audio_16k = _resample_to_16k_mono(audio, sr)
        cid = f"fleurs_{split_lang}_{i:03d}"
        wav_path = asr_dir / f"{cid}.wav"
        txt_path = asr_dir / f"{cid}.txt"
        duration_s, sha = _write_wav(wav_path, audio_16k)
        txt_path.write_text(clip.transcription + "\n", encoding="utf-8")
        rows.append(
            {
                "id": cid,
                "kind": "asr",
                "language": iso_lang,
                "source_corpus": f"{FLEURS_REPO}:{split_lang}:test@{revision}",
                "duration_seconds": round(duration_s, 3),
                "sha256": sha,
                "description": f"Fleurs {split_lang} test split, utterance {i:03d}",
            }
        )
        if (i + 1) % 25 == 0:
            logger.info("[%s] wrote %d / %d", split_lang, i + 1, len(clips))
    return rows


def _append_manifest_rows(manifest_path: Path, new_rows: list[dict]) -> None:
    """Append ``new_rows`` to ``samples:`` in ``manifest_path``.

    We deliberately avoid `yaml.safe_dump` of the whole file because it
    reflows comments + ordering — we write rows as YAML fragments
    appended to the file (preserving the existing manifest header and
    any prior entries).  The validator at
    ``benchmarks/_consent_check.py`` doesn't care about ordering.
    """
    import yaml

    text = manifest_path.read_text()
    if "\nsamples:" not in text and not text.rstrip().endswith("samples: []"):
        raise SystemExit(
            f"Manifest {manifest_path} does not appear to have a 'samples:' "
            f"key; refusing to corrupt it.  Inspect manually."
        )
    if text.rstrip().endswith("samples: []"):
        text = text.rstrip()[: -len("samples: []")] + "samples:\n"
    fragment_lines = ["\n# --- pulled by scripts/bench/pull_public_corpus.py ---"]
    for row in new_rows:
        # Render each row as a YAML list item with leading dash + indented keys.
        rendered = yaml.safe_dump([row], allow_unicode=True, sort_keys=False)
        fragment_lines.append(rendered.rstrip())
    manifest_path.write_text(text + "\n".join(fragment_lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir", type=Path, required=True)
    p.add_argument(
        "--languages",
        nargs="+",
        default=["ja_jp", "en_us"],
        help="Fleurs split lang codes to pull (default: ja_jp en_us).",
    )
    p.add_argument("--per-language", type=int, default=100)
    p.add_argument("--cache-dir", type=Path, default=Path("/data/huggingface"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "fixtures"
        / "meeting_consolidation"
        / "MANIFEST.yaml",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip downloads; only verify sha256 of every Fleurs row in the manifest.",
    )
    args = p.parse_args()

    target_dir = assert_offline_path(args.target_dir)
    asr_dir = target_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        return _verify(args.manifest, asr_dir)

    revision = _resolve_revision(FLEURS_REPO, repo_type="dataset")
    logger.info("Fleurs revision pinned at %s", revision)

    all_rows: list[dict] = []
    for split_lang in args.languages:
        iso_lang = split_lang.split("_", 1)[0]  # "ja_jp" → "ja"
        logger.info("Downloading Fleurs %s split=test ...", split_lang)
        tsv, tar = _download_split(FLEURS_REPO, split_lang, revision, args.cache_dir)
        transcriptions = _parse_fleurs_tsv(tsv, limit=args.per_language)
        logger.info("[%s] %d transcriptions selected", split_lang, len(transcriptions))
        clips = _iter_fleurs_clips(tar, transcriptions)
        logger.info("[%s] %d clips matched in tar", split_lang, len(clips))
        rows = _build_manifest_rows(
            clips,
            asr_dir=asr_dir,
            split_lang=split_lang,
            revision=revision,
            iso_lang=iso_lang,
        )
        all_rows.extend(rows)
        logger.info("[%s] wrote %d clips", split_lang, len(rows))

    if all_rows:
        _append_manifest_rows(args.manifest, all_rows)
        logger.info("Appended %d manifest rows to %s", len(all_rows), args.manifest)
    return 0


def _verify(manifest_path: Path, asr_dir: Path) -> int:
    import yaml

    data = yaml.safe_load(manifest_path.read_text()) or {}
    fail = 0
    for entry in data.get("samples", []) or []:
        if entry.get("kind") != "asr":
            continue
        if not str(entry.get("id", "")).startswith("fleurs_"):
            continue
        wav = asr_dir / f"{entry['id']}.wav"
        if not wav.exists():
            logger.error("MISSING %s", wav)
            fail += 1
            continue
        actual = hashlib.sha256(wav.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            logger.error(
                "DRIFT  %s — manifest=%s actual=%s",
                wav,
                entry["sha256"][:16],
                actual[:16],
            )
            fail += 1
        else:
            logger.info("OK     %s", wav.name)
    if fail:
        logger.error("verify failed on %d entries", fail)
        return 2
    logger.info("all Fleurs rows verified clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
