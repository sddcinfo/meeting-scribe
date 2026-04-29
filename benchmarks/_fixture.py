"""Shared fixture loader for Omni consolidation benchmarks.

Resolves samples by ID from MANIFEST.yaml, checksums them against the
on-disk files under /data/meeting-scribe-fixtures/ (or --fixture-dir),
and fails loudly if anything drifts. Never logs transcript text.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml required: pip install pyyaml") from exc


MANIFEST_PATH = Path(__file__).parent / "fixtures" / "meeting_consolidation" / "MANIFEST.yaml"


@dataclass(frozen=True)
class Sample:
    id: str
    kind: str
    language: str
    duration_seconds: float | None
    sha256: str
    description: str
    path: Path


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_samples(
    fixture_dir: Path,
    kind: str | None = None,
    language: str | None = None,
    *,
    verify_checksums: bool = True,
) -> list[Sample]:
    """Load and verify manifest samples from ``fixture_dir``.

    :param fixture_dir: path to the on-disk fixture root (e.g.
        /data/meeting-scribe-fixtures/).
    :param kind: optional filter — asr | tts_studio | tts_cloned_ref | translate.
    :param language: optional ISO-639-1 language filter.
    :param verify_checksums: when True (default), fail if any file's
        sha256 disagrees with the manifest.
    """
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    out: list[Sample] = []
    for entry in manifest.get("samples", []):
        if kind and entry["kind"] != kind:
            continue
        if language and entry.get("language") != language:
            continue
        path = fixture_dir / entry["kind"] / f"{entry['id']}.wav"
        if not path.exists():
            # Transcripts and ref JSON live next to the audio; harnesses
            # look them up themselves. Absence of audio is fatal.
            raise FileNotFoundError(f"Manifest sample {entry['id']!r} missing at {path}")
        if verify_checksums:
            actual = _sha256_file(path)
            if actual != entry["sha256"]:
                raise ValueError(
                    f"Checksum drift for {entry['id']!r}: "
                    f"manifest={entry['sha256']} actual={actual}"
                )
        out.append(
            Sample(
                id=entry["id"],
                kind=entry["kind"],
                language=entry.get("language", ""),
                duration_seconds=entry.get("duration_seconds"),
                sha256=entry["sha256"],
                description=entry.get("description", ""),
                path=path,
            )
        )
    return out


def add_fixture_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("/data/meeting-scribe-fixtures/"),
        help="Root of the on-disk fixture store. Falls back to the plan default.",
    )
