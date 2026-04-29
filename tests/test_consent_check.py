"""Unit tests for benchmarks/_consent_check.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from benchmarks._consent_check import check_manifest, enforce_consent


def _write_manifest(tmp: Path, samples: list[dict]) -> Path:
    p = tmp / "MANIFEST.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "samples": samples}))
    return p


def test_valid_cloned_ref_entry_passes(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "voice_donor_alice_001",
                "kind": "tts_cloned_ref",
                "language": "en",
                "duration_seconds": 6.0,
                "sha256": "0" * 64,
                "description": "CC-BY voice donation, donor alice",
                "consent": True,
                "consent_provenance": "Kyutai Voice Donation Project, donor 001",
            }
        ],
    )
    assert check_manifest(manifest) == []
    enforce_consent(manifest)


def test_missing_consent_field_fails(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "voice_unknown",
                "kind": "tts_cloned_ref",
                "sha256": "0" * 64,
                "description": "missing consent",
            }
        ],
    )
    violations = check_manifest(manifest)
    assert len(violations) == 2  # missing consent AND missing consent_provenance
    assert any("consent field missing or not literally true" in v for v in violations)
    with pytest.raises(SystemExit):
        enforce_consent(manifest)


def test_consent_false_fails(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "voice_revoked",
                "kind": "tts_cloned_ref",
                "sha256": "0" * 64,
                "description": "consent withdrawn",
                "consent": False,
                "consent_provenance": "Ticket SCRIBE-1234",
            }
        ],
    )
    violations = check_manifest(manifest)
    assert len(violations) == 1
    assert "consent field missing or not literally true" in violations[0]


def test_truthy_non_bool_consent_fails(tmp_path: Path) -> None:
    """Reject `consent: yes` / `consent: 1` — must be the literal Python True."""
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "voice_loose",
                "kind": "tts_cloned_ref",
                "sha256": "0" * 64,
                "description": "loose consent typing",
                "consent": 1,
                "consent_provenance": "Ticket SCRIBE-1234",
            }
        ],
    )
    violations = check_manifest(manifest)
    assert len(violations) == 1
    assert "not literally true" in violations[0]


def test_empty_provenance_fails(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "voice_no_prov",
                "kind": "tts_cloned_ref",
                "sha256": "0" * 64,
                "description": "consent yes, provenance missing",
                "consent": True,
                "consent_provenance": "   ",
            }
        ],
    )
    violations = check_manifest(manifest)
    assert len(violations) == 1
    assert "consent_provenance must be a non-empty string" in violations[0]


def test_non_cloned_ref_entry_ignored(tmp_path: Path) -> None:
    """ASR / TTS-studio / translate samples don't need consent fields."""
    manifest = _write_manifest(
        tmp_path,
        [
            {"id": "asr_clip_001", "kind": "asr", "sha256": "0" * 64, "description": "x"},
            {"id": "tts_studio_001", "kind": "tts_studio", "sha256": "0" * 64, "description": "x"},
            {"id": "translate_001", "kind": "translate", "sha256": "0" * 64, "description": "x"},
        ],
    )
    assert check_manifest(manifest) == []
    enforce_consent(manifest)
