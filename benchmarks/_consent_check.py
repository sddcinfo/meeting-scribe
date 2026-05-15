"""Consent validator for the 2026-Q2 model-challenger bench.

Voice-cloning references are only safe to bench if the speaker
explicitly consented to that use.  Every ``tts_cloned_ref`` entry in
``benchmarks/fixtures/meeting_consolidation/MANIFEST.yaml`` must carry
both ``consent: true`` and a non-empty ``consent_provenance`` string
(free-text identifier of where the consent record lives — e.g. an
internal ticket id, a contract section, a corpus licence reference).

This module is imported as the first step of every TTS bench script.
A missing or false field hard-fails the run before any synthesis.

CLI usage:

    python3 benchmarks/_consent_check.py [--manifest PATH]

Library usage:

    from benchmarks._consent_check import enforce_consent
    enforce_consent()  # raises SystemExit on any violation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml required: pip install pyyaml") from exc


DEFAULT_MANIFEST = Path(__file__).parent / "fixtures" / "meeting_consolidation" / "MANIFEST.yaml"


class ConsentViolation(Exception):
    """Raised when a tts_cloned_ref entry lacks consent metadata."""


def _validate_entry(entry: dict) -> list[str]:
    """Return a list of violation strings for one manifest entry; empty if OK."""
    if entry.get("kind") != "tts_cloned_ref":
        return []
    sample_id = entry.get("id", "<missing-id>")
    problems: list[str] = []
    if entry.get("consent") is not True:
        problems.append(
            f"sample {sample_id!r}: consent field missing or not literally true "
            f"(got {entry.get('consent')!r})"
        )
    provenance = entry.get("consent_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        problems.append(
            f"sample {sample_id!r}: consent_provenance must be a non-empty string "
            f"(got {provenance!r})"
        )
    return problems


def check_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    """Load ``manifest_path`` and return a list of violation strings."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text()) or {}
    violations: list[str] = []
    for entry in data.get("samples", []) or []:
        violations.extend(_validate_entry(entry))
    return violations


def enforce_consent(manifest_path: Path = DEFAULT_MANIFEST) -> None:
    """Raise SystemExit if any tts_cloned_ref entry lacks consent metadata."""
    violations = check_manifest(manifest_path)
    if not violations:
        return
    msg = "\n".join(["Consent check FAILED:"] + [f"  - {v}" for v in violations])
    raise SystemExit(msg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = p.parse_args(argv)
    enforce_consent(args.manifest)
    print(f"Consent check OK: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
