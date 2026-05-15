"""Symmetric redaction for the 2026-Q2 ASR bench supplemental corpus.

For internal meeting clips that contain PII, the WER gate is only
defensible if the **same redaction** applied to the reference
transcript is also applied to the model hypothesis before scoring.
Otherwise the audio still says "John Smith" but the reference says
"[NAME]" — Cohere and Qwen3 both transcribe "John Smith" and lose
points artificially.

The redaction map itself is **not committed** to the repo (it would
re-introduce the very PII we are protecting against).  This module
loads the map at runtime from a side-loaded YAML at
``$SCRIBE_REDACTION_MAP`` (default ``/data/meeting-scribe-fixtures/redaction-map.yaml``).
The map shape is::

    rules:
      - pattern: "John Smith"      # plain string OR /regex/
        mask: "[NAME]"
      - pattern: "/\\b\\d{4}-\\d{4}\\b/"
        mask: "[ACCOUNT]"

Both reference-prep and hypothesis-scoring import ``mask(text)`` and
get identical output.  The primary public-domain corpus skips this
module entirely — it has no PII to redact.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml required: pip install pyyaml") from exc


DEFAULT_MAP_PATH = Path(
    os.environ.get(
        "SCRIBE_REDACTION_MAP",
        "/data/meeting-scribe-fixtures/redaction-map.yaml",
    )
)


@dataclass(frozen=True)
class _Rule:
    matcher: re.Pattern[str]
    mask: str


def _compile_rules(raw: list[dict]) -> list[_Rule]:
    rules: list[_Rule] = []
    for entry in raw:
        pattern = entry["pattern"]
        mask = entry["mask"]
        if pattern.startswith("/") and pattern.endswith("/") and len(pattern) > 2:
            rules.append(_Rule(re.compile(pattern[1:-1]), mask))
        else:
            rules.append(_Rule(re.compile(re.escape(pattern)), mask))
    return rules


_RULES_CACHE: list[_Rule] | None = None


def load_rules(map_path: Path = DEFAULT_MAP_PATH) -> list[_Rule]:
    global _RULES_CACHE
    if _RULES_CACHE is not None:
        return _RULES_CACHE
    if not map_path.exists():
        raise FileNotFoundError(
            f"Redaction map not found at {map_path}.  Set SCRIBE_REDACTION_MAP "
            f"or place the map at the default offline location.  The redaction "
            f"map is never committed to the repo."
        )
    raw = yaml.safe_load(map_path.read_text()) or {}
    _RULES_CACHE = _compile_rules(raw.get("rules", []))
    return _RULES_CACHE


def mask(text: str, map_path: Path = DEFAULT_MAP_PATH) -> str:
    """Apply every rule in order.  Returns the masked text.

    Used identically on reference transcripts (at corpus prep time) and
    on model hypotheses (at scoring time) so WER is computed against
    aligned masked strings.
    """
    out = text
    for rule in load_rules(map_path):
        out = rule.matcher.sub(rule.mask, out)
    return out


def reset_cache_for_tests() -> None:
    """Test-only hook: clear the lazy rules cache so a fresh map can be loaded."""
    global _RULES_CACHE
    _RULES_CACHE = None
