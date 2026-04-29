"""Coverage gate for the server-side WebSocket event registry.

Three invariants enforced:

  1. Every value in ``WsEventType`` has a sample at
     ``tests/contracts/ws_event_samples/<type>.json``, the sample's
     ``type`` field matches, and the sample is parseable JSON.

  2. Every server-side call to ``_broadcast_json({...})`` uses a
     ``type`` value that lives in ``WsEventType`` (AST grep over
     ``src/meeting_scribe/``).

  3. The runtime guard in ``_broadcast_json`` raises in strict mode
     for any unknown ``type`` value (smoke test of the validator).

If a future change adds a new emit site without updating the registry
+ samples + JS handler cascades, one of these tests fails before the
PR can merge — converting the popout-clear-on-pulse class of bugs
from a runtime regression into a build-time error.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from meeting_scribe.ws.event_types import WS_EVENT_TYPES, WsEventType

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "tests" / "contracts" / "ws_event_samples"
SRC_DIR = REPO_ROOT / "src" / "meeting_scribe"


def test_every_registered_type_has_a_sample():
    """Every WsEventType value has a sample file with matching ``type``."""
    missing: list[str] = []
    mismatched: list[str] = []
    for et in WsEventType:
        path = SAMPLES_DIR / f"{et.value}.json"
        if not path.exists():
            missing.append(et.value)
            continue
        sample = json.loads(path.read_text())
        if sample.get("type") != et.value:
            mismatched.append(
                f"{et.value} (sample.type={sample.get('type')!r})"
            )
    assert not missing, (
        f"Missing samples for: {missing}. "
        f"Add tests/contracts/ws_event_samples/<type>.json for each."
    )
    assert not mismatched, (
        f"Sample 'type' fields don't match filename: {mismatched}"
    )


def test_every_sample_corresponds_to_a_registered_type():
    """Every sample file maps to a value in ``WsEventType``.

    Catches dangling samples for removed event types — keeps the
    sample dir an exact mirror of the registry.
    """
    if not SAMPLES_DIR.exists():
        pytest.skip("samples dir missing")
    orphans: list[str] = []
    for path in SAMPLES_DIR.glob("*.json"):
        slug = path.stem
        if slug not in WS_EVENT_TYPES:
            orphans.append(slug)
    assert not orphans, (
        f"Sample files for unregistered types: {orphans}. "
        f"Either add them to WsEventType or delete the samples."
    )


def _collect_broadcast_type_literals() -> dict[Path, set[str]]:
    """AST-walk ``src/`` and find every string literal at the ``type`` key
    of a dict passed to a function whose name contains ``broadcast``.

    Returns ``{file_path: {literal_type_value, ...}}``.
    """
    found: dict[Path, set[str]] = {}
    for path in SRC_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match foo.bar() or foo() where the name contains 'broadcast'
            fn_name = ""
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fn_name = node.func.attr
            if "broadcast" not in fn_name.lower():
                continue
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    for key, value in zip(arg.keys, arg.values, strict=False):
                        if (
                            isinstance(key, ast.Constant)
                            and key.value == "type"
                            and isinstance(value, ast.Constant)
                            and isinstance(value.value, str)
                        ):
                            found.setdefault(path, set()).add(value.value)
    return found


def test_every_broadcast_call_site_uses_a_registered_type():
    """Every ``_broadcast_json({"type": "...", ...})`` call site uses a
    ``type`` value present in ``WsEventType``.

    Catches the failure mode that the runtime validator only WARNs on
    in dev mode — at PR-time we want a hard fail.
    """
    found = _collect_broadcast_type_literals()
    offenders: dict[Path, set[str]] = {}
    for path, types in found.items():
        bad = {t for t in types if t not in WS_EVENT_TYPES}
        if bad:
            offenders[path] = bad
    assert not offenders, (
        f"Unregistered event types broadcast from these files: "
        f"{ {str(p.relative_to(REPO_ROOT)): list(v) for p, v in offenders.items()} }. "
        f"Add to meeting_scribe/ws/event_types.py + a sample file."
    )


def test_broadcast_strict_mode_rejects_unknown_type(monkeypatch):
    """``WS_EVENT_TYPES_STRICT=1`` turns the dev-mode warning into a hard
    error. The validator is sync and pure — call it directly so this
    test stays sync and doesn't touch the asyncio loop (which would
    conflict with pytest-playwright when run alongside browser tests).
    """
    from meeting_scribe.server_support.broadcast import _validate_event_type

    monkeypatch.setenv("WS_EVENT_TYPES_STRICT", "1")
    with pytest.raises(ValueError, match="unregistered event type"):
        _validate_event_type({"type": "this_is_not_a_real_event_type"})


def test_broadcast_strict_mode_accepts_registered_type(monkeypatch):
    """The strict guard must not flag a known type."""
    from meeting_scribe.server_support.broadcast import _validate_event_type

    monkeypatch.setenv("WS_EVENT_TYPES_STRICT", "1")
    # Should NOT raise.
    _validate_event_type({"type": WsEventType.SPEAKER_PULSE.value})


def test_validator_passes_untyped_payload(monkeypatch):
    """TranscriptEvent dicts (no `type` field) must always pass."""
    from meeting_scribe.server_support.broadcast import _validate_event_type

    monkeypatch.setenv("WS_EVENT_TYPES_STRICT", "1")
    _validate_event_type({"segment_id": "seg-1", "text": "hi", "language": "en"})
