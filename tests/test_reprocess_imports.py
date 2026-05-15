"""Regression: the reprocess speaker-data phase imports the right module.

2026-05-07: a full reprocess on meeting `e6b1a2a6-...` left
``timeline.json`` + ``speaker_lanes.json`` STALE because the reprocess
phase that regenerates them silently failed:

    WARNING reprocess: _generate_speaker_data/_generate_timeline
    failed: module 'meeting_scribe.server' has no attribute
    '_generate_speaker_data'

The functions had been moved to
``meeting_scribe.server_support.meeting_artifacts`` during the v2.0.0
refactor, but ``reprocess.py`` still loaded them off
``meeting_scribe.server`` via ``importlib.import_module(...)``. The
``except Exception`` swallowed the AttributeError, so the bug only
surfaced when the user reported "clicking the diarize lane doesn't
move the transcript anymore" — the new journal had fresh segment_ids,
the stale lanes had old ones, and the click handler's segment-id
lookup missed every time.

This test pins the imports so a future reorg can't silently re-break
it. If `_generate_speaker_data` / `_generate_timeline` move again,
this test fails LOUDLY at collection time (not at run time inside a
swallowed try/except).
"""

from __future__ import annotations

import importlib


def test_reprocess_imports_speaker_data_helpers_from_canonical_location() -> None:
    """``reprocess.py`` must be able to call ``_generate_speaker_data`` and
    ``_generate_timeline`` from ``server_support.meeting_artifacts`` —
    the home those functions moved to in the v2.0.0 refactor."""
    mod = importlib.import_module("meeting_scribe.server_support.meeting_artifacts")
    # Both helpers must exist + be callable.
    assert callable(mod._generate_speaker_data), (
        "meeting_scribe.server_support.meeting_artifacts._generate_speaker_data is "
        "missing or non-callable. reprocess.py imports it from here; if it moves "
        "again, fix reprocess.py and update this test."
    )
    assert callable(mod._generate_timeline), (
        "meeting_scribe.server_support.meeting_artifacts._generate_timeline is "
        "missing or non-callable. reprocess.py imports it from here; if it moves "
        "again, fix reprocess.py and update this test."
    )


def test_reprocess_speaker_data_phase_uses_canonical_imports() -> None:
    """Belt-and-braces: read reprocess.py source and confirm it
    imports from ``server_support.meeting_artifacts`` (not
    ``meeting_scribe.server``). The v2.0.0 refactor's leftover
    ``importlib.import_module('meeting_scribe.server')`` silently
    AttributeError'd inside an ``except`` — the test catches that
    pattern by name."""
    from pathlib import Path

    src = (Path(__file__).parent.parent / "src" / "meeting_scribe" / "reprocess.py").read_text()

    # The fix introduces this exact import block. Pin it.
    assert "from meeting_scribe.server_support.meeting_artifacts import" in src, (
        "reprocess.py speaker_data phase no longer imports from "
        "server_support.meeting_artifacts — that's where the helpers "
        "live. If this is intentional, update both the source and this "
        "test together."
    )
    # And it must NOT use the buggy importlib indirection that hid the
    # AttributeError last time.
    assert 'importlib.import_module("meeting_scribe.server")' not in src, (
        "reprocess.py is back to dynamic-importing meeting_scribe.server. "
        "That indirection let an AttributeError on `_generate_speaker_data` "
        "get swallowed by `except Exception`, leaving timeline.json + "
        "speaker_lanes.json stale on every full-reprocess. Use a direct "
        "import from server_support.meeting_artifacts."
    )
