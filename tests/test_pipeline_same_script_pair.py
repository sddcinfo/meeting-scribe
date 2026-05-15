"""Regression: same-script language pairs (en↔de, en↔fr, ...) must
NOT be remapped by the script-router.

`_detect_language_from_text` returns ``"en"`` for ANY Latin-script
text — it can't distinguish English from German from French. The
script-router in `pipeline.transcript_event._process_event` was
unconditionally trusting that signal, which forced German text in an
en↔de meeting to be re-labelled as English (and then translated to
German — a no-op, the user saw the German source mirrored back).

The fix is in transcript_event.py: when both languages in the active
pair share the same script class, the script-router has no useful
signal and we trust the ASR + lingua post-correction instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from meeting_scribe.models import TranscriptEvent
from meeting_scribe.pipeline import transcript_event as te


def _mk_event(text: str, lang: str) -> TranscriptEvent:
    return TranscriptEvent(
        segment_id="seg-test",
        text=text,
        language=lang,
        is_final=True,
        start_ms=0,
        end_ms=1000,
        utterance_end_at=1.0,
    )


class _StubMeeting:
    def __init__(self, language_pair):
        self.meeting_id = "mtg-test"
        self.language_pair = language_pair


@pytest.mark.asyncio
async def test_german_text_in_en_de_pair_not_remapped_to_en(monkeypatch):
    """The original bug: 'Ja, hundert Prozent' (German) in en↔de meeting
    was getting remapped from de → en because the Latin script lookup
    returned 'en' as the default for any Latin text. Must keep ASR's
    'de' label."""
    from meeting_scribe.runtime import state

    monkeypatch.setattr(state, "current_meeting", _StubMeeting(["en", "de"]))
    monkeypatch.setattr(state, "translation_queue", None)
    monkeypatch.setattr(state, "asr_backend", None)
    monkeypatch.setattr(state, "furigana_backend", None)
    monkeypatch.setattr(state, "_audio_out_clients", set())
    monkeypatch.setattr(state, "enrollment_store", SimpleNamespace(speakers={}))
    monkeypatch.setattr(state, "storage", SimpleNamespace(append_event=lambda *a, **k: None))

    ev = _mk_event("Ja, hundert Prozent, und so weiter.", "de")

    # Stub broadcast to a no-op
    async def _no_broadcast(*a, **k):
        return None

    with patch.object(te, "_broadcast", _no_broadcast):
        await te._process_event(ev)

    assert ev.language == "de", (
        f"German text in en↔de pair must keep ASR's 'de' label, got {ev.language!r}"
    )


@pytest.mark.asyncio
async def test_english_text_in_en_de_pair_not_remapped(monkeypatch):
    """Symmetric: English text in en↔de meeting keeps 'en'."""
    from meeting_scribe.runtime import state

    monkeypatch.setattr(state, "current_meeting", _StubMeeting(["en", "de"]))
    monkeypatch.setattr(state, "translation_queue", None)
    monkeypatch.setattr(state, "asr_backend", None)
    monkeypatch.setattr(state, "furigana_backend", None)
    monkeypatch.setattr(state, "_audio_out_clients", set())
    monkeypatch.setattr(state, "enrollment_store", SimpleNamespace(speakers={}))
    monkeypatch.setattr(state, "storage", SimpleNamespace(append_event=lambda *a, **k: None))

    ev = _mk_event("This is a regular English sentence.", "en")

    async def _no_broadcast(*a, **k):
        return None

    with patch.object(te, "_broadcast", _no_broadcast):
        await te._process_event(ev)

    assert ev.language == "en"


@pytest.mark.asyncio
async def test_japanese_text_in_ja_en_pair_still_remaps(monkeypatch):
    """The cross-script case (the original feature) still works:
    Japanese kana text in a ja↔en meeting that ASR mistagged as 'en'
    must be remapped to 'ja'."""
    from meeting_scribe.runtime import state

    monkeypatch.setattr(state, "current_meeting", _StubMeeting(["ja", "en"]))
    monkeypatch.setattr(state, "translation_queue", None)
    monkeypatch.setattr(state, "asr_backend", None)
    monkeypatch.setattr(state, "furigana_backend", None)
    monkeypatch.setattr(state, "_audio_out_clients", set())
    monkeypatch.setattr(state, "enrollment_store", SimpleNamespace(speakers={}))
    monkeypatch.setattr(state, "storage", SimpleNamespace(append_event=lambda *a, **k: None))

    ev = _mk_event("こんにちは、元気ですか", "en")  # ASR mistagged

    async def _no_broadcast(*a, **k):
        return None

    with patch.object(te, "_broadcast", _no_broadcast):
        await te._process_event(ev)

    # Mixed-script pair → script router IS useful → kana wins.
    assert ev.language == "ja"


@pytest.mark.asyncio
async def test_out_of_pair_label_in_same_script_pair_falls_back(monkeypatch):
    """If ASR returns an out-of-pair label (e.g. 'fr') in an en↔de
    meeting, the same-script-aware fallback pins to pair[0] (en)
    rather than calling the script-router (which would still say
    'en' because all are Latin) — same behavior, but explicit
    code path so future tweaks can be smarter."""
    from meeting_scribe.runtime import state

    monkeypatch.setattr(state, "current_meeting", _StubMeeting(["en", "de"]))
    monkeypatch.setattr(state, "translation_queue", None)
    monkeypatch.setattr(state, "asr_backend", None)
    monkeypatch.setattr(state, "furigana_backend", None)
    monkeypatch.setattr(state, "_audio_out_clients", set())
    monkeypatch.setattr(state, "enrollment_store", SimpleNamespace(speakers={}))
    monkeypatch.setattr(state, "storage", SimpleNamespace(append_event=lambda *a, **k: None))

    ev = _mk_event("Some Latin text.", "fr")  # 'fr' not in pair

    async def _no_broadcast(*a, **k):
        return None

    with patch.object(te, "_broadcast", _no_broadcast):
        await te._process_event(ev)

    assert ev.language in ("en", "de")
