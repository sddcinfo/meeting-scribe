"""Demand-driven multi-target translation + TTS fan-out.

Computes which target languages and TTS voice modes are actually
needed for a transcript event, given:

  * The meeting's configured ``language_pair`` (baseline — always
    translated so journal / captions / exports have it).
  * Live audio-out listeners' ``preferred_language`` values (optional —
    droppable under load).

Pulled out of ``server.py`` so the translation queue and the audio-out
WebSocket handler can both import these without circling back through
the server module.
"""

from __future__ import annotations

from meeting_scribe.models import TranscriptEvent
from meeting_scribe.runtime import state


def _norm_lang(code: str | None) -> str:
    """Normalize a language code to ISO 639-1 for comparison.

    ``"en-US"`` / ``"en_US"`` / ``"EN"`` → ``"en"``.

    Used everywhere we match a listener's preferred_language against a
    translation's target_language. Without normalization, a browser
    sending its locale (``"en-US"``) would never match the translation
    backend's output (``"en"``) and the listener would silently receive
    no audio.
    """
    if not code:
        return ""
    head = code.strip().split("-", 1)[0].split("_", 1)[0]
    return head.lower()


def _compute_translation_demand(
    event: TranscriptEvent,
) -> tuple[frozenset[str], frozenset[str]]:
    """Compute baseline and optional translation targets for a segment.

    Baseline = the meeting's ``language_pair`` cross-translation (always
    translated so journal/captions/exports have it). For monolingual
    meetings (length-1 ``language_pair``) the baseline is always empty —
    no translation work runs. Optional = live audio-out listener
    ``preferred_language`` values that are NOT in baseline and NOT
    equal to the source language. Optional targets are droppable under
    load.

    Under legacy mode the returned tuple still works: the queue uses
    baseline when demand is supplied, optional is dropped at synth time
    by the TTS listener filter.
    """
    source = _norm_lang(event.language)
    baseline: set[str] = set()
    if state.current_meeting and state.current_meeting.language_pair:
        pair = tuple(state.current_meeting.language_pair)
        if len(pair) == 2:
            a, b = _norm_lang(pair[0]), _norm_lang(pair[1])
            if source == a and b:
                baseline.add(b)
            elif source == b and a:
                baseline.add(a)
    optional: set[str] = set()
    for pref in state._audio_out_prefs.values():
        lang = _norm_lang(getattr(pref, "preferred_language", "") or "")
        if not lang or lang == source:
            continue
        if lang in baseline:
            continue
        optional.add(lang)
    return frozenset(baseline), frozenset(optional)


def _listener_tts_demand(target_lang: str) -> set[str]:
    """Return the set of voice_modes listeners want for ``target_lang``.

    Empty set means no listener wants TTS for this language → skip synth.
    Uses normalized language comparison so ``en-US`` listener matches
    ``en`` translation.
    """
    target_norm = _norm_lang(target_lang)
    modes: set[str] = set()
    for pref in state._audio_out_prefs.values():
        pref_lang = _norm_lang(getattr(pref, "preferred_language", "") or "")
        if pref_lang and pref_lang != target_norm:
            continue
        modes.add(getattr(pref, "voice_mode", "studio"))
    return modes
