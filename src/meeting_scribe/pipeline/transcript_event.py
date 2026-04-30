"""Live transcript-event pipeline: ASR-event → translate-queue → TTS.

Two entry points wire the live recording path together:

* ``_process_event`` — receives every ASR event (partial and final).
  Filters fillers, dedupes recent finals, language-corrects via
  script detection, runs enrolled-speaker matching, attaches
  diarization clusters (or queues for catch-up), persists to the
  journal, broadcasts to WS, then (if final) submits for translation
  + furigana annotation. The single most central function in the
  live path.

* ``_broadcast_translation`` — callback the translation queue invokes
  when each per-target translation finishes. Persists the translated
  event to the journal, broadcasts it to WS, then enqueues TTS if a
  listener is connected and the event still satisfies the deferral
  predicate.

Pulled out of ``server.py`` once the runtime state, audio output,
TTS worker pool, and meeting loops were all on canonical surfaces.
``state.config``, ``state.metrics``, ``state.translation_queue``,
``state.tts_backend``, ``state.diarize_backend``,
``state.enrollment_store``, and ``state.detected_speakers`` are the
only mutable state these functions touch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from meeting_scribe.audio.output_pipeline import _send_passthrough_audio
from meeting_scribe.models import (
    DetectedSpeaker,
    SpeakerAttribution,
    TranscriptEvent,
)
from meeting_scribe.runtime import state
from meeting_scribe.server_support.broadcast import _broadcast
from meeting_scribe.server_support.speaker_audio import _simple_embedding
from meeting_scribe.server_support.translation_demand import (
    _compute_translation_demand,
)
from meeting_scribe.tts.worker import _enqueue_tts

logger = logging.getLogger(__name__)


# Short-window text-hash dedup for ASR finals. Guards against
# scribe-asr emitting the same final under a fresh segment_id after a
# container restart or other flakiness (LocalAgreement revision dedup
# keys on seg_id and can't catch that case). See ``_process_event``.
_DEDUP_WINDOW_S = 8.0
_recent_finals: dict[tuple[str, str], tuple[float, str]] = {}


# TTS deferral feature flag. When True, ``_broadcast_translation``
# skips TTS on events with empty ``speakers[]`` and lets the catch-up
# loop fire TTS once it has attached a real speaker attribution.
# Defaults off in Part A; Part B of the 2026-04 speaker-separation
# refactor flips it on.
_TTS_DEFER_UNTIL_CATCH_UP = os.environ.get("SCRIBE_TTS_DEFER_UNTIL_CATCH_UP", "0") == "1"


async def _process_event(event: TranscriptEvent) -> None:
    """Store event, broadcast to WS clients, queue translation."""
    state.metrics.asr_events += 1
    state.metrics.last_asr_event_time = time.monotonic()
    if event.is_final:
        state.metrics.asr_finals += 1
        # W5: utterance-end-to-final latency for the dashboard tile.
        # `utterance_end_at` is the monotonic ts at which the speaker's
        # audio actually ended (audio_wall_at_start + end_ms/1000).
        # Difference vs the existing `last_asr_event_time` is that this
        # measures *speaker latency* (perceived "when did transcript
        # appear after I stopped talking"), not just inter-event delta.
        now_mono = time.monotonic()
        if event.utterance_end_at is not None:
            ms = (now_mono - event.utterance_end_at) * 1000
            if 0 <= ms < 60_000:  # filter clock-skew + outliers
                state.metrics.utterance_end_to_final_ms.append(ms)
        state.metrics.last_final_ts = now_mono
    else:
        state.metrics.asr_partials += 1

    # Filler-only finals: ASR frequently emits a final for a single
    # backchannel syllable ("ああ", "うん", "はい", "そう") between
    # real utterances. These carry no information, they inflate
    # translate cost, and they clutter the transcript. Drop them
    # here before any downstream fanout. The allowlist below matches
    # the most common JA + EN filler patterns; we keep text if it has
    # ≥ 2 CJK chars or ≥ 3 non-space ASCII chars outside the allowlist.
    if event.is_final and event.text:
        _filler = {
            "ああ",
            "ああ。",
            "うん",
            "うん。",
            "はい",
            "はい。",
            "えっと",
            "えっと。",
            "あ",
            "あ。",
            "そう",
            "そう。",
            "ね",
            "ね。",
            "なん",
            "なん。",
            "ええ",
            "ええ。",
            "uh",
            "uh.",
            "um",
            "um.",
            "ah",
            "ah.",
            "mm",
            "mm.",
            "yeah",
            "yeah.",
            "ok",
            "ok.",
        }
        normalized = event.text.strip().lower()
        if normalized in _filler or len(normalized) <= 1:
            state.metrics.asr_finals_filler_dropped = (
                getattr(state.metrics, "asr_finals_filler_dropped", 0) + 1
            )
            logger.info(
                "Filler final dropped: seg=%s text='%s'",
                event.segment_id,
                event.text[:40],
            )
            return

    # Short-window text-hash dedup on FINALS ONLY.
    # When scribe-asr recovers from a container restart (e.g. GPU
    # contention killed its worker), it can replay buffered audio
    # under fresh segment_ids so LocalAgreement's revision-based dedup
    # (keyed on seg_id) can't catch it. Drop a final whose exact
    # (language, normalized-text) we already saw within
    # _DEDUP_WINDOW_S, before it fans out to storage/WS/translate.
    if event.is_final and event.text:
        now = time.monotonic()
        key = (event.language or "", event.text.strip())
        prior = _recent_finals.get(key)
        if prior is not None and (now - prior[0]) < _DEDUP_WINDOW_S:
            state.metrics.asr_finals_deduped = getattr(state.metrics, "asr_finals_deduped", 0) + 1
            logger.warning(
                "Dup ASR final dropped: seg=%s prev_seg=%s dt=%.2fs text='%s'",
                event.segment_id,
                prior[1],
                now - prior[0],
                event.text,
            )
            return
        _recent_finals[key] = (now, event.segment_id)
        if len(_recent_finals) > 256:
            stale = [k for k, (ts, _) in _recent_finals.items() if now - ts > _DEDUP_WINDOW_S]
            for k in stale:
                _recent_finals.pop(k, None)

    # Enforce language pair: if ASR detected a language outside the
    # active pair (or couldn't detect one at all), remap it to the
    # best match so the frontend doesn't silently drop the event. The
    # renderer's filter `lang !== langA && lang !== langB` rejects
    # anything not in the pair — INCLUDING "unknown" — which used to
    # cause events to disappear.
    #
    # We *also* override the ASR's claimed language when the text's
    # script disagrees with it (e.g. ASR returns lang=en for kana-heavy
    # text, or lang=en for hangul "내장."). This is the gate that
    # decides whether furigana fires AND whether out-of-pair-script
    # text leaks into the wrong column, so a script/label mismatch
    # causes both bugs at once. The override fires whenever the
    # script and label disagree, *regardless* of whether the script's
    # language is in the active pair — out-of-pair script content is
    # then mapped to the best in-pair fallback (CJK→first CJK lang in
    # pair, else first non-CJK lang in pair).
    if state.current_meeting and hasattr(state.current_meeting, "language_pair") and event.text:
        pair = state.current_meeting.language_pair
        from meeting_scribe.backends.asr_filters import _detect_language_from_text

        script_lang = _detect_language_from_text(event.text)

        # Script class — Latin pairs (en/de/es/fr/it/pt/nl/pl/tr/...)
        # all collapse to "en" in _detect_language_from_text because
        # the heuristic only distinguishes by Unicode block, not by
        # n-grams. For pairs where BOTH languages live in the same
        # script class, the script-router has no signal: it would
        # always remap one side to the Latin default ("en"), which is
        # exactly the en↔de bug where German text got force-relabeled
        # to en and translated to itself. Trust the ASR + lingua
        # post-correction in that case.
        _CJK = {"ja", "zh", "ko"}
        _NON_LATIN_SINGLE_SCRIPT = {"ru", "uk", "ar", "th", "hi"}

        def _script_class(lang: str) -> str:
            if lang in _CJK:
                return "cjk"
            if lang in _NON_LATIN_SINGLE_SCRIPT:
                return lang  # each in its own bucket
            return "latin"

        pair_script_classes = {_script_class(lang) for lang in pair if lang}
        script_router_useful = len(pair_script_classes) > 1

        def _fallback_for_script(script: str) -> str:
            """Pick the closest in-pair language for an out-of-pair script."""
            cjk_langs = [lang for lang in pair if lang in _CJK]
            if script in _CJK and cjk_langs:
                return cjk_langs[0]
            non_cjk = [lang for lang in pair if lang not in _CJK]
            return non_cjk[0] if non_cjk else pair[0]

        needs_remap = event.language not in pair
        # Case 1: the script disagrees with the ASR label — trust the
        # script. If script_lang is in pair, use it directly.
        # Otherwise pick the closest in-pair fallback. This is how
        # "내장." tagged en gets re-routed to ja in a ja↔en meeting
        # instead of leaking into the English column. Skip when
        # script_lang is "unknown" (e.g. all punctuation / numbers) —
        # we have no signal to act on. ALSO skip when both pair
        # languages share the same script class (en↔de, en↔fr, etc.)
        # — the script detector has no signal to distinguish them.
        if (
            script_router_useful
            and not needs_remap
            and script_lang != "unknown"
            and script_lang != event.language
        ):
            target = script_lang if script_lang in pair else _fallback_for_script(script_lang)
            if target != event.language:
                logger.info(
                    "Language remap: ASR=%s → script=%s → %s (text=%r)",
                    event.language,
                    script_lang,
                    target,
                    event.text[:40],
                )
                event.language = target
        elif needs_remap:
            if script_router_useful and script_lang in pair:
                event.language = script_lang
            elif not script_router_useful:
                # Same-script pair with an out-of-pair ASR label — keep
                # the ASR call but pin to the first pair language as a
                # safe fallback. This is rare (lingua usually catches
                # it first) but happens when ASR returns e.g. "fr" on
                # an en↔de meeting.
                event.language = pair[0]
            else:
                event.language = _fallback_for_script(script_lang)

    # Speaker matching: compare audio chunk against enrolled speaker embeddings
    if event.is_final and event.text and state.enrollment_store.speakers:
        try:
            audio_chunk = getattr(state.asr_backend, "last_audio_chunk", None)
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Run embedding extraction off the event loop.
                if state._speaker_executor_singleton is None:
                    state._speaker_executor_singleton = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="speaker"
                    )
                _speaker_executor = state._speaker_executor_singleton

                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    _speaker_executor, _simple_embedding, audio_chunk
                )

                best_name = None
                best_score = 0.0
                best_eid = None
                from meeting_scribe.speaker.verification import cosine_similarity

                for (
                    eid,
                    name,
                    enrolled_emb,
                ) in state.enrollment_store.get_all_embeddings():
                    score = cosine_similarity(embedding, enrolled_emb)
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_eid = eid

                # ENRICH the diarization result with enrolled identity
                # rather than REPLACE it. Preserves per-speaker
                # cluster_ids so downstream timeline lanes and stats
                # show all speakers correctly. Threshold raised from
                # 0.3 → 0.55: 0.3 was too permissive, causing every
                # segment to be claimed by the first enrolled speaker.
                if best_score > 0.55 and best_name:
                    existing_speakers = list(event.speakers or [])
                    if existing_speakers:
                        # Keep original cluster_id, add identity on top
                        original = existing_speakers[0]
                        existing_speakers[0] = SpeakerAttribution(
                            cluster_id=original.cluster_id,
                            identity=best_name,
                            identity_confidence=best_score,
                            source="enrolled",
                        )
                    else:
                        existing_speakers = [
                            SpeakerAttribution(
                                cluster_id=0,
                                identity=best_name,
                                identity_confidence=best_score,
                                source="enrolled",
                            )
                        ]
                    event = event.model_copy(update={"speakers": existing_speakers})
                    existing = [s for s in state.detected_speakers if s.speaker_id == best_eid]
                    if existing:
                        existing[0].segment_count += 1
                        existing[0].last_seen_ms = event.end_ms
                    else:
                        state.detected_speakers.append(
                            DetectedSpeaker(
                                speaker_id=best_eid or "",
                                display_name=best_name,
                                matched_enrollment_id=best_eid,
                                match_confidence=best_score,
                                segment_count=1,
                                first_seen_ms=event.start_ms,
                                last_seen_ms=event.end_ms,
                            )
                        )
        except Exception as e:
            logger.debug("Speaker matching error: %s", e)

    # Automatic name discovery has been intentionally removed.
    # Names come from only two sources:
    #   1. Pre-meeting voice enrollment (explicit seat assignment) — handled above
    #   2. Explicit user corrections via the UI (update_segment_speaker endpoint)
    # Any speaker without a known identity will display as "Speaker {cluster_id}".
    #
    # Previously self-introduction detection + LLM name extraction
    # could silently auto-assign names to unnamed seats, which was
    # unreliable and confusing.

    # Diarization: attach speaker cluster_id from diarization backend.
    # First attempt is synchronous — if results are already cached we
    # use them. If not (diarization buffers audio and lags ASR), we
    # queue the event for the background catch-up loop which
    # re-attributes and re-broadcasts once diarization produces results.
    if (
        state.diarize_backend
        and event.is_final
        and hasattr(state.diarize_backend, "get_results_for_range")
        and not event.speakers  # don't overwrite enrolled identification
    ):
        try:
            results = state.diarize_backend.get_results_for_range(event.start_ms, event.end_ms)
            if results:
                event = event.model_copy(
                    update={
                        "speakers": [
                            SpeakerAttribution(
                                cluster_id=results[0].cluster_id,
                                source="diarization",
                            )
                        ]
                    }
                )
            elif event.text and len(event.text.strip()) > 0:
                # Queue for retroactive attribution by the catch-up loop
                state._pending_speaker_events[event.segment_id] = event
                state._pending_speaker_timestamps[event.segment_id] = time.monotonic()
                # Cap the pending queue (oldest 200 segments — ~10 minutes)
                while len(state._pending_speaker_events) > 200:
                    old_sid, _ = state._pending_speaker_events.popitem(last=False)
                    state._pending_speaker_timestamps.pop(old_sid, None)
        except Exception:
            pass

    # SPEAKER ATTRIBUTION DELIBERATELY NOT GUESSED HERE.
    #
    # Pre-2026-04 this code allocated a time-proximity "pseudo cluster"
    # (cluster_id ≥ 100) when diarization hadn't produced a result yet,
    # so the UI wouldn't show "Unknown". That was a guess — same
    # speaker across a silence gap might get a different pseudo-cluster,
    # or two different speakers might share one if they spoke
    # back-to-back.
    #
    # The user's directive for Part B of the 2026-04 refactor: ASR
    # should identify SPEECH only. The catch-up loop is the sole source
    # of speaker attribution, and it only runs when diarization has a
    # real answer. Segments broadcast here with empty speakers[] are
    # queued in state._pending_speaker_events above, and
    # ``_speaker_catchup_loop`` will broadcast a revised event (same
    # segment_id, incremented revision) once pyannote has assigned a
    # real cluster. The frontend dedups by segment_id, so the UI
    # updates in place: transcript line appears first with no speaker
    # badge, then the badge fills in seconds later.
    #
    # Enrollment matching runs earlier in this function (above). That
    # path is kept because enrolled-voice match is deterministic and
    # high-confidence — not a guess.

    if state.current_meeting:
        state.storage.append_event(state.current_meeting.meeting_id, event)

    await _broadcast(event)

    # Full interpretation mode: pass through original audio to clients
    # whose preferred language matches the speaker's language
    if event.is_final and event.text and state._audio_out_clients:
        audio_chunk = getattr(state.asr_backend, "last_audio_chunk", None)
        if audio_chunk is not None and len(audio_chunk) > 0:
            await _send_passthrough_audio(audio_chunk, event.language)

    if event.is_final and state.translation_queue:
        state.metrics.translations_submitted += 1
        logger.info(
            "Submitting for translation: seg=%s lang=%s text='%s'",
            event.segment_id,
            event.language,
            event.text[:60],
        )
        baseline, optional = _compute_translation_demand(event)
        await state.translation_queue.submit(
            event, baseline_targets=baseline, optional_targets=optional
        )
        # Flush merge gate immediately — don't hold segments waiting for merge
        await state.translation_queue.flush_merge_gate()

    # Furigana annotation — concurrent, non-blocking, best-effort.
    # Runs as a separate asyncio.create_task so it never delays the
    # translate submit. Uses its own httpx pool so the connection pool
    # is separate from translate_queue. Result is broadcast as an
    # event revision with a `furigana_html` field when it lands; the
    # UI picks it up and re-renders the transcript block in place.
    # Priority is intentionally +5 (lower than translate's -10) so
    # vLLM's priority scheduler preempts for the critical path under
    # contention.
    if (
        event.is_final
        and event.language == "ja"
        and event.text
        and state.furigana_backend is not None
    ):

        async def _annotate_and_broadcast(ev):
            try:
                html = await state.furigana_backend.annotate(ev.text)
                if not html:
                    logger.info(
                        "furigana: no annotation for seg=%s text=%s (no kanji or cache miss)",
                        ev.segment_id,
                        ev.text[:30],
                    )
                    return
                logger.info(
                    "furigana: seg=%s text=%s → %d chars",
                    ev.segment_id,
                    ev.text[:30],
                    len(html),
                )
                updated = ev.model_copy(
                    update={
                        "furigana_html": html,
                        "revision": (ev.revision or 0) + 1,
                    }
                )
                if state.current_meeting:
                    try:
                        state.storage.append_event(state.current_meeting.meeting_id, updated)
                    except Exception as exc:
                        logger.warning("furigana: journal append failed: %s", exc)
                await _broadcast(updated)
            except Exception as e:
                logger.warning("furigana task failed seg=%s: %s", ev.segment_id, e)

        task = asyncio.create_task(
            _annotate_and_broadcast(event),
            name=f"furigana-{event.segment_id}",
        )
        state._background_tasks.add(task)
        task.add_done_callback(state._background_tasks.discard)


async def _broadcast_translation(event: TranscriptEvent) -> None:
    """Callback from translation queue — track metrics, persist, and broadcast."""
    if event.translation:
        status = event.translation.status.value
        if status == "done":
            state.metrics.translations_completed += 1
            # Persist the translation to the journal
            if state.current_meeting:
                state.storage.append_event(state.current_meeting.meeting_id, event)
        elif status == "failed":
            state.metrics.translations_failed += 1

    # Broadcast to all clients (includes in_progress, done, failed, skipped)
    await _broadcast(event)

    # TTS: synthesize translated text in speaker's voice (background —
    # don't block translation queue). ONLY runs when at least one
    # listener is connected — no point burning GPU for nobody. Voice
    # references are still cached so listeners get speaker's voice
    # when they join later.
    #
    # When _TTS_DEFER_UNTIL_CATCH_UP is True (Part B of the 2026-04
    # speaker separation refactor), segments with empty event.speakers
    # are skipped at broadcast time and TTS is fired from the catch-up
    # loop instead, once the speaker attribution has been resolved by
    # diarization.
    if (
        state.tts_backend
        and state.tts_backend.available
        and event.translation
        and event.translation.status.value == "done"
        and event.translation.text
        and not (_TTS_DEFER_UNTIL_CATCH_UP and not event.speakers)
    ):
        # Cache voice reference synchronously regardless of listeners
        # (so future listeners get the speaker's voice immediately).
        #
        # Derive a stable speaker_id for the voice cache:
        #   1. The speaker's enrolled identity ("Tanaka") when set
        #   2. The diarization cluster_id ("cluster_0", "cluster_1")
        #      when the speaker is detected but not yet enrolled
        #   3. "default" only when there is no speaker attribution at all
        # The old code was `speaker.identity if speaker else "default"`,
        # which returned None for freshly-detected speakers (identity
        # is None on every new cluster until enrollment), and the
        # voice cache skipped them silently because `if audio_chunk
        # and speaker_id:` evaluates False on None. With no voice
        # cached, ``_do_tts_synthesis`` would fall through `if ref is
        # None: continue` for every segment and Listen would produce
        # nothing.
        speaker = event.speakers[0] if event.speakers else None
        if speaker is None:
            speaker_id = "default"
        elif speaker.identity:
            speaker_id = speaker.identity
        else:
            speaker_id = f"cluster_{speaker.cluster_id}"

        audio_chunk = getattr(state.asr_backend, "last_audio_chunk", None)
        if audio_chunk is not None:
            state.tts_backend.cache_voice(speaker_id, audio_chunk)

        # Skip synthesis entirely if no listeners — saves GPU
        if not state._audio_out_clients:
            return

        logger.info(
            "TTS fire: seg=%s speaker=%s text=%r",
            event.segment_id,
            speaker_id,
            (event.translation.text or "")[:60],
        )

        _enqueue_tts(event, speaker_id)
