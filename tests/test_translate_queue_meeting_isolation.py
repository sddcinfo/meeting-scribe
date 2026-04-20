"""Meeting-scoped live-history invariants for Phase B1.

These tests are the P0 guard from the plan: meeting-N history must not
be visible to meeting-N+1 after stop/start or dev-reset, and stale work
items must not contaminate a fresh generation.

Epoch semantics:
  * ``bind_meeting(mid)`` → epoch += 1
  * ``bind_meeting(None)`` → epoch += 1
  * ``clear_meeting(mid)`` → epoch += 1
  * dev_reset flow = bind(None) → flush → clear(mid) → bind(mid)
    → epoch advances by 3 so an item stamped pre-reset is stale on
    both the top-of-function and post-await checks.

The tests exercise ``TranslationQueue`` directly with a mock backend so
they don't need vLLM or the full server.
"""

from __future__ import annotations

import asyncio
import typing

import pytest

from meeting_scribe.models import TranscriptEvent
from meeting_scribe.translation.queue import TranslationQueue


class _RecordingBackend:
    """Records every call and returns a deterministic translation."""

    def __init__(self, hang_event: asyncio.Event | None = None):
        self.calls: list[dict] = []
        self._hang = hang_event

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        prior_context=None,
        meeting_id=None,
    ) -> str:
        self.calls.append(
            {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "prior_context": list(prior_context) if prior_context else None,
                "meeting_id": meeting_id,
            }
        )
        if self._hang is not None:
            await self._hang.wait()
        return f"{text}::{target_language}"


def _make_event(seg_id: str, text: str, lang: str = "ja") -> TranscriptEvent:
    return TranscriptEvent(
        segment_id=seg_id,
        revision=0,
        is_final=True,
        start_ms=0,
        end_ms=1000,
        language=lang,
        text=text,
    )


async def _wait_idle(q: TranslationQueue, timeout_s: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if q.is_idle():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("queue did not go idle in time")


class TestBindMeetingEpoch:
    def test_bind_unbind_clear_all_bump_epoch(self):
        q = TranslationQueue(on_result=None, languages=("ja", "en"))
        assert q._bind_epoch == 0
        q.bind_meeting("mtg-a")
        assert q._bind_epoch == 1
        q.bind_meeting(None)
        assert q._bind_epoch == 2
        q.clear_meeting("mtg-a")
        assert q._bind_epoch == 3
        q.bind_meeting("mtg-a")
        assert q._bind_epoch == 4

    def test_history_dict_created_on_bind(self):
        q = TranslationQueue(on_result=None, languages=("ja", "en"))
        q.bind_meeting("mtg-a", history_maxlen=2)
        assert "mtg-a" in q._live_history
        assert q._live_history_maxlen == 2

    def test_clear_meeting_pops_history(self):
        q = TranslationQueue(on_result=None, languages=("ja", "en"))
        q.bind_meeting("mtg-a", history_maxlen=2)
        q._live_history["mtg-a"][("ja", "en")] = object()  # sentinel
        q.clear_meeting("mtg-a")
        assert "mtg-a" not in q._live_history


class TestStopStartIsolation:
    """Meeting B must not see meeting A's live history."""

    @pytest.mark.asyncio
    async def test_meeting_b_sees_none_prior_context(self):
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            # Meeting A: three JA→EN utterances.
            q.bind_meeting("mtg-A", history_maxlen=2)
            for i, text in enumerate(["あ", "い", "う"]):
                await q.submit(_make_event(f"A-{i}", text))
            await q.flush_merge_gate()
            await _wait_idle(q)
            assert len(backend.calls) == 3

            # Stop: unbind + clear.
            q.bind_meeting(None)
            q.clear_meeting("mtg-A")

            # Meeting B: first utterance must see NO prior context.
            q.bind_meeting("mtg-B", history_maxlen=2)
            await q.submit(_make_event("B-0", "え"))
            await q.flush_merge_gate()
            await _wait_idle(q)

            b_call = backend.calls[-1]
            assert b_call["meeting_id"] == "mtg-B"
            assert b_call["prior_context"] is None
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_same_meeting_history_accumulates(self):
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            q.bind_meeting("mtg-A", history_maxlen=2)
            await q.submit(_make_event("A-0", "first"))
            await q.flush_merge_gate()
            await _wait_idle(q)
            # First call: no history yet.
            assert backend.calls[0]["prior_context"] is None

            await q.submit(_make_event("A-1", "second"))
            await q.flush_merge_gate()
            await _wait_idle(q)
            # Second call: sees the first utterance.
            assert backend.calls[1]["prior_context"] == [("first", "first::en")]

            await q.submit(_make_event("A-2", "third"))
            await q.flush_merge_gate()
            await _wait_idle(q)
            # Third call: sees the two newest.
            assert backend.calls[2]["prior_context"] == [
                ("first", "first::en"),
                ("second", "second::en"),
            ]
        finally:
            await q.stop()


class TestEnJaDoesNotUseHistory:
    """Inverse-direction invariant: EN→JA must never get prior_context
    even when the knob is on. The quality data from the sweep only
    justified JA→EN."""

    @pytest.mark.asyncio
    async def test_en_ja_prior_context_is_none(self):
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("en", "ja"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            q.bind_meeting("mtg-A", history_maxlen=2)
            for i, text in enumerate(["hello", "world"]):
                await q.submit(_make_event(f"A-{i}", text, lang="en"))
            await q.flush_merge_gate()
            await _wait_idle(q)

            for call in backend.calls:
                assert call["source_language"] == "en"
                assert call["target_language"] == "ja"
                assert call["prior_context"] is None, (
                    "EN→JA live path must remain stateless per plan B1 gate"
                )
        finally:
            await q.stop()


class TestContextWindowKnobOff:
    """When the knob is 0, no history should flow even on JA→EN."""

    @pytest.mark.asyncio
    async def test_knob_off_never_passes_prior_context(self):
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            q.bind_meeting("mtg-A", history_maxlen=0)  # knob off
            await q.submit(_make_event("A-0", "first"))
            await q.submit(_make_event("A-1", "second"))
            await q.flush_merge_gate()
            await _wait_idle(q)

            for call in backend.calls:
                assert call["prior_context"] is None
        finally:
            await q.stop()


class TestStaleItemDropAcrossStopStart:
    """P1 plan regression guard: an item enqueued under meeting A that
    is picked up by a worker after meeting A stops + meeting B starts
    must be dropped via ``_emit_skip`` — pre-flight epoch check."""

    @pytest.mark.asyncio
    async def test_pre_flight_drop_on_epoch_mismatch(self):
        backend = _RecordingBackend()
        results: list[TranscriptEvent] = []

        async def _on_result(ev: TranscriptEvent) -> None:
            results.append(ev)

        q = TranslationQueue(
            on_result=_on_result,
            languages=("ja", "en"),
            concurrency=1,
            timeout=2.0,
        )
        await q.start(backend)
        try:
            q.bind_meeting("mtg-A", history_maxlen=2)

            # Fabricate a stale _WorkItem by stamping it with an
            # epoch that will be wrong after the rebind.  The queue's
            # own submit() stamps with the current epoch, which would
            # stay valid; we need to construct the item directly to
            # simulate the race.
            from meeting_scribe.translation.queue import _WorkItem

            stale_event = _make_event("A-stale", "あ")
            stale_item = _WorkItem(
                event=stale_event,
                segment_id=stale_event.segment_id,
                revision_id=stale_event.revision,
                baseline_targets=frozenset({"en"}),
                meeting_id="mtg-A",
                bind_epoch=q._bind_epoch,  # captured while A is active
            )

            # Now stop → start → new meeting binds.
            q.bind_meeting(None)
            q.clear_meeting("mtg-A")
            q.bind_meeting("mtg-B", history_maxlen=2)

            # Feed the stale item as if a worker just pulled it.
            await q._translate_one(stale_item, "en", worker_id=99)

            # No backend call was made; the skip callback fired.
            assert backend.calls == []
            skipped = [
                r for r in results if r.translation and r.translation.status.value == "skipped"
            ]
            assert len(skipped) >= 1
        finally:
            await q.stop()


class TestStaleItemDropAcrossDevReset:
    """Same-meeting-id isolation: dev_reset keeps the meeting_id but
    cycles the epoch. An item enqueued pre-reset must still be dropped
    even though ``item.meeting_id == self._active_meeting_id``."""

    @pytest.mark.asyncio
    async def test_drop_despite_meeting_id_match(self):
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            q.bind_meeting("mtg-A", history_maxlen=2)

            from meeting_scribe.translation.queue import _WorkItem

            stale_event = _make_event("A-stale", "あ")
            pre_reset_epoch = q._bind_epoch
            stale_item = _WorkItem(
                event=stale_event,
                segment_id=stale_event.segment_id,
                revision_id=stale_event.revision,
                baseline_targets=frozenset({"en"}),
                meeting_id="mtg-A",
                bind_epoch=pre_reset_epoch,
            )

            # dev_reset flow: bind(None) → clear → bind(same id)
            q.bind_meeting(None)
            q.clear_meeting("mtg-A")
            q.bind_meeting("mtg-A", history_maxlen=2)

            # Same meeting_id now, but epoch advanced 3 ticks.
            assert q._active_meeting_id == "mtg-A"
            assert q._bind_epoch != pre_reset_epoch

            await q._translate_one(stale_item, "en", worker_id=42)
            assert backend.calls == []
        finally:
            await q.stop()


class TestInFlightDropAcrossDevReset:
    """Plan iter-4 finding #1 regression guard: an item that already
    passed the top-of-function epoch check but is mid-backend-await
    when dev_reset fires must be dropped via the post-await check —
    no history append, no done event."""

    @pytest.mark.asyncio
    async def test_post_await_drop_clears_history_and_emits_skip(self):
        hang = asyncio.Event()
        backend = _RecordingBackend(hang_event=hang)
        results: list[TranscriptEvent] = []

        async def _on_result(ev: TranscriptEvent) -> None:
            results.append(ev)

        q = TranslationQueue(
            on_result=_on_result,
            languages=("ja", "en"),
            concurrency=1,
            timeout=5.0,
        )
        await q.start(backend)
        try:
            q.bind_meeting("mtg-A", history_maxlen=2)

            # Submit under meeting A's epoch.  The backend will block
            # inside translate() until we set the hang event.
            await q.submit(_make_event("A-hanging", "ほげ"))
            await q.flush_merge_gate()

            # Wait until the worker has actually entered the backend call.
            async def _wait_call_started():
                for _ in range(200):
                    if backend.calls:
                        return
                    await asyncio.sleep(0.01)
                raise AssertionError("backend never called")

            await _wait_call_started()

            # dev_reset flow while the call is held open.
            q.bind_meeting(None)
            q.clear_meeting("mtg-A")
            q.bind_meeting("mtg-A", history_maxlen=2)

            # Release the backend.
            hang.set()
            await _wait_idle(q)

            # The backend did return, but the post-await check should
            # have dropped the result.  Asserts:
            #  (a) backend call was made (pre-flight passed).
            #  (b) No DONE event was emitted.
            #  (c) History for mtg-A is empty (post-reset).
            assert len(backend.calls) == 1
            done_events = [
                r for r in results if r.translation and r.translation.status.value == "done"
            ]
            assert done_events == []

            per_meeting = q._live_history.get("mtg-A") or {}
            assert per_meeting.get(("ja", "en")) in (None,)
        finally:
            hang.set()
            await q.stop()


class TestFragmentGating:
    """Phase B2: when ``fragment_gated=True``, only utterances that
    look fragmentary (short OR trailing in a bare particle) receive
    prior_context.  Short affirmatives remain cache-eligible."""

    @pytest.mark.asyncio
    async def test_non_fragment_utterance_skips_context(self):
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            q.bind_meeting("mtg-frag", history_maxlen=2, fragment_gated=True)
            # Seed history so the deque is non-empty.
            await q.submit(_make_event("A-0", "seed utterance that is long enough"))
            await q.flush_merge_gate()
            await _wait_idle(q)
            assert len(backend.calls) == 1
            # Second call: long utterance → gate rejects → no context.
            await q.submit(_make_event("A-1", "another long non-fragment utterance"))
            await q.flush_merge_gate()
            await _wait_idle(q)
            assert backend.calls[1]["prior_context"] is None
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_fragment_utterance_gets_context(self):
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            q.bind_meeting("mtg-frag", history_maxlen=2, fragment_gated=True)
            # First call: no history, no gate benefit, no context.
            await q.submit(_make_event("A-0", "最初の発言が十分に長いのでフラグメントではない"))
            await q.flush_merge_gate()
            await _wait_idle(q)
            # Second call: short (< 12 chars) → fragment → gets context.
            await q.submit(_make_event("A-1", "はい"))
            await q.flush_merge_gate()
            await _wait_idle(q)
            assert backend.calls[1]["prior_context"] is not None

            # Trailing particle should also count as fragmentary, even
            # though length alone wouldn't flag this one.
            await q.submit(_make_event("A-2", "その辺をクリアしないと、非常に"))
            await q.flush_merge_gate()
            await _wait_idle(q)
            assert backend.calls[2]["prior_context"] is not None
        finally:
            await q.stop()

    def test_looks_like_fragment_heuristic(self):
        assert TranslationQueue._looks_like_fragment("はい") is True  # short
        assert TranslationQueue._looks_like_fragment("その辺をクリアしないと、非常に") is True  # particle
        assert TranslationQueue._looks_like_fragment(
            "これは十分に長く、助詞で終わらない完結した発言です。"
        ) is False  # long + period tail
        assert TranslationQueue._looks_like_fragment("") is True  # empty counts as fragment


class TestRefinementPoolRead:
    """Phase B5: when the refinement worker runs on the SAME meeting,
    the live path prefers its tail over the queue's own history —
    higher-quality translations, same cross-meeting isolation
    guarantee (match on meeting_id)."""

    @pytest.mark.asyncio
    async def test_reads_refinement_pool_when_meeting_matches(self, monkeypatch):
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            q.bind_meeting("mtg-shared", history_maxlen=2)

            class _StubWorker:
                _meeting_id: str = "mtg-shared"
                _results: typing.ClassVar[list] = [
                    {
                        "language": "ja",
                        "text": "refined source",
                        "translation": {
                            "text": "refined target",
                            "target_language": "en",
                        },
                    }
                ]

            from meeting_scribe import server as server_mod

            monkeypatch.setattr(server_mod, "refinement_worker", _StubWorker())

            await q.submit(_make_event("A-0", "ほげ"))
            await q.flush_merge_gate()
            await _wait_idle(q)

            # Backend saw the refinement pool's entry as prior_context,
            # NOT the queue's own (still-empty) history.
            assert backend.calls[0]["prior_context"] == [
                ("refined source", "refined target")
            ]
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_ignores_refinement_pool_when_meeting_differs(self, monkeypatch):
        """Cross-meeting isolation: must NOT serve another meeting's
        refinement history as live context."""
        backend = _RecordingBackend()
        q = TranslationQueue(on_result=None, languages=("ja", "en"), concurrency=1, timeout=2.0)
        await q.start(backend)
        try:
            q.bind_meeting("mtg-current", history_maxlen=2)

            class _OtherMeetingWorker:
                _meeting_id: str = "mtg-OTHER"
                _results: typing.ClassVar[list] = [
                    {
                        "language": "ja",
                        "text": "other meeting source",
                        "translation": {
                            "text": "other meeting target",
                            "target_language": "en",
                        },
                    }
                ]

            from meeting_scribe import server as server_mod

            monkeypatch.setattr(server_mod, "refinement_worker", _OtherMeetingWorker())

            await q.submit(_make_event("A-0", "ほげ"))
            await q.flush_merge_gate()
            await _wait_idle(q)

            # No leak: current meeting's live call saw no prior_context.
            assert backend.calls[0]["prior_context"] is None
        finally:
            await q.stop()


class TestHistoryOnlyOnSuccess:
    """Timeouts and errors must not pollute the history."""

    @pytest.mark.asyncio
    async def test_timeout_skips_history_append(self):
        hang = asyncio.Event()  # never set → forces timeout
        backend = _RecordingBackend(hang_event=hang)
        q = TranslationQueue(
            on_result=None,
            languages=("ja", "en"),
            concurrency=1,
            timeout=0.2,  # short timeout to force the failure branch
        )
        await q.start(backend)
        try:
            q.bind_meeting("mtg-A", history_maxlen=2)
            await q.submit(_make_event("A-timeout", "あ"))
            await q.flush_merge_gate()
            # Wait for the worker to try, fail, and return.
            await asyncio.sleep(0.5)

            per_meeting = q._live_history.get("mtg-A") or {}
            assert per_meeting.get(("ja", "en")) in (None,)
        finally:
            hang.set()
            await q.stop()
