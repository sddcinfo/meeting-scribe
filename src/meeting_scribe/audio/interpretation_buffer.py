"""Consecutive room-speaker gate for bidirectional interpretation."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from meeting_scribe.models import TranscriptEvent
from meeting_scribe.runtime import state
from meeting_scribe.server_support.translation_demand import _norm_lang

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedItem:
    event: TranscriptEvent
    audio: np.ndarray
    source_lang: str
    target_lang: str
    speaker_id: str
    enqueued_monotonic: float
    utterance_end_at: float


class InterpretationBuffer:
    """Single FIFO of already-synthesized TTS held for room playback."""

    def __init__(
        self,
        *,
        pause_flush_ms: int = 2500,
        idle_drain_ms: int = 5000,
        max_buffer_ms: int = 30000,
        write_timeout_s: float = 5.0,
        self_echo_tail_s: float = 0.5,
    ) -> None:
        self.pause_flush_ms = pause_flush_ms
        self.idle_drain_ms = idle_drain_ms
        self.max_buffer_ms = max_buffer_ms
        self.write_timeout_s = write_timeout_s
        self.self_echo_tail_s = self_echo_tail_s
        self.release_generation = 0
        self.enabled = True
        self._items: deque[BufferedItem] = deque()
        self._lock = asyncio.Lock()
        self._release_task: asyncio.Task | None = None
        self._ticker_task: asyncio.Task | None = None
        self._inflight_item: BufferedItem | None = None
        self._inflight_done = asyncio.Event()
        self._inflight_done.set()

    async def append(
        self,
        *,
        event: TranscriptEvent,
        audio: np.ndarray,
        source_lang: str,
        target_lang: str,
        speaker_id: str,
    ) -> None:
        now = time.monotonic()
        item = BufferedItem(
            event=event,
            audio=audio,
            source_lang=_norm_lang(source_lang),
            target_lang=_norm_lang(target_lang),
            speaker_id=speaker_id,
            enqueued_monotonic=now,
            utterance_end_at=event.utterance_end_at or now,
        )
        async with self._lock:
            if not self.enabled:
                return
            self._items.append(item)
            self._ensure_ticker_locked()
            if self._queued_audio_ms_locked() >= self.max_buffer_ms:
                self._ensure_release_locked()

    async def cancel_all(self, *, clear: bool = True) -> int:
        async with self._lock:
            self.release_generation += 1
            dropped = len(self._items)
            if clear:
                self._items.clear()
            return dropped

    async def set_enabled(self, enabled: bool) -> int:
        async with self._lock:
            self.enabled = enabled
            if enabled:
                return 0
            self.release_generation += 1
            dropped = len(self._items)
            self._items.clear()
            return dropped

    async def filter_for_language_pair(self, pair: list[str] | tuple[str, ...]) -> int:
        keep = {_norm_lang(p) for p in pair}
        async with self._lock:
            self.release_generation += 1
            before = len(self._items)
            self._items = deque(
                item
                for item in self._items
                if item.source_lang in keep and item.target_lang in keep
            )
            return before - len(self._items)

    async def try_flush(self) -> bool:
        async with self._lock:
            if not self.enabled or not self._items:
                return False
            self._ensure_release_locked()
            return True

    async def drain_for_stop(self, budget_s: float = 10.0) -> int:
        deadline = time.monotonic() + budget_s
        await self.cancel_all(clear=False)
        if self._inflight_item is not None:
            try:
                await asyncio.wait_for(
                    self._inflight_done.wait(), timeout=max(0.0, deadline - time.monotonic())
                )
            except TimeoutError:
                async with self._lock:
                    dropped = len(self._items)
                    self._items.clear()
                    return dropped

        drained = 0
        while time.monotonic() < deadline:
            async with self._lock:
                if not self._items:
                    return drained
                item = self._items.popleft()
                self._inflight_item = item
                self._inflight_done.clear()
            ok = await self._emit_item(
                item, timeout_s=min(self.write_timeout_s, max(0.1, deadline - time.monotonic()))
            )
            async with self._lock:
                self._inflight_item = None
                self._inflight_done.set()
            if not ok:
                break
            drained += 1
        async with self._lock:
            dropped = len(self._items)
            self._items.clear()
        if dropped:
            logger.info("interpretation drain dropped %d queued items after budget", dropped)
        return drained

    def _ensure_ticker_locked(self) -> None:
        if self._ticker_task is None or self._ticker_task.done():
            self._ticker_task = asyncio.create_task(
                self._ticker(), name="interpretation-buffer-ticker"
            )
            state._background_tasks.add(self._ticker_task)
            self._ticker_task.add_done_callback(state._background_tasks.discard)

    def _ensure_release_locked(self) -> None:
        if self._release_task is None or self._release_task.done():
            self._release_task = asyncio.create_task(
                self._release(), name="interpretation-buffer-release"
            )
            state._background_tasks.add(self._release_task)
            self._release_task.add_done_callback(state._background_tasks.discard)

    def _queued_audio_ms_locked(self) -> float:
        return sum(len(item.audio) / 24000.0 * 1000.0 for item in self._items)

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(0.2)
            async with self._lock:
                if not self._items:
                    return
                now = time.monotonic()
                last_final = getattr(state.asr_backend, "last_final_monotonic", None)
                last_speech_audio = getattr(state, "last_speech_audio_ts", 0.0) or None
                quiet_anchor = (
                    max(ts for ts in (last_final, last_speech_audio) if ts is not None)
                    if (last_final is not None or last_speech_audio is not None)
                    else None
                )
                pause_ready = (
                    quiet_anchor is not None
                    and (now - quiet_anchor) * 1000.0 >= self.pause_flush_ms
                )
                idle_ready = (
                    quiet_anchor is None or (now - quiet_anchor) * 1000.0 >= self.idle_drain_ms
                )
                cap_ready = self._queued_audio_ms_locked() >= self.max_buffer_ms
                if pause_ready or idle_ready or cap_ready:
                    self._ensure_release_locked()

    async def _release(self) -> None:
        release_started = time.monotonic()
        async with self._lock:
            gen = self.release_generation
        while True:
            async with self._lock:
                if gen != self.release_generation or not self.enabled:
                    return
                if not self._items:
                    return
                item = self._items[0]
                if not self._eligible_room_sink_listeners(item):
                    return
                item = self._items.popleft()
                self._inflight_item = item
                self._inflight_done.clear()

            observed = getattr(state.asr_backend, "last_real_speech_observed_monotonic", None)
            if observed is not None and observed >= release_started:
                async with self._lock:
                    self.release_generation += 1
                    self._inflight_item = None
                    self._inflight_done.set()
                return

            ok = await self._emit_item(item, timeout_s=self.write_timeout_s)
            async with self._lock:
                self._inflight_item = None
                self._inflight_done.set()
                if ok and gen == self.release_generation:
                    self._record_playback_locked(item)
                elif not ok:
                    logger.info("interpretation room-sink emit failed; queued tail retained")
                    return

    async def _emit_item(self, item: BufferedItem, *, timeout_s: float) -> bool:
        from meeting_scribe.audio.output_pipeline import _deliver_audio_to_listener

        listeners = self._eligible_room_sink_listeners(item)
        if not listeners:
            return False

        wav_cache: dict = {}
        ok_any = False
        dead = []
        for ws, pref in listeners:
            try:
                ok = await asyncio.wait_for(
                    _deliver_audio_to_listener(ws, pref, item.audio, 24000, wav_cache),
                    timeout=timeout_s,
                )
            except TimeoutError:
                ok = False
            if ok:
                ok_any = True
            else:
                dead.append(ws)
        for ws in dead:
            state._audio_out_clients.discard(ws)
            state._audio_out_prefs.pop(ws, None)
            state.metrics.listener_removed_on_send_error += 1
        return ok_any

    def _eligible_room_sink_listeners(self, item: BufferedItem) -> list[tuple[object, object]]:
        listeners = []
        for ws in list(state._audio_out_clients):
            pref = state._audio_out_prefs.get(ws)
            if pref is None or pref.delivery_mode != "consecutive":
                continue
            if pref.transport != "room_sink":
                continue
            pref_lang = _norm_lang(pref.preferred_language)
            if pref_lang and pref_lang != item.target_lang:
                continue
            listeners.append((ws, pref))
        return listeners

    def _record_playback_locked(self, item: BufferedItem) -> None:
        duration_s = len(item.audio) / 24000.0
        until = time.monotonic() + duration_s + self.self_echo_tail_s
        for ws in state._audio_out_clients:
            pref = state._audio_out_prefs.get(ws)
            if pref is None or pref.transport != "room_sink":
                continue
            current = getattr(ws, "playback_until", 0.0)
            ws.playback_until = max(current, until)
            ws.last_played_target_lang = item.target_lang
            previous = getattr(ws, "last_played_text", "") or ""
            text = item.event.translation.text if item.event.translation else ""
            ws.last_played_text = " ".join((previous, text or "")).strip()[-2000:]
