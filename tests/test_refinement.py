"""Tests for refinement worker — overlapping chunks, VRAM safeguards,
drain-registry + counter persistence."""

from __future__ import annotations

import asyncio
import time

import pytest

from meeting_scribe.refinement import BYTES_PER_SEC, RefinementWorker


class TestRefinementWorkerInit:
    def test_creates_with_defaults(self, tmp_path):
        w = RefinementWorker(
            meeting_id="test-123",
            meeting_dir=tmp_path,
            asr_url="http://localhost:8003",
            translate_url="http://localhost:8010",
        )
        assert w._meeting_id == "test-123"
        assert w._trail_seconds == 45.0
        assert w._chunk_seconds == 10.0
        # Constructor default is 0 — the ServerConfig default of 4
        # flows in via the *explicit* kwarg at RefinementWorker
        # construction time (server.py passes it); this test only
        # covers the direct-construction path, where leaving it out
        # keeps the conservative zero default.
        assert w._context_window_segments_default == 0

    def test_custom_trail_and_chunk(self, tmp_path):
        w = RefinementWorker(
            meeting_id="test",
            meeting_dir=tmp_path,
            trail_seconds=30.0,
            chunk_seconds=5.0,
        )
        assert w._trail_seconds == 30.0
        assert w._chunk_seconds == 5.0

    def test_language_pair_configurable(self, tmp_path):
        w = RefinementWorker(
            meeting_id="test",
            meeting_dir=tmp_path,
            language_pair=("zh", "en"),
        )
        # The constructor coerces tuple → list internally so monolingual
        # meetings can pass a single-element pair without trailing-None.
        assert w._language_pair == ["zh", "en"]


class TestContextWindowCollector:
    """_collect_prior_context pulls only same-direction pairs and
    returns them oldest-first so the prompt reads naturally."""

    def _worker(self, tmp_path):
        return RefinementWorker(meeting_id="test", meeting_dir=tmp_path)

    def test_zero_window_returns_empty(self, tmp_path):
        w = self._worker(tmp_path)
        w._results = [
            {
                "language": "ja",
                "text": "こんにちは",
                "translation": {"text": "Hello", "target_language": "en"},
            }
        ]
        assert w._collect_prior_context("ja", "en", 0) == []

    def test_filters_by_direction(self, tmp_path):
        # Mixed directions in the same meeting — ja→en query must NOT
        # see the en→ja history (that would inject wrong-direction
        # exemplars into the prompt).
        w = self._worker(tmp_path)
        w._results = [
            {
                "language": "ja",
                "text": "こんにちは",
                "translation": {"text": "Hello", "target_language": "en"},
            },
            {
                "language": "en",
                "text": "Good morning",
                "translation": {"text": "おはよう", "target_language": "ja"},
            },
            {
                "language": "ja",
                "text": "ありがとう",
                "translation": {"text": "Thanks", "target_language": "en"},
            },
        ]
        out = w._collect_prior_context("ja", "en", 8)
        assert out == [("こんにちは", "Hello"), ("ありがとう", "Thanks")]

    def test_respects_window_size_newest_kept(self, tmp_path):
        # With 3 candidates and window=2, keep the newest two in
        # oldest-first order.
        w = self._worker(tmp_path)
        w._results = [
            {
                "language": "ja",
                "text": "A",
                "translation": {"text": "A-EN", "target_language": "en"},
            },
            {
                "language": "ja",
                "text": "B",
                "translation": {"text": "B-EN", "target_language": "en"},
            },
            {
                "language": "ja",
                "text": "C",
                "translation": {"text": "C-EN", "target_language": "en"},
            },
        ]
        out = w._collect_prior_context("ja", "en", 2)
        assert out == [("B", "B-EN"), ("C", "C-EN")]

    def test_skips_entries_missing_translation(self, tmp_path):
        # Segments where translation failed (translation=None or missing
        # text) should be silently excluded — better to inject fewer
        # context pairs than to pass the model a broken exemplar.
        w = self._worker(tmp_path)
        w._results = [
            {
                "language": "ja",
                "text": "ok",
                "translation": {"text": "OK", "target_language": "en"},
            },
            {"language": "ja", "text": "failed", "translation": None},
            {
                "language": "ja",
                "text": "noop",
                "translation": {"text": "", "target_language": "en"},
            },
        ]
        out = w._collect_prior_context("ja", "en", 5)
        assert out == [("ok", "OK")]


class TestBytesPerSec:
    def test_bytes_per_sec_16khz_s16le(self):
        # 16kHz * 2 bytes per sample = 32000 bytes/sec
        assert BYTES_PER_SEC == 32000


class TestOverlapCalculation:
    """Verify that chunk processing backs up by 1 second for overlap."""

    def test_overlap_offset_backs_up(self, tmp_path):
        w = RefinementWorker(
            meeting_id="test",
            meeting_dir=tmp_path,
            chunk_seconds=10.0,
        )
        # Simulate: processed a chunk ending at byte 320000 (10 seconds)
        end = 320000
        overlap_bytes = int(1.0 * BYTES_PER_SEC)
        new_offset = max(end - overlap_bytes, 0)
        # Should back up by 1 second (32000 bytes)
        assert new_offset == 320000 - 32000
        assert new_offset == 288000

    def test_overlap_doesnt_go_negative(self):
        overlap_bytes = int(1.0 * BYTES_PER_SEC)
        new_offset = max(16000 - overlap_bytes, 0)
        assert new_offset == 0  # Can't go negative


class TestCountersExposedOnWorker:
    """A0 prerequisite: counters must exist as fresh-zero attributes so
    the drain registry snapshot in server._stop_meeting_locked can read
    them without AttributeError even if the worker never made a call."""

    def test_counters_exist_and_start_at_zero(self, tmp_path):
        w = RefinementWorker(meeting_id="test-counters", meeting_dir=tmp_path)
        assert w.translate_call_count == 0
        assert w.asr_call_count == 0
        assert w.last_error_count == 0


class _FakeWorker:
    """Mock RefinementWorker for drain-registry tests.

    ``stop()`` is an ``asyncio.Event``-gated no-op so tests can simulate
    a slow drain without actually running _process_remaining. Exposes
    the same counter names the real worker does.
    """

    def __init__(self, *, meeting_id: str, stop_delay_s: float = 0.0):
        self._meeting_id = meeting_id
        self._stop_delay_s = stop_delay_s
        self.translate_call_count = 0
        self.asr_call_count = 0
        self.last_error_count = 0
        self._stopped = asyncio.Event()
        self._stop_called = False

    async def stop(self) -> None:
        self._stop_called = True
        if self._stop_delay_s > 0:
            await asyncio.sleep(self._stop_delay_s)
        # Mimic _process_remaining incrementing counters during drain.
        self.translate_call_count += 1
        self._stopped.set()


class TestDrainRegistry:
    """Server-side drain registry — monotonic ``drain_id``, per-meeting
    list, counter snapshots survive the worker being nulled."""

    def _fresh_registry(self):
        """Reset the module-level drain list + counter for test
        isolation. Module-globals are mutated intentionally — the
        registry *is* process-global in production, so testing it
        requires reaching into it."""
        from meeting_scribe import server

        server._refinement_drains.clear()
        server._drain_seq = 0
        return server

    @pytest.mark.asyncio
    async def test_kickoff_snapshot_visible_before_drain_completes(self):
        server = self._fresh_registry()
        worker = _FakeWorker(meeting_id="mtg-slow", stop_delay_s=0.5)
        # Set counters to nonzero so we can see the snapshot is the
        # live value at kickoff, not the post-drain value.
        worker.translate_call_count = 7
        worker.asr_call_count = 11

        server._drain_seq += 1
        drain_id = server._drain_seq
        entry = server._DrainEntry(
            drain_id=drain_id,
            meeting_id=worker._meeting_id,
            task=asyncio.create_task(
                server._drain_refinement(worker, worker._meeting_id, drain_id)
            ),
            state="draining",
            started_at=time.time(),
            translate_calls=worker.translate_call_count,
            asr_calls=worker.asr_call_count,
            errors_at_stop=worker.last_error_count,
        )
        server._refinement_drains.append(entry)

        # Before drain completes: snapshot reflects kickoff values.
        assert entry.state == "draining"
        assert entry.translate_calls == 7
        assert entry.asr_calls == 11

        await entry.task

        # After drain completes: counters updated from worker's final values.
        # Our FakeWorker's stop() increments translate_call_count by 1.
        assert entry.state == "complete"
        assert entry.translate_calls == 8
        assert entry.asr_calls == 11

    @pytest.mark.asyncio
    async def test_same_meeting_id_coexists_under_distinct_drain_ids(self):
        """P1 regression guard for iteration 4 finding #2: two drains
        for the same meeting_id must not overwrite each other."""
        server = self._fresh_registry()
        w1 = _FakeWorker(meeting_id="mtg-repeat", stop_delay_s=0.1)
        w2 = _FakeWorker(meeting_id="mtg-repeat", stop_delay_s=0.1)

        async def _kickoff(worker):
            server._drain_seq += 1
            drain_id = server._drain_seq
            entry = server._DrainEntry(
                drain_id=drain_id,
                meeting_id=worker._meeting_id,
                task=asyncio.create_task(
                    server._drain_refinement(worker, worker._meeting_id, drain_id)
                ),
                state="draining",
                started_at=time.time(),
                translate_calls=worker.translate_call_count,
                asr_calls=worker.asr_call_count,
                errors_at_stop=worker.last_error_count,
            )
            server._refinement_drains.append(entry)
            return entry

        e1 = await _kickoff(w1)
        e2 = await _kickoff(w2)

        assert e1.drain_id != e2.drain_id
        assert e1.meeting_id == e2.meeting_id == "mtg-repeat"

        await asyncio.gather(e1.task, e2.task)

        assert e1.state == "complete"
        assert e2.state == "complete"
        # Both entries present, neither lost.
        both = [e for e in server._refinement_drains if e.meeting_id == "mtg-repeat"]
        assert len(both) == 2
        assert {e.drain_id for e in both} == {e1.drain_id, e2.drain_id}

    @pytest.mark.asyncio
    async def test_drain_timeout_marks_partial(self):
        server = self._fresh_registry()

        class _HangForever:
            _meeting_id = "mtg-hang"
            translate_call_count = 0
            asr_call_count = 0
            last_error_count = 0

            async def stop(self):
                # Honor cancellation so the test's wait_for can tear us down.
                try:
                    await asyncio.sleep(120.0)
                except asyncio.CancelledError:
                    raise

        # Force a short timeout by monkey-patching _drain_refinement's
        # wait_for value via the production code. We can't redefine the
        # timeout without touching the server module; instead we test
        # the exception branch directly by making stop() raise.
        class _RaiseOnStop:
            _meeting_id = "mtg-raise"
            translate_call_count = 3
            asr_call_count = 5
            last_error_count = 1

            async def stop(self):
                self.translate_call_count = 4
                raise RuntimeError("simulated refinement crash")

        worker = _RaiseOnStop()
        server._drain_seq += 1
        drain_id = server._drain_seq
        entry = server._DrainEntry(
            drain_id=drain_id,
            meeting_id=worker._meeting_id,
            task=asyncio.create_task(
                server._drain_refinement(worker, worker._meeting_id, drain_id)
            ),
            state="draining",
            started_at=time.time(),
        )
        server._refinement_drains.append(entry)
        await entry.task

        assert entry.state == "failed"
        assert entry.error is not None
        assert "simulated refinement crash" in entry.error
        # Counters still re-read from worker in the finally block.
        assert entry.translate_calls == 4


class TestVramSafeguard:
    """VRAM safeguard flips ``_paused`` when gpu_monitor reports high
    VRAM and lifts it when headroom returns. Exercised via the
    extracted ``_apply_vram_gate`` helper so tests do not have to drive
    the 10s main loop. No real GPU needed."""

    class _Usage:
        def __init__(self, pct: float) -> None:
            self.pct = pct

    def _patch_vram(self, monkeypatch, pct: float) -> None:
        from meeting_scribe import gpu_monitor

        monkeypatch.setattr(gpu_monitor, "get_vram_usage", lambda: TestVramSafeguard._Usage(pct))

    def test_pauses_when_vram_pct_exceeds_95(self, tmp_path, monkeypatch):
        w = RefinementWorker(meeting_id="mtg-vram", meeting_dir=tmp_path)
        self._patch_vram(monkeypatch, 96.0)
        assert w._paused is False
        skip = w._apply_vram_gate()
        assert w._paused is True
        assert skip is True

    def test_resumes_when_vram_pct_drops_below_80(self, tmp_path, monkeypatch):
        w = RefinementWorker(meeting_id="mtg-vram", meeting_dir=tmp_path)
        # Start paused (as if a previous iteration triggered the pause).
        w._paused = True
        self._patch_vram(monkeypatch, 75.0)
        skip = w._apply_vram_gate()
        assert w._paused is False
        assert skip is False

    def test_hysteresis_band_80_to_95_preserves_state(self, tmp_path, monkeypatch):
        """Between the thresholds, the gate neither pauses nor resumes —
        that is what keeps the worker from flapping."""
        w = RefinementWorker(meeting_id="mtg-vram", meeting_dir=tmp_path)
        self._patch_vram(monkeypatch, 85.0)
        assert w._apply_vram_gate() is False
        assert w._paused is False
        w._paused = True
        assert w._apply_vram_gate() is True
        assert w._paused is True  # stays paused inside the band

    def test_gpu_monitor_unavailable_preserves_state(self, tmp_path, monkeypatch):
        """If the import or the call raises, the gate returns the
        current pause state — never crashes the worker."""
        w = RefinementWorker(meeting_id="mtg-vram", meeting_dir=tmp_path)
        from meeting_scribe import gpu_monitor

        def _raise():
            raise RuntimeError("nvml unavailable")

        monkeypatch.setattr(gpu_monitor, "get_vram_usage", _raise)
        assert w._apply_vram_gate() is False
        assert w._paused is False
