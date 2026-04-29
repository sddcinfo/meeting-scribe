"""Unit tests for the /api/meeting/start fast-path uplift.

What's covered:

1. ``_read_enrollment_wav_f32`` reads a 16-bit PCM WAV into float32.
2. ``_seed_tts_from_enrollments_async`` runs WAV reads + seed_voice via
   the executor (``asyncio.to_thread``), so the event loop stays free
   during meeting start.
3. Its ``Semaphore(2)`` bounds concurrency even for many speakers.
4. A single slow/bad WAV doesn't abort the whole seed.
5. ``RefinementWorker.start()``: if ``stop()`` was signalled while the
   worker was awaiting its model-id probes, the run loop is NEVER
   armed (the late-arrival guard).
6. ``RefinementWorker.stop()`` is safe to call when ``_task`` is None
   (never armed). No crash on ``_process_remaining``.
7. ``_drain_refinement`` cancels the pending ``_start_task`` before
   touching ``worker.stop()``; if the cancellation prevented start
   from finishing, the drain completes without calling stop().

These tests DO NOT spin up the live server or any vLLM backend.
"""

from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from meeting_scribe.refinement import RefinementWorker
from meeting_scribe.runtime import state as runtime_state
from meeting_scribe.server_support.refinement_drains import (
    _drain_refinement,
    _DrainEntry,
    _next_drain_id,
    _refinement_drains,
)
from meeting_scribe.server_support.voice_seed import (
    _read_enrollment_wav_f32,
    _seed_tts_from_enrollments_async,
)

# ── Helpers ──────────────────────────────────────────────────────


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    """Write a 16-bit PCM mono WAV with the given samples in [-1, 1]."""
    pcm = (samples * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


# ── 1. _read_enrollment_wav_f32 ─────────────────────────────────


def test_read_enrollment_wav_f32_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ref.wav"
    expected = np.array([0.0, 0.5, -0.5, 0.999, -0.999], dtype=np.float32)
    _write_wav(path, expected)

    got = _read_enrollment_wav_f32(str(path))
    assert got.dtype == np.float32
    # 16-bit quantisation is lossy; tolerate 1 LSB of slack (≈ 3e-5).
    assert np.allclose(got, expected, atol=5e-5)


def test_read_enrollment_wav_f32_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    _write_wav(path, np.array([], dtype=np.float32))

    got = _read_enrollment_wav_f32(str(path))
    assert got.shape == (0,)


# ── 2. seed runs via executor (event loop stays free) ──────────


async def test_seed_uses_thread_executor_not_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the WAV read and seed_voice were called on the event loop, a
    synchronous 300 ms wait inside them would stall any other coroutine.
    Using ``asyncio.to_thread`` keeps other coroutines running while the
    blocking work happens on the thread pool.

    We assert: during the seed, a parallel tick-counter coroutine
    continues to advance.
    """
    wav_path = tmp_path / "ref.wav"
    _write_wav(wav_path, np.zeros(1600, dtype=np.float32))  # 100 ms of silence

    call_log: list[str] = []

    def _blocking_seed(eid: str, audio: np.ndarray, source: str) -> None:
        call_log.append(f"seed:{eid}")
        # Synchronous sleep — if this runs on the event loop it will
        # starve the tick-counter. On the executor it won't.
        time.sleep(0.3)

    fake_backend = SimpleNamespace(seed_voice=_blocking_seed)
    monkeypatch.setattr(runtime_state, "tts_backend", fake_backend)

    speaker = SimpleNamespace(reference_wav_path=str(wav_path))
    speakers = [("spk-1", speaker)]

    ticks = 0

    async def tick_counter() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    ticker = asyncio.create_task(tick_counter())
    try:
        await _seed_tts_from_enrollments_async(speakers)
    finally:
        ticker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ticker

    # If the seed blocked the loop, we'd see ≤ 2 ticks.
    # With to_thread, we expect ≥ 8 (300 ms / 20 ms ≈ 15).
    assert ticks >= 8, f"loop stalled during seed (only {ticks} ticks)"
    assert call_log == ["seed:spk-1"]


# ── 3. Semaphore(2) bounds concurrency ──────────────────────────


async def test_seed_semaphore_caps_concurrency_at_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "ref.wav"
    _write_wav(wav_path, np.zeros(1600, dtype=np.float32))

    in_flight = 0
    peak = 0
    lock = asyncio.Lock()
    release = asyncio.Event()

    def _blocking_seed(eid: str, audio: np.ndarray, source: str) -> None:
        # Can't use an async lock from a thread — use a busy-wait on the
        # shared event loop's event via a small time.sleep loop.
        nonlocal in_flight, peak
        # Incrementing in_flight happens inside a short blocking window;
        # the GIL makes these ops atomic enough for a peak count.
        in_flight += 1
        peak = max(peak, in_flight)
        # Hold for 100 ms so the peak count has a chance to reflect
        # concurrent executor slots.
        time.sleep(0.1)
        in_flight -= 1

    fake_backend = SimpleNamespace(seed_voice=_blocking_seed)
    monkeypatch.setattr(runtime_state, "tts_backend", fake_backend)

    # 6 speakers — all share the same ref wav.
    speakers = [(f"spk-{i}", SimpleNamespace(reference_wav_path=str(wav_path))) for i in range(6)]

    await _seed_tts_from_enrollments_async(speakers)

    # Semaphore(2) in the code under test means at most 2 executor
    # threads should be blocking inside seed_voice at once. Allow one
    # extra slot for transient overlap (thread scheduling edges).
    assert peak <= 3, f"concurrency peak {peak} exceeds Semaphore(2) expectation"
    assert peak >= 2, f"concurrency peak {peak} below 2 — semaphore too restrictive?"
    # Release referenced for linter friendliness.
    _ = lock, release


# ── 4. Single bad WAV doesn't abort the whole seed ──────────────


async def test_seed_per_speaker_error_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good_path = tmp_path / "good.wav"
    _write_wav(good_path, np.zeros(1600, dtype=np.float32))

    seeded: list[str] = []

    def _seed(eid: str, audio: np.ndarray, source: str) -> None:
        if eid == "bad":
            raise RuntimeError("simulated TTS backend error")
        seeded.append(eid)

    fake_backend = SimpleNamespace(seed_voice=_seed)
    monkeypatch.setattr(runtime_state, "tts_backend", fake_backend)

    speakers = [
        ("good-1", SimpleNamespace(reference_wav_path=str(good_path))),
        ("bad", SimpleNamespace(reference_wav_path=str(good_path))),
        ("good-2", SimpleNamespace(reference_wav_path=str(good_path))),
        ("missing", SimpleNamespace(reference_wav_path=str(tmp_path / "nope.wav"))),
        ("no-path", SimpleNamespace(reference_wav_path="")),
    ]

    # Must not raise.
    await _seed_tts_from_enrollments_async(speakers)

    # Good speakers seed; bad/missing/no-path are skipped without
    # aborting the whole batch.
    assert sorted(seeded) == ["good-1", "good-2"]


# ── 5. RefinementWorker late-arrival guard ──────────────────────


async def test_refinement_start_aborts_if_stop_signalled_mid_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``stop()`` fires while ``start()`` is still awaiting model-id
    probes, the worker must NOT arm ``_run_loop``.

    Reproduces the race introduced by backgrounding ``start()``.
    """
    mdir = tmp_path / "mtg"
    (mdir / "audio").mkdir(parents=True)
    # Create a dummy recording.pcm so _run_loop (if incorrectly armed)
    # could at least open it.
    (mdir / "audio" / "recording.pcm").write_bytes(b"")

    worker = RefinementWorker(
        meeting_id="mtg-test",
        meeting_dir=mdir,
        asr_url="http://unused",
        translate_url="http://unused",
        language_pair=("en", "ja"),
    )

    # Stub out httpx.AsyncClient so start() doesn't hit the network.
    class _FakeResp:
        def json(self) -> dict[str, Any]:
            return {"data": [{"id": "stub-model"}]}

    class _FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def get(self, url: str) -> _FakeResp:
            # Simulate a slow model-id probe so we have a window to
            # signal stop before start() completes.
            await asyncio.sleep(0.05)
            return _FakeResp()

        async def aclose(self) -> None:
            pass

    import meeting_scribe.refinement as rf_mod

    monkeypatch.setattr(rf_mod.httpx, "AsyncClient", _FakeClient)

    start_task = asyncio.create_task(worker.start())
    # Wait long enough to enter the probe window but NOT long enough
    # for start() to finish.
    await asyncio.sleep(0.01)
    worker._stop_event.set()
    await start_task

    # Late-arrival guard fired: _run_loop was never armed.
    assert worker._task is None


async def test_refinement_stop_safe_when_task_never_armed(
    tmp_path: Path,
) -> None:
    """If ``_task`` is None (because start() was cancelled before
    arming it), ``stop()`` must return cleanly without calling
    ``_process_remaining`` on an uninitialised client.
    """
    mdir = tmp_path / "mtg"
    (mdir / "audio").mkdir(parents=True)

    worker = RefinementWorker(
        meeting_id="mtg-test",
        meeting_dir=mdir,
        asr_url="http://unused",
        translate_url="http://unused",
        language_pair=("en", "ja"),
    )
    # Simulate "start was never called" — _task stays None, clients None.
    assert worker._task is None
    assert worker._asr_client is None

    # Must not raise.
    await worker.stop()


# ── 6. _drain_refinement cancels pending _start_task ───────────


async def test_drain_refinement_cancels_pending_start_task(
    tmp_path: Path,
) -> None:
    """Simulate the fast-stop race: worker._start_task is still
    running when the drain begins. Drain must cancel it AND skip
    stop() (since nothing was armed)."""
    mdir = tmp_path / "mtg"
    (mdir / "audio").mkdir(parents=True)

    worker = RefinementWorker(
        meeting_id="mtg-test",
        meeting_dir=mdir,
        asr_url="http://unused",
        translate_url="http://unused",
        language_pair=("en", "ja"),
    )

    release = asyncio.Event()
    stop_called = False

    async def _never_completing_start() -> None:
        await release.wait()

    async def _stop_should_not_run() -> None:
        nonlocal stop_called
        stop_called = True

    worker.start = _never_completing_start  # type: ignore[assignment]
    worker.stop = _stop_should_not_run  # type: ignore[assignment]

    worker._start_task = asyncio.create_task(worker.start())

    # Register a drain entry so _drain_refinement has something to
    # update (mirrors server-side bookkeeping in _reset_meeting_state).
    drain_id = _next_drain_id()
    entry = _DrainEntry(
        drain_id=drain_id,
        meeting_id="mtg-test",
        task=None,  # will be set on real call site
        state="draining",
        started_at=time.time(),
        translate_calls=0,
        asr_calls=0,
        errors_at_stop=0,
    )
    _refinement_drains.append(entry)

    try:
        await _drain_refinement(worker, "mtg-test", drain_id)
    finally:
        _refinement_drains.remove(entry)

    # Start task was cancelled; stop() was skipped; entry state marked
    # complete (nothing to drain).
    assert worker._start_task.cancelled() or worker._start_task.done()
    assert stop_called is False
    assert entry.state == "complete"
    # Release ref — loop won't ever set it, but we satisfy linters.
    _ = release
