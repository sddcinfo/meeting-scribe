"""Lifecycle tests for :class:`ServerMicCapture` + ``reconcile_server_mic``.

Subprocess + ``_handle_audio`` are patched so the test runs offline,
on any machine, with no PipeWire backend. The contract being verified:

* ``start`` flips ``state.server_mic_active`` and prepends the
  ``[4B LE rate][PCM]`` wire shape to forwarded chunks.
* ``stop`` clears the flag and terminates the subprocess.
* ``reconcile_server_mic`` is idempotent + handles all transition
  pairs (none → running → retarget → stopped).
"""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, patch

import pytest

from meeting_scribe.audio import server_mic
from meeting_scribe.runtime import state


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test starts with a clean server_mic / server_mic_active."""
    saved_mic = getattr(state, "server_mic", None)
    saved_active = getattr(state, "server_mic_active", False)
    state.server_mic = None
    state.server_mic_active = False
    yield
    state.server_mic = saved_mic
    state.server_mic_active = saved_active


class _FakeStdout:
    """Synchronous read-yields-bytes-once-then-EOF stand-in for a Popen.stdout."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def read(self, n: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakePopen:
    """Lightweight replacement for ``subprocess.Popen``."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.stdout = _FakeStdout(chunks)
        self.terminated = False
        self._returncode: int | None = None
        # subprocess.run() constructs CompletedProcess(process.args, …) at
        # the end of its with-block, so any Popen stand-in needs this.
        self.args: list[str] = []

    def poll(self):
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            self._returncode = -15
        return self._returncode

    def kill(self) -> None:
        """Alias used by subprocess.run on its error/cleanup path."""
        self.terminate()

    def communicate(self, input=None, timeout=None):
        """Used by subprocess.run inside the context-manager flow."""
        return (b"", b"")

    # Context-manager protocol — subprocess.run() now uses ``with Popen(...)``
    # internally, so any monkeypatched stand-in must support it too.
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.terminate()


@pytest.mark.asyncio
async def test_start_sets_active_flag() -> None:
    capture = server_mic.ServerMicCapture(target_node="alsa_input.test")
    fake_popen = _FakePopen([b"\x00\x00" * 2400])

    with (
        patch.object(server_mic.subprocess, "Popen", return_value=fake_popen),
        patch("meeting_scribe.ws.audio_input._handle_audio", new_callable=AsyncMock) as mock_handle,
    ):
        await capture.start()
        # Yield long enough for the reader task to drain one chunk.
        await asyncio.sleep(0.05)
        assert state.server_mic_active is True
        assert mock_handle.called
        # First call's argv begins with the 48 kHz rate header.
        args, _ = mock_handle.call_args
        assert args[0][:4] == struct.pack("<I", server_mic.DEFAULT_CAPTURE_RATE)
        await capture.stop()
        assert state.server_mic_active is False


@pytest.mark.asyncio
async def test_stop_terminates_subprocess() -> None:
    capture = server_mic.ServerMicCapture(target_node="alsa_input.test")
    fake_popen = _FakePopen([b""])  # immediate EOF

    with (
        patch.object(server_mic.subprocess, "Popen", return_value=fake_popen),
        patch("meeting_scribe.ws.audio_input._handle_audio", new_callable=AsyncMock),
    ):
        await capture.start()
        await asyncio.sleep(0.05)
        await capture.stop()
        assert fake_popen.terminated or fake_popen.poll() is not None


@pytest.mark.asyncio
async def test_reconcile_starts_when_active_and_node_set() -> None:
    fake_popen = _FakePopen([b""])
    with (
        patch.object(server_mic.subprocess, "Popen", return_value=fake_popen),
        patch("meeting_scribe.ws.audio_input._handle_audio", new_callable=AsyncMock),
    ):
        await server_mic.reconcile_server_mic(mic_node="alsa_input.foo", mic_active=True)
        assert state.server_mic is not None
        assert state.server_mic.target_node == "alsa_input.foo"
        await state.server_mic.stop()
        state.server_mic = None


@pytest.mark.asyncio
async def test_reconcile_noop_when_inactive_and_no_capture() -> None:
    await server_mic.reconcile_server_mic(mic_node="", mic_active=False)
    assert state.server_mic is None


@pytest.mark.asyncio
async def test_reconcile_stops_when_toggled_off() -> None:
    fake_popen = _FakePopen([b""])
    with (
        patch.object(server_mic.subprocess, "Popen", return_value=fake_popen),
        patch("meeting_scribe.ws.audio_input._handle_audio", new_callable=AsyncMock),
    ):
        await server_mic.reconcile_server_mic(mic_node="alsa_input.foo", mic_active=True)
        assert state.server_mic is not None
        await server_mic.reconcile_server_mic(mic_node="alsa_input.foo", mic_active=False)
        assert state.server_mic is None
        assert state.server_mic_active is False


@pytest.mark.asyncio
async def test_reconcile_retargets_on_node_change() -> None:
    spawn_calls: list[str] = []

    def fake_spawn(self):
        # Sniff the target_node passed to argv reconstruction.
        spawn_calls.append(self.target_node)
        return _FakePopen([b""])

    with (
        patch.object(server_mic.ServerMicCapture, "_spawn", fake_spawn),
        patch("meeting_scribe.ws.audio_input._handle_audio", new_callable=AsyncMock),
    ):
        await server_mic.reconcile_server_mic(mic_node="alsa_input.A", mic_active=True)
        first = state.server_mic
        await server_mic.reconcile_server_mic(mic_node="alsa_input.B", mic_active=True)
        second = state.server_mic
        assert first is not second
        assert second.target_node == "alsa_input.B"
        await second.stop()
        state.server_mic = None
