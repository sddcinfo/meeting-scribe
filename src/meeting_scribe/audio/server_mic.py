"""Server-side microphone capture from a PipeWire source node into ASR.

Spawns ``pw-record --target=<node> --format=s16 --rate=48000
--channels=1 -`` and writes successive 50 ms PCM frames into
``ws.audio_input._handle_audio`` with the canonical
``[4B LE uint32 sample_rate][PCM]`` wire shape — exactly what the
browser AudioWorklet would push, so all downstream code (resampler,
audio writer, ASR) is unchanged.

When :class:`ServerMicCapture` is running, the WS handler in
:mod:`meeting_scribe.ws.audio_input` checks ``state.server_mic_active``
and drops inbound binary frames so the browser-mic stream and
server-mic stream don't both feed ASR concurrently (the operator
explicitly chose "server mic replaces browser mic" semantics).

Why USB capture beats HFP-mSBC for the meeting room
---------------------------------------------------
Devices like the Poly Sync 20-M expose a 48 kHz mono UAC2 stream with
on-device echo cancellation, noise suppression, and AGC. HFP-mSBC
maxes out at 16 kHz mono transparent SCO with no DSP. Trusting the
USB device's hardware DSP and resampling 48 kHz → 16 kHz server-side
with torchaudio Kaiser sinc gives noticeably cleaner ASR audio than
the BT path, and ASR is unaware of the change because the wire format
into ``_handle_audio`` is identical.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import subprocess
import time

from meeting_scribe.audio.audio_routing import (
    DEFAULT_CAPTURE_CHANNELS,
    DEFAULT_CAPTURE_RATE,
)
from meeting_scribe.audio.native_rate import detect_capture_rate
from meeting_scribe.runtime import state

logger = logging.getLogger(__name__)


# 50 ms frame at the chosen capture rate.  The rate is no longer a
# module-level constant because devices like the Dell SP325 advertise
# 16 kHz natively and degrade noticeably when forced to 48 kHz — see
# audio/native_rate.py.  Frame size + rate header are computed per
# ServerMicCapture instance below.
_FRAME_DURATION_S = 0.05
_LOCAL_TTS_ECHO_SUPPRESS = os.environ.get("SCRIBE_LOCAL_TTS_ECHO_SUPPRESS", "1") != "0"


def _frame_bytes_for(rate: int) -> int:
    """Bytes per 50 ms frame at ``rate`` (mono int16)."""
    return int(rate * _FRAME_DURATION_S) * 2 * DEFAULT_CAPTURE_CHANNELS


def _rate_header_for(rate: int) -> bytes:
    """Wire-protocol header prepended to each chunk: ``<I`` LE uint32."""
    return struct.pack("<I", rate)


class ServerMicCapture:
    """Long-lived pw-record subprocess + asyncio reader task.

    Lifecycle:

    * :meth:`start` spawns pw-record and the reader task. Idempotent.
    * The reader pulls ``_FRAME_BYTES`` at a time and forwards them to
      ``_handle_audio``. EOF / read error → respawn with a 1 s
      backoff (the device might have momentarily disappeared during a
      USB renumeration).
    * :meth:`stop` terminates the subprocess and awaits the reader.
      Idempotent.

    Single instance per process — owned by the lifespan and stored on
    ``state.server_mic`` so the admin route can stop+restart it when
    the operator changes the source.
    """

    def __init__(self, *, target_node: str, capture_rate: int | None = None) -> None:
        self.target_node = target_node
        # Per-device rate detection. SP325 → 16000 (native), Poly → 48000.
        # The caller can pin an explicit rate (env override / unit tests)
        # via the optional argument; production paths always autodetect.
        self.capture_rate = (
            capture_rate
            if capture_rate is not None
            else detect_capture_rate(target_node, default=DEFAULT_CAPTURE_RATE)
        )
        self.frame_bytes = _frame_bytes_for(self.capture_rate)
        self.rate_header = _rate_header_for(self.capture_rate)
        self.bytes_in = 0
        self.respawn_count = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._reader_task is not None and not self._reader_task.done():
            return
        self._stopping = False
        self._reader_task = asyncio.create_task(self._loop(), name="server-mic-loop")
        state.server_mic_active = True
        logger.info(
            "server-mic: started target=%s rate=%d (native-detected)",
            self.target_node,
            self.capture_rate,
        )

    async def stop(self) -> None:
        self._stopping = True
        state.server_mic_active = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:
                pass  # pw-record already exited — reader loop will observe EOF
        task = self._reader_task
        self._reader_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError, Exception:
                pass  # reader loop is being torn down; finally-block in _loop cleans pw-record
        self._proc = None
        logger.info("server-mic: stopped target=%s bytes_in=%d", self.target_node, self.bytes_in)

    def _spawn(self) -> subprocess.Popen[bytes] | None:
        argv = [
            "pw-record",
            f"--target={self.target_node}",
            "--format=s16",
            f"--rate={self.capture_rate}",
            f"--channels={DEFAULT_CAPTURE_CHANNELS}",
            "-",
        ]
        try:
            return subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError as e:
            logger.error("pw-record unavailable: %r", e)
            return None

    async def _loop(self) -> None:
        # Imported here to avoid the import cycle at module load
        # (ws/audio_input → routes → app → audio modules).
        from meeting_scribe.ws.audio_input import _handle_audio

        loop = asyncio.get_running_loop()
        while not self._stopping:
            self._proc = self._spawn()
            if self._proc is None or self._proc.stdout is None:
                # No pw-record binary or spawn failed — wait briefly
                # and retry, but bound the retry rate so a permanent
                # config error doesn't burn CPU.
                await asyncio.sleep(2.0)
                self.respawn_count += 1
                continue
            stdout = self._proc.stdout
            try:
                while not self._stopping:
                    chunk = await loop.run_in_executor(None, stdout.read, self.frame_bytes)
                    if not chunk:
                        break
                    self.bytes_in += len(chunk)
                    if _LOCAL_TTS_ECHO_SUPPRESS and time.monotonic() < getattr(
                        state, "local_tts_playback_until", 0.0
                    ):
                        state.metrics.local_tts_echo_suppressed_chunks += 1
                        continue
                    # Soft input mute (audio_routing): privacy pause
                    # honored at the server-mic boundary too. The
                    # ``_handle_audio`` path has the same gate, but
                    # skipping here saves the resample call and keeps
                    # the metric symmetric with the browser-mic path.
                    if state.mic_input_muted:
                        state.metrics.mic_muted_chunks_dropped += 1
                        continue
                    try:
                        await _handle_audio(self.rate_header + chunk)
                    except Exception:
                        logger.exception("server-mic: _handle_audio raised")
            finally:
                proc = self._proc
                self._proc = None
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=1)
                    except OSError, subprocess.TimeoutExpired:
                        pass  # pw-record already gone or refused to exit cleanly; respawn loop covers it
            if self._stopping:
                break
            self.respawn_count += 1
            logger.info("server-mic: pw-record exited, respawning (count=%d)", self.respawn_count)
            await asyncio.sleep(1.0)


async def reconcile_server_mic(*, mic_node: str, mic_active: bool) -> None:
    """Start, stop, or retarget the global :class:`ServerMicCapture`.

    Called from lifespan boot and from the admin route after a
    routing-config save. Idempotent on every transition pair.
    """
    current = getattr(state, "server_mic", None)
    desired = bool(mic_active and mic_node)

    if not desired:
        if current is not None:
            await current.stop()
            state.server_mic = None
        return

    # If a capture is already running on the right node, leave it alone.
    if current is not None and current.target_node == mic_node:
        if current._reader_task is None or current._reader_task.done():
            await current.start()
        return

    if current is not None:
        await current.stop()
    state.server_mic = ServerMicCapture(target_node=mic_node)
    await state.server_mic.start()
