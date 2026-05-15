"""Server-side audio-out client that plays TTS bytes through the local
PipeWire default sink.

Registered into ``state._audio_out_clients`` at server lifespan when
:func:`should_enable_local_sink` returns True. The fan-out in
:mod:`meeting_scribe.audio.output_pipeline` treats this listener like
any other audio-out consumer: same ``preferred_language`` /
``voice_mode`` / ``interpretation_mode`` filtering, same ``wav-pcm``
byte stream as a guest WebSocket would receive.

Implementation: a long-lived ``pw-cat --playback`` subprocess receives
raw PCM written to its stdin. Each incoming WAV frame is parsed via
:mod:`wave`, the RIFF header is stripped, and the PCM payload is
forwarded. On a broken pipe or process death we transparently respawn
on the next delivery — the subprocess is best-effort and a single
crashed pw-cat must not stall the fan-out.

The module is intentionally independent of the (still-unimplemented)
:mod:`meeting_scribe.audio.bt_bridge` state machine. A BT headset is
just one possible default sink; if the operator's BT device is paired
+ in HFP profile, the system default sink resolves to the BT speaker
node and TTS plays through it. ALSA / HDMI sinks are handled by the
same code path with no special-casing.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import wave

from meeting_scribe.audio.audio_routing import (
    SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE,
    SETTINGS_AUDIO_MEETING_MIC_ACTIVE,
    SETTINGS_AUDIO_MEETING_MIC_NODE,
    SETTINGS_AUDIO_ROOM_TTS_SINK_NODE,
    audio_nodes_share_physical_device,
)
from meeting_scribe.runtime import state
from meeting_scribe.server_support.sessions import ClientSession
from meeting_scribe.server_support.settings_store import (
    _effective_interpretation_enabled,
    _effective_tts_voice_mode,
    _load_settings_override,
)

logger = logging.getLogger(__name__)


# TTS output is 24 kHz mono int16 — see
# ``output_pipeline._send_audio_to_listeners`` (source_rate=24000) and
# ``_build_riff_wav`` (16-bit PCM mono). Any mismatch with the WAV
# header would be caught in :func:`_strip_wav_header` and skipped.
LOCAL_SINK_PCM_RATE = 24000
LOCAL_SINK_PCM_CHANNELS = 1
LOCAL_SINK_PCM_SAMPWIDTH = 2
LOCAL_SINK_WRITE_TIMEOUT_S = float(os.environ.get("SCRIBE_LOCAL_SINK_WRITE_TIMEOUT_S", "2.5"))
LOCAL_SINK_QUEUE_MAXSIZE = int(os.environ.get("SCRIBE_LOCAL_SINK_QUEUE_MAXSIZE", "64"))

# Settings keys — operator-tunable from the admin UI's Audio routing
# card. ``SETTINGS_LOCAL_SINK_TARGET`` is the legacy alias retained so
# pre-routing-UI deployments don't lose their pinned sink on upgrade.
SETTINGS_LOCAL_SINK_LANGUAGE = "local_sink_language"
SETTINGS_ADMIN_TTS_LANGUAGE = "admin_tts_language"
SETTINGS_ROOM_TTS_LANGUAGE = "room_tts_language"
SETTINGS_LOCAL_SINK_MODE = "local_sink_mode"
SETTINGS_LOCAL_SINK_TARGET = "local_sink_target_node"
LOCAL_SINK_ROLE_ADMIN = "admin_tts"
LOCAL_SINK_ROLE_ROOM = "room_tts"


def _resolve_sink_language(
    settings: dict,
    key: str,
    *,
    default: str = "en",
    allow_all: bool = False,
) -> str:
    """Return the target language for server-side room/headset TTS.

    Empty ``preferred_language`` means "synthesize every translated language".
    That is useful for a browser monitor, but not for a single physical room
    sink: in bidirectional interpretation it doubles TTS load and can play the
    source speaker's language back into the room. Keep the configured target
    language active even while consecutive interpretation buffering is enabled.
    Operators can still opt into all languages with ``all`` when debugging.
    """
    value = str(settings.get(key, default) or default).strip().lower()
    if value in {"all", "*", "any"}:
        if not allow_all:
            return default
        return ""
    return value or default


def resolve_local_sink_language(settings: dict) -> str:
    """Legacy/admin physical TTS target language."""
    return _resolve_sink_language(
        settings,
        SETTINGS_ADMIN_TTS_LANGUAGE
        if SETTINGS_ADMIN_TTS_LANGUAGE in settings
        else SETTINGS_LOCAL_SINK_LANGUAGE,
        default="en",
        allow_all=False,
    )


def resolve_room_sink_language(settings: dict) -> str:
    """In-room physical TTS target language; empty means all directions."""
    return _resolve_sink_language(
        settings, SETTINGS_ROOM_TTS_LANGUAGE, default="all", allow_all=True
    )


def _resolve_sink_target(settings: dict, role: str = LOCAL_SINK_ROLE_ADMIN) -> str | None:
    """Pick the configured PipeWire sink node name from settings.

    Priority: the new audio-routing key (set by the admin UI) wins over
    the legacy ``local_sink_target_node`` key. Empty string → None
    (PipeWire default sink).
    """
    if role == LOCAL_SINK_ROLE_ROOM:
        target = settings.get(SETTINGS_AUDIO_ROOM_TTS_SINK_NODE)
    else:
        target = settings.get(SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE)
        if not target:
            target = settings.get(SETTINGS_LOCAL_SINK_TARGET)
    return target or None


def _strip_wav_header(data: bytes) -> bytes | None:
    """Return the PCM payload of a 16-bit mono RIFF WAV, or None.

    The fan-out always emits 16-bit mono PCM (see
    ``output_pipeline._build_riff_wav``); anything else is unexpected
    and dropped rather than fed garbage to pw-cat.
    """
    if not data.startswith(b"RIFF"):
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getsampwidth() != LOCAL_SINK_PCM_SAMPWIDTH:
                return None
            if wf.getnchannels() != LOCAL_SINK_PCM_CHANNELS:
                return None
            return wf.readframes(wf.getnframes())
    except wave.Error, EOFError:
        return None


class LocalSinkListener:
    """Plays the audio fan-out's wav-pcm bytes through the local PipeWire sink.

    Implements the :class:`AudioListener` Protocol implicitly via
    :meth:`send_bytes` + :meth:`__hash__`. Holds a long-lived
    ``pw-cat --playback`` subprocess fed raw PCM. The serialization
    lock prevents two concurrent fan-out deliveries from interleaving
    PCM into the same stdin.
    """

    def __init__(
        self,
        *,
        target_node: str | None = None,
        role: str = LOCAL_SINK_ROLE_ADMIN,
        echo_guard: bool = False,
    ) -> None:
        self.audio_format = "wav-pcm"
        self.role = role
        self.echo_guard = echo_guard
        self.deliveries = 0
        self.respawn_count = 0
        self.playback_until = 0.0
        self.last_played_target_lang: str | None = None
        self.last_played_text = ""
        self._target_node = target_node or None
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[tuple[bytes, float]] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._queued_playback_until = 0.0
        self.dropped_buffers = 0

    def __hash__(self) -> int:
        return id(self)

    @property
    def cookies(self) -> dict[str, str]:
        return {}

    async def send_bytes(self, data: bytes) -> None:
        pcm = _strip_wav_header(data)
        if pcm is None:
            return
        duration_s = len(pcm) / (
            LOCAL_SINK_PCM_RATE * LOCAL_SINK_PCM_CHANNELS * LOCAL_SINK_PCM_SAMPWIDTH
        )
        now = asyncio.get_running_loop().time()
        self._queued_playback_until = max(self._queued_playback_until, now) + duration_s
        if self.echo_guard:
            state.local_tts_playback_until = max(
                getattr(state, "local_tts_playback_until", 0.0),
                self._queued_playback_until + 0.75,
            )
        queue = self._ensure_queue()
        try:
            queue.put_nowait((pcm, duration_s))
        except asyncio.QueueFull:
            self.dropped_buffers += 1
            logger.warning(
                "local-sink playback queue full; dropping newest TTS buffer "
                "queue_size=%d queue_max=%d dropped=%d",
                queue.qsize(),
                LOCAL_SINK_QUEUE_MAXSIZE,
                self.dropped_buffers,
            )
        self._ensure_writer_task()

    def _ensure_queue(self) -> asyncio.Queue[tuple[bytes, float]]:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=LOCAL_SINK_QUEUE_MAXSIZE)
        return self._queue

    def _ensure_writer_task(self) -> None:
        task = self._writer_task
        if task is None or task.done():
            self._writer_task = asyncio.create_task(self._writer_loop(), name="local-sink-writer")

    async def _writer_loop(self) -> None:
        queue = self._ensure_queue()
        while True:
            pcm, duration_s = await queue.get()
            try:
                async with self._lock:
                    timeout_s = max(LOCAL_SINK_WRITE_TIMEOUT_S, duration_s + 1.0)
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(self._write_pcm_sync, pcm),
                            timeout=timeout_s,
                        )
                        self.deliveries += 1
                    except TimeoutError:
                        logger.warning(
                            "local-sink pw-cat write timed out after %.1fs; respawning",
                            timeout_s,
                        )
                        self.respawn_count += 1
                        self._kill_proc()
                    except (BrokenPipeError, OSError) as e:
                        logger.info("local-sink pw-cat dead, respawning: %r", e)
                        self.respawn_count += 1
                        self._kill_proc()
            finally:
                queue.task_done()

    def _write_pcm_sync(self, pcm: bytes) -> None:
        proc = self._ensure_proc()
        if proc is None or proc.stdin is None:
            return
        proc.stdin.write(pcm)
        proc.stdin.flush()

    def _ensure_proc(self) -> subprocess.Popen[bytes] | None:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        argv = [
            "pw-cat",
            "--playback",
            f"--rate={LOCAL_SINK_PCM_RATE}",
            f"--channels={LOCAL_SINK_PCM_CHANNELS}",
            "--format=s16",
        ]
        if self._target_node:
            argv += ["--target", self._target_node]
        argv += ["-"]
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            logger.warning("pw-cat unavailable: %r", e)
            self._proc = None
            return None
        return self._proc

    def _kill_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass  # pw-cat already exited — nothing left to signal

    def retarget(self, target_node: str | None) -> None:
        """Update the target sink node + kill any live pw-cat.

        Next ``send_bytes`` respawns pw-cat against the new node — no
        listener re-registration in the fan-out, no race against TTS
        deliveries that arrive mid-switch (those go to the old node
        and the broken-pipe path catches up). Idempotent on repeated
        calls with the same target.
        """
        new_target = target_node or None
        if new_target == self._target_node and self._proc is not None:
            return
        self._target_node = new_target
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass  # pw-cat stdin already closed — next send_bytes will respawn on the new target

    async def drain(self) -> None:
        """Wait for queued local playback writes; used by focused tests."""
        if self._queue is not None:
            await self._queue.join()

    def shutdown(self) -> None:
        task = self._writer_task
        self._writer_task = None
        if task is not None:
            task.cancel()
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass  # pw-cat already gone — proceed to the wait/terminate fallback below
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except OSError, subprocess.TimeoutExpired:
                pass  # shutdown is best-effort; OS will reap pw-cat when this process exits


def should_enable_local_sink() -> bool:
    """Enable only when an explicit local TTS sink is configured.

    Never fall back to PipeWire's default sink here: on a meeting-room box the
    default may be the Poly speaker, so a browser/laptop mic can hear its own
    TTS output. Admin TTS must target an explicitly chosen private/headphone
    sink; in-room TTS must target an explicitly chosen room sink.
    """
    settings = _load_settings_override()
    if _resolve_sink_target(settings, LOCAL_SINK_ROLE_ADMIN):
        return True
    return bool(_resolve_safe_room_sink_target(settings))


def _resolve_safe_room_sink_target(settings: dict) -> str | None:
    target = _resolve_sink_target(settings, LOCAL_SINK_ROLE_ROOM)
    if not target:
        return None
    mic_node = str(settings.get(SETTINGS_AUDIO_MEETING_MIC_NODE) or "").strip()
    mic_active = bool(settings.get(SETTINGS_AUDIO_MEETING_MIC_ACTIVE, False))
    if mic_active and audio_nodes_share_physical_device(mic_node, target):
        return target
    logger.warning(
        "room TTS sink ignored until matching server mic is active: mic_active=%s mic=%s sink=%s",
        mic_active,
        mic_node or "<none>",
        target,
    )
    return None


def _local_sink_session_for_role(settings: dict, role: str) -> ClientSession:
    interpretation_enabled = _effective_interpretation_enabled()
    if role == LOCAL_SINK_ROLE_ROOM:
        return ClientSession(
            send_audio=True,
            voice_mode=_effective_tts_voice_mode(),
            preferred_language=resolve_room_sink_language(settings),
            interpretation_mode=settings.get(SETTINGS_LOCAL_SINK_MODE, "translation"),
            audio_format="wav-pcm",
            delivery_mode="consecutive" if interpretation_enabled else "simultaneous",
            transport="room_sink",
        )
    return ClientSession(
        send_audio=True,
        voice_mode=_effective_tts_voice_mode(),
        preferred_language=resolve_local_sink_language(settings),
        interpretation_mode=settings.get(SETTINGS_LOCAL_SINK_MODE, "translation"),
        audio_format="wav-pcm",
        delivery_mode="simultaneous",
        transport="bt_headset",
    )


def register_local_sink_listener(role: str = LOCAL_SINK_ROLE_ADMIN) -> LocalSinkListener:
    """Instantiate + register the local-sink listener into the audio fan-out.

    Reads listen prefs from the persisted settings store; defaults
    target English translation-only output suitable for a BT-mic
    interpreter scenario.
    """
    settings = _load_settings_override()
    listener = LocalSinkListener(
        target_node=_resolve_sink_target(settings, role),
        role=role,
        echo_guard=role == LOCAL_SINK_ROLE_ROOM,
    )
    pref = _local_sink_session_for_role(settings, role)
    state._audio_out_clients.add(listener)
    state._audio_out_prefs[listener] = pref
    logger.info(
        "local-sink listener registered: role=%s language=%s mode=%s voice=%s target=%s",
        listener.role,
        pref.preferred_language,
        pref.interpretation_mode,
        pref.voice_mode,
        listener._target_node or "<default>",
    )
    return listener


def ensure_local_sink_listener_registered() -> LocalSinkListener | None:
    """Ensure the configured room sink is present in the audio fan-out.

    Bluetooth/PipeWire route changes can leave the persisted room sink valid
    while the in-process listener has been removed or still points at an old
    node. This is deliberately cheap: it only touches in-memory listener state
    and does not spawn ``pw-cat`` until the next audio delivery.

    Symmetry guarantee: when ``should_enable_local_sink()`` flips back to
    False (operator clears ``room_sink_node`` / unchecks GB10 TTS), any
    previously-registered ``LocalSinkListener`` is purged. Without this,
    ``_listener_tts_demand`` keeps returning a non-empty set, the TTS
    producer keeps enqueueing segments that have no real sink to deliver
    to, ``state.tts_in_flight`` accumulates, and the health evaluator
    fires the "TTS stalled" badge despite the UI showing outputs off.
    """
    if not should_enable_local_sink():
        for client in list(state._audio_out_clients):
            if isinstance(client, LocalSinkListener):
                unregister_local_sink_listener(client)
        return None
    settings = _load_settings_override()
    desired_roles = []
    if _resolve_sink_target(settings, LOCAL_SINK_ROLE_ADMIN):
        desired_roles.append(LOCAL_SINK_ROLE_ADMIN)
    if _resolve_safe_room_sink_target(settings):
        desired_roles.append(LOCAL_SINK_ROLE_ROOM)
    first: LocalSinkListener | None = None
    for client in list(state._audio_out_clients):
        if not isinstance(client, LocalSinkListener):
            continue
        role = getattr(client, "role", LOCAL_SINK_ROLE_ADMIN)
        if role not in desired_roles:
            unregister_local_sink_listener(client)
            continue
        client.retarget(_resolve_sink_target(settings, role))
        pref = state._audio_out_prefs.get(client)
        if pref is not None:
            muted = getattr(pref, "delivery_mode", "simultaneous") == "drop"
            replacement = _local_sink_session_for_role(settings, role)
            pref.transport = replacement.transport
            pref.voice_mode = replacement.voice_mode
            pref.preferred_language = replacement.preferred_language
            pref.interpretation_mode = replacement.interpretation_mode
            if not muted:
                pref.delivery_mode = replacement.delivery_mode
        first = first or client
        if role in desired_roles:
            desired_roles.remove(role)
    for role in desired_roles:
        client = register_local_sink_listener(role)
        first = first or client
    return first


def unregister_local_sink_listener(listener: LocalSinkListener) -> None:
    """Remove the listener from the fan-out + tear down its subprocess.

    Safe to call multiple times; idempotent on the state set + dict.
    """
    state._audio_out_clients.discard(listener)
    state._audio_out_prefs.pop(listener, None)
    listener.shutdown()
