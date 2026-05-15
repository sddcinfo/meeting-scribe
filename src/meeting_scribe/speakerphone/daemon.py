"""Long-running speakerphone listener daemon.

Architecture (one asyncio loop):

* **udev monitor task** watches the ``input`` and ``hidraw`` subsystems
  for add/remove events. On each ``add`` whose VID:PID is in the
  catalog, runs the child-node settle loop and starts a per-device
  task. On ``remove`` cancels the matching device task.
* **per-device tasks** (one per connected speakerphone):
  * an evdev reader that walks key events through
    :class:`ButtonStateMachine` → action dispatch via
    :class:`UdsMeetingClient`.
  * an LED-state-machine driver that polls
    ``/api/internal/speakerphone/state`` every second and writes the
    resolved pattern to ``/dev/hidraw*``.

The daemon does **not** open ``/dev/input/event3`` (consumer page). The
Vol+/Vol−/Mute buttons keep flowing through the kernel + media-key
agent unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from meeting_scribe.speakerphone import (
    actions as sp_actions,
)
from meeting_scribe.speakerphone import (
    catalog as sp_catalog,
)
from meeting_scribe.speakerphone import (
    hid_leds,
)
from meeting_scribe.speakerphone import (
    led_state_machine as sp_sm,
)
from meeting_scribe.speakerphone import (
    mapping as sp_mapping,
)
from meeting_scribe.speakerphone.evdev_listener import (
    PressEvent,
    TelephonyEvdevReader,
)
from meeting_scribe.speakerphone.meeting_client import UdsMeetingClient

logger = logging.getLogger(__name__)

# Map from Linux input-event-codes (``KEY_*``) to abstract button names.
# Resolved at runtime via the ``evdev`` package so we don't hardcode
# numeric codes. The kernel maps HID Telephony usages to these:
#   Hook Switch (0x20) → KEY_PHONE (169)
#   Phone Mute   (0x2F) → KEY_MICMUTE (248)
# The Teams button on most Dell speakerphones rides the same usage as
# the Hook Switch or comes through a vendor-page channel; we treat the
# Hook Switch as "phone" by default and let the calibration command
# discover the actual Teams emission.
_KERNEL_KEY_TO_BUTTON_BY_NAME: dict[str, str] = {
    "KEY_PHONE": "phone",
    "KEY_MICMUTE": "phone_mute",
    # KEY_TEAMS doesn't exist; the Teams button shows up via the
    # vendor channel and is wired up in `capture-descriptor` once the
    # actual emission is identified.
}


def _resolve_keycode_map() -> dict[int, str]:
    """Translate the by-name map into numeric keycodes.

    Returns an empty dict if evdev isn't importable so the daemon can
    still load on a CI box without the kernel headers.
    """
    try:
        import evdev.ecodes as ec
    except Exception:
        logger.warning("evdev not available; daemon will not bind input keys")
        return {}

    out: dict[int, str] = {}
    for name, button in _KERNEL_KEY_TO_BUTTON_BY_NAME.items():
        code = getattr(ec, name, None)
        if code is not None:
            out[int(code)] = button
    return out


# ── Per-device task ─────────────────────────────────────────────────────


class DeviceSession:
    """Runs the evdev reader + LED loop for one connected speakerphone."""

    def __init__(
        self,
        *,
        device_key: str,
        evdev_path: Path,
        hidraw_path: Path,
        client: UdsMeetingClient,
        mapping_doc: dict[str, Any],
    ) -> None:
        self._device_key = device_key
        self._evdev_path = evdev_path
        self._hidraw_path = hidraw_path
        self._client = client
        self._mapping_doc = mapping_doc
        self._stop = asyncio.Event()

    def update_mapping(self, doc: dict[str, Any]) -> None:
        self._mapping_doc = doc

    async def run(self) -> None:
        try:
            import evdev
        except Exception:
            logger.exception("evdev unavailable; cannot bind %s", self._evdev_path)
            return

        # Opening evdev nodes immediately after the wideband HID
        # detach/reattach races with the udev rule that grants
        # plugdev rw on /dev/input/event* — the node exists ~6 ms
        # after rebind but the udev pass that sets group=plugdev can
        # take a few hundred ms more. Retry on EACCES so the gated
        # session start doesn't lose this fight every restart.
        evdev_device = await self._open_evdev_with_retry(
            evdev,
            str(self._evdev_path),
            label="telephony",
        )
        if evdev_device is None:
            return

        # Also try to open the consumer-page event node for the same
        # HID device (Vol+/Vol-/Mute keys). Non-grab — observation only.
        consumer_path = _find_consumer_event_path(self._device_key)
        consumer_device: Any = None
        if consumer_path is not None:
            consumer_device = await self._open_evdev_with_retry(
                evdev,
                str(consumer_path),
                label="consumer",
            )

        logger.info(
            "speakerphone: bound device %s @ telephony=%s consumer=%s hidraw=%s",
            self._device_key,
            self._evdev_path,
            consumer_path,
            self._hidraw_path,
        )

        reader_task = asyncio.create_task(
            self._run_evdev(evdev_device),
            name=f"sp-evdev-{self._device_key}",
        )
        led_task = asyncio.create_task(
            self._run_leds(),
            name=f"sp-led-{self._device_key}",
        )
        consumer_task: asyncio.Task | None = None
        if consumer_device is not None:
            consumer_task = asyncio.create_task(
                self._run_consumer_observer(consumer_device),
                name=f"sp-consumer-{self._device_key}",
            )

        all_tasks = [reader_task, led_task]
        if consumer_task is not None:
            all_tasks.append(consumer_task)
        try:
            await self._stop.wait()
        finally:
            for t in all_tasks:
                t.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)
            try:
                evdev_device.close()
            except Exception:
                pass  # node may already be gone after the wideband HID reattach; teardown is best-effort
            if consumer_device is not None:
                try:
                    consumer_device.close()
                except Exception:
                    pass  # consumer-page node may already be gone; observer task already cancelled

    async def _open_evdev_with_retry(
        self,
        evdev_mod: Any,
        path: str,
        *,
        label: str,
    ) -> Any:
        """Open an evdev node, retrying on ``EACCES``.

        Right after the SP325 wideband apply, the kernel destroys the
        HID-class input nodes (event3 / event7) and recreates them
        seconds later. Our udev rule grants the plugdev group rw on
        the recreated nodes, but the udev rule pass typically lags
        the inode creation by several hundred ms. Without this retry
        the daemon hits a one-shot ``PermissionError`` and never
        observes any button presses for the session lifetime.
        Discovered 2026-05-13.
        """
        # Schedule: 50ms, 150ms, 400ms, 1.0s, 2.5s — total ~4 s.
        # Generous enough for slow udev passes; if it's still failing
        # after 4 s there's a real permissions problem to surface.
        retry_delays = (0.05, 0.15, 0.4, 1.0, 2.5)
        for attempt, delay in enumerate(retry_delays, start=1):
            try:
                dev = evdev_mod.InputDevice(path)
                if attempt > 1:
                    logger.info(
                        "evdev %s open(%s) succeeded on retry %d",
                        label,
                        path,
                        attempt,
                    )
                return dev
            except PermissionError as e:
                logger.debug(
                    "evdev %s open(%s) attempt %d EACCES: %s",
                    label,
                    path,
                    attempt,
                    e,
                )
            except Exception:
                logger.exception(
                    "evdev %s open(%s) failed (non-retryable)",
                    label,
                    path,
                )
                return None
            await asyncio.sleep(delay)
        logger.error(
            "evdev %s open(%s) FAILED after %d attempts — check udev rule + plugdev membership",
            label,
            path,
            len(retry_delays),
        )
        return None

    def stop(self) -> None:
        self._stop.set()

    # ── evdev → action dispatch ────────────────────────────────────────

    async def _emit_press(self, event: PressEvent) -> None:
        device_block = self._mapping_doc.get("devices", {}).get(self._device_key, {})
        buttons = device_block.get("buttons", {})
        binding = buttons.get(event.button, {})
        action_name = binding.get(event.kind)
        if action_name is None:
            return
        handler = sp_actions.ACTIONS.get(action_name)
        if handler is None:
            logger.warning(
                "unknown action %r for %s/%s",
                action_name,
                event.button,
                event.kind,
            )
            return
        ctx = sp_actions.ActionContext(
            device_key=self._device_key,
            button=event.button,
            press_kind=event.kind,
        )
        try:
            await handler(self._client, ctx)
        except Exception:
            logger.exception(
                "action %s failed (device=%s button=%s)",
                action_name,
                self._device_key,
                event.button,
            )
        # Even on failure, report the press so the GUI can show it.
        try:
            await self._client.report_press(
                device_key=self._device_key,
                button=event.button,
                press_kind=event.kind,
            )
        except Exception:
            logger.exception("press-report failed (best-effort)")

        # Spoken feedback. The label_id reflects the post-action state
        # (e.g. "Mic muted" vs "Mic on") by re-reading the relevant
        # state from the server. Daemon ALWAYS fires this regardless
        # of the local cached ``button_feedback.enabled`` value — the
        # gate lives exclusively on the server (which re-reads
        # mapping.load() on every speak call), giving the zero-lag
        # enable/disable behaviour. Best-effort: never propagate.
        feedback_label = await self._resolve_telephony_feedback_label(event)
        if feedback_label is not None:
            await self._emit_feedback(feedback_label)

    async def _resolve_telephony_feedback_label(
        self,
        event: PressEvent,
    ) -> str | None:
        """Pick the feedback label_id for a telephony button press.

        Reads fresh state from the server so the announcement reflects
        the action's post-dispatch result. Returns ``None`` for buttons
        without a feedback mapping.
        """
        try:
            current = await self._client.get_state()
        except Exception:
            logger.exception("speakerphone feedback: state lookup failed")
            return None

        interp = current.get("interpretation", {}) or {}
        meeting = current.get("meeting", {}) or {}

        if event.button == "phone" and event.kind == "short":
            # tts_cycle just ran; announce the new direction
            new_lang = str(interp.get("room_tts_language", "all"))
            return f"tts_dir_{new_lang}"
        if event.button == "phone" and event.kind == "long":
            return "interp_on" if interp.get("enabled") else "interp_off"
        if event.button == "teams" and event.kind == "short":
            return "meeting_started" if meeting.get("recording") else "meeting_stopped"
        if event.button == "phone_mute" and event.kind == "short":
            return "mic_muted" if interp.get("mic_muted") else "mic_unmuted"
        return None

    async def _emit_feedback(self, label_id: str) -> None:
        """Fire-and-forget feedback request to the server.

        The daemon does NOT read ``button_feedback.enabled`` here —
        the server gate is the only enforcement point, so language /
        enable changes apply on the very next press with zero
        daemon-poll lag. Best-effort: log + swallow exceptions so a
        flaky TTS backend never crashes a button-press path.
        """
        try:
            await self._client.speak(label_id=label_id)
        except Exception:
            logger.exception(
                "speakerphone feedback: speak(%s) failed (best-effort)",
                label_id,
            )

    # ── Consumer-page observer (Vol+/Vol-/Mute) ─────────────────────

    async def _on_consumer_key(self, label: str) -> None:
        """Callback from ``ConsumerObserver`` for Vol+/Vol-/Mute.

        Consumer-page keys get the kernel media-key behaviour ONLY —
        no TTS announcement. The audio response IS the feedback:
        pressing Vol+ makes the room louder, pressing Mute makes it
        silent. Announcing "Volume up" out loud every press is more
        noise than signal (2026-05-13 — user feedback after listening
        to a queue full of "Volume up" repeats).

        TTS announcements are reserved for state changes the user
        can't sense from the audio itself — interpretation on/off,
        TTS direction, meeting recording start/stop, mic mute on/off.
        Those live in ``_emit_press`` for the telephony event path.

        The GB10 is headless and runs no desktop media-key agent, so
        without ``_apply_consumer_action`` pressing Vol+ on the SP325
        lands valid KEY_VOLUMEUP events on the kernel that nobody
        acts on — the daemon IS the media-key agent.
        """
        await self._apply_consumer_action(label)

    async def _apply_consumer_action(self, label: str) -> None:
        """Apply the volume/mute change via ``wpctl``.

        The daemon is the media-key agent on headless deployments
        (GB10). ``wpctl set-volume @DEFAULT_AUDIO_SINK@ 5%+ -l 1.0``
        increments by 5% with a 1.0 hard cap so successive Vol+
        presses can't overflow. ``set-mute toggle`` is the standard
        wpctl idiom for flipping the bit; we read the resulting state
        in ``_resolve_system_mute_state`` to pick the right
        announcement.

        Best-effort: a missing/slow wpctl never breaks the press
        path; we just log + swallow.
        """
        if label == "volume_up":
            args = ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%+", "-l", "1.0"]
        elif label == "volume_down":
            args = ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"]
        elif label == "system_mute_toggled":
            args = ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"]
        else:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=1.0)
            except TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass  # wpctl already exited between the timeout and the kill — log and move on
                logger.warning("consumer media-key: wpctl %s timed out", label)
                return
            if proc.returncode != 0:
                logger.warning(
                    "consumer media-key: wpctl %s rc=%d stderr=%r",
                    label,
                    proc.returncode,
                    (stderr or b"").decode("utf-8", "replace"),
                )
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                "consumer media-key: wpctl unavailable for %s: %s",
                label,
                e,
            )

    async def _run_consumer_observer(self, evdev_device: Any) -> None:
        from meeting_scribe.speakerphone.evdev_listener import (
            ConsumerObserver,
            resolve_consumer_key_to_label_map,
        )

        key_to_label = resolve_consumer_key_to_label_map()
        if not key_to_label:
            logger.info("consumer observer: no evdev keys resolved; skipping")
            return
        observer = ConsumerObserver(
            device=evdev_device,
            key_to_label=key_to_label,
            emit=self._on_consumer_key,
        )
        try:
            await observer.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "consumer observer crashed for %s",
                self._device_key,
            )

    async def _run_evdev(self, evdev_device: Any) -> None:
        keycode_map = _resolve_keycode_map()
        long_press_ms = int(self._mapping_doc.get("long_press_ms", 1000))
        reader = TelephonyEvdevReader(
            device=evdev_device,
            button_for_keycode=keycode_map,
            long_press_ms=long_press_ms,
            emit=self._emit_press,
        )
        try:
            await reader.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("evdev reader for %s crashed", self._device_key)

    # ── LED state machine driver ───────────────────────────────────────

    async def _run_leds(self) -> None:
        """Drive the Mute LED ring using the LED state machine.

        Polls the internal state endpoint at 1 Hz for fresh signals,
        resolves the pattern, and plays it through a
        :class:`hid_leds.PatternRunner`. Re-resolves every poll so a
        mode change (e.g. interpretation toggled off → backend_unready
        because the server briefly didn't respond) takes effect within
        ~1 s.
        """
        write = self._make_hidraw_writer()
        current_pattern: tuple[tuple[int, int], ...] | None = None
        runner: hid_leds.PatternRunner | None = None
        while not self._stop.is_set():
            try:
                signals = await self._collect_signals()
                cfg = self._mapping_doc.get("leds", {}).get("states", {})
                res = sp_sm.resolve(signals, cfg)
                schedule = hid_leds.resolve_pattern(res.pattern)
                if schedule != current_pattern:
                    if runner is not None:
                        runner.write_off()
                    runner = hid_leds.PatternRunner(schedule, write)
                    current_pattern = schedule
                delay = runner.tick()
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("LED loop tick failed; retrying in 1s")
                await asyncio.sleep(1.0)
        if runner is not None:
            runner.write_off()

    async def _collect_signals(self) -> sp_sm.SystemSignals:
        try:
            state = await self._client.get_state()
        except Exception:
            return sp_sm.SystemSignals(error=False, backend_unready=True)
        interp = state.get("interpretation", {})
        meeting = state.get("meeting", {})
        return sp_sm.SystemSignals(
            backend_unready=False,
            mic_muted=bool(interp.get("mic_muted", False)),
            recording=bool(meeting.get("recording", False)),
        )

    def _make_hidraw_writer(self):
        """Return a sync callable that writes the Mute LED on/off byte.

        On error (e.g. device disconnected), logs and degrades to a
        no-op so the LED task doesn't crash; the udev monitor will
        tear the session down via :meth:`stop` shortly.
        """
        fd: int | None = None
        path = self._hidraw_path

        def _open() -> int | None:
            nonlocal fd
            if fd is not None:
                return fd
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_NONBLOCK)
            except OSError as e:
                logger.warning("hidraw open %s failed: %r", path, e)
                fd = None
            return fd

        def write(on: bool) -> None:
            f = _open()
            if f is None:
                return
            try:
                os.write(f, hid_leds.mute_ring(on))
            except OSError as e:
                logger.warning("hidraw write %s failed: %r", path, e)
                # Reopen on next write.
                nonlocal_close(fd)

        def nonlocal_close(_fd: int | None) -> None:
            nonlocal fd
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass  # hidraw fd already invalid (device unplugged) — next write will reopen
                fd = None

        return write


# ── Device discovery (udev + child-node settle) ─────────────────────────


async def _wait_for_child_nodes(
    target_vid: int,
    target_pid: int,
    *,
    timeout_s: float = 5.0,
    poll_ms: int = 200,
) -> tuple[Path, Path] | None:
    """Look for the matching event* + hidraw* nodes for this device.

    Retries up to ``timeout_s`` seconds because the kernel can emit the
    USB-level add event before the event/hidraw children are populated.
    Returns ``(evdev_path, hidraw_path)`` on success, ``None`` on
    timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        evdev_path, hidraw_path = _scan_for_device(target_vid, target_pid)
        if evdev_path is not None and hidraw_path is not None:
            return evdev_path, hidraw_path
        await asyncio.sleep(poll_ms / 1000.0)
    return None


def _find_consumer_event_path(device_key: str) -> Path | None:
    """Locate the consumer-page event* node for ``device_key`` (vid:pid).

    Mirrors ``_scan_for_device`` but picks the input that has
    ``EV_KEY`` set and ``EV_LED`` *not* set (the consumer-control
    child — Vol+/Vol-/Mute keys live here). Returns ``None`` if no
    matching node can be found.
    """
    try:
        vid_s, pid_s = device_key.split(":")
        vid = int(vid_s, 16)
        pid = int(pid_s, 16)
    except ValueError, AttributeError:
        return None

    hid_root = Path("/sys/bus/hid/devices")
    if not hid_root.is_dir():
        return None
    for entry in hid_root.iterdir():
        try:
            uevent = (entry / "uevent").read_text()
        except OSError:
            continue
        line = next(
            (ln for ln in uevent.splitlines() if ln.startswith("HID_ID=")),
            "",
        )
        try:
            _bus, hid_vid_s, hid_pid_s = line.split("=", 1)[1].split(":")
            hid_vid = int(hid_vid_s, 16)
            hid_pid = int(hid_pid_s, 16)
        except ValueError, IndexError:
            continue
        if (hid_vid, hid_pid) != (vid, pid):
            continue
        input_dir = entry / "input"
        if not input_dir.is_dir():
            continue
        for inp in sorted(input_dir.iterdir()):
            if not inp.name.startswith("input"):
                continue
            try:
                ev_hex = (inp / "capabilities" / "ev").read_text().strip()
                ev_mask = int(ev_hex, 16)
            except OSError, ValueError:
                continue
            # Want EV_KEY (bit 1 = 0x02), do NOT want EV_LED (bit 17 = 0x20000).
            if not (ev_mask & 0x02):
                continue
            if ev_mask & 0x20000:
                continue
            for ev in inp.iterdir():
                if ev.name.startswith("event"):
                    return Path("/dev/input") / ev.name
    return None


def _guess_pipewire_source_name(device_key: str) -> str | None:
    """Map a catalog key (``vid:pid``) to the canonical PipeWire source name.

    PipeWire's ALSA naming convention for pro-audio sources is
    ``alsa_input.usb-<vendor>_<product>_<serial>-00.pro-input-0``.
    The serial comes from the device's iSerial descriptor — for the
    SP325 it's a fixed 16-zero string; future devices in the catalog
    may need a smarter lookup.

    Returns ``None`` when we don't have a known mapping; the compliance
    probe skips with a logged note in that case.
    """
    # Canonical names for the SP325/SP3022 product family.
    # Confirmed against live ``pw-cli info`` 2026-05-13.
    if device_key == "413c:8223":
        return "alsa_input.usb-Dell_Inc._Dell_SP325_Speakerphone_0000000000000000-00.pro-input-0"
    if device_key == "413c:8222":
        return "alsa_input.usb-Dell_Inc._Dell_SP3022_Speakerphone_0000000000000000-00.pro-input-0"
    return None


def _scan_for_device(vid: int, pid: int) -> tuple[Path | None, Path | None]:
    """Find the telephony event* + hidraw* nodes for a given VID:PID.

    Walks /sys to identify the right children. Telephony input is
    detected via the presence of an ``leds`` subdir (matches the SP325
    canonical layout where the telephony device has its own LED
    output report).
    """
    evdev_path: Path | None = None
    hidraw_path: Path | None = None
    # Find HID instances under /sys/bus/hid/devices/
    hid_root = Path("/sys/bus/hid/devices")
    if not hid_root.is_dir():
        return None, None
    for entry in hid_root.iterdir():
        try:
            uevent = (entry / "uevent").read_text()
        except OSError:
            continue
        if "HID_ID=" not in uevent:
            continue
        # HID_ID is "<bus>:<vid>:<pid>"
        line = next(
            (ln for ln in uevent.splitlines() if ln.startswith("HID_ID=")),
            "",
        )
        try:
            _bus, hid_vid_s, hid_pid_s = line.split("=", 1)[1].split(":")
            hid_vid = int(hid_vid_s, 16)
            hid_pid = int(hid_pid_s, 16)
        except ValueError, IndexError:
            continue
        if (hid_vid, hid_pid) != (vid, pid):
            continue
        # Look for a hidraw subdir.
        hidraw_dir = entry / "hidraw"
        if hidraw_dir.is_dir():
            for raw in hidraw_dir.iterdir():
                hidraw_path = Path("/dev") / raw.name
                break
        # Find the input/event* node with LED capabilities (the
        # telephony page child). The telephony input device has
        # ``EV_LED`` (bit 17 = 0x20000) set in its ``capabilities/ev``
        # bitmap because the same HID Report ID 5 carries both button
        # inputs and LED outputs — that's how we tell it apart from
        # the consumer-page input (volume keys) and the vendor-page
        # inputs. There is NO sysfs "leds" subdirectory; the LED
        # capability is exposed only via the event bitmap.
        input_dir = entry / "input"
        if input_dir.is_dir():
            for inp in sorted(input_dir.iterdir()):
                if not inp.name.startswith("input"):
                    continue
                try:
                    ev_hex = (inp / "capabilities" / "ev").read_text().strip()
                    ev_mask = int(ev_hex, 16)
                except OSError, ValueError:
                    continue
                # EV_LED is bit 17 (0x20000). Only the telephony
                # input has it set.
                if not (ev_mask & 0x20000):
                    continue
                for ev in inp.iterdir():
                    if ev.name.startswith("event"):
                        evdev_path = Path("/dev/input") / ev.name
                        break
                if evdev_path is not None:
                    break
        if evdev_path and hidraw_path:
            return evdev_path, hidraw_path
    return evdev_path, hidraw_path


# ── Daemon orchestrator ─────────────────────────────────────────────────


class SpeakerphoneDaemon:
    """Top-level run loop. Owns the udev monitor + per-device sessions."""

    def __init__(self) -> None:
        # Value is ``(session, task)`` where ``task`` is ``None`` while
        # the session is queued behind the wideband-apply gate (see
        # ``_gated_session_start``). ``_detach`` tolerates the None.
        self._sessions: dict[str, tuple[DeviceSession, asyncio.Task | None]] = {}
        self._client: UdsMeetingClient | None = None
        self._mapping_poll_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        sp_actions.assert_registry_complete()

    async def run(self) -> None:
        self._client = UdsMeetingClient()
        # Kick off the mapping-poll task so live edits in the GUI flow
        # into device sessions within ~5 s.
        self._mapping_poll_task = asyncio.create_task(
            self._poll_mapping(),
            name="sp-mapping-poll",
        )

        # Initial scan: pick up devices already attached at start.
        await self._scan_and_attach()

        try:
            await self._udev_monitor_loop()
        finally:
            self._stop.set()
            for key in list(self._sessions):
                await self._detach(key)
            if self._mapping_poll_task is not None:
                self._mapping_poll_task.cancel()
            if self._client is not None:
                await self._client.aclose()

    async def _udev_monitor_loop(self) -> None:
        """Watch the input + hidraw subsystems via pyudev.

        Falls back to a polling loop (every 5 s rescans /sys) if
        pyudev isn't available, so the daemon still works on a
        minimal system.
        """
        try:
            import pyudev
        except Exception:
            logger.warning("pyudev unavailable; falling back to 5 s polling")
            while not self._stop.is_set():
                await asyncio.sleep(5)
                await self._scan_and_attach()
            return

        ctx = pyudev.Context()
        monitor = pyudev.Monitor.from_netlink(ctx)
        monitor.filter_by("input")
        monitor.filter_by("hidraw")
        loop = asyncio.get_event_loop()

        def _enqueue(action: str, device: Any) -> None:
            # We don't care about properties here; trigger a rescan,
            # debounce naturally via the settle loop.
            asyncio.run_coroutine_threadsafe(self._scan_and_attach(), loop)

        observer = pyudev.MonitorObserver(monitor, _enqueue)
        observer.start()
        try:
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
        finally:
            observer.stop()

    async def _scan_and_attach(self) -> None:
        for key, profile in sp_catalog.CATALOG.items():
            if key in self._sessions:
                continue
            found = await _wait_for_child_nodes(profile.vid, profile.pid, timeout_s=0.4)
            if found is None:
                continue
            evdev_path, hidraw_path = found
            await self._attach(key, evdev_path, hidraw_path)

    async def _attach(self, key: str, evdev_path: Path, hidraw_path: Path) -> None:
        doc = sp_mapping.load()
        session = DeviceSession(
            device_key=key,
            evdev_path=evdev_path,
            hidraw_path=hidraw_path,
            client=self._client,
            mapping_doc=doc,
        )
        # Register the session WITHOUT a run-task yet. ``_scan_and_attach``
        # dedupes on ``key in self._sessions``, so this placeholder keeps
        # concurrent udev events (which fire during the HID detach/
        # reattach cycle inside apply_wideband) from double-attaching.
        self._sessions[key] = (session, None)

        # The startup ordering matters: ``apply_wideband_good`` detaches
        # the HID kernel driver while it sends SET_REPORT bytes, which
        # destroys /dev/input/event3 + event7 from under any evdev
        # readers that opened them. Pre-2026-05-13 the readers crashed
        # with ``Errno 19 No such device`` 37 ms after startup and were
        # never reattached for the rest of the session — buttons silently
        # dead. Run wideband FIRST, then open evdev once the nodes are
        # stable. Compliance probe stays in the background.
        asyncio.create_task(
            self._gated_session_start(key, session),
            name=f"sp-gate-{key}",
        )
        logger.info("speakerphone attached: %s", key)

    async def _gated_session_start(
        self,
        key: str,
        session: DeviceSession,
    ) -> None:
        """Run wideband apply, THEN open the evdev session.

        Prevents the Errno-19 race where HID detach inside the
        wideband apply destroys the evdev nodes under live readers.
        Compliance probe runs after the session opens, in the
        background, so it observes the post-apply DSP state.
        """
        if key in ("413c:8223", "413c:8205", "413c:8222"):
            await self._run_wideband_apply(key)

        # Evdev nodes are stable now — start the reader/observer/LED tasks.
        run_task = asyncio.create_task(
            session.run(),
            name=f"sp-session-run-{key}",
        )
        self._sessions[key] = (session, run_task)

        # Compliance probe in the background (regardless of wideband
        # success — operators still want to see the spectrum).
        asyncio.create_task(
            self._probe_compliance(key),
            name=f"sp-compliance-{key}",
        )

    async def _run_wideband_apply(self, key: str) -> None:
        """Apply SP325 wideband config off-loop, blocking until it's
        truly settled.

        Extracted from the old ``_apply_wideband_then_probe`` so that
        ``_gated_session_start`` can sequence "wideband done" before
        opening evdev, with compliance probing still backgrounded.
        """
        from meeting_scribe.speakerphone.sp325_hid import (
            Sp325Error,
            Sp325HidClient,
        )

        loop = asyncio.get_running_loop()

        def _apply() -> dict | None:
            try:
                with Sp325HidClient.open_default() as cli:
                    return cli.apply_wideband_good(settle_seconds=15.0)
            except Sp325Error as e:
                logger.warning("speakerphone: SP325 wideband apply failed: %s", e)
                return None

        result = await loop.run_in_executor(None, _apply)
        if result is not None:
            logger.info("speakerphone: SP325 wideband applied: %s", result)

    async def _apply_wideband_then_probe(self, key: str) -> None:
        """Run apply_wideband_good off the loop, then probe compliance.

        Sequenced so the compliance check sees the post-apply DSP
        state. ``apply_wideband_good`` includes a 15 s settle so
        compliance results reflect the real firmware mode, not the
        stale buffer.
        """
        from meeting_scribe.speakerphone.sp325_hid import (
            Sp325Error,
            Sp325HidClient,
        )

        loop = asyncio.get_running_loop()

        def _apply() -> dict | None:
            try:
                with Sp325HidClient.open_default() as cli:
                    return cli.apply_wideband_good(settle_seconds=15.0)
            except Sp325Error as e:
                logger.warning("speakerphone: SP325 wideband apply failed: %s", e)
                return None

        result = await loop.run_in_executor(None, _apply)
        if result is not None:
            logger.info("speakerphone: SP325 wideband applied: %s", result)

        # Always run the compliance probe — even when apply fails, the
        # operator should see the spectral signature so they know
        # whether to intervene.
        await self._probe_compliance(key)

    async def _probe_compliance(self, key: str) -> None:
        from meeting_scribe.speakerphone import compliance

        # Source-node name follows the standard PipeWire ALSA naming
        # convention for pro-audio. The daemon doesn't know the exact
        # node name from the catalog alone, so we derive it from the
        # ALSA card+device pulled out of /sys. For now, try the canonical
        # SP325 node name; if other devices show up in the catalog we'll
        # generalize.
        node_name = _guess_pipewire_source_name(key)
        if node_name is None:
            logger.info(
                "compliance: skipping %s — no resolved PipeWire source name",
                key,
            )
            return

        min_hbp, min_rolloff = compliance.expected_thresholds(key)
        # Run the blocking spectrum analysis off the event loop so the
        # daemon's hotplug/LED tasks keep ticking.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compliance.probe_device(
                node_name,
                min_high_band_pct=min_hbp,
                min_rolloff_pct=min_rolloff,
            ),
        )
        log_method = {
            "pass": logger.info,
            "warn": logger.warning,
            "fail": logger.error,
        }.get(result.status, logger.info)
        log_method(
            "speakerphone compliance %s: device=%s high_band_pct=%s%% "
            "rolloff_3400=%s%% rms=%s reason=%s",
            result.status,
            key,
            result.high_band_pct,
            result.rolloff_3400hz_pct,
            result.rms,
            result.reason,
        )

    async def _detach(self, key: str) -> None:
        entry = self._sessions.pop(key, None)
        if entry is None:
            return
        session, task = entry
        session.stop()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5)
            except TimeoutError:
                task.cancel()
        logger.info("speakerphone detached: %s", key)

    async def _poll_mapping(self) -> None:
        last_etag: str | None = None
        while not self._stop.is_set():
            try:
                doc = sp_mapping.load()
                etag = sp_mapping.compute_etag(doc)
                if etag != last_etag:
                    last_etag = etag
                    for session, _task in self._sessions.values():
                        session.update_mapping(doc)
            except Exception:
                logger.exception("mapping poll failed (will retry)")
            await asyncio.sleep(5.0)
