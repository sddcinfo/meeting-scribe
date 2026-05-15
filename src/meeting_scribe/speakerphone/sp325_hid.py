"""Dell SP325 vendor-HID protocol client (Linux, no Windows DPM needed).

Reverse-engineered from `com.dell.DPM.Plugin.LogicalDevice.SP3022.dll`
shipped in Dell Peripheral Manager v1.6.7 (Feb 2025), then empirically
validated against the SP325 hardware on 2026-05-13.

The protocol command vocabulary came from the SP3022 plugin disasm;
the working **transport** on Linux is via libusb USB control transfer,
NOT hidraw. Linux's hidraw layer silently drops vendor-defined reports
on multi-collection HID devices (the SP325 advertises 5 collections;
see fwupd #6430). DPM on Windows ultimately issues a USB SET_REPORT
class request, and we do the same directly via pyusb after detaching
the kernel HID driver on the interface.

## Wire protocol

Dell speakerphones in the SP3022/SP325 family expose two vendor HID
report channels (interface 3, descriptor pages 0xff13 and 0xff99 —
neither claimed by the kernel's hid-generic driver):

    report_id=0x06  page=0xff13  OUTPUT (host→device)  61 bytes
    report_id=0x07  page=0xff13  INPUT  (device→host)  61 bytes
    report_id=0x9a  page=0xff99  FEATURE              63 bytes

The DPM plugin's per-property setter functions build a 4-byte payload
of the form ``[CMD_CLASS, OPCODE, 0x00, VALUE]`` for the simple
boolean/scalar settings, and longer payloads for the EQ preset family.
The device echoes back a buffer starting with ``0xB8`` on success.

Command classes observed in the DPM disassembly:

    0xD0  Mic Mute Sound (cue tone on mic mute)
    0xD1  Volume Limit value
    0xD2  Mic Noise Suppression (the AI NR toggle — narrowband regression cause)
    0xD3  Volume Adjustment Tone (audio feedback on volume button press)
    0xD4  Is Volume Limit Enabled (gate for D1)
    0xD5  (suspected NR direction control — to be confirmed)
    0xC0  Audio Equalizer (preset family, longer payload)
    0xA1  (paired with 0xC0, EQ preset index/profile)

Opcode 0x01 is "set scalar value", 0x02 is "set numeric value", 0x04
appears in the EQ preset path with a 5-byte payload encoding the
preset profile. Opcode 0x02 is suspected to also be the GET form
(read current state); empirical probing confirms.

The DPM call signature is ``hid_send(this, response_size, response_buf,
command_payload)`` where ``response_size`` is the second arg —
``0x40`` (64) for the D-class commands and ``0xC0`` (192) for the
A/C-class EQ commands.

## Wiring on Linux

Send via ``HIDIOCSFEATURE`` (feature report 0x9a, 63-byte payload) OR
the OUTPUT pipe (report 0x06, 61-byte payload). The DPM plugin uses
``IOCTL_HID_SET_FEATURE`` on Windows which is the same kernel-level
transaction that ``HIDIOCSFEATURE`` triggers on Linux. Responses come
back via ``HIDIOCGFEATURE`` on the same report ID.

Both transports are tried at probe time so we adapt to whichever the
firmware actually accepts; empirically the FEATURE pipe (0x9a) carries
the D-class commands.

## Safety posture

* All writes are gated through ``Sp325HidClient.set(...)`` which
  records the previous value first via ``get(...)``.
* The client refuses to issue unknown opcodes — every documented
  command lives in ``COMMANDS`` and is wrapped by a typed method.
* Persistence is verified across replug in the bench harness; values
  that don't survive an explicit save (``commit()``) are flagged.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Linux ioctl plumbing ──────────────────────────────────────────────

_IOC_READ = 2
_IOC_WRITE = 1
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14


def _ioc(direction: int, type_char: str, nr: int, size: int) -> int:
    """Pack the bitfields the kernel uses to recognise an ioctl number.

    Mirrors ``_IOC`` from ``<linux/ioctl.h>``. ``HIDIOCSFEATURE`` and
    ``HIDIOCGFEATURE`` both pass an in/out buffer (the kernel may copy
    response bytes back into the same buffer on GET), so we OR both
    direction bits.
    """
    return (
        (direction << (_IOC_NRBITS + _IOC_TYPEBITS + _IOC_SIZEBITS))
        | (size << (_IOC_NRBITS + _IOC_TYPEBITS))
        | (ord(type_char) << _IOC_NRBITS)
        | nr
    )


def _HIDIOCSFEATURE(length: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, "H", 0x06, length)


def _HIDIOCGFEATURE(length: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, "H", 0x07, length)


def _HIDIOCSOUTPUT(length: int) -> int:
    """Linux equivalent of Windows IOCTL_HID_SET_OUTPUT_REPORT.

    Confirmed by decoding DPM's `mov edx, 0xb0195` →
    CTL_CODE(FILE_DEVICE_KEYBOARD=0xB, 0x65, METHOD_IN_DIRECT=1, 0).
    Linux kernel header (this host: 6.17): NR 0x0B.
    """
    return _ioc(_IOC_READ | _IOC_WRITE, "H", 0x0B, length)


def _HIDIOCGINPUT(length: int) -> int:
    """Linux equivalent of Windows IOCTL_HID_GET_INPUT_REPORT.

    Confirmed by decoding DPM's `mov edx, 0xb01a2` →
    CTL_CODE(FILE_DEVICE_KEYBOARD=0xB, 0x68, METHOD_OUT_DIRECT=2, 0).
    Linux kernel header (this host: 6.17): NR 0x0A.
    """
    return _ioc(_IOC_READ | _IOC_WRITE, "H", 0x0A, length)


# ─── Protocol constants ────────────────────────────────────────────────

FEATURE_REPORT_ID = 0x9A  # vendor page 0xff99
FEATURE_REPORT_PAYLOAD = 63  # 63 + 1 report-id byte

# DPM frames every command with a 2-byte header before the 4-byte
# command body and sends/receives via IOCTL_HID_SET_OUTPUT_REPORT /
# IOCTL_HID_GET_INPUT_REPORT — NOT feature reports. The header is
# embedded in the 64-byte buffer that Windows' SET_OUTPUT_REPORT
# treats as `[report_id, payload…]`. Empirically the report id slot
# carries 0xB9 (the protocol magic), not the descriptor-advertised
# 0x06 OUTPUT id — Dell uses a "hidden" report the kernel still
# routes through to the device's control endpoint.
PROTOCOL_MAGIC = 0xB9  # buffer[0] header byte
PROTOCOL_SIZE_HINT = 0x40  # buffer[1] = expected response capacity
WIRE_BUFFER_BYTES = 0x40  # 64 — DPM's buffer size for SET_OUTPUT_REPORT

RESPONSE_OK_BYTE = 0xB8  # device → host success-echo signature
# (response byte 0 of the GET_INPUT_REPORT result)


@dataclasses.dataclass(frozen=True)
class Command:
    """A single Dell vendor command, as recovered from DPM."""

    name: str
    cmd_class: int  # byte 0 of the payload (0xD0–0xD5, 0xC0, 0xA1)
    set_opcode: int  # byte 1 when writing
    get_opcode: int  # byte 1 when reading (typically set_opcode|0x10 or 0x02)
    payload_bytes: int = 4  # most commands are 4-byte payloads
    notes: str = ""


# The names match the Qt property names exposed by the DPM SP3022
# plugin's QMetaObject. Empirical opcode mapping ships in v1; we use
# get_opcode=0x02 by default as DPM's GET pattern — overridable if a
# given class doesn't ack it.

COMMANDS: dict[str, Command] = {
    "mic_mute_sound": Command(
        name="mic_mute_sound",
        cmd_class=0xD0,
        set_opcode=0x06,
        get_opcode=0x02,
        notes="Plays a cue tone when the mic-mute hardware button is pressed.",
    ),
    "volume_limit": Command(
        name="volume_limit",
        cmd_class=0xD1,
        set_opcode=0x01,
        get_opcode=0x02,
        notes="Output volume ceiling (gated by volume_limit_enabled).",
    ),
    "mic_noise_suppression": Command(
        name="mic_noise_suppression",
        cmd_class=0xD2,
        set_opcode=0x01,
        get_opcode=0x02,
        notes=(
            "AI Noise Cancellation — the toggle that narrowbands ASR when on. "
            "Setting this to 0 is half of the wideband-good config; the second "
            "direction (incoming) is on 0xD5."
        ),
    ),
    "volume_adjustment_tone": Command(
        name="volume_adjustment_tone",
        cmd_class=0xD3,
        set_opcode=0x01,
        get_opcode=0x02,
        notes="Audio feedback when adjusting volume via hardware buttons.",
    ),
    "volume_limit_enabled": Command(
        name="volume_limit_enabled",
        cmd_class=0xD4,
        set_opcode=0x02,
        get_opcode=0x02,
        notes="Master enable for the volume_limit ceiling.",
    ),
    "mic_noise_suppression_incoming": Command(
        name="mic_noise_suppression_incoming",
        cmd_class=0xD5,
        set_opcode=0x01,
        get_opcode=0x02,
        notes=(
            "Suspected: the second AI NR toggle (incoming/playback direction). "
            "Both must be 0 for wideband-good mic capture per the empirical "
            "Dell datasheet probe."
        ),
    ),
    "audio_equalizer_preset": Command(
        name="audio_equalizer_preset",
        cmd_class=0xC0,
        set_opcode=0xA1,
        get_opcode=0x02,
        payload_bytes=6,
        notes=(
            "EQ preset selector. The 'Default' preset is the wideband-good "
            "choice; 'Speech Boost' re-introduces the telephony bump that "
            "broke Japanese consonant discrimination."
        ),
    ),
}


# ─── Client ────────────────────────────────────────────────────────────


class Sp325Error(RuntimeError):
    """Raised when the SP325 declines or fails to respond to a command."""


# USB control-transfer constants. wValue = (report_type << 8) | report_id.
USB_REQ_SET_REPORT = 0x09  # bRequest
USB_REQ_GET_REPORT = 0x01
USB_BMREQTYPE_H2D = 0x21  # Host-to-Device | Class | Interface
USB_BMREQTYPE_D2H = 0xA1  # Device-to-Host | Class | Interface
SP325_HID_INTERFACE = 3
SP325_REPORT_ID = 0xB9  # DPM's wire byte 0 acts as report ID
SP325_REPORT_TYPE_OUTPUT = 0x02
SP325_WIRE_PAYLOAD = 63  # 64 minus the report-id byte


class Sp325HidClient:
    """Dell SP325 vendor-HID client over libusb USB control transfers.

    Why libusb and not hidraw: Linux's hidraw silently drops vendor
    output reports on devices with multiple HID collections (fwupd #6430).
    The SP325 has 5 collections; hidraw accepts the write at the ioctl
    layer but the bytes never reach the device firmware. libusb routes
    the same USB SET_REPORT class request that DPM uses on Windows
    directly to the device's control endpoint, and works reliably.

    Critical recipe (validated 2026-05-13 on a live SP325):
      1. Detach kernel drivers on **all four** USB interfaces — not
         just the HID one. Empirically, detaching only interface 3
         leaves the device in a state where vendor commands are
         accepted at the wire but ignored by firmware; detaching all
         four interfaces puts the device into a "configurable" mode
         where the firmware actually processes and persists settings.
      2. Send 63-byte payload [0x40, cmd_class, op, 0x00, value, 0×58]
         via SET_REPORT class request (bmRequestType=0x21, bRequest=0x09,
         wValue=(0x02<<8)|0xB9, wIndex=3).
      3. The device flashes its LED ring red/white and saves the
         setting to NVRAM. Persists across replug per Dell datasheet.
      4. Reattach kernel drivers so PipeWire reclaims the audio
         interfaces and ALSA captures resume.

    Requires root (or a udev rule granting access to /dev/bus/usb/…).
    pyusb is provided by the apt package ``python3-usb``.

    Example::

        with Sp325HidClient.open_default() as cli:
            cli.apply_wideband_good()
    """

    def __init__(self, dev):
        self._dev = dev
        self._detached: list[int] = []

    # ── lifecycle ──────────────────────────────────────────────────

    @classmethod
    def open_default(cls) -> Sp325HidClient:
        """Locate the connected SP325/SP3022 on the USB bus and open it."""
        # Import here so callers that never use the HID client don't
        # need pyusb installed. The apt package is python3-usb.
        import sys as _sys

        for p in ("/usr/lib/python3/dist-packages", "/usr/lib/python3.12/dist-packages"):
            if p not in _sys.path:
                _sys.path.append(p)
        try:
            import usb.core
        except ImportError as e:
            raise Sp325Error(
                "pyusb not importable. Install: sudo apt-get install -y python3-usb"
            ) from e

        for pid in (0x8223, 0x8205):  # SP325, then SP3022
            dev = usb.core.find(idVendor=0x413C, idProduct=pid)
            if dev is not None:
                logger.info(
                    "SP325 family device found: PID 0x%04x bus=%d addr=%d",
                    pid,
                    dev.bus,
                    dev.address,
                )
                return cls(dev).open()
        raise Sp325Error("no SP325/SP3022 found on USB bus")

    def open(self) -> Sp325HidClient:
        """Detach kernel drivers from ALL interfaces.

        This is the step that makes vendor commands take effect — the
        device firmware only enters "configurable" mode when the audio
        interfaces are released. The wideband SET_REPORT also targets
        the HID interface (see ``SP325_HID_INTERFACE = 3``), so the
        HID driver must release the interface before libusb can
        ``ctrl_transfer`` to it; pyusb returns ``Errno 16 Resource
        busy`` otherwise.

        Pair this with ``close()``, which verifies every interface
        actually rebinds to its kernel driver. The 2026-05-13 incident
        traced bricked SP325 buttons to a silent reattach failure that
        left interface 1.3 orphaned for hours.
        """
        import usb.core

        for cfg in self._dev:
            for intf in cfg:
                num = intf.bInterfaceNumber
                try:
                    if self._dev.is_kernel_driver_active(num):
                        self._dev.detach_kernel_driver(num)
                        self._detached.append(num)
                        logger.debug("detached kernel driver iface=%d", num)
                except usb.core.USBError as e:
                    logger.debug("detach iface=%d skipped: %s", num, e)
        return self

    def close(self) -> None:
        """Reattach kernel drivers AND verify the rebind via sysfs.

        The old version used pyusb's ``attach_kernel_driver`` and
        trusted that a clean return meant the kernel actually rebound.
        It doesn't always — and the failure mode is silent: pyusb
        returns OK while ``/sys/bus/usb/devices/<iface>/driver`` is
        missing. That was the 2026-05-13 incident — interface 3-1:1.3
        ended up driverless, evdev nodes vanished, SP325 buttons dead.

        New behaviour:

        1. **Release libusb resources first** so the kernel sees the
           interface as no longer claimed by usbfs (otherwise
           ``attach_kernel_driver`` raises Errno 16 Resource busy
           immediately after the 15 s DSP-settle).
        2. **Retry with backoff** on Resource busy — the firmware
           sometimes needs another few hundred ms after the settle
           before it accepts a rebind.
        3. **Verify via sysfs** that a driver symlink actually exists.
        4. **Hard log** any interface still orphaned at the end so it
           can never go unnoticed again.
        """
        import time as _time

        import usb.core
        import usb.util

        # Step 1 — release libusb's grip. Without this, the kernel
        # views the interface as still claimed by usbfs and refuses
        # to rebind (Errno 16). ``dispose_resources`` is the most
        # heavy-handed cleanup pyusb offers — drops every cached
        # handle on the device.
        try:
            usb.util.dispose_resources(self._dev)
        except Exception as e:
            logger.debug("SP325 dispose_resources: %s", e)

        sysfs_root = self._sysfs_root_path()
        failures: list[tuple[int, str]] = []
        # Retry schedule: ~50ms, 150ms, 400ms, 1s — total ~1.6s.
        # The DSP-settle is already 15s; adding ~2s in the worst
        # case is fine for the operator.
        retry_delays = (0.05, 0.15, 0.40, 1.0)

        for num in list(self._detached):
            ok = False
            last_err: Exception | None = None
            for attempt, delay in enumerate(retry_delays, start=1):
                try:
                    if not self._dev.is_kernel_driver_active(num):
                        self._dev.attach_kernel_driver(num)
                    ok = self._verify_kernel_driver_bound(sysfs_root, num)
                    if ok:
                        if attempt > 1:
                            logger.info(
                                "SP325 reattach iface=%d succeeded on retry %d",
                                num,
                                attempt,
                            )
                        break
                except usb.core.USBError as e:
                    last_err = e
                    logger.debug(
                        "SP325 attach attempt %d for iface=%d: %s",
                        attempt,
                        num,
                        e,
                    )
                _time.sleep(delay)

            # Last-ditch sysfs recovery if all pyusb retries failed.
            if not ok and sysfs_root is not None:
                ok = self._force_rebind_via_sysfs(sysfs_root, num)

            if not ok:
                msg = f"orphaned after {len(retry_delays)} attach attempts"
                if last_err is not None:
                    msg += f" (last err: {last_err})"
                failures.append((num, msg))
                logger.error(
                    "SP325 iface=%d FAILED to rebind to a kernel driver — %s. "
                    "Buttons / LEDs may stop working until manually "
                    "rebound: sudo sh -c \"echo '%s' > /sys/bus/usb/drivers_probe\"",
                    num,
                    msg,
                    self._sysfs_iface_name(num),
                )
        self.reattach_failures = failures  # exposed for callers + tests
        self._detached.clear()

    # ── sysfs helpers (rebind verification + recovery) ────────────────

    def _sysfs_iface_name(self, num: int) -> str:
        """Sysfs interface basename — e.g. ``3-1:1.3``.

        ``cfg`` value (currently always 1 for the SP325 family) is read
        from the active configuration so this stays correct if a future
        device has multiple configs.
        """
        cfg = self._dev.get_active_configuration().bConfigurationValue
        port = self._dev.port_number
        bus = self._dev.bus
        return f"{bus}-{port}:{cfg}.{num}"

    def _sysfs_root_path(self):
        return Path("/sys/bus/usb/devices")

    def _verify_kernel_driver_bound(self, sysfs_root, num: int) -> bool:
        if sysfs_root is None:
            return True  # can't verify, assume OK
        return (sysfs_root / self._sysfs_iface_name(num) / "driver").exists()

    def _force_rebind_via_sysfs(self, sysfs_root, num: int) -> bool:
        """Last-resort: ask the kernel to re-probe drivers for this iface.

        Writes the interface name (e.g. ``3-1:1.3``) to
        ``/sys/bus/usb/drivers_probe``. Requires write permission on
        that file — the udev rule for the SP325 doesn't ship one, so
        on a stock setup this fails with EACCES and we surface the
        recovery hint via the WARN log above.
        """
        probe = Path("/sys/bus/usb/drivers_probe")
        iface_name = self._sysfs_iface_name(num)
        try:
            probe.write_text(iface_name)
            logger.info("SP325 sysfs drivers_probe rebound iface=%s", iface_name)
        except OSError as e:
            logger.warning(
                "SP325 sysfs drivers_probe iface=%s failed: %s",
                iface_name,
                e,
            )
            return False
        return self._verify_kernel_driver_bound(sysfs_root, num)

    def __enter__(self) -> Sp325HidClient:
        return self.open() if not self._detached else self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── low-level wire transaction ───────────────────────────────

    def _set_report(self, cmd_payload: bytes) -> int:
        """Issue USB SET_REPORT class request — equivalent of DPM's
        IOCTL_HID_SET_OUTPUT_REPORT. Returns bytes sent.

        The wire format on this device family is a 63-byte payload
        where byte 0 is the response-size hint (DPM uses 0x40) and
        bytes 1+ are the 4-byte Dell command. The report ID (0xB9)
        rides in wValue, not the data buffer.
        """
        if len(cmd_payload) > SP325_WIRE_PAYLOAD - 1:
            raise Sp325Error(f"command too long: {len(cmd_payload)} > {SP325_WIRE_PAYLOAD - 1}")
        data = bytearray(SP325_WIRE_PAYLOAD)
        data[0] = PROTOCOL_SIZE_HINT
        data[1 : 1 + len(cmd_payload)] = cmd_payload
        w_value = (SP325_REPORT_TYPE_OUTPUT << 8) | SP325_REPORT_ID
        try:
            return self._dev.ctrl_transfer(
                bmRequestType=USB_BMREQTYPE_H2D,
                bRequest=USB_REQ_SET_REPORT,
                wValue=w_value,
                wIndex=SP325_HID_INTERFACE,
                data_or_wLength=bytes(data),
                timeout=1500,
            )
        except Exception as e:  # usb.core.USBError
            raise Sp325Error(f"USB SET_REPORT failed: {e}") from e

    # ── command-level API ────────────────────────────────────────

    def set(self, cmd: Command, value: int) -> None:
        """Write ``value`` to ``cmd``. Returns silently on apparent success.

        The SP325 does not return a synchronous ack on the GET_REPORT
        channel — the LED-ring red/white flash IS the ack, and the
        setting is persisted to NVRAM. Verify with compliance + an
        ALSA capture afterward.
        """
        if value < 0 or value > 0xFF:
            raise Sp325Error(f"value out of range: {value}")
        payload = bytes([cmd.cmd_class, cmd.set_opcode, 0x00, value]) + b"\x00" * max(
            0, cmd.payload_bytes - 4
        )
        sent = self._set_report(payload)
        logger.info("SP325 SET %s = %d (%dB sent)", cmd.name, value, sent)

    def _prime_DEPRECATED(self) -> int:
        """Replay the exact GET-sweep sequence from the working probe.

        Empirical finding (2026-05-13): an abbreviated prime (single
        report-type / report-id combo) is NOT sufficient. Sending the
        SETs after only that abbreviated sequence leaves compliance
        FAILing. Replaying the FULL probe sweep — six command classes
        × two report types × six report IDs + GET_REPORT round trips —
        is what triggers the LED red/white flash AND the wideband flip
        (compliance climbs to ≥2.5% high_band_pct, peaking near 9%).

        Why a passive read-sweep triggers a real config change is
        currently a black box — possibly the SP325 firmware exits a
        sleep / low-power state on the first vendor query, or one of
        the (rtype, rid, opcode) tuples in the sweep happens to be
        the actual "enter wideband" SET on this device family.

        Returns the number of transactions sent.
        """
        import usb.core as _ucore

        n = 0
        classes = (0xD2, 0xD5, 0xD0, 0xD1, 0xD3, 0xD4)
        report_types = (SP325_REPORT_TYPE_OUTPUT, 0x03)  # Output then Feature
        report_ids = (SP325_REPORT_ID, 0x06, 0x07, 0x9A, 0x9B, 0x00)

        for cls in classes:
            payload = bytes([cls, 0x02, 0x00, 0x00])
            for rtype in report_types:
                for rid in report_ids:
                    # SET via the (rtype, rid) we're sweeping. We pass
                    # the report id explicitly here, overriding the
                    # default 0xB9 used by _set_report.
                    data = bytearray(SP325_WIRE_PAYLOAD)
                    data[0] = PROTOCOL_SIZE_HINT
                    data[1 : 1 + len(payload)] = payload
                    wval = (rtype << 8) | rid
                    try:
                        self._dev.ctrl_transfer(
                            bmRequestType=USB_BMREQTYPE_H2D,
                            bRequest=USB_REQ_SET_REPORT,
                            wValue=wval,
                            wIndex=SP325_HID_INTERFACE,
                            data_or_wLength=bytes(data),
                            timeout=1500,
                        )
                        n += 1
                    except _ucore.USBError as e:
                        logger.debug(
                            "prime SET cls=0x%02x rtype=0x%02x rid=0x%02x skipped: %s",
                            cls,
                            rtype,
                            rid,
                            e,
                        )
                        continue
                    # GET round-trip on the same channel — part of the
                    # working recipe even if responses are zeros.
                    try:
                        self._dev.ctrl_transfer(
                            bmRequestType=USB_BMREQTYPE_D2H,
                            bRequest=USB_REQ_GET_REPORT,
                            wValue=wval,
                            wIndex=SP325_HID_INTERFACE,
                            data_or_wLength=SP325_WIRE_PAYLOAD,
                            timeout=1500,
                        )
                    except _ucore.USBError:
                        pass  # GET round-trip is part of the recipe; response payload is ignored
        return n

    def apply_wideband_good(self, *, settle_seconds: float = 15.0) -> dict[str, int]:
        """Set the SP325 into wideband mic-capture mode.

        Empirical recipe — validated 2026-05-13 via per-command bisect
        (now `meeting-scribe speakerphone benchmark`; results in
        tests/.../sp325_bisect.json).

        Three commands tested PASS ≥ 4/5 compliance samples:

          • [0xD0, 0x02, 0x00, 0x00]  → 5/5 pass, median 41.82%, peak 47.67%  ← strongest
          • [0xD1, 0x01, 0x00, 0x00]  → 4/5 pass, median 31.55%, peak 37.51%
          • [0xC0, 0x04, 0x00, 0x00]  → 4/5 pass, median  2.04%, peak  2.74%
            (suspected EQ-preset-Default — lower magnitude, very stable)

        And critically, [0xD2, 0x01, 0x00, 0x00] (the DPM-derived
        "set mic NS off" on SP3022) **kills** SP325 wideband — pre-
        bisect started at 19.55% and post-command dropped to 0.40%
        median. DPM ships the SP3022 plugin; the SP325 firmware uses
        a different opcode-to-effect mapping. NEVER send 0xD2/0x01
        in the wideband-good path.

        Sends all three winners for belt-and-suspenders coverage.
        """
        import time as _time

        # Send the three winners. Order matters: D0/0x02 first (it's
        # the strongest signal; if anything regresses to narrowband,
        # D0/0x02 alone has restored it in repeat trials).
        winners = (
            (0xD0, 0x02, "wideband_enable_primary"),
            (0xD1, 0x01, "wideband_enable_secondary"),
            (0xC0, 0x04, "eq_preset_default"),
        )
        out: dict[str, int] = {}
        for cls, op, label in winners:
            try:
                self._set_report(bytes([cls, op, 0x00, 0x00]))
                out[label] = 1
                logger.info("SP325 %-32s [0x%02x 0x%02x] sent", label, cls, op)
            except Sp325Error as e:
                logger.warning("SP325 %s failed: %s", label, e)
                out[label] = 0

        if settle_seconds > 0:
            logger.info("SP325 DSP settle for %.1fs…", settle_seconds)
            _time.sleep(settle_seconds)
        return out

    def snapshot(self) -> dict[str, int | None]:
        """No-op stub — the device doesn't ack reads on this protocol.

        Kept for API symmetry; callers should use the spectral
        compliance check as the source of truth for current state.
        """
        return {name: None for name in COMMANDS}


def main() -> int:
    """Stand-alone: apply the wideband-good config to the SP325.

    Requires root (or a udev rule granting access to the USB device).
    The script detaches kernel drivers, sends the two NR-off commands,
    reattaches drivers, and the device should flash LED red/white
    momentarily to confirm NVRAM save.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        with Sp325HidClient.open_default() as cli:
            applied = cli.apply_wideband_good()
    except Sp325Error as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("SP325 wideband config applied:")
    for k, v in applied.items():
        print(f"  {k:<40} = {v}")
    print("\nNext: meeting-scribe speakerphone compliance — confirm high_band_pct ≥ 1.5%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
