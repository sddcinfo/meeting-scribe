"""HID report-descriptor decoder for telephony speakerphones.

Two responsibilities:

1. Parse a raw HID report descriptor into a structured list of report
   metadata (report ID → list of usages and bit layout). We only handle
   the subset we actually care about — Consumer (0x0C), Telephony
   (0x0B), LED (0x08), and vendor pages (0xFF13/0xFF14/0xFF99 on the
   SP325, treated as opaque byte blobs). Enough to assert in tests that
   a known descriptor decodes to the expected usages, and enough to
   detect when a future device deviates.

2. Encode/decode the SP325's Report ID 5 — the load-bearing telephony +
   LED report. Bit layout (1 byte after the report ID):

   * **Input (button states sent by device)**:
     - bit 0: Hook Switch (Phone / call button)
     - bit 1: Phone Mute (mic mute button)
     - bit 2: Redial
     - bit 3: Phone Key
     - bit 4: Line Busy Tone
     - bit 5: Speaker (Phone Speaker)
     - bit 6: Phone usage (0x50)
     - bit 7: padding (Button page filler)

   * **Output (LED states the host writes)**:
     - bit 0: Off-Hook LED
     - bit 1: Mute LED  ← the only one physically present on the SP325
     - bit 2: Ring LED
     - bit 3: Hold LED
     - bit 4: Microphone LED
     - bit 5: Off-Line LED
     - bits 6-7: padding

The SP325 also has a Teams button that may not show up as a standard
HID telephony usage; many Teams-certified devices route it through the
vendor page (0xFF13/0xFF99) as an opaque byte sequence the host has to
pattern-match. We expose ``decode_vendor_report`` as a passthrough so
the calibration command (``speakerphone capture-descriptor``) can dump
those reports for the operator to inspect.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum

# Captured canonical descriptor bytes for the Dell SP325 (413c:8223),
# dumped 2026-05-13 from /sys/class/hidraw/hidraw0/device/report_descriptor
# on a live device. Kept in-tree as a test fixture so
# ``test_speakerphone_descriptor_decode.py`` does not need hardware. Any
# device whose descriptor matches this exactly is guaranteed to round-trip
# through ``decode_telephony_report`` and ``encode_led_output_report``.
SP325_DESCRIPTOR: bytes = bytes.fromhex(
    "050c0901a101850c1500250109e909ea09e209cd09b509b609b009b1"
    "750195088102c00613ff0901a101150026ff00850609007508953d91"
    "02850709007508953d8102c00614ff0901a101150026ff0085100900"
    "750895058102c00699ff0901a101859a150026ff00953f75080903b1"
    "02859b150025019501750109048106750195078101c0050b0905a101"
    "1500250175018505092095018122092f95018106092409210997092a"
    "0950950581060907050909017501950181020508850509170909091"
    "809200921092a9506912295029101c0"
)


class TelephonyButton(IntEnum):
    """Bit positions in the Report ID 5 input byte (telephony page)."""

    HOOK_SWITCH = 0  # the "phone" / call button
    PHONE_MUTE = 1
    REDIAL = 2
    PHONE_KEY = 3
    LINE_BUSY = 4
    SPEAKER = 5
    PHONE = 6
    BUTTON1_PAD = 7  # filler bit from Button page; not a real button


# Public stable names used everywhere (mapping schema, dispatcher, tests).
# Maps the abstract button name to the bit position emitted by the
# device. Anything emitted on a bit we don't recognise is ignored.
BUTTON_BITS: dict[str, TelephonyButton] = {
    "phone": TelephonyButton.HOOK_SWITCH,
    "phone_mute": TelephonyButton.PHONE_MUTE,
    "redial": TelephonyButton.REDIAL,
    "phone_key": TelephonyButton.PHONE_KEY,
    "line_busy": TelephonyButton.LINE_BUSY,
    "speaker": TelephonyButton.SPEAKER,
    "phone_usage": TelephonyButton.PHONE,
}


class LedBit(IntEnum):
    """Bit positions in the Report ID 5 output byte (LED page)."""

    OFF_HOOK = 0
    MUTE = 1
    RING = 2
    HOLD = 3
    MICROPHONE = 4
    OFF_LINE = 5


# The SP325 only has one physical LED ring (the Mute LED).
LED_NAMES: dict[str, LedBit] = {
    "off_hook_led": LedBit.OFF_HOOK,
    "mute_led": LedBit.MUTE,
    "ring_led": LedBit.RING,
    "hold_led": LedBit.HOLD,
    "microphone_led": LedBit.MICROPHONE,
    "off_line_led": LedBit.OFF_LINE,
}


# ── Report ID 5 (telephony) input/output codec ──────────────────────────


def decode_telephony_report(payload: bytes) -> dict[str, bool]:
    """Decode the byte that follows Report ID 5 into a button-name map.

    Returns a dict ``{button_name: pressed?}`` for every button we
    recognise. The padding bit is dropped. Accepts the bare data byte
    (no report ID prefix); pass ``payload[1:2]`` from a hidraw read.
    """
    if not payload:
        raise ValueError("empty telephony payload")
    byte = payload[0]
    return {name: bool(byte & (1 << int(bit))) for name, bit in BUTTON_BITS.items()}


def pressed_buttons(payload: bytes) -> set[str]:
    """Convenience: return only the names whose bit is currently set."""
    return {name for name, on in decode_telephony_report(payload).items() if on}


def encode_led_output_report(states: dict[str, bool]) -> bytes:
    """Pack a ``{led_name: on?}`` map into the single output byte.

    Unknown LED names are silently ignored (defensive against a future
    device adding LEDs we don't ship support for). Bits not mentioned
    default to off. The caller is responsible for prepending the
    report ID and writing to ``/dev/hidraw*``.
    """
    byte = 0
    for name, on in states.items():
        bit = LED_NAMES.get(name)
        if bit is None:
            continue
        if on:
            byte |= 1 << int(bit)
    return bytes((byte,))


# ── Light-weight descriptor parsing (just enough for the test fixture) ──


@dataclass(frozen=True)
class ReportEntry:
    """One report ID worth of decoded usages."""

    report_id: int
    direction: str  # "input" | "output" | "feature"
    usage_page: int
    usages: tuple[int, ...] = field(default_factory=tuple)
    bit_size: int = 0
    bit_count: int = 0


def _iter_items(blob: bytes) -> Iterable[tuple[int, int, int]]:
    """Yield (item_tag_type, item_size, payload_int) tuples.

    Short-form items only — sufficient for every descriptor a USB
    speakerphone we care about will emit. Long-form (rare) raises.
    """
    i = 0
    while i < len(blob):
        prefix = blob[i]
        i += 1
        if prefix == 0xFE:
            raise NotImplementedError("long-form HID items unsupported")
        size = prefix & 0x03
        # The spec encodes 0/1/2/4-byte payloads via the low 2 bits.
        if size == 3:
            size = 4
        if i + size > len(blob):
            raise ValueError("truncated HID descriptor")
        payload = int.from_bytes(blob[i : i + size], "little")
        # Sign-extend for signed Local item payloads if needed — not
        # used by our subset, so leave it unsigned.
        yield prefix & 0xFC, size, payload
        i += size


def parse_descriptor(blob: bytes) -> list[ReportEntry]:
    """Walk a raw descriptor and surface the (report_id, page, usages) tuples.

    Returns one entry per Input/Output/Feature item, preserving the
    encounter order. Usages are accumulated since the previous main
    item (matching the HID spec's Local-item-pool semantics) and then
    consumed/cleared on each main item.

    This decoder targets the speakerphone subset; it is **not** a
    full HID parser. Items it doesn't understand are skipped.
    """
    entries: list[ReportEntry] = []
    usage_page = 0
    report_id = 0
    report_size = 0
    report_count = 0
    pending_usages: list[int] = []

    for tag, size, value in _iter_items(blob):
        # Global items
        if tag == 0x04:  # Usage Page
            # Some devices encode usage-page as 2 bytes for vendor pages
            # (0xFF13 etc). Only retain the low 16 bits.
            usage_page = value & 0xFFFF
        elif tag == 0x84:  # Report ID
            report_id = value
        elif tag == 0x74:  # Report Size
            report_size = value
        elif tag == 0x94:  # Report Count
            report_count = value
        # Local items
        elif tag == 0x08:  # Usage
            pending_usages.append(value)
        # Main items
        elif tag in (0x80, 0x90, 0xB0):  # Input / Output / Feature
            direction = {0x80: "input", 0x90: "output", 0xB0: "feature"}[tag]
            entries.append(
                ReportEntry(
                    report_id=report_id,
                    direction=direction,
                    usage_page=usage_page,
                    usages=tuple(pending_usages),
                    bit_size=report_size,
                    bit_count=report_count,
                ),
            )
            pending_usages.clear()
        elif tag == 0xA0:  # Collection — pending usages are consumed
            pending_usages.clear()
        elif tag == 0xC0:  # End Collection — nothing to capture
            pass
        # Everything else: skip silently. Sufficient for our subset.
        _ = size  # size is informational; payload already decoded
    return entries


def describe(entries: list[ReportEntry]) -> str:
    """Render decoded entries as a human-readable multi-line string.

    Used by ``meeting-scribe speakerphone capture-descriptor`` so an
    operator can see at a glance what the device exposes.
    """
    lines: list[str] = []
    for e in entries:
        usage_hex = ", ".join(f"0x{u:02x}" for u in e.usages) or "(none)"
        lines.append(
            f"report_id=0x{e.report_id:02x} dir={e.direction:<7} "
            f"page=0x{e.usage_page:04x} "
            f"size={e.bit_size:>2}b count={e.bit_count:>2} usages=[{usage_hex}]"
        )
    return "\n".join(lines)
