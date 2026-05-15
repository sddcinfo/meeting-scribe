"""Known USB speakerphone catalog.

Keyed by ``"vid:pid"`` strings (lowercase 4-hex-digit each) so the SPA can
render them in URLs/JSON without escaping. Each entry maps to a stable
device profile name + the canonical mapping/LED defaults the
``mapping.py`` module ships when a sidecar JSON is created from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    """A catalog entry for one VID:PID."""

    vid: int
    pid: int
    name: str
    # input event device whose key codes carry the telephony buttons.
    # On the SP325 this is the device with the "leds" handler in
    # /proc/bus/input/devices.
    telephony_node_hint: str = "telephony"
    # Whether the device's hidraw node carries vendor-page reports that
    # may contain a separate Teams-button bit (decoded at runtime via
    # capture-descriptor). Best-effort flag — defaults to True since most
    # Teams-certified speakerphones do.
    has_vendor_teams_channel: bool = True


# Lowercase "vid:pid" → profile. Add new entries here; the rest of the
# daemon switches behavior off the resolved profile, not the raw IDs.
CATALOG: dict[str, DeviceProfile] = {
    "413c:8223": DeviceProfile(
        vid=0x413C,
        pid=0x8223,
        name="Dell Pro SP325 Speakerphone",
    ),
    # SP3022 — predecessor; same HID-telephony pattern, ships in the
    # catalog from day one so users with the older device aren't a
    # special case. The descriptor differs slightly but the daemon only
    # uses the standard telephony usage IDs so it works without code
    # changes.
    "413c:8222": DeviceProfile(
        vid=0x413C,
        pid=0x8222,
        name="Dell Pro SP3022 Speakerphone",
    ),
}


def vid_pid_key(vid: int, pid: int) -> str:
    """Render a VID:PID pair into the canonical ``"vid:pid"`` form."""
    return f"{vid:04x}:{pid:04x}"


def lookup(vid: int, pid: int) -> DeviceProfile | None:
    """Return the catalog entry for ``vid:pid`` or ``None`` if unknown."""
    return CATALOG.get(vid_pid_key(vid, pid))


def known_vids() -> set[int]:
    """All VIDs the daemon should pay attention to (udev filter)."""
    return {entry.vid for entry in CATALOG.values()}
