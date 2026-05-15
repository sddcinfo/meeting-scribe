"""USB-speakerphone integration (Dell SP325 + future Teams-certified devices).

A standalone subsystem that bridges the device's HID telephony page (buttons +
LEDs) to meeting-scribe runtime state. Architecture: kernel evdev/hidraw on
one side, meeting-scribe HTTP API over a Unix-domain socket on the other.

The package is split into pure-userspace logic (catalog, descriptor, mapping,
hid_leds, actions, led_state_machine) and I/O drivers (evdev_listener, daemon,
service_install) so the load-bearing decisions can be tested without hardware.

Public entry points live in ``meeting_scribe.cli.speakerphone`` and the
HTTP routers under ``meeting_scribe.routes.{admin,internal}_speakerphone``.
"""

from __future__ import annotations
