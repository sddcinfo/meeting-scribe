"""HDMI kiosk: cage + chromium runtime + the in-process state it owns.

The kiosk role is the GB10-local cage chromium session that mirrors
the laptop operator's admin pop-out onto the connected HDMI display.

Surface:
  * :mod:`meeting_scribe.kiosk.nonces` - in-process single-use nonce
    store used by ``/kiosk-bootstrap`` to gate cookie issuance.
  * :mod:`meeting_scribe.kiosk.runtime` - the long-running Python
    process started by ``cage`` that drives wlr-randr, launches
    chromium, and owns the DPMS idle timer.
  * :mod:`meeting_scribe.kiosk.hdmi_status` - read-only helper that
    surfaces the ``/run/meeting-scribe/hdmi-status.json`` blob the
    runtime publishes; admin REST reads it for the settings panel.
"""
