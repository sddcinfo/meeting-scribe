"""Read-only access to ``/run/meeting-scribe/hdmi-status.json``.

The kiosk-runtime (cage session) writes this file on startup and on
every HDMI hot-plug event. It's the single source of truth for
"what does wlr-randr think is connected and at which mode". The
admin REST handler reads this file when populating the HDMI Display
settings panel's status block + available-modes dropdown.

Shape (written by :mod:`meeting_scribe.kiosk.runtime`)::

    {
      "connected": true,
      "current_mode": "3840x2160@60.000Hz",
      "available_modes": ["3840x2160@60.000Hz", "2560x1440@60.000Hz", ...],
      "rotation": 0,
      "enabled": true,
      "edid_name": "ACME 4K 27",
      "updated_at": 1747171200.0
    }

If the file is missing (kiosk service not started yet) or stale
(older than ~10 minutes), callers get a sentinel response so the
admin UI can render a "no kiosk running" placeholder instead of a
500 error.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default path matches the runtime's writer; override via the env var
# for tests or development without the systemd unit running.
_DEFAULT_PATH = Path("/run/meeting-scribe/hdmi-status.json")

# After this many seconds we treat the file as stale and surface
# ``connected=false`` to the admin. The runtime re-writes on every
# udev hotplug AND on a 5 min watchdog tick, so 10 min is generous.
_STALE_THRESHOLD_SECONDS = 600


def _path() -> Path:
    """Resolve the status-file path, allowing env-var override."""
    import os

    raw = os.environ.get("MEETING_SCRIBE_HDMI_STATUS_PATH")
    return Path(raw) if raw else _DEFAULT_PATH


def _sentinel() -> dict[str, Any]:
    """Return the "no kiosk available" placeholder shape."""
    return {
        "connected": False,
        "current_mode": None,
        "available_modes": [],
        "rotation": 0,
        "enabled": False,
        "edid_name": None,
        "updated_at": None,
        "source": "sentinel",
    }


def read_status() -> dict[str, Any]:
    """Read the live HDMI status blob.

    Returns the sentinel shape when the file is missing, unreadable,
    malformed, or stale. Never raises — admin UI must always get a
    well-formed response so the settings tab renders.
    """
    path = _path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _sentinel()
    except OSError as exc:
        logger.warning("hdmi_status: read failed (%s): %s", path, exc)
        return _sentinel()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("hdmi_status: JSON parse failed (%s): %s", path, exc)
        return _sentinel()

    if not isinstance(data, dict):
        logger.warning("hdmi_status: payload is not a dict (%s)", path)
        return _sentinel()

    updated_at = data.get("updated_at")
    if isinstance(updated_at, (int, float)) and time.time() - updated_at > _STALE_THRESHOLD_SECONDS:
        out = _sentinel()
        out["updated_at"] = updated_at
        out["source"] = "stale"
        return out

    # Normalize shape with explicit defaults so the admin UI can
    # render every field unconditionally.
    return {
        "connected": bool(data.get("connected", False)),
        "current_mode": data.get("current_mode") or None,
        "available_modes": list(data.get("available_modes") or []),
        "rotation": int(data.get("rotation", 0)),
        "enabled": bool(data.get("enabled", False)),
        "edid_name": data.get("edid_name") or None,
        "updated_at": updated_at,
        "source": "live",
    }


def is_mode_supported(mode: str) -> bool:
    """True when ``mode`` is in the current available-modes list.

    Used by the settings PUT validator to reject modes that wlr-randr
    can't actually apply. ``"auto"`` always validates (kiosk-runtime
    will pick the EDID-preferred mode).
    """
    if mode == "auto":
        return True
    status = read_status()
    return mode in status.get("available_modes", [])
