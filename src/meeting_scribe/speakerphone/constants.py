"""Single source of truth for cross-module constants.

Anything that appears in daemon code, server-side handler code, the SPA UI
copy, and tests goes here — never duplicated. If you find yourself wanting
to define the same number twice, add it to this module instead.
"""

from __future__ import annotations

from typing import Final

# Long-press threshold for the Phone button. Daemon, UI ("hold ≥1 s"), and
# tests all read from here. User-configurable via the Hardware tab; this is
# the default if the mapping config doesn't override it.
DEFAULT_LONG_PRESS_MS: Final[int] = 1000

# Mapping config defaults used by `mapping.py` when the sidecar JSON is
# missing or has gaps. Kept in sync with the v1 schema in the plan.
MAPPING_SCHEMA_VERSION: Final[int] = 1

# Action vocabulary the daemon can dispatch. Anything outside this set is a
# 400 from the mapping PATCH/PUT endpoints.
ACTION_REGISTRY: Final[frozenset[str]] = frozenset(
    {
        "noop",
        "tts_cycle",
        "interpretation_toggle",
        "meeting_record_toggle",
        "mic_mute_toggle",
    },
)

# Buttons the daemon owns (subset of the HID telephony page). Consumer-page
# keys (Vol+/Vol-/sys-mute on event3) are deliberately absent; they flow
# through the kernel + media-key agent unchanged.
DAEMON_BUTTONS: Final[frozenset[str]] = frozenset(
    {
        "phone",
        "teams",
        "phone_mute",
    },
)

# OS-managed buttons surfaced as read-only rows in the Hardware tab. They
# never appear in the mapping schema; this list exists purely for the SPA
# to render the informational "Handled by OS" section.
OS_HANDLED_BUTTONS: Final[tuple[tuple[str, str], ...]] = (
    ("volume_up", "PipeWire default-sink volume up"),
    ("volume_down", "PipeWire default-sink volume down"),
    ("system_mute", "PipeWire default-sink mute toggle"),
)

# LED state-machine state names + canonical priority order (highest first).
# The state machine resolves to the first state whose `enabled` is True;
# its `pattern` drives the Mute LED ring.
LED_STATES: Final[tuple[str, ...]] = (
    "error",
    "backend_unready",
    "mic_muted",
    "recording",
    "idle_ready",
)

# LED patterns expressed as (on_ms, off_ms) pairs that loop. A single
# (on_ms, 0) is solid; (0, off_ms) is fully off; longer pairs blink.
LED_PATTERNS: Final[dict[str, tuple[tuple[int, int], ...]]] = {
    "off": ((0, 1000),),
    "solid": ((1000, 0),),
    "slow_blink": ((500, 500),),
    "blink": ((250, 250),),
    "fast_blink": ((125, 125),),
    "very_fast_blink": ((62, 63),),
    "slow_pulse": ((200, 1800),),
    "double_blink": ((125, 125, 125, 625),),
}

# Default mapping the descriptor module ships when no sidecar exists yet.
DEFAULT_LED_STATE_BEHAVIOR: Final[dict[str, dict[str, object]]] = {
    "error": {"enabled": True, "pattern": "very_fast_blink"},
    "backend_unready": {"enabled": True, "pattern": "fast_blink"},
    "mic_muted": {"enabled": True, "pattern": "solid"},
    "recording": {"enabled": True, "pattern": "slow_pulse"},
    "idle_ready": {"enabled": True, "pattern": "off"},
}

# Default per-meeting profile applied by Teams-from-idle. Lives in the
# sidecar mapping JSON (canonical home; not mirrored to settings_store).
DEFAULT_MEETING_PROFILE: Final[dict[str, object]] = {
    "name": "EN/JA bidirectional",
    "languages": ["en", "ja"],
    "interpretation_enabled": True,
    "room_tts_language": "all",
    "admin_tts_language": "en",
    "title_template": "Quick meeting — {timestamp}",
}

# Default button-feedback config in the sidecar mapping JSON. The
# server-side ``apply_speak`` re-reads this on every call so changes
# apply on the very next press with no daemon-poll lag. ``enabled``
# defaults True so a fresh install auditions audibly (the operator
# can disable it from the Hardware tab once they decide it's noise).
DEFAULT_BUTTON_FEEDBACK: Final[dict[str, object]] = {
    "enabled": True,
    "language": "en",
    "overrides": {},  # label_id → {lang_code: text}
}
