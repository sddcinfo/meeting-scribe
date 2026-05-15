"""Priority-ordered LED state machine.

A single :class:`Resolver` instance decides what the Mute LED should be
doing right now based on five inputs: error flag, backend readiness,
mic mute state, meeting-recording state, and the per-state config from
the mapping document. It does not write hidraw — the daemon's LED task
plugs the resolved pattern into a :class:`PatternRunner`.

State priority (highest first):

1. ``error`` — daemon caught an exception talking to the backend.
2. ``backend_unready`` — readiness endpoint is not OK.
3. ``mic_muted`` — meeting-scribe reports the mic muted.
4. ``recording`` — an active meeting is recording.
5. ``idle_ready`` — fall-through.

A state that is configured ``enabled=False`` in the mapping is skipped
(the next-highest enabled state wins). ``idle_ready`` is the floor; if
the operator disables every other state, ``idle_ready`` still resolves
so the LED has *something* to do.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from meeting_scribe.speakerphone.constants import LED_PATTERNS, LED_STATES


@dataclass(frozen=True)
class SystemSignals:
    """Inputs into the state machine. All optional; defaults = idle/ready."""

    error: bool = False
    backend_unready: bool = False
    mic_muted: bool = False
    recording: bool = False


@dataclass(frozen=True)
class LedResolution:
    """The state machine's decision for this tick."""

    state: str
    pattern: str


_DEFAULT_PATTERNS: Mapping[str, str] = {
    "error": "very_fast_blink",
    "backend_unready": "fast_blink",
    "mic_muted": "solid",
    "recording": "slow_pulse",
    "idle_ready": "off",
}


def resolve(
    signals: SystemSignals,
    state_config: Mapping[str, Mapping[str, object]] | None = None,
) -> LedResolution:
    """Pick the active state + pattern from ``signals`` and config.

    ``state_config`` is the ``leds.states`` block from the mapping
    document. If absent or partial, defaults from
    :data:`_DEFAULT_PATTERNS` fill the gaps so the daemon never crashes
    on a malformed sidecar.
    """
    cfg = state_config or {}
    active_predicates = {
        "error": signals.error,
        "backend_unready": signals.backend_unready,
        "mic_muted": signals.mic_muted,
        "recording": signals.recording,
        "idle_ready": True,  # always active as the floor
    }
    for state_name in LED_STATES:
        if not active_predicates[state_name]:
            continue
        behavior = cfg.get(state_name, {})
        enabled = behavior.get("enabled", True)
        if not isinstance(enabled, bool) or not enabled:
            # Operator disabled this state — fall through to the next.
            continue
        pattern_name = behavior.get("pattern")
        if not isinstance(pattern_name, str) or pattern_name not in LED_PATTERNS:
            pattern_name = _DEFAULT_PATTERNS.get(state_name, "off")
        return LedResolution(state=state_name, pattern=pattern_name)
    # Safety net: every state was disabled including idle_ready. Should
    # never happen because validate() requires an enabled bool, but
    # defend against config corruption.
    return LedResolution(state="idle_ready", pattern="off")


__all__ = ["LedResolution", "SystemSignals", "resolve"]
