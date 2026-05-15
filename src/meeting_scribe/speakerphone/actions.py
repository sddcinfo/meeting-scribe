"""Action registry + cycle math for the speakerphone daemon.

This module is the boundary between "button press detected" and "make
the HTTP request". The actions themselves are async callables that take
a :class:`MeetingClient`-shaped object plus a context dict; they're
registered by name in :data:`ACTIONS` and dispatched by the daemon.

Cycle math (Phone-button short-press) lives in :func:`next_tts_state`
so the test suite can exercise every transition without spinning up
the daemon or HTTP machinery.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from meeting_scribe.speakerphone.constants import ACTION_REGISTRY

# ── HTTP client contract the daemon hands to action handlers ───────────


class MeetingClient(Protocol):
    """Minimal interface action handlers depend on.

    The daemon's real client (``meeting_client.UdsMeetingClient``) wraps
    ``httpx.AsyncClient`` with a UDS transport; the test suite provides
    a stub that records calls. Keeping the surface narrow means
    test stubs stay tiny.
    """

    async def get_state(self) -> dict[str, Any]: ...

    async def set_interpretation(
        self,
        *,
        enabled: bool | None = None,
        room_tts_language: str | None = None,
    ) -> dict[str, Any]: ...

    async def toggle_mic_mute(self) -> dict[str, Any]: ...

    async def toggle_meeting_record(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ActionContext:
    """Per-call context passed to every action handler.

    ``device_key`` is the ``vid:pid`` of the device whose button was
    pressed (lets handlers attribute actions to a specific device if
    multiple are connected). ``button`` is the abstract button name
    (``"phone"``, ``"teams"``, ``"phone_mute"``). ``press_kind`` is
    ``"short"`` or ``"long"``.
    """

    device_key: str
    button: str
    press_kind: str


# ── TTS cycle math ──────────────────────────────────────────────────────


def next_tts_state(
    current_language: str,
    interpretation_enabled: bool,
    meeting_languages: list[str],
) -> tuple[bool, str]:
    """Compute the next ``(enabled, room_tts_language)`` after a Phone-button short press.

    Args:
        current_language: Current ``room_tts_language`` (e.g. ``"en"``,
            ``"ja"``, ``"all"``).
        interpretation_enabled: Whether interpretation is currently on.
            If False, the next state always enables interpretation at the
            first language (the user pressed a button — they want sound).
        meeting_languages: The two-letter codes configured for the
            current meeting. Two languages → 3-state cycle
            ``[lang_a, lang_b, "all"]``. One language → 2-state cycle
            ``[lang_a, "all"]``. Zero or three+ → fall back to
            ``["all"]`` to avoid surprising the user.

    Returns:
        ``(enabled, next_language)`` to POST to the interpretation
        endpoint.
    """
    if not interpretation_enabled:
        # Re-enable at the *first* meeting language (or "all" if no pair).
        first = meeting_languages[0] if meeting_languages else "all"
        return True, first

    cycle = _build_cycle(meeting_languages)
    try:
        idx = cycle.index(current_language)
    except ValueError:
        # Current language isn't in the cycle (e.g. operator picked a
        # third language in the GUI). Start fresh at index 0.
        return True, cycle[0]
    return True, cycle[(idx + 1) % len(cycle)]


def _build_cycle(meeting_languages: list[str]) -> list[str]:
    """Return the rotation order for a Phone-button cycle.

    Always returns a non-empty list. ``"all"`` is the last element when
    the meeting has 1+ languages so users hit it after picking each
    individual direction.
    """
    langs = [lang for lang in meeting_languages if isinstance(lang, str) and len(lang) == 2]
    seen: list[str] = []
    for lang in langs:
        if lang not in seen:
            seen.append(lang)
    seen = seen[:2]
    if not seen:
        return ["all"]
    return [*seen, "all"]


# ── Action handlers ────────────────────────────────────────────────────


async def _action_noop(client: MeetingClient, ctx: ActionContext) -> None:
    """Do nothing. Used when an operator wants to disable a button."""
    _ = client, ctx
    return None


async def _action_tts_cycle(client: MeetingClient, ctx: ActionContext) -> None:
    """Advance the TTS direction one step in the meeting-language cycle."""
    state = await client.get_state()
    interp = state.get("interpretation", {})
    enabled = bool(interp.get("enabled", False))
    current = str(interp.get("room_tts_language", "all"))
    languages = state.get("meeting_languages") or []
    new_enabled, new_lang = next_tts_state(current, enabled, list(languages))
    await client.set_interpretation(
        enabled=new_enabled,
        room_tts_language=new_lang,
    )


async def _action_interpretation_toggle(
    client: MeetingClient,
    ctx: ActionContext,
) -> None:
    """Flip interpretation on/off.

    On re-enable, the server restores ``interpretation_last_room_tts_language``
    — we never pass an explicit direction here. On disable, the daemon
    is satisfied with ``enabled=false`` and lets the server preserve
    the stored direction.
    """
    state = await client.get_state()
    interp = state.get("interpretation", {})
    enabled = bool(interp.get("enabled", False))
    if enabled:
        await client.set_interpretation(enabled=False)
    else:
        # Send ``enabled=True`` without ``room_tts_language`` so the server
        # applies ``interpretation_last_room_tts_language`` itself.
        await client.set_interpretation(enabled=True)


async def _action_meeting_record_toggle(
    client: MeetingClient,
    ctx: ActionContext,
) -> None:
    """Start a meeting from idle (using the default profile) or stop the active one.

    The server-side handler decides start vs. stop based on current
    state and applies the default profile in a single transaction.
    """
    await client.toggle_meeting_record()


async def _action_mic_mute_toggle(
    client: MeetingClient,
    ctx: ActionContext,
) -> None:
    """Toggle the mic mute state."""
    await client.toggle_mic_mute()


# Registry keyed by the action names that appear in the mapping schema.
# Anything outside this map is rejected by mapping validation; the
# daemon will refuse to dispatch an unknown action defensively too.
ACTIONS: dict[str, Callable[[MeetingClient, ActionContext], Awaitable[None]]] = {
    "noop": _action_noop,
    "tts_cycle": _action_tts_cycle,
    "interpretation_toggle": _action_interpretation_toggle,
    "meeting_record_toggle": _action_meeting_record_toggle,
    "mic_mute_toggle": _action_mic_mute_toggle,
}


def assert_registry_complete() -> None:
    """Sanity check: every action name in :data:`ACTION_REGISTRY` is
    implemented by a callable in :data:`ACTIONS`.

    Called by the daemon at startup so a typo in the registry causes
    a clear error instead of a 500 deep inside a button-press path.
    """
    missing = ACTION_REGISTRY - set(ACTIONS)
    extra = set(ACTIONS) - ACTION_REGISTRY
    if missing or extra:
        raise RuntimeError(
            f"action registry/implementation mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)}",
        )


__all__ = [
    "ACTIONS",
    "ActionContext",
    "MeetingClient",
    "assert_registry_complete",
    "next_tts_state",
]
