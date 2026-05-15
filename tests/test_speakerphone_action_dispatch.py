"""Tests for the speakerphone action registry + cycle math."""

from __future__ import annotations

import pytest

from meeting_scribe.speakerphone import actions
from meeting_scribe.speakerphone.constants import ACTION_REGISTRY


class FakeClient:
    """Minimal MeetingClient stub that records calls."""

    def __init__(self, state: dict) -> None:
        self._state = state
        self.calls: list[tuple[str, dict]] = []

    async def get_state(self) -> dict:
        self.calls.append(("get_state", {}))
        return self._state

    async def set_interpretation(
        self,
        *,
        enabled=None,
        room_tts_language=None,
    ) -> dict:
        body = {"enabled": enabled, "room_tts_language": room_tts_language}
        self.calls.append(("set_interpretation", body))
        # Mirror the server's effect on local state for cycle tests.
        if enabled is not None:
            self._state.setdefault("interpretation", {})["enabled"] = enabled
        if room_tts_language is not None:
            self._state.setdefault("interpretation", {})["room_tts_language"] = room_tts_language
        return self._state

    async def toggle_mic_mute(self) -> dict:
        self.calls.append(("toggle_mic_mute", {}))
        return {}

    async def toggle_meeting_record(self) -> dict:
        self.calls.append(("toggle_meeting_record", {}))
        return {}


def test_registry_complete() -> None:
    actions.assert_registry_complete()


def test_action_registry_covers_constants_registry() -> None:
    assert set(actions.ACTIONS.keys()) == set(ACTION_REGISTRY)


# ── next_tts_state cycle math ──────────────────────────────────────────


def test_cycle_en_ja_starts_at_en_when_starting_state_is_all() -> None:
    enabled, lang = actions.next_tts_state("all", True, ["en", "ja"])
    assert enabled is True
    assert lang == "en"


def test_cycle_en_ja_advances_en_to_ja() -> None:
    enabled, lang = actions.next_tts_state("en", True, ["en", "ja"])
    assert enabled is True
    assert lang == "ja"


def test_cycle_en_ja_advances_ja_to_all() -> None:
    enabled, lang = actions.next_tts_state("ja", True, ["en", "ja"])
    assert enabled is True
    assert lang == "all"


def test_cycle_en_ja_wraps_all_back_to_en() -> None:
    _enabled, lang = actions.next_tts_state("all", True, ["en", "ja"])
    assert lang == "en"  # since cycle is [en, ja, all]


def test_disabled_state_reenables_at_first_meeting_language() -> None:
    enabled, lang = actions.next_tts_state("ja", False, ["en", "ja"])
    assert enabled is True
    assert lang == "en"


def test_single_language_meeting_cycles_two_states() -> None:
    # 1 lang → cycle [lang_a, all]
    _enabled, lang = actions.next_tts_state("en", True, ["en"])
    assert lang == "all"
    _enabled, lang = actions.next_tts_state("all", True, ["en"])
    assert lang == "en"


def test_no_languages_falls_back_to_all_only() -> None:
    _enabled, lang = actions.next_tts_state("all", True, [])
    assert lang == "all"


def test_cycle_handles_current_lang_not_in_cycle() -> None:
    # Operator picked "fr" via the GUI but the meeting is EN/JA. The
    # daemon shouldn't crash — it just restarts at index 0.
    _enabled, lang = actions.next_tts_state("fr", True, ["en", "ja"])
    assert lang == "en"


def test_cycle_deduplicates_repeated_meeting_languages() -> None:
    _enabled, lang = actions.next_tts_state("all", True, ["en", "en"])
    # ["en", "en"] dedupes to ["en"] → cycle [en, all]
    assert lang == "en"


# ── Action handler integration with FakeClient ─────────────────────────


@pytest.mark.asyncio
async def test_tts_cycle_calls_set_interpretation_with_next_state() -> None:
    client = FakeClient(
        {
            "interpretation": {"enabled": True, "room_tts_language": "en"},
            "meeting_languages": ["en", "ja"],
        },
    )
    ctx = actions.ActionContext(device_key="413c:8223", button="phone", press_kind="short")
    await actions.ACTIONS["tts_cycle"](client, ctx)
    set_calls = [c for c in client.calls if c[0] == "set_interpretation"]
    assert set_calls == [
        ("set_interpretation", {"enabled": True, "room_tts_language": "ja"}),
    ]


@pytest.mark.asyncio
async def test_interpretation_toggle_disables_when_currently_enabled() -> None:
    client = FakeClient(
        {"interpretation": {"enabled": True, "room_tts_language": "ja"}},
    )
    ctx = actions.ActionContext(device_key="413c:8223", button="phone", press_kind="long")
    await actions.ACTIONS["interpretation_toggle"](client, ctx)
    last = client.calls[-1]
    assert last == ("set_interpretation", {"enabled": False, "room_tts_language": None})


@pytest.mark.asyncio
async def test_interpretation_toggle_reenables_without_specifying_direction() -> None:
    # When re-enabling, the action MUST NOT pass room_tts_language —
    # the server applies the persisted last direction. This is the
    # contract that lets "long-press off, long-press on, get the same
    # direction back" work even after a meeting-scribe restart.
    client = FakeClient(
        {"interpretation": {"enabled": False, "room_tts_language": "ja"}},
    )
    ctx = actions.ActionContext(device_key="413c:8223", button="phone", press_kind="long")
    await actions.ACTIONS["interpretation_toggle"](client, ctx)
    last = client.calls[-1]
    assert last == ("set_interpretation", {"enabled": True, "room_tts_language": None})


@pytest.mark.asyncio
async def test_meeting_record_toggle_dispatches_to_client() -> None:
    client = FakeClient({})
    ctx = actions.ActionContext(device_key="413c:8223", button="teams", press_kind="short")
    await actions.ACTIONS["meeting_record_toggle"](client, ctx)
    assert ("toggle_meeting_record", {}) in client.calls


@pytest.mark.asyncio
async def test_mic_mute_toggle_dispatches_to_client() -> None:
    client = FakeClient({})
    ctx = actions.ActionContext(
        device_key="413c:8223",
        button="phone_mute",
        press_kind="short",
    )
    await actions.ACTIONS["mic_mute_toggle"](client, ctx)
    assert ("toggle_mic_mute", {}) in client.calls


@pytest.mark.asyncio
async def test_noop_makes_no_calls() -> None:
    client = FakeClient({})
    ctx = actions.ActionContext(device_key="413c:8223", button="phone", press_kind="short")
    await actions.ACTIONS["noop"](client, ctx)
    assert client.calls == []
