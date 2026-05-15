"""Tests for PATCH/PUT semantics on the speakerphone mapping document.

Regression target: per-field PATCH must never clobber sibling nested
fields, and full-document PUT must enforce ETag.
"""

from __future__ import annotations

import pytest

from meeting_scribe.speakerphone import mapping


def _doc() -> dict:
    return mapping.default_document()


def test_patch_phone_short_does_not_clobber_phone_long() -> None:
    doc = _doc()
    original_long = doc["devices"]["413c:8223"]["buttons"]["phone"]["long"]
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "replace",
                "path": "/devices/413c:8223/buttons/phone/short",
                "value": "noop",
            },
        ],
    )
    assert new_doc["devices"]["413c:8223"]["buttons"]["phone"]["short"] == "noop"
    # Sibling field preserved.
    assert new_doc["devices"]["413c:8223"]["buttons"]["phone"]["long"] == original_long


def test_patch_phone_does_not_clobber_teams_or_phone_mute() -> None:
    doc = _doc()
    teams_before = doc["devices"]["413c:8223"]["buttons"]["teams"]
    mute_before = doc["devices"]["413c:8223"]["buttons"]["phone_mute"]
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "replace",
                "path": "/devices/413c:8223/buttons/phone/short",
                "value": "noop",
            },
        ],
    )
    assert new_doc["devices"]["413c:8223"]["buttons"]["teams"] == teams_before
    assert new_doc["devices"]["413c:8223"]["buttons"]["phone_mute"] == mute_before


def test_patch_does_not_mutate_input_document() -> None:
    doc = _doc()
    snapshot = mapping.compute_etag(doc)
    mapping.apply_patch(
        doc,
        [
            {
                "op": "replace",
                "path": "/devices/413c:8223/buttons/phone/short",
                "value": "noop",
            },
        ],
    )
    # Input must be unchanged.
    assert mapping.compute_etag(doc) == snapshot


def test_patch_can_edit_led_state_pattern() -> None:
    doc = _doc()
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "replace",
                "path": "/leds/states/backend_unready/pattern",
                "value": "very_fast_blink",
            },
        ],
    )
    assert new_doc["leds"]["states"]["backend_unready"]["pattern"] == "very_fast_blink"
    # Other LED states untouched.
    assert (
        new_doc["leds"]["states"]["recording"]["pattern"]
        == doc["leds"]["states"]["recording"]["pattern"]
    )


def test_patch_can_edit_default_profile_languages() -> None:
    doc = _doc()
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "replace",
                "path": "/default_meeting_profile/languages",
                "value": ["en", "fr"],
            },
            # Keep room_tts_language consistent so post-patch validation passes.
            {
                "op": "replace",
                "path": "/default_meeting_profile/room_tts_language",
                "value": "en",
            },
        ],
    )
    assert new_doc["default_meeting_profile"]["languages"] == ["en", "fr"]


def test_patch_rejects_unknown_path() -> None:
    doc = _doc()
    with pytest.raises(mapping.MappingValidationError) as excinfo:
        mapping.apply_patch(
            doc,
            [
                {
                    "op": "replace",
                    "path": "/devices/413c:8223/buttons/phone/triple_click",
                    "value": "noop",
                },
            ],
        )
    # The error should echo the offending path so the SPA can show it.
    assert "triple_click" in str(excinfo.value) or "allow-list" in str(excinfo.value)


def test_patch_rejects_value_outside_action_registry() -> None:
    doc = _doc()
    with pytest.raises(mapping.MappingValidationError):
        mapping.apply_patch(
            doc,
            [
                {
                    "op": "replace",
                    "path": "/devices/413c:8223/buttons/phone/short",
                    "value": "summon_the_kraken",
                },
            ],
        )


def test_replace_full_with_correct_etag_succeeds() -> None:
    doc = _doc()
    etag = mapping.compute_etag(doc)
    new_doc = mapping.default_document()
    new_doc["long_press_ms"] = 1500
    replaced = mapping.replace_full(doc, new_doc, etag=etag)
    assert replaced["long_press_ms"] == 1500


def test_replace_full_with_stale_etag_raises() -> None:
    doc = _doc()
    with pytest.raises(mapping.StaleEtagError):
        mapping.replace_full(doc, mapping.default_document(), etag="deadbeef")


# ── button_feedback patch path ──────────────────────────────────────


def test_patch_replace_button_feedback_enabled_works() -> None:
    doc = _doc()
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "replace",
                "path": "/button_feedback/enabled",
                "value": False,
            },
        ],
    )
    assert new_doc["button_feedback"]["enabled"] is False
    # Sibling untouched.
    assert new_doc["button_feedback"]["language"] == "en"


def test_patch_replace_button_feedback_language_works() -> None:
    doc = _doc()
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "replace",
                "path": "/button_feedback/language",
                "value": "ja",
            },
        ],
    )
    assert new_doc["button_feedback"]["language"] == "ja"


def test_patch_first_override_for_label_via_parent_path() -> None:
    """SPA's first-save path: ``add /…/overrides/<label>`` with a
    ``{<lang>: <text>}`` value materializes the parent atomically.

    This is the load-bearing test for the codex P1 fix — RFC 6902
    strict means we MUST use the parent path for the first override
    per label, not the leaf path.
    """
    doc = _doc()
    assert doc["button_feedback"]["overrides"] == {}
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "add",
                "path": "/button_feedback/overrides/volume_up",
                "value": {"en": "Loud"},
            },
        ],
    )
    assert new_doc["button_feedback"]["overrides"] == {
        "volume_up": {"en": "Loud"},
    }


def test_patch_subsequent_override_via_leaf_path() -> None:
    """After the parent exists, add a new lang via leaf-path add."""
    doc = _doc()
    doc["button_feedback"]["overrides"] = {"volume_up": {"en": "Loud"}}
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "add",
                "path": "/button_feedback/overrides/volume_up/ja",
                "value": "うるさい",
            },
        ],
    )
    assert new_doc["button_feedback"]["overrides"]["volume_up"] == {
        "en": "Loud",
        "ja": "うるさい",
    }


def test_patch_replace_existing_override_leaf() -> None:
    doc = _doc()
    doc["button_feedback"]["overrides"] = {"volume_up": {"en": "Loud"}}
    new_doc = mapping.apply_patch(
        doc,
        [
            {
                "op": "replace",
                "path": "/button_feedback/overrides/volume_up/en",
                "value": "Even louder",
            },
        ],
    )
    assert new_doc["button_feedback"]["overrides"]["volume_up"]["en"] == "Even louder"


def test_patch_leaf_add_fails_when_parent_label_missing() -> None:
    """Strict RFC 6902 — naively trying to add a leaf without parent fails.

    The SPA is expected to use the parent-path shape for first-time
    saves. This test guards against accidentally implementing
    auto-create semantics (which would mask a class of UI bugs).
    """
    doc = _doc()
    assert "volume_up" not in doc["button_feedback"]["overrides"]
    with pytest.raises(mapping.MappingValidationError) as excinfo:
        mapping.apply_patch(
            doc,
            [
                {
                    "op": "add",
                    "path": "/button_feedback/overrides/volume_up/en",
                    "value": "Loud",
                },
            ],
        )
    msg = str(excinfo.value)
    # Error must clearly identify the missing parent so the SPA
    # operator can correct the patch shape.
    assert "volume_up" in msg


def test_patch_op_replace_fails_when_leaf_does_not_exist() -> None:
    """Strict RFC 6902 — ``replace`` requires an existing leaf."""
    doc = _doc()
    assert "volume_up" not in doc["button_feedback"]["overrides"]
    with pytest.raises(mapping.MappingValidationError):
        mapping.apply_patch(
            doc,
            [
                {
                    "op": "replace",
                    "path": "/button_feedback/overrides/volume_up",
                    "value": {"en": "Loud"},
                },
            ],
        )


def test_patch_rejects_remove_op() -> None:
    """`add` is now allowed (first-override-for-label case), but `remove` is still rejected."""
    doc = _doc()
    with pytest.raises(mapping.MappingValidationError):
        mapping.apply_patch(
            doc,
            [
                {
                    "op": "remove",
                    "path": "/button_feedback/enabled",
                },
            ],
        )


def test_patch_unknown_label_in_override_value_rejected_by_validate() -> None:
    """Validate runs after the patch — an unknown label_id is rejected."""
    doc = _doc()
    with pytest.raises(mapping.MappingValidationError):
        mapping.apply_patch(
            doc,
            [
                {
                    "op": "add",
                    "path": "/button_feedback/overrides/fly_to_mars",
                    "value": {"en": "Launch!"},
                },
            ],
        )
