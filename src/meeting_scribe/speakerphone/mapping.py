"""Sidecar JSON store for speakerphone configuration.

The full Hardware-tab document lives here and is the *only* canonical
home for:

* per-device button mappings (Phone / Teams / Phone-Mute → action)
* per-LED-state behavior (pattern + enabled)
* the default meeting profile applied by Teams-from-idle
* the long-press threshold

`meeting-toggle` reads ``default_meeting_profile`` from this file via the
same module the GUI edits, so there is no second source of truth and no
cross-file consistency dance. ETag covers the whole document.

Two write modes:

* :func:`apply_patch` — RFC 6902 JSON-Patch, single-field surgical edit,
  validated path-by-path against an allow-list. The SPA uses this for
  every dropdown / checkbox change.
* :func:`replace_full` — full-document replace, ``If-Match`` enforced
  via :func:`compute_etag`. Used by the "Reset to defaults" button.

Persistence is atomic (tmp + ``os.replace``); concurrent readers either
see the old document or the new, never a partial.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from meeting_scribe.speakerphone import catalog
from meeting_scribe.speakerphone.constants import (
    ACTION_REGISTRY,
    DAEMON_BUTTONS,
    DEFAULT_BUTTON_FEEDBACK,
    DEFAULT_LED_STATE_BEHAVIOR,
    DEFAULT_LONG_PRESS_MS,
    DEFAULT_MEETING_PROFILE,
    LED_PATTERNS,
    LED_STATES,
    MAPPING_SCHEMA_VERSION,
)
from meeting_scribe.speakerphone.labels import CANONICAL_LABELS


# Sidecar JSON path. Honors XDG_CONFIG_HOME so the customer-side
# meeting-scribe install (which sets it explicitly) stays inside its
# own config root, and so tests can redirect via tmp_path.
def _config_root() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "meeting-scribe"


def default_path() -> Path:
    return _config_root() / "speakerphone.json"


# ── Document construction ──────────────────────────────────────────────


def _default_device_block(profile: catalog.DeviceProfile) -> dict[str, Any]:
    """The canonical per-device block for a freshly-installed device.

    Phone → short-press TTS cycle, long-press interpretation toggle.
    Teams → short-press meeting-record toggle.
    Phone Mute → mic-mute toggle.
    Mute LED → state-machine driven.
    """
    return {
        "name": profile.name,
        "buttons": {
            # Each button gets both 'short' and 'long' keys (with 'noop'
            # for unbound presses) so the SPA's RFC 6902 PATCH path can
            # always target an existing leaf — 'replace' on a missing
            # key would 400. Defaults match the plan's Phone-cycle +
            # Teams-record + Mute-toggle behavior.
            "phone": {
                "short": "tts_cycle",
                "long": "interpretation_toggle",
            },
            "teams": {
                "short": "meeting_record_toggle",
                "long": "noop",
            },
            "phone_mute": {
                "short": "mic_mute_toggle",
                "long": "noop",
            },
        },
        "leds": {
            "mute_led": {"state_machine": "default"},
        },
    }


def default_document() -> dict[str, Any]:
    """Build the canonical fresh-install mapping document.

    Returns a deepcopy so callers can freely mutate without leaking
    edits back into module-level state.
    """
    return copy.deepcopy(
        {
            "version": MAPPING_SCHEMA_VERSION,
            "long_press_ms": DEFAULT_LONG_PRESS_MS,
            "devices": {
                key: _default_device_block(profile) for key, profile in catalog.CATALOG.items()
            },
            "leds": {
                "states": dict(DEFAULT_LED_STATE_BEHAVIOR),
            },
            "default_meeting_profile": dict(DEFAULT_MEETING_PROFILE),
            "button_feedback": dict(DEFAULT_BUTTON_FEEDBACK),
        },
    )


def load(path: Path | None = None) -> dict[str, Any]:
    """Read the sidecar JSON, or return defaults if missing.

    Missing fields are filled from :func:`default_document` so the GUI
    can render every section even on a fresh install. Schema-version
    mismatch raises ``ValueError`` — migrations live elsewhere.
    """
    path = path or default_path()
    if not path.exists():
        return default_document()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be an object")
    version = raw.get("version")
    if version != MAPPING_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema version {version!r} != expected {MAPPING_SCHEMA_VERSION}",
        )
    return _merge_with_defaults(raw)


def _merge_with_defaults(doc: dict[str, Any]) -> dict[str, Any]:
    """Fill gaps in a loaded doc with default values.

    Keeps user edits; only adds keys the loaded document lacks. Does
    not deep-merge inside devices to avoid silently introducing fresh
    buttons the user expected to be absent.
    """
    defaults = default_document()
    out: dict[str, Any] = copy.deepcopy(defaults)
    for top_key, value in doc.items():
        if top_key == "devices" and isinstance(value, dict):
            out_devices = dict(defaults["devices"])
            out_devices.update(value)
            out["devices"] = out_devices
        else:
            out[top_key] = value
    return out


def save(doc: Mapping[str, Any], path: Path | None = None) -> None:
    """Atomically write the document to disk.

    Creates the parent directory if absent. Throws if the document
    fails :func:`validate`.
    """
    validate(doc)
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    body = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def compute_etag(doc: Mapping[str, Any]) -> str:
    """Stable ETag for the document. Used by PUT's ``If-Match``."""
    body = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:32]


# ── Validation ──────────────────────────────────────────────────────────


_VID_PID_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{4}$")


class MappingValidationError(ValueError):
    """Raised when a document or patch op violates the schema."""


def validate(doc: Mapping[str, Any]) -> None:
    """Validate a full mapping document. Raise on the first problem.

    Checks: schema version, long-press range, devices contain only
    daemon-owned buttons mapped to known actions, LED states are
    recognised + use known patterns, default meeting profile fields
    are well-formed.
    """
    if doc.get("version") != MAPPING_SCHEMA_VERSION:
        raise MappingValidationError(
            f"version must be {MAPPING_SCHEMA_VERSION}",
        )

    lp = doc.get("long_press_ms")
    if not isinstance(lp, int) or lp < 200 or lp > 5000:
        raise MappingValidationError(
            "long_press_ms must be an int in [200, 5000]",
        )

    devices = doc.get("devices")
    if not isinstance(devices, dict):
        raise MappingValidationError("devices must be an object")
    for key, dev in devices.items():
        if not _VID_PID_RE.match(key):
            raise MappingValidationError(f"device key {key!r} is not vid:pid")
        _validate_device(key, dev)

    led_states = doc.get("leds", {}).get("states", {})
    if not isinstance(led_states, dict):
        raise MappingValidationError("leds.states must be an object")
    for state_name, behavior in led_states.items():
        if state_name not in LED_STATES:
            raise MappingValidationError(
                f"unknown LED state {state_name!r} (known: {sorted(LED_STATES)})",
            )
        if not isinstance(behavior, dict):
            raise MappingValidationError(
                f"leds.states.{state_name} must be an object",
            )
        if not isinstance(behavior.get("enabled"), bool):
            raise MappingValidationError(
                f"leds.states.{state_name}.enabled must be bool",
            )
        pattern = behavior.get("pattern")
        if pattern not in LED_PATTERNS:
            raise MappingValidationError(
                f"leds.states.{state_name}.pattern={pattern!r} not in {sorted(LED_PATTERNS)}",
            )

    profile = doc.get("default_meeting_profile")
    if not isinstance(profile, dict):
        raise MappingValidationError("default_meeting_profile must be an object")
    _validate_profile(profile)

    feedback = doc.get("button_feedback")
    if not isinstance(feedback, dict):
        raise MappingValidationError("button_feedback must be an object")
    _validate_button_feedback(feedback)


def _validate_device(key: str, dev: Any) -> None:
    if not isinstance(dev, dict):
        raise MappingValidationError(f"devices[{key}] must be an object")
    buttons = dev.get("buttons", {})
    if not isinstance(buttons, dict):
        raise MappingValidationError(f"devices[{key}].buttons must be an object")
    for btn_name, binding in buttons.items():
        if btn_name not in DAEMON_BUTTONS:
            raise MappingValidationError(
                f"button {btn_name!r} is not daemon-owned (allowed: {sorted(DAEMON_BUTTONS)})",
            )
        if not isinstance(binding, dict):
            raise MappingValidationError(
                f"devices[{key}].buttons[{btn_name}] must be an object",
            )
        for press_kind, action in binding.items():
            if press_kind not in {"short", "long"}:
                raise MappingValidationError(
                    f"devices[{key}].buttons[{btn_name}]: "
                    f"press kind {press_kind!r} must be 'short' or 'long'",
                )
            if action not in ACTION_REGISTRY:
                raise MappingValidationError(
                    f"devices[{key}].buttons[{btn_name}].{press_kind}: "
                    f"action {action!r} not in registry "
                    f"({sorted(ACTION_REGISTRY)})",
                )


def _validate_profile(profile: Mapping[str, Any]) -> None:
    languages = profile.get("languages")
    if not isinstance(languages, list) or not (1 <= len(languages) <= 2):
        raise MappingValidationError(
            "default_meeting_profile.languages must be a 1- or 2-element list",
        )
    if any(not isinstance(code, str) or len(code) != 2 for code in languages):
        raise MappingValidationError(
            "default_meeting_profile.languages entries must be 2-letter codes",
        )
    if len(set(languages)) != len(languages):
        raise MappingValidationError(
            "default_meeting_profile.languages must be unique",
        )
    if not isinstance(profile.get("interpretation_enabled"), bool):
        raise MappingValidationError(
            "default_meeting_profile.interpretation_enabled must be bool",
        )
    room = profile.get("room_tts_language")
    if not isinstance(room, str) or room not in (*languages, "all"):
        raise MappingValidationError(
            "default_meeting_profile.room_tts_language must be 'all' or "
            "one of the profile languages",
        )
    admin = profile.get("admin_tts_language")
    if admin is not None and admin not in languages:
        raise MappingValidationError(
            "default_meeting_profile.admin_tts_language, if set, must be "
            "one of the profile languages",
        )


def _tts_native_codes() -> frozenset[str]:
    """Cached lookup of language codes Qwen3-TTS can synthesize.

    Sourced from :mod:`meeting_scribe.languages.LANGUAGE_REGISTRY` —
    the same registry the rest of meeting-scribe uses for direction
    selection. Filtered by ``tts_native=True`` because the
    button-feedback path runs through the TTS backend; an ASR-only
    code would have no synthesis path.
    """
    from meeting_scribe.languages import LANGUAGE_REGISTRY

    return frozenset(code for code, lang in LANGUAGE_REGISTRY.items() if lang.tts_native)


def _validate_button_feedback(feedback: Mapping[str, Any]) -> None:
    """Validate the ``button_feedback`` block.

    Schema:
    * ``enabled``: bool
    * ``language``: str in the TTS-native subset of LANGUAGE_REGISTRY
    * ``overrides``: dict mapping label_id (in CANONICAL_LABELS) to a
      dict of {lang_code: text}; lang_code must be TTS-native; text
      must be a non-empty string.

    Catches operator typos in label_id (which would silently never play)
    and rejects override entries in languages the TTS backend can't
    synthesize (which would surface as a confusing "fell back to
    English" at playback time).
    """
    if not isinstance(feedback.get("enabled"), bool):
        raise MappingValidationError("button_feedback.enabled must be bool")

    tts_langs = _tts_native_codes()

    language = feedback.get("language")
    if not isinstance(language, str) or language not in tts_langs:
        raise MappingValidationError(
            f"button_feedback.language={language!r} not in TTS-native set {sorted(tts_langs)}",
        )

    overrides = feedback.get("overrides")
    if not isinstance(overrides, dict):
        raise MappingValidationError("button_feedback.overrides must be an object")

    for label_id, per_label in overrides.items():
        if label_id not in CANONICAL_LABELS:
            raise MappingValidationError(
                f"button_feedback.overrides: unknown label_id {label_id!r} "
                f"(allowed: {sorted(CANONICAL_LABELS)})",
            )
        if not isinstance(per_label, dict):
            raise MappingValidationError(
                f"button_feedback.overrides[{label_id}] must be an object",
            )
        for lang_code, text in per_label.items():
            if lang_code not in tts_langs:
                raise MappingValidationError(
                    f"button_feedback.overrides[{label_id}]: "
                    f"language {lang_code!r} not in TTS-native set",
                )
            if not isinstance(text, str) or not text.strip():
                raise MappingValidationError(
                    f"button_feedback.overrides[{label_id}][{lang_code}] "
                    "must be a non-empty string",
                )


# ── Patch application (RFC 6902 subset) ─────────────────────────────────

# JSON Pointer paths the SPA is allowed to PATCH. The set is closed; an
# op against any other path is a 400. Each entry is a regex that must
# match the full path string.
_ALLOWED_PATCH_PATHS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^/long_press_ms$"),
    re.compile(
        r"^/devices/[0-9a-f]{4}:[0-9a-f]{4}/buttons/"
        r"(phone|teams|phone_mute)/(short|long)$",
    ),
    re.compile(r"^/leds/states/[a-z_]+/(enabled|pattern)$"),
    re.compile(
        r"^/default_meeting_profile/(name|interpretation_enabled|"
        r"room_tts_language|admin_tts_language|title_template)$"
    ),
    re.compile(r"^/default_meeting_profile/languages$"),
    # button_feedback subtree (per the plan)
    re.compile(r"^/button_feedback/(enabled|language)$"),
    # First-override-for-this-label: SPA emits this to materialize the
    # parent object atomically (strict RFC 6902 `add` requires the
    # parent to exist when writing a leaf).
    re.compile(r"^/button_feedback/overrides/[a-z0-9_]+$"),
    # Subsequent per-language override: parent already exists.
    re.compile(r"^/button_feedback/overrides/[a-z0-9_]+/[a-z]{2}$"),
)


def _path_allowed(path: str) -> bool:
    return any(rx.match(path) for rx in _ALLOWED_PATCH_PATHS)


def _resolve_pointer(doc: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    """Walk a JSON Pointer to the parent of the leaf and return (parent, key).

    Pointer syntax follows RFC 6901: ``/a/b/c`` → doc["a"]["b"], leaf
    "c". Tilde-escapes are not used by our allowed paths so we don't
    decode them.
    """
    if not path.startswith("/"):
        raise MappingValidationError(f"path {path!r} must start with /")
    parts = path.split("/")[1:]
    node: Any = doc
    for segment in parts[:-1]:
        if not isinstance(node, dict) or segment not in node:
            raise MappingValidationError(f"path {path!r}: missing segment {segment!r}")
        node = node[segment]
    if not isinstance(node, dict):
        raise MappingValidationError(f"path {path!r}: parent is not an object")
    return node, parts[-1]


def apply_patch(
    doc: dict[str, Any],
    ops: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply a JSON-Patch op list.

    Supported ops (strict RFC 6902 — no auto-create of intermediates):

    * ``replace`` — leaf must already exist.
    * ``add`` — used by the SPA's first-override-for-this-label save:
      ``add /button_feedback/overrides/<label_id>`` writes a new dict
      where one didn't exist before, materializing the parent
      atomically. The SPA is responsible for choosing the parent-path
      shape vs the leaf-path shape based on the current state. A
      naive ``add`` to a leaf path whose parent is missing raises
      ``MappingValidationError`` — by design, so we don't silently
      drift into auto-create semantics.

    Validates each op against the allow-list, then validates the
    post-application document. Returns a new dict; never mutates the
    input.
    """
    if not isinstance(ops, list):
        raise MappingValidationError("PATCH body must be a list of ops")
    out = copy.deepcopy(doc)
    for op in ops:
        if not isinstance(op, dict):
            raise MappingValidationError("each op must be an object")
        op_kind = op.get("op")
        if op_kind not in ("replace", "add"):
            raise MappingValidationError(
                f"op {op_kind!r} not supported; only 'replace' and 'add'",
            )
        path = op.get("path")
        if not isinstance(path, str):
            raise MappingValidationError("op.path must be a string")
        if not _path_allowed(path):
            raise MappingValidationError(
                f"PATCH path {path!r} not in the allow-list",
            )
        parent, key = _resolve_pointer(out, path)
        # Strict RFC 6902: replace requires an existing leaf; add allows new keys.
        if op_kind == "replace" and key not in parent:
            raise MappingValidationError(
                f"PATCH replace on {path!r}: key {key!r} does not exist (use op=add to create)",
            )
        parent[key] = op.get("value")
    validate(out)
    return out


def replace_full(
    doc: dict[str, Any],
    new_doc: Mapping[str, Any],
    etag: str,
) -> dict[str, Any]:
    """Full-document replace with ``If-Match`` enforcement.

    Returns the new document if the etag matches; raises
    :class:`StaleEtagError` otherwise.
    """
    expected = compute_etag(doc)
    if etag != expected:
        raise StaleEtagError(expected_etag=expected, actual_etag=etag)
    new_doc = dict(new_doc)
    validate(new_doc)
    return new_doc


class StaleEtagError(MappingValidationError):
    """Raised by :func:`replace_full` when the If-Match etag is stale."""

    def __init__(self, *, expected_etag: str, actual_etag: str) -> None:
        super().__init__(
            f"If-Match etag {actual_etag!r} is stale; current etag is {expected_etag!r}",
        )
        self.expected_etag = expected_etag
        self.actual_etag = actual_etag
