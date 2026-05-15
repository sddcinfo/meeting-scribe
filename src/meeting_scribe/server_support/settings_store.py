"""Persisted admin-UI settings overrides + regdomain/timezone resolution.

The admin UI's settings endpoint writes a small JSON file at
``~/.config/meeting-scribe/settings.json``; reads are mtime-cached so
the ``/api/status`` poll loop doesn't re-decode JSON on every hit.

The ``_effective_*`` helpers resolve the live value for regdomain and
timezone using a fixed precedence:

  1. ``state.config.X`` (env var or programmatic override at startup)
  2. The persisted override file
  3. A built-in fallback (``"JP"`` for regdomain, ``""`` for timezone)

Pulled out of ``server.py`` so the to-be-extracted admin route module
and the WiFi subsystem can both reach these without importing the
server module itself.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from meeting_scribe.runtime import state

logger = logging.getLogger(__name__)

# Persisted overrides — survives process restart. The admin UI's
# settings endpoint writes here. Env vars still override.
SETTINGS_OVERRIDE_FILE = Path.home() / ".config" / "meeting-scribe" / "settings.json"

# Default regdomain used when config hasn't been initialized yet (e.g.
# during unit tests that don't run the lifespan startup).
_DEFAULT_REGDOMAIN = "JP"

_DEFAULT_TIMEZONE = ""  # empty = use the server's local time

_settings_cache: dict | None = None
_settings_cache_mtime: float = 0.0
_legacy_migration_attempted: bool = False


def _migrate_legacy_settings(current: dict) -> dict:
    """One-shot legacy-key migration.

    Today's only legacy key: ``audio_meeting_sink_node`` (pre-routing
    card). Copy the value to the new key only if the new key is empty,
    then rename the legacy key to ``audio_admin_tts_sink_node_legacy_backup``
    so a determined operator can hand-restore on rollback. The backup
    key is never read by code.

    Returns the migrated dict (caller is responsible for persisting it).
    Idempotent: a second call is a no-op once the legacy key is gone.
    """
    from meeting_scribe.audio.audio_routing import (
        LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE,
        LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE_BACKUP,
        SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE,
    )

    if LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE not in current:
        return current

    legacy_value = current.pop(LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE)

    # New key wins; legacy only fills an empty/missing new key.
    if not current.get(SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE) and legacy_value:
        current[SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE] = legacy_value

    # Write-once backup. If a backup already exists from a prior
    # migration, leave it alone — first migration is the authoritative
    # rollback breadcrumb.
    if legacy_value and LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE_BACKUP not in current:
        current[LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE_BACKUP] = legacy_value

    logger.info(
        "settings_store: migrated %s -> %s (backup retained as %s)",
        LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE,
        SETTINGS_AUDIO_ADMIN_TTS_SINK_NODE,
        LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE_BACKUP,
    )
    return current


def _maybe_migrate_and_persist(current: dict) -> dict:
    """If the loaded dict contains legacy keys, migrate + write back.

    Runs at most once per process via ``_legacy_migration_attempted``;
    the cache layer also keeps it from re-running on every read.
    """
    global _legacy_migration_attempted
    from meeting_scribe.audio.audio_routing import (
        LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE,
    )

    if _legacy_migration_attempted:
        return current
    _legacy_migration_attempted = True

    if LEGACY_SETTINGS_AUDIO_MEETING_SINK_NODE not in current:
        return current

    migrated = _migrate_legacy_settings(dict(current))

    # Persist the migrated form so subsequent reads see the new shape
    # and the next process boot doesn't re-migrate from a stale file.
    import json as _json

    try:
        SETTINGS_OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_OVERRIDE_FILE.with_suffix(
            SETTINGS_OVERRIDE_FILE.suffix + f".tmp.{os.getpid()}"
        )
        tmp.write_text(_json.dumps(migrated, indent=2) + "\n")
        tmp.replace(SETTINGS_OVERRIDE_FILE)
    except OSError:
        logger.exception("settings_store: migration write failed; keeping in-memory copy")
    return migrated


def _load_settings_override() -> dict:
    """Read persisted admin-UI settings overrides. Best-effort.

    Cached by file mtime — safe for the ``/api/status`` hot path
    (~3 s poll). Invalidated on write via ``_save_settings_override``.
    """
    global _settings_cache, _settings_cache_mtime
    import json as _json

    if not SETTINGS_OVERRIDE_FILE.exists():
        _settings_cache = {}
        return {}
    try:
        mtime = SETTINGS_OVERRIDE_FILE.stat().st_mtime
        if _settings_cache is not None and mtime == _settings_cache_mtime:
            return _settings_cache
        data = _json.loads(SETTINGS_OVERRIDE_FILE.read_text())
        result = data if isinstance(data, dict) else {}
        result = _maybe_migrate_and_persist(result)
        _settings_cache = result
        # Re-stat in case the migration write bumped the mtime.
        _settings_cache_mtime = SETTINGS_OVERRIDE_FILE.stat().st_mtime
        return result
    except OSError, _json.JSONDecodeError:
        return _settings_cache or {}


def _save_settings_override(updates: dict) -> None:
    """Merge ``updates`` into the persisted settings override file."""
    global _settings_cache, _settings_cache_mtime
    import json as _json

    SETTINGS_OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    current = _load_settings_override()
    current.update(updates)
    tmp = SETTINGS_OVERRIDE_FILE.with_suffix(SETTINGS_OVERRIDE_FILE.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(_json.dumps(current, indent=2) + "\n")
    tmp.replace(SETTINGS_OVERRIDE_FILE)
    _settings_cache = current
    _settings_cache_mtime = SETTINGS_OVERRIDE_FILE.stat().st_mtime


def _effective_regdomain() -> str:
    """Return the regulatory domain to enforce.

    Priority:
      1. Live ``state.config.wifi_regdomain`` (env var, profile,
         persisted settings override).
      2. Persisted override file (read directly if config isn't
         initialized).
      3. ``_DEFAULT_REGDOMAIN`` fallback.

    The returned code is upper-cased to match what ``iw`` expects.
    """
    from_config = getattr(state.config, "wifi_regdomain", None) if state.config else None
    if isinstance(from_config, str) and from_config.strip():
        return from_config.strip().upper()

    override = _load_settings_override().get("wifi_regdomain")
    if isinstance(override, str) and override.strip():
        return override.strip().upper()
    return _DEFAULT_REGDOMAIN


def _regdomain_modprobe_path(country: str) -> Path:
    """Canonical modprobe conf path for a given 2-letter country code."""
    safe = country.strip().upper() or _DEFAULT_REGDOMAIN
    return Path("/etc/modprobe.d") / f"cfg80211-{safe.lower()}.conf"


# Curated list of 2-letter ISO 3166-1 country codes the WiFi card
# supports. The MT7925e mt7925e driver + cfg80211 regdb cover every ISO
# country, but surfacing all ~250 in a dropdown is unusable. This list
# is the "useful demo deployment" subset; add a country via PR or the
# ``SCRIBE_WIFI_REGDOMAIN`` env var (the env var accepts any valid code).
_WIFI_REGDOMAIN_OPTIONS: tuple[tuple[str, str], ...] = (
    ("JP", "Japan"),
    ("US", "United States"),
    ("CA", "Canada"),
    ("GB", "United Kingdom"),
    ("IE", "Ireland"),
    ("DE", "Germany"),
    ("FR", "France"),
    ("IT", "Italy"),
    ("ES", "Spain"),
    ("PT", "Portugal"),
    ("NL", "Netherlands"),
    ("BE", "Belgium"),
    ("LU", "Luxembourg"),
    ("CH", "Switzerland"),
    ("AT", "Austria"),
    ("SE", "Sweden"),
    ("NO", "Norway"),
    ("FI", "Finland"),
    ("DK", "Denmark"),
    ("IS", "Iceland"),
    ("PL", "Poland"),
    ("CZ", "Czechia"),
    ("SK", "Slovakia"),
    ("HU", "Hungary"),
    ("GR", "Greece"),
    ("RO", "Romania"),
    ("BG", "Bulgaria"),
    ("EE", "Estonia"),
    ("LV", "Latvia"),
    ("LT", "Lithuania"),
    ("AU", "Australia"),
    ("NZ", "New Zealand"),
    ("SG", "Singapore"),
    ("HK", "Hong Kong"),
    ("TW", "Taiwan"),
    ("KR", "South Korea"),
    ("CN", "China"),
    ("IN", "India"),
    ("TH", "Thailand"),
    ("MY", "Malaysia"),
    ("ID", "Indonesia"),
    ("PH", "Philippines"),
    ("VN", "Vietnam"),
    ("AE", "United Arab Emirates"),
    ("SA", "Saudi Arabia"),
    ("IL", "Israel"),
    ("TR", "Turkey"),
    ("ZA", "South Africa"),
    ("BR", "Brazil"),
    ("MX", "Mexico"),
    ("AR", "Argentina"),
    ("CL", "Chile"),
    ("CO", "Colombia"),
)


def _is_valid_regdomain(code: str) -> bool:
    """Return True if ``code`` is in the curated supported-country list."""
    return code.upper() in {c for c, _ in _WIFI_REGDOMAIN_OPTIONS}


def _effective_timezone() -> str:
    """Return the display timezone to use (IANA name) or '' for local.

    Priority mirrors ``_effective_regdomain``:
      1. ``state.config.timezone``
      2. persisted override file
      3. default (empty string — server local time).
    """
    from_config = getattr(state.config, "timezone", None) if state.config else None
    if isinstance(from_config, str) and from_config.strip():
        return from_config.strip()

    override = _load_settings_override().get("timezone")
    if isinstance(override, str) and override.strip():
        return override.strip()
    return _DEFAULT_TIMEZONE


def _effective_tts_voice_mode() -> str:
    """Return the server-wide default TTS voice mode.

    "studio" — Qwen3-TTS named speaker per language (fast, commercial-safe).
    "cloned" — clone each meeting participant's voice (slower, personal).
    Individual listeners can still override per-session via WS
    ``set_voice``.
    """
    override = _load_settings_override().get("tts_voice_mode")
    if isinstance(override, str) and override in ("studio", "cloned"):
        return override
    return "studio"


def _effective_interpretation_enabled() -> bool:
    """Return whether bidirectional room interpretation is enabled.

    Default-off by design; the room sink keeps legacy simultaneous
    behavior until the operator explicitly enables this persisted flag.
    """
    override = _load_settings_override().get("interpretation_enabled")
    return override if isinstance(override, bool) else False


def _effective_interpretation_last_room_tts_language() -> str:
    """Return the last active room TTS direction (en/ja/.../"all").

    Persisted server-side under ``interpretation_last_room_tts_language``
    in ``settings.json`` so a long-press-off followed by a long-press-on
    (possibly across a meeting-scribe restart) restores exactly the
    direction the user had selected. Default is ``"all"`` so a brand-new
    install still has a sane fallback when no direction has been picked
    yet.
    """
    override = _load_settings_override().get("interpretation_last_room_tts_language")
    if isinstance(override, str) and override:
        return override
    return "all"


def _effective_interpretation_pause_flush_ms() -> int:
    override = _load_settings_override().get("interpretation_pause_flush_ms")
    if isinstance(override, int) and 100 <= override <= 30000:
        return override
    return 2500


def _effective_interpretation_idle_drain_ms() -> int:
    override = _load_settings_override().get("interpretation_idle_drain_ms")
    if isinstance(override, int) and 100 <= override <= 60000:
        return override
    return 5000


# ── WAN (upstream WiFi / gateway) settings ──────────────────────
#
# v1 schema. See docs/plans/wifi-wan-gateway.md for the full rationale.
# Storage is split across these top-level keys in ``settings.json``:
#
#   wan_egress_mode           "block" | "gateway"   (default "block")
#   wan_active_profile_id     uuid4 str | None
#   wan_wired_profile_name    str | None       (NM connection name we claimed on enP7s7)
#   wan_wired_metric_original int | None       (prior route-metric so a future restore is possible)
#   wan_profiles              list[WanProfile]
#   wan_captive_state         { wlan_sta: { connectivity, portal_url, probed_at } }
#
# A WanProfile is a dict:
#   { id: uuid4-str, ssid: str, bssid: str|None, psk_ref: str, regdomain: str|None,
#     last_seen: iso8601|None }

SETTINGS_WAN_EGRESS_MODE = "wan_egress_mode"
SETTINGS_WAN_EGRESS_MODE_SOURCE = "wan_egress_mode_source"
SETTINGS_WAN_ACTIVE_PROFILE_ID = "wan_active_profile_id"
SETTINGS_WAN_WIRED_PROFILE_NAME = "wan_wired_profile_name"
SETTINGS_WAN_WIRED_METRIC_ORIGINAL = "wan_wired_metric_original"
SETTINGS_WAN_PROFILES = "wan_profiles"
SETTINGS_WAN_CAPTIVE_STATE = "wan_captive_state"

# ``block``    — zero-egress hotspot (meeting-privacy default)
# ``gateway``  — unconditional pass-through (admin SSID, no per-client gating)
# ``captive``  — per-IP allowlist: only signed-in admins reach WAN; everything
#                else gets REDIRECT'd to the local captive sub-app (Phase H)
_VALID_WAN_EGRESS_MODES: frozenset[str] = frozenset({"block", "gateway", "captive"})

# Source of the persisted egress-mode value:
#   "default" — never explicitly set (or set by the Phase H.6 migration ladder)
#   "operator" — explicitly chosen via CLI / REST / UI / settings-page save
# Allows the AP-up migration to flip block→captive ONLY when the operator
# hasn't already made a choice. Once source becomes "operator" it stays
# sticky so future defaults migrations can't silently overwrite a deliberate
# pick.
_VALID_WAN_EGRESS_MODE_SOURCES: frozenset[str] = frozenset({"default", "operator"})

# Matches RFC 4122 lowercase canonical uuid4 form (the form ``uuid.uuid4()``
# emits via ``str()``). Profile ids are user-visible and must never be
# truncated in display (per feedback_no_id_truncation).
_UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

# colon-separated 6-octet hex MAC (e.g. AA:BB:CC:DD:EE:FF). Upper- or lower-case.
_BSSID_RE = re.compile(r"^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$")

# psk_ref keys in the age store are SCREAMING_SNAKE_CASE (matches the
# existing convention: GH_TOKEN, YUNOMOTOCHO_PSK, CLOUDFLARE_API_TOKEN).
_PSK_REF_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

# NM ``wifi.band`` values; ``auto`` is our internal sentinel meaning
# "omit the property so the supplicant picks across both bands".
_VALID_WAN_BANDS: frozenset[str] = frozenset({"auto", "a", "bg"})

# IEEE 802.11 SSID — 1..32 octets. We additionally require valid UTF-8
# str (which Python str already is) and disallow embedded NULs.
_SSID_MAX_OCTETS = 32


def _is_valid_egress_mode(value: object) -> bool:
    return isinstance(value, str) and value in _VALID_WAN_EGRESS_MODES


def _is_valid_egress_mode_source(value: object) -> bool:
    return isinstance(value, str) and value in _VALID_WAN_EGRESS_MODE_SOURCES


def _is_valid_uuid4_str(value: object) -> bool:
    return isinstance(value, str) and bool(_UUID4_RE.match(value))


def _is_valid_psk_ref(value: object) -> bool:
    return isinstance(value, str) and bool(_PSK_REF_RE.match(value))


def _is_valid_bssid(value: object) -> bool:
    return isinstance(value, str) and bool(_BSSID_RE.match(value))


def _is_valid_wan_band(value: object) -> bool:
    return isinstance(value, str) and value in _VALID_WAN_BANDS


def _is_valid_ssid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if "\x00" in value:
        return False
    octets = value.encode("utf-8")
    return 1 <= len(octets) <= _SSID_MAX_OCTETS


def _is_valid_wan_profile(value: object) -> bool:
    """Return True iff ``value`` is a well-formed WanProfile dict.

    ``psk_ref`` is **optional**. A profile with ``psk_ref=None`` (or the
    field absent) is treated as an OPEN network (no auth, no PSK
    lookup, no ``wifi-sec.*`` on the NM profile). The CLI/REST layer is
    responsible for surfacing this explicitly so it can never become an
    accidental fallback for forgotten PSK refs — only operators who saw
    the OPEN security badge in the scan should be able to create one.
    """
    if not isinstance(value, dict):
        return False
    if not _is_valid_uuid4_str(value.get("id")):
        return False
    if not _is_valid_ssid(value.get("ssid")):
        return False
    psk_ref = value.get("psk_ref")
    if psk_ref is not None and not _is_valid_psk_ref(psk_ref):
        return False
    bssid = value.get("bssid")
    if bssid is not None and not _is_valid_bssid(bssid):
        return False
    band = value.get("band")
    # Older profiles may omit ``band`` entirely; treat that as auto.
    if band is not None and not _is_valid_wan_band(band):
        return False
    regdomain = value.get("regdomain")
    if regdomain is not None and not (
        isinstance(regdomain, str) and _is_valid_regdomain(regdomain)
    ):
        return False
    last_seen = value.get("last_seen")
    return not (last_seen is not None and not isinstance(last_seen, str))


def _effective_wan_egress_mode() -> str:
    """Return the active WAN egress mode. Defaults to ``"block"``.

    ``"block"`` reproduces today's zero-egress hotspot posture (meeting
    privacy). ``"gateway"`` permits AP→WAN forwarding with explicit
    MASQUERADE; ``"captive"`` gates that forwarding on a per-IP allowlist
    populated by admin sign-in (see ``docs/plans/wifi-wan-gateway.md``
    Phase H).
    """
    override = _load_settings_override().get(SETTINGS_WAN_EGRESS_MODE)
    if _is_valid_egress_mode(override):
        return override  # type: ignore[return-value]
    return "block"


def _wan_egress_mode_source() -> str:
    """Return the persisted source of the current egress mode.

    ``"operator"`` means the value was set explicitly via CLI / REST /
    UI; the AP-up migration ladder must leave it alone. ``"default"``
    is the legacy / never-touched state — the ladder may upgrade it
    (``block`` → ``captive``) once per AP bring-up.
    """
    override = _load_settings_override().get(SETTINGS_WAN_EGRESS_MODE_SOURCE)
    if _is_valid_egress_mode_source(override):
        return override  # type: ignore[return-value]
    return "default"


def _effective_wan_active_profile_id() -> str | None:
    override = _load_settings_override().get(SETTINGS_WAN_ACTIVE_PROFILE_ID)
    if override is None:
        return None
    if _is_valid_uuid4_str(override):
        return override  # type: ignore[return-value]
    return None


def _load_wan_profiles() -> list[dict]:
    """Return the persisted list of WAN profiles. Filters out malformed entries.

    Returns a shallow-copied list so callers can mutate safely without
    touching the cache.
    """
    raw = _load_settings_override().get(SETTINGS_WAN_PROFILES)
    if not isinstance(raw, list):
        return []
    return [dict(p) for p in raw if _is_valid_wan_profile(p)]


def _find_wan_profile_by_id(profile_id: str) -> dict | None:
    for prof in _load_wan_profiles():
        if prof.get("id") == profile_id:
            return prof
    return None


def _find_wan_profiles_by_ssid(ssid: str) -> list[dict]:
    """All profiles matching ``ssid``. Used for disambiguation in the CLI."""
    return [p for p in _load_wan_profiles() if p.get("ssid") == ssid]


def _save_wan_profile(profile: dict) -> None:
    """Insert or update a WAN profile (keyed by ``id``).

    Raises ``ValueError`` if the profile fails validation — settings
    must never contain a malformed profile (the validators are the
    contract enforced at every write).
    """
    if not _is_valid_wan_profile(profile):
        raise ValueError(f"malformed wan profile: {profile!r}")
    profiles = _load_wan_profiles()
    profile_id = profile["id"]
    profiles = [p for p in profiles if p.get("id") != profile_id]
    profiles.append(dict(profile))
    _save_settings_override({SETTINGS_WAN_PROFILES: profiles})


def _delete_wan_profile(profile_id: str) -> bool:
    """Remove a WAN profile by id. Returns True iff something was deleted."""
    profiles = _load_wan_profiles()
    remaining = [p for p in profiles if p.get("id") != profile_id]
    if len(remaining) == len(profiles):
        return False
    _save_settings_override({SETTINGS_WAN_PROFILES: remaining})
    return True


def _set_wan_active_profile_id(profile_id: str | None) -> None:
    """Persist (or clear) the active WAN profile id.

    Validates that ``profile_id`` either refers to an existing profile
    or is ``None``. Refuses to set a non-existent id.
    """
    if profile_id is None:
        _save_settings_override({SETTINGS_WAN_ACTIVE_PROFILE_ID: None})
        return
    if not _is_valid_uuid4_str(profile_id):
        raise ValueError(f"not a uuid4: {profile_id!r}")
    if _find_wan_profile_by_id(profile_id) is None:
        raise ValueError(f"no wan profile with id {profile_id!r}")
    _save_settings_override({SETTINGS_WAN_ACTIVE_PROFILE_ID: profile_id})


def _set_wan_egress_mode(mode: str, *, source: str = "operator") -> None:
    """Persist the WAN egress mode + its provenance.

    ``source="operator"`` (the default) is what every external caller
    should pass — CLI, REST, UI, settings-page saves. Once persisted as
    ``operator`` the value is sticky against future ``default`` migrations.

    ``source="default"`` is reserved for the AP-up migration ladder
    that upgrades a never-touched ``block`` to ``captive``. Callers
    outside the migration path MUST NOT pass this — doing so would let
    a code-internal default silently overwrite an operator's pick on
    the next migration tick.
    """
    if not _is_valid_egress_mode(mode):
        raise ValueError(f"invalid wan_egress_mode: {mode!r}")
    if not _is_valid_egress_mode_source(source):
        raise ValueError(f"invalid wan_egress_mode_source: {source!r}")
    _save_settings_override(
        {
            SETTINGS_WAN_EGRESS_MODE: mode,
            SETTINGS_WAN_EGRESS_MODE_SOURCE: source,
        }
    )


def _load_wan_wired_state() -> tuple[str | None, int | None]:
    """Return ``(connection_name, prior_metric)`` for the claimed wired profile.

    Either component may be ``None`` if we have never claimed the wired
    profile (the WAN feature has never been used) or the previous metric
    was unknown.
    """
    over = _load_settings_override()
    name = over.get(SETTINGS_WAN_WIRED_PROFILE_NAME)
    metric = over.get(SETTINGS_WAN_WIRED_METRIC_ORIGINAL)
    if not isinstance(name, str):
        name = None
    if not isinstance(metric, int):
        metric = None
    return name, metric


def _save_wan_wired_state(name: str, original_metric: int | None) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"wired profile name must be a non-empty str: {name!r}")
    if original_metric is not None and not isinstance(original_metric, int):
        raise ValueError(f"original_metric must be int or None: {original_metric!r}")
    _save_settings_override(
        {
            SETTINGS_WAN_WIRED_PROFILE_NAME: name,
            SETTINGS_WAN_WIRED_METRIC_ORIGINAL: original_metric,
        }
    )


def _load_wan_captive_state() -> dict:
    """Return the cached per-interface captive-portal probe results."""
    raw = _load_settings_override().get(SETTINGS_WAN_CAPTIVE_STATE)
    if not isinstance(raw, dict):
        return {}
    # Only return entries that look sane — defensive against hand-edits.
    out: dict[str, dict] = {}
    for iface, entry in raw.items():
        if not isinstance(iface, str) or not isinstance(entry, dict):
            continue
        connectivity = entry.get("connectivity")
        portal_url = entry.get("portal_url")
        probed_at = entry.get("probed_at")
        out[iface] = {
            "connectivity": connectivity if isinstance(connectivity, str) else "unknown",
            "portal_url": portal_url if isinstance(portal_url, str) else None,
            "probed_at": probed_at if isinstance(probed_at, str) else None,
        }
    return out


def _save_wan_captive_state(
    iface: str, *, connectivity: str, portal_url: str | None, probed_at: str
) -> None:
    """Update the captive-state cache for one interface."""
    current = _load_wan_captive_state()
    current[iface] = {
        "connectivity": connectivity,
        "portal_url": portal_url,
        "probed_at": probed_at,
    }
    _save_settings_override({SETTINGS_WAN_CAPTIVE_STATE: current})


# ── HDMI kiosk display settings ──────────────────────────────────
#
# Persisted by the admin HDMI Display tab; consumed by the
# kiosk-runtime via inotify on the settings file.
#
# Mode IDs are wlr-randr-format strings (e.g. ``3840x2160@60.000Hz``).
# The runtime publishes the current connector's modes to
# ``/run/meeting-scribe/hdmi-status.json``; admin REST validates new
# values against that list. ``"auto"`` always validates and tells the
# runtime to skip the ``--mode`` flag so cage uses the EDID-preferred
# mode.

_HDMI_ROTATION_OPTIONS: tuple[int, ...] = (0, 90, 180, 270)


def _is_valid_hdmi_rotation(value: object) -> bool:
    """True for ``0|90|180|270`` (degrees; cage / wlr-randr semantics)."""
    if not isinstance(value, int):
        return False
    return value in _HDMI_ROTATION_OPTIONS


def _is_valid_hdmi_mode(value: object) -> bool:
    """True for ``"auto"`` or a mode advertised by the kiosk-runtime.

    The runtime writes the available-modes list to
    ``/run/meeting-scribe/hdmi-status.json`` on every wlr-randr probe;
    we accept any string in that list AND ``"auto"`` (cage's
    EDID-preferred default).
    """
    if not isinstance(value, str) or not value:
        return False
    if value == "auto":
        return True
    try:
        from meeting_scribe.kiosk.hdmi_status import is_mode_supported

        return is_mode_supported(value)
    except Exception:
        # If the kiosk module is unavailable for any reason, accept
        # any non-empty string here and rely on the runtime to no-op
        # on an unknown mode. Better to persist a setting that the
        # user obviously typed correctly than to 400 because the
        # kiosk service is not up yet.
        return True


def _is_valid_hdmi_idle_sleep(value: object) -> bool:
    """True for integers in ``[0, 240]`` (minutes; 0 = never sleep)."""
    return isinstance(value, int) and 0 <= value <= 240


def _is_valid_popout_layout(value: object) -> bool:
    """Layout payload must be ``null`` (clear) or a JSON object.

    Fine-grained shape validation lives in the client storage module
    (``popout-layout-storage.js``); the server's job is to be the
    monotonic version arbiter, not to second-guess pane trees.
    """
    return value is None or isinstance(value, dict)


def _effective_hdmi_enabled() -> bool:
    """Default ``True`` so the kiosk auto-starts the first time a cable
    is plugged in. Operators opt out via Settings."""
    return bool((_load_settings_override() or {}).get("hdmi_enabled", True))


def _effective_hdmi_mode() -> str:
    return str((_load_settings_override() or {}).get("hdmi_mode", "auto"))


def _effective_hdmi_rotation() -> int:
    return int((_load_settings_override() or {}).get("hdmi_rotation", 0))


def _effective_hdmi_idle_sleep_minutes() -> int:
    return int((_load_settings_override() or {}).get("hdmi_idle_sleep_minutes", 0))
