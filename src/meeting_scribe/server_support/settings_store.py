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

import os
from pathlib import Path

from meeting_scribe.runtime import state

# Persisted overrides — survives process restart. The admin UI's
# settings endpoint writes here. Env vars still override.
SETTINGS_OVERRIDE_FILE = Path.home() / ".config" / "meeting-scribe" / "settings.json"

# Default regdomain used when config hasn't been initialized yet (e.g.
# during unit tests that don't run the lifespan startup).
_DEFAULT_REGDOMAIN = "JP"

_DEFAULT_TIMEZONE = ""  # empty = use the server's local time

_settings_cache: dict | None = None
_settings_cache_mtime: float = 0.0


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
        _settings_cache = result
        _settings_cache_mtime = mtime
        return result
    except (OSError, _json.JSONDecodeError):
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


def _is_dev_mode() -> bool:
    """Return True if dev mode is enabled (SSID rotation skipped)."""
    override = _load_settings_override().get("dev_mode")
    if isinstance(override, bool):
        return override
    return os.environ.get("SCRIBE_DEV_MODE", "0") == "1"


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
