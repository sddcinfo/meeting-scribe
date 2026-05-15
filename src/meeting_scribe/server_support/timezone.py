"""IANA timezone helpers.

Used by the settings API (admin selects a timezone for the device) and
by anything that needs to validate user-supplied timezone names. Kept
out of ``server.py`` so route modules can import without dragging the
full server graph in.
"""

from __future__ import annotations


def _timezone_options() -> list[str]:
    """Return all IANA timezone names the runtime knows about.

    Filters out bare legacy aliases (``UTC``, ``GMT``, ``EST`` …) that
    don't follow the ``Region/City`` format — those are confusing in a
    dropdown next to modern names. Sorted alphabetically.
    """
    from zoneinfo import available_timezones

    names = available_timezones()
    # Keep only names with a '/' — drops legacy shortcuts like UTC, GMT,
    # EST, MST, Zulu, etc. UTC is re-added at the top for visibility.
    tz_list = sorted(n for n in names if "/" in n)
    return ["UTC", *tz_list]


def _is_valid_timezone(name: str) -> bool:
    """Return True if ``name`` is a valid IANA timezone name."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(name)
        return True
    except ZoneInfoNotFoundError, ValueError:
        return False
