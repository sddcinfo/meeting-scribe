"""WiFi regulatory-domain helpers.

Three layers of defense against ending up on the wrong regdomain
(which silently caps 5 GHz AP transmit power and prevents clients
from associating):

1. ``/etc/modprobe.d/cfg80211-<code>.conf`` sets ``ieee80211_regdom``
   at module load time so boots with meeting-scribe installed start
   in the configured country.
2. ``_ensure_regdomain()`` runs ``iw reg set <code>`` before every AP
   rotation and at server startup, and verifies via ``iw reg get``
   that the setting took effect.
3. The verify step logs an error if the code couldn't be set — no
   silent drift.

Pulled out of ``server.py`` so route modules (``routes/admin.py``)
and the AP-control bring-up path can both reach these helpers
without circling back through the server module.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from meeting_scribe.server_support.settings_store import (
    _effective_regdomain,
    _regdomain_modprobe_path,
)

logger = logging.getLogger(__name__)


def _current_regdomain() -> str | None:
    """Return the current 2-letter country code from ``iw reg get``.

    Prefers the **phy#0** block over the **global** block when both are
    present. When STA is associated to an upstream AP, the upstream's
    Country Information Element (IE) overrides the global regdomain to
    a numeric code (e.g. ``98``); but the phy0 setting we explicitly
    applied via ``iw reg set`` stays correct. AP TX power is governed
    by phy0's regdomain, so phy0 is the right value to check.

    Returns the country code (``JP``, ``US``, ``00``, …) or ``None`` if
    ``iw`` isn't available or the output can't be parsed.
    """
    try:
        result = subprocess.run(
            ["iw", "reg", "get"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    # Walk every block (`global`, `phy#0`, …). Prefer phy#0's country
    # over global's when both exist.
    blocks: dict[str, str] = {}  # block_name → country code
    current_block: str | None = None
    for raw in result.stdout.splitlines():
        stripped = raw.strip()
        if stripped == "global" or stripped.startswith("phy#"):
            current_block = stripped
            continue
        if stripped.startswith("country "):
            token = stripped.split(":", 1)[0].removeprefix("country ").strip()
            if token and current_block is not None and current_block not in blocks:
                blocks[current_block] = token
    # Prefer phy#0 if available; fall back to global; finally any block.
    if "phy#0" in blocks:
        return blocks["phy#0"]
    if "global" in blocks:
        return blocks["global"]
    if blocks:
        return next(iter(blocks.values()))
    return None


def _ensure_regdomain() -> bool:
    """Ensure the WiFi regulatory domain matches ``_effective_regdomain()``.

    Runs ``sudo iw reg set <code>`` and verifies the result via ``iw reg get``.
    Returns True on success, False otherwise. Never raises — failures are
    logged and the caller decides whether to abort.
    """
    import time

    target = _effective_regdomain()
    current = _current_regdomain()
    if current == target:
        return True

    try:
        subprocess.run(
            ["sudo", "iw", "reg", "set", target],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.error(
            "failed to invoke 'iw reg set %s': %s (current=%r)",
            target,
            exc,
            current,
        )
        return False

    # Verify the setting actually took effect. The kernel may reject
    # the change (e.g. on some MT7925 firmware versions) or silently
    # revert due to a concurrent scan result.
    time.sleep(0.2)
    after = _current_regdomain()
    if after == target:
        if current != after:
            logger.info("regdomain set to %s (was %r)", target, current)
        return True

    logger.error(
        "regdomain set to %s failed: iw reg get still reports %r — "
        "5 GHz AP transmit power will be capped; clients cannot connect",
        target,
        after,
    )
    return False


def _ensure_regdomain_persistent() -> bool:
    """Install ``/etc/modprobe.d/cfg80211-<code>.conf`` so cfg80211 boots
    in the configured country. Idempotent — rewrites only if the target
    code has changed. Returns True on success, False on failure (e.g.
    no sudo, write error). Never raises.
    """
    target = _effective_regdomain()
    path = _regdomain_modprobe_path(target)
    body = f"options cfg80211 ieee80211_regdom={target}\n"

    # If an existing file for a DIFFERENT country is present, remove it so
    # we don't end up with two conflicting modprobe entries.
    try:
        for existing in Path("/etc/modprobe.d").glob("cfg80211-*.conf"):
            if existing == path:
                continue
            try:
                subprocess.run(
                    ["sudo", "rm", "-f", str(existing)],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
    except OSError:
        pass

    # Skip the write if the content is already correct.
    try:
        if path.exists() and path.read_text() == body:
            return True
    except OSError:
        pass

    try:
        result = subprocess.run(
            ["sudo", "tee", str(path)],
            input=body,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("could not write %s: %s", path, exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "sudo tee %s failed (rc=%d): %s",
            path,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return True
