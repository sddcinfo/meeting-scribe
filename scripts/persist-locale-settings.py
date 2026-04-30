#!/usr/bin/env python3
"""Persist locale settings (regdomain + timezone) into meeting-scribe's
user settings file.

Called by ``bootstrap.sh`` once the regular user has confirmed (or
overridden) the auto-detected defaults. Writes to
``$XDG_CONFIG_HOME/meeting-scribe/settings.json`` so the running
server picks them up via the existing settings_store loader on its
next read — no env-var plumbing needed.

Usage:
    persist-locale-settings.py <regdomain> <timezone>

Both arguments are required. We deliberately do not auto-detect from
inside this script — bootstrap is the single source of truth for what
the customer chose, and writing the values it owns means re-running
bootstrap can't drift out of sync with the on-disk settings.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <regdomain> <timezone>", file=sys.stderr)
        sys.exit(2)
    regdomain, timezone = sys.argv[1], sys.argv[2]

    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    target = base / "meeting-scribe" / "settings.json"
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)

    existing: dict[str, str] = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            # Corrupt / unreadable — overwrite. The values we're about
            # to write are derivable from bootstrap so a clobber is
            # safer than a partial-merge with garbage.
            existing = {}

    existing.update({"wifi_regdomain": regdomain, "timezone": timezone})
    target.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"[bootstrap] persisted regdomain={regdomain} timezone={timezone} → {target}")


if __name__ == "__main__":
    main()
