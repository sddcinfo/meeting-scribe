#!/usr/bin/env python3
"""Print the deterministic admin password + guest PIN for the sidecar.

These are computed from the same ``appliance_pin()`` source production
uses, so they're identical between the two instances. Output:

    admin password: DellMeetingAdmin<NNNN>
    guest PIN:      <NNNN>

The sidecar's state dir at /tmp/meeting-scribe-sidecar-state was seeded
by ``scripts/sidecar.py start`` with the HMAC of this admin password,
so /auth accepts it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from meeting_scribe.cli._common import appliance_pin
from meeting_scribe.setup_state import _mint_admin_password


def main() -> int:
    pin = appliance_pin()
    pw = _mint_admin_password()
    print(f"admin password: {pw}")
    print(f"guest PIN:      {pin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
