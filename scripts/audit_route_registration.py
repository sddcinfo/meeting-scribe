#!/usr/bin/env python3
"""One-shot: list every route on the live FastAPI app.

Used to triage the 2026-05-07 BT-route 404 — the curl probe only
saw the 302 from the auth gate (which fires before route
resolution), so it falsely confirmed the route was registered.
This script imports the actual app and dumps the route table so
"is the bt/scan route registered?" has a real answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from meeting_scribe import server


def main() -> int:
    paths = sorted({getattr(r, "path", str(r)) for r in server.app.routes})
    print(f"total routes: {len(paths)}")
    print()
    print("== /api/admin/* routes ==")
    for p in paths:
        if p.startswith("/api/admin"):
            print(f"  {p}")
    print()
    print("== bt-related routes ==")
    bt_paths = [p for p in paths if "bt" in p.lower()]
    if not bt_paths:
        print("  (none — admin_bt's router is NOT included in the app)")
    for p in bt_paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
