#!/usr/bin/env python3
"""Mark every meta.json that's still in `recording` state as `stopped`.

Used by /full clean restart workflow so the server doesn't auto-resume
a half-finished meeting on the next start.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "meetings"
fixed = 0
for meta_path in ROOT.glob("*/meta.json"):
    try:
        data = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"skip {meta_path}: {e}", file=sys.stderr)
        continue
    if data.get("state") in ("recording", "created"):
        data["state"] = "stopped"
        meta_path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"marked stopped: {meta_path.parent.name}")
        fixed += 1
print(f"{fixed} meetings normalized")
