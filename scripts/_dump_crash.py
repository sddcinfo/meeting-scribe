#!/usr/bin/env python3
"""Print the crash field from /api/status if present."""
import json
import ssl
import urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
r = urllib.request.urlopen("https://localhost:8080/api/status", timeout=3, context=ctx)
data = json.loads(r.read())
crash = (data.get("metrics") or {}).get("crash")
print(json.dumps(crash, indent=2) if crash else "(no crash recorded)")
