#!/usr/bin/env python3
"""Print listener-side delivery metrics from /api/status."""
import json
import ssl
import urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
r = urllib.request.urlopen("https://localhost:8080/api/status", timeout=2, context=ctx)
data = json.loads(r.read())
m = data.get("metrics") or {}
tts = m.get("tts") or {}
listener = m.get("listener") or {}
print("tts.submitted:    ", tts.get("submitted"))
print("tts.delivered:    ", tts.get("delivered"))
print("tts.drops:        ", tts.get("drops"))
print("tts.timeouts:     ", tts.get("timeouts"))
print("listener.connected:", listener.get("connected"))
print("listener.deliveries:", listener.get("deliveries"))
print("listener.send_failed:", listener.get("send_failed"))
print("listener.removed_on_send_error:", listener.get("removed_on_send_error"))
print("listener.send_ms:  ", listener.get("send_ms"))
print("audio_out_connections:", data.get("audio_out_connections"))
print("connections:      ", data.get("connections"))
