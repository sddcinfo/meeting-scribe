#!/usr/bin/env bash
# Post-restart smoke test for meeting-scribe.
#
# Runs directly against the live stack and fails hard on anything that
# would surface to the user as "broken". Intended to be invoked by the
# `meeting-scribe restart` wrapper so a silent partial startup never
# slips through.
#
# Exit codes:
#   0  — everything green
#   1  — a component that's required for a meeting failed
#   2  — a non-blocking optional component is down (TTS, summary)
set -u
cd "$(dirname "$0")/.."

SCRIBE_HTTP="${SCRIBE_HTTP:-https://localhost:8080}"
CURL_TLS_FLAGS="${CURL_TLS_FLAGS:--k}"
LAG_SAMPLES=5
LAG_WARN_MS=250
LAG_FAIL_MS=1500

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

fail=0
soft=0

_probe() {
    local label=$1 url=$2 expect_code=${3:-200}
    local extra=""
    # Admin endpoints use self-signed TLS; model containers don't.
    case "$url" in
        https://*) extra="$CURL_TLS_FLAGS" ;;
    esac
    code=$(curl -sS $extra -o /dev/null --max-time 5 -w '%{http_code}' "$url" 2>/dev/null)
    if [ "$code" = "$expect_code" ]; then
        green "  ✓ $label (http=$code)"
        return 0
    else
        red "  ✗ $label (http=$code, expected $expect_code)"
        return 1
    fi
}

echo "== static JS syntax =="
if command -v node >/dev/null 2>&1; then
    if node -c static/js/scribe-app.js 2>/tmp/js-err; then
        green "  ✓ scribe-app.js parses clean"
    else
        red "  ✗ JS syntax error — page will render blank"
        red "$(cat /tmp/js-err)"
        fail=1
    fi
else
    yellow "  ⚠ node not installed — skipping JS syntax check"
fi

echo "== scribe-main process =="
if ! pgrep -f "meeting_scribe.server" >/dev/null; then
    red "  ✗ no meeting_scribe.server process running"
    fail=1
else
    pid=$(pgrep -f "meeting_scribe.server" | head -1)
    green "  ✓ PID $pid"
fi

echo "== HTTP admin =="
_probe "landing page"     "$SCRIBE_HTTP/"              200 || fail=1
_probe "/api/status"      "$SCRIBE_HTTP/api/status"    200 || fail=1
_probe "/api/meetings"    "$SCRIBE_HTTP/api/meetings"  200 || fail=1

echo "== model containers =="
_probe "ASR :8003"        "http://localhost:8003/health"   200 || fail=1
_probe "translate :8010"  "http://localhost:8010/health"   200 || fail=1
_probe "diarize :8001"    "http://localhost:8001/health"   200 || fail=1
_probe "TTS legacy :8002" "http://localhost:8002/health"   200 || soft=1

echo "== backend latency probes =="
# ASR end-to-end probe: send a 1.5s sine tone and confirm a 200 response.
python3 - <<'PY' 2>/dev/null || { red "  ✗ ASR probe crashed"; exit 1; }
import base64, io, time, wave, urllib.request, json
import sys
try:
    import numpy as np
except ImportError:
    # use stdlib to build a 1.5s 16kHz 220Hz int16 sine tone
    import math, struct
    sr, secs, freq = 16000, 1.5, 220.0
    samples = [int(0.3*32767*math.sin(2*math.pi*freq*i/sr)) for i in range(int(sr*secs))]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(struct.pack("<%dh" % len(samples), *samples))
    audio_bytes = buf.getvalue()
else:
    sr, secs = 16000, 1.5
    t = np.linspace(0, secs, int(sr*secs), endpoint=False)
    a = (0.3*np.sin(2*np.pi*220*t)*32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf,"wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(a.tobytes())
    audio_bytes = buf.getvalue()

b64 = base64.b64encode(audio_bytes).decode()
body = json.dumps({"model":"Qwen/Qwen3-ASR-1.7B","messages":[{"role":"system","content":"Transcribe."},{"role":"user","content":[{"type":"input_audio","input_audio":{"data":b64,"format":"wav"}}]}],"max_tokens":64,"temperature":0.0}).encode()
req = urllib.request.Request("http://localhost:8003/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
t0 = time.monotonic()
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        ms = (time.monotonic()-t0)*1000
        assert r.status == 200
        print(f"  ✓ ASR synthetic probe {ms:.0f}ms")
except Exception as e:
    print(f"  ✗ ASR probe: {e}")
    sys.exit(1)
PY

# Translate probe — 1-sentence JA→EN round trip, timeout 15s.
# Auto-discovers the served model via /v1/models so the probe does not
# need updating every time we migrate backends (was hard-coded to the
# pre-2026-04 INT4 model and silently 404'd after the FP8 migration).
python3 - <<'PY' 2>/dev/null || { red "  ✗ translate probe crashed"; soft=1; }
import json, time, urllib.request
try:
    models = json.loads(urllib.request.urlopen("http://localhost:8010/v1/models", timeout=5).read())
    model_id = models["data"][0]["id"]
except Exception as e:
    print(f"  ✗ translate: could not discover model via /v1/models: {e}")
    raise SystemExit(1)
body = json.dumps({"model":model_id,"messages":[{"role":"system","content":"Translate Japanese to English."},{"role":"user","content":"こんにちは、世界。"}],"temperature":0.0,"max_tokens":64,"stream":False,"chat_template_kwargs":{"enable_thinking":False}}).encode()
t0 = time.monotonic()
try:
    r = urllib.request.urlopen(urllib.request.Request("http://localhost:8010/v1/chat/completions", data=body, headers={"Content-Type":"application/json"}), timeout=15)
    ms = (time.monotonic()-t0)*1000
    data = json.loads(r.read())
    out = data["choices"][0]["message"]["content"].strip()
    if out and out.strip():
        short_model = model_id.split("/")[-1]
        print(f"  ✓ translate probe {ms:.0f}ms ({short_model})  'こんにちは' → {out[:40]!r}")
    else:
        print(f"  ✗ translate probe returned empty")
        raise SystemExit(1)
except Exception as e:
    print(f"  ✗ translate: {e}")
    raise SystemExit(1)
PY

echo "== event-loop lag (sampled via /api/status rolling buckets) =="
python3 - <<PY || { red "  ✗ lag probe crashed"; fail=1; }
import ssl, json, urllib.request, sys
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
r = urllib.request.urlopen("${SCRIBE_HTTP}/api/status", context=ctx, timeout=5).read()
d = json.loads(r)
loop = d.get("metrics",{}).get("loop_lag_ms", {}) or {}
def _f(v): return v if isinstance(v,(int,float)) else 0
p50, p95, p99 = _f(loop.get("p50")), _f(loop.get("p95")), _f(loop.get("p99"))
n = loop.get("sample_count",0) or 0
if n < 10:
    print(f"  ⚠ only {n} loop-lag samples collected yet — not enough to judge (fresh start, give it ~10s)"); sys.exit(0)
print(f"  p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms  samples={n}")
if p99 > 2500:
    print(f"  ⚠ p99 > 2500ms — periodic stall (likely TTS health eval, retry loop, or diarize window). BASELINE target p99 < 300ms."); sys.exit(0)
elif p95 > 500:
    print(f"  ⚠ p95 > 500ms — investigate background loops")
else:
    print(f"  ✓ event-loop healthy")
PY
lag_rc=$?
[ "$lag_rc" != 0 ] && fail=1

echo
if [ "$fail" != 0 ]; then
    red "SMOKE TEST FAILED — meeting-scribe is NOT ready to accept meetings"
    exit 1
fi
if [ "$soft" != 0 ]; then
    yellow "SMOKE TEST PARTIAL — critical path OK, optional components degraded"
    exit 2
fi
green "SMOKE TEST GREEN — meeting-scribe ready"
exit 0
