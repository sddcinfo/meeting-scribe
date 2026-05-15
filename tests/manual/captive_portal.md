# Captive-portal real-device verification

Per-OS expected behavior, run on real hardware (no emulator counts).

## iPhone (iOS 17+)
1. Settings → Wi-Fi → join `meeting-scribe` (or whatever SSID is configured).
2. Within ~2s the captive sheet should pop with the meeting portal.
3. Tap "Join Meeting" → sheet dismisses, blue checkmark appears next to the SSID.
4. Open Safari, navigate to `https://example.com` → should load with the
   network's interception (or fail fast — no 30s hang).

**Failure modes:**
- Captive sheet never appears → CNA probe is failing, check
  `/api/diag/log/server` for `/hotspot-detect.html` requests.
- Sheet stays open after tap → `_captive_acked` set isn't being
  populated; check the index route's `_captive_ack(request)` call.
- 30s hang on any request → an iptables rule reverted from REJECT to
  DROP. Run `tests/test_iptables_rules.py` to verify.

## Android (12+)
1. Settings → Network → Wi-Fi → join the SSID.
2. Notification "Sign in to Wi-Fi network" within ~2s.
3. Tap → portal opens in a Chromium webview.
4. Tap "Join" → "Sign in" notification dismisses, network shows as connected.

## Windows (11)
1. Settings → Network & internet → Wi-Fi → join the SSID.
2. Browser-style window pops within ~3s with the portal.
3. Click "Join" → window dismisses, "internet access" badge appears.

## Pass criteria — last-verified

Bump `tests/manual/.last_verified.json` with `scripts/manual_test_status.py --bump captive_portal_real_devices` after each successful run.
