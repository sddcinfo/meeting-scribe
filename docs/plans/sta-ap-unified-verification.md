# STA + AP unified hotspot — hardware verification checklist

End-to-end checks that need a real GB10 (MT7925 radio + paired BT
device + hotspot client). Each check maps to a section of the
implementation plan.

## 0. Pre-conditions

```bash
# Branch deployed end-to-end.
git fetch origin
git checkout feature/sta-ap-unified-2026-05-04
sudo bash bootstrap.sh

# Cert generated as v1.0 leaf-only.
meeting-scribe cert-fingerprint        # → 64-char hex; matches /admin/bootstrap header
sudo openssl x509 -in /var/lib/meeting-scribe/certs/cert.pem -text -noout | grep -A 1 "Subject Alternative Name"
# expect: IP Address:10.42.0.1
```

## 1. Cookie + login flow

```bash
# Reset to a clean shell. Open the bootstrap form in the admin laptop's
# browser at https://10.42.0.1/admin/bootstrap.
# Verify:
#   • Cookie is Secure; HttpOnly; SameSite=Strict; Path=/
#   • The 3-part cookie format (issued.session_id.hmac).
#   • Restart the server: previously-issued cookies fail verification
#     (per-boot subkey rotation).

sudo systemctl restart meeting-scribe.service
# Visit / again — guest UI; sign-in form re-mints cookie.
```

## 2. Logout

```bash
# Sign in as admin. Open two tabs of the admin UI, both authed.
# Logout from tab A.
# • Tab A redirects to / (guest UI).
# • Tab B's next request returns guest UI within 5 s (cookie revoked).
# • Browser back button after logout shows guest UI (Cache-Control:
#   no-store, private + Vary: Cookie).
```

## 3. CSP + XSS hardening

```bash
# Admin browser DevTools console, on https://10.42.0.1/.
# • No CSP violation reports.
# • DOM-sink lint (if enabled): every page renders BT names + transcripts
#   via textContent — never innerHTML.
```

## 4. Position-1 firewall invariant

```bash
sudo iptables -L FORWARD -n --line-numbers | head -3
# expect: line 1 jumps to MS_FWD with ms-fw-managed comment

# Displace our jump.
sudo iptables -I FORWARD 1 -s 192.0.2.0/24 -j ACCEPT
meeting-scribe wifi bypass             # legacy rollback path
meeting-scribe wifi unbypass 2>&1 || true
journalctl -u meeting-scribe-server --since "1 minute ago" \
    | grep -E "drift: managed jump not at position 1"
ls /var/lib/meeting-scribe/STA-DEGRADED   # present

sudo iptables -D FORWARD -s 192.0.2.0/24 -j ACCEPT
sudo rm /var/lib/meeting-scribe/STA-DEGRADED
meeting-scribe wifi unbypass

# Docker-style rule at position 2 must NOT trip drift:
sudo iptables -I FORWARD 2 -s 192.0.2.0/24 -j ACCEPT
meeting-scribe wifi bypass
meeting-scribe wifi unbypass            # succeeds
sudo iptables -L FORWARD -n --line-numbers | grep "192.0.2.0/24"  # still present
sudo iptables -D FORWARD -s 192.0.2.0/24 -j ACCEPT
```

## 5. Combined IPv4 restore

```bash
sudo strace -f -e execve -p $(pgrep -f meeting-scribe-server) 2>&1 \
    | grep iptables-restore &
STRACE_PID=$!
meeting-scribe wifi bypass
meeting-scribe wifi unbypass
sudo kill $STRACE_PID 2>/dev/null
# expect: ONE iptables-restore (filter+nat together) + ONE ip6tables-restore
```

## 6. End-to-end no-egress invariant (THE PRODUCT TEST)

From a hotspot client (phone or test machine joined to the AP):

```bash
# AP up, STA down.
ping -c 1 1.1.1.1     # FAIL

# AP up, STA up.
sudo meeting-scribe wifi sta connect yunomotocho
ping -c 1 1.1.1.1     # STILL FAIL (egress isolation)
curl https://example.com   # STILL FAIL

# Reconnect cycles: still FAIL.
# Forced rollback: still FAIL (rollback either succeeds OR refuses with
# STA-DEGRADED; egress always blocked).
```

## 7. BT bridge (Ray-Ban Meta or Mijia)

```bash
meeting-scribe bt status
meeting-scribe bt pair AA:BB:CC:DD:EE:FF
meeting-scribe bt status     # Idle (A2DP)
# Start a meeting; verify TTS plays through glasses (mSBC).
meeting-scribe bt mic on     # → MicLive in <1s; live transcript appears.
meeting-scribe bt mic off    # → A2DP restored.

# Disconnect glasses physically while in MicLive → bridge enters
# Disconnected with link-loss + retry.
# Reconnect → bridge auto-resumes to MicLive (NOT Idle) because
# bt_input_active was True when the link dropped.
```

## 8. Trust install round-trip

On the admin laptop:

```bash
# USB transfer path:
meeting-scribe trust-install --from-pem /Volumes/STICK/cert.pem
# Subsequent visits to https://10.42.0.1 show NO cert warning.

# Fingerprint-confirmation path (after trust-uninstall):
meeting-scribe trust-uninstall <appliance_id>
meeting-scribe trust-install --confirm-fingerprint
# Type a wrong fingerprint → installer aborts; cert NOT in trust store.
# Type the right fingerprint → cert installed; subsequent visits clean.
```

## 9. Privileged helper isolation

```bash
# As the meeting-scribe service user — root_only verbs are rejected.
sudo -u meeting-scribe \
    python -c 'import asyncio, meeting_scribe.helper_client as hc; \
    asyncio.run(hc.regdomain_set(country="US"))'
# expect: HelperError(code="cli_only_verb")

# As root — the verb runs.
sudo python -c 'import asyncio, meeting_scribe.helper_client as hc; \
    print(asyncio.run(hc.regdomain_set(country="US")))'
```

## 10. Admission control + auth rate limit

```bash
# 10 fake guests over /api/ws/view from one IP — 65th connection refused
# with HTTP 503 (Plan §A.7 — pre-upgrade).
# Saturate the cap; admin sign-in still succeeds (auth-aware exemption).

# Hostile auth source — 5 wrong attempts → per-IP backoff to 60 s.
# Different IP from a clean source → still 200 within 100 ms (no global
# lockout).
```

The full pytest suite (1544 unit tests at the time of this writing)
runs via:

```bash
scripts/run-pytest.sh
```

Hardware-gated tests are marked ``@pytest.mark.gb10_hardware`` so the
dev box's run skips them; the real GB10 runs them with that marker
explicitly selected.
