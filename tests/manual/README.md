# Manual-test runbook

Things that genuinely can't be automated. Cap: 15 minutes total.
Each item has a "last-verified" stamp; run `scripts/manual_test_status.py`
to see what's stale (warning at 30 days; release CI fails at 30 days
on tagged release commits).

To bump a stamp after running an item:

```
scripts/manual_test_status.py --bump <slug>
```

## Items

### iphone_hotspot_join — `slug: iphone_hotspot_join`
**What:** Open WiFi on a real iPhone, join the meeting hotspot, verify
captive sheet pops within ~2s and presents the portal.
**Why automation can't:** Real iOS CNA + connectivity probe interplay
isn't faithfully modeled by Playwright WebKit. See
`tests/test_captive_portal_logic.py` for the unit-level coverage.
**Pass criteria:**
- Captive sheet appears within 5s of joining.
- Tapping into the portal page dismisses the sheet within 2s.
- Loading any other URL (e.g. `https://example.com`) gets RST'd within 1s.

### gb10_model_swap — `slug: gb10_model_swap`
**What:** `meeting-scribe down` then `meeting-scribe up`, verify all
three backends (ASR, translate, TTS) come up green.
**Why automation can't:** No GPU runner in CI; backends require GB10
+ vLLM container init.
**Pass criteria:**
- `scripts/check_backend_apis.py` returns exit 0 within 5 minutes of `up`.
- `meeting-scribe status` shows all three backends ready.

### two_hour_meeting_smoke — `slug: two_hour_meeting_smoke`
**What:** Run a real 2-hour meeting end-to-end. Memory-leak smoke.
**Why automation can't:** Wall-clock duration. Use real audio.
**Pass criteria:**
- No memory growth above 4 GB resident over the meeting.
- Finalization completes within 60s.
- Generated summary is non-empty + bilingual.

### listen_mode_audio_out — `slug: listen_mode_audio_out`
**What:** Open the popout in Listen mode on a guest device, verify
TTS audio actually plays for incoming translations.
**Why automation can't:** Real audio output device required; MSE
encoder smoke is in `tests/browser/test_mse_client_contract.py` but
that doesn't exercise the speaker.
**Pass criteria:**
- Audio plays within ~2s of a translation completing.
- Audio is intelligible (no corruption, no codec gap).

### captive_portal_real_devices — `slug: captive_portal_real_devices`
**What:** Join the hotspot on iPhone, Android, and Windows. Verify the
captive sheet behavior on each. See `tests/manual/captive_portal.md`
for the per-OS expected behavior.

---

## Why this exists

Automated tests cover the seams meeting-scribe has historically broken
at (cross-window state sync, WS lifecycle, event dedup, language
routing, captive-portal logic). The runbook covers what can't be
automated without GPUs / real radios / real devices. Don't add items
here that could be a unit/integration test.
