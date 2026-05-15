# Manual Device Test Card

These tests require real devices and cannot be automated with headless Chromium.
Run after any change to the MSE pipeline or guest.html audio path.

## 1. iPhone Safari MSE acceptance

- Hard-reload guest page on iPhone
- Silent switch in silent mode
- Tap Listen
- Verify: `path === "mse"` in /api/diag/listeners
- Verify: `format_acked === true`
- Verify: audio audible through speaker

## 2. iPhone background-tab continuity

- Start listening (MSE path active)
- Lock screen or switch to another app
- Wait 30 seconds
- Return to Safari
- Verify: audio resumes (or note if it does not)

## 3. Android Chrome smoke

- Open guest page in Chrome on Android
- Tap Listen
- Verify: MSE path activates (`path === "mse"` in diag)
- Verify: audio plays through speaker
