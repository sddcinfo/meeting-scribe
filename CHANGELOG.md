# Changelog

All notable changes to meeting-scribe are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.4.0] — 2026-04-16

### Added
- **WiFi hotspot management**: Full AP lifecycle ported from sddc-cli into meeting-scribe (`meeting_scribe/wifi.py`). Three modes: off, meeting (rotating SSID + captive portal + guest isolation), admin (fixed SSID, admin UI reachable over WiFi). No sddc-cli dependency required.
- **CLI: `meeting-scribe wifi up/down/status`**: Manage WiFi AP from the meeting-scribe CLI. `--mode admin` for fixed SSID admin access, `--mode meeting` for rotating guest SSID with captive portal.
- **Admin settings panel: WiFi controls**: Mode toggle, admin SSID, write-only admin password in the settings slide-over. Mode changes trigger async AP reconfiguration (HTTP 202 + background task with rollback).
- **WiFi auto-bring-up on boot**: If `wifi_mode` in settings is not `"off"`, meeting-scribe automatically brings up the AP during server lifespan startup.
- **Admin-mode firewall**: Explicit allowlist (DHCP, DNS, HTTP, admin HTTPS 8080) with default-deny. Same security posture as meeting mode but with port 8080 allowed.
- **WPA3-SAE enforcement**: All WiFi modes use WPA3-Personal with PMF required. WPA2-only devices cannot associate.
- **Captive portal improvements**: DHCP option 114 (RFC 8910) for instant portal detection on iOS 14+/Android 11+. Two-phase probe responses (redirect until portal loaded, then success for CNA dismissal). Port 443 REJECT with TCP RST (HSTS-preloaded domains cannot be intercepted).
- **WiFi status from live radio**: `wifi status` reads from nmcli/wpa_cli, not just the state file. Shows desired vs live mode, security (key_mgmt from wpa_supplicant), regdomain drift detection, client count.
- **`wifi.py` test suite**: 46 tests covering nmcli helpers, captive portal, firewalls, regdomain, admin mode, config validation, rollback snapshot integrity.

### Changed
- **Captive portal probes**: Flipped from "dismiss CNA" to "open CNA" — all OS probes now return 302 redirect to portal (was returning Success/204). CNA dismisses after client loads the portal page (IP-keyed acknowledgement).
- **Settings API**: GET `/api/admin/settings` now returns `wifi_mode`, `wifi_active`, `wifi_ssid`, `wifi_security`, `admin_ssid`, `admin_password_set` (boolean, never the password). PUT accepts `wifi_mode`, `admin_ssid`, `admin_password`.
- **No TLS on port 443**: Deleted `captive-portal-443.py` HTTPS MITM redirector. Port 443 from the hotspot subnet is now REJECT with TCP RST. HSTS-preloaded domains (apple.com, google.com) were causing browser hangs.

### Deprecated
- **`sddc gb10 hotspot up/down/status`**: Prints deprecation warning. Will be removed in a future release. Use `meeting-scribe wifi up/down/status`.

## [1.3.0] — 2026-04-09

### Added
- **Multi-language support**: Language registry with 20 languages, configurable language pairs per meeting
- **Real-time interpretation**: Per-client language preference via WebSocket, TTS audio streaming to listeners
- **Voice cloning TTS**: Qwen3-TTS-12Hz-0.6B-Base via faster-qwen3-tts container (CUDA graph optimized)
- **1:1 conversation mode**: Full-screen split view for 2-person bilingual conversations
- **Metrics split-view**: Side panel with real-time VRAM, ASR, translation stats for demo showcase
- **Captive portal breakout**: Portal page for hotspot clients to escape CNA sandbox into real browser
- **Multi-browser sync**: Late-joining WebSocket clients receive journal replay (last 500 events)
- **Audio drift detection**: Wall clock vs audio byte monitoring with WebSocket warnings
- **Parallel backend loading**: TTS, diarization, name extractor init via asyncio.gather
- **CLI: `meeting-scribe setup`**: First-time setup wizard (venv, HF_TOKEN, certs, Docker)
- **CLI: `meeting-scribe health`**: Full backend + GPU + container health check
- **CLI: `meeting-scribe diagnose`**: System diagnostic report
- **ASR in docker-compose**: Qwen3-ASR-1.7B now managed by compose (was orphan container)
- **HF_TOKEN handling**: Setup command, .env loading, encrypted credentials

### Changed
- Translation `with_translation()` now accepts explicit `target_language` (removed hardcoded ja/en swap)
- ASR language detection delegates to `asr_filters` for multi-script support (was ja/en only)
- WhisperLiveKit normalizes detected languages via `languages.py` registry (was ja/en whitelist)
- Removed `supported_languages` config field (unused, replaced by language registry)
- TTS container: custom `faster-qwen3-tts` image replaces non-functional vLLM TTS service
- ASR recipe: added `--enforce-eager --load-format safetensors` to match working config

### Fixed
- TTS container crash: `Qwen/Qwen3-TTS` model ID didn't exist (now `Qwen/Qwen3-TTS-12Hz-0.6B-Base`)
- TTS container driver incompatibility: new container uses `nvcr.io/nvidia/pytorch:25.01-py3` (works with driver 580.x)
- ASR not managed by docker-compose (was manually started orphan container)
- Stale PID file preventing server restart

## [1.2.0] — 2026-04-07

### Added
- GB10 GPU production deployment (migrated from macOS MLX)
- Qwen3-ASR-1.7B via vLLM for real-time multilingual ASR
- Qwen3.5-35B-A3B-FP8 translation via vLLM with TurboQuant
- pyannote.audio diarization with Sortformer backend
- Rolling refinement worker (trails ~45s behind live transcription)
- Translation persistence to journal (fixed pre-existing bug)
- WiFi hotspot + captive portal via `sddc gb10 hotspot up`
- Finalization modal with progress steps and ETA
- Meeting summary generation (topics, action items, decisions)
- Speaker timeline with zoom (Ctrl+scroll, 1x-20x)
- Seamless full-meeting audio playback with transcript sync
- Layout modes (hide live, hide table, compact, pop-out)
- Guest view (guest.html) for read-only hotspot clients
- How-it-works page rewritten for GB10

## [1.0.0] — 2026-03

### Added
- Initial release with real-time bilingual transcription
- WebSocket-based audio streaming from browser microphone
- Speaker enrollment with voice reference recording
- Virtual room layout with drag-and-drop seats
- Three-column transcript view (Live, Language A, Language B)
- Meeting recording and playback
