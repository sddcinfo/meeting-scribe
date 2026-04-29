# Changelog

All notable changes to meeting-scribe are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.5.0] — 2026-04-29

### Added
- **Diarization on pyannote 4.0 (`speaker-diarization-community-1`)**: production swap from pyannote 3.x. Single-speaker exclusive timeline emitted alongside the standard segmentation; per-segment confidence scoring.
- **`meeting-scribe validate`**: comprehensive end-to-end validation command that exercises the full pipeline against a fixture audio sample and emits a structured pass/fail report.
- **Live transcript popout**: dedicated browser popout window with auto-reconnecting WebSocket, connection-state pill, resizable slide pane, and admin-side slide controls.
- **Eager summary**: summaries pre-compute during recording; finalize completes faster.
- **Cancel meeting**: cancel button deletes all artifacts on the spot.
- **Slide translation parallelism**: adaptive batching + 4-way concurrent LibreOffice renders. ~6× faster than per-slide on a 20-slide deck. `SCRIBE_SLIDE_RENDER_PARALLELISM` (default 4, clamped to [1, 16]).
- **Refinement worker (off by default)**: rolling polished transcript trailing 45s behind live. `SCRIBE_ENABLE_REFINEMENT=1` to enable. Strongly recommended to pair with a separate `SCRIBE_TRANSLATE_OFFLINE_VLLM_URL` to avoid degrading live-path latency.
- **Split-backend env vars**: `SCRIBE_TRANSLATE_REALTIME_VLLM_URL` (smaller live-path model) and `SCRIBE_TRANSLATE_OFFLINE_VLLM_URL` (refinement-path model) now honoured by `_apply_env`.

### Changed
- **Modular package layout**: `server.py` decomposed from a 11k-line monolith into focused packages — `routes/` (API surface), `runtime/` (lifespan, state, metrics, init, background loops, health monitors, net), `pipeline/` (transcript-event router, diarize, quality, speaker_attach), `server_support/` (helpers), `ws/` (WebSocket handlers), `tts/worker.py`, `audio/output_pipeline.py`, `middlewares.py`, `hotspot/` (AP lifecycle, captive portal). `server.py` is now <800 LOC and only constructs the FastAPI app + wires routers.
- **CLI split**: `cli.py` (3.4k LOC, 52 commands) atomically refactored into a 13-module `cli/` package — one module per topic group (`benchmark`, `config`, `gb10`, `library`, `lifecycle`, `meetings`, `precommit`, `queue`, `setup`, `terminal`, `versions`, `wifi`). Click registration unchanged; subcommand resolution loads only the relevant module.
- **`hotspot/ap_control.py` ↔ `wifi.py` deduplication**: nmcli + state helpers consolidated (-263 LOC).
- **Single canonical `atomic_write_json`**: `util/atomic_io.py` is the source of truth; the four prior copies in `wifi.py`, `hotspot/ap_control.py`, `slides/job.py`, `slides/worker.py` are gone.
- **No back-compat shims in `server.py`**: 5 `noqa: E402, F401` re-exports removed; downstream callers retargeted to canonical paths.
- **Live-path script-router fix**: same-script language pairs (e.g. en↔de) no longer go through the kana/Latin script router.
- **Finalize correctness**: deferred A2 step 3 applied; A5 flipped to shadow by default for safer rollouts.
- **vLLM image consolidation**: dropped the custom `ghcr.io/bjk110/vllm-spark:turboquant` build path. ASR now runs on `scribe-vllm-asr:latest` — a 4-line Dockerfile under `containers/vllm-asr/` that layers `vllm[audio]` extras (soundfile, av, soxr) on top of stock `vllm/vllm-openai:latest`. Translation continues to run on stock `vllm/vllm-openai` via autosre. Measured on 100 English fleurs samples: identical WER (p50 0.2609, p95 0.4751 — bit-equal), p50 latency −3.9% (687→660 ms), p95 latency −19.2% (1238→1000 ms). `SCRIBE_VLLM_IMAGE` env override remains for opt-in custom builds.
- **Python floor bumped to 3.14**: matches `pyproject.toml requires-python` and `.mise.toml`.

### Removed
- **Ad-hoc TurboQuant build flow**: `--turboquant` flag on `meeting-scribe gb10 setup`, the `build_vllm_image()` SSH+clone+docker-build helper, and the `docker_image_turboquant` field in `GB10Config`. Anyone who needs a custom vLLM image can set `SCRIBE_VLLM_IMAGE` directly.
- **Omni unified-backend scaffolding (unshipped)**: `containers/omni/` (Dockerfile + acceptance/build pin docs), the `omni-unified` compose service under the `omni-spike` profile, the `omni_asr_url` / `omni_tts_url` / `omni_translate_url` config fields and their `_apply_env` reads, the `omni_*_url` fallthroughs in `runtime/init.py` and `routes/meeting_lifecycle.py`, the 7 contract tests under `tests/omni/`, and the Omni references in module docstrings/comments. The consolidation never reached its acceptance gates (the rows in `containers/omni/ACCEPTANCE.md` were all blank); production has been on dedicated ASR + TTS + translate containers throughout. Re-add as a separate spike branch if revived.
- **`vllm-tts` migration target (unshipped)**: `containers/tts-vllm/` (Dockerfile + Dockerfile.spike + README), the `vllm-tts` compose service under the `vllm-omni` profile, and the `qwen3-tts-vllm.yaml` recipe + its three test references. Same pattern as Omni — the migration parallel-run never started; production runs `qwen3-tts` + `qwen3-tts-2` on the dedicated faster-qwen3-tts container.

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
