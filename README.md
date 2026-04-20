# Meeting Scribe

Real-time multilingual meeting transcription running on the NVIDIA GB10 (DGX Spark). Transcribes and translates live speech between any pair of 20 supported languages (10 of them TTS-capable for synthesized interpretation audio), identifies speakers via diarization, streams the interpretation audio track to guests over a local WiFi hotspot, and records the full meeting.

Live demo + deep-dive: <https://sddcinfo.github.io/meeting-scribe/>

## Model Stack

All models run locally on a single GB10 node (aarch64 Linux, 128 GB unified memory, CUDA 13.0) — no cloud dependency:

| Component | Model | Backend | Default port |
|-----------|-------|---------|------|
| ASR | Qwen3-ASR-1.7B | vLLM | 8003 |
| Translation | Qwen3.6-35B-A3B-FP8 | vLLM | 8010 |
| TTS (interpretation audio) | Qwen3-TTS-12Hz-0.6B-Base | faster-qwen3-tts | 8002 |
| Diarization | pyannote.audio 3.x | Custom container | 8001 |
| Name extraction | Qwen3.6-35B-A3B-FP8 (reuses translation vLLM) | vLLM | 8010 |
| Refinement worker — *off by default* | Qwen3.6-35B-A3B-FP8 (reuses translation vLLM) | vLLM | 8010 |

The 35B FP8 translation model is the heaviest component (≈35 GB VRAM when loaded + KV cache). Combined footprint runs ≈43 GB, well inside the GB10's 128 GB unified pool. The translate vLLM endpoint is also the primary sharing point if you run [auto-sre](https://github.com/sddcinfo/auto-sre) on the same box — both point at `:8010`.

## Features

- **Multi-language ASR** — Qwen3-ASR with 52-language support + auto-detection
- **Configurable language pairs** — Any pair from **20 supported languages**. 10 of those are TTS-capable (English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian) and unlock the interpretation-audio feature end-to-end. The other 10 (Dutch, Arabic, Thai, Vietnamese, Indonesian, Malay, Hindi, Turkish, Polish, Ukrainian) work for ASR + translate only — no synthesized interpretation audio. Default pair is `en,ja`; set `SCRIBE_LANGUAGE_PAIR` to change.
- **Interpretation audio** — For TTS-capable language pairs only. Near-real-time translated audio track, per-client language preference, streamed to hotspot guests over a dedicated WebSocket.
- **Speaker diarization** — pyannote-based speaker identification (real speakers, not just "speaker 1 / speaker 2")
- **1:1 conversation mode** — Full-screen split for 2-person bilingual conversations
- **Metrics dashboard** — Split-view real-time performance stats (memory, ASR, translation latency)
- **LLM name extraction** — Detects speaker names from conversation context
- **Bilingual transcript view** — Every utterance shown in its original language side-by-side with the translation
- **Slide translation** — Upload a PPTX; the deck is translated slide-by-slide and rendered progressively during the meeting (adaptive batching + concurrent LibreOffice renders for fast first-paint)
- **Audio recording** — Time-aligned PCM with segment-level playback and a podcast-style player
- **Room setup** — Drag-and-drop table/seat editor with 8 presets
- **Meeting history** — Browse and replay past meetings with full transcript and audio
- **WiFi hotspot** — QR code for guest device access with captive portal breakout
- **Refinement worker** *(optional, off by default)* — Rolling polished transcript trailing 45 s behind live. On a single shared translate backend this can degrade live-path latency; keep it off unless you have a separate vLLM instance for the refinement path (`SCRIBE_TRANSLATE_OFFLINE_VLLM_URL`).
- **Multi-browser sync** — Late-joining clients receive a journal replay
- **Zero build step** — Static HTML/CSS/JS, instant refresh during development

## Quick Start

Two commands on a fresh GB10 (NVIDIA DGX Spark, aarch64 Linux with CUDA):

```bash
git clone https://github.com/sddcinfo/meeting-scribe.git
cd meeting-scribe && ./bootstrap.sh
```

`bootstrap.sh` gates on platform, installs OS packages (ffmpeg, libportaudio,
etc.), pins Python via mise (or uses system python3.11+), creates a `.venv`,
editable-installs meeting-scribe, then hands off to `meeting-scribe setup`
for HF token / TLS certs / container stack / systemd registration.

Manual install (for dev machines where you want to manage the pieces yourself):

```bash
git clone https://github.com/sddcinfo/meeting-scribe.git
cd meeting-scribe
python3 -m venv .venv
.venv/bin/pip install -e .

# Configure environment
cp .env.gb10 .env
$EDITOR .env

# Start model backends (ASR + TTS + diarization containers)
.venv/bin/meeting-scribe gb10 up

# Start the translate vLLM separately. Two options:
#   (a) with the auto-sre helper (recommended):
#       autosre start            # https://github.com/sddcinfo/auto-sre
#   (b) without auto-sre, run vLLM directly (see docker-compose.gb10.yml
#       or the vLLM docs for the Qwen3.6-35B-A3B-FP8 flags).

# Run the scribe server
.venv/bin/meeting-scribe start
# Admin UI:     https://127.0.0.1:8080  (also on LAN management IP)
# Guest portal: http://127.0.0.1/       (also on hotspot AP IP 10.42.0.1)
```

## Architecture

```
Browser AudioWorklet (native-rate s16le PCM + sample-rate header)
  --> WebSocket
  --> FastAPI Server
        --> torchaudio resample to 16kHz (Kaiser sinc, GPU)
        --> Qwen3-ASR (vLLM, port 8003) -- real-time transcription
        --> Qwen3.6-35B-A3B-FP8 (vLLM, port 8010) -- multilingual translation + name extraction (shared)
        --> pyannote (port 8001) -- speaker diarization
        --> Qwen3-TTS (faster-qwen3-tts, port 8002) -- interpretation audio synthesis
        --> AudioWriter (time-aligned PCM recording)
  --> WebSocket (TranscriptEvent JSON) --> host transcript view
  --> WebSocket (audio-out WAV frames) --> hotspot guest listeners
```

## Configuration

All settings via environment variables (see `.env.gb10` for the full template):

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRIBE_PROFILE` | | Set to `gb10` for production mode |
| `SCRIBE_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for network access) |
| `SCRIBE_PORT` | `8080` | Server port |
| `SCRIBE_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | ASR model served by vLLM |
| `SCRIBE_TRANSLATE_BACKEND` | `vllm` | Translation backend |
| `SCRIBE_TRANSLATE_VLLM_URL` | `http://localhost:8010` | vLLM translation endpoint (shared with autosre) |
| `SCRIBE_DIARIZE` | `true` | Enable pyannote speaker diarization |
| `SCRIBE_DIARIZE_URL` | `http://localhost:8001` | pyannote endpoint |
| `SCRIBE_TTS_VLLM_URL` | `http://localhost:8002` | TTS endpoint (faster-qwen3-tts) |
| `SCRIBE_LANGUAGE_PAIR` | `en,ja` | Default meeting language pair (comma-separated ISO 639-1 codes, any pair from the 20 supported languages) |
| `SCRIBE_TRANSLATE_REALTIME_VLLM_URL` | | Optional: smaller model for live translation |
| `SCRIBE_TRANSLATE_OFFLINE_VLLM_URL` | | Optional: separate vLLM endpoint for the refinement worker. Strongly recommended if `SCRIBE_ENABLE_REFINEMENT=1` — running refinement on the same backend as live-path translation degrades live p95/p99. |
| `SCRIBE_ENABLE_REFINEMENT` | `false` | Enable the refinement worker (rolling polished transcript, trails 45 s). Off by default — see the Refinement feature note above. |
| `SCRIBE_SLIDE_RENDER_PARALLELISM` | `4` | Concurrent LibreOffice slide-render processes. Clamped to [1, 16]. Measured sweet spot on GB10 is 4. |
| `SCRIBE_NAME_EXTRACTION` | `auto` | LLM-based speaker name extraction |
| `HF_CACHE_DIR` | `/data/huggingface` | HuggingFace model cache directory |

### Docker Compose

The `docker-compose.gb10.yml` file manages all model backends:

```bash
# Start all backends
docker compose -f docker-compose.gb10.yml up -d

# Or use the CLI
meeting-scribe gb10 up

# Check status
docker compose -f docker-compose.gb10.yml ps
```

## CLI

```bash
meeting-scribe start [--port 8080] [--debug] [--foreground]
meeting-scribe stop
meeting-scribe restart            # with smoke-test
meeting-scribe status
meeting-scribe logs [-f]
meeting-scribe gb10 up            # Start model backend containers
meeting-scribe gb10 down          # Stop model backend containers
meeting-scribe wifi up --mode {admin,meeting}   # Bring up WiFi hotspot
meeting-scribe wifi down
meeting-scribe precommit          # Scan the working tree for sensitive data before commit
```

## Per-Meeting Artifacts

```
meetings/{id}/
  meta.json              # Meeting state + metadata
  room.json              # Table + seat layout
  speakers.json          # Enrolled speaker embeddings
  journal.jsonl          # Append-only transcript events
  polished.json          # Refinement worker output
  detected_speakers.json # Auto-detected speakers
  timeline.json          # Segment manifest for podcast player
  audio/recording.pcm    # Time-aligned s16le 16kHz mono
```

## WiFi Hotspot

Meeting-scribe manages the WiFi AP directly — no separate `sddc-cli` dependency required. The GB10's MT7925 radio supports one AP at a time in three modes:

| Mode | SSID | Captive Portal | Admin UI over WiFi |
|------|------|---------------|--------------------|
| **off** | No AP | No | No |
| **meeting** | Rotating per session (e.g. "Dell Demo A7F3") | Yes — auto-redirect on connect | No — LAN only |
| **admin** | Fixed (configurable, default "Dell Admin") | No | Yes — `https://10.42.0.1:8080/` |

### CLI

```bash
# Admin mode — fixed SSID, admin UI reachable over WiFi
meeting-scribe wifi up --mode admin

# Meeting mode — rotating SSID, captive portal, guest isolation
meeting-scribe wifi up --mode meeting

# Check live status (reads from nmcli/wpa_cli, not just state file)
meeting-scribe wifi status

# Tear down (persists wifi_mode=off, survives reboot)
meeting-scribe wifi down
```

### Admin Settings Panel

The WiFi mode, admin SSID, and admin password are configurable from the admin UI settings panel at `https://<mgmt-ip>:8080/` (gear icon). Changing the mode triggers an async AP reconfiguration (returns HTTP 202, UI polls until the switch completes). The admin password is write-only — it is never returned in API GET responses.

### Auto-Bring-Up on Boot

If `wifi_mode` in `~/.config/meeting-scribe/settings.json` is not `"off"`, meeting-scribe automatically brings up the WiFi AP in the configured mode during server startup (in the FastAPI lifespan). No separate `hotspot up` step is needed.

### Security

- **WPA3-SAE** with PMF required on all modes (IEEE 802.11w). Devices older than ~2019 (iOS 13, Android 10, Windows 10 1903) cannot associate.
- **Admin mode firewall**: allows DHCP (67), DNS (53), HTTP (80), admin HTTPS (8080). Rejects port 443 with TCP RST. Default-deny everything else from the hotspot subnet.
- **Meeting mode firewall**: same as admin but port 8080 is REJECT (admin UI blocked from WiFi clients). Port 443 is REJECT. Captive portal DNS wildcard + DHCP option 114 (RFC 8910) for instant portal detection.

### Settings Storage

| Field | Location | Description |
|-------|----------|-------------|
| `wifi_mode` | `~/.config/meeting-scribe/settings.json` | `"off"`, `"meeting"`, or `"admin"` |
| `admin_ssid` | `~/.config/meeting-scribe/settings.json` | Fixed SSID for admin mode (default: "Dell Admin") |
| `admin_password` | `~/.config/meeting-scribe/settings.json` | WPA3 passphrase for admin mode (8-63 chars, auto-generated on first use) |
| `wifi_regdomain` | `~/.config/meeting-scribe/settings.json` | 2-letter country code (default: JP) |

## License

MIT — see [LICENSE](LICENSE). Anyone is free to use, fork, and modify this
software.

## Contributions

This repository is published for consumption, not co-development. Pull
requests, feature requests, and issues from external contributors are **not
accepted**. Fork freely — you own your fork.
