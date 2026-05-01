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
| Diarization | pyannote.audio 4.0 (`speaker-diarization-community-1`) | Custom container | 8001 |
| Name extraction | Qwen3.6-35B-A3B-FP8 (reuses translation vLLM) | vLLM | 8010 |
| Refinement worker — *off by default* | Qwen3.6-35B-A3B-FP8 (reuses translation vLLM) | vLLM | 8010 |

The 35B FP8 translation model is the heaviest component (≈35 GB VRAM when loaded + KV cache). Combined footprint runs ≈43 GB, well inside the GB10's 128 GB unified pool. The translate vLLM endpoint is also the primary sharing point if you run [auto-sre](https://github.com/sddcinfo/auto-sre) on the same box — both point at `:8010`.

## Features

- **Multi-language ASR** — Qwen3-ASR with 52-language support + auto-detection. Per-language quality verified on 19 of the 20 (see [`benchmarks/results/asr-language-matrix.md`](benchmarks/results/asr-language-matrix.md) — most languages clear ≤5% p50 normalized error on Fleurs). Malay (`ms`) is best-effort: Fleurs has no Malay split, so we score Indonesian (`id`) as a proxy.
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

## Prerequisites

A "fresh GB10" for this README means:

- **NVIDIA GB10** (DGX Spark) running aarch64 Linux. `nvidia-smi` must work — driver ≥ 580 + CUDA 13.
- **Docker** installed, with your user in the `docker` group so `docker ps` runs without sudo. Bootstrap does not configure docker for you.
- **A HuggingFace token** with the gated models below accepted in a browser. Bootstrap prompts for it (or pre-set `HF_TOKEN` for unattended installs).
- **`sudo` access** for the invoking user. Bootstrap runs as root and configures everything else (apt packages, capability grants, scoped passwordless sudo for the service, systemd units, wifi radio, regdomain).

### HuggingFace gated models

Open each URL once in a browser, click *Agree and access*:

- <https://huggingface.co/pyannote/speaker-diarization-community-1> — diarization (required, fails the loudest)
- <https://huggingface.co/pyannote/segmentation-3.0> — diarization dependency
- <https://huggingface.co/Qwen/Qwen3-ASR-1.7B> — ASR
- <https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8> — translation (heaviest pull)
- <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base> — TTS

Then create a token at <https://huggingface.co/settings/tokens> with *Read* scope.

### First-run timing

Plan **~30–50 minutes** of disk I/O on first install: ~24 GB for
the `vllm/vllm-openai` base image, ~90 GB of HF model weights
(ASR + TTS + diarization + the autosre-managed Qwen3.6-35B-A3B-FP8
translate model — `meeting-scribe gb10 pull-models` defaults to
`--include-shared` so the install is fully self-contained), plus
container build layers and editable-install layers. Subsequent
runs are idempotent and re-use everything cached.

## Quick Start

One canonical command on a fresh GB10:

```bash
git clone https://github.com/sddcinfo/meeting-scribe.git
cd meeting-scribe && sudo ./bootstrap.sh
```

That's the wipe-and-reinstall flow. Bootstrap exits non-zero if any
post-install validation phase fails — there is no "partial install"
state where the customer has to chase down what went wrong.

What `sudo ./bootstrap.sh` does, in order:

1. **Platform gate** — aarch64 Linux + `nvidia-smi` available.
2. **Locale prompts** — country code (wifi regdomain), timezone,
   default language pair. Defaults are auto-detected from the
   system; press Enter to accept or type to override. Pass
   `--yes` (or set `MEETING_SCRIBE_YES=1`) to accept defaults
   for unattended installs.
3. **OS packages** — `apt install` ffmpeg, libportaudio, rfkill, iw, etc.
4. **WiFi radio** — `rfkill unblock wifi` + `nmcli radio wifi on`
   (two independent kill switches). Persists the regdomain so
   subsequent reboots keep the right country code.
5. **Scoped sudoers** — installs `/etc/sudoers.d/meeting-scribe`
   granting passwordless sudo *only* on `nmcli`, `iw`, `rfkill`,
   `iptables`, `ip6tables`, `setcap`. Validated with `visudo -c`
   before being moved into place.
6. **Python toolchain + venv** — runs as the invoking user via
   `sudo -u $SUDO_USER` so file ownership is correct. mise pins
   Python 3.14, creates `.venv`, editable-installs meeting-scribe.
7. **Sister-clone auto-sre** — clones into `../auto-sre`, sets up
   its venv, installs the auto-sre user systemd unit so the
   translate vLLM autostarts on boot.
8. **App-layer config** — `meeting-scribe setup` runs (HF token
   prompt if not in env, TLS cert generation, port-80
   `cap_net_bind_service`).
9. **HF model pre-pull** — downloads ~80 GB of model weights to
   `/data/huggingface` so containers boot offline.
10. **Container builds** — pyannote-diarize, qwen3-tts, vllm-asr.
11. **Backend startup** — `meeting-scribe gb10 up` brings the four
    in-tree containers to healthy.
12. **systemd units** — installs `meeting-scribe.service` as a
    user unit, runs `loginctl enable-linger` so it autostarts at
    boot before any console login.
13. **Server start + acceptance gate** — starts
    `meeting-scribe.service` and runs `meeting-scribe validate
    --customer-flow` (see [Verify](#verify) below). Bootstrap
    exits with the validator's return code.

The only manual step left after bootstrap returns is **`autosre
start --no-scribe`** to cold-load the 35 B FP8 translate model (3-7
minutes). Or just reboot — the autosre unit is enabled, so systemd
brings it up automatically on next boot.

### Verify

The acceptance gate runs at the end of bootstrap. Re-run it any
time on the device:

```bash
meeting-scribe validate --customer-flow
```

Seven phases, each guarding a specific failure mode that has bitten
us in production:

| Phase | What it checks |
|-------|----------------|
| `systemd_unit` | `meeting-scribe.service` is `active` under the user manager |
| `wifi_radio` | rfkill clear AND `nmcli radio wifi=enabled` (both kill switches) |
| `sudoers_d` | `/etc/sudoers.d/meeting-scribe` is installed |
| `multipart_dep` | `python_multipart` importable (PPTX upload depends on it) |
| `status_latency` | 3 back-to-back `GET /api/status`, p95 < 500ms |
| `slides_upload` | Real multipart POST of `tests/fixtures/test_slides.pptx`; fails only on the python-multipart assertion |
| `meeting_qr` | Starts a temp meeting, polls `/api/meeting/wifi` for up to 60s, asserts SSID + password + QR SVG come back, cancels the meeting |

`VALIDATE GREEN` = customer-ready. Any `FAIL` exits non-zero with
a remediation hint. `meeting-scribe validate --quick` is the older
backend-liveness sweep and is still available for quick diagnostic
sweeps that don't need to start a meeting.

### Unattended installs (CI / re-imaging)

```bash
HF_TOKEN=hf_xxx MEETING_SCRIBE_YES=1 sudo -E ./bootstrap.sh
```

`-E` preserves `HF_TOKEN` through the `sudo` boundary; `MEETING_SCRIBE_YES=1`
accepts every prompt with its auto-detected default. Bootstrap still exits
non-zero if `validate --customer-flow` fails, so CI gets a hard signal
either way.

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

## Customer Install Media

Build a wipe-and-reinstall USB / PXE bundle that runs the entire flow above
without any human keystrokes after the F7 boot prompt — useful for shipping
a turnkey GB10 to a customer or re-imaging a borked system. Implemented as
three `sddc` subcommands; see `~/.claude/plans/hashed-giggling-rivest.md`
for the full design.

### One-time prerequisites (on the build machine)

```bash
sudo apt install -y xorriso whois        # whois → mkpasswd
```

### Three commands end-to-end

```bash
# 1. Fetch the Dell DGX OS recovery ISO (~9.21 GB, public Akamai mirror)
sddc dgx-iso fetch \
  --service-tag <YOUR-SERVICE-TAG> \
  --url 'https://dl.dell.com/FOLDER<id>M/1/Dell_Pro_Max_with_GB10_FCM1253_<region>_DGX_OS7_A01_Recovery_Image.iso' \
  --save-mapping \
  --out ~/Downloads/dgx-os.iso
# Find the URL on dell.com/support → enter service tag → Drivers & Downloads
# → "Operating System" → DGX OS Reinstallation Image. The first run with
# --save-mapping caches it; subsequent fetches just need --service-tag.

# 2. Build the customized install ISO
sddc build-customer-iso \
  --base-iso ~/Downloads/dgx-os.iso \
  --out      ./dist/meeting-scribe-customer.iso \
  --scribe-ref main
# Default: SSH disabled on the installed system. Verify success via local
# console (cat /var/log/meeting-scribe-firstboot.json). Add
# `--ssh-key ~/.ssh/id_ed25519.pub` for permanent remote access, or
# `--ephemeral-validation-key` for a one-shot key auto-removed after success.

# 3. Flash to USB (or stage dist/pxe-bundle/ on a netboot HTTP root)
sddc flash-usb /dev/sdX ./dist/meeting-scribe-customer.iso
```

Plug the USB into the customer GB10, press **F7** at the Dell logo, select
the USB. Walk away. ≤30 minutes later: `validate --customer-flow` has run
under the installer, the result is at `/var/log/meeting-scribe-firstboot.json`
on the installed disk, and the system is ready for the one remaining manual
step (`autosre start --no-scribe` with `HF_TOKEN` for the first vLLM cold-load).

### What the ISO does

1. Subiquity autoinstall partitions the disk and lays down DGX OS 7.x with
   `linux-nvidia-64k`, the operator's chosen locale/keyboard, and the
   `meetingscribe` user (force-change console password, no remote password).
2. A one-shot `meeting-scribe-firstboot.service` waits for connectivity
   (DNS + curl probe of GitHub/PyPI/Docker Hub/mise + `apt-get update`),
   clones meeting-scribe at the **commit SHA pinned at build time**,
   verifies the SHA matches, then runs `bootstrap.sh` in unattended mode
   and `validate --customer-flow` as the acceptance gate.
3. On success it writes `/var/lib/meeting-scribe/firstboot.done` and
   disables itself. Transient network failures retry up to 10× per hour
   via `Restart=on-failure`.

### Security defaults

- **No remote SSH on fresh install.** `sshd` is masked unless you opt in.
- **No vendor SSH keys baked in.** Every authorized key is explicit
  (`--ssh-key` repeatable; no auto-discovery from `~/.ssh/`).
- **Pinned-SHA supply chain.** The build resolves the requested ref to a
  commit SHA via `git ls-remote` and bakes that exact SHA into the
  firstboot script. Firstboot aborts with exit 11 if the cloned HEAD
  doesn't match — moved tags or repo takeover fail closed.
- **No blanket sudo grant during install.** Firstboot runs as root and
  uses `bootstrap.sh`'s native dual-mode (`SUDO_USER` env) to drop
  privileges. There's no temporary `/etc/sudoers.d/...` NOPASSWD entry
  that could leak on failure.
- **Apt connectivity probed by apt itself.** `apt-get update` honors
  every `Acquire::*::Proxy` setting natively — including path-prefixed
  proxies that a curl-based probe would mishandle.

### PXE bundle

`dist/pxe-bundle/` contains `customer.iso`, `vmlinuz`, `initrd`,
`user-data`, `meta-data`, `boot.cfg.snippet`, and a `README.md` documenting
how to wire it into iPXE or GRUB-via-HTTP. Same artifact either way — the
kernel/initrd boot, then mount the served ISO via `iso-scan/filename=`.

### Secure Boot

**No need to disable it.** The build preserves the Secure Boot trust chain
intact:

- shim (MS-signed `bootaa64.efi`) — byte-identical to base ISO
- grub (Canonical-signed `grubaa64.efi`) — byte-identical
- MOK Manager (`mmaa64.efi`) — byte-identical
- Kernel (Canonical-signed `/casper/vmlinuz`) — byte-identical

Ubuntu's chain validates the **binaries** (firmware → shim → grub → kernel
→ modules). `grub.cfg` and the autoinstall payload we add are plain text,
not signature-verified — so our edits don't break anything. The build
asserts byte-identity of every signed binary as a final acceptance gate
and fails loudly if any of them changed (catches future regressions where
someone accidentally swaps a signed file).

### Troubleshooting on the customer side

If the install never finishes: SSH or console in, then:

```bash
cat /var/log/meeting-scribe-firstboot.json | jq .
journalctl -u meeting-scribe-firstboot.service --no-pager
cat /var/log/meeting-scribe-firstboot.log | tail -100
```

Exit codes: `10` = connectivity exhausted retries, `11` = clone SHA mismatch
(supply-chain failure), `12` = `validate --customer-flow` did not pass,
`13` = no apt sources configured (structural). The unit is `Restart=on-failure`,
so transient connectivity failures retry up to 10× per hour automatically.

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
meeting-scribe restart                  # with smoke-test
meeting-scribe status
meeting-scribe logs [-f]
meeting-scribe gb10 up                  # Start model backend containers
meeting-scribe gb10 down                # Stop model backend containers
meeting-scribe gb10 status              # Health check all backends
meeting-scribe gb10 pull-models         # Download required HF models
meeting-scribe wifi up --mode {admin,meeting}   # Bring up WiFi hotspot
meeting-scribe wifi down
meeting-scribe wifi status              # Live nmcli/wpa_cli state
meeting-scribe validate [--quick|--full|--e2e]  # End-to-end backend health + quality probes
meeting-scribe demo-smoke [--host HOST]  # Real e2e gate: meeting + PPTX upload + render + translate
meeting-scribe hf-probe                 # Validate HF token + model EULA acceptance
meeting-scribe doctor watch-pressure    # Tail /proc/pressure/memory with severity classification
meeting-scribe versions list -m <id>    # List reprocess snapshots for a meeting
meeting-scribe versions diff -m <id>    # Diff a snapshot against the current state
meeting-scribe precommit                # Scan the working tree for sensitive data before commit
```

### Pre-flight: HuggingFace credential check

Before any 30 GB model download, validate your token and per-model
EULA acceptance — saves you a 20-minute round-trip on a 401/403:

```bash
# Interactive (reads HF_TOKEN from .env, prints the EULA URLs you may need):
meeting-scribe hf-probe

# Orchestrator/CI (token via stdin, machine-readable JSON, exit 0/64/65):
printf '%s' "$HF_TOKEN" | meeting-scribe hf-probe --read-token-from-stdin --json
```

The `setup` first-run prompt now lists every gated-model URL and
validates the token you paste before saving it to `.env`. See
plan §1 of `plans/steady-plotting-eich.md` for the full design.

### Real demo gate: `demo-smoke`

`meeting-scribe validate --quick` only checks backend liveness.
`demo-smoke` drives the actual demo path: starts a meeting, uploads
a real PPTX fixture, polls until the deck renders + translates, then
fires a live translate probe at the autosre vLLM. Target wall-time
≤30 s on warm GPU. Use this as the acceptance gate after every cold
install or post-upgrade restart:

```bash
meeting-scribe demo-smoke --host 192.168.1.100
```

## Per-Meeting Artifacts

```
meetings/{id}/
  meta.json                      # Meeting state + metadata
  room.json                      # Table + seat layout
  speakers.json                  # Enrolled speaker embeddings
  journal.jsonl                  # Append-only transcript events
  detected_speakers.json         # Auto-detected speakers
  timeline.json                  # Segment manifest for podcast player
  summary.json                   # Eager-computed AI summary (refreshed on finalize)
  speaker_lanes.json             # Per-speaker timeline lanes (+ _exclusive variant)
  polished.json                  # Refinement worker output (when SCRIBE_ENABLE_REFINEMENT=1)
  audio/
    recording.pcm                # Time-aligned s16le 16kHz mono
  slides/                        # Present only if a deck was uploaded
    {deck_id}/                   # Source PPTX, rendered thumbnails, translated text
  versions/                      # Reprocess snapshots (one subdir per run)
    {ts}__{label}/
      manifest.json              # Snapshot inputs + git hash
      journal.jsonl              # Snapshot of pre-reprocess artifacts
      summary.json
      timeline.json
      detected_speakers.json
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

### Third-party model licenses

The MIT license above covers **this repository's source code only**. The
HuggingFace models that meeting-scribe pulls and runs are governed by their
own upstream licenses, summarized below.

> **Scope:** this table reflects the **specific model IDs and versions
> currently wired into this repo** (see [Model Stack](#model-stack) and
> [HuggingFace gated models](#huggingface-gated-models)). If you swap a
> model for a different revision, sibling variant (e.g. `-Instruct` vs
> `-Base`), or a successor release, **re-verify the license** — both Qwen
> and Cohere maintain other model lines under non-Apache terms (custom
> "Qwen License", C4AI CC-BY-NC, etc.) and pyannote ships premium
> pipelines under separate commercial terms. Always check the model card
> before upgrading.

| Model | License | Gated | Commercial use | Key obligations |
|-------|---------|-------|----------------|-----------------|
| `pyannote/speaker-diarization-community-1` | **CC-BY-4.0** | Yes (contact-info acceptance) | ✅ | Attribution required; credit pyannote and link the license |
| `Qwen/Qwen3-ASR-1.7B` | **Apache-2.0** | No | ✅ | Preserve license + NOTICE on redistribution |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | **Apache-2.0** | No | ✅ | Preserve license + NOTICE on redistribution |
| `Qwen/Qwen3.6-35B-A3B-FP8` | **Apache-2.0** | No | ✅ | Preserve license + NOTICE on redistribution |
| `CohereLabs/cohere-transcribe-03-2026` *(bench-only, not in production stack)* | **Apache-2.0** | Yes (contact-info acceptance) | ✅ | Preserve license + NOTICE; contact-info acceptance is a separate click-through, not a license restriction |

Notes:

- **All five permit commercial use** with attribution. None are copyleft
  or NonCommercial.
- **Two are gated** (pyannote, Cohere). A HuggingFace token with prior
  in-browser acceptance is required — see
  [HuggingFace gated models](#huggingface-gated-models). The `-release`
  suffix sometimes seen on the Cohere ID is not the canonical model card;
  the bare `cohere-transcribe-03-2026` is the gated artifact.
- **CC-BY-4.0 (pyannote) is the only attribution-strict license** in the
  set — if you ship a product UI, an "About" / credits screen
  acknowledging pyannote satisfies it.
- The Cohere model is currently used **only in the `2026-Q2` benchmark
  Track A** (`containers/cohere-transcribe/`, `docker-compose.gb10.yml`)
  and is not loaded by the production stack.

## Contributions

This repository is published for consumption, not co-development. Pull
requests, feature requests, and issues from external contributors are **not
accepted**. Fork freely — you own your fork.
