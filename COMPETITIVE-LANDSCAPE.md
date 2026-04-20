# meeting-scribe Competitive Landscape Analysis

*Last updated: 2026-04-09*

## Executive Summary

**No product — commercial or open-source — combines all of meeting-scribe's capabilities in one package.** The intersection of real-time transcription + real-time translation + voice cloning + self-hosted + GPU-optimized edge deployment is genuine whitespace.

---

## Head-to-Head Comparisons

### vs. Cloud Commercial Products

| Feature | meeting-scribe | Otter.ai | Fireflies | Fathom | Granola | Gong |
|---|---|---|---|---|---|---|
| Real-time transcription | Yes | Yes | Yes | Yes | Yes | Yes |
| Speaker diarization | Yes (pyannote) | Yes | Yes | Yes | Yes | Yes |
| AI summaries + action items | Yes | Yes | Yes | Yes | Yes | Yes |
| **Real-time translation** | **Yes (20 langs)** | Limited | No | No | No | No |
| **Voice cloning TTS** | **Yes (zero-shot)** | No | No | No | No | No |
| **Per-client language stream** | **Yes** | No | No | No | No | No |
| **100% self-hosted** | **Yes** | No | No | No | No | No |
| **No data leaves device** | **Yes** | No | No | No | Partial | No |
| Physical room layout editor | Yes | No | No | No | No | No |
| WiFi hotspot + captive portal | Yes | No | No | No | No | No |
| Audio playback with sync | Yes | Yes | Yes | Yes | No | Yes |
| No bot in meeting | Yes (local audio) | No (bot) | No (bot) | No (bot) | Yes | No (bot) |
| Pricing | Free (MIT) | $8-20/mo | $10-19/mo | Free-$15/mo | $14-35/mo | ~$250/user/mo |

### vs. Open-Source Alternatives

| Feature | meeting-scribe | Meetily (7K stars) | Char (Hyprnote) | TellMeMo | WhisperX |
|---|---|---|---|---|---|
| Real-time transcription | Yes | Yes | Yes | Yes | Yes |
| Speaker diarization | Yes | Yes | No | No | Yes |
| AI summaries | Yes | Yes (Ollama/Claude) | Yes | Yes | No |
| **Real-time translation** | **Yes (20 langs)** | No | No | No | No |
| **Voice cloning TTS** | **Yes** | No | No | No | No |
| **Per-client interpretation** | **Yes** | No | No | No | No |
| **Physical room layout** | **Yes** | No | No | No | No |
| **Captive portal + hotspot** | **Yes** | No | No | No | No |
| **Rolling refinement** | **Yes** | No | No | No | No |
| GPU-optimized deployment | Yes (GB10 128GB) | Partial (CUDA) | No (local CPU) | No (cloud API) | Yes (CUDA) |
| Complete meeting app | Yes | Yes | Yes | Yes | No (library) |
| Web UI | Yes | Yes (Tauri) | No (native Mac) | Yes | No |

### vs. Real-Time Translation Products

| Feature | meeting-scribe | DeepL Voice | Wordly | KUDO | Interprefy |
|---|---|---|---|---|---|
| Real-time translation | Yes | Yes | Yes | Yes | Yes |
| **Voice cloning output** | **Yes** | No | No | No | No |
| Self-hosted | Yes | No | No | No | No |
| Transcription + recording | Yes | Partial | Yes | Yes | Yes |
| Speaker diarization | Yes | No | No | No | No |
| AI summaries | Yes | No | No | No | No |
| Language count | 20 | ~15 | 3000+ pairs | 200+ | 200+ |
| Offline capable | Yes | No | No | No | No |
| Pricing | Free (MIT) | Enterprise | $0.08-0.30/word | Enterprise | Enterprise |

### vs. NVIDIA Riva (Closest Infrastructure Comparison)

| Feature | meeting-scribe | NVIDIA Riva |
|---|---|---|
| GPU-accelerated ASR/TTS/NMT | Yes | Yes |
| Complete meeting application | Yes | No (toolkit only) |
| Meeting UI / room layout | Yes | No |
| Speaker diarization | Yes | Yes |
| Voice cloning | Yes (zero-shot, 3s reference) | No (requires custom training) |
| Self-hosted | Yes | Yes |
| Edge deployment ready | Yes (GB10) | Yes (Jetson/data center) |
| Open source | Yes (MIT) | No (proprietary SDK) |
| Setup effort | `meeting-scribe setup` | Significant integration work |

---

## Features Unique to meeting-scribe

No competitor offers any of the following:

1. **Zero-shot voice cloning in translated speech** — Translations spoken back in the original speaker's voice from a 3-second reference. DeepL Voice and Wordly output generic TTS voices.

2. **Physical room layout editor with 8 presets** — Designed for in-person meetings with a physical room concept (boardroom, round, square, rectangle, classroom, U-shape, pods, free-form). No competitor models the physical space.

3. **WiFi hotspot + captive portal + QR code guest access** — A complete "bring your own network" appliance mode. Guests connect to the device's WiFi and get a read-only transcript view. Compatible with Apple/Windows/Android captive portal detection.

4. **Rolling refinement worker** — Trails 45s behind live transcription to re-transcribe with full audio context, then re-translates polished segments. This "draft then polish" pipeline is unique.

5. **GB10 edge appliance deployment** — Purpose-built for a single 128GB unified-memory GPU node as a standalone meeting device. No competitor targets this form factor.

6. **Per-client real-time interpretation with voice cloning** — Each browser client picks their language and receives an independent TTS audio stream in the speaker's cloned voice. This is simultaneous interpretation, self-hosted, with voice cloning — nothing else comes close.

---

## Competitor Detail

### Tier 1 — Cloud Market Leaders

**Fireflies.ai** — Recording, transcription, AI summaries, action items, CRM automation (Salesforce, HubSpot). 60+ languages. Free (800 min storage); Pro $10/user/mo; Business $19/user/mo. Integrates with Zoom, Meet, Teams, Webex, Slack, and 50+ apps.

**Otter.ai** — Real-time transcription, conversational search, highlights, speaker diarization. English primary with limited live translation (Spanish/French/German to English). Free (300 min/mo); Pro $8.33/mo; Business $20/user/mo.

**Fathom** — Transcription, AI summaries, action items, CRM sync. 28+ languages. Free forever for individuals (unlimited); Team $15/user/mo. Originally Zoom-centric, now broader.

**Granola** — Desktop AI notepad, local audio capture, no visible bot. Raised $125M Series C at $1.5B valuation (March 2026). Audio not stored; only text retained. Free (basic); Business $14/user/mo; Enterprise $35/user/mo.

### Tier 2 — Strong Contenders

**Krisp** — Noise cancellation (on-device) + transcription (16 languages) + AI summaries. Works system-wide with any conferencing app. Free (60 min/day); Core $8/user/mo.

**tl;dv** — Recording, transcription, video clips, CRM integration. 30+ languages. Most generous free tier for recordings. GDPR compliant (EU data storage). Free (unlimited); Pro $18/user/mo.

**Fellow** — Meeting agendas, action item tracking, AI notes, 1-on-1 management. SOC 2/GDPR. Free; Team $7/user/mo; Business $15/user/mo. 50+ native integrations.

**Read.ai** — Engagement analytics (participation, talk time, sentiment), summaries, action items. Pro ~$15/user/mo. Unique angle on meeting engagement measurement.

### Tier 3 — Specialized / Niche

**Avoma** — Full meeting lifecycle, conversation intelligence, revenue intelligence, MEDDIC/SPICED coaching. $29-39/recorder/mo. Sales team focus.

**Sembly** — Transcription, summaries, speaker ID, 45+ languages. $10-20/mo.

**Notta** — 58+ language transcription, real-time translation to 42+ languages, offline mode. Free (120 min/mo); Pro ~$26/mo.

**Tactiq** — Chrome extension (no bot), real-time transcription, AI insights. Free (10 transcripts/mo); Pro $8-12/user/mo.

### Tier 4 — Enterprise Revenue Intelligence

**Gong** — Revenue intelligence, deal tracking, forecasting, conversation analytics, 70+ languages. ~$250/user/mo + $5K-$50K annual platform fee. The 800-pound gorilla.

**Chorus (ZoomInfo)** — Conversation intelligence, 60+ languages. Pro $10/user/mo; Enterprise $39/user/mo. 50-60% cheaper than Gong.

### Open-Source Alternatives

**Meetily** (MIT, 7K+ GitHub stars) — Real-time transcription (Whisper/Parakeet), speaker diarization, Ollama/Claude summarization, Rust backend. Fully self-hosted. Min 8GB RAM; recommended 16GB+/GPU. No translation.

**Char** (GPL-3.0, formerly Hyprnote) — Local-first AI notepad, real-time transcription, plugin system, BYOLLM via Ollama. Native Mac app; Windows/Linux Q2 2026. Free (local); managed $25/mo.

**TellMeMo** (open source) — Real-time Q&A during meetings, RAG search across meetings, action tracking, Docker deployment. ~$0.006/min via cloud API.

**WhisperX** (BSD-4) — 70x realtime transcription, word-level timestamps, speaker diarization (pyannote), VAD. Library/toolkit, not a complete meeting app.

**faster-whisper** (MIT) — CTranslate2-based Whisper, 4x faster, lower memory. Foundation for many other tools.

**Vosk** (Apache 2.0) — Lightweight offline recognition, 20+ languages, runs on Raspberry Pi. Lower accuracy than Whisper.

### Real-Time Translation Products

**DeepL Voice** — Real-time speech translation in meetings, each participant sees translated captions. Cloud but privacy-conscious (data deleted after session). Microsoft Teams, Zoom. Enterprise pricing.

**Wordly** — First 100% AI simultaneous interpretation platform, 3,000+ language pairs. Cloud. $0.08-0.30/word.

**KUDO** — AI + human interpreter hybrid, 200 languages, QR code access. Cloud. Enterprise pricing.

**Interprefy** — Remote simultaneous interpretation, AI speech translation, hybrid AI/human. Cloud. Enterprise pricing.

### GPU / Edge AI Infrastructure

**NVIDIA Riva** — GPU-accelerated multilingual speech microservices (ASR/TTS/NMT), 26+ languages. Data center, on-premise, cloud, edge. Enterprise pricing. Most mature enterprise edge speech stack, but toolkit only — no meeting UX.

**NVIDIA Parakeet** — Speed-optimized ASR models. TDT 0.6B: RTFx >2,000. RNNT 1.1B: 1.8% WER on LibriSpeech. TDT v3: 25 European languages. Self-hosted via NIM containers.

---

## Where Competitors Have the Edge

| Gap | Who Does It Better | Severity |
|---|---|---|
| Language count | Wordly (3000+ pairs), Gong (70+), Fireflies (60+) | Medium — 20 covers most business needs |
| Calendar/conferencing integrations | Fireflies, Otter, Fathom (Zoom/Meet/Teams bots) | Low — meeting-scribe targets in-person, not virtual |
| CRM sync | Gong, Avoma, Fireflies (Salesforce, HubSpot) | Low — different use case |
| Mobile apps | Otter, Fireflies, Plaud NotePin | Medium — no native mobile client |
| Multi-meeting search/RAG | TellMeMo, Otter ("ask questions about all meetings") | Low-Medium |
| Team collaboration | Granola Spaces, Fellow, tl;dv clips | Low — different audience |
| Concurrent meetings | All cloud tools (unlimited) | Medium — single meeting limitation |
| Setup simplicity | All cloud tools (sign up and go) | Expected for self-hosted |

---

## Market Position

meeting-scribe occupies a completely unserved intersection:

```
                    Cloud ────────────────── Self-Hosted
                      │                          │
  Transcription-only  │  Otter, Fireflies        │  Meetily, Char
                      │  Fathom, Granola         │  WhisperX
                      │                          │
  + Translation       │  DeepL Voice, Wordly     │  ← EMPTY →
                      │  KUDO, Interprefy        │
                      │                          │
  + Voice Cloning     │  ← EMPTY →               │  meeting-scribe ★
  + Room Layout       │                          │
  + Edge Appliance    │                          │
```

The closest approximation would be integrating NVIDIA Riva + a custom meeting UI + SeamlessM4T + a TTS voice cloning model — months of integration work that meeting-scribe packages as a single `meeting-scribe setup` command.

---

## Key Market Trends (2026)

- **Privacy demand surging**: The Cluely data breach (2025, 83K users' recordings exposed) accelerated local-first demand. Global privacy regulations now cover ~75% of world population.
- **Local quality at parity**: On-device transcription accuracy has reached cloud parity in 2026 thanks to models like Whisper Large-v3 Turbo, Parakeet, and Qwen3-ASR.
- **Granola's $1.5B valuation** validates the "no bot, local capture" approach.
- **Enterprise translation tools remain cloud-only and expensive** — no self-hosted option exists for real-time meeting translation besides meeting-scribe.
- **Edge AI hardware maturing**: NVIDIA Jetson T4000 (1200 FP4 TFLOPs, 64GB), GB10/DGX Spark (128GB unified) make on-device inference practical for full meeting stacks.
