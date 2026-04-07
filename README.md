# Meeting Scribe

Real-time bilingual (JA/EN) meeting transcription with automatic translation, voice cloning, speaker identification, and full audio recording.

## Features

- **Real-time ASR** — Qwen3-ASR-0.6B (MLX) with auto language detection (52 languages)
- **Bidirectional translation** — NLLB-200 1.3B (CPU) or vLLM (GPU) for JA↔EN
- **Voice cloning TTS** — Speaks translations in the original speaker's voice (Qwen3-TTS)
- **3-column live view** — Live (both languages), English-only, Japanese-only
- **Speaker identification** — Auto-detects self-introductions ("my name is X")
- **Audio recording** — Time-aligned PCM with segment-level playback + podcast player
- **Room setup** — Drag-and-drop table/seat editor with 8 presets
- **Meeting history** — Browse, replay past meetings with full transcript + audio
- **Zero build step** — Static HTML/CSS/JS, instant refresh during development

## Platform Capabilities

| Feature | MacBook (MLX) | GB10 (Primary) |
|---------|--------------|----------------|
| ASR | Qwen3-ASR 0.6B | vLLM Qwen3-ASR 0.6B/1.7B |
| Translation | NLLB-200 1.3B (CPU, 500ms) | vLLM Qwen3.5-35B (GPU, 400ms) |
| Voice Clone TTS | Qwen3-TTS 0.6B (~1-2s) | vLLM-Omni Qwen3-TTS (97ms) |
| Diarization | Name detection only | Sortformer (real speaker ID) |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sddcinfo/meeting-scribe.git
cd meeting-scribe
python3 -m venv .venv
.venv/bin/pip install -e .

# Convert translation model (one-time)
.venv/bin/ct2-transformers-converter \
  --model facebook/nllb-200-distilled-1.3B \
  --output_dir ~/.cache/nllb-200-distilled-1.3B-ct2-int8 \
  --quantization int8

# Start
.venv/bin/meeting-scribe start
# Open http://localhost:8080
```

## Architecture

```
Browser (AudioWorklet, 16kHz s16le)
  → WebSocket
  → FastAPI Server
    → Qwen3-ASR (MLX, batch inference)
    → NLLB-200 translation (CTranslate2, CPU)
    → AudioWriter (time-aligned PCM)
  → WebSocket (TranscriptEvent JSON)
  → 3-column GridRenderer
```

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRIBE_PORT` | 8080 | Server port |
| `SCRIBE_ASR_MODEL` | medium | Whisper model (tiny/medium/large-v3-turbo) |
| `SCRIBE_TRANSLATE_BACKEND` | ct2 | Translation: ct2, vllm, or ollama |
| `SCRIBE_TRANSLATE_VLLM_URL` | http://localhost:8000 | vLLM endpoint (for GPU translation) |
| `SCRIBE_DIARIZE` | false | Enable speaker diarization (requires GPU) |

## CLI

```bash
.venv/bin/meeting-scribe start [--port 8080] [--debug]
.venv/bin/meeting-scribe stop
.venv/bin/meeting-scribe status
.venv/bin/meeting-scribe logs [-f]
```

## Tests

```bash
# Start server first, then:
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -v
```

19 tests covering: server health, room layout CRUD, meeting lifecycle, EN/JA transcription, translation, audio recording/playback, language filtering, hallucination detection, speaker name detection.

## GB10 Production

For NVIDIA DGX Spark (GB10) with vLLM:

```bash
SCRIBE_TRANSLATE_BACKEND=vllm \
SCRIBE_TRANSLATE_VLLM_URL=http://<gb10-ip>:8000 \
SCRIBE_ASR_MODEL=large-v3-turbo \
.venv/bin/meeting-scribe start
```

## Per-Meeting Artifacts

```
meetings/{id}/
  meta.json              # Meeting state + metadata
  room.json              # Table + seat layout
  speakers.json          # Enrolled speaker embeddings
  journal.jsonl          # Append-only transcript events
  detected_speakers.json # Auto-detected speakers
  timeline.json          # Segment manifest for podcast player
  audio/recording.pcm    # Time-aligned s16le 16kHz mono
```

## License

MIT
