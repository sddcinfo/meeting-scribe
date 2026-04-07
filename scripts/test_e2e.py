"""End-to-end test: send audio via WebSocket, verify transcript events come back.

Generates a test tone (or uses a provided WAV file), packages it in the
wire format, sends via WebSocket, and prints received transcript events.

Usage:
    python scripts/test_e2e.py                    # Generate test tone
    python scripts/test_e2e.py --file speech.wav  # Use real audio file
    python scripts/test_e2e.py --tts "こんにちは"   # Use macOS TTS
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

SERVER = "http://localhost:8080"
WS_URL = "ws://localhost:8080/api/ws"


def generate_test_tone(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a simple sine wave test tone."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz A4
    return audio


def generate_tts_audio(text: str, sample_rate: int = 16000) -> np.ndarray:
    """Use macOS say command to generate speech audio."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        tmp_aiff = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    # macOS say → AIFF
    subprocess.run(["say", "-v", "Kyoko", "-o", tmp_aiff, text], check=True)
    # Convert to WAV
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEF32", "-c", "1", tmp_aiff, tmp_wav],
        check=True,
    )

    audio, sr = sf.read(tmp_wav, dtype="float32")
    if sr != sample_rate:
        # Simple resample via interpolation
        import torch
        import torchaudio

        tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(tensor).squeeze(0).numpy()

    return audio


def load_wav_file(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load a WAV file and resample to target rate."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    if sr != sample_rate:
        import torch
        import torchaudio

        tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(tensor).squeeze(0).numpy()

    return audio


def pack_audio_message(
    audio: np.ndarray,
    sample_rate: int,
    sample_offset: int,
    stream_epoch: int,
) -> bytes:
    """Pack audio into the wire format the server expects."""
    header = struct.pack("<IqI", sample_rate, sample_offset, stream_epoch)
    payload = audio.astype(np.float32).tobytes()
    return header + payload


async def run_test(audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Send audio via WebSocket and print received events."""
    import websockets

    # 1. Start a meeting
    req = urllib.request.Request(f"{SERVER}/api/meeting/start", method="POST")
    resp = urllib.request.urlopen(req)
    meeting = json.loads(resp.read())
    print(f"Meeting started: {meeting['meeting_id'][:8]}... (resumed={meeting.get('resumed')})")

    # 2. Connect WebSocket
    events_received = []

    async with websockets.connect(WS_URL) as ws:
        print(f"WebSocket connected")

        # Start receiving in background
        async def receive_events():
            try:
                async for msg in ws:
                    event = json.loads(msg)
                    events_received.append(event)
                    text = event.get("text", "")
                    lang = event.get("language", "?")
                    final = "FINAL" if event.get("is_final") else "partial"
                    trans = event.get("translation")
                    trans_text = ""
                    if trans and trans.get("text"):
                        trans_text = f" → [{trans['target_language']}] {trans['text']}"

                    print(f"  [{lang}] ({final}) {text}{trans_text}")
            except Exception:
                pass

        recv_task = asyncio.create_task(receive_events())

        # 3. Send audio in chunks (~250ms each)
        chunk_samples = int(sample_rate * 0.25)
        stream_epoch = 12345
        sample_offset = 0

        total_chunks = len(audio) // chunk_samples
        print(f"Sending {len(audio)/sample_rate:.1f}s of audio ({total_chunks} chunks)...")

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            if len(chunk) == 0:
                break

            msg = pack_audio_message(chunk, sample_rate, sample_offset, stream_epoch)
            await ws.send(msg)
            sample_offset += len(chunk)

            # Pace like real-time
            await asyncio.sleep(0.25)

        print(f"Audio sent. Waiting for ASR + translation results (up to 30s)...")

        # 4. Wait for processing (translation needs time, especially on first call)
        await asyncio.sleep(30)

        # 5. Close
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    # 6. Stop meeting
    req = urllib.request.Request(f"{SERVER}/api/meeting/stop", method="POST")
    resp = urllib.request.urlopen(req)
    print(f"\nMeeting stopped. Received {len(events_received)} transcript events.")

    if not events_received:
        print("\n⚠ NO EVENTS RECEIVED — ASR pipeline may not be producing output")
        print("Check /tmp/meeting-scribe.log for details")
    else:
        print("\n✓ Pipeline working!")


def main():
    parser = argparse.ArgumentParser(description="E2E test for meeting-scribe")
    parser.add_argument("--file", help="Path to a WAV file to use as input")
    parser.add_argument("--tts", help="Text to speak via macOS TTS (Japanese)")
    parser.add_argument("--tone", action="store_true", help="Generate a test tone (default)")
    args = parser.parse_args()

    if args.file:
        print(f"Loading audio from: {args.file}")
        audio = load_wav_file(args.file)
    elif args.tts:
        print(f"Generating TTS: '{args.tts}'")
        audio = generate_tts_audio(args.tts)
    else:
        print("Generating 5s test tone (440Hz)...")
        audio = generate_test_tone(duration=5.0)

    print(f"Audio: {len(audio)} samples, {len(audio)/16000:.1f}s")

    asyncio.run(run_test(audio))


if __name__ == "__main__":
    main()
