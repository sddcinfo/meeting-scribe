"""Smoke-test an ASR endpoint against a single WAV clip.

Sends ``benchmarks/asr_accuracy_latency.py``'s OpenAI-shape request
to ``--url`` and prints the transcript + total_ms.  Used to confirm
a fresh sidecar is wired correctly before the full corpus run.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="e.g. http://localhost:8013")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--language", default=None, help="Optional language hint, e.g. 'en' / 'ja'.")
    p.add_argument(
        "--model",
        default="auto",
        help="Model field in the OpenAI body (Qwen3-ASR rejects 'auto'; pass 'Qwen/Qwen3-ASR-1.7B').",
    )
    args = p.parse_args()

    audio_b64 = base64.b64encode(args.audio.read_bytes()).decode()
    body: dict = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Transcribe in the spoken language."},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                ],
            },
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    if args.language:
        body["language"] = args.language

    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    text = data["choices"][0]["message"]["content"]
    print(json.dumps({"total_ms": round(elapsed_ms, 1), "transcript": text}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
