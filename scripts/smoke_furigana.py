#!/usr/bin/env python3
"""Smoke-test the furigana backend: feed kanji, expect <ruby> markup."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_scribe.backends.furigana import FuriganaBackend


async def main() -> int:
    backend = FuriganaBackend()
    await backend.start()
    cases = ["会議", "東京", "私は田中です", "明日の予定", "こんにちは"]
    failures = 0
    for text in cases:
        html = await backend.annotate(text)
        ok = bool(html) and "<ruby>" in html
        marker = "✓" if ok else "✗"
        print(f"{marker} {text!r:25} → {html or '(empty)'}")
        if not ok:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
