#!/usr/bin/env python3
"""Generate summary for a single meeting without re-running diarization."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    meeting_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not meeting_id:
        print("Usage: generate-summary.py <meeting_id>")
        sys.exit(1)

    meeting_dir = Path(__file__).parent.parent / "meetings" / meeting_id
    if not meeting_dir.exists():
        print(f"Meeting not found: {meeting_dir}")
        sys.exit(1)

    from meeting_scribe.summary import generate_summary

    result = await generate_summary(meeting_dir)
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    print(f"Summary generated: {meeting_dir / 'summary.json'}")
    print(f"  Events: {result.get('metadata', {}).get('num_segments')}")
    print(f"  Duration: {result.get('metadata', {}).get('duration_min')} min")
    print(f"  Generation: {result.get('metadata', {}).get('generation_ms')} ms")


asyncio.run(main())
