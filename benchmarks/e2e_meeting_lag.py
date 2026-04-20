"""End-to-end meeting lag harness.

Plays a 15-minute recording into a live scribe-main via /api/meeting/ingest
(or an equivalent audio-in WS), then scrapes `/api/status` for the
`metrics.end_to_end_lag_ms` rolling bucket and records p50/p95 per lang.
Deliberately leaves the playback mechanism pluggable because on-GB10
ingestion path may be WS-audio vs HTTP — filled in during Phase A once
the live harness is wired.

This skeleton sets up the scaffolding; the actual playback is
`TODO(phase-a)`.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from benchmarks._fixture import add_fixture_arg


async def run(scribe_url: str, fixture_dir: Path, out: Path) -> None:
    # Phase A implementation lands once a consented 15-min recording is
    # added to the manifest under kind=meeting_e2e. Until then, fail
    # fast with a clear message rather than emit fake numbers.
    raise SystemExit(
        "e2e_meeting_lag harness skeleton. Populate the manifest with a "
        "kind=meeting_e2e sample and implement the playback loop before "
        "Phase A baseline can run."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scribe-url", required=True, help="e.g. http://localhost:7800")
    add_fixture_arg(p)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(run(args.scribe_url, args.fixture_dir, args.out))


if __name__ == "__main__":
    main()
