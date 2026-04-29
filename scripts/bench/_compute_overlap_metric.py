"""Compute the Track C time-weighted overlap-resolution metric from
the saved /v1/diarize JSON dumps.

Inputs:
* baseline JSON  (pyannote 3.1 standard segmentation, from port 8001)
* challenger JSON (community-1 with X-Include-Exclusive=true, from port 8014;
  must contain ``exclusive_segments``)

Computes ``overlap_resolved_seconds / total_overlap_seconds`` against
the 0.30 pass threshold from plans/stateful-marinating-whistle.md C5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bench.diarize_compare import OVERLAP_PASS_THRESHOLD, Segment, overlap_metric


def _load(path: Path, key: str) -> list[Segment]:
    raw = json.loads(path.read_text())
    return [
        Segment(s["speaker_id"], float(s["start"]), float(s["end"])) for s in raw.get(key, [])
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--challenger", type=Path, required=True)
    args = p.parse_args()

    base_segs = _load(args.baseline, "segments")
    chal_excl = _load(args.challenger, "exclusive_segments")

    metric = overlap_metric(base_segs, chal_excl)
    metric["pass_threshold"] = OVERLAP_PASS_THRESHOLD
    print(json.dumps(metric, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
