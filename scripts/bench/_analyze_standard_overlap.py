"""Quick analysis of standard /v1/diarize JSON dumps from 3.1 vs c-1.

Inputs are the response bodies from POST /v1/diarize (both pipelines
serve the same shape).  Computes:

* total overlap seconds within each pipeline's output (sweep-line)
* per-pipeline speaker count, audio duration
* a side-by-side summary so we can sanity-check before running the
  full diarize_compare.py inside the c-1 container for exclusive mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bench.diarize_compare import Segment, _overlap_intervals


def _load_segments(path: Path) -> tuple[list[Segment], dict]:
    raw = json.loads(path.read_text())
    segs = [Segment(s["speaker_id"], float(s["start"]), float(s["end"])) for s in raw["segments"]]
    meta = {
        "num_speakers": raw.get("num_speakers"),
        "audio_duration_s": raw.get("audio_duration_s"),
        "processing_ms": raw.get("processing_ms"),
        "segment_count": len(segs),
    }
    return segs, meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--challenger", type=Path, required=True)
    args = p.parse_args()

    base_segs, base_meta = _load_segments(args.baseline)
    chal_segs, chal_meta = _load_segments(args.challenger)

    base_overlaps = _overlap_intervals(base_segs)
    chal_overlaps = _overlap_intervals(chal_segs)

    base_overlap_s = sum(b - a for a, b in base_overlaps)
    chal_overlap_s = sum(b - a for a, b in chal_overlaps)

    summary = {
        "baseline (pyannote 3.1)": {**base_meta, "overlap_s": round(base_overlap_s, 1)},
        "challenger (community-1, standard)": {
            **chal_meta,
            "overlap_s": round(chal_overlap_s, 1),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
