"""Aggregate per-meeting JSON output of asr_meeting_realworld.py
into the final report.  Used to recover from a transient aggregation
bug without re-running the bench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bench.asr_meeting_realworld import (  # noqa: E402
    _percentile,
    _render_report,
    _summarize,
)


def main() -> int:
    in_dir = Path(sys.argv[1])
    report_path = Path(sys.argv[2])

    per_meeting: list[dict] = []
    for path in sorted(in_dir.glob("*.json")):
        if path.name.endswith("aggregate.json"):
            continue
        per_meeting.append(json.loads(path.read_text()))

    # Annotate each row with its meeting id so the top-disagreements
    # table can cite it (the per-meeting JSON doesn't carry it).
    for m in per_meeting:
        for r in m["rows"]:
            r["meeting_id"] = m["meeting_id"]

    all_rows = [r for m in per_meeting for r in m["rows"]]
    overall = _summarize(all_rows)
    speech_rows = [r for r in all_rows if r["rms"] >= 0.005]
    top = sorted(
        speech_rows, key=lambda r: r["char_disagreement_frac"], reverse=True
    )[:30]

    report = _render_report(per_meeting, overall, top)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
