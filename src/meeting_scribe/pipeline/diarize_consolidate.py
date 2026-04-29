"""Fast-path diarization consolidation from incremental journal data.

Background: ``backends/diarize_sortformer.py`` runs rolling-window
diarization continuously during a recording. Each event's
``speakers[*].cluster_id`` reflects the live cluster assignment at
write time. At finalize, the current code path re-runs diarization on
the FULL pcm (``_diarize_full_audio``), which can take 1–4 minutes.

This module exposes a fast-path consolidator that builds the same
``diarize_segments`` shape directly from the existing event records,
skipping the full-pass.

The fast-path is gated on env ``SCRIBE_FINALIZE_DIARIZE_FAST``:
  * ``shadow`` (default) — run BOTH; finalize uses full-pass; logs
    a structured comparison line per finalize. No behavior change.
  * ``on`` — fast-path only; full-pass skipped.
  * ``off`` — full-pass only; no comparison logging.

Promotion to ``on`` should happen after the shadow-mode comparison
shows < 1% segment-level disagreement on at least 5 representative
meetings. Default is ``shadow`` so we accumulate that data on every
real finalize without changing behavior.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def consolidate_from_events(events_by_sid: dict[str, dict]) -> list[dict[str, Any]]:
    """Build a diarize_segments-shaped list from existing event records.

    Each input event may have a ``speakers`` list of
    ``{cluster_id, start_ms, end_ms, score?}`` entries already
    populated by the live rolling diarizer. This function flattens
    those into the same shape that ``_diarize_full_audio`` returns:
    one segment per (event, speaker) crossing.

    Returns ``[]`` if no events have speaker assignments — caller
    should fall back to the full-pass in that case.
    """
    out: list[dict[str, Any]] = []
    for ev in events_by_sid.values():
        speakers = ev.get("speakers") or []
        if not speakers:
            continue
        for sp in speakers:
            cid = sp.get("cluster_id")
            if cid is None:
                continue
            out.append(
                {
                    "cluster_id": cid,
                    "start_ms": sp.get("start_ms", ev.get("start_ms", 0)),
                    "end_ms": sp.get("end_ms", ev.get("end_ms", 0)),
                    "score": sp.get("score"),
                }
            )
    return out


def compare_diarize_results(
    fast: list[dict[str, Any]],
    full: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute structured comparison between fast-path and full-pass.

    Used by SHADOW mode logging to produce a single line we can grep
    for a promotion decision. Compares (a) speaker counts and (b)
    per-segment cluster_id disagreement based on time-overlap
    alignment.

    Returns a dict with: full_speakers, fast_speakers, full_segments,
    fast_segments, segment_disagree_rate (float in 0..1).
    """
    full_speakers = len({s["cluster_id"] for s in full}) if full else 0
    fast_speakers = len({s["cluster_id"] for s in fast}) if fast else 0

    if not full:
        # Can't compute disagreement without a full-pass baseline.
        return {
            "full_speakers": 0,
            "fast_speakers": fast_speakers,
            "full_segments": 0,
            "fast_segments": len(fast),
            "segment_disagree_rate": None,
        }

    # Naive alignment: for each fast segment, find the full segment with
    # max time-overlap and compare cluster_id.
    disagree = 0
    matched = 0
    for fs in fast:
        fs_start = fs.get("start_ms", 0)
        fs_end = fs.get("end_ms", 0)
        if fs_end <= fs_start:
            continue
        best_overlap = 0
        best_cid = None
        for ps in full:
            o_start = max(fs_start, ps.get("start_ms", 0))
            o_end = min(fs_end, ps.get("end_ms", 0))
            overlap = max(0, o_end - o_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cid = ps.get("cluster_id")
        if best_overlap > 0:
            matched += 1
            if best_cid != fs.get("cluster_id"):
                disagree += 1

    rate = (disagree / matched) if matched else 1.0
    return {
        "full_speakers": full_speakers,
        "fast_speakers": fast_speakers,
        "full_segments": len(full),
        "fast_segments": len(fast),
        "segment_disagree_rate": round(rate, 4),
    }
