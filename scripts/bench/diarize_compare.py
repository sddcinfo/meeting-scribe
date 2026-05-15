"""Track C: standalone pyannote 3.1 vs community-1 segmentation A/B.

The production diarize container (``containers/pyannote/server.py``)
serializes only the standard ``annotation.itertracks(yield_label=True)``
output, dropping community-1's new ``DiarizeOutput.exclusive_diarization``
field on the floor.  That field is the whole reason to evaluate
community-1, so this script bypasses the HTTP layer entirely:

* Loads each pretrained pipeline directly via
  ``Pipeline.from_pretrained(<id>, revision=<sha>)`` — pinned per S3.
* Runs both on the cloned meeting audio.
* Emits ``speaker_diarization`` (standard) and, for community-1,
  ``exclusive_diarization`` (the new feature) as separate JSON arrays.
* Computes the time-weighted overlap-resolution metric used by the
  Track C decision gate: of the seconds where 3.1's standard output
  has ≥ 2 overlapping speakers, what fraction does community-1's
  exclusive mode assign to a single speaker?

Output JSON shape (per meeting)::

    {
      "meeting_id": "...",
      "audio_path": "...",
      "audio_duration_s": 1234.5,
      "pyannote_3_1": {
        "pipeline_id": "pyannote/speaker-diarization-3.1",
        "revision": "<pinned sha>",
        "speaker_diarization": [{"speaker_id": int, "start": float, "end": float}, ...]
      },
      "community_1": {
        "pipeline_id": "pyannote/speaker-diarization-community-1",
        "revision": "<pinned sha>",
        "speaker_diarization": [...],
        "exclusive_diarization": [...]      # may be null if the field is absent
      },
      "overlap_metric": {
        "total_overlap_seconds": float,
        "overlap_resolved_seconds": float,
        "fraction_resolved": float,
        "pass_threshold": 0.30,
        "pass": bool
      }
    }

Output paths are validated against the offline-only rule
(``benchmarks/_bench_paths.assert_offline_path``) so per-meeting JSON
never lands in the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# benchmarks/_bench_paths.py is the canonical assert_offline_path() helper.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks._bench_paths import assert_offline_path

OVERLAP_PASS_THRESHOLD = 0.30


@dataclass(frozen=True)
class Segment:
    speaker_id: int
    start: float
    end: float


# ---------------------------------------------------------------------------
# Pipeline loading + segmentation extraction
# ---------------------------------------------------------------------------


def _load_pipeline(model_id: str, revision: str | None, hf_token: str | None):
    # Imported lazily so the script can be `--help`'d without torch installed.
    from pyannote.audio import Pipeline  # type: ignore

    kwargs: dict = {}
    if revision:
        # pyannote's Pipeline.from_pretrained forwards **kwargs into HF
        # snapshot_download via huggingface_hub; revision is the standard knob.
        kwargs["revision"] = revision
    if hf_token:
        kwargs["token"] = hf_token
    return Pipeline.from_pretrained(model_id, **kwargs)


def _annotation_to_segments(annotation) -> list[Segment]:
    out: list[Segment] = []
    label_to_id: dict[str, int] = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        sid = label_to_id.setdefault(speaker, len(label_to_id))
        out.append(Segment(speaker_id=sid, start=float(turn.start), end=float(turn.end)))
    return out


def _run_pipeline(pipeline, audio_path: Path) -> tuple[list[Segment], list[Segment] | None]:
    """Returns (standard_segments, exclusive_segments | None)."""
    raw = pipeline(str(audio_path))
    # 3.x returns Annotation directly; 4.x wraps in DiarizeOutput.
    from pyannote.core import Annotation  # type: ignore

    if isinstance(raw, Annotation):
        return _annotation_to_segments(raw), None
    standard = _annotation_to_segments(raw.speaker_diarization)
    exclusive = None
    if (
        hasattr(raw, "exclusive_speaker_diarization")
        and raw.exclusive_speaker_diarization is not None
    ):
        exclusive = _annotation_to_segments(raw.exclusive_speaker_diarization)
    return standard, exclusive


# ---------------------------------------------------------------------------
# Time-weighted overlap-resolution metric
# ---------------------------------------------------------------------------


def _overlap_intervals(segments: list[Segment]) -> list[tuple[float, float]]:
    """Time intervals (in seconds) where ≥ 2 segments overlap.

    Implemented via a sweep-line over (time, +1/-1) events; each
    interval where the running count ≥ 2 is an overlap window.
    """
    events: list[tuple[float, int]] = []
    for s in segments:
        events.append((s.start, +1))
        events.append((s.end, -1))
    events.sort()

    overlaps: list[tuple[float, float]] = []
    count = 0
    overlap_start: float | None = None
    for t, delta in events:
        prev = count
        count += delta
        if prev < 2 and count >= 2:
            overlap_start = t
        elif prev >= 2 and count < 2 and overlap_start is not None:
            overlaps.append((overlap_start, t))
            overlap_start = None
    return overlaps


def _segments_for_speakers_at(time_window: tuple[float, float], segs: list[Segment]) -> set[int]:
    """Speakers active at any point within ``time_window``."""
    a, b = time_window
    return {s.speaker_id for s in segs if s.end > a and s.start < b}


def overlap_metric(standard_3_1: list[Segment], exclusive_c_1: list[Segment] | None) -> dict:
    """Of the seconds where 3.1 has ≥ 2 overlapping speakers, what fraction
    does community-1's exclusive mode assign to exactly one speaker?

    Both sides are time-weighted, so segmentation-boundary differences
    between the two backends don't bias the count.
    """
    overlaps = _overlap_intervals(standard_3_1)
    total_overlap_s = sum(b - a for a, b in overlaps)
    if exclusive_c_1 is None or total_overlap_s == 0.0:
        return {
            "total_overlap_seconds": total_overlap_s,
            "overlap_resolved_seconds": 0.0,
            "fraction_resolved": 0.0,
            "pass_threshold": OVERLAP_PASS_THRESHOLD,
            "pass": False,
            "note": "exclusive_diarization unavailable"
            if exclusive_c_1 is None
            else "no overlaps in 3.1",
        }

    # For each overlap window, walk the exclusive segments and count
    # how much time within the window has exactly one assigned speaker.
    resolved_s = 0.0
    for win_start, win_end in overlaps:
        # Build sub-events restricted to this window from the exclusive segs.
        sub_events: list[tuple[float, int]] = []
        for seg in exclusive_c_1:
            a = max(seg.start, win_start)
            b = min(seg.end, win_end)
            if b > a:
                sub_events.append((a, +1))
                sub_events.append((b, -1))
        sub_events.sort()
        count = 0
        last_t = win_start
        for t, delta in sub_events:
            if count == 1:
                resolved_s += t - last_t
            count += delta
            last_t = t
        # Trailing piece up to win_end with current count.
        if count == 1:
            resolved_s += win_end - last_t
    fraction = resolved_s / total_overlap_s if total_overlap_s else 0.0
    return {
        "total_overlap_seconds": total_overlap_s,
        "overlap_resolved_seconds": resolved_s,
        "fraction_resolved": fraction,
        "pass_threshold": OVERLAP_PASS_THRESHOLD,
        "pass": fraction >= OVERLAP_PASS_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Per-meeting bench driver + JSON emission
# ---------------------------------------------------------------------------


def _segments_to_json(segs: list[Segment]) -> list[dict]:
    return [{"speaker_id": s.speaker_id, "start": s.start, "end": s.end} for s in segs]


def _segments_to_rttm(segs: list[Segment], meeting_id: str) -> str:
    lines = []
    for s in segs:
        dur = max(0.0, s.end - s.start)
        lines.append(
            f"SPEAKER {meeting_id} 1 {s.start:.3f} {dur:.3f} <NA> <NA> spk{s.speaker_id} <NA> <NA>"
        )
    return "\n".join(lines) + "\n"


def run_one(
    *,
    meeting_id: str,
    audio_path: Path,
    out_dir: Path,
    rev_3_1: str | None,
    rev_c_1: str | None,
    hf_token: str | None,
    write_rttm: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{meeting_id}] loading pyannote 3.1 (revision={rev_3_1})", flush=True)
    pipe_3_1 = _load_pipeline("pyannote/speaker-diarization-3.1", rev_3_1, hf_token)
    t0 = time.monotonic()
    std_3_1, _ = _run_pipeline(pipe_3_1, audio_path)
    elapsed_3_1 = time.monotonic() - t0
    print(f"[{meeting_id}] 3.1 ran in {elapsed_3_1:.1f}s, {len(std_3_1)} segments", flush=True)

    print(f"[{meeting_id}] loading pyannote community-1 (revision={rev_c_1})", flush=True)
    pipe_c_1 = _load_pipeline("pyannote/speaker-diarization-community-1", rev_c_1, hf_token)
    t0 = time.monotonic()
    std_c_1, exc_c_1 = _run_pipeline(pipe_c_1, audio_path)
    elapsed_c_1 = time.monotonic() - t0
    exc_label = f"+{len(exc_c_1)} exclusive" if exc_c_1 is not None else "no exclusive field"
    print(
        f"[{meeting_id}] c-1 ran in {elapsed_c_1:.1f}s, "
        f"{len(std_c_1)} standard segments, {exc_label}",
        flush=True,
    )

    metric = overlap_metric(std_3_1, exc_c_1)

    # Optional RTTM dump — useful for later DER scoring against an RTTM label.
    if write_rttm:
        (out_dir / "3_1.rttm").write_text(_segments_to_rttm(std_3_1, meeting_id))
        (out_dir / "c_1_standard.rttm").write_text(_segments_to_rttm(std_c_1, meeting_id))
        if exc_c_1 is not None:
            (out_dir / "c_1_exclusive.rttm").write_text(_segments_to_rttm(exc_c_1, meeting_id))

    record = {
        "meeting_id": meeting_id,
        "audio_path": str(audio_path),
        "pyannote_3_1": {
            "pipeline_id": "pyannote/speaker-diarization-3.1",
            "revision": rev_3_1,
            "elapsed_s": elapsed_3_1,
            "speaker_diarization": _segments_to_json(std_3_1),
        },
        "community_1": {
            "pipeline_id": "pyannote/speaker-diarization-community-1",
            "revision": rev_c_1,
            "elapsed_s": elapsed_c_1,
            "speaker_diarization": _segments_to_json(std_c_1),
            "exclusive_diarization": _segments_to_json(exc_c_1) if exc_c_1 is not None else None,
        },
        "overlap_metric": metric,
    }

    out_json = out_dir / "segments.json"
    out_json.write_text(json.dumps(record, indent=2))
    print(f"[{meeting_id}] wrote {out_json}", flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meeting-id", required=True, help="Source meeting id (used in RTTM lines).")
    p.add_argument("--audio", type=Path, required=True, help="Path to the meeting audio file.")
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory; must resolve outside the repo "
        "(e.g. /data/meeting-scribe-fixtures/bench-runs/2026-Q2/diarize_community1/<id>/).",
    )
    p.add_argument("--rev-3-1", default=None, help="Pinned HF revision SHA for pyannote 3.1.")
    p.add_argument(
        "--rev-c-1", default=None, help="Pinned HF revision SHA for pyannote community-1."
    )
    p.add_argument("--hf-token", default=None, help="HuggingFace token (or via $HF_TOKEN).")
    p.add_argument("--write-rttm", action="store_true")
    args = p.parse_args(argv)

    out_dir = assert_offline_path(args.out_dir)
    if not args.audio.exists():
        raise SystemExit(f"audio not found: {args.audio}")

    import os

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    record = run_one(
        meeting_id=args.meeting_id,
        audio_path=args.audio,
        out_dir=out_dir,
        rev_3_1=args.rev_3_1,
        rev_c_1=args.rev_c_1,
        hf_token=hf_token,
        write_rttm=args.write_rttm,
    )
    metric = record["overlap_metric"]
    print(
        f"\noverlap_metric: total_overlap={metric['total_overlap_seconds']:.1f}s "
        f"resolved={metric['overlap_resolved_seconds']:.1f}s "
        f"fraction={metric['fraction_resolved']:.2%} "
        f"pass={metric['pass']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
