#!/usr/bin/env python3
"""Deep quality-gate validator for a meeting directory.

Runs a suite of structural + alignment + speaker-mapping checks against
a meeting's artifacts (audio + journal + timeline + detected_speakers +
lanes + meta) and reports each check as pass/fail with metrics.

Designed to be:
- Importable as a library (``validate_meeting(meeting_dir)`` returns a
  ``ValidationResult`` dict with per-check flags and metrics).
- Runnable as a CLI (``python validate_meeting.py <meeting_id_or_dir>``)
  that prints a readable report and exits non-zero on any failure.
- Diffable across runs — output is deterministic JSON, suitable for
  checking in as a baseline and comparing against future reprocess
  results.

Checks (in order):
  1. Audio + timeline alignment (duration_ms drift)
  2. seq_index density (contiguous from 1)
  3. speaker_lanes has no "0" key (orphan lane leak)
  4. journal cluster_ids all exist in detected
  5. No empty-speaker events in the journal
  6. 30 s coverage buckets (transcript present throughout audio)
  7. Name preservation (only if journal.jsonl.bak exists) —
     what fraction of old unique names appear in detected, plus
     per-event identity accuracy for old named time ranges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
BUCKET_MS = 30_000
DRIFT_TOLERANCE_MS = 10_000  # transcript can lag audio by up to 10 s of tail silence


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    meeting_id: str
    audio_ms: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "audio_ms": self.audio_ms,
            "all_passed": self.all_passed,
            "checks": [asdict(c) for c in self.checks],
        }


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_journal(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _best_finals_by_id(journal_path: Path) -> dict[str, dict]:
    """Dedup by segment_id keeping highest revision — matches server/reprocess semantics."""
    best: dict[str, dict] = {}
    for e in _iter_journal(journal_path):
        if e.get("type") == "speaker_correction":
            continue
        if not e.get("is_final") or not e.get("text"):
            continue
        sid = e.get("segment_id")
        if not sid:
            continue
        if sid not in best or e.get("revision", 0) >= best[sid].get("revision", 0):
            best[sid] = e
    return best


def _check_alignment(m: Path, audio_ms: int) -> CheckResult:
    tl = _load_json(m / "timeline.json")
    duration_ms = tl.get("duration_ms", 0)
    segs = tl.get("segments", [])
    last_end = max((s.get("end_ms", 0) for s in segs), default=0)

    # Duration match: timeline.duration_ms should be ~= audio_ms
    duration_drift = duration_ms - audio_ms
    # Tail: last segment's end can be earlier than audio end (tail
    # silence is expected up to ~5 s) but never significantly larger
    # than that. A tail_gap of 44 minutes means the transcript stops
    # mid-meeting — that's the diarize-failure signature we caught
    # on f38d5807's first diarize-driven run.
    tail_gap = audio_ms - last_end
    # Real meetings often end with 30-90 s of "thanks, bye" silence
    # plus a slow hangup. 120 s is generous enough to tolerate those
    # without silently accepting a mid-meeting diarize gap (which
    # would manifest as coverage failures AND a huge tail gap).
    MAX_TAIL_GAP_MS = 120_000

    ok = abs(duration_drift) <= 500 and tail_gap >= -200 and tail_gap <= MAX_TAIL_GAP_MS
    return CheckResult(
        name="alignment",
        passed=ok,
        detail=(
            f"duration_drift={duration_drift:+d}ms tail_gap={tail_gap}ms "
            f"last_end={last_end}ms audio={audio_ms}ms"
        ),
        metrics={
            "timeline_duration_ms": duration_ms,
            "audio_ms": audio_ms,
            "duration_drift_ms": duration_drift,
            "last_segment_end_ms": last_end,
            "tail_gap_ms": tail_gap,
            "segment_count": len(segs),
        },
    )


def _check_seq_density(m: Path) -> CheckResult:
    ds = _load_json(m / "detected_speakers.json")
    seq = sorted(s.get("seq_index", -1) for s in ds)
    expected = list(range(1, len(ds) + 1))
    passed = seq == expected
    return CheckResult(
        name="seq_density",
        passed=passed,
        detail=f"seq={seq} expected={expected}",
        metrics={"seq_indices": seq, "speaker_count": len(ds)},
    )


def _check_no_orphan_lane(m: Path) -> CheckResult:
    lanes = _load_json(m / "speaker_lanes.json")
    keys = sorted(lanes.keys())
    has_zero = "0" in lanes
    return CheckResult(
        name="no_orphan_lane",
        passed=not has_zero,
        detail=f"lane_keys={keys}",
        metrics={"lane_keys": keys},
    )


def _check_journal_detected_agreement(m: Path) -> CheckResult:
    ds = _load_json(m / "detected_speakers.json")
    valid_seqs = {s.get("seq_index") for s in ds}
    best = _best_finals_by_id(m / "journal.jsonl")
    clusters_seen: Counter[int] = Counter()
    for e in best.values():
        sp = e.get("speakers") or []
        if sp:
            clusters_seen[sp[0].get("cluster_id")] += 1
    unknown = [c for c in clusters_seen if c not in valid_seqs]
    passed = len(unknown) == 0
    return CheckResult(
        name="journal_detected_agreement",
        passed=passed,
        detail=f"journal_clusters={dict(clusters_seen)} unknown={unknown}",
        metrics={
            "journal_cluster_dist": dict(clusters_seen),
            "unknown_cluster_ids": unknown,
            "valid_seq_indices": sorted(valid_seqs),
        },
    )


def _check_no_empty_speakers(m: Path) -> CheckResult:
    best = _best_finals_by_id(m / "journal.jsonl")
    empty = [sid for sid, e in best.items() if not (e.get("speakers") or [])]
    passed = len(empty) == 0
    return CheckResult(
        name="no_empty_speakers",
        passed=passed,
        detail=f"empty_events={len(empty)}/{len(best)}",
        metrics={"empty_count": len(empty), "total_finals": len(best)},
    )


def _check_coverage(m: Path, audio_ms: int) -> CheckResult:
    tl = _load_json(m / "timeline.json")
    segs = tl.get("segments", [])
    buckets = Counter(s.get("start_ms", 0) // BUCKET_MS for s in segs)
    n_buckets = (audio_ms // BUCKET_MS) + 1
    empty_buckets = [i for i in range(n_buckets) if buckets.get(i, 0) == 0]
    # Up to 1 empty bucket at the very end (tail silence) is fine
    passed = len([b for b in empty_buckets if b < n_buckets - 1]) == 0
    return CheckResult(
        name="coverage",
        passed=passed,
        detail=f"empty_buckets={len(empty_buckets)}/{n_buckets}",
        metrics={
            "total_buckets": n_buckets,
            "empty_bucket_count": len(empty_buckets),
            "empty_bucket_indices": empty_buckets,
        },
    )


def _check_name_preservation(m: Path) -> CheckResult:
    bak = m / "journal.jsonl.bak"
    if not bak.exists():
        return CheckResult(
            name="name_preservation",
            passed=True,
            detail="no .bak — skipped (meeting had no prior corrections)",
            metrics={"skipped": True},
        )

    old_seg_by_id: dict[str, dict] = {}
    old_corrs = []
    for e in _iter_journal(bak):
        if e.get("type") == "speaker_correction":
            old_corrs.append(e)
        elif e.get("is_final") and e.get("text"):
            sid = e.get("segment_id")
            if sid:
                old_seg_by_id[sid] = e

    old_named_ranges: list[tuple[int, int, str]] = []
    old_name_counts: Counter = Counter()
    for c in old_corrs:
        seg = old_seg_by_id.get(c.get("segment_id"))
        if not seg:
            continue
        name = c.get("speaker_name") or ""
        if not name:
            continue
        old_named_ranges.append((seg.get("start_ms", 0), seg.get("end_ms", 0), name))
        old_name_counts[name] += 1

    unique_old_names = set(old_name_counts.keys())
    if not unique_old_names:
        return CheckResult(
            name="name_preservation",
            passed=True,
            detail=".bak had no usable corrections",
            metrics={"original_unique_names": 0},
        )

    # What fraction of old unique names now appear somewhere in detected OR per-event identities?
    new_best = _best_finals_by_id(m / "journal.jsonl")
    names_in_detected = {
        s.get("display_name")
        for s in _load_json(m / "detected_speakers.json")
        if s.get("display_name") and not s["display_name"].startswith("Speaker ")
    }
    names_in_events = set()
    for e in new_best.values():
        sp = e.get("speakers") or []
        if sp:
            ident = sp[0].get("identity") or sp[0].get("display_name")
            if ident and not ident.startswith("Speaker "):
                names_in_events.add(ident)
    preserved_any = names_in_detected | names_in_events
    covered = unique_old_names & preserved_any
    missing = unique_old_names - preserved_any

    # Per-event accuracy: for each old named range, does the new event
    # at that time range carry the matching identity?
    accurate = 0
    checked = 0
    for old_start, old_end, name in old_named_ranges:
        old_dur = max(1, old_end - old_start)
        best_match = None
        best_ov = 0
        for e in new_best.values():
            ns, ne = e.get("start_ms", 0), e.get("end_ms", 0)
            ov = max(0, min(ne, old_end) - max(ns, old_start))
            if ov > best_ov:
                best_ov = ov
                best_match = e
        if not best_match or best_ov < 0.3 * old_dur:
            continue
        checked += 1
        sp = best_match.get("speakers") or []
        ident = (sp[0].get("identity") or sp[0].get("display_name")) if sp else None
        if ident == name:
            accurate += 1

    accuracy_pct = (accurate / checked * 100) if checked else 0.0
    # Gate: ≥80% coverage of unique names AND ≥70% per-event accuracy
    # (imperfect because per-event matching depends on diarize boundaries,
    #  but 70% is a reasonable floor — a regression from 95% will still fire)
    coverage_pct = len(covered) / len(unique_old_names) * 100
    passed = coverage_pct >= 80 and accuracy_pct >= 70

    return CheckResult(
        name="name_preservation",
        passed=passed,
        detail=(
            f"name_coverage={coverage_pct:.0f}% ({len(covered)}/{len(unique_old_names)}) "
            f"per_event_accuracy={accuracy_pct:.0f}% ({accurate}/{checked})"
        ),
        metrics={
            "original_unique_names": sorted(unique_old_names),
            "preserved_names": sorted(covered),
            "missing_names": sorted(missing),
            "name_coverage_pct": round(coverage_pct, 1),
            "per_event_accurate": accurate,
            "per_event_checked": checked,
            "per_event_accuracy_pct": round(accuracy_pct, 1),
        },
    )


def validate_meeting(meeting_dir: Path) -> ValidationResult:
    """Run all checks. Does not raise — result carries pass/fail per check."""
    meeting_id = meeting_dir.name
    pcm = meeting_dir / "audio" / "recording.pcm"
    audio_ms = (
        int(os.path.getsize(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000) if pcm.exists() else 0
    )

    result = ValidationResult(meeting_id=meeting_id, audio_ms=audio_ms)
    checks = [
        lambda: _check_alignment(meeting_dir, audio_ms),
        lambda: _check_seq_density(meeting_dir),
        lambda: _check_no_orphan_lane(meeting_dir),
        lambda: _check_journal_detected_agreement(meeting_dir),
        lambda: _check_no_empty_speakers(meeting_dir),
        lambda: _check_coverage(meeting_dir, audio_ms),
        lambda: _check_name_preservation(meeting_dir),
    ]
    for fn in checks:
        try:
            result.checks.append(fn())
        except Exception as e:
            result.checks.append(
                CheckResult(
                    name=fn.__name__ if hasattr(fn, "__name__") else "unknown",
                    passed=False,
                    detail=f"EXCEPTION: {type(e).__name__}: {e}",
                )
            )
    return result


def _resolve_meeting_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    # Assume it's a meeting_id under ./meetings/
    candidate = Path("meetings") / arg
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"meeting not found: {arg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a meeting's reprocess output.")
    ap.add_argument("meeting", help="meeting_id or full path to meeting directory")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    meeting_dir = _resolve_meeting_dir(args.meeting)
    result = validate_meeting(meeting_dir)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n=== {result.meeting_id}  audio={result.audio_ms / 60000:.2f}min ===\n")
        for c in result.checks:
            mark = "PASS" if c.passed else "FAIL"
            print(f"  [{mark}] {c.name:<28} {c.detail}")
        print()
        print("OVERALL:", "PASS" if result.all_passed else "FAIL")
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
