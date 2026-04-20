#!/usr/bin/env python3
"""Independent cross-file speaker alignment audit.

The regular validator in ``validate_meeting.py`` checks each file for
internal consistency. This script goes further: it cross-references
the THREE places the UI reads speakers from and asserts they agree
segment-by-segment.

The three files:

  1. ``detected_speakers.json`` — the participant list shown in the
     UI sidebar / table-chair view. Each entry has a ``seq_index``
     (1..N), a ``cluster_id``, and a ``display_name``.

  2. ``speaker_lanes.json`` — the per-speaker timeline. Dict keyed by
     ``str(seq_index)`` → list of ``{start_ms, end_ms, segment_id}``
     entries. The UI renders one horizontal lane per key.

  3. ``journal.jsonl`` — the authoritative transcript. Each final
     event has ``speakers[0].cluster_id`` (a seq_index after the
     reprocess remapping) + optional ``speakers[0].identity``.

Invariants this script checks (all must hold):

  A. Every ``seq_index`` in (1) has a lane in (2) with ≥1 entry.
  B. Every lane key in (2) matches a ``seq_index`` in (1).
  C. Every journal event's ``cluster_id`` equals a ``seq_index`` in (1).
  D. Every lane entry's ``segment_id`` exists in the journal with a
     matching ``cluster_id``.
  E. Every journal event with a speaker has exactly one corresponding
     lane entry (same seq_index, same segment_id).
  F. Segment counts per speaker match across all three files.
  G. Total speaking time per speaker: ``detected.total_speaking_ms``
     equals the sum of ``(end - start)`` across that speaker's journal
     events.
  H. Timeline coverage: every journal final event has a corresponding
     entry in ``timeline.json``'s segments list.

Any failure is reported with file paths + offending id(s) so it's
obvious which file to inspect.

Usage::

    ./cross_check_speakers.py <meeting_id>
    ./cross_check_speakers.py meetings/<meeting_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Finding:
    severity: str  # "FAIL" | "WARN"
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.detail}"


@dataclass
class Report:
    meeting_id: str
    findings: list[Finding] = field(default_factory=list)

    def fail(self, check: str, detail: str) -> None:
        self.findings.append(Finding("FAIL", check, detail))

    def warn(self, check: str, detail: str) -> None:
        self.findings.append(Finding("WARN", check, detail))

    @property
    def failures(self) -> int:
        return sum(1 for f in self.findings if f.severity == "FAIL")

    @property
    def ok(self) -> bool:
        return self.failures == 0


def _load_journal_finals(path: Path) -> list[dict]:
    """Dedup by segment_id keeping highest revision. Returns only
    final events with non-empty text (what the UI renders)."""
    best: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "speaker_correction":
                continue
            if not e.get("is_final") or not e.get("text"):
                continue
            sid = e.get("segment_id")
            if not sid:
                continue
            if sid not in best or e.get("revision", 0) >= best[sid].get("revision", 0):
                best[sid] = e
    return list(best.values())


def cross_check(meeting_dir: Path) -> Report:
    report = Report(meeting_id=meeting_dir.name)

    detected_path = meeting_dir / "detected_speakers.json"
    lanes_path = meeting_dir / "speaker_lanes.json"
    journal_path = meeting_dir / "journal.jsonl"
    timeline_path = meeting_dir / "timeline.json"

    for p in (detected_path, lanes_path, journal_path, timeline_path):
        if not p.exists():
            report.fail("file_missing", f"{p.name} does not exist")
    if report.failures:
        return report

    detected = json.loads(detected_path.read_text())
    lanes = json.loads(lanes_path.read_text())
    journal = _load_journal_finals(journal_path)
    timeline = json.loads(timeline_path.read_text()).get("segments", [])

    # Index by seq_index / cluster_id
    detected_by_seq = {s["seq_index"]: s for s in detected}
    detected_seqs = set(detected_by_seq.keys())

    lane_seqs = set()
    for key in lanes.keys():
        try:
            lane_seqs.add(int(key))
        except ValueError:
            report.fail("lanes_nonint_key", f"non-integer lane key: {key!r}")

    # Journal: the UI reads speakers[0].cluster_id AFTER the post-
    # reprocess remap, which means cluster_id IS the seq_index here.
    journal_by_sid: dict[str, dict] = {}
    journal_clusters: Counter = Counter()
    for e in journal:
        sid = e["segment_id"]
        journal_by_sid[sid] = e
        sp = e.get("speakers") or []
        if sp:
            cid = sp[0].get("cluster_id")
            if cid is not None:
                journal_clusters[cid] += 1

    timeline_by_sid: dict[str, dict] = {}
    for seg in timeline:
        sid = seg.get("segment_id")
        if sid:
            timeline_by_sid[sid] = seg

    # A. Every seq_index in detected has at least one lane entry.
    for seq in detected_seqs:
        key = str(seq)
        if key not in lanes or not lanes[key]:
            report.fail(
                "detected_without_lane",
                f"speaker seq={seq} name={detected_by_seq[seq].get('display_name')!r} "
                f"has no lane entries",
            )

    # B. Every lane key corresponds to a detected seq.
    for seq in lane_seqs:
        if seq not in detected_seqs:
            report.fail(
                "lane_without_detected",
                f"lane key {seq} has no matching detected_speakers entry",
            )

    # C. Every journal cluster_id is a valid seq.
    for cid, count in journal_clusters.items():
        if cid not in detected_seqs:
            report.fail(
                "journal_cluster_not_detected",
                f"cluster_id={cid} appears on {count} journal events "
                f"but is not in detected_speakers.json",
            )

    # D + E. Every lane entry points to a journal event with matching seq.
    lane_segment_cluster: dict[str, int] = {}
    for key, entries in lanes.items():
        try:
            lane_seq = int(key)
        except ValueError:
            continue
        for entry in entries:
            sid = entry.get("segment_id")
            if not sid:
                report.fail("lane_entry_no_segment_id", f"lane {key} has entry with no segment_id")
                continue
            if sid in lane_segment_cluster:
                if lane_segment_cluster[sid] != lane_seq:
                    report.fail(
                        "segment_in_multiple_lanes",
                        f"segment_id={sid} appears in lanes {lane_segment_cluster[sid]} "
                        f"AND {lane_seq}",
                    )
            else:
                lane_segment_cluster[sid] = lane_seq
            if sid not in journal_by_sid:
                report.fail(
                    "lane_segment_not_in_journal",
                    f"lane {key} references segment_id={sid} not found in journal",
                )
                continue
            j_sp = (journal_by_sid[sid].get("speakers") or [{}])[0]
            j_cid = j_sp.get("cluster_id")
            if j_cid != lane_seq:
                report.fail(
                    "lane_journal_cluster_mismatch",
                    f"segment_id={sid} lane={lane_seq} journal_cluster={j_cid}",
                )

    # F. Every journal event with a speaker has exactly one lane entry.
    journal_events_by_seq: Counter = Counter()
    for e in journal:
        sp = e.get("speakers") or []
        if not sp:
            continue
        cid = sp[0].get("cluster_id")
        if cid is None:
            continue
        journal_events_by_seq[cid] += 1
        sid = e["segment_id"]
        if sid not in lane_segment_cluster:
            report.fail(
                "journal_event_without_lane",
                f"seg={sid} cluster={cid} text={e.get('text','')!r} "
                f"— no lane entry",
            )

    # G. detected.segment_count equals journal count per speaker.
    for seq, info in detected_by_seq.items():
        declared = info.get("segment_count", 0)
        actual = journal_events_by_seq.get(seq, 0)
        if declared != actual:
            report.fail(
                "segment_count_mismatch",
                f"speaker seq={seq} detected_says={declared} journal_says={actual}",
            )

    # H. detected.total_speaking_ms matches sum of journal (end-start).
    journal_total_ms: dict[int, int] = {}
    for e in journal:
        sp = e.get("speakers") or []
        if not sp:
            continue
        cid = sp[0].get("cluster_id")
        if cid is None:
            continue
        journal_total_ms[cid] = journal_total_ms.get(cid, 0) + max(
            0, e.get("end_ms", 0) - e.get("start_ms", 0)
        )
    for seq, info in detected_by_seq.items():
        declared = info.get("total_speaking_ms", 0)
        actual = journal_total_ms.get(seq, 0)
        # Allow 100ms slack for rounding
        if abs(declared - actual) > 100:
            report.fail(
                "speaking_ms_mismatch",
                f"speaker seq={seq} detected={declared}ms journal={actual}ms "
                f"diff={declared - actual:+d}ms",
            )

    # I. Every journal final has a corresponding timeline segment.
    for sid in journal_by_sid:
        if sid not in timeline_by_sid:
            report.fail(
                "journal_event_not_in_timeline",
                f"seg={sid} in journal but missing from timeline.json",
            )

    # J. Every timeline segment references a real journal event.
    for sid in timeline_by_sid:
        if sid not in journal_by_sid:
            report.fail(
                "timeline_segment_without_journal",
                f"seg={sid} in timeline but not in journal",
            )

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-file speaker alignment audit.")
    ap.add_argument("meeting", help="meeting_id or full path")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    md = Path(args.meeting)
    if not md.is_dir():
        md = Path("meetings") / args.meeting
    if not md.is_dir():
        raise SystemExit(f"meeting not found: {args.meeting}")

    report = cross_check(md)

    if args.json:
        print(json.dumps(
            {
                "meeting_id": report.meeting_id,
                "ok": report.ok,
                "failures": report.failures,
                "findings": [
                    {"severity": f.severity, "check": f.check, "detail": f.detail}
                    for f in report.findings
                ],
            },
            indent=2,
        ))
    else:
        print(f"\n=== Cross-check: {report.meeting_id} ===\n")
        if not report.findings:
            print("  All invariants hold. (A–J)")
        else:
            by_check: dict[str, list[Finding]] = {}
            for f in report.findings:
                by_check.setdefault(f.check, []).append(f)
            for check, items in sorted(by_check.items()):
                mark = "FAIL" if items[0].severity == "FAIL" else "WARN"
                print(f"  [{mark}] {check}  ×{len(items)}")
                for item in items[:5]:
                    print(f"      {item.detail}")
                if len(items) > 5:
                    print(f"      ... ({len(items) - 5} more)")
        print()
        print("RESULT:", "PASS" if report.ok else f"FAIL ({report.failures} failures)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
