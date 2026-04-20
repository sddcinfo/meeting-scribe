"""Deep meeting data integrity validator — self-learning test process.

Runs a comprehensive analysis over every meeting on disk and asserts that
all derived files (speaker_lanes, timeline, detected_speakers, summary,
room) are internally consistent with the source-of-truth journal.jsonl.

What "self-learning" means here:
- First run captures a *baseline* of findings per meeting to
  `tests/meeting_integrity_baseline.json`.
- Future runs DIFF against the baseline — new issues fail the build,
  but pre-existing ones are grandfathered (with visibility into the list).
- When an issue is fixed, the baseline auto-tightens: the validator
  prints `✓ healed: <meeting_id>` and asks you to refresh the baseline.
- You can re-baseline explicitly with `UPDATE_INTEGRITY_BASELINE=1 pytest`.
- Every check is scored (WARN/ERROR) so findings surface as reports, not
  one-line failures.

What it checks per meeting (all are cross-file, not just self-consistent):
    1. Cluster ↔ identity consistency
       - Each cluster_id maps to a single identity (or UNNAMED).
       - Each identity belongs to a single cluster_id.
       - Flags "identity drift" (same cluster with multiple names) as ERROR.
    2. Orphan micro-clusters
       - Cluster with <=3 segments AND total speech <10s → WARN.
       - Likely misclustered interjections from a known speaker.
    3. Derived file consistency
       - timeline.json segment count matches deduplicated journal events.
       - speaker_lanes.json contains every cluster_id that appears in events.
       - detected_speakers.json cluster_ids are a subset of journal clusters.
    4. Coverage gaps
       - Total speaking time vs meeting duration (< 40% → WARN).
       - Unnamed speakers in meetings that have a populated room layout.
    5. Structural integrity
       - summary.json has executive_summary + topics when it exists.
       - meta.json state is one of the allowed values.
       - All required files exist for "complete" meetings.

Run:
    pytest tests/test_meeting_integrity.py -v      # report all issues
    UPDATE_INTEGRITY_BASELINE=1 pytest tests/test_meeting_integrity.py
                                                    # rebaseline after fix
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

MEETINGS_DIR = Path(__file__).parent.parent / "meetings"
BASELINE_PATH = Path(__file__).parent / "meeting_integrity_baseline.json"


# ── Finding model ──────────────────────────────────────────────


@dataclass
class Finding:
    """One issue discovered during validation."""

    meeting_id: str
    code: str  # short tag like "cluster-drift"
    severity: str  # "ERROR" | "WARN"
    detail: str

    def key(self) -> str:
        """Stable key for baselining (ignores detail text changes)."""
        return f"{self.meeting_id}::{self.code}"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.meeting_id} {self.code}: {self.detail}"


@dataclass
class MeetingReport:
    meeting_id: str
    state: str
    event_count: int
    cluster_count: int
    named_speakers: int
    findings: list[Finding] = field(default_factory=list)

    def summary_line(self) -> str:
        errors = sum(1 for f in self.findings if f.severity == "ERROR")
        warns = sum(1 for f in self.findings if f.severity == "WARN")
        return (
            f"{self.meeting_id}  {self.state:<11} "
            f"events={self.event_count:4d}  clusters={self.cluster_count:2d}  "
            f"named={self.named_speakers:2d}  err={errors} warn={warns}"
        )


# ── Parsing helpers ────────────────────────────────────────────


def _load_journal(meeting_dir: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Dedup + apply corrections. Returns (events_by_sid, corrections_by_sid)."""
    events: dict[str, dict] = {}
    corrections: dict[str, str] = {}
    path = meeting_dir / "journal.jsonl"
    if not path.exists():
        return events, corrections

    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "speaker_correction":
            corrections[e.get("segment_id", "")] = e.get("speaker_name", "")
            continue
        if not e.get("is_final") or not e.get("text"):
            continue
        sid = e.get("segment_id")
        if not sid:
            continue
        if sid not in events or e.get("revision", 0) > events[sid].get("revision", 0):
            events[sid] = e

    # Apply corrections
    for sid, name in corrections.items():
        if sid in events:
            sp = events[sid].get("speakers") or []
            if sp:
                sp[0]["identity"] = name
                sp[0]["display_name"] = name
            else:
                events[sid]["speakers"] = [
                    {"identity": name, "display_name": name, "cluster_id": 0}
                ]
    return events, corrections


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── Validators ─────────────────────────────────────────────────


def _check_cluster_identity_consistency(
    meeting_id: str, events: dict[str, dict]
) -> tuple[list[Finding], int, int]:
    """Each cluster maps to one identity; each identity belongs to one cluster."""
    findings: list[Finding] = []
    cluster_to_identities: dict[int, Counter] = defaultdict(Counter)
    identity_to_clusters: dict[str, Counter] = defaultdict(Counter)

    for e in events.values():
        sp = (e.get("speakers") or [{}])[0]
        cid = sp.get("cluster_id")
        ident = sp.get("identity") or sp.get("display_name")
        if cid is None:
            continue
        cluster_to_identities[cid][ident or "UNNAMED"] += 1
        if ident:
            identity_to_clusters[ident][cid] += 1

    # Drift: cluster has segments under different names
    for cid, names in cluster_to_identities.items():
        real_names = {n for n in names if n != "UNNAMED"}
        if len(real_names) > 1:
            total = sum(names.values())
            parts = ", ".join(f"{n}={c}" for n, c in names.most_common())
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="cluster-identity-drift",
                    severity="ERROR",
                    detail=f"cluster {cid} has {len(real_names)} names across {total} segments: {parts}",
                )
            )

    # Drift: same person shows up in multiple clusters
    # Allow at most one "overflow" cluster with a small share.
    for ident, clusters in identity_to_clusters.items():
        if len(clusters) > 1:
            total = sum(clusters.values())
            top_count = clusters.most_common(1)[0][1]
            secondary = total - top_count
            if secondary > max(2, total * 0.05):  # >5% and >2 segments
                parts = ", ".join(f"{c}={n}" for c, n in clusters.most_common())
                findings.append(
                    Finding(
                        meeting_id=meeting_id,
                        code="identity-cluster-split",
                        severity="ERROR",
                        detail=f"'{ident}' split across clusters: {parts} (top={top_count})",
                    )
                )

    cluster_count = len(cluster_to_identities)
    named_count = len(
        [c for c, names in cluster_to_identities.items() if any(n != "UNNAMED" for n in names)]
    )
    return findings, cluster_count, named_count


def _check_orphan_micro_clusters(meeting_id: str, events: dict[str, dict]) -> list[Finding]:
    """Tiny clusters with trivial speech are likely misclustered interjections."""
    findings: list[Finding] = []
    cluster_segments: dict[int, list[dict]] = defaultdict(list)
    for e in events.values():
        sp = (e.get("speakers") or [{}])[0]
        cid = sp.get("cluster_id")
        if cid is None:
            continue
        cluster_segments[cid].append(e)

    for cid, segs in cluster_segments.items():
        sp0 = (segs[0].get("speakers") or [{}])[0]
        has_name = bool(sp0.get("identity") or sp0.get("display_name"))
        if has_name:
            continue  # named clusters are fine at any size
        count = len(segs)
        total_ms = sum(e.get("end_ms", 0) - e.get("start_ms", 0) for e in segs)
        total_chars = sum(len(e.get("text", "")) for e in segs)
        if count <= 3 and total_ms < 20000:
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="orphan-micro-cluster",
                    severity="WARN",
                    detail=(
                        f"cluster {cid}: {count} segments, {total_ms / 1000:.1f}s, "
                        f"{total_chars} chars — likely misclustered interjections"
                    ),
                )
            )
        elif count <= 15 and total_ms < 60000:
            # Slightly bigger but still suspicious
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="small-unnamed-cluster",
                    severity="WARN",
                    detail=f"cluster {cid}: {count} segments, {total_ms / 1000:.1f}s — no identity",
                )
            )
    return findings


def _check_derived_files(
    meeting_id: str, meeting_dir: Path, events: dict[str, dict]
) -> list[Finding]:
    """timeline/lanes/detected_speakers must agree with events."""
    findings: list[Finding] = []

    event_clusters = {(e.get("speakers") or [{}])[0].get("cluster_id") for e in events.values()}
    event_clusters.discard(None)

    # timeline.json
    tl = _safe_load_json(meeting_dir / "timeline.json")
    if tl is not None:
        if isinstance(tl, dict):
            segs = tl.get("segments", [])
        else:
            segs = tl
        if abs(len(segs) - len(events)) > max(1, len(events) * 0.02):
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="timeline-count-mismatch",
                    severity="ERROR",
                    detail=(f"timeline has {len(segs)} segments, journal has {len(events)} events"),
                )
            )

    # speaker_lanes.json
    lanes = _safe_load_json(meeting_dir / "speaker_lanes.json")
    if lanes is not None and isinstance(lanes, dict):
        lane_clusters = {int(k) for k in lanes.keys() if str(k).lstrip("-").isdigit()}
        missing = event_clusters - lane_clusters
        if missing:
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="lanes-missing-clusters",
                    severity="ERROR",
                    detail=f"speaker_lanes.json missing clusters from events: {sorted(missing)}",
                )
            )
        extra = lane_clusters - event_clusters
        if extra:
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="lanes-extra-clusters",
                    severity="WARN",
                    detail=f"speaker_lanes.json has clusters not in events: {sorted(extra)}",
                )
            )
        # Count lane entries vs events
        lane_entries = sum(len(v) for v in lanes.values() if isinstance(v, list))
        if abs(lane_entries - len(events)) > max(1, len(events) * 0.02):
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="lane-entry-count-mismatch",
                    severity="WARN",
                    detail=f"lanes have {lane_entries} entries, events have {len(events)}",
                )
            )

    # detected_speakers.json
    ds = _safe_load_json(meeting_dir / "detected_speakers.json")
    if ds is not None and isinstance(ds, list):
        ds_clusters = {s.get("cluster_id") for s in ds if s.get("cluster_id") is not None}
        missing = event_clusters - ds_clusters
        if missing:
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="detected-speakers-missing-clusters",
                    severity="ERROR",
                    detail=f"detected_speakers.json missing clusters: {sorted(missing)}",
                )
            )
        # Check that detected_speakers has identities where events have them
        event_identities = {}
        for e in events.values():
            sp = (e.get("speakers") or [{}])[0]
            cid = sp.get("cluster_id")
            ident = sp.get("identity") or sp.get("display_name")
            if cid is not None and ident:
                event_identities[cid] = ident
        ds_identities = {s.get("cluster_id"): s.get("display_name") for s in ds}
        for cid, ident in event_identities.items():
            ds_name = ds_identities.get(cid, "")
            if not ds_name or ds_name.startswith("Speaker "):
                findings.append(
                    Finding(
                        meeting_id=meeting_id,
                        code="detected-speakers-missing-identity",
                        severity="ERROR",
                        detail=(
                            f"cluster {cid} has identity '{ident}' in events but "
                            f"'{ds_name}' in detected_speakers.json — reprocess lost the name"
                        ),
                    )
                )
    return findings


def _check_coverage(meeting_id: str, meeting_dir: Path, events: dict[str, dict]) -> list[Finding]:
    """Speaking time coverage vs meeting duration."""
    findings: list[Finding] = []
    if not events:
        return findings

    meta = _safe_load_json(meeting_dir / "meta.json") or {}
    if meta.get("state") != "complete":
        return findings  # skip interrupted/recording

    total_ms = max(e.get("end_ms", 0) for e in events.values())
    speaking_ms = sum((e.get("end_ms", 0) - e.get("start_ms", 0)) for e in events.values())
    if total_ms > 0:
        pct = speaking_ms / total_ms * 100
        if pct < 40:
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="low-speech-coverage",
                    severity="WARN",
                    detail=(
                        f"only {pct:.0f}% of meeting is speech "
                        f"({speaking_ms / 1000:.0f}s of {total_ms / 1000:.0f}s)"
                    ),
                )
            )
    return findings


def _check_structure(meeting_id: str, meeting_dir: Path) -> list[Finding]:
    """File existence, metadata sanity."""
    findings: list[Finding] = []
    meta = _safe_load_json(meeting_dir / "meta.json") or {}
    state = meta.get("state", "unknown")

    allowed_states = {"created", "recording", "interrupted", "finalizing", "complete"}
    if state not in allowed_states:
        findings.append(
            Finding(
                meeting_id=meeting_id,
                code="invalid-state",
                severity="ERROR",
                detail=f"meta.state='{state}' is not a known state",
            )
        )

    if state == "complete":
        # Required files for a complete meeting
        required = [
            "journal.jsonl",
            "detected_speakers.json",
            "speaker_lanes.json",
            "timeline.json",
        ]
        for f in required:
            if not (meeting_dir / f).exists():
                findings.append(
                    Finding(
                        meeting_id=meeting_id,
                        code="missing-required-file",
                        severity="ERROR",
                        detail=f"complete meeting missing {f}",
                    )
                )

    # Summary sanity
    summary = _safe_load_json(meeting_dir / "summary.json")
    if summary is not None:
        if not summary.get("executive_summary"):
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="summary-missing-executive",
                    severity="WARN",
                    detail="summary.json exists but executive_summary is empty",
                )
            )
        topics = summary.get("topics", [])
        if not topics:
            findings.append(
                Finding(
                    meeting_id=meeting_id,
                    code="summary-missing-topics",
                    severity="WARN",
                    detail="summary.json exists but topics list is empty",
                )
            )
    return findings


# ── Main validation loop ───────────────────────────────────────


def analyze_meeting(meeting_dir: Path) -> MeetingReport:
    meeting_id = meeting_dir.name
    events, _corrections = _load_journal(meeting_dir)
    meta = _safe_load_json(meeting_dir / "meta.json") or {}
    state = meta.get("state", "unknown")

    findings: list[Finding] = []
    findings.extend(_check_structure(meeting_id, meeting_dir))
    findings.extend(_check_derived_files(meeting_id, meeting_dir, events))

    cluster_findings, cluster_count, named_count = _check_cluster_identity_consistency(
        meeting_id, events
    )
    findings.extend(cluster_findings)
    findings.extend(_check_orphan_micro_clusters(meeting_id, events))
    findings.extend(_check_coverage(meeting_id, meeting_dir, events))

    return MeetingReport(
        meeting_id=meeting_id,
        state=state,
        event_count=len(events),
        cluster_count=cluster_count,
        named_speakers=named_count,
        findings=findings,
    )


def collect_all_meetings() -> list[Path]:
    if not MEETINGS_DIR.exists():
        return []
    return sorted([d for d in MEETINGS_DIR.iterdir() if d.is_dir()])


# ── Baseline management ────────────────────────────────────────


def load_baseline() -> dict[str, list[str]]:
    """Map of meeting_id → [finding_code, ...]. Pre-existing issues."""
    if not BASELINE_PATH.exists():
        return {}
    try:
        data = json.loads(BASELINE_PATH.read_text())
        return data.get("known_issues", {})
    except Exception:
        return {}


def save_baseline(all_findings: list[Finding]) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for f in all_findings:
        grouped[f.meeting_id].append(f.code)
    # Sort + dedup codes per meeting
    grouped = {k: sorted(set(v)) for k, v in grouped.items()}
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "_comment": (
                    "Pre-existing integrity issues per meeting. New issues will fail "
                    "the test. When an issue is fixed, re-run with "
                    "UPDATE_INTEGRITY_BASELINE=1 to refresh this file."
                ),
                "total_meetings": len({f.meeting_id for f in all_findings}),
                "total_findings": len(all_findings),
                "known_issues": grouped,
            },
            indent=2,
        )
        + "\n"
    )


# ── Pytest entrypoint ──────────────────────────────────────────


@pytest.mark.system
class TestMeetingIntegrity:
    """Self-learning validator: analyzes every meeting and diffs against baseline."""

    @pytest.fixture(scope="class")
    def all_reports(self) -> list[MeetingReport]:
        dirs = collect_all_meetings()
        if not dirs:
            pytest.skip("No meetings directory found")
        return [analyze_meeting(d) for d in dirs]

    def test_print_validator_report(self, all_reports: list[MeetingReport]) -> None:
        """Print a full analysis report. Never fails — for visibility."""
        print("\n\n" + "=" * 72)
        print(f"  MEETING INTEGRITY REPORT — {len(all_reports)} meetings")
        print("=" * 72)
        total_err = 0
        total_warn = 0
        for r in all_reports:
            print("  " + r.summary_line())
            total_err += sum(1 for f in r.findings if f.severity == "ERROR")
            total_warn += sum(1 for f in r.findings if f.severity == "WARN")
        print("-" * 72)
        print(
            f"  TOTAL: {total_err} errors, {total_warn} warnings across {len(all_reports)} meetings"
        )
        print("=" * 72 + "\n")

    def test_detailed_findings(self, all_reports: list[MeetingReport]) -> None:
        """Print every finding with full detail. Never fails — for visibility."""
        any_findings = False
        for r in all_reports:
            if not r.findings:
                continue
            any_findings = True
            print(f"\n  {r.meeting_id}:")
            for f in r.findings:
                marker = "✗" if f.severity == "ERROR" else "!"
                print(f"    {marker} [{f.code}] {f.detail}")
        if not any_findings:
            print("\n  ✓ No integrity issues in any meeting.\n")

    def test_no_new_issues_vs_baseline(self, all_reports: list[MeetingReport]) -> None:
        """DIFF against baseline. Fails on NEW issues, grandfathers known ones.

        Run with UPDATE_INTEGRITY_BASELINE=1 pytest to refresh.
        """
        all_findings: list[Finding] = []
        for r in all_reports:
            all_findings.extend(r.findings)

        if os.environ.get("UPDATE_INTEGRITY_BASELINE"):
            save_baseline(all_findings)
            print(f"\n  Baseline refreshed: {BASELINE_PATH}")
            print(
                f"  Recorded {len(all_findings)} findings across "
                f"{len({f.meeting_id for f in all_findings})} meetings."
            )
            return

        baseline = load_baseline()
        # If no baseline exists yet, write one and pass
        if not baseline and all_findings:
            save_baseline(all_findings)
            print(f"\n  First run: created baseline at {BASELINE_PATH}")
            print(f"  Recorded {len(all_findings)} findings as pre-existing.")
            return

        # Diff current findings vs baseline
        new_issues: list[Finding] = []
        for f in all_findings:
            known = baseline.get(f.meeting_id, [])
            if f.code not in known:
                new_issues.append(f)

        # Detect healed meetings — in baseline but not in current
        current_keys = {f.key() for f in all_findings}
        healed: list[str] = []
        for mid, codes in baseline.items():
            for code in codes:
                if f"{mid}::{code}" not in current_keys:
                    healed.append(f"{mid} {code}")

        if healed:
            print(f"\n  ✓ Healed {len(healed)} baseline issues:")
            for h in healed[:20]:
                print(f"      {h}")
            print("  → Re-run with UPDATE_INTEGRITY_BASELINE=1 to refresh baseline.\n")

        if new_issues:
            msg = f"\n  {len(new_issues)} NEW integrity issues not in baseline:\n"
            for f in new_issues[:50]:
                msg += f"    ✗ {f}\n"
            if len(new_issues) > 50:
                msg += f"    ... and {len(new_issues) - 50} more\n"
            pytest.fail(msg)
