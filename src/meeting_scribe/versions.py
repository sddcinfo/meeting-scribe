"""Reprocess version snapshots + diff tooling.

Each time a meeting is reprocessed (re-run ASR + translation + diarization
from raw audio) we snapshot the prior derived artifacts into
``meetings/{id}/versions/{label}/`` so we can A/B compare runs.

Layout::

    meetings/{id}/
    ├── journal.jsonl              # current (latest)
    ├── summary.json
    ├── timeline.json
    ├── detected_speakers.json
    └── versions/
        ├── 2026-04-17T13-15-22__pre-dutch-fix/
        │   ├── journal.jsonl
        │   ├── summary.json
        │   ├── timeline.json
        │   ├── detected_speakers.json
        │   └── manifest.json
        └── 2026-04-17T14-02-08__post-dutch-fix/
            └── ...

The manifest captures the inputs that drove the run (model ids, language
pair, expected_speakers, code git hash, wall-clock per phase) so two
versions are comparable beyond just the content diff.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Files that get versioned. Listed explicitly so we don't accidentally
# snapshot huge audio files or slide decks (those are upstream inputs,
# not reprocess outputs).
SNAPSHOT_FILES: tuple[str, ...] = (
    "journal.jsonl",
    "summary.json",
    "timeline.json",
    "detected_speakers.json",
    "speaker_lanes.json",
    "speakers.json",
)


def _versions_dir(meeting_dir: Path) -> Path:
    return meeting_dir / "versions"


def _slugify(label: str) -> str:
    """Coerce a free-form label into a filesystem-safe slug."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", label.strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:48] or "v"


def _git_commit() -> str:
    """Best-effort git HEAD commit hash for the manifest. Empty on failure."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=str(Path(__file__).resolve().parent),
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return ""


def snapshot_meeting(
    meeting_dir: Path,
    label: str | None = None,
    *,
    inputs: dict[str, Any] | None = None,
) -> Path | None:
    """Snapshot the current derived artifacts into ``versions/{label}/``.

    Called BEFORE a reprocess run begins, so the previous outputs are
    preserved when the new ones overwrite the top-level files.

    ``inputs`` should contain whatever drove this run (model ids, language
    pair, expected_speakers, asr_url, etc.) — captured into manifest.json
    so a later diff can correlate output deltas with input changes.

    Returns the snapshot directory path, or None if there's nothing worth
    snapshotting (no journal yet — first-ever reprocess).
    """
    journal = meeting_dir / "journal.jsonl"
    if not journal.exists():
        return None

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H-%M-%S")
    suffix = f"__{_slugify(label)}" if label else ""
    snap_dir = _versions_dir(meeting_dir) / f"{ts}{suffix}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in SNAPSHOT_FILES:
        src = meeting_dir / name
        if src.exists():
            shutil.copy2(src, snap_dir / name)
            copied.append(name)

    manifest = {
        "label": label or "",
        "snapshot_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "source": "auto:reprocess",
        "files": copied,
        "git_commit": _git_commit(),
        "inputs": inputs or {},
    }
    (snap_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    logger.info(
        "Snapshot created: %s (%d files, label=%r)",
        snap_dir.name, len(copied), label,
    )
    return snap_dir


def list_versions(meeting_dir: Path) -> list[dict[str, Any]]:
    """List snapshots for a meeting, newest first."""
    vdir = _versions_dir(meeting_dir)
    if not vdir.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(vdir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {"_parse_error": True}
        out.append(
            {
                "name": d.name,
                "path": str(d),
                "manifest": manifest,
            }
        )
    return out


# ── Metric extraction (one number / structure per artifact) ─────


def _summarize_journal(path: Path) -> dict[str, Any]:
    """Per-version journal stats: counts, language tag distribution, durations."""
    if not path.exists():
        return {"present": False}
    finals: list[dict] = []
    by_segment: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "speaker_correction":
            continue
        if not e.get("is_final"):
            continue
        sid = e.get("segment_id")
        if not sid:
            continue
        # Highest-revision wins for same segment_id
        prev = by_segment.get(sid)
        if not prev or e.get("revision", 0) > prev.get("revision", 0):
            by_segment[sid] = e

    finals = list(by_segment.values())
    lang_counts: dict[str, int] = {}
    chars_total = 0
    translated_count = 0
    cluster_ids: set[int] = set()
    speech_ms = 0
    for e in finals:
        lang = e.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        chars_total += len(e.get("text") or "")
        tr = e.get("translation") or {}
        if tr.get("text"):
            translated_count += 1
        for sp in e.get("speakers") or []:
            cid = sp.get("cluster_id")
            if cid is not None and cid > 0:
                cluster_ids.add(cid)
        speech_ms += max(0, e.get("end_ms", 0) - e.get("start_ms", 0))

    return {
        "present": True,
        "segment_count": len(finals),
        "language_counts": lang_counts,
        "translated_segments": translated_count,
        "translation_coverage": (
            round(translated_count / len(finals), 4) if finals else 0.0
        ),
        "total_text_chars": chars_total,
        "unique_clusters_in_journal": len(cluster_ids),
        "total_speech_ms": speech_ms,
    }


def _summarize_speakers(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    try:
        speakers = json.loads(path.read_text())
    except Exception:
        return {"present": True, "_parse_error": True}
    if not isinstance(speakers, list):
        return {"present": True, "count": 0}
    return {
        "present": True,
        "count": len(speakers),
        "labels": [s.get("display_name") for s in speakers if isinstance(s, dict)],
        "segment_counts_per_speaker": [
            s.get("segment_count", 0) for s in speakers if isinstance(s, dict)
        ],
    }


def _summarize_summary_doc(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return {"present": True, "_parse_error": True}
    if not isinstance(doc, dict):
        return {"present": True, "_invalid_shape": True}
    return {
        "present": True,
        "executive_summary_chars": len(doc.get("executive_summary") or ""),
        "key_insights_count": len(doc.get("key_insights") or []),
        "action_items_count": len(doc.get("action_items") or []),
        "topics_count": len(doc.get("topics") or []),
        "decisions_count": len(doc.get("decisions") or []),
        "key_quotes_count": len(doc.get("key_quotes") or []),
        "schema_version": doc.get("schema_version"),
    }


def _summarize_timeline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return {"present": True, "_parse_error": True}
    segs = doc.get("segments") or []
    return {
        "present": True,
        "duration_ms": doc.get("duration_ms", 0),
        "segment_count": len(segs),
        "unique_speaker_ids": sorted({s.get("speaker_id") for s in segs if s.get("speaker_id")}),
    }


def _metrics_for(version_dir: Path) -> dict[str, Any]:
    """All metrics for one version (or one meeting's current state)."""
    return {
        "journal": _summarize_journal(version_dir / "journal.jsonl"),
        "speakers": _summarize_speakers(version_dir / "detected_speakers.json"),
        "summary": _summarize_summary_doc(version_dir / "summary.json"),
        "timeline": _summarize_timeline(version_dir / "timeline.json"),
    }


def metrics_for_current(meeting_dir: Path) -> dict[str, Any]:
    """Metrics for the CURRENT (latest) state of a meeting."""
    return _metrics_for(meeting_dir)


def metrics_for_version(meeting_dir: Path, version_name: str) -> dict[str, Any]:
    """Metrics for a specific snapshot version."""
    return _metrics_for(_versions_dir(meeting_dir) / version_name)


# ── Diff (verdict per dimension) ────────────────────────────────


def _verdict(delta: float, *, higher_better: bool, threshold: float = 0.05) -> str:
    """Verdict for a numeric delta: better/worse/same.

    ``threshold`` is the minimum relative-change magnitude to count as
    a real difference (5% by default — anything smaller is "same").
    """
    if abs(delta) < threshold:
        return "same"
    improved = (delta > 0) == higher_better
    return "better" if improved else "worse"


def _rel(a: float, b: float) -> float:
    """Relative change from a (baseline) to b (compare)."""
    if a == 0 and b == 0:
        return 0.0
    if a == 0:
        return 1.0 if b > 0 else -1.0
    return (b - a) / abs(a)


def diff_versions(
    baseline: dict[str, Any],
    compare: dict[str, Any],
) -> dict[str, Any]:
    """Produce a structured diff between two versions' metrics.

    Each dimension is graded better/worse/same, but the assumption of
    "better" is conservative: more transcript content + better translation
    coverage + LO reasonable speaker count is considered better. Wild
    increases (e.g. 6 → 18 speakers) are flagged as worse since real
    meetings have a fixed cap.
    """
    bj = baseline.get("journal", {})
    cj = compare.get("journal", {})
    bs = baseline.get("speakers", {})
    cs = compare.get("speakers", {})
    bm = baseline.get("summary", {})
    cm = compare.get("summary", {})

    out: dict[str, Any] = {"dimensions": {}, "totals": {"better": 0, "worse": 0, "same": 0}}

    def _grade(key: str, baseline_v: float, compare_v: float, higher_better: bool):
        delta = _rel(baseline_v, compare_v)
        verdict = _verdict(delta, higher_better=higher_better)
        out["dimensions"][key] = {
            "baseline": baseline_v,
            "compare": compare_v,
            "delta_rel": round(delta, 4),
            "verdict": verdict,
        }
        out["totals"][verdict] += 1

    _grade("transcript.segment_count", bj.get("segment_count", 0), cj.get("segment_count", 0), higher_better=True)
    _grade("transcript.total_text_chars", bj.get("total_text_chars", 0), cj.get("total_text_chars", 0), higher_better=True)
    _grade("transcript.translation_coverage", bj.get("translation_coverage", 0), cj.get("translation_coverage", 0), higher_better=True)
    _grade("transcript.total_speech_ms", bj.get("total_speech_ms", 0), cj.get("total_speech_ms", 0), higher_better=True)
    _grade("speakers.count", bs.get("count", 0), cs.get("count", 0), higher_better=False)  # fewer is usually better (less over-clustering)
    _grade("summary.key_insights", bm.get("key_insights_count", 0), cm.get("key_insights_count", 0), higher_better=True)
    _grade("summary.action_items", bm.get("action_items_count", 0), cm.get("action_items_count", 0), higher_better=True)
    _grade("summary.executive_chars", bm.get("executive_summary_chars", 0), cm.get("executive_summary_chars", 0), higher_better=True)

    # Language distribution change — surface as info, not graded
    out["language_distribution"] = {
        "baseline": bj.get("language_counts", {}),
        "compare": cj.get("language_counts", {}),
    }

    return out
