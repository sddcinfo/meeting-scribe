"""Journal replay harness — pipes scrubbed journals through the JS-side
SegmentStore and asserts deterministic structural snapshots.

Why a structural snapshot, not a text diff: synthetic placeholder text
isn't load-bearing — what matters is segment count, segment_id ordering,
language column routing, and translation-state preservation across
reorderings. The snapshot captures those invariants per fixture.

Why this layer: it catches the bugs that motivated the SegmentStore
regression suite (translation lost to furigana race, speakers wiped by
later furigana, control-event phantom segments) AT THE WHOLE-MEETING
SCALE, not just per-segment. A regression in the merge logic or
listener fan-out shows up here as a snapshot diff with hundreds of
differences instead of a single-test failure.

Snapshot files live alongside the fixtures at
``tests/fixtures/journals/<name>.snapshot.json``. Regenerate with
``UPDATE_SNAPSHOTS=1 pytest tests/test_journal_replay.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "journals"
JS_REPLAY = REPO_ROOT / "tests" / "js" / "_journal_replay_runner.mjs"


def _node_available() -> bool:
    return shutil.which("node") is not None


def _replay_journal(fixture: Path) -> dict:
    """Run the JS replay runner against ``fixture`` and return its
    JSON-serialized SegmentStore digest.

    The runner is a thin wrapper around the production
    ``segment-store.js`` — same code that runs in the popout.
    """
    proc = subprocess.run(
        ["node", str(JS_REPLAY), str(fixture)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"journal replay runner failed for {fixture.name}:\n"
            f"  stdout: {proc.stdout}\n"
            f"  stderr: {proc.stderr}"
        )
    return json.loads(proc.stdout)


def _digest_to_snapshot(digest: dict) -> dict:
    """Convert the runner's raw digest into the snapshot shape.

    The snapshot is structural ONLY — segment_ids, language routing,
    translation-state availability, speaker-cluster ids. The actual
    text is intentionally omitted because it's synthetic placeholder
    content; structural drift is what we care about.
    """
    return {
        "segment_count": digest["count"],
        "order": digest["order"],
        "by_segment": {
            sid: {
                "language": s.get("language"),
                "is_final": s.get("is_final"),
                "has_translation": bool(s.get("translation")),
                "translation_status": (s.get("translation") or {}).get("status"),
                "translation_target_language": (s.get("translation") or {}).get("target_language"),
                "speakers": [
                    {"cluster_id": sp.get("cluster_id"), "source": sp.get("source")}
                    for sp in (s.get("speakers") or [])
                ],
                "revision": s.get("revision"),
            }
            for sid, s in digest["segments"].items()
        },
    }


@pytest.fixture(scope="module")
def fixtures_present() -> list[Path]:
    return sorted(FIXTURES.glob("*.jsonl"))


def test_fixture_corpus_present(fixtures_present):
    """Sanity: at least the 5 named fixtures exist."""
    names = {p.stem for p in fixtures_present}
    expected = {
        "monolingual_en",
        "bilingual_en_ja",
        "same_script_en_de",
        "with_speaker_remap",
        "with_translation_failure",
    }
    missing = expected - names
    assert not missing, (
        f"expected fixtures missing: {missing}. "
        f"Run scripts/scrub_journal.py to regenerate."
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "monolingual_en",
        "bilingual_en_ja",
        "same_script_en_de",
        "with_speaker_remap",
        "with_translation_failure",
    ],
)
def test_journal_replay_snapshot(fixture_name):
    """For each scrubbed fixture, replay through SegmentStore and
    compare against the committed snapshot.

    A diff means the SegmentStore merge logic, listener fan-out, or
    a related code path has changed in a way that affects the
    converged state — investigate before regenerating the snapshot.
    """
    if not _node_available():
        pytest.skip("node not on PATH")

    fixture = FIXTURES / f"{fixture_name}.jsonl"
    if not fixture.exists():
        pytest.skip(f"fixture missing: {fixture}")

    snapshot_path = FIXTURES / f"{fixture_name}.snapshot.json"
    digest = _replay_journal(fixture)
    actual = _digest_to_snapshot(digest)

    if os.environ.get("UPDATE_SNAPSHOTS") == "1" or not snapshot_path.exists():
        snapshot_path.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n")
        if not os.environ.get("UPDATE_SNAPSHOTS"):
            pytest.skip(f"baseline snapshot created at {snapshot_path}")
        return

    expected = json.loads(snapshot_path.read_text())
    if actual != expected:
        # Provide a focused diff hint — full diff is too noisy for a
        # 100+ segment fixture.
        a_count = actual["segment_count"]
        e_count = expected["segment_count"]
        diff_segments: list[str] = []
        for sid in actual["by_segment"]:
            if sid in expected["by_segment"] and actual["by_segment"][sid] != expected["by_segment"][sid]:
                diff_segments.append(sid)
                if len(diff_segments) >= 5:
                    break
        msg = (
            f"snapshot diff for {fixture_name}:\n"
            f"  segment_count: actual={a_count} expected={e_count}\n"
            f"  drifted segments (first 5): {diff_segments}\n"
            f"  Re-run with UPDATE_SNAPSHOTS=1 if the drift is intentional."
        )
        assert actual == expected, msg
