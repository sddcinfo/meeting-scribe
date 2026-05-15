"""Ingest a recorded meeting fixture into meeting-scribe's storage.

Creates a complete meeting with journal, audio, room layout, and metadata
that can be reviewed in the meeting history UI.

Usage:
    PYTHONPATH=src .venv/bin/python3 scripts/ingest_recording.py \
        --fixture ~/test-fixtures/90min_english_2026-04-07 \
        --speakers "Brad,John,Leon,Joel,Mark"
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@click.command()
@click.option(
    "--fixture",
    required=True,
    type=click.Path(exists=True),
    help="Path to recorded meeting fixture directory",
)
@click.option(
    "--speakers",
    default="",
    help="Comma-separated speaker names (for room layout seats)",
)
@click.option(
    "--meetings-dir",
    default=None,
    help="Meeting storage directory (default: ./meetings)",
)
def ingest(fixture: str, speakers: str, meetings_dir: str | None) -> None:
    """Ingest a recorded meeting fixture for review in the UI."""
    fixture_path = Path(fixture)
    storage_dir = Path(meetings_dir) if meetings_dir else Path(__file__).parent.parent / "meetings"

    # Validate fixture
    journal_path = fixture_path / "journal.jsonl"
    pcm_path = fixture_path / "audio" / "recording.pcm"
    meta_path = fixture_path / "meta.json"

    if not journal_path.exists():
        click.secho(f"No journal.jsonl in {fixture_path}", fg="red")
        sys.exit(1)
    if not pcm_path.exists():
        click.secho(f"No audio/recording.pcm in {fixture_path}", fg="red")
        sys.exit(1)

    # Load journal
    with open(journal_path) as f:
        segments = [json.loads(line) for line in f]
    finals = [s for s in segments if s["is_final"] and s["text"].strip()]

    click.echo(f"Fixture: {fixture_path.name}")
    click.echo(f"  Segments: {len(finals)} final")
    click.echo(f"  Duration: {finals[-1]['end_ms'] / 1000 / 60:.1f} min")
    click.echo(f"  Audio: {pcm_path.stat().st_size / 1024 / 1024:.0f} MB")

    # Load or create metadata
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meeting_id = meta.get("meeting_id", str(uuid.uuid4()))
    else:
        meeting_id = str(uuid.uuid4())
        meta = {}

    # Create meeting directory
    meeting_dir = storage_dir / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = meeting_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Copy audio
    click.echo(f"  Copying audio ({pcm_path.stat().st_size / 1024 / 1024:.0f} MB)...")
    shutil.copy2(pcm_path, audio_dir / "recording.pcm")

    # Copy journal
    shutil.copy2(journal_path, meeting_dir / "journal.jsonl")

    # Create metadata
    meta_out = {
        "meeting_id": meeting_id,
        "state": "complete",
        "created_at": meta.get(
            "created_at",
            datetime.now(UTC).isoformat(),
        ),
        "organizer_token_hash": "",
        "invite_code_hash": "",
        "max_attendees": 10,
        "audio_sample_rate": 16000,
        "enrolled_speakers": {},
    }
    with open(meeting_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    # Create room layout with speaker seats
    speaker_names = [s.strip() for s in speakers.split(",") if s.strip()]
    if not speaker_names:
        speaker_names = _detect_speakers(finals)

    seats = []
    n = len(speaker_names)
    for i, name in enumerate(speaker_names):
        # Arrange seats around a rectangle
        if i < n // 2:
            x = 20 + (60 * i / max(n // 2, 1))
            y = 28
        else:
            x = 20 + (60 * (i - n // 2) / max(n - n // 2, 1))
            y = 72
        seats.append(
            {
                "seat_id": str(uuid.uuid4()),
                "x": round(x, 1),
                "y": round(y, 1),
                "enrollment_id": None,
                "speaker_name": name,
            }
        )

    room = {
        "preset": "rectangle",
        "tables": [
            {
                "table_id": str(uuid.uuid4()),
                "x": 50.0,
                "y": 50.0,
                "width": 44.0,
                "height": 22.0,
                "border_radius": 3.0,
                "label": "",
            }
        ],
        "seats": seats,
    }
    with open(meeting_dir / "room.json", "w") as f:
        json.dump(room, f, indent=2)

    # Generate timeline from journal
    timeline = []
    for seg in finals:
        timeline.append(
            {
                "segment_id": seg["segment_id"],
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg["text"],
                "language": seg["language"],
                "speakers": seg.get("speakers", []),
            }
        )
    with open(meeting_dir / "timeline.json", "w") as f:
        json.dump(timeline, f)

    # Create detected speakers file
    detected = []
    for name in speaker_names:
        detected.append(
            {
                "display_name": name,
                "matched_enrollment_id": None,
                "match_confidence": 0.0,
                "segment_count": 0,
                "first_seen_ms": 0,
                "last_seen_ms": finals[-1]["end_ms"],
            }
        )
    with open(meeting_dir / "detected_speakers.json", "w") as f:
        json.dump(detected, f, indent=2)

    # Empty speakers enrollment
    with open(meeting_dir / "speakers.json", "w") as f:
        json.dump([], f)

    click.secho(f"\nMeeting ingested: {meeting_id}", fg="green")
    click.echo(f"  Storage: {meeting_dir}")
    click.echo(f"  Speakers: {', '.join(speaker_names)}")
    click.echo(f"  Segments: {len(finals)}")
    click.echo(f"  Duration: {finals[-1]['end_ms'] / 1000 / 60:.1f} min")
    click.echo("\n  View at: https://<host>:8080 → Meeting History")


def _detect_speakers(segments: list[dict]) -> list[str]:
    """Detect speaker names from transcript content."""
    all_text = " ".join(s["text"] for s in segments)

    # Known names from the recording analysis
    candidates = ["Brad", "John", "Leon", "Joel", "Mark", "Angela", "Rosa", "Gary"]
    found = []
    for name in candidates:
        count = all_text.count(name)
        if count >= 3:  # Mentioned 3+ times = likely a participant
            found.append((name, count))

    found.sort(key=lambda x: -x[1])
    names = [name for name, _ in found]

    if not names:
        names = ["Speaker 1", "Speaker 2", "Speaker 3"]

    return names


if __name__ == "__main__":
    ingest()
