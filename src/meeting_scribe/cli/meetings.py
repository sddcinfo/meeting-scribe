"""Meeting-scoped commands: finalize / reprocess / full-reprocess / cleanup / reprocess-summaries.

All commands here operate on per-meeting directories under
``<meetings_root>/<id>/``. The shared ``_resolve_meeting_ids``
helper accepts literal ids, ``-`` for stdin, and ``@file.txt`` for
file-based id lists so the commands compose with
``meeting-scribe library ls --ids-only`` pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import (
    PROJECT_ROOT,
    _parse_duration_spec,
)


@cli.command()
@click.option("--meeting-id", "-m", default=None, help="Specific meeting ID")
def finalize(meeting_id: str | None) -> None:
    """Finalize interrupted meetings: generate timeline and summary.

    Finds meetings stuck in 'interrupted', 'recording', 'stopped', or
    'reprocessing' state and completes their post-processing (timeline,
    summary, state -> complete). The 'reprocessing' branch recovers
    meetings whose reprocess was killed mid-flight (for example by a
    systemd stop timeout); the journal and artifacts are intact in that
    case, only the state flag needs to be flipped back.
    """
    import asyncio

    from meeting_scribe.summary import generate_summary

    meetings_dir = PROJECT_ROOT / "meetings"
    if not meetings_dir.exists():
        click.secho("No meetings directory found.", fg="red")
        return

    dirs = [meetings_dir / meeting_id] if meeting_id else sorted(meetings_dir.iterdir())

    finalized = 0
    for meeting_dir in dirs:
        if not meeting_dir.is_dir():
            continue
        meta_path = meeting_dir / "meta.json"
        if not meta_path.exists():
            continue

        import json as _json

        meta = _json.loads(meta_path.read_text())
        state = meta.get("state", "")
        mid = meeting_dir.name

        # `stopped` is a legacy state from an older schema (pre-MeetingState
        # enum cleanup) that still shows up on meetings recorded before the
        # refactor. Semantically it's the same as `interrupted` — a meeting
        # whose recording ended without a clean finalize — so fold it in
        # here instead of silently skipping and leaving the user stuck.
        # `reprocessing` is a transient state set by reprocess_meeting() at
        # step 0 and cleared at step 7. If the server is killed in between
        # (e.g. systemd TimeoutStopUSec fires while a long reprocess is
        # holding the event loop), the flag stays. Journal + derived
        # artifacts are left untouched until later phases, so the meeting
        # is still viewable; flip the flag back to 'complete' here.
        if state not in ("interrupted", "recording", "stopped", "reprocessing"):
            if meeting_id:
                click.echo(f"  {mid}: state is '{state}', not recoverable")
            continue

        journal = meeting_dir / "journal.jsonl"
        if not journal.exists():
            click.echo(f"  {mid}: no journal, skipping")
            continue

        click.echo(f"  {mid}: finalizing (was '{state}')...")

        # Generate timeline if missing
        timeline_path = meeting_dir / "timeline.json"
        if not timeline_path.exists():
            events = []
            for line in journal.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = _json.loads(line)
                except Exception:
                    continue
                if e.get("is_final") and e.get("text"):
                    sp = e.get("speakers", [])
                    events.append(
                        {
                            "segment_id": e.get("segment_id", ""),
                            "start_ms": e.get("start_ms", 0),
                            "end_ms": e.get("end_ms", 0),
                            "language": e.get("language", "unknown"),
                            "speaker_id": sp[0].get("cluster_id", 0) if sp else 0,
                            "text": e.get("text", "")[:100],
                        }
                    )
            pcm = meeting_dir / "audio" / "recording.pcm"
            duration_ms = int(pcm.stat().st_size / 32000 * 1000) if pcm.exists() else 0
            timeline_path.write_text(
                _json.dumps({"duration_ms": duration_ms, "segments": events}, indent=2)
            )
            click.echo(f"    timeline: {len(events)} segments")

        # Generate summary if missing
        summary_path = meeting_dir / "summary.json"
        if not summary_path.exists():
            try:
                summary = asyncio.run(
                    generate_summary(meeting_dir, vllm_url="http://localhost:8010")
                )
                if "error" not in summary:
                    click.secho(f"    summary: {len(summary.get('topics', []))} topics", fg="green")
                else:
                    click.secho(f"    summary: {summary['error']}", fg="yellow")
            except Exception as e:
                click.secho(f"    summary failed: {e}", fg="yellow")

        # Update state to complete
        meta["state"] = "complete"
        meta_path.write_text(_json.dumps(meta, indent=2))
        click.secho("    state → complete", fg="green")
        finalized += 1

    if finalized:
        click.secho(f"Finalized {finalized} meeting(s).", fg="green")
    elif not meeting_id:
        click.echo("No interrupted meetings found.")


def _resolve_meeting_ids(
    values: tuple[str, ...],
    meetings_dir: Path,
    require_explicit: bool,
) -> list[Path]:
    """Resolve ``-m`` option values to a list of meeting directories.

    Supports three forms, mixable in a single invocation:
      - literal meeting id (full UUID or 8-char prefix): ``-m 415bfa55``
      - stdin marker: ``-m -`` reads one id per line from stdin
      - file reference: ``-m @file.txt`` reads one id per line from file

    Blank lines and lines starting with ``#`` are ignored in both stdin
    and file forms so that the output of something like
    ``meeting-scribe library ls --ids-only`` pipes cleanly, and so users
    can annotate their id lists without breaking parsing.

    When ``values`` is empty and ``require_explicit`` is False, returns
    every meeting dir that has a ``journal.jsonl``. When empty and
    ``require_explicit`` is True, raises ``click.UsageError`` — used by
    destructive commands like ``full-reprocess`` that must never
    silently operate on every meeting.
    """
    ids: list[str] = []
    for v in values:
        if v == "-":
            for line in sys.stdin.read().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
            continue
        if v.startswith("@"):
            f = Path(v[1:]).expanduser()
            if not f.exists():
                raise click.BadParameter(f"id file not found: {f}")
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
            continue
        ids.append(v)

    if not ids:
        if require_explicit:
            raise click.UsageError("no meeting ids given; use -m <id>, -m -, or -m @file.txt")
        return sorted(
            [d for d in meetings_dir.iterdir() if d.is_dir() and (d / "journal.jsonl").exists()],
            key=lambda d: d.stat().st_mtime,
        )

    resolved: list[Path] = []
    missing: list[str] = []
    for mid in ids:
        exact = meetings_dir / mid
        if exact.exists() and exact.is_dir():
            resolved.append(exact)
            continue
        # Prefix match (8-char or similar).
        candidates = [d for d in meetings_dir.iterdir() if d.is_dir() and d.name.startswith(mid)]
        if len(candidates) == 1:
            resolved.append(candidates[0])
        elif len(candidates) == 0:
            missing.append(mid)
        else:
            raise click.BadParameter(f"ambiguous id {mid!r}: matches {len(candidates)} meetings")
    if missing:
        raise click.BadParameter(f"meeting(s) not found: {', '.join(missing)}")
    return resolved


@cli.command()
@click.option(
    "--meeting-id",
    "-m",
    "meeting_ids",
    multiple=True,
    help=(
        "Specific meeting id (repeatable). Also accepts `-m -` to read ids "
        "from stdin and `-m @file.txt` to read from a file. Default: all meetings."
    ),
)
@click.option(
    "--vllm-url", default="http://localhost:8010", help="vLLM endpoint for summary generation"
)
def reprocess(meeting_ids: tuple[str, ...], vllm_url: str) -> None:
    """Re-process meetings: generate summaries and polished transcripts.

    Re-runs AI summary generation and refinement on completed meetings.
    Useful after model upgrades or when processing was interrupted.

    Pipe from `library ls --ids-only` for batch runs:
        meeting-scribe library ls --state complete --max-events 300 --ids-only \\
          | meeting-scribe reprocess -m -
    """
    import asyncio

    from meeting_scribe.summary import generate_summary

    meetings_dir = PROJECT_ROOT / "meetings"
    if not meetings_dir.exists():
        click.secho("No meetings directory found.", fg="red")
        return

    dirs = _resolve_meeting_ids(meeting_ids, meetings_dir, require_explicit=False)
    if not dirs:
        click.echo("No meetings matched.")
        return

    click.echo(f"Re-processing {len(dirs)} meeting(s)...")

    for meeting_dir in dirs:
        mid = meeting_dir.name
        journal = meeting_dir / "journal.jsonl"
        if not journal.exists():
            click.echo(f"  {mid}: no journal, skipping")
            continue

        existing_summary = meeting_dir / "summary.json"
        if existing_summary.exists():
            click.echo(f"  {mid}: summary already exists, regenerating...")
        else:
            click.echo(f"  {mid}: generating summary...")

        from meeting_scribe.server_support.summary_status import (
            SummaryStatus,
            classify_summary_error,
            next_attempt_id,
            write_status,
        )

        attempt = next_attempt_id(meeting_dir)
        write_status(
            meeting_dir,
            SummaryStatus.GENERATING,
            attempt_id=attempt,
            journal_path=journal,
        )
        try:
            summary = asyncio.run(generate_summary(meeting_dir, vllm_url=vllm_url))
            if isinstance(summary, dict) and "error" in summary:
                code = classify_summary_error(summary)
                write_status(
                    meeting_dir,
                    SummaryStatus.ERROR,
                    attempt_id=attempt,
                    journal_path=journal,
                    error_code=code,
                )
                click.secho(f"  {mid}: {code.value}", fg="yellow")
            else:
                write_status(
                    meeting_dir,
                    SummaryStatus.COMPLETE,
                    attempt_id=attempt,
                    journal_path=journal,
                )
                topics = len(summary.get("topics", []))
                actions = len(summary.get("action_items", []))
                click.secho(f"  {mid}: {topics} topics, {actions} action items", fg="green")
        except Exception as e:
            code = classify_summary_error(e)
            write_status(
                meeting_dir,
                SummaryStatus.ERROR,
                attempt_id=attempt,
                journal_path=journal,
                error_code=code,
            )
            click.secho(f"  {mid}: failed ({code.value})", fg="red")

    click.secho("Re-processing complete!", fg="green")


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would happen without changing anything")
@click.option(
    "--yes", "-y", is_flag=True, help="Apply all actions without interactive confirmation"
)
@click.option(
    "--vllm-url", default="http://localhost:8010", help="vLLM endpoint for summary generation"
)
@click.option(
    "--min-events",
    type=int,
    default=0,
    help="Delete meetings with fewer than N journal events (default 0 = conservative). "
    "Applies to all meetings regardless of state. Use e.g. --min-events 200 to prune low-signal meetings.",
)
@click.option(
    "--older-than",
    default=None,
    help="Also delete meetings older than SPEC (e.g. 7d, 48h, 2w). Stacks with --min-events "
    "— a meeting must satisfy BOTH thresholds to be deleted when both are given.",
)
@click.option(
    "--exclude",
    default=None,
    help="Comma-separated list of meeting IDs (or 8-char prefixes) to protect from any action.",
)
@click.option(
    "--archive",
    "archive_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Instead of permanently deleting, move meetings to this directory. "
        "Created on demand. Any existing entry with the same id is overwritten. "
        "Must not be inside the meetings dir (would cause re-audit loops). "
        "Use this for reversible cleanup runs on a cron."
    ),
)
def cleanup(
    dry_run: bool,
    yes: bool,
    vllm_url: str,
    min_events: int,
    older_than: str | None,
    exclude: str | None,
    archive_dir: Path | None,
) -> None:
    """Audit and clean up the meeting library.

    Actions performed (in order):
      1. Finalize interrupted/stopped meetings with substantial content (>60s audio, >5 events)
      2. Regenerate missing summaries for completed meetings with content
      3. Delete meetings below the event threshold and/or older than age threshold
         (default: only 0-event empties >1h old)
      4. Delete corrupt meeting directories (missing/unparseable meta.json)

    Flags:
      --min-events N      prune meetings with < N journal events
      --older-than SPEC   prune meetings older than SPEC (7d, 48h, 2w, 30m)
      --exclude ID[,ID]   protect specific meetings (full UUID or 8-char prefix)

    When both --min-events and --older-than are given, a meeting must satisfy
    BOTH thresholds to be deleted (AND semantics — conservative default).

    Legacy `stopped` state (pre-enum-cleanup schema) is treated as
    `interrupted` for finalization. High-audio / low-event meetings are
    flagged as suspected ASR failures before deletion.

    Use --dry-run to preview. Use --yes to skip the confirmation prompt.
    """
    import asyncio
    import json as _json
    import shutil as _shutil

    from meeting_scribe.config import ServerConfig
    from meeting_scribe.storage import MeetingStorage
    from meeting_scribe.summary import generate_summary

    older_than_hours = _parse_duration_spec(older_than)
    excluded: set[str] = set()
    if exclude:
        excluded = {x.strip() for x in exclude.split(",") if x.strip()}

    def _is_excluded(meeting_id: str) -> bool:
        return any(meeting_id == e or meeting_id.startswith(e) for e in excluded)

    config = ServerConfig.from_env()
    storage = MeetingStorage(config)
    audit = storage.audit_meetings()

    # Validate archive destination up front so we fail fast, before any
    # mutation. Refuse dirs inside the meetings root because the next
    # audit pass would scan them as bogus meetings.
    archive_resolved: Path | None = None
    if archive_dir is not None:
        archive_resolved = archive_dir.expanduser().resolve()
        meetings_root_resolved = Path(config.meetings_dir).resolve()
        try:
            archive_resolved.relative_to(meetings_root_resolved)
            raise click.BadParameter(
                f"--archive path {archive_resolved} is inside the meetings dir "
                f"{meetings_root_resolved}; pick a location outside it"
            )
        except ValueError:
            pass  # not inside meetings dir — good

    # Skip the currently-recording meeting
    active_file = Path("/tmp/meeting-scribe-active.json")
    active_id: str | None = None
    if active_file.exists():
        try:
            active_id = _json.loads(active_file.read_text()).get("meeting_id")
        except Exception:
            pass

    # Classify meetings. Note: `audit_meetings()` silently skips dirs with
    # missing/corrupt meta.json, so we enumerate the meetings root directly
    # to catch those as a separate "corrupt" bucket.
    to_finalize: list[dict] = []
    to_regen_summary: list[dict] = []
    to_delete: list[dict] = []
    to_delete_corrupt: list[Path] = []
    # Meetings that would be deleted but have >=60s of audio — almost
    # always an ASR failure, not a garbage recording. Surfaced as a
    # warning in the report so the user can rescue them with
    # `meeting-scribe full-reprocess` before deciding to delete.
    suspected_asr_failures: list[dict] = []

    audited_ids = {Path(m["meeting_dir"]).name for m in audit}
    meetings_root = Path(config.meetings_dir)
    if meetings_root.exists():
        for d in sorted(meetings_root.iterdir()):
            if (
                d.is_dir()
                and d.name not in audited_ids
                and d.name != "__pycache__"
                and not _is_excluded(d.name)
            ):
                to_delete_corrupt.append(d)

    for m in audit:
        if m["meeting_id"] == active_id:
            continue  # Never touch the active meeting
        if m["state"] == "recording":
            continue
        if _is_excluded(m["meeting_id"]):
            continue

        # Build the delete predicate. When both --min-events and --older-than
        # are set, require BOTH (AND semantics) — that matches user intent
        # of "trim small old meetings" rather than "trim anything small OR
        # anything old", which would be surprising and hard to undo.
        events_below = min_events > 0 and m["journal_lines"] < min_events
        age_above = older_than_hours is not None and m["age_hours"] > older_than_hours

        if min_events > 0 and older_than_hours is not None:
            delete_me = events_below and age_above
        elif min_events > 0:
            delete_me = events_below
        elif older_than_hours is not None:
            delete_me = age_above
        else:
            # Legacy conservative fallback: 0 events, 0 audio, >1h old.
            delete_me = m["audio_duration_s"] < 5 and m["journal_lines"] == 0 and m["age_hours"] > 1

        if delete_me:
            to_delete.append(m)
            # Suspected ASR failure: ≥60s of audio but a transcription rate
            # below ~5 events/min. Normal Japanese/English speech produces
            # 10–60 finalized events per minute; anything under 5 almost
            # always means the ASR backend was offline or misconfigured
            # when the meeting was recorded. We deliberately DO NOT key
            # this off the user's `--min-events` threshold — that value
            # expresses "noise level I don't care about", not "ASR was
            # broken", and conflating them would cry wolf on every prune.
            if m["audio_duration_s"] >= 60:
                events_per_min = m["journal_lines"] / (m["audio_duration_s"] / 60)
                if events_per_min < 5:
                    suspected_asr_failures.append(m)
            continue

        # Legacy `stopped` state is folded in with `interrupted` here (same
        # semantics — meeting ended without clean finalize). finalize command
        # also accepts it for standalone reruns.
        if (
            m["state"] in ("interrupted", "stopped")
            and m["audio_duration_s"] >= 60
            and m["journal_lines"] >= 5
        ):
            to_finalize.append(m)
            continue

        if m["state"] == "complete" and not m["has_summary"] and m["journal_lines"] >= 5:
            to_regen_summary.append(m)

    # Print audit
    click.secho("─" * 72, fg="cyan")
    click.secho(f"Meeting Library Audit — {len(audit)} total meetings", bold=True)
    click.secho("─" * 72, fg="cyan")
    action_word = "archive" if archive_resolved else "delete"
    click.echo(f"  Active (skipped):          {'1' if active_id else '0'}")
    click.echo(f"  To finalize (interrupted): {len(to_finalize)}")
    click.echo(f"  To regenerate summary:     {len(to_regen_summary)}")
    if min_events > 0:
        click.echo(f"  To {action_word} (<{min_events} events):    {len(to_delete)}")
    else:
        click.echo(f"  To {action_word} (empty):         {len(to_delete)}")
    click.echo(f"  To {action_word} (corrupt):       {len(to_delete_corrupt)}")
    if archive_resolved:
        click.secho(f"  Archive destination:       {archive_resolved}", fg="cyan")

    if to_finalize:
        click.echo()
        click.secho("Interrupted meetings to finalize:", fg="yellow")
        for m in to_finalize:
            click.echo(
                f"  {m['meeting_id']}  {m['audio_duration_s']:.0f}s audio  "
                f"{m['journal_lines']} events  age={m['age_hours']:.0f}h"
            )

    if to_regen_summary:
        click.echo()
        click.secho("Complete meetings missing summary:", fg="yellow")
        for m in to_regen_summary:
            click.echo(
                f"  {m['meeting_id']}  {m['audio_duration_s']:.0f}s audio  "
                f"{m['journal_lines']} events"
            )

    if to_delete:
        click.echo()
        verb = "archive" if archive_resolved else "delete"
        label = (
            f"Meetings to {verb} (<{min_events} events):"
            if min_events > 0
            else f"Empty meetings to {verb}:"
        )
        click.secho(label, fg="red")
        for m in to_delete:
            click.echo(
                f"  {m['meeting_id']}  state={m['state']}  "
                f"audio={m['audio_duration_s']:.0f}s  events={m['journal_lines']}  "
                f"age={m['age_hours']:.0f}h"
            )

    if to_delete_corrupt:
        click.echo()
        click.secho("Corrupt meeting dirs to delete (missing/unparseable meta.json):", fg="red")
        for d in to_delete_corrupt:
            try:
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            except OSError:
                size = 0
            click.echo(f"  {d.name}  size={size // 1024}KB")

    if suspected_asr_failures:
        click.echo()
        click.secho(
            "⚠  Suspected ASR failures — would be deleted but have substantial audio:",
            fg="yellow",
            bold=True,
        )
        click.echo("   These meetings have ≥60s of recorded audio but very few journal")
        click.echo("   events. Likely ASR backend was offline/misconfigured when they")
        click.echo("   were recorded. Consider `meeting-scribe full-reprocess -m <id>`")
        click.echo("   before deletion, or `--exclude <id>` to protect them.")
        for m in suspected_asr_failures:
            click.echo(
                f"     {m['meeting_id']}  audio={m['audio_duration_s']:.0f}s  "
                f"events={m['journal_lines']}"
            )

    if excluded:
        click.echo()
        click.secho(
            f"Protected by --exclude (skipped from all actions): {len(excluded)} id(s)",
            fg="cyan",
        )

    if dry_run:
        click.echo()
        click.secho("(dry-run — no changes made)", fg="cyan")
        return

    if not (to_finalize or to_regen_summary or to_delete or to_delete_corrupt):
        click.echo()
        click.secho("Nothing to do — meeting library is clean!", fg="green")
        return

    if not yes:
        click.echo()
        if not click.confirm("Apply these changes?", default=False):
            click.echo("Aborted.")
            return

    # Apply: finalize + regen summary (both go through generate_summary)
    finalized = 0
    for m in to_finalize + to_regen_summary:
        meeting_dir = m["meeting_dir"]
        mid = m["meeting_id"]
        click.echo(f"  {mid}: generating summary...")

        # Generate summary (LLM call)
        try:
            summary = asyncio.run(generate_summary(meeting_dir, vllm_url=vllm_url))
            if "error" in summary:
                click.secho(f"  {mid}: summary failed: {summary['error']}", fg="yellow")
                continue

            # Transition state to complete if it was interrupted or
            # the legacy `stopped`. Leaving a finalized summary attached
            # to an "interrupted" meeting is a UI lie — the meeting's
            # output is done, the state should reflect that.
            meta_path = meeting_dir / "meta.json"
            if meta_path.exists():
                meta = _json.loads(meta_path.read_text())
                if meta.get("state") in ("interrupted", "stopped"):
                    meta["state"] = "complete"
                    meta_path.write_text(_json.dumps(meta, indent=2))

            topics = len(summary.get("topics", []))
            actions = len(summary.get("action_items", []))
            click.secho(
                f"  {mid}: ✓ {topics} topics, {actions} action items",
                fg="green",
            )
            finalized += 1
        except Exception as e:
            click.secho(f"  {mid}: failed ({e})", fg="red")

    # Helper: archive-or-delete. Archive is a reversible move to an
    # external directory; if an entry with the same name already exists
    # in the archive (re-running cleanup hit the same meeting twice)
    # we overwrite it — the newer copy is always the one the user cares
    # about, and refusing would leave the user with a surprise error
    # mid-run.
    def _remove_dir(src: Path) -> None:
        if archive_resolved is None:
            _shutil.rmtree(src)
            return
        archive_resolved.mkdir(parents=True, exist_ok=True)
        dest = archive_resolved / src.name
        if dest.exists():
            _shutil.rmtree(dest)
        _shutil.move(str(src), str(dest))

    verb_past = "archived" if archive_resolved else "deleted"

    # Apply: remove below-threshold meetings
    removed = 0
    for m in to_delete:
        meeting_dir = m["meeting_dir"]
        mid = m["meeting_id"]
        try:
            _remove_dir(meeting_dir)
            reason = f"<{min_events} events" if min_events > 0 else "empty"
            click.secho(f"  {mid}: {verb_past} ({reason})", fg="red")
            removed += 1
        except Exception as e:
            click.secho(f"  {mid}: {verb_past[:-1]} failed ({e})", fg="red")

    # Apply: remove corrupt directories
    removed_corrupt = 0
    for d in to_delete_corrupt:
        try:
            _remove_dir(d)
            click.secho(f"  {d.name}: {verb_past} (corrupt)", fg="red")
            removed_corrupt += 1
        except Exception as e:
            click.secho(f"  {d.name}: {verb_past[:-1]} failed ({e})", fg="red")

    click.echo()
    click.secho("─" * 72, fg="cyan")
    click.secho(
        f"Cleanup complete: {finalized} finalized, {removed} {verb_past}, "
        f"{removed_corrupt} corrupt {verb_past}",
        fg="green",
        bold=True,
    )
    if archive_resolved:
        click.secho(f"Archive: {archive_resolved}", fg="cyan")


@cli.command("full-reprocess")
@click.option(
    "--meeting-id",
    "-m",
    "meeting_ids",
    multiple=True,
    required=True,
    help=(
        "Meeting id to fully reprocess (repeatable). Also accepts `-m -` "
        "to read from stdin and `-m @file.txt` to read from a file. "
        "Required — full-reprocess never operates on the whole library silently."
    ),
)
@click.option("--asr-url", default="http://localhost:8003", help="ASR endpoint")
@click.option("--translate-url", default="http://localhost:8010", help="Translation endpoint")
@click.option(
    "--expected-speakers",
    type=click.IntRange(1, 12),
    default=None,
    help=(
        "Pin the speaker count when known. Constrains pyannote per-chunk "
        "and forces the cluster collapse to exactly N speakers."
    ),
)
def full_reprocess(
    meeting_ids: tuple[str, ...],
    asr_url: str,
    translate_url: str,
    expected_speakers: int | None,
) -> None:
    """Fully reprocess one or more meetings — re-run ASR + translation on raw audio.

    Reads the raw PCM recording, re-transcribes with Qwen3-ASR,
    translates all segments, and regenerates timeline + summary for
    each meeting. Original journal is backed up as journal.jsonl.bak.

    Examples:
      meeting-scribe full-reprocess -m 415bfa55
      meeting-scribe full-reprocess -m 415bfa55 -m dba10719
      meeting-scribe library ls --state complete --max-events 50 --ids-only \\
        | meeting-scribe full-reprocess -m -     # rescue suspected ASR failures
    """
    import asyncio
    import json as _json

    from meeting_scribe.config import ServerConfig
    from meeting_scribe.reprocess import reprocess_meeting
    from meeting_scribe.storage import MeetingStorage

    meetings_dir = PROJECT_ROOT / "meetings"
    dirs = _resolve_meeting_ids(meeting_ids, meetings_dir, require_explicit=True)

    # The CLI reprocess path runs outside the server process, so we
    # build a standalone MeetingStorage rooted at the same meetings
    # directory. transition_state() only touches meta.json + an
    # in-process journal registry, so the fresh instance is safe
    # even if the live server is also running (both serialize via
    # per-file writes).
    storage = MeetingStorage(ServerConfig(meetings_dir=meetings_dir))

    successes = 0
    failures: list[tuple[str, str]] = []

    for meeting_dir in dirs:
        meeting_id = meeting_dir.name
        pcm = meeting_dir / "audio" / "recording.pcm"
        if not pcm.exists():
            click.secho(f"{meeting_id}: no recording.pcm found — skipping.", fg="yellow")
            failures.append((meeting_id, "no recording.pcm"))
            continue

        duration_s = pcm.stat().st_size / (16000 * 2)

        # Read languages per meeting — mixing e.g. ja/en and it/en in
        # one batch must still respect each meeting's original config.
        # Length-1 (monolingual) meetings are respected too — reprocess
        # skips the translation pass end-to-end.
        meta_path = meeting_dir / "meta.json"
        language_pair: list[str] = ["en", "ja"]
        if meta_path.exists():
            try:
                meta_data = _json.loads(meta_path.read_text())
                lp = meta_data.get("language_pair", ["en", "ja"])
                from meeting_scribe.languages import is_valid_languages

                if isinstance(lp, list) and is_valid_languages(lp):
                    language_pair = list(lp)
            except Exception:
                pass

        click.secho(f"Full reprocess: {meeting_id}", fg="cyan", bold=True)
        click.echo(f"  Audio: {duration_s:.0f}s ({pcm.stat().st_size / 1024 / 1024:.0f}MB)")
        click.echo(f"  ASR: {asr_url}")
        click.echo(f"  Translation: {translate_url}")
        click.echo(f"  Languages: {'/'.join(language_pair)}")

        def on_progress(step: int, total: int, msg: str) -> None:
            click.echo(f"  [{step}/{total}] {msg}")

        try:
            result = asyncio.run(
                reprocess_meeting(
                    meeting_dir,
                    storage,
                    asr_url=asr_url,
                    translate_url=translate_url,
                    language_pair=language_pair,
                    on_progress=on_progress,
                    expected_speakers=expected_speakers,
                )
            )
        except Exception as e:
            click.secho(f"  Crashed: {e}", fg="red")
            failures.append((meeting_id, str(e)))
            continue

        if "error" in result:
            click.secho(f"  Error: {result['error']}", fg="red")
            failures.append((meeting_id, result["error"]))
            continue

        click.secho(
            f"  Done: {result['segments']} segments, {result['translated']} translated, "
            f"{result.get('speakers', 0)} speakers, "
            f"summary={'yes' if result.get('has_summary') else 'no'}",
            fg="green",
        )
        successes += 1
        click.echo()

    if len(dirs) > 1:
        click.echo()
        click.secho("─" * 72, fg="cyan")
        click.secho(
            f"Full reprocess complete: {successes} ok, {len(failures)} failed",
            fg="green" if not failures else "yellow",
            bold=True,
        )
        for mid, reason in failures:
            click.echo(f"  {mid}  {reason}")


@cli.command("reprocess-summaries")
@click.option(
    "--dry-run", is_flag=True, help="List meetings that would be processed without calling the LLM"
)
@click.option(
    "--resume", is_flag=True, help="Skip meetings already processed (from checkpoint state)"
)
@click.option("--vllm-url", default="http://localhost:8010", help="vLLM endpoint URL")
def reprocess_summaries(dry_run: bool, resume: bool, vllm_url: str) -> None:
    """Regenerate all meeting summaries with the v2 schema.

    Processes each completed meeting sequentially. Backs up existing summaries,
    validates v2 output before replacing, and checkpoints progress for resumability.
    """
    import asyncio
    import json as _json
    import shutil

    from meeting_scribe.config import ServerConfig
    from meeting_scribe.summary import generate_summary

    config = ServerConfig.from_env()
    meetings_dir = config.meetings_dir
    state_path = meetings_dir / "reprocess-state.json"

    if not meetings_dir.exists():
        click.secho("No meetings directory found", fg="red")
        raise SystemExit(1)

    # Load checkpoint state
    done_ids: set[str] = set()
    if resume and state_path.exists():
        with open(state_path) as f:
            state = _json.load(f)
            done_ids = set(state.get("completed", []))
        click.echo(f"Resuming: {len(done_ids)} meetings already processed")

    # Find all completed meetings with transcripts
    candidates = []
    for d in sorted(meetings_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        journal_path = d / "journal.jsonl"
        if not meta_path.exists() or not journal_path.exists():
            continue
        try:
            meta = _json.loads(meta_path.read_text())
        except _json.JSONDecodeError:
            continue
        if meta.get("state") != "complete":
            continue
        if resume and d.name in done_ids:
            continue
        candidates.append(d)

    if not candidates:
        click.secho("No meetings to process", fg="yellow")
        return

    click.echo(f"Found {len(candidates)} meeting(s) to reprocess")

    if dry_run:
        for d in candidates:
            summary_path = d / "summary.json"
            status = "has summary" if summary_path.exists() else "no summary"
            click.echo(f"  {d.name} ({status})")
        click.echo(f"\nDry run: {len(candidates)} meetings would be processed")
        return

    # Process sequentially
    succeeded = 0
    failed = 0
    failed_ids: list[str] = []

    for i, meeting_dir in enumerate(candidates, 1):
        meeting_id = meeting_dir.name
        summary_path = meeting_dir / "summary.json"
        backup_path = meeting_dir / "summary.v1.bak.json"

        click.echo(f"[{i}/{len(candidates)}] {meeting_id}: ", nl=False)

        try:
            # Backup existing summary
            if summary_path.exists() and not backup_path.exists():
                shutil.copy2(summary_path, backup_path)

            # Generate new summary (writes to summary.json internally)
            # We need to intercept and validate, so we generate to tmp first
            result = asyncio.run(
                generate_summary(
                    meeting_dir=meeting_dir,
                    vllm_url=vllm_url,
                )
            )

            if result.get("error"):
                click.secho(f"FAILED ({result['error']})", fg="red")
                failed += 1
                failed_ids.append(meeting_id)
                # Restore backup if summary.json was overwritten
                if backup_path.exists() and not summary_path.exists():
                    shutil.copy2(backup_path, summary_path)
                continue

            # Validate v2 fields are present
            if not result.get("key_insights") or not result.get("named_entities"):
                click.secho("FAILED (missing v2 fields)", fg="red")
                failed += 1
                failed_ids.append(meeting_id)
                # Restore backup
                if backup_path.exists():
                    shutil.copy2(backup_path, summary_path)
                continue

            click.secho("OK", fg="green")
            succeeded += 1

            # Update checkpoint
            done_ids.add(meeting_id)
            state_path.write_text(
                _json.dumps(
                    {
                        "completed": sorted(done_ids),
                        "last_updated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    indent=2,
                )
                + "\n"
            )

        except Exception as e:
            click.secho(f"FAILED ({e})", fg="red")
            failed += 1
            failed_ids.append(meeting_id)
            # Restore backup
            if backup_path.exists() and summary_path.exists():
                # generate_summary already wrote; restore backup
                shutil.copy2(backup_path, summary_path)

    click.echo()
    click.secho(
        f"Done: {succeeded} succeeded, {failed} failed", fg="green" if failed == 0 else "yellow"
    )
    if failed_ids:
        click.echo("Failed meetings:")
        for mid in failed_ids:
            click.echo(f"  {mid}")
