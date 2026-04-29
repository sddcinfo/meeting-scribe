"""library sub-group: ls / stats / verify (read-only meeting inspection)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import _parse_duration_spec


@cli.group()
def library() -> None:
    """Read-only meeting library inspection (ls, stats, verify)."""


def _library_audit() -> list[dict]:
    from meeting_scribe.config import ServerConfig
    from meeting_scribe.storage import MeetingStorage

    cfg = ServerConfig.from_env()
    return MeetingStorage(cfg).audit_meetings()


def _apply_library_filters(
    rows: list[dict],
    state: str | None,
    min_events: int | None,
    max_events: int | None,
    older_than: str | None,
    newer_than: str | None,
) -> list[dict]:
    older_h = _parse_duration_spec(older_than)
    newer_h = _parse_duration_spec(newer_than)
    out = []
    for m in rows:
        if state and m["state"] != state:
            continue
        if min_events is not None and m["journal_lines"] < min_events:
            continue
        if max_events is not None and m["journal_lines"] >= max_events:
            continue
        if older_h is not None and m["age_hours"] < older_h:
            continue
        if newer_h is not None and m["age_hours"] > newer_h:
            continue
        out.append(m)
    return out


@library.command("ls")
@click.option(
    "--state",
    type=click.Choice(["recording", "interrupted", "complete", "stopped", "finalizing"]),
    default=None,
    help="Filter by state",
)
@click.option("--min-events", type=int, default=None, help="Only meetings with ≥ N events")
@click.option("--max-events", type=int, default=None, help="Only meetings with < N events")
@click.option("--older-than", default=None, help="Only meetings older than SPEC (e.g. 7d, 48h)")
@click.option("--newer-than", default=None, help="Only meetings newer than SPEC")
@click.option(
    "--ids-only",
    is_flag=True,
    help="Emit only meeting IDs, one per line (for piping into xargs).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def library_ls(
    state: str | None,
    min_events: int | None,
    max_events: int | None,
    older_than: str | None,
    newer_than: str | None,
    ids_only: bool,
    as_json: bool,
) -> None:
    """List meetings with filters — pipeable into reprocess/finalize/etc.

    Examples:
      meeting-scribe library ls --state interrupted --min-events 200
      meeting-scribe library ls --older-than 30d --max-events 50 --ids-only \
        | xargs -I{} meeting-scribe cleanup --exclude {}   # dry inspection
    """
    import json as _json

    rows = _apply_library_filters(
        _library_audit(), state, min_events, max_events, older_than, newer_than
    )
    if ids_only:
        for m in rows:
            click.echo(m["meeting_id"])
        return
    if as_json:
        serial = [
            {k: (str(v) if hasattr(v, "__fspath__") else v) for k, v in m.items()} for m in rows
        ]
        click.echo(_json.dumps(serial, indent=2))
        return
    click.echo(f"{'ID':<38} {'STATE':<12} {'EVENTS':>7} {'AUDIO':>8} {'AGE':>8}  SUMMARY")
    for m in rows:
        click.echo(
            f"{m['meeting_id']:<38} {m['state']:<12} "
            f"{m['journal_lines']:>7} {m['audio_duration_s']:>7.0f}s "
            f"{m['age_hours']:>7.0f}h  {'yes' if m['has_summary'] else 'no'}"
        )
    click.echo(f"\n{len(rows)} meeting(s)")


@library.command("stats")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def library_stats(as_json: bool) -> None:
    """Aggregate library stats: state counts, disk usage, event histogram."""
    import json as _json
    from collections import Counter

    rows = _library_audit()
    if not rows:
        click.echo("Library is empty.")
        return

    by_state = Counter(m["state"] for m in rows)
    events = [m["journal_lines"] for m in rows]
    audio_s = [m["audio_duration_s"] for m in rows]

    total_bytes = 0
    audio_bytes = 0
    journal_bytes = 0
    summary_bytes = 0
    other_bytes = 0
    for m in rows:
        d = Path(m["meeting_dir"])
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            total_bytes += sz
            name = f.name
            if name == "summary.json":
                summary_bytes += sz
            elif name == "journal.jsonl":
                journal_bytes += sz
            elif f.suffix in (".pcm", ".wav", ".mp3", ".opus"):
                audio_bytes += sz
            else:
                other_bytes += sz

    buckets = [
        (0, 50),
        (50, 100),
        (100, 200),
        (200, 500),
        (500, 1000),
        (1000, None),
    ]
    histogram: list[dict[str, Any]] = []
    for lo, hi in buckets:
        if hi is None:
            n = sum(1 for e in events if e >= lo)
            label = f"{lo}+"
        else:
            n = sum(1 for e in events if lo <= e < hi)
            label = f"{lo}-{hi}"
        histogram.append({"range": label, "count": n})

    sorted_events = sorted(events)
    stats: dict[str, Any] = {
        "total_meetings": len(rows),
        "by_state": dict(by_state),
        "events": {
            "min": min(events),
            "max": max(events),
            "mean": round(sum(events) / len(events), 1),
            "median": sorted_events[len(events) // 2],
        },
        "audio_seconds": {
            "min": round(min(audio_s), 1),
            "max": round(max(audio_s), 1),
            "mean": round(sum(audio_s) / len(audio_s), 1),
            "total_hours": round(sum(audio_s) / 3600, 2),
        },
        "disk_bytes": {
            "total": total_bytes,
            "audio": audio_bytes,
            "journal": journal_bytes,
            "summary": summary_bytes,
            "other": other_bytes,
        },
        "event_histogram": histogram,
    }

    if as_json:
        click.echo(_json.dumps(stats, indent=2))
        return

    def _mb(b: int) -> str:
        return f"{b / 1024 / 1024:7.1f} MB"

    click.secho(f"Meeting Library Stats — {len(rows)} meetings", bold=True)
    click.echo()
    click.echo("By state:")
    for s, n in sorted(by_state.items(), key=lambda x: -x[1]):
        click.echo(f"  {s:<14} {n}")
    click.echo()
    click.echo(
        f"Events:  min={stats['events']['min']}  "
        f"median={stats['events']['median']}  "
        f"mean={stats['events']['mean']}  "
        f"max={stats['events']['max']}"
    )
    click.echo(
        f"Audio:   total={stats['audio_seconds']['total_hours']}h  "
        f"mean={stats['audio_seconds']['mean']}s  "
        f"max={stats['audio_seconds']['max']}s"
    )
    click.echo()
    click.echo("Disk:")
    click.echo(f"  total    {_mb(total_bytes)}")
    click.echo(f"  audio    {_mb(audio_bytes)}")
    click.echo(f"  journal  {_mb(journal_bytes)}")
    click.echo(f"  summary  {_mb(summary_bytes)}")
    click.echo(f"  other    {_mb(other_bytes)}")
    click.echo()
    click.echo("Event-count histogram:")
    max_count = max((h["count"] for h in histogram), default=0)
    for h in histogram:
        bar_len = int(40 * h["count"] / max_count) if max_count else 0
        bar = "█" * bar_len
        click.echo(f"  {h['range']:>12}  {h['count']:>4}  {bar}")


@library.command("verify")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def library_verify(as_json: bool) -> None:
    """Integrity check — parseable meta, valid JSONL, audio sanity, state consistency.

    Read-only. Reports findings; does not mutate. Run `meeting-scribe cleanup`
    or `meeting-scribe finalize` to act on the findings.
    """
    import json as _json

    from meeting_scribe.config import ServerConfig

    cfg = ServerConfig.from_env()
    rows = _library_audit()
    audited = {Path(m["meeting_dir"]).name for m in rows}
    issues: list[dict] = []

    meetings_root = Path(cfg.meetings_dir)
    if meetings_root.exists():
        for d in sorted(meetings_root.iterdir()):
            if not d.is_dir() or d.name == "__pycache__":
                continue
            if d.name not in audited:
                issues.append(
                    {
                        "meeting_id": d.name,
                        "severity": "error",
                        "issue": "missing or unparseable meta.json",
                    }
                )

    for m in rows:
        mid = m["meeting_id"]
        d = Path(m["meeting_dir"])

        jp = d / "journal.jsonl"
        if jp.exists():
            bad = 0
            with jp.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _json.loads(line)
                    except _json.JSONDecodeError:
                        bad += 1
            if bad:
                issues.append(
                    {
                        "meeting_id": mid,
                        "severity": "error",
                        "issue": f"{bad} malformed journal line(s)",
                    }
                )

        if m["journal_lines"] >= 50 and m["audio_duration_s"] < 1:
            issues.append(
                {
                    "meeting_id": mid,
                    "severity": "warning",
                    "issue": f"{m['journal_lines']} events but 0s audio",
                }
            )

        if m["state"] in ("interrupted", "stopped") and m["has_summary"] and m["has_timeline"]:
            issues.append(
                {
                    "meeting_id": mid,
                    "severity": "warning",
                    "issue": (
                        f"state={m['state']} but already has summary+timeline "
                        "(run `meeting-scribe finalize` to transition to complete)"
                    ),
                }
            )

        if m["state"] == "stopped":
            issues.append(
                {
                    "meeting_id": mid,
                    "severity": "warning",
                    "issue": "legacy 'stopped' state (run `meeting-scribe finalize`)",
                }
            )

    if as_json:
        click.echo(
            _json.dumps(
                {
                    "checked": len(rows),
                    "issue_count": len(issues),
                    "issues": issues,
                },
                indent=2,
            )
        )
        return

    click.secho(f"Verified {len(rows)} meetings — {len(issues)} issue(s)", bold=True)
    if not issues:
        click.secho("  all clean", fg="green")
        return
    for iss in issues:
        color = "red" if iss["severity"] == "error" else "yellow"
        click.secho(
            f"  [{iss['severity']:<7}] {iss['meeting_id']}  {iss['issue']}",
            fg=color,
        )
