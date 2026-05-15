"""versions sub-group: list + diff reprocess snapshots."""

from __future__ import annotations

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import PROJECT_ROOT


@cli.group()
def versions() -> None:
    """Reprocess version snapshots — list + diff past runs against current.

    Every reprocess auto-snapshots the prior journal/summary/timeline/
    speakers into ``meetings/{id}/versions/{ts}__pre-reprocess/``. Use
    these commands to compare runs and judge whether a code/model change
    actually improved transcription quality.
    """


@versions.command("list")
@click.option("-m", "--meeting-id", required=True, help="Meeting id to list versions for")
def versions_list(meeting_id: str) -> None:
    """List snapshots for a meeting (newest first)."""
    from meeting_scribe.versions import list_versions

    meetings_dir = PROJECT_ROOT / "meetings"
    mdir = meetings_dir / meeting_id
    if not mdir.is_dir():
        click.secho(f"Meeting not found: {meeting_id}", fg="red")
        return
    rows = list_versions(mdir)
    if not rows:
        click.echo("(no snapshots — this meeting hasn't been reprocessed yet)")
        return
    click.secho(f"{len(rows)} snapshot(s) for {meeting_id}:", fg="cyan", bold=True)
    for r in rows:
        m = r["manifest"]
        label = m.get("label") or "(no label)"
        ts = m.get("snapshot_at_utc", "")
        commit = (m.get("git_commit") or "")[:8] or "-"
        inputs = m.get("inputs") or {}
        lp = ",".join(inputs.get("language_pair", []) or [])
        es = inputs.get("expected_speakers")
        click.echo(
            f"  {r['name']}"
            f"\n    label={label}  ts={ts}  git={commit}"
            f"  language_pair={lp or '-'}  expected_speakers={es if es else '-'}"
        )


@versions.command("diff")
@click.option("-m", "--meeting-id", required=True, help="Meeting id")
@click.option(
    "--baseline",
    default=None,
    help="Snapshot dir name to use as baseline (default: most recent snapshot)",
)
@click.option(
    "--compare",
    default=None,
    help="Snapshot dir name to compare to baseline (default: current state)",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def versions_diff(
    meeting_id: str,
    baseline: str | None,
    compare: str | None,
    as_json: bool,
) -> None:
    """Compare two versions of a meeting (or the latest snapshot vs current).

    By default: ``--baseline`` is the most recent snapshot and ``--compare``
    is the current top-level state. So after running reprocess, this
    command tells you "did the new run improve things vs the prior run".
    """
    import json as _json

    from meeting_scribe.versions import (
        diff_versions,
        list_versions,
        metrics_for_current,
        metrics_for_version,
    )

    meetings_dir = PROJECT_ROOT / "meetings"
    mdir = meetings_dir / meeting_id
    if not mdir.is_dir():
        click.secho(f"Meeting not found: {meeting_id}", fg="red")
        return

    snaps = list_versions(mdir)
    if not snaps and baseline is None:
        click.secho("(no snapshots yet — run reprocess once to create one)", fg="yellow")
        return

    if baseline is None:
        baseline = snaps[0]["name"]
    base_metrics = metrics_for_version(mdir, baseline)
    base_label = baseline

    if compare is None:
        cmp_metrics = metrics_for_current(mdir)
        cmp_label = "(current)"
    else:
        cmp_metrics = metrics_for_version(mdir, compare)
        cmp_label = compare

    diff = diff_versions(base_metrics, cmp_metrics)

    if as_json:
        click.echo(
            _json.dumps(
                {"baseline": base_label, "compare": cmp_label, "diff": diff},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    click.secho(f"Diff: {base_label}  →  {cmp_label}", fg="cyan", bold=True)
    click.echo()
    color_for = {"better": "green", "worse": "red", "same": "white"}
    sign_for = {"better": "▲", "worse": "▼", "same": "·"}
    for key, info in diff["dimensions"].items():
        verdict = info["verdict"]
        delta_pct = info["delta_rel"] * 100.0
        click.secho(
            f"  {sign_for[verdict]} {key:<40} {info['baseline']!s:>10}  →  {info['compare']!s:<10}"
            f"  ({delta_pct:+.1f}%)  [{verdict}]",
            fg=color_for[verdict],
        )
    t = diff["totals"]
    click.echo()
    click.secho(
        f"Totals: {t['better']} better · {t['worse']} worse · {t['same']} same",
        bold=True,
    )
    if diff.get("language_distribution"):
        click.echo()
        click.secho("Language tag distribution:", fg="cyan")
        click.echo(f"  baseline: {diff['language_distribution']['baseline']}")
        click.echo(f"  compare:  {diff['language_distribution']['compare']}")
