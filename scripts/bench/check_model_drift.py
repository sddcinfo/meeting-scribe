"""Drift watcher for HuggingFace model artifacts.

Companion to ``scripts/bench/check_stale_pins.py``: that script catches
PyPI library drift; this one catches *pretrained model* drift.  The
2026-Q2 retrospective documented the gap directly — pyannote.audio
4.0.4 was current on PyPI for weeks while we were still loading the
older `pyannote/speaker-diarization-3.1` pipeline.  Library version
and pretrained-model version are separate axes; both need watching.

Reads `scripts/bench/watchlist.yaml` and emits a Markdown report:

* ``models[]`` — for each pinned id, queries `HfApi.list_models` for
  siblings in the family glob.  Flags any sibling whose `lastModified`
  is newer than the pinned id by ≥ ``--stale-days``.
* ``closed_source_watchlist[]`` — probes each repo for any weights
  file (`*.safetensors`, `*.bin`).  When one appears, that's the
  "open weights have shipped" signal we're waiting for.

Exit codes:
  0  — no drift, no new open weights
  1  — at least one model is stale OR a closed-source candidate just
       opened (CI workflow uses this to open / update an issue)
  2  — error talking to HF Hub (network, auth)
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

WATCHLIST_DEFAULT = Path(__file__).resolve().parent / "watchlist.yaml"


# -- weights-file globs used to decide "open weights have shipped" -----
WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.ckpt", "*.pth", "*.gguf")


@dataclass(frozen=True)
class _PinnedModel:
    id: str
    family: str
    role: str


@dataclass(frozen=True)
class _ClosedSourceCandidate:
    id: str
    note: str


@dataclass
class _DriftRow:
    pinned_id: str
    role: str
    latest_sibling_id: str | None
    latest_sibling_modified: datetime | None
    days_newer: int | None
    error: str | None


@dataclass
class _OpenWeightsRow:
    repo_id: str
    note: str
    has_weights: bool
    weights_count: int
    error: str | None


def _load_watchlist(path: Path) -> tuple[list[_PinnedModel], list[_ClosedSourceCandidate]]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"watchlist not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    models = [
        _PinnedModel(id=m["id"], family=m["family"], role=m.get("role", "unspecified"))
        for m in (data.get("models") or [])
    ]
    closed = [
        _ClosedSourceCandidate(id=c["id"], note=c.get("note", ""))
        for c in (data.get("closed_source_watchlist") or [])
    ]
    return models, closed


def _model_last_modified(model_id: str, api) -> datetime | None:
    try:
        info = api.model_info(model_id)
    except Exception:
        return None
    last = getattr(info, "last_modified", None) or getattr(info, "lastModified", None)
    if last is None:
        return None
    if isinstance(last, str):
        return datetime.fromisoformat(last.replace("Z", "+00:00"))
    if isinstance(last, datetime):
        return last if last.tzinfo else last.replace(tzinfo=UTC)
    return None


def _evaluate_pinned(model: _PinnedModel, api) -> _DriftRow:
    """Return a drift row for ``model``."""
    pinned_modified = _model_last_modified(model.id, api)
    if pinned_modified is None:
        return _DriftRow(
            pinned_id=model.id,
            role=model.role,
            latest_sibling_id=None,
            latest_sibling_modified=None,
            days_newer=None,
            error="pinned model not found on HF Hub",
        )

    # Family glob → fetch the namespace and filter.
    namespace = model.family.rsplit("/", 1)[0]
    glob = model.family.rsplit("/", 1)[1]
    try:
        # `list_models(author=...)` is paginated; cap at 200 to avoid
        # runaway iteration on big orgs (Qwen has thousands).
        siblings = list(api.list_models(author=namespace, limit=200, full=False))
    except Exception as exc:
        return _DriftRow(
            pinned_id=model.id,
            role=model.role,
            latest_sibling_id=None,
            latest_sibling_modified=None,
            days_newer=None,
            error=f"list_models failed: {type(exc).__name__}: {exc}",
        )

    candidates = []
    for sib in siblings:
        sib_id = sib.modelId if hasattr(sib, "modelId") else sib.id
        if not fnmatch.fnmatch(sib_id, model.family):
            continue
        if sib_id == model.id:
            continue
        last = getattr(sib, "last_modified", None) or getattr(sib, "lastModified", None)
        if isinstance(last, str):
            last = datetime.fromisoformat(last.replace("Z", "+00:00"))
        if last is None:
            continue
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        candidates.append((sib_id, last))

    if not candidates:
        return _DriftRow(
            pinned_id=model.id,
            role=model.role,
            latest_sibling_id=None,
            latest_sibling_modified=None,
            days_newer=0,
            error=None,
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    latest_id, latest_mod = candidates[0]
    days_newer = (latest_mod - pinned_modified).days
    return _DriftRow(
        pinned_id=model.id,
        role=model.role,
        latest_sibling_id=latest_id,
        latest_sibling_modified=latest_mod,
        days_newer=days_newer,
        error=None,
    )


def _evaluate_closed_source(candidate: _ClosedSourceCandidate, api) -> _OpenWeightsRow:
    try:
        info = api.model_info(candidate.id)
    except Exception as exc:
        # 404 / RepositoryNotFoundError is the EXPECTED state for a
        # closed-source watchlist entry — that's literally the signal
        # we're waiting to disappear.  Treat it as "still closed", not
        # an error.  Anything else (network, auth, rate limit) is a
        # real error worth surfacing.
        msg = f"{type(exc).__name__}: {exc}".splitlines()[0]
        is_not_found = (
            "RepositoryNotFoundError" in type(exc).__name__
            or "404" in str(exc)
        )
        if is_not_found:
            return _OpenWeightsRow(
                repo_id=candidate.id,
                note=candidate.note,
                has_weights=False,
                weights_count=0,
                error=None,
            )
        return _OpenWeightsRow(
            repo_id=candidate.id,
            note=candidate.note,
            has_weights=False,
            weights_count=0,
            error=msg,
        )
    siblings = getattr(info, "siblings", None) or []
    files = [getattr(s, "rfilename", "") for s in siblings]
    weights = [
        f for f in files if any(fnmatch.fnmatch(f, pat) for pat in WEIGHT_GLOBS)
    ]
    return _OpenWeightsRow(
        repo_id=candidate.id,
        note=candidate.note,
        has_weights=bool(weights),
        weights_count=len(weights),
        error=None,
    )


def render_report(
    drift_rows: list[_DriftRow],
    open_rows: list[_OpenWeightsRow],
    *,
    stale_days: int,
) -> tuple[str, bool]:
    """Return ``(markdown, any_action_required)``.

    ``any_action_required`` is True when at least one drift row crosses
    the staleness floor OR at least one closed-source candidate just
    flipped to having open weights.
    """
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append(f"# Model drift report — {today}")
    lines.append("")
    lines.append(f"Stale threshold: **≥ {stale_days} days** newer than the pinned model.")
    lines.append("")

    drift_actionable = [
        r for r in drift_rows if r.days_newer is not None and r.days_newer >= stale_days
    ]
    drift_current = [
        r
        for r in drift_rows
        if r.error is None and (r.days_newer is None or r.days_newer < stale_days)
    ]
    drift_errored = [r for r in drift_rows if r.error]
    open_actionable = [r for r in open_rows if r.has_weights]
    open_idle = [r for r in open_rows if not r.has_weights and r.error is None]
    open_errored = [r for r in open_rows if r.error]

    lines.append(
        f"- Pinned models: **{len(drift_rows)}** — drift "
        f"actionable: **{len(drift_actionable)}**, current: **{len(drift_current)}**"
        + (f", errored: **{len(drift_errored)}**" if drift_errored else "")
    )
    lines.append(
        f"- Closed-source watchlist: **{len(open_rows)}** — newly open: "
        f"**{len(open_actionable)}**, still closed: **{len(open_idle)}**"
        + (f", errored: **{len(open_errored)}**" if open_errored else "")
    )
    lines.append("")

    if drift_actionable:
        lines.append(f"## Pinned model drift — {len(drift_actionable)} stale")
        lines.append("")
        lines.append("| Role | Pinned id | Latest sibling | Sibling lastModified | Days newer |")
        lines.append("|---|---|---|---|---:|")
        for r in sorted(drift_actionable, key=lambda r: -(r.days_newer or 0)):
            mod = r.latest_sibling_modified.isoformat() if r.latest_sibling_modified else "?"
            lines.append(
                f"| `{r.role}` | `{r.pinned_id}` | `{r.latest_sibling_id}` | {mod} | {r.days_newer} |"
            )
        lines.append("")

    if open_actionable:
        lines.append(f"## Closed-source candidates that just opened — {len(open_actionable)}")
        lines.append("")
        lines.append("| Repo | Weights count | Note |")
        lines.append("|---|---:|---|")
        for r in sorted(open_actionable, key=lambda r: r.repo_id):
            lines.append(f"| `{r.repo_id}` | {r.weights_count} | {r.note} |")
        lines.append("")

    if drift_current:
        lines.append(f"## Pinned models — {len(drift_current)} current")
        lines.append("")
        lines.append("<details><summary>Show</summary>")
        lines.append("")
        lines.append("| Role | Pinned id | Latest sibling | Days newer |")
        lines.append("|---|---|---|---:|")
        for r in sorted(drift_current, key=lambda r: r.pinned_id.lower()):
            sib = r.latest_sibling_id or "—"
            days = r.days_newer if r.days_newer is not None else "?"
            lines.append(f"| `{r.role}` | `{r.pinned_id}` | `{sib}` | {days} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if open_idle:
        lines.append(f"## Closed-source candidates still closed — {len(open_idle)}")
        lines.append("")
        lines.append("<details><summary>Show</summary>")
        lines.append("")
        lines.append("| Repo | Note |")
        lines.append("|---|---|")
        for r in sorted(open_idle, key=lambda r: r.repo_id):
            lines.append(f"| `{r.repo_id}` | {r.note} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if drift_errored or open_errored:
        lines.append(
            f"## Errored — {len(drift_errored) + len(open_errored)} repo(s) failed to query"
        )
        lines.append("")
        lines.append("| Repo | Error |")
        lines.append("|---|---|")
        for r in drift_errored:
            lines.append(f"| `{r.pinned_id}` | {r.error} |")
        for r in open_errored:
            lines.append(f"| `{r.repo_id}` | {r.error} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_This report never auto-bumps anything.  Each candidate "
        "promotion still goes through the bench-validation discipline of "
        "`reports/2026-Q2-bench/2026-Q2-RETROSPECTIVE.md` (Tier 1 / Tier 2)._"
    )
    return "\n".join(lines) + "\n", bool(drift_actionable or open_actionable)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watchlist", type=Path, default=WATCHLIST_DEFAULT)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--stale-days", type=int, default=30)
    args = p.parse_args(argv)

    try:
        models, closed = _load_watchlist(args.watchlist)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Watchlist: {len(models)} pinned model(s), {len(closed)} closed-source candidate(s).",
        file=sys.stderr,
    )

    from huggingface_hub import HfApi

    api = HfApi()
    drift_rows = [_evaluate_pinned(m, api) for m in models]
    open_rows = [_evaluate_closed_source(c, api) for c in closed]

    report, actionable = render_report(drift_rows, open_rows, stale_days=args.stale_days)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"Wrote report to {args.report}", file=sys.stderr)
    else:
        print(report)

    if any(r.error for r in drift_rows + open_rows):
        return 2
    return 1 if actionable else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "WEIGHT_GLOBS",
    "_PinnedModel",
    "_ClosedSourceCandidate",
    "_DriftRow",
    "_OpenWeightsRow",
    "_load_watchlist",
    "_evaluate_pinned",
    "_evaluate_closed_source",
    "render_report",
    "main",
    # Tests reach for these helpers; importing via importlib in tests
    # uses these names.
    "datetime",
    "timedelta",
]
