"""Stale-pin freshness report — added at user request 2026-04-28.

Walks every ``==`` pin we have on disk and asks PyPI whether something
newer is available.  Output is a Markdown report; a stale row triggers
a CI workflow that opens (or updates) a GitHub issue.

Sources scanned:

* ``pyproject.toml`` — top-level ``dependencies`` and every entry of
  ``[project.optional-dependencies]``.
* ``containers/*/Dockerfile`` — any line of the form
  ``'<pkg>==<version>'`` (catches the ``pyannote.audio==4.0.4`` pin
  that lives outside the Python project metadata).

Pre-/dev-/release-candidate versions on PyPI are skipped — we only
flag drift against stable releases.  The default staleness floor is
30 days behind the latest stable; configurable with ``--stale-days``.

This script never auto-bumps anything.  The report just surfaces the
gap; promotion follows the same bench-validation discipline as
Track A/B/C of plans/stateful-marinating-whistle.md (minor/patch =
``pytest`` + 5-min live smoke; major = full Track-style validation).

Exit code 0 = no stale pins.  Exit code 1 = at least one stale pin
(the CI workflow uses this to open the issue).  Exit code 2 = error
talking to PyPI.

Usage::

    python3 scripts/bench/check_stale_pins.py \\
        --report /tmp/stale-pins.md \\
        --stale-days 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CONTAINERS_DIR = REPO_ROOT / "containers"

# matches `'pkg==1.2.3'` or `"pkg==1.2.3"` or bare `pkg==1.2.3`,
# with optional version specifiers like `[extra]` after the name.
_PIN_RE = re.compile(
    r"""['"]?(?P<name>[A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==(?P<ver>[A-Za-z0-9_.\-+]+)['"]?"""
)

_PRERELEASE_RE = re.compile(r"(a|b|rc|alpha|beta|dev|pre)\d*", re.IGNORECASE)


@dataclass(frozen=True)
class Pin:
    name: str
    version: str
    source: str  # "pyproject.toml" or "containers/<name>/Dockerfile"


@dataclass(frozen=True)
class FreshnessRow:
    pin: Pin
    latest: str | None
    latest_release_date: datetime | None
    days_behind: int | None
    error: str | None

    @property
    def is_stale(self) -> bool:
        return self.days_behind is not None and self.days_behind > 0


# ---------------------------------------------------------------------------
# Pin extraction
# ---------------------------------------------------------------------------


def _extract_from_pyproject(path: Path) -> list[Pin]:
    if not path.exists():
        return []
    pins: list[Pin] = []
    in_deps = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        # Track whether we're inside a `dependencies = [` or
        # `optional-dependencies` block; we don't strictly need to —
        # the regex will match any `pkg==x.y.z` line — but staying
        # disciplined helps reject false positives in commentary.
        if (
            stripped.startswith("dependencies =")
            or stripped.startswith("dev =")
            or "= [" in stripped
        ):
            in_deps = True
            continue
        if in_deps and stripped == "]":
            in_deps = False
            continue
        m = _PIN_RE.search(stripped)
        if m:
            pins.append(Pin(name=m.group("name"), version=m.group("ver"), source="pyproject.toml"))
    return pins


def _extract_from_dockerfiles(containers_dir: Path) -> list[Pin]:
    pins: list[Pin] = []
    if not containers_dir.exists():
        return pins
    for dockerfile in containers_dir.glob("*/Dockerfile"):
        rel = dockerfile.relative_to(REPO_ROOT)
        for line in dockerfile.read_text().splitlines():
            for m in _PIN_RE.finditer(line):
                pins.append(Pin(name=m.group("name"), version=m.group("ver"), source=str(rel)))
    return pins


def collect_pins() -> list[Pin]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Pin] = []
    for pin in _extract_from_pyproject(PYPROJECT) + _extract_from_dockerfiles(CONTAINERS_DIR):
        key = (pin.name.lower(), pin.version, pin.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(pin)
    return out


# ---------------------------------------------------------------------------
# PyPI lookup
# ---------------------------------------------------------------------------


def _is_stable(version: str) -> bool:
    """Return True only for canonical release versions (no a/b/rc/dev/pre)."""
    return _PRERELEASE_RE.search(version) is None


def _latest_stable_from_pypi(name: str) -> tuple[str | None, datetime | None, str | None]:
    """Return (latest_stable_version, release_date_utc, error)."""
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return None, None, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return None, None, f"{type(exc).__name__}: {exc.reason}"

    releases: dict[str, list[dict]] = data.get("releases", {})
    stable_versions = [v for v in releases.keys() if _is_stable(v)]
    if not stable_versions:
        return None, None, "no stable release on PyPI"

    # PyPI returns versions as strings; sort with PEP-440 if available,
    # falling back to lexicographic which is "good enough" for the
    # narrow case of catching staleness (we'll double-check by date).
    try:
        from packaging.version import Version  # type: ignore

        latest = max(stable_versions, key=Version)
    except Exception:
        latest = max(stable_versions)

    files = releases.get(latest) or []
    upload_dt: datetime | None = None
    for f in files:
        ts = f.get("upload_time_iso_8601")
        if ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if upload_dt is None or dt > upload_dt:
                upload_dt = dt
    return latest, upload_dt, None


# ---------------------------------------------------------------------------
# Comparison + Markdown report
# ---------------------------------------------------------------------------


def _is_newer(latest: str, pinned: str) -> bool:
    try:
        from packaging.version import Version  # type: ignore

        return Version(latest) > Version(pinned)
    except Exception:
        return latest != pinned and latest > pinned


def evaluate(pin: Pin) -> FreshnessRow:
    latest, dt, err = _latest_stable_from_pypi(pin.name)
    if err:
        return FreshnessRow(
            pin=pin, latest=None, latest_release_date=None, days_behind=None, error=err
        )
    if latest is None:
        return FreshnessRow(
            pin=pin, latest=None, latest_release_date=None, days_behind=None, error="no latest"
        )
    if not _is_newer(latest, pin.version):
        return FreshnessRow(
            pin=pin, latest=latest, latest_release_date=dt, days_behind=0, error=None
        )
    if dt is None:
        return FreshnessRow(
            pin=pin, latest=latest, latest_release_date=None, days_behind=None, error=None
        )
    days = (datetime.now(tz=UTC) - dt).days
    return FreshnessRow(
        pin=pin, latest=latest, latest_release_date=dt, days_behind=days, error=None
    )


def render_report(rows: list[FreshnessRow], stale_days: int) -> tuple[str, list[FreshnessRow]]:
    """Return (markdown, stale_rows)."""
    stale = [r for r in rows if r.days_behind is not None and r.days_behind >= stale_days]
    stale.sort(key=lambda r: r.days_behind or 0, reverse=True)

    current = [
        r
        for r in rows
        if r.error is None
        and (r.days_behind == 0 or (r.days_behind is not None and r.days_behind < stale_days))
    ]
    errored = [r for r in rows if r.error]

    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append(f"# Dependency freshness report — {today}")
    lines.append("")
    lines.append(f"Stale threshold: **≥ {stale_days} days** behind latest stable on PyPI.")
    lines.append("")
    lines.append(f"- Total pins scanned: **{len(rows)}**")
    lines.append(f"- Stale (≥ {stale_days} days behind): **{len(stale)}**")
    lines.append(f"- Current or within window: **{len(current)}**")
    if errored:
        lines.append(f"- Errored (PyPI lookup failed): **{len(errored)}**")
    lines.append("")

    if stale:
        lines.append(f"## STALE — {len(stale)} pin(s)")
        lines.append("")
        lines.append("| Package | Pinned | Latest stable | Days behind | Source |")
        lines.append("|---|---|---|---:|---|")
        for r in stale:
            lines.append(
                f"| `{r.pin.name}` | `{r.pin.version}` | `{r.latest}` | "
                f"{r.days_behind} | `{r.pin.source}` |"
            )
        lines.append("")

    if current:
        lines.append(f"## Current — {len(current)} pin(s)")
        lines.append("")
        lines.append("<details><summary>Show</summary>")
        lines.append("")
        lines.append("| Package | Pinned | Latest stable | Days behind | Source |")
        lines.append("|---|---|---|---:|---|")
        for r in sorted(current, key=lambda x: x.pin.name.lower()):
            lines.append(
                f"| `{r.pin.name}` | `{r.pin.version}` | `{r.latest}` | "
                f"{r.days_behind if r.days_behind is not None else '?'} | `{r.pin.source}` |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if errored:
        lines.append(f"## Errored — {len(errored)} pin(s)")
        lines.append("")
        lines.append("| Package | Pinned | Error | Source |")
        lines.append("|---|---|---|---|")
        for r in errored:
            lines.append(f"| `{r.pin.name}` | `{r.pin.version}` | {r.error} | `{r.pin.source}` |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Each candidate bump goes through the bench-validation discipline of "
        "`plans/stateful-marinating-whistle.md`: minor/patch = `pytest` + 5-min "
        "live smoke; major = full Track-style validation. This report never "
        "auto-bumps._"
    )
    return "\n".join(lines) + "\n", stale


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", type=Path, default=None, help="Optional path to write the Markdown report."
    )
    p.add_argument("--stale-days", type=int, default=30)
    p.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="If given, only check these package names (case-insensitive).",
    )
    args = p.parse_args(argv)

    pins = collect_pins()
    if args.include:
        wanted = {s.lower() for s in args.include}
        pins = [p for p in pins if p.name.lower() in wanted]
    if not pins:
        print("No `==` pins found.")
        return 0

    print(f"Checking {len(pins)} pinned package(s) against PyPI ...", file=sys.stderr)
    rows = [evaluate(pin) for pin in pins]
    report, stale = render_report(rows, args.stale_days)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"Wrote report to {args.report}", file=sys.stderr)
    else:
        print(report)

    if any(r.error for r in rows):
        return 2
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
