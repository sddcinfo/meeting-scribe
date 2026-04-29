"""Unit tests for scripts/bench/check_stale_pins.py — pure logic only.

Network-touching code (PyPI lookup) is exercised manually; the unit
tests cover the pin extractor + the staleness comparator + the
report renderer for current/stale/errored rows.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench" / "check_stale_pins.py"

spec = importlib.util.spec_from_file_location("check_stale_pins", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
# Register before exec so the dataclass machinery can resolve cls.__module__
# back to this module while @dataclass decorators run.
sys.modules["check_stale_pins"] = mod
spec.loader.exec_module(mod)


def test_pin_regex_matches_quoted_and_bare_forms() -> None:
    cases = [
        ('"fastapi==0.135.3",', "fastapi", "0.135.3"),
        ("'pyannote.audio==4.0.4'", "pyannote.audio", "4.0.4"),
        ("torch==2.11.0", "torch", "2.11.0"),
        ('"uvicorn[standard]==0.44.0",', "uvicorn", "0.44.0"),
    ]
    for line, expected_name, expected_ver in cases:
        m = mod._PIN_RE.search(line)
        assert m is not None, f"no match in {line!r}"
        assert m.group("name") == expected_name
        assert m.group("ver") == expected_ver


def test_prerelease_versions_are_skipped() -> None:
    assert not mod._is_stable("4.0.0a1")
    assert not mod._is_stable("2.0.0rc3")
    assert not mod._is_stable("0.99.dev1")
    assert mod._is_stable("4.0.4")
    assert mod._is_stable("2.11.0")


def test_extract_pins_from_synthetic_pyproject(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text(
        "[project]\n"
        "dependencies = [\n"
        '    "fastapi==0.135.3",\n'
        '    "torch==2.11.0",\n'
        '    "click>=8.0",\n'  # not a pin — should be skipped
        "]\n"
        "[project.optional-dependencies]\n"
        "dev = [\n"
        '    "pytest==9.0.3",\n'
        "]\n"
    )
    pins = mod._extract_from_pyproject(pp)
    names = {p.name for p in pins}
    assert names == {"fastapi", "torch", "pytest"}
    assert all(p.source == "pyproject.toml" for p in pins)


def test_extract_pins_from_synthetic_dockerfile(tmp_path: Path, monkeypatch) -> None:
    containers = tmp_path / "containers"
    (containers / "pyannote").mkdir(parents=True)
    (containers / "pyannote" / "Dockerfile").write_text(
        "FROM python:3.14\n"
        "RUN pip install --no-cache-dir \\\n"
        "    'pyannote.audio==4.0.4' \\\n"
        "    'fastapi==0.135.3'\n"
    )
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "CONTAINERS_DIR", containers)
    pins = mod._extract_from_dockerfiles(containers)
    names = {p.name for p in pins}
    assert names == {"pyannote.audio", "fastapi"}


def test_render_report_buckets_correctly() -> None:
    Pin = mod.Pin
    Row = mod.FreshnessRow
    now = datetime.now(tz=UTC)
    rows = [
        # current
        Row(
            pin=Pin("fastapi", "0.135.3", "pyproject.toml"),
            latest="0.135.3",
            latest_release_date=now - timedelta(days=10),
            days_behind=0,
            error=None,
        ),
        # stale (90 days behind)
        Row(
            pin=Pin("torch", "2.10.0", "pyproject.toml"),
            latest="2.11.0",
            latest_release_date=now - timedelta(days=90),
            days_behind=90,
            error=None,
        ),
        # errored
        Row(
            pin=Pin("madeup-package", "9.9.9", "pyproject.toml"),
            latest=None,
            latest_release_date=None,
            days_behind=None,
            error="HTTP 404",
        ),
    ]
    report, stale = mod.render_report(rows, stale_days=30)
    assert "STALE — 1 pin" in report
    assert "Current — 1 pin" in report
    assert "Errored — 1 pin" in report
    assert "torch" in report and "2.10.0" in report and "2.11.0" in report
    assert len(stale) == 1
    assert stale[0].pin.name == "torch"


def test_pin_within_window_is_not_stale() -> None:
    Pin = mod.Pin
    Row = mod.FreshnessRow
    now = datetime.now(tz=UTC)
    rows = [
        Row(
            pin=Pin("pkg", "1.0.0", "pyproject.toml"),
            latest="1.0.1",
            latest_release_date=now - timedelta(days=15),
            days_behind=15,
            error=None,
        ),
    ]
    _, stale = mod.render_report(rows, stale_days=30)
    assert stale == []
