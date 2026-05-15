"""Unit tests for scripts/bench/check_model_drift.py — pure logic only.

Network-touching code (HF Hub queries) is exercised by the CI workflow;
these unit tests cover the watchlist parser, the drift / open-weights
classifier, and the report renderer with mocked rows.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench" / "check_model_drift.py"

spec = importlib.util.spec_from_file_location("check_model_drift", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["check_model_drift"] = mod
spec.loader.exec_module(mod)


def _write_watchlist(tmp: Path, models: list[dict], closed: list[dict]) -> Path:
    p = tmp / "watchlist.yaml"
    p.write_text(yaml.safe_dump({"models": models, "closed_source_watchlist": closed}))
    return p


def test_watchlist_parser_normalises_role(tmp_path: Path) -> None:
    p = _write_watchlist(
        tmp_path,
        [{"id": "x/y", "family": "x/*"}],
        [{"id": "z/w", "note": "watch"}],
    )
    models, closed = mod._load_watchlist(p)
    assert len(models) == 1
    assert models[0].role == "unspecified"
    assert closed[0].note == "watch"


def test_watchlist_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod._load_watchlist(tmp_path / "absent.yaml")


def test_render_report_drift_actionable() -> None:
    now = datetime.now(tz=UTC)
    drift_rows = [
        mod._DriftRow(
            pinned_id="org/foo-3.1",
            role="prod_diarize",
            latest_sibling_id="org/foo-community-1",
            latest_sibling_modified=now - timedelta(days=10),
            days_newer=45,
            error=None,
        ),
        mod._DriftRow(
            pinned_id="org/bar",
            role="prod_asr",
            latest_sibling_id=None,
            latest_sibling_modified=None,
            days_newer=0,
            error=None,
        ),
    ]
    report, action = mod.render_report(drift_rows, [], stale_days=30)
    assert action is True
    assert "Pinned model drift — 1 stale" in report
    assert "org/foo-community-1" in report
    assert "Pinned models — 1 current" in report


def test_render_report_open_weights_actionable() -> None:
    open_rows = [
        mod._OpenWeightsRow(
            repo_id="FunAudioLLM/CosyVoice",
            note="Watch for 3.5 open release",
            has_weights=True,
            weights_count=4,
            error=None,
        ),
        mod._OpenWeightsRow(
            repo_id="bosonai/higgs-audio",
            note="Watch for V3",
            has_weights=False,
            weights_count=0,
            error=None,
        ),
    ]
    report, action = mod.render_report([], open_rows, stale_days=30)
    assert action is True
    assert "just opened — 1" in report
    assert "FunAudioLLM/CosyVoice" in report
    assert "still closed — 1" in report


def test_render_report_no_action_when_clean() -> None:
    drift_rows = [
        mod._DriftRow(
            pinned_id="org/foo",
            role="prod",
            latest_sibling_id=None,
            latest_sibling_modified=None,
            days_newer=0,
            error=None,
        ),
    ]
    open_rows = [
        mod._OpenWeightsRow(
            repo_id="org/closed",
            note="watching",
            has_weights=False,
            weights_count=0,
            error=None,
        ),
    ]
    _, action = mod.render_report(drift_rows, open_rows, stale_days=30)
    assert action is False


def test_render_report_errored_rows_listed() -> None:
    drift_rows = [
        mod._DriftRow(
            pinned_id="org/missing",
            role="prod",
            latest_sibling_id=None,
            latest_sibling_modified=None,
            days_newer=None,
            error="HTTP 404",
        ),
    ]
    open_rows = [
        mod._OpenWeightsRow(
            repo_id="org/network-fail",
            note="x",
            has_weights=False,
            weights_count=0,
            error="ConnectionError: refused",
        ),
    ]
    report, _ = mod.render_report(drift_rows, open_rows, stale_days=30)
    assert "Errored — 2 repo(s)" in report
    assert "HTTP 404" in report
    assert "ConnectionError" in report


def test_weight_globs_cover_safetensors_and_bin() -> None:
    import fnmatch

    assert any(fnmatch.fnmatch("model.safetensors", g) for g in mod.WEIGHT_GLOBS)
    assert any(fnmatch.fnmatch("pytorch_model.bin", g) for g in mod.WEIGHT_GLOBS)
    assert any(fnmatch.fnmatch("Q4_K_M.gguf", g) for g in mod.WEIGHT_GLOBS)
    assert not any(fnmatch.fnmatch("README.md", g) for g in mod.WEIGHT_GLOBS)
    assert not any(fnmatch.fnmatch("config.json", g) for g in mod.WEIGHT_GLOBS)
