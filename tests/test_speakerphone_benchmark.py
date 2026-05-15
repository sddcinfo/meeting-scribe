"""Unit tests for the SP325 benchmark sweep harness.

These tests run without hardware. The orchestration is driven through
stub seams (``_sleep``, ``_apply``, ``_sample``) so the per-cell loop,
sort order, JSON schema, and winner-detection logic are exercised
deterministically.

The real-hardware integration test belongs in a separate suite gated
behind a USB-presence check; including it here would mean the unit
lane couldn't run on a CI agent that lacks the speakerphone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_scribe.speakerphone import benchmark
from meeting_scribe.speakerphone.compliance import ComplianceResult

# ─── Fixtures + stubs ──────────────────────────────────────────────────


def _make_sample(hbp: float, rolloff: float = 5.0, rms: float = -30.0) -> ComplianceResult:
    return ComplianceResult(
        status="pass" if hbp >= 1.5 else "fail",
        high_band_pct=hbp,
        rolloff_3400hz_pct=rolloff,
        rms=rms,
        reason="stub",
        bands_pct={},
    )


@pytest.fixture
def no_sleep():
    """Replace time.sleep with a no-op so sweeps don't actually wait."""
    return lambda _: None


@pytest.fixture
def cell_sequencer():
    """Build a `_sample` stub that returns a scripted high_band_pct per cell.

    Usage:
        seq = cell_sequencer({0: [10, 12, 11, 10, 11], 1: [0.3, 0.4, 0.2, 0.1, 0.0], ...})
        ... pass seq.fn into run_sweep as _sample ...
    """

    class _Seq:
        def __init__(self, plan: dict[int, list[float]]):
            self.plan = plan
            self.cell_idx = -1
            self.call_n = 0
            self._pre = True  # first call per cell is the pre-trial baseline

        def fn(self, node, *, capture_seconds, min_high_band_pct, min_rolloff_pct):
            if self._pre:
                self.cell_idx += 1
                self._pre = False
                # Baseline reads use a fixed "drifted" value 0.4 unless explicitly scripted.
                pre = self.plan.get(self.cell_idx, [None])[0] if "pre" not in self.plan else None
                return _make_sample(self.plan.get(self.cell_idx, [0.4])[0] if False else 0.4)
            # Subsequent reads pull from the scripted list, looping past pre.
            vals = self.plan.get(self.cell_idx, [0.0])
            idx = self.call_n
            v = vals[idx] if idx < len(vals) else vals[-1]
            self.call_n += 1
            if self.call_n >= 5:  # default samples_per_cell
                self.call_n = 0
                self._pre = True
            return _make_sample(v)

    return _Seq


# ─── Schema + winner detection ─────────────────────────────────────────


def test_cell_result_to_dict_round_trips() -> None:
    r = benchmark.CellResult(
        cell=benchmark.Cell(0xD0, 0x02, 0x00, "wideband_enable_primary"),
        pre_high_band_pct=0.4,
        pre_rolloff_3400_pct=2.0,
        samples_high_band_pct=[41.8, 39.2, 40.0, 38.5, 42.0],
        samples_rolloff_3400_pct=[15.0, 14.8, 15.2, 14.5, 15.1],
        samples_rms=[-30.0, -31.0, -30.5, -30.8, -30.2],
        median_high_band_pct=40.0,
        max_high_band_pct=42.0,
        median_rolloff_3400_pct=15.0,
        num_pass=5,
    )
    d = r.to_dict()
    assert d["cell"]["label"] == "wideband_enable_primary"
    assert d["median_high_band_pct"] == 40.0
    assert d["is_winner"] is True
    assert d["num_pass"] == 5
    assert d["error"] is None
    # Round-trip through JSON.
    assert json.loads(json.dumps(d))["is_winner"] is True


def test_is_winner_requires_no_error() -> None:
    r = benchmark.CellResult(
        cell=benchmark.Cell(0xD0, 0x02, 0x00, "wideband_enable_primary"),
        pre_high_band_pct=0.4,
        pre_rolloff_3400_pct=2.0,
        samples_high_band_pct=[41.0] * 5,
        samples_rolloff_3400_pct=[15.0] * 5,
        samples_rms=[-30.0] * 5,
        median_high_band_pct=41.0,
        max_high_band_pct=41.0,
        median_rolloff_3400_pct=15.0,
        num_pass=5,
        error="USB error",
    )
    assert r.is_winner is False


def test_is_winner_needs_at_least_four_passes() -> None:
    base_kwargs = dict(
        cell=benchmark.Cell(0xD0, 0x02, 0x00, "w"),
        pre_high_band_pct=0.4,
        pre_rolloff_3400_pct=2.0,
        samples_high_band_pct=[41.0] * 5,
        samples_rolloff_3400_pct=[15.0] * 5,
        samples_rms=[-30.0] * 5,
        median_high_band_pct=41.0,
        max_high_band_pct=41.0,
        median_rolloff_3400_pct=15.0,
    )
    assert benchmark.CellResult(**base_kwargs, num_pass=4).is_winner
    assert benchmark.CellResult(**base_kwargs, num_pass=3).is_winner is False


# ─── Run-sweep orchestration ───────────────────────────────────────────


def test_run_sweep_orchestrates_per_cell_phases(no_sleep) -> None:
    cells = (
        benchmark.Cell(0xD0, 0x02, 0x00, "winner_likely"),
        benchmark.Cell(0xD2, 0x01, 0x00, "mic_ns_set_off_dpm"),
    )

    apply_calls: list[benchmark.Cell] = []

    def _apply(cell: benchmark.Cell) -> None:
        apply_calls.append(cell)

    # First cell: 5 samples all PASS (winner).
    # Second cell: 5 samples all FAIL (regression).
    sample_script = iter(
        [41.0, 40.0, 39.5, 41.5, 40.0]  # winner samples
        + [0.4, 0.3, 0.2, 0.5, 0.6]  # regression samples
    )

    def _sample(node, *, capture_seconds, min_high_band_pct, min_rolloff_pct):
        # First call per cell is the pre baseline — give a quiescent value.
        if not hasattr(_sample, "_state"):
            _sample._state = {"pre_pending": True, "i": 0}  # type: ignore[attr-defined]
        st = _sample._state  # type: ignore[attr-defined]
        if st["pre_pending"]:
            st["pre_pending"] = False
            return _make_sample(0.4)
        v = next(sample_script)
        st["i"] += 1
        if st["i"] >= 5:
            st["i"] = 0
            st["pre_pending"] = True
        return _make_sample(v)

    report = benchmark.run_sweep(
        "test-node",
        cells=cells,
        baseline_seconds=1.0,
        settle_seconds=1.0,
        samples_per_cell=5,
        sample_gap_seconds=0.1,
        capture_seconds=1.0,
        _sleep=no_sleep,
        _apply=_apply,
        _sample=_sample,
    )

    assert len(report.cells) == 2
    assert apply_calls == list(cells)

    winner = next(r for r in report.cells if r.cell.label == "winner_likely")
    assert winner.num_pass == 5
    assert winner.is_winner is True
    assert winner.median_high_band_pct == 40.0

    regression = next(r for r in report.cells if r.cell.label == "mic_ns_set_off_dpm")
    assert regression.num_pass == 0
    assert regression.is_winner is False
    assert regression.median_high_band_pct < 1.5


def test_run_sweep_records_apply_errors(no_sleep) -> None:
    cells = (benchmark.Cell(0xD0, 0x02, 0x00, "winner_likely"),)

    def _apply(cell):
        raise RuntimeError("no SP325 found")

    def _sample(node, **kw):
        return _make_sample(0.4)

    report = benchmark.run_sweep(
        "test-node",
        cells=cells,
        baseline_seconds=0.1,
        settle_seconds=0.1,
        samples_per_cell=2,
        sample_gap_seconds=0.0,
        capture_seconds=0.5,
        _sleep=no_sleep,
        _apply=_apply,
        _sample=_sample,
    )
    [r] = report.cells
    assert r.error is not None
    assert "no SP325 found" in r.error
    assert r.is_winner is False


def test_sweep_report_sorts_cells_by_median_descending() -> None:
    report = benchmark.SweepReport(
        started_at="t0",
        ended_at="t1",
        device_key="413c:8223",
        pipewire_node="x",
        firmware_observed=None,
        baseline_seconds=0,
        settle_seconds=0,
        samples_per_cell=1,
        sample_gap_seconds=0,
        capture_seconds=0,
        min_high_band_pct=1.5,
        min_rolloff_3400_pct=2.0,
        cells=[
            benchmark.CellResult(
                cell=benchmark.Cell(0xD2, 0x01, 0x00, "regression"),
                pre_high_band_pct=0.4,
                pre_rolloff_3400_pct=2.0,
                samples_high_band_pct=[0.4],
                samples_rolloff_3400_pct=[2.0],
                samples_rms=[-30.0],
                median_high_band_pct=0.4,
                max_high_band_pct=0.4,
                median_rolloff_3400_pct=2.0,
                num_pass=0,
            ),
            benchmark.CellResult(
                cell=benchmark.Cell(0xD0, 0x02, 0x00, "winner"),
                pre_high_band_pct=0.4,
                pre_rolloff_3400_pct=2.0,
                samples_high_band_pct=[41.0],
                samples_rolloff_3400_pct=[15.0],
                samples_rms=[-30.0],
                median_high_band_pct=41.0,
                max_high_band_pct=41.0,
                median_rolloff_3400_pct=15.0,
                num_pass=1,
            ),
        ],
    )
    d = report.to_dict()
    assert d["cells"][0]["cell"]["label"] == "winner"
    assert d["cells"][1]["cell"]["label"] == "regression"
    assert d["winners"] == []  # num_pass=1 isn't ≥4


def test_write_report_creates_json_and_markdown(tmp_path: Path) -> None:
    report = benchmark.SweepReport(
        started_at="t0",
        ended_at="t1",
        device_key="413c:8223",
        pipewire_node="x",
        firmware_observed=None,
        baseline_seconds=1.0,
        settle_seconds=1.0,
        samples_per_cell=5,
        sample_gap_seconds=1.0,
        capture_seconds=3.0,
        min_high_band_pct=1.5,
        min_rolloff_3400_pct=2.0,
        cells=[
            benchmark.CellResult(
                cell=benchmark.Cell(0xD0, 0x02, 0x00, "wideband_enable_primary"),
                pre_high_band_pct=0.4,
                pre_rolloff_3400_pct=2.0,
                samples_high_band_pct=[41.0] * 5,
                samples_rolloff_3400_pct=[15.0] * 5,
                samples_rms=[-30.0] * 5,
                median_high_band_pct=41.0,
                max_high_band_pct=41.0,
                median_rolloff_3400_pct=15.0,
                num_pass=5,
            ),
        ],
    )
    out = tmp_path / "sp325-benchmark-test.json"
    benchmark.write_report(report, out)
    assert out.exists()
    md = out.with_suffix(".md")
    assert md.exists()
    j = json.loads(out.read_text())
    assert j["winners"] == ["wideband_enable_primary"]
    md_text = md.read_text()
    assert "wideband_enable_primary" in md_text
    assert "★" in md_text  # winner star


def test_default_cells_cover_known_winners_and_regressions() -> None:
    """The DEFAULT_CELLS matrix must include the empirically-proven set
    so a future firmware that changes the device's opcode mapping is
    detected by the sweep."""
    labels = {c.label for c in benchmark.DEFAULT_CELLS}
    # The three winners that produced the 2026-05-13 wideband-good config.
    assert "wideband_enable_primary" in labels
    assert "wideband_enable_secondary" in labels
    assert "eq_preset_default" in labels
    # The DPM-derived "set NS off" pair that REGRESSES SP325 — sweep
    # must include them so the gate trips if firmware reverses this.
    assert "mic_ns_set_off_dpm" in labels
    assert "mic_ns_incoming_set_off_dpm" in labels


def test_default_report_path_is_under_speakerphone_quality(tmp_path: Path) -> None:
    p = benchmark.default_report_path(tmp_path)
    assert p.parent == tmp_path / "speakerphone-quality"
    assert p.name.startswith("sp325-benchmark-")
    assert p.suffix == ".json"


def test_apply_cell_imports_sp325hid_lazily() -> None:
    """Module import must NOT require pyusb (tests run on hosts without
    python3-usb)."""
    import importlib

    # benchmark.py imports compliance + stdlib only at top level.
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "meeting_scribe"
        / "speakerphone"
        / "benchmark.py"
    ).read_text()
    head = src.split("def ", 1)[0]
    assert "import usb" not in head
    assert "sp325_hid" not in head
    # The lazy import lives inside _apply_cell.
    importlib.reload(benchmark)  # round-trip is a no-op import sanity check
