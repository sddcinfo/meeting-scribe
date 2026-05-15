"""SP325 settings benchmark sweep.

Per-command sweep harness for SP325 vendor settings. Found the
wideband-good winners (D0/0x02, D1/0x01, C0/0x04). Driven by
``meeting-scribe speakerphone benchmark``.

For each (cmd_class, opcode, value) cell:

  1. Wait ``baseline_seconds`` so the device drifts back to its
     natural state. Sample compliance ONCE as the pre-trial baseline.
  2. Open the Sp325HidClient (detaches kernel drivers from all four
     USB interfaces — required for the firmware to enter "configurable"
     mode and persist the setting; see ``Sp325HidClient.open``).
  3. Send the single SET_REPORT carrying the cell's payload.
  4. Close the client (reattaches drivers; PipeWire reclaims the audio
     interfaces).
  5. Wait ``settle_seconds`` for the DSP buffer to flush.
  6. Capture ``samples_per_cell`` compliance probes spaced by
     ``sample_gap_seconds``. Record the high_band_pct + rolloff_3400_pct
     of each, plus the median, max, and pass count (samples meeting
     ``min_high_band_pct``).

Result table is sorted by median high_band_pct, descending. The
WINNER for each command class shows ≥4/5 PASS with stable median —
not a single spike.

The output JSON is written to ``reports/speakerphone-quality/`` so
the matrix sits next to the existing capture artifacts. Schema is
stable; re-running on a new firmware revision produces a new file
with the same shape, making trend comparison straightforward.

The harness deliberately does NOT shell out to the CLI for each
measurement (the scratch bisect did). Direct ``probe_device`` calls
cut the per-cell overhead from ~5 s (subprocess + auth roundtrip)
to ~3 s (capture + analyze).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compliance import (
    ComplianceResult,
    expected_thresholds,
    probe_device,
)

logger = logging.getLogger(__name__)


# ─── Trial definitions ─────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Cell:
    """One sweep cell — a single command to apply + measure."""

    cmd_class: int
    opcode: int
    value: int = 0x00
    label: str = ""

    @property
    def key(self) -> str:
        return f"cls=0x{self.cmd_class:02x} op=0x{self.opcode:02x} val=0x{self.value:02x}"


# Default sweep covers:
#  • the three KNOWN winners from the 2026-05-13 bisect (must continue
#    to PASS or the device firmware has changed underfoot),
#  • the DPM-derived "set mic NS off" pair on 0xD2/0xD5 (must FAIL on
#    SP325 — they regress to narrowband; the harness encodes the
#    negative case so a future firmware that reverses this is caught),
#  • a representative slice of the unmapped 0xD0–0xD4 opcodes for
#    coverage drift,
#  • the two EQ-class variants we have a wire format for.
DEFAULT_CELLS: tuple[Cell, ...] = (
    Cell(0xD0, 0x02, 0x00, "wideband_enable_primary"),
    Cell(0xD1, 0x01, 0x00, "wideband_enable_secondary"),
    Cell(0xC0, 0x04, 0x00, "eq_preset_default"),
    Cell(0xD2, 0x01, 0x00, "mic_ns_set_off_dpm"),  # KILLS wideband — verify it still regresses.
    Cell(0xD5, 0x01, 0x00, "mic_ns_incoming_set_off_dpm"),
    Cell(0xD0, 0x01, 0x00, "d0_op1_unknown"),
    Cell(0xD1, 0x02, 0x00, "d1_op2_unknown"),
    Cell(0xD3, 0x01, 0x00, "vol_tone_set"),
    Cell(0xD4, 0x02, 0x00, "vol_limit_enabled"),
    Cell(0xC0, 0x01, 0x00, "eq_op1_unknown"),
)


# ─── Result schema ─────────────────────────────────────────────────────


@dataclasses.dataclass
class CellResult:
    cell: Cell
    pre_high_band_pct: float
    pre_rolloff_3400_pct: float
    samples_high_band_pct: list[float]
    samples_rolloff_3400_pct: list[float]
    samples_rms: list[float]
    median_high_band_pct: float
    max_high_band_pct: float
    median_rolloff_3400_pct: float
    num_pass: int
    error: str | None = None

    @property
    def is_winner(self) -> bool:
        return self.num_pass >= 4 and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": dataclasses.asdict(self.cell),
            "pre_high_band_pct": round(self.pre_high_band_pct, 2),
            "pre_rolloff_3400_pct": round(self.pre_rolloff_3400_pct, 2),
            "samples_high_band_pct": [round(v, 2) for v in self.samples_high_band_pct],
            "samples_rolloff_3400_pct": [round(v, 2) for v in self.samples_rolloff_3400_pct],
            "samples_rms": [round(v, 1) for v in self.samples_rms],
            "median_high_band_pct": round(self.median_high_band_pct, 2),
            "max_high_band_pct": round(self.max_high_band_pct, 2),
            "median_rolloff_3400_pct": round(self.median_rolloff_3400_pct, 2),
            "num_pass": self.num_pass,
            "is_winner": self.is_winner,
            "error": self.error,
        }


@dataclasses.dataclass
class SweepReport:
    """Top-level benchmark output. One JSON file = one SweepReport."""

    started_at: str
    ended_at: str
    device_key: str
    pipewire_node: str
    firmware_observed: str | None
    baseline_seconds: float
    settle_seconds: float
    samples_per_cell: int
    sample_gap_seconds: float
    capture_seconds: float
    min_high_band_pct: float
    min_rolloff_3400_pct: float
    cells: list[CellResult]

    def to_dict(self) -> dict[str, Any]:
        cells_sorted = sorted(self.cells, key=lambda r: r.median_high_band_pct, reverse=True)
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "device_key": self.device_key,
            "pipewire_node": self.pipewire_node,
            "firmware_observed": self.firmware_observed,
            "parameters": {
                "baseline_seconds": self.baseline_seconds,
                "settle_seconds": self.settle_seconds,
                "samples_per_cell": self.samples_per_cell,
                "sample_gap_seconds": self.sample_gap_seconds,
                "capture_seconds": self.capture_seconds,
                "min_high_band_pct": self.min_high_band_pct,
                "min_rolloff_3400_pct": self.min_rolloff_3400_pct,
            },
            "cells": [r.to_dict() for r in cells_sorted],
            "winners": [r.cell.label or r.cell.key for r in cells_sorted if r.is_winner],
        }


# ─── Sweep runner ──────────────────────────────────────────────────────


def _sample_compliance(
    pipewire_node: str,
    *,
    capture_seconds: float,
    min_high_band_pct: float,
    min_rolloff_pct: float,
) -> ComplianceResult:
    """Single compliance probe. Wraps ``probe_device`` so callers can
    monkeypatch ONE function in tests without standing up a real WAV
    pipeline."""
    return probe_device(
        pipewire_node,
        capture_seconds=capture_seconds,
        min_high_band_pct=min_high_band_pct,
        min_rolloff_pct=min_rolloff_pct,
    )


def _apply_cell(cell: Cell) -> None:
    """Open Sp325HidClient, send one SET_REPORT, close.

    Errors propagate as ``Sp325Error`` so the trial can record the
    failure and move on instead of aborting the whole sweep.
    """
    # Imported lazily so module import doesn't require pyusb when the
    # test suite exercises the schema + sort logic without hardware.
    from .sp325_hid import Sp325HidClient

    with Sp325HidClient.open_default() as cli:
        cli._set_report(bytes([cell.cmd_class, cell.opcode, cell.value, 0x00]))


def run_sweep(
    pipewire_node: str,
    *,
    cells: Sequence[Cell] = DEFAULT_CELLS,
    device_key: str = "413c:8223",
    baseline_seconds: float = 30.0,
    settle_seconds: float = 15.0,
    samples_per_cell: int = 5,
    sample_gap_seconds: float = 1.0,
    capture_seconds: float = 3.0,
    min_high_band_pct: float | None = None,
    min_rolloff_pct: float | None = None,
    progress: Any = None,
    _sleep=time.sleep,
    _apply=_apply_cell,
    _sample=_sample_compliance,
) -> SweepReport:
    """Run the full settings sweep. Returns a SweepReport.

    ``_sleep``, ``_apply``, ``_sample`` are seams for testing; the
    default behaviour calls real hardware. A test passes stub fns to
    exercise the orchestration without USB / mic capture.

    ``progress`` is an optional callable ``(i, n, cell, phase)`` —
    receives lifecycle ticks for CLI rendering.
    """
    if min_high_band_pct is None or min_rolloff_pct is None:
        thresholds = expected_thresholds(device_key)
        min_high_band_pct = min_high_band_pct if min_high_band_pct is not None else thresholds[0]
        min_rolloff_pct = min_rolloff_pct if min_rolloff_pct is not None else thresholds[1]

    started = datetime.now(UTC).isoformat(timespec="seconds")
    results: list[CellResult] = []

    n = len(cells)
    for i, cell in enumerate(cells, 1):
        if progress:
            progress(i, n, cell, "baseline")
        logger.info("sweep %d/%d %s — baseline settle %.1fs", i, n, cell.key, baseline_seconds)
        _sleep(baseline_seconds)

        pre = _sample(
            pipewire_node,
            capture_seconds=capture_seconds,
            min_high_band_pct=min_high_band_pct,
            min_rolloff_pct=min_rolloff_pct,
        )

        if progress:
            progress(i, n, cell, "apply")

        err: str | None = None
        try:
            _apply(cell)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.warning("sweep %s apply failed: %s", cell.key, err)

        if progress:
            progress(i, n, cell, "settle")
        _sleep(settle_seconds)

        samples_hbp: list[float] = []
        samples_roll: list[float] = []
        samples_rms: list[float] = []
        num_pass = 0
        for s in range(samples_per_cell):
            if progress:
                progress(i, n, cell, f"sample_{s + 1}")
            res = _sample(
                pipewire_node,
                capture_seconds=capture_seconds,
                min_high_band_pct=min_high_band_pct,
                min_rolloff_pct=min_rolloff_pct,
            )
            samples_hbp.append(res.high_band_pct)
            samples_roll.append(res.rolloff_3400hz_pct)
            samples_rms.append(res.rms)
            if res.high_band_pct >= min_high_band_pct:
                num_pass += 1
            if s < samples_per_cell - 1:
                _sleep(sample_gap_seconds)

        results.append(
            CellResult(
                cell=cell,
                pre_high_band_pct=pre.high_band_pct,
                pre_rolloff_3400_pct=pre.rolloff_3400hz_pct,
                samples_high_band_pct=samples_hbp,
                samples_rolloff_3400_pct=samples_roll,
                samples_rms=samples_rms,
                median_high_band_pct=statistics.median(samples_hbp) if samples_hbp else 0.0,
                max_high_band_pct=max(samples_hbp) if samples_hbp else 0.0,
                median_rolloff_3400_pct=(statistics.median(samples_roll) if samples_roll else 0.0),
                num_pass=num_pass,
                error=err,
            )
        )

    ended = datetime.now(UTC).isoformat(timespec="seconds")
    return SweepReport(
        started_at=started,
        ended_at=ended,
        device_key=device_key,
        pipewire_node=pipewire_node,
        firmware_observed=None,  # USB doesn't expose FW — behavioral check only.
        baseline_seconds=baseline_seconds,
        settle_seconds=settle_seconds,
        samples_per_cell=samples_per_cell,
        sample_gap_seconds=sample_gap_seconds,
        capture_seconds=capture_seconds,
        min_high_band_pct=min_high_band_pct,
        min_rolloff_3400_pct=min_rolloff_pct,
        cells=results,
    )


# ─── Output rendering ──────────────────────────────────────────────────


def render_markdown_table(report: SweepReport) -> str:
    """Pretty-print the sweep result as a Markdown table sorted by median."""
    lines = [
        "# SP325 settings sweep",
        "",
        f"started_at: {report.started_at}",
        f"ended_at:   {report.ended_at}",
        f"device:     {report.device_key}  source: {report.pipewire_node}",
        f"thresholds: high_band_pct ≥ {report.min_high_band_pct}%  rolloff ≥ {report.min_rolloff_3400_pct}%",
        "",
        "| label                            | class | op  | pre%  | samples high_band       | median | max  | pass/N | winner |",
        "|----------------------------------|-------|-----|-------|-------------------------|--------|------|--------|--------|",
    ]
    cells_sorted = sorted(report.cells, key=lambda r: r.median_high_band_pct, reverse=True)
    for r in cells_sorted:
        label = (r.cell.label or "—")[:32]
        samp = " ".join(f"{v:>5.2f}" for v in r.samples_high_band_pct)
        winner = "★" if r.is_winner else ""
        if r.error:
            line = (
                f"| {label:<32} | 0x{r.cell.cmd_class:02x}  | 0x{r.cell.opcode:02x} | "
                f"  —   | ERROR: {r.error[:50]} |        |      |        |        |"
            )
        else:
            line = (
                f"| {label:<32} | 0x{r.cell.cmd_class:02x}  | 0x{r.cell.opcode:02x} | "
                f"{r.pre_high_band_pct:>5.2f} | {samp:<23} | "
                f"{r.median_high_band_pct:>6.2f} | {r.max_high_band_pct:>4.2f} | "
                f"{r.num_pass}/{report.samples_per_cell:<6}| {winner:^6} |"
            )
        lines.append(line)
    return "\n".join(lines)


def default_report_path(reports_root: Path) -> Path:
    """Return the conventional ``reports/speakerphone-quality/sp325-benchmark-<ts>.json`` path."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return reports_root / "speakerphone-quality" / f"sp325-benchmark-{ts}.json"


def write_report(report: SweepReport, out_path: Path) -> Path:
    """Persist the sweep as JSON + a sibling .md summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown_table(report) + "\n")
    return out_path
