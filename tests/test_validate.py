"""Tests for ``meeting-scribe validate`` orchestrator wiring.

The phase functions themselves hit live backends (HTTP), so the tests
focus on the parts that don't: dataclass behavior, baseline loading
+ env overrides, the report shape, and the entrypoint's error handling
+ phase-selection semantics. Backend probes are stubbed via monkeypatch
on the module-level phase functions rather than mocking httpx.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from meeting_scribe import validate
from meeting_scribe.validate import PhaseResult, ValidateReport

# ──────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────


class TestPhaseResult:
    def test_basic_construction(self):
        p = PhaseResult(name="x", status="pass", elapsed_ms=12.3)
        assert p.name == "x"
        assert p.status == "pass"
        assert p.elapsed_ms == 12.3
        assert p.detail == ""
        assert p.metrics == {}

    def test_metrics_default_is_isolated_per_instance(self):
        # Regression guard: dataclass field(default_factory=dict)
        a = PhaseResult(name="a", status="pass", elapsed_ms=0)
        b = PhaseResult(name="b", status="pass", elapsed_ms=0)
        a.metrics["k"] = 1
        assert b.metrics == {}


class TestValidateReportPassed:
    def test_all_pass(self):
        r = ValidateReport(
            started_at=0,
            finished_at=1,
            mode="quick",
            hardware_class="gb10",
            phases=[
                PhaseResult("a", "pass", 1),
                PhaseResult("b", "pass", 1),
            ],
        )
        assert r.passed is True

    def test_any_fail(self):
        r = ValidateReport(
            started_at=0,
            finished_at=1,
            mode="quick",
            hardware_class="gb10",
            phases=[
                PhaseResult("a", "pass", 1),
                PhaseResult("b", "fail", 1),
            ],
        )
        assert r.passed is False

    def test_skip_does_not_fail(self):
        # `skip` is non-fatal — passes if no `fail`
        r = ValidateReport(
            started_at=0,
            finished_at=1,
            mode="full",
            hardware_class="gb10",
            phases=[
                PhaseResult("a", "pass", 1),
                PhaseResult("b", "skip", 1, "fixture missing"),
            ],
        )
        assert r.passed is True

    def test_no_phases_passes(self):
        r = ValidateReport(
            started_at=0, finished_at=0, mode="quick", hardware_class="gb10", phases=[]
        )
        assert r.passed is True


class TestValidateReportToJson:
    def test_round_trip(self):
        r = ValidateReport(
            started_at=100.5,
            finished_at=200.5,
            mode="full",
            hardware_class="gb10",
            phases=[
                PhaseResult("liveness", "pass", 12.0, "all OK"),
                PhaseResult("asr", "fail", 99.0, "HTTP 500"),
            ],
        )
        out = r.to_json()
        assert out["mode"] == "full"
        assert out["hardware_class"] == "gb10"
        assert out["passed"] is False
        assert len(out["phases"]) == 2
        assert out["phases"][0]["name"] == "liveness"
        assert out["phases"][1]["status"] == "fail"
        # Must be JSON-serialisable
        json.dumps(out)


# ──────────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────────


class TestLoadBaselines:
    def test_returns_empty_when_baseline_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(validate, "_BASELINES_PATH", tmp_path / "missing.json")
        assert validate._load_baselines("gb10") == {}

    def test_reads_hardware_section(self, monkeypatch, tmp_path):
        p = tmp_path / "baselines.json"
        p.write_text(
            json.dumps(
                {
                    "gb10": {"asr": {"p95_ms": 800}},
                    "x86": {"asr": {"p95_ms": 1500}},
                }
            )
        )
        monkeypatch.setattr(validate, "_BASELINES_PATH", p)
        out = validate._load_baselines("gb10")
        assert out == {"asr": {"p95_ms": 800}}

    def test_unknown_hardware_class_returns_empty(self, monkeypatch, tmp_path):
        p = tmp_path / "baselines.json"
        p.write_text(json.dumps({"gb10": {"asr": {"p95_ms": 800}}}))
        monkeypatch.setattr(validate, "_BASELINES_PATH", p)
        assert validate._load_baselines("nonexistent") == {}

    def test_env_override_replaces_baseline_value(self, monkeypatch, tmp_path):
        p = tmp_path / "baselines.json"
        p.write_text(json.dumps({"gb10": {"asr": {"p95_ms": 800}}}))
        monkeypatch.setattr(validate, "_BASELINES_PATH", p)
        monkeypatch.setenv("SCRIBE_VALIDATE_ASR_P95_MS", "1234")
        out = validate._load_baselines("gb10")
        assert out["asr"]["p95_ms"] == 1234.0

    def test_env_override_creates_section(self, monkeypatch, tmp_path):
        # Section absent in file but env-only — still appears in output.
        p = tmp_path / "baselines.json"
        p.write_text(json.dumps({"gb10": {}}))
        monkeypatch.setattr(validate, "_BASELINES_PATH", p)
        monkeypatch.setenv("SCRIBE_VALIDATE_TTS_P95_TTFA_MS", "750")
        out = validate._load_baselines("gb10")
        assert out["tts"]["p95_ttfa_ms"] == 750.0

    def test_unparseable_env_override_is_ignored(self, monkeypatch, tmp_path):
        p = tmp_path / "baselines.json"
        p.write_text(json.dumps({"gb10": {"asr": {"p95_ms": 800}}}))
        monkeypatch.setattr(validate, "_BASELINES_PATH", p)
        monkeypatch.setenv("SCRIBE_VALIDATE_ASR_P95_MS", "not-a-number")
        out = validate._load_baselines("gb10")
        # Falls back to file value
        assert out["asr"]["p95_ms"] == 800


# ──────────────────────────────────────────────────────────────────
# Orchestrator entrypoint — phase selection + error handling
# ──────────────────────────────────────────────────────────────────


def _stub_phase(name: str, status: str = "pass"):
    async def _fn(*args, **kwargs):
        return PhaseResult(name=name, status=status, elapsed_ms=0.1)

    return _fn


class TestRunValidate:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown validate mode"):
            asyncio.run(validate.run_validate(mode="bogus"))

    def test_quick_mode_runs_only_liveness_and_furigana(self, monkeypatch, tmp_path):
        # Stub all phase functions
        monkeypatch.setattr(validate, "_phase_liveness", _stub_phase("liveness"))
        monkeypatch.setattr(validate, "_phase_furigana", _stub_phase("furigana"))
        monkeypatch.setattr(validate, "_phase_asr_latency", _stub_phase("asr_latency"))
        monkeypatch.setattr(validate, "_phase_translate_latency", _stub_phase("translate_latency"))
        monkeypatch.setattr(validate, "_phase_diarize_latency", _stub_phase("diarize_latency"))
        monkeypatch.setattr(validate, "_phase_tts_ttfa", _stub_phase("tts_ttfa"))
        monkeypatch.setattr(validate, "_phase_e2e_lag", _stub_phase("e2e_lag"))
        # Steer the diagnostics-write target at a temp dir
        monkeypatch.setenv("SCRIBE_MEETINGS_DIR", str(tmp_path / "meetings"))

        report = asyncio.run(validate.run_validate(mode="quick", json_only=True))
        names = [p.name for p in report.phases]
        assert names == ["liveness", "furigana"]
        assert report.passed is True
        assert report.mode == "quick"

    def test_full_mode_adds_quality_phases(self, monkeypatch, tmp_path):
        monkeypatch.setattr(validate, "_phase_liveness", _stub_phase("liveness"))
        monkeypatch.setattr(validate, "_phase_furigana", _stub_phase("furigana"))
        monkeypatch.setattr(validate, "_phase_asr_latency", _stub_phase("asr_latency"))
        monkeypatch.setattr(validate, "_phase_translate_latency", _stub_phase("translate_latency"))
        monkeypatch.setattr(validate, "_phase_diarize_latency", _stub_phase("diarize_latency"))
        monkeypatch.setattr(validate, "_phase_tts_ttfa", _stub_phase("tts_ttfa"))
        monkeypatch.setattr(validate, "_phase_e2e_lag", _stub_phase("e2e_lag"))
        monkeypatch.setenv("SCRIBE_MEETINGS_DIR", str(tmp_path / "meetings"))

        report = asyncio.run(validate.run_validate(mode="full", json_only=True))
        names = [p.name for p in report.phases]
        assert names == [
            "liveness",
            "furigana",
            "asr_latency",
            "translate_latency",
            "diarize_latency",
            "tts_ttfa",
        ]

    def test_e2e_mode_appends_e2e_lag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(validate, "_phase_liveness", _stub_phase("liveness"))
        monkeypatch.setattr(validate, "_phase_furigana", _stub_phase("furigana"))
        monkeypatch.setattr(validate, "_phase_asr_latency", _stub_phase("asr_latency"))
        monkeypatch.setattr(validate, "_phase_translate_latency", _stub_phase("translate_latency"))
        monkeypatch.setattr(validate, "_phase_diarize_latency", _stub_phase("diarize_latency"))
        monkeypatch.setattr(validate, "_phase_tts_ttfa", _stub_phase("tts_ttfa"))
        monkeypatch.setattr(validate, "_phase_e2e_lag", _stub_phase("e2e_lag"))
        monkeypatch.setenv("SCRIBE_MEETINGS_DIR", str(tmp_path / "meetings"))

        report = asyncio.run(validate.run_validate(mode="e2e", json_only=True))
        assert report.phases[-1].name == "e2e_lag"

    def test_failed_phase_propagates_to_report(self, monkeypatch, tmp_path):
        monkeypatch.setattr(validate, "_phase_liveness", _stub_phase("liveness", "fail"))
        monkeypatch.setattr(validate, "_phase_furigana", _stub_phase("furigana"))
        monkeypatch.setenv("SCRIBE_MEETINGS_DIR", str(tmp_path / "meetings"))

        report = asyncio.run(validate.run_validate(mode="quick", json_only=True))
        assert report.passed is False

    def test_writes_diagnostics_json(self, monkeypatch, tmp_path):
        monkeypatch.setattr(validate, "_phase_liveness", _stub_phase("liveness"))
        monkeypatch.setattr(validate, "_phase_furigana", _stub_phase("furigana"))
        meetings_dir = tmp_path / "meetings"
        monkeypatch.setenv("SCRIBE_MEETINGS_DIR", str(meetings_dir))

        asyncio.run(validate.run_validate(mode="quick", json_only=True))
        # Report lands at <meetings_dir>/../diagnostics/validate-*.json
        diag_dir = Path(meetings_dir).parent / "diagnostics"
        reports = list(diag_dir.glob("validate-*.json"))
        assert len(reports) == 1
        body = json.loads(reports[0].read_text())
        assert body["mode"] == "quick"
        assert body["passed"] is True


# ──────────────────────────────────────────────────────────────────
# Print helper — graceful on non-tty
# ──────────────────────────────────────────────────────────────────


class TestPrintPhase:
    def test_pass_status_renders(self, capsys, monkeypatch):
        # Force non-tty so we get the plain ASCII path (no ANSI)
        import sys

        monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
        validate._print_phase(PhaseResult("liveness", "pass", 5.0, "all OK"))
        out = capsys.readouterr().out
        assert "PASS" in out
        assert "liveness" in out

    def test_fail_status_renders(self, capsys, monkeypatch):
        import sys

        monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
        validate._print_phase(PhaseResult("asr", "fail", 999.0, "HTTP 500"))
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "HTTP 500" in out
