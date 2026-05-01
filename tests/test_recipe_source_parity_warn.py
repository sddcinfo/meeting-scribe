"""Tests for the boot-time recipe-source-drift warning helper.

Companion to ``tests/test_recipes.py::TestComposeRecipeDriftGuard``:
that test class is the strict CI gate, this module exercises the
runtime helper that emits WARNING log lines without aborting boot.

The runtime helper exists because the CI test only catches drift
between committed sources — it does not catch a host that booted
from a stale image or a hand-edited compose without a CI cycle.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from meeting_scribe.infra import compose as compose_mod


def _load_compose_dict() -> dict:
    return yaml.safe_load(compose_mod.COMPOSE_FILE.read_text())


def _write_compose_dict(tmp_path: Path, data: dict, monkeypatch) -> None:
    """Write ``data`` to a temp compose file and point the helper at it."""
    target = tmp_path / "docker-compose.gb10.yml"
    target.write_text(yaml.safe_dump(data, sort_keys=False))
    monkeypatch.setattr(compose_mod, "COMPOSE_FILE", target)


class TestAssertRecipeSourceParity:
    def test_real_compose_has_no_drift(self):
        """Sanity: the live compose + recipes are aligned today."""
        mismatches = compose_mod.assert_recipe_source_parity()
        assert mismatches == [], f"unexpected drift: {mismatches}"

    def test_detects_asr_gpu_mem_drift(self, tmp_path, monkeypatch):
        """Inject a deliberate ASR --gpu-memory-utilization mismatch."""
        data = _load_compose_dict()
        cmd = data["services"]["vllm-asr"]["command"]
        if isinstance(cmd, list):
            text = " ".join(str(p) for p in cmd)
        else:
            text = str(cmd)
        bad = text.replace("--gpu-memory-utilization 0.10", "--gpu-memory-utilization 0.42")
        assert bad != text, "expected the substitution to land — compose value drifted?"
        data["services"]["vllm-asr"]["command"] = bad
        _write_compose_dict(tmp_path, data, monkeypatch)

        mismatches = compose_mod.assert_recipe_source_parity()
        assert any(
            m.service == "vllm-asr" and m.field == "gpu-memory-utilization" for m in mismatches
        ), f"expected gpu-memory-utilization drift, got {mismatches}"

    def test_detects_diarize_max_speakers_drift(self, tmp_path, monkeypatch):
        data = _load_compose_dict()
        env = data["services"]["pyannote-diarize"]["environment"]
        for i, entry in enumerate(env):
            if isinstance(entry, str) and entry.startswith("DIARIZE_MAX_SPEAKERS="):
                env[i] = "DIARIZE_MAX_SPEAKERS=99"
                break
        else:
            pytest.fail("DIARIZE_MAX_SPEAKERS not found in compose env")
        _write_compose_dict(tmp_path, data, monkeypatch)

        mismatches = compose_mod.assert_recipe_source_parity()
        assert any(
            m.service == "pyannote-diarize" and m.field == "DIARIZE_MAX_SPEAKERS"
            for m in mismatches
        ), f"expected DIARIZE_MAX_SPEAKERS drift, got {mismatches}"

    def test_detects_tts_model_drift(self, tmp_path, monkeypatch):
        data = _load_compose_dict()
        env = data["services"]["qwen3-tts"]["environment"]
        for i, entry in enumerate(env):
            if isinstance(entry, str) and entry.startswith("TTS_MODEL="):
                env[i] = "TTS_MODEL=Qwen/Wrong-Model-Name"
                break
        else:
            pytest.fail("TTS_MODEL not found in compose env")
        _write_compose_dict(tmp_path, data, monkeypatch)

        mismatches = compose_mod.assert_recipe_source_parity()
        assert any(m.service == "qwen3-tts" and m.field == "TTS_MODEL" for m in mismatches), (
            f"expected TTS_MODEL drift, got {mismatches}"
        )


class TestWarnOnRecipeSourceDrift:
    def test_emits_warning_per_mismatch(self, tmp_path, monkeypatch, caplog):
        """The boot-time helper logs one WARNING per source mismatch."""
        data = _load_compose_dict()
        cmd = data["services"]["vllm-asr"]["command"]
        text = " ".join(str(p) for p in cmd) if isinstance(cmd, list) else str(cmd)
        data["services"]["vllm-asr"]["command"] = text.replace(
            "--gpu-memory-utilization 0.10",
            "--gpu-memory-utilization 0.42",
        )
        _write_compose_dict(tmp_path, data, monkeypatch)

        with caplog.at_level(logging.WARNING, logger="meeting_scribe.infra.compose"):
            compose_mod.warn_on_recipe_source_drift()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("recipe drift" in r.message and "vllm-asr" in r.message for r in warnings), (
            f"expected drift WARNING, got: {[r.message for r in warnings]}"
        )

    def test_no_warning_when_aligned(self, caplog):
        """No warnings when compose and recipes are aligned (the live state)."""
        with caplog.at_level(logging.WARNING, logger="meeting_scribe.infra.compose"):
            compose_mod.warn_on_recipe_source_drift()

        drift_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "recipe drift" in r.message
        ]
        assert drift_warnings == [], (
            f"unexpected drift warnings: {[r.message for r in drift_warnings]}"
        )

    def test_helper_swallows_exceptions(self, tmp_path, monkeypatch, caplog):
        """A broken compose file must not abort boot — it logs and returns."""
        target = tmp_path / "docker-compose.gb10.yml"
        target.write_text("not: valid: yaml: [unclosed")
        monkeypatch.setattr(compose_mod, "COMPOSE_FILE", target)

        with caplog.at_level(logging.WARNING, logger="meeting_scribe.infra.compose"):
            compose_mod.warn_on_recipe_source_drift()  # must not raise

        assert any("could not parse" in r.message for r in caplog.records)
