"""Tests for the recipe YAML loading system."""

from __future__ import annotations

import pytest

from meeting_scribe.recipes import all_model_ids, list_recipes, load_recipe


class TestRecipeLoader:
    """Recipe discovery and loading."""

    def test_list_recipes_returns_all(self):
        recipes = list_recipes()
        assert len(recipes) >= 3
        assert "qwen3.6-35b-translation" in recipes
        assert "sortformer-4spk" in recipes
        assert "qwen3-asr-vllm" in recipes

    def test_load_by_filename(self):
        recipe = load_recipe("qwen3.6-35b-translation")
        assert recipe["model_key"] == "translation"
        assert recipe["model_id"] == "Qwen/Qwen3.6-35B-A3B-FP8"
        assert recipe["port"] == 8010

    def test_load_by_model_key(self):
        recipe = load_recipe("translation")
        assert recipe["model_id"] == "Qwen/Qwen3.6-35B-A3B-FP8"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_recipe("nonexistent-model")


class TestRecipeContent:
    """Recipe content validation."""

    def test_translation_recipe(self):
        r = load_recipe("translation")
        assert r["mode"] == "shared"
        assert r["tensor_parallel"] == 1
        assert r["port"] == 8010
        # Production-locked on Qwen3.6-35B-A3B-FP8 (native Qwen FP8)
        # since 2026-04-25; the prior 3.5-INT4 / Intel AutoRound stack
        # used quantization=inc and gpu_memory_utilization=0.30.
        assert r["model_id"] == "Qwen/Qwen3.6-35B-A3B-FP8"
        assert r["quantization"] == "fp8"

    def test_diarization_recipe(self):
        r = load_recipe("diarization")
        assert r["port"] == 8001
        assert r["container_type"] == "pyannote"

    def test_asr_vllm_recipe(self):
        r = load_recipe("asr-vllm")
        assert r["port"] == 8003
        assert "ASR" in r["model_id"] or "asr" in r["model_id"].lower()


class TestAllModelIds:
    """Model ID collection for pull-models."""

    def test_all_model_ids_returns_list(self):
        ids = all_model_ids()
        assert isinstance(ids, list)
        assert len(ids) >= 3

    def test_all_model_ids_excludes_shared(self):
        """The translation recipe is mode=shared (autosre owns it).
        all_model_ids() must filter it out so customer pull-models
        doesn't waste ~35 GB downloading weights it'll never serve."""
        ids = all_model_ids()
        assert "Qwen/Qwen3.6-35B-A3B-FP8" not in ids
        # ASR + TTS + diarization are scribe's responsibility and stay in.
        assert any("ASR" in mid for mid in ids)
        assert any("TTS" in mid for mid in ids)
        assert any("speaker-diarization" in mid for mid in ids)

    def test_no_port_collision(self):
        """Verify all recipes use distinct ports."""
        recipes = list_recipes()
        ports = {}
        for name in recipes:
            r = load_recipe(name)
            port = r.get("port")
            if port is not None:
                assert port not in ports, f"Port {port} collision: {name} vs {ports[port]}"
                ports[port] = name
