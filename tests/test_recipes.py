"""Tests for the recipe YAML loading system."""

from __future__ import annotations

import pytest

from meeting_scribe.recipes import all_model_ids, list_recipes, load_recipe


class TestRecipeLoader:
    """Recipe discovery and loading."""

    def test_list_recipes_returns_all(self):
        recipes = list_recipes()
        assert len(recipes) >= 4
        assert "qwen3.5-35b-translation" in recipes
        assert "sortformer-4spk" in recipes
        assert "qwen3-asr-vllm" in recipes
        assert "qwen3-tts-vllm" in recipes

    def test_load_by_filename(self):
        recipe = load_recipe("qwen3.5-35b-translation")
        assert recipe["model_key"] == "translation"
        assert recipe["model_id"] == "Intel/Qwen3.5-35B-A3B-int4-AutoRound"
        assert recipe["port"] == 8010

    def test_load_by_model_key(self):
        recipe = load_recipe("translation")
        assert recipe["model_id"] == "Intel/Qwen3.5-35B-A3B-int4-AutoRound"

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
        assert r["gpu_memory_utilization"] == 0.30
        assert r["quantization"] == "inc"

    def test_tts_recipe(self):
        r = load_recipe("tts-vllm")
        # Parallel-run port while legacy scribe-tts/scribe-tts-2 own 8002/8012.
        assert r["port"] == 8022
        assert r["container_type"] == "vllm"
        assert "Qwen3-TTS" in r["model_id"]

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

    def test_all_model_ids_includes_translation(self):
        ids = all_model_ids()
        assert any("Qwen3.5" in mid for mid in ids)

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
