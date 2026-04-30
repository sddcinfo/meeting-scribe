"""Tests for the recipe YAML loading system."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

from meeting_scribe.recipes import all_model_ids, list_recipes, load_recipe

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = REPO_ROOT / "docker-compose.gb10.yml"


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


def _extract_vllm_flags(compose_command: list[str] | str) -> dict[str, str]:
    """Parse the `vllm serve ...` cmd from a docker-compose service.command
    block into a {flag_name: value} dict for assertion."""
    if isinstance(compose_command, list):
        text = " ".join(str(part) for part in compose_command)
    else:
        text = str(compose_command)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = shlex.split(text)
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            name = tok.lstrip("-")
            if "=" in name:
                key, _, val = name.partition("=")
                flags[key] = val
                i += 1
                continue
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[name] = tokens[i + 1]
                i += 2
                continue
            flags[name] = ""
        i += 1
    return flags


def _extract_compose_env(env_block: list[str] | None) -> dict[str, str]:
    """Parse a docker-compose service.environment list of ``KEY=VALUE``
    strings into a ``{key: value}`` dict. Resolves shell-expansion
    defaults — ``${VAR:-default}`` becomes ``default``; ``${VAR}`` (no
    default) becomes the empty string. Used by the drift guard for
    diarize and TTS, whose containers are configured via env vars
    rather than vLLM command flags."""
    if not env_block:
        return {}
    out: dict[str, str] = {}
    for entry in env_block:
        if not isinstance(entry, str) or "=" not in entry:
            continue
        key, _, val = entry.partition("=")
        val = val.strip()
        m = re.match(r"^\$\{[^:}]+:-([^}]*)\}$", val)
        if m:
            val = m.group(1)
        elif val.startswith("${") and val.endswith("}"):
            val = ""
        out[key.strip()] = val
    return out


class TestComposeRecipeDriftGuard:
    """Catch the class of regression where docker-compose.gb10.yml hardcodes
    a vLLM flag value that diverges from the canonical recipe.

    Concrete past failure (2026-04-30): compose hardcoded
    `--gpu-memory-utilization 0.10` for scribe-asr while the recipe
    specified `0.04`. Under the shared 35B+autosre VRAM load this
    over-allocation triggered cuBLAS workspace contention that killed
    both ASR and pyannote diarization mid-meeting. The recipe author
    had even left an inline warning at recipes/qwen3-asr-vllm.yaml:14
    documenting that exact failure mode."""

    @pytest.fixture(scope="class")
    def asr_compose_flags(self) -> dict[str, str]:
        compose = yaml.safe_load(COMPOSE_PATH.read_text())
        cmd = compose["services"]["vllm-asr"]["command"]
        return _extract_vllm_flags(cmd)

    def test_asr_gpu_memory_utilization_matches_recipe(self, asr_compose_flags):
        recipe = load_recipe("asr-vllm")
        compose_val = float(asr_compose_flags["gpu-memory-utilization"])
        assert compose_val == pytest.approx(recipe["gpu_memory_utilization"]), (
            f"docker-compose.gb10.yml has --gpu-memory-utilization={compose_val} "
            f"for scribe-asr but the recipe (qwen3-asr-vllm.yaml) says "
            f"{recipe['gpu_memory_utilization']}. The recipe is the source of "
            f"truth — see its inline comment for the failure mode at "
            f"higher values under the shared-VRAM stack."
        )

    def test_asr_max_model_len_matches_recipe(self, asr_compose_flags):
        recipe = load_recipe("asr-vllm")
        assert int(asr_compose_flags["max-model-len"]) == recipe["max_model_len"]

    def test_asr_max_num_seqs_matches_recipe(self, asr_compose_flags):
        recipe = load_recipe("asr-vllm")
        assert int(asr_compose_flags["max-num-seqs"]) == recipe["max_num_seqs"]

    def test_asr_port_matches_recipe(self, asr_compose_flags):
        recipe = load_recipe("asr-vllm")
        assert int(asr_compose_flags["port"]) == recipe["port"]

    # ---------- Diarize + TTS use environment: blocks, not vllm flags ---------

    @pytest.fixture(scope="class")
    def diarize_compose_env(self) -> dict[str, str]:
        compose = yaml.safe_load(COMPOSE_PATH.read_text())
        env = compose["services"]["pyannote-diarize"]["environment"]
        return _extract_compose_env(env)

    @pytest.fixture(scope="class")
    def tts_compose_env(self) -> dict[str, str]:
        compose = yaml.safe_load(COMPOSE_PATH.read_text())
        env = compose["services"]["qwen3-tts"]["environment"]
        return _extract_compose_env(env)

    @pytest.fixture(scope="class")
    def tts2_compose_env(self) -> dict[str, str]:
        compose = yaml.safe_load(COMPOSE_PATH.read_text())
        env = compose["services"]["qwen3-tts-2"]["environment"]
        return _extract_compose_env(env)

    def test_diarize_port_matches_recipe(self, diarize_compose_env):
        recipe = load_recipe("diarization")
        assert int(diarize_compose_env["DIARIZE_PORT"]) == recipe["port"]

    def test_diarize_max_speakers_matches_recipe(self, diarize_compose_env):
        recipe = load_recipe("diarization")
        assert int(diarize_compose_env["DIARIZE_MAX_SPEAKERS"]) == recipe["max_speakers"]

    def test_diarize_pipeline_id_matches_recipe(self, diarize_compose_env):
        recipe = load_recipe("diarization")
        assert diarize_compose_env["DIARIZE_PIPELINE_ID"] == recipe["model_id"]

    def test_tts_port_matches_recipe(self, tts_compose_env):
        recipe = load_recipe("tts")
        assert int(tts_compose_env["TTS_PORT"]) == recipe["port"]

    def test_tts_model_matches_recipe(self, tts_compose_env):
        recipe = load_recipe("tts")
        assert tts_compose_env["TTS_MODEL"] == recipe["model_id"]

    def test_tts2_replica_uses_distinct_port(self, tts_compose_env, tts2_compose_env):
        # Replica MUST NOT share the primary's port (would collide on host network).
        assert tts2_compose_env["TTS_PORT"] != tts_compose_env["TTS_PORT"]

    def test_tts2_replica_uses_same_model_as_recipe(self, tts2_compose_env):
        # Both replicas must serve the same model (the recipe).
        recipe = load_recipe("tts")
        assert tts2_compose_env["TTS_MODEL"] == recipe["model_id"]
