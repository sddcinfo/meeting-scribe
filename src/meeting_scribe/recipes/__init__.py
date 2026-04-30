"""Embedded model recipes for meeting-scribe on GB10.

Recipes are YAML files defining model configs, vLLM parameters,
and container settings. Pattern adapted from auto-sre.
"""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

RECIPES_DIR = Path(__file__).parent


def list_recipes() -> list[str]:
    """List available recipe short names."""
    return sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))


def load_recipe(name: str) -> dict:
    """Load a recipe by short name or model key.

    Args:
        name: Recipe filename stem (e.g., "qwen3.6-35b-translation")
              or model_key value (e.g., "translation").

    Returns:
        Recipe dict with all fields.

    Raises:
        FileNotFoundError: If recipe doesn't exist.
    """
    # Try direct filename match
    path = RECIPES_DIR / f"{name}.yaml"
    if path.exists():
        return _load_yaml(path)

    # Try matching by model_key field
    for recipe_path in RECIPES_DIR.glob("*.yaml"):
        recipe = _load_yaml(recipe_path)
        if recipe.get("model_key") == name:
            return recipe

    available = ", ".join(list_recipes())
    msg = f"Recipe '{name}' not found. Available: {available}"
    raise FileNotFoundError(msg)


def all_model_ids(include_shared: bool = False) -> list[str]:
    """HuggingFace model IDs that meeting-scribe needs cached locally.

    Args:
        include_shared: When False (default — preserves the
            pre-2026-04-30 behaviour for any caller that depended
            on it), skip recipes flagged ``mode: shared`` (the
            autosre-owned translation model). When True, include
            them — required by the customer-install bootstrap
            because customer GB10s always run autosre alongside
            meeting-scribe and a wipe-and-reinstall MUST land a
            fully-cached /data/huggingface so neither stack races
            a download on first start.

    The original docstring noted the shared-skip was to "not waste
    ~17 GB on a model that runs out of process anyway". That
    rationale held when customer devices weren't running their own
    autosre instance. After the customer-install path was unified
    in bootstrap.sh:292+, customer devices DO run autosre locally,
    and the 35B becomes mandatory cargo on the device — wiping +
    reinstalling without it leaves autosre crash-looping on first
    boot with `LocalEntryNotFoundError` because `HF_HUB_OFFLINE=1`
    blocks the recovery download.
    """
    ids = []
    for recipe_path in sorted(RECIPES_DIR.glob("*.yaml")):
        recipe = _load_yaml(recipe_path)
        if not include_shared and recipe.get("mode") == "shared":
            continue
        model_id = recipe.get("model_id")
        if model_id:
            ids.append(model_id)
    return ids


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    text = path.read_text()
    if not text.strip():
        return {}
    result = yaml.safe_load(text)
    if not isinstance(result, dict):
        return {}
    return result
