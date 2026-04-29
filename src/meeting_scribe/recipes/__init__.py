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


def all_model_ids() -> list[str]:
    """HuggingFace model IDs that meeting-scribe is responsible for pulling.

    Skips recipes flagged ``mode: shared`` — those are managed by an
    external service (auto-sre) and should not land in meeting-scribe's
    /data/huggingface cache. Pulling them here just wastes ~17 GB of
    customer-device disk on a model that runs out of process anyway.
    """
    ids = []
    for recipe_path in sorted(RECIPES_DIR.glob("*.yaml")):
        recipe = _load_yaml(recipe_path)
        if recipe.get("mode") == "shared":
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
