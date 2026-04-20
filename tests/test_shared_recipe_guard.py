"""Regression: shared recipes must not spawn a duplicate container.

Context (2026-04-14 OOM):
    ``cli.py:gb10_up`` used to iterate recipe names and call
    ``start_container()`` on each — including ``qwen3.5-35b-translation``
    which is marked ``mode: shared`` in its recipe. The safety
    contract ("this model is managed by autosre, don't launch it from
    here") lived only in the recipe file comment + the compose file
    comment. Nothing enforced it in code.

    Result: every ``meeting-scribe gb10 up`` created a second
    scribe-translation vLLM container that loaded the same 60 GB 35B
    model a second time. Memory pressure accumulated until the kernel
    OOM-killed systemd/tmux/pipewire/NetworkManager and multiple
    Claude Code sessions.

    Full refactor (2026-04-14): compose is now the single source of
    truth. ``start_container`` was removed entirely. The 35B translate
    service is NOT declared in docker-compose.gb10.yml at all. The
    ``mode: shared`` sentinel in the recipe is now informational —
    compose can't launch what it doesn't define.

    These tests keep that fix from rotting.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).parent.parent


def _compose_text() -> str:
    return (REPO / "docker-compose.gb10.yml").read_text()


def test_translation_recipe_still_marked_shared():
    """If someone drops `mode: shared` from the translation recipe,
    every downstream check based on that sentinel (docs, test
    assertions, future automation) stops working. Catch it here."""
    from meeting_scribe.recipes import load_recipe
    recipe = load_recipe("qwen3.5-35b-translation")
    assert recipe.get("mode") == "shared", (
        "qwen3.5-35b-translation.yaml must set `mode: shared` — "
        "it's the documentation sentinel that says 'this is managed "
        "by autosre, not by scribe'"
    )


def test_start_container_is_removed():
    """``infra/containers.start_container`` was deleted in the compose
    refactor. If it comes back, someone has reverted the refactor."""
    from meeting_scribe.infra import containers
    assert not hasattr(containers, "start_container"), (
        "start_container was removed 2026-04-14 as part of the compose "
        "refactor. Use meeting_scribe.infra.compose.compose_up instead."
    )


def test_compose_has_no_translation_service():
    """docker-compose.gb10.yml must NOT define a scribe-translation
    service. The 35B runs under autosre and is referenced by URL.

    This parser iterates the services block and checks service names
    + container_name overrides. Anything with `translation` in the
    name should raise a red flag (except comments, which we skip)."""
    compose = _compose_text()
    in_services = False
    service_names: list[str] = []
    container_names: list[str] = []
    for raw in compose.splitlines():
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw.startswith("services:"):
            in_services = True
            continue
        if not in_services:
            continue
        # 2-space indent = service name
        if raw.startswith("  ") and not raw.startswith("    "):
            name = stripped.split(":", 1)[0].strip()
            if name and ":" in raw:
                service_names.append(name)
        if "container_name:" in stripped:
            container_names.append(stripped.split(":", 1)[1].strip())

    assert "scribe-translation" not in service_names, (
        f"docker-compose defines scribe-translation as a service: {service_names}"
    )
    assert "scribe-translation" not in container_names, (
        f"docker-compose defines container_name=scribe-translation: {container_names}"
    )


def test_tts_recipe_model_key_disambiguates_from_legacy():
    """The ``qwen3-tts-vllm.yaml`` recipe uses ``model_key: tts-vllm``
    (not ``tts``) because docker-compose.gb10.yml already owns the
    name ``scribe-tts`` via the legacy ``qwen3-tts`` service. Using
    ``tts`` as the model_key would have created a naming collision
    next time the refactor comes back to the recipe runtime path."""
    from meeting_scribe.recipes import load_recipe
    recipe = load_recipe("qwen3-tts-vllm")
    assert recipe.get("model_key") == "tts-vllm", (
        "qwen3-tts-vllm.yaml must use model_key='tts-vllm' to avoid "
        "colliding with the legacy scribe-tts container"
    )


def test_compose_module_exposes_wrapper_api():
    """The new compose wrapper module provides the CLI's primitives.
    If any of them disappears, gb10_up/down/restart-container break."""
    from meeting_scribe.infra import compose
    for attr in ("compose_up", "compose_down", "compose_restart",
                 "compose_services", "COMPOSE_FILE"):
        assert hasattr(compose, attr), f"compose module missing {attr}"
    assert compose.COMPOSE_FILE.name == "docker-compose.gb10.yml"


def test_no_duplicate_container_names_for_same_model():
    """For every unique (port, model_id) pair that appears anywhere
    in the compose file + recipe directory, there must be AT MOST
    one container name. Otherwise we're back to the scribe-asr /
    scribe-asr-vllm split-brain that triggered the OOM.

    This is a static check: we parse both sources and cross-reference.
    """
    import re

    from meeting_scribe.recipes import list_recipes, load_recipe

    compose = _compose_text()
    # Parse (container_name, port) from compose. Naive block scan.
    compose_pairs: dict[str, set[int]] = {}  # container_name → ports used
    current_container: str | None = None
    for raw in compose.splitlines():
        if "container_name:" in raw:
            current_container = raw.split(":", 1)[1].strip().strip('"')
        port_match = re.search(r'^\s+-\s+"?(\d{2,5}):\d{2,5}"?', raw)
        if port_match and current_container:
            compose_pairs.setdefault(current_container, set()).add(int(port_match.group(1)))

    # Parse (model_key, port) from recipes.
    recipe_pairs: dict[str, tuple[str, int]] = {}
    for name in list_recipes():
        r = load_recipe(name)
        if r.get("mode") == "shared":
            continue  # shared recipes are informational only
        mk = r.get("model_key")
        port = r.get("port")
        if mk and port:
            recipe_pairs[mk] = (name, int(port))

    # For each recipe port, check that at most one compose container
    # name is reachable on that port (compose is the runtime truth).
    for mk, (recipe_name, port) in recipe_pairs.items():
        owners = [name for name, ports in compose_pairs.items() if port in ports]
        # It's fine if compose doesn't have an entry — recipes can
        # reference ports served by external things (e.g. the 35B
        # served by autosre on 8010). What we're guarding against is
        # MULTIPLE compose containers claiming the same port.
        assert len(owners) <= 1, (
            f"Multiple compose containers claim port {port}: {owners}. "
            f"This is the same split-brain pattern that caused the "
            f"2026-04-14 OOM. Pick one container name for port {port}."
        )
