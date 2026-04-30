"""Thin wrapper around ``docker compose -f docker-compose.gb10.yml``.

Meeting-scribe used to have two parallel launch systems:

  1. ``docker-compose.gb10.yml`` — compose file with container_names,
     volumes, ports, labels for willfarrell/autoheal, profiles for
     gated services (e.g. ``bench`` for benchmark sidecars), and a
     single source of truth for the stack topology.

  2. ``src/meeting_scribe/recipes/*.yaml`` + ``infra/containers.py`` —
     a parallel path that reimplemented half of the above by reading
     recipes and calling ``docker run`` directly.

The two paths produced containers with DIFFERENT names for the same
model (e.g. ``scribe-asr`` from compose, ``scribe-asr-vllm`` from the
recipe) and neither cleaned up the other's containers. On 2026-04-14
we hit a system-RAM OOM because the recipe path was launching a
duplicate of the 35B translate model every time ``meeting-scribe gb10
up`` ran, while autosre's 35B was already serving the same port.

The refactor (2026-04-14):

  - Compose is the single source of truth. Every container the stack
    runs lives in ``docker-compose.gb10.yml``.
  - This module is a thin wrapper around ``docker compose`` subprocess
    calls, exposing ``compose_up``, ``compose_down``, ``compose_restart``,
    and ``compose_services`` for the CLI to call.
  - ``infra/containers.py`` is now limited to docker-level helpers
    (``list_containers``, ``pull_models``) used for pre-flight tasks
    before compose runs.
  - The recipe files stay useful for:
      - pull-models (iterates recipe model_ids)
      - test assertions (port expectations, model_id stability)
    but are NOT used by the runtime launch path.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = _REPO_ROOT / "docker-compose.gb10.yml"


def _compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE), *args]


def _run(cmd: list[str], timeout: int) -> None:
    logger.info("compose: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"docker compose failed (rc={result.returncode}): {stderr}")


def compose_up(
    services: list[str] | None = None,
    profiles: list[str] | None = None,
    pull: str | None = None,
) -> None:
    """Bring up the stack (detached). Optional service + profile filters.

    Without filters, starts every service in the compose file EXCEPT
    those gated behind a profile (e.g. ``funcosyvoice`` under the
    ``bench`` profile stays off until explicitly requested).

    Args:
        pull: Docker compose pull policy (``"never"``, ``"always"``,
              ``"missing"``). ``None`` keeps Docker's default (pull if
              missing). Pass ``"never"`` from the systemd boot path to
              avoid hanging on network-unavailable cold boots.
    """
    cmd: list[str] = []
    if profiles:
        for p in profiles:
            cmd.extend(["--profile", p])
    up_args = ["up", "-d"]
    if pull:
        up_args.extend(["--pull", pull])
    cmd = _compose_cmd(*cmd, *up_args)
    if services:
        cmd.extend(services)
    _run(cmd, timeout=600)  # 10 min — create can be slow on cold boot


def compose_down(remove_volumes: bool = False) -> None:
    """Stop + remove the stack containers.

    ``remove_volumes=False`` by default — HF model cache volumes are
    too expensive to rebuild (model re-download).
    """
    args = ["down"]
    if remove_volumes:
        args.append("-v")
    _run(_compose_cmd(*args), timeout=300)


def compose_restart(service: str, timeout: int = 30, recreate: bool = False) -> None:
    """Restart a single compose service.

    Default mode is `docker compose restart`, which preserves the running
    container's environment, image, and volumes — fast (~5s) and right for
    CUDA-context recovery.

    Pass ``recreate=True`` for a full recreate via
    ``docker compose up -d --force-recreate``, which is required when the
    compose file's environment / image / volume / port spec has changed.
    A plain restart silently keeps the OLD env even if the file was edited.
    """
    if recreate:
        # `up -d --force-recreate` alone fails with "container is running:
        # stop the container before removing" on host-network services where
        # the network reuse logic gets confused. The reliable shape is
        # explicit stop → rm → up. Each step is bounded by the same timeout
        # budget so a hung container can't park us indefinitely.
        _run(_compose_cmd("stop", "-t", str(timeout), service), timeout=timeout + 30)
        _run(_compose_cmd("rm", "-f", service), timeout=timeout + 30)
        _run(_compose_cmd("up", "-d", "--no-deps", service), timeout=timeout + 120)
    else:
        _run(_compose_cmd("restart", "-t", str(timeout), service), timeout=timeout + 60)


def compose_services() -> list[str]:
    """Return the list of service names declared in the compose file
    (including profile-gated ones)."""
    result = subprocess.run(
        _compose_cmd("config", "--services"),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return []
    return [s.strip() for s in result.stdout.splitlines() if s.strip()]


# ─────────────────────────────────────────────────────────────────────
# Recipe ↔ compose source-drift detector (boot-time warn).
#
# Compose is the runtime source of truth (per the 2026-04-14 refactor
# above). The recipe yamls under src/meeting_scribe/recipes/ are kept
# as documentation + pull-models drivers + drift sentinels.
#
# This helper is called once during server startup and emits a WARNING
# log line per source-yaml mismatch. Intentionally NON-FATAL: an
# operator may have tuned compose during incident response without yet
# updating the recipe — refusing boot here would block recovery in
# exactly the wrong moment. The CI test
# (tests/test_recipes.py::TestComposeRecipeDriftGuard) is the strict
# gate for PRs.
#
# Concrete past failure (2026-04-30): compose hardcoded
# `--gpu-memory-utilization 0.10` for vllm-asr while the recipe said
# `0.04`. The CI test now catches that class of drift; this runtime
# helper additionally surfaces drift on a host that booted from a
# stale image or a hand-edited compose without a CI cycle.
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecipeMismatch:
    service: str
    field: str
    compose_value: str
    recipe_value: str

    def __str__(self) -> str:
        return (
            f"source drift: {self.service} {self.field} "
            f"compose={self.compose_value!r} recipe={self.recipe_value!r}"
        )


def _extract_vllm_flags(compose_command: list[str] | str) -> dict[str, str]:
    """Parse a compose service.command (vllm serve …) into a flag dict."""
    if isinstance(compose_command, list):
        text = " ".join(str(p) for p in compose_command)
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
    """Parse a compose service.environment list into a dict; strip
    `${VAR:-default}` shell-expansion to just the default value."""
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


def assert_recipe_source_parity() -> list[RecipeMismatch]:
    """Compare each compose service's perf-sensitive fields against the
    recipe yaml that documents the intended values. Return a list of
    mismatches (empty if fully aligned).

    Caller is responsible for the policy decision (warn vs fail). At
    server startup we log a WARNING per mismatch and continue booting
    — see module docstring for the rationale.

    Imports the recipe loader lazily so this module stays usable in
    contexts that don't have the recipes package available (e.g.
    customer-install pre-flight)."""
    import yaml  # local import — avoid forcing yaml on all compose users

    from meeting_scribe.recipes import load_recipe

    try:
        compose = yaml.safe_load(COMPOSE_FILE.read_text())
    except (FileNotFoundError, yaml.YAMLError) as exc:
        logger.warning("could not parse %s for source-drift check: %s", COMPOSE_FILE, exc)
        return []

    services = compose.get("services", {}) if isinstance(compose, dict) else {}
    mismatches: list[RecipeMismatch] = []

    # ASR — vllm command flags
    asr = services.get("vllm-asr") or {}
    asr_flags = _extract_vllm_flags(asr.get("command", []))
    if asr_flags:
        recipe = load_recipe("asr-vllm")
        for compose_key, recipe_key in [
            ("gpu-memory-utilization", "gpu_memory_utilization"),
            ("max-model-len", "max_model_len"),
            ("max-num-seqs", "max_num_seqs"),
            ("port", "port"),
        ]:
            if compose_key not in asr_flags:
                continue
            cv = asr_flags[compose_key]
            rv = recipe.get(recipe_key)
            if rv is None:
                continue
            # Numeric compare where both sides are numbers; otherwise string.
            try:
                if float(cv) != float(rv):
                    mismatches.append(
                        RecipeMismatch("vllm-asr", compose_key, cv, str(rv))
                    )
            except (TypeError, ValueError):
                if cv != str(rv):
                    mismatches.append(
                        RecipeMismatch("vllm-asr", compose_key, cv, str(rv))
                    )

    # Diarize — environment vars
    diar = services.get("pyannote-diarize") or {}
    diar_env = _extract_compose_env(diar.get("environment"))
    if diar_env:
        recipe = load_recipe("diarization")
        for env_key, recipe_key in [
            ("DIARIZE_PORT", "port"),
            ("DIARIZE_MAX_SPEAKERS", "max_speakers"),
            ("DIARIZE_PIPELINE_ID", "model_id"),
        ]:
            if env_key not in diar_env:
                continue
            cv = diar_env[env_key]
            rv = recipe.get(recipe_key)
            if rv is None:
                continue
            try:
                if float(cv) != float(rv):
                    mismatches.append(
                        RecipeMismatch("pyannote-diarize", env_key, cv, str(rv))
                    )
            except (TypeError, ValueError):
                if cv != str(rv):
                    mismatches.append(
                        RecipeMismatch("pyannote-diarize", env_key, cv, str(rv))
                    )

    # TTS primary replica — environment vars; `qwen3-tts-2` has a distinct
    # port by design (8012 vs 8002), so we only check the primary's port
    # against the recipe; for tts-2 we just check the model.
    for tts_svc in ("qwen3-tts", "qwen3-tts-2"):
        tts = services.get(tts_svc) or {}
        tts_env = _extract_compose_env(tts.get("environment"))
        if not tts_env:
            continue
        recipe = load_recipe("tts")
        if tts_svc == "qwen3-tts" and "TTS_PORT" in tts_env:
            cv, rv = tts_env["TTS_PORT"], recipe.get("port")
            if rv is not None and int(cv) != int(rv):
                mismatches.append(
                    RecipeMismatch(tts_svc, "TTS_PORT", cv, str(rv))
                )
        if "TTS_MODEL" in tts_env:
            cv, rv = tts_env["TTS_MODEL"], recipe.get("model_id")
            if rv is not None and cv != str(rv):
                mismatches.append(
                    RecipeMismatch(tts_svc, "TTS_MODEL", cv, str(rv))
                )

    return mismatches


def warn_on_recipe_source_drift() -> None:
    """Boot-time hook: assert parity, emit one WARNING log line per
    mismatch, never raise. Safe to call from any startup path."""
    try:
        mismatches = assert_recipe_source_parity()
    except Exception as exc:  # noqa: BLE001
        logger.warning("recipe-parity check raised %s: %s", type(exc).__name__, exc)
        return
    for m in mismatches:
        logger.warning("recipe drift: %s", m)
