"""vLLM translation backend — multi-language translation via OpenAI-compat API.

Uses vLLM's OpenAI-compatible chat endpoint for high-quality translation.
Supports any language pair from the language registry.
Leverages vLLM's continuous batching for concurrent translation requests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import httpx

from meeting_scribe import runtime_config
from meeting_scribe.backends.base import TranslateBackend
from meeting_scribe.languages import get_translation_prompt

logger = logging.getLogger(__name__)

# Shared JSONL log read by the autosre TUI Recent Requests panel.
# Same schema as anthropic_proxy's proxy-requests.jsonl.
_TRANS_LOG_PATH = Path.home() / ".local" / "share" / "autosre" / "scribe-translations.jsonl"


_TRANSLATE_LOG_SCHEMA_VERSION = 1


def _fingerprint_prior_context(
    prior_context: list[tuple[str, str]] | None,
) -> str:
    """Deterministic short hash of a ``prior_context`` list for cache keying.

    JSON-encodes the list as ``[[source, target], ...]`` with
    ``ensure_ascii=True`` so every byte that could collide (newline,
    backslash, quote, any non-ASCII) is escaped to ``\\uXXXX``.  That
    makes the serialisation lossless — two distinct context lists
    cannot ever serialise to the same bytes — and byte-identical
    across locales.  The resulting hash is truncated to 16 hex chars
    which is plenty of namespace for per-meeting LRU entries.

    Returns an empty string for ``None`` / empty context so the cache
    key reduces to the pre-B3 shape and existing short-phrase entries
    still hit.
    """
    if not prior_context:
        return ""
    payload = json.dumps(
        [[s, t] for s, t in prior_context],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _log_translation(
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
    translated: str,
    elapsed_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    kind: str = "translate",
    source: str = "live",
    meeting_id: str = "",
) -> None:
    """Fire-and-forget JSONL logger. Never raises.

    ``kind`` — always ``"translate"`` from this path; reserved for future
    row types that may share the file.  ``source`` tags the caller
    (``"live"`` vs ``"refinement"``) so the validation harness can
    attribute load by path.  ``meeting_id`` is the active meeting UUID —
    empty string when no meeting is bound (e.g. warmup calls).  All three
    are required by ``translate_stats.py`` for strict filtering so any
    future row that lacks them will be rejected as malformed rather than
    silently polluting the aggregation.
    """
    try:
        _TRANS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps(
            {
                "ts": time.time(),
                "schema_version": _TRANSLATE_LOG_SCHEMA_VERSION,
                "kind": kind,
                "source": source,
                "meeting_id": meeting_id,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_ms": round(elapsed_ms, 1),
                "tps": round(output_tokens / (elapsed_ms / 1000), 1)
                if elapsed_ms > 0 and output_tokens > 0
                else 0,
                "max_tokens": 256,
                "stream": False,
                "tools": 0,
                "thinking": False,
                "messages": 2,
                "prompt_prefix": f"[{source_lang}→{target_lang}] {text[:80]}",
                "response_prefix": translated[:200],
            }
        )
        with _TRANS_LOG_PATH.open("a") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # Never let logging break translation


class VllmTranslateBackend(TranslateBackend):
    """Translation via vLLM OpenAI-compatible chat endpoint.

    Works with any LLM served by vLLM. For best JA↔EN results, use
    a multilingual model like qwen3.5 or nemotron.

    The vLLM server handles batching — multiple concurrent translate()
    calls are batched automatically for maximum throughput.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8010",
        model: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> None:
        # ``_static_base_url`` is the value from ServerConfig at startup.
        # It's the fallback when the hot-reload ``translate_url`` knob is
        # unset.  ``_base_url`` tracks the currently-active URL for the
        # model auto-detect cache — see ``_resolve_base_url`` for the
        # per-request read that drives the Phase 7 rollback knob.
        self._static_base_url = base_url.rstrip("/")
        self._base_url = self._static_base_url
        self._model = model  # None = auto-detect from vLLM
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

        # LRU over (source_lang, target_lang, normalized_text) → translation.
        # Real meetings are full of repeated short phrases ("はい", "そうそう",
        # "OK", "I see") — caching the last N saves ~15-20 % of translate
        # calls in the sample we measured. OrderedDict is cheap enough.
        from collections import OrderedDict

        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._cache_max = 256
        self._cache_hits = 0
        self._cache_misses = 0

    async def start(self) -> None:
        """Initialize HTTP client and verify vLLM is reachable."""
        self._client = httpx.AsyncClient(timeout=self._timeout)

        # Auto-detect model if not specified
        if not self._model:
            try:
                resp = await self._client.get(f"{self._base_url}/v1/models")
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    self._model = models[0]["id"]
                    logger.info("vLLM auto-detected model: %s", self._model)
                else:
                    raise RuntimeError("No models available on vLLM server")
            except httpx.HTTPError as e:
                raise RuntimeError(f"Cannot connect to vLLM at {self._base_url}: {e}") from e

        # Health check
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            logger.info("vLLM connected: %s (model=%s)", self._base_url, self._model)
        except httpx.HTTPError:
            logger.warning("vLLM health check failed (may still work)")

    async def stop(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _resolve_base_url(self) -> str:
        """Return the live base URL, re-detecting the model if the hot-
        reload knob switched endpoints since the last request.

        This is the per-request read that the Phase 7 rollback procedure
        depends on: ``meeting-scribe config set translate_url ...`` +
        SIGHUP flips the runtime-config value, and the next translate()
        call picks it up here.  A URL change invalidates the cached
        ``self._model`` because a different endpoint likely serves a
        different model.
        """
        override = runtime_config.get("translate_url")
        current = (override or self._static_base_url).rstrip("/")
        if current != self._base_url:
            logger.info(
                "translate backend URL changed: %s → %s (re-detecting model)",
                self._base_url,
                current,
            )
            self._base_url = current
            self._model = None
        if self._model is None and self._client is not None:
            try:
                resp = await self._client.get(f"{self._base_url}/v1/models")
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    self._model = models[0]["id"]
                    logger.info("vLLM auto-detected model: %s", self._model)
            except httpx.HTTPError:
                logger.warning(
                    "Model auto-detect failed against %s; keeping previous id=%s",
                    self._base_url,
                    self._model,
                )
        return self._base_url

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        prior_context: list[tuple[str, str]] | None = None,
        meeting_id: str | None = None,
    ) -> str:
        """Translate text via vLLM chat completion.

        Uses the OpenAI-compatible /v1/chat/completions endpoint.
        vLLM's continuous batching handles concurrent requests efficiently.

        ``prior_context`` — when a non-empty list is passed, each tuple
        ``(source, translation)`` is folded into the system prompt as
        rolling meeting context by :func:`get_translation_prompt`.  The
        live path leaves this as ``None`` to honour the latency SLO;
        the refinement worker passes the tail of its own already-
        refined history when the
        ``refinement_context_window_segments`` knob is on.

        ``meeting_id`` — optional meeting UUID that the caller (usually
        ``TranslationQueue``) supplies so the JSONL log row can be
        attributed to the correct meeting.  Left ``None`` for callers
        outside of an active meeting (startup warmup, ad-hoc tests).
        """
        if self._client is None:
            msg = "vLLM client not initialized"
            raise RuntimeError(msg)

        # Side-effect: updates self._base_url and re-detects self._model
        # if the hot-reload translate_url knob switched endpoints.
        await self._resolve_base_url()

        # LRU fast-path: exact repeat of an already-translated phrase
        # skips the vLLM call entirely.  Normalize by trimming +
        # collapsing inner whitespace so "はい 。" and "はい。" share a
        # cache slot.
        #
        # Phase B3 — context-aware cache key.  The key now includes a
        # fingerprint of ``prior_context`` so a repeated utterance with
        # the same surrounding history still hits the cache, but a
        # repeated utterance with a *different* history is treated as
        # a different lookup.  The fingerprint uses JSON serialization
        # (``ensure_ascii=True``) before the hash so delimiter
        # characters in the utterance can never alias two distinct
        # histories to the same bytes.  When ``prior_context`` is
        # ``None`` the hash slot is empty and the key reduces to the
        # pre-B3 shape — existing short-phrase cache entries still hit.
        normalized_text = " ".join((text or "").strip().split())
        ctx_hash = _fingerprint_prior_context(prior_context)
        cache_key = (source_language, target_language, normalized_text, ctx_hash)
        if normalized_text:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                self._cache_hits += 1
                return cached
            self._cache_misses += 1
        else:
            cache_key = None  # type: ignore[assignment]

        system = get_translation_prompt(
            source_language, target_language, prior_context=prior_context
        )

        t0 = time.monotonic()

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": text},
                    ],
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "stream": False,
                    # LATENCY SLO: the live translate path MUST NOT flip
                    # enable_thinking=True.  Reasoning-mode responses add
                    # 3-5 s per utterance; the live path has to keep pace
                    # with speaker audio (new utterance every ~400-600 ms).
                    # Reasoning is allowed on post-process paths (slide
                    # translation, summaries) — see `project_scribe_
                    # reasoning_path_split` in auto-memory.  Any change
                    # here must be matched by a corresponding test update
                    # in tests/test_translate_vllm_latency_slo.py.
                    "chat_template_kwargs": {"enable_thinking": False},
                    # Priority ordering on the shared vLLM coder (lower = higher
                    # priority under --scheduling-policy=priority):
                    #   -10  live translation (this)
                    #    10  Claude Code coding agent (autosre anthropic proxy)
                    #    20  plan-review runner (autosre)
                    # Translation is always highest so real-time meeting audio
                    # preempts any coding or review workload on the GPU.
                    "priority": -10,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            translated = data["choices"][0]["message"]["content"].strip()

            elapsed_ms = (time.monotonic() - t0) * 1000
            usage = data.get("usage", {}) or {}
            logger.debug(
                "vLLM translated [%s→%s] %.0fms: '%s' → '%s'",
                source_language,
                target_language,
                elapsed_ms,
                text[:40],
                translated[:40],
            )

            # Log to shared JSONL so autosre TUI can show translation requests
            _log_translation(
                model=self._model or "unknown",
                source_lang=source_language,
                target_lang=target_language,
                text=text,
                translated=translated,
                elapsed_ms=elapsed_ms,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                kind="translate",
                source="live",
                meeting_id=meeting_id or "",
            )

            if not translated:
                msg = "Empty translation from vLLM"
                raise RuntimeError(msg)

            # Stash in the LRU for next-identical-phrase hits.  The
            # key is context-aware under B3 so a later call with the
            # same text AND the same rolling history can hit, while a
            # call with a different history gets a fresh vLLM
            # translation — no risk of serving a wrong-topic cached
            # answer.
            if cache_key is not None and cache_key[2]:
                self._cache[cache_key] = translated
                if len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)

            return translated

        except httpx.TimeoutException as e:
            raise TimeoutError(f"vLLM translation timed out after {self._timeout}s") from e
        except httpx.HTTPStatusError as e:
            msg = f"vLLM API error: {e.response.status_code}"
            raise RuntimeError(msg) from e
