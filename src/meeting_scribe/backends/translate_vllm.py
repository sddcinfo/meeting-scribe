"""vLLM translation backend — LLM-quality JA↔EN translation via OpenAI-compat API.

Uses vLLM's OpenAI-compatible chat endpoint for high-quality translation.
Supports any model served by vLLM (qwen3.5, nemotron, gemma4, etc.).
Leverages vLLM's continuous batching for concurrent translation requests.

For GB10: vLLM serves the model, this backend sends translation requests.
The same endpoint works for single-node or TP=2 cluster — vLLM abstracts it.
"""

from __future__ import annotations

import logging
import time

import httpx

from meeting_scribe.backends.base import TranslateBackend

logger = logging.getLogger(__name__)

JA_TO_EN_SYSTEM = (
    "You are a professional Japanese-to-English translator for live meeting transcription. "
    "Translate the following Japanese text to natural, fluent English. "
    "Preserve the meaning, tone, and speaker's intent. "
    "Output ONLY the English translation, nothing else. No explanations."
)

EN_TO_JA_SYSTEM = (
    "You are a professional English-to-Japanese translator for live meeting transcription. "
    "Translate the following English text to natural, fluent Japanese. "
    "Preserve the meaning, tone, and speaker's intent. "
    "Output ONLY the Japanese translation, nothing else. No explanations."
)


class VllmTranslateBackend(TranslateBackend):
    """Translation via vLLM OpenAI-compatible chat endpoint.

    Works with any LLM served by vLLM. For best JA↔EN results, use
    a multilingual model like qwen3.5 or nemotron.

    The vLLM server handles batching — multiple concurrent translate()
    calls are batched automatically for maximum throughput.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model  # None = auto-detect from vLLM
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

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

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text via vLLM chat completion.

        Uses the OpenAI-compatible /v1/chat/completions endpoint.
        vLLM's continuous batching handles concurrent requests efficiently.
        """
        if self._client is None:
            msg = "vLLM client not initialized"
            raise RuntimeError(msg)

        if source_language == "ja" and target_language == "en":
            system = JA_TO_EN_SYSTEM
        elif source_language == "en" and target_language == "ja":
            system = EN_TO_JA_SYSTEM
        else:
            msg = f"Unsupported language pair: {source_language} → {target_language}"
            raise ValueError(msg)

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
                },
            )
            resp.raise_for_status()
            data = resp.json()
            translated = data["choices"][0]["message"]["content"].strip()

            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.debug(
                "vLLM translated [%s→%s] %.0fms: '%s' → '%s'",
                source_language,
                target_language,
                elapsed_ms,
                text[:40],
                translated[:40],
            )

            if not translated:
                msg = "Empty translation from vLLM"
                raise RuntimeError(msg)

            return translated

        except httpx.TimeoutException as e:
            raise TimeoutError(f"vLLM translation timed out after {self._timeout}s") from e
        except httpx.HTTPStatusError as e:
            msg = f"vLLM API error: {e.response.status_code}"
            raise RuntimeError(msg) from e
