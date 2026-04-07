"""Ollama translation backend (POC).

Uses a local Ollama instance for JA↔EN translation.
Model: qwen3.5:9b (or configurable).
"""

from __future__ import annotations

import logging

import httpx

from meeting_scribe.backends.base import TranslateBackend

logger = logging.getLogger(__name__)

# System prompts for translation
JA_TO_EN_SYSTEM = (
    "You are a professional Japanese-to-English translator. "
    "Translate the following Japanese text to natural, fluent English. "
    "Preserve the meaning and tone. Output ONLY the translation, nothing else."
)

EN_TO_JA_SYSTEM = (
    "You are a professional English-to-Japanese translator. "
    "Translate the following English text to natural, fluent Japanese. "
    "Preserve the meaning and tone. Output ONLY the translation, nothing else."
)


class OllamaTranslateBackend(TranslateBackend):
    """Translation via local Ollama instance.

    POC backend for MacBook. Uses chat completion API.
    """

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Initialize the HTTP client and verify Ollama is reachable."""
        self._client = httpx.AsyncClient(timeout=self._timeout)

        # Verify Ollama is running
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            logger.info("Ollama connected: %s", self._base_url)
        except httpx.HTTPError as e:
            logger.warning("Ollama not reachable at %s: %s", self._base_url, e)

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text via Ollama chat API.

        Args:
            text: Source text to translate.
            source_language: "ja" or "en".
            target_language: "ja" or "en".

        Returns:
            Translated text.

        Raises:
            TimeoutError: If Ollama doesn't respond within timeout.
            RuntimeError: If the Ollama API returns an error.
        """
        if self._client is None:
            msg = "Ollama client not initialized. Call start() first."
            raise RuntimeError(msg)

        if source_language == "ja" and target_language == "en":
            system_prompt = JA_TO_EN_SYSTEM
        elif source_language == "en" and target_language == "ja":
            system_prompt = EN_TO_JA_SYSTEM
        else:
            msg = f"Unsupported language pair: {source_language} → {target_language}"
            raise ValueError(msg)

        try:
            resp = await self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    "stream": False,
                    "keep_alive": "30m",  # Keep model loaded in memory
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            translated = data.get("message", {}).get("content", "").strip()

            if not translated:
                msg = "Empty translation response"
                raise RuntimeError(msg)

            return translated

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Ollama translation timed out after {self._timeout}s") from e
        except httpx.HTTPStatusError as e:
            msg = f"Ollama API error: {e.response.status_code}"
            raise RuntimeError(msg) from e
