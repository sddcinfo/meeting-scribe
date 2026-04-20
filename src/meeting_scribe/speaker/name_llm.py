"""LLM-based speaker name extraction.

Replaces regex heuristic with neural understanding via vLLM.
Reuses the translation vLLM endpoint (Qwen3.5-35B) — no extra model needed.
Falls back to regex if LLM is unavailable.

Uses max_tokens=20 to keep latency minimal (name extraction, not generation).
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a name extraction assistant. Given a transcript utterance, determine if \
the speaker is introducing themselves. If yes, extract ONLY their name. \
If no self-introduction is present, respond with exactly "null".

Rules:
- Return only the name (e.g., "Brad", "田中", "Tanaka")
- Handle both English and Japanese introductions
- For Japanese names, return the name in the original script
- Do NOT return titles, honorifics, or extra text
- If ambiguous, return "null"

Examples:
- "My name is Brad" → "Brad"
- "Hi, I'm Tanaka from engineering" → "Tanaka"
- "田中です" → "田中"
- "私はブラッドです" → "ブラッド"
- "Let's start the meeting" → "null"
- "Today we'll discuss the roadmap" → "null"\
"""


class LLMNameExtractor:
    """Extract speaker names from utterances using an LLM.

    Connects to a vLLM endpoint via OpenAI-compatible API.
    Designed to reuse the translation endpoint (no extra model).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8010",
        model: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client: httpx.AsyncClient | None = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    async def start(self) -> None:
        """Connect to endpoint and auto-detect model."""
        self._client = httpx.AsyncClient(timeout=10.0)

        try:
            resp = await self._client.get(f"{self._base_url}/v1/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models and not self._model:
                self._model = models[0]["id"]
            self._available = True
            logger.info("LLM name extractor ready: model=%s", self._model)
        except (httpx.HTTPError, KeyError, IndexError) as e:
            logger.info("LLM name extractor unavailable: %s", e)
            self._available = False

    async def stop(self) -> None:
        """Release HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._available = False

    async def extract_name(self, text: str, language: str = "auto") -> str | None:
        """Extract a speaker's name from a transcript utterance.

        Args:
            text: The transcript text to analyze.
            language: Source language ("ja", "en", or "auto").

        Returns:
            Extracted name string, or None if no self-introduction detected.
        """
        if not self._available or not self._client or not text.strip():
            return None

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": text.strip()},
                    ],
                    "max_tokens": 20,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    # Name extraction runs inline during a live meeting
                    # to catch "Hi, I'm Brad" self-intros. Same tier as
                    # live translation (-10) — it's on the critical path
                    # for the UI showing a name before the speaker moves
                    # on. Was -5 (priority inversion vs refinement -8).
                    "priority": -10,
                },
            )
            resp.raise_for_status()
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except (httpx.HTTPError, KeyError, IndexError) as e:
            logger.debug("LLM name extraction failed: %s", e)
            return None

        # Parse response
        if not content or content.lower() in ("null", "none", "n/a", ""):
            return None

        # Validate: name should be short (1-20 chars), no sentences
        name = content.strip().strip('"').strip("'")
        if len(name) > 20 or (" " in name and len(name.split()) > 3):
            logger.debug("LLM returned invalid name: '%s'", name)
            return None

        return name
