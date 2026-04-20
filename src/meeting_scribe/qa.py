"""Streaming Q&A over meeting transcripts.

Allows users to ask specific questions about a completed meeting.
Answers are grounded exclusively in the transcript text — no summary.json
is used as context, ensuring all answers cite what was actually said.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from meeting_scribe.summary import build_transcript_text

logger = logging.getLogger(__name__)

# Per-meeting concurrency guard: only one Q&A query at a time per meeting.
# Prevents vLLM queue flooding from rapid-fire questions.
_in_flight: set[str] = set()

QA_SYSTEM_PROMPT = """\
You are answering questions about a meeting based on its transcript. Follow these rules strictly:

1. Answer using ONLY information from the transcript below. Do not infer, speculate, or add information that is not explicitly stated.
2. Quote speakers by name when attributing statements or actions.
3. Cite approximate timestamps (e.g., "at ~5.2m") when referencing specific parts of the discussion.
4. If the answer is not in the transcript, say so clearly: "This was not discussed in the meeting."
5. Be specific — use exact names, numbers, and details from the transcript.
6. For list-type questions, format your answer as a clear numbered or bulleted list.
7. Keep answers focused and direct. Do not repeat the question back."""


async def ask_meeting_question(
    meeting_dir: Path,
    question: str,
    vllm_url: str = "http://localhost:8010",
    model: str | None = None,
    max_transcript_chars: int = 100_000,
) -> AsyncIterator[str]:
    """Stream answer chunks for a Q&A query about a meeting.

    Yields SSE-formatted lines: 'data: {"type": "chunk", "text": "..."}'
    and a final 'data: {"type": "done", "full_text": "..."}' line.

    Raises:
        ValueError: If meeting has no transcript or a query is already in-flight.
    """
    meeting_id = meeting_dir.name

    if meeting_id in _in_flight:
        yield f'data: {json.dumps({"type": "error", "text": "A question is already being answered for this meeting. Please wait."})}\n\n'
        return

    _in_flight.add(meeting_id)
    try:
        events, transcript = build_transcript_text(meeting_dir, max_transcript_chars)

        if not events:
            yield f'data: {json.dumps({"type": "error", "text": "No transcript available for this meeting."})}\n\n'
            return

        logger.info(
            "Q&A start: meeting=%s, question=%r, transcript_events=%d, transcript_chars=%d",
            meeting_id, question, len(events), len(transcript),
        )

        user_prompt = f"""Meeting transcript:
{transcript}

Question: {question}"""

        async with httpx.AsyncClient(base_url=vllm_url, timeout=120) as client:
            if not model:
                try:
                    resp = await client.get("/v1/models")
                    model = resp.json()["data"][0]["id"]
                except Exception:
                    model = "Qwen/Qwen3.5-35B-A3B-FP8"

            logger.info("Q&A using model=%s, vllm_url=%s", model, vllm_url)

            t0 = time.monotonic()
            full_text = []
            chunk_count = 0

            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "repetition_penalty": 1.15,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "priority": -10,
                },
                timeout=120,
            ) as stream:
                async for line in stream.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        finish_reason = chunk["choices"][0].get("finish_reason")
                        if content:
                            chunk_count += 1
                            full_text.append(content)
                            yield f'data: {json.dumps({"type": "chunk", "text": content})}\n\n'
                        if finish_reason:
                            logger.info("Q&A finish_reason=%s at chunk %d", finish_reason, chunk_count)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

            elapsed_ms = (time.monotonic() - t0) * 1000
            final_text = "".join(full_text)
            logger.info(
                "Q&A answered in %.0fms (%d chunks, %d chars answer, meeting=%s)",
                elapsed_ms,
                chunk_count,
                len(final_text),
                meeting_id,
            )
            logger.info("Q&A full answer for meeting=%s:\n%s", meeting_id, final_text)
            yield f'data: {json.dumps({"type": "done", "full_text": final_text})}\n\n'

    except Exception as e:
        logger.error("Q&A failed for meeting %s: %s", meeting_id, e)
        yield f'data: {json.dumps({"type": "error", "text": f"Q&A failed: {e}"})}\n\n'
    finally:
        _in_flight.discard(meeting_id)
