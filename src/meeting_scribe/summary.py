"""AI-powered meeting summary generation via Qwen3.5.

Reads the meeting transcript from journal.jsonl, sends a structured prompt
to the vLLM translation endpoint (Qwen3.5-35B), and generates a comprehensive
meeting summary with key insights, named entities, categorized action items,
and speaker statistics.

Output saved as summary.json in the meeting directory.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ── Shared transcript builder ────────────────────────────────────────────

# Conservative chars-per-token estimate for mixed CJK/English transcripts.
# CJK ≈ 1-2 chars/token, English ≈ 4 chars/token. 3 is a safe middle.
_CHARS_PER_TOKEN = 3

# Token budget: 128K context − 3K prompt − 8K output = 117K for transcript.
# At 3 chars/token → 100K chars max.
_DEFAULT_MAX_TRANSCRIPT_CHARS = 100_000


def build_transcript_text(
    meeting_dir: Path,
    max_chars: int = _DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> tuple[list[dict], str]:
    """Load and format the meeting transcript from journal.jsonl.

    Returns:
        Tuple of (deduplicated events sorted chronologically, formatted transcript text).
        The transcript text is truncated to max_chars using middle-truncation
        to preserve both the opening and closing of the meeting.
    """
    journal_path = meeting_dir / "journal.jsonl"
    speakers_path = meeting_dir / "detected_speakers.json"

    if not journal_path.exists():
        return [], ""

    # Load transcript events (finals only). Journals are append-only —
    # the same segment_id shows up multiple times as revisions attach
    # diarization / speaker corrections / remapped cluster_ids. We want
    # ONLY the highest revision per segment so downstream speaker stats
    # reflect the final post-finalize state.
    best: dict[str, dict] = {}
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not e.get("is_final") or not e.get("text"):
            continue
        sid = e.get("segment_id")
        if not sid:
            continue
        rev = e.get("revision", 0)
        if sid not in best or rev > best[sid].get("revision", 0):
            best[sid] = e
    events = sorted(best.values(), key=lambda e: e.get("start_ms", 0))

    if not events:
        return [], ""

    # Load speaker info for fallback names
    speakers: list[dict] = []
    if speakers_path.exists():
        with contextlib.suppress(json.JSONDecodeError):
            speakers = json.loads(speakers_path.read_text())

    speaker_stats = _calculate_speaker_stats(events, speakers)
    default_speaker = speaker_stats[0]["name"] if speaker_stats else "Speaker 1"

    # Build transcript lines
    transcript_lines = []
    for e in events:
        sp = e.get("speakers", [])
        speaker_name = (
            sp[0].get(
                "identity", sp[0].get("display_name", f"Speaker {sp[0].get('cluster_id', '?')}")
            )
            if sp
            else default_speaker
        )
        time_min = e.get("start_ms", 0) / 60000
        lang = e.get("language", "?")
        text = e.get("text", "")
        transcript_lines.append(f"[{time_min:.1f}m] [{lang.upper()}] {speaker_name}: {text}")

    transcript = "\n".join(transcript_lines)

    # Middle-truncation: keep first 40% + last 40% to preserve both
    # opening context and closing wrap-up (where action items live).
    if len(transcript) > max_chars:
        logger.warning(
            "Transcript truncated: %d chars → %d chars (middle-truncation)",
            len(transcript),
            max_chars,
        )
        head_size = int(max_chars * 0.4)
        tail_size = int(max_chars * 0.4)
        transcript = (
            transcript[:head_size]
            + "\n\n... (transcript middle section omitted for length) ...\n\n"
            + transcript[-tail_size:]
        )

    return events, transcript


# ── Summary prompt ────────────────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """\
You are a senior executive assistant who has sat in thousands of board meetings, \
strategy sessions, and cross-functional reviews. You produce meeting intelligence \
that executives actually read — not generic bullet points, but the kind of \
structured analysis that reveals what really happened, what was decided, and \
what needs to happen next.

You will receive a timestamped, speaker-attributed meeting transcript. Analyze it \
thoroughly and produce a JSON response with the following structure. Every field \
is required (use empty arrays/objects if nothing applies).

{
  "schema_version": 2,

  "executive_summary": "A rich 3-5 sentence overview that captures the meeting's \
purpose, key outcomes, and strategic significance. Not a list of topics — a \
narrative that an executive could read in 30 seconds and understand what happened.",

  "key_insights": [
    {
      "title": "Thematic title for this insight",
      "description": "Multi-paragraph analysis. Include specific examples, company \
names, account names, and proof points mentioned in the meeting. Provide context \
on why this matters and what the implications are. Each insight should be 2-4 \
paragraphs of substantive analysis, not a single sentence.",
      "speakers": ["Names of speakers who contributed to this insight"]
    }
  ],

  "action_items": [
    {
      "category": "A thematic category grouping related actions (e.g., 'Account \
Strategy', 'Internal Enablement', 'AI & Tooling', 'Headcount & Roles', \
'Governance & Process')",
      "task": "Detailed description of what needs to be done, with enough context \
that someone reading this weeks later would understand the action",
      "assignee": "Person name if explicitly assigned, or null",
      "due": "Date if mentioned, or null"
    }
  ],

  "named_entities": {
    "companies": ["Every company, organization, or system integrator mentioned"],
    "people": ["Every person mentioned by name"],
    "products": ["Every product, platform, technology, or tool mentioned"],
    "accounts": ["Every customer account or prospect mentioned"]
  },

  "decisions": ["Explicit decisions that were made during the meeting"],

  "questions": ["Unresolved or important questions that were raised but not answered"],

  "key_quotes": [
    {
      "text": "The exact or near-exact quote",
      "speaker": "Who said it",
      "context": "Brief context for why this quote matters"
    }
  ],

  "topics": [
    {
      "title": "Topic name",
      "description": "What was discussed under this topic",
      "timestamp_min": 0.0
    }
  ]
}

Rules:
- Extract EVERY named entity mentioned — companies, people, products, accounts. \
Do not omit any.
- Group action items by strategic theme, not chronologically. Each action item \
should have enough context to stand alone.
- Key insights should be substantive multi-paragraph analyses, not summaries. \
Include specific examples, numbers, and named entities from the transcript.
- For multilingual meetings, analyze content in all languages. Report in English.
- Only assign an action item to someone if they were explicitly asked or \
volunteered to do it.
- If no decisions, action items, or other fields exist, use empty arrays.
- Topics should be in chronological order with approximate timestamps.
- Respond ONLY with valid JSON. No markdown, no explanation, no preamble."""

SUMMARY_USER_PROMPT = """\
Meeting details:
- Duration: {duration_min:.1f} minutes
- Speakers: {num_speakers} ({speaker_names})
- Languages: {languages}
- Total segments: {num_segments}

Transcript (speaker-attributed, chronological):
{transcript}"""


# ── Summary generation ────────────────────────────────────────────────────


async def _call_vllm_summary(
    vllm_url: str,
    user_prompt: str,
    model: str | None = None,
    priority: int = -10,
    system_prompt: str | None = None,
    max_tokens: int = 8192,
    enable_thinking: bool = True,
    timeout: float = 180,
) -> tuple[dict | str | None, str | None, float, str]:
    """Low-level vLLM call for summary generation.

    Returns:
        (parsed_summary_or_raw_text, error_string, elapsed_ms, model_used)
    """
    async with httpx.AsyncClient(base_url=vllm_url, timeout=timeout) as client:
        if not model:
            try:
                resp = await client.get("/v1/models")
                model = resp.json()["data"][0]["id"]
            except Exception:
                model = "Qwen/Qwen3.5-35B-A3B-FP8"

        t0 = time.monotonic()
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt or SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    "priority": priority,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            return None, f"LLM call failed: {e}", elapsed_ms, model

    elapsed_ms = (time.monotonic() - t0) * 1000

    # For chunk summaries (plain text), return raw text
    if system_prompt and system_prompt != SUMMARY_SYSTEM_PROMPT:
        return raw.strip(), None, elapsed_ms, model

    # For final summary, parse JSON
    summary = _parse_json_response(raw)
    if not summary:
        return None, "Failed to parse summary JSON", elapsed_ms, model

    return summary, None, elapsed_ms, model


def _build_user_prompt(
    events: list[dict],
    transcript: str,
    speakers: list[dict] | None = None,
) -> str:
    """Build the user prompt from events and transcript text."""
    speaker_stats = _calculate_speaker_stats(events, speakers or [])
    speaker_names = ", ".join(s["name"] for s in speaker_stats) if speaker_stats else "Unknown"

    duration_ms = max(e.get("end_ms", 0) for e in events) if events else 0
    languages = set(e.get("language", "unknown") for e in events)

    return SUMMARY_USER_PROMPT.format(
        duration_min=duration_ms / 60000,
        num_speakers=len(speaker_stats)
        or len(
            set(
                e.get("speakers", [{}])[0].get("cluster_id", 0) for e in events if e.get("speakers")
            )
        ),
        speaker_names=speaker_names,
        languages=", ".join(sorted(languages - {"unknown"})) or "auto-detected",
        num_segments=len(events),
        transcript=transcript,
    )


_CHUNK_SUMMARY_PROMPT = """\
Summarize this section of a meeting transcript concisely. Include:
- Key topics discussed
- Decisions made
- Action items mentioned
- Important quotes

Return a plain text summary, 200-400 words. No JSON."""


async def _summarize_chunks(
    events: list[dict],
    transcript: str,
    vllm_url: str,
    model: str | None,
    speakers: list[dict],
    priority: int,
) -> tuple[str, str | None]:
    """For large transcripts, split into chunks, summarize each, then merge.

    Returns (merged_transcript_for_final_summary, model_used).
    """
    # Split transcript into ~30K char chunks (fits easily in context)
    chunk_size = 30_000
    chunks = []
    lines = transcript.split("\n")
    current_chunk: list[str] = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(line)
        current_len += len(line) + 1
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    if len(chunks) <= 1:
        return transcript, None  # No chunking needed

    logger.info("Chunked summary: %d chunks of ~%dK chars each", len(chunks), chunk_size // 1000)

    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info("Summarizing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        result, error, _elapsed_ms, used_model = await _call_vllm_summary(
            vllm_url,
            f"Meeting transcript section {i + 1} of {len(chunks)}:\n\n{chunk}",
            model=model,
            priority=priority,
            system_prompt=_CHUNK_SUMMARY_PROMPT,
            max_tokens=1024,
            enable_thinking=False,
            timeout=60,
        )
        if error:
            # Use raw chunk text as fallback
            chunk_summaries.append(f"[Section {i + 1} summary unavailable]\n{chunk[:500]}...")
            logger.warning("Chunk %d summary failed: %s", i + 1, error)
        else:
            # result is a dict from JSON parse — but we asked for plain text
            # The raw response is what we want
            chunk_summaries.append(
                f"=== Section {i + 1} ===\n{result if isinstance(result, str) else json.dumps(result)}"
            )
        model = used_model or model

    # Merge chunk summaries into a condensed transcript for the final summary
    merged = "\n\n".join(chunk_summaries)
    # Cap at 50K chars for the final summary prompt
    if len(merged) > 50_000:
        merged = merged[:25_000] + "\n\n... (sections omitted) ...\n\n" + merged[-25_000:]

    return merged, model


async def generate_summary(
    meeting_dir: Path,
    vllm_url: str = "http://localhost:8010",
    model: str | None = None,
    max_transcript_chars: int = _DEFAULT_MAX_TRANSCRIPT_CHARS,
    priority: int = -10,
) -> dict:
    """Generate an AI meeting summary from the transcript.

    For large transcripts (> 50K chars), uses chunked summarization:
    split into ~30K char chunks, summarize each, then generate the
    final structured summary from the chunk summaries.
    """
    events, transcript = build_transcript_text(meeting_dir, max_transcript_chars)

    if not events:
        return {"error": "No transcript segments"}

    # Speaker stats
    speakers_path = meeting_dir / "detected_speakers.json"
    speakers: list[dict] = []
    if speakers_path.exists():
        with contextlib.suppress(json.JSONDecodeError):
            speakers = json.loads(speakers_path.read_text())

    # For large transcripts, chunk-summarize first to fit in context
    final_transcript = transcript
    if len(transcript) > 50_000:
        logger.info(
            "Large transcript (%d chars, %d events) — using chunked summarization",
            len(transcript),
            len(events),
        )
        final_transcript, chunk_model = await _summarize_chunks(
            events,
            transcript,
            vllm_url,
            model,
            speakers,
            priority,
        )
        if chunk_model:
            model = chunk_model

    user_prompt = _build_user_prompt(events, final_transcript, speakers)
    summary, error, elapsed_ms, model = await _call_vllm_summary(
        vllm_url,
        user_prompt,
        model=model,
        priority=priority,
    )

    if error:
        logger.error("Summary generation failed: %s", error)
        return {"error": error}

    logger.info("Summary generated in %.0fms (%d chars prompt)", elapsed_ms, len(user_prompt))

    # _call_vllm_summary returns (dict | str | None, ...); the error path
    # above has caught str + None cases, so summary is a dict from here.
    assert isinstance(summary, dict)

    # Ensure schema_version is set
    summary["schema_version"] = 2

    # Add speaker stats and metadata
    speaker_stats = _calculate_speaker_stats(events, speakers)
    duration_ms = max(e.get("end_ms", 0) for e in events) if events else 0
    languages = set(e.get("language", "unknown") for e in events)
    summary["speaker_stats"] = speaker_stats
    summary["metadata"] = {
        "meeting_id": meeting_dir.name,
        "duration_min": round(duration_ms / 60000, 1),
        "num_segments": len(events),
        "num_speakers": len(speaker_stats),
        "languages": sorted(languages - {"unknown"}),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generation_ms": round(elapsed_ms),
        "model": model,
    }

    # Save to disk
    summary_path = meeting_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    logger.info("Summary saved: %s", summary_path)

    return summary


async def generate_draft_summary(
    meeting_dir: Path,
    vllm_url: str = "http://localhost:8010",
    model: str | None = None,
    max_transcript_chars: int = _DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> tuple[dict | None, int]:
    """Generate a draft summary during live recording (low priority).

    Uses priority 5 so live translation (priority -10) always wins on
    the shared vLLM instance. The draft is NOT saved to disk — the caller
    caches it in memory and decides whether to promote it at stop time.

    Returns:
        (summary_dict_or_None, event_count)
    """
    events, transcript = build_transcript_text(meeting_dir, max_transcript_chars)
    if not events:
        return None, 0

    user_prompt = _build_user_prompt(events, transcript)
    summary, error, elapsed_ms, model = await _call_vllm_summary(
        vllm_url,
        user_prompt,
        model=model,
        priority=5,
    )

    if error:
        logger.warning("Draft summary failed (non-fatal): %s", error)
        return None, len(events)

    logger.info(
        "Draft summary generated in %.0fms (%d events, %d chars prompt)",
        elapsed_ms,
        len(events),
        len(user_prompt),
    )

    # Error path above handled the str/None cases; summary is dict here.
    assert isinstance(summary, dict)

    # Attach minimal metadata for cache-hit decisions at stop time
    summary["schema_version"] = 2
    speaker_stats = _calculate_speaker_stats(events, [])
    duration_ms = max(e.get("end_ms", 0) for e in events) if events else 0
    languages = set(e.get("language", "unknown") for e in events)
    summary["speaker_stats"] = speaker_stats
    summary["metadata"] = {
        "meeting_id": meeting_dir.name,
        "duration_min": round(duration_ms / 60000, 1),
        "num_segments": len(events),
        "num_speakers": len(speaker_stats),
        "languages": sorted(languages - {"unknown"}),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generation_ms": round(elapsed_ms),
        "model": model,
        "is_draft": True,
    }

    return summary, len(events)


# ── Helpers ───────────────────────────────────────────────────────────────


def _calculate_speaker_stats(events: list[dict], speakers: list[dict]) -> list[dict]:
    """Calculate per-speaker statistics from transcript events."""
    stats: dict[str, dict] = {}

    for e in events:
        sp = e.get("speakers", [])
        if not sp:
            continue
        s = sp[0]
        cluster_id = s.get("cluster_id", 0)
        name = s.get("identity") or s.get("display_name") or f"Speaker {cluster_id}"

        if name not in stats:
            stats[name] = {"name": name, "cluster_id": cluster_id, "segments": 0, "speaking_ms": 0}

        stats[name]["segments"] += 1
        duration = e.get("end_ms", 0) - e.get("start_ms", 0)
        stats[name]["speaking_ms"] += max(0, duration)

    # Calculate percentages
    total_ms = sum(s["speaking_ms"] for s in stats.values())
    result = []
    for s in sorted(stats.values(), key=lambda x: -x["speaking_ms"]):
        result.append(
            {
                "name": s["name"],
                "segments": s["segments"],
                "speaking_seconds": round(s["speaking_ms"] / 1000),
                "pct": round(s["speaking_ms"] / total_ms * 100, 1) if total_ms > 0 else 0,
            }
        )

    return result


def _parse_json_response(raw: str) -> dict | None:
    """Extract JSON from LLM response, handling thinking tags, markdown code blocks."""
    raw = raw.strip()

    # Strip Qwen3.5 thinking tags
    import re

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Also strip "Thinking Process:" preambles
    if "Thinking Process:" in raw:
        idx = raw.rfind("}")
        if idx >= 0:
            # Find the JSON portion after the thinking
            json_start = raw.find("{")
            if json_start >= 0:
                raw = raw[json_start:]

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```" in raw:
        start = raw.find("```json")
        if start >= 0:
            start = raw.find("\n", start) + 1
        else:
            start = raw.find("```") + 3
            start = raw.find("\n", start) + 1
        end = raw.rfind("```")
        if start > 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

    # Try finding first { to last }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None
