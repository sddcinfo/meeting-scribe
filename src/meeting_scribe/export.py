"""Meeting export utilities.

Generates downloadable exports in various formats:
- Markdown (summary + transcript)
- Plain text (filtered by language)
- ZIP archive (all meeting artifacts)
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _format_timestamp(ms: int) -> str:
    """Format milliseconds as HH:MM:SS."""
    total_s = ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _load_events_with_corrections(journal_path: Path) -> list[dict]:
    """Load journal events, dedup per segment_id, apply speaker corrections.

    Under multi-target fan-out the journal contains multiple translation
    lines per segment (one per target language). This loader returns ONE
    dict per segment_id with all translations merged into a
    ``translations`` dict keyed by ``target_language``. The single
    ``translation`` slot is preserved (last-done wins) so callers that
    know about only one translation still work. Events are emitted in
    first-seen order to preserve meeting chronology.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []
    corrections: dict[str, str] = {}  # segment_id → speaker_name

    for raw in journal_path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if entry.get("type") == "speaker_correction":
            corrections[entry["segment_id"]] = entry["speaker_name"]
            continue
        if not (entry.get("is_final") and entry.get("text")):
            continue

        sid = entry.get("segment_id")
        if not sid:
            continue

        existing = merged.get(sid)
        if existing is None:
            # First sighting — seed with a shallow copy and init translations map.
            seed = dict(entry)
            seed["translations"] = {}
            tr = entry.get("translation") or {}
            tlang = tr.get("target_language") or ""
            if tlang and tr.get("text"):
                seed["translations"][tlang] = tr
            merged[sid] = seed
            order.append(sid)
        else:
            # Higher-rev ASR text replaces source; later translations fold in.
            if entry.get("revision", 0) >= existing.get("revision", 0):
                existing["text"] = entry.get("text", existing.get("text", ""))
                existing["revision"] = entry.get("revision", existing.get("revision", 0))
                existing["language"] = entry.get("language", existing.get("language"))
                if entry.get("speakers"):
                    existing["speakers"] = entry["speakers"]
            tr = entry.get("translation") or {}
            tlang = tr.get("target_language") or ""
            if tlang and tr.get("text"):
                existing["translations"][tlang] = tr
                # Keep the flat .translation slot populated with the last
                # non-empty done translation for single-target consumers.
                existing["translation"] = tr

    # Apply speaker corrections
    for sid, event in merged.items():
        if sid in corrections:
            new_name = corrections[sid]
            speakers = event.get("speakers", [])
            if speakers:
                speakers[0]["identity"] = new_name
                speakers[0]["display_name"] = new_name

    return [merged[sid] for sid in order]


def transcript_to_text(journal_path: Path, lang: str | None = None) -> str:
    """Convert journal.jsonl to plain text, optionally filtered by language.

    Args:
        journal_path: Path to journal.jsonl
        lang: "en" or "ja" to filter. None for all languages.

    Returns:
        Plain text transcript with speaker attribution and timestamps.
    """
    events = _load_events_with_corrections(journal_path)
    lines = []

    for e in events:
        event_lang = e.get("language", "unknown")
        text = e.get("text", "")
        translations = e.get("translations") or {}
        # Back-compat: legacy entries only have the flat .translation slot.
        if not translations:
            legacy = e.get("translation") or {}
            tlang = legacy.get("target_language") or ""
            if tlang and legacy.get("text"):
                translations = {tlang: legacy}

        # Determine what to include based on language filter
        if lang:
            if event_lang == lang:
                output_text = text
            elif lang in translations and translations[lang].get("text"):
                output_text = translations[lang]["text"]
            else:
                continue
        else:
            output_text = text
            for tlang in sorted(translations):
                trans_text = translations[tlang].get("text")
                if trans_text:
                    output_text += f" [{tlang}: {trans_text}]"

        # Format with speaker and timestamp
        speakers = e.get("speakers", [])
        if speakers:
            speaker = (
                speakers[0].get("identity")
                or speakers[0].get("display_name")
                or f"Speaker {speakers[0].get('cluster_id', '?')}"
            )
        else:
            speaker = "Unknown"

        timestamp = _format_timestamp(e.get("start_ms", 0))
        lines.append(f"[{timestamp}] {speaker}: {output_text}")

    return "\n".join(lines)


def meeting_to_markdown(meeting_dir: Path) -> str:
    """Generate a Markdown export with summary and transcript."""
    lines = []

    # Header
    meta_path = meeting_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        lines.append(f"# Meeting {meta.get('meeting_id', 'Unknown')}")
        lines.append("")
        lines.append(f"**Date:** {meta.get('created_at', 'Unknown')}")
        # Calculate duration from audio if available
        pcm_path = meeting_dir / "audio" / "recording.pcm"
        if pcm_path.exists():
            dur_s = pcm_path.stat().st_size / (SAMPLE_RATE * 2)
            lines.append(f"**Duration:** {_format_timestamp(int(dur_s * 1000))}")
        lines.append("")

    # Summary
    summary_path = meeting_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        is_v2 = bool(summary.get("key_insights"))

        if summary.get("executive_summary"):
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(summary["executive_summary"])
            lines.append("")

        if is_v2 and summary.get("named_entities"):
            entities = summary["named_entities"]
            has_any = any(entities.get(k) for k in ("companies", "people", "products", "accounts"))
            if has_any:
                lines.append("## Named Entities")
                lines.append("")
                for category in ("companies", "people", "products", "accounts"):
                    items = entities.get(category, [])
                    if items:
                        label = category.capitalize()
                        lines.append(f"**{label}:** {', '.join(items)}")
                lines.append("")

        if is_v2 and summary.get("key_insights"):
            lines.append("## Key Insights")
            lines.append("")
            for i, insight in enumerate(summary["key_insights"], 1):
                speakers = insight.get("speakers", [])
                speaker_str = f" ({', '.join(speakers)})" if speakers else ""
                lines.append(f"### {i}. {insight['title']}{speaker_str}")
                lines.append("")
                lines.append(insight.get("description", ""))
                lines.append("")

        if summary.get("action_items"):
            lines.append("## Action Items")
            lines.append("")
            if is_v2:
                # Group by category
                by_cat: dict[str, list] = {}
                for a in summary["action_items"]:
                    cat = a.get("category", "General")
                    by_cat.setdefault(cat, []).append(a)
                for cat, items in by_cat.items():
                    lines.append(f"### {cat}")
                    lines.append("")
                    for a in items:
                        assignee = f" @{a['assignee']}" if a.get("assignee") else ""
                        due = f" (due: {a['due']})" if a.get("due") else ""
                        lines.append(f"- [ ] {a['task']}{assignee}{due}")
                    lines.append("")
            else:
                for a in summary["action_items"]:
                    assignee = f" @{a['assignee']}" if a.get("assignee") else ""
                    due = f" (due: {a['due']})" if a.get("due") else ""
                    lines.append(f"- [ ] {a['task']}{assignee}{due}")
                lines.append("")

        if summary.get("decisions"):
            lines.append("## Decisions")
            lines.append("")
            for d in summary["decisions"]:
                lines.append(f"- {d}")
            lines.append("")

        if summary.get("key_quotes"):
            lines.append("## Key Quotes")
            lines.append("")
            for q in summary["key_quotes"]:
                if isinstance(q, str):
                    lines.append(f'> "{q}"')
                else:
                    speaker = q.get("speaker", "")
                    context = q.get("context", "")
                    attr = f" — {speaker}" if speaker else ""
                    if context:
                        attr += f" ({context})"
                    lines.append(f'> "{q.get("text", "")}"{attr}')
                lines.append("")

        if summary.get("topics"):
            lines.append("## Topics")
            lines.append("")
            for t in summary["topics"]:
                lines.append(f"- **{t['title']}**: {t.get('description', '')}")
            lines.append("")

        if summary.get("speaker_stats"):
            lines.append("## Speaker Statistics")
            lines.append("")
            lines.append("| Speaker | Segments | Speaking Time | % |")
            lines.append("|---------|----------|-------------|---|")
            for s in summary["speaker_stats"]:
                lines.append(
                    f"| {s['name']} | {s['segments']} | {s['speaking_seconds']}s | {s['pct']}% |"
                )
            lines.append("")

    # Transcript
    journal_path = meeting_dir / "journal.jsonl"
    if journal_path.exists():
        lines.append("## Full Transcript")
        lines.append("")
        lines.append(transcript_to_text(journal_path))

    return "\n".join(lines)


def meeting_to_zip(meeting_dir: Path) -> bytes:
    """Create a ZIP archive of all meeting artifacts."""
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        meeting_id = meeting_dir.name

        # JSON files
        for name in [
            "meta.json",
            "room.json",
            "journal.jsonl",
            "timeline.json",
            "summary.json",
            "polished.json",
            "detected_speakers.json",
            "speaker_lanes.json",
        ]:
            path = meeting_dir / name
            if path.exists():
                zf.writestr(f"{meeting_id}/{name}", path.read_text())

        # Audio as WAV (convert from PCM)
        pcm_path = meeting_dir / "audio" / "recording.pcm"
        if pcm_path.exists():
            try:
                import numpy as np

                raw = np.frombuffer(pcm_path.read_bytes(), dtype=np.int16)
                audio = raw.astype(np.float32) / 32768.0
                wav_buf = io.BytesIO()
                sf.write(wav_buf, audio, SAMPLE_RATE, format="WAV")
                zf.writestr(f"{meeting_id}/recording.wav", wav_buf.getvalue())
            except Exception as e:
                logger.warning("Failed to convert PCM to WAV for ZIP: %s", e)
                # Include raw PCM as fallback
                zf.write(str(pcm_path), f"{meeting_id}/recording.pcm")

        # Markdown summary
        md = meeting_to_markdown(meeting_dir)
        zf.writestr(f"{meeting_id}/meeting-notes.md", md)

    return buf.getvalue()
