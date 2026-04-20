"""CLI entry point for the sandboxed slides worker container.

Runs the full pipeline: validate -> render originals -> extract text.
Translation happens server-side via the TranslationQueue, so this
container only handles the PPTX-touching stages.

After translation completes, the server invokes this again with --mode=reinsert
to reinsert translated text and render the translated slides.

Usage:
    python -m meeting_scribe.slides.convert_main \\
        --input=/input/source.pptx \\
        --output=/output \\
        --max-slides=100 \\
        --max-output-mb=200

    python -m meeting_scribe.slides.convert_main \\
        --mode=reinsert \\
        --input=/input/source.pptx \\
        --output=/output \\
        --translations=/input/translations.json \\
        --max-output-mb=200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from meeting_scribe.slides.convert import (
    MAX_OUTPUT_MB,
    MAX_SLIDES,
    convert_pptx_to_images,
    extract_text_from_pptx,
    reinsert_translated_text,
    render_translated_to_images,
    validate_pptx_contents,
    write_text_extract,
)
from meeting_scribe.slides.models import SlideMeta, StageProgress, StageStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _write_meta(meta: SlideMeta, output_dir: Path) -> None:
    """Write meta.json atomically."""
    tmp = output_dir / "meta.json.tmp"
    target = output_dir / "meta.json"
    tmp.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2))
    tmp.rename(target)


def _run_extract(args: argparse.Namespace) -> int:
    """Run validate -> render originals -> extract text."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = SlideMeta(deck_id="worker")  # deck_id set by job.py on the server side

    # Stage 1: Validate
    meta.stage = "validating"
    meta.stages["validating"].status = StageStatus.IN_PROGRESS
    _write_meta(meta, output_dir)

    result = validate_pptx_contents(input_path)
    if not result.valid:
        meta.stages["validating"] = StageProgress(
            status=StageStatus.FAILED,
            error=result.error,
        )
        meta.error = result.error
        _write_meta(meta, output_dir)
        logger.error("Validation failed: %s", result.error)
        return 1

    meta.total_slides = result.slide_count
    meta.stages["validating"].status = StageStatus.DONE
    _write_meta(meta, output_dir)
    logger.info("Validation passed: %d slides", result.slide_count)

    # Stage 2: Render originals
    meta.stage = "rendering_original"
    meta.stages["rendering_original"].status = StageStatus.IN_PROGRESS
    _write_meta(meta, output_dir)

    original_dir = output_dir / "original"
    original_dir.mkdir(exist_ok=True)

    try:
        count = convert_pptx_to_images(input_path, original_dir)
        meta.total_slides = count
        meta.stages["rendering_original"] = StageProgress(
            status=StageStatus.DONE,
            progress=f"{count}/{count}",
        )
        _write_meta(meta, output_dir)
        logger.info("Rendered %d original slides", count)
    except Exception as exc:
        meta.stages["rendering_original"] = StageProgress(
            status=StageStatus.FAILED,
            error=str(exc),
        )
        meta.error = str(exc)
        _write_meta(meta, output_dir)
        logger.error("Rendering failed: %s", exc)
        return 1

    # Stage 3: Extract text
    meta.stage = "extracting_text"
    meta.stages["extracting_text"].status = StageStatus.IN_PROGRESS
    _write_meta(meta, output_dir)

    try:
        slides = extract_text_from_pptx(input_path)
        write_text_extract(slides, output_dir / "text_extract.json")
        meta.stages["extracting_text"].status = StageStatus.DONE
        _write_meta(meta, output_dir)

        total_runs = sum(len(s.runs) for s in slides)
        logger.info("Extracted %d text runs from %d slides", total_runs, len(slides))
    except Exception as exc:
        meta.stages["extracting_text"] = StageProgress(
            status=StageStatus.FAILED,
            error=str(exc),
        )
        meta.error = str(exc)
        _write_meta(meta, output_dir)
        logger.error("Text extraction failed: %s", exc)
        return 1

    return 0


def _run_reinsert(args: argparse.Namespace) -> int:
    """Run reinsert translated text -> render translated slides."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    translations_path = Path(args.translations)

    # Read translations
    translated_runs = json.loads(translations_path.read_text())

    # Read current meta
    meta_path = output_dir / "meta.json"
    if meta_path.exists():
        meta_dict = json.loads(meta_path.read_text())
        meta = SlideMeta(deck_id=meta_dict.get("deck_id", "worker"))
        meta.total_slides = meta_dict.get("total_slides", 0)
        meta.source_lang = meta_dict.get("source_lang", "")
        meta.target_lang = meta_dict.get("target_lang", "")
        meta.started_at = meta_dict.get("started_at", "")
        # Restore completed stage statuses
        for stage_name in ("validating", "rendering_original", "extracting_text", "translating"):
            if stage_name in meta_dict.get("stages", {}):
                s = meta_dict["stages"][stage_name]
                meta.stages[stage_name] = StageProgress(
                    status=StageStatus(s["status"]),
                    progress=s.get("progress"),
                )
    else:
        meta = SlideMeta(deck_id="worker")

    # Stage 5: Reinsert
    meta.stage = "reinserting"
    meta.stages["reinserting"].status = StageStatus.IN_PROGRESS
    _write_meta(meta, output_dir)

    translated_pptx = output_dir / "translated.pptx"
    try:
        reinsert_translated_text(input_path, translated_runs, translated_pptx)
        meta.stages["reinserting"].status = StageStatus.DONE
        _write_meta(meta, output_dir)
        logger.info("Reinserted translations into PPTX")
    except Exception as exc:
        meta.stages["reinserting"] = StageProgress(
            status=StageStatus.FAILED,
            error=str(exc),
        )
        meta.error = str(exc)
        _write_meta(meta, output_dir)
        logger.error("Reinsertion failed: %s", exc)
        return 1

    # Stage 6: Render translated
    meta.stage = "rendering_translated"
    meta.stages["rendering_translated"].status = StageStatus.IN_PROGRESS
    _write_meta(meta, output_dir)

    translated_dir = output_dir / "translated"
    translated_dir.mkdir(exist_ok=True)

    try:
        count = render_translated_to_images(translated_pptx, translated_dir)
        meta.stages["rendering_translated"] = StageProgress(
            status=StageStatus.DONE,
            progress=f"{count}/{count}",
        )
        from datetime import UTC, datetime

        meta.completed_at = datetime.now(UTC).isoformat()
        meta.stage = "complete"
        _write_meta(meta, output_dir)
        logger.info("Rendered %d translated slides", count)
    except Exception as exc:
        meta.stages["rendering_translated"] = StageProgress(
            status=StageStatus.FAILED,
            error=str(exc),
        )
        meta.error = str(exc)
        _write_meta(meta, output_dir)
        logger.error("Translated rendering failed: %s", exc)
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Slides worker — sandboxed PPTX processing")
    parser.add_argument("--input", required=True, help="Path to source .pptx")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        default="extract",
        choices=["extract", "reinsert"],
        help="extract: validate+render+extract; reinsert: reinsert+render",
    )
    parser.add_argument("--translations", help="Path to translations JSON (reinsert mode)")
    parser.add_argument("--max-slides", type=int, default=MAX_SLIDES)
    parser.add_argument("--max-output-mb", type=int, default=MAX_OUTPUT_MB)
    args = parser.parse_args()

    # Apply overrides
    import meeting_scribe.slides.convert as conv

    conv.MAX_SLIDES = args.max_slides
    conv.MAX_OUTPUT_MB = args.max_output_mb

    if args.mode == "extract":
        rc = _run_extract(args)
    elif args.mode == "reinsert":
        if not args.translations:
            parser.error("--translations required for reinsert mode")
        rc = _run_reinsert(args)
    else:
        rc = 1

    sys.exit(rc)


if __name__ == "__main__":
    main()
