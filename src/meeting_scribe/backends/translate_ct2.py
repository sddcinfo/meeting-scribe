"""CTranslate2 + NLLB-200 translation backend.

Uses NLLB-200-distilled-600M (int8) for fast bidirectional JA↔EN translation.
CPU-only — does not compete with Whisper for Metal GPU.

Typical latency: ~200ms per sentence on Apple Silicon.
Memory: ~2 GB RSS.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from meeting_scribe.backends.base import TranslateBackend

logger = logging.getLogger(__name__)

# NLLB-200 language codes
_LANG_MAP = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
}

_DEFAULT_MODEL_DIR = Path.home() / ".cache" / "nllb-200-distilled-1.3B-ct2-int8"
_FALLBACK_MODEL_DIR = Path.home() / ".cache" / "nllb-200-distilled-600M-ct2-int8"


class CTranslate2TranslateBackend(TranslateBackend):
    """Translation via CTranslate2 + NLLB-200.

    Uses 1.3B (higher quality) with 600M fallback.
    CPU-only inference, ~500ms per sentence.
    Coexists with Whisper on Metal without contention.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cpu",
        beam_size: int = 4,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._device = device
        self._beam_size = beam_size
        self._translator = None
        self._tokenizer = None

    async def start(self) -> None:
        """Load the CTranslate2 model and tokenizer."""
        import ctranslate2
        from transformers import AutoTokenizer

        # Try configured dir, then fallback
        model_dir = self._model_dir
        if not model_dir.exists():
            model_dir = _FALLBACK_MODEL_DIR
        if not model_dir.exists():
            msg = (
                f"NLLB CT2 model not found. Convert with:\n"
                "  ct2-transformers-converter --model facebook/nllb-200-distilled-1.3B "
                f"--output_dir {_DEFAULT_MODEL_DIR} --quantization int8"
            )
            raise FileNotFoundError(msg)

        # Detect model size from path for tokenizer
        hf_model = (
            "facebook/nllb-200-distilled-1.3B"
            if "1.3B" in str(model_dir)
            else "facebook/nllb-200-distilled-600M"
        )

        t0 = time.monotonic()
        self._translator = ctranslate2.Translator(
            str(model_dir),
            device=self._device,
            compute_type="int8",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_model, src_lang="eng_Latn", local_files_only=True
        )
        load_ms = (time.monotonic() - t0) * 1000
        logger.info("NLLB CT2 loaded in %.0fms from %s", load_ms, model_dir)

    async def stop(self) -> None:
        """Release resources."""
        self._translator = None
        self._tokenizer = None

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text using NLLB-200.

        Args:
            text: Source text.
            source_language: "ja" or "en".
            target_language: "ja" or "en".

        Returns:
            Translated text.
        """
        if self._translator is None or self._tokenizer is None:
            msg = "CTranslate2 translator not initialized"
            raise RuntimeError(msg)

        src_lang = _LANG_MAP.get(source_language)
        tgt_lang = _LANG_MAP.get(target_language)
        if not src_lang or not tgt_lang:
            msg = f"Unsupported language pair: {source_language} → {target_language}"
            raise ValueError(msg)

        t0 = time.monotonic()

        # Set source language on tokenizer
        self._tokenizer.src_lang = src_lang

        # Tokenize
        tokens = self._tokenizer(text, return_tensors=None)["input_ids"]
        token_strs = self._tokenizer.convert_ids_to_tokens(tokens)

        # Translate
        results = self._translator.translate_batch(
            [token_strs],
            target_prefix=[[tgt_lang]],
            beam_size=self._beam_size,
            max_input_length=512,
            max_decoding_length=512,
        )

        # Decode
        output_tokens = results[0].hypotheses[0]
        # Remove the target language prefix token
        if output_tokens and output_tokens[0] == tgt_lang:
            output_tokens = output_tokens[1:]

        translated = self._tokenizer.decode(
            self._tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "Translated [%s→%s] %.0fms: '%s' → '%s'",
            source_language,
            target_language,
            elapsed_ms,
            text[:40],
            translated[:40],
        )

        return translated
