"""Top-level HTML view routes + the standalone voice-clone demo POST.

Plain-old page renders that hand pre-cached HTML from
``server_support.page_cache._HTML`` to ``HTMLResponse``. The single
``/api/demo/voice-clone`` POST is grouped here because it backs
``/demo/voice-clone`` and shares no state with the live meeting
pipeline — it does its own ASR → translate → TTS round-trip against
``state.translate_backend`` / ``state.tts_backend`` and is independent
of any active meeting.

The ``index`` route is the one slightly load-bearing handler: it
inspects ``_is_guest_scope(request)`` to decide whether to serve the
portal/guest pages or the admin index, and ACKs hotspot clients so
captive-portal probes flip to "online".
"""

from __future__ import annotations

import base64
import io
import logging

import fastapi
import httpx
import numpy as np
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.captive_ack import _captive_ack
from meeting_scribe.server_support.page_cache import _HTML
from meeting_scribe.server_support.request_scope import _is_guest_scope

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def index(request: fastapi.Request) -> HTMLResponse:
    if _is_guest_scope(request):
        # Any hit on the portal root from a hotspot client ACKs them
        # so subsequent captive-portal probes return "not captive" and
        # the CNA sheet (iOS) / sign-in notification (Android) dismisses.
        # Without this the CNA stays open forever because iOS keeps
        # polling hotspot-detect.html and never sees the Success body.
        _captive_ack(request)
        if request.cookies.get("scribe_portal") == "done":
            return HTMLResponse(_HTML.get("guest", ""))
        return HTMLResponse(_HTML.get("portal", ""))
    return HTMLResponse(_HTML.get("index", ""))


@router.get("/reader")
async def reader_view(request: fastapi.Request) -> HTMLResponse:
    """Large-font, text-only reader view for guest displays (iPad, TV)."""
    return HTMLResponse(_HTML.get("reader", ""))


@router.get("/demo")
@router.get("/demo/")
async def demo_landing() -> HTMLResponse:
    """Demo landing page — links to all interactive demos."""
    return HTMLResponse(_HTML.get("demo", ""))


@router.get("/demo/guest")
async def demo_guest_page() -> HTMLResponse:
    """Preview of the guest view from the admin side."""
    return HTMLResponse(_HTML.get("guest", ""))


@router.get("/demo/voice-clone")
async def demo_voice_clone_page() -> HTMLResponse:
    """Standalone voice clone demo — own mic, own pipeline."""
    return HTMLResponse(_HTML.get("voice-clone", ""))


@router.post("/api/demo/voice-clone")
async def demo_voice_clone(request: fastapi.Request) -> JSONResponse:
    """Standalone voice clone: voice reference + text → translate → TTS.

    JSON body:
      - audio_b64: base64 s16le 16kHz mono PCM (voice reference)
      - text: text to translate and speak (if empty, ASR transcribes the audio)
      - target_language: BCP-47 target language code

    Returns JSON with original_text, translated_text, audio_b64 (s16le PCM),
    source_language, target_language, sample_rate.
    Completely independent of any running meeting.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    audio_b64 = body.get("audio_b64", "")
    text = body.get("text", "").strip()
    tgt_lang = body.get("target_language", "en")

    if not audio_b64:
        return JSONResponse({"error": "audio_b64 is required"}, status_code=400)

    # Decode voice reference from base64 s16le PCM
    try:
        raw = base64.b64decode(audio_b64)
        pcm = np.frombuffer(raw, dtype=np.int16)
        voice_f32 = pcm.astype(np.float32) / 32768.0
    except Exception as e:
        return JSONResponse({"error": f"Could not decode audio: {e}"}, status_code=400)

    if len(voice_f32) < 8000:
        return JSONResponse(
            {"error": "Voice reference too short (need at least 0.5s)"}, status_code=400
        )

    # If no text provided, transcribe the voice reference via ASR
    src_lang = None
    if not text:
        import soundfile as sf  # type: ignore[import-untyped]

        wav_buf = io.BytesIO()
        sf.write(wav_buf, voice_f32, 16000, format="WAV")
        asr_b64 = base64.b64encode(wav_buf.getvalue()).decode()

        try:
            async with httpx.AsyncClient(timeout=30) as c:
                asr_resp = await c.post(
                    f"{state.config.asr_vllm_url}/v1/chat/completions",
                    json={
                        "model": "Qwen/Qwen3-ASR-1.7B",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {"data": asr_b64, "format": "wav"},
                                    },
                                    {"type": "text", "text": "<|startoftranscript|>"},
                                ],
                            }
                        ],
                        "max_tokens": 512,
                        "temperature": 0.0,
                    },
                )
                asr_resp.raise_for_status()
                text = asr_resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return JSONResponse({"error": f"ASR failed: {e}"}, status_code=502)

    if not text:
        return JSONResponse(
            {"error": "No text to translate (speech not detected)"}, status_code=400
        )

    # Detect source language
    cjk = sum(1 for ch in text if "　" <= ch <= "鿿" or "＀" <= ch <= "￯")
    src_lang = "ja" if cjk > len(text) * 0.3 else "en"
    if tgt_lang == src_lang:
        tgt_lang = "en" if src_lang == "ja" else "ja"

    # Translate
    try:
        if state.translate_backend:
            translated = await state.translate_backend.translate(text, src_lang, tgt_lang)
        else:
            translated = text
    except Exception as e:
        return JSONResponse({"error": f"Translation failed: {e}"}, status_code=502)

    # TTS — synthesize in the speaker's cloned voice
    audio_out_b64 = None
    sample_rate = 16000
    try:
        if state.tts_backend and state.tts_backend.available:
            tts_audio = await state.tts_backend.synthesize(
                text=translated,
                language=tgt_lang,
                voice_reference=voice_f32,
            )
            out_pcm = (np.clip(tts_audio, -1.0, 1.0) * 32767).astype(np.int16)
            audio_out_b64 = base64.b64encode(out_pcm.tobytes()).decode()
    except Exception as e:
        logger.warning("Demo voice-clone TTS failed: %s", e)

    from meeting_scribe.languages import is_tts_native

    return JSONResponse(
        {
            "original_text": text,
            "translated_text": translated,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "audio_b64": audio_out_b64,
            "sample_rate": sample_rate,
            "tts_supported": is_tts_native(tgt_lang),
        }
    )


@router.get("/how-it-works")
async def how_it_works() -> RedirectResponse:
    return RedirectResponse("/static/how-it-works.html")
