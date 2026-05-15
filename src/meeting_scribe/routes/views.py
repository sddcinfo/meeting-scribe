"""Top-level HTML view routes + the standalone voice-clone demo POST.

Plain-old page renders that hand pre-cached HTML from
``server_support.page_cache._HTML`` to ``HTMLResponse``. The single
``/api/demo/voice-clone`` POST is grouped here because it backs
``/demo/voice-clone`` and shares no state with the live meeting
pipeline — it does its own ASR → translate → TTS round-trip against
``state.translate_backend`` / ``state.tts_backend`` and is independent
of any active meeting.

The ``index`` route is the one slightly load-bearing handler: it
inspects ``has_admin_session(request)`` to decide whether to serve
the portal/guest pages or the admin index, and ACKs hotspot clients
so captive-portal probes flip to "online". The cookie distinction
replaces the old subnet/scheme heuristic — v1.0 puts every client on
a single HTTPS origin.
"""

from __future__ import annotations

import base64
import io
import logging

import fastapi
import httpx
import numpy as np
from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.admin_guard import has_admin_session
from meeting_scribe.server_support.captive_ack import _captive_ack
from meeting_scribe.server_support.page_cache import _HTML

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=None)
async def index(request: fastapi.Request) -> HTMLResponse | RedirectResponse:
    """Pick the rendered page from the caller's cookie state.

    Order matters:
      1. ``scribe_admin`` cookie → admin index.
      2. Hotspot clients are ACK'd so subsequent captive probes return
         "online" — without that the iOS CNA never dismisses.
      3. ``ms_guest`` cookie (Phase H captive-gateway, post-PIN entry)
         OR legacy ``scribe_portal=done`` cookie → guest page. This
         must run BEFORE the admin-mode redirect; otherwise a guest
         who authed via the port-80 PIN flow ends up bounced back to
         ``/auth`` in admin mode and never sees the meeting view (GB10
         phone test 2026-05-13).
      4. Admin mode with NO cookie → ``/auth`` so the operator goes
         straight to sign-in without an interstitial.
      5. Anything else → portal landing (the rotating-SSID default).
    """
    if has_admin_session(request):
        return HTMLResponse(_HTML.get("index", ""))
    _captive_ack(request)
    # Authenticated guest? Render the guest meeting view regardless
    # of AP mode. Three accepted signals (any one is enough):
    #   * ``ms_guest`` cookie (post-PIN canonical-HTTPS auth).
    #   * Legacy ``scribe_portal=done`` cookie.
    #   * IP in the ``ms-allowed-guests`` ipset. Required for the
    #     Phase H captive-gateway flow: the port-80 captive sub-app
    #     can't set a cookie that crosses into Safari, so the first
    #     HTTPS request arrives with NO cookie. The hotspot_guard
    #     middleware mints one on the response, but this handler
    #     runs first and would otherwise fall through to the
    #     admin-mode ``/auth`` redirect (GB10 phone test 2026-05-13:
    #     "authed as guest but got a differently styled auth page").
    from meeting_scribe.routes.guest_auth import verify_guest_cookie

    guest_signal = (
        verify_guest_cookie(request.cookies.get("ms_guest"))
        or request.cookies.get("scribe_portal") == "done"
    )
    if not guest_signal:
        try:
            from meeting_scribe.server_support import firewall_allowlist

            client = getattr(request, "client", None)
            ip = client.host if client is not None else ""
            if ip and firewall_allowlist.is_guest(ip):
                guest_signal = True
        except Exception:
            logger.exception("views.index: guest ipset lookup failed")
    if guest_signal:
        return HTMLResponse(_HTML.get("guest", ""))
    if _admin_mode_ap_active():
        return RedirectResponse("/auth", status_code=302)
    return HTMLResponse(_HTML.get("portal", ""))


def _admin_mode_ap_active() -> bool:
    """True when the persisted AP mode is ``admin`` (fixed SSID, no portal).

    Read-only, defensive: any settings_store failure falls back to
    ``False`` so a misconfigured box still renders portal.html instead
    of returning a 500. The check is cheap (mtime-cached settings
    read) so it's safe to call on every ``/`` hit.
    """
    try:
        from meeting_scribe.server_support.settings_store import _load_settings_override

        return _load_settings_override().get("wifi_mode") == "admin"
    except Exception:
        return False


@router.get("/reader")
async def reader_view(request: fastapi.Request) -> HTMLResponse:
    """Large-font, text-only reader view for guest displays (iPad, TV)."""
    return HTMLResponse(_HTML.get("reader", ""))


@router.get("/kiosk", response_model=None)
async def kiosk_view(request: fastapi.Request) -> HTMLResponse | JSONResponse:
    """HDMI kiosk mirror of the laptop admin pop-out.

    Gated by the ``scribe_kiosk`` cookie (issued only at
    ``/kiosk-bootstrap`` on the loopback listener) OR by an admin
    cookie when the request truly arrived on the kiosk listener (lets
    operators preview the kiosk via an SSH port-forward during dev).

    Renders the admin index template with ``data-role="kiosk"``
    injected into the body tag so the CSS cascade in ``_state.css``
    can hide admin chrome and reveal the kiosk-only idle splash.
    """
    from meeting_scribe.auth.roles import Role

    role = getattr(request.state, "role", Role.GUEST)
    via_kiosk_listener = getattr(request.state, "via_kiosk_listener", False)
    if role != Role.KIOSK and not (via_kiosk_listener and role == Role.ADMIN):
        return JSONResponse({"error": "kiosk_view_forbidden"}, status_code=403)

    html = _HTML.get("index", "")
    if not html:
        return JSONResponse({"error": "index_unavailable"}, status_code=503)
    # Inject ``data-role="kiosk"`` onto the body tag so the cascade
    # in ``_state.css`` can hide admin chrome and reveal the kiosk
    # idle splash. Handles both ``<body>`` (current template) and
    # ``<body class="...">`` (future template variants).
    if "<body>" in html:
        html = html.replace("<body>", '<body data-role="kiosk">', 1)
    else:
        html = html.replace("<body ", '<body data-role="kiosk" ', 1)
    # No-store: chromium snap was caching the kiosk page in its
    # ``~/snap/chromium/common/kiosk-profile`` disk cache, so even a
    # full restart of the kiosk service kept showing stale HTML
    # (the old ``<img src=dell-wordmark.svg>`` block painting as a
    # solid white rectangle). The kiosk URL never changes - bumping
    # a ``?v=`` would just kick the can - so kill the cache outright.
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


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


# ── Editorial pages ──
#
# The three editorial pages (how-it-works, benchmarking,
# hardware-scaling) share a canonical trailing-slash URL convention
# that works identically on both the FastAPI server and on the GitHub
# Pages mirror. The HTML uses relative ``../static/...`` paths for
# CSS / JS / data so the same file renders correctly under
# ``/how-it-works/`` here and under ``/meeting-scribe/how-it-works/``
# on Pages.
#
# No-slash variants 308-redirect to the canonical trailing-slash form
# so legacy / ergonomic URLs keep working but every cross-nav link
# stays on the canonical URL.

from meeting_scribe.server_support.page_cache import _LAST_STATIC_DIR


def _editorial_file(name: str) -> FileResponse:
    """Serve a static/<name>.html file at a slug URL.

    Reads STATIC_DIR off the page cache (populated at server startup)
    so we don't have to thread it through the router. Falls back to a
    relative path lookup in tests where the cache hasn't been primed.
    """
    if _LAST_STATIC_DIR is not None:
        return FileResponse(_LAST_STATIC_DIR / name)
    from pathlib import Path

    return FileResponse(Path("static") / name)


@router.get("/how-it-works/", include_in_schema=False)
async def how_it_works_page() -> FileResponse:
    return _editorial_file("how-it-works.html")


@router.get("/how-it-works", include_in_schema=False)
async def how_it_works_no_slash() -> RedirectResponse:
    return RedirectResponse("/how-it-works/", status_code=308)


@router.get("/benchmarking/", include_in_schema=False)
async def benchmarking_page() -> FileResponse:
    return _editorial_file("benchmarking.html")


@router.get("/benchmarking", include_in_schema=False)
async def benchmarking_no_slash() -> RedirectResponse:
    return RedirectResponse("/benchmarking/", status_code=308)


@router.get("/hardware-scaling/", include_in_schema=False)
async def hardware_scaling_page() -> FileResponse:
    return _editorial_file("hardware-scaling.html")


@router.get("/hardware-scaling", include_in_schema=False)
async def hardware_scaling_no_slash() -> RedirectResponse:
    return RedirectResponse("/hardware-scaling/", status_code=308)
