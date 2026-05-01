"""``meeting-scribe demo-smoke`` — real end-to-end demo gate.

Plan §4.1: drives the actual demo path against the running stack
(start meeting → upload PPTX → wait for translated render → translate
sanity check → cancel). Catches regressions that
`validate --customer-flow` misses because it only verifies HTTP-level
plumbing, not real model output.

Designed to run on a warm GB10 in ≤30 s. Reuses the canned fixture at
`tests/fixtures/validate/slides_test.pptx` so the regression set is
the same one the rest of the suite exercises.

Usage:
    meeting-scribe demo-smoke
    meeting-scribe demo-smoke --host 192.168.1.100 --port 8080
"""

from __future__ import annotations

import json
import sys
import time

import click
import httpx

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import DEFAULT_PORT, PROJECT_ROOT

_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "validate"
_PPTX = _FIXTURE_DIR / "slides_test.pptx"
_EXPECTED_PHRASES = _FIXTURE_DIR / "expected_phrases.json"


def _human(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _load_expected_phrases() -> dict:
    if not _EXPECTED_PHRASES.is_file():
        return {}
    return json.loads(_EXPECTED_PHRASES.read_text())


@cli.command("demo-smoke")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Admin host. Use the GB10's IP for remote runs.")
@click.option("--port", default=DEFAULT_PORT, show_default=True,
              help="Admin HTTPS port (defaults to meeting-scribe's 8080).")
@click.option("--source-lang", default="en", show_default=True,
              help="Slide source language for the translate gate.")
@click.option("--target-lang", default="ja", show_default=True,
              help="Slide target language. Default ja matches the typical "
                   "demo (English deck → Japanese translation).")
@click.option("--total-budget", default=60, show_default=True,
              help="Hard timeout for the whole sequence in seconds.")
@click.option("--insecure/--no-insecure", default=True, show_default=True,
              help="Skip TLS verification (admin uses a self-signed cert).")
def demo_smoke(
    host: str,
    port: int,
    source_lang: str,
    target_lang: str,
    total_budget: int,
    insecure: bool,
) -> None:
    """Drive the live demo path end-to-end.

    Sequence:
      1. POST /api/meeting/start
      2. POST /api/meetings/{id}/slides/upload (real fixture PPTX)
      3. Poll /api/meetings/{id}/slides until stage=complete
      4. Translate sanity check via the autosre vLLM proxy (port 8010)
      5. POST /api/meeting/cancel

    Exit 0 on green, non-zero on any gate failure with a structured
    error summary so a regression is diagnosable from the log alone.
    """
    if not _PPTX.is_file():
        click.secho(f"FIXTURE MISSING: {_PPTX}", fg="red", err=True)
        sys.exit(2)

    base = f"https://{host}:{port}"
    deadline = time.monotonic() + total_budget
    started = time.monotonic()

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    client = httpx.Client(verify=not insecure, timeout=15.0)
    meeting_id: str | None = None
    try:
        # ─── 1. Start meeting ───────────────────────────────────────
        click.echo("[demo-smoke] 1/4 starting meeting…")
        r = client.post(f"{base}/api/meeting/start", json={})
        if r.status_code != 200:
            click.secho(
                f"FAIL: /api/meeting/start returned {r.status_code} "
                f"({r.text[:200]})", fg="red", err=True,
            )
            sys.exit(1)
        meeting_id = r.json().get("meeting_id")
        if not meeting_id:
            click.secho(f"FAIL: no meeting_id in response: {r.text[:200]}",
                        fg="red", err=True)
            sys.exit(1)
        click.echo(f"           meeting_id={meeting_id}")

        # ─── 2. Upload PPTX ─────────────────────────────────────────
        click.echo(f"[demo-smoke] 2/4 uploading {_PPTX.name} ({_PPTX.stat().st_size} B)…")
        upload_t = time.monotonic()
        with _PPTX.open("rb") as f:
            r = client.post(
                f"{base}/api/meetings/{meeting_id}/slides/upload",
                files={"file": (_PPTX.name, f, "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
                data={"source_lang": source_lang, "target_lang": target_lang},
            )
        if r.status_code != 200:
            click.secho(
                f"FAIL: slides/upload returned {r.status_code} "
                f"({r.text[:300]})", fg="red", err=True,
            )
            sys.exit(1)
        deck = r.json()
        click.echo(f"           accepted, deck_id={deck.get('deck_id')}")

        # ─── 3. Poll for translated render ──────────────────────────
        click.echo("[demo-smoke] 3/4 waiting for slide processing…")
        last_stage = ""
        while remaining() > 0:
            r = client.get(f"{base}/api/meetings/{meeting_id}/slides")
            if r.status_code != 200:
                click.secho(
                    f"FAIL: slides poll returned {r.status_code} "
                    f"({r.text[:200]})", fg="red", err=True,
                )
                sys.exit(1)
            payload = r.json()
            stage = payload.get("stage", "")
            if stage != last_stage:
                click.echo(f"           stage={stage}")
                last_stage = stage
            if stage == "complete":
                stages = payload.get("stages", {})
                tr_stage = stages.get("translating", {})
                tr_progress = tr_stage.get("progress", "")
                click.secho(
                    f"           ✓ render+translate complete "
                    f"({tr_progress} runs translated, "
                    f"{_human(time.monotonic() - upload_t)})",
                    fg="green",
                )
                break
            if stage == "error" or payload.get("error"):
                err = payload.get("error") or "(no detail)"
                click.secho(f"FAIL: slide processing errored: {err}",
                            fg="red", err=True)
                sys.exit(1)
            time.sleep(0.5)
        else:
            click.secho(
                f"FAIL: slide processing did not reach stage=complete within "
                f"{total_budget}s (last stage={last_stage})",
                fg="red", err=True,
            )
            sys.exit(1)

        # ─── 4. Translate sanity probe ──────────────────────────────
        # Hit the autosre vLLM proxy directly with a small ja→en
        # translation. Independent of the slide pipeline; catches the
        # case where slides translate via a cache hit but the live
        # translation path is broken (or vice-versa).
        click.echo("[demo-smoke] 4/4 live translate probe…")
        translate_url = f"http://{host}:8010/v1/chat/completions"
        try:
            r = client.post(
                translate_url,
                json={
                    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
                    "messages": [
                        {"role": "system",
                         "content": "Translate the user's message to English. "
                                    "Respond with the translation only."},
                        {"role": "user", "content": "こんにちは、世界。"},
                    ],
                    "max_tokens": 64,
                    "temperature": 0.0,
                    # Disable thinking — Qwen3.6 reasoning consumes the
                    # whole token budget on a translation prompt and
                    # leaves `content` empty. Real-time utterance path
                    # already disables reasoning per project memory
                    # (scribe_reasoning_path_split).
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except httpx.RequestError as e:
            click.secho(
                f"FAIL: translate probe could not reach {translate_url}: {e}",
                fg="red", err=True,
            )
            sys.exit(1)
        if r.status_code != 200:
            click.secho(
                f"FAIL: translate probe returned {r.status_code} "
                f"({r.text[:200]})", fg="red", err=True,
            )
            sys.exit(1)
        choices = r.json().get("choices") or []
        msg = (choices[0] or {}).get("message", {}) if choices else {}
        # vLLM returns either "content" (default) or "reasoning_content" +
        # empty content on reasoning models. Demo-smoke checks the visible
        # translation in either field — the operator only cares that the
        # pipeline produced something usable.
        tr_text = (msg.get("content") or msg.get("reasoning_content") or "")
        tr_text = tr_text.strip() if isinstance(tr_text, str) else ""
        if not tr_text:
            click.secho(
                f"FAIL: translate probe returned empty content. Response: "
                f"{json.dumps(r.json(), ensure_ascii=False)[:300]}",
                fg="red", err=True,
            )
            sys.exit(1)
        # Truncate long output (reasoning models can produce paragraphs).
        preview = tr_text if len(tr_text) <= 100 else tr_text[:97] + "…"
        click.secho(f"           ✓ translate: 'こんにちは、世界。' → '{preview}'",
                    fg="green")

        click.secho(
            f"\n[demo-smoke] GREEN — total {_human(time.monotonic() - started)}",
            fg="green", bold=True,
        )

    finally:
        # Always cancel the meeting, even on failure, so the box is left
        # in a clean state for the next demo.
        if meeting_id:
            try:
                client.post(f"{base}/api/meeting/cancel", json={})
            except Exception:
                pass
        client.close()
