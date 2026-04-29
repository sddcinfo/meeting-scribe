#!/usr/bin/env python3
"""Validate Meeting Scribe UI styling rules.

Rules (see STYLING.md for rationale):

  1. No window.alert / window.confirm / window.prompt (or bare forms)
     in static/js/**. One narrow exemption for slide-viewer.js's
     fallback line when scribe-app.js has not been loaded.
  2. No em-dashes (U+2014) in user-visible HTML text. Em-dashes inside
     HTML comments are allowed.
  3. Modal title + message selectors must carry user-select: text so
     every diagnostic is copyable.
  4. .modal-confirm-message must allow long-token wrapping
     (overflow-wrap: anywhere OR word-break: break-word).
  5. Every acronym used as plain text in how-it-works.html body must
     have a <div class="glossary-item"> entry in the Terminology
     section. Matches the set defined in _HOW_IT_WORKS_ACRONYMS.

Exit 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_JS = ROOT / "static" / "js"
STATIC_CSS = ROOT / "static" / "css"
HOW_IT_WORKS = ROOT / "static" / "how-it-works.html"
INDEX_HTML = ROOT / "static" / "index.html"

# Acronyms we expect to see in how-it-works.html body copy. Each must
# have a matching <div class="glossary-item"> in the Terminology
# section. Keep this list sorted and bilingual-neutral (Japanese uses
# the same acronyms in text).
_HOW_IT_WORKS_ACRONYMS = {
    "ASR",
    "TTS",
    "vLLM",
    "LLM",
    "PCM",
    "VAD",
    "SLO",
    "MoE",
    "FP8",
    "BF16",
    "GEMM",
    "SM121",
    "KV cache",
    "VRAM",
    "WebSocket",
}

# HTML files to scan for em-dashes. Add to this list when a new public
# page is added to static/.
_USER_FACING_HTML = [
    "static/how-it-works.html",
    "static/index.html",
    "static/reader.html",
    "static/404.html",
]


# ── helpers ──────────────────────────────────────────────────────────


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def _strip_line_comment(line: str) -> str:
    """Return the code portion of a JS line, minus any trailing //-comment.

    Preserves the line identity (no newline changes) so caller-side
    line numbers stay accurate. Uses a cheap quote-balance heuristic
    to avoid truncating at `//` inside a string literal.
    """
    idx = line.find("//")
    if idx < 0:
        return line
    prefix = line[:idx]
    if prefix.count('"') % 2 == 0 and prefix.count("'") % 2 == 0:
        return prefix
    return line


def _is_inside_block_comment_at(text: str, pos: int) -> bool:
    """True if character offset `pos` falls inside an unterminated
    `/* ... */` block. Used by the popup check to skip matches that
    sit entirely inside a block comment, without shifting line numbers
    via regex substitution.
    """
    opens = 0
    i = 0
    while i < pos:
        if text[i : i + 2] == "/*":
            opens += 1
            i += 2
            end = text.find("*/", i)
            if end == -1:
                return True  # unterminated — everything after is in-comment
            if end >= pos:
                return True
            i = end + 2
            opens -= 1
        else:
            i += 1
    return False


def _strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


# ── checks ───────────────────────────────────────────────────────────


def check_no_native_popups() -> list[tuple[str, int, str]]:
    """Rule 1: no window.alert / confirm / prompt in static/js."""
    errors: list[tuple[str, int, str]] = []
    if not STATIC_JS.is_dir():
        return errors

    pattern = re.compile(r"\b(?:window\.)?(alert|confirm|prompt)\s*\(")

    for js_path in sorted(STATIC_JS.rglob("*.js")):
        rel = js_path.relative_to(ROOT).as_posix()
        raw = _read(js_path)
        # Track the running character offset so we can ask whether a
        # given match sits inside a multi-line /* ... */ block.
        offset = 0
        for i, line in enumerate(raw.splitlines(keepends=True), start=1):
            code = _strip_line_comment(line.rstrip("\n"))
            for m in pattern.finditer(code):
                absolute = offset + m.start()
                if _is_inside_block_comment_at(raw, absolute):
                    continue
                # Allow only the narrow slide-viewer fallback where
                # window.alertDialog is checked first and window.alert
                # is the else branch.
                is_slide_viewer = rel.endswith("slide-viewer.js")
                is_alertdialog_fallback = "window.alertDialog" in code and "window.alert" in code
                if is_slide_viewer and is_alertdialog_fallback:
                    continue
                errors.append((rel, i, f"native popup call: {code.strip()[:120]}"))
                break  # one error per line is enough
            offset += len(line)
    return errors


def check_no_em_dashes_in_html() -> list[tuple[str, int, str]]:
    """Rule 2: no em-dashes in rendered HTML text.

    Skipped contexts (em-dashes are allowed here because they never
    render as user-facing prose):
      - inside <!-- ... --> HTML comments
      - inside // and /* ... */ comments inside <script>/<style> blocks
      - inside placeholder glyphs that are replaced at runtime
        (id="mc-..." metric cards where the em-dash is a loading dash
        that JavaScript overwrites on first paint)
    """
    errors: list[tuple[str, int, str]] = []
    for rel in _USER_FACING_HTML:
        path = ROOT / rel
        if not path.exists():
            continue
        raw = path.read_text()
        offset = 0
        in_script_or_style = False
        tag_close_pattern: str | None = None
        for i, line in enumerate(raw.splitlines(keepends=True), start=1):
            line_no_newline = line.rstrip("\n")

            # Track whether we're inside <script>/<style>. These blocks
            # often contain JS/CSS comments with em-dashes that never
            # render as prose.
            if not in_script_or_style:
                if re.search(r"<script\b", line_no_newline, re.IGNORECASE):
                    in_script_or_style = True
                    tag_close_pattern = r"</script>"
                elif re.search(r"<style\b", line_no_newline, re.IGNORECASE):
                    in_script_or_style = True
                    tag_close_pattern = r"</style>"

            # Strip trailing //-comments on JS-inside-HTML lines so an
            # em-dash sitting only in a comment is ignored. Keep the
            # stripped text for em-dash detection.
            scan = line_no_newline
            if in_script_or_style:
                scan = _strip_line_comment(scan)

            # Allow em-dashes inside HTML comments. Check against the
            # absolute-offset block-comment scan so line numbers stay
            # accurate.
            if "—" in scan:
                # Is every occurrence of "—" inside a comment?
                any_outside_comment = False
                for match in re.finditer(r"—", scan):
                    abs_pos = offset + match.start()
                    # Treat JS /* ... */ blocks inside <script> the same
                    # way as HTML comments — both are "authored comment"
                    # contexts where the em-dash is not a display glyph.
                    if in_script_or_style and _is_inside_block_comment_at(raw, abs_pos):
                        continue
                    # HTML <!-- ... --> scan.
                    if _is_inside_html_comment_at(raw, abs_pos):
                        continue
                    # Placeholder dash in metric cards, e.g.
                    # id="mc-asr">—< — the em-dash is a loading glyph
                    # that JS overwrites on paint. Narrow exemption.
                    if re.search(r'id="mc-[a-z]+">[^<]*—', scan):
                        continue
                    any_outside_comment = True
                    break
                if any_outside_comment:
                    errors.append((rel, i, f"em-dash in user-facing text: {scan.strip()[:120]}"))

            if (
                in_script_or_style
                and tag_close_pattern
                and re.search(tag_close_pattern, line_no_newline, re.IGNORECASE)
            ):
                in_script_or_style = False
                tag_close_pattern = None

            offset += len(line)
    return errors


def _is_inside_html_comment_at(text: str, pos: int) -> bool:
    """True if character offset `pos` falls inside an unterminated
    `<!-- ... -->` span. Walk-based, same line-preserving semantics as
    `_is_inside_block_comment_at`."""
    i = 0
    while i < pos:
        if text[i : i + 4] == "<!--":
            i += 4
            end = text.find("-->", i)
            if end == -1:
                return True
            if end >= pos:
                return True
            i = end + 3
        else:
            i += 1
    return False


def check_modal_text_selectable() -> list[tuple[str, int, str]]:
    """Rule 3: .modal-confirm-message + .modal-confirm-title must be
    user-selectable so errors are copyable.

    Checks the BASE selector only (exact match on the class). Variant
    rules like `.modal-confirm-message.pre` inherit `user-select` from
    the base block, so they don't need to redeclare it. The base block
    must declare `user-select: text` explicitly.
    """
    errors: list[tuple[str, int, str]] = []
    css_path = STATIC_CSS / "style.css"
    if not css_path.exists():
        errors.append(("static/css/style.css", 0, "style.css not found"))
        return errors

    raw = css_path.read_text()

    # Only match exact base-selector rules (not variants such as
    # `.modal-confirm-message.pre`). A variant would fail the header
    # equality check below.
    rule_re = re.compile(r"(\.modal-confirm-(?:message|title))(\s*)\{([^}]*)\}", re.DOTALL)
    seen = {"message": False, "title": False}
    for match in rule_re.finditer(raw):
        header = match.group(1)
        body = match.group(3)
        kind = "message" if header.endswith("message") else "title"
        if "user-select" not in body or "text" not in body:
            line_no = raw[: match.start()].count("\n") + 1
            errors.append(
                ("static/css/style.css", line_no, f"{header}: missing user-select: text (rule 3)")
            )
        seen[kind] = True

    for kind, found in seen.items():
        if not found:
            errors.append(
                ("static/css/style.css", 0, f".modal-confirm-{kind} base rule not found (rule 3)")
            )
    return errors


def check_modal_message_wraps_long_tokens() -> list[tuple[str, int, str]]:
    """Rule 4: .modal-confirm-message must allow long-token wrapping."""
    errors: list[tuple[str, int, str]] = []
    css_path = STATIC_CSS / "style.css"
    if not css_path.exists():
        return errors

    raw = css_path.read_text()
    match = re.search(r"\.modal-confirm-message\b[^{]*\{([^}]*)\}", raw, re.DOTALL)
    if not match:
        errors.append(("static/css/style.css", 0, ".modal-confirm-message rule not found (rule 4)"))
        return errors

    body = match.group(1)
    has_wrap = ("overflow-wrap" in body and "anywhere" in body) or (
        "word-break" in body and "break-word" in body
    )
    if not has_wrap:
        line_no = raw[: match.start()].count("\n") + 1
        errors.append(
            (
                "static/css/style.css",
                line_no,
                ".modal-confirm-message missing overflow-wrap: anywhere "
                "or word-break: break-word (rule 4)",
            )
        )
    return errors


def check_no_document_keydown_in_dialogs() -> list[tuple[str, int, str]]:
    """Rule 7: dialog primitives must not attach document-level keydown
    listeners (leaks across modal pops; caused double-resolve in the
    pre-stack implementation). Use card._onClose + input-scoped
    keydown instead.

    Scans the bodies of alertDialog, confirmDialog, and promptDialog
    inside static/js/scribe-app.js. Boundaries are detected by the
    next top-level function/const/class declaration, which avoids
    the regex-literal brace-counting pitfall of a naive depth walker.
    """
    errors: list[tuple[str, int, str]] = []
    js_path = STATIC_JS / "scribe-app.js"
    if not js_path.exists():
        return errors
    raw = js_path.read_text()

    dialog_names = ("alertDialog", "confirmDialog", "promptDialog")
    header_re = re.compile(
        r"^(?:async\s+)?function\s+(\w+)\s*\(",
        re.MULTILINE,
    )
    top_level_re = re.compile(
        r"^(?:async\s+function|function|class|const|let|var)\s+",
        re.MULTILINE,
    )

    for name in dialog_names:
        # Start: `async function <name>(`.
        start_re = re.compile(
            r"^async\s+function\s+" + re.escape(name) + r"\s*\(",
            re.MULTILINE,
        )
        m = start_re.search(raw)
        if not m:
            errors.append(
                (js_path.relative_to(ROOT).as_posix(), 0, f"{name} definition not found (rule 7)")
            )
            continue
        # End: the next top-level declaration after this one. That line
        # starts at column zero with function/class/const/let/var —
        # reliable for scribe-app.js's flat top-level style.
        next_m = top_level_re.search(raw, m.end())
        body_end = next_m.start() if next_m else len(raw)
        body = raw[m.end() : body_end]
        for bad in re.finditer(
            r"document\.addEventListener\s*\(\s*['\"]keydown['\"]",
            body,
        ):
            src_offset = m.end() + bad.start()
            line_no = raw[:src_offset].count("\n") + 1
            errors.append(
                (
                    js_path.relative_to(ROOT).as_posix(),
                    line_no,
                    f"{name} attaches document keydown listener — "
                    f"use card._onClose instead (rule 7)",
                )
            )
    # Reference the overall-shape header pattern above to quiet pyflakes;
    # the walker doesn't need it directly.
    _ = header_re
    return errors


def check_how_it_works_glossary() -> list[tuple[str, int, str]]:
    """Rule 5: every acronym in how-it-works body needs a glossary item."""
    errors: list[tuple[str, int, str]] = []
    html = _read(HOW_IT_WORKS)
    if not html:
        return errors

    # Extract every <div class="glossary-item" ...> <dt>TERM</dt>
    items = re.findall(
        r'<div\s+class="glossary-item"[^>]*>\s*<dt>([^<]+)</dt>',
        html,
    )
    present = {t.strip() for t in items}

    # Split the document at the glossary section so we only scan body
    # copy for acronym mentions (otherwise the glossary itself would
    # satisfy every rule trivially).
    gloss_idx = html.find('<section class="glossary"')
    body = html[:gloss_idx] if gloss_idx != -1 else html

    for acronym in _HOW_IT_WORKS_ACRONYMS:
        if acronym in body and acronym not in present:
            errors.append(
                (
                    HOW_IT_WORKS.relative_to(ROOT).as_posix(),
                    0,
                    f"acronym '{acronym}' used in body but missing from glossary (rule 5)",
                )
            )
    return errors


# ── main ─────────────────────────────────────────────────────────────


CHECKS = [
    ("no-native-popups (rule 1)", check_no_native_popups),
    ("no-em-dashes-in-html (rule 2)", check_no_em_dashes_in_html),
    ("modal-text-selectable (rule 3)", check_modal_text_selectable),
    ("modal-wraps-long-tokens (rule 4)", check_modal_message_wraps_long_tokens),
    ("glossary-coverage (rule 6)", check_how_it_works_glossary),
    ("no-doc-keydown-in-dialogs (rule 7)", check_no_document_keydown_in_dialogs),
]


def validate() -> list[tuple[str, str, int, str]]:
    all_errors: list[tuple[str, str, int, str]] = []
    for name, fn in CHECKS:
        try:
            for path, line, msg in fn():
                all_errors.append((name, path, line, msg))
        except Exception as exc:  # check crash counts as a failure
            all_errors.append((name, "<check>", 0, f"check crashed: {exc}"))
    return all_errors


def main() -> None:
    errors = validate()
    if errors:
        print(f"UI style validation FAILED ({len(errors)} issue{'s' if len(errors) != 1 else ''}):")
        for rule, path, line, msg in errors:
            loc = f"{path}:{line}" if line else path
            print(f"  [{rule}] {loc} — {msg}")
        print()
        print("See STYLING.md for the full rule set.")
        sys.exit(1)
    print("UI style validation: all checks passed")
    for name, _ in CHECKS:
        print(f"  ok {name}")


if __name__ == "__main__":
    main()
