# Meeting Scribe UI Styling Guide

The rules in this document are enforced by `scripts/check_ui_style.py`.
A violation blocks `meeting-scribe precommit` and any commit that runs
through the pre-commit hook at `.githooks/pre-commit`.

The goal is one coherent editorial aesthetic: no browser-default chrome
leaking into the app, no unstyled pop-ups, every diagnostic readable
and copyable.

## 1. No native browser pop-ups

`window.alert`, `window.confirm`, and `window.prompt` are banned in
application JavaScript under `static/js/`. Every notice, confirmation,
and input prompt must go through the styled primitives in
`static/js/scribe-app.js`:

- `alertDialog(title, message, okText = 'OK')` returns `Promise<void>`
- `confirmDialog(title, message, confirmText, danger)` returns
  `Promise<boolean>`
- `promptDialog(title, message, options)` returns `Promise<string | null>`
  (`null` = cancelled, matching the `window.prompt` contract)

All three are exposed on `window` so modules loaded after `scribe-app.js`
(e.g. `slide-viewer.js` on `reader.html`) can reach them without an ES
module import.

### Narrow exemption

A single line in `slide-viewer.js` checks `window.alertDialog` first and
falls back to `window.alert` if it is absent. This covers the case where
`slide-viewer.js` is loaded in a context where `scribe-app.js` is not,
so a file-type rejection never silently succeeds. The validator allows
this one line and nothing else.

## 2. No em-dashes in user-facing text

Em-dash (`—`, U+2014) is banned from user-visible text in HTML files.
Use one of these instead:

- `·` (middle dot) for bullet-like section labels ("01 · Intro")
- `:` (colon) to introduce a list or label
- `;` (semicolon) or `.` to break an independent clause
- `,` or parentheses for appositions and asides

Em-dashes inside `<!-- ... -->` HTML comments are allowed because they
never render. The validator skips them automatically.

## 3. Modal content must be selectable

Every modal body (`.modal-confirm-message` and `.modal-confirm-title`)
must be selectable and copyable, because any error message can end up
in a bug report. In `static/css/style.css` these selectors must carry
`user-select: text` and `-webkit-user-select: text`.

## 4. Long error messages must wrap and scroll

`.modal-confirm-message` must carry `overflow-wrap: anywhere` (or
equivalent `word-break: break-word`) and a `max-height` with
`overflow-y: auto`, so that a multi-line traceback or a very long path
cannot blow out the card layout.

`alertDialog` auto-promotes messages over 160 characters or with a
newline into a wider `.modal-card.confirm.wide` card with the `.pre`
body variant (mono font, pre-wrap whitespace) and renders a **Copy**
button. Keep that auto-detection in place; do not bypass it by manually
calling `showModal` with hand-written HTML for long errors.

## 5. Bilingual parity on how-it-works

`static/how-it-works.html` pairs `class="lang-en"` and
`class="lang-ja"` blocks side by side. The counts must match exactly:
every English block needs a Japanese counterpart and vice versa. This
rule is enforced by the existing `scripts/validate_how_it_works.py`.

## 6. Glossary-acronym coverage

Every acronym that appears as plain text in the body of
`static/how-it-works.html` (for example `SLO`, `GEMM`, `BF16`) must
have a `.glossary-item` definition in the Terminology section. Adding a
new acronym to the body without adding it to the glossary is a
regression.

## 7. Multi-level modals

`showModal` is a single-overlay stack. Opening a second modal while
one is already visible (for example the Meeting Actions modal opening
a `confirmDialog`) pushes a new card onto the stack; the outer card
is hidden in place but its DOM and event listeners survive untouched.
`closeModal` pops one card — the topmost — and re-activates the
previous one. `closeAllModals` walks the stack to empty.

Rules every dialog / caller must follow:

1. **Escape key**: closes the topmost modal only. Outer modals
   survive. The built-in `_modalEscHandler` ignores Escape while focus
   is inside an `<input>`, `<textarea>`, or `<select>` so an input
   with its own Escape handler (e.g. `promptDialog`) can cancel
   itself without double-popping the stack.
2. **Backdrop click**: same as Escape — closes only the topmost.
3. **`card._onClose` hook**: every close path (explicit button,
   Escape, backdrop, programmatic `closeModal`) fires the topmost
   card's `_onClose` hook exactly once before the DOM is torn down.
   Dialog primitives (`alertDialog`, `confirmDialog`, `promptDialog`)
   set `_onClose` so Escape / backdrop dismissals resolve their
   promise with a sensible default (`null`, `false`, or `undefined`)
   and awaiting callers never hang.
4. **No document-level keydown listeners inside dialog primitives**:
   use the `_onClose` hook instead. Document-level listeners from
   prior versions leaked across modal pops and caused double-resolve
   bugs when a stacked dialog's key handler fired on the outer
   modal's input.
5. **Close the whole stack before a long-running action**: for
   actions where the outer modal would be stale after the operation
   (delete, reprocess, re-diarize), call `closeAllModals()` after
   validation and before the `fetch`. The completion / failure alert
   is the user's only required feedback.
6. **`window.closeAllModals` is globally available**: call it from
   inline `onclick` handlers on hand-written modal HTML when the
   whole stack should collapse.

These rules are covered by `tests/js/modal-stack.test.mjs` (10 node
tests) and enforced statically by rule 7 in
`scripts/check_ui_style.py`.

## Running the checks locally

```bash
python3 scripts/check_ui_style.py              # all rules, exit 0/1
meeting-scribe precommit                       # includes UI + secrets
```

## Enabling the pre-commit hook

Once per clone, point git at the versioned hook directory:

```bash
git config core.hooksPath .githooks
```

After that, every `git commit` runs `.githooks/pre-commit`, which
invokes `scripts/check_ui_style.py` (blocking) and
`scripts/validate_how_it_works.py` (blocking). The sensitive-data
scanner runs as advisory in the hook because it still has known
false positives; CI should gate on it once those are triaged.

To bypass the hook in a true emergency, use `git commit --no-verify`.
Bypassing should be exceptional — CI must still catch the violation
before merge.

## Adding a new rule

1. Add a `check_*` function to `scripts/check_ui_style.py` that returns
   a list of `(file, line, message)` tuples, one per violation.
2. Register the function in the `CHECKS` list at the top of
   `validate()`.
3. Add a one-paragraph section above explaining the intent.
4. Run `python3 scripts/check_ui_style.py` against a clean tree to
   confirm zero violations.
