# Styling — Tailwind v4 + component partials

The admin SPA + static pages share a single Tailwind v4 build pipeline.
Source files live under `static/css/src/`, output lands at
`static/css/dist/*.css`, and the HTML pages link the dist artifacts
with a `?v=<short-hash>` cache-bust managed by `scripts/build_css.py`.

## Layout

```
static/css/
├── src/
│   ├── lib/
│   │   ├── tokens.css         # unified @theme — colour, font, motion tokens
│   │   └── base.css           # the @import "tailwindcss" entry + @source globs
│   ├── components/
│   │   ├── _state.css         # body-state cascades (recording, popout-view,
│   │   │                      #   lang-mode, hide-furigana, …)
│   │   ├── _settings-tabs.css # Settings slide-over tab nav + panes
│   │   ├── _record-controls.css
│   │   ├── _audio-meter.css
│   │   ├── _backend-pills.css
│   │   ├── _speaker-chip.css
│   │   ├── _toast.css
│   │   ├── _meetings-panel.css
│   │   ├── _modal.css
│   │   ├── _header.css
│   │   ├── _meeting-banner.css
│   │   └── _control-bar.css   # (plus the rest under static/css/src/components/)
│   ├── app.css                # admin SPA entry — @imports every component above
│   ├── reader.css             # /reader entry
│   ├── guest.css              # /guest entry
│   ├── portal.css             # captive-portal entry
│   ├── setup.css              # setup wizard entry
│   ├── voice-clone.css        # /demo/voice-clone entry
│   ├── demo.css               # /demo landing entry
│   └── how-it-works.css       # /how-it-works entry
└── dist/                      # committed Tailwind-built output
    ├── app.css
    ├── …
    └── manifest.json          # basename → short sha256 (cache-bust key)
```

## Token namespace

The unified theme tokens live in `lib/tokens.css` and use the `--color-*`
prefix:

```css
--color-bg-deep       /* admin SPA page background       */
--color-bg-surface    /* card / panel background         */
--color-bg-raised     /* hover / pressed-cell background */
--color-border        /* base border                     */
--color-border-dim    /* subtle separator                */
--color-text-primary  --color-text-secondary  --color-text-muted  --color-text-accent
--color-recording     --color-recording-bg    /* live-meeting red */
--color-success       --color-warning         --color-speaker-{1,2,3,4}
```

`--space-{xs,sm,md,lg,xl}`, `--font-{ui,ja,mono}`, `--duration-*`, and
`--ease-out` are also in the unified `@theme`.

## Body-state cascades

`static/css/src/components/_state.css` is the **only** file that may
declare ancestor-state cascade rules of the form
`body.something .descendant { … }`. Single source of truth — every
recording, popout-view, language-mode, monolingual-slide, view-only,
furigana-hide cascade lives there.

If you find yourself wanting to add a new ancestor-state cascade in a
feature CSS file, stop and add it to `_state.css` instead.

## Cache-bust

`scripts/build_css.py` runs after every source edit:

```
scripts/build_css.py            # mode=build (default) — write dist + rewrite ?v= in HTML
scripts/build_css.py --mode check  # CI gate — fails if dist or HTML drift from source
```

The build writes `static/css/dist/manifest.json` mapping basename →
first 12 hex chars of the sha256, then walks every `static/**/*.html`
and rewrites any `?v=…` query on a link whose path matches
`/css/dist/<basename>` to the new hash. The `--mode check` lane is
wired into `.githooks/pre-push` and `.github/workflows/tests.yml` so a
stale dist or HTML can't ship.

The Tailwind binary lives at `.tools/tailwindcss` (gitignored). It is
acquired by `scripts/install_tailwind.py` which verifies a per-platform
sha256 from the committed `scripts/tailwind_versions.json`. Tailwind
itself is **not** a pip dependency — the binary is sufficient.

## Adding a new shared component partial

1. Create `static/css/src/components/_<name>.css`. Start with the
   standard header:

   ```css
   /* Meeting Scribe — <description>.
    *
    * Brief explanation of what's in here and which feature owns it.
    */

   @reference "tailwindcss";
   @import "../lib/tokens.css";
   ```

2. Write the rules. Use `--color-*`, `--space-*`, `--font-*` tokens.
   If the component has body-state cascades, move them to `_state.css`
   instead — don't duplicate.

3. Import the new partial from `static/css/src/app.css`:

   ```css
   @import "./components/_<name>.css";
   ```

4. `python3 scripts/build_css.py` to rebuild + bump the cache-bust hash.

5. Live-verify against the running admin URL with Playwright's
   `getComputedStyle` — don't trust visual matching alone.

6. Commit. CI gates: `scripts/ci_local.py` (mirrors the GH workflow).

## Visual QA viewports

Canonical list lives in `static/TESTING_VIEWPORTS.md`.

## UI lint rules (enforced by `scripts/check_ui_style.py`)

The rules in this section block commits via `meeting-scribe precommit`
or `.githooks/pre-commit`. The goal: one coherent editorial aesthetic;
no browser-default chrome leaking into the app, no unstyled pop-ups,
every diagnostic readable and copyable.

### 1. No native browser pop-ups

`window.alert`, `window.confirm`, and `window.prompt` are banned in
application JavaScript under `static/js/`. Every notice, confirmation,
and input prompt must go through the styled primitives:

- `alertDialog(title, message, okText = 'OK')` returns `Promise<void>`
- `confirmDialog(title, message, confirmText, danger)` returns
  `Promise<boolean>`
- `promptDialog(title, message, options)` returns `Promise<string | null>`

All three are exposed on `window` so modules loaded after the admin SPA
boot can reach them without an ES module import.

Narrow exemption: a single line in `slide-viewer.js` checks
`window.alertDialog` first and falls back to `window.alert` if it is
absent. The validator allows this one line and nothing else.

### 2. No em-dashes in user-facing text

Em-dash (`—`, U+2014) is banned from user-visible text in HTML files.
Use `·` (middle dot), `:`, `;`, `.`, `,`, or parentheses instead.
Em-dashes inside `<!-- ... -->` HTML comments are allowed.

### 3. Modal content must be selectable

Every modal body (`.modal-confirm-message` and `.modal-confirm-title`)
must carry `user-select: text` and `-webkit-user-select: text`, because
any error message can end up in a bug report.

### 4. Long error messages must wrap and scroll

`.modal-confirm-message` must carry `overflow-wrap: anywhere` (or
equivalent `word-break: break-word`) and a `max-height` with
`overflow-y: auto`, so that a multi-line traceback or a very long path
cannot blow out the card layout.

`alertDialog` auto-promotes messages over 160 characters or with a
newline into a wider `.modal-card.confirm.wide` card with the `.pre`
body variant and renders a **Copy** button. Don't bypass the
auto-detection.

### 5. Bilingual parity on how-it-works

`static/how-it-works.html` pairs `class="lang-en"` and `class="lang-ja"`
blocks side by side. The counts must match exactly. Enforced by
`scripts/validate_how_it_works.py`.

### 6. Glossary-acronym coverage

Every acronym in the body of `static/how-it-works.html` (`SLO`, `GEMM`,
`BF16`, etc.) must have a `.glossary-item` definition in the Terminology
section.

### 7. Multi-level modals

`showModal` is a single-overlay stack. Opening a second modal while one
is visible pushes a new card; `closeModal` pops the topmost;
`closeAllModals` empties.

Rules every dialog / caller must follow:

1. **Escape key** closes the topmost modal only. Outer modals survive.
   The built-in `_modalEscHandler` ignores Escape while focus is inside
   an `<input>`, `<textarea>`, or `<select>`.
2. **Backdrop click**: same as Escape — closes only the topmost.
3. **`card._onClose` hook**: every close path (button, Escape, backdrop,
   programmatic `closeModal`) fires the topmost card's `_onClose` hook
   exactly once before teardown. Dialog primitives set `_onClose` so
   Escape / backdrop dismissals resolve their promise with a sensible
   default and awaiting callers never hang.
4. **No document-level keydown listeners** inside dialog primitives —
   use the `_onClose` hook instead.
5. **Close the whole stack before a long-running action**: for actions
   where the outer modal would be stale after the operation (delete,
   reprocess, re-diarize), call `closeAllModals()` after validation and
   before the `fetch`.
6. **`window.closeAllModals` is globally available** — callable from
   inline `onclick` handlers on hand-written modal HTML.

Covered by `tests/js/modal-stack.test.mjs` and rule 7 in
`scripts/check_ui_style.py`.

### Running the checks locally

```bash
python3 scripts/check_ui_style.py              # all rules, exit 0/1
meeting-scribe precommit                       # includes UI + secrets
```

### Adding a new rule

1. Add a `check_*` function to `scripts/check_ui_style.py` that returns
   a list of `(file, line, message)` tuples, one per violation.
2. Register the function in the `CHECKS` list at the top of `validate()`.
3. Add a section above explaining the intent.
4. Run `python3 scripts/check_ui_style.py` against a clean tree to
   confirm zero violations.
