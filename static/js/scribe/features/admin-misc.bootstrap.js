// Meeting Scribe — assorted admin SPA boot bits.
//
// Collects four small, independent DOM bootstraps. They share two
// things and nothing else:
//
//   1. They have to fire after the DOM is parsed.
//   2. They're admin-only — the popout window doesn't see these
//      controls. Each block self-gates on `?popout=view` to keep that
//      contract.
//
// The four blocks, in evaluation order:
//
//   * Transcript scroll-direction toggle (#btn-scroll-dir) — persists
//     newest-first / oldest-first across reloads via localStorage.
//   * Keyboard shortcut: Space bar play/pause the audio player when
//     the user isn't typing in an input.
//   * beforeunload warning — block accidental tab close while
//     recording or mid-start (server-side finalize completes regardless
//     of browser presence, so no warning otherwise).
//   * SegmentStore subscribe bridge into the 1:1 conversation mode
//     renderer (forwards every final segment event into
//     `window._oneOnOneRender` when 1:1 mode is active).
//
// Dependency surface:
//   state.js — `store` (SegmentStore singleton)
// Window globals it consumes:
//   `window._gridRenderer`, `window.audioPlayer`,
//   `window._oneOnOneActive`, `window._oneOnOneRender`

import { store, timer } from "../state.js";

const POPOUT_MODE = new URLSearchParams(location.search).get("popout");

// ── 1. Transcript scroll-direction toggle ────────────────────────────
(function () {
  if (POPOUT_MODE) return;
  const btn = document.getElementById("btn-scroll-dir");
  if (!btn) return;
  // Restore saved preference (default: newest first, matches
  // CompactGridRenderer init).
  const saved = localStorage.getItem("scribe_transcript_direction");
  if (saved === "oldest") {
    btn.textContent = "↓ Oldest first";
    btn.dataset.direction = "oldest";
    // Apply after gridRenderer is initialized (it defaults to newestFirst=true)
    setTimeout(() => {
      if (window._gridRenderer && window._gridRenderer._newestFirst) {
        window._gridRenderer.toggleDirection();
      }
    }, 100);
  } else {
    btn.textContent = "↑ Newest first";
    btn.dataset.direction = "newest";
  }
  btn.addEventListener("click", () => {
    if (!window._gridRenderer) return;
    const newestFirst = window._gridRenderer.toggleDirection();
    btn.textContent = newestFirst ? "↑ Newest first" : "↓ Oldest first";
    btn.dataset.direction = newestFirst ? "newest" : "oldest";
    localStorage.setItem(
      "scribe_transcript_direction",
      newestFirst ? "newest" : "oldest",
    );
  });
})();

// ── 2. Keyboard shortcut: Space = audio player play/pause ────────────
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.code === "Space" && window.audioPlayer?.audio?.src) {
    e.preventDefault();
    window.audioPlayer.togglePlay();
  }
});

// ── 3. beforeunload warning — admin only, while recording ────────────
if (!POPOUT_MODE) {
  window.addEventListener("beforeunload", (e) => {
    if (
      document.body.classList.contains("recording")
      || document.body.classList.contains("starting")
    ) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
}

// ── 4. #btn-clear — wipe transcript + segment store + timer reset ────
// Dev-only debug control surfaced on the admin SPA. Wipes the live
// transcript grid, the SegmentStore, and the meeting timer in one
// click. Popout windows render no such button; the listener bails
// quietly if the element isn't in the DOM.
document.getElementById("btn-clear")?.addEventListener("click", () => {
  if (window._gridRenderer) window._gridRenderer._clear(true);
  store.clear();
  timer.reset();
});

// ── 5. SegmentStore → 1:1 conversation renderer bridge ───────────────
// The 1:1 mode bootstrap (features/one-on-one.bootstrap.js) publishes
// `window._oneOnOneRender`. Bridging through the global keeps the
// publish order intact (registry → bootstrap → this dispatcher).
store.subscribe((segId, event) => {
  if (
    window._oneOnOneActive?.()
    && event.is_final
    && window._oneOnOneRender
  ) {
    window._oneOnOneRender(event);
  }
});
