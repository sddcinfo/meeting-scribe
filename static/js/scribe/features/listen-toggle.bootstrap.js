// Meeting Scribe — Listen toggle wiring.
//
// Wires the header's "Listen" button — the operator-facing toggle for
// the real-time interpretation TTS audio output. Reads
// `window.audioOutListener` (stamped by features/audio-out.bootstrap.js,
// which loads earlier in the bootstrap chain) and orchestrates:
//   * Synchronous AudioContext priming (critical — must happen
//     inside the user-gesture stack, never after an await).
//   * /api/languages dropdown population on first open.
//   * Default language pick = the "other" language in the pair.
//   * Listen language + mode change handlers.
//
// Also wires the `/#listen` auto-start path used by the demo links.

import { getLangB as _getLangB } from "../lib/lang-helpers.js";

// Auto-start Listen if navigated via /#listen (e.g. from /demo page).
if (location.hash === "#listen") {
  const _waitForRecording = setInterval(() => {
    if (document.body.classList.contains("recording")) {
      clearInterval(_waitForRecording);
      document.getElementById("btn-listen")?.click();
      history.replaceState(null, "", location.pathname);
    }
  }, 1000);
  setTimeout(() => clearInterval(_waitForRecording), 30000);
}

document.getElementById("btn-listen")?.addEventListener("click", async () => {
  const btn = document.getElementById("btn-listen");
  const langSelect = document.getElementById("listen-lang");
  const modeSelect = document.getElementById("listen-mode");

  if (window.audioOutListener.enabled) {
    window.audioOutListener.stop();
    btn.classList.remove("active");
    langSelect.style.display = "none";
    modeSelect.style.display = "none";
  } else {
    // CRITICAL: prime the AudioContext SYNCHRONOUSLY here, BEFORE any
    // await. If we let `window.audioOutListener.start()` create the
    // context after the language fetch resolves, the user-gesture
    // context is already gone and the resume() runs in a deferred
    // microtask that browsers refuse to honor. The result is a
    // permanently-suspended AudioContext that decodes every incoming
    // WAV into a muted destination — server says "delivered", user
    // hears nothing.
    window.audioOutListener.primeAudioContext();

    if (langSelect.options.length === 0) {
      try {
        const resp = await fetch("/api/languages");
        const data = await resp.json();
        (data.languages || []).forEach((l) => {
          const opt = document.createElement("option");
          opt.value = l.code;
          opt.textContent = l.native_name || l.name;
          langSelect.appendChild(opt);
        });
        const langB = _getLangB();
        if (langB) langSelect.value = langB;
      } catch {
        /* ignore */
      }
    }
    const lang = langSelect.value || _getLangB() || "en";
    const mode = modeSelect.value || "translation";
    await window.audioOutListener.start(lang, mode);
    btn.classList.add("active");
    langSelect.style.display = "";
    modeSelect.style.display = "";
  }
});

document.getElementById("listen-lang")?.addEventListener("change", (e) => {
  window.audioOutListener.setLanguage(e.target.value);
});

document.getElementById("listen-mode")?.addEventListener("change", (e) => {
  window.audioOutListener.setMode(e.target.value);
});
