// Meeting Scribe — SP325 wideband-mode chip wiring (admin SPA only).
//
// Polls /api/admin/speakerphone/wideband-status every 60 s and renders
// a small chip inside the Audio popover. Operators can hit Re-apply to
// re-run the vendor HID command sequence on demand. Visibility-only:
// nothing in this module blocks any meeting flow.

const STATUS_URL = "/api/admin/speakerphone/wideband-status";
const APPLY_URL = "/api/admin/speakerphone/wideband-apply";
const POLL_MS = 60_000;

function _el(id) {
  return document.getElementById(id);
}

function _renderStatus(payload) {
  const root = _el("audio-pop-wideband");
  const valEl = _el("wideband-value");
  if (!root || !valEl) return;

  const status = payload?.status || "unavailable";
  root.dataset.status = status;

  if (status === "pass") {
    const hb = (payload.high_band_pct ?? 0).toFixed(2);
    valEl.textContent = `wideband · ${hb}% high-band`;
    valEl.title = payload.reason || "";
  } else if (status === "warn") {
    const hb = (payload.high_band_pct ?? 0).toFixed(2);
    valEl.textContent = `partial · ${hb}% high-band`;
    valEl.title = payload.reason || "";
  } else if (status === "fail") {
    const hb = (payload.high_band_pct ?? 0).toFixed(2);
    valEl.textContent = `narrowband · ${hb}% high-band — re-apply`;
    valEl.title = payload.reason || "";
  } else {
    valEl.textContent = payload?.reason || "unavailable";
    valEl.title = "";
  }
}

async function _poll() {
  try {
    const r = await fetch(STATUS_URL, { credentials: "same-origin" });
    if (!r.ok) {
      _renderStatus({ status: "unavailable", reason: `HTTP ${r.status}` });
      return;
    }
    const data = await r.json();
    _renderStatus(data);
  } catch (e) {
    _renderStatus({ status: "unavailable", reason: String(e) });
  }
}

async function _apply() {
  const btn = _el("wideband-apply-btn");
  const valEl = _el("wideband-value");
  if (!btn || !valEl) return;

  btn.disabled = true;
  const previousLabel = btn.textContent;
  btn.textContent = "Applying…";
  valEl.textContent = "applying… (15 s DSP settle)";

  try {
    const r = await fetch(APPLY_URL, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
    });
    const data = await r.json();
    if (!r.ok || !data.ok) {
      valEl.textContent = `apply failed: ${data.error || `HTTP ${r.status}`}`;
      _el("audio-pop-wideband")?.setAttribute("data-status", "fail");
    } else {
      valEl.textContent = "applied — verifying…";
      // Trigger a fresh status poll after a short delay so the
      // 15 s DSP settle has time to flush into the compliance read.
      setTimeout(_poll, 18_000);
    }
  } catch (e) {
    valEl.textContent = `apply error: ${e}`;
  } finally {
    btn.disabled = false;
    btn.textContent = previousLabel;
  }
}

// ──────────────────────────────────────────────────────────────────
// Bootstrap
// ──────────────────────────────────────────────────────────────────

const trigger = _el("btn-audio-routing");
const applyBtn = _el("wideband-apply-btn");

// Lazy-poll: only fire when the popover is opened so we don't ping the
// admin endpoint while the user is on landing / room setup. The first
// poll-on-open populates the chip; subsequent polls run every 60 s
// while the popover stays open. Closes ⇒ stop polling.
let _pollHandle = null;

function _startPolling() {
  if (_pollHandle) return;
  _poll();
  _pollHandle = setInterval(_poll, POLL_MS);
}

function _stopPolling() {
  if (_pollHandle) {
    clearInterval(_pollHandle);
    _pollHandle = null;
  }
}

if (trigger) {
  // Hook into the existing aria-expanded transition the audio-pop
  // bootstrap toggles. A MutationObserver is cheaper than wrapping
  // the click handler and respects every other open/close path
  // (Escape, outside-click, programmatic).
  const obs = new MutationObserver(() => {
    const open = trigger.getAttribute("aria-expanded") === "true";
    if (open) _startPolling();
    else _stopPolling();
  });
  obs.observe(trigger, { attributes: true, attributeFilter: ["aria-expanded"] });
}

if (applyBtn) {
  applyBtn.addEventListener("click", _apply);
}
