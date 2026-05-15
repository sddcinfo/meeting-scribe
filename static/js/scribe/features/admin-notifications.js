// Meeting Scribe — admin notifications banner.
//
// Surfaces server-side ``admin_notifications`` (from /api/status) using
// the existing ``.meeting-banner`` component for visual consistency.
// Producers live in audio_routing.reconcile_audio_routing() and write
// rows via server_support.admin_notifications.put_notification().
//
// Three notification kinds today:
//
//   * mic_rebound       — informational. "Mic was auto-rebound from X to Y."
//                         Dismiss is fire-and-forget.
//   * mic_ambiguous     — actionable. "Multiple mics match — pick one."
//                         Action button opens the admin audio route page.
//   * mic_unresolved    — informational. "Mic disconnected; will auto-
//                         restore when it returns."
//   * mic_capture_failed — actionable. "Mic capture refused to start."
//                         Action button is "Retry" (re-POST /api/admin/audio/route).
//
// Only the newest un-dismissed notification is shown at any one time —
// these are operator-facing alerts about the audio routing state, not
// a queue. Picking the newest covers the case where a single device
// flap produces unresolved→rebound; only the rebound row stays visible.

const BANNER_ID = "admin-notification-banner";

// Notification kinds that the operator never needs to act on — auto-
// dismiss after a beat so the banner doesn't squat on the meeting view.
// Actionable kinds (mic_ambiguous, mic_capture_failed) stay until the
// operator explicitly resolves them.
const INFO_KINDS = new Set(["mic_rebound", "mic_unresolved"]);
const AUTO_DISMISS_MS = 8000;

function ensureAdminBannerEl() {
  let el = document.getElementById(BANNER_ID);
  if (el) return el;
  el = document.createElement("div");
  el.id = BANNER_ID;
  // Reuse .meeting-banner so the CSS already in the bundle styles it.
  // The tone modifier is chosen per-notification by renderAdminNotifications:
  //   .info       — benign, slate (mic_rebound, mic_unresolved)
  //   .return     — amber, fallback for unknown kinds
  //   .reconnecting (not used here) — red alert.
  el.className = "meeting-banner";
  el.setAttribute("role", "status");
  el.setAttribute("aria-live", "polite");
  el.innerHTML = `
    <span class="meeting-banner-dot" aria-hidden="true"></span>
    <span class="meeting-banner-label"></span>
    <button class="meeting-banner-btn admin-notification-action" type="button"></button>
    <button class="meeting-banner-btn admin-notification-dismiss" type="button">Dismiss</button>
  `;
  document.body.appendChild(el);
  return el;
}

function _hideAdminBanner() {
  const el = document.getElementById(BANNER_ID);
  if (!el) return;
  el.classList.remove("visible");
}

function _formatLabel(item) {
  switch (item.kind) {
    case "mic_rebound":
      return `Microphone auto-rebound — ${shortNode(item.mic_from)} → ${shortNode(item.mic_to)}`;
    case "mic_ambiguous":
      return `Microphone ambiguous — ${item.candidates?.length ?? "?"} sources match the configured device. Pick one to continue.`;
    case "mic_unresolved":
      return `Microphone disconnected — will auto-restore when it returns (${shortNode(item.mic_node)})`;
    case "mic_capture_failed":
      return `Microphone capture failed to start: ${item.detail || "unknown error"}`;
    default:
      return item.detail || item.kind || "Admin notice";
  }
}

function shortNode(name) {
  if (!name) return "(unknown)";
  // PipeWire node names are long; show the trailing device-specific portion.
  const m = /alsa_(?:input|output)\.(?:usb-)?(.+?)(?:-\d{2})?\.([^.]+)(?:\.\d+)?$/.exec(name);
  if (m) return `${m[1]} (${m[2]})`;
  if (name.startsWith("bluez_")) {
    const macMatch = /bluez_(?:input|output)\.([0-9A-Fa-f_:]+)/.exec(name);
    if (macMatch) return `BT ${macMatch[1].replace(/_/g, ":")}`;
  }
  // Fallback — chop to last component.
  return name.length > 64 ? name.slice(0, 32) + "…" + name.slice(-16) : name;
}

function _actionFor(item) {
  switch (item.kind) {
    case "mic_ambiguous":
    case "mic_capture_failed":
      return {
        label: "Open audio routing",
        run: () => {
          // The admin audio route card lives in the admin settings panel.
          // Routes are SPA-internal so a hash navigation is enough.
          if (location.hash !== "#settings/audio") {
            location.hash = "#settings/audio";
          }
        },
      };
    case "mic_rebound":
    case "mic_unresolved":
    default:
      return null;
  }
}

async function _dismiss(kind, dismissBtn) {
  if (dismissBtn) dismissBtn.disabled = true;
  try {
    const resp = await fetch(
      `/api/admin/notifications/${encodeURIComponent(kind)}/dismiss`,
      { method: "POST", credentials: "same-origin" },
    );
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  } catch (err) {
    if (dismissBtn) dismissBtn.disabled = false;
    console.warn("admin-notifications: dismiss failed", err);
    return;
  }
  // Hide locally without waiting for the next /api/status poll.
  _hideAdminBanner();
}

// Tracks the kind currently shown so the auto-dismiss timer doesn't
// race a re-render when /api/status delivers the same row again.
let _shownKind = null;
let _autoDismissTimer = null;

/**
 * Render the highest-priority un-dismissed admin notification.
 *
 * Called from checkStatus() on every /api/status response with
 * ``data.admin_notifications``. An empty list hides the banner.
 */
export function renderAdminNotifications(items) {
  if (!Array.isArray(items) || items.length === 0) {
    _hideAdminBanner();
    _shownKind = null;
    if (_autoDismissTimer) {
      clearTimeout(_autoDismissTimer);
      _autoDismissTimer = null;
    }
    return;
  }
  // Server already orders newest-first; pick the head.
  const item = items[0];
  const el = ensureAdminBannerEl();
  el.classList.add("visible");

  // Tone modifier: slate ``.info`` for benign rebound / unresolved
  // notices, amber ``.return`` for everything else (actionable
  // ambiguous / capture_failed prompts that the operator needs to
  // notice). Strip both classes first so the class state is fresh
  // on every render.
  el.classList.remove("info", "return");
  el.classList.add(INFO_KINDS.has(item.kind) ? "info" : "return");

  const labelEl = el.querySelector(".meeting-banner-label");
  const actionBtn = el.querySelector(".admin-notification-action");
  const dismissBtn = el.querySelector(".admin-notification-dismiss");

  if (labelEl) labelEl.textContent = _formatLabel(item);

  const action = _actionFor(item);
  if (actionBtn) {
    if (action) {
      actionBtn.textContent = action.label;
      actionBtn.style.display = "";
      actionBtn.onclick = action.run;
    } else {
      actionBtn.style.display = "none";
      actionBtn.onclick = null;
    }
  }

  if (dismissBtn) {
    dismissBtn.disabled = false;
    dismissBtn.onclick = () => _dismiss(item.kind, dismissBtn);
  }

  // Auto-dismiss informational notifications after a beat so they
  // don't squat on the meeting view. Re-arm only if the kind changed
  // (avoids restarting the countdown on every /api/status tick).
  if (_autoDismissTimer && _shownKind !== item.kind) {
    clearTimeout(_autoDismissTimer);
    _autoDismissTimer = null;
  }
  if (INFO_KINDS.has(item.kind) && !_autoDismissTimer) {
    _autoDismissTimer = setTimeout(() => {
      _autoDismissTimer = null;
      _dismiss(item.kind, null);
    }, AUTO_DISMISS_MS);
  }
  _shownKind = item.kind;
}
