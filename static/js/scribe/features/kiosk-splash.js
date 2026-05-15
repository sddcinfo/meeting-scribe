// Meeting Scribe - HDMI kiosk idle splash population.
//
// On every page load that has ``body[data-role="kiosk"]`` set
// (server-side injected by the ``/kiosk`` view handler), fetch the
// appliance PIN from the narrow ``/api/kiosk/settings`` projection
// and paint it into ``#kiosk-idle-splash-pin`` in display-size
// numerals. Subscribes to the meeting-state WS events so the splash
// transitions to/from a fading state when a meeting starts/stops -
// the cascade in ``components/_state.css`` does the actual hide.
//
// Pure module: no top-level side effects. The bootstrap counterpart
// (``kiosk-splash.bootstrap.js``) wires the initial paint + the WS
// event listeners.

import { esc } from "../lib/escape.js";

const API = "";

let _renderedPin = null;

export async function refreshKioskSplashPin() {
  const target = document.getElementById("kiosk-idle-splash-pin");
  if (!target) return;
  try {
    // Prefer the loopback-only narrow projection when we're actually
    // a kiosk; fall through to the admin payload (works during admin
    // SSH-port-forward previews).
    let pin = "";
    try {
      const r = await fetch(`${API}/api/kiosk/settings`);
      if (r.ok) {
        const j = await r.json();
        pin = j.appliance_pin || "";
      }
    } catch {
      // ignored
    }
    if (!pin) {
      const r = await fetch(`${API}/api/admin/settings`);
      if (r.ok) {
        const j = await r.json();
        pin = j.appliance_pin || "";
      }
    }
    if (!pin || pin === _renderedPin) return;
    _renderedPin = pin;
    target.textContent = pin;
    // ARIA label so screen readers announce it on appear.
    target.setAttribute("aria-label", `Meeting ID ${esc(pin)}`);
  } catch (err) {
    console.warn("[kiosk-splash] refresh failed:", err);
  }
}
