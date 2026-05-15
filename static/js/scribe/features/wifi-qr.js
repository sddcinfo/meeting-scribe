// Meeting Scribe - meeting PIN header chip.
//
// The hotspot SSID is open so "scan to join" QR codes are redundant;
// this surface persistently shows the 4-digit appliance PIN ("1618")
// that doubles as the SSID suffix AND the guest auth PIN. The DOM
// target is #header-qr so popout / mirror references continue to
// resolve.
//
// Bootstrap half (wifi-qr.bootstrap.js) wires the initial paint + a
// light 30 s polling interval to follow factory_reset rotations.

import { esc } from "../lib/escape.js";

const API = "";

let _lastRenderedPin = null;

export async function refreshWifiQR(retries = 15, delayMs = 2000) {
  const target = document.getElementById("header-qr");
  if (!target) return;
  try {
    // Try the admin settings payload first - it carries appliance_pin
    // for both meeting and idle states.
    const settingsResp = await fetch(`${API}/api/admin/settings`);
    if (!settingsResp.ok) {
      if (retries > 0) setTimeout(() => refreshWifiQR(retries - 1, delayMs), delayMs);
      return;
    }
    const settings = await settingsResp.json();
    const pin = settings.appliance_pin || "";
    if (!pin) {
      target.innerHTML = "";
      _lastRenderedPin = null;
      return;
    }
    if (pin === _lastRenderedPin) return;
    _lastRenderedPin = pin;
    // Editorial layout: a small chip plus a callout reveal-on-hover.
    // Tailwind utility classes align with the header components in
    // static/css/src/components/.
    target.innerHTML = `
      <div class="header-pin-chip" role="status" aria-label="Meeting ID">
        <span class="header-pin-label">Meeting ID</span>
        <span class="header-pin-value">${esc(pin)}</span>
      </div>
    `;
  } catch {
    if (retries > 0) setTimeout(() => refreshWifiQR(retries - 1, delayMs), delayMs);
  }
}
