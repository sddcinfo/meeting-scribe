// Meeting Scribe — WiFi QR refresh (bootstrap).
//
// Wires the initial paint + the 10 s polling interval. The recording
// start/stop handlers call `refreshWifiQR()` by bare identifier; the
// window publish below is what they resolve against.

import { refreshWifiQR } from "./wifi-qr.js";

window.refreshWifiQR = refreshWifiQR;

refreshWifiQR();

// Poll every 10 s so every open view converges to the same live SSID.
// If rotation happens mid-session this catches up within 10 s.
setInterval(() => refreshWifiQR(0, 0), 10000);
