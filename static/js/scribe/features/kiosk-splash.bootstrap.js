// Meeting Scribe - HDMI kiosk idle splash boot wiring.
//
// Runs only when the body carries ``data-role="kiosk"``. Paints the
// appliance PIN once on load, then refreshes every 30 s (cheap
// belt-and-suspenders against the appliance_id rotating via
// factory_reset). The splash visibility is controlled entirely by
// the cascade in ``components/_state.css``; this bootstrap only
// owns the PIN text.

import { refreshKioskSplashPin } from "./kiosk-splash.js";

(() => {
  if (typeof document === "undefined") return;
  if (document.body.dataset.role !== "kiosk") return;
  refreshKioskSplashPin();
  setInterval(refreshKioskSplashPin, 30_000);
})();
