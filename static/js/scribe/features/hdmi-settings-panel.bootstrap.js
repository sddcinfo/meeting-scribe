// Meeting Scribe - HDMI settings panel bootstrap.
//
// Wires the panel once the settings tab machinery is in the DOM. Also
// re-runs the hydrate on every "settings tab opened" event so the
// status block shows fresh wlr-randr output every time the operator
// visits the tab.

import { initHdmiPanel } from "./hdmi-settings-panel.js";

(() => {
  if (typeof document === "undefined") return;
  // Initial wire-up.
  initHdmiPanel();
  // Re-hydrate when the user clicks the Display tab so the status
  // block isn't stale.
  const tabBtn = document.getElementById("settings-tab-display");
  if (tabBtn && !tabBtn.dataset.kioskBound) {
    tabBtn.dataset.kioskBound = "1";
    tabBtn.addEventListener("click", () => {
      // Re-init reads + binds; safe to call repeatedly.
      initHdmiPanel();
    });
  }
})();
