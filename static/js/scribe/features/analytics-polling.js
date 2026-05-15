// Meeting Scribe — analytics polling (pure module).
//
// 2-second `/api/status` poll that drives the live metrics dashboard
// when `body.metrics-split` is on. Only fires the network call when
// the dashboard is visible; otherwise the polling loop is a cheap
// no-op. Stopped by `stopAnalytics()` when the meeting ends.

import { updateMetricsDashboard } from "./metrics-dashboard.js";

let _analyticsInterval = null;

export function startAnalytics() {
  if (_analyticsInterval) return;
  _analyticsInterval = setInterval(async () => {
    try {
      const data = await (await fetch("/api/status")).json();
      // Update the togglable metrics dashboard if it's visible. The
      // header has no inline metric chips anymore — everything lives
      // in the dashboard panel (live meetings) or the finalization
      // modal (past).
      if (document.body.classList.contains("metrics-split")) {
        updateMetricsDashboard(data);
      }
    } catch {
      /* swallow — next tick retries */
    }
  }, 2000);
}

export function stopAnalytics() {
  if (_analyticsInterval) {
    clearInterval(_analyticsInterval);
    _analyticsInterval = null;
  }
}
