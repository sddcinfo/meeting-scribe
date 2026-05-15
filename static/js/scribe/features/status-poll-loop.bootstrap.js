// Meeting Scribe — status-poll tick loop.
//
// Runs the /api/status `checkStatus()` paint on a tiered cadence:
//   * 2 s while recording or mid-start (busy)
//   * 10 s otherwise (idle)
//   * paused when the tab is hidden (`document.hidden`)
//   * immediate tick on visibilitychange (when the tab regains focus)
//   * skipped entirely on popout windows (their own status surface
//     lives in the popout SPA)
//
// The poll runs as a setTimeout chain rather than setInterval so each
// tick's actual run time doesn't compound — the next delay is scheduled
// from the END of the previous tick, not from a fixed wall clock.
//
// Dependency surface:
//   features/status-poll.js — `checkStatus()`

import { checkStatus } from "./status-poll.js";

const POPOUT_MODE = new URLSearchParams(location.search).get("popout");

let _pollTimer = null;

function _scheduleNextStatusTick() {
  if (document.hidden || POPOUT_MODE) return;
  const busy = document.body.classList.contains("recording")
            || document.body.classList.contains("starting");
  const delay = busy ? 2000 : 10000;
  _pollTimer = setTimeout(_statusTick, delay);
}

async function _statusTick() {
  try { await checkStatus(); } catch {}
  _scheduleNextStatusTick();
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  if (_pollTimer) {
    clearTimeout(_pollTimer);
    _pollTimer = null;
  }
  _statusTick();
});

if (!POPOUT_MODE) {
  checkStatus().then(_scheduleNextStatusTick);
}
