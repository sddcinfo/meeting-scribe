// Meeting Scribe — SegmentStore baseline subscribers + test hook.
//
// Two cross-cutting subscribers that have to wire into the live
// SegmentStore as early as the rest of the SPA boots:
//
//   1. `#segment-count` chip — admin SPA's footer meter UI shows
//      "N segments" live. Popout windows render no such element;
//      the listener guards against null so it doesn't throw on every
//      fan-out. (Before SegmentStore added per-listener try/catch a
//      throw here aborted iteration and silently broke every
//      subsequent subscriber — including the CompactGridRenderer one.)
//
//   2. Detected-speakers tracker — any new REAL `cluster_id` seen in
//      an event registers in the speaker registry, which drives the
//      virtual-table seat list. Pseudo-clusters (time_proximity
//      fallbacks) are ignored so transient "Speaker 1 → Speaker 2"
//      shuffles during the catch-up loop don't leak into the UI.
//
// Plus the `?test=1` URL-flag test hook that publishes
// `window.__test_store` + ingestion counters for the
// cross-window-sync browser tests. (Production sessions never carry
// `?test=1`, so this is a strict no-op there.)
//
// `_refreshDetectedSpeakersStrip` lives in the admin SPA boot
// orchestrator because it closes over the `roomSetup` singleton
// constructed there; we take it via a one-shot
// `configureSegmentStoreSubscribers({…})` so the bootstrap can call
// it at the right point without circular imports.

import { store } from "../state.js";
import {
  _speakerRegistry,
  getSpeakerDisplayName,
  isPseudoCluster as _isPseudoCluster,
  refreshTranscriptSpeakerLabels as _refreshTranscriptSpeakerLabels,
} from "./speaker-registry.js";

let _deps = null;

export function configureSegmentStoreSubscribers(deps) {
  _deps = deps;
}

// Browser-test hook. Exposed only when `?test=1` is in the URL — keeps
// the store off `window` in normal sessions but lets Playwright tests
// observe `window.__test_store` and friends without monkey-patching
// modules. Used by tests/browser/test_cross_window_sync.py.
if (new URLSearchParams(location.search).get("test") === "1") {
  window.__test_store = store;
  window.__test_ingest_count = 0;
  window.__test_msg_log = [];
}

// 1. Segment-count chip updater (admin SPA footer meter).
store.subscribe(() => {
  const el = document.getElementById("segment-count");
  if (el) el.textContent = `${store.count} segments`;
});

// 2. CompactGridRenderer feed — forwards every ingested event into the
//    grid so the admin transcript surface stays current. Active-speaker
//    highlighting on the table strip is driven exclusively by the
//    server's `speaker_pulse` broadcast (every 200ms), so we
//    deliberately don't add a second sticky-timer path here that would
//    race the pulse loop and leave the previous speaker lit during
//    back-and-forth.
store.subscribe((id, evt) => {
  if (window._gridRenderer) window._gridRenderer.update(id, evt);
});

// 3. Detected-speakers tracker → speaker registry → table strip.
store.subscribe((id, evt) => {
  if (!evt || !evt.speakers?.length) return;
  const s = evt.speakers[0];
  const cid = s.cluster_id;
  if (cid == null) return;
  if (_isPseudoCluster(cid)) return; // transient — don't register
  const wasKnown = _speakerRegistry.clusters.has(cid);
  const prevName = _speakerRegistry.clusters.get(cid)?.displayName;
  // Register / update (honors explicit names too)
  const newName = getSpeakerDisplayName(cid, s.identity || s.display_name);
  if (!wasKnown || prevName !== newName) {
    // New speaker detected OR renamed — refresh the live speaker strip
    _deps?.refreshDetectedSpeakersStrip?.();
    // Also refresh already-rendered transcript blocks so their speaker
    // labels reflect the current registry state
    _refreshTranscriptSpeakerLabels();
  }
});
