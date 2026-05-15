// Meeting Scribe — live stats DOM helpers.
//
// Pure renderers for the header-stats popover plus the segment-count
// chip on the meeting control bar. Takes the segment store and the
// shared `state` object as args so this module stays free of any
// hidden module-scope coupling.
//
// The mic-source indicator that used to live here (applyMicSourceIndicator
// + the `[data-mic-source="server"]` data-attr toggle + the "Server mic"
// pill) was retired alongside the meter-bar going server-driven: the
// `mic_level` WS event now drives the bar directly from the actual
// active mic source, so a "browser vs server" disambiguation pill is
// no longer needed.

export function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function updateLiveStats(pct, ttsMetrics, store, state) {
  setText("segment-count", `${store.count} segments`);
  setText("stat-segs", String(store.count));
  setText("stat-ws", String(state.wsMessageCount));
  setText("stat-chunks", String(state.audioChunkCount));
  if (pct != null) setText("stat-mic", `${pct}%`);

  if (ttsMetrics) {
    const q = ttsMetrics.queue_depth || 0;
    const qMax = ttsMetrics.queue_maxsize || 0;
    const busy = ttsMetrics.workers_busy || 0;
    const conc = ttsMetrics.container_concurrency || ttsMetrics.workers_total || 0;
    const lagP95 = (ttsMetrics.end_to_end_lag_ms || {}).p95;
    const drops = Object.values(ttsMetrics.drops || {}).reduce(
      (a, b) => a + (b || 0),
      0,
    );
    setText("stat-tts-queue", `${q}/${qMax}`);
    setText("stat-tts-busy", `${busy}/${conc}`);
    setText("stat-tts-lag", lagP95 == null ? "-" : `${Math.round(lagP95)}ms`);
    setText("stat-tts-drops", String(drops));
    const dot = document.getElementById("stats-health-dot");
    if (dot) {
      const health = ttsMetrics.health || "healthy";
      dot.classList.toggle("visible", health !== "healthy" || q > 0 || busy > 0);
      dot.classList.toggle("warn", health === "degraded");
      dot.classList.toggle("err", health === "stalled");
    }
  }
}
