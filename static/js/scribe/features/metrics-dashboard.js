// Meeting Scribe — Metrics dashboard (pure module).
//
// Re-paints the slide-over metrics panel ("ASR / Translate / Voice
// Activity / WS / Backend health") from a /api/status payload.
// Called from the /api/status polling loop whenever the metrics
// dashboard is open. Pure DOM updater — no fetch, no state, no
// window publish.

/**
 * Show / hide the slide-over metrics dashboard and mirror its
 * visibility on body.metrics-split + the #btn-metrics aria-state.
 *
 * Called from three lifecycle paths (recording start, finalize,
 * view-mode entry) and from the btn-metrics click handler in the
 * control-bar-toggles bootstrap.
 */
export function setMetricsVisible(visible, button) {
  const dash = document.getElementById("metrics-dashboard");
  const btn = button || document.getElementById("btn-metrics");
  document.body.classList.toggle("metrics-split", !!visible);
  if (dash) dash.style.display = visible ? "" : "none";
  if (btn) {
    btn.classList.toggle("active-toggle", !!visible);
    btn.setAttribute("aria-expanded", String(!!visible));
  }
}

export function updateMetricsDashboard(data) {
  const m = data.metrics || {};
  const gpu = data.gpu;
  const b = data.backends || {};

  const set = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  // ASR
  const asrEps = (m.asr_eps || 0).toFixed(1);
  set("mc-asr", `${m.asr_finals || 0}`);
  set("mc-asr-sub", `${asrEps} evt/s`);

  // Translation
  const avgMs = m.avg_translation_ms || 0;
  set(
    "mc-trans",
    `${m.translations_completed || 0}/${m.translations_submitted || 0}`,
  );
  set("mc-trans-sub", avgMs > 0 ? `avg ${avgMs}ms` : "");

  // VRAM bar
  const vramPct = gpu ? gpu.vram_pct || 0 : 0;
  set(
    "mc-vram",
    gpu
      ? `${gpu.vram_used_mb || 0}MB / ${gpu.vram_total_mb || 0}MB (${vramPct.toFixed(0)}%)`
      : "—",
  );
  const vramBar = document.getElementById("mc-vram-bar");
  if (vramBar) {
    vramBar.style.width = `${vramPct}%`;
    vramBar.className = `metric-bar ${vramPct > 90 ? "metric-bar-danger" : vramPct > 75 ? "metric-bar-warn" : ""}`;
  }

  // Audio
  set("mc-audio", `${(m.audio_s || 0).toFixed(0)}s`);

  // Connections
  const wsCount = (data.connections || 0) + (data.audio_out_connections || 0);
  set("mc-health", `${wsCount}`);

  // Backends
  const backendStr = Object.entries(b)
    .map(
      ([k, v]) =>
        `<span class="backend-badge ${v ? "backend-on" : "backend-off"}">${k}</span>`,
    )
    .join(" ");
  const backendsEl = document.getElementById("mc-backends");
  if (backendsEl) backendsEl.innerHTML = backendStr;

  // W5 — reliability tiles. Color-coded via CSS class so warn/crit
  // states pop visually before they kill a meeting.
  const setColored = (id, text, level) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.className = `metric-card-value ${level === "crit" ? "metric-value-crit" : level === "warn" ? "metric-value-warn" : ""}`;
  };

  // ASR RTT p95 — backend control-path latency. Warn >800 ms,
  // crit >2000 ms.
  const asrRtt = (m.asr_request_rtt_ms && m.asr_request_rtt_ms.p95) || null;
  if (asrRtt === null) {
    setColored("mc-asr-rtt-p95", "—", "ok");
  } else {
    const lvl = asrRtt > 2000 ? "crit" : asrRtt > 800 ? "warn" : "ok";
    setColored("mc-asr-rtt-p95", `${asrRtt.toFixed(0)}ms`, lvl);
  }

  // Watchdog fires per minute. Warn >0.5/min, crit >2/min.
  const wdRate = m.watchdog_fires_per_min || 0;
  const wdLvl = wdRate > 2 ? "crit" : wdRate > 0.5 ? "warn" : "ok";
  setColored("mc-watchdog-fires", `${wdRate}`, wdLvl);

  // Time since last ASR final. Warn >5 s, crit >15 s. null on a fresh
  // meeting (no finals yet) — render '—' rather than alarming.
  const tsf = m.time_since_last_final_s;
  if (tsf === null || tsf === undefined) {
    setColored("mc-since-final", "—", "ok");
  } else {
    const lvl = tsf > 15 ? "crit" : tsf > 5 ? "warn" : "ok";
    setColored("mc-since-final", `${tsf.toFixed(1)}s`, lvl);
  }

  // GPU free MB. Warn <8 GB, crit <4 GB. Inverse of the other tiles.
  const freeMb = gpu ? gpu.vram_free_mb || 0 : null;
  if (freeMb === null) {
    setColored("mc-gpu-free", "—", "ok");
  } else {
    const lvl = freeMb < 4096 ? "crit" : freeMb < 8192 ? "warn" : "ok";
    const gb = (freeMb / 1024).toFixed(1);
    setColored("mc-gpu-free", `${gb} GB`, lvl);
  }
}
