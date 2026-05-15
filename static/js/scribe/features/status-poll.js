// Meeting Scribe — /api/status poll handler.
//
// `checkStatus()` is the top-level admin-SPA status renderer. Driven
// from the 2-tier polling loop in the admin SPA boot orchestrator
// (2s while recording, 10s idle, paused on document.hidden, immediate
// tick on visibilitychange). On each tick it:
//
//   - Repaints the four backend-status pills (ASR, Translate,
//     Diarize, TTS) — colour + textContent + data-state
//   - Surfaces the TTS-stalled / -degraded / -circuit-broken sub-states
//   - Renders the admin notifications banner (mic auto-rebind /
//     ambiguous / unresolved / capture_failed)
//   - Mirrors the live elapsed time + the per-language WPM read-out
//     into the meeting-stats footer
//   - Tracks the start-button label so it's never stuck on "Starting…"
//     after an async failure
//   - Hands control off to the meeting-reconcile state machine
//     (`reconciler.reconcile(data)`), which decides whether to start
//     recording, attach the view-only WS, or display the takeover banner.
//
// On 401/403 the per-boot HMAC subkey has rotated (service restart,
// factory_reset). `_maybeRedirectOnAuthFailure` drops the cookie and
// hard-redirects to `/auth` so the operator immediately re-signs in
// instead of staring at a wall of 401s.
//
// Dependency surface — small + clean:
//   state.js          — `state`, `store`, `audio` singletons
//   live-stats.js     — `updateLiveStats` (admin-side stats footer)
//   admin-notifications.js — `renderAdminNotifications`
//
// `reconciler` is a let-binding in the admin SPA boot orchestrator
// (reassigned after construction). Configured here via the lazy-getter
// pattern so the async body always resolves the current reference.

import { audio, state, store } from "../state.js";
import { updateLiveStats as _updateLiveStatsRaw } from "./live-stats.js";
import { renderAdminNotifications } from "./admin-notifications.js";

const API = "";

let _deps = null;

export function configureStatusPoll(deps) {
  _deps = deps;
}

// Auth-failure handler shared between checkStatus and any other
// fetch that needs to detect "service restarted, my cookie no longer
// signs". On 401/403 the per-boot HMAC subkey rotated; the operator
// must re-sign in. Drop the cookie so the next reload doesn't re-send
// it, then redirect to ``/auth`` so the reconnect-after-restart flow
// is automatic instead of "operator stares at a console flooded with
// 401s and wonders why nothing works".
let _authRedirectInFlight = false;
function _maybeRedirectOnAuthFailure(resp) {
  if (!resp) return false;
  if (resp.status !== 401 && resp.status !== 403) return false;
  if (_authRedirectInFlight) return true;
  _authRedirectInFlight = true;
  // Best-effort: clear any client-side state that's now meaningless.
  try { document.cookie = 'scribe_admin=; Max-Age=0; Path=/; SameSite=Strict; Secure'; } catch {}
  // Skip the redirect if we're already on /auth (the sign-in form
  // itself uses /api/admin/authorize which never 401s through this
  // path; this guard is purely defensive against future callsites).
  if (location.pathname.startsWith('/auth')) return true;
  // Stash the current URL so post-sign-in can return to it. ``next``
  // is the existing convention used by ``routes/views.py``.
  const next = encodeURIComponent(location.pathname + location.search + location.hash);
  location.replace(`/auth?next=${next}`);
  return true;
}

// Thin wrapper so the body below can call `updateLiveStats(pct, tts)`
// without threading `store` / `state` through every call site.
function updateLiveStats(pct, ttsMetrics) {
  return _updateLiveStatsRaw(pct, ttsMetrics, store, state);
}

export async function checkStatus() {
  try {
    const resp = await fetch(`${API}/api/status`, { credentials: 'same-origin' });
    if (_maybeRedirectOnAuthFailure(resp)) return;
    const data = await resp.json();
    // Admin notifications banner (mic auto-rebind / ambiguous /
    // unresolved / capture_failed). Producer is
    // audio_routing.reconcile_audio_routing(); the SPA shows the
    // newest un-dismissed entry via the existing .meeting-banner CSS.
    if (Array.isArray(data.admin_notifications)) {
      try {
        renderAdminNotifications(data.admin_notifications);
      } catch (err) {
        console.warn('admin-notifications render failed', err);
      }
    }
    const pills = document.getElementById('backend-pills');
    const details = data.backend_details || {};
    const backendEntries = [
      ['ASR', 'asr', 'Speech recognition'],
      ['Translate', 'translate', 'Translation model'],
      ['Diarize', 'diarize', 'Speaker diarization'],
      ['TTS', 'tts', 'Interpretation audio'],
    ];
    // TTS health overlay — surfaces real-time lag/queue/drop state on top
    // of the basic "backend ready" flag. Computed server-side by
    // _tts_health_evaluator; this UI only renders the committed state.
    const ttsMetrics = (data.metrics && data.metrics.tts) || {};
    const ttsHealth = ttsMetrics.health || 'healthy';
    const ttsLagP95 = (ttsMetrics.end_to_end_lag_ms || {}).p95;
    const ttsDrops = ttsMetrics.drops || {};
    const ttsDropTotal = Object.values(ttsDrops).reduce((a, b) => a + (b || 0), 0);
    updateLiveStats(null, ttsMetrics);

    pills.innerHTML = backendEntries
      .map(([label, key, desc]) => {
        const d = details[key] || { ready: data.backends[key], status: data.backends[key] ? 'active' : 'down' };
        const ok = d.ready;
        let pillClass, icon, title;
        if (ok) {
          pillClass = 'active';
          icon = '';
          title = d.detail ? `${desc} — ${d.detail}` : `${desc} — active`;
        } else if (d.status === 'loading') {
          pillClass = 'loading';
          icon = ' ◐';
          title = d.detail ? `${desc} — ${d.detail}` : `${desc} — loading`;
        } else if (d.status === 'error') {
          pillClass = 'down';
          icon = ' ⚠';
          title = d.detail ? `${desc} — ERROR: ${d.detail}` : `${desc} — error`;
        } else if (d.status === 'restarting') {
          pillClass = 'loading';
          icon = ' ↻';
          title = `${desc} — restarting`;
        } else if (d.status === 'not started') {
          pillClass = 'down';
          icon = ' ✕';
          title = `${desc} — not started`;
        } else {
          pillClass = 'down';
          icon = ' ✕';
          title = d.detail ? `${desc} — ${d.detail}` : `${desc} — ${d.status || 'unavailable'}`;
        }
        // TTS health state overrides color when the backend is nominally
        // up but falling behind — degraded (amber) or stalled (red).
        if (key === 'tts' && ok) {
          if (ttsHealth === 'stalled') {
            pillClass = 'down tts-stalled';
            icon = ' ⚠';
          } else if (ttsHealth === 'degraded') {
            pillClass = 'loading tts-degraded';
            icon = ' ⚠';
          }
          const lagStr = (ttsLagP95 != null) ? ` · lag p95=${Math.round(ttsLagP95)}ms` : '';
          const q = ttsMetrics.queue_depth || 0;
          const qMax = ttsMetrics.queue_maxsize || 0;
          const busy = ttsMetrics.workers_busy || 0;
          const conc = ttsMetrics.container_concurrency || 0;
          title = `TTS — ${ttsHealth}${lagStr} · queue ${q}/${qMax} · busy ${busy}/${conc} · drops ${ttsDropTotal}`;
        }
        const detailSpan = (!ok && d.detail) ? `<span class="pill-detail">${d.detail}</span>` : '';
        return `<span class="pill ${pillClass}" title="${title}">${label}${icon}${detailSpan}</span>`;
      }).join('');

    // Stall toast: when TTS flips to "stalled", show a prominent one-line
    // warning. Auto-clears when health returns to healthy. We track the
    // last-seen state on a module-level var so transitions fire only once.
    if (window._lastTtsHealth !== ttsHealth) {
      if (ttsHealth === 'stalled') {
        const tb = document.getElementById('tts-toast');
        if (tb) {
          tb.textContent = 'TTS stalled — Listen may miss audio until backend recovers';
          tb.classList.add('visible');
        }
      } else if (window._lastTtsHealth === 'stalled' && ttsHealth === 'healthy') {
        const tb = document.getElementById('tts-toast');
        if (tb) { tb.classList.remove('visible'); tb.textContent = ''; }
      }
      window._lastTtsHealth = ttsHealth;
    }

    // ── Pre-emptive Start-button gate (W4 Fix #3) ─────────────────
    // Disable the Start button + show a "Backends warming up…" banner
    // while any REQUIRED backend (ASR or translate) is not yet ready.
    // The operator gets a visible "wait" affordance instead of clicking
    // Start during cold-start and discovering the 503 the hard way.
    // The backend's wait-with-deadline preflight covers the subtler
    // case of a backend that's "ready" by /v1/models but slow on the
    // first inference call; together they hide cold-start latency from
    // the operator entirely.
    {
      const asrReady = !!(details.asr && details.asr.ready);
      const translateReady = !!(details.translate && details.translate.ready);
      const allRequiredReady = asrReady && translateReady;
      const banner = document.getElementById('backends-warming-banner');
      const bannerDetail = document.getElementById('backends-warming-detail');
      const startBtn = document.getElementById('btn-start-meeting');
      if (banner && startBtn) {
        if (allRequiredReady) {
          banner.style.display = 'none';
          // Don't override `disabled` set by other gates (e.g. mid-start
          // double-click guard) — only re-enable if the gate set it.
          if (startBtn.dataset.gatedByWarmup === '1') {
            startBtn.disabled = false;
            delete startBtn.dataset.gatedByWarmup;
          }
        } else {
          const waiting = [];
          if (!asrReady) waiting.push(`ASR (${(details.asr && details.asr.detail) || 'loading'})`);
          if (!translateReady) waiting.push(`Translate (${(details.translate && details.translate.detail) || 'loading'})`);
          if (bannerDetail) bannerDetail.textContent = waiting.join(' · ');
          banner.style.display = '';
          startBtn.disabled = true;
          startBtn.dataset.gatedByWarmup = '1';
        }
      }
    }

    // Crash red-dot on the server pill (if backend reported an unhandled
    // exception from a background task). Sanitised — only ts/component/code.
    const crash = (data.metrics || {}).crash;
    if (crash && window._lastCrashCode !== crash.code) {
      const tb = document.getElementById('tts-toast');
      if (tb) {
        tb.textContent = `server crash in ${crash.component} · code ${crash.code}`;
        tb.classList.add('visible', 'crash');
      }
      window._lastCrashCode = crash.code;
    }
    // ── Backend detail panel ──────────────────────────────────
    const bdPanel = document.getElementById('backend-detail-panel');
    if (bdPanel) {
      bdPanel.innerHTML = backendEntries.map(([label, key, desc]) => {
        const d = details[key] || {};
        const dotCls = d.ready ? 'active' : (d.status === 'loading' ? 'loading' : (d.status === 'error' ? 'error' : 'down'));
        let rows = '';
        if (d.model) rows += `<div class="bd-row"><span class="bd-label">Model</span><span class="bd-val">${d.model}</span></div>`;
        if (d.url) rows += `<div class="bd-row"><span class="bd-label">URL</span><span class="bd-val">${d.url}</span></div>`;
        if (d.consecutive_failures > 0) rows += `<div class="bd-row"><span class="bd-label">Failures</span><span class="bd-val bd-val-crit">${d.consecutive_failures}</span></div>`;
        // TTS-specific metrics
        if (key === 'tts') {
          const t = ttsMetrics;
          rows += `<div class="bd-row"><span class="bd-label">Queue</span><span class="bd-val">${t.queue_depth || 0}/${t.queue_maxsize || 0}</span></div>`;
          rows += `<div class="bd-row"><span class="bd-label">Workers</span><span class="bd-val">${t.workers_busy || 0}/${t.workers_total || 0}</span></div>`;
          rows += `<div class="bd-row"><span class="bd-label">Delivered</span><span class="bd-val">${t.delivered || 0}</span></div>`;
          if (ttsDropTotal > 0) rows += `<div class="bd-row"><span class="bd-label">Drops</span><span class="bd-val bd-val-crit">${ttsDropTotal}</span></div>`;
          const lag = (t.end_to_end_lag_ms || {}).p95;
          if (lag != null) rows += `<div class="bd-row"><span class="bd-label">Lag P95</span><span class="bd-val">${Math.round(lag)}ms</span></div>`;
        }
        rows += `<div class="bd-row"><span class="bd-label">Status</span><span class="bd-val">${d.status || 'unknown'}</span></div>`;
        const errDiv = (!d.ready && d.detail) ? `<div class="bd-error">${d.detail}</div>` : '';
        return `<div class="bd-card"><div class="bd-name"><span class="bd-dot ${dotCls}"></span>${label}</div>${rows}${errDiv}</div>`;
      }).join('');
    }

    // ── System resources panel ─────────────────────────────────
    const sysPanel = document.getElementById('system-resources-panel');
    const sys = data.system;
    if (sysPanel && sys) {
      // Threshold → CSS class instead of inline ``style="color:..."`` —
      // the strict CSP (``style-src 'self'``) blocks every inline style
      // attribute, and the previous template injected one on every
      // status poll, producing a continuous spam of CSP violation
      // errors in the operator console. ``.bd-val-warn`` and
      // ``.bd-val-crit`` are the equivalent classes in style.css.
      const cpuCls = sys.cpu_pct > 90 ? ' class="bd-val bd-val-crit"' : sys.cpu_pct > 70 ? ' class="bd-val bd-val-warn"' : ' class="bd-val"';
      const memCls = sys.mem_pct > 90 ? ' class="bd-val bd-val-crit"' : sys.mem_pct > 75 ? ' class="bd-val bd-val-warn"' : ' class="bd-val"';
      let sysHtml = `<div class="bd-card"><div class="bd-name"><span class="bd-dot active"></span>System</div>`;
      sysHtml += `<div class="bd-row"><span class="bd-label">CPU</span><span${cpuCls}>${sys.cpu_pct}%</span></div>`;
      sysHtml += `<div class="bd-row"><span class="bd-label">Memory</span><span${memCls}>${Math.round(sys.mem_used_mb/1024)}/${Math.round(sys.mem_total_mb/1024)} GB (${sys.mem_pct}%)</span></div>`;
      sysHtml += `<div class="bd-row"><span class="bd-label">Load</span><span class="bd-val">${sys.load.join(' / ')}</span></div>`;
      const days = Math.floor(sys.uptime_s / 86400);
      const hrs = Math.floor((sys.uptime_s % 86400) / 3600);
      sysHtml += `<div class="bd-row"><span class="bd-label">Uptime</span><span class="bd-val">${days}d ${hrs}h</span></div>`;
      sysHtml += `</div>`;

      // Per-container resource cards
      if (sys.containers && sys.containers.length > 0) {
        for (const c of sys.containers) {
          const cCpuCls = c.cpu_pct > 200 ? ' class="bd-val bd-val-crit"' : c.cpu_pct > 100 ? ' class="bd-val bd-val-warn"' : ' class="bd-val"';
          sysHtml += `<div class="bd-card"><div class="bd-name">${c.name}</div>`;
          sysHtml += `<div class="bd-row"><span class="bd-label">CPU</span><span${cCpuCls}>${c.cpu_pct}%</span></div>`;
          sysHtml += `<div class="bd-row"><span class="bd-label">Memory</span><span class="bd-val">${c.mem_mb > 1024 ? (c.mem_mb/1024).toFixed(1) + ' GB' : c.mem_mb + ' MB'}</span></div>`;
          sysHtml += `<div class="bd-row"><span class="bd-label">PIDs</span><span class="bd-val">${c.pids}</span></div>`;
          sysHtml += `</div>`;
        }
      }
      sysPanel.innerHTML = sysHtml;
    }

    const m = data.metrics || {};
    // REQUIRED = ASR + Translation. Diarize/TTS are nice-to-have but not gating.
    // This matches the server-side gate in /api/meeting/start.
    const ready = data.backends.asr && data.backends.translate;
    const _btnRecord = document.getElementById('btn-record');
    if (_btnRecord) {
      _btnRecord.disabled = !ready;
      _btnRecord.title = ready
        ? 'Start recording'
        : 'Required backends not ready yet — meeting start is blocked';
    }

    // Show clear message when backends are loading (any of the 4, not just required)
    const notReady = backendEntries
      .filter(([, key]) => !(details[key] || {}).ready);
    if (notReady.length > 0) {
      const waiting = notReady
        .map(([label, key]) => {
          const d = details[key] || {};
          const err = d.status === 'error' ? ' ERROR' : '';
          return d.detail ? `${label}${err}: ${d.detail}` : `${label}${err}`;
        }).join(' · ');
      const prefix = ready
        ? 'Ready (optional backends loading)'
        : 'Waiting for backends';
      document.getElementById('status-line').textContent = `${prefix}: ${waiting}`;
    }

    // State transitions (rehydration, banner, title, ownership) are
    // delegated to the reconciler; see static/js/meeting-reconcile.js.
    // When backends aren't ready yet, keep the old status text so the
    // user sees "Warming up..." instead of a silent idle state.
    if (!(data.meeting?.state === 'recording') && !ready) {
      document.getElementById('status-line').textContent = 'Warming up...';
    } else if (!(data.meeting?.state === 'recording')) {
      document.getElementById('status-line').textContent = 'Ready';
    }
    // Lazy getter — `reconciler` is constructed by the admin SPA boot
    // after this module evaluates. Resolve at call time so the
    // binding is always current.
    const reconciler = _deps?.getReconciler?.();
    if (reconciler) {
      await reconciler.reconcile(data);
    }
  } catch (err) {
    console.error('checkStatus failed:', err);
    document.getElementById('status-line').textContent = 'Server unreachable';
    const _btnRecord = document.getElementById('btn-record');
    if (_btnRecord) _btnRecord.disabled = true;
  }
}
