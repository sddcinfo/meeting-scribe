/* Diagnostics panel — system health + log viewer.
 *
 * First-class native UI for inspecting backend health, GPU/system
 * resources, container state, recent warnings/errors and the live
 * server log. Surfaces everything that previously required SSH or
 * the embedded terminal so users can investigate issues from the
 * browser. All data sources are admin-gated server-side.
 *
 * Tabs:
 *   • Health      — /api/status snapshot, polled every 5 s
 *   • Issues      — /api/diag/issues, polled every 5 s
 *   • Server log  — /api/diag/log/server/stream (SSE) live tail
 *   • Meeting log — /api/meetings/{id}/terminal-log on demand
 */

(function () {
  'use strict';

  const HEALTH_POLL_MS = 5000;
  const ISSUES_POLL_MS = 5000;

  const $ = (id) => document.getElementById(id);

  const state = {
    panelOpen: false,
    activeTab: 'health',
    healthTimer: null,
    issuesTimer: null,
    issuesSinceId: 0,
    issuesAccum: [],
    sse: null,
    forbidden: false,
    lastStatus: null,
  };

  // ── Boot ────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);

  function init() {
    const btn = $('btn-diagnostics');
    const close = $('btn-diagnostics-close');
    const backdrop = $('diagnostics-backdrop');
    if (!btn || !close || !backdrop) return;

    btn.addEventListener('click', open);
    close.addEventListener('click', closePanel);
    backdrop.addEventListener('click', closePanel);
    document.addEventListener('keydown', onKey);

    // Tab switcher
    document.querySelectorAll('.diag-tab').forEach((tab) => {
      tab.addEventListener('click', () => activateTab(tab.dataset.tab));
    });

    // Toolbar wiring
    const reauth = $('diag-go-authorize');
    if (reauth) reauth.addEventListener('click', () => {
      closePanel();
      const sb = $('btn-settings');
      if (sb) sb.click();
    });

    // Issues controls
    $('diag-issues-level')?.addEventListener('change', renderIssues);
    $('diag-issues-component')?.addEventListener('input', renderIssues);
    $('diag-issues-refresh')?.addEventListener('click', () => fetchIssues({ reset: false }));
    $('diag-issues-clear')?.addEventListener('click', () => {
      state.issuesAccum = [];
      renderIssues();
    });
    $('diag-issues-autorefresh')?.addEventListener('change', (e) => {
      if (e.target.checked) startIssuesPoll();
      else stopIssuesPoll();
    });

    // Server-log controls
    $('diag-srvlog-level')?.addEventListener('change', refetchServerLogTail);
    $('diag-srvlog-search')?.addEventListener('input', debounce(refetchServerLogTail, 250));
    $('diag-srvlog-live')?.addEventListener('change', (e) => {
      if (e.target.checked) startServerLogStream();
      else stopServerLogStream();
    });
    $('diag-srvlog-wrap')?.addEventListener('change', (e) => {
      const v = $('diag-srvlog-viewer');
      if (v) v.dataset.wrap = e.target.checked ? 'true' : 'false';
    });

    // Meeting-log controls
    $('diag-mtglog-select')?.addEventListener('change', loadMeetingLog);
    $('diag-mtglog-refresh')?.addEventListener('click', loadMeetingLog);
    $('diag-mtglog-wrap')?.addEventListener('change', (e) => {
      const v = $('diag-mtglog-viewer');
      if (v) v.dataset.wrap = e.target.checked ? 'true' : 'false';
    });

    // Background health probe — runs even when the panel is closed so
    // the header dot can flag trouble before the user opens diagnostics.
    pollHealthDot();
    setInterval(pollHealthDot, HEALTH_POLL_MS);
  }

  // ── Panel open/close ────────────────────────────────────────────
  function open() {
    const panel = $('diagnostics-panel');
    const backdrop = $('diagnostics-backdrop');
    const btn = $('btn-diagnostics');
    if (!panel || !backdrop) return;
    panel.classList.add('open');
    backdrop.classList.add('open');
    panel.setAttribute('aria-hidden', 'false');
    if (btn) btn.setAttribute('aria-expanded', 'true');
    state.panelOpen = true;
    activateTab(state.activeTab);
    panel.focus();
  }

  function closePanel() {
    const panel = $('diagnostics-panel');
    const backdrop = $('diagnostics-backdrop');
    const btn = $('btn-diagnostics');
    if (!panel || !backdrop) return;
    panel.classList.remove('open');
    backdrop.classList.remove('open');
    panel.setAttribute('aria-hidden', 'true');
    if (btn) btn.setAttribute('aria-expanded', 'false');
    state.panelOpen = false;
    stopHealthPoll();
    stopIssuesPoll();
    stopServerLogStream();
  }

  function onKey(e) {
    if (e.key === 'Escape' && state.panelOpen) closePanel();
  }

  // ── Tabs ────────────────────────────────────────────────────────
  function activateTab(name) {
    state.activeTab = name;
    document.querySelectorAll('.diag-tab').forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    document.querySelectorAll('.diag-tab-panel').forEach((p) => {
      p.classList.toggle('active', p.dataset.tabPanel === name);
    });
    // Stop any tab-specific work that's now in the background.
    stopHealthPoll();
    stopIssuesPoll();
    stopServerLogStream();

    if (name === 'health') {
      pollHealth();
      state.healthTimer = setInterval(pollHealth, HEALTH_POLL_MS);
    } else if (name === 'issues') {
      fetchIssues({ reset: true });
      if ($('diag-issues-autorefresh')?.checked) startIssuesPoll();
    } else if (name === 'server-log') {
      refetchServerLogTail();
      if ($('diag-srvlog-live')?.checked) startServerLogStream();
    } else if (name === 'meeting-log') {
      loadMeetingsList();
    }
  }

  // ── 403 handling ────────────────────────────────────────────────
  function setForbidden(yes) {
    state.forbidden = yes;
    const banner = $('diag-forbidden');
    const body = $('diag-body');
    if (banner) banner.hidden = !yes;
    if (body) body.style.display = yes ? 'none' : '';
  }

  async function adminFetch(url, opts) {
    const r = await fetch(url, { credentials: 'include', ...(opts || {}) });
    if (r.status === 403) {
      setForbidden(true);
      throw new Error('forbidden');
    }
    setForbidden(false);
    return r;
  }

  // ── Health tab ──────────────────────────────────────────────────
  function stopHealthPoll() {
    if (state.healthTimer) {
      clearInterval(state.healthTimer);
      state.healthTimer = null;
    }
  }

  async function pollHealth() {
    try {
      const r = await adminFetch('/api/status');
      const data = await r.json();
      state.lastStatus = data;
      renderHealth(data);
      updateHeaderDot(data);
    } catch (e) {
      if (e.message !== 'forbidden') console.warn('[diag] health poll failed', e);
    }
  }

  // Lightweight version that runs continuously to drive the header dot.
  // Falls back to /api/status (which returns a guest-safe payload for
  // non-admins) so we never end up with the dot frozen at "unknown".
  async function pollHealthDot() {
    try {
      const r = await fetch('/api/status', { credentials: 'include' });
      if (!r.ok) return;
      const data = await r.json();
      updateHeaderDot(data);
    } catch {}
  }

  function updateHeaderDot(data) {
    const dot = $('diag-health-dot');
    if (!dot) return;
    let level = 'ok';
    if (data?.crash) level = 'crit';
    const backends = (data?.backend_details) || data?.backends || {};
    for (const k of Object.keys(backends)) {
      const b = backends[k] || {};
      if (b.consecutive_failures && b.consecutive_failures > 0) level = (level === 'crit') ? 'crit' : 'warn';
      if (b.ready === false) level = (level === 'crit') ? 'crit' : 'warn';
    }
    dot.dataset.level = level;
  }

  function renderHealth(data) {
    // Crash banner
    const crashBanner = $('diag-crash-banner');
    const crashDetail = $('diag-crash-detail');
    if (crashBanner && crashDetail) {
      if (data?.crash) {
        crashBanner.hidden = false;
        const c = data.crash;
        crashDetail.textContent = ` Component: ${c.component || '?'} · code ${c.code || '?'} · t+${(c.ts || 0).toFixed(1)}s`;
      } else {
        crashBanner.hidden = true;
      }
    }

    // Backends
    const grid = $('diag-backends-grid');
    if (grid) {
      const backends = data?.backend_details || {};
      const keys = Object.keys(backends).sort();
      if (!keys.length) {
        grid.innerHTML = '<div class="diag-empty">No backend details available.</div>';
      } else {
        grid.innerHTML = keys.map((k) => backendCardHTML(k, backends[k])).join('');
      }
    }

    // System
    const sys = $('diag-system-grid');
    if (sys) {
      const s = data?.system || {};
      const g = data?.gpu || {};
      sys.innerHTML = [
        statCardHTML('CPU', `${(s.cpu_pct ?? 0).toFixed?.(1) ?? s.cpu_pct ?? 0}%`, gauge(s.cpu_pct)),
        statCardHTML('Memory', `${fmtMb(s.mem_used_mb)} / ${fmtMb(s.mem_total_mb)}`, gauge(s.mem_pct)),
        statCardHTML('GPU VRAM', `${fmtMb(g.vram_used_mb)} / ${fmtMb(g.vram_total_mb)}`, gauge(g.vram_pct)),
        statCardHTML('Load avg', s.load ? s.load.map((x) => x.toFixed(2)).join(' / ') : '—'),
        statCardHTML('Uptime', s.uptime_s ? fmtDuration(s.uptime_s) : '—'),
      ].join('');
    }

    // Containers
    const cont = $('diag-containers-grid');
    if (cont) {
      const list = data?.system?.containers || [];
      if (!list.length) {
        cont.innerHTML = '<div class="diag-empty">No container metrics.</div>';
      } else {
        cont.innerHTML = list.map(containerCardHTML).join('');
      }
    }

    // Loop lag
    const loop = $('diag-loop-grid');
    if (loop) {
      const l = data?.loop_lag_ms || {};
      loop.innerHTML = [
        statCardHTML('p50', l.p50 != null ? `${l.p50.toFixed(0)} ms` : '—'),
        statCardHTML('p95', l.p95 != null ? `${l.p95.toFixed(0)} ms` : '—', l.p95 > 250 ? 'warn' : 'ok'),
        statCardHTML('p99', l.p99 != null ? `${l.p99.toFixed(0)} ms` : '—', l.p99 > 500 ? 'warn' : 'ok'),
        statCardHTML('max', l.max != null ? `${l.max.toFixed(0)} ms` : '—', l.max > 1500 ? 'crit' : (l.max > 500 ? 'warn' : 'ok')),
      ].join('');
    }
  }

  function backendCardHTML(name, b) {
    const ready = b?.ready === true;
    const fails = b?.consecutive_failures || 0;
    const level = ready ? (fails ? 'warn' : 'ok') : 'crit';
    const detail = b?.detail ? esc(b.detail) : '';
    const model = b?.model ? `<div class="diag-card-line"><span>model</span><code>${esc(b.model)}</code></div>` : '';
    const url = b?.url ? `<div class="diag-card-line"><span>url</span><code>${esc(b.url)}</code></div>` : '';
    const failsLine = fails ? `<div class="diag-card-line"><span>consecutive failures</span><strong>${fails}</strong></div>` : '';
    return `
      <div class="diag-card" data-level="${level}">
        <div class="diag-card-head">
          <span class="diag-card-dot"></span>
          <span class="diag-card-title">${esc(name)}</span>
          <span class="diag-card-status">${ready ? 'ready' : 'not ready'}</span>
        </div>
        ${detail ? `<div class="diag-card-detail">${detail}</div>` : ''}
        ${model}${url}${failsLine}
      </div>
    `;
  }

  function containerCardHTML(c) {
    const status = (c.status || '').toLowerCase();
    let level = 'ok';
    if (status.includes('exit') || status.includes('dead')) level = 'crit';
    else if (status.includes('restart') || status.includes('unhealthy')) level = 'warn';
    return `
      <div class="diag-card" data-level="${level}">
        <div class="diag-card-head">
          <span class="diag-card-dot"></span>
          <span class="diag-card-title">${esc(c.name || c.id || 'container')}</span>
          <span class="diag-card-status">${esc(c.status || '')}</span>
        </div>
        ${c.cpu_pct != null ? `<div class="diag-card-line"><span>CPU</span><strong>${(+c.cpu_pct).toFixed(1)}%</strong></div>` : ''}
        ${c.mem_mb != null ? `<div class="diag-card-line"><span>Mem</span><strong>${fmtMb(c.mem_mb)}</strong></div>` : ''}
        ${c.image ? `<div class="diag-card-line"><span>image</span><code>${esc(c.image)}</code></div>` : ''}
      </div>
    `;
  }

  function statCardHTML(label, value, levelOrPct) {
    let level = 'ok';
    let bar = '';
    if (typeof levelOrPct === 'number') {
      if (levelOrPct >= 90) level = 'crit';
      else if (levelOrPct >= 70) level = 'warn';
      bar = `<div class="diag-card-bar"><span style="width:${Math.min(100, Math.max(0, levelOrPct))}%"></span></div>`;
    } else if (typeof levelOrPct === 'string') {
      level = levelOrPct;
    }
    return `
      <div class="diag-card diag-card-stat" data-level="${level}">
        <div class="diag-card-stat-label">${esc(label)}</div>
        <div class="diag-card-stat-value">${esc(String(value))}</div>
        ${bar}
      </div>
    `;
  }

  function gauge(pct) {
    const n = typeof pct === 'number' ? pct : Number(pct) || 0;
    return n;
  }

  // ── Issues tab ──────────────────────────────────────────────────
  function startIssuesPoll() {
    stopIssuesPoll();
    state.issuesTimer = setInterval(() => fetchIssues({ reset: false }), ISSUES_POLL_MS);
  }
  function stopIssuesPoll() {
    if (state.issuesTimer) {
      clearInterval(state.issuesTimer);
      state.issuesTimer = null;
    }
  }

  async function fetchIssues({ reset }) {
    if (reset) {
      state.issuesSinceId = 0;
      state.issuesAccum = [];
    }
    const params = new URLSearchParams();
    params.set('limit', '500');
    if (state.issuesSinceId) params.set('since_id', String(state.issuesSinceId));
    try {
      const r = await adminFetch(`/api/diag/issues?${params}`);
      const data = await r.json();
      const arr = data?.events || [];
      if (arr.length) {
        state.issuesAccum = state.issuesAccum.concat(arr);
        if (state.issuesAccum.length > 1000) {
          state.issuesAccum = state.issuesAccum.slice(-1000);
        }
        state.issuesSinceId = Math.max(state.issuesSinceId, ...arr.map((e) => e.id || 0));
      }
      renderIssues();
      updateIssuesBadge();
    } catch (e) {
      if (e.message !== 'forbidden') console.warn('[diag] issues fetch failed', e);
    }
  }

  function renderIssues() {
    const list = $('diag-issues-list');
    if (!list) return;
    const minLevel = $('diag-issues-level')?.value || 'WARNING';
    const comp = ($('diag-issues-component')?.value || '').trim().toLowerCase();
    const order = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
    const minIdx = order.indexOf(minLevel);
    const filtered = state.issuesAccum.filter((e) => {
      if (order.indexOf(e.level) < minIdx) return false;
      if (comp && !(e.logger || '').toLowerCase().includes(comp)) return false;
      return true;
    });
    if (!filtered.length) {
      list.innerHTML = '<div class="diag-empty">No issues match the current filter.</div>';
      return;
    }
    list.innerHTML = filtered.slice().reverse().map(issueRowHTML).join('');
  }

  function issueRowHTML(e) {
    const level = (e.level || 'INFO').toLowerCase();
    const ts = new Date(e.ts * 1000);
    const tsStr = ts.toLocaleTimeString([], { hour12: false }) + '.' + String(ts.getMilliseconds()).padStart(3, '0');
    const exc = e.exc ? `<pre class="diag-issue-exc">${esc(e.exc)}</pre>` : '';
    return `
      <div class="diag-issue-row" data-level="${esc(level)}">
        <span class="diag-issue-time">${tsStr}</span>
        <span class="diag-issue-level">${esc(e.level || '')}</span>
        <code class="diag-issue-logger">${esc(e.logger || '')}</code>
        <span class="diag-issue-msg">${esc(e.message || '')}</span>
        ${exc}
      </div>
    `;
  }

  function updateIssuesBadge() {
    const badge = $('diag-issues-badge');
    if (!badge) return;
    const last5min = Date.now() / 1000 - 300;
    const recent = state.issuesAccum.filter((e) => e.ts >= last5min).length;
    if (recent > 0) {
      badge.textContent = String(recent);
      badge.style.display = 'inline-block';
    } else {
      badge.style.display = 'none';
    }
  }

  // ── Server log tab ──────────────────────────────────────────────
  async function refetchServerLogTail() {
    const v = $('diag-srvlog-viewer');
    if (!v) return;
    const params = new URLSearchParams();
    params.set('lines', '500');
    const lvl = $('diag-srvlog-level')?.value;
    const search = $('diag-srvlog-search')?.value;
    if (lvl) params.set('level', lvl);
    if (search) params.set('search', search);
    try {
      const r = await adminFetch(`/api/diag/log/server?${params}`);
      const text = await r.text();
      v.innerHTML = '';
      for (const line of text.split('\n')) {
        if (!line) continue;
        v.appendChild(logLineEl(line));
      }
      maybeAutoscroll(v);
    } catch (e) {
      if (e.message !== 'forbidden') console.warn('[diag] server log fetch failed', e);
    }
  }

  function startServerLogStream() {
    stopServerLogStream();
    if (state.forbidden) return;
    try {
      const sse = new EventSource('/api/diag/log/server/stream', { withCredentials: true });
      sse.onmessage = (ev) => {
        const v = $('diag-srvlog-viewer');
        if (!v) return;
        const line = ev.data;
        const search = ($('diag-srvlog-search')?.value || '').toLowerCase();
        if (search && !line.toLowerCase().includes(search)) return;
        const lvl = $('diag-srvlog-level')?.value;
        if (lvl && !lineMeetsLevel(line, lvl)) return;
        v.appendChild(logLineEl(line));
        // Cap retained lines so the DOM doesn't balloon during long sessions.
        while (v.childNodes.length > 5000) v.removeChild(v.firstChild);
        maybeAutoscroll(v);
      };
      sse.onerror = () => {
        // 403s arrive as opaque network errors; flip into the forbidden
        // banner so the user understands why no log is appearing.
        setForbidden(true);
        stopServerLogStream();
      };
      state.sse = sse;
    } catch (e) {
      console.warn('[diag] SSE failed', e);
    }
  }

  function stopServerLogStream() {
    if (state.sse) {
      try { state.sse.close(); } catch {}
      state.sse = null;
    }
  }

  function lineMeetsLevel(line, minLevel) {
    const order = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
    const minIdx = order.indexOf(minLevel);
    for (let i = minIdx; i < order.length; i++) {
      if (line.includes(' ' + order[i] + ' ')) return true;
    }
    return false;
  }

  function logLineEl(line) {
    const div = document.createElement('div');
    div.className = 'diag-log-line';
    let level = 'info';
    if (line.includes(' CRITICAL ')) level = 'critical';
    else if (line.includes(' ERROR ')) level = 'error';
    else if (line.includes(' WARNING ')) level = 'warning';
    else if (line.includes(' DEBUG ')) level = 'debug';
    div.dataset.level = level;
    div.textContent = line;
    return div;
  }

  function maybeAutoscroll(v) {
    if ($('diag-srvlog-autoscroll')?.checked) v.scrollTop = v.scrollHeight;
  }

  // ── Meeting log tab ─────────────────────────────────────────────
  async function loadMeetingsList() {
    const sel = $('diag-mtglog-select');
    if (!sel) return;
    try {
      const r = await adminFetch('/api/meetings');
      const data = await r.json();
      const arr = Array.isArray(data) ? data : (data?.meetings || []);
      sel.innerHTML = '<option value="">— Select a meeting —</option>' +
        arr.map((m) => {
          const id = m.id || m.meeting_id || '';
          const label = m.title || m.summary || m.started_at || id;
          return `<option value="${esc(id)}">${esc(label)} · ${esc(id)}</option>`;
        }).join('');
    } catch (e) {
      if (e.message !== 'forbidden') console.warn('[diag] meetings list failed', e);
    }
  }

  async function loadMeetingLog() {
    const sel = $('diag-mtglog-select');
    const v = $('diag-mtglog-viewer');
    if (!sel || !v) return;
    const id = sel.value;
    if (!id) {
      v.innerHTML = '<div class="diag-empty">Select a meeting to view its terminal log.</div>';
      return;
    }
    v.innerHTML = '<div class="diag-empty">Loading…</div>';
    try {
      const r = await adminFetch(`/api/meetings/${encodeURIComponent(id)}/terminal-log`);
      if (r.status === 404) {
        v.innerHTML = '<div class="diag-empty">No terminal log captured for this meeting.</div>';
        return;
      }
      const text = await r.text();
      v.innerHTML = '';
      for (const line of text.split('\n')) {
        if (!line) continue;
        const div = document.createElement('div');
        div.className = 'diag-log-line';
        // Strip ANSI sequences for safety; preserving colour would need
        // a parser and the meeting-log viewer is fine as plain text.
        div.textContent = stripAnsi(line);
        v.appendChild(div);
      }
      v.scrollTop = v.scrollHeight;
    } catch (e) {
      if (e.message !== 'forbidden') v.innerHTML = '<div class="diag-empty">Failed to load log.</div>';
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────
  function esc(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function fmtMb(n) {
    if (n == null) return '—';
    if (n >= 1024) return `${(n / 1024).toFixed(1)} GB`;
    return `${Math.round(n)} MB`;
  }
  function fmtDuration(s) {
    s = Number(s) || 0;
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (d) return `${d}d ${h}h`;
    if (h) return `${h}h ${m}m`;
    return `${m}m`;
  }
  function debounce(fn, ms) {
    let t = null;
    return (...args) => {
      if (t) clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  }
  // ANSI CSI sequences (ESC [ … m) — strip for the meeting log viewer.
  function stripAnsi(s) {
    return s.replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, '');
  }
})();
