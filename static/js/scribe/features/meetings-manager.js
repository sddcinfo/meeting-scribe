// Meeting Scribe — Meetings manager (history panel).
//
// The slide-over list of past meetings + the review-mode entry path.
// The class takes a `deps` bag via the constructor so the admin SPA
// boot orchestrator can pass in the surfaces that live there:
//
//   * exitLiveMeetingView, _resumeMeeting — recording lifecycle
//   * _renderSummaryPanel, _renderFinalizationSummary — summary UI
//   * _updateColumnHeaders, _defaultLanguagePair — language config
//   * showFinalizationSummaryFor — finalization modal reopen
//   * findSpeakerSegments, showSpeakerModal, renderSpeakerTimeline —
//     thin wrappers that bind the shared `store` + the refresh hook
//     into the underlying speaker-review feature
//
// Everything else (modal-system, meeting-tools modals, popout-window,
// room-editor, bg-finalize-toast, metrics-dashboard, meetings-panel,
// speaker-registry, speaker-palette, compact-grid, lang-helpers,
// time-format, meeting-url, escape, state.js, mic-warmup) is imported
// directly below.
//
// The constructor wires the tab switcher + the "new meeting" button
// + kicks the first refresh + sets the 10s polling interval.

import { _enc } from "../lib/meeting-url.js";
import { esc } from "../lib/escape.js";
import { getLangA as _getLangA, getLangB as _getLangB } from "../lib/lang-helpers.js";
import { formatTime } from "../lib/time-format.js";
import { getSpeakerColor, SPEAKER_COLORS } from "../lib/speaker-palette.js";
import { state, store } from "../state.js";
import { resetSpeakerRegistry as _resetSpeakerRegistry } from "./speaker-registry.js";
import {
  alertDialog,
  closeModal,
  confirmDialog,
  promptDialog,
  showModal,
} from "./modal-system.js";
import {
  isMeetingsPanelOpen,
  toggleMeetingsPanel,
} from "./meetings-panel.js";
import {
  openMeetingToolsModal as _openMeetingToolsModal,
  openSlideAssetsModal as _openSlideAssetsModal,
} from "./meeting-tools-modal.js";
import { openPopout as _openPopout } from "./popout-window.js";
import { openRoomEditor as _openRoomEditor } from "./room-editor.js";
import { CompactGridRenderer } from "./compact-grid.js";
import { setMetricsVisible } from "./metrics-dashboard.js";
import { micWarmup } from "./mic-warmup.js";
import {
  _bgFinalizeTrackers,
  _finalizingMeetings,
  _renderBackgroundFinalizeToast,
  _sidecarToToastMsg,
} from "./bg-finalize-toast.js";

const API = "";

export class MeetingsManager {
  constructor(deps) {
    this._deps = deps;
    this.listEl = document.getElementById('meetings-list');
    this.btnNew = document.getElementById('btn-new-meeting');
    this.viewingMeetingId = null;
    // Tab state — "all" keeps strict chronological order so the timeline
    // doesn't reshuffle when a meeting is favorited. "favorites" is a
    // separate view for the user's curated demo/reference picks.
    this.activeTab = (() => {
      try { return localStorage.getItem('meetings_tab') || 'all'; } catch { return 'all'; }
    })();
    this._lastMeetings = [];

    // Wire the tab switcher buttons
    const tabs = document.querySelectorAll('.meetings-tab');
    tabs.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.tab === this.activeTab);
      btn.setAttribute('aria-selected', btn.dataset.tab === this.activeTab ? 'true' : 'false');
      btn.addEventListener('click', () => {
        this.activeTab = btn.dataset.tab;
        try { localStorage.setItem('meetings_tab', this.activeTab); } catch {}
        tabs.forEach((b) => {
          b.classList.toggle('active', b.dataset.tab === this.activeTab);
          b.setAttribute('aria-selected', b.dataset.tab === this.activeTab ? 'true' : 'false');
        });
        this._render(this._lastMeetings);
      });
    });

    this.btnNew.addEventListener('click', () => {
      // Free navigation: even while recording the user can reach the
      // setup screen. The server's idempotent start fast-path handles
      // any accidental re-start attempts; the return banner remains
      // visible so the user can always get back to the live view.
      if (document.body.classList.contains('recording')
       || document.body.classList.contains('meeting-active')) {
        this._deps.exitLiveMeetingView();
      }
      this.showSetup(); toggleMeetingsPanel();
    });
    this.refresh();
    setInterval(() => this.refresh(), 10000);
  }

  async refresh() {
    try {
      const resp = await fetch(`${API}/api/meetings`);
      const data = await resp.json();
      this._lastMeetings = data.meetings || [];
      // Hydrate the Phase B finalize trackers + corner toasts from the
      // server-side sidecar. After a page reload the WS connection is
      // fresh and we never saw the in-flight events that built up
      // ``_bgFinalizeTrackers`` in memory; without this the toast
      // would silently disappear while Phase B kept running in the
      // background. We do it for every row that carries the sidecar
      // (in-flight OR failure-terminal) so the operator sees a
      // consistent picture across reload.
      for (const m of this._lastMeetings) {
        if (m.phase_b_progress) {
          const synthetic = _sidecarToToastMsg(m.meeting_id, m.phase_b_progress);
          if (synthetic) _renderBackgroundFinalizeToast(synthetic);
        }
      }
      this._render(this._lastMeetings);
    } catch {}
  }

  _render(meetings) {
    this.listEl.innerHTML = '';

    // Apply the active tab filter. The list comes from the server already
    // sorted newest-first; we DON'T re-sort favorites to the top here so
    // toggling a star never reshuffles the timeline. Favorites tab just
    // narrows the view to the starred subset.
    if (this.activeTab === 'favorites') {
      meetings = meetings.filter((m) => m.is_favorite);
      if (meetings.length === 0) {
        this.listEl.innerHTML = '<div style="padding:1rem;color:var(--text-muted);font-size:0.75rem;text-align:center;">No starred meetings yet — tap a star to add one.</div>';
        return;
      }
    }

    if (meetings.length === 0) {
      this.listEl.innerHTML = '<div style="padding:1rem;color:var(--text-muted);font-size:0.75rem;text-align:center;">No meetings yet</div>';
      return;
    }

    for (const m of meetings) {
      const item = document.createElement('div');
      item.className = `meeting-item${this.viewingMeetingId === m.meeting_id ? ' active' : ''}`;
      const date = m.created_at ? new Date(m.created_at) : null;
      const dateStr = date ? `${date.toLocaleDateString()} ${date.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}` : 'Unknown';
      // Build action buttons
      let summaryBtn = '';
      if (m.state === 'complete') {
        if (m.has_summary) {
          summaryBtn = '<button class="meeting-summary btn-ghost" title="View finalization summary"><span class="check-mark">✓</span> Summary</button>';
        } else if (m.event_count > 0) {
          summaryBtn = '<button class="meeting-summary btn-ghost meeting-no-summary" title="No summary — click to generate">⊘ Summary</button>';
        }
      }
      const viewBtn = m.event_count > 0 && m.state !== 'recording' && m.state !== 'finalizing'
        ? '<button class="meeting-view btn-ghost" title="View meeting">View</button>'
        : '';
      const resumeBtn = m.state === 'interrupted'
        ? '<button class="meeting-resume btn-ghost" title="Resume recording">Resume</button>'
        : '';
      // Phase B: show live progress in the meetings list. The toast is
      // ephemeral (one per session), but the list is the durable place
      // an operator looks to see "is my prior meeting still finalizing?".
      // The progress map is keyed by meeting_id and updated by the
      // background_finalize_progress WS handler.
      let finalizingProgress = '';
      if (m.state === 'finalizing') {
        const tracker = _bgFinalizeTrackers.get(m.meeting_id);
        const step = tracker?.step ?? 0;
        const total = tracker?.total ?? 7;
        const label = tracker?.label || 'Finalizing...';
        const pct = total > 0 ? Math.min(100, Math.round((step / total) * 100)) : 0;
        finalizingProgress = `
          <div class="meeting-finalizing-progress" data-meeting-id="${esc(m.meeting_id)}">
            <span class="meeting-finalizing-label" title="${esc(label)}">${esc(label)}</span>
            <span class="meeting-finalizing-bar"><span class="meeting-finalizing-fill" style="width:${pct}%"></span></span>
            <span class="meeting-finalizing-step">${step}/${total}</span>
          </div>
        `;
      }
      // Interrupted meetings with events but no summary can be finalized
      // without resuming — just close them out and run the summarizer.
      const finalizeBtn = m.state === 'interrupted' && m.event_count > 0 && !m.has_summary
        ? '<button class="meeting-finalize btn-ghost" title="End and finalize (summary + cleanup)">End</button>'
        : '';

      // Short summary preview — truncated to ~120 chars, full text in tooltip
      const fullSummary = m.executive_summary || '';
      const topicsStr = (m.topics || []).join(' • ');
      const shortSummary = fullSummary
        ? (fullSummary.length > 120 ? fullSummary.slice(0, 117) + '…' : fullSummary)
        : '';
      const summaryTitle = fullSummary
        + (topicsStr ? `\n\nTopics: ${topicsStr}` : '');
      const summaryPreview = shortSummary
        ? `<div class="meeting-item-summary" title="${esc(summaryTitle)}">${esc(shortSummary)}</div>`
        : '';

      const starBtn = `<button class="meeting-star${m.is_favorite ? ' starred' : ''}" title="${m.is_favorite ? 'Unstar — remove from favorites' : 'Star — mark as useful demo / reference'}" aria-pressed="${m.is_favorite ? 'true' : 'false'}">${m.is_favorite ? '★' : '☆'}</button>`;
      // Reprocess buttons — only on complete meetings with events.
      // - Re-diarize: fast, just re-runs full-audio diarization + speaker
      //   collapse on the existing transcript (~2-3 min for 60-min audio).
      // - Reprocess: slow, re-runs ASR + translation + diarization from
      //   raw audio (~10-15 min for 60-min audio). Use when you want
      //   higher-quality transcript text in addition to speaker fixes.
      const canReprocess = m.state === 'complete' && m.event_count > 0;
      // Show the tools menu (⋯) for any non-live meeting with a journal,
      // so a meeting whose reprocess was killed mid-flight can still be
      // recovered (Retry reprocess, view versions, delete). Without this
      // the user has no escape hatch when state gets stuck at
      // 'reprocessing' or 'interrupted'.
      const canShowTools = m.event_count > 0 && m.state !== 'recording';
      const rediarizeBtn = canReprocess
        ? '<button class="meeting-rediarize btn-ghost" title="Re-run diarization + speaker consolidation (fast, ~2-3 min for 60-min audio). Optional speaker count.">Re-diarize</button>'
        : '';
      const reprocessBtn = canReprocess
        ? '<button class="meeting-reprocess btn-ghost" title="Full reprocess: re-run ASR + translation + diarization from raw audio (slow, ~10-15 min for 60-min audio). Use for higher-quality transcript.">Reprocess</button>'
        : '';
      // Slides button: only when this meeting has a deck on disk. Opens
      // the original-language PDF in a new tab — uses the same deck that
      // was uploaded during the meeting.
      const slidesBtn = m.has_slides
        ? '<button class="meeting-slides btn-ghost" title="Open the slide deck uploaded during this meeting">Slides</button>'
        : '';
      // Tools menu: hides the destructive / heavy admin actions
      // (Re-diarize, Reprocess, Versions, Delete) behind a single "⋯"
      // button so the row stays uncluttered and the primary actions
      // (View / Summary / Slides) stay obvious.
      const toolsBtn = '<button class="meeting-tools btn-ghost" title="More actions: Re-diarize, Reprocess, Versions, Delete" aria-label="More actions">⋯</button>';
      item.innerHTML = `
        <div class="meeting-item-row">
          ${starBtn}
          <div class="meeting-item-content">
            <div class="meeting-item-head">
              <span class="meeting-item-date">${dateStr}</span>
              <span class="meeting-item-info">
                <span class="meeting-item-state ${m.state}">${m.state}</span>
                ${m.event_count > 0 ? `${m.event_count} events` : ''}
              </span>
            </div>
            ${finalizingProgress}
            ${summaryPreview}
          </div>
          ${viewBtn}
          ${summaryBtn}
          ${slidesBtn}
          ${resumeBtn}
          ${finalizeBtn}
          ${canShowTools ? toolsBtn : '<button class="meeting-delete" title="Delete meeting">&times;</button>'}
        </div>
      `;
      item.querySelector('.meeting-item-content').addEventListener('click', () => this.viewMeeting(m.meeting_id));
      item.querySelector('.meeting-view')?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.viewMeeting(m.meeting_id);
      });
      item.querySelector('.meeting-summary')?.addEventListener('click', (e) => {
        e.stopPropagation();
        this._deps.showFinalizationSummaryFor(m.meeting_id);
      });
      item.querySelector('.meeting-resume')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        await this._deps._resumeMeeting(m.meeting_id);
        if (isMeetingsPanelOpen()) toggleMeetingsPanel();
        this.refresh();
      });
      item.querySelector('.meeting-finalize')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.textContent = 'Ending…';
        try {
          const resp = await fetch(`${API}/api/meetings/${_enc(m.meeting_id)}/finalize`, {
            method: 'POST',
          });
          if (!resp.ok) throw new Error(await resp.text());
          // Refresh the list — the meeting state should transition to "complete"
          // once the summarizer finishes. Poll every 3s.
          this.refresh();
          const poll = setInterval(async () => {
            await this.refresh();
            const resp2 = await fetch(`${API}/api/meetings`);
            const data = await resp2.json();
            const updated = (data.meetings || []).find(x => x.meeting_id === m.meeting_id);
            if (updated && updated.state === 'complete') {
              clearInterval(poll);
            }
          }, 3000);
          setTimeout(() => clearInterval(poll), 120000); // hard stop after 2 min
        } catch (err) {
          showModal(`<div class="modal-confirm-title">Finalization failed</div><div class="modal-confirm-message">${esc(String(err.message || err))}</div><div class="modal-confirm-actions"><button class="modal-btn" onclick="closeModal()">OK</button></div>`, 'confirm');
          btn.disabled = false;
          btn.textContent = 'End';
        }
      });
      item.querySelector('.meeting-delete')?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteMeeting(m.meeting_id);
      });
      // Tools menu (⋯) — opens an actions modal that explains each
      // operation. Keeps Re-diarize / Reprocess / Versions / Delete out
      // of the row so the row stays clean.
      item.querySelector('.meeting-tools')?.addEventListener('click', (e) => {
        e.stopPropagation();
        _openMeetingToolsModal(m, this);
      });
      // Slides: chooser modal — original PDF for quick viewing, plus
      // the source + translated PPTX downloads for editing or sharing.
      // translated.pptx 404s for monolingual decks; the row is probed
      // on open and shown as disabled in that case.
      item.querySelector('.meeting-slides')?.addEventListener('click', (e) => {
        e.stopPropagation();
        _openSlideAssetsModal(m);
      });

      // Re-diarize: fast path — just re-runs diarize + speaker consolidation
      item.querySelector('.meeting-rediarize')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const raw = await promptDialog(
          'Re-diarize meeting',
          'Pin a speaker count when known (recommended for over-clustered meetings). Leave blank to let the model decide.',
          {
            placeholder: 'Speaker count (1–12) or blank',
            confirmText: 'Re-diarize',
            type: 'number',
            inputMode: 'numeric',
            min: 1,
            max: 12,
            help: 'Keeps the transcript text. Replaces speaker labels only.',
          }
        );
        if (raw === null) return;
        const expected = raw === '' ? null : parseInt(raw, 10);
        if (raw !== '' && (!Number.isFinite(expected) || expected < 1 || expected > 12)) {
          await alertDialog('Invalid count', 'Speaker count must be a number between 1 and 12, or blank.');
          return;
        }
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.textContent = 'Re-diarizing…';
        try {
          const qs = expected != null ? `?expected_speakers=${expected}` : '';
          const resp = await fetch(`${API}/api/meetings/${_enc(m.meeting_id)}/finalize${qs}`, { method: 'POST' });
          if (!resp.ok) throw new Error(await resp.text());
          const result = await resp.json();
          this.refresh();
          await alertDialog(
            'Re-diarize complete',
            `${result?.diarization?.unique_speakers ?? '?'} speakers detected from ${result?.diarization?.segments ?? '?'} diarize segments.`,
          );
        } catch (err) {
          await alertDialog('Re-diarize failed', String(err.message || err));
        } finally {
          btn.disabled = false;
          btn.textContent = 'Re-diarize';
        }
      });

      // Reprocess: slow path — re-runs ASR + translation + diarize
      item.querySelector('.meeting-reprocess')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const raw = await promptDialog(
          'Full reprocess from raw audio',
          'Re-runs ASR + translation + diarization for a higher-quality transcript. Slow: about 10–15 minutes for a 60-minute meeting. The current journal is backed up as journal.jsonl.bak.',
          {
            placeholder: 'Speaker count (1–12) or blank',
            confirmText: 'Reprocess',
            type: 'number',
            inputMode: 'numeric',
            min: 1,
            max: 12,
            help: 'Pin a speaker count when known, or leave blank to let pyannote decide.',
          }
        );
        if (raw === null) return;
        const expected = raw === '' ? null : parseInt(raw, 10);
        if (raw !== '' && (!Number.isFinite(expected) || expected < 1 || expected > 12)) {
          await alertDialog('Invalid count', 'Speaker count must be a number between 1 and 12, or blank.');
          return;
        }
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.textContent = 'Reprocessing…';
        try {
          const qs = expected != null ? `?expected_speakers=${expected}` : '';
          // No timeout on the fetch — server holds the connection until
          // the full pipeline completes, which can take 10+ minutes.
          const resp = await fetch(`${API}/api/meetings/${_enc(m.meeting_id)}/reprocess${qs}`, { method: 'POST' });
          if (!resp.ok) throw new Error(await resp.text());
          const result = await resp.json();
          this.refresh();
          const segs = result?.segments ?? '?';
          const tr = result?.translated ?? '?';
          const sp = result?.speakers ?? '?';
          await alertDialog('Reprocess complete', `${segs} segments, ${tr} translated, ${sp} speakers detected.`);
        } catch (err) {
          await alertDialog('Reprocess failed', String(err.message || err));
        } finally {
          btn.disabled = false;
          btn.textContent = 'Reprocess';
        }
      });
      item.querySelector('.meeting-star')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const btn = e.currentTarget;
        const next = !m.is_favorite;
        // Optimistic update so the star feels instant — revert on failure.
        btn.disabled = true;
        btn.classList.toggle('starred', next);
        btn.textContent = next ? '★' : '☆';
        btn.setAttribute('aria-pressed', next ? 'true' : 'false');
        try {
          const resp = await fetch(`${API}/api/meetings/${_enc(m.meeting_id)}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_favorite: next }),
          });
          if (!resp.ok) throw new Error(await resp.text());
          m.is_favorite = next;
          // Re-render so favorites re-sort to the top of the list
          this.refresh();
        } catch (err) {
          // Revert optimistic UI and surface the error
          btn.classList.toggle('starred', !next);
          btn.textContent = !next ? '★' : '☆';
          btn.setAttribute('aria-pressed', !next ? 'true' : 'false');
          showModal(`<div class="modal-confirm-title">Could not update favorite</div><div class="modal-confirm-message">${esc(String(err.message || err))}</div><div class="modal-confirm-actions"><button class="modal-btn" onclick="closeModal()">OK</button></div>`, 'confirm');
        } finally {
          btn.disabled = false;
        }
      });
      this.listEl.appendChild(item);
    }
  }

  async viewMeeting(meetingId) {
    // Navigation is allowed while a meeting is recording — the live
    // pipeline stays alive, the reconciler's return banner lets the
    // user come back. We detach live rendering so the review view
    // can safely own the shared store.
    if (document.body.classList.contains('recording')
     || document.body.classList.contains('meeting-active')) {
      this._deps.exitLiveMeetingView();
    }

    // If this meeting is still being finalized, reopen the finalization modal
    if (_finalizingMeetings.has(meetingId)) {
      this._reopenFinalizationModal(meetingId);
      return;
    }
    this.viewingMeetingId = meetingId;
    this.refresh();
    // Close the panel if it's open (don't open it if closed)
    if (isMeetingsPanelOpen()) toggleMeetingsPanel();

    try {
      const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}`);
      const data = await resp.json();

      // Hydrate the Phase B finalize tracker for this meeting BEFORE
      // the banner render so the patcher has data to paint. If the
      // sidecar isn't present (meeting fully finalized) the banner
      // stays hidden via display:none.
      const banner = document.getElementById('meeting-detail-finalize-banner');
      if (banner) {
        banner.dataset.meetingId = meetingId;
        if (data.phase_b_progress) {
          const synthetic = _sidecarToToastMsg(meetingId, data.phase_b_progress);
          if (synthetic) _renderBackgroundFinalizeToast(synthetic);
        } else {
          // Stale tracker from a previous detail view: clear so the
          // banner doesn't paint into the wrong meeting.
          _bgFinalizeTrackers.delete(meetingId);
          banner.style.display = 'none';
        }
      }

      // Clear stale state from previous meeting view. Cluster_ids are
      // meeting-local, so without resetting the speaker registry the
      // first-seen-order sequential labels ("Speaker 1", "Speaker 2"...)
      // and any custom-renamed names from the previous meeting leak into
      // this one. We also have to drop the one-on-one mode assignments
      // (also keyed on the previous meeting's cluster_ids) and null the
      // audio player's timeline so a fast meeting switch can't cross-wire
      // old segments with new audio.
      _resetSpeakerRegistry();
      window.audioPlayer._timeline = null;
      window.audioPlayer.currentSegmentId = null;
      store.clear();

      document.getElementById('speaker-timeline').style.display = 'none';
      document.getElementById('speaker-timeline-lanes').innerHTML = '';
      document.getElementById('speaker-timeline-times').innerHTML = '';
      document.getElementById('meeting-summary-panel').style.display = 'none';
      document.getElementById('player-bar').style.display = 'none';
      window.audioPlayer.hide();

      // Metrics split-view is only meaningful for live meetings — clear it on
      // entering review so the 320px right-rail reservation doesn't hide the
      // new top summary bar under the pinned dashboard.
      setMetricsVisible(false);

      // Show meeting mode with the transcript grid for past meeting replay
      document.getElementById('landing-mode').style.display = 'none';
      document.getElementById('room-setup').style.display = 'none';
      document.getElementById('view-mode').style.display = 'none';
      document.getElementById('meeting-mode').style.display = '';

      // Initialize grid renderer with meeting ID for playback
      window._gridRenderer = new CompactGridRenderer(document.getElementById("transcript-grid"), meetingId, formatTime);
      // Finished-meeting default: ALWAYS oldest-first with the viewport
      // anchored to the FIRST segment.
      //
      // Two things need to happen here that the live-view code path gets
      // for free:
      //
      // 1. Flip the renderer direction so new blocks get appended at the
      //    end (oldest-to-newest reading order).
      // 2. Disable auto-scroll. The renderer's default update() calls
      //    `scrollTop = scrollHeight` after each event ingest so the
      //    LATEST block stays visible — that's right for a live stream
      //    but wrong for review mode, where we want the oldest block to
      //    stay at the top while every later event fills in below.
      //    Without this, the last ingested event during initial load
      //    yanks the viewport to the bottom and the user sees "newest
      //    at bottom" while scrolled to it, which reads as "reversed".
      {
        const isFinished0 = data.meta?.state && data.meta.state !== 'recording';
        if (isFinished0 && window._gridRenderer) {
          window._gridRenderer.toggleDirection();
          window._gridRenderer._autoScroll = false;
          requestAnimationFrame(() => {
            const g = document.getElementById('transcript-grid');
            if (g) g.scrollTop = 0;
          });
        }
      }

      // Show podcast player + speaker timeline if audio/timeline exists
      try {
        const tlResp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/timeline`);
        if (tlResp.ok) {
          const tl = await tlResp.json();
          if (tl.duration_ms > 0) {
            window.audioPlayer.loadMeeting(meetingId, tl.duration_ms, tl.segments);
            // Render speaker timeline lanes
            if (tl.speaker_lanes && Object.keys(tl.speaker_lanes).length > 0) {
              this._deps.renderSpeakerTimeline(tl.speaker_lanes, tl.duration_ms, tl.speakers || [], meetingId);
            }
          }
        }
      } catch {}

      // Set meeting start time for wall clock display
      if (data.meta?.created_at) {
        state.meetingStartWallMs = new Date(data.meta.created_at).getTime();
      }

      // Set language pair from meeting metadata. Length 1 = monolingual
      // (store a single-code string so downstream helpers can decide mode).
      if (data.meta?.language_pair && data.meta.language_pair.length >= 1) {
        state.currentLanguagePair = data.meta.language_pair.join(',');
        this._deps._updateColumnHeaders();
      }

      // Load events into store → flows to all transcript columns
      if (data.events?.length > 0) {
        for (const event of data.events) {
          store.ingest(event);
        }
      }

      // Render table strip if room layout exists
      const strip = document.getElementById('meeting-table-strip');
      strip.innerHTML = '';
      if (data.room && (data.room.tables?.length || data.room.seats?.length)) {
        strip.style.display = '';
        for (const t of data.room.tables || []) {
          const el = document.createElement('div');
          el.className = 'strip-table';
          el.style.cssText = `left:${t.x}%;top:${t.y}%;width:${t.width}%;height:${t.height}%;border-radius:${t.border_radius}%`;
          strip.appendChild(el);
        }
        // Build name→cluster_id map from events so seat colors match the transcript
        const nameToClusterId = {};
        for (const e of data.events || []) {
          const sp = e.speakers?.[0];
          if (sp) {
            const key = sp.identity || sp.display_name;
            if (key && !(key in nameToClusterId)) {
              nameToClusterId[key] = sp.cluster_id;
            }
          }
        }
        for (let si = 0; si < (data.room.seats || []).length; si++) {
          const s = data.room.seats[si];
          // Color by cluster_id (canonical) — falls back to seat index if unknown
          const clusterId = nameToClusterId[s.speaker_name];
          const color = clusterId != null ? getSpeakerColor(clusterId) : SPEAKER_COLORS[si % SPEAKER_COLORS.length];
          const el = document.createElement('div');
          el.className = `strip-seat enrolled`;
          el.dataset.speakerId = String(si);
          el.dataset.clusterId = String(clusterId ?? '');
          el.style.cssText = `left:${s.x}%;top:${s.y}%;--seat-color:${color}`;
          el.innerHTML = `<span class="strip-seat-num">${esc(s.speaker_name?.[0] || (si+1).toString())}</span><span class="strip-seat-name">${esc(s.speaker_name || '')}</span>`;
          // Click to show speaker's segments
          const seatName = s.speaker_name || `Speaker ${si+1}`;
          el.addEventListener('click', () => {
            this._deps.showSpeakerModal(seatName, color, this._deps.findSpeakerSegments(seatName), meetingId);
          });
          strip.appendChild(el);
        }
      } else {
        strip.style.display = 'none';
      }

      // Hide control bar for replay — review tools live in the summary bar
      document.getElementById('control-bar').style.display = 'none';
      const stateLabel = data.meta?.state === 'interrupted' ? ' · interrupted' : '';
      const nameA = state.languageNames[_getLangA()]?.name || _getLangA().toUpperCase();
      const nameB = state.languageNames[_getLangB()]?.name || _getLangB().toUpperCase();
      const savedDir = localStorage.getItem('scribe_transcript_direction');
      const dirInitialLabel = savedDir === 'oldest' ? '↓ Oldest' : '↑ Newest';

      // Status line is now a single short label — the event count and meeting
      // id live inside the summary modal where they belong.
      document.getElementById('status-line').textContent = `Review${stateLabel}`;

      // Populate the top summary bar with the review toolbar. The bar is
      // always shown in review mode, even if no summary exists yet; the
      // "View Summary" button is appended afterwards by _renderSummaryPanel
      // only when the summary fetch succeeds.
      //
      // The "id: <uuid>" chip is a click-to-copy so users reviewing a
      // meeting can identify which one they were looking at when
      // something goes wrong (re-finalize failure, download link issue).
      const summaryPanel = document.getElementById('meeting-summary-panel');
      summaryPanel.innerHTML =
        `<span class="summary-bar-icon" aria-hidden="true">📄</span>` +
        `<span class="summary-bar-title">Meeting Summary</span>` +
        `<span class="summary-bar-id" id="rv-meeting-id" title="Click to copy meeting ID" style="cursor:pointer;font-family:ui-monospace,Menlo,monospace;font-size:0.78rem;opacity:0.7;margin-right:0.5rem">id: ${esc(meetingId)}</span>` +
        `<div class="summary-bar-tools">` +
          `<button class="btn-ghost" id="rv-col-a" title="Show only ${nameA}">${nameA}</button>` +
          `<button class="btn-ghost" id="rv-col-b" title="Show only ${nameB}">${nameB}</button>` +
          `<button class="btn-ghost" id="rv-table" title="Toggle virtual table">Table</button>` +
          `<button class="btn-ghost" id="rv-scroll-dir" title="Toggle transcript order (newest first / oldest first)">${dirInitialLabel}</button>` +
          `<button class="btn-ghost" id="rv-live" title="Open pop-out translation view">Pop-out</button>` +
          `<button class="btn-ghost" id="rv-edit-layout" title="Edit layout + assign detected voices to seats">Edit Layout</button>` +
          `<button class="btn-ghost" id="btn-reprocess" title="Re-run ASR, diarization, translation and regenerate summary from the original audio">Re-finalize</button>` +
          (data.meta?.state === 'interrupted' ? `<button class="btn-ghost" id="btn-resume" style="background:var(--success);color:#fff">Resume</button>` : '') +
        `</div>`;
      summaryPanel.style.display = 'flex';

      // Column selectors — toggle between: both → only-a → only-b → both
      document.getElementById('rv-col-a')?.addEventListener('click', (e) => {
        const wasActive = document.body.classList.contains('show-only-a');
        document.body.classList.remove('show-only-a', 'show-only-b');
        document.getElementById('rv-col-b')?.classList.remove('active-toggle');
        if (!wasActive) {
          document.body.classList.add('show-only-a');
          e.target.classList.add('active-toggle');
        } else {
          e.target.classList.remove('active-toggle');
        }
      });
      document.getElementById('rv-col-b')?.addEventListener('click', (e) => {
        const wasActive = document.body.classList.contains('show-only-b');
        document.body.classList.remove('show-only-a', 'show-only-b');
        document.getElementById('rv-col-a')?.classList.remove('active-toggle');
        if (!wasActive) {
          document.body.classList.add('show-only-b');
          e.target.classList.add('active-toggle');
        } else {
          e.target.classList.remove('active-toggle');
        }
      });

      document.getElementById('rv-live')?.addEventListener('click', (e) => {
        _openPopout(e.target, meetingId);
      });
      document.getElementById('rv-table')?.addEventListener('click', (e) => { document.body.classList.toggle('hide-table'); e.target.classList.toggle('active-toggle'); });
      document.getElementById('rv-scroll-dir')?.addEventListener('click', (e) => {
        if (!window._gridRenderer) return;
        const newestFirst = window._gridRenderer.toggleDirection();
        e.target.textContent = newestFirst ? '↑ Newest' : '↓ Oldest';
        localStorage.setItem('scribe_transcript_direction', newestFirst ? 'newest' : 'oldest');
      });
      document.getElementById('rv-edit-layout')?.addEventListener('click', () => {
        _openRoomEditor('review', meetingId);
      });
      document.getElementById('btn-reprocess')?.addEventListener('click', async () => {
        const btn = document.getElementById('btn-reprocess');
        btn.disabled = true;
        btn.textContent = 'Re-finalizing…';
        document.getElementById('status-line').textContent = 'Re-finalizing: diarization, speaker data, summary...';
        try {
          // Re-finalize = finalize again: full-audio diarization + regen
          // speaker_lanes + regen detected_speakers + regen summary.
          // Use this after fixing bugs/config — same pipeline as the initial
          // finalize but forced to regenerate everything.
          const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/finalize?force=true`, {
            method: 'POST',
          });
          if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${err}`);
          }
          const result = await resp.json();
          console.log('Re-finalize result:', result);

          // Warn clearly if the audio is corrupted at the recording level
          const aq = result.audio_quality;
          if (aq && !aq.usable) {
            showModal(`
              <div class="modal-card-header"><h2>Audio quality warning</h2></div>
              <div style="padding:1rem 1.25rem;font-size:0.85rem;line-height:1.5">
                <p style="margin-bottom:0.75rem">
                  <strong>This recording is ${aq.zero_fill_pct}% silence-filled</strong>,
                  with the longest gap being ${(aq.longest_zero_run_ms/1000).toFixed(1)}s.
                </p>
                <p style="margin-bottom:0.75rem">
                  An earlier audio writer inserted zero-gaps whenever WebSocket
                  chunks arrived later than wall clock. Speaker separation for
                  this meeting is fundamentally limited — only
                  ${result.diarization?.unique_speakers || '?'} speaker(s) could
                  be clustered from the corrupted audio.
                </p>
                <p style="color:var(--text-muted)">
                  New meetings recorded after the audio writer fix will be clean.
                </p>
              </div>
              <div class="modal-card-footer">
                <button class="btn btn-primary" onclick="closeModal()">OK</button>
              </div>
            `);
          }

          const dz = result.diarization || {};
          document.getElementById('status-line').textContent =
            `Re-finalized ${meetingId}: ${dz.segments || 0} diarization segments, ${dz.unique_speakers || 0} speakers`;
          this.viewMeeting(meetingId);
        } catch (e) {
          document.getElementById('status-line').textContent = `Re-finalize failed: ${e.message}`;
          btn.disabled = false;
          btn.textContent = 'Re-finalize';
        }
      });
      document.getElementById('btn-resume')?.addEventListener('click', () => this._deps._resumeMeeting(meetingId));

      // Click-to-copy for the review-mode meeting-id chip. Keep in sync
      // with the equivalent handler in _renderFinalizationSummary().
      document.getElementById('rv-meeting-id')?.addEventListener('click', async (e) => {
        const chip = e.currentTarget;
        const original = chip.textContent;
        try {
          if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(meetingId);
          } else {
            const ta = document.createElement('textarea');
            ta.value = meetingId;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            ta.remove();
          }
          chip.textContent = 'copied ✓';
        } catch {
          chip.textContent = 'copy failed';
        }
        setTimeout(() => { chip.textContent = original; }, 1200);
      });

      // Load summary if available — must run AFTER the summary bar is
      // populated with rv-* tools above, because _renderSummaryPanel
      // appends the "View Summary" button to the existing
      // .summary-bar-tools row. Running it earlier would have nothing
      // to append to and then the later innerHTML= would wipe the row.
      try {
        const sumResp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/summary`);
        if (sumResp.ok) {
          const summary = await sumResp.json();
          this._deps._renderSummaryPanel(summary, meetingId);
        }
      } catch {}
    } catch (err) {
      document.getElementById('status-line').textContent = `Error: ${err.message}`;
    }
  }

  _reopenFinalizationModal(meetingId) {
    // Close meetings panel if open
    if (isMeetingsPanelOpen()) toggleMeetingsPanel();

    const tracker = _finalizingMeetings.get(meetingId);
    if (!tracker) return;

    // Must stay in sync with the 6-step flow emitted by /api/meeting/stop
    // (see stopMeeting() above).
    const steps = [
      { label: 'Flushing speech recognition' },
      { label: 'Completing translations' },
      { label: 'Saving speaker data' },
      { label: 'Running full-audio diarization' },
      { label: 'Generating timeline' },
      { label: 'Generating meeting summary' },
    ];

    const card = showModal(`
      <div class="finalize-modal">
        <div class="finalize-header">
          <div class="finalize-header-content">
            <div class="finalize-pulse"></div>
            <div>
              <h3>Finalizing Meeting</h3>
              <p class="finalize-subtitle" id="finalize-subtitle">${tracker.label || 'Processing...'}</p>
            </div>
          </div>
          <button class="finalize-close" id="finalize-close-btn" title="Close">&times;</button>
        </div>
        <div class="finalize-progress-track">
          <div class="finalize-progress-fill" id="finalize-progress-fill" style="width:${Math.min(100, (tracker.step / 6) * 100)}%"></div>
        </div>
        <div class="finalize-steps" id="finalize-steps">
          ${steps.map((s, i) => `
            <div class="finalize-step${(i + 1) < tracker.step ? ' done' : (i + 1) === tracker.step ? ' active' : ''}" data-step="${i + 1}">
              <div class="step-indicator">
                <div class="step-ring">
                  <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="none" stroke-width="1.5"/></svg>
                  <span class="step-check">&#10003;</span>
                </div>
                ${i < steps.length - 1 ? '<div class="step-connector"></div>' : ''}
              </div>
              <span class="step-label">${s.label}</span>
            </div>
          `).join('')}
        </div>
        <div class="finalize-eta" id="finalize-eta"></div>
        <div class="finalize-summary" id="finalize-summary" style="display:none"></div>
      </div>
    `, 'finalize');

    card.querySelector('#finalize-close-btn')?.addEventListener('click', () => closeModal());

    // Listen for updates via the original WS (if still alive)
    const ws = tracker.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      const origOnMessage = ws.onmessage;
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'finalize_progress') {
            // Server sends TWO step=6 messages: the first starts summary
            // generation (no summary field), the second carries the
            // summary and signals completion. Only the second one is
            // terminal — see the matching logic in stopMeeting().
            const isCompletion = msg.step >= 6 && (msg.summary !== undefined || msg.meeting_id);

            const pct = isCompletion ? 100 : Math.min(100, (msg.step / 6) * 100);
            const fill = card.querySelector('#finalize-progress-fill');
            if (fill) fill.style.width = `${pct}%`;

            card.querySelectorAll('.finalize-step').forEach(el => {
              const s = parseInt(el.dataset.step);
              el.classList.toggle('done', isCompletion ? true : s < msg.step);
              el.classList.toggle('active', isCompletion ? false : s === msg.step);
            });

            const subtitle = card.querySelector('#finalize-subtitle');
            if (subtitle && msg.label) subtitle.textContent = msg.label;

            const etaEl = card.querySelector('#finalize-eta');
            if (etaEl && msg.eta_seconds > 0) {
              etaEl.textContent = `Estimated ${msg.eta_seconds}s remaining`;
              etaEl.classList.add('visible');
            } else if (etaEl) {
              etaEl.classList.remove('visible');
            }

            // Update tracker
            tracker.step = msg.step;
            tracker.label = msg.label || '';

            if (isCompletion) {
              _finalizingMeetings.delete(meetingId);
              const pulse = card.querySelector('.finalize-pulse');
              if (pulse) pulse.classList.add('complete');
              if (subtitle) subtitle.textContent = 'Meeting finalized';
              if (msg.summary) this._deps._renderFinalizationSummary(msg.summary, meetingId);
            }
          }
        } catch {}
      };
    }
  }

  async deleteMeeting(meetingId) {
    const ok = await confirmDialog(
      'Delete Meeting?',
      'This will permanently delete the meeting and all its recordings, transcripts, and translations. This cannot be undone.',
    );
    if (!ok) return;
    try {
      const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}`, { method: 'DELETE' });
      if (!resp.ok) { const e = await resp.json(); console.warn('Delete failed:', e.error); return; }
      if (this.viewingMeetingId === meetingId) this.showSetup();
      this.refresh();
    } catch (err) { console.warn('Delete error:', err); }
  }

  showSetup() {
    this.viewingMeetingId = null;
    this.refresh();
    // Prime the microphone the moment the setup panel becomes visible so
    // the first chair tap captures audio instantly instead of waiting on
    // getUserMedia + AudioContext init (~500-1500 ms cold). Fire-and-forget;
    // _enrollSeat falls back to a per-call acquisition if priming hasn't
    // resolved yet (or if the user denies the permission prompt).
    micWarmup.prime();
    document.getElementById('landing-mode').style.display = 'none';
    document.getElementById('room-setup').style.display = '';
    document.getElementById('meeting-mode').style.display = 'none';
    document.getElementById('view-mode').style.display = 'none';
    document.getElementById('speaker-timeline').style.display = 'none';
    document.getElementById('meeting-summary-panel').style.display = 'none';
    setMetricsVisible(false);
    document.body.classList.remove('show-only-a', 'show-only-b', 'hide-live', 'hide-table');
    window.audioPlayer.hide();
    store.clear();
    document.getElementById('status-line').textContent = 'Ready';
    // Reset Start button in case a previous start attempt failed
    const startBtn = document.getElementById('btn-start-meeting');
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.textContent = startBtn.dataset.origText || 'Start Meeting';
    }
    // The setup screen is always bilingual. If the user is returning from
    // a mono meeting (started via the landing page's "Quick start English"
    // button), reset both dropdowns back to the server's configured default
    // pair — it's always possible to get back to a multi-language meeting
    // from here. Single-language is accessible ONLY via the landing page
    // quick-start, which bypasses this screen entirely.
    const selA = document.getElementById('lang-a-select');
    const selB = document.getElementById('lang-b-select');
    const selector = document.getElementById('language-selector');
    if (selA && selB) {
      const [defA, defB] = this._deps._defaultLanguagePair();
      const langCodes = [...selB.options].map(o => o.value).filter(Boolean);
      // Prefer server defaults; degrade to the first two distinct codes
      // the dropdowns know about if those aren't present (e.g. a very
      // restricted /api/languages response).
      const pickA = langCodes.includes(defA) ? defA : (langCodes[0] || defA);
      let pickB = langCodes.includes(defB) && defB !== pickA
        ? defB
        : (langCodes.find(v => v !== pickA) || defB);
      selA.value = pickA;
      selB.value = pickB;
      state.currentLanguagePair = `${pickA},${pickB}`;
      if (selector) selector.classList.remove('mono');
      this._deps._updateColumnHeaders();
    }
  }
}

