// Meeting Scribe — Speaker review (pure module).
//
// Three related public functions for the speaker-detail review surface:
//
//   findSpeakerSegments(speakerName, store)
//     → returns all events in `store.segments` matching the given
//       speaker name (by identity, display_name, cluster id from
//       the registry, or the `Speaker N` numeric pattern).
//
//   showSpeakerModal(speakerName, color, segments, meetingId, hooks)
//     → mounts the "click a speaker, see all their segments" modal.
//       hooks = { store, onSpeakerRegistryChanged }
//       The rename flow inside the modal writes through to the
//       speaker registry, posts to `/api/meetings/<mid>/events/<sid>/speaker`,
//       and calls `onSpeakerRegistryChanged()` so the transcript +
//       detected-speakers strip repaint.
//
//   renderSpeakerTimeline(speakerLanes, durationMs, speakers, meetingId, hooks)
//     → paints the per-speaker horizontal timeline above the
//       transcript in review mode. Clicking a lane label opens the
//       speaker modal; clicking a block (or empty track area) seeks
//       the audio + scrolls the transcript to the matching row.
//
// Bottom-up topology: every dep is either imported from another pure
// module (esc / formatTime / speaker-registry primitives / audio-player /
// modal-system / meeting-url), a window-contract read
// (`window.audioPlayer`), or passed in via the `hooks.store` /
// `hooks.onSpeakerRegistryChanged` parameters.

import { esc } from "../lib/escape.js";
import { formatTime } from "../lib/time-format.js";
import { _enc, _meetingsUrl } from "../lib/meeting-url.js";
import { getSpeakerColor } from "../lib/speaker-palette.js";
import {
  _speakerRegistry,
  getSpeakerDisplayName,
  getSpeakerSeqIndex,
  renameSpeaker,
} from "./speaker-registry.js";
import { SEEK_LEADIN_MS } from "./audio-player.js";
import { showModal, closeModal } from "./modal-system.js";

export function findSpeakerSegments(speakerName, store) {
  // Resolve the speakerName back to a cluster_id via the display registry
  // so we also catch entries whose speakers[0].display_name was never
  // populated (older journals).
  let targetClusterId = null;
  for (const [cid, entry] of _speakerRegistry.clusters) {
    if (entry?.displayName === speakerName) {
      targetClusterId = cid;
      break;
    }
    if (!entry?.displayName && `Speaker ${entry?.seqIndex}` === speakerName) {
      targetClusterId = cid;
      break;
    }
  }
  const nameMatch = speakerName.match(/^Speaker (\d+)$/);
  const nameClusterStr = nameMatch ? nameMatch[1] : null;

  const out = [];
  for (const ev of store.segments.values()) {
    const s = ev.speakers?.[0];
    if (!s) {
      // Attributed elsewhere (e.g. just the display name from finalize)
      if ((ev.speaker_name || ev.display_name) === speakerName) out.push(ev);
      continue;
    }
    if (s.identity === speakerName || s.display_name === speakerName) {
      out.push(ev);
      continue;
    }
    if (targetClusterId != null && s.cluster_id === targetClusterId) {
      out.push(ev);
      continue;
    }
    if (nameClusterStr && String(s.cluster_id) === nameClusterStr) {
      out.push(ev);
      continue;
    }
  }
  out.sort((a, b) => (a.start_ms || 0) - (b.start_ms || 0));
  return out;
}

export function showSpeakerModal(speakerName, speakerColor, segments, meetingId, hooks = {}) {
  const { store, onSpeakerRegistryChanged } = hooks;
  const _onChanged = onSpeakerRegistryChanged || (() => {});

  const totalSegs = segments.length;
  const totalMs = segments.reduce((s, e) => s + (e.end_ms - e.start_ms), 0);
  const totalSec = Math.round(totalMs / 1000);
  const languages = [...new Set(segments.map((e) => e.language).filter((l) => l !== "unknown"))];
  const mid = meetingId || window.audioPlayer?.meetingId || "";

  const segHtml = segments
    .map((e, i) => {
      const time = formatTime(e.start_ms);
      const lang =
        e.language !== "unknown" ? `<span class="seg-lang">${e.language.toUpperCase()}</span>` : "";
      const srcBody = e.furigana_html || esc(e.text);
      const trBody =
        e.translation?.furigana_html || (e.translation?.text ? esc(e.translation.text) : "");
      const trans = e.translation?.text
        ? `<div class="speaker-seg-translation">${trBody}</div>`
        : '<div class="speaker-seg-translation" style="color:var(--text-muted);font-size:0.8rem">awaiting translation...</div>';
      const playBtn = mid
        ? `<button class="speaker-seg-play" data-idx="${i}" title="Play this segment">▶</button>`
        : "";
      return `<div class="speaker-segment" data-start="${e.start_ms}" data-end="${e.end_ms}" data-segment-id="${e.segment_id || ""}">
      <div class="speaker-seg-left">
        ${playBtn}
        <div>
          <div class="speaker-seg-time">${time} ${lang}</div>
          <div class="speaker-seg-text">${srcBody}</div>
        </div>
      </div>
      <div class="speaker-seg-right">${trans}</div>
    </div>`;
    })
    .join("");

  const playAllBtn =
    mid && totalSegs > 0
      ? `<button class="modal-btn" id="speaker-play-all" style="margin-left:auto">▶ Play All</button>`
      : "";

  const card = showModal(
    `
    <div class="speaker-modal-header">
      <div class="speaker-modal-dot" style="background:${speakerColor}"></div>
      <div class="speaker-modal-name" id="speaker-modal-current-name">${esc(speakerName)}</div>
      <button class="btn-ghost speaker-rename-btn" id="speaker-rename-btn" title="Rename speaker">Rename</button>
      <div class="speaker-modal-stats">
        <span>${totalSegs} segments</span>
        <span>${totalSec}s speaking</span>
        <span>${languages.join(", ") || "—"}</span>
        ${playAllBtn}
      </div>
      <button class="speaker-modal-close" id="speaker-modal-close-btn">&times;</button>
    </div>
    <div class="speaker-modal-body">${
      segHtml ||
      `
      <div class="speaker-modal-empty">
        <h4>No segments found for ${esc(speakerName)}</h4>
        <div>This usually means the meeting hasn't been finalized yet,
          or the speaker label in the timeline lane no longer matches
          any transcript entry. Try refreshing the meeting view.</div>
      </div>`
    }</div>
  `,
    "speaker-modal",
  );

  card.querySelector("#speaker-modal-close-btn")?.addEventListener("click", closeModal);

  const renameBtn = card.querySelector("#speaker-rename-btn");
  const nameEl = card.querySelector("#speaker-modal-current-name");
  renameBtn?.addEventListener("click", () => {
    const currentName = nameEl?.textContent?.trim() || speakerName;
    const inputHtml = `
      <input type="text" class="speaker-rename-input" id="speaker-rename-input" value="${esc(currentName)}" />
      <button class="btn-ghost" id="speaker-rename-save" style="color:var(--success);font-weight:600">Save</button>
      <button class="btn-ghost" id="speaker-rename-cancel" style="color:var(--text-muted)">Cancel</button>
    `;
    nameEl.style.display = "none";
    renameBtn.style.display = "none";
    const container = document.createElement("div");
    container.className = "speaker-rename-row";
    container.innerHTML = inputHtml;
    nameEl.after(container);
    const input = container.querySelector("#speaker-rename-input");
    input.focus();
    input.select();

    const doRename = () => {
      const newName = input.value.trim();
      container.remove();
      nameEl.style.display = "";
      renameBtn.style.display = "";
      if (!newName || newName === currentName) return;

      nameEl.textContent = newName;

      // Collect every cluster_id this speaker maps to. Usually just one,
      // but a speaker can span multiple clusters after a diarize remap.
      const clusterIds = new Set();
      for (const s of segments) {
        const sp = (s.speakers || [])[0];
        if (sp && sp.cluster_id != null) {
          clusterIds.add(sp.cluster_id);
        }
      }
      // Also sweep the in-memory store so we catch clusters whose
      // events are in store but weren't in the segments list passed
      // to this modal (e.g. revisions that landed after the modal opened).
      if (store?.segments) {
        for (const [, ev] of store.segments) {
          if (ev.speakers?.length > 0) {
            const s = ev.speakers[0];
            if (s.identity === currentName || s.display_name === currentName) {
              s.identity = newName;
              s.display_name = newName;
              if (s.cluster_id != null) clusterIds.add(s.cluster_id);
            }
          }
        }
      }

      // Stamp the new name into the speaker registry.
      for (const cid of clusterIds) {
        renameSpeaker(cid, newName);
      }

      // Tell the caller the registry changed so it can repaint the
      // transcript + detected-speakers strip.
      _onChanged();

      // Timeline lane labels are drawn once per meeting load — update
      // them in place since they don't go through the registry.
      document.querySelectorAll(".speaker-timeline-lane-label").forEach((el) => {
        if (el.textContent === currentName) el.textContent = newName;
      });

      // Persist via API — send old_name so server can update all meeting files.
      if (mid && segments.length > 0) {
        const firstSeg = segments.find((s) => s.segment_id);
        if (firstSeg) {
          fetch(_meetingsUrl(mid, `/events/${_enc(firstSeg.segment_id)}/speaker`), {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ speaker_name: newName, old_name: currentName }),
          }).catch(() => {});
        }
        for (const seg of segments.slice(1)) {
          if (seg.segment_id) {
            fetch(_meetingsUrl(mid, `/events/${_enc(seg.segment_id)}/speaker`), {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ speaker_name: newName }),
            }).catch(() => {});
          }
        }
      }
    };

    container.querySelector("#speaker-rename-save").addEventListener("click", doRename);
    container.querySelector("#speaker-rename-cancel").addEventListener("click", () => {
      container.remove();
      nameEl.style.display = "";
      renameBtn.style.display = "";
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doRename();
      if (e.key === "Escape") {
        container.remove();
        nameEl.style.display = "";
        renameBtn.style.display = "";
      }
    });
  });

  if (mid) {
    card.querySelectorAll(".speaker-seg-play").forEach((btn) => {
      btn.addEventListener("click", () => {
        const seg = btn.closest(".speaker-segment");
        const startMs = parseInt(seg.dataset.start);
        const endMs = parseInt(seg.dataset.end);
        const segmentId = seg.dataset.segmentId || undefined;
        window.audioPlayer?.playSegment(
          mid,
          Math.max(0, startMs - SEEK_LEADIN_MS),
          endMs,
          segmentId,
        );
        card.querySelectorAll(".speaker-segment.playing").forEach((s) => s.classList.remove("playing"));
        seg.classList.add("playing");
      });
    });

    card.querySelector("#speaker-play-all")?.addEventListener("click", async () => {
      for (const seg of card.querySelectorAll(".speaker-segment")) {
        const startMs = parseInt(seg.dataset.start);
        const endMs = parseInt(seg.dataset.end);
        card.querySelectorAll(".speaker-segment.playing").forEach((s) => s.classList.remove("playing"));
        seg.classList.add("playing");
        seg.scrollIntoView({ behavior: "smooth", block: "center" });
        window.audioPlayer?.playSegment(mid, startMs, endMs);
        await new Promise((r) => {
          if (window.audioPlayer?.audio) window.audioPlayer.audio.onended = r;
          setTimeout(r, endMs - startMs + 1000);
        });
      }
    });
  }
}

/**
 * Open a modal to rename a speaker cluster from the detected-speakers
 * strip. POSTs to the cluster rename endpoint and updates the
 * client-side registry. `meetingId` falls back to
 * `window._gridRenderer?._meetingId` when omitted.
 *
 * hooks = { onSpeakerRegistryChanged }
 */
export function openSpeakerRenameModal(clusterId, currentName, color, hooks = {}) {
  const meetingId = hooks.meetingId ?? window._gridRenderer?._meetingId;
  const onChanged = hooks.onSpeakerRegistryChanged || (() => {});
  const seqIndex = getSpeakerSeqIndex(clusterId);
  showModal(`
    <div class="modal-card-header"><h2>Rename speaker</h2></div>
    <form id="rename-speaker-form" style="padding:1rem 1.25rem">
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem">
        <div style="width:40px;height:40px;border-radius:50%;background:${color};color:#fff;
                    display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1rem">
          ${seqIndex}
        </div>
        <div>
          <div style="font-size:0.85rem;font-weight:600">Currently: ${esc(currentName)}</div>
          <div style="font-size:0.7rem;color:var(--text-muted)">Cluster ${clusterId}</div>
        </div>
      </div>
      <label style="font-size:0.75rem;font-weight:500;color:var(--text-secondary);
                    display:block;margin-bottom:0.4rem">New name</label>
      <input type="text" id="rename-speaker-input" value="${esc(currentName)}"
             style="width:100%;padding:0.5rem 0.7rem;font-size:0.9rem;
                    border:1px solid var(--border);border-radius:4px;
                    background:var(--bg-surface);color:var(--text-primary)" />
      <div style="display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem">
        <button type="button" class="btn btn-ghost" onclick="closeModal()">Cancel</button>
        <button type="submit" class="btn btn-primary">Rename</button>
      </div>
    </form>
  `);
  const input = document.getElementById("rename-speaker-input");
  input?.focus();
  input?.select();
  document.getElementById("rename-speaker-form")?.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const newName = input.value.trim();
    if (!newName || newName === currentName) {
      closeModal();
      return;
    }
    // Optimistic client-side update
    renameSpeaker(clusterId, newName);
    onChanged();
    closeModal();

    if (meetingId) {
      try {
        const resp = await fetch(
          _meetingsUrl(meetingId, `/clusters/${clusterId}/name`),
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ speaker_name: newName }),
          },
        );
        if (!resp.ok) {
          console.warn("Rename persistence failed:", resp.status, await resp.text());
        }
      } catch (e) {
        console.warn("Rename request failed:", e);
      }
    }
  });
}

export function renderSpeakerTimeline(speakerLanes, durationMs, speakers, meetingId, hooks = {}) {
  const { store, onSpeakerRegistryChanged } = hooks;

  const container = document.getElementById("speaker-timeline");
  const lanesEl = document.getElementById("speaker-timeline-lanes");
  const timesEl = document.getElementById("speaker-timeline-times");
  const cursorEl = document.getElementById("speaker-timeline-cursor");

  if (!container || !lanesEl) return;

  container.style.display = "";
  lanesEl.innerHTML = "";
  timesEl.innerHTML = "";

  let zoomLevel = 1;
  const LABEL_WIDTH = 70;

  // Sort speakers by total speaking time (most talkative first)
  const sortedSpeakers = Object.entries(speakerLanes)
    .map(([id, blocks]) => {
      const serverName = speakers.find((s) => String(s.cluster_id) === id)?.display_name;
      const name = getSpeakerDisplayName(Number(id), serverName);
      return {
        id,
        blocks,
        totalMs: blocks.reduce((s, b) => s + (b.end_ms - b.start_ms), 0),
        name,
      };
    })
    .sort((a, b) => b.totalMs - a.totalMs);

  sortedSpeakers.forEach((speaker) => {
    const color = getSpeakerColor(speaker.id);
    const lane = document.createElement("div");
    lane.className = "speaker-timeline-lane";
    lane.dataset.clusterId = String(speaker.id);

    const label = document.createElement("div");
    label.className = "speaker-timeline-lane-label";
    label.textContent = speaker.name;
    label.style.color = color;
    label.style.cursor = "pointer";
    label.title = "Click to view speaker details";
    label.addEventListener("click", () => {
      showSpeakerModal(speaker.name, color, findSpeakerSegments(speaker.name, store), meetingId, {
        store,
        onSpeakerRegistryChanged,
      });
    });
    lane.appendChild(label);

    const track = document.createElement("div");
    track.className = "speaker-timeline-lane-track";

    for (const block of speaker.blocks) {
      const left = (block.start_ms / durationMs) * 100;
      const width = Math.max(0.15, ((block.end_ms - block.start_ms) / durationMs) * 100);
      const el = document.createElement("div");
      el.className = "speaker-timeline-block";
      el.style.left = `${left}%`;
      el.style.width = `${width}%`;
      el.style.background = color;
      el.dataset.segmentId = block.segment_id || "";
      el.dataset.startMs = String(block.start_ms);
      el.dataset.endMs = String(block.end_ms);
      el.title = `${speaker.name}: ${formatTime(block.start_ms)} - ${formatTime(block.end_ms)} — click to view transcript`;
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const seekStart = Math.max(0, block.start_ms - SEEK_LEADIN_MS);
        if (window.audioPlayer?._fullMeetingSrc) {
          window.audioPlayer.seekFullMeeting(seekStart, block.segment_id);
        } else {
          window.audioPlayer?.playSegment(meetingId, seekStart, block.end_ms, block.segment_id);
        }
        const row =
          document.querySelector(`.compact-block[data-segment-id="${block.segment_id}"]`) ||
          [...document.querySelectorAll("[data-segment-ids]")].find((e) =>
            e.dataset.segmentIds?.split(",").includes(block.segment_id),
          );
        if (row) {
          row.scrollIntoView({ behavior: "smooth", block: "center" });
          window.audioPlayer?._clearHighlight();
          row.classList.add("playing");
        }
      });
      track.appendChild(el);
    }

    track.addEventListener("click", (e) => {
      if (e.target === track) {
        const rect = track.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / (rect.width * zoomLevel);
        const clickMs = Math.min(pct, 1) * durationMs;
        const candidates = speaker.blocks;
        if (candidates && candidates.length) {
          const nearest = candidates.reduce((best, b) =>
            Math.abs(b.start_ms - clickMs) < Math.abs(best.start_ms - clickMs) ? b : best,
          );
          const seekStart = Math.max(0, nearest.start_ms - SEEK_LEADIN_MS);
          if (window.audioPlayer?._fullMeetingSrc) {
            window.audioPlayer.seekFullMeeting(seekStart, nearest.segment_id);
          } else {
            window.audioPlayer?.playSegment(meetingId, seekStart, nearest.end_ms, nearest.segment_id);
          }
          const row =
            document.querySelector(`.compact-block[data-segment-id="${nearest.segment_id}"]`) ||
            [...document.querySelectorAll("[data-segment-ids]")].find((el) =>
              el.dataset.segmentIds?.split(",").includes(nearest.segment_id),
            );
          if (row) {
            row.scrollIntoView({ behavior: "smooth", block: "center" });
            window.audioPlayer?._clearHighlight();
            row.classList.add("playing");
          }
        } else if (window.audioPlayer?._fullMeetingSrc) {
          window.audioPlayer.seekFullMeeting(clickMs);
        }
      }
    });

    lane.appendChild(track);
    lanesEl.appendChild(lane);
  });

  const applyZoom = () => {
    lanesEl.querySelectorAll(".speaker-timeline-lane-track").forEach((track) => {
      track.style.transform = `scaleX(${zoomLevel})`;
      track.style.transformOrigin = "left";
    });
    updateTimeMarkers();
  };

  const updateTimeMarkers = () => {
    timesEl.innerHTML = "";
    const numMarkers = Math.max(4, Math.min(12, Math.round(6 * zoomLevel)));
    for (let i = 0; i <= numMarkers; i++) {
      const ms = (i / numMarkers) * durationMs;
      const span = document.createElement("span");
      span.textContent = formatTime(ms);
      timesEl.appendChild(span);
    }
    timesEl.style.transform = `scaleX(${zoomLevel})`;
    timesEl.style.transformOrigin = "left";
  };
  updateTimeMarkers();

  container.addEventListener(
    "wheel",
    (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.2 : 0.83;
        zoomLevel = Math.max(1, Math.min(20, zoomLevel * factor));
        applyZoom();
      }
    },
    { passive: false },
  );

  const updateCursor = () => {
    if (
      window.audioPlayer?.audio?.duration &&
      window.audioPlayer.audio.currentTime > 0
    ) {
      const pct = window.audioPlayer.audio.currentTime / window.audioPlayer.audio.duration;
      const trackEl = lanesEl.querySelector(".speaker-timeline-lane-track");
      const baseWidth = trackEl ? trackEl.offsetWidth / zoomLevel : 1;
      const cursorX = LABEL_WIDTH + pct * baseWidth * zoomLevel;
      cursorEl.style.left = `${cursorX}px`;
      cursorEl.style.display = "";

      if (zoomLevel > 1 && container.scrollWidth > container.clientWidth) {
        const viewLeft = container.scrollLeft;
        const viewRight = viewLeft + container.clientWidth;
        if (cursorX < viewLeft + 50 || cursorX > viewRight - 50) {
          container.scrollLeft = cursorX - container.clientWidth / 2;
        }
      }
    } else {
      cursorEl.style.display = "none";
    }
    requestAnimationFrame(updateCursor);
  };
  requestAnimationFrame(updateCursor);
}
