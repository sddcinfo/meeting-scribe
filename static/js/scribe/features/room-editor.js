// Meeting Scribe — Room Editor Overlay (mid-meeting + review).
//
// Owns the full-screen room-editor canvas (#room-editor-overlay) +
// the live/review cluster sidebar (#cluster-chips). The pure module
// here holds every function; the sibling `room-editor.bootstrap.js`
// wires the three DOM event listeners + publishes
// `window._onRoomLayoutUpdate`.
//
// External callers (the `meetings-list` row-action button, the
// `#meeting-table-strip` click in the bootstrap, the WS broadcast
// path that fires `_onRoomLayoutUpdate`) reach the editor through
// the named exports below. The admin SPA boot orchestrator calls
// `configureRoomEditor` once to inject:
//
//   * the singleton `roomSetup` instance,
//   * the `openSpeakerRenameModal` wrapper that binds the
//     speaker-strip refresh hook,
//   * the `refreshDetectedSpeakersStrip` helper (DOM repaint of the
//     in-meeting "detected speakers" strip).
//
// Speaker-registry helpers (`getAllSpeakers`, `getSpeakerColor`,
// `getSpeakerDisplayName`, `renameSpeaker`, `_speakerRegistry`,
// `refreshTranscriptSpeakerLabels`), modal helpers (`promptDialog`,
// `confirmDialog`), and the tiny `_enc` / `esc` helpers are imported
// directly — they're already pure modules.

import { _enc } from "../lib/meeting-url.js";
import { esc } from "../lib/escape.js";
import { getSpeakerColor } from "../lib/speaker-palette.js";
import {
  _speakerRegistry,
  getAllSpeakers,
  getSpeakerDisplayName,
  refreshTranscriptSpeakerLabels,
  renameSpeaker,
} from "./speaker-registry.js";
import { confirmDialog, promptDialog } from "./modal-system.js";

const API = "";

let _roomSetup = null;
let _openSpeakerRenameModalHook = () => {};
let _refreshDetectedSpeakersStripHook = () => {};

/**
 * One-shot init called by the admin SPA boot orchestrator. Must run
 * before any DOM handler tries to open the editor; the bootstrap
 * that wires those handlers loads after the orchestrator, so the
 * dependency is satisfied by load order.
 */
export function configureRoomEditor({
  roomSetup,
  openSpeakerRenameModal,
  refreshDetectedSpeakersStrip,
}) {
  _roomSetup = roomSetup;
  if (typeof openSpeakerRenameModal === "function") {
    _openSpeakerRenameModalHook = openSpeakerRenameModal;
  }
  if (typeof refreshDetectedSpeakersStrip === "function") {
    _refreshDetectedSpeakersStripHook = refreshDetectedSpeakersStrip;
  }
}


export function openRoomEditor(mode, meetingId) {
  const roomSetup = _roomSetup;
  if (!roomSetup) return;

  const overlay = document.getElementById("room-editor-overlay");
  const canvas = document.getElementById("room-editor-canvas");
  const sidebar = document.getElementById("room-editor-sidebar");
  const titleEl = document.getElementById("room-editor-title-text");
  const badge = document.getElementById("room-editor-mode-badge");

  if (!overlay || !canvas) return;

  titleEl.textContent =
    mode === "review"
      ? "Edit Layout & Assign Voices"
      : mode === "live"
        ? "Edit Room & Assign Live Voices"
        : "Edit Room Layout";
  badge.className = `room-editor-mode-badge ${mode}`;
  badge.textContent = mode === "live" ? "LIVE" : mode === "review" ? "REVIEW" : "";
  // Show sidebar for BOTH live and review modes so the user can assign
  // detected voices to seats mid-meeting.
  sidebar.style.display = mode === "review" || mode === "live" ? "" : "none";

  if (mode === "review" && meetingId) {
    // Load persisted room.json for this past meeting.
    fetch(`${API}/api/meetings/${_enc(meetingId)}/room`)
      .then((r) => (r.ok ? r.json() : null))
      .then((layout) => {
        if (layout) {
          roomSetup.tables.forEach((t) => t.element?.remove());
          roomSetup.seats.forEach((s) => s.element?.remove());
          roomSetup.tables = [];
          roomSetup.seats = [];
          (layout.tables || []).forEach((t) => {
            const tbl = {
              tableId: t.table_id,
              x: t.x,
              y: t.y,
              width: t.width,
              height: t.height,
              borderRadius: t.border_radius,
              label: t.label || "",
              element: null,
            };
            roomSetup.tables.push(tbl);
            roomSetup._renderTable(tbl);
            roomSetup._applyTable(tbl);
          });
          (layout.seats || []).forEach((s) => {
            const seat = {
              seatId: s.seat_id,
              x: s.x,
              y: s.y,
              name: s.speaker_name || "",
              enrolled: !!s.enrollment_id,
              enrollmentId: s.enrollment_id || null,
              element: null,
            };
            roomSetup.seats.push(seat);
            roomSetup._renderSeat(seat);
            roomSetup._applySeat(seat);
          });
        }
        roomSetup.mount(canvas, mode, { meetingId });
        _populateClusterChips(meetingId);
      })
      .catch(() => {
        roomSetup.mount(canvas, mode, { meetingId });
      });
  } else if (mode === "live" && meetingId) {
    // For an active meeting: keep the user's existing room layout but
    // (1) ensure there's a seat for every detected speaker, and
    // (2) populate the sidebar with currently detected voices.
    roomSetup.mount(canvas, mode, { meetingId });
    // If the user hasn't set up any table yet, give them a default one
    // so new seats can be arranged around it.
    if (roomSetup.tables.length === 0) {
      roomSetup.addTable();
    }
    roomSetup.reconcileSeatsToDetectedSpeakers();
    _populateClusterChipsLive(meetingId);
  } else {
    roomSetup.mount(canvas, mode, { meetingId });
  }

  overlay.style.display = "flex";

  overlay.querySelectorAll(".room-editor-actions .preset-btn").forEach((btn) => {
    btn.onclick = () => roomSetup.applyPreset(btn.dataset.preset);
  });
  document.getElementById("room-editor-add-table").onclick = () => roomSetup.addTable();
  document.getElementById("room-editor-add-seat").onclick = () => roomSetup.addSeatAtCenter();
}


export function closeRoomEditor() {
  const roomSetup = _roomSetup;
  if (!roomSetup) return;

  const overlay = document.getElementById("room-editor-overlay");
  if (!overlay || overlay.style.display === "none") return;
  overlay.style.display = "none";
  roomSetup.unmount();
  // Stop the live-chip refresh loop if it was running.
  if (window._liveChipRefreshTimer) {
    clearInterval(window._liveChipRefreshTimer);
    window._liveChipRefreshTimer = null;
  }
  if (document.getElementById("meeting-mode")?.style.display !== "none") {
    roomSetup._renderTableStrip();
  }
}


/**
 * Populate the cluster sidebar with LIVE detected speakers from the
 * client-side speaker registry. Refreshes every 2s while the editor
 * is open so new speakers appear as they're detected.
 */
function _populateClusterChipsLive(meetingId) {
  const roomSetup = _roomSetup;
  const chipsEl = document.getElementById("cluster-chips");
  if (!chipsEl) return;

  const render = () => {
    const speakers = getAllSpeakers();
    if (speakers.length === 0) {
      chipsEl.innerHTML =
        '<div style="color:var(--text-muted);font-size:0.75rem;padding:0.5rem">' +
        "No voices detected yet. Speakers will appear here as they speak.</div>";
      return;
    }

    const boundNames = new Set(roomSetup.seats.map((s) => s.name).filter(Boolean));
    chipsEl.innerHTML = "";

    speakers.forEach((sp) => {
      const clusterId = sp.clusterId;
      const name = sp.displayName;
      const color = getSpeakerColor(clusterId);
      const isBound = boundNames.has(name);

      const chip = document.createElement("div");
      chip.className = `cluster-chip${isBound ? " cluster-chip-bound" : ""}`;
      chip.draggable = !isBound;
      chip.dataset.clusterId = String(clusterId);
      chip.dataset.displayName = name;
      chip.style.setProperty("--chip-color", color);
      chip.innerHTML = `
        <div class="cluster-chip-swatch"></div>
        <div class="cluster-chip-body">
          <div class="cluster-chip-name">${esc(name)}</div>
          <div class="cluster-chip-stats">Speaker ${sp.seqIndex}${sp.hasCustomName ? " · named" : ""}</div>
        </div>
        <button class="cluster-chip-rename" title="Rename this voice">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
        </button>
      `;

      chip.querySelector(".cluster-chip-rename")?.addEventListener("click", (e) => {
        e.stopPropagation();
        _openSpeakerRenameModalHook(clusterId, name, color);
      });

      if (!isBound) {
        chip.addEventListener("dragstart", (ev) => {
          chip.classList.add("dragging");
          ev.dataTransfer.setData(
            "text/plain",
            JSON.stringify({
              cluster_id: clusterId,
              display_name: name,
            }),
          );
          ev.dataTransfer.effectAllowed = "move";
        });
        chip.addEventListener("dragend", () => {
          chip.classList.remove("dragging");
          document
            .querySelectorAll(".seat-node.drop-target")
            .forEach((el) => el.classList.remove("drop-target"));
        });
      }

      chip.addEventListener("dblclick", () => {
        _openSpeakerRenameModalHook(clusterId, name, color);
      });

      chipsEl.appendChild(chip);
    });

    _wireSeatDropTargets(meetingId);
  };

  render();
  // Auto-refresh every 2s so new speakers show up as they join. Also
  // reconcile the canvas seats so each new speaker gets a new seat.
  if (window._liveChipRefreshTimer) clearInterval(window._liveChipRefreshTimer);
  window._liveChipRefreshTimer = setInterval(() => {
    roomSetup.reconcileSeatsToDetectedSpeakers();
    render();
    _wireSeatDropTargets(meetingId);
  }, 2000);
}


async function _populateClusterChips(meetingId) {
  const roomSetup = _roomSetup;
  const chipsEl = document.getElementById("cluster-chips");
  if (!chipsEl) return;
  chipsEl.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem">Loading...</div>';

  try {
    const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/speakers`);
    const data = await resp.json();
    const speakers = data.speakers || [];
    if (speakers.length === 0) {
      chipsEl.innerHTML =
        '<div style="color:var(--text-muted);font-size:0.75rem">No voices detected.</div>';
      return;
    }

    const boundNames = new Set(roomSetup.seats.map((s) => s.name).filter(Boolean));

    chipsEl.innerHTML = "";
    speakers.forEach((sp, i) => {
      const clusterId = sp.cluster_id ?? sp.speaker_id ?? i;
      // Unified naming — ignores generic "Speaker N" server labels.
      const name = getSpeakerDisplayName(Number(clusterId), sp.display_name);
      const color = getSpeakerColor(Number(clusterId));
      const segCount = sp.segment_count || 0;
      const totalMs = sp.total_speaking_ms || 0;
      const totalSec = Math.round(totalMs / 1000);
      const timeStr =
        totalSec >= 60 ? `${Math.floor(totalSec / 60)}m ${totalSec % 60}s` : `${totalSec}s`;
      const isBound = boundNames.has(name);

      const firstSeenMs = sp.first_seen_ms || 0;

      const chip = document.createElement("div");
      chip.className = `cluster-chip${isBound ? " cluster-chip-bound" : ""}`;
      chip.draggable = !isBound;
      chip.dataset.clusterId = String(clusterId);
      chip.dataset.displayName = name;
      chip.dataset.firstSeenMs = String(firstSeenMs);
      chip.style.setProperty("--chip-color", color);
      chip.innerHTML = `
        <div class="cluster-chip-swatch"></div>
        <div class="cluster-chip-body">
          <div class="cluster-chip-name">${esc(name)}</div>
          <div class="cluster-chip-stats">${segCount} segments · ${timeStr}</div>
        </div>
        <button class="cluster-chip-play" title="Play a 4s sample">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
        </button>
      `;

      // Play a 4-second sample starting at first_seen_ms.
      chip.querySelector(".cluster-chip-play")?.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!window.audioPlayer?.audio) return;
        window.audioPlayer.seekFullMeeting?.(firstSeenMs);
        // Auto-pause after ~4s.
        clearTimeout(window._clusterSampleTimer);
        window._clusterSampleTimer = setTimeout(() => {
          if (window.audioPlayer?.audio && !window.audioPlayer.audio.paused) {
            window.audioPlayer.audio.pause();
            const btn = document.getElementById("player-play");
            if (btn) {
              btn.textContent = "▶";
              btn.classList.remove("playing");
            }
          }
        }, 4000);
      });

      if (!isBound) {
        chip.addEventListener("dragstart", (e) => {
          chip.classList.add("dragging");
          e.dataTransfer.setData(
            "text/plain",
            JSON.stringify({
              cluster_id: Number(clusterId),
              display_name: name,
            }),
          );
          e.dataTransfer.effectAllowed = "move";
        });
        chip.addEventListener("dragend", () => {
          chip.classList.remove("dragging");
          document
            .querySelectorAll(".seat-node.drop-target")
            .forEach((el) => el.classList.remove("drop-target"));
        });
      }

      chip.addEventListener("dblclick", async () => {
        const newName = await promptDialog(
          "Rename voice",
          "Set a display name for this voice cluster. Used wherever the cluster appears (transcript, timeline, speaker lanes).",
          {
            initialValue: name,
            placeholder: "Voice name",
            confirmText: "Rename",
          },
        );
        if (!newName || newName === name) return;
        _assignCluster(meetingId, Number(clusterId), null, newName);
      });

      chipsEl.appendChild(chip);
    });

    _wireSeatDropTargets(meetingId);
  } catch (_e) {
    chipsEl.innerHTML =
      '<div style="color:var(--text-muted);font-size:0.75rem">Error loading voices</div>';
  }
}


function _wireSeatDropTargets(meetingId) {
  const roomSetup = _roomSetup;
  document.querySelectorAll("#room-editor-canvas .seat-node").forEach((seatEl) => {
    seatEl.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      seatEl.classList.add("drop-target");
    });
    seatEl.addEventListener("dragleave", () => {
      seatEl.classList.remove("drop-target");
    });
    seatEl.addEventListener("drop", async (e) => {
      e.preventDefault();
      seatEl.classList.remove("drop-target");
      try {
        const data = JSON.parse(e.dataTransfer.getData("text/plain"));
        const seatId = seatEl.dataset.seatId;
        if (!seatId) return;
        await _assignCluster(meetingId, data.cluster_id, seatId, data.display_name);
      } catch {
        /* ignore malformed payload */
      }
    });

    // Right-click a bound seat → unbind the cluster.
    seatEl.addEventListener("contextmenu", async (e) => {
      const seatId = seatEl.dataset.seatId;
      if (!seatId) return;
      const seat = roomSetup.seats.find((s) => s.seatId === seatId);
      if (!seat || !seat.name) return; // Only unbind if there's something bound.
      e.preventDefault();
      if (
        !(await confirmDialog(
          "Unbind speaker?",
          `Remove "${esc(seat.name)}" from this seat?`,
          "Unbind",
          true,
        ))
      ) {
        return;
      }
      await _unbindCluster(meetingId, seatId);
    });
  });
}


async function _unbindCluster(meetingId, seatId) {
  const roomSetup = _roomSetup;
  const seat = roomSetup.seats.find((s) => s.seatId === seatId);
  if (!seat) return;

  // Fetch the current speakers list to find which cluster this name maps to.
  let clusterId = null;
  try {
    const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/speakers`);
    const data = await resp.json();
    const match = (data.speakers || []).find((sp) => (sp.display_name || "") === seat.name);
    if (match) {
      clusterId = match.cluster_id ?? match.speaker_id ?? null;
    }
  } catch {
    /* fall through to local-only unbind */
  }

  // Clear the seat name locally and persist via /room/layout.
  seat.name = "";
  const nameEl = seat.element?.querySelector(".seat-name");
  if (nameEl) nameEl.textContent = "";
  roomSetup._persistLayout();

  // Reset the cluster's display name in the client registry, then sync
  // the backend with the sequential fallback produced by
  // getSpeakerDisplayName.
  if (clusterId != null) {
    const cid = Number(clusterId);
    // Clear any user-assigned name and re-derive the sequential label.
    const entry = _speakerRegistry.clusters.get(cid);
    if (entry) entry.displayName = null;
    _speakerRegistry.clusters.set(cid, entry || { seqIndex: _speakerRegistry.nextIndex++ });
    const defaultName = getSpeakerDisplayName(cid);
    try {
      await fetch(`${API}/api/meetings/${_enc(meetingId)}/speakers/assign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cluster_id: cid,
          seat_id: null,
          display_name: defaultName,
        }),
      });
    } catch {
      /* persist failure is non-fatal — UI already cleared */
    }
    refreshTranscriptSpeakerLabels();
    _refreshDetectedSpeakersStripHook();
  }

  // Refresh chip sidebar.
  await _populateClusterChips(meetingId);
}


async function _assignCluster(meetingId, clusterId, seatId, displayName) {
  const roomSetup = _roomSetup;
  try {
    const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/speakers/assign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cluster_id: clusterId,
        seat_id: seatId,
        display_name: displayName,
      }),
    });
    if (!resp.ok) {
      console.error("Assign failed:", await resp.text());
      return;
    }
    if (seatId) {
      const seat = roomSetup.seats.find((s) => s.seatId === seatId);
      if (seat) {
        seat.name = displayName;
        const nameEl = seat.element?.querySelector(".seat-name");
        if (nameEl) nameEl.textContent = displayName;
      }
    }
    // Also update the client-side registry so the transcript + seat strip
    // reflect the new name immediately.
    renameSpeaker(clusterId, displayName);
    refreshTranscriptSpeakerLabels();
    // Refresh sidebar using whichever populator is appropriate.
    const badge = document.getElementById("room-editor-mode-badge");
    const isLive = badge?.classList.contains("live");
    if (isLive) {
      _populateClusterChipsLive(meetingId);
    } else {
      await _populateClusterChips(meetingId);
    }
  } catch (e) {
    console.error("Assign error:", e);
  }
}


/**
 * Refresh mini strip when the server broadcasts a layout update.
 * Stamped onto `window._onRoomLayoutUpdate` by the bootstrap module.
 */
export function applyRoomLayoutUpdate(layout) {
  const roomSetup = _roomSetup;
  if (!roomSetup || !layout) return;
  roomSetup.tables.forEach((t) => t.element?.remove());
  roomSetup.seats.forEach((s) => s.element?.remove());
  roomSetup.tables = [];
  roomSetup.seats = [];
  (layout.tables || []).forEach((t) => {
    const tbl = {
      tableId: t.table_id,
      x: t.x,
      y: t.y,
      width: t.width,
      height: t.height,
      borderRadius: t.border_radius,
      label: t.label || "",
      element: null,
    };
    roomSetup.tables.push(tbl);
    roomSetup._renderTable(tbl);
    roomSetup._applyTable(tbl);
  });
  (layout.seats || []).forEach((s) => {
    const seat = {
      seatId: s.seat_id,
      x: s.x,
      y: s.y,
      name: s.speaker_name || "",
      enrolled: !!s.enrollment_id,
      enrollmentId: s.enrollment_id || null,
      element: null,
    };
    roomSetup.seats.push(seat);
    roomSetup._renderSeat(seat);
    roomSetup._applySeat(seat);
  });
  roomSetup._renderTableStrip();
}
