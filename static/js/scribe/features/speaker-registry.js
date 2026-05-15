// Meeting Scribe — Speaker registry (pure module).
//
// Maps raw cluster_ids (which may be large pseudo-IDs like 100, 101
// from the time-proximity fallback, or arbitrary numbers from
// diarization) to friendly sequential labels: "Speaker 1",
// "Speaker 2"… in first-seen order. Also honors user-assigned names
// (from the rename modal).
//
// Used everywhere a speaker needs to be displayed so the UI is
// consistent: transcript blocks, timeline lanes, speaker modals,
// participants panel, room strip, finalize summary.
//
// The registry itself is module-private state — consumers go through
// the named exports. Reset is exposed because the meeting-lifecycle
// code needs to clear the registry between meetings (otherwise
// sequential indices accumulate across meeting starts).

/** The registry map. Exported so `findSpeakerSegments` can walk
 *  entries by display name. */
export const _speakerRegistry = {
  // clusterId → {seqIndex, displayName}
  clusters: new Map(),
  nextIndex: 1,
};

/** True if a server-supplied name is a generic "Speaker <number>"
 *  label (e.g. "Speaker 0", "Speaker 101") that carries no human
 *  context and should be regenerated from the client-side sequential
 *  registry instead. */
export function isGenericSpeakerLabel(name) {
  if (!name) return true;
  // Strict: matches "Speaker 12", "Speaker 101", ignores anything else.
  return /^Speaker\s+\d+$/.test(String(name).trim());
}

/** True if the cluster_id is a pseudo / placeholder cluster (>=100).
 *  DEAD CODE AFTER 2026-04 SPEAKER-SEPARATION REFACTOR — the server
 *  no longer emits pseudo clusters. ``_process_event`` broadcasts
 *  events with ``speakers: []`` when diarization hasn't caught up,
 *  and ``_speaker_catchup_loop`` rebroadcasts a revised event once
 *  pyannote has a real cluster. This check is kept as defensive
 *  handling of any residual pseudo-cluster events from a rolling
 *  restart (old backend still running while the new frontend loads).
 *  Safe to remove in a future cleanup pass once no pseudo-cluster
 *  events are seen in the wild for a few meetings. */
export function isPseudoCluster(clusterId) {
  return clusterId != null && Number(clusterId) >= 100;
}

/** Get the display name for a cluster_id.
 *  Precedence:
 *    1. Human-assigned name (from rename modal) — persistent across
 *       renders
 *    2. Explicit identity from backend (real name like "Brad", "田中")
 *    3. Sequential "Speaker N" from the registry (first-seen order)
 *    4. "Speaker ?" for null / pseudo / unresolved clusters
 *
 *  Events with no speaker attribution (empty ``speakers`` array on
 *  the wire) arrive with clusterId === null. After the 2026-04
 *  speaker separation refactor, that is the common case for the
 *  first 0–2 s of any segment — ASR has produced text but
 *  diarization hasn't caught up yet. The catch-up loop will send a
 *  revised event within a few hundred ms with the real cluster_id,
 *  and the registry assigns a sequential index at that point. Show
 *  "Speaker ?" in the interim, not "Unknown", so the placeholder is
 *  visually identical to pseudo-cluster rendering and users don't
 *  see a scary "Unknown" flash. */
export function getSpeakerDisplayName(clusterId, explicitName) {
  if (clusterId == null) return "Speaker ?";
  // Pseudo-cluster: transient placeholder — don't allocate a seq index.
  if (isPseudoCluster(clusterId)) {
    const existing = _speakerRegistry.clusters.get(clusterId);
    if (existing?.displayName) return existing.displayName;
    return "Speaker ?";
  }
  let entry = _speakerRegistry.clusters.get(clusterId);
  if (!entry) {
    entry = { seqIndex: _speakerRegistry.nextIndex++, displayName: null };
    _speakerRegistry.clusters.set(clusterId, entry);
  }
  // Trust the server-supplied name. Post-2026-04 the server remaps
  // raw diarize cluster_ids to a stable seq_index at finalize, so
  // "Speaker 3" from the server IS the canonical label for cluster 3
  // across the transcript, timeline lanes, participants panel, and
  // summary — NOT a generic fallback to be renumbered client-side.
  // The previous isGenericSpeakerLabel gate was silently renumbering
  // speakers based on UI iteration order, which diverged from the
  // server's seq_index. Real user-set names still win because they
  // flow through ``renameSpeaker(cluster_id, name)`` on
  // ``speaker_rename`` WS events and get stamped into entry.displayName
  // below.
  if (explicitName) {
    entry.displayName = explicitName;
  }
  return entry.displayName || `Speaker ${entry.seqIndex}`;
}

/** Get the sequential number assigned to this cluster (1, 2, 3…) —
 *  useful for coloring/avatar text regardless of whether the speaker
 *  has been named. Pseudo-clusters return 0 (no seat). */
export function getSpeakerSeqIndex(clusterId) {
  if (clusterId == null) return 0;
  if (isPseudoCluster(clusterId)) return 0;
  const entry = _speakerRegistry.clusters.get(clusterId);
  if (entry) return entry.seqIndex;
  // Auto-register.
  const seq = _speakerRegistry.nextIndex++;
  _speakerRegistry.clusters.set(clusterId, {
    seqIndex: seq,
    displayName: null,
  });
  return seq;
}

/** Assign a user-chosen display name to a cluster. Returns the new
 *  name. */
export function renameSpeaker(clusterId, newName) {
  if (clusterId == null) return null;
  const trimmed = (newName || "").trim();
  if (!trimmed) return null;
  const entry = _speakerRegistry.clusters.get(clusterId) || {};
  entry.displayName = trimmed;
  if (entry.seqIndex == null) entry.seqIndex = _speakerRegistry.nextIndex++;
  _speakerRegistry.clusters.set(clusterId, entry);
  return trimmed;
}

/** Walk through all REAL speakers (not pseudo-clusters) — used by the
 *  virtual table to render one seat per detected cluster in
 *  first-seen order. */
export function getAllSpeakers() {
  return [..._speakerRegistry.clusters.entries()]
    .filter(([clusterId]) => !isPseudoCluster(clusterId))
    .map(([clusterId, entry]) => ({
      clusterId,
      seqIndex: entry.seqIndex,
      displayName: entry.displayName || `Speaker ${entry.seqIndex}`,
      hasCustomName: !!entry.displayName,
    }))
    .sort((a, b) => a.seqIndex - b.seqIndex);
}

/** Reset the registry (between meetings). */
export function resetSpeakerRegistry() {
  _speakerRegistry.clusters.clear();
  _speakerRegistry.nextIndex = 1;
}

/**
 * Walk rendered transcript blocks and update their speaker labels +
 * colors from the current speaker registry. Used after a rename so
 * previously rendered blocks show the new name without a full reload.
 * Pure DOM — no callers need to inject state.
 */
import { getSpeakerColor as _getSpeakerColor } from "../lib/speaker-palette.js";
export function refreshTranscriptSpeakerLabels() {
  document.querySelectorAll(".compact-block[data-cluster-id]").forEach((row) => {
    const cid = parseInt(row.dataset.clusterId, 10);
    if (isNaN(cid)) return;
    const newName = getSpeakerDisplayName(cid);
    const speakerEl = row.querySelector(".compact-speaker");
    if (speakerEl && speakerEl.textContent !== newName) {
      speakerEl.textContent = newName;
    }
    const color = _getSpeakerColor(cid);
    row.style.setProperty("--speaker-color", color);
    if (speakerEl) speakerEl.style.color = color;
  });
}
