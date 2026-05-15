// Meeting Scribe — Speaker color palette.
//
// Canonical speaker color assignment — keyed by cluster_id so the
// same speaker always gets the same color across:
//   * the timeline lanes (speaker-timeline feature)
//   * the transcript blocks (compact-grid feature)
//   * the speaker modals (speaker-registry feature)
//   * the room strip (room-setup feature)
//   * scrub-bar bands (audio-player feature)
//
// Using cluster_id directly (not sorted order) ensures stability
// across re-sorts. The palette is intentionally small + warm — a
// larger palette risks colors that look identical at the chip size.

export const SPEAKER_COLORS = [
  "#c45d20",
  "#1a6fb5",
  "#2a8540",
  "#9b2d7b",
  "#8b6914",
  "#b52d2d",
  "#2d6b5e",
  "#6b3fa0",
];

export function getSpeakerColor(clusterId) {
  if (clusterId == null) return SPEAKER_COLORS[0];
  const idx = Math.abs(parseInt(clusterId, 10) || 0) % SPEAKER_COLORS.length;
  return SPEAKER_COLORS[idx];
}
