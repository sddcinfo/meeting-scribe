// MSE audio helpers extracted from guest.html for testability.
//
// Contains the pure function `resolveLiveTarget` which determines the
// relationship between the audio element's currentTime and the
// SourceBuffer's buffered ranges. Used by _mseDrainQueue to make
// trim/seek decisions.
//
// Also exports the constants that govern MSE buffer management.

export const MSE_MAX_BUFFERED_S = 12.0;
export const MSE_RE_ANCHOR_S = 30.0;
export const MSE_MIME = 'audio/mp4; codecs="mp4a.40.2"';

/**
 * Analyse buffered ranges relative to currentTime.
 *
 * @param {SourceBuffer} sb - SourceBuffer (or any object with a `.buffered` TimeRanges)
 * @param {number} ct - currentTime of the audio element
 * @returns {{ kind: 'live'|'forward'|'behind'|'none', start: number, end: number }}
 */
export function resolveLiveTarget(sb, ct) {
  if (!sb.buffered || sb.buffered.length === 0) {
    return { kind: 'none', start: 0, end: 0 };
  }
  let forward = null, behind = null;
  for (let i = 0; i < sb.buffered.length; i++) {
    const s = sb.buffered.start(i);
    const e = sb.buffered.end(i);
    if (ct >= s && ct <= e) return { kind: 'live', start: s, end: e };
    if (s > ct && (forward === null || s < forward.start)) forward = { start: s, end: e };
    if (e < ct && (behind === null || e > behind.end))    behind  = { start: s, end: e };
  }
  if (forward) return { kind: 'forward', start: forward.start, end: forward.end };
  if (behind)  return { kind: 'behind',  start: behind.start,  end: behind.end };
  return { kind: 'none', start: 0, end: 0 };
}
