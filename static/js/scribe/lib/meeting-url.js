// Meeting-URL helpers used by every feature module that calls
// `/api/meetings/<mid>/...`. Escapes `mid` so CodeQL's
// `js/client-side-request-forgery` taint check is satisfied at every
// call site without 27 inline `encodeURIComponent` repetitions. Pass
// the suffix already-encoded if it contains a second user-controlled
// segment (e.g. segment_id).

export const _enc = encodeURIComponent;

export function _meetingsUrl(mid, suffix = "") {
  return `/api/meetings/${_enc(mid)}${suffix}`;
}
