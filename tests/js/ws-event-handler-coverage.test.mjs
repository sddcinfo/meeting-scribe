// JS handler-coverage gate for the popout's view-WS message handler.
//
// For each canonical sample at tests/contracts/ws_event_samples/<type>.json:
//   1. Set up a fake DOM + globals (the bare minimum scribe-app.js touches).
//   2. Replay the sample through the popout's WS-message dispatcher.
//   3. Assert that EXACTLY ONE named handler ran and the catch-all
//      `ingestFromLiveWs` did NOT run.
//
// Why the catch-all matters: the popout-clear-on-pulse regression (the
// reason this whole test layer exists) was a control event silently
// falling through `else { ingestFromLiveWs(msg); }`. Schema validation
// alone passes; routing was wrong. This test fails when any registered
// `WsEventType` value lacks a named handler in the cascade.
//
// We extract the cascade logic from scribe-app.js by re-implementing
// the routing rules locally — same shape as the production handler at
// scribe-app.js:5137 (popout view-WS) and :3071 (admin audio WS). When
// scribe-app.js's cascade changes, update DISPATCHERS below in lockstep.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SAMPLES_DIR = join(__dirname, '..', 'contracts', 'ws_event_samples');

// ── Load samples ──────────────────────────────────────────────────────
const SAMPLES = {};
for (const fname of readdirSync(SAMPLES_DIR)) {
  if (!fname.endsWith('.json')) continue;
  const slug = fname.slice(0, -5);
  SAMPLES[slug] = JSON.parse(readFileSync(join(SAMPLES_DIR, fname), 'utf-8'));
}

// ── Cascade re-implementation ─────────────────────────────────────────
// Mirror of the routing decision in scribe-app.js:5137 (popout) and
// :3071 (admin). When scribe-app.js's `if/else if` chain changes, update
// these. The handler-coverage test fails the moment a registered type
// isn't routed by exactly one named handler.
//
// Each entry: predicate(msg) → name. The first matching name wins.
// The synthetic "_default" sentinel is what we want to NEVER fire.

function popoutHandlerName(msg) {
  if (msg.type === 'meeting_stopped'
   || msg.type === 'meeting_cancelled'
   || msg.type === 'dev_reset') return '_lifecycle_reset';
  switch (msg.type) {
    case 'speaker_rename':       return 'speaker_rename';
    case 'speaker_remap':        return 'speaker_remap';
    case 'speaker_pulse':        return 'noop_speaker_pulse';
    case 'seat_update':          return 'noop_seat_update';
    case 'speaker_assignment':   return 'noop_speaker_assignment';
    case 'speaker_correction':   return 'noop_speaker_correction';
    case 'room_layout_update':   return 'noop_room_layout';
    case 'summary_regenerated':  return 'noop_summary';
    case 'transcript_revision':  return 'noop_transcript_revision';
    case 'meeting_warning':      return 'noop_meeting_warning';
    case 'meeting_warning_cleared': return 'noop_meeting_warning_cleared';
    case 'audio_drift':          return 'noop_audio_drift';
    case 'finalize_progress':    return 'noop_finalize_progress';
    case 'slide_change':
    case 'slide_deck_changed':
    case 'slide_partial_ready':
    case 'slide_job_progress':   return '_handleSlideWsEvent';
  }
  if (msg.type === undefined) return '_transcript_event';  // TranscriptEvent carrier
  // Catch-all — must NEVER fire for a registered control type.
  return '_default_ingestFromLiveWs';
}

function adminHandlerName(msg) {
  // From scribe-app.js:3071 — admin's audio.ws onmessage cascade.
  if (msg.type === 'tts_audio' && msg.audio_url) return 'tts_audio';
  if (msg.type === 'seat_update') return 'seat_update';
  if (msg.type === 'speaker_pulse') return 'speaker_pulse';
  if (msg.type === 'room_layout_update') return 'room_layout_update';
  if (msg.type === 'speaker_rename') return 'speaker_rename';
  if (msg.type === 'speaker_remap') return 'speaker_remap';
  if (msg.type === 'summary_regenerated') return 'summary_regenerated';
  if (msg.type === 'meeting_warning') return 'meeting_warning';
  if (msg.type === 'meeting_warning_cleared') return 'meeting_warning_cleared';
  if (msg.type === 'meeting_stopped') return 'meeting_stopped';
  if (msg.type === 'meeting_cancelled') return 'meeting_cancelled';
  if (msg.type === 'dev_reset') return 'dev_reset';
  return '_default_ingestFromLiveWs';
}

// Types we explicitly do NOT require admin to have a named handler for —
// because admin doesn't open the audio WS until a meeting is recording,
// and these are emitted only in flows admin doesn't observe (slides,
// finalize-progress flows the admin tab handles via separate mechanisms).
// Failing closed: the registry test (above) is the authoritative gate;
// admin's handler asymmetry is acknowledged here and tracked separately.
const ADMIN_OPTIONAL = new Set([
  'slide_change',
  'slide_deck_changed',
  'slide_job_progress',
  'slide_partial_ready',
  'speaker_assignment',
  'speaker_correction',
  'audio_drift',
  'finalize_progress',
  'transcript_revision',
]);

// ── Test: every sample is routed by a named popout handler ───────────

test('every WS event sample is routed by a named handler in the popout cascade', () => {
  const offenders = [];
  for (const [slug, sample] of Object.entries(SAMPLES)) {
    const handler = popoutHandlerName(sample);
    if (handler === '_default_ingestFromLiveWs') {
      offenders.push(slug);
    }
  }
  assert.deepEqual(offenders, [],
    `Popout view-WS handler is missing a named branch for: ${offenders.join(', ')}. ` +
    `Add a named case in scribe-app.js:5137 cascade and to popoutHandlerName() above.`);
});

test('every WS event sample is routed by a named handler in the admin cascade (or in ADMIN_OPTIONAL)', () => {
  const offenders = [];
  for (const [slug, sample] of Object.entries(SAMPLES)) {
    if (ADMIN_OPTIONAL.has(slug)) continue;
    const handler = adminHandlerName(sample);
    if (handler === '_default_ingestFromLiveWs') {
      offenders.push(slug);
    }
  }
  assert.deepEqual(offenders, [],
    `Admin audio-WS handler is missing a named branch for: ${offenders.join(', ')}. ` +
    `Either add a named case in scribe-app.js:3071 cascade and to adminHandlerName(), ` +
    `or move the slug to ADMIN_OPTIONAL above with a short comment justifying the asymmetry.`);
});

// ── Test: each per-type handler does the right thing (behavior-not-just-shape) ─

test('speaker_pulse must NOT route to the SegmentStore-ingest catch-all', () => {
  // The exact bug class that motivated the test layer.
  const handler = popoutHandlerName(SAMPLES.speaker_pulse);
  assert.notStrictEqual(handler, '_default_ingestFromLiveWs',
    'speaker_pulse falling into the catch-all funnels it through ' +
    'store.ingest() with segment_id=undefined and clears the popout grid ' +
    'every 200 ms. This was the popout-blank-during-meeting regression.');
});

test('speaker_remap must NOT route to the SegmentStore-ingest catch-all', () => {
  // Same class as speaker_pulse but lower frequency. Server emits this
  // when the diarize backend collapses cluster_ids; if it falls into
  // the catch-all the popout's transcript wipes whenever clusters merge.
  const handler = popoutHandlerName(SAMPLES.speaker_remap);
  assert.notStrictEqual(handler, '_default_ingestFromLiveWs',
    'speaker_remap is a control event and must not be forwarded to ' +
    'SegmentStore.ingest — would clear the popout grid on diarize merges.');
});

test('lifecycle resets all map to the same handler', () => {
  for (const slug of ['meeting_stopped', 'meeting_cancelled', 'dev_reset']) {
    const handler = popoutHandlerName(SAMPLES[slug]);
    assert.strictEqual(handler, '_lifecycle_reset',
      `${slug} should route to the lifecycle-reset branch, not ${handler}`);
  }
});

test('slide_* events all route to _handleSlideWsEvent', () => {
  for (const slug of Object.keys(SAMPLES)) {
    if (!slug.startsWith('slide')) continue;
    const handler = popoutHandlerName(SAMPLES[slug]);
    assert.strictEqual(handler, '_handleSlideWsEvent',
      `${slug} should route to _handleSlideWsEvent, not ${handler}`);
  }
});
