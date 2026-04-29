// Regression tests for SegmentStore (static/js/segment-store.js).
//
// Run:  node --test tests/js/
//
// These tests exist because the SegmentStore has been the source of three
// distinct production bugs where transcript dimensions silently disappeared:
//
//   1. Translation dropped when furigana arrived first (revision gate
//      rejected the translation rebroadcast because furigana had already
//      bumped revision).
//   2. Furigana dropped when speaker catch-up arrived first (equal-revision
//      gate didn't admit furigana_html as a valid update).
//   3. Speakers wiped by a higher-revision furigana update that didn't
//      carry them.
//
// All three boil down to the same root cause: the server pushes updates
// from independent background loops that don't coordinate revision numbers,
// and the client store needs to take the union across arrival orders.
//
// If any of these tests start failing, the SegmentStore has likely been
// "simplified" back into a stale-revision-drops-everything model. Read the
// comment block at the top of segment-store.js before changing them.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SegmentStore } from '../../static/js/segment-store.js';

// ── Helpers ─────────────────────────────────────────────────────────────────

const SEG = 'seg-abc';

const asrFinal = (overrides = {}) => ({
  segment_id: SEG,
  revision: 0,
  is_final: true,
  start_ms: 1000,
  end_ms: 2000,
  language: 'ja',
  text: '私は田中です',
  speakers: [],
  translation: null,
  furigana_html: null,
  ...overrides,
});

const furiganaUpdate = (overrides = {}) => ({
  ...asrFinal(),
  revision: 1,
  furigana_html: '<ruby>私<rt>わたし</rt></ruby>は<ruby>田中<rt>たなか</rt></ruby>です',
  ...overrides,
});

const speakerUpdate = (overrides = {}) => ({
  ...asrFinal(),
  revision: 1,
  speakers: [{ cluster_id: 4, identity: 'Tanaka', source: 'enrolled' }],
  ...overrides,
});

// `with_translation` on the server does NOT bump revision, so the
// translation worker rebroadcasts at the original revision (0). This is
// the exact wire shape that triggered the production bug.
const translationDone = (overrides = {}) => ({
  ...asrFinal(),
  revision: 0,
  translation: {
    status: 'done',
    text: 'I am Tanaka',
    target_language: 'en',
  },
  ...overrides,
});

const translationInProgress = (overrides = {}) => ({
  ...asrFinal(),
  revision: 0,
  translation: { status: 'in_progress', text: null, target_language: 'en' },
  ...overrides,
});

// ── Single-update sanity ────────────────────────────────────────────────────

test('initial ASR final lands on an empty store', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  const seg = store.segments.get(SEG);
  assert.equal(seg.text, '私は田中です');
  assert.equal(seg.revision, 0);
  assert.equal(seg.is_final, true);
  assert.equal(store.count, 1);
});

test('higher-revision update replaces text/revision', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal({ revision: 0, text: 'partial' }));
  store.ingest(asrFinal({ revision: 1, text: 'final' }));
  assert.equal(store.segments.get(SEG).text, 'final');
  assert.equal(store.segments.get(SEG).revision, 1);
});

test('lower-revision text update is ignored', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal({ revision: 5, text: 'newer' }));
  store.ingest(asrFinal({ revision: 1, text: 'older' }));
  assert.equal(store.segments.get(SEG).text, 'newer');
});

// ── REGRESSION: translation dropped by furigana race ───────────────────────
//
// The bug: ASR final arrives at rev=0. Furigana annotation finishes ~50 ms
// later and rebroadcasts at rev=1. Translation finishes ~300 ms later and
// rebroadcasts at rev=0 (because `with_translation` doesn't bump revision).
// A naive "drop if existing.rev > new.rev" gate dropped the translation
// silently, so every Japanese segment showed kanji + ruby but no English
// translation.

test('regression: translation arriving AFTER furigana is preserved', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());                // rev=0, no furigana, no translation
  store.ingest(furiganaUpdate());          // rev=1, furigana_html set
  store.ingest(translationDone());         // rev=0, translation.text set

  const seg = store.segments.get(SEG);
  assert.equal(seg.translation?.text, 'I am Tanaka', 'translation must survive lower-rev arrival');
  assert.ok(seg.furigana_html?.includes('<ruby>'), 'furigana must still be present');
  assert.equal(seg.revision, 1, 'highest revision wins for the primary dimension');
});

test('regression: translation arriving BEFORE furigana is preserved', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  store.ingest(translationDone());
  store.ingest(furiganaUpdate());

  const seg = store.segments.get(SEG);
  assert.equal(seg.translation?.text, 'I am Tanaka');
  assert.ok(seg.furigana_html?.includes('<ruby>'));
});

// ── REGRESSION: furigana dropped by speaker catch-up race ──────────────────
//
// Both furigana and speaker catch-up bump revision to N+1. Whichever lands
// second was being dropped by the equal-revision gate because furigana_html
// and speakers weren't in the gate's "is this a meaningful update" check.

test('regression: furigana arriving AFTER same-rev speaker update is preserved', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  store.ingest(speakerUpdate());           // rev=1 with speakers
  store.ingest(furiganaUpdate());          // rev=1 with furigana

  const seg = store.segments.get(SEG);
  assert.ok(seg.furigana_html?.includes('<ruby>'), 'furigana must land on top of speaker update');
  assert.equal(seg.speakers?.[0]?.identity, 'Tanaka', 'speakers must still be present');
});

test('regression: speaker update arriving AFTER same-rev furigana is preserved', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  store.ingest(furiganaUpdate());          // rev=1 with furigana
  store.ingest(speakerUpdate());           // rev=1 with speakers

  const seg = store.segments.get(SEG);
  assert.ok(seg.furigana_html?.includes('<ruby>'));
  assert.equal(seg.speakers?.[0]?.identity, 'Tanaka');
});

// ── REGRESSION: a dimension carried by an earlier update must not be
// silently wiped by a later update that simply omits it ────────────────────

test('regression: speakers from earlier update survive a later furigana update', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  store.ingest(speakerUpdate({ revision: 1 }));
  store.ingest(furiganaUpdate({ revision: 2 }));   // rev bump, no speakers field

  const seg = store.segments.get(SEG);
  assert.equal(seg.speakers?.[0]?.identity, 'Tanaka', 'speakers must NOT be wiped by a later furigana-only update');
  assert.ok(seg.furigana_html?.includes('<ruby>'));
});

test('regression: furigana from earlier update survives a later speaker update', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  store.ingest(furiganaUpdate({ revision: 1 }));
  store.ingest(speakerUpdate({ revision: 2 }));

  const seg = store.segments.get(SEG);
  assert.ok(seg.furigana_html?.includes('<ruby>'));
  assert.equal(seg.speakers?.[0]?.identity, 'Tanaka');
});

test('regression: translation from earlier update survives a later furigana update', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  store.ingest(translationDone());
  store.ingest(furiganaUpdate({ revision: 2 }));

  const seg = store.segments.get(SEG);
  assert.equal(seg.translation?.text, 'I am Tanaka');
  assert.ok(seg.furigana_html?.includes('<ruby>'));
});

// ── Translation status progression ──────────────────────────────────────────

test('translation in_progress → done both land', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  store.ingest(translationInProgress());
  assert.equal(store.segments.get(SEG).translation?.status, 'in_progress');
  store.ingest(translationDone());
  assert.equal(store.segments.get(SEG).translation?.text, 'I am Tanaka');
  assert.equal(store.segments.get(SEG).translation?.status, 'done');
});

test('a stale translation rebroadcast does not erase a completed translation', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  store.ingest(translationDone());
  // A later in_progress update — must not erase the completed text.
  store.ingest(translationInProgress());
  assert.equal(store.segments.get(SEG).translation?.text, 'I am Tanaka');
});

// ── Same-revision is_final flip ────────────────────────────────────────────

test('partial → same-revision final flips is_final', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal({ revision: 0, is_final: false, text: 'partial' }));
  store.ingest(asrFinal({ revision: 0, is_final: true, text: 'final' }));
  const seg = store.segments.get(SEG);
  assert.equal(seg.is_final, true);
  assert.equal(seg.text, 'final');
});

// ── Subscribers are notified ───────────────────────────────────────────────

test('subscribers fire on ingest', () => {
  const store = new SegmentStore();
  let calls = 0;
  let lastSeg = null;
  store.subscribe((id, seg, isNew) => {
    calls++;
    lastSeg = seg;
  });
  store.ingest(asrFinal());
  store.ingest(furiganaUpdate());
  store.ingest(translationDone());
  assert.equal(calls, 3);
  assert.equal(lastSeg.translation?.text, 'I am Tanaka');
  assert.ok(lastSeg.furigana_html?.includes('<ruby>'));
});

// ── clear ──────────────────────────────────────────────────────────────────

test('clear() empties the store and notifies subscribers', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  let cleared = false;
  store.subscribe((id) => {
    if (id === null) cleared = true;
  });
  store.clear();
  assert.equal(store.count, 0);
  assert.equal(cleared, true);
});

// ── Multi-target translation fan-out ──────────────────────────────────────
//
// Under the demand-driven pipeline the server sends one event per
// (segment_id, target_lang). The store must accumulate these into
// `translations[target_lang]` while keeping `.translation` populated
// with the most-complete copy for legacy render paths.

test('multi-target: two target-lang events accumulate under translations', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal({ language: 'en' }));
  store.ingest(translationDone({ translation: { status: 'done', text: '田中です', target_language: 'ja' } }));
  store.ingest(translationDone({ translation: { status: 'done', text: 'Je suis Tanaka', target_language: 'fr' } }));

  const seg = store.segments.get(SEG);
  assert.equal(seg.translations?.ja?.text, '田中です');
  assert.equal(seg.translations?.fr?.text, 'Je suis Tanaka');
  // Flat slot holds the last one that landed.
  assert.equal(seg.translation?.text, 'Je suis Tanaka');
});

test('multi-target: same target updating in_progress → done overwrites', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal({ language: 'en' }));
  store.ingest(translationDone({ translation: { status: 'in_progress', text: null, target_language: 'ja' } }));
  store.ingest(translationDone({ translation: { status: 'done', text: '田中です', target_language: 'ja' } }));

  const seg = store.segments.get(SEG);
  assert.equal(seg.translations?.ja?.status, 'done');
  assert.equal(seg.translations?.ja?.text, '田中です');
});

test('multi-target: translations survive a higher-rev update that omits them', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal({ language: 'en' }));
  store.ingest(translationDone({ translation: { status: 'done', text: '田中です', target_language: 'ja' } }));
  store.ingest(translationDone({ translation: { status: 'done', text: 'Je suis Tanaka', target_language: 'fr' } }));
  // Higher-rev furigana-only rebroadcast — no translation field.
  store.ingest(furiganaUpdate({ revision: 2, translation: null }));

  const seg = store.segments.get(SEG);
  assert.equal(seg.translations?.ja?.text, '田中です');
  assert.equal(seg.translations?.fr?.text, 'Je suis Tanaka');
});

// ── REGRESSION: control messages must not phantom-clear the renderer ──────
//
// The popout's /api/ws/view handler only filters lifecycle + speaker_rename
// + slide_* messages — every other control broadcast (speaker_pulse,
// seat_update, room_layout_update, speaker_remap, summary_regenerated,
// meeting_warning, audio_drift, tts_audio, …) falls through to
// `ingestFromLiveWs → store.ingest`. Those messages have no segment_id.
//
// If `ingest()` doesn't gate on segment_id, it fires listeners with
// segment_id=undefined; CompactGridRenderer.update treats the falsy id as
// a "store cleared" signal and calls _clear(), wiping the popout's grid
// every 200 ms (speaker_pulse cadence during an active meeting). The
// popout then only ever shows the sliver of utterances received between
// pulses — visibly diverging from the main window's full transcript.
//
// The explicit-clear path goes through `clear()`, which still notifies
// listeners with (null, null, false) — so this guard does not break
// intentional resets.
test('regression: control messages without segment_id are ignored', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  let calls = 0;
  let lastId;
  store.subscribe((id) => { calls++; lastId = id; });

  // speaker_pulse shape — reaches store.ingest via the popout fallthrough.
  store.ingest({ type: 'speaker_pulse', active_speakers: [], timestamp_ms: 1234 });
  store.ingest({ type: 'seat_update', speaker_name: 'Tanaka' });
  store.ingest({ type: 'audio_drift', drift_ms: 600 });
  store.ingest({ type: 'speaker_remap', renames: { 5: 4 } });

  assert.equal(calls, 0, 'control messages must not fire transcript listeners');
  assert.equal(store.count, 1, 'store must not gain phantom segments');
  assert.equal(store.segments.has(undefined), false,
    'control messages must not insert an undefined-keyed segment');

  // Sanity: explicit clear() still notifies (id===null) so the renderer
  // can reset on real resets.
  store.clear();
  assert.equal(lastId, null);
});

// ── REGRESSION: listener fan-out is isolated; one throw must not abort the
// rest of the listener chain ─────────────────────────────────────────────
//
// Found in `tests/browser/test_cross_window_sync.py` while testing the
// popout: a separate listener (the admin `#segment-count` chip updater)
// threw `Cannot set properties of null` because that element doesn't exist
// in popout-mode DOM. Set iteration aborted on the throw, and the next
// listener — the CompactGridRenderer subscription — never ran. Popout
// transcript stayed permanently empty even though events DID land in
// the store.
//
// Each subscriber must be isolated so one buggy/wrong-context listener
// can't take down the rest of the fan-out.
test('regression: a throwing listener does not block subsequent listeners', () => {
  const store = new SegmentStore();
  const calls = [];
  store.subscribe(() => { throw new Error('listener 0 boom'); });
  store.subscribe((id) => { calls.push(['l1', id]); });
  store.subscribe((id) => { calls.push(['l2', id]); });

  // Suppress the expected console.error so the test output stays clean.
  const origError = console.error;
  console.error = () => {};
  try {
    store.ingest(asrFinal());
  } finally {
    console.error = origError;
  }

  assert.deepEqual(calls, [['l1', SEG], ['l2', SEG]],
    'subscribers after a throwing one must still fire');
});

test('regression: a throwing listener on clear() does not block subsequent listeners', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal());
  const cleared = [];
  store.subscribe(() => { throw new Error('boom on clear'); });
  store.subscribe((id) => { cleared.push(id); });

  const origError = console.error;
  console.error = () => {};
  try {
    store.clear();
  } finally {
    console.error = origError;
  }

  assert.deepEqual(cleared, [null], 'clear() must still notify all subscribers');
});

// ── Multi-segment isolation ────────────────────────────────────────────────

test('updates on one segment do not leak into another', () => {
  const store = new SegmentStore();
  store.ingest(asrFinal({ segment_id: 'a' }));
  store.ingest(asrFinal({ segment_id: 'b' }));
  store.ingest(furiganaUpdate({ segment_id: 'a' }));
  store.ingest(translationDone({ segment_id: 'b' }));

  assert.ok(store.segments.get('a').furigana_html);
  assert.equal(store.segments.get('a').translation, null);
  assert.equal(store.segments.get('b').translation?.text, 'I am Tanaka');
  assert.equal(store.segments.get('b').furigana_html, null);
});
