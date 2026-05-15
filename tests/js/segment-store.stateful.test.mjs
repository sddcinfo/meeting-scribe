// Stateful / property-style regression tests for SegmentStore.
//
// The named regression tests in `segment-store.test.mjs` cover three
// historical bug shapes (translation lost to furigana race; speakers
// wiped by furigana update; control event clearing the grid). This
// suite generates RANDOM action sequences over the same primitive
// operations and asserts the same invariants — so a future race
// condition outside the named cases still trips a test.
//
// Approach: define a small set of action generators, fire N sequences
// of varying length, and after each sequence assert global invariants
// (final translation always survives if it was ever set; speakers
// survive a later furigana-only update; control events never produce
// phantom segments; etc.).
//
// 200 sequences × ~12 actions each = ~2400 ingestions per run; node:test
// completes the suite in <100 ms.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SegmentStore } from '../../static/js/segment-store.js';

// ── Random helpers ───────────────────────────────────────────────────

let _rng = 1;
function _seed(s) { _rng = s >>> 0; }
function _rand() {
  // Deterministic PRNG so failures are reproducible.
  _rng = (_rng * 1664525 + 1013904223) >>> 0;
  return _rng / 0x100000000;
}
function _pick(arr) { return arr[Math.floor(_rand() * arr.length)]; }
function _int(min, max) { return min + Math.floor(_rand() * (max - min + 1)); }

// ── Event factories ──────────────────────────────────────────────────

const SEGMENT_IDS = ['seg-A', 'seg-B', 'seg-C'];
const LANGS = ['en', 'ja', 'de', 'fr'];

function asrPartial({ seg, rev = 0 }) {
  return {
    segment_id: seg, revision: rev, is_final: false,
    start_ms: _int(0, 100000), end_ms: _int(0, 100000),
    language: _pick(LANGS), text: 'partial text', speakers: [],
  };
}

function asrFinal({ seg, rev = 0 }) {
  return {
    segment_id: seg, revision: rev, is_final: true,
    start_ms: _int(0, 100000), end_ms: _int(0, 100000),
    language: _pick(LANGS), text: 'final text', speakers: [],
  };
}

function translationInProgress({ seg, rev = 0, target = 'ja' }) {
  return {
    ...asrFinal({ seg, rev }),
    translation: { status: 'in_progress', text: null, target_language: target },
  };
}

function translationDone({ seg, rev = 0, target = 'ja' }) {
  return {
    ...asrFinal({ seg, rev }),
    translation: { status: 'done', text: 'translated', target_language: target },
  };
}

function furigana({ seg, rev = 1 }) {
  return {
    ...asrFinal({ seg, rev }),
    furigana_html: '<ruby>あ<rt>a</rt></ruby>',
  };
}

function speakerAttach({ seg, rev = 1, cluster = 1 }) {
  return {
    ...asrFinal({ seg, rev }),
    speakers: [{ cluster_id: cluster, identity: 'Tanaka', source: 'enrolled' }],
  };
}

function controlEventNoSegmentId() {
  // The popout-clear-on-pulse class — fired through ingestFromLiveWs
  // before the SegmentStore guard was added. Must NOT add anything.
  return {
    type: _pick(['speaker_pulse', 'audio_drift', 'meeting_warning', 'finalize_progress']),
    timestamp_ms: _int(0, 100000),
  };
}

const ACTIONS = [
  (seg) => asrPartial({ seg, rev: 0 }),
  (seg) => asrFinal({ seg, rev: 0 }),
  (seg) => asrFinal({ seg, rev: 1 }),
  (seg) => asrFinal({ seg, rev: 2 }),
  (seg) => translationInProgress({ seg, rev: 0 }),
  (seg) => translationDone({ seg, rev: 0 }),
  (seg) => translationDone({ seg, rev: 0, target: 'fr' }),
  (seg) => furigana({ seg, rev: 1 }),
  (seg) => furigana({ seg, rev: 2 }),
  (seg) => speakerAttach({ seg, rev: 1 }),
  (seg) => speakerAttach({ seg, rev: 2 }),
  () => controlEventNoSegmentId(),
];

// ── Invariants ───────────────────────────────────────────────────────

function assertInvariants(store, history) {
  // 1. Phantom: control events with no segment_id MUST NOT create
  //    entries in the store.
  assert.equal(store.segments.has(undefined), false,
    'phantom segment with undefined key created');
  assert.equal(store.segments.has(null), false,
    'phantom segment with null key created');

  // 2. Per-segment integrity:
  for (const [sid, rec] of store.segments) {
    assert.ok(sid, 'segment_id must be truthy');
    assert.equal(rec.segment_id, sid, 'record.segment_id must match key');
    assert.equal(typeof rec.revision, 'number', 'revision must be numeric');
    assert.ok(!Number.isNaN(rec.revision), 'revision must not be NaN');
  }

  // 3. Translation preservation: if any translation_done was ever
  //    ingested for a segment, the final translation.text must not
  //    be null (regression — translation was being wiped by furigana race).
  const lastDoneTranslation = new Map(); // segment_id → text
  for (const ev of history) {
    if (ev.translation && ev.translation.status === 'done' && ev.translation.text) {
      lastDoneTranslation.set(ev.segment_id, ev.translation.text);
    }
  }
  for (const [sid, expectedText] of lastDoneTranslation) {
    const rec = store.segments.get(sid);
    if (!rec) continue; // segment never reached the store
    const trText = rec.translation && rec.translation.text;
    assert.ok(trText, `translation.text wiped for ${sid} — last done was ${JSON.stringify(expectedText)}`);
  }

  // 4. Speaker preservation: if any speaker_attach was ever ingested
  //    with a non-empty speakers array, the final record's speakers
  //    array must also be non-empty.
  const everHadSpeakers = new Set();
  for (const ev of history) {
    if (ev.segment_id && (ev.speakers || []).length > 0) {
      everHadSpeakers.add(ev.segment_id);
    }
  }
  for (const sid of everHadSpeakers) {
    const rec = store.segments.get(sid);
    if (!rec) continue;
    assert.ok((rec.speakers || []).length > 0,
      `speakers wiped for ${sid} — final state has no speakers`);
  }

  // 5. Furigana preservation: same shape.
  const everHadFurigana = new Set();
  for (const ev of history) {
    if (ev.segment_id && ev.furigana_html) {
      everHadFurigana.add(ev.segment_id);
    }
  }
  for (const sid of everHadFurigana) {
    const rec = store.segments.get(sid);
    if (!rec) continue;
    assert.ok(rec.furigana_html, `furigana_html wiped for ${sid}`);
  }
}

// ── Property test ────────────────────────────────────────────────────

test('property: random sequences satisfy SegmentStore invariants', () => {
  const N_SEQUENCES = 200;
  const failures = [];

  for (let s = 0; s < N_SEQUENCES; s++) {
    _seed(s + 1);
    const seqLen = _int(5, 25);
    const store = new SegmentStore();
    const history = [];

    // Suppress expected console.error from the per-listener try/catch
    // (no listeners attached here, but the guard is still active).
    const origError = console.error;
    console.error = () => {};

    try {
      for (let i = 0; i < seqLen; i++) {
        const seg = _pick(SEGMENT_IDS);
        const make = _pick(ACTIONS);
        const ev = make(seg);
        history.push(ev);
        try { store.ingest(ev); } catch (e) {
          failures.push({ seed: s + 1, step: i, action: ev, error: String(e) });
        }
      }
      assertInvariants(store, history);
    } catch (e) {
      failures.push({ seed: s + 1, error: String(e), historyLen: history.length });
    } finally {
      console.error = origError;
    }
  }

  assert.deepEqual(failures, [],
    `property test failed for ${failures.length} of ${N_SEQUENCES} sequences. ` +
    `First failure: ${JSON.stringify(failures[0], null, 2)}`);
});
