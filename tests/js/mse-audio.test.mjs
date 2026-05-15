// Tests for resolveLiveTarget (static/js/mse-audio.js).
//
// Run:  node --test tests/js/mse-audio.test.mjs
//
// This function determines the relationship between an audio element's
// currentTime and the SourceBuffer's buffered ranges. It drives the
// trim/seek decisions in _mseDrainQueue (guest.html). The canonical
// implementation lives in guest.html's inline script; this module is
// the testable extraction.
//
// If any of these tests fail, the MSE drain queue's gap-seek and
// trim logic may be making wrong decisions — verify against the
// inline copy in guest.html before fixing.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveLiveTarget, MSE_MAX_BUFFERED_S, MSE_RE_ANCHOR_S, MSE_MIME } from '../../static/js/mse-audio.js';

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * Build a fake SourceBuffer-shaped object with a `.buffered` TimeRanges.
 * @param {Array<[number, number]>} ranges - [[start, end], ...]
 */
function fakeSb(ranges) {
  return {
    buffered: {
      length: ranges.length,
      start(i) { return ranges[i][0]; },
      end(i) { return ranges[i][1]; },
    },
  };
}

// ── Constants ───────────────────────────────────────────────────────

test('constants match expected values', () => {
  assert.equal(MSE_MAX_BUFFERED_S, 12.0);
  assert.equal(MSE_RE_ANCHOR_S, 30.0);
  assert.equal(MSE_MIME, 'audio/mp4; codecs="mp4a.40.2"');
});

// ── resolveLiveTarget ───────────────────────────────────────────────

test('empty buffered returns none', () => {
  const result = resolveLiveTarget(fakeSb([]), 5.0);
  assert.equal(result.kind, 'none');
  assert.equal(result.start, 0);
  assert.equal(result.end, 0);
});

test('null buffered returns none', () => {
  const result = resolveLiveTarget({ buffered: null }, 5.0);
  assert.equal(result.kind, 'none');
});

test('currentTime inside range returns live', () => {
  const result = resolveLiveTarget(fakeSb([[2.0, 10.0]]), 5.0);
  assert.equal(result.kind, 'live');
  assert.equal(result.start, 2.0);
  assert.equal(result.end, 10.0);
});

test('currentTime before range returns forward', () => {
  const result = resolveLiveTarget(fakeSb([[5.0, 10.0]]), 2.0);
  assert.equal(result.kind, 'forward');
  assert.equal(result.start, 5.0);
  assert.equal(result.end, 10.0);
});

test('currentTime after range returns behind', () => {
  const result = resolveLiveTarget(fakeSb([[2.0, 5.0]]), 10.0);
  assert.equal(result.kind, 'behind');
  assert.equal(result.start, 2.0);
  assert.equal(result.end, 5.0);
});

test('multiple ranges picks nearest forward', () => {
  // Three ranges: [1,3], [5,7], [10,12]. ct=4.0 is in the gap
  // between [1,3] and [5,7]. Nearest forward range is [5,7].
  const result = resolveLiveTarget(fakeSb([[1, 3], [5, 7], [10, 12]]), 4.0);
  assert.equal(result.kind, 'forward');
  assert.equal(result.start, 5);
  assert.equal(result.end, 7);
});

test('multiple ranges picks nearest behind when all before ct', () => {
  // ct=15.0. All ranges are before it. Should pick the closest (latest end).
  const result = resolveLiveTarget(fakeSb([[1, 3], [5, 7], [10, 12]]), 15.0);
  assert.equal(result.kind, 'behind');
  assert.equal(result.start, 10);
  assert.equal(result.end, 12);
});

test('currentTime at range start boundary is live', () => {
  // ct exactly equals start of range — the >= comparison includes it.
  const result = resolveLiveTarget(fakeSb([[5.0, 10.0]]), 5.0);
  assert.equal(result.kind, 'live');
  assert.equal(result.start, 5.0);
  assert.equal(result.end, 10.0);
});

test('currentTime at range end boundary is live', () => {
  // ct exactly equals end of range — the <= comparison includes it.
  const result = resolveLiveTarget(fakeSb([[5.0, 10.0]]), 10.0);
  assert.equal(result.kind, 'live');
  assert.equal(result.start, 5.0);
  assert.equal(result.end, 10.0);
});
