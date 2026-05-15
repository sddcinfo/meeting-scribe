// Regression tests for Phase 1.3: meeting-controls hydration race
// (the reload-during-recording bug — see plan
// `bubbly-swinging-bird.md` §1.3).
//
// Symptom in production: the user starts a meeting, reloads the
// page, and the "Stop" button is gone — clicking it starts a NEW
// meeting instead. Root cause: the start/stop button at #btn-record
// reads ``document.body.classList.contains('recording')`` to decide
// its action. Pre-fix, that class was set only by ``startRecording()``
// in scribe-app.js — so a reload re-entered ``meeting-active`` mode
// (via ``enterLiveMeetingMode``) but never re-added ``recording``.
//
// Fix: the reconciler is the single owner of meeting state and
// authoritatively syncs ``body.classList.recording`` with
// ``status.meeting.state === 'recording'``. These tests lock that
// contract so a future refactor that pulls the class management out
// of the reconciler regresses red.
//
// Run:  node --test tests/js/reconcile.race.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createReconciler } from '../../static/js/meeting-reconcile.js';

// Local copies of the shims from reconcile.test.mjs, kept self-
// contained so this file doesn't grow a cross-module dependency.

function makeStorageShim() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
  };
}

function makeBodyShim() {
  const classes = new Set();
  return {
    classList: {
      add: (c) => classes.add(c),
      remove: (...cs) => { for (const c of cs) classes.delete(c); },
      contains: (c) => classes.has(c),
      _raw: classes,
    },
  };
}

function makeDocShim() {
  return { body: makeBodyShim() };
}

function defaultDeps(overrides = {}) {
  const doc = makeDocShim();
  const storage = makeStorageShim();
  const calls = {
    startRecording: [],
    attachViewOnlyWs: [],
    detachViewOnlyWs: [],
    loadMeetingJournal: [],
    resetStore: [],
    setStoreLive: [],
    renderTableStrip: [],
    showMeetingMode: [],
    showBanner: [],
    setTitle: [],
    onFinalizeCleanup: [],
  };
  const defaults = {
    doc,
    storage,
    fetchFn: async () => ({ json: async () => ({}) }),
    getAudioWsState: () => 'closed',
    startRecording: async (resume) => { calls.startRecording.push(resume); },
    attachViewOnlyWs: (mid) => { calls.attachViewOnlyWs.push(mid); },
    detachViewOnlyWs: () => { calls.detachViewOnlyWs.push(true); },
    loadMeetingJournal: async (mid) => {
      calls.loadMeetingJournal.push(mid);
      return { meta: { meeting_id: mid }, events: [] };
    },
    resetStore: () => { calls.resetStore.push(true); },
    setStoreLive: (enabled) => { calls.setStoreLive.push(enabled); },
    renderTableStrip: () => { calls.renderTableStrip.push(true); },
    showMeetingMode: () => { calls.showMeetingMode.push(true); },
    showBanner: (state) => { calls.showBanner.push(state); },
    setTitle: (t) => { calls.setTitle.push(t); },
    onFinalizeCleanup: () => {
      calls.onFinalizeCleanup.push(true);
      doc.body.classList.remove('recording', 'meeting-active', 'starting');
    },
    apiBase: '',
    popoutMode: false,
  };
  return { deps: { ...defaults, ...overrides }, calls, doc, storage };
}

// ── 1. The actual user bug — reload mid-meeting must add 'recording' ──

test("reload mid-meeting: server says recording → body.recording is set", async () => {
  const { deps, doc } = defaultDeps();
  // Fresh page load: no body classes set — exactly what scribe-app.js
  // sees right after a reload while the server-side meeting is still
  // active. ``startRecording()`` (which adds ``recording``) was NOT
  // called on this tab; the reconciler is the only path that can
  // catch up to server state.
  assert.equal(doc.body.classList.contains('recording'), false);
  assert.equal(doc.body.classList.contains('meeting-active'), false);

  const r = createReconciler(deps);
  await r.reconcile({
    meeting: { id: 'mtg-reload-1', state: 'recording' },
    connections: 0,
  });

  // Pre-fix this assertion failed: ``meeting-active`` was added but
  // ``recording`` stayed off, leaving the start/stop button stuck on
  // "Start". This is the regression guard.
  assert.equal(doc.body.classList.contains('recording'), true,
    "after reconciling with state='recording', body.classList must " +
    "contain 'recording' so #btn-record renders as a Stop button.");
  // ``meeting-active`` is added by enterLiveMeetingMode and was
  // already correct pre-fix; assert it for completeness.
  assert.equal(doc.body.classList.contains('meeting-active'), true);
});

// ── 2. Idempotent: a second reconcile with the same status doesn't toggle ──

test("repeated reconciles with state='recording' keep the class set", async () => {
  const { deps, doc } = defaultDeps();
  const r = createReconciler(deps);
  const status = {
    meeting: { id: 'mtg-1', state: 'recording' },
    connections: 0,
  };
  await r.reconcile(status);
  assert.equal(doc.body.classList.contains('recording'), true);
  // Subsequent polls (every 2 s during recording per scribe-app.js)
  // must not toggle the class.
  await r.reconcile(status);
  await r.reconcile(status);
  assert.equal(doc.body.classList.contains('recording'), true);
});

// ── 3. Server says no meeting → cleanup path removes the class ────────

test("server says no meeting → onFinalizeCleanup clears 'recording'", async () => {
  const { deps, calls, doc } = defaultDeps();
  // Simulate a tab that was recording: classes already on the body.
  doc.body.classList.add('recording');
  doc.body.classList.add('meeting-active');

  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: null, state: null }, connections: 0 });

  assert.equal(calls.onFinalizeCleanup.length, 1,
    'transitioning from recording → no-meeting must call onFinalizeCleanup');
  // The shim's onFinalizeCleanup mirrors the production version
  // (scribe-app.js:6890-6896), which removes all three classes.
  assert.equal(doc.body.classList.contains('recording'), false);
  assert.equal(doc.body.classList.contains('meeting-active'), false);
});

// ── 4. Stale poll arriving with state='idle' after a real start —
// the ordering matters for the start-then-stale-poll race ──

test("stale poll with no meeting after a real start does NOT remove the class out of order", async () => {
  // This case codifies the existing reconciler invariant: as long as
  // the body has not yet been told to record (no 'recording' class),
  // a stale poll arriving with state=null does not call
  // onFinalizeCleanup (the only writer that removes the class). The
  // teardown path is gated on ``bodyRecording``, so a no-op state
  // transition is genuinely a no-op.
  const { deps, calls } = defaultDeps();
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: null, state: null }, connections: 0 });
  assert.equal(calls.onFinalizeCleanup.length, 0,
    'no-meeting-from-no-meeting must not trigger cleanup; the ' +
    'cleanup path is reserved for the active→idle transition.');
});
