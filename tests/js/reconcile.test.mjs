// Regression tests for the meeting-reconciliation state machine
// (static/js/meeting-reconcile.js).
//
// Run:  node --test tests/js/
//
// These tests exist because the client-side state machine has been
// the source of four distinct UX bugs:
//
//   1. Sometimes clicking Start doesn't actually start the meeting
//      (silent failure, stuck button).
//   2. Reloading the page while a meeting is running drops the user
//      back to the setup screen (no rehydration self-heal).
//   3. Navigating away during recording leaves no way to return,
//      and any observer tab would silently steal the mic.
//   4. A stuck "Reconnecting" banner could survive past the meeting
//      it described, because its clear path was only ws.onopen.
//
// All four boil down to: there was no single authority for client
// state, so every new call site re-invented its own heuristic. This
// module IS that authority, and these tests lock down its contract.
//
// If any of these tests start failing, read the plan file
// (smooth-dancing-cake.md) before "simplifying" the module.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createReconciler } from '../../static/js/meeting-reconcile.js';

// ── Harness ────────────────────────────────────────────────────

function makeStorageShim() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
    clear: () => { map.clear(); },
    _raw: map,
  };
}

function makeBodyShim() {
  const classes = new Set();
  return {
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
      toggle: (c, force) => {
        if (force === true) { classes.add(c); return true; }
        if (force === false) { classes.delete(c); return false; }
        if (classes.has(c)) { classes.delete(c); return false; }
        classes.add(c); return true;
      },
      _raw: classes,
    },
  };
}

function makeDocShim() {
  return {
    body: makeBodyShim(),
  };
}

function defaultDeps(overrides = {}) {
  const doc = makeDocShim();
  const storage = makeStorageShim();

  // Spies
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
    fetchFn: [],
  };

  const defaults = {
    doc,
    storage,
    fetchFn: async (...a) => { calls.fetchFn.push(a); return { json: async () => ({}) }; },
    getAudioWsState: () => 'closed',
    startRecording: async (resume) => { calls.startRecording.push(resume); },
    attachViewOnlyWs: (mid) => { calls.attachViewOnlyWs.push(mid); },
    detachViewOnlyWs: () => { calls.detachViewOnlyWs.push(true); },
    loadMeetingJournal: async (mid) => {
      calls.loadMeetingJournal.push(mid);
      return { meta: { meeting_id: mid, created_at: 0 }, events: [] };
    },
    resetStore: () => { calls.resetStore.push(true); },
    setStoreLive: (enabled) => { calls.setStoreLive.push(enabled); },
    renderTableStrip: () => { calls.renderTableStrip.push(true); },
    showMeetingMode: () => { calls.showMeetingMode.push(true); },
    showBanner: (state) => { calls.showBanner.push(state); },
    setTitle: (t) => { calls.setTitle.push(t); },
    onFinalizeCleanup: () => { calls.onFinalizeCleanup.push(true); },
    apiBase: '',
    popoutMode: false,
  };

  return { deps: { ...defaults, ...overrides }, calls, doc, storage };
}

// ── 1. Rehydration fires exactly once on idle→recording ────────

test('idle → recording rehydration calls enterLiveMeetingMode once', async () => {
  const { deps, calls } = defaultDeps();
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.equal(calls.loadMeetingJournal.length, 1);
  assert.equal(calls.loadMeetingJournal[0], 'mtg-1');
});

// ── 2. Concurrent reconciles guarded by _rehydrationInFlight ───

test('concurrent reconciles trigger rebuild only once', async () => {
  let release;
  const journalCalls = [];
  const { deps } = defaultDeps({
    loadMeetingJournal: async (mid) => {
      journalCalls.push(mid);
      await new Promise((res) => { release = res; });
      return { meta: { meeting_id: mid, created_at: 0 }, events: [] };
    },
  });
  const r = createReconciler(deps);
  const status = { meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 };
  const p1 = r.reconcile(status);
  const p2 = r.reconcile(status);
  // The custom loadMeetingJournal synchronously constructs the
  // pending Promise at p1 start, so `release` is now bound to that
  // Promise's resolver. p2's reconcile should have seen either body.
  // meeting-active (set by p1 inside enterLiveMeetingMode) OR the
  // _rehydrationInFlight guard, and skipped the rebuild.
  release();
  await Promise.all([p1, p2]);
  assert.equal(journalCalls.length, 1, `journal called ${journalCalls.length} times`);
});

// ── 3/4. No rehydration when meeting-active or starting set ────

test('no rehydration when body.meeting-active already set', async () => {
  const { deps, calls, doc } = defaultDeps();
  doc.body.classList.add('meeting-active');
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.equal(calls.loadMeetingJournal.length, 0);
});

test('no rehydration when body.starting is set', async () => {
  const { deps, calls, doc } = defaultDeps();
  doc.body.classList.add('starting');
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  // Rehydration is gated by meeting-active OR starting; the current
  // dispatch logic skips enter when starting is set.
  assert.equal(calls.loadMeetingJournal.length, 0);
});

// ── 5-7. Banner precedence ─────────────────────────────────────

test('banner return-state visible when recording + !meeting-active + !starting + !reconnecting', async () => {
  const { deps, calls } = defaultDeps();
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 1 });
  // After the rehydration transitions the body into meeting-active,
  // banner should NOT be visible (last call should be null). But an
  // earlier call to showBanner during the transition flow may have
  // emitted a non-null state. Assert ANY call emitted the return state
  // before meeting-active was set, then hidden after.
  const states = calls.showBanner.map((s) => (s ? s.mode : null));
  assert.ok(states.includes('return'), `expected some 'return' state in ${JSON.stringify(states)}`);
  assert.equal(states[states.length - 1], null);
});

test('banner hidden when body.meeting-active set and not reconnecting', async () => {
  const { deps, calls, doc } = defaultDeps();
  doc.body.classList.add('meeting-active');
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 1 });
  const last = calls.showBanner[calls.showBanner.length - 1];
  assert.equal(last, null);
});

test('reconnect banner wins precedence even with meeting-active set', async () => {
  const { deps, calls, doc } = defaultDeps();
  doc.body.classList.add('meeting-active');
  const r = createReconciler(deps);
  r.setReconnecting(true);
  // setReconnecting eagerly re-renders the banner.
  const last = calls.showBanner[calls.showBanner.length - 1];
  assert.ok(last && last.mode === 'reconnecting');
});

// ── 8. Title reflects state ────────────────────────────────────

test('document.title flips on recording and back to idle', async () => {
  const { deps, calls } = defaultDeps();
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.ok(calls.setTitle.some((t) => t === '● mtg-1 · Meeting Scribe'));
  await r.reconcile({ meeting: null, connections: 0 });
  assert.equal(calls.setTitle[calls.setTitle.length - 1], 'Meeting Scribe');
});

// ── 9/9b/9c. Matrix case A: pipeline open / connecting / starting flag ─

test('matrix A: pipeline open → no transport action', async () => {
  const { deps, calls } = defaultDeps({ getAudioWsState: () => 'open' });
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 1 });
  assert.equal(calls.startRecording.length, 0);
  assert.equal(calls.attachViewOnlyWs.length, 0);
});

test('matrix A: pipeline connecting (fresh-start race) → no transport action', async () => {
  const { deps, calls, storage } = defaultDeps({ getAudioWsState: () => 'connecting' });
  const r = createReconciler(deps);
  // Simulate: a fresh start just claimed ownership, so _startInitiatedByThisTab=true.
  r.claimOwnership('mtg-1');
  // Server status briefly shows connections=0 (WS mid-handshake).
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.equal(calls.startRecording.length, 0);
  assert.equal(calls.attachViewOnlyWs.length, 0);
  assert.ok(storage._raw.has('meetingScribe.recorderOwnership'));
});

test('matrix A: ws closed but _startInitiatedByThisTab set → still no transport action', async () => {
  const { deps, calls } = defaultDeps({ getAudioWsState: () => 'closed' });
  const r = createReconciler(deps);
  r.claimOwnership('mtg-1');
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.equal(calls.startRecording.length, 0);
  assert.equal(calls.attachViewOnlyWs.length, 0);
});

// ── 10-12. Matrix cases B / C / D ──────────────────────────────

test('matrix B: closed + owns + connections=0 → startRecording(true)', async () => {
  const { deps, calls, storage } = defaultDeps({ getAudioWsState: () => 'closed' });
  // Token is pre-existing (tab was the recorder before reload).
  storage.setItem('meetingScribe.recorderOwnership', JSON.stringify({
    meetingId: 'mtg-1', claimedAt: 0,
  }));
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.equal(calls.startRecording.length, 1);
  assert.equal(calls.startRecording[0], true);
  assert.equal(calls.attachViewOnlyWs.length, 0);
});

test('matrix C: closed + owns + connections>0 → view-only + pendingReclaim', async () => {
  const { deps, calls, storage } = defaultDeps({ getAudioWsState: () => 'closed' });
  storage.setItem('meetingScribe.recorderOwnership', JSON.stringify({
    meetingId: 'mtg-1', claimedAt: 0,
  }));
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 1 });
  assert.equal(calls.startRecording.length, 0);
  assert.equal(calls.attachViewOnlyWs.length, 1);
  assert.equal(r._inspect().pendingReclaim, true);
});

test('matrix D: closed + !owns → view-only only', async () => {
  const { deps, calls } = defaultDeps({ getAudioWsState: () => 'closed' });
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 1 });
  assert.equal(calls.startRecording.length, 0);
  assert.equal(calls.attachViewOnlyWs.length, 1);
  assert.equal(r._inspect().pendingReclaim, false);
});

// ── 13. Upgrade path: pendingReclaim → reclaim on next tick ───

test('upgrade: pendingReclaim + connections=0 triggers startRecording', async () => {
  const { deps, calls, storage } = defaultDeps({ getAudioWsState: () => 'closed' });
  storage.setItem('meetingScribe.recorderOwnership', JSON.stringify({
    meetingId: 'mtg-1', claimedAt: 0,
  }));
  const r = createReconciler(deps);
  // Enter via case C (connections=1).
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 1 });
  assert.equal(r._inspect().pendingReclaim, true);
  // Next tick: stale socket gone.
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.equal(calls.startRecording.length, 1);
  assert.equal(calls.startRecording[0], true);
  assert.equal(calls.detachViewOnlyWs.length, 1);
  assert.equal(r._inspect().pendingReclaim, false);
});

// ── 14. Upgrade retry: rejection keeps pendingReclaim true ────

test('upgrade retry: startRecording rejection leaves pendingReclaim=true', async () => {
  let shouldReject = true;
  const { deps, storage } = defaultDeps({
    getAudioWsState: () => 'closed',
    startRecording: async () => {
      if (shouldReject) throw new Error('mic busy');
    },
  });
  storage.setItem('meetingScribe.recorderOwnership', JSON.stringify({
    meetingId: 'mtg-1', claimedAt: 0,
  }));
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 1 });
  // First upgrade attempt fails.
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.equal(r._inspect().pendingReclaim, true);
  // Next tick succeeds.
  shouldReject = false;
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  assert.equal(r._inspect().pendingReclaim, false);
});

// ── 15. Stale token cleared when server shows no meeting ──────

test('server-no-meeting releases stale ownership token', async () => {
  const { deps, storage } = defaultDeps();
  storage.setItem('meetingScribe.recorderOwnership', JSON.stringify({
    meetingId: 'mtg-old', claimedAt: 0,
  }));
  const r = createReconciler(deps);
  await r.reconcile({ meeting: null, connections: 0 });
  assert.equal(storage.getItem('meetingScribe.recorderOwnership'), null);
});

// ── 16. setStoreLive gating during rebuild ───────────────────

test('setStoreLive(false) called before journal load; true only after rebuild', async () => {
  let order = [];
  const { deps } = defaultDeps({
    setStoreLive: (enabled) => { order.push(`live:${enabled}`); },
    loadMeetingJournal: async (mid) => {
      order.push('journal-load');
      return { meta: { meeting_id: mid, created_at: 0 }, events: [] };
    },
    renderTableStrip: () => { order.push('render-table'); },
  });
  const r = createReconciler(deps);
  await r.reconcile({ meeting: { id: 'mtg-1', state: 'recording' }, connections: 0 });
  // Expected: live:false → journal-load → render-table → live:true
  const firstLiveFalse = order.indexOf('live:false');
  const lastLiveTrue = order.lastIndexOf('live:true');
  const journalIdx = order.indexOf('journal-load');
  const renderIdx = order.indexOf('render-table');
  assert.ok(firstLiveFalse >= 0, 'setStoreLive(false) was not called');
  assert.ok(lastLiveTrue >= 0, 'setStoreLive(true) was not called');
  assert.ok(firstLiveFalse < journalIdx, 'live:false must precede journal-load');
  assert.ok(journalIdx < renderIdx, 'journal-load must precede render-table');
  assert.ok(renderIdx < lastLiveTrue, 'render-table must precede live:true');
});

// ── 17. checkStatus timeout-safety ───────────────────────────

test('checkStatus aborts on timeout and clears _statusInFlight', async () => {
  // Fake fetch that never resolves until aborted.
  let abortReceived = false;
  const { deps } = defaultDeps({
    fetchFn: (url, { signal }) => new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => {
        abortReceived = true;
        const err = new Error('aborted');
        err.name = 'AbortError';
        reject(err);
      });
    }),
  });
  const r = createReconciler(deps);
  // Shrink the timeout so the test runs quickly — monkey-patch the
  // module constant by constructing a fresh reconciler with a fake
  // fetch that triggers abort immediately via the AbortController.
  // But we exported the timeout as an internal — the simplest way is
  // to call checkStatus, then immediately inspect that it's queued.
  assert.equal(r._inspect().statusInFlight, false);
  const p = r.checkStatus();
  // After the microtask runs, _statusInFlight should be true.
  await Promise.resolve();
  assert.equal(r._inspect().statusInFlight, true);
  // Force a synthetic abort via the test-only pathway: we can't
  // manipulate time, but we CAN assert the default behaviour by
  // awaiting with a tighter timeout that hits the 5s boundary. To
  // avoid a 5s test runtime, use a custom fetch that immediately
  // throws AbortError when signal fires; we'll just directly call
  // signal.abort() via the inspection path — but the reconciler
  // doesn't expose the controller. So: drop this test to a smoke
  // check that _statusInFlight self-clears on any fetch rejection.
  const p2 = null;
  // Trigger resolution: simulate a network failure.
  // (The fake fetch above never resolves; simulate abort immediately.)
  // We manually invoke the internal abort via a second fake fetch:
  // This is a limitation of the current harness; a full test would
  // mock Date.now / setTimeout. Accept the 5s wait for now:
  await Promise.race([p, new Promise((res) => setTimeout(res, 5200))]);
  assert.ok(abortReceived, 'AbortController.abort() was not called');
  assert.equal(r._inspect().statusInFlight, false);
});

// ── 18. clearReconnectState removes stuck state ──────────────

test('clearReconnectState clears body class and re-renders banner', async () => {
  const { deps, calls } = defaultDeps();
  const r = createReconciler(deps);
  r.setReconnecting(true);
  const beforeClear = calls.showBanner[calls.showBanner.length - 1];
  assert.ok(beforeClear && beforeClear.mode === 'reconnecting');
  r.clearReconnectState();
  const afterClear = calls.showBanner[calls.showBanner.length - 1];
  // Reconnecting banner is cleared. Banner should be null (no
  // recording status was ever set in this test).
  assert.equal(afterClear, null);
});

// ── 19. returnToMeeting is pass-through ──────────────────────

test('returnToMeeting produces same transport trace as enterLiveMeetingMode', async () => {
  // Scenario A: direct enterLiveMeetingMode (observer, case D).
  const a = defaultDeps({ getAudioWsState: () => 'closed' });
  const rA = createReconciler(a.deps);
  await rA.enterLiveMeetingMode('mtg-1', { resetStore: true });

  // Scenario B: via returnToMeeting.
  const b = defaultDeps({ getAudioWsState: () => 'closed' });
  const rB = createReconciler(b.deps);
  await rB.returnToMeeting('mtg-1');

  // Both should have loaded the journal and attached view-only WS.
  assert.equal(a.calls.loadMeetingJournal.length, b.calls.loadMeetingJournal.length);
  assert.equal(a.calls.attachViewOnlyWs.length, b.calls.attachViewOnlyWs.length);
  assert.equal(a.calls.startRecording.length, b.calls.startRecording.length);
});
