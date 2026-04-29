// Meeting state reconciliation + per-tab recorder ownership.
//
// This module owns every client-side state transition for live meetings:
//
//   - Periodic /api/status polling with AbortController timeout
//   - Deciding when to enter, re-enter, or exit live meeting mode
//   - Transport choice (recorder vs view-only) via a single shared
//     predicate so the ownership matrix is evaluated consistently
//   - Banner precedence (reconnect > return/view > hidden)
//   - Per-tab sessionStorage ownership token (which tab "owns" the mic)
//   - Store live-ingestion gating so non-meeting views can't be
//     corrupted by late-arriving WS events
//
// Everything is dependency-injected via ``createReconciler(deps)``.
// scribe-app.js wires real DOM / sessionStorage / fetch / AudioPipeline
// primitives; tests wire in-memory spies.
//
// Why a separate module: scribe-app.js is thousands of lines of
// side-effectful top-level code; loading it in a node --test run is
// not feasible. By keeping the state machine self-contained here we
// can exercise every branch in isolation.

const OWNERSHIP_KEY = "meetingScribe.recorderOwnership";
const STATUS_TIMEOUT_MS = 5000;

/**
 * Build a reconciler instance.
 *
 * @param {Object} deps — injected primitives
 * @param {Document} deps.doc
 * @param {Storage} deps.storage — session-scoped storage (sessionStorage in prod)
 * @param {typeof fetch} deps.fetchFn
 * @param {() => 'open' | 'connecting' | 'closed'} deps.getAudioWsState
 * @param {(resume: boolean) => Promise<void>} deps.startRecording
 * @param {(meetingId: string) => void} deps.attachViewOnlyWs
 * @param {() => void} deps.detachViewOnlyWs
 * @param {(meetingId: string) => Promise<{meta: Object, events: any[]}>} deps.loadMeetingJournal
 * @param {() => void} deps.resetStore
 * @param {(enabled: boolean) => void} deps.setStoreLive
 * @param {() => void} deps.renderTableStrip
 * @param {() => void} deps.showMeetingMode
 * @param {(state: BannerState | null) => void} deps.showBanner
 * @param {(text: string) => void} deps.setTitle
 * @param {() => void} deps.onFinalizeCleanup
 * @param {string} [deps.apiBase]
 * @param {boolean} [deps.popoutMode]
 */
export function createReconciler(deps) {
  const {
    doc,
    storage,
    fetchFn,
    getAudioWsState,
    startRecording,
    attachViewOnlyWs,
    detachViewOnlyWs,
    loadMeetingJournal,
    resetStore,
    setStoreLive,
    renderTableStrip,
    showMeetingMode,
    showBanner,
    setTitle,
    onFinalizeCleanup,
    apiBase = "",
    popoutMode = false,
  } = deps;

  // ── Module-local state ────────────────────────────────────────
  let _rehydrationInFlight = false;
  let _statusInFlight = false;
  let _startInitiatedByThisTab = false;
  let _pendingReclaim = false;
  let _reconnecting = false;
  let _lastStatus = null;

  // ── Ownership token helpers ───────────────────────────────────
  // All reads/writes go through the injected ``storage`` so tests can
  // exercise the same code path with an in-memory shim.

  function _ownsRecorder(meetingId) {
    try {
      const raw = storage.getItem(OWNERSHIP_KEY);
      const t = raw ? JSON.parse(raw) : null;
      return !!(t && t.meetingId === meetingId);
    } catch {
      return false;
    }
  }

  function claimOwnership(meetingId) {
    storage.setItem(
      OWNERSHIP_KEY,
      JSON.stringify({ meetingId, claimedAt: Date.now() }),
    );
    _startInitiatedByThisTab = true;
  }

  function releaseOwnership() {
    storage.removeItem(OWNERSHIP_KEY);
    _startInitiatedByThisTab = false;
  }

  // ── Single shared transport predicate ─────────────────────────
  // Every transport-decision point in the module funnels through this
  // so 'open', 'connecting', and the fresh-start transient flag are
  // always treated identically. Three narrow `state === 'open'` checks
  // anywhere else in this file would be a bug.

  function _tabOwnsOrAcquiringTransport() {
    const state = getAudioWsState();
    return (
      state === "open" ||
      state === "connecting" ||
      _startInitiatedByThisTab === true
    );
  }

  // ── Banner precedence ─────────────────────────────────────────

  function _renderBanner(status) {
    if (popoutMode) {
      showBanner(null);
      return;
    }
    // State 1: reconnect (highest precedence; shown even when
    // meeting-active so the user knows the live WS is down).
    if (_reconnecting) {
      showBanner({
        mode: "reconnecting",
        label: "Reconnecting to meeting…",
        button: null,
        onClick: null,
      });
      return;
    }
    // State 2: return / view (shown only when recorder is active on
    // the server AND this tab is not already on the meeting view).
    const mid = status?.meeting?.id;
    const isRecording = status?.meeting?.state === "recording";
    const inMeetingView = doc.body.classList.contains("meeting-active");
    const isStarting = doc.body.classList.contains("starting");
    if (isRecording && mid && !inMeetingView && !isStarting) {
      const owns = _ownsRecorder(mid);
      showBanner({
        mode: "return",
        label: "Meeting in progress",
        button: owns ? "Return" : "View",
        onClick: () => returnToMeeting(mid),
      });
      return;
    }
    showBanner(null);
  }

  // Public hooks for AudioPipeline to flag reconnect state.
  function setReconnecting(v) {
    _reconnecting = !!v;
    _renderBanner(_lastStatus);
  }

  // Must be invoked by every terminal teardown path (stop / cancel /
  // finalize / permanent ws close / pipeline dispose) so a stuck
  // reconnect banner can never survive past the meeting it described.
  function clearReconnectState() {
    _reconnecting = false;
    _renderBanner(_lastStatus);
  }

  // ── Core transition: enterLiveMeetingMode ─────────────────────
  // Canonical ordering per plan §3b. setStoreLive(false) stays in
  // effect through the ENTIRE rebuild; only step 9 re-enables.

  async function enterLiveMeetingMode(meetingId, opts = {}) {
    const { resetStore: shouldReset = true } = opts;
    if (_rehydrationInFlight) return;
    _rehydrationInFlight = true;
    try {
      // 2. suppress live ingestion across the rebuild
      setStoreLive(false);
      // 3. body class flags
      doc.body.classList.add("meeting-active");
      doc.body.classList.remove("starting");
      doc.body.classList.remove("off-meeting-view");
      // 4. DOM panel toggles
      showMeetingMode();
      // 5. optionally reset the store
      if (shouldReset) {
        resetStore();
      }
      // 6. hydrate from the journal (authoritative; discards any stray
      // events that arrived while the store was detached)
      try {
        await loadMeetingJournal(meetingId);
      } catch (e) {
        // Non-fatal: the live WS will start adding events again once
        // transport is up. Still proceed with transport choice so the
        // user at least sees the "Recording" state; next poll will
        // retry the journal load via future rehydrations.
        // eslint-disable-next-line no-console
        console.warn("journal load failed during enterLiveMeetingMode", e);
      }
      // 7. layout
      renderTableStrip();
      // 8. transport choice via the shared predicate
      const connections = _lastStatus?.connections ?? 0;
      if (_tabOwnsOrAcquiringTransport()) {
        // Case A: pipeline already live / mid-handshake / fresh-start
        // in flight on this tab. Nothing to do.
        _pendingReclaim = false;
      } else if (_ownsRecorder(meetingId) && connections === 0) {
        // Case B: recorder tab post-reload, stale socket already gone.
        try {
          await startRecording(true);
          _pendingReclaim = false;
        } catch {
          // mic-fail: degrade to view-only, do not retry forever.
          attachViewOnlyWs(meetingId);
          _pendingReclaim = false;
        }
      } else if (_ownsRecorder(meetingId) && connections > 0) {
        // Case C: recorder tab post-reload, stale socket still there.
        // Start view-only; the upgrade branch will reclaim once the
        // old socket disappears from /api/status.
        attachViewOnlyWs(meetingId);
        _pendingReclaim = true;
      } else {
        // Case D: observer tab, no ownership.
        attachViewOnlyWs(meetingId);
        _pendingReclaim = false;
      }
      // 9. re-enable live ingestion — SINGLE place.
      setStoreLive(true);
    } finally {
      _rehydrationInFlight = false;
    }
  }

  // Banner click / explicit return. Pass-through by design — the
  // transport decision is made inside ``enterLiveMeetingMode``.
  async function returnToMeeting(meetingId) {
    await enterLiveMeetingMode(meetingId, { resetStore: true });
  }

  // Observer tab takes over recorder role.
  async function takeoverRecorder(meetingId) {
    releaseOwnership();
    claimOwnership(meetingId);
    await enterLiveMeetingMode(meetingId, { resetStore: true });
  }

  // ── Reconcile: dispatch-only. No transport heuristic here. ────

  async function reconcile(status) {
    _lastStatus = status;
    const mid = status?.meeting?.id || null;
    const isRecording = status?.meeting?.state === "recording";
    const inMeetingView = doc.body.classList.contains("meeting-active");
    const isStarting = doc.body.classList.contains("starting");
    const bodyRecording = doc.body.classList.contains("recording");

    // Title update (always driven by server state).
    if (isRecording && mid) {
      setTitle(`● ${mid} · Meeting Scribe`);
    } else {
      setTitle("Meeting Scribe");
    }

    // Server says no meeting, client thinks it is recording → server-
    // side finalize or crash. Converge.
    if (!isRecording && bodyRecording) {
      onFinalizeCleanup();
      releaseOwnership();
      clearReconnectState();
      _renderBanner(status);
      return;
    }

    // Server says recording. Decide whether to enter / upgrade / noop.
    if (isRecording && mid) {
      // Cold entry: we are not already in meeting view, and not
      // mid-start. Let ``enterLiveMeetingMode`` pick transport.
      if (!inMeetingView && !isStarting) {
        _renderBanner(status);
        await enterLiveMeetingMode(mid, { resetStore: true });
        _renderBanner(_lastStatus);
        return;
      }
      // Upgrade path: we are in meeting view in view-only mode, we
      // own the token, and the stale recorder socket has dropped.
      // Reclaim the mic. Gated by the same shared predicate so the
      // fresh-start window or a connecting WS doesn't trip us into
      // opening a second recorder.
      if (
        inMeetingView &&
        _pendingReclaim &&
        _ownsRecorder(mid) &&
        (status.connections ?? 0) === 0 &&
        !_tabOwnsOrAcquiringTransport()
      ) {
        try {
          await startRecording(true);
          detachViewOnlyWs();
          _pendingReclaim = false;
        } catch {
          _pendingReclaim = true;
        }
      }
      _renderBanner(status);
      return;
    }

    // Server says no meeting and client is also idle. Nothing to do
    // except banner hide + stale-token cleanup.
    if (!isRecording) {
      // If we hold a token for a meeting the server no longer knows
      // about, release it.
      try {
        const raw = storage.getItem(OWNERSHIP_KEY);
        if (raw) releaseOwnership();
      } catch {
        /* ignore */
      }
    }
    _renderBanner(status);
  }

  // ── checkStatus: timeout-safe polling ─────────────────────────

  async function checkStatus() {
    if (_statusInFlight) return null;
    _statusInFlight = true;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), STATUS_TIMEOUT_MS);
    try {
      const r = await fetchFn(`${apiBase}/api/status`, { signal: ctl.signal });
      const data = await r.json();
      await reconcile(data);
      return data;
    } catch (e) {
      // Timeouts + network errors are recoverable — the next scheduled
      // poll retries naturally. Log at debug level if a logger is
      // available; otherwise stay silent so the console stays clean.
      if (typeof console !== "undefined" && console.debug) {
        console.debug(
          "checkStatus failed:",
          e?.name === "AbortError" ? "timeout" : e?.message || e,
        );
      }
      return null;
    } finally {
      clearTimeout(timer);
      _statusInFlight = false;
    }
  }

  return {
    reconcile,
    enterLiveMeetingMode,
    returnToMeeting,
    takeoverRecorder,
    claimOwnership,
    releaseOwnership,
    clearReconnectState,
    setReconnecting,
    checkStatus,
    // test-only introspection (not load-bearing for production):
    _inspect: () => ({
      rehydrationInFlight: _rehydrationInFlight,
      statusInFlight: _statusInFlight,
      startInitiatedByThisTab: _startInitiatedByThisTab,
      pendingReclaim: _pendingReclaim,
      reconnecting: _reconnecting,
      lastStatus: _lastStatus,
    }),
    _ownsRecorder,
  };
}
