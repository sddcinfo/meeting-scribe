// Meeting Scribe admin SPA entry point.
//
// Loaded by static/index.html as the single <script type="module">.
// Evaluation order:
//
//   1. ./state.js — exports the cross-module `state` object + installs the
//      `window.current_meeting_id` accessor shim. Must run before
//      admin-boot so the window contract is in place before app code
//      touches it.
//   2. Pure feature modules (side-effect-free exports only).
//   3. ./features/admin-boot.js — boot orchestrator that constructs the
//      legacy singletons (roomSetup, reconciler, meetingsMgr) and wires
//      the feature-module imports to the configure() deps bags.
//   4. Post-boot bootstrap initializers — wire `window.*` contracts and
//      `document.addEventListener` side effects in the order they need
//      to fire after admin-boot's body.
//
// Each bootstrap.js file either:
//   * Owns a `window.*` contract (per docs/scribe-window-contract.md) →
//     publishes via `window.X = X` at its top level.
//   * Reads/writes shared state via `import { state } from "../state.js"`.

import "./state.js";
// Pure feature modules — side-effect-free exports only, safe to
// evaluate before admin-boot's body runs.
import "./features/error-reporter.js";
import "./features/wifi-qr.js";
import "./features/mic-warmup.js";
import "./features/modal-system.js";
import "./features/audio-player.js";
import "./features/speaker-registry.js";
import "./features/metrics-dashboard.js";
import "./features/audio-out.js";
import "./features/qa-panel.js";
import "./features/finalize-summary.js";
import "./features/compact-grid.js";
import "./features/one-on-one.js";
import "./features/meeting-timer.js";
import "./features/audio-pipeline.js";
import "./features/speaker-review.js";
import "./features/bg-finalize-toast.js";
import "./features/popout-window.js";
import "./features/meeting-tools-modal.js";
import "./features/analytics-polling.js";
import "./features/meeting-banner.js";
import "./features/room-editor.js";
import "./features/meetings-panel.js";
import "./features/room-setup.js";
import "./features/live-session.js";
import "./features/view-only-ws.js";
import "./features/navigation.js";
import "./features/recording-lifecycle.js";
import "./features/meetings-manager.js";
import "./features/kiosk-splash.js";
import "./features/hdmi-settings-panel.js";
import "./features/admin-slide-bar.js";
import "./features/meeting-controls.js";
// Boot orchestrator: constructs the three SPA singletons (roomSetup,
// reconciler, meetingsMgr) and wires feature modules into their
// configure() deps bags.
import "./features/admin-boot.js";
// Post-boot bootstrap initializers — fire AFTER admin-boot's body so
// the window.* publish order is consistent.
import "./features/error-reporter.bootstrap.js";
import "./features/wifi-qr.bootstrap.js";
import "./features/modal-system.bootstrap.js";
import "./features/audio-player.bootstrap.js";
import "./features/audio-out.bootstrap.js";
import "./features/one-on-one.bootstrap.js";
import "./features/listen-toggle.bootstrap.js";
import "./features/audio-pop.bootstrap.js";
import "./features/sp325-wideband.bootstrap.js";
import "./features/room-editor.bootstrap.js";
import "./features/control-bar-toggles.bootstrap.js";
import "./features/stats-popover.bootstrap.js";
import "./features/meetings-panel.bootstrap.js";
import "./features/kiosk-splash.bootstrap.js";
import "./features/hdmi-settings-panel.bootstrap.js";
import "./features/admin-slide-bar.bootstrap.js";
// Settings panel (gear-icon slide-over) — wires the DOM listeners +
// initial /api/admin/settings fetch + every Save handler. No internal
// guard needed; the panel's DOM nodes only exist on the admin SPA.
import "./features/settings-panel.bootstrap.js";
// Pop-out SPA — gates internally on `?popout=view`, no-ops on admin.
import "./features/popout-spa.bootstrap.js";
// Setup-screen language A/B dropdowns. Self-gates on `?popout=view`.
import "./features/language-loader.bootstrap.js";
// Small admin-only bootstraps (scroll-direction toggle, Space=play,
// beforeunload warning, SegmentStore→1:1 dispatcher).
import "./features/admin-misc.bootstrap.js";
// /api/status tiered poll loop (admin-only, self-gates on `?popout=view`).
import "./features/status-poll-loop.bootstrap.js";
// Segment-store baseline subscribers (#segment-count chip + speaker
// tracker) + `?test=1` window-contract hook. admin-boot configures
// the strip-refresh dep when roomSetup is constructed; the actual
// `store.subscribe` calls fire post-boot from the bootstrap module.
import "./features/segment-store-subscribers.bootstrap.js";
// admin-navigation.bootstrap.js is imported from admin-boot directly
// (named imports) since the boot needs to be sequenced explicitly
// after meetingsMgr + roomSetup construction.
