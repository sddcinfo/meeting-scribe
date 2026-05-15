# Admin SPA window contract

Inventory of every `window.*` property and `window`-dispatched custom
event that crosses the boundary between `static/js/scribe/` and
external consumers (sibling classic scripts, tests, slide viewer,
etc.). External callers depend on these names — renames here are
breaking changes.

## Outbound — `scribe/` features publish; external code consumes

| Window name | Type | External consumers | Owning feature |
|---|---|---|---|
| `window.reportClientError` | function | `tests/browser/*` (some) | error-reporter |
| `window.closeAllModals` | function | (internal only) | modal-system |
| `window.alertDialog` | function | `static/js/slide-viewer.js`, `static/js/terminal-panel.js` | modal-system |
| `window.confirmDialog` | function | (internal) | modal-system |
| `window.promptDialog` | function | `static/js/terminal-panel.js` | modal-system |
| `window._gridRenderer` | object | `tests/browser/test_cross_window_sync.py`, `test_mobile_viewports.py`, `test_visual_regression.py` | compact-grid |
| `window.audioPlayer` | object | (internal cross-feature) | audio-player |
| `window.current_meeting_id` | string (read+write) | tests (some), cross-feature | shared — backed by `state.current_meeting_id` via `Object.defineProperty` shim in `state.js` |
| `window._onRoomLayoutUpdate` | function | (internal) | room-editor |
| `window._onSummaryRegenerated` | function | (internal) | finalize-summary |
| `window._lastTtsHealth` | string | (internal, scribe only) | status / finalize |
| `window.__test_store` | object | `tests/browser/test_cross_window_sync.py` | segment-store / test hooks |
| `window.__test_ingest_count` | counter | `test_cross_window_sync.py` | test hooks |
| `window.__test_msg_log` | array | `test_cross_window_sync.py` | test hooks |

Custom events dispatched on `window`:

| Event name | Owning feature |
|---|---|
| `scribe-ws-message` (test hook) | ws |
| `meeting-scribe:interpretation-status` | audio-out |
| `meeting-scribe:bt-status` | status / ws |

## Inbound — `scribe/` consumes; produced by classic sibling scripts

`window.PopoutLayout`, `window.PopoutLayoutPresets`,
`window.PopoutLayoutStorage`, `window.PopoutLayoutResizer`,
`window.PopoutPanelRegistry`, `window.PopoutLayoutRender`,
`window._adminAudioCard`, `window.closeDiagnosticsPanel`.

These nine scripts are loaded as classic `<script>` tags (no `type=
"module"`) so they execute before the deferred `scribe/index.js` module
and publish their APIs onto `window.*` at parse time.

Custom events listened for (consumed):
`meeting-scribe:close-diagnostics`, `meeting-scribe:close-settings`
(both emitted by `static/js/diagnostics-panel.js`).
