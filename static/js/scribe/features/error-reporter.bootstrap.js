// Meeting Scribe — Error reporter (bootstrap).
//
// Loaded by static/js/scribe/index.js. Stamps the global
// ``window.reportClientError`` contract and wires the three listeners
// that translate browser-emitted error events into
// /api/diag/client-error POSTs.
//
// Window-contract surface (per docs/scribe-window-contract.md):
//   * window.reportClientError(kind, message, stack, extra)
//
// Consumers: tests/browser/*, slide-viewer.js, terminal-panel.js
// (anywhere a non-fatal client error needs to surface in the
// server-side ring buffer).

import {
  looksLikeExtensionNoise,
  reportClientError,
} from "./error-reporter.js";

window.reportClientError = reportClientError;

window.addEventListener("error", (ev) => {
  reportClientError("uncaught", ev?.message || String(ev), ev?.error?.stack, {
    filename: ev?.filename || null,
    lineno: ev?.lineno || null,
    colno: ev?.colno || null,
  });
});

window.addEventListener("unhandledrejection", (ev) => {
  const reason = ev?.reason;
  reportClientError(
    "unhandled-rejection",
    reason && reason.message ? reason.message : String(reason),
    reason && reason.stack,
    null,
  );
});

window.addEventListener("securitypolicyviolation", (ev) => {
  if (looksLikeExtensionNoise(ev.blockedURI)) return; // 1Password etc.
  reportClientError(
    "csp-violation",
    `${ev.violatedDirective}: ${ev.blockedURI || "(inline)"}`,
    null,
    {
      blockedURI: ev.blockedURI || null,
      violatedDirective: ev.violatedDirective || null,
      effectiveDirective: ev.effectiveDirective || null,
      sourceFile: ev.sourceFile || null,
      lineNumber: ev.lineNumber || null,
      columnNumber: ev.columnNumber || null,
      sample: (ev.sample || "").slice(0, 500),
    },
  );
});
