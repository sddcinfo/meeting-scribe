// Meeting Scribe — Error reporter (pure module).
//
// Side-effect-free exports only. The bootstrap half (window publish +
// addEventListener registrations) lives in error-reporter.bootstrap.js.
//
// Posts uncaught errors + unhandled promise rejections + CSP violations
// + manual reportClientError() calls to /api/diag/client-error so
// server-side tooling can investigate UI bugs without asking the
// operator to open DevTools.

const API = "";

const CLIENT_ID = (() => {
  try {
    let id = sessionStorage.getItem("meetingScribe.clientId");
    if (!id) {
      id =
        (crypto?.randomUUID && crypto.randomUUID()) ||
        "cid-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
      sessionStorage.setItem("meetingScribe.clientId", id);
    }
    return id;
  } catch {
    return "cid-anon";
  }
})();

const PAGE_TAG = (() => {
  if (location.pathname.includes("reader.html")) return "reader";
  if (location.pathname.includes("guest.html")) return "guest";
  if (location.pathname.includes("portal.html")) return "portal";
  if (location.pathname.includes("how-it-works.html")) return "how-it-works";
  // The setup wizard has its own setup-wizard.js bundle. "admin" covers
  // index.html (with or without popout).
  if (new URLSearchParams(location.search).get("popout")) return "popout";
  return "admin";
})();

// CSP violations include the blocked URI (the font CDN URL, the inline
// style hash) — that's exactly the info the server-side ring buffer
// needs to triage the kind of error that doesn't surface as a normal
// JS exception.
//
// Extension-noise filter: 1Password and other browser extensions inject
// content scripts that pull fonts/icons/etc from third-party CDNs. CSP
// correctly blocks them, but the violation reports flood our server
// ring buffer with junk that has nothing to do with our code. The
// blocklist below is the small set of third-party hosts we've actually
// observed extensions hitting — keep it tight; we want to hear about
// anything novel because that *would* be a real leak.
const CSP_EXT_NOISE_HOSTS = new Set([
  "fonts.gstatic.com",
  "fonts.googleapis.com",
  "cache.agilebits.com",
]);

export function looksLikeExtensionNoise(blockedURI) {
  if (!blockedURI) return false;
  try {
    const url = new URL(blockedURI);
    return CSP_EXT_NOISE_HOSTS.has(url.host);
  } catch {
    return false;
  }
}

export function reportClientError(kind, message, stack, extra) {
  try {
    const body = {
      client_id: CLIENT_ID,
      page: PAGE_TAG,
      kind: String(kind || "manual"),
      message: String(message || ""),
      stack: typeof stack === "string" ? stack : (stack && stack.stack) || "",
      url: location.href,
      user_agent: navigator.userAgent,
      viewport: { w: innerWidth, h: innerHeight },
      context: extra || null,
    };
    // ``keepalive`` lets the report survive a navigation that fires
    // an error during unload; small body so it fits the 64 KB cap.
    fetch(`${API}/api/diag/client-error`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* swallow — reporter must never crash the app */
  }
}
