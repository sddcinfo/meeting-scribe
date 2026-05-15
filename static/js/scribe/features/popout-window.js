// Meeting Scribe — pop-out window opener.
//
// `openPopout(btnEl, meetingId)` spawns a fresh tab with
// `?popout=view#meeting/<id>&tabId=<uuid>` — the popout's bootstrap
// reads the hash to take ownership of its own session storage so it
// can't reattach to the inline admin tab's tmux session.
//
// State is module-private (the cached `_livePopout` reference). With
// `noopener` set on window.open the popup runs fully isolated and we
// can only poll its `.closed` state when window.open returns a usable
// handle.

let _livePopout = null;

export function openPopout(btnEl, meetingId) {
  if (_livePopout && !_livePopout.closed) {
    _livePopout.focus();
    return;
  }
  // Mint a fresh per-tab id for the popout's terminal (if it ever
  // attaches one) so it can't reattach to the inline admin tab's
  // tmux session and clip its viewport. Three layers of isolation:
  //   1. `noopener` severs window.opener AND gives the new context
  //      fresh sessionStorage, so the popout can't inherit our token.
  //   2. URL hash carries an explicit tabId — the popout's bootstrap
  //      writes this into its own sessionStorage BEFORE first attach,
  //      overriding any storage that did leak.
  //   3. terminal-panel.js's own _resolveTabSessionId() generates a
  //      fresh UUID on first load if neither layer above provided one.
  const tabId =
    (crypto.randomUUID && crypto.randomUUID().replace(/-/g, "")) ||
    ("t" + Math.random().toString(36).slice(2) + Date.now().toString(36)).slice(0, 40);
  const meetingHash = meetingId ? `meeting/${meetingId}` : "";
  const hashParts = [meetingHash, `tabId=${tabId}`].filter(Boolean);
  const hash = hashParts.length ? `#${hashParts.join("&")}` : "";
  _livePopout = window.open(
    `${location.origin}${location.pathname}?popout=view${hash}`,
    "_blank",
    `noopener,popup,width=${Math.round(
      window.screen.availWidth * 0.8,
    )},height=${Math.round(window.screen.availHeight * 0.6)},menubar=no,toolbar=no,location=no,status=no`,
  );
  if (btnEl) btnEl.classList.add("active-toggle");
  // With noopener, `_livePopout` is null — we can't poll its `.closed`
  // state from here. Drop the focus-on-second-click affordance for the
  // popout (the user can simply click the button to open another). The
  // active-toggle class is cleared on next page load or on a manual
  // toggle from the operator.
  if (!_livePopout) {
    return;
  }
  const checkClosed = setInterval(() => {
    if (_livePopout?.closed) {
      clearInterval(checkClosed);
      _livePopout = null;
      if (btnEl) btnEl.classList.remove("active-toggle");
    }
  }, 1000);
}
