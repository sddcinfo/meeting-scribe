// Meeting Scribe — Control-row-secondary button handlers (boot).
//
// The compact strip of toggles beneath the primary record-bar:
//   #btn-metrics       → metrics-dashboard slide-over
//   #btn-toggle-live   → pop-out translation view
//   #btn-col-a         → show-only language A
//   #btn-col-b         → show-only language B
//   #btn-toggle-table  → hide-table layout flag
//   #btn-toggle-terminal → lazy-load inline TerminalPanel
//
// Each handler is independent — no shared module-scope state — and
// the metrics-dashboard click reaches into ``setMetricsVisible`` from
// the metrics-dashboard module.
//
// Loaded once the admin SPA boot has run so window.TerminalPanel /
// window.refresh hooks are available by the time these listeners
// fire on user input.

import { setMetricsVisible } from "./metrics-dashboard.js";
import { openPopout } from "./popout-window.js";

const API = "";

// ─── Metrics dashboard ──────────────────────────────────────
document.getElementById("btn-metrics")?.addEventListener("click", (e) => {
  const btn = e.currentTarget;
  const isVisible = document.body.classList.contains("metrics-split");
  setMetricsVisible(!isVisible, btn);
});

// ─── "Live" pop-out translation view ────────────────────────
document.getElementById("btn-toggle-live")?.addEventListener("click", (e) => {
  openPopout(e.target);
});

// ─── Column selectors (active meeting) ──────────────────────
document.getElementById("btn-col-a")?.addEventListener("click", (e) => {
  const wasActive = document.body.classList.contains("show-only-a");
  document.body.classList.remove("show-only-a", "show-only-b");
  document.getElementById("btn-col-b")?.classList.remove("active-toggle");
  if (!wasActive) {
    document.body.classList.add("show-only-a");
    e.target.classList.add("active-toggle");
  } else {
    e.target.classList.remove("active-toggle");
  }
});

document.getElementById("btn-col-b")?.addEventListener("click", (e) => {
  const wasActive = document.body.classList.contains("show-only-b");
  document.body.classList.remove("show-only-a", "show-only-b");
  document.getElementById("btn-col-a")?.classList.remove("active-toggle");
  if (!wasActive) {
    document.body.classList.add("show-only-b");
    e.target.classList.add("active-toggle");
  } else {
    e.target.classList.remove("active-toggle");
  }
});

// ─── Virtual-table toggle ───────────────────────────────────
document.getElementById("btn-toggle-table")?.addEventListener("click", (e) => {
  // Flip hide-table, then set active-toggle to match VISIBLE state
  // (active = table is showing).
  const nowHidden = document.body.classList.toggle("hide-table");
  e.target.classList.toggle("active-toggle", !nowHidden);
});

// ─── Admin "translation + terminal" view ────────────────────
// Toggle the in-flow terminal panel above the transcript (mirrors
// the slide-bar's role). The terminal-panel module is lazy-loaded;
// we stash the instance on window so a second toggle reuses the
// same scrollback buffer.
document.getElementById("btn-toggle-terminal")?.addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  const host = document.getElementById("admin-terminal-host");
  if (!host) return;
  const wasActive = document.body.classList.contains("view-terminal-host-active");

  if (wasActive) {
    document.body.classList.remove("view-terminal-host-active");
    btn.classList.remove("active-toggle");
    if (
      window._adminInlineTerminal &&
      typeof window._adminInlineTerminal.hide === "function"
    ) {
      window._adminInlineTerminal.hide();
    }
    return;
  }

  // Lazy-load + mount on first activation.
  btn.disabled = true;
  try {
    if (!window.TerminalPanel) {
      await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "/static/js/terminal-panel.js?v=2";
        s.async = false;
        s.onload = resolve;
        s.onerror = () => reject(new Error("terminal-panel.js failed to load"));
        document.head.appendChild(s);
      });
    }
    if (!window._adminInlineTerminal) {
      const panel = new window.TerminalPanel({
        apiBase: API,
        wsBase: API.replace(/^http/, "ws"),
      });
      await panel.mount();
      window._adminInlineTerminal = panel;
    }
    // Re-parent the panel root into our host container — it may have
    // been auto-inserted near <main> by the panel's mount() fallback.
    const root = window._adminInlineTerminal._root;
    if (root && root.parentNode !== host) {
      host.appendChild(root);
    }
    document.body.classList.add("view-terminal-host-active");
    btn.classList.add("active-toggle");
    window._adminInlineTerminal.show();
  } catch (err) {
    console.warn("terminal toggle failed:", err);
    document.body.classList.remove("view-terminal-host-active");
    btn.classList.remove("active-toggle");
  } finally {
    btn.disabled = false;
  }
});
