// Meeting Scribe — Modal system (pure module).
//
// Stack-based modal infrastructure consumed by the admin SPA. Five
// public entry points:
//
//   showModal(html, cssClass='')          → returns the card element
//   closeModal()                          → pops the top card
//   closeAllModals()                      → bounded loop, empties stack
//   alertDialog(title, message, okText)   → Promise<void>
//   confirmDialog(title, message, confirmText, danger) → Promise<bool>
//   promptDialog(title, message, options) → Promise<string|null>
//
// Stack discipline: each showModal call hides the previous active
// card (if any), creates a sibling node, and registers the document
// Escape handler. closeModal pops one and reactivates the previous;
// the document Escape handler is removed only when the stack drains
// to empty. The .modal-card-active class is the single source of
// truth for "which card is on top".
//
// Cancel-resolution hook: dialog primitives set card._onClose so that
// every dismissal path (Escape, backdrop, explicit closeModal, parent
// closeAllModals) converges on a single cancel value. The hook is
// cleared by the confirm/cancel buttons before they call closeModal()
// so they don't double-resolve.
//
// Window-contract surface (per docs/scribe-window-contract.md):
//   window.alertDialog · window.confirmDialog · window.promptDialog
//   · window.closeAllModals
//
// Consumers: tests/browser/*, slide-viewer.js, terminal-panel.js,
// and the admin SPA's feature-specific dialogs (showSpeakerModal etc).

import { esc } from "../lib/escape.js";

export function showModal(html, cssClass = "") {
  const overlay = document.getElementById("modal-overlay");
  const rootCard = document.getElementById("modal-card");

  // Hide any currently-active card so the new one takes over. Hidden
  // cards stay in the DOM (and keep their listeners) until they pop.
  const prevActive = overlay.querySelector(".modal-card-active");
  if (prevActive) {
    prevActive.classList.remove("modal-card-active");
    prevActive.setAttribute("aria-hidden", "true");
    prevActive.style.display = "none";
  }

  // First modal: reuse the static root card. Stacked modal: create a
  // new sibling so the outer card's DOM + handlers survive untouched.
  let card;
  if (!prevActive) {
    card = rootCard;
  } else {
    card = document.createElement("div");
    overlay.appendChild(card);
  }
  card.className = `modal-card modal-card-active ${cssClass}`;
  card.innerHTML = html;
  card.style.display = "";
  card.removeAttribute("aria-hidden");
  card._onClose = null; // set by dialog primitives for cancel cleanup

  // Explicit ``flex`` — not ``''`` — because the CSS default for
  // ``.modal-overlay`` is ``display: none`` (defense against any path
  // that clears the inline ``style="display:none"`` and lets the
  // overlay paint a gray haze over the entire viewport on a fresh
  // page load with no modal showing). ``closeModal`` flips back to
  // ``'none'``.
  overlay.style.display = "flex";
  overlay.onclick = (e) => {
    if (e.target === overlay) closeModal();
  };
  document.addEventListener("keydown", _modalEscHandler);
  return card;
}

export function closeModal() {
  const overlay = document.getElementById("modal-overlay");
  const rootCard = document.getElementById("modal-card");
  const active = overlay.querySelector(".modal-card-active");
  if (!active) {
    // Nothing active — ensure overlay is down and bail.
    overlay.style.display = "none";
    document.removeEventListener("keydown", _modalEscHandler);
    overlay.onclick = null;
    return;
  }

  // Fire the cleanup hook BEFORE we tear DOM down so handlers can
  // inspect the card if they need to. Errors here must never block
  // the close — a hanging stack would trap the user.
  if (typeof active._onClose === "function") {
    try {
      active._onClose();
    } catch (e) {
      console.error("modal _onClose threw", e);
    }
  }
  active._onClose = null;
  active.classList.remove("modal-card-active");

  if (active === rootCard) {
    // Root card stays in the DOM (it's the static shell). Clear it.
    active.innerHTML = "";
    active.style.display = "none";
  } else {
    // Stacked sibling: remove entirely.
    active.remove();
  }

  // Pop: re-activate the most recently hidden card with real content,
  // or close the overlay if the stack is empty.
  const all = Array.from(overlay.querySelectorAll(".modal-card"));
  const hidden = all.filter(
    (c) => c.style.display === "none" && c.innerHTML.trim() !== "",
  );
  if (hidden.length > 0) {
    const top = hidden[hidden.length - 1];
    top.classList.add("modal-card-active");
    top.style.display = "";
    top.removeAttribute("aria-hidden");
  } else {
    overlay.style.display = "none";
    document.removeEventListener("keydown", _modalEscHandler);
    overlay.onclick = null;
  }
}

// Escape-key handler. Only closes the TOP modal (closeModal pops one).
// Repeated Escape presses walk the stack down to empty.
function _modalEscHandler(e) {
  if (e.key !== "Escape") return;
  // If focus is inside an input/select/textarea, let the element handle
  // Escape itself first. promptDialog's input attaches its own keydown
  // on the input and will cancel there; we stop the document handler
  // from firing a duplicate closeModal.
  const tag = (document.activeElement && document.activeElement.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
    return;
  }
  closeModal();
}

// Close every card in the stack. Used when an action mutates the
// underlying data in a way that makes outer cards stale (for example
// deleting a meeting while its tools modal is open).
export function closeAllModals() {
  const overlay = document.getElementById("modal-overlay");
  // Bounded loop: each iteration pops one card, overlay hides when empty.
  // The cap protects against any future bug that fails to advance state.
  for (let i = 0; i < 16; i++) {
    if (overlay.style.display === "none") break;
    if (!overlay.querySelector(".modal-card-active")) break;
    closeModal();
  }
}

// Styled Yes/No confirm. Resolves true on confirm, false on cancel,
// Escape, backdrop click, or any non-confirm close. Uses the card
// _onClose hook so every dismissal path converges on a single cancel
// value — the awaiting caller never hangs.
export async function confirmDialog(title, message, confirmText = "Delete", danger = true) {
  return new Promise((resolve) => {
    const card = showModal(
      `
      <div class="modal-confirm-title">${title}</div>
      <div class="modal-confirm-message">${message}</div>
      <div class="modal-confirm-actions">
        <button class="modal-btn" id="modal-cancel">Cancel</button>
        <button class="modal-btn ${danger ? "danger" : ""}" id="modal-confirm">${confirmText}</button>
      </div>
    `,
      "confirm",
    );
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    card._onClose = () => finish(false);
    card.querySelector("#modal-cancel").onclick = () => {
      closeModal();
    };
    card.querySelector("#modal-confirm").onclick = () => {
      card._onClose = null; // skip default cancel-resolution
      closeModal();
      finish(true);
    };
    card.querySelector("#modal-confirm").focus();
  });
}

// Styled single-button notice. Replaces window.alert so every diagnostic
// lives in the same visual language as the rest of the UI. Resolves
// when the user acknowledges (button, Enter, Escape, backdrop click —
// all treated as dismissals since there is nothing to cancel).
//
// Auto-promotes to a wider "pre" variant when the message looks like
// a stack trace / error body (multi-line, contains JSON/traceback
// markers, or exceeds a short-prose length). The pre variant wraps
// the message in a mono, scrollable, selectable region and adds a
// Copy button so the full string can be captured for bug reports.
export async function alertDialog(title, message, okText = "OK") {
  const msg = String(message ?? "");
  const isLong = msg.length > 160 || msg.includes("\n");
  const looksTraceback = /Traceback|^\s*at |\bError:|\bException:|HTTP \d{3}|<\?xml|^\s*\{/m.test(msg);
  const pre = isLong || looksTraceback;
  return new Promise((resolve) => {
    const body = pre
      ? `<div class="modal-confirm-message pre">${esc(msg)}</div>`
      : `<div class="modal-confirm-message">${esc(msg).replace(/\n/g, "<br>")}</div>`;
    const copyBtn = pre
      ? `<button type="button" class="modal-copy-btn" id="modal-copy" title="Copy message to clipboard">
           <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
           <span>Copy</span>
         </button>`
      : "";
    const cssClass = pre ? "confirm wide" : "confirm";
    const card = showModal(
      `
      <div class="modal-confirm-title">${esc(title)}</div>
      ${body}
      <div class="modal-confirm-actions">
        ${copyBtn}
        <button class="modal-btn primary" id="modal-ok">${esc(okText)}</button>
      </div>
    `,
      cssClass,
    );
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    // _onClose ensures Enter-via-focused-OK, Escape, backdrop click,
    // and explicit closeModal() all converge on resolve(). No
    // document-level keydown listener is needed — the standard
    // _modalEscHandler pops the stack and triggers this hook.
    card._onClose = finish;
    card.querySelector("#modal-ok").onclick = () => {
      card._onClose = null;
      closeModal();
      finish();
    };
    card.querySelector("#modal-ok").focus();
    const copy = card.querySelector("#modal-copy");
    if (copy) {
      copy.onclick = async (ev) => {
        ev.stopPropagation();
        const label = copy.querySelector("span");
        const fallback = () => {
          const ta = document.createElement("textarea");
          ta.value = msg;
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.select();
          try {
            document.execCommand("copy");
          } finally {
            document.body.removeChild(ta);
          }
        };
        try {
          if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(msg);
          } else {
            fallback();
          }
          copy.classList.add("copied");
          if (label) label.textContent = "Copied";
          setTimeout(() => {
            copy.classList.remove("copied");
            if (label) label.textContent = "Copy";
          }, 1400);
        } catch {
          if (label) label.textContent = "Copy failed";
        }
      };
    }
  });
}

// Styled text-input prompt. Resolves to the trimmed string on confirm,
// or null on cancel / Escape / backdrop click (matching the window.prompt
// contract so callers can do `if (raw === null) return` unchanged).
// Options: placeholder, initialValue, confirmText, type ('text'|'number'),
// min/max (for type=number), inputMode, help (extra hint text).
export async function promptDialog(title, message, options = {}) {
  const {
    placeholder = "",
    initialValue = "",
    confirmText = "OK",
    cancelText = "Cancel",
    type = "text",
    min,
    max,
    inputMode,
    help = "",
  } = options;
  return new Promise((resolve) => {
    const extraAttrs = [
      type ? `type="${esc(type)}"` : "",
      placeholder ? `placeholder="${esc(placeholder)}"` : "",
      min != null ? `min="${esc(String(min))}"` : "",
      max != null ? `max="${esc(String(max))}"` : "",
      inputMode ? `inputmode="${esc(inputMode)}"` : "",
    ]
      .filter(Boolean)
      .join(" ");
    const card = showModal(
      `
      <div class="modal-confirm-title">${esc(title)}</div>
      <div class="modal-confirm-message">${esc(message).replace(/\n/g, "<br>")}</div>
      <input class="modal-input" id="modal-input" ${extraAttrs} value="${esc(initialValue)}" autocomplete="off" spellcheck="false">
      ${help ? `<div class="modal-input-help">${esc(help)}</div>` : ""}
      <div class="modal-confirm-actions">
        <button class="modal-btn" id="modal-cancel">${esc(cancelText)}</button>
        <button class="modal-btn primary" id="modal-confirm">${esc(confirmText)}</button>
      </div>
    `,
      "confirm",
    );
    const input = card.querySelector("#modal-input");
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    // Default on any non-confirm close (Escape from outside the input,
    // backdrop, explicit closeModal): resolve null, matching
    // window.prompt.
    card._onClose = () => finish(null);
    const ok = () => {
      card._onClose = null;
      const v = input.value.trim();
      closeModal();
      finish(v);
    };
    const cancel = () => {
      closeModal();
      /* _onClose resolves null */
    };
    card.querySelector("#modal-confirm").onclick = ok;
    card.querySelector("#modal-cancel").onclick = cancel;
    // Input-scoped Enter/Escape. Escape inside the input takes priority
    // over the document-level _modalEscHandler (which ignores Escape
    // while focus is inside an input), so this cleanly cancels only
    // the prompt without walking farther up the stack.
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        ok();
      } else if (e.key === "Escape") {
        e.preventDefault();
        cancel();
      }
    });
    // Defer focus so the slide-up animation doesn't clobber the
    // selection.
    setTimeout(() => {
      input.focus();
      input.select();
    }, 50);
  });
}
