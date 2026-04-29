// Regression tests for the modal stack (static/js/scribe-app.js).
//
// Run:  node --test tests/js/
//
// The scribe client can open one modal on top of another — most visibly
// when the Meeting Actions (tools) modal opens a confirmDialog /
// promptDialog / alertDialog. The rules enforced here (also documented
// in STYLING.md):
//
//   1. showModal on an already-open modal hides the outer card in
//      place and activates a new sibling. Outer DOM + listeners
//      survive untouched.
//   2. closeModal pops ONE card: the topmost. Outer cards reappear
//      unless the caller walks the whole stack via closeAllModals.
//   3. Every close path (explicit, Escape, backdrop) fires the
//      topmost card's _onClose hook once and only once. Dialog
//      primitives use this to resolve their promise on cancellation,
//      so awaiting callers never hang.
//   4. Dialog primitives clear _onClose before their own explicit
//      close (confirm / OK button click) so the hook does not
//      double-resolve their promise.
//
// The modal core lives inside scribe-app.js (not its own module), so
// we import only what we need by reading the source and evaluating
// the relevant slice against a tiny DOM shim. This keeps the test
// independent of the huge downstream app imports.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(
  resolve(here, '../../static/js/scribe-app.js'),
  'utf8',
);

// Extract the modal core block (showModal, closeModal, _modalEscHandler,
// closeAllModals). The slice is bounded by the header comment and the
// final `window.closeAllModals = closeAllModals;` line so the extract
// stays stable under unrelated edits above/below.
function sliceModalCore(src) {
  const start = src.indexOf('// ─── Modal System');
  const tail = 'window.closeAllModals = closeAllModals;';
  const end = src.indexOf(tail);
  assert.ok(start !== -1, 'modal-system header not found in scribe-app.js');
  assert.ok(end !== -1, 'closeAllModals export marker not found in scribe-app.js');
  return src.slice(start, end + tail.length);
}

// Build a minimal DOM shim that supports the operations the modal
// core touches: getElementById, querySelector/querySelectorAll with
// the two class selectors we use, classList, setAttribute /
// removeAttribute, event handling (add/remove listeners), innerHTML,
// style.display, appendChild, remove.
function makeDom() {
  let listeners = [];  // document-level keydown listeners

  const newEl = (tag = 'div', id = null) => {
    const el = {
      tagName: tag.toUpperCase(),
      id: id || '',
      className: '',
      _innerHTML: '',
      style: { display: 'none' },
      children: [],
      _attrs: {},
      _listeners: {},
      _onClose: null,
      classList: {
        add(c) {
          const set = new Set(el.className.split(' ').filter(Boolean));
          set.add(c);
          el.className = [...set].join(' ');
        },
        remove(c) {
          const set = new Set(el.className.split(' ').filter(Boolean));
          set.delete(c);
          el.className = [...set].join(' ');
        },
        contains(c) {
          return el.className.split(' ').includes(c);
        },
      },
      setAttribute(k, v) { el._attrs[k] = String(v); },
      removeAttribute(k) { delete el._attrs[k]; },
      getAttribute(k) { return el._attrs[k] ?? null; },
      get innerHTML() { return el._innerHTML; },
      set innerHTML(v) { el._innerHTML = String(v); },
      appendChild(child) { el.children.push(child); child.parentEl = el; return child; },
      remove() {
        if (el.parentEl) {
          el.parentEl.children = el.parentEl.children.filter(c => c !== el);
          el.parentEl = null;
        }
      },
      // Query: support '.modal-card-active' and '.modal-card'. Scans
      // descendants depth-first; no CSS parsing beyond the two selectors
      // we actually use.
      querySelector(sel) { return _querySelector(el, sel); },
      querySelectorAll(sel) { return _querySelectorAll(el, sel); },
      addEventListener() { /* card-level listeners not exercised here */ },
      removeEventListener() {},
      onclick: null,
    };
    return el;
  };

  function _matches(el, sel) {
    if (sel.startsWith('.')) {
      return el.className.split(' ').includes(sel.slice(1));
    }
    if (sel.startsWith('#')) return el.id === sel.slice(1);
    return false;
  }
  function _querySelectorAll(root, sel, out = []) {
    for (const c of root.children) {
      if (_matches(c, sel)) out.push(c);
      _querySelectorAll(c, sel, out);
    }
    return out;
  }
  function _querySelector(root, sel) {
    return _querySelectorAll(root, sel)[0] || null;
  }

  const overlay = newEl('div', 'modal-overlay');
  const rootCard = newEl('div', 'modal-card');
  overlay.appendChild(rootCard);

  const document = {
    getElementById(id) {
      if (id === 'modal-overlay') return overlay;
      if (id === 'modal-card') return rootCard;
      return null;
    },
    addEventListener(ev, fn) {
      if (ev !== 'keydown') return;
      if (!listeners.includes(fn)) listeners.push(fn);
    },
    removeEventListener(ev, fn) {
      if (ev !== 'keydown') return;
      listeners = listeners.filter(f => f !== fn);
    },
    createElement(tag) { return newEl(tag); },
    activeElement: { tagName: 'BODY' },
    _fireKeydown(key) {
      const event = { key, preventDefault() {} };
      for (const fn of [...listeners]) fn(event);
    },
    _listenerCount: () => listeners.length,
  };

  const windowObj = {};
  const consoleObj = {
    error(..._args) { /* swallow; tests assert state not output */ },
  };

  return { document, overlay, rootCard, windowObj, consoleObj };
}

function loadModalCore() {
  const core = sliceModalCore(SRC);
  const dom = makeDom();
  const { document, overlay, rootCard, windowObj, consoleObj } = dom;
  const closureFactory = new Function(
    'document', 'window', 'console',
    `${core}
     return { showModal, closeModal, closeAllModals, _modalEscHandler };`,
  );
  const mod = closureFactory(document, windowObj, consoleObj);
  return { ...mod, overlay, rootCard, document };
}

// ── Tests ──────────────────────────────────────────────────────────────────

test('showModal opens on the root card and activates it', () => {
  const { showModal, overlay, rootCard } = loadModalCore();
  const card = showModal('<p>first</p>', 'confirm');
  assert.equal(card, rootCard);
  assert.ok(card.classList.contains('modal-card-active'));
  assert.equal(overlay.style.display, '');
  assert.equal(card.innerHTML, '<p>first</p>');
});

test('nested showModal stacks on a new sibling; outer DOM survives', () => {
  const { showModal, overlay, rootCard } = loadModalCore();
  const outer = showModal('<p>outer</p>', 'confirm');
  const inner = showModal('<p>inner</p>', 'confirm');
  // Two cards in the overlay; root holds outer content; inner is sibling.
  assert.notEqual(outer, inner);
  assert.equal(outer, rootCard);
  assert.equal(overlay.children.length, 2);
  // Outer is hidden in place; inner is active.
  assert.equal(outer.style.display, 'none');
  assert.ok(!outer.classList.contains('modal-card-active'));
  assert.ok(inner.classList.contains('modal-card-active'));
  // Outer DOM content survived.
  assert.equal(outer.innerHTML, '<p>outer</p>');
});

test('closeModal pops only the topmost; outer reappears', () => {
  const { showModal, closeModal, overlay, rootCard } = loadModalCore();
  showModal('<p>outer</p>', 'confirm');
  showModal('<p>inner</p>', 'confirm');
  closeModal();
  // Inner sibling removed; outer reactivated.
  assert.equal(overlay.children.length, 1);
  assert.equal(overlay.children[0], rootCard);
  assert.ok(rootCard.classList.contains('modal-card-active'));
  assert.equal(rootCard.innerHTML, '<p>outer</p>');
  assert.equal(rootCard.style.display, '');
});

test('closeModal on the only modal hides the overlay and clears the card', () => {
  const { showModal, closeModal, overlay, rootCard } = loadModalCore();
  showModal('<p>only</p>', 'confirm');
  closeModal();
  assert.equal(overlay.style.display, 'none');
  assert.equal(rootCard.innerHTML, '');
  assert.ok(!rootCard.classList.contains('modal-card-active'));
});

test('_onClose fires exactly once per close (button path)', () => {
  const { showModal, closeModal } = loadModalCore();
  let closeCount = 0;
  const card = showModal('<p>body</p>', 'confirm');
  card._onClose = () => { closeCount += 1; };
  closeModal();
  assert.equal(closeCount, 1);
  // Second close on an already-closed overlay is a no-op; hook does not refire.
  closeModal();
  assert.equal(closeCount, 1);
});

test('Escape fires _onClose and pops the top card only', () => {
  const { showModal, document } = loadModalCore();
  const outerCloses = [];
  const innerCloses = [];
  const outer = showModal('<p>outer</p>', 'confirm');
  outer._onClose = () => outerCloses.push('outer');
  const inner = showModal('<p>inner</p>', 'confirm');
  inner._onClose = () => innerCloses.push('inner');
  // First Escape pops inner only.
  document._fireKeydown('Escape');
  assert.deepEqual(innerCloses, ['inner']);
  assert.deepEqual(outerCloses, []);
  // Second Escape pops outer.
  document._fireKeydown('Escape');
  assert.deepEqual(outerCloses, ['outer']);
});

test('Escape is ignored while focus is inside an INPUT element', () => {
  const { showModal, document } = loadModalCore();
  let closed = false;
  const card = showModal('<input id="modal-input">', 'confirm');
  card._onClose = () => { closed = true; };
  document.activeElement = { tagName: 'INPUT' };
  document._fireKeydown('Escape');
  // The document-level handler must not pop the stack while an input
  // has focus — promptDialog's input-scoped Escape handler owns that
  // interaction and would otherwise double-fire.
  assert.equal(closed, false);
  document.activeElement = { tagName: 'BODY' };
});

test('backdrop click closes the top modal via overlay.onclick', () => {
  const { showModal, overlay } = loadModalCore();
  let closed = false;
  const card = showModal('<p>body</p>', 'confirm');
  card._onClose = () => { closed = true; };
  // Simulate a backdrop click. The handler ignores clicks whose target
  // is a child card; it fires only when target === overlay.
  overlay.onclick({ target: overlay });
  assert.equal(closed, true);
});

test('closeAllModals walks the stack to empty', () => {
  const { showModal, closeAllModals, overlay, rootCard } = loadModalCore();
  const outer = showModal('<p>outer</p>', 'confirm');
  const middle = showModal('<p>middle</p>', 'confirm');
  const inner = showModal('<p>inner</p>', 'confirm');
  const closes = [];
  outer._onClose = () => closes.push('outer');
  middle._onClose = () => closes.push('middle');
  inner._onClose = () => closes.push('inner');
  closeAllModals();
  assert.deepEqual(closes, ['inner', 'middle', 'outer']);
  assert.equal(overlay.style.display, 'none');
  assert.equal(rootCard.innerHTML, '');
});

test('document keydown listener is removed when overlay closes', () => {
  const { showModal, closeModal, document } = loadModalCore();
  assert.equal(document._listenerCount(), 0);
  showModal('<p>a</p>', 'confirm');
  assert.equal(document._listenerCount(), 1);
  showModal('<p>b</p>', 'confirm');
  // Stacking reuses the same document listener — one is enough.
  assert.equal(document._listenerCount(), 1);
  closeModal();
  assert.equal(document._listenerCount(), 1); // outer still active
  closeModal();
  assert.equal(document._listenerCount(), 0); // overlay hidden
});
