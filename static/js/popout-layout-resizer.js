/* Shared resize-gutter drag helper.
 *
 * A single implementation that replaces the duplicated mousedown/move/up
 * patterns in terminal-panel._onResizeStart and scribe-app's slide-viewer
 * resize. Emits rAF-throttled ratio updates while dragging, commits the
 * final ratio on mouseup, clamps to [0.15, 0.85], supports Shift-snap
 * and Escape-cancel.
 */

(function () {
  'use strict';

  const { RATIO_MIN, RATIO_MAX, clampRatio } = window.PopoutLayout;

  function createResizer(gutterEl, splitEl, opts) {
    const { dir, onCommit, onUpdate } = opts;
    const startRatio = () => Number(splitEl.dataset.ratio || '0.5');

    function applyRatio(ratio) {
      const r = clampRatio(ratio);
      splitEl.dataset.ratio = String(r);
      // Children with --slot-flex controls the two panes' flex-grow.
      const slots = splitEl.querySelectorAll(':scope > .lyt-slot');
      if (slots.length === 2) {
        slots[0].style.setProperty('--slot-flex', String(r));
        slots[1].style.setProperty('--slot-flex', String(1 - r));
      }
      if (onUpdate) onUpdate(r);
    }

    function onMouseDown(ev) {
      if (ev.button !== 0) return;
      ev.preventDefault();
      const rect = splitEl.getBoundingClientRect();
      const initial = startRatio();
      let current = initial;
      let rafPending = false;
      document.body.style.cursor = dir === 'h' ? 'ew-resize' : 'ns-resize';
      document.body.style.userSelect = 'none';

      const onMove = (e) => {
        let raw;
        if (dir === 'h') {
          raw = (e.clientX - rect.left) / rect.width;
        } else {
          raw = (e.clientY - rect.top) / rect.height;
        }
        if (e.shiftKey) raw = Math.round(raw * 20) / 20; // 5% snap
        current = Math.max(RATIO_MIN, Math.min(RATIO_MAX, raw));
        if (!rafPending) {
          rafPending = true;
          requestAnimationFrame(() => {
            rafPending = false;
            applyRatio(current);
          });
        }
      };
      const onUp = () => {
        cleanup();
        if (onCommit) onCommit(current);
      };
      const onKey = (e) => {
        if (e.key === 'Escape') {
          applyRatio(initial);
          current = initial;
          cleanup();
        }
      };
      const cleanup = () => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        document.removeEventListener('keydown', onKey);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
      document.addEventListener('keydown', onKey);
    }

    gutterEl.addEventListener('mousedown', onMouseDown);

    return {
      destroy: () => gutterEl.removeEventListener('mousedown', onMouseDown),
      applyRatio,
    };
  }

  window.PopoutLayoutResizer = { createResizer };
})();
