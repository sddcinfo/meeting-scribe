// Meeting Scribe — RoomSetup canvas + seat editor.
//
// The data-driven room layout authoring surface (canvas + tables +
// seats + speaker enrollment + presets).
//
// The class takes a `deps` bag via the constructor for the two
// bindings it touches in the admin SPA boot orchestrator:
//
//   * startRecording — kicked from RoomSetup.startMeeting after the
//     setup transitions are wrapped up. Function reference is stable
//     at construction time.
//   * getReconciler() — RoomSetup constructs before the reconciler
//     does (createReconciler runs once all the helpers it needs are
//     in scope). The lazy getter form defers the binding until the
//     user actually clicks the start button, by which point
//     reconciler is set.
//
// Everything else (state, store, _enc, esc, modal-system,
// speaker-registry, speaker-palette, lang-helpers, time-format,
// compact-grid, mic-warmup) imports directly from the corresponding
// feature/lib modules.

import { _enc } from "../lib/meeting-url.js";
import { esc } from "../lib/escape.js";
import { formatTime } from "../lib/time-format.js";
import { getSpeakerColor, SPEAKER_COLORS } from "../lib/speaker-palette.js";
import { state, store } from "../state.js";
import { CompactGridRenderer } from "./compact-grid.js";
import { confirmDialog } from "./modal-system.js";
import { micWarmup } from "./mic-warmup.js";
import {
  _speakerRegistry,
  getAllSpeakers,
  getSpeakerDisplayName,
} from "./speaker-registry.js";

const API = "";

export class RoomSetup {
  constructor(deps) {
    this._deps = deps;
    this._persistTimer = null;
    this.tables = [];
    this.seats = [];
    this.preset = 'rectangle';
    this.selected = null; // { type: 'seat'|'table', id: string }
    this.mode = 'setup'; // 'setup' | 'live' | 'review'
    this.meetingId = null; // set when mounting for live/review
    this.defaultCanvas = document.getElementById('room-canvas');
    this.canvas = this.defaultCanvas;
    this.btnStart = document.getElementById('btn-start-meeting');
    this.hintEl = document.getElementById('setup-hint');

    document.getElementById('btn-add-seat').addEventListener('click', () => this.addSeatAtCenter());
    // Add Table now also places the requested number of seats around the
    // new table in a single click. Falls back to 0 extra seats if the input
    // is missing/invalid.
    document.getElementById('btn-add-table')?.addEventListener('click', () => {
      const input = document.getElementById('seat-count-input');
      let n = parseInt(input?.value, 10);
      if (!Number.isFinite(n) || n < 0) n = 0;
      if (n > 20) n = 20;
      this.addTableWithSeats(n);
    });
    document.getElementById('btn-clear-layout')?.addEventListener('click', () => this.clearLayout());
    this.btnStart.addEventListener('click', async () => {
      // Debounce: disable immediately so double-clicks can't fire two
      // concurrent startMeeting() calls. Button state is restored by
      // startMeeting's own try/finally regardless of success / failure
      // (see RoomSetup.startMeeting — this is the bulletproof path for
      // both sync and async errors from startRecording).
      if (this.btnStart.disabled) return;
      this.btnStart.disabled = true;
      this.btnStart.dataset.origText = this.btnStart.textContent;
      this.btnStart.textContent = 'Starting…';
      try {
        await this.startMeeting();
      } catch (e) {
        // startMeeting's finally already restored btnStart; just
        // surface the error for visibility (sentry-style) without
        // re-touching the DOM.
        console.error('startMeeting failed:', e);
      }
    });

    // Single-language meetings are only accessible via the "Quick start
    // English" button on the landing page — the setup screen is always
    // bilingual. If users want mono they go back to the landing page.

    // Clicking a preset button jumps the seat-count input to the layout's
    // typical size (read from data-default-seats on the button) and then
    // applies the preset. Users can still override by editing the number.
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const defaultN = parseInt(btn.dataset.defaultSeats, 10);
        if (Number.isFinite(defaultN) && defaultN > 0) {
          const input = document.getElementById('seat-count-input');
          if (input) input.value = String(defaultN);
        }
        this.applyPreset(btn.dataset.preset);
      });
    });

    // Live re-apply: changing the seat count re-runs the currently active
    // preset so users see the layout update as they adjust the number.
    const seatInput = document.getElementById('seat-count-input');
    seatInput?.addEventListener('input', () => {
      if (!this.preset) return;
      // Debounce so typing "12" doesn't re-render once for "1" then again
      // for "12"; users see the result after a short pause.
      clearTimeout(this._seatInputDebounce);
      this._seatInputDebounce = setTimeout(() => {
        if (this.preset) this.applyPreset(this.preset);
      }, 180);
    });

    // Click canvas to deselect — rebound in mount() if canvas changes
    this._bindCanvasClick();

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Don't handle if typing in an input
      if (e.target.tagName === 'INPUT') return;

      if ((e.key === 'Delete' || e.key === 'Backspace') && this.selected) {
        e.preventDefault();
        if (this.selected.type === 'seat') this.removeSeat(this.selected.id);
        else if (this.selected.type === 'table') this.removeTable(this.selected.id);
        this._deselect();
      }

      if (e.key === 'Escape') this._deselect();
    });

    this._loadLayout();
  }

  _bindCanvasClick() {
    // Remove previous listener if any (we reattach on mount)
    if (this._canvasClickHandler) {
      this.canvas.removeEventListener('click', this._canvasClickHandler);
    }
    this._canvasClickHandler = (e) => {
      if (e.target === this.canvas) this._deselect();
    };
    this.canvas.addEventListener('click', this._canvasClickHandler);
  }

  /**
   * Reparent the editor into a different container.
   * Mode determines behavior:
   *   'setup'  — default setup page, with toolbar + start button
   *   'live'   — mid-meeting overlay, live sync, persists to meeting
   *   'review' — past-meeting edit, cluster chips, persists to meeting
   */
  mount(containerEl, mode = 'setup', opts = {}) {
    this.mode = mode;
    this.meetingId = opts.meetingId || null;

    // Reparent: move every rendered DOM element into the new container
    if (containerEl && containerEl !== this.canvas) {
      containerEl.innerHTML = '';
      for (const t of this.tables) {
        if (t.element) containerEl.appendChild(t.element);
      }
      for (const s of this.seats) {
        if (s.element) containerEl.appendChild(s.element);
      }
      this.canvas = containerEl;
      this._bindCanvasClick();
    }

    // Refresh the empty-canvas affordance for the new container
    this._updateEmptyOverlay();
  }

  /**
   * Restore the editor back to the default setup canvas.
   */
  unmount() {
    this.mount(this.defaultCanvas, 'setup');
  }

  // ── Selection ──────────────────────────────────────

  _select(type, id) {
    this._deselect();
    this.selected = { type, id };
    const el = type === 'seat'
      ? this.seats.find(s => s.seatId === id)?.element
      : this.tables.find(t => t.tableId === id)?.element;
    el?.classList.add('selected');
  }

  _deselect() {
    this.canvas.querySelectorAll('.selected').forEach(el => el.classList.remove('selected'));
    this.selected = null;
  }

  // ── Presets ────────────────────────────────────────

  /**
   * Apply a room preset. The number of seats comes from the shared
   * seat-count input (the one alongside the Add Table button) so both
   * entry points — the canvas empty-state overlay and the footer preset
   * row — produce the same result. Pass an explicit `seatCount` to
   * override.
   */
  applyPreset(preset, seatCount) {
    this.preset = preset;
    document.querySelectorAll('.preset-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.preset === preset)
    );

    // Clear current scene
    this.tables.forEach(t => t.element?.remove());
    this.seats.forEach(s => s.element?.remove());
    this.tables = [];
    this.seats = [];

    // Read seat count from the shared input if not provided explicitly
    let n = seatCount;
    if (n == null) {
      const input = document.getElementById('seat-count-input');
      const parsed = parseInt(input?.value, 10);
      n = Number.isFinite(parsed) && parsed > 0 ? parsed : 6;
    }
    if (n < 1) n = 1;
    if (n > 20) n = 20;

    const P = this._presetData(preset, n);
    P.tables.forEach(t => { this.tables.push(t); this._renderTable(t); this._applyTable(t); });
    P.seats.forEach(s => { this.seats.push(s); this._renderSeat(s); this._applySeat(s); });
    this._updateHint();
    this._persistLayout();
  }

  _id() { return crypto.randomUUID(); }

  _presetData(preset, n = 6) {
    const T = (x, y, w, h, br, label) => ({
      tableId: this._id(), x, y, width: w, height: h, borderRadius: br, label: label || '', element: null
    });
    const S = (x, y) => ({
      seatId: this._id(), x, y, name: '', enrolled: false, enrollmentId: null, element: null
    });

    switch (preset) {
      case 'boardroom':
        return { tables: [T(50, 50, 50, 28, 50, '')], seats: this._ellipseSeats(50, 50, 35, 22, n) };
      case 'round':
        return { tables: [T(50, 50, 30, 30, 50, '')], seats: this._ellipseSeats(50, 50, 26, 26, n) };
      case 'square':
        return { tables: [T(50, 50, 28, 28, 3, '')], seats: this._rectSeats(50, 50, 14, 14, n) };
      case 'rectangle':
        return { tables: [T(50, 50, 44, 22, 3, '')], seats: this._rectSeats(50, 50, 22, 11, n) };
      case 'classroom': {
        // Rows of up to 4 desks facing a front table. Each row is centred
        // on its own actual seat count so a partial last row doesn't end
        // up flush-left, and a single seat lands at the horizontal centre
        // instead of x=20.
        const perRow = Math.min(4, Math.max(1, Math.ceil(n / 2)));
        const numRows = Math.ceil(n / perRow);
        const rowWidth = 60; // usable horizontal span (20..80)
        const seats = [];
        for (let i = 0; i < n; i++) {
          const row = Math.floor(i / perRow);
          const col = i % perRow;
          const isLastRow = row === numRows - 1;
          const rowSeats = isLastRow && n % perRow !== 0 ? n % perRow : perRow;
          const step = rowWidth / rowSeats;
          const x = 20 + step * (col + 0.5);
          const y = 35 + row * 18;
          seats.push(S(x, Math.min(90, y)));
        }
        return { tables: [T(50, 15, 60, 5, 2, 'FRONT')], seats };
      }
      case 'u_shape': {
        // Three linear tables forming a U that opens upwards. Participants
        // sit INSIDE the U facing the centre — seats previously landed on
        // the outside (x=12, x=88, y=85), which placed them off-canvas
        // relative to the three tables. The inside-facing coordinates are
        // x=30 (right of left table), y=68 (above bottom table), and x=70
        // (left of right table).
        let nLeft = Math.round(n * 0.4);
        let nRight = Math.round(n * 0.4);
        let nBottom = n - nLeft - nRight;
        if (nBottom < 0) { nBottom = 0; nLeft = Math.floor(n / 2); nRight = n - nLeft; }
        const seats = [];
        // Left wall, walking top→bottom so seat order feels natural
        for (let i = 0; i < nLeft; i++) {
          const y = 30 + (i + 0.5) * (40 / Math.max(nLeft, 1));
          seats.push(S(30, y));
        }
        // Bottom of the U (inside, above the horizontal table)
        for (let i = 0; i < nBottom; i++) {
          const x = 30 + (i + 0.5) * (40 / Math.max(nBottom, 1));
          seats.push(S(x, 68));
        }
        // Right wall, walking bottom→top so the outer ring wraps cleanly
        for (let i = 0; i < nRight; i++) {
          const y = 70 - (i + 0.5) * (40 / Math.max(nRight, 1));
          seats.push(S(70, y));
        }
        return {
          tables: [T(22, 50, 8, 45, 2, ''), T(50, 76, 48, 6, 2, ''), T(78, 50, 8, 45, 2, '')],
          seats,
        };
      }
      case 'pods': {
        // Four pods in a 2x2 grid. The previous version stepped through
        // (top, right, bottom) only (indexInPod * π/2 - π/2), so each pod
        // topped out at 3 seats in an L-shape. Now each pod owns its
        // share of seats (round-robined so n=5 becomes 2/1/1/1) and the
        // seats are distributed evenly around the pod's full circle.
        const podCenters = [[30, 35], [70, 35], [30, 70], [70, 70]];
        const seats = [];
        for (let pod = 0; pod < 4; pod++) {
          const seatsInPod = Math.floor(n / 4) + (pod < n % 4 ? 1 : 0);
          if (seatsInPod === 0) continue;
          const [cx, cy] = podCenters[pod];
          for (let j = 0; j < seatsInPod; j++) {
            const angle = (2 * Math.PI * j) / seatsInPod - Math.PI / 2;
            seats.push(S(
              Math.max(5, Math.min(95, cx + 14 * Math.cos(angle))),
              Math.max(5, Math.min(95, cy + 12 * Math.sin(angle))),
            ));
          }
        }
        return {
          tables: [T(30, 35, 20, 16, 3, 'A'), T(70, 35, 20, 16, 3, 'B'), T(30, 70, 20, 16, 3, 'C'), T(70, 70, 20, 16, 3, 'D')],
          seats,
        };
      }
      case 'freeform':
      default: {
        // Free-form: sprinkle N seats in a loose grid, no table. Using
        // (i + 0.5) slot-centring so a single seat lands at (50, 50)
        // instead of the top-left corner.
        const cols = Math.ceil(Math.sqrt(n));
        const rows = Math.ceil(n / cols);
        const seats = [];
        for (let i = 0; i < n; i++) {
          const col = i % cols;
          const row = Math.floor(i / cols);
          const x = 20 + ((col + 0.5) / cols) * 60;
          const y = 25 + ((row + 0.5) / rows) * 50;
          seats.push(S(x, y));
        }
        return { tables: [], seats };
      }
    }
  }

  _ellipseSeats(cx, cy, rx, ry, n) {
    const off = n === 2 ? Math.PI / 2 : -Math.PI / 2;
    return Array.from({ length: n }, (_, i) => {
      const a = off + (2 * Math.PI * i) / n;
      return { seatId: this._id(), x: cx + rx * Math.cos(a), y: cy + ry * Math.sin(a), name: '', enrolled: false, enrollmentId: null, element: null };
    });
  }

  _rectSeats(cx, cy, hw, hh, n) {
    if (n <= 1) return [{ seatId: this._id(), x: cx, y: cy - hh - 6, name: '', enrolled: false, enrollmentId: null, element: null }];
    // Walk the perimeter with the slot-centre offset `(i + 0.5) / n`, so
    // n=4 on a square yields one seat at the midpoint of each edge
    // (top, right, bottom, left) instead of four corner-biased points.
    const perim = 2 * (2 * hw + 2 * hh);
    return Array.from({ length: n }, (_, i) => {
      let d = ((i + 0.5) / n) * perim, x, y;
      if (d < 2 * hw) {
        // Top edge
        x = cx - hw + d;
        y = cy - hh - 6;
      } else if (d < 2 * hw + 2 * hh) {
        // Right edge
        d -= 2 * hw;
        x = cx + hw + 6;
        y = cy - hh + d;
      } else if (d < 4 * hw + 2 * hh) {
        // Bottom edge
        d -= 2 * hw + 2 * hh;
        x = cx + hw - d;
        y = cy + hh + 6;
      } else {
        // Left edge
        d -= 4 * hw + 2 * hh;
        x = cx - hw - 6;
        y = cy + hh - d;
      }
      return { seatId: this._id(), x, y, name: '', enrolled: false, enrollmentId: null, element: null };
    });
  }

  // ── Table management ───────────────────────────────

  addTable() {
    const t = { tableId: this._id(), x: 50, y: 50, width: 22, height: 14, borderRadius: 8, label: '', element: null };
    this.tables.push(t);
    this._renderTable(t);
    this._applyTable(t);
    this._persistLayout();
    this._updateHint();
    return t;
  }

  /**
   * Wipe all tables and seats. Shows the empty-state overlay again so
   * users can pick a fresh layout. Also clears the active preset so the
   * seat-count input stops live-re-rendering against a stale preset.
   */
  clearLayout() {
    this.tables.forEach(t => t.element?.remove());
    this.seats.forEach(s => s.element?.remove());
    this.tables = [];
    this.seats = [];
    this.preset = '';
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    this._persistLayout();
    this._updateHint();
  }

  /**
   * Add a new table and arrange `seatCount` seats evenly around it in a
   * single action, so users can say "I want a round table with 6 seats"
   * without clicking Add Seat repeatedly.
   *
   * Reuses _nextSeatPosition() so the placement algorithm matches the
   * existing auto-layout used when reconciling detected speakers.
   */
  addTableWithSeats(seatCount) {
    const table = this.addTable();
    if (!table || !seatCount || seatCount <= 0) return;
    // _nextSeatPosition computes positions against this.tables[0]. The new
    // table was just pushed so it's only this.tables[0] when it's the first
    // table; otherwise place seats around it explicitly using the same
    // ellipse layout as _nextSeatPosition.
    const isFirst = this.tables[0] === table;
    for (let i = 0; i < seatCount; i++) {
      let x, y;
      if (isFirst) {
        ({ x, y } = this._nextSeatPosition(i, seatCount));
      } else {
        const cx = table.x + table.width / 2;
        const cy = table.y + table.height / 2;
        const rx = table.width / 2 + 8;
        const ry = table.height / 2 + 10;
        const angle = (2 * Math.PI * i) / seatCount - Math.PI / 2;
        x = Math.max(5, Math.min(95, cx + rx * Math.cos(angle)));
        y = Math.max(5, Math.min(95, cy + ry * Math.sin(angle)));
      }
      const seat = {
        seatId: this._id(),
        x, y,
        name: '',
        enrolled: false,
        enrollmentId: null,
        element: null,
      };
      this.seats.push(seat);
      this._renderSeat(seat);
      this._applySeat(seat);
    }
    this._persistLayout();
    this._updateHint();
  }

  /**
   * Ensure the canvas has one seat per detected speaker and that the seat
   * names match the registry. Called when the room editor is opened in
   * "live" mode and whenever a new speaker is detected while the editor
   * is open. Seats are added around the first table (or in a grid if
   * no table exists). Existing seats are kept and bound to speakers in
   * first-seen order.
   */
  reconcileSeatsToDetectedSpeakers() {
    if (typeof getAllSpeakers !== 'function') return;
    const speakers = getAllSpeakers();
    if (speakers.length === 0) return;

    // 1. If we have fewer seats than speakers, ADD seats around the first
    //    table (or in a grid if there's no table).
    while (this.seats.length < speakers.length) {
      const idx = this.seats.length;
      const { x, y } = this._nextSeatPosition(idx, speakers.length);
      const seat = {
        seatId: this._id(),
        x, y,
        name: '',
        enrolled: false,
        enrollmentId: null,
        element: null,
      };
      this.seats.push(seat);
      this._renderSeat(seat);
      this._applySeat(seat);
    }

    // 2. Bind each seat to the speaker at the same index (first-seen order)
    //    so the seat label reflects the live speaker name. We keep existing
    //    names if the seat was already bound to that cluster.
    speakers.forEach((sp, i) => {
      const seat = this.seats[i];
      if (!seat) return;
      if (seat.name !== sp.displayName) {
        seat.name = sp.displayName;
        const nameEl = seat.element?.querySelector('.seat-name');
        if (nameEl) nameEl.textContent = sp.displayName;
      }
      // Stamp the cluster_id on the DOM so downstream code can map seat→speaker
      if (seat.element) {
        seat.element.dataset.clusterId = String(sp.clusterId);
      }
    });

    this._persistLayout();
    this._updateHint();
  }

  /** Compute the next seat position when auto-adding seats around a table. */
  _nextSeatPosition(index, totalWanted) {
    // Prefer arranging around the first table
    const table = this.tables[0];
    if (table) {
      const cx = table.x + table.width / 2;
      const cy = table.y + table.height / 2;
      const rx = table.width / 2 + 8;
      const ry = table.height / 2 + 10;
      // Spread evenly around an ellipse
      const n = Math.max(totalWanted, 1);
      const angle = (2 * Math.PI * index) / n - Math.PI / 2;
      const x = Math.max(5, Math.min(95, cx + rx * Math.cos(angle)));
      const y = Math.max(5, Math.min(95, cy + ry * Math.sin(angle)));
      return { x, y };
    }
    // No table — grid layout
    const cols = Math.ceil(Math.sqrt(totalWanted));
    const rows = Math.ceil(totalWanted / cols);
    const col = index % cols;
    const row = Math.floor(index / cols);
    const x = 15 + (70 / Math.max(cols - 1, 1)) * col;
    const y = 15 + (70 / Math.max(rows - 1, 1)) * row;
    return { x, y };
  }

  removeTable(tableId) {
    const idx = this.tables.findIndex(t => t.tableId === tableId);
    if (idx === -1) return;
    this.tables[idx].element?.remove();
    this.tables.splice(idx, 1);
    this._persistLayout();
    this._updateHint();
  }

  _renderTable(table) {
    const el = document.createElement('div');
    el.className = 'table-obj';
    el.innerHTML = `
      <button class="table-remove" title="Remove table">&times;</button>
      <div class="table-surface"><div class="table-grain"></div><div class="table-label">${table.label}</div></div>
      <div class="table-resize"></div>
    `;
    this.canvas.appendChild(el);
    table.element = el;

    el.querySelector('.table-remove').addEventListener('click', async e => {
      e.stopPropagation();
      const ok = await confirmDialog('Remove Table?', 'Remove this table from the room?', 'Remove');
      if (!ok) return;
      this.removeTable(table.tableId);
    });
    this._makeTableDraggable(table);
    this._makeTableResizable(table);
  }

  _applyTable(t) {
    if (!t.element) return;
    t.element.style.left = `${t.x}%`;
    t.element.style.top = `${t.y}%`;
    t.element.style.width = `${t.width}%`;
    t.element.style.height = `${t.height}%`;
    t.element.querySelector('.table-surface').style.borderRadius = `${t.borderRadius}%`;
  }

  _makeTableDraggable(table) {
    const surf = table.element.querySelector('.table-surface');
    let active = false, sx, sy, spx, spy;
    surf.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      this._select('table', table.tableId);
      active = true; surf.setPointerCapture(e.pointerId); table.element.classList.add('dragging');
      sx = e.clientX; sy = e.clientY; spx = table.x; spy = table.y;
    });
    surf.addEventListener('pointermove', e => {
      if (!active) return;
      const r = this.canvas.getBoundingClientRect();
      table.x = Math.max(5, Math.min(95, spx + (e.clientX - sx) / r.width * 100));
      table.y = Math.max(5, Math.min(95, spy + (e.clientY - sy) / r.height * 100));
      this._applyTable(table);
    });
    surf.addEventListener('pointerup', () => { if (!active) return; active = false; table.element.classList.remove('dragging'); this._persistLayout(); });
  }

  _makeTableResizable(table) {
    const h = table.element.querySelector('.table-resize');
    let active = false, sx, sy, sw, sh;
    h.addEventListener('pointerdown', e => {
      e.stopPropagation(); active = true; h.setPointerCapture(e.pointerId);
      sx = e.clientX; sy = e.clientY; sw = table.width; sh = table.height;
    });
    h.addEventListener('pointermove', e => {
      if (!active) return;
      const r = this.canvas.getBoundingClientRect();
      table.width = Math.max(6, Math.min(90, sw + (e.clientX - sx) / r.width * 200));
      table.height = Math.max(4, Math.min(90, sh + (e.clientY - sy) / r.height * 200));
      this._applyTable(table);
    });
    h.addEventListener('pointerup', () => { if (!active) return; active = false; this._persistLayout(); });
  }

  // ── Seat management ────────────────────────────────

  addSeatAtCenter() {
    // Generate candidate positions around ALL tables
    const candidates = [];
    for (const t of this.tables) {
      const isRound = t.borderRadius >= 30;
      if (isRound) {
        // Elliptical: generate 12 positions around the ellipse
        const rx = t.width / 2 + 7, ry = t.height / 2 + 7;
        for (let i = 0; i < 12; i++) {
          const a = (2 * Math.PI * i) / 12 - Math.PI / 2;
          candidates.push({ x: t.x + rx * Math.cos(a), y: t.y + ry * Math.sin(a) });
        }
      } else {
        // Rectangular: generate positions along edges with offset
        const hw = t.width / 2 + 7, hh = t.height / 2 + 7;
        // Long sides (top & bottom) get more candidates
        const longSide = t.width >= t.height;
        const nLong = 6, nShort = 3;
        // Top edge
        for (let i = 0; i < (longSide ? nLong : nShort); i++) {
          const frac = (i + 0.5) / (longSide ? nLong : nShort);
          candidates.push({ x: t.x - hw + frac * 2 * hw, y: t.y - hh });
        }
        // Bottom edge
        for (let i = 0; i < (longSide ? nLong : nShort); i++) {
          const frac = (i + 0.5) / (longSide ? nLong : nShort);
          candidates.push({ x: t.x - hw + frac * 2 * hw, y: t.y + hh });
        }
        // Left edge
        for (let i = 0; i < (longSide ? nShort : nLong); i++) {
          const frac = (i + 0.5) / (longSide ? nShort : nLong);
          candidates.push({ x: t.x - hw, y: t.y - hh + frac * 2 * hh });
        }
        // Right edge
        for (let i = 0; i < (longSide ? nShort : nLong); i++) {
          const frac = (i + 0.5) / (longSide ? nShort : nLong);
          candidates.push({ x: t.x + hw, y: t.y - hh + frac * 2 * hh });
        }
      }
    }

    // Fallback: if no tables, use grid
    if (candidates.length === 0) {
      for (let cx = 15; cx <= 85; cx += 10)
        for (let cy = 15; cy <= 85; cy += 10)
          candidates.push({ x: cx, y: cy });
    }

    // Clamp to canvas bounds
    const clamped = candidates.map(c => ({
      x: Math.max(5, Math.min(95, c.x)),
      y: Math.max(5, Math.min(95, c.y)),
    }));

    // Pick the candidate farthest from all existing seats
    let x = 50, y = 50;
    if (this.seats.length > 0) {
      let best = -1;
      for (const c of clamped) {
        const d = Math.min(...this.seats.map(s => Math.hypot(s.x - c.x, s.y - c.y)));
        if (d > best) { best = d; x = c.x; y = c.y; }
      }
    } else {
      x = clamped[0].x; y = clamped[0].y;
    }

    const s = { seatId: this._id(), x, y, name: '', enrolled: false, enrollmentId: null, element: null };
    this.seats.push(s);
    this._renderSeat(s);
    this._applySeat(s);
    this._updateHint();
    this._persistLayout();
  }

  removeSeat(seatId) {
    const idx = this.seats.findIndex(s => s.seatId === seatId);
    if (idx === -1) return;
    this.seats[idx].element?.remove();
    this.seats.splice(idx, 1);
    this._updateHint();
    this._persistLayout();
  }

  _renderSeat(seat) {
    const ci = this.seats.indexOf(seat);
    // design-time: no cluster_id yet, color by seat index
    const color = SPEAKER_COLORS[ci % SPEAKER_COLORS.length];
    const circ = 2 * Math.PI * 34;
    const n = document.createElement('div');
    n.className = 'seat-node';
    n.dataset.seatId = seat.seatId;
    n.style.setProperty('--seat-color', color);
    n.innerHTML = `
      <button class="seat-remove" title="Remove">&times;</button>
      <div class="seat-avatar" title="Click chair to enroll voice">
        <span class="seat-index">${ci + 1}</span>
        <svg class="seat-progress" viewBox="0 0 72 72">
          <circle class="progress-bg" cx="36" cy="36" r="34"/>
          <circle class="progress-fill" cx="36" cy="36" r="34" stroke-dasharray="${circ}" stroke-dashoffset="${circ}"/>
        </svg>
      </div>
      <span class="seat-name-label" title="Click name to edit">${seat.name || 'Click chair to enroll'}</span>
      <span class="seat-status-label">${seat.enrolled ? 'Enrolled' : ''}</span>
    `;
    this.canvas.appendChild(n);
    seat.element = n;
    if (seat.enrolled) n.classList.add('enrolled');
    // Name label is always clickable (both pre- and post-enrollment) — gives
    // a fast "just type a name" path without interrupting voice capture.
    const nameLabel = n.querySelector('.seat-name-label');
    nameLabel.style.cursor = 'pointer';
    nameLabel.addEventListener('click', (e) => {
      e.stopPropagation();
      this._editName(seat);
    });
    n.querySelector('.seat-remove').addEventListener('click', async e => {
      e.stopPropagation();
      if (seat.enrolled) {
        const ok = await confirmDialog('Remove Seat?', `Remove ${seat.name || 'this seat'} from the room?`, 'Remove');
        if (!ok) return;
      }
      this.removeSeat(seat.seatId);
    });
    this._makeSeatDraggable(seat);
  }

  _applySeat(s) { if (s.element) { s.element.style.left = `${s.x}%`; s.element.style.top = `${s.y}%`; } }

  _makeSeatDraggable(seat) {
    const av = seat.element.querySelector('.seat-avatar');
    let on = false, sx, sy, spx, spy, moved;
    av.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      this._select('seat', seat.seatId);
      on = true; moved = 0; av.setPointerCapture(e.pointerId);
      sx = e.clientX; sy = e.clientY; spx = seat.x; spy = seat.y;
      seat.element.style.transition = 'none'; seat.element.style.zIndex = '10';
    });
    av.addEventListener('pointermove', e => {
      if (!on) return;
      const r = this.canvas.getBoundingClientRect();
      const dx = e.clientX - sx, dy = e.clientY - sy;
      moved += Math.abs(dx) + Math.abs(dy);
      seat.x = Math.max(3, Math.min(97, spx + dx / r.width * 100));
      seat.y = Math.max(3, Math.min(97, spy + dy / r.height * 100));
      seat.element.style.left = `${seat.x}%`; seat.element.style.top = `${seat.y}%`;
    });
    av.addEventListener('pointerup', () => {
      if (!on) return; on = false;
      seat.element.style.transition = ''; seat.element.style.zIndex = '';
      if (moved < 10) this._seatAction(seat); else this._persistLayout();
    });
  }

  // ── Seat actions ────────────────────────────────────

  async _seatAction(seat) {
    // Chair tap → always start voice enrollment. The name label handles
    // name-only edits on its own click path. Tapping an already-enrolled
    // chair re-enrolls (overwrites) so a user who mis-captured can retry
    // without finding a separate "re-enroll" button.
    this._enrollSeat(seat);
  }

  async _enrollSeat(seat) {
    // Dynamic-length enrollment: record continuously, probe the server for
    // a self-stated name every ~1.2s, and stop as soon as the name is
    // detected (or the user clicks the seat again, or we hit the safety
    // ceiling). Replaces the old fixed 2s capture.
    const nameLabel = seat.element.querySelector('.seat-name-label');
    const statusLabel = seat.element.querySelector('.seat-status-label');
    const pf = seat.element.querySelector('.progress-fill');
    const circ = 2 * Math.PI * 34;
    // Probe early and often. Detection latency is dominated by ASR round-trip
    // (~300-500 ms on GB10), so a 500 ms probe interval naturally throttles
    // to back-to-back probes without overlap. First probe at 1000 ms gives
    // enough audio for a self-introduction in any language (Japanese names
    // like "私は田中です" need ~1s minimum for ASR to produce usable text;
    // the server's detect-name endpoint requires ≥0.8s / 25600 bytes).
    // MAX_MS is a safety ceiling — a silent mic can't hang the UI.
    const MIN_MS = 1000;
    const MAX_MS = 15000;
    const PROBE_INTERVAL_MS = 500;

    seat.element.classList.remove('enrolled');
    seat.element.classList.add('enrolling');
    nameLabel.textContent = 'Say your name...';
    statusLabel.textContent = 'listening';
    pf.style.strokeDashoffset = circ;

    // Manual-stop: clicking the seat while enrolling finalizes immediately
    // with whatever's been captured so far. Bind once, and clear on exit.
    //
    // The seat is started from a pointerup handler on .seat-avatar. The
    // browser then dispatches a *synthetic click* on the same element right
    // after pointerup. Without a guard, that synthetic click hits the
    // capture-phase stopHandler we register here and sets manualStop=true on
    // the very first 100 ms loop tick, so enrollment ends instantly. Ignore
    // any click that lands within the first 400 ms — that is comfortably
    // longer than the synthetic-click delay but well below the time it would
    // take a user to deliberately tap the seat a second time to abort.
    let manualStop = false;
    const stopArmedAt = performance.now() + 400;
    const stopHandler = (e) => {
      if (performance.now() < stopArmedAt) return;
      e.stopPropagation();
      manualStop = true;
    };
    seat.element.addEventListener('click', stopHandler, { capture: true });

    // Reuse the warmed-up mic when available so the user gets instant capture
    // instead of waiting ~500-1500 ms for getUserMedia + AudioContext boot.
    // Falls back to a one-shot acquisition if priming hasn't completed yet
    // (e.g. very fast click before the warm-up promise resolved).
    let proc = null;
    let usedWarmup = false;
    let localAc = null;
    let localStream = null;
    let localSource = null;
    if (micWarmup.primed) {
      usedWarmup = true;
      if (micWarmup.ac.state === 'suspended') {
        try { await micWarmup.ac.resume(); } catch {}
      }
    }
    const acRef = () => usedWarmup ? micWarmup.ac : localAc;
    const sourceRef = () => usedWarmup ? micWarmup.source : localSource;

    const chunks = [];
    // Build an Int16 PCM buffer from the accumulated Float32 chunks at 16kHz
    const buildPcm16 = () => {
      const totLen = chunks.reduce((s, c) => s + c.length, 0);
      if (!totLen) return new Int16Array(0);
      const full = new Float32Array(totLen);
      let off = 0;
      for (const c of chunks) { full.set(c, off); off += c.length; }
      const ratio = acRef().sampleRate / 16000;
      const outLen = Math.floor(totLen / ratio);
      const a16k = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) a16k[i] = full[Math.min(Math.floor(i * ratio), totLen - 1)];
      const s16 = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const v = Math.max(-1, Math.min(1, a16k[i]));
        s16[i] = v < 0 ? v * 32768 : v * 32767;
      }
      return s16;
    };

    try {
      if (!usedWarmup) {
        localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        localAc = new AudioContext();
        if (localAc.state === 'suspended') await localAc.resume();
        localSource = localAc.createMediaStreamSource(localStream);
      }
      proc = acRef().createScriptProcessor(4096, 1, 1);
      const startedAt = performance.now();
      proc.onaudioprocess = e => {
        chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        const elMs = performance.now() - startedAt;
        // Ring fill: grows toward MAX_MS, resetting visually when name lands
        pf.style.strokeDashoffset = circ * (1 - Math.min(1, elMs / MAX_MS));
      };
      sourceRef().connect(proc);
      proc.connect(acRef().destination);

      // Probe loop — runs concurrently with capture. As soon as the probe
      // returns a name, we paint the chair with that name *immediately* and
      // wake the loop via a Promise so the user sees instant feedback rather
      // than waiting for the next 100 ms tick.
      let detectedName = null;
      let lastProbeAt = 0;
      let probeInFlight = false;
      let wakeLoop;
      const wakePromise = () => new Promise(r => { wakeLoop = r; });
      let waiter = wakePromise();
      const sleep = (ms) => new Promise(r => setTimeout(r, ms));

      const onNameDetected = (name) => {
        if (detectedName) return;
        detectedName = name;
        // Instant feedback — paint the chair before we even stop the mic so
        // the user knows we heard them. Final classes/state get applied below
        // after the embedding extraction returns.
        nameLabel.textContent = name;
        statusLabel.textContent = 'captured';
        pf.style.strokeDashoffset = 0;
        if (wakeLoop) wakeLoop();
      };

      while (!manualStop && !detectedName) {
        const elMs = performance.now() - startedAt;
        if (elMs >= MAX_MS) break;

        if (elMs >= MIN_MS && !probeInFlight && (elMs - lastProbeAt) >= PROBE_INTERVAL_MS) {
          probeInFlight = true;
          lastProbeAt = elMs;
          const s16 = buildPcm16();
          statusLabel.textContent = `${(elMs / 1000).toFixed(1)}s · listening`;
          (async () => {
            try {
              const resp = await fetch(`${API}/api/room/enroll/detect-name`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/octet-stream' },
                body: s16.buffer,
              });
              if (resp.ok) {
                const j = await resp.json();
                if (j.name) onNameDetected(j.name);
              }
            } catch {}
            probeInFlight = false;
            if (wakeLoop) wakeLoop();
          })();
        } else if (!detectedName) {
          statusLabel.textContent = `${(elMs / 1000).toFixed(1)}s · listening`;
        }
        // Wake on probe completion or after 100 ms, whichever comes first.
        await Promise.race([waiter, sleep(100)]);
        waiter = wakePromise();
      }

      // Stop capture. Detach the ScriptProcessor but leave the warmed-up
      // mic stream + AudioContext alive so the next chair tap is also
      // instant. Local (non-warm-up) resources are torn down in `finally`.
      try { proc.disconnect(); } catch {}
      try { sourceRef()?.disconnect(proc); } catch {}

      const s16final = buildPcm16();
      if (s16final.length < 16000) {
        // <1s captured — abort rather than poison the embedding store
        throw new Error('Too short');
      }

      // If we already painted a detected name, leave it visible during the
      // embedding extraction instead of flashing "Processing..." over it.
      if (!detectedName) {
        nameLabel.textContent = 'Processing...';
        statusLabel.textContent = '';
      } else {
        statusLabel.textContent = 'saving voice…';
      }
      // If we already have a name from the probe, pin it so the final
      // enrollment call doesn't re-run ASR unnecessarily.
      const q = detectedName ? `?name=${encodeURIComponent(detectedName)}` : '';
      const resp = await fetch(`${API}/api/room/enroll${q}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: s16final.buffer,
      });
      if (!resp.ok) throw new Error((await resp.json()).error || 'Failed');
      const result = await resp.json();
      seat.enrolled = true;
      seat.enrollmentId = result.enrollment_id;
      seat.name = result.name;
      seat.element.classList.remove('enrolling');
      seat.element.classList.add('enrolled');
      nameLabel.textContent = result.name;
      nameLabel.title = 'Click to edit name';
      nameLabel.style.cursor = 'pointer';
      // The persistent click-to-edit listener was attached in _renderSeat;
      // no need to re-add here.
      statusLabel.textContent = 'Enrolled';
      pf.style.strokeDashoffset = 0;
    } catch (err) {
      nameLabel.textContent = 'Tap to retry';
      statusLabel.textContent = err.message || 'Error';
      seat.element.classList.remove('enrolling');
      pf.style.strokeDashoffset = circ;
      // Ensure we released any lingering media resources on error paths
      try { proc?.disconnect(); } catch {}
      try { sourceRef()?.disconnect(); } catch {}
    } finally {
      seat.element.removeEventListener('click', stopHandler, { capture: true });
      // Tear down the local fallback resources only — leave the warm-up
      // singleton alive so the next chair tap is also instant.
      if (!usedWarmup) {
        try { localSource?.disconnect(); } catch {}
        if (localStream) localStream.getTracks().forEach(t => t.stop());
        if (localAc && localAc.state !== 'closed') {
          try { await localAc.close(); } catch {}
        }
      }
    }
    this._updateHint();
    this._persistLayout();
  }

  _editName(seat) {
    const lbl = seat.element.querySelector('.seat-name-label');
    const cur = seat.name || '';
    const inp = document.createElement('input');
    inp.className = 'seat-name-input';
    inp.value = cur;
    inp.placeholder = 'Enter name';
    inp.maxLength = 20;
    lbl.replaceWith(inp);
    inp.focus();
    if (cur) inp.select();
    let cancelled = false;
    const fin = () => {
      const nm = cancelled ? cur : (inp.value.trim() || cur);
      seat.name = nm;
      // Promote the seat to "set up" when a name is entered on a previously
      // blank chair — matches the old modal's "Just enter a name" path.
      if (nm && !seat.enrolled) {
        seat.enrolled = true;
        seat.element.classList.add('enrolled');
        const status = seat.element.querySelector('.seat-status-label');
        if (status) status.textContent = 'Named';
      }
      const l = document.createElement('span');
      l.className = 'seat-name-label';
      l.textContent = nm || 'Click chair to enroll';
      l.title = nm ? 'Click to edit name' : 'Click chair to enroll voice';
      l.style.cursor = 'pointer';
      l.addEventListener('click', (e) => {
        e.stopPropagation();
        this._editName(seat);
      });
      inp.replaceWith(l);
      if (seat.enrollmentId && nm && nm !== cur) {
        fetch(`${API}/api/room/enroll/rename?id=${seat.enrollmentId}&name=${encodeURIComponent(nm)}`, { method: 'POST' }).catch(() => {});
      }
      this._updateHint();
      this._persistLayout();
    };
    inp.addEventListener('blur', fin);
    inp.addEventListener('keydown', e => {
      if (e.key === 'Enter') inp.blur();
      if (e.key === 'Escape') { cancelled = true; inp.blur(); }
    });
  }

  // ── Persist / Load ─────────────────────────────────

  _persistLayout() {
    clearTimeout(this._persistTimer);
    this._persistTimer = setTimeout(() => {
      const body = {
        preset: this.preset,
        tables: this.tables.map(t => ({
          table_id: t.tableId, x: +t.x.toFixed(1), y: +t.y.toFixed(1),
          width: +t.width.toFixed(1), height: +t.height.toFixed(1),
          border_radius: +t.borderRadius.toFixed(1), label: t.label || '',
        })),
        seats: this.seats.map(s => ({
          seat_id: s.seatId, x: +s.x.toFixed(1), y: +s.y.toFixed(1),
          enrollment_id: s.enrollmentId || null, speaker_name: s.name || '',
        })),
      };
      // In live/review mode, persist to the meeting's room.json.
      // In setup mode, persist to the session draft.
      const url = (this.mode === 'live' || this.mode === 'review') && this.meetingId
        ? `${API}/api/meetings/${_enc(this.meetingId)}/room/layout`
        : `${API}/api/room/layout`;
      fetch(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).catch(() => {});
    }, 300);
  }

  async _loadLayout() {
    try {
      const resp = await fetch(`${API}/api/room/layout`);
      const data = await resp.json();
      if (data.tables?.length > 0 || data.seats?.length > 0) {
        this.preset = data.preset || 'boardroom';
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.toggle('active', b.dataset.preset === this.preset));
        (data.tables || []).forEach(t => {
          const table = { tableId: t.table_id, x: t.x, y: t.y, width: t.width, height: t.height, borderRadius: t.border_radius, label: t.label || '', element: null };
          this.tables.push(table); this._renderTable(table); this._applyTable(table);
        });
        (data.seats || []).forEach(s => {
          const seat = { seatId: s.seat_id, x: s.x, y: s.y, name: s.speaker_name || '', enrolled: !!s.enrollment_id, enrollmentId: s.enrollment_id || null, element: null };
          this.seats.push(seat); this._renderSeat(seat); this._applySeat(seat);
        });
      } else {
        // Empty canvas by default — user adds a table/preset or starts directly
        this.preset = '';
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      }
    } catch {
      this.preset = '';
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    }
    this._updateHint();
  }

  _updateHint() {
    const en = this.seats.filter(s => s.enrolled).length, tot = this.seats.length;
    this.btnStart.disabled = false; // Always allow starting
    if (tot === 0 && this.tables.length === 0) {
      this.hintEl.textContent = 'Click Start Meeting, or pick a layout / add seats first';
    }
    else if (tot === 0) { this.hintEl.textContent = 'Add seats or start directly'; }
    else if (en === tot) { this.hintEl.textContent = 'All enrolled — ready to start'; }
    else if (en > 0) { this.hintEl.textContent = `${en}/${tot} enrolled — click seats to set up, or start now`; }
    else { this.hintEl.textContent = 'Click seats to enroll, or start meeting now'; }
    // Refresh empty-canvas affordance in sync with hint
    this._updateEmptyOverlay();
  }

  _updateEmptyOverlay() {
    if (!this.canvas) return;
    const isEmpty = this.tables.length === 0 && this.seats.length === 0;
    let overlay = this.canvas.querySelector('.canvas-empty-overlay');

    if (!isEmpty) {
      overlay?.remove();
      return;
    }
    if (overlay) return; // Already present

    overlay = document.createElement('div');
    overlay.className = 'canvas-empty-overlay';
    overlay.innerHTML = `
      <div class="empty-title">Start with a layout</div>
      <div class="empty-sub">Pick a preset, add a table, or start an empty meeting</div>
      <div class="empty-actions">
        <button class="empty-btn" data-action="rectangle" title="Conference table">
          <svg width="26" height="18" viewBox="0 0 18 12" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="1" y="1" width="16" height="10" rx="1.5"/></svg>
          <span>Conference</span>
        </button>
        <button class="empty-btn" data-action="round" title="Round table">
          <svg width="22" height="22" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="7" cy="7" r="6"/></svg>
          <span>Round</span>
        </button>
        <button class="empty-btn" data-action="boardroom" title="Oval boardroom">
          <svg width="26" height="18" viewBox="0 0 18 12" fill="none" stroke="currentColor" stroke-width="1.2"><ellipse cx="9" cy="6" rx="8" ry="5"/></svg>
          <span>Boardroom</span>
        </button>
        <button class="empty-btn" data-action="add-table" title="Add a single table">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="6" width="18" height="12" rx="3"/><path d="M12 10v4M10 12h4"/></svg>
          <span>+ Table</span>
        </button>
        <button class="empty-btn" data-action="add-seat" title="Add a single seat">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="8" r="4"/><path d="M4 21v-1a8 8 0 0 1 16 0v1"/></svg>
          <span>+ Seat</span>
        </button>
        <button class="empty-btn empty-btn-primary" data-action="start-empty" title="Start without a layout">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
          <span>Start Empty</span>
        </button>
      </div>
    `;
    overlay.querySelectorAll('.empty-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const action = btn.dataset.action;
        if (action === 'add-table') {
          // Use the bulk add so the shared seat-count input applies
          // regardless of whether the user clicked here or in the footer.
          const input = document.getElementById('seat-count-input');
          let n = parseInt(input?.value, 10);
          if (!Number.isFinite(n) || n < 0) n = 0;
          if (n > 20) n = 20;
          this.addTableWithSeats(n);
        } else if (action === 'add-seat') {
          this.addSeatAtCenter();
        } else if (action === 'start-empty') {
          // Hard-reset the room layout: even though startMeeting()
          // adds the body.hide-table class, any tables/seats the
          // user laid down before opening the empty-state overlay
          // are still in this.tables / this.seats and would render
          // into the meeting-table-strip the moment the user toggles
          // the Table button on. "Start Empty" should mean what it
          // says — there is no room to remember. Clearing here is
          // the explicit, defensive form of "no virtual table".
          this.tables = [];
          this.seats = [];
          this.startMeeting();
        } else {
          this.applyPreset(action);
        }
      });
    });
    this.canvas.appendChild(overlay);
  }

  async startMeeting() {
    // Mark the fresh-start window BEFORE the async start POST. The
    // reconciler's shared transport predicate reads
    // _startInitiatedByThisTab (set via claimOwnership after the POST
    // returns) but `body.starting` is the sibling signal that keeps
    // navigation guards / reconcile's rehydration branch inert until
    // body.recording is set.
    document.body.classList.add('starting');

    // Transition to meeting mode. #landing-mode is hidden unconditionally
    // here (in addition to room-setup / view-mode) so a mid-start click
    // on Home never leaves landing visible on top of a running meeting.
    document.getElementById('landing-mode').style.display = 'none';
    document.getElementById('room-setup').style.display = 'none';
    document.getElementById('view-mode').style.display = 'none';
    document.getElementById('meeting-mode').style.display = '';
    document.getElementById('control-bar').style.display = '';
    document.body.classList.add('meeting-active');

    // NOTE: ``hide-table`` is intentionally NOT applied here — the
    // fresh-meeting cleanup inside ``startRecording`` (search "Wipe
    // layout overlays from the previous meeting") explicitly removes
    // hide-table along with the rest of the layout body classes to
    // prevent inheritance from a previous session. Adding it here
    // would be wiped milliseconds later. We apply it AFTER
    // startRecording returns instead — see the post-await block
    // below. The 2026-05-07 "virtual table is still there when
    // starting an empty meeting" report was this exact race.
    const tableBtn = document.getElementById('btn-toggle-table');
    if (tableBtn) tableBtn.classList.remove('active-toggle');

    // Clear any stale timeline/player from a previously viewed meeting
    document.getElementById('speaker-timeline').style.display = 'none';
    document.getElementById('player-bar').style.display = 'none';
    window.audioPlayer.hide();

    // Render miniature table strip (hidden by default via hide-table class,
    // but rendered so it's instantly ready when the user toggles it on)
    this._renderTableStrip();

    // Initialize transcript column renderers (global subscription handles delivery)
    window._gridRenderer = new CompactGridRenderer(document.getElementById("transcript-grid"), null, formatTime);

    try {
      await this._deps.startRecording(false);
      // Success signal: startRecording adds body.recording on success
      // and swallows its own errors via its catch block. Only claim
      // ownership when we can observe that class — that's the one
      // reliable indicator a real recorder pipeline is now up on this
      // tab. (window.current_meeting_id may be stale from a previous
      // start if the current attempt failed without clearing it.)
      if (document.body.classList.contains('recording')
       && window.current_meeting_id
       && this._deps.getReconciler()) {
        this._deps.getReconciler().claimOwnership(window.current_meeting_id);
      }
      // Apply hide-table AFTER the fresh-meeting cleanup inside
      // startRecording has run (it removes hide-table along with the
      // other layout overlays to prevent inheritance from a previous
      // meeting). Virtual table stays hidden by default on new meetings;
      // operator turns it on via the Table button in the control bar.
      if (document.body.classList.contains('recording')) {
        document.body.classList.add('hide-table');
        const tableBtn2 = document.getElementById('btn-toggle-table');
        if (tableBtn2) tableBtn2.classList.remove('active-toggle');
      }
    } finally {
      // Bulletproof button restore: even if startRecording threw after
      // advancing past the sync try/catch in the outer click handler,
      // this block always clears .starting. If the start succeeded we
      // are now in body.recording; if it failed, startRecording's own
      // catch block has rolled the meeting-mode UI back to room-setup.
      document.body.classList.remove('starting');
      if (this.btnStart) {
        this.btnStart.disabled = false;
        this.btnStart.textContent = this.btnStart.dataset.origText || 'Start Meeting';
      }
    }
  }

  _renderTableStrip() {
    const strip = document.getElementById('meeting-table-strip');
    strip.innerHTML = '';

    // Create an inner container that matches the setup canvas aspect ratio,
    // then scale it to fit the strip. This keeps proportions identical.
    const inner = document.createElement('div');
    inner.className = 'strip-inner';
    strip.appendChild(inner);

    // Render tables
    for (const t of this.tables) {
      const el = document.createElement('div');
      el.className = 'strip-table';
      el.style.left = `${t.x}%`;
      el.style.top = `${t.y}%`;
      el.style.width = `${t.width}%`;
      el.style.height = `${t.height}%`;
      el.style.borderRadius = `${t.borderRadius}%`;
      inner.appendChild(el);
    }

    // SEAT RENDERING — two modes:
    //
    // 1. Active meeting: overlay detected speakers (from _speakerRegistry)
    //    onto the layout seats in first-seen order. Falls back to design-time
    //    placeholder names if no speakers are detected yet.
    //
    // 2. Design-time (setup mode): use the layout's own seat data as-is.
    const inMeeting = document.body.classList.contains('recording')
                   || document.body.classList.contains('meeting-active');
    const detected = inMeeting ? getAllSpeakers() : [];

    // Make sure we have enough seat SLOTS for every detected speaker —
    // if detection finds 6 speakers but the layout only has 4, we still
    // show all 6 by overflowing into synthesized positions.
    const layoutSeats = this.seats;
    const totalSlots = Math.max(layoutSeats.length, detected.length);

    for (let i = 0; i < totalSlots; i++) {
      const layoutSeat = layoutSeats[i];
      const detectedSpeaker = detected[i];

      // Position: prefer layout seat coords; overflow speakers get
      // auto-positioned along the bottom edge.
      let x, y;
      if (layoutSeat) {
        x = layoutSeat.x;
        y = layoutSeat.y;
      } else {
        const overflow = i - layoutSeats.length;
        x = 10 + (overflow * 18) % 80;
        y = 88;
      }

      // Seat content: detected speaker (if active meeting) else layout name
      const clusterId = detectedSpeaker?.clusterId ?? null;
      const seqIndex = detectedSpeaker?.seqIndex ?? (i + 1);
      const nameFromSpeaker = detectedSpeaker?.displayName;
      const nameFromLayout = layoutSeat?.name;
      const displayName = nameFromSpeaker || nameFromLayout || '';
      // A seat is considered "enrolled" (has a real human name) when the
      // display name isn't the fallback "Speaker N" pattern. Layout
      // enrollment is older design-time state; we also honor it.
      const hasRealName = displayName && !/^Speaker\s+\d+$/i.test(displayName.trim());
      const isEnrolled = layoutSeat?.enrolled || hasRealName;
      // design-time: no cluster_id yet — fall back to seat index colour
      const color = clusterId != null ? getSpeakerColor(clusterId) : SPEAKER_COLORS[i % SPEAKER_COLORS.length];

      const el = document.createElement('div');
      el.className = `strip-seat${isEnrolled ? ' enrolled' : ''}${detectedSpeaker ? ' detected' : ''}`;
      el.dataset.speakerId = String(i);
      if (clusterId != null) el.dataset.clusterId = String(clusterId);
      el.style.left = `${x}%`;
      el.style.top = `${y}%`;
      el.style.setProperty('--seat-color', color);

      const hoverHint = detectedSpeaker
        ? (isEnrolled
            ? `Click to rename ${displayName}`
            : `Click to name Speaker ${seqIndex}`)
        : 'Click to edit seat';
      el.title = hoverHint;

      // Use the seq index as the avatar number so the seat matches the
      // "Speaker 1 / 2 / 3" naming in the transcript. Name label is
      // always rendered: the CSS adds a "tap to name" hint when empty.
      el.innerHTML = `<span class="strip-seat-num">${seqIndex}</span>` +
        `<span class="strip-seat-name">${hasRealName ? esc(displayName) : ''}</span>`;

      el.addEventListener('click', () => {
        if (detectedSpeaker) {
          // Active-meeting path: open rename modal tied to cluster_id
          _openSpeakerRenameModal(clusterId, displayName, color);
        } else {
          // Design-time or historical fallback
          const seatName = layoutSeat?.name || `Speaker ${i+1}`;
          showSpeakerModal(seatName, color,
            findSpeakerSegments(seatName), window._gridRenderer?._meetingId);
        }
      });
      inner.appendChild(el);
    }
  }
}
