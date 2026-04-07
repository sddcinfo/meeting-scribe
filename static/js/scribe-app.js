/**
 * Meeting Scribe — main application controller.
 *
 * Event-driven architecture:
 *   AudioWorklet → WebSocket (binary) → Server ASR/Translate → WebSocket (JSON) → SegmentStore → DOM
 *
 * SegmentStore: keyed by segment_id, tracks highest revision per segment.
 * DOM updates are batched via requestAnimationFrame for smooth rendering.
 */

const API = '';
const WS_PROTO = location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${WS_PROTO}//${location.host}/api/ws`;

// ─── Room Setup — Data-Driven Tables + Presets ──────────────

const SPEAKER_COLORS = ['#c45d20', '#1a6fb5', '#2a8540', '#9b2d7b', '#8b6914', '#b52d2d', '#2d6b5e', '#6b3fa0'];
let _persistTimer = null;

// ─── Modal System ───────────────────────────────────────────

function showModal(html, cssClass = '') {
  const overlay = document.getElementById('modal-overlay');
  const card = document.getElementById('modal-card');
  card.className = `modal-card ${cssClass}`;
  card.innerHTML = html;
  overlay.style.display = '';
  // Click backdrop to close
  overlay.onclick = (e) => { if (e.target === overlay) closeModal(); };
  document.addEventListener('keydown', _modalEscHandler);
  return card;
}

function closeModal() {
  document.getElementById('modal-overlay').style.display = 'none';
  document.removeEventListener('keydown', _modalEscHandler);
}

function _modalEscHandler(e) { if (e.key === 'Escape') closeModal(); }

async function confirmDialog(title, message, confirmText = 'Delete', danger = true) {
  return new Promise(resolve => {
    const card = showModal(`
      <div class="modal-confirm-title">${title}</div>
      <div class="modal-confirm-message">${message}</div>
      <div class="modal-confirm-actions">
        <button class="modal-btn" id="modal-cancel">Cancel</button>
        <button class="modal-btn ${danger ? 'danger' : ''}" id="modal-confirm">${confirmText}</button>
      </div>
    `, 'confirm');
    card.querySelector('#modal-cancel').onclick = () => { closeModal(); resolve(false); };
    card.querySelector('#modal-confirm').onclick = () => { closeModal(); resolve(true); };
    card.querySelector('#modal-confirm').focus();
  });
}

function showSpeakerModal(speakerName, speakerColor, segments, meetingId) {
  const totalSegs = segments.length;
  const totalMs = segments.reduce((s, e) => s + (e.end_ms - e.start_ms), 0);
  const totalSec = Math.round(totalMs / 1000);
  const languages = [...new Set(segments.map(e => e.language).filter(l => l !== 'unknown'))];
  const mid = meetingId || audioPlayer.meetingId || '';

  const segHtml = segments.map((e, i) => {
    const time = formatTime(e.start_ms);
    const lang = e.language !== 'unknown' ? `<span class="seg-lang">${e.language.toUpperCase()}</span>` : '';
    const trans = e.translation?.text
      ? `<div class="speaker-seg-translation">${esc(e.translation.text)}</div>`
      : '<div class="speaker-seg-translation" style="color:var(--text-muted);font-size:0.8rem">awaiting translation...</div>';
    const playBtn = mid
      ? `<button class="speaker-seg-play" data-idx="${i}" title="Play this segment">▶</button>`
      : '';
    return `<div class="speaker-segment" data-start="${e.start_ms}" data-end="${e.end_ms}">
      <div class="speaker-seg-left">
        ${playBtn}
        <div>
          <div class="speaker-seg-time">${time} ${lang}</div>
          <div class="speaker-seg-text">${esc(e.text)}</div>
        </div>
      </div>
      <div class="speaker-seg-right">${trans}</div>
    </div>`;
  }).join('');

  const playAllBtn = mid && totalSegs > 0
    ? `<button class="modal-btn" id="speaker-play-all" style="margin-left:auto">▶ Play All</button>`
    : '';

  const card = showModal(`
    <div class="speaker-modal-header">
      <div class="speaker-modal-dot" style="background:${speakerColor}"></div>
      <div class="speaker-modal-name">${esc(speakerName)}</div>
      <div class="speaker-modal-stats">
        <span>${totalSegs} segments</span>
        <span>${totalSec}s speaking</span>
        <span>${languages.join(', ') || '—'}</span>
        ${playAllBtn}
      </div>
      <button class="speaker-modal-close" id="speaker-modal-close-btn">&times;</button>
    </div>
    <div class="speaker-modal-body">${segHtml || '<div style="padding:2rem;text-align:center;color:var(--text-muted)">No segments yet</div>'}</div>
  `, 'speaker-modal');

  // Close button
  card.querySelector('#speaker-modal-close-btn')?.addEventListener('click', closeModal);

  // Wire play buttons
  if (mid) {
    card.querySelectorAll('.speaker-seg-play').forEach(btn => {
      btn.addEventListener('click', () => {
        const seg = btn.closest('.speaker-segment');
        const startMs = parseInt(seg.dataset.start);
        const endMs = parseInt(seg.dataset.end);
        audioPlayer.playSegment(mid, startMs, endMs);
        // Highlight this segment
        card.querySelectorAll('.speaker-segment.playing').forEach(s => s.classList.remove('playing'));
        seg.classList.add('playing');
      });
    });

    // Play all sequentially
    card.querySelector('#speaker-play-all')?.addEventListener('click', async () => {
      for (const seg of card.querySelectorAll('.speaker-segment')) {
        const startMs = parseInt(seg.dataset.start);
        const endMs = parseInt(seg.dataset.end);
        card.querySelectorAll('.speaker-segment.playing').forEach(s => s.classList.remove('playing'));
        seg.classList.add('playing');
        seg.scrollIntoView({ behavior: 'smooth', block: 'center' });
        audioPlayer.playSegment(mid, startMs, endMs);
        // Wait for playback to finish
        await new Promise(r => {
          audioPlayer.audio.onended = r;
          setTimeout(r, (endMs - startMs) + 1000); // fallback timeout
        });
      }
    });
  }
}

class RoomSetup {
  constructor() {
    this.tables = [];
    this.seats = [];
    this.preset = 'rectangle';
    this.selected = null; // { type: 'seat'|'table', id: string }
    this.canvas = document.getElementById('room-canvas');
    this.btnStart = document.getElementById('btn-start-meeting');
    this.hintEl = document.getElementById('setup-hint');

    document.getElementById('btn-add-seat').addEventListener('click', () => this.addSeatAtCenter());
    document.getElementById('btn-add-table')?.addEventListener('click', () => this.addTable());
    this.btnStart.addEventListener('click', () => this.startMeeting());

    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => this.applyPreset(btn.dataset.preset));
    });

    // Click canvas to deselect
    this.canvas.addEventListener('click', (e) => {
      if (e.target === this.canvas) this._deselect();
    });

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

  applyPreset(preset) {
    this.preset = preset;
    document.querySelectorAll('.preset-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.preset === preset)
    );

    // Clear current scene
    this.tables.forEach(t => t.element?.remove());
    this.seats.forEach(s => s.element?.remove());
    this.tables = [];
    this.seats = [];

    const P = this._presetData(preset);
    P.tables.forEach(t => { this.tables.push(t); this._renderTable(t); this._applyTable(t); });
    P.seats.forEach(s => { this.seats.push(s); this._renderSeat(s); this._applySeat(s); });
    this._updateHint();
    this._persistLayout();
  }

  _id() { return crypto.randomUUID().slice(0, 8); }

  _presetData(preset) {
    const T = (x, y, w, h, br, label) => ({
      tableId: this._id(), x, y, width: w, height: h, borderRadius: br, label: label || '', element: null
    });
    const S = (x, y) => ({
      seatId: this._id(), x, y, name: '', enrolled: false, enrollmentId: null, element: null
    });

    switch (preset) {
      case 'boardroom':
        return { tables: [T(50, 50, 50, 28, 50, '')], seats: this._ellipseSeats(50, 50, 35, 22, 2) };
      case 'round':
        return { tables: [T(50, 50, 30, 30, 50, '')], seats: this._ellipseSeats(50, 50, 26, 26, 2) };
      case 'square':
        return { tables: [T(50, 50, 28, 28, 3, '')], seats: [S(50, 22), S(50, 78)] };
      case 'rectangle':
        return { tables: [T(50, 50, 44, 22, 3, '')], seats: [S(50, 28), S(50, 72)] };
      case 'classroom':
        return {
          tables: [T(50, 15, 60, 5, 2, 'FRONT')],
          seats: [S(35, 38), S(65, 38)],
        };
      case 'u_shape':
        return {
          tables: [T(22, 50, 8, 45, 2, ''), T(50, 76, 48, 6, 2, ''), T(78, 50, 8, 45, 2, '')],
          seats: [S(12, 40), S(88, 40)],
        };
      case 'pods':
        return {
          tables: [T(30, 35, 20, 16, 3, 'A'), T(70, 35, 20, 16, 3, 'B'), T(30, 70, 20, 16, 3, 'C'), T(70, 70, 20, 16, 3, 'D')],
          seats: [S(22, 22), S(78, 22)],
        };
      case 'freeform':
      default:
        return { tables: [], seats: [S(35, 45), S(65, 55)] };
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
    const perim = 2 * (2 * hw + 2 * hh);
    return Array.from({ length: n }, (_, i) => {
      let d = (i / n) * perim, x, y;
      if (d < 2 * hw) { x = cx - hw + d; y = cy - hh - 6; }
      else if (d < 2 * hw + 2 * hh) { d -= 2 * hw; x = cx + hw + 6; y = cy - hh + d; }
      else if (d < 4 * hw + 2 * hh) { d -= 2 * hw + 2 * hh; x = cx + hw - d; y = cy + hh + 6; }
      else { d -= 4 * hw + 2 * hh; x = cx - hw - 6; y = cy + hh - d; }
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
  }

  removeTable(tableId) {
    const idx = this.tables.findIndex(t => t.tableId === tableId);
    if (idx === -1) return;
    this.tables[idx].element?.remove();
    this.tables.splice(idx, 1);
    this._persistLayout();
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
    const color = SPEAKER_COLORS[ci % SPEAKER_COLORS.length];
    const circ = 2 * Math.PI * 34;
    const n = document.createElement('div');
    n.className = 'seat-node';
    n.dataset.seatId = seat.seatId;
    n.style.setProperty('--seat-color', color);
    n.innerHTML = `
      <button class="seat-remove" title="Remove">&times;</button>
      <div class="seat-avatar" title="Click to enroll">
        <span class="seat-index">${ci + 1}</span>
        <svg class="seat-progress" viewBox="0 0 72 72">
          <circle class="progress-bg" cx="36" cy="36" r="34"/>
          <circle class="progress-fill" cx="36" cy="36" r="34" stroke-dasharray="${circ}" stroke-dashoffset="${circ}"/>
        </svg>
      </div>
      <span class="seat-name-label">${seat.name || 'Click to enroll'}</span>
      <span class="seat-status-label">${seat.enrolled ? 'Enrolled' : ''}</span>
    `;
    this.canvas.appendChild(n);
    seat.element = n;
    if (seat.enrolled) {
      n.classList.add('enrolled');
      n.querySelector('.seat-name-label').style.cursor = 'pointer';
      n.querySelector('.seat-name-label').addEventListener('click', () => this._editName(seat), { once: true });
    }
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
    if (seat.enrolled) {
      // Already enrolled — edit name
      this._editName(seat);
      return;
    }
    // Show choice: enroll voice or just name
    const card = showModal(`
      <div class="modal-confirm-title">Set Up Seat</div>
      <div class="modal-confirm-message">How would you like to identify this speaker?</div>
      <div class="modal-confirm-actions" style="flex-direction:column;gap:8px;">
        <button class="modal-btn" id="modal-enroll" style="width:100%;text-align:left;">
          🎤 Enroll Voice (2s recording)
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:2px;">Records voice for speaker identification</div>
        </button>
        <button class="modal-btn" id="modal-name" style="width:100%;text-align:left;">
          ✏️ Just enter a name
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:2px;">Name only, no voice enrollment</div>
        </button>
        <button class="modal-btn" id="modal-cancel" style="width:100%;">Cancel</button>
      </div>
    `, 'confirm');
    card.querySelector('#modal-enroll').onclick = () => { closeModal(); this._enrollSeat(seat); };
    card.querySelector('#modal-name').onclick = () => { closeModal(); this._nameSeat(seat); };
    card.querySelector('#modal-cancel').onclick = () => closeModal();
  }

  _nameSeat(seat) {
    const nameLabel = seat.element.querySelector('.seat-name-label');
    const input = document.createElement('input');
    input.className = 'seat-name-input';
    input.placeholder = 'Enter name';
    input.maxLength = 20;
    nameLabel.replaceWith(input);
    input.focus();
    const finish = () => {
      const name = input.value.trim();
      if (!name) { input.replaceWith(nameLabel); return; }
      seat.name = name;
      seat.enrolled = true; // Mark as set up (no voice, but named)
      seat.element.classList.add('enrolled');
      const label = document.createElement('span');
      label.className = 'seat-name-label';
      label.textContent = name;
      label.title = 'Click to edit';
      label.style.cursor = 'pointer';
      label.addEventListener('click', () => this._editName(seat), { once: true });
      input.replaceWith(label);
      seat.element.querySelector('.seat-status-label').textContent = 'Named';
      this._updateHint();
      this._persistLayout();
    };
    input.addEventListener('blur', finish);
    input.addEventListener('keydown', e => { if (e.key === 'Enter') input.blur(); if (e.key === 'Escape') { input.value = ''; input.blur(); } });
  }

  async _enrollSeat(seat) {
    const nameLabel = seat.element.querySelector('.seat-name-label');
    const statusLabel = seat.element.querySelector('.seat-status-label');
    const pf = seat.element.querySelector('.progress-fill');
    const circ = 2 * Math.PI * 34;
    seat.element.classList.remove('enrolled'); seat.element.classList.add('enrolling');
    nameLabel.textContent = 'Say your name...'; statusLabel.textContent = '5s'; pf.style.strokeDashoffset = circ;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ac = new AudioContext(); if (ac.state === 'suspended') await ac.resume();
      const src = ac.createMediaStreamSource(stream);
      const proc = ac.createScriptProcessor(4096, 1, 1);
      const chunks = []; const dur = 2;
      proc.onaudioprocess = e => {
        chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        const el = chunks.length * 4096 / ac.sampleRate;
        pf.style.strokeDashoffset = circ * (1 - Math.min(1, el / dur));
        const rem = Math.ceil(dur - el); statusLabel.textContent = rem > 0 ? `${rem}s` : '';
      };
      src.connect(proc); proc.connect(ac.destination);
      await new Promise(r => setTimeout(r, dur * 1000));
      proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t => t.stop());
      const tot = chunks.reduce((s, c) => s + c.length, 0);
      const full = new Float32Array(tot); let off = 0;
      for (const c of chunks) { full.set(c, off); off += c.length; }
      const ratio = ac.sampleRate / 16000; const outLen = Math.floor(tot / ratio);
      const a16k = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) a16k[i] = full[Math.min(Math.floor(i * ratio), tot - 1)];
      await ac.close();
      const s16 = new Int16Array(a16k.length);
      for (let i = 0; i < a16k.length; i++) { const v = Math.max(-1, Math.min(1, a16k[i])); s16[i] = v < 0 ? v * 32768 : v * 32767; }
      nameLabel.textContent = 'Processing...'; statusLabel.textContent = '';
      const resp = await fetch(`${API}/api/room/enroll`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: s16.buffer });
      if (!resp.ok) throw new Error((await resp.json()).error || 'Failed');
      const result = await resp.json();
      seat.enrolled = true; seat.enrollmentId = result.enrollment_id; seat.name = result.name;
      seat.element.classList.remove('enrolling'); seat.element.classList.add('enrolled');
      nameLabel.textContent = result.name; nameLabel.title = 'Click to edit'; nameLabel.style.cursor = 'pointer';
      nameLabel.addEventListener('click', () => this._editName(seat), { once: true });
      statusLabel.textContent = 'Enrolled'; pf.style.strokeDashoffset = 0;
    } catch (err) {
      nameLabel.textContent = 'Tap to retry'; statusLabel.textContent = err.message || 'Error';
      seat.element.classList.remove('enrolling'); pf.style.strokeDashoffset = circ;
    }
    this._updateHint(); this._persistLayout();
  }

  _editName(seat) {
    const lbl = seat.element.querySelector('.seat-name-label'); const cur = seat.name;
    const inp = document.createElement('input'); inp.className = 'seat-name-input'; inp.value = cur; inp.maxLength = 20;
    lbl.replaceWith(inp); inp.focus(); inp.select();
    const fin = () => {
      const nm = inp.value.trim() || cur; seat.name = nm;
      const l = document.createElement('span'); l.className = 'seat-name-label'; l.textContent = nm;
      l.title = 'Click to edit'; l.style.cursor = 'pointer';
      l.addEventListener('click', () => this._editName(seat), { once: true });
      inp.replaceWith(l);
      if (seat.enrollmentId && nm !== cur) fetch(`${API}/api/room/enroll/rename?id=${seat.enrollmentId}&name=${encodeURIComponent(nm)}`, { method: 'POST' }).catch(() => {});
      this._persistLayout();
    };
    inp.addEventListener('blur', fin); inp.addEventListener('keydown', e => { if (e.key === 'Enter') inp.blur(); });
  }

  // ── Persist / Load ─────────────────────────────────

  _persistLayout() {
    clearTimeout(_persistTimer);
    _persistTimer = setTimeout(() => {
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
      fetch(`${API}/api/room/layout`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).catch(() => {});
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
        this.applyPreset('rectangle');
      }
    } catch { this.applyPreset('rectangle'); }
    this._updateHint();
  }

  _updateHint() {
    const en = this.seats.filter(s => s.enrolled).length, tot = this.seats.length;
    this.btnStart.disabled = false; // Always allow starting
    if (tot === 0) { this.hintEl.textContent = 'Add seats or start directly'; }
    else if (en === tot) { this.hintEl.textContent = 'All enrolled — ready to start'; }
    else if (en > 0) { this.hintEl.textContent = `${en}/${tot} enrolled — click seats to set up, or start now`; }
    else { this.hintEl.textContent = 'Click seats to enroll, or start meeting now'; }
  }

  startMeeting() {
    // Transition to meeting mode
    document.getElementById('room-setup').style.display = 'none';
    document.getElementById('view-mode').style.display = 'none';
    document.getElementById('meeting-mode').style.display = '';
    document.getElementById('control-bar').style.display = '';
    document.body.classList.add('meeting-active');

    // Render miniature table strip
    this._renderTableStrip();

    // Initialize 3-column renderers (global subscription handles delivery)
    window._gridRenderer = new GridRenderer(document.getElementById('transcript-grid'));

    startRecording(false);
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

    // Render seats
    this.seats.forEach((s, i) => {
      const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
      const el = document.createElement('div');
      el.className = `strip-seat${s.enrolled ? ' enrolled' : ''}`;
      el.style.left = `${s.x}%`;
      el.style.top = `${s.y}%`;
      el.style.borderColor = s.enrolled ? color : '';
      el.innerHTML = `<span style="color:${color}">${i + 1}</span><span class="strip-seat-name">${esc(s.name || '')}</span>`;
      el.addEventListener('click', () => {
        const segs = [...store.segments.values()].filter(ev =>
          ev.speakers?.some(sp => sp.identity === s.name || sp.cluster_id === i)
        );
        showSpeakerModal(s.name || `Speaker ${i+1}`, color, segs, window._gridRenderer?._meetingId);
      });
      inner.appendChild(el);
    });
  }
}

const roomSetup = new RoomSetup();

// ─── Segment Store ───────────────────────────────────────────
// Reactive store keyed by segment_id. Only highest revision shown.

class SegmentStore {
  constructor() {
    this.segments = new Map();
    this.order = [];
    this._listeners = new Set();
  }

  subscribe(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }

  ingest(event) {
    const { segment_id, revision } = event;
    const existing = this.segments.get(segment_id);
    if (existing && existing.revision > revision) return;
    if (existing && existing.revision === revision) {
      const hasNewFinal = !existing.is_final && event.is_final;
      const hasNewTranslation = event.translation?.text && !existing.translation?.text;
      const hasTranslationUpdate = event.translation?.status && event.translation.status !== existing.translation?.status;
      if (!hasNewFinal && !hasNewTranslation && !hasTranslationUpdate) return;
    }
    const isNew = !existing;
    this.segments.set(segment_id, event);
    if (isNew) this.order.push(segment_id);
    for (const fn of this._listeners) fn(segment_id, event, isNew);
  }

  clear() { this.segments.clear(); this.order = []; for (const fn of this._listeners) fn(null, null, false); }
  get count() { return this.segments.size; }
}

// SPEAKER_COLORS defined in Room Setup section above

// ─── Audio Player ───────────────────────────────────────────

class AudioPlayer {
  constructor() {
    this.audio = new Audio();
    this.meetingId = null;
    this.currentSegmentId = null;
    this._bar = document.getElementById('player-bar');
    this._playBtn = document.getElementById('player-play');
    this._scrub = document.getElementById('player-scrub');
    this._current = document.getElementById('player-current');
    this._durationEl = document.getElementById('player-duration');
    this._speed = document.getElementById('player-speed');

    this._playBtn.addEventListener('click', () => this.togglePlay());
    this._scrub.addEventListener('input', () => {
      if (this.audio.duration) this.audio.currentTime = (this._scrub.value / 100) * this.audio.duration;
    });
    this._speed.addEventListener('change', () => { this.audio.playbackRate = parseFloat(this._speed.value); });
    this.audio.addEventListener('timeupdate', () => this._onTimeUpdate());
    this.audio.addEventListener('ended', () => this._onEnded());
  }

  /** Play a specific segment from a meeting */
  playSegment(meetingId, startMs, endMs, segmentId) {
    this.meetingId = meetingId;
    this.audio.src = `${API}/api/meetings/${meetingId}/audio?start_ms=${startMs}&end_ms=${endMs}`;
    this.audio.playbackRate = parseFloat(this._speed.value);
    this.audio.play();
    this._highlightRow(segmentId);
    this._playBtn.textContent = '⏸';
    this._playBtn.classList.add('playing');
  }

  /** Load full meeting audio for podcast-style playback */
  loadMeeting(meetingId, durationMs, timeline) {
    this.meetingId = meetingId;
    this._timeline = timeline || null;
    this.audio.src = `${API}/api/meetings/${meetingId}/audio`;
    this._durationEl.textContent = this._fmt(durationMs);
    this._scrub.value = 0;
    this._current.textContent = '00:00';
    this._bar.style.display = '';
    document.body.classList.add('has-player');
  }

  togglePlay() {
    if (this.audio.paused) { this.audio.play(); this._playBtn.textContent = '⏸'; this._playBtn.classList.add('playing'); }
    else { this.audio.pause(); this._playBtn.textContent = '▶'; this._playBtn.classList.remove('playing'); }
  }

  hide() {
    this.audio.pause();
    this._bar.style.display = 'none';
    document.body.classList.remove('has-player');
    this._clearHighlight();
  }

  _onTimeUpdate() {
    if (!this.audio.duration) return;
    const pct = (this.audio.currentTime / this.audio.duration) * 100;
    this._scrub.value = pct;
    this._current.textContent = this._fmt(this.audio.currentTime * 1000);

    // Scroll transcript to current playback position
    if (this._timeline && window._gridRenderer) {
      const currentMs = this.audio.currentTime * 1000;
      const seg = this._timeline.find(s => currentMs >= s.start_ms && currentMs <= s.end_ms);
      if (seg && seg._segmentId) {
        const row = document.querySelector(`.grid-row[data-segment-id="${seg._segmentId}"]`);
        if (row && !row.classList.contains('playing')) {
          this._clearHighlight();
          row.classList.add('playing');
          row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }
  }

  _onEnded() {
    this._playBtn.textContent = '▶';
    this._playBtn.classList.remove('playing');
    this._clearHighlight();
  }

  _highlightRow(segmentId) {
    this._clearHighlight();
    if (segmentId) {
      this.currentSegmentId = segmentId;
      document.querySelector(`.grid-row[data-segment-id="${segmentId}"]`)?.classList.add('playing');
    }
  }

  _clearHighlight() {
    document.querySelectorAll('.grid-row.playing').forEach(r => r.classList.remove('playing'));
    this.currentSegmentId = null;
  }

  _fmt(ms) {
    const s = Math.floor((ms || 0) / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }
}

const audioPlayer = new AudioPlayer();

// ─── Column Renderer (for 3-column meeting mode) ────────────

/**
 * GridRenderer — renders all 3 columns as a single unified grid.
 * Each segment_id gets one grid row with 3 cells (Live, EN, JA).
 * Rows are always aligned because they share the same grid row height.
 */
class GridRenderer {
  constructor(gridEl, meetingId) {
    this.gridEl = gridEl;
    this._meetingId = meetingId || null;
    this.rows = new Map();
    this._autoScroll = true;
    this.gridEl.addEventListener('scroll', () => {
      const { scrollTop, scrollHeight, clientHeight } = this.gridEl;
      this._autoScroll = scrollHeight - scrollTop - clientHeight < 60;
    });
  }

  update(segmentId, event) {
    if (!segmentId) { this._clear(); return; }
    if (!event.text) return;

    const time = formatTime(event.start_ms || 0);
    const lang = event.language;
    const tr = event.translation?.text || '';

    if (this.rows.has(segmentId)) {
      // Update existing row
      const r = this.rows.get(segmentId);
      this._fillCells(r, event, time, lang, tr);
    } else {
      // Create new grid row with 3 cells
      const row = document.createElement('div');
      row.className = `grid-row${event.is_final ? '' : ' partial'}`;
      row.dataset.segmentId = segmentId;

      const liveCell = document.createElement('div');
      liveCell.className = 'grid-cell';
      const enCell = document.createElement('div');
      enCell.className = 'grid-cell';
      const jaCell = document.createElement('div');
      jaCell.className = 'grid-cell';

      row.appendChild(liveCell);
      row.appendChild(enCell);
      row.appendChild(jaCell);
      this.gridEl.appendChild(row);

      const r = { row, liveCell, enCell, jaCell };
      this.rows.set(segmentId, r);
      this._fillCells(r, event, time, lang, tr);

      this._wireRowInteractions(row, segmentId, event);
    }

    if (this._autoScroll) {
      requestAnimationFrame(() => { this.gridEl.scrollTop = this.gridEl.scrollHeight; });
    }
  }

  _fillCells(r, event, time, lang, tr) {
    const langBadge = lang !== 'unknown' ? `<span class="seg-lang">${lang.toUpperCase()}</span>` : '';
    r.row.className = `grid-row${event.is_final ? '' : ' partial'}`;

    // Speaker badge
    let speakerBadge = '';
    if (event.speakers?.length > 0) {
      const s = event.speakers[0];
      const name = s.identity || `S${(s.cluster_id || 0) + 1}`;
      const color = SPEAKER_COLORS[(s.cluster_id || 0) % SPEAKER_COLORS.length];
      speakerBadge = `<span class="seg-speaker-badge" style="color:${color}"><span class="seg-speaker-dot" style="background:${color}"></span>${esc(name)}</span>`;
    }

    // Live cell — shows speaker + original text
    r.liveCell.innerHTML = `<div class="col-seg-meta"><span class="seg-time">${time}</span>${speakerBadge}${langBadge}</div><div class="col-seg-text">${esc(event.text)}</div>`;

    // EN cell — EN original or JA→EN translation
    if (lang === 'en') {
      r.enCell.innerHTML = `<div class="col-seg-meta"><span class="seg-time">${time}</span></div><div class="col-seg-text">${esc(event.text)}</div>`;
      r.jaCell.innerHTML = tr
        ? `<div class="col-seg-meta"><span class="seg-time">${time}</span></div><div class="col-seg-text ja">${esc(tr)}</div>`
        : `<div class="col-seg-meta"><span class="seg-time">${time}</span></div><div class="col-seg-text" style="color:var(--text-muted)">...</div>`;
    } else if (lang === 'ja') {
      r.jaCell.innerHTML = `<div class="col-seg-meta"><span class="seg-time">${time}</span></div><div class="col-seg-text ja">${esc(event.text)}</div>`;
      r.enCell.innerHTML = tr
        ? `<div class="col-seg-meta"><span class="seg-time">${time}</span></div><div class="col-seg-text">${esc(tr)}</div>`
        : `<div class="col-seg-meta"><span class="seg-time">${time}</span></div><div class="col-seg-text" style="color:var(--text-muted)">...</div>`;
    } else {
      // Unknown language — show in live only
      r.enCell.innerHTML = '';
      r.jaCell.innerHTML = '';
    }
  }

  _wireRowInteractions(row, segmentId, event) {
    // Timestamps → play audio
    row.querySelectorAll('.seg-time').forEach(el => {
      el.addEventListener('click', () => {
        if (this._meetingId && event.start_ms != null && event.end_ms != null) {
          audioPlayer.playSegment(this._meetingId, event.start_ms, event.end_ms, segmentId);
        }
      });
    });
    // Speaker badges → speaker modal
    row.querySelectorAll('.seg-speaker-badge').forEach(el => {
      el.addEventListener('click', () => {
        const name = el.textContent.trim();
        const clusterId = 0;
        const color = el.style.color || SPEAKER_COLORS[0];
        const segs = [...store.segments.values()].filter(ev =>
          ev.speakers?.some(s => s.identity === name)
        );
        showSpeakerModal(name, color, segs, this._meetingId);
      });
    });
  }

  _clear() {
    this.rows.clear();
    this.gridEl.innerHTML = '';
  }
}

// ─── Audio Capture ───────────────────────────────────────────

class AudioPipeline {
  constructor() { this.ws = null; this.audioCtx = null; this.workletNode = null; this.stream = null; this.analyser = null; this.running = false; }

  async start(onEvent) {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }
    });
    this.audioCtx = new AudioContext();
    if (this.audioCtx.state === 'suspended') await this.audioCtx.resume();
    const sampleRate = this.audioCtx.sampleRate;
    this.analyser = this.audioCtx.createAnalyser(); this.analyser.fftSize = 256;
    await this.audioCtx.audioWorklet.addModule('/static/js/audio-worklet.js');
    this.workletNode = new AudioWorkletNode(this.audioCtx, 'scribe-audio-processor', { processorOptions: { sampleRate } });
    const source = this.audioCtx.createMediaStreamSource(this.stream);
    source.connect(this.analyser); source.connect(this.workletNode);
    this.ws = new WebSocket(WS_URL); this.ws.binaryType = 'arraybuffer';
    this.ws.onmessage = (evt) => {
      wsMessageCount++;
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'tts_audio' && msg.audio_url) {
          // Auto-play TTS audio (voice-cloned translated speech)
          if (!document.getElementById('tts-mute')?.checked) {
            audioPlayer.audio.src = `${API}${msg.audio_url}`;
            audioPlayer.audio.play().catch(() => {});
          }
        } else if (msg.type === 'seat_update') {
          // Update specific seat name in the table strip
          const seats = document.querySelectorAll('.strip-seat');
          let updated = false;
          seats.forEach(s => {
            const nameSpan = s.querySelector('.strip-seat-name');
            if (nameSpan && !nameSpan.textContent.trim() && !updated) {
              nameSpan.textContent = msg.speaker_name;
              s.classList.add('enrolled');
              updated = true;
            }
          });
        } else {
          onEvent(msg);
        }
      } catch(e) { console.warn('WS parse:', e); }
    };
    await new Promise((resolve, reject) => { this.ws.onopen = resolve; this.ws.onerror = reject; });
    this.workletNode.port.onmessage = (evt) => { if (this.ws?.readyState === WebSocket.OPEN) { audioChunkCount++; this.ws.send(evt.data); } };
    this.running = true;
  }

  async stop() {
    this.running = false; this.workletNode?.disconnect();
    if (this.audioCtx) await this.audioCtx.close();
    this.stream?.getTracks().forEach(t => t.stop()); this.ws?.close();
    this.workletNode = null; this.audioCtx = null; this.stream = null; this.ws = null; this.analyser = null;
  }

  getLevel() {
    if (!this.analyser) return 0;
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let sum = 0; for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    return Math.min(Math.sqrt(sum / data.length) * 8, 1);
  }
}

// ─── Timer + Helpers ─────────────────────────────────────────

class Timer {
  constructor(el) { this.el = el; this.startTime = 0; this.interval = null; }
  start() { this.startTime = Date.now(); this.interval = setInterval(() => this._tick(), 1000); }
  stop() { clearInterval(this.interval); }
  reset() { this.stop(); this.el.textContent = '00:00'; }
  _tick() {
    const s = Math.floor((Date.now() - this.startTime) / 1000);
    this.el.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }
}

function esc(text) { const d = document.createElement('div'); d.textContent = text; return d.innerHTML; }
// Meeting start wall clock (set when recording begins)
let _meetingStartWallMs = 0;

function formatTime(offsetMs) {
  if (_meetingStartWallMs > 0) {
    // Show wall clock time in browser timezone (e.g., JST)
    const wallMs = _meetingStartWallMs + (offsetMs || 0);
    const d = new Date(wallMs);
    const h = String(d.getHours()).padStart(2, '0');
    const m = String(d.getMinutes()).padStart(2, '0');
    const s = String(d.getSeconds()).padStart(2, '0');
    return `${h}:${m}:${s}`;
  }
  // Fallback: elapsed time (mm:ss)
  const total = Math.floor((offsetMs || 0) / 1000);
  const m = String(Math.floor(total / 60)).padStart(2, '0');
  const s = String(total % 60).padStart(2, '0');
  return `${m}:${s}`;
}

// ─── Init ────────────────────────────────────────────────────

const store = new SegmentStore();
const audio = new AudioPipeline();
const timer = new Timer(document.getElementById('timer'));

// Single global subscription that delegates to active column renderers
store.subscribe(() => { document.getElementById('segment-count').textContent = `${store.count} segments`; });
store.subscribe((id, evt) => {
  if (window._gridRenderer) window._gridRenderer.update(id, evt);
});

let meterRaf = null, wsMessageCount = 0, audioChunkCount = 0;

function updateMeter() {
  const pct = Math.round(audio.getLevel() * 100);
  document.getElementById('meter-bar').style.width = `${pct}%`;
  document.getElementById('segment-count').textContent = `${store.count} segs | ${wsMessageCount} ws | ${audioChunkCount} chunks | ${pct}% mic`;
  if (audio.running) meterRaf = requestAnimationFrame(updateMeter);
}

async function checkStatus() {
  try {
    const resp = await fetch(`${API}/api/status`);
    const data = await resp.json();
    const pills = document.getElementById('backend-pills');
    pills.innerHTML = [['ASR', data.backends.asr], ['Translate', data.backends.translate], ['Diarize', data.backends.diarize]]
      .map(([n, ok]) => `<span class="pill ${ok ? 'active' : 'inactive'}">${n}</span>`).join('');
    const m = data.metrics || {};
    const info = m.asr_load_ms > 0 ? ` (ASR ${(m.asr_load_ms/1000).toFixed(1)}s, NLLB ${(m.ollama_warmup_ms/1000).toFixed(1)}s)` : '';
    const ready = data.backends.asr && data.backends.translate;
    btnRecord.disabled = !ready;

    if (data.meeting?.state === 'recording' && ready) {
      // Active meeting found — reconnect to meeting mode
      const mid = data.meeting.id;
      document.getElementById('status-line').textContent = `Reconnecting to ${mid.slice(0, 8)}...`;
      document.getElementById('room-setup').style.display = 'none';
      document.getElementById('view-mode').style.display = 'none';
      document.getElementById('meeting-mode').style.display = '';
      document.body.classList.add('meeting-active');

      // Initialize column renderers (global subscription handles delivery)
      window._gridRenderer = new GridRenderer(document.getElementById('transcript-grid'));

      // Load existing transcript from journal
      try {
        const meetResp = await fetch(`${API}/api/meetings/${mid}`);
        const meetData = await meetResp.json();
        if (meetData.meta?.created_at) _meetingStartWallMs = new Date(meetData.meta.created_at).getTime();
        if (meetData.events) {
          for (const event of meetData.events) {
            store.ingest(event);
          }
        }
      } catch {}

      // Render table strip from saved layout
      roomSetup._renderTableStrip();

      await startRecording(true);
    } else {
      document.getElementById('status-line').textContent = !ready ? 'Warming up...' : `Ready${info}`;
    }
  } catch { document.getElementById('status-line').textContent = 'Server unreachable'; btnRecord.disabled = true; }
}

const btnRecord = document.getElementById('btn-record');
btnRecord.addEventListener('click', async () => {
  if (document.body.classList.contains('recording')) await stopRecording();
  else await startRecording(false);
});

async function validateMic() {
  const el = document.getElementById('status-line'); el.textContent = 'Checking mic...';
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext(); if (ctx.state === 'suspended') await ctx.resume();
  const src = ctx.createMediaStreamSource(stream); const an = ctx.createAnalyser(); an.fftSize = 2048; src.connect(an);
  let max = 0; const buf = new Float32Array(an.fftSize);
  for (let i = 0; i < 10; i++) { await new Promise(r => setTimeout(r, 50)); an.getFloatTimeDomainData(buf); max = Math.max(max, buf.reduce((m, v) => Math.max(m, Math.abs(v)), 0)); }
  stream.getTracks().forEach(t => t.stop()); await ctx.close();
  if (max < 0.001) { el.textContent = 'Mic silent! Check input volume'; throw new Error('Mic silent'); }
  el.textContent = `Mic OK (${(max * 100).toFixed(1)}%)`; return true;
}

async function startRecording(isResume) {
  try {
    if (!isResume) await validateMic();
    const resp = await fetch(`${API}/api/meeting/start`, { method: 'POST' });
    const data = await resp.json();
    if (!isResume && !data.resumed) store.clear();
    wsMessageCount = 0; audioChunkCount = 0;
    document.body.classList.add('recording');
    document.getElementById('status-line').textContent = `Recording ${data.meeting_id.slice(0, 8)}`;
    if (window._gridRenderer) window._gridRenderer._meetingId = data.meeting_id;
    audioPlayer.meetingId = data.meeting_id;
    _meetingStartWallMs = Date.now();
    timer.start();
    await audio.start((event) => store.ingest(event));
    updateMeter(); startAnalytics();
  } catch (err) { console.error('Start failed:', err); document.getElementById('status-line').textContent = `Error: ${err.message}`; }
}

async function stopRecording() {
  timer.stop(); cancelAnimationFrame(meterRaf);
  document.getElementById('meter-bar').style.width = '0%';
  document.getElementById('status-line').textContent = 'Finalizing...';
  audio.workletNode?.disconnect(); audio.stream?.getTracks().forEach(t => t.stop());
  if (audio.audioCtx) await audio.audioCtx.close();
  audio.audioCtx = null; audio.stream = null; audio.workletNode = null; audio.analyser = null;

  // Get meeting ID before stopping
  const statusResp = await fetch(`${API}/api/status`);
  const statusData = await statusResp.json();
  const meetingId = statusData.meeting?.id;

  await fetch(`${API}/api/meeting/stop`, { method: 'POST' });
  audio.ws?.close(); audio.ws = null; audio.running = false;
  stopAnalytics(); document.body.classList.remove('recording');
  document.body.classList.remove('meeting-active');

  // Stay on the meeting — hide control bar, show podcast player
  document.getElementById('control-bar').style.display = 'none';

  if (meetingId) {
    document.getElementById('status-line').textContent = `Meeting complete — ${meetingId.slice(0, 8)}`;
    // Load podcast player for the just-finished meeting
    try {
      const tlResp = await fetch(`${API}/api/meetings/${meetingId}/timeline`);
      if (tlResp.ok) {
        const tl = await tlResp.json();
        if (tl.duration_ms > 0) audioPlayer.loadMeeting(meetingId, tl.duration_ms);
      }
    } catch {}
    // Update grid renderer with meeting ID for playback
    if (window._gridRenderer) window._gridRenderer._meetingId = meetingId;
    // Refresh meetings list
    meetingsMgr?.refresh();
  } else {
    document.getElementById('status-line').textContent = 'Stopped';
  }
}

document.getElementById('btn-clear').addEventListener('click', () => { store.clear(); timer.reset(); });

let analyticsInterval = null;
function startAnalytics() {
  analyticsInterval = setInterval(async () => {
    try {
      const data = (await (await fetch(`${API}/api/status`)).json());
      const m = data.metrics || {};
      document.getElementById('metric-audio').textContent = `${(m.audio_s || 0).toFixed(0)}s`;
      document.getElementById('metric-asr').textContent = `${m.asr_eps || 0} e/s`;
      document.getElementById('metric-finals').textContent = `${m.asr_finals || 0} final`;
      document.getElementById('metric-trans').textContent = `${m.translations_completed || 0}/${m.translations_submitted || 0}`;
      document.getElementById('metric-avg-trans').textContent = m.avg_translation_ms > 0 ? `${m.avg_translation_ms}ms` : '';
    } catch {}
  }, 2000);
}
function stopAnalytics() { clearInterval(analyticsInterval); }

checkStatus();

// ─── Hamburger Menu ──────────────────────────────────────────

const hamburgerBtn = document.getElementById('btn-hamburger');
const meetingsPanel = document.getElementById('meetings-panel');
const meetingsBackdrop = document.getElementById('meetings-backdrop');

function toggleMeetingsPanel() {
  const open = meetingsPanel.classList.toggle('open');
  meetingsBackdrop.classList.toggle('open', open);
}

hamburgerBtn.addEventListener('click', toggleMeetingsPanel);
meetingsBackdrop.addEventListener('click', toggleMeetingsPanel);

// ─── Meetings History ────────────────────────────────────────

class MeetingsManager {
  constructor() {
    this.listEl = document.getElementById('meetings-list');
    this.btnNew = document.getElementById('btn-new-meeting');
    this.viewingMeetingId = null;

    this.btnNew.addEventListener('click', () => {
      if (document.body.classList.contains('recording')) { toggleMeetingsPanel(); return; }
      this.showSetup(); toggleMeetingsPanel();
    });
    this.refresh();
    setInterval(() => this.refresh(), 10000);
  }

  async refresh() {
    try {
      const resp = await fetch(`${API}/api/meetings`);
      const data = await resp.json();
      this._render(data.meetings || []);
    } catch {}
  }

  _render(meetings) {
    this.listEl.innerHTML = '';

    if (meetings.length === 0) {
      this.listEl.innerHTML = '<div style="padding:1rem;color:var(--text-muted);font-size:0.75rem;text-align:center;">No meetings yet</div>';
      return;
    }

    for (const m of meetings) {
      const item = document.createElement('div');
      item.className = `meeting-item${this.viewingMeetingId === m.meeting_id ? ' active' : ''}`;
      const date = m.created_at ? new Date(m.created_at) : null;
      const dateStr = date ? `${date.toLocaleDateString()} ${date.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}` : 'Unknown';
      item.innerHTML = `
        <div class="meeting-item-row">
          <div class="meeting-item-content">
            <div class="meeting-item-date">${dateStr}</div>
            <div class="meeting-item-info">
              <span class="meeting-item-state ${m.state}">${m.state}</span>
              ${m.event_count > 0 ? `${m.event_count} events` : ''}
            </div>
          </div>
          <button class="meeting-delete" title="Delete meeting">&times;</button>
        </div>
      `;
      item.querySelector('.meeting-item-content').addEventListener('click', () => this.viewMeeting(m.meeting_id));
      item.querySelector('.meeting-delete').addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteMeeting(m.meeting_id);
      });
      this.listEl.appendChild(item);
    }
  }

  async viewMeeting(meetingId) {
    // Don't navigate away during active recording
    if (document.body.classList.contains('recording')) {
      toggleMeetingsPanel();
      return;
    }
    this.viewingMeetingId = meetingId;
    this.refresh();
    toggleMeetingsPanel();

    try {
      const resp = await fetch(`${API}/api/meetings/${meetingId}`);
      const data = await resp.json();

      // Show meeting mode with 3 columns for past meeting replay
      document.getElementById('room-setup').style.display = 'none';
      document.getElementById('view-mode').style.display = 'none';
      document.getElementById('meeting-mode').style.display = '';

      // Initialize grid renderer with meeting ID for playback
      window._gridRenderer = new GridRenderer(document.getElementById('transcript-grid'), meetingId);
      store.clear();

      // Show podcast player if audio/timeline exists
      try {
        const tlResp = await fetch(`${API}/api/meetings/${meetingId}/timeline`);
        if (tlResp.ok) {
          const tl = await tlResp.json();
          if (tl.duration_ms > 0) audioPlayer.loadMeeting(meetingId, tl.duration_ms);
        }
      } catch {}

      // Set meeting start time for wall clock display
      if (data.meta?.created_at) {
        _meetingStartWallMs = new Date(data.meta.created_at).getTime();
      }

      // Load events into store → flows to all 3 columns
      if (data.events?.length > 0) {
        for (const event of data.events) {
          store.ingest(event);
        }
      }

      // Render table strip if room layout exists
      const strip = document.getElementById('meeting-table-strip');
      strip.innerHTML = '';
      if (data.room && (data.room.tables?.length || data.room.seats?.length)) {
        strip.style.display = '';
        for (const t of data.room.tables || []) {
          const el = document.createElement('div');
          el.className = 'strip-table';
          el.style.cssText = `left:${t.x}%;top:${t.y}%;width:${t.width}%;height:${t.height}%;border-radius:${t.border_radius}%`;
          strip.appendChild(el);
        }
        for (const s of data.room.seats || []) {
          const color = SPEAKER_COLORS[(data.room.seats.indexOf(s)) % SPEAKER_COLORS.length];
          const el = document.createElement('div');
          el.className = `strip-seat${s.enrollment_id ? ' enrolled' : ''}`;
          el.style.cssText = `left:${s.x}%;top:${s.y}%;border-color:${s.enrollment_id ? color : ''}`;
          el.innerHTML = `<span style="color:${color}">${esc(s.speaker_name?.[0] || '?')}</span><span class="strip-seat-name">${esc(s.speaker_name || '')}</span>`;
          strip.appendChild(el);
        }
      } else {
        strip.style.display = 'none';
      }

      // Hide control bar for replay (no stop button)
      document.getElementById('control-bar').style.display = 'none';
      document.getElementById('status-line').textContent = `Viewing ${meetingId.slice(0, 8)} · ${data.total_events || 0} events`;
    } catch (err) {
      document.getElementById('status-line').textContent = `Error: ${err.message}`;
    }
  }

  async deleteMeeting(meetingId) {
    const ok = await confirmDialog(
      'Delete Meeting?',
      'This will permanently delete the meeting and all its recordings, transcripts, and translations. This cannot be undone.',
    );
    if (!ok) return;
    try {
      const resp = await fetch(`${API}/api/meetings/${meetingId}`, { method: 'DELETE' });
      if (!resp.ok) { const e = await resp.json(); console.warn('Delete failed:', e.error); return; }
      if (this.viewingMeetingId === meetingId) this.showSetup();
      this.refresh();
    } catch (err) { console.warn('Delete error:', err); }
  }

  showSetup() {
    this.viewingMeetingId = null;
    this.refresh();
    document.getElementById('room-setup').style.display = '';
    document.getElementById('meeting-mode').style.display = 'none';
    document.getElementById('view-mode').style.display = 'none';
    audioPlayer.hide();
    store.clear();
    document.getElementById('status-line').textContent = 'Ready';
  }
}

const meetingsMgr = new MeetingsManager();
