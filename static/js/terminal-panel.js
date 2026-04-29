/* meeting-scribe terminal panel — xterm.js over a single-use-ticket WS.
 *
 * Exports `window.TerminalPanel`. The scribe-app.js popout loader
 * instantiates one on first toggle and keeps the same instance around.
 *
 * Design notes:
 *   · Auth is a two-factor handshake: admin cookie (from /admin/bootstrap)
 *     plus a single-use ticket minted over REST.
 *   · Binary frames carry stdin ('I' prefix) and stdout ('O' prefix).
 *     JSON text frames carry attach/resize/ack/ping/status/error/bye/pong.
 *   · Client-side flow control sends cumulative-byte ACKs so a malicious
 *     or stuck client can't OOM the server. The server closes the
 *     session on overrun; we'd rather take a repaint than a kill.
 */

(() => {
  'use strict';

  const PREFIX_OUTPUT_CODE = 0x4f; // 'O'
  const PREFIX_INPUT = 'I';

  // Hotspot has no internet — everything is vendored.
  const XTERM_BASE = '/static/vendor/xterm';
  const XTERM_VER = '6.0.0';

  const PASTE_CHUNK_BYTES = 200 * 1024;     // stay safely under server 256K cap
  const RESIZE_DEBOUNCE_MS = 150;
  const STATUS_INTERVAL_MS = 500;
  const ACK_MIN_BYTES = 16 * 1024;          // matches LOW_WATER on the server
  const MAX_RECONNECT_ATTEMPTS = 5;

  const FONT_SIZE_DEFAULT = 13;
  const FONT_SIZE_MIN = 9;
  const FONT_SIZE_MAX = 26;

  // Distinct dark theme tuned against the scribe popout's cream surfaces.
  // Graphite base, amber accent, cyan for web links. Not "VS Code dark".
  const THEME_GRAPHITE = {
    background: '#0b0e14',
    foreground: '#e6e1cf',
    cursor: '#ffb454',
    cursorAccent: '#0b0e14',
    selectionBackground: '#2d3847',
    selectionForeground: '#f6f5ef',
    black:   '#3b4048', red:     '#f07178', green:   '#aad94c',
    yellow:  '#ffb454', blue:    '#59c2ff', magenta: '#d2a6ff',
    cyan:    '#95e6cb', white:   '#c7c7c7',
    brightBlack:   '#6c7380', brightRed:     '#ff8f40',
    brightGreen:   '#b8cc52', brightYellow:  '#ffc66d',
    brightBlue:    '#73d0ff', brightMagenta: '#d4bfff',
    brightCyan:    '#95e6cb', brightWhite:   '#ffffff',
  };

  // localStorage keys — per-browser, not per-session.
  const LS = {
    visible: 'terminal_visible',
    height: 'terminal_height',
    sessionName: 'terminal_session_name',
    fontSize: 'terminal_font_size',
  };

  // ── Tiny DOM helper ─────────────────────────────────────────────
  function el(tag, attrs = {}, children = []) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'text') n.textContent = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k === 'class') n.className = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const c of children) {
      if (c != null) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return n;
  }

  // ── Strict-order lazy load of xterm + addons ────────────────────
  let _bundleReady = null;
  function loadXtermBundle() {
    if (_bundleReady) return _bundleReady;
    // CSS first (non-blocking, but we still wait on load)
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = `${XTERM_BASE}/xterm.css?v=${XTERM_VER}`;
    document.head.appendChild(css);

    const scripts = [
      'xterm.js',
      'addon-fit.js',
      'addon-unicode-graphemes.js',
      'addon-web-links.js',
      'addon-search.js',
      'addon-clipboard.js',
      'addon-serialize.js',
      'addon-webgl.js',
    ];
    _bundleReady = scripts.reduce((chain, fname) => chain.then(() =>
      new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = `${XTERM_BASE}/${fname}?v=${XTERM_VER}`;
        s.async = false;   // preserve order
        s.onload = () => resolve();
        s.onerror = () => reject(new Error(`failed to load ${fname}`));
        document.head.appendChild(s);
      })
    ), Promise.resolve());
    return _bundleReady;
  }

  // ── Panel class ─────────────────────────────────────────────────
  class TerminalPanel {
    constructor({ apiBase, wsBase } = {}) {
      this.apiBase = apiBase || window.location.origin;
      this.wsBase = wsBase || this.apiBase.replace(/^http/, 'ws');

      this._mounted = false;
      this._visible = false;
      this._term = null;
      this._fit = null;
      this._webgl = null;
      this._search = null;
      this._serialize = null;

      this._ws = null;
      this._attached = false;
      this._reconnectAttempts = 0;
      this._reconnectTimer = null;
      this._paused = false;

      // Cumulative flow-control counters
      this._bytesRendered = 0;
      this._bytesRenderedAcked = 0;
      this._ackScheduled = false;

      // For status-line display (updated from server status frames + our side)
      this._bytesIn = 0;
      this._bytesOutSent = 0;
      this._connectionState = 'idle';  // idle | connecting | authorizing | live | paused | reconnecting | offline

      this._tmuxSession = localStorage.getItem(LS.sessionName) || 'scribe';
      this._resizeObs = null;
      this._resizeTimer = null;
      this._statusTimer = null;
    }

    // ── Lifecycle ────────────────────────────────────────────────
    async mount() {
      if (this._mounted) return;
      if (this._mountPromise) return this._mountPromise;
      this._mountPromise = (async () => {
        await loadXtermBundle();
        this._buildDom();
        this._initTerminal();
        this._installResizeObserver();
        this._installShortcuts();
        this._mounted = true;
      })();
      try { await this._mountPromise; }
      finally { this._mountPromise = null; }
      // NOTE: do NOT auto-show from inside mount(). The caller (scribe-app.js)
      // owns the auto-restore decision. Auto-showing here caused show() to run
      // twice in quick succession, which raced _connect() — two WebSockets
      // attached to the same tmux session → shell echo reached xterm twice
      // → every keystroke rendered doubled.
    }

    toggle() { (this._visible ? this.hide : this.show).call(this); }

    show() {
      if (!this._mounted) { this.mount().then(() => this.show()); return; }
      this._visible = true;
      this._root.hidden = false;
      localStorage.setItem(LS.visible, '1');
      requestAnimationFrame(() => {
        if (this._fit) { try { this._fit.fit(); } catch {} }
        this._term && this._term.focus();
      });
      // An explicit show() is a user intent — reset the reconnect budget
      // so a prior "gave up" state doesn't permanently wedge the panel.
      this._reconnectAttempts = 0;
      if (!this._ws) this._connect();
      if (!this._statusTimer) {
        this._statusTimer = setInterval(() => this._paintStatus(), STATUS_INTERVAL_MS);
      }
    }

    /** Explicit user-initiated retry. Clears state and kicks a fresh connect. */
    reconnectNow() {
      clearTimeout(this._reconnectTimer);
      this._reconnectAttempts = 0;
      this._closeWs();
      if (this._term) {
        try { this._term.write('\r\n\x1b[38;5;214m[ reconnecting… ]\x1b[0m\r\n'); } catch {}
      }
      this._connect();
    }

    /** Kill the backing tmux session AND hide the panel.
     *
     *  Wired to the X button. Opening the panel again creates a fresh
     *  session via ``new-session -A`` on the next attach. Alt+X is the
     *  soft-hide variant that leaves the session running.
     */
    async closeAndKill() {
      clearTimeout(this._reconnectTimer);
      const sess = this._tmuxSession || 'scribe';
      try {
        await fetch(`${this.apiBase}/api/terminal/sessions/${encodeURIComponent(sess)}/reset`, {
          method: 'POST', credentials: 'include',
        });
      } catch (e) {
        console.warn('[terminal] reset failed', e);
      }
      this._closeWs();
      this._reconnectAttempts = 0;
      this.hide();
    }

    hide() {
      this._visible = false;
      if (this._root) this._root.hidden = true;
      localStorage.setItem(LS.visible, '0');
    }

    focus() { this._term && this._term.focus(); }

    destroy() {
      clearInterval(this._statusTimer);
      clearTimeout(this._reconnectTimer);
      this._closeWs();
      try { this._webgl && this._webgl.dispose(); } catch {}
      try { this._term && this._term.dispose(); } catch {}
      if (this._root && this._root.parentNode) this._root.parentNode.removeChild(this._root);
      this._mounted = false;
    }

    setTmuxSession(name) {
      if (!/^[a-zA-Z0-9_\-]{1,32}$/.test(name)) return false;
      if (name === this._tmuxSession) return true;
      this._tmuxSession = name;
      localStorage.setItem(LS.sessionName, name);
      this._sessionPickBtn.textContent = name;
      // Reconnect with new session name.
      this._closeWs();
      this._connect();
      return true;
    }

    // ── DOM ──────────────────────────────────────────────────────
    _buildDom() {
      // X kills the current tmux session AND hides the panel. The
      // rationale is user intent: X on a panel means "I'm done with
      // that thing", not "minimize it while it keeps running". Alt
      // + X is the soft-hide (keep session alive).
      const closeBtn = el('button', {
        class: 'term-close',
        title: 'Close session (Alt+X to hide without killing)',
        text: '×',
        onclick: (ev) => {
          if (ev.altKey) { this.hide(); return; }
          this.closeAndKill();
        },
      });

      this._sessionPickBtn = el('button', {
        class: 'term-session-pick',
        title: 'Switch tmux session',
        text: this._tmuxSession,
        onclick: () => this._promptSession(),
      });

      this._hostSpan = el('span', { class: 'term-status-host', text: this._deriveHost() });
      this._dimsSpan = el('span', { class: 'term-dims', text: '— × —' });
      this._bytesSpan = el('span', { class: 'term-bytes', text: '↓0 ↑0' });
      this._dot = el('span', { class: 'term-dot', 'aria-hidden': 'true' });
      this._dotLabel = el('span', { class: 'term-dot-label', text: 'idle' });

      const searchBtn = el('button', {
        class: 'term-tool',
        title: 'Search (Ctrl+Shift+F)',
        text: '⌕',
        onclick: () => this._toggleSearch(),
      });

      const saveBtn = el('button', {
        class: 'term-tool',
        title: 'Save scrollback (Ctrl+Shift+K)',
        text: '↓',
        onclick: () => this._downloadScrollback(),
      });

      const zoomOutBtn = el('button', {
        class: 'term-tool term-zoom-out',
        title: 'Decrease font (Ctrl + -)',
        text: 'A−',
        onclick: () => this.adjustFontSize(-1),
      });
      const zoomInBtn = el('button', {
        class: 'term-tool term-zoom-in',
        title: 'Increase font (Ctrl + =)',
        text: 'A+',
        onclick: () => this.adjustFontSize(1),
      });

      this._status = el('div', { class: 'term-status', role: 'status', 'data-state': 'idle' }, [
        this._hostSpan,
        el('span', { class: 'term-sep', 'aria-hidden': 'true', text: '·' }),
        this._sessionPickBtn,
        el('span', { class: 'term-sep', 'aria-hidden': 'true', text: '·' }),
        this._dimsSpan,
        el('span', { class: 'term-sep', 'aria-hidden': 'true', text: '·' }),
        this._bytesSpan,
        el('span', { class: 'term-spacer' }),
        zoomOutBtn, zoomInBtn, searchBtn, saveBtn,
        el('span', { class: 'term-dotwrap' }, [this._dot, this._dotLabel]),
        closeBtn,
      ]);

      // Search overlay (hidden until toggled)
      this._searchInput = el('input', {
        type: 'text',
        class: 'term-search-input',
        placeholder: 'find…',
        spellcheck: 'false',
        oninput: () => this._doSearch(),
        onkeydown: (e) => {
          if (e.key === 'Enter') { this._doSearch(e.shiftKey ? -1 : 1); e.preventDefault(); }
          if (e.key === 'Escape') { this._toggleSearch(false); e.preventDefault(); }
        },
      });
      this._searchBar = el('div', { class: 'term-search', 'data-open': '0' }, [
        this._searchInput,
        el('button', { class: 'term-tool', text: '↑', title: 'Previous', onclick: () => this._doSearch(-1) }),
        el('button', { class: 'term-tool', text: '↓', title: 'Next', onclick: () => this._doSearch(1) }),
        el('button', { class: 'term-tool', text: '×', title: 'Close', onclick: () => this._toggleSearch(false) }),
      ]);

      this._mount = el('div', { class: 'term-mount' });
      this._body = el('div', { class: 'term-body' }, [this._mount, this._searchBar]);

      // Auth overlay for the 403-not-authorized case.
      this._authOverlay = el('div', { class: 'term-auth-overlay', hidden: '' }, [
        el('div', { class: 'term-auth-card' }, [
          el('div', { class: 'term-auth-eyebrow', text: 'Terminal · authorization required' }),
          el('div', { class: 'term-auth-head', text: 'This browser has not been authorized for the embedded terminal.' }),
          el('div', { class: 'term-auth-body', text: 'Paste the admin secret once to mint a 7-day signed cookie for this device.' }),
          el('a', {
            class: 'term-auth-cta',
            href: '/admin/bootstrap',
            target: '_blank',
            rel: 'noopener',
            text: 'Open bootstrap →',
          }),
        ]),
      ]);
      this._body.appendChild(this._authOverlay);

      // Offline overlay — shown when we've exhausted reconnect attempts.
      // Gives the user a manual retry button so they don't have to reload
      // the whole page.
      this._offlineOverlay = el('div', { class: 'term-offline-overlay', hidden: '' }, [
        el('div', { class: 'term-offline-card' }, [
          el('div', { class: 'term-offline-eyebrow', text: 'Terminal · disconnected' }),
          this._offlineHead = el('div', { class: 'term-offline-head', text: 'Connection lost.' }),
          el('div', { class: 'term-offline-body', text: 'The shell dropped or tmux exited. Click to start a fresh session.' }),
          el('button', {
            class: 'term-offline-cta',
            type: 'button',
            text: 'Reconnect',
            onclick: () => this.reconnectNow(),
          }),
        ]),
      ]);
      this._body.appendChild(this._offlineOverlay);

      const handleGrip = el('div', { class: 'term-resize-grip' });
      this._resizeHandle = el('div', {
        class: 'term-resize-handle',
        title: 'Drag to resize',
        onmousedown: (e) => this._onResizeStart(e),
      }, [handleGrip]);

      this._root = el('div', { class: 'terminal-panel', hidden: '' }, [
        this._resizeHandle,
        this._status,
        this._body,
      ]);
      const savedH = parseInt(localStorage.getItem(LS.height) || '320', 10);
      this._body.style.height = `${Math.max(140, Math.min(savedH, Math.round(window.innerHeight * 0.7)))}px`;

      // Insertion into the DOM is handled by the layout renderer (see
      // popout-panel-registry). We stash the root element and let the
      // caller (or the registry) pick the mount point. Legacy fallback:
      // if there's no registry loaded (e.g. a standalone test harness),
      // fall back to the old auto-insert-after-main behavior.
      if (!window.PopoutPanelRegistry) {
        const main = document.querySelector('main');
        const slides = document.querySelector('.popout-slides');
        const reference = slides ? slides.nextSibling : (main ? main.nextSibling : null);
        if (reference) reference.parentNode.insertBefore(this._root, reference);
        else document.body.appendChild(this._root);
      }
    }

    _initTerminal() {
      const FONT_SIZE = this._clampFontSize(
        parseInt(localStorage.getItem(LS.fontSize) || String(FONT_SIZE_DEFAULT), 10)
      );
      this._fontSize = FONT_SIZE;
      const Terminal = window.Terminal;
      const term = new Terminal({
        fontFamily: "'JetBrains Mono', 'SF Mono', Menlo, ui-monospace, monospace",
        fontSize: FONT_SIZE,
        lineHeight: 1.2,
        cursorBlink: true,
        cursorStyle: 'block',
        scrollback: 10000,
        allowProposedApi: true,
        macOptionIsMeta: true,
        convertEol: false,
        theme: THEME_GRAPHITE,
      });
      this._term = term;

      this._fit = new window.FitAddon.FitAddon();
      term.loadAddon(this._fit);

      if (window.Unicode11Addon && window.Unicode11Addon.Unicode11Addon) {
        term.loadAddon(new window.Unicode11Addon.Unicode11Addon());
      } else if (window.UnicodeGraphemesAddon && window.UnicodeGraphemesAddon.UnicodeGraphemesAddon) {
        term.loadAddon(new window.UnicodeGraphemesAddon.UnicodeGraphemesAddon());
      }

      if (window.WebLinksAddon) {
        term.loadAddon(new window.WebLinksAddon.WebLinksAddon((ev, uri) => {
          window.open(uri, '_blank', 'noopener,noreferrer');
        }));
      }

      if (window.SearchAddon) {
        this._search = new window.SearchAddon.SearchAddon();
        term.loadAddon(this._search);
      }
      if (window.ClipboardAddon) term.loadAddon(new window.ClipboardAddon.ClipboardAddon());
      if (window.SerializeAddon) {
        this._serialize = new window.SerializeAddon.SerializeAddon();
        term.loadAddon(this._serialize);
      }

      term.open(this._mount);
      try { term.unicode.activeVersion = '11'; } catch {}

      // WebGL goes LAST — needs DOM.
      if (window.WebglAddon) {
        try {
          this._webgl = new window.WebglAddon.WebglAddon();
          this._webgl.onContextLoss(() => {
            try { this._webgl.dispose(); } catch {}
            this._webgl = null;
          });
          term.loadAddon(this._webgl);
        } catch (e) {
          console.warn('WebGL renderer unavailable — falling back to canvas', e);
        }
      }

      try { this._fit.fit(); } catch {}

      // Wire up input → WS
      term.onData((data) => this._sendStdin(data));
      term.attachCustomKeyEventHandler((ev) => {
        if (ev.type !== 'keydown') return true;
        // Ctrl+=/Ctrl++/Ctrl+-/Ctrl+0 — terminal font zoom. These are
        // ALSO the browser's page-zoom shortcuts, so we have to call
        // preventDefault() ourselves — xterm's handler only prevents
        // default when it CONSUMES the key, and we're returning false
        // to tell it "don't forward this to the PTY". Without the
        // explicit preventDefault, Ctrl+0 would reset browser zoom
        // (blowing away window-scoped state like _panel).
        if (ev.ctrlKey && !ev.shiftKey && !ev.altKey && !ev.metaKey) {
          if (ev.key === '=' || ev.key === '+') {
            ev.preventDefault();
            this.adjustFontSize(1);
            return false;
          }
          if (ev.key === '-' || ev.key === '_') {
            ev.preventDefault();
            this.adjustFontSize(-1);
            return false;
          }
          if (ev.key === '0') {
            ev.preventDefault();
            this.setFontSize(FONT_SIZE_DEFAULT);
            return false;
          }
        }
        if (!ev.ctrlKey || !ev.shiftKey) return true;
        if (ev.key === 'C' || ev.key === 'c') return false;   // let clipboard addon copy
        if (ev.key === 'V' || ev.key === 'v') return false;   // paste
        if (ev.key === 'F' || ev.key === 'f') { this._toggleSearch(true); return false; }
        if (ev.key === 'K' || ev.key === 'k') { this._downloadScrollback(); return false; }
        return true;
      });
    }

    // ── Font zoom ────────────────────────────────────────────────
    _clampFontSize(n) {
      const v = Number.isFinite(n) ? Math.round(n) : FONT_SIZE_DEFAULT;
      return Math.max(FONT_SIZE_MIN, Math.min(FONT_SIZE_MAX, v));
    }
    setFontSize(px) {
      const next = this._clampFontSize(px);
      if (next === this._fontSize) return next;
      this._fontSize = next;
      localStorage.setItem(LS.fontSize, String(next));
      if (this._term) {
        try { this._term.options.fontSize = next; } catch {}
        // Run TWO refits: one on the next frame (so xterm's renderer
        // has a chance to re-measure the new cell width/height) and a
        // follow-up after the debounce in case the first ran before
        // the web-font finished hydrating. Without the rAF wait the
        // fit addon occasionally computed cols/rows against stale cell
        // metrics.
        requestAnimationFrame(() => {
          try {
            if (this._fit) this._fit.fit();
          } catch {}
          this._scheduleRefit();
        });
      }
      // Give the settings panel (or anyone else) a chance to react.
      try { window.dispatchEvent(new CustomEvent('terminal:font-size', { detail: { size: next } })); } catch {}
      return next;
    }
    adjustFontSize(delta) { return this.setFontSize((this._fontSize || FONT_SIZE_DEFAULT) + delta); }
    getFontSize() { return this._fontSize || FONT_SIZE_DEFAULT; }

    _installResizeObserver() {
      this._resizeObs = new ResizeObserver(() => this._scheduleRefit());
      this._resizeObs.observe(this._body);
      window.addEventListener('resize', () => this._scheduleRefit());
    }

    _installShortcuts() {
      // No global listener here — the toggle (Ctrl+Shift+T) is owned by
      // scribe-app.js because xterm has focus when the panel is open.
    }

    _scheduleRefit() {
      clearTimeout(this._resizeTimer);
      this._resizeTimer = setTimeout(() => {
        if (!this._fit) return;
        try {
          this._fit.fit();
          const dims = this._term ? { cols: this._term.cols, rows: this._term.rows } : null;
          if (dims) {
            this._dimsSpan.textContent = `${dims.cols}×${dims.rows}`;
            this._sendText({ type: 'resize', cols: dims.cols, rows: dims.rows });
          }
        } catch {}
      }, RESIZE_DEBOUNCE_MS);
    }

    // ── Resize drag handle ───────────────────────────────────────
    _onResizeStart(e) {
      if (e.button !== 0) return;
      e.preventDefault();
      const startY = e.clientY;
      const startH = this._body.getBoundingClientRect().height;
      const minH = 140;
      const maxH = Math.round(window.innerHeight * 0.75);
      document.body.style.cursor = 'ns-resize';
      const onMove = (ev) => {
        const dy = startY - ev.clientY; // drag up → grow
        const h = Math.max(minH, Math.min(maxH, startH + dy));
        this._body.style.height = `${h}px`;
      };
      const onUp = () => {
        document.body.style.cursor = '';
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        localStorage.setItem(LS.height, `${Math.round(this._body.getBoundingClientRect().height)}`);
        this._scheduleRefit();
      };
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    }

    // ── Connection lifecycle ─────────────────────────────────────
    async _connect() {
      // Guard against concurrent re-entry. A prior version returned early
      // only when `_ws` was already set, but `_ws` isn't set until after an
      // async ticket fetch — so back-to-back show() calls would both race
      // past the guard and spin up two WebSockets attached to the same tmux
      // session. tmux then mirrored shell echo to both clients, which the
      // xterm rendered as doubled keystrokes.
      if (this._ws || this._connectInflight) return this._connectInflight;
      this._connectInflight = this._doConnect();
      try { await this._connectInflight; }
      finally { this._connectInflight = null; }
    }

    async _doConnect() {
      this._setState('connecting');
      let ticket;
      try {
        const r = await fetch(`${this.apiBase}/api/terminal/ticket`, {
          method: 'POST', credentials: 'include',
        });
        if (r.status === 403) {
          this._showAuthPrompt();
          this._setState('offline', 'not authorized');
          return;
        }
        if (!r.ok) throw new Error(`ticket: ${r.status}`);
        const body = await r.json();
        ticket = body.ticket;
      } catch (e) {
        console.warn('[terminal] ticket mint failed', e);
        this._scheduleReconnect();
        return;
      }
      this._hideAuthPrompt();

      const wsUrl = `${this.wsBase}/api/ws/terminal`;
      let ws;
      try { ws = new WebSocket(wsUrl); }
      catch (e) { console.warn('[terminal] ws construct failed', e); this._scheduleReconnect(); return; }
      this._ws = ws;
      ws.binaryType = 'arraybuffer';
      ws.addEventListener('open', () => this._onWsOpen(ticket));
      ws.addEventListener('message', (e) => this._onWsMessage(e));
      ws.addEventListener('close', (e) => this._onWsClose(e));
      ws.addEventListener('error', () => {/* handled via close */});
    }

    _onWsOpen(ticket) {
      const attach = {
        type: 'attach',
        ticket,
        tmux_session: this._tmuxSession,
        cols: this._term.cols,
        rows: this._term.rows,
        term: 'xterm-256color',
      };
      this._ws.send(JSON.stringify(attach));
    }

    _onWsMessage(e) {
      const data = e.data;
      if (typeof data === 'string') return this._onTextFrame(data);
      // Binary output frame. data is ArrayBuffer.
      const buf = new Uint8Array(data);
      if (buf.length === 0) return;
      if (buf[0] === PREFIX_OUTPUT_CODE) {
        const payload = buf.subarray(1);
        // term.write accepts a completion callback; use it to count rendered bytes.
        this._term.write(payload, () => {
          this._bytesRendered += payload.length;
          this._scheduleAck();
        });
      }
    }

    _onWsClose(e) {
      this._ws = null;
      this._attached = false;
      if (e && e.code === 4401) {
        this._showAuthPrompt();
        this._setState('offline', 'not authorized');
        return;
      }
      if (!this._visible) return;  // don't reconnect if the user hid the panel
      this._setState('reconnecting');
      this._scheduleReconnect();
    }

    _scheduleReconnect() {
      if (this._reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        this._setState('offline', 'gave up');
        this._showOfflinePrompt();
        return;
      }
      const delay = Math.min(10000, 1000 * (2 ** this._reconnectAttempts));
      this._reconnectAttempts += 1;
      this._setState('reconnecting', `retry ${Math.round(delay / 1000)}s`);
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = setTimeout(() => this._connect(), delay);
    }

    _closeWs() {
      if (this._ws) {
        try { this._ws.close(1000, 'client'); } catch {}
        this._ws = null;
      }
    }

    _onTextFrame(text) {
      let msg;
      try { msg = JSON.parse(text); } catch { return; }
      if (!msg || typeof msg !== 'object') return;
      switch (msg.type) {
        case 'attached':
          this._reconnectAttempts = 0;
          this._attached = true;
          this._setState('live');
          this._hideOfflinePrompt();
          this._bytesRendered = 0;
          this._bytesRenderedAcked = 0;
          this._dimsSpan.textContent = `${msg.cols}×${msg.rows}`;
          // Send a resize right away to ensure server matches actual client size.
          if (this._term) {
            if (this._term.cols !== msg.cols || this._term.rows !== msg.rows) {
              this._sendText({ type: 'resize', cols: this._term.cols, rows: this._term.rows });
            }
          }
          break;
        case 'status':
          this._bytesIn = msg.bytes_in || 0;
          this._bytesOutSent = msg.bytes_sent_total || 0;
          this._paused = !!msg.paused;
          this._setState(this._paused ? 'paused' : 'live');
          break;
        case 'error':
          if (msg.code === 'auth') this._showAuthPrompt();
          this._setState('offline', msg.message || msg.code || 'error');
          break;
        case 'bye':
          // `pty_exited` is the "user typed `exit` in the shell" path —
          // that's intentional, not a failure. Reset the retry budget so
          // the next reconnect gets a clean 5 attempts and respawns tmux.
          if (msg.reason === 'pty_exited') {
            this._reconnectAttempts = 0;
            this._lastBye = 'pty_exited';
            this._setState('reconnecting', 'shell exited — respawning');
          } else {
            this._setState('offline', msg.reason || 'bye');
          }
          break;
        case 'pong':
          break;
      }
    }

    _showOfflinePrompt() {
      if (!this._offlineOverlay) return;
      this._offlineOverlay.hidden = false;
      if (this._offlineHead) {
        this._offlineHead.textContent =
          this._lastBye === 'pty_exited'
            ? 'Shell exited — click to start a fresh one.'
            : 'Connection lost.';
      }
    }
    _hideOfflinePrompt() {
      if (!this._offlineOverlay) return;
      this._offlineOverlay.hidden = true;
    }

    // ── Outbound: stdin + resize + ack ───────────────────────────
    _sendStdin(data) {
      if (!this._ws || this._ws.readyState !== 1) return;
      const enc = new TextEncoder().encode(data);
      const prefix = new Uint8Array([PREFIX_INPUT.charCodeAt(0)]);
      // Chunk oversize payloads so one paste can't trip the server's 256K cap.
      let offset = 0;
      while (offset < enc.length) {
        const slice = enc.subarray(offset, offset + PASTE_CHUNK_BYTES);
        const frame = new Uint8Array(1 + slice.length);
        frame.set(prefix, 0);
        frame.set(slice, 1);
        try { this._ws.send(frame); } catch (e) { console.warn('[terminal] send fail', e); return; }
        offset += slice.length;
      }
    }

    _sendText(obj) {
      if (!this._ws || this._ws.readyState !== 1) return;
      try { this._ws.send(JSON.stringify(obj)); } catch {}
    }

    _scheduleAck() {
      if (this._ackScheduled) return;
      this._ackScheduled = true;
      requestAnimationFrame(() => {
        this._ackScheduled = false;
        const unacked = this._bytesRendered - this._bytesRenderedAcked;
        if (unacked < ACK_MIN_BYTES && !this._paused) return;
        this._sendText({ type: 'ack', bytes_total: this._bytesRendered });
        this._bytesRenderedAcked = this._bytesRendered;
      });
    }

    // ── Auth UX ──────────────────────────────────────────────────
    _showAuthPrompt() {
      if (!this._authOverlay) return;
      this._authOverlay.hidden = false;
    }
    _hideAuthPrompt() {
      if (!this._authOverlay) return;
      this._authOverlay.hidden = true;
    }

    // ── Search ───────────────────────────────────────────────────
    _toggleSearch(open) {
      const shouldOpen = typeof open === 'boolean' ? open : this._searchBar.dataset.open !== '1';
      this._searchBar.dataset.open = shouldOpen ? '1' : '0';
      if (shouldOpen) {
        this._searchInput.focus();
        this._searchInput.select();
      } else {
        this._term && this._term.focus();
      }
    }
    _doSearch(direction) {
      if (!this._search) return;
      const needle = this._searchInput.value;
      if (!needle) return;
      if (direction === -1) this._search.findPrevious(needle, { caseSensitive: false, wholeWord: false });
      else this._search.findNext(needle, { caseSensitive: false, wholeWord: false });
    }

    // ── Scrollback export ────────────────────────────────────────
    _downloadScrollback() {
      if (!this._serialize) return;
      const text = this._serialize.serialize();
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      a.download = `scribe-terminal-${this._tmuxSession}-${stamp}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    }

    // ── Session picker ───────────────────────────────────────────
    _promptSession() {
      const name = window.prompt(
        'tmux session name\n(letters, digits, _ or -, max 32 chars)',
        this._tmuxSession
      );
      if (!name) return;
      if (!this.setTmuxSession(name)) {
        window.alert(`invalid session name: ${name}`);
      }
    }

    // ── Status line ──────────────────────────────────────────────
    _setState(state, note) {
      this._connectionState = state;
      if (this._status) {
        this._status.dataset.state = state;
      }
      if (this._dotLabel) {
        this._dotLabel.textContent = note ? `${state} · ${note}` : state;
      }
      this._paintStatus();
    }

    _paintStatus() {
      if (!this._bytesSpan) return;
      this._bytesSpan.textContent = `↓${_fmtBytes(this._bytesOutSent)} ↑${_fmtBytes(this._bytesIn)}`;
      if (this._term && this._dimsSpan) {
        const d = `${this._term.cols}×${this._term.rows}`;
        if (this._dimsSpan.textContent !== d) this._dimsSpan.textContent = d;
      }
    }

    _deriveHost() {
      const h = window.location.host || 'localhost';
      return `scribe@${h.replace(/:\d+$/, '')}`;
    }
  }

  function _fmtBytes(n) {
    if (!n || n < 0) return '0';
    if (n < 1024) return String(n);
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}k`;
    return `${(n / 1024 / 1024).toFixed(1)}M`;
  }

  window.TerminalPanel = TerminalPanel;
})();
