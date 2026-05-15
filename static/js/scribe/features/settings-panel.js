// Meeting Scribe — Settings panel (gear icon → slide-over).
//
// Hosts the appliance-wide configuration surface: WiFi mode + admin SSID +
// passphrase + regulatory domain, timezone, TTS voice mode, per-output
// (admin / room) TTS language, terminal font, plus the read-only admin
// password + guest PIN reveal flow.
//
// Dependency surface (all named imports):
//   state.js          — `state`, `store`, `audio` singletons
//   modal-system.js   — `confirmDialog` (used for the kiosk-mode confirm)
//
// Window globals it touches:
//   `window.dispatchEvent` for the cross-tab terminal-font event
//   `window._terminalPanel` (reads only — the lazy-loaded popout terminal panel
//   uses the same `setFontSize` setter so admin + popout stay in lockstep)
//   `window.isSecureContext` (read) for the credential clipboard fallback

import { audio, state, store } from "../state.js";
import { confirmDialog } from "./modal-system.js";

const API = "";

export function bootSettingsPanel() {
  const panel = document.getElementById('settings-panel');
  const backdrop = document.getElementById('settings-backdrop');
  const openBtn = document.getElementById('btn-settings');
  const closeBtn = document.getElementById('btn-settings-close');
  const saveBtn = document.getElementById('btn-settings-save');
  const wifiModeSelect = document.getElementById('setting-wifi-mode');
  const adminSsidInput = document.getElementById('setting-admin-ssid');
  const adminPasswordInput = document.getElementById('setting-admin-password');
  const adminPwToggle = document.getElementById('btn-toggle-admin-pw');
  const wifiLiveStatus = document.getElementById('wifi-live-status');
  const regdomainSelect = document.getElementById('setting-wifi-regdomain');
  const regdomainLive = document.getElementById('regdomain-live');
  const timezoneSelect = document.getElementById('setting-timezone');
  const voiceModeSelect = document.getElementById('setting-tts-voice-mode');
  const adminTtsLanguageSelect = document.getElementById('audio-admin-language-select');
  const roomTtsLanguageSelect = document.getElementById('audio-room-language-select');
  const status = document.getElementById('settings-status');
  if (!panel || !openBtn || !regdomainSelect || !timezoneSelect) return;

  // Password show/hide toggle
  if (adminPwToggle && adminPasswordInput) {
    adminPwToggle.addEventListener('click', () => {
      const showing = adminPasswordInput.type === 'text';
      adminPasswordInput.type = showing ? 'password' : 'text';
      adminPwToggle.textContent = showing ? 'Show' : 'Hide';
    });
  }

  let loaded = false;
  let lastLoadedData = null;
  let wifiModeDirty = false;

  const setStatus = (msg, cls = '') => {
    status.textContent = msg || '';
    status.className = 'settings-status' + (cls ? ' ' + cls : '');
  };

  const populate = (data) => {
    lastLoadedData = data;

    // Device pin + appliance id (read-only). Pin is the 4 digits that
    // appear in the SSID suffix and as the guest PIN; combined with
    // ``DellMeetingAdmin`` it yields the admin password. Full appliance
    // id (16 hex) shown for fleet correlation (no truncation).
    const devPinLine = document.querySelector('.device-pin-line');
    if (devPinLine) {
      devPinLine.textContent = data.appliance_pin
        ? `${data.appliance_pin}  (admin password: DellMeetingAdmin${data.appliance_pin})`
        : '(not minted yet — run meeting-scribe setup)';
    }
    const devIdLine = document.querySelector('.device-id-line');
    if (devIdLine) {
      devIdLine.textContent = data.appliance_id || '(not minted yet)';
    }

    // WiFi mode select. This is the operator's desired persisted mode,
    // not the transient live AP state. Live can briefly report "off"
    // during AP rotation/restart; reflecting that into the form caused
    // unrelated settings saves to persist wifi_mode=off.
    const liveMode = data.wifi_mode_live;
    const persistedMode = data.wifi_mode;
    // The dropdown only offers off/meeting/admin; persisted values
    // outside that set fall back to admin while live mismatch remains
    // visible in the status line below.
    const optionCodes = (data.wifi_mode_options || []).map(o => o.code);
    const dropdownMode = optionCodes.includes(persistedMode) ? persistedMode : 'admin';
    if (wifiModeSelect) {
      wifiModeSelect.innerHTML = '';
      for (const opt of data.wifi_mode_options || []) {
        const el = document.createElement('option');
        el.value = opt.code;
        el.textContent = opt.name;
        if (opt.code === dropdownMode) el.selected = true;
        wifiModeSelect.appendChild(el);
      }
      wifiModeDirty = false;
    }

    // Admin SSID
    if (adminSsidInput) adminSsidInput.value = data.admin_ssid || '';

    // Admin password — write-only, show placeholder if set
    if (adminPasswordInput) {
      adminPasswordInput.value = '';
      adminPasswordInput.placeholder = data.admin_password_set
        ? 'Leave blank to keep current'
        : 'Set a password (8-63 chars)';
    }

    // WiFi live status line
    if (wifiLiveStatus) {
      if (data.wifi_active && data.wifi_ssid) {
        const sec = data.wifi_security;
        const km = sec ? sec.key_mgmt || '?' : '?';
        // The dropdown only offers off/meeting/admin. "setup" is a
        // system-managed first-touch state — when the box is in
        // setup mode the dropdown can't represent it, so call it
        // out explicitly here. Mismatch flag stays so the styling
        // signals "this isn't what you'd expect."
        const SELECTABLE_MODES = ['off', 'meeting', 'admin'];
        let modeNote = '';
        let mismatch = false;
        if (liveMode === 'setup') {
          modeNote = '  ·  mode: first-touch setup (pick a mode below to leave it)';
          mismatch = true;
        } else if (liveMode && persistedMode && liveMode !== persistedMode) {
          modeNote = `  ·  saved: ${persistedMode}`;
          mismatch = true;
        } else if (liveMode) {
          modeNote = `  ·  mode: ${liveMode}`;
          mismatch = !SELECTABLE_MODES.includes(liveMode);
        }
        wifiLiveStatus.textContent = `Live: ${data.wifi_ssid} (${km})${modeNote}`;
        wifiLiveStatus.classList.toggle('mismatch', mismatch);
      } else {
        wifiLiveStatus.textContent = 'WiFi AP is off';
        wifiLiveStatus.classList.add('mismatch');
      }
    }

    regdomainSelect.innerHTML = '';
    for (const opt of data.wifi_regdomain_options || []) {
      const el = document.createElement('option');
      el.value = opt.code;
      el.textContent = `${opt.code} · ${opt.name}`;
      if (opt.code === data.wifi_regdomain) el.selected = true;
      regdomainSelect.appendChild(el);
    }

    timezoneSelect.innerHTML = '';
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = '— Server local time —';
    if (!data.timezone) blank.selected = true;
    timezoneSelect.appendChild(blank);
    for (const tz of data.timezone_options || []) {
      const el = document.createElement('option');
      el.value = tz;
      el.textContent = tz;
      if (tz === data.timezone) el.selected = true;
      timezoneSelect.appendChild(el);
    }


    if (voiceModeSelect) {
      voiceModeSelect.innerHTML = '';
      for (const opt of data.tts_voice_mode_options || []) {
        const el = document.createElement('option');
        el.value = opt.code;
        el.textContent = opt.name;
        if (opt.code === data.tts_voice_mode) el.selected = true;
        voiceModeSelect.appendChild(el);
      }
    }

    if (adminTtsLanguageSelect) {
      adminTtsLanguageSelect.innerHTML = '';
      for (const opt of data.local_sink_language_options || []) {
        const el = document.createElement('option');
        el.value = opt.code;
        el.textContent = opt.name;
        if (opt.code === (data.admin_tts_language || data.local_sink_language)) el.selected = true;
        adminTtsLanguageSelect.appendChild(el);
      }
    }

    if (roomTtsLanguageSelect) {
      roomTtsLanguageSelect.innerHTML = '';
      const all = document.createElement('option');
      all.value = 'all';
      all.textContent = 'All translated languages';
      if ((data.room_tts_language || 'all') === 'all') all.selected = true;
      roomTtsLanguageSelect.appendChild(all);
      for (const opt of data.local_sink_language_options || []) {
        const el = document.createElement('option');
        el.value = opt.code;
        el.textContent = opt.name;
        if (opt.code === data.room_tts_language) el.selected = true;
        roomTtsLanguageSelect.appendChild(el);
      }
    }

    if (data.wifi_regdomain_current && data.wifi_regdomain_current !== data.wifi_regdomain) {
      regdomainLive.textContent =
        `Live radio: ${data.wifi_regdomain_current} — will switch to ${data.wifi_regdomain} on next hotspot rotation`;
      regdomainLive.classList.add('mismatch');
    } else if (data.wifi_regdomain_current) {
      regdomainLive.textContent = `Live radio: ${data.wifi_regdomain_current}`;
      regdomainLive.classList.remove('mismatch');
    } else {
      regdomainLive.textContent = '';
    }
  };

  const load = async () => {
    setStatus('Loading…');
    try {
      const resp = await fetch(`${API}/api/admin/settings`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      populate(data);
      loaded = true;
      setStatus('');
    } catch (err) {
      setStatus('Could not load settings', 'error');
    }
  };

  const save = async () => {
    if (!loaded || !lastLoadedData) return;
    const body = {
      wifi_regdomain: regdomainSelect.value,
      timezone: timezoneSelect.value,
    };
    if (voiceModeSelect && voiceModeSelect.value) {
      body.tts_voice_mode = voiceModeSelect.value;
    }
    if (adminTtsLanguageSelect && adminTtsLanguageSelect.value) {
      body.admin_tts_language = adminTtsLanguageSelect.value;
    }
    if (roomTtsLanguageSelect && roomTtsLanguageSelect.value) {
      body.room_tts_language = roomTtsLanguageSelect.value;
    }
    // WiFi mode + admin creds. Only send wifi_mode when the operator
    // explicitly changes it; otherwise transient live-state UI refreshes
    // must not alter AP mode during an unrelated settings save.
    if (wifiModeSelect && wifiModeDirty) body.wifi_mode = wifiModeSelect.value;
    if (adminSsidInput && adminSsidInput.value.trim()) {
      body.admin_ssid = adminSsidInput.value.trim();
    }
    if (adminPasswordInput && adminPasswordInput.value) {
      body.admin_password = adminPasswordInput.value;
    }

    // Warn if switching away from admin while likely connected over WiFi
    const oldMode = lastLoadedData.wifi_mode;
    const newMode = body.wifi_mode;
    if (wifiModeDirty && oldMode === 'admin' && newMode !== 'admin') {
      const ok = await confirmDialog('Switch WiFi mode?', 'Switching away from admin mode will disconnect you from WiFi.', 'Continue', false);
      if (!ok) return;
    }

    setStatus('Applying…');
    saveBtn.disabled = true;
    try {
      const resp = await fetch(`${API}/api/admin/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await resp.json().catch(() => ({}));

      // 202 = async WiFi mode switch in progress
      if (resp.status === 202) {
        setStatus(`Switching WiFi to ${data.wifi_mode || newMode}…`, 'ok');
        // Poll until live state matches desired, or timeout
        let attempts = 0;
        const poll = setInterval(async () => {
          attempts++;
          try {
            const pollResp = await fetch(`${API}/api/admin/settings`);
            if (pollResp.ok) {
              const pollData = await pollResp.json();
              populate(pollData);
              if (pollData.wifi_active === (newMode !== 'off') || attempts > 20) {
                clearInterval(poll);
                setStatus('WiFi mode switched', 'ok');
                setTimeout(() => setStatus(''), 3000);
              }
            }
          } catch { /* ignore poll errors */ }
          if (attempts > 20) {
            clearInterval(poll);
            setStatus('WiFi switch may still be in progress', 'error');
          }
        }, 2000);
        return;
      }

      if (!resp.ok) {
        setStatus(data.error || `HTTP ${resp.status}`, 'error');
        return;
      }
      populate(data);
      if (data.runtime_ok === false) {
        setStatus(
          `Saved, but 'iw reg set ${data.wifi_regdomain}' did not take effect`,
          'error',
        );
      } else {
        setStatus('Saved', 'ok');
        setTimeout(() => {
          if (status.textContent === 'Saved') setStatus('');
        }, 2500);
      }
    } catch {
      setStatus('Network error', 'error');
    } finally {
      saveBtn.disabled = false;
    }
  };

  // ─── Tab nav (Network/Audio/Display/Terminal/Credentials) ───
  // The settings panel body is split into 5 panes; one pane shows at a
  // time. The last non-Credentials tab persists across panel open/close
  // so operators don't re-navigate every time; Credentials is
  // deliberately excluded so reopening never auto-surfaces the
  // password / PIN reveal surface.
  const tabsNav = panel.querySelector('.settings-tabs-nav');
  const tabButtons = tabsNav
    ? Array.from(tabsNav.querySelectorAll('.settings-tab'))
    : [];
  const tabPanes = Array.from(panel.querySelectorAll('.settings-tab-pane'));
  const LAST_TAB_KEY = 'scribe.settings.last_tab';

  const activateTab = (tabName, { persist = true, focusTab = false } = {}) => {
    if (!tabName) return false;
    let activated = false;
    for (const btn of tabButtons) {
      const isActive = btn.dataset.tab === tabName;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
      btn.tabIndex = isActive ? 0 : -1;
      if (isActive) {
        activated = true;
        if (focusTab) btn.focus();
      }
    }
    if (!activated) return false;
    for (const pane of tabPanes) {
      const isActive = pane.dataset.tab === tabName;
      pane.classList.toggle('is-active', isActive);
      if (isActive) {
        pane.removeAttribute('hidden');
      } else {
        pane.setAttribute('hidden', '');
      }
    }
    if (persist && tabName !== 'credentials') {
      try { localStorage.setItem(LAST_TAB_KEY, tabName); } catch {}
    }
    if (tabName === 'credentials') {
      onCredentialsActivated();
    } else {
      // Leaving Credentials → mask the password and forget any cached
      // value so the next Reveal re-fetches against the server.
      maskAdminPassword();
    }
    return true;
  };

  // ─── Credentials tab (admin password + guest PIN) ───
  // Lazy-fetch with split-endpoint discipline so the reusable admin
  // password is never pulled into the JS layer without an explicit
  // Reveal click. PIN hydrates on tab open because it's already public
  // via the SSID suffix.
  const credAdminPwEl = document.getElementById('cred-admin-pw');
  const credAdminRevealBtn = document.getElementById('btn-cred-admin-reveal');
  const credAdminCopyBtn = document.getElementById('btn-cred-admin-copy');
  const credGuestPinEl = document.getElementById('cred-guest-pin');
  const credGuestCopyBtn = document.getElementById('btn-cred-pin-copy');
  const credStatusEl = document.getElementById('cred-status');
  const CRED_MASK = '••••••••••••••';
  let adminPasswordCache = null;
  let guestPinCache = null;
  let guestPinFetchInFlight = false;
  let credStatusClearTimer = null;

  const setCredStatus = (msg, tone = '') => {
    if (!credStatusEl) return;
    credStatusEl.textContent = msg || '';
    if (tone) {
      credStatusEl.setAttribute('data-tone', tone);
    } else {
      credStatusEl.removeAttribute('data-tone');
    }
    if (credStatusClearTimer) {
      clearTimeout(credStatusClearTimer);
      credStatusClearTimer = null;
    }
    if (msg) {
      credStatusClearTimer = setTimeout(() => {
        if (credStatusEl) {
          credStatusEl.textContent = '';
          credStatusEl.removeAttribute('data-tone');
        }
      }, 4000);
    }
  };

  const maskAdminPassword = () => {
    adminPasswordCache = null;
    if (credAdminPwEl) {
      credAdminPwEl.textContent = CRED_MASK;
      credAdminPwEl.setAttribute('data-hidden', '1');
      credAdminPwEl.setAttribute('aria-label', 'Admin login password (hidden)');
    }
    if (credAdminRevealBtn) {
      credAdminRevealBtn.textContent = 'Show';
      credAdminRevealBtn.setAttribute('aria-pressed', 'false');
    }
  };

  const revealAdminPassword = async () => {
    if (!credAdminPwEl || !credAdminRevealBtn) return;
    if (adminPasswordCache) {
      // Defensive: cache invariant is "if non-null we already revealed";
      // nothing to do.
      return;
    }
    credAdminRevealBtn.disabled = true;
    try {
      const resp = await fetch('/api/admin/admin-password', {
        cache: 'no-store',
        credentials: 'same-origin',
      });
      if (!resp.ok) {
        setCredStatus(`Reveal failed (${resp.status})`, 'err');
        return;
      }
      const data = await resp.json();
      adminPasswordCache = data.password;
      credAdminPwEl.textContent = data.password;
      credAdminPwEl.setAttribute('data-hidden', '0');
      credAdminPwEl.setAttribute('aria-label', 'Admin login password');
      credAdminRevealBtn.textContent = 'Hide';
      credAdminRevealBtn.setAttribute('aria-pressed', 'true');
    } catch (exc) {
      setCredStatus(`Reveal failed: ${exc.message || exc}`, 'err');
    } finally {
      credAdminRevealBtn.disabled = false;
    }
  };

  const hydrateGuestPin = async () => {
    if (!credGuestPinEl) return;
    if (guestPinCache) {
      credGuestPinEl.textContent = guestPinCache;
      return;
    }
    if (guestPinFetchInFlight) return;
    guestPinFetchInFlight = true;
    try {
      const resp = await fetch('/api/admin/guest-pin', {
        credentials: 'same-origin',
      });
      if (!resp.ok) {
        setCredStatus(`Guest PIN load failed (${resp.status})`, 'err');
        return;
      }
      const data = await resp.json();
      guestPinCache = data.pin;
      credGuestPinEl.textContent = data.pin || '––––';
    } catch (exc) {
      setCredStatus(`Guest PIN load failed: ${exc.message || exc}`, 'err');
    } finally {
      guestPinFetchInFlight = false;
    }
  };

  const copyToClipboard = async (value, label) => {
    if (!value) {
      setCredStatus(`${label} not available — reveal first`, 'err');
      return;
    }
    // Try the modern API first; falls back to execCommand for non-
    // secure contexts (HTTPS with an untrusted self-signed cert).
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(value);
        setCredStatus(`${label} copied`, 'ok');
        return;
      } catch {
        // Fall through to execCommand.
      }
    }
    const ta = document.createElement('textarea');
    ta.value = value;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '-9999px';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch {
      ok = false;
    }
    document.body.removeChild(ta);
    if (ok) {
      setCredStatus(`${label} copied`, 'ok');
    } else {
      setCredStatus(`Copy failed — select the value and copy manually`, 'err');
    }
  };

  function onCredentialsActivated() {
    hydrateGuestPin();
  }

  if (credAdminRevealBtn) {
    credAdminRevealBtn.addEventListener('click', () => {
      if (credAdminPwEl && credAdminPwEl.getAttribute('data-hidden') === '0') {
        maskAdminPassword();
      } else {
        revealAdminPassword();
      }
    });
  }
  if (credAdminCopyBtn) {
    credAdminCopyBtn.addEventListener('click', async () => {
      if (!adminPasswordCache) {
        // Auto-reveal so the operator only has to click Copy once.
        await revealAdminPassword();
      }
      if (adminPasswordCache) {
        await copyToClipboard(adminPasswordCache, 'Admin password');
      }
    });
  }
  if (credGuestCopyBtn) {
    credGuestCopyBtn.addEventListener('click', async () => {
      if (!guestPinCache) {
        await hydrateGuestPin();
      }
      if (guestPinCache) {
        await copyToClipboard(guestPinCache, 'Guest PIN');
      }
    });
  }

  const restoreLastTab = () => {
    let target = null;
    try { target = localStorage.getItem(LAST_TAB_KEY); } catch {}
    if (target === 'credentials' || !target) target = 'network';
    activateTab(target, { persist: false });
  };

  if (tabsNav) {
    tabsNav.addEventListener('click', (e) => {
      const btn = e.target.closest('.settings-tab');
      if (!btn || !btn.dataset.tab) return;
      activateTab(btn.dataset.tab);
    });
    tabsNav.addEventListener('keydown', (e) => {
      if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key)) return;
      const current = tabButtons.findIndex(b => b.classList.contains('is-active'));
      if (current < 0) return;
      let next;
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        next = (current + 1) % tabButtons.length;
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        next = (current - 1 + tabButtons.length) % tabButtons.length;
      } else if (e.key === 'Home') {
        next = 0;
      } else {
        next = tabButtons.length - 1;
      }
      e.preventDefault();
      activateTab(tabButtons[next].dataset.tab, { focusTab: true });
    });
  }

  const open = async () => {
    window.dispatchEvent(new CustomEvent('meeting-scribe:close-diagnostics'));
    panel.classList.add('open');
    backdrop.classList.add('open');
    panel.setAttribute('aria-hidden', 'false');
    openBtn.setAttribute('aria-expanded', 'true');
    if (!loaded) await load();
    restoreLastTab();
    // Defer focus to allow the transition to start. When the restored
    // tab is Network the original focus (first dropdown) is preserved;
    // otherwise focus lands on the active tab button so the keyboard
    // operator can read what tab they're on.
    setTimeout(() => {
      const activeTab = tabButtons.find(b => b.classList.contains('is-active'));
      if (activeTab && activeTab.dataset.tab !== 'network') {
        activeTab.focus();
        return;
      }
      const firstField =
        regdomainSelect.disabled ? timezoneSelect : regdomainSelect;
      firstField.focus();
    }, 60);
  };

  const close = (restoreFocus = true) => {
    panel.classList.remove('open');
    backdrop.classList.remove('open');
    panel.setAttribute('aria-hidden', 'true');
    openBtn.setAttribute('aria-expanded', 'false');
    // Mask the admin password before the panel slides shut so the next
    // open starts hidden — defense against a "left the panel up on a
    // shared screen" scenario.
    maskAdminPassword();
    if (restoreFocus) openBtn.focus();
  };

  openBtn.addEventListener('click', () => {
    if (panel.classList.contains('open')) close();
    else open();
  });
  closeBtn.addEventListener('click', close);
  backdrop.addEventListener('click', close);
  saveBtn.addEventListener('click', save);
  if (wifiModeSelect) {
    wifiModeSelect.addEventListener('change', () => {
      wifiModeDirty = true;
    });
  }
  window.addEventListener('meeting-scribe:close-settings', () => close(false));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && panel.classList.contains('open')) {
      e.preventDefault();
      close();
    }
  });

  panel.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'Enter' && !e.isComposing) {
      // Don't trigger on select (browser handles Enter to open dropdown).
      if (e.target.tagName !== 'SELECT' && !e.target.closest('.term-font-row')) {
        e.preventDefault();
        save();
      }
    }
  });

  // Font-size slider — keyboard shortcuts Ctrl+/−/0 still work inside xterm.
  const termFontSlider  = document.getElementById('setting-term-font-size');
  const termFontValue   = document.getElementById('term-font-value');
  const termFontDecBtn  = document.getElementById('btn-term-font-dec');
  const termFontIncBtn  = document.getElementById('btn-term-font-inc');

  const clampFont = (n) => Math.max(9, Math.min(26, Math.round(n)));
  const applyTerminalFont = (px, { push = true } = {}) => {
    const v = clampFont(px);
    if (termFontSlider && String(termFontSlider.value) !== String(v)) {
      termFontSlider.value = String(v);
    }
    if (termFontValue) termFontValue.textContent = `${v}px`;
    if (push) {
      try { localStorage.setItem('terminal_font_size', String(v)); } catch {}
      if (window._terminalPanel && typeof window._terminalPanel.setFontSize === 'function') {
        window._terminalPanel.setFontSize(v);
      }
    }
  };
  // Initial paint from localStorage.
  applyTerminalFont(parseInt(localStorage.getItem('terminal_font_size') || '13', 10), { push: false });
  if (termFontSlider) {
    termFontSlider.addEventListener('input', () => applyTerminalFont(termFontSlider.value));
  }
  if (termFontDecBtn) termFontDecBtn.addEventListener('click', () => applyTerminalFont(clampFont((termFontSlider && +termFontSlider.value) || 13) - 1));
  if (termFontIncBtn) termFontIncBtn.addEventListener('click', () => applyTerminalFont(clampFont((termFontSlider && +termFontSlider.value) || 13) + 1));
  // React when the terminal panel itself changes font size (e.g. Ctrl+=).
  window.addEventListener('terminal:font-size', (e) => {
    if (e && e.detail && typeof e.detail.size === 'number') {
      applyTerminalFont(e.detail.size, { push: false });
    }
  });

}
