// Meeting Scribe - HDMI Display settings tab wiring.
//
// Loads the current values + connector status from
// GET /api/admin/settings, populates the form, and PUTs the diff
// back when Apply is clicked. The kiosk-runtime watches
// settings.json via mtime and applies via wlr-randr within ~1 s.
//
// Pure module: no top-level side effects. Bootstrap counterpart is
// .bootstrap.js (binds DOM events once the settings tab is in the DOM).

import { esc } from "../lib/escape.js";

const API = "";
const ROTATION_OPTIONS = [0, 90, 180, 270];

/** Fetch the live settings payload (includes hdmi_status). */
async function _loadSettings() {
  const resp = await fetch(`${API}/api/admin/settings`);
  if (!resp.ok) throw new Error(`settings GET ${resp.status}`);
  return resp.json();
}

/** Populate the form + status block. */
function _hydrateForm(settings) {
  const statusEl = document.getElementById("hdmi-status-indicator");
  const enabledEl = document.getElementById("setting-hdmi-enabled");
  const modeEl = document.getElementById("setting-hdmi-mode");
  const rotEl = document.getElementById("setting-hdmi-rotation");
  const sleepEl = document.getElementById("setting-hdmi-idle-sleep");
  if (!modeEl || !rotEl || !sleepEl) return;

  const status = settings.hdmi_status || {};
  if (statusEl) {
    const connected = status.connected ? "connected" : "disconnected";
    const mode = status.current_mode || "(no mode)";
    statusEl.textContent = `Status: ${connected} - ${esc(mode)}`;
  }

  if (enabledEl) enabledEl.checked = Boolean(settings.hdmi_enabled);

  // Rebuild the mode dropdown so we always show the live list.
  while (modeEl.options.length > 1) modeEl.remove(1);
  for (const mode of status.available_modes || []) {
    const opt = document.createElement("option");
    opt.value = mode;
    opt.textContent = mode;
    modeEl.appendChild(opt);
  }
  modeEl.value = settings.hdmi_mode || "auto";

  rotEl.value = String(
    ROTATION_OPTIONS.includes(settings.hdmi_rotation) ? settings.hdmi_rotation : 0
  );
  sleepEl.value = String(Math.max(0, Math.min(240, settings.hdmi_idle_sleep_minutes || 0)));
}

/** Read form values back into a JSON body the PUT understands. */
function _collectForm() {
  const enabledEl = document.getElementById("setting-hdmi-enabled");
  const modeEl = document.getElementById("setting-hdmi-mode");
  const rotEl = document.getElementById("setting-hdmi-rotation");
  const sleepEl = document.getElementById("setting-hdmi-idle-sleep");
  return {
    hdmi_enabled: Boolean(enabledEl && enabledEl.checked),
    hdmi_mode: modeEl ? modeEl.value : "auto",
    hdmi_rotation: rotEl ? Number(rotEl.value) : 0,
    hdmi_idle_sleep_minutes: sleepEl ? Number(sleepEl.value) : 0,
  };
}

async function _applyForm() {
  const statusEl = document.getElementById("hdmi-apply-status");
  if (statusEl) statusEl.textContent = "Saving...";
  try {
    const body = _collectForm();
    const resp = await fetch(`${API}/api/admin/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      credentials: "same-origin",
    });
    if (!resp.ok) {
      const text = await resp.text();
      if (statusEl) statusEl.textContent = `Save failed: ${esc(text.slice(0, 200))}`;
      return;
    }
    const data = await resp.json();
    _hydrateForm(data);
    if (statusEl) statusEl.textContent = "Saved.";
  } catch (err) {
    if (statusEl) statusEl.textContent = `Save failed: ${esc(String(err))}`;
  }
}

async function _refreshStatus() {
  try {
    const settings = await _loadSettings();
    _hydrateForm(settings);
  } catch (err) {
    console.warn("[hdmi-settings] refresh failed", err);
  }
}

export function initHdmiPanel() {
  const root = document.getElementById("settings-pane-display");
  if (!root) return;
  const refreshBtn = document.getElementById("btn-hdmi-refresh");
  const applyBtn = document.getElementById("btn-hdmi-apply");
  if (refreshBtn && !refreshBtn.dataset.bound) {
    refreshBtn.dataset.bound = "1";
    refreshBtn.addEventListener("click", _refreshStatus);
  }
  if (applyBtn && !applyBtn.dataset.bound) {
    applyBtn.dataset.bound = "1";
    applyBtn.addEventListener("click", _applyForm);
  }
  _refreshStatus();
}
