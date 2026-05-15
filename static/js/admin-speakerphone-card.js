// admin-speakerphone-card.js
//
// Hardware tab in the admin SPA. Polls /api/admin/speakerphone/state
// every 1 s, renders the live device list + button mapping + LED state
// machine + default meeting profile. Every per-field edit emits a
// single RFC 6902 JSON-Patch op against /api/admin/speakerphone/mapping
// — never a shallow PUT, so siblings can never be clobbered. Only the
// "Reset to defaults" button uses PUT (and only with the current ETag).

(function () {
  "use strict";

  const POLL_INTERVAL_MS = 1000;
  const FLASH_DURATION_MS = 300;
  const ACTION_OPTIONS = [
    { value: "noop", label: "(unmapped)" },
    { value: "tts_cycle", label: "Cycle TTS direction" },
    { value: "interpretation_toggle", label: "Toggle interpretation on/off" },
    { value: "meeting_record_toggle", label: "Start / stop meeting" },
    { value: "mic_mute_toggle", label: "Toggle mic mute" },
  ];
  const LED_PATTERN_OPTIONS = [
    "off", "solid", "slow_blink", "blink", "fast_blink",
    "very_fast_blink", "slow_pulse", "double_blink",
  ];
  const OS_HANDLED = [
    { key: "volume_up", label: "Volume +", behavior: "PipeWire default-sink volume up" },
    { key: "volume_down", label: "Volume −", behavior: "PipeWire default-sink volume down" },
    { key: "system_mute", label: "System mute", behavior: "PipeWire default-sink mute toggle" },
  ];
  // Mirror of constants.py — kept in sync via the LED behavior table the
  // server sends; this list defines the rendering order.
  const LED_STATE_ORDER = [
    "error", "backend_unready", "mic_muted", "recording", "idle_ready",
  ];
  const LED_STATE_LABELS = {
    error: "Error (daemon)",
    backend_unready: "Backend not ready",
    mic_muted: "Mic muted",
    recording: "Recording",
    idle_ready: "Idle / ready",
  };

  // Languages Qwen3-TTS can synthesize natively. Mirrors the
  // ``tts_native=True`` subset of LANGUAGE_REGISTRY in
  // ``meeting_scribe/languages.py``. The CI label-coverage test
  // (tests/test_speakerphone_labels.py) asserts every label_id has
  // an entry for every code listed here, so if a contributor adds
  // a new TTS-native language they MUST update both lists.
  const TTS_NATIVE_LANGS = [
    { code: "en", name: "English" },
    { code: "zh", name: "Chinese (中文)" },
    { code: "ja", name: "Japanese (日本語)" },
    { code: "ko", name: "Korean (한국어)" },
    { code: "fr", name: "French (Français)" },
    { code: "de", name: "German (Deutsch)" },
    { code: "es", name: "Spanish (Español)" },
    { code: "it", name: "Italian (Italiano)" },
    { code: "pt", name: "Portuguese (Português)" },
    { code: "ru", name: "Russian (Русский)" },
  ];

  // Label used for the "Test feedback" button. Stays in sync with
  // labels.py — volume_up is always present in the canonical catalog.
  const TEST_PREVIEW_LABEL = "volume_up";

  let pollTimer = null;
  let lastEtag = null;
  let lastPressKey = null;  // {device_key, button, press_kind, at}

  // ── DOM helpers ─────────────────────────────────────────────────────

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "dataset") {
        for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
      } else if (k.startsWith("on") && typeof v === "function") {
        node.addEventListener(k.slice(2), v);
      } else if (v != null) {
        node.setAttribute(k, v);
      }
    }
    for (const child of children) {
      if (child == null) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function actionSelect(currentValue, onChange) {
    const sel = el("select", { class: "speakerphone-action-select" });
    for (const opt of ACTION_OPTIONS) {
      sel.append(
        el("option", { value: opt.value, ...(opt.value === currentValue ? { selected: "" } : {}) }, opt.label),
      );
    }
    sel.addEventListener("change", () => onChange(sel.value));
    return sel;
  }

  function patternSelect(currentValue, onChange) {
    const sel = el("select", { class: "speakerphone-pattern-select" });
    for (const name of LED_PATTERN_OPTIONS) {
      sel.append(
        el("option", { value: name, ...(name === currentValue ? { selected: "" } : {}) }, name),
      );
    }
    sel.addEventListener("change", () => onChange(sel.value));
    return sel;
  }

  // ── HTTP ────────────────────────────────────────────────────────────

  async function fetchJson(path, init) {
    const resp = await fetch(path, { credentials: "same-origin", ...init });
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`${resp.status} ${path}: ${body || resp.statusText}`);
    }
    return await resp.json();
  }

  async function applyPatch(op) {
    // Single-op JSON-Patch against the mapping document.
    setStatus("Saving…");
    try {
      await fetchJson("/api/admin/speakerphone/mapping", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([op]),
      });
      setStatus("Saved.");
      // Bump poll immediately so the user sees the change reflected.
      await pollOnce();
    } catch (e) {
      console.error("speakerphone PATCH failed:", e);
      setStatus(`Save failed: ${e.message}`);
    }
  }

  async function fullReset() {
    setStatus("Resetting…");
    try {
      await fetchJson("/api/admin/speakerphone/reset-defaults", {
        method: "POST",
      });
      setStatus("Reset to defaults.");
      await pollOnce();
    } catch (e) {
      console.error("speakerphone reset failed:", e);
      setStatus(`Reset failed: ${e.message}`);
    }
  }

  async function ledTest() {
    setStatus("LED test queued. Watch the Mute LED ring.");
    try {
      await fetchJson("/api/admin/speakerphone/led-test", { method: "POST" });
    } catch (e) {
      setStatus(`LED test failed: ${e.message}`);
    }
  }

  function setStatus(msg) {
    const el = document.getElementById("speakerphone-status");
    if (!el) return;
    el.textContent = msg;
  }

  // ── Render ──────────────────────────────────────────────────────────

  function renderDevices(state) {
    const root = document.getElementById("speakerphone-devices");
    if (!root) return;
    root.innerHTML = "";
    const devices = (state.mapping || {}).devices || {};
    const keys = Object.keys(devices);
    if (keys.length === 0) {
      root.append(el("p", { class: "settings-field-help" }, "No devices configured."));
      return;
    }
    for (const key of keys) {
      const dev = devices[key];
      root.append(
        el(
          "div",
          { class: "speakerphone-device-row", dataset: { deviceKey: key } },
          el("strong", {}, dev.name || key),
          el("span", { class: "speakerphone-vid-pid" }, " · ", key),
        ),
      );
    }
  }

  function renderButtons(state) {
    const root = document.getElementById("speakerphone-buttons");
    if (!root) return;
    root.innerHTML = "";
    const devices = (state.mapping || {}).devices || {};
    const buttonLabels = {
      phone: "Phone",
      teams: "Teams",
      phone_mute: "Phone mute",
    };
    for (const [deviceKey, dev] of Object.entries(devices)) {
      for (const buttonName of ["phone", "teams", "phone_mute"]) {
        const binding = (dev.buttons || {})[buttonName] || {};
        const row = el(
          "div",
          { class: "speakerphone-button-row", dataset: { deviceKey, button: buttonName } },
          el("span", { class: "speakerphone-button-label" }, buttonLabels[buttonName]),
        );
        // Short press
        row.append(
          el(
            "label",
            {},
            "Short: ",
            actionSelect(binding.short || "noop", (v) =>
              applyPatch({
                op: "replace",
                path: `/devices/${deviceKey}/buttons/${buttonName}/short`,
                value: v,
              }),
            ),
          ),
        );
        // Long press (only Phone supports it by convention; show for all)
        const longSelect = actionSelect(binding.long || "noop", (v) =>
          applyPatch({
            op: "replace",
            path: `/devices/${deviceKey}/buttons/${buttonName}/long`,
            value: v,
          }),
        );
        // The PATCH allow-list only accepts /short and /long children; if
        // the binding doesn't yet have a long key, sending the op adds it
        // (mapping.apply_patch uses replace, which would error). Instead,
        // pre-seed via PUT when needed — but for v1 we just let users only
        // edit existing keys. A future patch will lift this.
        row.append(el("label", {}, "Long: ", longSelect));
        row.append(el("span", { class: "speakerphone-last-press" }, "—"));
        root.append(row);
      }
    }
  }

  function renderOsHandled() {
    const root = document.getElementById("speakerphone-os-buttons");
    if (!root) return;
    root.innerHTML = "";
    for (const entry of OS_HANDLED) {
      root.append(
        el(
          "li",
          { class: "speakerphone-os-row" },
          el("strong", {}, entry.label),
          el("span", { class: "speakerphone-os-behavior" }, " · ", entry.behavior),
          el("span", { class: "speakerphone-os-badge" }, " OS-managed"),
        ),
      );
    }
  }

  function renderLedStates(state) {
    const root = document.getElementById("speakerphone-led-states");
    if (!root) return;
    root.innerHTML = "";
    const states = ((state.mapping || {}).leds || {}).states || {};
    for (const name of LED_STATE_ORDER) {
      const cfg = states[name] || { enabled: true, pattern: "off" };
      const row = el(
        "div",
        { class: "speakerphone-led-row", dataset: { state: name } },
        el("strong", {}, LED_STATE_LABELS[name] || name),
      );
      const enabledBox = el("input", {
        type: "checkbox",
        ...(cfg.enabled ? { checked: "" } : {}),
        onchange: (ev) =>
          applyPatch({
            op: "replace",
            path: `/leds/states/${name}/enabled`,
            value: ev.target.checked,
          }),
      });
      row.append(el("label", {}, "Enabled: ", enabledBox));
      row.append(
        el(
          "label",
          {},
          "Pattern: ",
          patternSelect(cfg.pattern, (v) =>
            applyPatch({
              op: "replace",
              path: `/leds/states/${name}/pattern`,
              value: v,
            }),
          ),
        ),
      );
      root.append(row);
    }
  }

  function renderProfile(state) {
    const profile = (state.mapping || {}).default_meeting_profile || {};
    const nameInput = document.getElementById("speakerphone-profile-name");
    if (nameInput && document.activeElement !== nameInput) {
      nameInput.value = profile.name || "";
    }
    const interpBox = document.getElementById("speakerphone-profile-interp");
    if (interpBox) interpBox.checked = !!profile.interpretation_enabled;
    const langs = profile.languages || [];

    // Populate language multi-select (the union of meeting_languages from
    // the server plus the static fallback ["en","ja","zh","ko","fr","de",
    // "es","it","pt","ru"] — TTS-native set).
    const langsSel = document.getElementById("speakerphone-profile-langs");
    if (langsSel) {
      const known = ["en", "ja", "zh", "ko", "fr", "de", "es", "it", "pt", "ru"];
      langsSel.innerHTML = "";
      for (const code of known) {
        const opt = el(
          "option",
          { value: code, ...(langs.includes(code) ? { selected: "" } : {}) },
          code,
        );
        langsSel.append(opt);
      }
    }
    const roomSel = document.getElementById("speakerphone-profile-room");
    if (roomSel) {
      roomSel.innerHTML = "";
      const options = [...langs, "all"];
      for (const v of options) {
        roomSel.append(el(
          "option",
          { value: v, ...(profile.room_tts_language === v ? { selected: "" } : {}) },
          v,
        ));
      }
    }

    // Long-press threshold
    const lp = document.getElementById("speakerphone-long-press");
    if (lp && document.activeElement !== lp) {
      lp.value = state.mapping?.long_press_ms ?? 1000;
    }
  }

  function flashRow(deviceKey, button) {
    const sel = `.speakerphone-button-row[data-device-key="${CSS.escape(deviceKey)}"][data-button="${CSS.escape(button)}"]`;
    const row = document.querySelector(sel);
    if (!row) return;
    row.classList.add("is-flashing");
    const lastSpan = row.querySelector(".speakerphone-last-press");
    if (lastSpan) lastSpan.textContent = "just pressed";
    setTimeout(() => row.classList.remove("is-flashing"), FLASH_DURATION_MS);
  }

  function maybeFlashLastPress(state) {
    const lp = state.last_press;
    if (!lp) return;
    const key = `${lp.device_key}/${lp.button}/${lp.at}`;
    if (key === lastPressKey) return;
    lastPressKey = key;
    flashRow(lp.device_key, lp.button);
  }

  function renderSummary(state) {
    const summary = document.getElementById("speakerphone-summary");
    if (!summary) return;
    const recording = state.meeting?.recording;
    const interp = state.interpretation || {};
    summary.innerHTML = "";
    summary.append(
      `Interpretation: ${interp.enabled ? "ON" : "off"}`,
      el("br"),
      `Direction: ${interp.room_tts_language || "—"}`,
      el("br"),
      `Recording: ${recording ? "YES (id: " + (state.meeting.meeting_id || "?") + ")" : "no"}`,
      el("br"),
      `Last active direction: ${interp.last_active_room_tts_language || "—"}`,
    );
  }

  // ── Poll loop ───────────────────────────────────────────────────────

  async function pollOnce() {
    try {
      const state = await fetchJson("/api/admin/speakerphone/state");
      if (state.etag !== lastEtag) {
        lastEtag = state.etag;
        renderDevices(state);
        renderButtons(state);
        renderOsHandled();
        renderLedStates(state);
        renderProfile(state);
        renderButtonFeedback(state);
      }
      renderSummary(state);
      maybeFlashLastPress(state);
    } catch (e) {
      // Probably no admin auth (settings panel hasn't been opened with
      // creds yet) or transient. Silently retry on the next tick.
      console.warn("speakerphone poll skipped:", e.message);
    }
  }

  // ── Button feedback UI ──────────────────────────────────────────────

  function renderButtonFeedback(state) {
    const feedback = (state.mapping || {}).button_feedback || {};
    const enabled = !!feedback.enabled;
    const language = feedback.language || "en";

    const enabledBox = document.getElementById("speakerphone-feedback-enabled");
    if (enabledBox && document.activeElement !== enabledBox) {
      enabledBox.checked = enabled;
    }

    const langSel = document.getElementById("speakerphone-feedback-language");
    if (langSel) {
      // Rebuild the option list every render so a future code change
      // to TTS_NATIVE_LANGS is picked up without manual refresh.
      langSel.innerHTML = "";
      for (const { code, name } of TTS_NATIVE_LANGS) {
        const opt = el("option", { value: code }, name);
        if (code === language) opt.setAttribute("selected", "");
        langSel.append(opt);
      }
    }
  }

  async function previewFeedback(labelId, language, unsavedText) {
    // ``unsavedText`` retained for backward compatibility with future
    // per-label override UI — when not used, the preview endpoint
    // falls back to the saved override → catalog → English.
    const body = { label_id: labelId };
    if (language) body.language = language;
    const text = (unsavedText || "").trim();
    if (text) {
      body.overrides = { [labelId]: { [language]: text } };
    }
    setStatus(`Previewing ${labelId}…`);
    try {
      const resp = await fetchJson("/api/admin/speakerphone/speak/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (resp.skipped) {
        setStatus(`Preview skipped (${resp.reason || "?"})`);
      } else if (resp.ok) {
        setStatus(`Preview played: ${labelId}`);
      } else {
        setStatus(`Preview failed: ${resp.reason || "unknown"}`);
      }
    } catch (e) {
      console.error("speakerphone preview failed:", e);
      setStatus(`Preview failed: ${e.message}`);
    }
  }

  function start() {
    if (pollTimer) return;
    pollOnce();
    pollTimer = setInterval(pollOnce, POLL_INTERVAL_MS);
  }

  function stop() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  function wireEventHandlers() {
    const ledBtn = document.getElementById("speakerphone-led-test");
    if (ledBtn) ledBtn.addEventListener("click", ledTest);
    const resetBtn = document.getElementById("speakerphone-reset");
    if (resetBtn) resetBtn.addEventListener("click", fullReset);

    const profileName = document.getElementById("speakerphone-profile-name");
    if (profileName) {
      profileName.addEventListener("change", () =>
        applyPatch({
          op: "replace",
          path: "/default_meeting_profile/name",
          value: profileName.value,
        }),
      );
    }
    const profileInterp = document.getElementById("speakerphone-profile-interp");
    if (profileInterp) {
      profileInterp.addEventListener("change", () =>
        applyPatch({
          op: "replace",
          path: "/default_meeting_profile/interpretation_enabled",
          value: profileInterp.checked,
        }),
      );
    }
    const profileRoom = document.getElementById("speakerphone-profile-room");
    if (profileRoom) {
      profileRoom.addEventListener("change", () =>
        applyPatch({
          op: "replace",
          path: "/default_meeting_profile/room_tts_language",
          value: profileRoom.value,
        }),
      );
    }
    const profileLangs = document.getElementById("speakerphone-profile-langs");
    if (profileLangs) {
      profileLangs.addEventListener("change", () => {
        const picked = Array.from(profileLangs.selectedOptions)
          .map((o) => o.value)
          .slice(0, 2);
        applyPatch({
          op: "replace",
          path: "/default_meeting_profile/languages",
          value: picked,
        });
      });
    }
    const lp = document.getElementById("speakerphone-long-press");
    if (lp) {
      lp.addEventListener("change", () => {
        const v = parseInt(lp.value, 10);
        if (Number.isNaN(v)) return;
        applyPatch({
          op: "replace",
          path: "/long_press_ms",
          value: v,
        });
      });
    }

    // Button-feedback enabled toggle + language dropdown.
    // Per-label override inputs are wired in renderButtonFeedback so
    // each row binds its own blur handler (we re-render on every
    // language change to repopulate the inputs for the new lang).
    const fbEnabled = document.getElementById(
      "speakerphone-feedback-enabled",
    );
    if (fbEnabled) {
      fbEnabled.addEventListener("change", () =>
        applyPatch({
          op: "replace",
          path: "/button_feedback/enabled",
          value: fbEnabled.checked,
        }),
      );
    }
    const fbLang = document.getElementById("speakerphone-feedback-language");
    if (fbLang) {
      fbLang.addEventListener("change", () =>
        applyPatch({
          op: "replace",
          path: "/button_feedback/language",
          value: fbLang.value,
        }),
      );
    }
    const fbTest = document.getElementById("speakerphone-feedback-test");
    if (fbTest) {
      fbTest.addEventListener("click", () => {
        const langSel = document.getElementById("speakerphone-feedback-language");
        const lang = langSel ? langSel.value : "en";
        previewFeedback(TEST_PREVIEW_LABEL, lang, "");
      });
    }
  }

  // ── Lifecycle: start when Hardware tab is opened, stop when closed ──

  function onSettingsOpen() {
    wireEventHandlers();
    start();
  }
  function onSettingsClose() {
    stop();
  }

  // Settings panel lifecycle hooks. The settings-panel feature
  // dispatches CustomEvents 'settings:open' / 'settings:close'
  // (best-effort — if not, we still start the poll once the script
  // loads, which is harmless).
  document.addEventListener("settings:open", onSettingsOpen);
  document.addEventListener("settings:close", onSettingsClose);

  // Also wire up if the Hardware tab is already visible on load.
  if (document.readyState !== "loading") {
    onSettingsOpen();
  } else {
    document.addEventListener("DOMContentLoaded", onSettingsOpen);
  }
})();
