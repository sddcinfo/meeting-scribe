// Admin Audio routing card — CSP-clean external script.
//
// Renders two device dropdowns (meeting microphone + meeting playback)
// driven by /api/admin/audio/devices. Posts changes to
// /api/admin/audio/route. The mic-active toggle is a separate field so
// the operator can stage a node selection before flipping the
// browser-mic-replaced semantics on (the server-side capture takes
// over the moment the toggle is on AND a non-empty mic_node is
// configured).
//
// Status messaging mirrors admin-bt-card.js:
//   - "Audio stack unavailable"   → fetch network error
//   - "Not authorized"             → 401/403
//   - "Audio backend error"        → 5xx
//   - "<N> source / <M> sink devices" → healthy
//
// All DOM writes go through textContent (no innerHTML) — device
// descriptions can include attacker-influenced strings via the device's
// USB descriptor.

(function () {
  const roots = [
    document.getElementById("audio-routing-card"),
    document.getElementById("setup-audio-routing-card"),
  ].filter(Boolean);
  if (!roots.length) return;

  roots.forEach(initAudioRoutingCard);

  // Setup-mode draft buffer. The wizard accumulates changes locally and
  // hands them to the recording-lifecycle Start call as a single
  // ``audio_config`` payload. Settings mode keeps its per-field PUT.
  // Module-scoped so ``window._adminAudioCard`` getters can read it
  // without an extra wiring step.
  const _setupDraft = {};

  function _applyDraftField(patch) {
    Object.assign(_setupDraft, patch);
  }

  function _readSetupDraft() {
    // Strip empty-string fields so the server side validator doesn't
    // get a noisy patch — empty string means "default" anyway and
    // never overrides an existing persisted value at the API edge.
    const out = {};
    for (const [k, v] of Object.entries(_setupDraft)) {
      if (typeof v === "string" && !v) continue;
      out[k] = v;
    }
    return out;
  }

  // Public surface used by recording-lifecycle's ``startRecording`` to
  // build the atomic ``audio_config`` payload on the Start request.
  window._adminAudioCard = window._adminAudioCard || {};
  window._adminAudioCard.getSetupDraft = _readSetupDraft;

  function initAudioRoutingCard(root) {
  const mode = root.dataset.mode === "setup" ? "setup" : "settings";
  const statusLine = root.querySelector(".audio-routing-status");
  const micSelect = root.querySelector(".audio-mic-select, #audio-mic-select");
  const adminSinkSelect = root.querySelector(".audio-admin-sink-select, #audio-admin-sink-select");
  const roomSinkSelect = root.querySelector(".audio-room-sink-select, #audio-room-sink-select");
  const adminLangSettingsSelect = root.querySelector(".audio-admin-language-select, #audio-admin-language-select");
  const roomLangSettingsSelect = root.querySelector(".audio-room-language-select, #audio-room-language-select");
  const voiceModeSelect = root.querySelector(".audio-voice-mode-select");
  const micActiveToggle = root.querySelector(".audio-mic-active-toggle");
  const micActiveStatus = root.querySelector(".audio-mic-active-status");
  const refreshBtn = root.querySelector(".audio-refresh-btn");
  const refreshStatus = root.querySelector(".audio-refresh-status");
  const meetingInterpretationRoot = document.getElementById("meeting-interpretation-controls");
  const interpretationToggle = root.querySelector(".interpretation-enabled-toggle");
  const meetingInterpretationToggle = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".meeting-interpretation-enabled-toggle");
  const interpretationPauseSec = root.querySelector(".interpretation-pause-sec");
  const interpretationIdleSec = root.querySelector(".interpretation-idle-sec");
  const interpretationStatusEls = Array.from(document.querySelectorAll(".interpretation-status"));
  const meetingAdminSinkSelect = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".meeting-admin-sink-select");
  const meetingAdminLangSelect = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".meeting-admin-lang-select");
  const meetingRoomSinkSelect = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".meeting-room-sink-select");
  const meetingRoomLangSelect = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".meeting-room-lang-select");
  const meetingVoiceModeSelect = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".meeting-voice-mode-select");
  const muteMicBtn = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".interpretation-mute-mic");
  const muteRoomBtn = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".interpretation-mute-room");
  const muteWebBtn = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".interpretation-mute-web");
  const muteBtBtn = meetingInterpretationRoot && meetingInterpretationRoot.querySelector(".interpretation-mute-bt");

  let pollHandle = null;
  let interpretationSnapshot = null;

  function setStatus(text, kind) {
    if (!statusLine) return;
    statusLine.textContent = text;
    statusLine.classList.toggle("ok", kind === "ok");
    statusLine.classList.toggle("err", kind === "err");
    statusLine.classList.toggle("warn", kind === "warn");
  }

  function setRefreshStatus(text, kind) {
    if (!refreshStatus) return;
    refreshStatus.textContent = text;
    refreshStatus.classList.toggle("ok", kind === "ok");
    refreshStatus.classList.toggle("err", kind === "err");
  }

  function setMicActiveStatus(text, kind) {
    if (!micActiveStatus) return;
    micActiveStatus.textContent = text;
    micActiveStatus.classList.toggle("ok", kind === "ok");
    micActiveStatus.classList.toggle("err", kind === "err");
  }

  function setInterpretationStatus(text, kind) {
    interpretationStatusEls.forEach(function (el) {
      el.textContent = text;
      el.classList.toggle("ok", kind === "ok");
      el.classList.toggle("err", kind === "err");
    });
  }

  function msToSeconds(ms, fallbackMs) {
    const raw = Number(ms || fallbackMs);
    return (raw / 1000).toFixed(raw % 1000 === 0 ? 0 : 1);
  }

  function secondsToMs(value, fallbackMs) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds <= 0) return fallbackMs;
    return Math.round(seconds * 1000);
  }

  function classBadge(deviceClass) {
    if (deviceClass === "usb") return "USB";
    if (deviceClass === "bluetooth") return "BT";
    if (deviceClass === "hdmi") return "HDMI";
    return "ALSA";
  }

  // Replace a select's <option> children. Preserves the leading
  // "default" option (value="") that the HTML provides as a placeholder.
  function fillSelect(select, devices, currentValue) {
    if (!select) return;
    // Drop everything past the first <option>.
    while (select.children.length > 1) {
      select.removeChild(select.lastChild);
    }
    devices.forEach(function (d) {
      const opt = document.createElement("option");
      opt.value = d.node_name;
      const badge = classBadge(d.device_class);
      const star = d.is_default ? " ★" : "";
      opt.textContent = "[" + badge + "] " + (d.description || d.node_name) + star;
      select.appendChild(opt);
    });
    // Apply current selection. If the persisted value isn't in the
    // refreshed device list (e.g. unplugged USB), keep the value but
    // mark it visually so the operator notices.
    if (currentValue && !devices.some(function (d) { return d.node_name === currentValue; })) {
      const opt = document.createElement("option");
      opt.value = currentValue;
      opt.textContent = "(missing) " + currentValue;
      opt.dataset.missing = "1";
      select.appendChild(opt);
    }
    select.value = currentValue || "";
  }

  // Per-device volume strip. Renders one row per USB / Bluetooth
  // source + sink so the operator can adjust the Poly Sync's mic gain
  // and speaker volume without leaving the admin tab. Built-in HDMI /
  // generic ALSA devices are skipped — they're rarely the meeting
  // device and add visual noise.
  const _volumeDebounceMs = 220;
  const _volumeTimers = new Map(); // node_name → setTimeout handle
  // node_name → list of row elements (one per .audio-device-volumes
  // placeholder — settings card, setup card, in-meeting popover).
  const _volumeRows = new Map();

  function _isControllableDevice(d) {
    if (!d || typeof d.node_name !== "string") return false;
    if (d.volume == null) return false; // wpctl get-volume failed
    return d.device_class === "usb" || d.device_class === "bluetooth";
  }

  // Mirror of audio_routing._physical_audio_device_id — collapses an
  // input/output PipeWire node-name pair to a single physical-device key
  // so the UI can render one row per Poly / headset instead of two.
  function _physicalDeviceId(nodeName) {
    if (typeof nodeName !== "string") return "";
    const name = nodeName.trim();
    if (!name) return "";
    const bluez = name.match(/^bluez_(?:input|output)\.([A-Fa-f0-9_:-]+)/);
    if (bluez) return "bluez:" + bluez[1].replace(/_/g, ":").toLowerCase();
    const usb = name.match(/^alsa_(?:input|output)\.usb-(.+?)-\d{2}\./);
    if (usb) return "usb:" + usb[1].toLowerCase();
    const pci = name.match(/^alsa_(?:input|output)\.pci-(.+?)\./);
    if (pci) return "pci:" + pci[1].toLowerCase();
    return name.toLowerCase();
  }

  // Strip the per-endpoint suffix from a node description so the Poly's
  // mic + speaker rows can collapse to a single header line.
  function _trimEndpointSuffix(desc) {
    if (typeof desc !== "string") return "";
    return desc
      .replace(/\s+(?:Mono|Stereo|Analog Stereo|Mono Fallback|Mono fallback|HFP|HFP-AG|A2DP|A2DP Sink|A2DP Source)\s*$/i, "")
      .trim();
  }

  function _groupByPhysicalDevice(controllable) {
    const groups = new Map();
    controllable.forEach(function (d) {
      const id = _physicalDeviceId(d.node_name) || d.node_name;
      let group = groups.get(id);
      if (!group) {
        group = {
          id: id,
          label: _trimEndpointSuffix(d.description || d.node_name) || (d.description || d.node_name),
          device_class: d.device_class,
          source: null,
          sink: null,
        };
        groups.set(id, group);
      }
      if (d.kind === "source") group.source = d;
      else if (d.kind === "sink") group.sink = d;
      // If the trimmed label was lost (e.g. only a sink was seen
      // first), upgrade to the source's longer label when available.
      const trimmed = _trimEndpointSuffix(d.description || d.node_name);
      if (trimmed && trimmed.length > group.label.length) group.label = trimmed;
    });
    return Array.from(groups.values());
  }

  function _renderEndpointSlider(d, kindLabel) {
    const wrap = document.createElement("div");
    wrap.className = "audio-device-endpoint";
    wrap.dataset.nodeName = d.node_name;
    wrap.dataset.kind = d.kind;

    const tag = document.createElement("span");
    tag.className = "audio-device-endpoint-kind";
    tag.textContent = kindLabel;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "150";
    slider.step = "1";
    slider.className = "audio-device-volume-slider";
    slider.setAttribute(
      "aria-label",
      kindLabel + " volume for " + (d.description || d.node_name),
    );

    const readout = document.createElement("span");
    readout.className = "audio-device-volume-readout";

    const muteBtn = document.createElement("button");
    muteBtn.type = "button";
    muteBtn.className = "audio-device-volume-mute";
    muteBtn.title = "Toggle mute";

    wrap.appendChild(tag);
    wrap.appendChild(slider);
    wrap.appendChild(readout);
    wrap.appendChild(muteBtn);

    slider.addEventListener("input", function () {
      readout.textContent = slider.value + "%";
    });
    slider.addEventListener("change", function () {
      _scheduleVolumeWrite(d.node_name, Number(slider.value) / 100);
    });
    muteBtn.addEventListener("click", function () {
      const muted = wrap.classList.contains("is-muted");
      _writeDeviceVolume({ node_name: d.node_name, muted: !muted });
    });

    return wrap;
  }

  function _renderDeviceGroupRow(group) {
    const row = document.createElement("div");
    row.className = "audio-device-volume-row audio-device-grouped";
    row.dataset.physicalId = group.id;

    const name = document.createElement("span");
    name.className = "audio-device-volume-name";
    const kind = document.createElement("span");
    kind.className = "audio-device-volume-kind";
    kind.textContent = "[" + classBadge(group.device_class) + "]";
    name.appendChild(kind);
    name.appendChild(document.createTextNode(group.label));

    row.appendChild(name);

    const endpoints = document.createElement("div");
    endpoints.className = "audio-device-endpoints";
    if (group.sink) endpoints.appendChild(_renderEndpointSlider(group.sink, "Speaker"));
    if (group.source) endpoints.appendChild(_renderEndpointSlider(group.source, "Mic"));
    row.appendChild(endpoints);

    return row;
  }

  function _applyEndpointState(wrap, d) {
    if (!wrap) return;
    const slider = wrap.querySelector(".audio-device-volume-slider");
    const readout = wrap.querySelector(".audio-device-volume-readout");
    const muteBtn = wrap.querySelector(".audio-device-volume-mute");
    const muted = !!d.muted;
    const pct = Math.round(Number(d.volume || 0) * 100);
    if (slider && document.activeElement !== slider) {
      slider.value = String(pct);
    }
    if (readout) readout.textContent = pct + "%" + (muted ? " (muted)" : "");
    if (muteBtn) {
      muteBtn.textContent = muted ? "Unmute" : "Mute";
      muteBtn.classList.toggle("is-muted", muted);
    }
    wrap.classList.toggle("is-muted", muted);
  }

  function _applyGroupRowState(row, group) {
    Array.from(row.querySelectorAll(".audio-device-endpoint")).forEach(function (wrap) {
      const name = wrap.dataset.nodeName;
      const d = (group.sink && group.sink.node_name === name)
        ? group.sink
        : (group.source && group.source.node_name === name ? group.source : null);
      if (d) _applyEndpointState(wrap, d);
    });
  }

  function _scheduleVolumeWrite(nodeName, volume) {
    const existing = _volumeTimers.get(nodeName);
    if (existing) clearTimeout(existing);
    const timer = setTimeout(function () {
      _volumeTimers.delete(nodeName);
      _writeDeviceVolume({ node_name: nodeName, volume: volume });
    }, _volumeDebounceMs);
    _volumeTimers.set(nodeName, timer);
  }

  function _writeDeviceVolume(payload) {
    return fetch("/api/admin/audio/volume", {
      method: "POST",
      credentials: "include",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .then(function (body) {
        if (!body || !body.node_name) return;
        // Mirror the new state to every endpoint widget for this node
        // across all volume strips (settings / setup / popover).
        const wraps = document.querySelectorAll(
          '.audio-device-endpoint[data-node-name="' + CSS.escape(body.node_name) + '"]'
        );
        wraps.forEach(function (wrap) { _applyEndpointState(wrap, body); });
      })
      .catch(function () {
        setRefreshStatus("Volume change failed", "err");
        setTimeout(function () { setRefreshStatus("", null); }, 2000);
      });
  }

  function _renderEmptyState(host) {
    host.innerHTML = "";
    const empty = document.createElement("p");
    empty.className = "audio-device-volumes-empty";
    empty.textContent = "No USB/Bluetooth audio device detected.";
    host.appendChild(empty);
  }

  function renderDeviceVolumes(devices) {
    const hosts = Array.from(document.querySelectorAll(".audio-device-volumes"));
    if (!hosts.length) return;
    const all = (devices.sources || []).concat(devices.sinks || []);
    const controllable = all.filter(_isControllableDevice);

    if (!controllable.length) {
      hosts.forEach(_renderEmptyState);
      _volumeRows.clear();
      return;
    }

    const groups = _groupByPhysicalDevice(controllable);
    const liveIds = new Set(groups.map(function (g) { return g.id; }));

    groups.forEach(function (group) {
      hosts.forEach(function (host) {
        let row = host.querySelector(
          '.audio-device-volume-row[data-physical-id="' + CSS.escape(group.id) + '"]'
        );
        if (!row) {
          row = _renderDeviceGroupRow(group);
          host.appendChild(row);
        } else {
          // Endpoint set may have changed (BT headset switching profiles
          // mid-call). Cheaper to rebuild the inner column than to
          // diff individual sliders.
          const expected = (group.sink ? 1 : 0) + (group.source ? 1 : 0);
          if (row.querySelectorAll(".audio-device-endpoint").length !== expected) {
            const fresh = _renderDeviceGroupRow(group);
            row.replaceWith(fresh);
            row = fresh;
          }
        }
        _applyGroupRowState(row, group);
      });
    });

    hosts.forEach(function (host) {
      Array.from(host.children).forEach(function (child) {
        const id = child.dataset && child.dataset.physicalId;
        if (id && !liveIds.has(id)) child.remove();
        else if (!id) child.remove(); // empty-state placeholder
      });
    });
  }

  function fillLanguageSelect(select, options, currentValue) {
    if (!select) return;
    select.innerHTML = "";
    options.forEach(function (opt) {
      const el = document.createElement("option");
      el.value = opt.code;
      el.textContent = opt.name;
      if (opt.code === currentValue) el.selected = true;
      select.appendChild(el);
    });
    select.value = currentValue || "en";
  }

  function setBackendKind(resp) {
    if (resp.status === 401 || resp.status === 403) return "auth";
    if (resp.status >= 500) return "server";
    if (!resp.ok) return "client";
    return "ok";
  }

  function refresh() {
    fetch("/api/admin/audio/devices", { credentials: "include" })
      .then(function (resp) {
        const kind = setBackendKind(resp);
        if (kind === "auth") {
          setStatus("Not authorized — admin cookie missing or expired", "err");
          return null;
        }
        if (kind === "server") {
          setStatus("Audio backend error (HTTP " + resp.status + ")", "err");
          return null;
        }
        if (kind === "client") {
          setStatus("Audio devices request failed (HTTP " + resp.status + ")", "err");
          return null;
        }
        return resp.json();
      })
      .then(function (body) {
        if (!body) return;
        const sources = (body.devices && Array.isArray(body.devices.sources)) ? body.devices.sources : [];
        const sinks = (body.devices && Array.isArray(body.devices.sinks)) ? body.devices.sinks : [];
        const sel = body.selection || {};
        fillSelect(micSelect, sources, sel.mic_node);
        fillSelect(adminSinkSelect, sinks, sel.admin_sink_node);
        fillSelect(roomSinkSelect, sinks, sel.room_sink_node);
        fillSelect(meetingAdminSinkSelect, sinks, sel.admin_sink_node);
        fillSelect(meetingRoomSinkSelect, sinks, sel.room_sink_node);
        renderDeviceVolumes(body.devices || { sources: [], sinks: [] });
        if (micActiveToggle) micActiveToggle.checked = !!sel.mic_active;
        const ssLive = !!sel.server_mic_active_live;
        setStatus(
          sources.length + " source · " + sinks.length + " sink device" + (sinks.length === 1 ? "" : "s"),
          "ok",
        );
        if (sel.mic_active && sel.mic_node) {
          // Use the friendly device description (matches what the
          // dropdown shows) instead of the raw ALSA node_name, so
          // the captured-from status reads as a UI sentence rather
          // than a 60-char USB device path.
          const _selected = sources.find(function (d) {
            return d.node_name === sel.mic_node;
          });
          const _friendly = (_selected && _selected.description) || sel.mic_node;
          setMicActiveStatus(
            ssLive ? "Capturing from " + _friendly : "Configured but capture not yet live",
            ssLive ? "ok" : "err",
          );
        } else if (sel.mic_active && !sel.mic_node) {
          setMicActiveStatus("Toggle on but no microphone selected — pick one above.", "err");
        } else {
          setMicActiveStatus("");
        }
      })
      .catch(function () {
        setStatus("Audio stack unavailable (network error)", "err");
      });
    refreshInterpretation();
  }

  function countsText(counts) {
    const room = counts && counts.room_sink ? counts.room_sink : { total: 0, active: 0, muted: 0 };
    const web = counts && counts.web_browser ? counts.web_browser : { total: 0, active: 0, muted: 0 };
    const admin = counts && counts.admin_monitor ? counts.admin_monitor : { total: 0, active: 0, muted: 0 };
    const bt = counts && counts.bt_headset ? counts.bt_headset : { total: 0, active: 0, muted: 0 };
    return "GB10 " + room.active + "/" + room.total +
      " · Web " + (web.active + admin.active) + "/" + (web.total + admin.total) +
      " · BT " + bt.active + "/" + bt.total;
  }

  function applyInterpretation(body) {
    if (!body) return;
    interpretationSnapshot = body;
    if (interpretationToggle) interpretationToggle.checked = !!body.enabled;
    if (meetingInterpretationToggle) meetingInterpretationToggle.checked = !!body.enabled;
    if (interpretationPauseSec) interpretationPauseSec.value = msToSeconds(body.pause_flush_ms, 2500);
    if (interpretationIdleSec) interpretationIdleSec.value = msToSeconds(body.idle_drain_ms, 5000);
    const langOptions = body.local_sink_language_options || [];
    const roomOptions = [{ code: "all", name: "All" }].concat(langOptions);
    fillLanguageSelect(adminLangSettingsSelect, langOptions, body.admin_tts_language || "en");
    fillLanguageSelect(meetingAdminLangSelect, langOptions, body.admin_tts_language || "en");
    fillLanguageSelect(roomLangSettingsSelect, roomOptions, body.room_tts_language || "all");
    fillLanguageSelect(meetingRoomLangSelect, roomOptions, body.room_tts_language || "all");
    if (voiceModeSelect) {
      fillLanguageSelect(
        voiceModeSelect,
        body.tts_voice_mode_options || [],
        body.tts_voice_mode || "studio",
      );
    }
    if (meetingVoiceModeSelect) {
      fillLanguageSelect(
        meetingVoiceModeSelect,
        body.tts_voice_mode_options || [],
        body.tts_voice_mode || "studio",
      );
    }
    const counts = body.listener_counts || {};
    const room = counts.room_sink || { muted: 0 };
    const web = counts.web_browser || { muted: 0 };
    const admin = counts.admin_monitor || { muted: 0 };
    const bt = counts.bt_headset || { muted: 0 };
    if (muteRoomBtn) muteRoomBtn.textContent = room.muted ? "Unmute GB10" : "Mute GB10";
    if (muteWebBtn) muteWebBtn.textContent = (web.muted || admin.muted) ? "Unmute web" : "Mute web";
    if (muteBtBtn) muteBtBtn.textContent = bt.muted ? "Unmute BT" : "Mute BT";
    if (muteMicBtn) {
      const micMuted = !!body.mic_muted;
      muteMicBtn.textContent = micMuted ? "Unmute mic" : "Mute mic";
      muteMicBtn.classList.toggle("is-muted-input", micMuted);
    }
    setInterpretationStatus(countsText(body.listener_counts || {}) + " · " + (body.enabled ? "consecutive GB10 output" : "off"), "ok");
  }

  function refreshInterpretation() {
    if (!interpretationToggle && !meetingInterpretationRoot) return;
    fetch("/api/admin/audio/interpretation", { credentials: "include" })
      .then(function (resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .then(applyInterpretation)
      .catch(function () {
        setInterpretationStatus("Interpretation controls unavailable", "err");
      });
  }

  function postInterpretation(patch) {
    return fetch("/api/admin/audio/interpretation", {
      method: "POST",
      credentials: "include",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(patch),
    })
      .then(function (resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .then(function (body) {
        applyInterpretation(body);
        return body;
      })
      .catch(function () {
        setInterpretationStatus("Save failed", "err");
      });
  }

  function postRoute(patch, statusEl) {
    return fetch("/api/admin/audio/route", {
      method: "POST",
      credentials: "include",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(patch),
    })
      .then(function (resp) {
        const kind = setBackendKind(resp);
        if (kind === "auth") {
          if (statusEl) statusEl(":Not authorized", "err");
          return null;
        }
        if (kind === "server" || kind === "client") {
          return resp
            .json()
            .catch(function () {
              return {};
            })
            .then(function (body) {
              const error = body && body.error ? body.error : "Save failed (HTTP " + resp.status + ")";
              if (statusEl) statusEl(error, "err");
              return null;
            });
        }
        return resp.json();
      })
      .then(function (body) {
        if (!body) return;
        if (statusEl) statusEl("Saved.", "ok");
        // Pull fresh device + selection state to reflect any
        // server-side reconciliation (e.g. server_mic_active flipping).
        setTimeout(refresh, 200);
      })
      .catch(function () {
        if (statusEl) statusEl("Network error", "err");
      });
  }

  // Setup mode accumulates to a draft (no PUT). Settings mode keeps
  // the per-field POST so the persistent default updates live. The
  // ``setInterpretationStatus`` callback is reused as a tiny "Saved"
  // signal — settings mode shows it; setup mode shows nothing since
  // changes don't persist yet.
  function applyChange(patch, statusEl) {
    if (mode === "setup") {
      _applyDraftField(patch);
      return;
    }
    postRoute(patch, statusEl);
  }

  if (micSelect) {
    micSelect.addEventListener("change", function () {
      applyChange({ mic_node: micSelect.value || "" }, setMicActiveStatus);
    });
  }
  if (adminSinkSelect) {
    adminSinkSelect.addEventListener("change", function () {
      applyChange({ admin_sink_node: adminSinkSelect.value || "" }, setRefreshStatus);
    });
  }
  if (roomSinkSelect) {
    roomSinkSelect.addEventListener("change", function () {
      applyChange({ room_sink_node: roomSinkSelect.value || "" }, setRefreshStatus);
    });
  }
  if (meetingAdminSinkSelect) {
    meetingAdminSinkSelect.addEventListener("change", function () {
      applyChange({ admin_sink_node: meetingAdminSinkSelect.value || "" }, setInterpretationStatus);
    });
  }
  if (meetingRoomSinkSelect) {
    meetingRoomSinkSelect.addEventListener("change", function () {
      applyChange({ room_sink_node: meetingRoomSinkSelect.value || "" }, setInterpretationStatus);
    });
  }
  if (adminLangSettingsSelect) {
    adminLangSettingsSelect.addEventListener("change", function () {
      postInterpretation({ admin_tts_language: adminLangSettingsSelect.value || "en" });
    });
  }
  if (roomLangSettingsSelect) {
    roomLangSettingsSelect.addEventListener("change", function () {
      postInterpretation({ room_tts_language: roomLangSettingsSelect.value || "all" });
    });
  }
  if (meetingAdminLangSelect) {
    meetingAdminLangSelect.addEventListener("change", function () {
      postInterpretation({ admin_tts_language: meetingAdminLangSelect.value || "en" });
    });
  }
  if (meetingRoomLangSelect) {
    meetingRoomLangSelect.addEventListener("change", function () {
      postInterpretation({ room_tts_language: meetingRoomLangSelect.value || "all" });
    });
  }
  if (voiceModeSelect) {
    voiceModeSelect.addEventListener("change", function () {
      const value = voiceModeSelect.value === "cloned" ? "cloned" : "studio";
      postInterpretation({ tts_voice_mode: value });
    });
  }
  if (meetingVoiceModeSelect) {
    meetingVoiceModeSelect.addEventListener("change", function () {
      const value = meetingVoiceModeSelect.value === "cloned" ? "cloned" : "studio";
      postInterpretation({ tts_voice_mode: value });
    });
  }
  if (micActiveToggle) {
    micActiveToggle.addEventListener("change", function () {
      applyChange({ mic_active: !!micActiveToggle.checked }, setMicActiveStatus);
    });
  }
  if (refreshBtn) {
    refreshBtn.addEventListener("click", function () {
      setRefreshStatus("Refreshing…", "ok");
      refresh();
      setTimeout(function () { setRefreshStatus("", null); }, 1200);
    });
  }
  if (interpretationToggle) {
    interpretationToggle.addEventListener("change", function () {
      postInterpretation({ enabled: !!interpretationToggle.checked });
    });
  }
  if (meetingInterpretationToggle) {
    meetingInterpretationToggle.addEventListener("change", function () {
      postInterpretation({ enabled: !!meetingInterpretationToggle.checked });
    });
  }
  if (interpretationPauseSec) {
    interpretationPauseSec.addEventListener("change", function () {
      postInterpretation({ pause_flush_ms: secondsToMs(interpretationPauseSec.value, 2500) });
    });
  }
  if (interpretationIdleSec) {
    interpretationIdleSec.addEventListener("change", function () {
      postInterpretation({ idle_drain_ms: secondsToMs(interpretationIdleSec.value, 5000) });
    });
  }
  if (muteMicBtn) {
    muteMicBtn.addEventListener("click", function () {
      const currentlyMuted = !!(interpretationSnapshot && interpretationSnapshot.mic_muted);
      fetch("/api/admin/audio/mic", {
        method: "POST",
        credentials: "include",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ muted: !currentlyMuted }),
      })
        .then(function (resp) {
          if (!resp.ok) throw new Error("HTTP " + resp.status);
          return resp.json();
        })
        .then(applyInterpretation)
        .catch(function () {
          setInterpretationStatus("Mic mute failed", "err");
        });
    });
  }
  if (muteRoomBtn) {
    muteRoomBtn.addEventListener("click", function () {
      const c = interpretationSnapshot && interpretationSnapshot.listener_counts;
      const muted = c && c.room_sink && c.room_sink.muted;
      postInterpretation({ mute: muted ? "unmute_room_speaker" : "mute_room_speaker" });
    });
  }
  if (muteWebBtn) {
    muteWebBtn.addEventListener("click", function () {
      const c = interpretationSnapshot && interpretationSnapshot.listener_counts;
      const muted = c && ((c.web_browser && c.web_browser.muted) || (c.admin_monitor && c.admin_monitor.muted));
      postInterpretation({ mute: muted ? "unmute_web" : "mute_web" });
    });
  }
  if (muteBtBtn) {
    muteBtBtn.addEventListener("click", function () {
      const c = interpretationSnapshot && interpretationSnapshot.listener_counts;
      const muted = c && c.bt_headset && c.bt_headset.muted;
      postInterpretation({ mute: muted ? "unmute_bt_headsets" : "mute_bt_headsets" });
    });
  }

  function start() {
    refresh();
    pollHandle = setInterval(refresh, 8000);
  }

  function stop() {
    if (pollHandle != null) {
      clearInterval(pollHandle);
      pollHandle = null;
    }
  }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      stop();
    } else if (pollHandle == null) {
      start();
    }
  });

  window.addEventListener("meeting-scribe:interpretation-status", function (ev) {
    applyInterpretation(ev.detail);
  });

  start();
  }

  // In-meeting "Vol" popover wiring. Lives outside the per-card init
  // because it's a singleton owned by the meeting control bar — no
  // per-mode duplication.
  function initMeetingVolumePopover() {
    const wrap = document.querySelector(".meeting-device-volumes-popover");
    if (!wrap) return;
    const toggle = wrap.querySelector(".meeting-device-volumes-toggle");
    const panel = wrap.querySelector(".meeting-device-volumes-panel");
    if (!toggle || !panel) return;

    function setOpen(open) {
      wrap.dataset.open = open ? "true" : "false";
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "");
      }
    }

    toggle.addEventListener("click", function (ev) {
      ev.stopPropagation();
      setOpen(wrap.dataset.open !== "true");
    });

    document.addEventListener("click", function (ev) {
      if (wrap.dataset.open !== "true") return;
      if (wrap.contains(ev.target)) return;
      setOpen(false);
    });

    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && wrap.dataset.open === "true") {
        setOpen(false);
        toggle.focus();
      }
    });
  }

  initMeetingVolumePopover();
})();
