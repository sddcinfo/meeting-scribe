// Admin BT card — CSP-clean external script (Plan §B.7b).
//
// Talks to /api/admin/bt/* over fetch (cookie auth carried automatically).
// All DOM writes go through textContent — never innerHTML — to satisfy
// the XSS hardening rule for attacker-controllable BT device names.
//
// Status messaging is user-actionable, not a single catch-all:
//   - "Bluetooth stack unavailable"  → fetch network error
//   - "Not authorized"               → 401/403 (cookie expired / not admin)
//   - "Bluetooth backend error"      → 5xx (bluetoothctl failed)
//   - "Adapter off"                  → adapter is powered=false
//   - "Adapter on · N paired"        → healthy, with paired count
//
// Scan flow drives pairing from the UI so the operator never has to
// drop into the CLI to read off a MAC. See bt_scan() in
// meeting_scribe/bt.py + GET /api/admin/bt/scan.

(function () {
  const root = document.getElementById("bt-card");
  if (!root) return;

  const statusLine = root.querySelector(".bt-status-line");
  const devicesList = root.querySelector(".bt-devices-list");
  const micToggle = root.querySelector(".bt-mic-toggle");
  const micStatus = root.querySelector(".bt-mic-status");
  const scanBtn = root.querySelector(".bt-scan-btn");
  const scanResultsList = root.querySelector(".bt-scan-results");
  const scanStatus = root.querySelector(".bt-scan-status");

  let pollHandle = null;
  let ws = null;
  let wsRetryHandle = null;

  function setStatus(text, kind) {
    if (!statusLine) return;
    statusLine.textContent = text;
    statusLine.classList.toggle("ok", kind === "ok");
    statusLine.classList.toggle("err", kind === "err");
    statusLine.classList.toggle("warn", kind === "warn");
  }

  function setScanStatus(text, kind) {
    if (!scanStatus) return;
    scanStatus.textContent = text;
    scanStatus.classList.toggle("ok", kind === "ok");
    scanStatus.classList.toggle("err", kind === "err");
  }

  function clearChildren(el) {
    while (el && el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  function deviceStateLabel(device) {
    if (device.connected) return "Connected";
    if (device.paired) return "Paired";
    if (device.trusted) return "Known";
    return "Seen";
  }

  function refreshSoon(delay) {
    setTimeout(refresh, delay || 300);
  }

  function postDeviceAction(path, body, button, busyText, doneDelay) {
    if (button) {
      button.disabled = true;
      button.textContent = busyText;
    }
    fetch(path, {
      method: "POST",
      credentials: "include",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body || {}),
    })
      .then(function (resp) {
        if (resp.ok) {
          refreshSoon(doneDelay || 500);
        } else if (button) {
          button.disabled = false;
          button.textContent = resp.status === 401 || resp.status === 403 ? "Not authorized" : "Failed";
        }
      })
      .catch(function () {
        if (button) {
          button.disabled = false;
          button.textContent = "Network error";
        }
      });
  }

  function renderDevices(devices) {
    if (!devicesList) return;
    clearChildren(devicesList);
    devices.forEach(function (device) {
      const li = document.createElement("li");
      li.className = "bt-device";

      const meta = document.createElement("span");
      meta.className = "bt-paired-meta";
      const name = document.createElement("span");
      name.className = "bt-name";
      name.textContent = device.name || "(unknown)";
      const mac = document.createElement("span");
      mac.className = "bt-mac";
      mac.textContent = device.mac + " · " + deviceStateLabel(device);
      meta.appendChild(name);
      meta.appendChild(mac);

      const actions = document.createElement("span");
      actions.className = "bt-device-actions";

      const connectBtn = document.createElement("button");
      connectBtn.type = "button";
      connectBtn.className = "btn-ghost bt-connect-btn";
      connectBtn.textContent = device.connected ? "Disconnect" : "Connect";
      connectBtn.addEventListener("click", function () {
        postDeviceAction(
          device.connected ? "/api/admin/bt/disconnect" : "/api/admin/bt/connect",
          device.connected ? { mac: device.mac } : { mac: device.mac },
          connectBtn,
          device.connected ? "Disconnecting…" : "Connecting…",
          700,
        );
      });

      const forgetBtn = document.createElement("button");
      forgetBtn.type = "button";
      forgetBtn.className = "btn-ghost bt-forget-btn";
      forgetBtn.textContent = "Forget";
      forgetBtn.title = "Remove pairing — re-pairing requires putting the device in pairing mode again";
      forgetBtn.addEventListener("click", function () {
        const label = device.name || device.mac;
        if (!window.confirm("Forget " + label + "? You'll need to re-pair to use it again.")) {
          return;
        }
        postDeviceAction("/api/admin/bt/forget", { mac: device.mac }, forgetBtn, "Forgetting…", 300);
      });

      actions.appendChild(connectBtn);
      actions.appendChild(forgetBtn);
      li.appendChild(meta);
      li.appendChild(actions);
      devicesList.appendChild(li);
    });
  }

  function renderScanResults(devices) {
    if (!scanResultsList) return;
    clearChildren(scanResultsList);
    if (!devices.length) {
      const li = document.createElement("li");
      li.className = "bt-scan-empty";
      li.textContent = "No nearby devices found. Make sure the device is in pairing mode and try again.";
      scanResultsList.appendChild(li);
      return;
    }
    devices.forEach(function (device) {
      const li = document.createElement("li");
      li.className = "bt-device bt-scan-row";

      const meta = document.createElement("span");
      meta.className = "bt-scan-meta";
      const name = document.createElement("span");
      name.className = "bt-name";
      name.textContent = device.name || "(unknown)";
      const mac = document.createElement("span");
      mac.className = "bt-mac";
      mac.textContent = device.mac;
      meta.appendChild(name);
      meta.appendChild(mac);

      const pairBtn = document.createElement("button");
      pairBtn.type = "button";
      pairBtn.className = "btn-ghost bt-pair-btn";
      pairBtn.textContent = "Pair";
      pairBtn.addEventListener("click", function () {
        pairBtn.disabled = true;
        pairBtn.textContent = "Pairing…";
        fetch("/api/admin/bt/pair", {
          method: "POST",
          credentials: "include",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ mac: device.mac }),
        })
          .then(function (resp) {
            if (resp.ok) {
              pairBtn.textContent = "Paired ✓";
              pairBtn.classList.add("paired");
              // Refresh status so the device moves to the known list.
              refreshSoon(500);
            } else if (resp.status === 401 || resp.status === 403) {
              pairBtn.textContent = "Not authorized";
              pairBtn.disabled = false;
            } else {
              pairBtn.textContent = "Pair failed";
              pairBtn.disabled = false;
            }
          })
          .catch(function () {
            pairBtn.textContent = "Network error";
            pairBtn.disabled = false;
          });
      });

      li.appendChild(meta);
      li.appendChild(pairBtn);
      scanResultsList.appendChild(li);
    });
  }

  function setBackendKind(resp) {
    // Map an HTTP response shape to a status kind for the badge.
    if (resp.status === 401 || resp.status === 403) return "auth";
    if (resp.status >= 500) return "server";
    if (!resp.ok) return "client";
    return "ok";
  }

  function renderStatus(body) {
    if (!body) return;
    const devices = Array.isArray(body.devices) ? body.devices : [];
    const paired = Number.isFinite(body.paired_count) ? body.paired_count : devices.filter(function (d) { return d.paired; }).length;
    const connected = Number.isFinite(body.connected_count) ? body.connected_count : devices.filter(function (d) { return d.connected; }).length;
    if (body.powered === true) {
      if (connected > 0) {
        setStatus("Adapter on · " + connected + " connected · " + devices.length + " known", "ok");
      } else if (devices.length === 0) {
        setStatus("Adapter on · no known devices", "ok");
      } else {
        setStatus("Adapter on · " + paired + " paired · " + devices.length + " known", "ok");
      }
    } else {
      setStatus("Adapter off — power on Bluetooth on the host", "warn");
    }
    renderDevices(devices);
  }

  function refresh() {
    fetch("/api/admin/bt/status", { credentials: "include" })
      .then(function (resp) {
        const kind = setBackendKind(resp);
        if (kind === "auth") {
          setStatus("Not authorized — admin cookie missing or expired", "err");
          return null;
        }
        if (kind === "server") {
          setStatus("Bluetooth backend error (HTTP " + resp.status + ")", "err");
          return null;
        }
        if (kind === "client") {
          setStatus("Bluetooth status request failed (HTTP " + resp.status + ")", "err");
          return null;
        }
        return resp.json();
      })
      .then(function (body) {
        renderStatus(body);
      })
      .catch(function () {
        setStatus("Bluetooth stack unavailable (network error)", "err");
      });
  }

  function setMic(enabled) {
    fetch("/api/admin/bt/mic", {
      method: "POST",
      credentials: "include",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ enabled: !!enabled }),
    })
      .then(function (resp) {
        if (resp.ok) {
          if (micStatus) {
            micStatus.textContent = enabled ? "Mic on" : "Mic off";
          }
        } else if (resp.status === 401 || resp.status === 403) {
          if (micStatus) micStatus.textContent = "Not authorized";
        } else {
          if (micStatus) {
            micStatus.textContent = "Mic toggle failed (HTTP " + resp.status + ")";
          }
        }
      })
      .catch(function () {
        if (micStatus) {
          micStatus.textContent = "Network error";
        }
      });
  }

  function startScan() {
    if (!scanBtn) return;
    scanBtn.disabled = true;
    scanBtn.textContent = "Scanning…";
    setScanStatus("Scanning ~10 s — put the device in pairing mode now.", "ok");
    clearChildren(scanResultsList);
    fetch("/api/admin/bt/scan?timeout=10", { credentials: "include" })
      .then(function (resp) {
        scanBtn.disabled = false;
        scanBtn.textContent = "Scan again";
        if (!resp.ok) {
          if (resp.status === 401 || resp.status === 403) {
            setScanStatus("Not authorized — re-open Settings.", "err");
          } else {
            setScanStatus("Scan failed (HTTP " + resp.status + ")", "err");
          }
          return null;
        }
        return resp.json();
      })
      .then(function (body) {
        if (!body) return;
        const devices = Array.isArray(body.devices) ? body.devices : [];
        renderScanResults(devices);
        setScanStatus(
          devices.length
            ? "Found " + devices.length + " device" + (devices.length === 1 ? "" : "s") + "."
            : "Scan complete — no nearby pairing-mode devices.",
          "ok",
        );
      })
      .catch(function () {
        scanBtn.disabled = false;
        scanBtn.textContent = "Scan again";
        setScanStatus("Network error contacting BT API.", "err");
      });
  }

  if (micToggle) {
    micToggle.addEventListener("change", function () {
      setMic(micToggle.checked);
    });
  }
  if (scanBtn) {
    scanBtn.addEventListener("click", startScan);
  }

  function start() {
    refresh();
    startEventStream();
    pollHandle = setInterval(refresh, 30000);
  }

  function stop() {
    if (pollHandle != null) {
      clearInterval(pollHandle);
      pollHandle = null;
    }
    stopEventStream();
  }

  function stopEventStream() {
    if (wsRetryHandle != null) {
      clearTimeout(wsRetryHandle);
      wsRetryHandle = null;
    }
    if (ws) {
      ws.onclose = null;
      ws.close();
      ws = null;
    }
  }

  function startEventStream() {
    if (ws && ws.readyState !== WebSocket.CLOSED) return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/api/ws/view");
    ws.onmessage = function (evt) {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "bt_status") {
          renderStatus(msg.status || msg);
        }
      } catch (err) {}
    };
    ws.onclose = function () {
      ws = null;
      if (!document.hidden && wsRetryHandle == null) {
        wsRetryHandle = setTimeout(function () {
          wsRetryHandle = null;
          startEventStream();
        }, 2000);
      }
    };
    ws.onerror = function () {
      try { ws.close(); } catch (err) {}
    };
  }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      stop();
    } else if (pollHandle == null) {
      start();
    }
  });

  window.addEventListener("meeting-scribe:bt-status", function (evt) {
    renderStatus(evt.detail);
  });

  start();
})();
