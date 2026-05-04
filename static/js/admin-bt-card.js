// Admin BT card — CSP-clean external script (Plan §B.7b).
//
// Talks to /api/admin/bt/* over fetch (cookie auth carried automatically).
// All DOM writes go through textContent — never innerHTML — to satisfy
// the XSS hardening rule for attacker-controllable BT device names.

(function () {
  const root = document.getElementById("bt-card");
  if (!root) return;

  const statusLine = root.querySelector(".bt-status-line");
  const devicesList = root.querySelector(".bt-devices-list");
  const micToggle = root.querySelector(".bt-mic-toggle");
  const micStatus = root.querySelector(".bt-mic-status");

  let pollHandle = null;

  function setStatus(text, ok) {
    if (!statusLine) return;
    statusLine.textContent = text;
    statusLine.classList.toggle("ok", !!ok);
    statusLine.classList.toggle("err", !ok);
  }

  function renderDevices(devices) {
    if (!devicesList) return;
    while (devicesList.firstChild) {
      devicesList.removeChild(devicesList.firstChild);
    }
    devices.forEach(function (device) {
      const li = document.createElement("li");
      li.className = "bt-device";
      const name = document.createElement("span");
      name.className = "bt-name";
      name.textContent = device.name || "(unknown)";
      const mac = document.createElement("span");
      mac.className = "bt-mac";
      mac.textContent = device.mac;
      li.appendChild(name);
      li.appendChild(mac);
      devicesList.appendChild(li);
    });
  }

  function refresh() {
    fetch("/api/admin/bt/status", { credentials: "include" })
      .then(function (resp) {
        if (!resp.ok) {
          setStatus("Bluetooth stack unavailable", false);
          return null;
        }
        return resp.json();
      })
      .then(function (body) {
        if (!body) return;
        setStatus(
          body.powered ? "Adapter powered" : "Adapter off",
          body.powered === true,
        );
        renderDevices(Array.isArray(body.devices) ? body.devices : []);
      })
      .catch(function () {
        setStatus("Network error contacting BT API", false);
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
        } else {
          if (micStatus) {
            micStatus.textContent = "Mic toggle failed";
          }
        }
      })
      .catch(function () {
        if (micStatus) {
          micStatus.textContent = "Network error";
        }
      });
  }

  if (micToggle) {
    micToggle.addEventListener("change", function () {
      setMic(micToggle.checked);
    });
  }

  function start() {
    refresh();
    pollHandle = setInterval(refresh, 5000);
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

  start();
})();
