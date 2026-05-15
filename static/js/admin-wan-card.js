// Admin WAN card — CSP-clean external script, vanilla DOM.
//
// Mirrors the admin-bt-card.js pattern: short-circuits if the card
// section is absent (e.g. on guest pages), uses textContent for all
// attacker-controllable strings, and threads cookie auth through fetch
// automatically. No innerHTML on any path.
//
// Talks to /api/admin/wan/* (see meeting_scribe/routes/admin_wan.py).
//
// Status pill semantics:
//   - "Network unavailable"   → fetch network error
//   - "Not authorized"        → 401/403
//   - "Backend error"         → 5xx
//   - per-iface badges        → up/down + connectivity tier

(function () {
  const root = document.getElementById("wan-card");
  if (!root) return;

  const statusLine = root.querySelector(".wan-status-line");
  const wiredLine = root.querySelector(".wan-wired-line");
  const wifiLine = root.querySelector(".wan-wifi-line");
  const portalLink = root.querySelector(".wan-portal-link");
  const profilesList = root.querySelector(".wan-profiles-list");
  const ssidInput = root.querySelector(".wan-ssid-input");
  const pskRefInput = root.querySelector(".wan-pskref-input");
  const bandRadios = root.querySelectorAll('input[name="wan-band"]');
  const addBtn = root.querySelector(".wan-add-btn");
  const addAndConnectBtn = root.querySelector(".wan-add-and-connect-btn");
  const addStatus = root.querySelector(".wan-add-status");
  const scanBtn = root.querySelector(".wan-scan-btn");
  const scanStatus = root.querySelector(".wan-scan-status");
  const scanResults = root.querySelector(".wan-scan-results");
  const upBtn = root.querySelector(".wan-up-btn");
  const downBtn = root.querySelector(".wan-down-btn");
  const actionStatus = root.querySelector(".wan-action-status");

  function getBandValue() {
    for (const r of bandRadios) if (r.checked) return r.value;
    return "auto";
  }

  function setBandValue(v) {
    for (const r of bandRadios) r.checked = (r.value === v);
  }

  let pollHandle = null;

  function clearChildren(el) {
    while (el && el.firstChild) el.removeChild(el.firstChild);
  }

  function setStatus(el, text, kind) {
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("ok", kind === "ok");
    el.classList.toggle("err", kind === "err");
    el.classList.toggle("warn", kind === "warn");
  }

  async function api(path, opts) {
    try {
      const resp = await fetch(path, Object.assign({ credentials: "same-origin" }, opts || {}));
      return resp;
    } catch (_e) {
      return null; // network error
    }
  }

  function describeWired(w) {
    if (!w) return "—";
    const parts = [];
    parts.push(w.up ? "UP" : "DOWN");
    if (w.iface) parts.push(w.iface);
    if (w.lease) parts.push(`lease=${w.lease}`);
    if (w.route_metric != null) parts.push(`metric=${w.route_metric}`);
    if (w.default_route) parts.push("default-route");
    return parts.join("  ");
  }

  function describeWifi(w) {
    if (!w) return "—";
    const parts = [];
    parts.push(w.up ? "UP" : "DOWN");
    if (w.ssid) parts.push(w.ssid);
    if (w.bssid) parts.push(w.bssid);
    if (typeof w.signal_dbm === "number") parts.push(`${w.signal_dbm} dBm`);
    if (w.lease) parts.push(`lease=${w.lease}`);
    if (w.route_metric != null) parts.push(`metric=${w.route_metric}`);
    if (w.default_route) parts.push("default-route");
    if (w.connectivity) parts.push(`conn=${w.connectivity}`);
    return parts.join("  ");
  }

  async function refreshStatus() {
    const resp = await api("/api/admin/wan/status");
    if (!resp) return setStatus(statusLine, "Network unavailable", "err");
    if (resp.status === 401 || resp.status === 403) return setStatus(statusLine, "Not authorized", "err");
    if (!resp.ok) return setStatus(statusLine, "Backend error", "err");
    const data = await resp.json();
    setStatus(statusLine, `egress=${data.egress_mode}  active_default=${data.active_default || "(none)"}`, "ok");
    wiredLine.textContent = describeWired(data.wired);
    wifiLine.textContent = describeWifi(data.wifi);

    clearChildren(portalLink);
    const url = data.wifi && data.wifi.portal_url;
    if (url) {
      const label = document.createElement("span");
      label.textContent = "Captive portal: ";
      portalLink.appendChild(label);
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = url; // full URL, no truncation
      portalLink.appendChild(a);
    }
  }

  async function refreshProfiles() {
    const resp = await api("/api/admin/wan/profiles");
    if (!resp || !resp.ok) return;
    const data = await resp.json();
    clearChildren(profilesList);
    const profiles = data.profiles || [];
    for (const p of profiles) {
      const li = document.createElement("li");
      li.className = "settings-bt-row";

      const labelWrap = document.createElement("div");
      const ssid = document.createElement("strong");
      ssid.textContent = p.ssid;
      labelWrap.appendChild(ssid);
      const id = document.createElement("div");
      id.className = "settings-field-help";
      id.textContent = `id ${p.id}`; // FULL id, never truncated
      labelWrap.appendChild(id);
      if (p.psk_ref) {
        const ref = document.createElement("div");
        ref.className = "settings-field-help";
        ref.textContent = `psk_ref ${p.psk_ref}`;
        labelWrap.appendChild(ref);
      } else {
        const ref = document.createElement("div");
        ref.className = "settings-field-help";
        ref.textContent = "OPEN (no PSK)";
        labelWrap.appendChild(ref);
      }
      const band = (p.band || "auto").toLowerCase();
      const bandLabel = band === "a" ? "5 GHz only"
        : band === "bg" ? "2.4 GHz only"
        : "Band: Auto";
      const bandRow = document.createElement("div");
      bandRow.className = "settings-field-help";
      bandRow.textContent = bandLabel;
      labelWrap.appendChild(bandRow);
      if (p.bssid) {
        const b = document.createElement("div");
        b.className = "settings-field-help";
        b.textContent = `bssid ${p.bssid} (pinned)`;
        labelWrap.appendChild(b);
      }
      if (data.active_id === p.id) {
        const tag = document.createElement("span");
        tag.className = "settings-field-help";
        tag.textContent = " (active)";
        ssid.appendChild(tag);
      }
      li.appendChild(labelWrap);

      const actions = document.createElement("div");
      const setActive = document.createElement("button");
      setActive.type = "button";
      setActive.className = "btn-ghost";
      setActive.textContent = "Set active";
      setActive.addEventListener("click", async () => {
        const r = await api(`/api/admin/wan/profiles/${encodeURIComponent(p.id)}/set-active`, { method: "POST" });
        if (r && r.ok) refreshProfiles();
      });
      actions.appendChild(setActive);

      const del = document.createElement("button");
      del.type = "button";
      del.className = "btn-ghost";
      del.textContent = "Delete";
      del.addEventListener("click", async () => {
        const r = await api(`/api/admin/wan/profiles/${encodeURIComponent(p.id)}`, { method: "DELETE" });
        if (r && r.ok) refreshProfiles();
      });
      actions.appendChild(del);

      li.appendChild(actions);
      profilesList.appendChild(li);
    }
  }

  // Open-network mode is set by the scan "Use" button when the chosen
  // SSID is OPEN. The PSK input becomes disabled and the Add flow posts
  // ``open=true`` instead of a psk_ref. Tracked on the root element so
  // a re-render of the form doesn't lose state.
  function isOpenMode() {
    return root.dataset.openMode === "1";
  }

  function setOpenMode(on) {
    root.dataset.openMode = on ? "1" : "";
    if (!pskRefInput) return;
    pskRefInput.disabled = !!on;
    pskRefInput.placeholder = on ? "(open network — no PSK)" : "PSK_REF (SCREAMING_SNAKE)";
    if (on) pskRefInput.value = "";
  }

  async function addProfile() {
    const ssid = (ssidInput.value || "").trim();
    const pskRef = (pskRefInput.value || "").trim();
    const openMode = isOpenMode();
    if (!ssid) {
      setStatus(addStatus, "SSID is required", "err");
      return null;
    }
    if (!openMode && !pskRef) {
      setStatus(addStatus, "PSK_REF is required (or click Use on an OPEN network)", "err");
      return null;
    }
    const band = getBandValue();
    const body = openMode
      ? { ssid, open: true, band }
      : { ssid, psk_ref: pskRef, band };
    const r = await api("/api/admin/wan/profiles", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r) {
      setStatus(addStatus, "Network error", "err");
      return null;
    }
    if (r.status === 422) {
      const errBody = await r.json().catch(() => ({}));
      setStatus(addStatus, `psk_ref ${errBody.psk_ref || pskRef} not found in credentials store`, "err");
      return null;
    }
    if (!r.ok) {
      setStatus(addStatus, `Failed (${r.status})`, "err");
      return null;
    }
    const created = await r.json();
    return created.profile;
  }

  // SSID input edits by hand should drop the open-mode badge — the
  // operator is no longer using the OPEN network they picked from scan.
  if (ssidInput) {
    ssidInput.addEventListener("input", () => {
      if (isOpenMode()) setOpenMode(false);
    });
  }

  if (addBtn) addBtn.addEventListener("click", async () => {
    const prof = await addProfile();
    if (!prof) return;
    setStatus(addStatus, `Added (id ${prof.id}).`, "ok");
    ssidInput.value = "";
    pskRefInput.value = "";
    setOpenMode(false);
    refreshProfiles();
  });

  if (addAndConnectBtn) addAndConnectBtn.addEventListener("click", async () => {
    const prof = await addProfile();
    if (!prof) return;
    setStatus(addStatus, `Added; setting active and connecting…`, "warn");
    // Set the new profile active, then bring up. Two round-trips so
    // a failure mid-flow leaves clear state for the operator.
    const sa = await api(
      `/api/admin/wan/profiles/${encodeURIComponent(prof.id)}/set-active`,
      { method: "POST" }
    );
    if (!sa || !sa.ok) {
      setStatus(addStatus, "set-active failed", "err");
      refreshProfiles();
      return;
    }
    const up = await api("/api/admin/wan/up", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: prof.id }),
    });
    if (!up) {
      setStatus(addStatus, "WAN up: network error", "err");
    } else if (up.ok) {
      setStatus(addStatus, "WAN is up.", "ok");
      ssidInput.value = "";
      pskRefInput.value = "";
      setOpenMode(false);
    } else if (up.status === 422) {
      setStatus(addStatus, "PSK resolve failed — check psk_ref + age key.", "err");
    } else {
      setStatus(addStatus, `WAN up failed (${up.status})`, "err");
    }
    refreshProfiles();
    refreshStatus();
  });

  if (scanBtn) scanBtn.addEventListener("click", async () => {
    setStatus(scanStatus, "Scanning…", "warn");
    const r = await api("/api/admin/wan/scan");
    clearChildren(scanResults);
    if (!r || !r.ok) return setStatus(scanStatus, "Scan failed", "err");
    const data = await r.json();
    const networks = data.networks || [];
    setStatus(scanStatus, `Found ${networks.length} networks`, "ok");
    for (const n of networks) {
      const li = document.createElement("li");
      li.className = "settings-bt-row wan-scan-row";

      const labelWrap = document.createElement("div");
      const main = document.createElement("strong");
      main.textContent = n.ssid || "(hidden)";
      labelWrap.appendChild(main);

      const detail = document.createElement("div");
      detail.className = "settings-field-help";
      // Bands ("a"/"bg") -> "5 GHz"/"2.4 GHz"; show both if dual-band.
      const bandsHuman = (n.bands || [])
        .map(b => (b === "a" ? "5 GHz" : "2.4 GHz"))
        .join(" + ");
      const sec = n.security === "wpa" ? "WPA" : "OPEN";
      const sig = typeof n.best_signal_dbm === "number"
        ? `${Math.round(n.best_signal_dbm)} dBm`
        : "?";
      detail.textContent = `${bandsHuman}  ·  ${n.ap_count} AP${n.ap_count === 1 ? "" : "s"}  ·  ${sig}  ·  ${sec}`;
      labelWrap.appendChild(detail);
      li.appendChild(labelWrap);

      const actions = document.createElement("div");
      const useBtn = document.createElement("button");
      useBtn.type = "button";
      useBtn.className = "btn-ghost";
      useBtn.textContent = "Use";
      useBtn.addEventListener("click", () => {
        ssidInput.value = n.ssid || "";
        // Pre-select the strongest band (or Auto if dual-band is
        // available and roaming is fine). The radio still lets the
        // operator override.
        if ((n.bands || []).length > 1) {
          setBandValue("auto");
        } else if (n.best_signal_band === "a") {
          setBandValue("a");
        } else if (n.best_signal_band === "bg") {
          setBandValue("bg");
        }
        // Open networks flip the form into open mode: PSK input is
        // disabled, Add posts {open:true}. WPA networks keep the
        // password flow.
        if (n.security === "open") {
          setOpenMode(true);
          setStatus(addStatus, `Pre-filled OPEN network: ${n.ssid}`, "ok");
        } else {
          setOpenMode(false);
          if (pskRefInput) pskRefInput.focus();
          setStatus(addStatus, `Pre-filled from scan: ${n.ssid}`, "ok");
        }
      });
      actions.appendChild(useBtn);
      li.appendChild(actions);

      scanResults.appendChild(li);
    }
  });

  if (upBtn) upBtn.addEventListener("click", async () => {
    setStatus(actionStatus, "Bringing up active WAN…", "warn");
    // Look up active id from /profiles so the operator doesn't have to retype.
    const pres = await api("/api/admin/wan/profiles");
    if (!pres || !pres.ok) return setStatus(actionStatus, "No profile list", "err");
    const pdata = await pres.json();
    if (!pdata.active_id) return setStatus(actionStatus, "No active profile — use Set active", "err");
    const r = await api("/api/admin/wan/up", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: pdata.active_id }),
    });
    if (!r) return setStatus(actionStatus, "Network error", "err");
    if (r.ok) {
      setStatus(actionStatus, "WAN up.", "ok");
      refreshStatus();
    } else if (r.status === 422) {
      setStatus(actionStatus, "PSK resolve failed — check psk_ref + age key.", "err");
    } else {
      setStatus(actionStatus, `WAN up failed (${r.status})`, "err");
    }
  });

  if (downBtn) downBtn.addEventListener("click", async () => {
    setStatus(actionStatus, "Tearing down…", "warn");
    const r = await api("/api/admin/wan/down", { method: "POST" });
    if (r && r.ok) {
      setStatus(actionStatus, "WAN down.", "ok");
      refreshStatus();
    } else {
      setStatus(actionStatus, "Tear-down failed", "err");
    }
  });

  // ── Egress mode (block / gateway / captive) ────────────────
  const egressRadios = root.querySelectorAll('input[name="wan-egress-mode"]');
  const egressSourceLine = root.querySelector(".wan-egress-source");

  async function refreshEgressMode() {
    const r = await api("/api/admin/wan/mode");
    if (!r || !r.ok) {
      if (egressSourceLine) egressSourceLine.textContent = "source: unknown";
      return;
    }
    const data = await r.json();
    for (const radio of egressRadios) radio.checked = (radio.value === data.mode);
    if (egressSourceLine) {
      const label = data.source === "operator"
        ? "set by operator"
        : "shipped default";
      egressSourceLine.textContent = `source: ${label}`;
    }
  }

  for (const radio of egressRadios) {
    radio.addEventListener("change", async () => {
      if (!radio.checked) return;
      const r = await api("/api/admin/wan/mode", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ mode: radio.value }),
      });
      if (!r) {
        setStatus(addStatus, "Mode change: network error", "err");
        await refreshEgressMode();
        return;
      }
      if (!r.ok) {
        setStatus(addStatus, `Mode change failed (${r.status})`, "err");
        await refreshEgressMode();
        return;
      }
      const data = await r.json();
      if (data.warning) {
        setStatus(addStatus, data.warning, "warn");
      } else {
        setStatus(addStatus, `Egress mode now ${data.mode}.`, "ok");
      }
      await refreshEgressMode();
      await refreshStatus();
    });
  }

  // First load + 30s poll.
  refreshStatus();
  refreshProfiles();
  refreshEgressMode();
  pollHandle = window.setInterval(() => {
    refreshStatus();
  }, 30000);

  window.addEventListener("beforeunload", () => {
    if (pollHandle) window.clearInterval(pollHandle);
  });
})();
