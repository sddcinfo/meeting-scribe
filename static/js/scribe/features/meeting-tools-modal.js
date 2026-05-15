// Meeting Scribe — meeting-tools + slide-assets modals.
//
// Two modal renderers triggered from the meetings panel's per-row
// ⋯ menu:
//
//   openSlideAssetsModal(meeting)
//     → "Download / open" the meeting's slide artifacts: source
//       .pptx, original .pdf, translated .pdf. Probes the translated
//       PDF with HEAD before exposing the button (monolingual decks
//       and unfinished bulk renders disable that row).
//
//   openMeetingToolsModal(meeting, mgr)
//     → Re-diarize / Reprocess / Versions / Delete actions on a past
//       meeting. Re-diarize and Reprocess prompt for an optional
//       expected speaker count; Versions opens a diff view; Delete
//       requires a destructive confirm before tearing down the
//       meeting directory. `mgr` is the MeetingsManager instance —
//       called as `mgr?.refresh()` after each mutation.
//
// Pure module: every dep is imported (esc, _enc, modal-system).

import { esc } from "../lib/escape.js";
import { _enc } from "../lib/meeting-url.js";
import {
  showModal,
  closeModal,
  closeAllModals,
  alertDialog,
  confirmDialog,
  promptDialog,
} from "./modal-system.js";

export function openSlideAssetsModal(meeting) {
  const m = meeting || {};
  const mid = m.meeting_id || "";
  const srcPptxUrl = `/api/meetings/${_enc(mid)}/slides/source.pptx`;
  const origPdfUrl = `/api/meetings/${_enc(mid)}/slides/original.pdf`;
  const transPdfUrl = `/api/meetings/${_enc(mid)}/slides/translated.pdf`;

  const card = showModal(
    `
    <div class="finalize-modal">
      <div class="finalize-header">
        <div class="finalize-header-content">
          <div>
            <h3>Slide deck</h3>
            <p class="finalize-subtitle">${esc(mid)}</p>
          </div>
        </div>
        <button class="finalize-close" id="slides-close-btn" title="Close">&times;</button>
      </div>
      <div class="finalize-summary">
        <div class="meeting-tool-list">
          <div class="meeting-tool-item">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Original deck (PPTX)</div>
              <div class="meeting-tool-desc">
                The unmodified source file the presenter uploaded.
              </div>
            </div>
            <a class="modal-btn" href="${srcPptxUrl}" download="presentation.pptx">Download</a>
          </div>
          <div class="meeting-tool-item">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Original deck (PDF)</div>
              <div class="meeting-tool-desc">
                Lossless render of the source deck. Opens in a new tab.
              </div>
            </div>
            <button class="modal-btn" data-action="pdf">Open</button>
          </div>
          <div class="meeting-tool-item" data-row="translated">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Translated deck (PDF)</div>
              <div class="meeting-tool-desc" id="slides-translated-desc">
                Lossless render of the deck with translated text reinserted
                into each shape. Useful for handing off to a presenter.
              </div>
            </div>
            <button class="modal-btn" id="slides-translated-btn" data-action="translated-pdf">Open</button>
          </div>
        </div>
      </div>
    </div>
  `,
    "finalize",
  );

  card.querySelector("#slides-close-btn")?.addEventListener("click", closeModal);
  card.querySelector('[data-action="pdf"]')?.addEventListener("click", () => {
    window.open(origPdfUrl, "_blank", "noopener");
  });
  card.querySelector('[data-action="translated-pdf"]')?.addEventListener("click", () => {
    window.open(transPdfUrl, "_blank", "noopener");
  });

  // Probe the translated PDF — disable the row when the file is absent
  // (monolingual deck, or the bulk render hasn't completed yet).
  fetch(transPdfUrl, { method: "HEAD" })
    .then((resp) => {
      if (resp.ok) return;
      const btn = card.querySelector("#slides-translated-btn");
      const desc = card.querySelector("#slides-translated-desc");
      if (btn) {
        btn.classList.add("disabled");
        btn.setAttribute("aria-disabled", "true");
        btn.textContent = "Unavailable";
        btn.onclick = (e) => {
          e.preventDefault();
        };
      }
      if (desc) {
        desc.textContent =
          resp.status === 404
            ? "Not available for this deck (monolingual, or translation not produced yet)."
            : `Not available right now (HTTP ${resp.status}).`;
      }
    })
    .catch(() => {
      /* network blip — leave button enabled, fall through to open */
    });
}

// ── Meeting tools modal ──────────────────────────────────────
// Houses Re-diarize, Reprocess, Versions, Delete behind a single ⋯
// button on each meeting row. Each action is presented with a clear
// description, an estimate of how long it takes, and a confirm step
// for destructive ones. Keeps the row clean and the consequences obvious.
export function openMeetingToolsModal(meeting, mgr) {
  const m = meeting || {};
  const card = showModal(
    `
    <div class="finalize-modal">
      <div class="finalize-header">
        <div class="finalize-header-content">
          <div>
            <h3>Meeting actions</h3>
            <p class="finalize-subtitle">${esc(m.meeting_id || "")}</p>
          </div>
        </div>
        <button class="finalize-close" id="tools-close-btn" title="Close">&times;</button>
      </div>
      <div class="finalize-summary" id="tools-body">
        <div class="meeting-tool-list">
          <div class="meeting-tool-item" data-action="rediarize">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Re-diarize speakers</div>
              <div class="meeting-tool-desc">
                Re-runs full-audio diarization + speaker consolidation on the existing
                transcript. Use when speaker labels look wrong (over-clustered, missing
                speakers, fragments). Optional: pin the expected speaker count.
                <span class="meeting-tool-meta">~2-3 min for 60-min audio · keeps a snapshot</span>
              </div>
            </div>
            <button class="modal-btn" data-action="rediarize-go">Run</button>
          </div>
          <div class="meeting-tool-item" data-action="reprocess">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Full reprocess from raw audio</div>
              <div class="meeting-tool-desc">
                Re-runs ASR + translation + diarization end-to-end from
                <code>recording.pcm</code>. Use when transcript text quality is poor
                (e.g. wrong language detected). Auto-snapshots the current run so you
                can compare via Versions afterwards.
                <span class="meeting-tool-meta">~10-15 min for 60-min audio · destructive but versioned</span>
              </div>
            </div>
            <button class="modal-btn" data-action="reprocess-go">Run</button>
          </div>
          <div class="meeting-tool-item" data-action="versions">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Versions / Compare runs</div>
              <div class="meeting-tool-desc">
                Each reprocess auto-snapshots the prior outputs. View past versions and
                see a per-dimension diff (segment count, language tags, translation
                coverage, speaker count, summary structure) so you can judge whether a
                change actually improved quality.
              </div>
            </div>
            <button class="modal-btn" data-action="versions-go">Open</button>
          </div>
          <div class="meeting-tool-item meeting-tool-danger" data-action="delete">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Delete meeting</div>
              <div class="meeting-tool-desc">
                Permanently removes the meeting directory — transcript, audio, slides,
                summary, all snapshots. Cannot be undone.
              </div>
            </div>
            <button class="modal-btn modal-btn-danger" data-action="delete-go">Delete</button>
          </div>
        </div>
      </div>
    </div>
  `,
    "finalize",
  );
  card.querySelector("#tools-close-btn")?.addEventListener("click", closeModal);

  async function _runRediarize() {
    const raw = await promptDialog(
      "Re-diarize meeting",
      "Pin a speaker count when known (recommended for over-clustered meetings). Leave blank to let the model decide.",
      {
        placeholder: "Speaker count (1–12) or blank",
        confirmText: "Re-diarize",
        type: "number",
        inputMode: "numeric",
        min: 1,
        max: 12,
        help: "Runs diarization + speaker consolidation on the existing transcript. Keeps a snapshot.",
      },
    );
    if (raw === null) return;
    const expected = raw === "" ? null : parseInt(raw, 10);
    if (raw !== "" && (!Number.isFinite(expected) || expected < 1 || expected > 12)) {
      await alertDialog(
        "Invalid count",
        "Speaker count must be a number between 1 and 12, or blank.",
      );
      return;
    }
    closeAllModals();
    try {
      const qs = expected != null ? `?expected_speakers=${expected}` : "";
      const resp = await fetch(`/api/meetings/${_enc(m.meeting_id)}/finalize${qs}`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error(await resp.text());
      const result = await resp.json();
      mgr?.refresh();
      await alertDialog(
        "Re-diarize complete",
        `${result?.diarization?.unique_speakers ?? "?"} speakers detected from ${result?.diarization?.segments ?? "?"} diarize segments.`,
      );
    } catch (err) {
      await alertDialog("Re-diarize failed", String(err.message || err));
    }
  }

  async function _runReprocess() {
    const raw = await promptDialog(
      "Full reprocess from raw audio",
      "Re-runs ASR + translation + diarization for a higher-quality transcript. Slow: about 10–15 minutes for a 60-minute meeting. Snapshots the current journal automatically.",
      {
        placeholder: "Speaker count (1–12) or blank",
        confirmText: "Reprocess",
        type: "number",
        inputMode: "numeric",
        min: 1,
        max: 12,
        help: "Pin a speaker count when known, or leave blank to let pyannote decide.",
      },
    );
    if (raw === null) return;
    const expected = raw === "" ? null : parseInt(raw, 10);
    if (raw !== "" && (!Number.isFinite(expected) || expected < 1 || expected > 12)) {
      await alertDialog(
        "Invalid count",
        "Speaker count must be a number between 1 and 12, or blank.",
      );
      return;
    }
    closeAllModals();
    try {
      const qs = expected != null ? `?expected_speakers=${expected}` : "";
      const resp = await fetch(`/api/meetings/${_enc(m.meeting_id)}/reprocess${qs}`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error(await resp.text());
      const result = await resp.json();
      mgr?.refresh();
      const segs = result?.segments ?? "?";
      const tr = result?.translated ?? "?";
      const sp = result?.speakers ?? "?";
      await alertDialog(
        "Reprocess complete",
        `${segs} segments, ${tr} translated, ${sp} speakers detected. The previous run is saved as a version; open Tools > Versions to compare.`,
      );
    } catch (err) {
      await alertDialog("Reprocess failed", String(err.message || err));
    }
  }

  async function _showVersions() {
    closeAllModals();
    let resp;
    try {
      resp = await fetch(`/api/meetings/${_enc(m.meeting_id)}/versions`);
    } catch (e) {
      await alertDialog("Versions unavailable", String(e));
      return;
    }
    const data = await resp.json().catch(() => ({}));
    const versions = (data && data.versions) || [];
    if (!versions.length) {
      await alertDialog(
        "No versions yet",
        "Run a reprocess first. Each reprocess auto-snapshots the prior run so you can compare them here.",
      );
      return;
    }
    let diffHtml = "";
    try {
      const dResp = await fetch(`/api/meetings/${_enc(m.meeting_id)}/versions/diff`);
      if (dResp.ok) {
        const dData = await dResp.json();
        const dims = (dData && dData.diff && dData.diff.dimensions) || {};
        const totals = (dData && dData.diff && dData.diff.totals) || {};
        const sign = { better: "▲", worse: "▼", same: "·" };
        const color = { better: "#10b981", worse: "#ef4444", same: "#9a9aa2" };
        const rows = Object.entries(dims)
          .map(([k, v]) => {
            const pct = (v.delta_rel * 100).toFixed(1);
            return `<tr>
            <td style="padding:0.25rem 0.5rem;color:${color[v.verdict]}">${sign[v.verdict]}</td>
            <td style="padding:0.25rem 0.5rem;font-family:var(--font-mono);font-size:0.78rem">${esc(k)}</td>
            <td style="padding:0.25rem 0.5rem;text-align:right;font-family:var(--font-mono)">${esc(String(v.baseline))}</td>
            <td style="padding:0.25rem 0.5rem;text-align:center;color:#9a9aa2">→</td>
            <td style="padding:0.25rem 0.5rem;text-align:right;font-family:var(--font-mono)">${esc(String(v.compare))}</td>
            <td style="padding:0.25rem 0.5rem;text-align:right;color:${color[v.verdict]};font-family:var(--font-mono)">${pct}%</td>
          </tr>`;
          })
          .join("");
        diffHtml = `
          <div style="margin-top:1rem">
            <div style="font-weight:600;margin-bottom:0.4rem">Diff: ${esc(dData.baseline)} → ${esc(dData.compare)}</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">${rows}</table>
            <div style="margin-top:0.5rem;font-size:0.75rem;color:var(--text-secondary)">
              <span style="color:#10b981">▲ ${totals.better || 0} better</span> ·
              <span style="color:#ef4444">▼ ${totals.worse || 0} worse</span> ·
              <span style="color:#9a9aa2">· ${totals.same || 0} same</span>
            </div>
          </div>`;
      }
    } catch {}
    const versionsHtml = versions
      .map((v) => {
        const meta = v.manifest || {};
        const inputs = meta.inputs || {};
        return `<div style="padding:0.4rem 0.5rem;border-bottom:1px solid var(--border);font-size:0.78rem">
        <div style="font-family:var(--font-mono);font-weight:600">${esc(v.name)}</div>
        <div style="color:var(--text-secondary);margin-top:0.15rem">
          ${esc(meta.snapshot_at_utc || "")} · git=${esc((meta.git_commit || "").slice(0, 8) || "-")} · pair=${esc((inputs.language_pair || []).join(","))} · expected_speakers=${esc(String(inputs.expected_speakers || "-"))}
        </div>
      </div>`;
      })
      .join("");
    showModal(
      `
      <div class="finalize-modal">
        <div class="finalize-header">
          <div class="finalize-header-content"><div><h3>Versions</h3><p class="finalize-subtitle">${esc(m.meeting_id || "")}</p></div></div>
          <button class="finalize-close" onclick="closeModal()" title="Close">&times;</button>
        </div>
        <div class="finalize-summary">
          ${diffHtml}
          <div style="margin-top:1rem;font-weight:600">Snapshots (${versions.length})</div>
          ${versionsHtml}
        </div>
      </div>
    `,
      "finalize",
    );
  }

  async function _confirmDelete() {
    const go = await confirmDialog(
      "Delete meeting?",
      `<code style="font-family:var(--font-mono);font-size:0.82em">${esc(m.meeting_id || "")}</code><br><br>This permanently removes the transcript, audio, slides, summary, and all snapshots. The action cannot be undone.`,
      "Delete",
      true,
    );
    if (!go) return;
    closeAllModals();
    try {
      const r = await fetch(`/api/meetings/${_enc(m.meeting_id)}`, { method: "DELETE" });
      if (!r.ok) throw new Error(await r.text());
      mgr?.refresh();
    } catch (e) {
      await alertDialog("Delete failed", String(e.message || e));
      return;
    }
  }

  card.querySelector('[data-action="rediarize-go"]').addEventListener("click", _runRediarize);
  card.querySelector('[data-action="reprocess-go"]').addEventListener("click", _runReprocess);
  card.querySelector('[data-action="versions-go"]').addEventListener("click", _showVersions);
  card.querySelector('[data-action="delete-go"]').addEventListener("click", _confirmDelete);
}
