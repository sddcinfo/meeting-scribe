// Meeting Scribe — Finalize summary renderer.
//
// Paints the post-meeting summary card (executive summary, key
// insights, named entities, action items, decisions, quotes, topics,
// speaker stats, download buttons, Q&A panel) into the
// #finalize-summary slot.
//
// Pure rendering — takes the summary payload + meeting id + a small
// hooks object so the module has no back-deps on the admin SPA's
// language helpers or meetings manager.

import { esc } from "../lib/escape.js";
import { closeModal } from "./modal-system.js";
import { initQaPanel } from "./qa-panel.js";

const _enc = encodeURIComponent;

/**
 * @param {object} summary — server-emitted summary JSON.
 * @param {string} meetingId — meeting id (uuid).
 * @param {object} [hooks]
 * @param {string} [hooks.langA] — bilingual A code (e.g. "en").
 * @param {string} [hooks.langB] — bilingual B code (e.g. "ja").
 * @param {(meetingId: string) => void} [hooks.onViewMeeting] —
 *   invoked when the operator clicks "View Meeting".
 */
/**
 * Inject the "View Summary" entry-point button into the past-meeting
 * review toolbar (#meeting-summary-panel > .summary-bar-tools). The
 * finalization modal is the single source of truth for summary content;
 * this button is just the entry point.
 *
 * Idempotent — calling it twice (e.g. after re-finalize) does NOT
 * double-render the button.
 *
 * @param {object} summary — server-emitted summary JSON (or null/error).
 * @param {string} meetingId — meeting id (uuid).
 * @param {object} [hooks]
 * @param {(meetingId: string) => void} [hooks.onOpenSummary] —
 *   invoked when the operator clicks "View Summary"
 *   (typically `showFinalizationSummaryFor(meetingId)`).
 */
export function renderSummaryPanel(summary, meetingId, hooks = {}) {
  const panel = document.getElementById("meeting-summary-panel");
  if (!panel || !summary || summary.error) return;
  const tools = panel.querySelector(".summary-bar-tools");
  if (!tools) return;

  // Don't inject twice if called again (e.g. after re-finalize)
  if (tools.querySelector("#summary-bar-open")) return;

  const btn = document.createElement("button");
  btn.className = "btn-ghost summary-bar-btn";
  btn.id = "summary-bar-open";
  btn.title = "Open the full meeting summary with topics, decisions, action items, and downloads";
  btn.textContent = "View Summary";
  btn.addEventListener("click", () => hooks.onOpenSummary?.(meetingId));
  tools.appendChild(btn);
}

export function renderFinalizationSummary(summary, meetingId, hooks = {}) {
  const el = document.getElementById("finalize-summary");
  if (!el) return;
  if (summary.error) {
    el.innerHTML = `<p class="finalize-error">${esc(summary.error || "Summary generation failed")}</p>`;
    el.style.display = "";
    return;
  }

  // Collapse the progress steps with animation
  const stepsEl = el.closest(".finalize-modal")?.querySelector("#finalize-steps");
  if (stepsEl) stepsEl.classList.add("collapsed");
  const etaEl = el.closest(".finalize-modal")?.querySelector("#finalize-eta");
  if (etaEl) etaEl.classList.remove("visible");

  const meta = summary.metadata || {};
  const durationMin = meta.duration_min || 0;
  const durationStr =
    durationMin >= 60
      ? `${Math.floor(durationMin / 60)}h ${Math.round(durationMin % 60)}m`
      : `${Math.round(durationMin)}m`;
  const numSpeakers = meta.num_speakers || (summary.speaker_stats || []).length || 0;
  const isV2 = !!summary.key_insights;
  const langA = (hooks.langA || "en").toLowerCase();
  const langB = (hooks.langB || "ja").toLowerCase();

  // ── Shared components ──
  const speakerColors = [
    "var(--speaker-1)",
    "var(--speaker-2)",
    "var(--speaker-3)",
    "var(--speaker-4)",
    "#8b6914",
    "#b52d2d",
  ];
  const stats = (summary.speaker_stats || [])
    .map((s, i) => {
      const barColor = speakerColors[i % speakerColors.length];
      return `<div class="speaker-stat" style="animation-delay:${i * 80}ms">
      <div class="speaker-stat-dot" style="background:${barColor}"></div>
      <span class="speaker-stat-name">${esc(s.name)}</span>
      <div class="speaker-stat-bar">
        <div class="speaker-stat-fill" style="width:${Math.max(3, s.pct)}%;background:${barColor}"></div>
      </div>
      <span class="speaker-stat-pct">${Math.round(s.pct)}%</span>
    </div>`;
    })
    .join("");

  let mainContent = "";

  if (isV2) {
    // ── V2: Rich insights, categorized actions, named entities, attributed quotes ──
    const insights = (summary.key_insights || [])
      .map((insight, i) => {
        const speakerPills = (insight.speakers || [])
          .map((sp) => `<span class="insight-speaker-pill">${esc(sp)}</span>`)
          .join("");
        const descParas = esc(insight.description || "")
          .split(/\n\n|\n/)
          .filter(Boolean)
          .map((p) => `<p>${p}</p>`)
          .join("");
        return `<div class="insight-card" style="animation-delay:${i * 80}ms">
        <div class="insight-header">
          <span class="insight-number">${String(i + 1).padStart(2, "0")}</span>
          <h5 class="insight-title">${esc(insight.title)}</h5>
        </div>
        <div class="insight-body">${descParas}</div>
        ${speakerPills ? `<div class="insight-speakers">${speakerPills}</div>` : ""}
      </div>`;
      })
      .join("");

    const entities = summary.named_entities || {};
    const entitySections = [];
    for (const [category, items] of Object.entries(entities)) {
      if (items && items.length > 0) {
        const label = category.charAt(0).toUpperCase() + category.slice(1);
        const pills = items
          .map(
            (item) =>
              `<span class="entity-pill entity-${esc(category)}">${esc(item)}</span>`,
          )
          .join("");
        entitySections.push(`<div class="entity-group">
          <span class="entity-label">${esc(label)}</span>
          <div class="entity-pills">${pills}</div>
        </div>`);
      }
    }
    const entitiesHtml = entitySections.join("");

    const actionsByCategory = {};
    for (const a of summary.action_items || []) {
      const cat = a.category || "General";
      if (!actionsByCategory[cat]) actionsByCategory[cat] = [];
      actionsByCategory[cat].push(a);
    }
    let actionIdx = 0;
    const categorizedActions = Object.entries(actionsByCategory)
      .map(([cat, items]) => {
        const itemsHtml = items
          .map((a) => {
            actionIdx++;
            return `<li class="action-item" style="animation-delay:${actionIdx * 50}ms">
          <span class="action-check"></span>
          <div class="action-body">
            <span>${esc(a.task)}</span>
            ${a.assignee ? `<span class="action-assignee">${esc(a.assignee)}</span>` : ""}
          </div>
        </li>`;
          })
          .join("");
        return `<div class="action-category-group">
        <div class="action-category-header">${esc(cat)}</div>
        <ul class="action-list">${itemsHtml}</ul>
      </div>`;
      })
      .join("");

    const quotes = (summary.key_quotes || [])
      .map((q, i) => {
        const text = typeof q === "string" ? q : q.text;
        const speaker = typeof q === "object" ? q.speaker : null;
        const context = typeof q === "object" ? q.context : null;
        return `<div class="quote-card" style="animation-delay:${i * 60}ms">
        <div class="quote-mark">&ldquo;</div>
        <div class="quote-body">
          <p class="quote-text">${esc(text || "")}</p>
          ${
            speaker || context
              ? `<div class="quote-attribution">
            ${speaker ? `<span class="quote-speaker">${esc(speaker)}</span>` : ""}
            ${context ? `<span class="quote-context">${esc(context)}</span>` : ""}
          </div>`
              : ""
          }
        </div>
      </div>`;
      })
      .join("");

    const decisions = (summary.decisions || [])
      .map(
        (d, i) =>
          `<li style="animation-delay:${i * 60}ms"><span class="decision-bullet"></span>${esc(d)}</li>`,
      )
      .join("");

    const topics = (summary.topics || [])
      .map(
        (t, i) =>
          `<li class="topic-item" style="animation-delay:${i * 60}ms">
        <span class="topic-marker">${String(i + 1).padStart(2, "0")}</span>
        <div>
          <strong>${esc(t.title)}</strong>
          <p>${esc(t.description || "")}</p>
        </div>
      </li>`,
      )
      .join("");

    mainContent = `
      <div class="summary-scroll summary-v2">
        <div class="summary-section summary-overview">
          <h4>Executive Summary</h4>
          <p>${esc(summary.executive_summary || "")}</p>
        </div>
        ${entitiesHtml ? `<div class="summary-section summary-entities"><h4>Mentioned</h4><div class="entities-grid">${entitiesHtml}</div></div>` : ""}
        ${insights ? `<div class="summary-section summary-insights"><h4>Key Insights</h4><div class="insights-grid">${insights}</div></div>` : ""}
        ${categorizedActions ? `<div class="summary-section summary-actions-section"><h4>Action Items</h4>${categorizedActions}</div>` : ""}
        ${decisions ? `<div class="summary-section"><h4>Decisions</h4><ul class="decision-list">${decisions}</ul></div>` : ""}
        ${quotes ? `<div class="summary-section summary-quotes"><h4>Key Quotes</h4><div class="quotes-grid">${quotes}</div></div>` : ""}
        ${topics ? `<div class="summary-section"><h4>Topics</h4><ul class="topic-list">${topics}</ul></div>` : ""}
        ${stats ? `<div class="summary-section"><h4>Speaker Participation</h4><div class="speaker-stats-grid">${stats}</div></div>` : ""}
      </div>`;
  } else {
    // ── V1 fallback: simple flat rendering for legacy summary payloads ──
    const topics = (summary.topics || [])
      .map(
        (t, i) =>
          `<li class="topic-item" style="animation-delay:${i * 60}ms">
        <span class="topic-marker">${String(i + 1).padStart(2, "0")}</span>
        <div>
          <strong>${esc(t.title)}</strong>
          <p>${esc(t.description || "")}</p>
        </div>
      </li>`,
      )
      .join("");

    const decisions = (summary.decisions || [])
      .map(
        (d, i) =>
          `<li style="animation-delay:${i * 60}ms"><span class="decision-bullet"></span>${esc(d)}</li>`,
      )
      .join("");

    const actions = (summary.action_items || [])
      .map(
        (a, i) =>
          `<li class="action-item" style="animation-delay:${i * 60}ms">
        <span class="action-check"></span>
        <div class="action-body">
          <span>${esc(a.task)}</span>
          ${a.assignee ? `<span class="action-assignee">${esc(a.assignee)}</span>` : ""}
        </div>
      </li>`,
      )
      .join("");

    mainContent = `
      <div class="summary-scroll">
        <div class="summary-section summary-overview">
          <h4>Summary</h4>
          <p>${esc(summary.executive_summary || "")}</p>
        </div>
        ${topics ? `<div class="summary-section"><h4>Topics</h4><ul class="topic-list">${topics}</ul></div>` : ""}
        ${decisions ? `<div class="summary-section"><h4>Decisions</h4><ul class="decision-list">${decisions}</ul></div>` : ""}
        ${actions ? `<div class="summary-section"><h4>Action Items</h4><ul class="action-list">${actions}</ul></div>` : ""}
        ${stats ? `<div class="summary-section"><h4>Speaker Participation</h4><div class="speaker-stats-grid">${stats}</div></div>` : ""}
      </div>`;
  }

  el.innerHTML = `
    <div class="summary-meta-bar">
      <span class="meta-chip">${durationStr}</span>
      <span class="meta-chip">${numSpeakers} speaker${numSpeakers !== 1 ? "s" : ""}</span>
      <span class="meta-chip">${meta.num_segments || "—"} segments</span>
      <span class="meta-chip">${(meta.languages || []).map((l) => l.toUpperCase()).join(" / ")}</span>
      ${meetingId ? `<span class="meta-chip meta-chip-id" id="finalize-meeting-id" title="Click to copy meeting ID" style="cursor:pointer;font-family:ui-monospace,Menlo,monospace">id: ${esc(meetingId)}</span>` : ""}
    </div>
    ${mainContent}
    <div class="summary-qa" id="summary-qa">
      <div class="qa-messages" id="qa-messages"></div>
      <div class="qa-input-row">
        <input type="text" id="qa-input" placeholder="Ask about this meeting..." autocomplete="off" />
        <button class="qa-send-btn" id="qa-send">Ask</button>
      </div>
    </div>
    <div class="summary-actions">
      <div class="summary-actions-left">
        <button class="modal-btn btn-download" onclick="window.open('/api/meetings/${_enc(meetingId)}/export?format=md')">Markdown</button>
        <button class="modal-btn btn-download" onclick="window.open('/api/meetings/${_enc(meetingId)}/export?format=txt&lang=${langA}')">${langA.toUpperCase()}</button>
        <button class="modal-btn btn-download" onclick="window.open('/api/meetings/${_enc(meetingId)}/export?format=txt&lang=${langB}')">${langB.toUpperCase()}</button>
        <button class="modal-btn btn-download" onclick="window.open('/api/meetings/${_enc(meetingId)}/export?format=zip')">ZIP</button>
      </div>
      <div class="summary-actions-right">
        <button class="modal-btn btn-primary" id="finalize-done-btn">View Meeting</button>
      </div>
    </div>
  `;
  el.style.display = "";

  initQaPanel(meetingId);

  el.querySelector("#finalize-done-btn")?.addEventListener("click", () => {
    closeModal();
    if (meetingId && typeof hooks.onViewMeeting === "function") {
      setTimeout(() => hooks.onViewMeeting(meetingId), 300);
    }
  });

  const idChip = el.querySelector("#finalize-meeting-id");
  if (idChip && meetingId) {
    idChip.addEventListener("click", async () => {
      const originalText = idChip.textContent;
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(meetingId);
        } else {
          const ta = document.createElement("textarea");
          ta.value = meetingId;
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          ta.remove();
        }
        idChip.textContent = "copied ✓";
        setTimeout(() => {
          idChip.textContent = originalText;
        }, 1200);
      } catch {
        idChip.textContent = "copy failed";
        setTimeout(() => {
          idChip.textContent = originalText;
        }, 1200);
      }
    });
  }
}
