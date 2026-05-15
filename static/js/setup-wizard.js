/* Meeting Scribe — Setup wizard client (single-page, no rotation)
 *
 * Wires tap-to-copy on every `.cred-row` (the whole row is the tap
 * target, even when the user lands a finger off the value text)
 * and the Done / Cancel buttons. POST /api/setup/finish marks
 * setup-complete; after that the box no longer routes through
 * the wizard.
 *
 * Pure browser, zero remote dependencies.
 */

'use strict';

async function postJson(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  let data = null;
  try { data = await resp.json(); } catch (_) { /* empty */ }
  return { ok: resp.ok, status: resp.status, data: data || {} };
}

function setStatus(text, kind) {
  const node = document.getElementById('status');
  if (!node) return;
  node.textContent = text || '';
  node.className = '';
  if (kind) node.classList.add(kind);
}

function copyValueOf(row) {
  const target = row.querySelector('.copy-target');
  if (!target) return null;
  return target.dataset.copy || target.textContent.trim();
}

function flashCopied(row) {
  row.classList.add('copied');
  setTimeout(() => row.classList.remove('copied'), 1800);
}

function fallbackSelect(row) {
  const target = row.querySelector('.copy-target');
  if (!target) return;
  const range = document.createRange();
  range.selectNodeContents(target);
  const sel = window.getSelection();
  if (!sel) return;
  sel.removeAllRanges();
  sel.addRange(range);
}

function wireCopyTargets() {
  for (const row of document.querySelectorAll('.cred-row')) {
    row.setAttribute('role', 'button');
    row.setAttribute('tabindex', '0');
    const handler = async () => {
      const value = copyValueOf(row);
      if (!value) return;
      try {
        await navigator.clipboard.writeText(value);
        flashCopied(row);
      } catch (_) {
        fallbackSelect(row);
      }
    };
    row.addEventListener('click', handler);
    row.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        handler();
      }
    });
  }
}

async function confirmDialog(message) {
  // Tiny native-<dialog> confirm helper — the wizard is a standalone
  // page with no dependency on the admin SPA's modal stack. Returns a
  // Promise<boolean>.
  return new Promise((resolve) => {
    const dlg = document.createElement('dialog');
    dlg.className = 'wizard-confirm-dialog';
    dlg.innerHTML = `
      <p class="wizard-confirm-message"></p>
      <div class="wizard-confirm-actions">
        <button type="button" class="btn-secondary" data-confirm-cancel>Keep going</button>
        <button type="button" class="btn-primary" data-confirm-ok>Cancel setup</button>
      </div>`;
    dlg.querySelector('.wizard-confirm-message').textContent = message;
    document.body.appendChild(dlg);
    const finish = (ok) => {
      try { dlg.close(); } catch (_) { /* noop */ }
      dlg.remove();
      resolve(ok);
    };
    dlg.querySelector('[data-confirm-ok]').addEventListener('click', () => finish(true));
    dlg.querySelector('[data-confirm-cancel]').addEventListener('click', () => finish(false));
    dlg.addEventListener('cancel', (ev) => { ev.preventDefault(); finish(false); });
    dlg.showModal();
  });
}

async function cancelSetup() {
  if (!(await confirmDialog('Cancel setup and start over?'))) return;
  await postJson('/api/setup/cancel');
  window.location.assign('/setup');
}

async function finishSetup() {
  const finishBtn = document.getElementById('finish');
  if (finishBtn) finishBtn.disabled = true;

  // Submit hidden autofill form to trigger the browser's
  // password-save prompt.
  const form = document.getElementById('autofill-form');
  if (form && typeof form.requestSubmit === 'function') {
    try { form.requestSubmit(); } catch (_) { /* noop */ }
  }

  const { ok, status, data } = await postJson('/api/setup/finish');
  if (!ok) {
    setStatus(data.error || ('Server returned ' + status), 'error');
    if (finishBtn) finishBtn.disabled = false;
    return;
  }

  setStatus('Setup complete — taking you to sign-in.', 'ok');
  setTimeout(() => window.location.assign('/auth'), 1500);
}

function boot() {
  wireCopyTargets();
  const finishBtn = document.getElementById('finish');
  if (finishBtn) finishBtn.addEventListener('click', finishSetup);
  const cancelBtn = document.getElementById('cancel');
  if (cancelBtn) cancelBtn.addEventListener('click', cancelSetup);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
