// components.js — utility/helper functions loaded first, before charts.js and app.js
// All code is global scope (no ES modules).
// NOTE: _globalAuthEnabled, _globalBindHost, and toggleFullscreen live in app.js

function $(id) {
  return document.getElementById(id);
}

// ── Toasts ───────────────────────────────────────────────────────────────────
// kinds: info | success | error | warning. Dismissable, capped stack.
function toast(msg, kind = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  // Cap the stack at 5 — drop the oldest.
  while (container.children.length >= 5) container.firstChild.remove();
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  const text = document.createElement('span');
  text.textContent = msg;
  const close = document.createElement('button');
  close.className = 'toast-close';
  close.setAttribute('aria-label', 'Dismiss');
  close.textContent = '✕';
  close.onclick = function () { dismiss(); };
  el.appendChild(text);
  el.appendChild(close);
  container.appendChild(el);
  let gone = false;
  function dismiss() {
    if (gone) return;
    gone = true;
    el.classList.add('leaving');
    setTimeout(() => el.remove(), 250);
  }
  setTimeout(dismiss, kind === 'error' ? 8000 : 4000);
}

function fmt(n) {
  if (typeof n !== 'number') return n;
  return n.toLocaleString();
}

function fmtMib(n) {
  if (n == null) return '—';
  return Math.round(n).toLocaleString() + ' MiB';
}

function fmtSize(n) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return n.toFixed(1) + ' ' + units[i];
}

function fmtTimings(t) {
  if (t.error) {
    return '<div class="error-text">Error: ' + t.error + '</div>';
  }
  return '<table class="kv-table" style="margin-top:8px">' +
    '<tr><td>Prompt tokens</td><td>' + fmt(t.prompt_tokens) + '</td></tr>' +
    '<tr><td>Generated tokens</td><td>' + fmt(t.completion_tokens) + '</td></tr>' +
    '<tr><td>Prompt speed (tok/s)</td><td>' + (t.prompt_per_second != null ? Number(t.prompt_per_second).toFixed(1) : '—') + '</td></tr>' +
    '<tr><td>Generation speed (tok/s)</td><td>' + (t.predicted_per_second != null ? Number(t.predicted_per_second).toFixed(1) : '—') + '</td></tr>' +
    '</table>';
}

// ── Modal ────────────────────────────────────────────────────────────────────
function showModal(html) {
  let m = document.getElementById('modal');
  if (!m) {
    m = document.createElement('div');
    m.id = 'modal';
    document.body.appendChild(m);
  }
  m.className = 'modal';
  m.style.display = 'flex';
  document.body.classList.add('modal-open');
  m.innerHTML =
    '<div>' +
    '<button onclick="closeModal()" class="btn-muted" style="float:right;margin-bottom:8px">Close</button>' +
    html +
    '</div>';
  requestAnimationFrame(function () {
    const box = m.firstElementChild;
    if (box) box.classList.add('modal-open');
  });
}

function closeModal() {
  const m = document.getElementById('modal');
  if (m) m.style.display = 'none';
  document.body.classList.remove('modal-open');
}

document.addEventListener('keydown', function (e) {
  if (e.key !== 'Escape') return;
  const m = document.getElementById('modal');
  if (m && m.style.display !== 'none') closeModal();
  const c = document.getElementById('confirmOverlay');
  if (c) c._cancel && c._cancel();
});

// ── confirmDialog — styled, promise-based replacement for confirm() ─────────
// Usage: if (!(await confirmDialog('Delete this model?'))) return;
function confirmDialog(message, opts) {
  opts = opts || {};
  return new Promise(function (resolve) {
    const old = document.getElementById('confirmOverlay');
    if (old) old.remove();
    const overlay = document.createElement('div');
    overlay.id = 'confirmOverlay';
    overlay.className = 'modal';
    overlay.style.display = 'flex';

    const box = document.createElement('div');
    box.className = 'confirm-dialog';
    const title = document.createElement('h3');
    title.textContent = opts.title || 'Are you sure?';
    const body = document.createElement('p');
    body.textContent = message;
    const footer = document.createElement('div');
    footer.className = 'btn-group confirm-footer';
    const cancel = document.createElement('button');
    cancel.className = 'btn-muted';
    cancel.textContent = opts.cancelLabel || 'Cancel';
    const ok = document.createElement('button');
    ok.className = opts.danger === false ? 'primary' : 'btn-danger';
    ok.textContent = opts.okLabel || 'Confirm';
    footer.appendChild(cancel);
    footer.appendChild(ok);
    box.appendChild(title);
    box.appendChild(body);
    box.appendChild(footer);
    overlay.appendChild(box);

    function done(result) {
      overlay.remove();
      document.body.classList.remove('modal-open');
      resolve(result);
    }
    overlay._cancel = function () { done(false); };
    cancel.onclick = function () { done(false); };
    ok.onclick = function () { done(true); };
    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) done(false);
    });

    document.body.appendChild(overlay);
    document.body.classList.add('modal-open');
    requestAnimationFrame(function () { box.classList.add('modal-open'); });
    ok.focus();
  });
}
