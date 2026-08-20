// app.js — LLM Control Center main application module
// Loaded AFTER components.js and charts.js
// All functions are global scope; uses async/await throughout.

'use strict';

// ── Global state ─────────────────────────────────────────────────────────────
var _globalAuthEnabled = false;
var _globalBindHost = '127.0.0.1';
var _autoLogsInterval = null;
var _vramEstimateTimer = null;
var _activeTab = 'dashboard';
// Follow the host used to open the manager instead of baking in an address
// that changes whenever DHCP assigns this Mac a new lease.
const LAN_HOST = window.location.hostname || '127.0.0.1';

// ── API wrapper ───────────────────────────────────────────────────────────────
// Returns parsed JSON on success, {error: msg} on any failure. Never throws.
async function api(url, method, body) {
  method = method || 'GET';
  body = body || null;
  try {
    const opts = { method: method, headers: { 'Content-Type': 'application/json' } };
    if (body !== null) opts.body = JSON.stringify(body);
    const r = await fetch(url, opts);
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch (_) { data = text; }
    if (!r.ok) {
      const msg = (data && data.detail) || (data && data.error) || r.statusText || ('HTTP ' + r.status);
      toast(msg, 'error');
      return { error: msg };
    }
    return data;
  } catch (e) {
    toast(e.message || 'Network error', 'error');
    return { error: e.message || 'Network error' };
  }
}

// ── setVal — animated text update ────────────────────────────────────────────
function setVal(id, val) {
  var el = document.getElementById(id);
  if (!el) return;
  var s = String(val == null ? '—' : val);
  if (el.textContent !== s) {
    el.textContent = s;
    el.classList.remove('value-updated');
    void el.offsetWidth; // force reflow
    el.classList.add('value-updated');
  }
}

// ── toggleFullscreen ──────────────────────────────────────────────────────────
function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(function() {});
  } else {
    document.exitFullscreen().catch(function() {});
  }
}

// ── initTabs ──────────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.nav-item').forEach(function(item) {
    item.addEventListener('click', function() {
      var tab = this.dataset.tab;
      _activeTab = tab;
      document.querySelectorAll('.nav-item').forEach(function(i) { i.classList.remove('active'); });
      document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
      this.classList.add('active');
      var tabEl = document.getElementById('tab-' + tab);
      if (tabEl) tabEl.classList.add('active');

      if (tab === 'dashboard') { refreshAll(); loadVramChart(); }
      if (tab === 'services')  { loadServices(); }
      if (tab === 'models')    { loadModels(); loadTasks(); loadHfCliStatus(); }
      if (tab === 'runtime')   { loadConfig(); }
      if (tab === 'chat')      { chatFocus(); }
      if (tab === 'benchmark') { loadRecentResults(); _populateCalcModel(); }
      if (tab === 'power')     { refreshPower(); loadGpuProcesses(); loadSystemInfo(); loadTelemetryHealth(); }
      if (tab === 'monitor')   { loadMonitor(); } else { if (typeof monitorStop === 'function') monitorStop(); }
      if (tab === 'logs')      { loadLogs(); }
      if (tab === 'settings')  { loadSettings(); }
    });
  });
}

// loadVramChart lives in charts.js (single renderer with tooltips/crosshair).

// ── refreshAll ────────────────────────────────────────────────────────────────
async function refreshAll() {
  if (document.hidden) return;
  var results = await Promise.all([
    api('/api/status'),
    api('/api/session/stats')
  ]);
  var status = results[0];
  var session = results[1];

  // Update status pills
  var pillsEl = document.getElementById('statusPills');
  if (pillsEl) {
    if (status && !status.error) {
      var healthy = !!(status.service && status.service.healthy);
      var active  = !!(status.service && status.service.active);
      var vramStr = (status.gpu && status.gpu.used_mib) ? (status.gpu.used_mib / 1024).toFixed(1) + ' GiB' : '—';
      var pillColor = healthy ? 'green' : (active ? 'yellow' : 'red');
      var pillLabel = healthy ? '● RUNNING' : (active ? '◉ STARTING' : '○ STOPPED');
      var engine = status.config ? String(status.config.engine || '') : '';
      var enginePill = engine
        ? '<span class="pill ' + (engine.indexOf('mlx') === 0 ? 'purple' : 'cyan') + '">' + (engine.indexOf('mlx') === 0 ? 'MLX' : 'llama.cpp') + '</span>'
        : '';
      pillsEl.innerHTML =
        '<span class="pill ' + pillColor + '">' + pillLabel + '</span>' + enginePill +
        '<span class="pill muted">' + vramStr + ' unified GPU memory</span>' +
        (status.gpu && status.gpu.temp_c != null ? '<span class="pill purple">' + status.gpu.temp_c + '°C</span>' : '');
    } else {
      pillsEl.innerHTML = '<span class="pill red">× disconnected</span>';
    }
  }

  // Warning rail
  if (status && !status.error) {
    var managerHealth = null;
    var mh = await api('/api/manager/health');
    if (mh && !mh.error) managerHealth = mh;
    renderWarningRail(managerHealth, status);
  }

  // GPU stats
  if (status && !status.error && status.gpu) {
    var g = status.gpu;
    var usedGiB = g.used_mib != null ? (g.used_mib / 1024).toFixed(1) : '—';
    setVal('vramValue',  usedGiB);
    setVal('powerValue', g.power_w != null ? g.power_w.toFixed(0) : '—');
    setVal('tempValue',  g.temp_c  != null ? g.temp_c : '—');

    var vramUsedEl = document.getElementById('vramUsed');
    var vramTotalEl = document.getElementById('vramTotal');
    if (vramUsedEl)  vramUsedEl.textContent  = g.used_mib  != null ? Math.round(g.used_mib)  + ' MiB' : '—';
    if (vramTotalEl) vramTotalEl.textContent  = g.total_mib != null ? Math.round(g.total_mib) + ' MiB' : '—';

    var pdEl = document.getElementById('powerDisplay');
    if (pdEl) pdEl.textContent = g.power_w != null ? g.power_w.toFixed(0) + ' W' : '';

    var vramBar = document.getElementById('vramBar');
    if (vramBar && g.total_mib) {
      var pct = (g.used_mib / g.total_mib * 100);
      vramBar.style.width = pct.toFixed(1) + '%';
      vramBar.style.background = pct > 95
        ? 'linear-gradient(90deg,var(--red),#f87171)'
        : pct > 80
        ? 'linear-gradient(90deg,var(--yellow),#fcd34d)'
        : 'linear-gradient(90deg,var(--green),#34d399,var(--cyan))';
    }
  }

  // Context
  if (status && !status.error && status.config && status.config.ctx) {
    var ctx = status.config.ctx;
    setVal('ctxValue', ctx >= 1024 ? Math.round(ctx / 1024) + 'K' : ctx);
  }

  // Service status text
  var svcStatusEl = document.getElementById('serviceStatus');
  if (svcStatusEl && status && !status.error) {
    var svc = status.service || {};
    svcStatusEl.textContent =
      'PID: ' + (svc.pid || '—') + '  API: ' + (svc.api_base || '—');
  }

  // Session stats
  _updateSessionStats(session);

  // Config summary
  if (status && !status.error && status.config) {
    var cfgSumEl = document.getElementById('configSummary');
    if (cfgSumEl) {
      var c = status.config;
      cfgSumEl.innerHTML = Object.entries(c).filter(function(kv) {
        return kv[1] !== null && kv[1] !== '';
      }).map(function(kv) {
        var k = kv[0], v = kv[1];
        var display = v;
        if (typeof v === 'boolean') display = v ? '<span class="success-text">ON</span>' : '<span class="muted">off</span>';
        else if (typeof v === 'number') display = fmt(v);
        return '<tr><td>' + k + '</td><td>' + display + '</td></tr>';
      }).join('');
    }
  }
}

function _updateSessionStats(session) {
  if (!session || session.error) return;
  var isMlx = String(session.engine || '').indexOf('mlx') === 0;
  setVal('statsCheckpoints', session.checkpoints_created || 0);
  setVal('statsRestored',    session.checkpoints_restored || 0);
  setVal('statsForced',      session.forced_reprocess || 0);
  setVal('statsAvgLat', isMlx
    ? fmt(session.last_cached_tokens || 0) + ' cached'
    : (session.avg_prompt_ms != null ? Math.round(session.avg_prompt_ms) + ' ms' : '—'));
  setVal('statsP95', isMlx
    ? fmt(session.last_replayed_tokens || 0) + ' replayed'
    : (session.p95_prompt_ms != null ? Math.round(session.p95_prompt_ms) + ' ms' : '—'));
}

// ── refreshStats (session stats only, faster poll) ────────────────────────────
async function refreshStats() {
  if (document.hidden) return;
  var session = await api('/api/session/stats');
  _updateSessionStats(session);
}

// ── renderWarningRail ─────────────────────────────────────────────────────────
function renderWarningRail(managerHealth, status) {
  var rail = document.getElementById('warningRail');
  if (!rail) return;
  var warnings = [];

  if (status && status.service && !status.service.active) {
    warnings.push('<span class="pill red">⚠ Service stopped</span>');
  }
  if (managerHealth && managerHealth.drift && managerHealth.drift > 2) {
    warnings.push('<span class="pill yellow">⚠ launchd drift detected (' + managerHealth.drift.toFixed(1) + 's)</span>');
  }
  if (!_globalAuthEnabled && _globalBindHost === '0.0.0.0') {
    warnings.push('<span class="pill red">⚠ Auth disabled on 0.0.0.0 — exposed to network!</span>');
  }
  if (status && status.gpu && status.gpu.total_mib) {
    var vramPct = (status.gpu.used_mib / status.gpu.total_mib) * 100;
    if (vramPct > 95) {
      warnings.push('<span class="pill red">⚠ VRAM critical (' + vramPct.toFixed(1) + '%)</span>');
    }
  }

  rail.innerHTML = warnings.join(' ');
  rail.style.display = warnings.length ? 'flex' : 'none';
}

// ── serviceAction ─────────────────────────────────────────────────────────────
async function serviceAction(action) {
  if (action === 'stop' || action === 'restart' || action === 'clean') {
    var ok = await confirmDialog(
      action.charAt(0).toUpperCase() + action.slice(1) +
      ' the llama server? Active clients will see errors, and the next prompt re-prefills its full context.',
      { title: 'Service ' + action, okLabel: action.charAt(0).toUpperCase() + action.slice(1) });
    if (!ok) return;
  }
  var endpoint = action === 'clean' ? 'restart-clean' : action;
  var r = await api('/api/service/' + endpoint, 'POST');
  if (r && r.error) return; // api() already toasted the failure
  toast('Service ' + action + ' OK', 'success');
  setTimeout(refreshAll, 2000);
}

// ── resetStats ────────────────────────────────────────────────────────────────
async function resetStats() {
  await api('/api/service/reset-stats', 'POST');
  toast('Session stats reset', 'info');
  await refreshStats();
}

// ── loadServices: launchd controller + Metal runtime ─────────────────────────
async function loadServices() {
  var results = await Promise.all([api('/api/services/stack'), api('/api/service/version')]);
  var stack = results[0];
  var ver = results[1];

  var bEl = document.getElementById('stackBuild');
  if (bEl && ver && !ver.error) bEl.textContent = '— llama build ' + (ver.build || '?') + ', ctx ' + (ver.context || '?');

  var tbody = document.querySelector('#servicesTable tbody');
  if (!tbody) return;
  if (!Array.isArray(stack) || stack.error) { tbody.innerHTML = '<tr><td colspan="6">failed to load stack</td></tr>'; return; }

  tbody.innerHTML = stack.map(function(s) {
    var pill = s.active === 'active' ? 'green' : (s.active === 'activating' ? 'yellow' : 'red');
    var health = s.healthy === null ? '—' : (s.healthy ? '<span class="pill green">healthy</span>' : '<span class="pill red">unhealthy</span>');
    var since = s.since ? '<div class="muted" style="font-size:9px">' + s.since.replace(/^\w+ /, '').replace(/ [A-Z]+$/, '') + '</div>' : '';
    var isMgr = s.key === 'manager';
    var btn = function(action, cls, label, disabled) {
      return '<button onclick="stackAction(\'' + s.key + '\',\'' + action + '\')" class="' + cls + '" style="font-size:10px;padding:2px 6px"' + (disabled ? ' disabled' : '') + '>' + label + '</button>';
    };
    return '<tr>' +
      '<td><b>' + s.key + '</b><div class="muted" style="font-size:9px">' + s.desc + '</div></td>' +
      '<td><span class="pill ' + pill + '">' + s.active + '</span>' + since + '</td>' +
      '<td>' + health + '</td>' +
      '<td class="font-mono">' + s.port + '</td>' +
      '<td class="font-mono" style="font-size:10px">' + (s.pid || '—') + '</td>' +
      '<td>' + btn('start', 'success', '▶', isMgr) + ' ' + btn('stop', 'danger', '■', isMgr) + ' ' +
               btn('restart', 'warn', '⟳', false) + ' ' +
               '<button onclick="stackLogs(\'' + s.key + '\')" class="muted" style="font-size:10px;padding:2px 6px">logs</button></td>' +
      '</tr>';
  }).join('');

  // LAN snippets
  _renderLanSnippets();
  loadNetworkClients();
}

async function stackAction(key, action) {
  if (key === 'qwen' && (action === 'stop' || action === 'restart')) {
    if (!(await confirmDialog(action + ' the llama server? Active clients (OpenCode/Codex) will see errors, and the next prompt re-prefills its full context.'))) return;
  }
  var r = await api('/api/services/stack/' + key + '/' + action, 'POST');
  if (r && r.ok) toast(key + ' ' + action + ' OK', 'success');
  else toast(key + ' ' + action + ' failed: ' + ((r && (r.stderr || r.error)) || '?'), 'error');
  setTimeout(loadServices, 1500);
}

async function stackLogs(key) {
  var pre = document.getElementById('svcLogView');
  if (!pre) return;
  pre.style.display = 'block';
  pre.textContent = 'loading ' + key + ' logs…';
  var r = await api('/api/services/stack/' + key + '/logs?lines=150');
  pre.textContent = (r && r.content) ? r.content : '(no logs)';
  pre.scrollTop = pre.scrollHeight;
}

// ── Deep research from the web (task-tracked) ────────────────────────────────
var _researchPoll = null;
async function startResearch() {
  var q = (document.getElementById('researchQ') || {}).value || '';
  if (q.trim().length < 8) { toast('Enter a research question', 'error'); return; }
  if (!(await confirmDialog('Deep research occupies the llama server for minutes to tens of minutes. Coding clients will queue behind it. Start?'))) return;
  var body = {
    question: q.trim(),
    depth: parseInt((document.getElementById('researchDepth') || {}).value) || 3,
    sources: parseInt((document.getElementById('researchSources') || {}).value) || 15,
    max_tokens: parseInt((document.getElementById('researchMaxTok') || {}).value) || 4096,
  };
  var r = await api('/api/research', 'POST', body);
  if (!r || !r.task) { toast('Failed to start research', 'error'); return; }
  toast('Research started', 'success');
  var st = document.getElementById('researchStatus');
  var t0 = Date.now();
  if (_researchPoll) clearInterval(_researchPoll);
  _researchPoll = setInterval(async function() {
    var t = await api('/api/tasks/' + r.task);
    if (!t || t.error) return;
    var mins = ((Date.now() - t0) / 60000).toFixed(1);
    var lastLog = (t.log && t.log.length) ? t.log[t.log.length - 1] : '';
    if (t.status === 'done') {
      clearInterval(_researchPoll);
      var f = (t.result && t.result.file) || '';
      if (st) st.innerHTML = '✅ done in ' + ((t.result && t.result.seconds) || '?') + 's — <a href="#" onclick="viewResult(\'' + f + '\');return false">' + f + '</a> (also in Benchmark ▸ Recent Results)';
      loadRecentResults && loadRecentResults();
    } else if (t.status === 'error') {
      clearInterval(_researchPoll);
      if (st) st.textContent = '❌ failed after ' + mins + ' min: ' + lastLog;
    } else {
      if (st) st.textContent = '⏳ running ' + mins + ' min… ' + lastLog;
    }
  }, 5000);
}

// ── optillm quick technique test ─────────────────────────────────────────────
async function optillmTest() {
  var approach = (document.getElementById('optApproach') || {}).value || 'none';
  var prompt = (document.getElementById('optPrompt') || {}).value || 'What is 17*23?';
  var out = document.getElementById('optTestOut');
  if (out) { out.style.display = 'block'; out.textContent = 'running ' + approach + '… (moa ≈ 1-2 min on big models)'; }
  var r = await api('/api/optillm/test', 'POST', { approach: approach, prompt: prompt, max_tokens: 300 });
  if (out) {
    out.textContent = (r && r.ok)
      ? '[' + r.seconds + 's] ' + r.content
      : 'FAILED [' + ((r && r.seconds) || '?') + 's]: ' + ((r && r.error) || 'unknown');
  }
}

async function _renderLanSnippets() {
  var el = document.getElementById('lanSnippets');
  if (!el) return;
  var results = await Promise.all([api('/api/config'), api('/api/models/active')]);
  var cfg = results[0];
  var active = results[1];
  if (!cfg || cfg.error) return;
  var isMlx = String(cfg.engine || '').indexOf('mlx') === 0;
  var port = isMlx ? 8098 : 8097;
  var capacity = isMlx ? Number(cfg.mlx_preallocate_kv_size || 0) : Number(cfg.ctx || 0);
  var outputLimit = 8192;
  var inputLimit = Math.max(0, capacity - outputLimit);
  var providerId = isMlx ? 'local-mlx' : 'local';
  // mlx-vlm treats the request's model field as a loadable repo/path. Send
  // the exact preloaded model id so it reuses weights instead of unloading
  // them and trying to load the placeholder "default_model".
  var modelId = (active && active.models && active.models[0]) ||
    (isMlx ? cfg.mlx_model_path : cfg.model) || 'default_model';
  var opencode = JSON.stringify({
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      [providerId]: {
        "npm": "@ai-sdk/openai-compatible",
        "name": "M1 Max " + (isMlx ? "MLX" : "llama.cpp"),
        "options": {
          "baseURL": "http://" + LAN_HOST + ":" + port + "/v1",
          "apiKey": "sk-opencode",
          "timeout": 2400000,
          "chunkTimeout": 2400000
        },
        "models": {
          [modelId]: {
            "name": "Local " + (isMlx ? "MLX" : "llama.cpp"),
            "limit": {"context": inputLimit, "output": outputLimit}
          }
        }
      }
    },
    "model": providerId + "/" + modelId,
    "small_model": providerId + "/" + modelId,
    "compaction": {"auto": false, "prune": false, "reserved": outputLimit},
    "agent": {"title": {"disable": true}},
    "permission": {"*": "allow", "task": "deny"}
  }, null, 2);
  var continuedev = JSON.stringify({
    "models": [{
      "title": "Local LLM",
      "provider": "openai",
      "model": "local",
      "apiBase": "http://" + LAN_HOST + ":" + port + "/v1",
      "apiKey": "local"
    }]
  }, null, 2);
  el.innerHTML =
    '<h3 style="font-size:12px;color:var(--accent);margin-bottom:6px">OpenCode config snippet</h3>' +
    '<pre style="background:rgba(0,0,0,.4);padding:8px;border-radius:4px;font-size:11px;overflow:auto">' + opencode + '</pre>' +
    '<h3 style="font-size:12px;color:var(--accent);margin:10px 0 6px">Continue.dev config snippet</h3>' +
    '<pre style="background:rgba(0,0,0,.4);padding:8px;border-radius:4px;font-size:11px;overflow:auto">' + continuedev + '</pre>';
}

// ── loadNetworkClients ────────────────────────────────────────────────────────
async function loadNetworkClients() {
  var data = await api('/api/network/clients');
  var el = document.getElementById('networkClients');
  if (!el) return;
  if (!data || data.error || !data.connections) { el.textContent = 'No data.'; return; }
  var targetPorts = ['8090', '8097', '8098'];
  var conns = data.connections.filter(function(c) {
    return targetPorts.some(function(p) { return c.local && c.local.includes(':' + p); });
  });
  if (!conns.length) { el.innerHTML = '<span class="muted">No active connections on LLM ports.</span>'; return; }
  el.innerHTML = '<table class="data" style="font-size:11px"><thead><tr><th>Local</th><th>State</th></tr></thead><tbody>' +
    conns.map(function(c) { return '<tr><td class="font-mono">' + c.local + '</td><td>' + c.state + '</td></tr>'; }).join('') +
    '</tbody></table>';
}

// ── loadModels ────────────────────────────────────────────────────────────────
async function loadModels() {
  var results = await Promise.all([api('/api/models'), api('/api/status'), api('/api/models/active')]);
  var models = results[0];
  var status = results[1];
  var activeModels = (results[2] && results[2].models) || [];
  if (!models || models.error) return;

  var currentPath = status && status.config ? status.config.model : null;
  // Try to match by model name suffix
  var tbody = document.querySelector('#modelsTable tbody');
  if (tbody) {
    var mlxPath = status && status.config ? status.config.mlx_model_path : '';
    var engineNow = status && status.config ? status.config.engine : 'llama.cpp';
    tbody.innerHTML = models.map(function(m) {
      var isMlx = m.engine === 'mlx';
      var isActive = isMlx
        ? (activeModels.indexOf(m.path) !== -1 || (engineNow && engineNow.indexOf('mlx') === 0 && mlxPath && m.path === mlxPath))
        : (currentPath && (m.path === currentPath || m.name === currentPath || m.path.endsWith(currentPath)) && (!engineNow || engineNow.indexOf('mlx') !== 0));
      var sizeStr = m.size_mib
        ? (m.size_mib / 1024).toFixed(2) + ' GB'
        : m.size_bytes ? (m.size_bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB' : '—';
      var activePill = isActive ? ' <span class="pill green" style="font-size:10px">active</span>' : '';
      var enginePill = isMlx
        ? '<span class="pill purple" style="font-size:10px">MLX</span>'
        : '<span class="pill cyan" style="font-size:10px">llama.cpp</span>';
      return '<tr>' +
        '<td>' + m.name + activePill + '</td>' +
        '<td>' + enginePill + '</td>' +
        '<td class="font-mono">' + sizeStr + '</td>' +
        '<td>' + (m.arch || '—') + '</td>' +
        '<td>' + (m.quant || '—') + '</td>' +
        '<td>' +
          '<button onclick="modelInfo(\'' + _esc(m.path) + '\')" class="btn-muted" style="font-size:11px;padding:2px 5px">Info</button> ' +
          '<button onclick="useModel(\'' + _esc(m.path) + '\')"  class="btn-success" style="font-size:11px;padding:2px 5px">Use</button> ' +
          '<button onclick="modelDelete(\'' + _esc(m.path) + '\')" class="btn-danger" style="font-size:11px;padding:2px 5px">Del</button>' +
        '</td>' +
        '</tr>';
    }).join('');
  }

  // Populate calcModel select
  _populateCalcModel(models);
}

function _esc(s) {
  return String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

function _populateCalcModel(models) {
  var calc = document.getElementById('calcModel');
  if (!calc) return;
  if (models) {
    calc.innerHTML = models.filter(function(m) { return !m.is_drafter; }).map(function(m) {
      return '<option value="' + m.path + '">' + m.name + '</option>';
    }).join('');
  } else {
    api('/api/models').then(function(ms) {
      if (!ms || ms.error) return;
      calc.innerHTML = ms.filter(function(m) { return !m.is_drafter; }).map(function(m) {
        return '<option value="' + m.path + '">' + m.name + '</option>';
      }).join('');
    });
  }
}

// ── modelInfo ─────────────────────────────────────────────────────────────────
async function modelInfo(path) {
  var meta = await api('/api/models/info?path=' + encodeURIComponent(path));
  if (!meta || meta.error) { toast('Cannot load model info', 'error'); return; }
  showModal(
    '<h3 style="margin-bottom:10px;color:var(--accent)">Model Info</h3>' +
    '<table class="kv-table">' +
    Object.entries(meta).map(function(kv) {
      return '<tr><td>' + kv[0] + '</td><td style="word-break:break-all">' + kv[1] + '</td></tr>';
    }).join('') +
    '</table>'
  );
}

// ── useModel ──────────────────────────────────────────────────────────────────
async function useModel(path) {
  if (!(await confirmDialog('Switch to this model and restart service?'))) return;
  var r = await api('/api/models/use', 'POST', { path: path });
  if (r && r.error) return;
  toast('Switching model — restarting…', 'success');
  setTimeout(refreshAll, 3000);
}

// ── modelDelete ───────────────────────────────────────────────────────────────
async function modelDelete(path) {
  if (!(await confirmDialog('Permanently delete this model file?\n' + path))) return;
  var r = await api('/api/models', 'DELETE', { path: path });
  if (!r || r.error) return;  // api() already toasts errors (incl. 409 active-model guard)
  var freed = r.freed_bytes ? ' — freed ' + (r.freed_bytes / 1e9).toFixed(2) + ' GB' : '';
  toast('Model deleted' + freed, 'success');
  loadModels();
}

// ── searchHF ──────────────────────────────────────────────────────────────────
async function searchHF() {
  var el = document.getElementById('hfSearch');
  if (!el) return;
  var q = el.value.trim();
  if (!q) return;
  var resultsEl = document.getElementById('hfResults');
  if (resultsEl) resultsEl.innerHTML = '<div class="spinner"></div> Searching…';
  var res = await api('/api/hf/search?q=' + encodeURIComponent(q));
  if (!res || res.error || !Array.isArray(res)) {
    if (resultsEl) resultsEl.innerHTML = '<span style="color:var(--red)">Search failed</span>';
    return;
  }
  if (!res.length) { if (resultsEl) resultsEl.innerHTML = '<span class="muted">No results.</span>'; return; }
  if (resultsEl) {
    resultsEl.innerHTML =
      '<table class="data" style="margin-top:10px">' +
      '<thead><tr><th>Repo</th><th>Downloads</th><th>Likes</th><th>Action</th></tr></thead>' +
      '<tbody>' +
      res.map(function(r) {
        return '<tr>' +
          '<td>' + r.modelId + '</td>' +
          '<td>' + (r.downloads || 0).toLocaleString() + '</td>' +
          '<td>' + (r.likes || 0).toLocaleString() + '</td>' +
          '<td><button onclick="hfBrowse(\'' + _esc(r.modelId) + '\')" class="btn-muted">Browse</button></td>' +
          '</tr>';
      }).join('') +
      '</tbody></table>';
  }
}

// ── hfDirectDownload ──────────────────────────────────────────────────────────
async function hfDirectDownload() {
  var el = document.getElementById('hfSearch');
  var repo = el ? el.value.trim() : '';
  if (!repo) { toast('Enter a HF repo ID', 'error'); return; }
  hfBrowse(repo);
}

// ── hfBrowse ──────────────────────────────────────────────────────────────────
async function hfBrowse(repo) {
  toast('Browsing ' + repo + '…');
  var files = await api('/api/hf/files?repo=' + encodeURIComponent(repo));
  if (!files || files.error) { toast('Cannot list files: ' + (files && files.error ? files.error : 'error'), 'error'); return; }
  var ggufs = files.filter(function(f) { return f.path && f.path.endsWith('.gguf'); });
  if (!ggufs.length) { toast('No .gguf files found in ' + repo, 'error'); return; }
  var html =
    '<table class="data">' +
    '<thead><tr><th>File</th><th>Size</th><th>Action</th></tr></thead>' +
    '<tbody>' +
    ggufs.map(function(f) {
      return '<tr>' +
        '<td style="word-break:break-all">' + f.path + '</td>' +
        '<td class="font-mono">' + (f.size ? (f.size / 1024 / 1024 / 1024).toFixed(2) + ' GB' : '—') + '</td>' +
        '<td><button onclick="hfDownload(\'' + _esc(repo) + '\',\'' + _esc(f.path) + '\')" class="btn-success">Download</button></td>' +
        '</tr>';
    }).join('') +
    '</tbody></table>';
  var resultsEl = document.getElementById('hfResults');
  if (resultsEl) resultsEl.innerHTML = html;
}

// ── hfDownload ────────────────────────────────────────────────────────────────
async function hfDownload(repo, file) {
  toast('Starting download: ' + file);
  var r = await api('/api/hf/download', 'POST', { repo: repo, file: file });
  if (!r || r.error) return;
  var resultsEl = document.getElementById('hfResults');
  if (resultsEl) resultsEl.innerHTML = '<div class="muted">Download task started. Check Tasks panel for progress.</div>';
  loadTasks();
}

// ── loadTasks ─────────────────────────────────────────────────────────────────
async function loadTasks() {
  var tasks = await api('/api/tasks');
  var el = document.getElementById('tasksList');
  var cnt = document.getElementById('taskCount');
  if (!el) return;

  if (!tasks || tasks.error || !Array.isArray(tasks) || !tasks.length) {
    el.innerHTML = '<span class="muted">No tasks yet.</span>';
    if (cnt) cnt.textContent = '0';
    return;
  }

  var running = tasks.filter(function(t) { return t.status === 'running' || t.status === 'stalled'; });
  if (cnt) cnt.textContent = running.length ? (running.length + ' running') : tasks.length;

  el.innerHTML =
    '<table class="data" style="font-size:11px">' +
    '<thead><tr><th>Status</th><th>File</th><th style="min-width:220px">Progress</th><th></th></tr></thead>' +
    '<tbody>' +
    tasks.slice(0, 20).map(function(t) {
      var st = t.status || 'unknown';
      var pillColor = st === 'done' ? 'green' : (st === 'running' ? 'yellow' : (st === 'failed' || st === 'cancelled' ? 'red' : 'muted'));
      var fname = (t.file || t.id || '').split('/').pop().substring(0, 40);
      var cancelBtn = (st === 'running') ? '<button onclick="cancelTask(\'' + t.id + '\')" class="btn-danger" style="font-size:10px;padding:2px 5px">Cancel</button>' : '';
      // Progress bar + size / speed / ETA
      var pct = (t.progress != null) ? t.progress : (st === 'done' ? 100 : 0);
      var dl = t.downloaded_bytes, tot = t.total_bytes;
      var sizeStr = (dl != null && tot) ? (dl / 1e9).toFixed(2) + ' / ' + (tot / 1e9).toFixed(2) + ' GB'
                  : (dl != null && dl > 0 ? (dl / 1e9).toFixed(2) + ' GB' : '');
      var spd = (st === 'running' && t.speed_bps) ? (t.speed_bps / 1e6).toFixed(1) + ' MB/s' : '';
      var eta = (st === 'running' && t.eta_s != null && t.eta_s > 0) ? 'ETA ' + (typeof _fmtUptime === 'function' ? _fmtUptime(t.eta_s) : t.eta_s + 's') : '';
      var barColor = st === 'done' ? '#34d399' : (st === 'failed' || st === 'cancelled' ? '#ef4444' : '#22d3ee');
      var sub = [pct + '%', sizeStr, spd, eta].filter(Boolean).join(' · ');
      var lastLog = (t.log && t.log.length ? t.log[t.log.length - 1] : '').substring(0, 80);
      var bar =
        '<div class="progress" style="height:7px;border-radius:4px;background:rgba(255,255,255,.12);overflow:hidden">' +
          '<div style="height:100%;width:' + pct + '%;background:' + barColor + ';transition:width .6s"></div>' +
        '</div>' +
        '<div style="font-size:9px;color:var(--muted);margin-top:2px">' + (sub || '—') + '</div>' +
        (lastLog ? '<div style="font-size:9px;color:var(--muted);opacity:.7;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + lastLog.replace(/"/g, '&quot;') + '">' + lastLog + '</div>' : '');
      return '<tr>' +
        '<td><span class="pill ' + pillColor + '" style="font-size:10px">' + st + '</span></td>' +
        '<td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + (t.file || '') + '">' + fname + '</td>' +
        '<td>' + bar + '</td>' +
        '<td>' + cancelBtn + '</td>' +
        '</tr>';
    }).join('') +
    '</tbody></table>';

  // Auto-refresh while any task is running so the bars animate
  if (typeof _tasksTimer !== 'undefined' && _tasksTimer) { clearTimeout(_tasksTimer); _tasksTimer = null; }
  if (running.length) { _tasksTimer = setTimeout(loadTasks, 2000); }
}
var _tasksTimer = null;

// ── cancelTask ────────────────────────────────────────────────────────────────
async function cancelTask(tid) {
  if (!(await confirmDialog('Cancel task ' + tid + '?'))) return;
  await api('/api/tasks/' + tid + '/cancel', 'POST');
  toast('Cancel requested', 'info');
  setTimeout(loadTasks, 1000);
}

// ── loadHfCliStatus ───────────────────────────────────────────────────────────
async function loadHfCliStatus() {
  var s = await api('/api/hf/cli-status');
  var el = document.getElementById('hfCliStatus');
  if (!el) return;
  if (!s || s.error) { el.textContent = 'Could not load HF CLI status.'; return; }
  el.innerHTML =
    '<div>CLI: ' + (s.cli_name || '(not found)') + ' @ ' + (s.cli_path || '—') + '</div>' +
    '<div>Version: ' + (s.version || '—') + '</div>' +
    '<div>Token: ' + (s.token_present ? '<span class="success-text">✓ present</span>' : '<span class="warn-text">✗ not set</span>') + '</div>' +
    '<div>Cache: ' + (s.cache_path || '—') + '</div>';
}

// ── loadConfig ────────────────────────────────────────────────────────────────
async function loadConfig() {
  var results = await Promise.all([api('/api/config'), api('/api/models')]);
  var data = results[0];
  var models = results[1];
  if (!data || data.error) return;

  // Populate the MLX engine panel from the full list before filtering.
  _populateMlxPanel(data, Array.isArray(models) && !models.error ? models : []);

  // Model select — this form configures the llama.cpp engine;
  // the MLX panel alongside configures the MLX engine.
  var modelSel = document.getElementById('cfg-model');
  if (modelSel && Array.isArray(models) && !models.error) {
    models = models.filter(function(m) { return m.engine !== 'mlx'; });
    modelSel.innerHTML = models.map(function(m) {
      return '<option value="' + m.path + '"' + (m.path === data.model_path ? ' selected' : '') + '>' +
        m.name + (m.size_mib ? ' (' + (m.size_mib / 1024).toFixed(1) + 'GB)' : '') +
        (m.arch ? ' — ' + m.arch : '') + '</option>';
    }).join('');
  }

  // KV cache select
  var kvKSel = document.getElementById('cfg-kv-k');
  var kvVSel = document.getElementById('cfg-kv-v');
  if (kvKSel && kvVSel && data.kv_types) {
    var options = Object.entries(data.kv_types).map(function(kv) {
      var key = kv[0], info = kv[1];
      var label = Array.isArray(info) ? info[0] : (info.desc || key);
      return '<option value="' + key + '">' + key + ' — ' + label + '</option>';
    }).join('');
    kvKSel.innerHTML = options; kvVSel.innerHTML = options;
    kvKSel.value = data.kv_k; kvVSel.value = data.kv_v;
  }

  function setInput(id, val) {
    var el = document.getElementById(id);
    if (el) el.value = val != null ? val : '';
  }

  setInput('cfg-ctx',         data.ctx);
  setInput('cfg-batch',       data.batch);
  setInput('cfg-ubatch',      data.ubatch);
  setInput('cfg-mmproj',      data.mmproj_path || '');
  setInput('cfg-image-min-tokens', data.image_min_tokens || '');
  setInput('cfg-image-max-tokens', data.image_max_tokens || '');
  setInput('cfg-graph-cap',   data.graph_cap);
  setInput('cfg-ckpt',        data.ctx_chk || '0');
  setInput('cfg-ckpt-every',  data.checkpoint_every != null ? data.checkpoint_every : '512');
  setInput('cfg-cache-ram',   data.cache_ram != null ? data.cache_ram : '512');
  setInput('cfg-rope-scale',  data.rope_scale || '');
  setInput('cfg-yarn-ctx',    data.yarn_orig_ctx || '');

  updateCommandPreview();
  updateVramEstimate();
  loadVisionMode();
}

// ── MLX engine panel ──────────────────────────────────────────────────────────
function _populateMlxPanel(cfg, models) {
  var sel = document.getElementById('mlx-model');
  if (sel) {
    var mlx = models.filter(function(m) { return m.engine === 'mlx'; });
    sel.innerHTML = mlx.length
      ? mlx.map(function(m) {
          return '<option value="' + m.path + '"' + (m.path === cfg.mlx_model_path ? ' selected' : '') + '>' +
            m.name + ' (' + (m.size_mib / 1024).toFixed(1) + ' GB' + (m.quant ? ', ' + m.quant : '') + ')</option>';
        }).join('')
      : '<option value="">— no MLX models downloaded —</option>';
  }
  var mode = document.getElementById('mlx-kv-mode');
  if (mode) mode.value = String(cfg.mlx_kv_mode || ('native' + (cfg.mlx_kv_bits || 4)));
  var group = document.getElementById('mlx-kv-group-size');
  if (group) group.value = String(cfg.mlx_kv_group_size || 64);
  var prefill = document.getElementById('mlx-prefill-step');
  if (prefill) prefill.value = String(cfg.mlx_prefill_step_size || 2048);
  var adaptive = document.getElementById('mlx-adaptive-prefill');
  if (adaptive) adaptive.checked = cfg.mlx_adaptive_prefill !== false;
  var experimental = document.getElementById('mlx-experimental-context');
  if (experimental) experimental.checked = cfg.mlx_experimental_context === true;
  var preallocate = document.getElementById('mlx-preallocate-kv-size');
  if (preallocate) preallocate.value = String(
    cfg.mlx_preallocate_kv_size === undefined ? 131072 : cfg.mlx_preallocate_kv_size
  );
  var cacheMib = document.getElementById('mlx-prompt-cache-mib');
  if (cacheMib) cacheMib.value = String(cfg.mlx_prompt_cache_mib || 2048);
  var reuse = document.getElementById('mlx-session-reuse');
  if (reuse) reuse.checked = cfg.mlx_session_reuse !== false;

  var isMlx = String(cfg.engine || '').indexOf('mlx') === 0;
  var badge = document.getElementById('engineBadge');
  if (badge) {
    badge.textContent = isMlx ? 'MLX-LM — port 8098' : 'llama.cpp — port 8097';
    badge.className = 'pill ' + (isMlx ? 'purple' : 'cyan');
  }
  var detail = document.getElementById('engineDetail');
  if (detail) detail.textContent = isMlx
    ? (String(cfg.mlx_model_path || '').split('/').pop() || '(no model)') +
      ' · KV ' + (cfg.mlx_kv_mode || ('native' + (cfg.mlx_kv_bits || 4))) +
      ' · fixed ' + fmt(cfg.mlx_preallocate_kv_size || 0) +
      ' · group ' + (cfg.mlx_kv_group_size || 64)
    : (cfg.model || '') + ' · ctx ' + fmt(cfg.ctx) + ' · KV ' + cfg.kv_k + '/' + cfg.kv_v;
  var lp = document.getElementById('llamaActivePill');
  var mp = document.getElementById('mlxActivePill');
  if (lp) lp.innerHTML = isMlx ? '' : '<span class="pill green" style="font-size:10px">active engine</span>';
  if (mp) mp.innerHTML = isMlx ? '<span class="pill green" style="font-size:10px">active engine</span>' : '';
  updateMlxPreview();
}

function updateMlxPreview() {
  var el = document.getElementById('mlxCmdPreview');
  if (!el) return;
  var sel = document.getElementById('mlx-model');
  var path = sel ? sel.value : '';
  var mode = (document.getElementById('mlx-kv-mode') || {}).value || 'native4';
  var bits = mode === 'native8' ? '8' : '4';
  var group = (document.getElementById('mlx-kv-group-size') || {}).value || '64';
  var prefill = (document.getElementById('mlx-prefill-step') || {}).value || '2048';
  var adaptive = (document.getElementById('mlx-adaptive-prefill') || {}).checked;
  var preallocate = parseInt(
    (document.getElementById('mlx-preallocate-kv-size') || {}).value
  ) || 0;
  var cacheMib = parseInt((document.getElementById('mlx-prompt-cache-mib') || {}).value) || 2048;
  var reuse = (document.getElementById('mlx-session-reuse') || {}).checked;
  var fixedTail = reuse
    ? ' --preallocate-kv-size ' + preallocate + ' --session-reuse --session-checkpoints 1'
    : ' --preallocate-kv-size ' + preallocate + ' --prompt-cache-size 0 --prompt-cache-bytes 0';
  el.textContent = path
    ? 'python -m local_llm_control.mlx_quant_server --kv-bits ' + bits + ' --kv-mode ' + mode + ' --kv-group-size ' + group +
      ' --model ' + path + ' --host 0.0.0.0 --port 8098 --decode-concurrency 1 --prompt-concurrency 1' +
      ' --prefill-step-size ' + prefill +
      (adaptive ? '' : ' --no-adaptive-prefill') +
      (preallocate
        ? fixedTail
        : ' --prompt-cache-size 1 --prompt-cache-bytes ' + (cacheMib * 1048576))
    : '— no MLX model selected —';
  loadMlxMemoryPlan();
}

var mlxPlanRequest = 0;
async function loadMlxMemoryPlan() {
  var target = document.getElementById('mlxMemoryPlan');
  var model = document.getElementById('mlx-model');
  if (!target || !model || !model.value) return;
  var context = parseInt((document.getElementById('mlx-preallocate-kv-size') || {}).value) || 0;
  var mode = (document.getElementById('mlx-kv-mode') || {}).value || 'native4';
  var group = parseInt((document.getElementById('mlx-kv-group-size') || {}).value) || 64;
  if (!context) {
    target.className = 'font-sm muted mt-2';
    target.textContent = 'Dynamic KV grows on demand; choose a fixed context for a startup memory safety plan.';
    return;
  }
  var request = ++mlxPlanRequest;
  target.className = 'font-sm muted mt-2';
  target.textContent = 'Calculating fixed-arena memory plan…';
  var query = '?path=' + encodeURIComponent(model.value) +
    '&context=' + context + '&kv_mode=' + encodeURIComponent(mode) +
    '&group_size=' + group + '&checkpoints=1';
  var plan = await api('/api/mlx/memory-plan' + query);
  if (request !== mlxPlanRequest) return;
  if (!plan || plan.error) {
    target.className = 'font-sm red mt-2';
    target.textContent = 'Memory plan unavailable' + (plan && plan.error ? ': ' + plan.error : '.');
    return;
  }
  if (!plan.model_weights_gib) {
    target.className = 'font-sm warn mt-2';
    target.textContent = 'Weights are incomplete or still downloading; starting is blocked until all shards arrive.';
    return;
  }
  target.className = 'font-sm mt-2 ' + (plan.safe ? 'green' : 'red');
  target.innerHTML =
    '<strong>' + (plan.safe ? 'SAFE' : 'UNSAFE') + '</strong>' +
    ' · weights ' + Number(plan.model_weights_gib).toFixed(2) + ' GiB' +
    ' + KV ' + Number(plan.kv_gib).toFixed(2) + ' GiB' +
    ' · estimated peak ' + Number(plan.peak_gib).toFixed(2) + ' / ' +
    Number(plan.safe_peak_gib).toFixed(1) + ' GiB' +
    ' · recommended maximum ' + fmt(plan.recommended_context);
}

async function applyMlx(start) {
  var sel = document.getElementById('mlx-model');
  var path = sel ? sel.value : '';
  if (!path) { toast('Download an MLX model first (Models tab → HuggingFace)', 'error'); return; }
  var updates = {
    LOCAL_LLM_ENGINE: 'mlx-lm',
    LOCAL_LLM_MLX_MODEL_PATH: path,
    LOCAL_LLM_MLX_KV_BITS: ((document.getElementById('mlx-kv-mode') || {}).value === 'native8') ? '8' : '4',
    LOCAL_LLM_MLX_KV_MODE: (document.getElementById('mlx-kv-mode') || {}).value || 'native4',
    LOCAL_LLM_MLX_KV_GROUP_SIZE: (document.getElementById('mlx-kv-group-size') || {}).value || '64',
    LOCAL_LLM_MLX_PREALLOCATE_KV_SIZE: (document.getElementById('mlx-preallocate-kv-size') || {}).value || '0',
    LOCAL_LLM_MLX_PREFILL_STEP_SIZE: (document.getElementById('mlx-prefill-step') || {}).value || '2048',
    LOCAL_LLM_MLX_ADAPTIVE_PREFILL: (document.getElementById('mlx-adaptive-prefill') || {}).checked ? 'true' : 'false',
    LOCAL_LLM_MLX_EXPERIMENTAL_CONTEXT: (document.getElementById('mlx-experimental-context') || {}).checked ? 'true' : 'false',
    LOCAL_LLM_MLX_PROMPT_CACHE_MIB: (document.getElementById('mlx-prompt-cache-mib') || {}).value || '2048',
    LOCAL_LLM_MLX_SESSION_REUSE: (document.getElementById('mlx-session-reuse') || {}).checked ? 'true' : 'false',
  };
  var r = await api('/api/config', 'PUT', updates);
  if (r && r.error) return;
  toast('MLX engine configured', 'success');
  if (start) {
    if (await confirmDialog('Start the MLX runtime now? Any running model will be stopped first.',
                            { title: 'Start MLX', danger: false, okLabel: 'Start' })) {
      await api('/api/service/restart', 'POST');
      toast('MLX runtime starting…', 'info');
      setTimeout(refreshAll, 4000);
    }
  }
  loadConfig();
}

async function loadVisionMode() {
  var data = await api('/api/vision-mode');
  var el = document.getElementById('visionModeStatus');
  if (!el) return;
  if (!data || data.error) {
    el.textContent = 'Could not load vision mode.';
    return;
  }
  var mode = data.enabled
    ? '<span class="pill green">Vision/OCR enabled</span>'
    : '<span class="pill muted">Text max-context</span>';
  var mm = data.enabled ? ('<div>mmproj: <span class="font-mono">' + data.mmproj_path + '</span></div>') : '';
  el.innerHTML = mode +
    '<div>ctx ' + data.ctx + ', batch ' + data.batch + ', ubatch ' + data.ubatch +
    (data.image_min_tokens ? ', image min ' + data.image_min_tokens : '') +
    (data.image_max_tokens ? ', image max ' + data.image_max_tokens : '') + '</div>' + mm;
}

async function setVisionMode(enabled, restart) {
  var msg = enabled
    ? 'Enable Vision/OCR? This requires a matching mmproj and restarts llama.cpp.'
    : 'Disable Vision/OCR? This restarts llama without mmproj and restores the 395k text context.';
  if (restart && !(await confirmDialog(msg))) return;
  var r = await api('/api/vision-mode', 'POST', { enabled: !!enabled, restart: !!restart });
  if (!r || r.error) {
    toast('Vision mode switch failed: ' + ((r && r.error) || 'unknown'), 'error');
    return;
  }
  toast((enabled ? 'Vision/OCR enabled' : 'Text max-context enabled') + (restart ? '; restarting…' : ''), 'success');
  await loadConfig();
  await loadVisionMode();
  if (restart) setTimeout(refreshAll, 5000);
}

// ── Apple Silicon presets measured on this machine ───────────────────────────
function modelProfileFor(name) {
  var n = (name || '').toLowerCase();
  if (/a3b|moe|35b/.test(n)) return {
    key: '35b', ctx: '262144', batch: '2048', ubatch: '512', kv_k: 'q8_0', kv_v: 'turbo4', moe: true,
    note: '35B Q4_K_P default — native 262K, q8 K + Turbo4 V, measured 44.84 tok/s.'
  };
  if (/27b/.test(n)) return {
    key: '27b', ctx: '196608', batch: '1024', ubatch: '256', kv_k: 'q8_0', kv_v: 'turbo4', moe: false,
    note: '27B Q6 default — tested ~200K (196608) context, quality-first q8 K + Turbo4 V, 10.09 tok/s.'
  };
  return null;
}

function _applyProfileObj(p) {
  if (!p) { toast('No preset for this model', 'info'); return; }
  function setVal(id, v) { var e = document.getElementById(id); if (e) e.value = v; }
  setVal('cfg-ctx', p.ctx); setVal('cfg-batch', p.batch); setVal('cfg-ubatch', p.ubatch);
  var kvK = document.getElementById('cfg-kv-k'); if (kvK) kvK.value = p.kv_k;
  var kvV = document.getElementById('cfg-kv-v'); if (kvV) kvV.value = p.kv_v;
  var nEl = document.getElementById('presetNote'); if (nEl) nEl.textContent = p.note;
  toast('Applied ' + p.key.toUpperCase() + ' best-performance preset — review VRAM estimate, then Save & Restart', 'success');
  updateCommandPreview(); updateVramEstimate();
}

// apply preset for whatever model is currently selected in the dropdown
function applyModelProfile() {
  var sel = document.getElementById('cfg-model');
  var name = (sel && sel.options[sel.selectedIndex]) ? sel.options[sel.selectedIndex].text : '';
  _applyProfileObj(modelProfileFor(name));
}

// apply a named preset (35b/27b) and select a matching model if present
function applyNamedProfile(which) {
  var sel = document.getElementById('cfg-model');
  if (sel) {
    var frag = which === '35b' ? /a3b|moe|35b/i : /27b/i;
    var match = Array.from(sel.options).find(function (o) { return frag.test(o.text); });
    if (match) sel.value = match.value;
  }
  var p = modelProfileFor(which === '35b' ? 'a3b 35b' : '27b');
  _applyProfileObj(p);
}

// model dropdown changed → auto-apply that model's preset (seamless switching)
function onModelChange() {
  applyModelProfile();
}

// ── saveConfig ────────────────────────────────────────────────────────────────
async function saveConfig(restart) {
  function gv(id) { var el = document.getElementById(id); return el ? el.value : ''; }

  var modelPath = gv('cfg-model');
  var kvKType   = gv('cfg-kv-k');
  var kvVType   = gv('cfg-kv-v');

  var updates = {
    LOCAL_LLM_QWEN_CTX:                  gv('cfg-ctx'),
    LOCAL_LLM_QWEN_BATCH:                gv('cfg-batch'),
    LOCAL_LLM_QWEN_UBATCH:               gv('cfg-ubatch'),
    LOCAL_LLM_QWEN_MMPROJ_PATH:          gv('cfg-mmproj'),
    LOCAL_LLM_QWEN_IMAGE_MIN_TOKENS:      gv('cfg-image-min-tokens'),
    LOCAL_LLM_QWEN_IMAGE_MAX_TOKENS:      gv('cfg-image-max-tokens'),
    GGML_CUDA_GRAPH_CACHE_MAX:           gv('cfg-graph-cap'),
    LOCAL_LLM_QWEN_CTX_CHECKPOINTS:      gv('cfg-ckpt'),
    LOCAL_LLM_QWEN_CHECKPOINT_EVERY_NT:  gv('cfg-ckpt-every'),
    LOCAL_LLM_QWEN_CACHE_RAM:            gv('cfg-cache-ram'),
  };

  if (gv('cfg-rope-scale')) updates.LOCAL_LLM_QWEN_ROPE_SCALE    = gv('cfg-rope-scale');
  if (gv('cfg-yarn-ctx'))   updates.LOCAL_LLM_QWEN_YARN_ORIG_CTX = gv('cfg-yarn-ctx');

  if (kvKType) updates.LOCAL_LLM_QWEN_CACHE_TYPE_K = kvKType;
  if (kvVType) updates.LOCAL_LLM_QWEN_CACHE_TYPE_V = kvVType;

  if (modelPath) {
    var models = await api('/api/models');
    if (Array.isArray(models) && !models.error) {
      var m = models.find(function(x) { return x.path === modelPath; });
      if (m) {
        updates.LOCAL_LLM_ENGINE          = 'llama.cpp';
        updates.LOCAL_LLM_QWEN_MODEL      = m.name;
        updates.LOCAL_LLM_QWEN_MODEL_PATH = m.path;
        var isMoE = m.name.includes('A3B') || m.name.toLowerCase().includes('moe');
        updates.LOCAL_LLM_QWEN_CONTEXT_OVERRIDE_KEY = isMoE ? 'qwen35moe.context_length' : 'qwen35.context_length';
      }
    }
  }

  var r = await api('/api/config', 'PUT', updates);
  if (r && r.error) return;
  toast('Config saved!', 'success');
  updateCommandPreview();

  if (restart) {
    await api('/api/service/restart-clean', 'POST');
    toast('Service restarting…', 'info');
    setTimeout(refreshAll, 3000);
  }
}

// ── updateCommandPreview ──────────────────────────────────────────────────────
function updateCommandPreview() {
  var el = document.getElementById('cmdPreview');
  if (!el) return;
  function gv(id) { var e = document.getElementById(id); return e ? e.value : ''; }
  function go(id) { var e = document.getElementById(id); return e ? e.options[e.selectedIndex] : null; }

  var modelOpt = go('cfg-model');
  var modelPath = modelOpt ? modelOpt.value : '/models/model.gguf';
  var ctx      = gv('cfg-ctx') || '131072';
  var kvK      = gv('cfg-kv-k') || 'q8_0';
  var kvV      = gv('cfg-kv-v') || 'turbo4';
  var batch    = gv('cfg-batch') || '256';
  var ubatch   = gv('cfg-ubatch') || '64';
  var mmproj   = gv('cfg-mmproj');
  var imgMin   = gv('cfg-image-min-tokens');
  var imgMax   = gv('cfg-image-max-tokens');
  var graphCap = gv('cfg-graph-cap') || '4';
  var ckpt     = gv('cfg-ckpt') || '0';
  var ckptEvery = gv('cfg-ckpt-every') || '256';
  var cacheRam = gv('cfg-cache-ram') || '512';
  var ropeSc   = gv('cfg-rope-scale');
  var yarnCtx  = gv('cfg-yarn-ctx');
  var moe      = /a3b|moe/i.test(modelOpt ? modelOpt.text : '');

  var cmd = './llama-server' +
    ' --model ' + modelPath +
    ' --ctx-size ' + ctx +
    ' --cache-type-k ' + kvK +
    ' --cache-type-v ' + kvV +
    ' --batch-size ' + batch +
    ' --ubatch-size ' + ubatch +
    ' --gpu-layers 99' +
    ' --host 0.0.0.0' +
    ' --port 8097' +
    ' --flash-attn on';
  if (mmproj) cmd += ' --mmproj ' + mmproj + ' --mmproj-offload';
  if (mmproj && imgMin) cmd += ' --image-min-tokens ' + imgMin;
  if (mmproj && imgMax) cmd += ' --image-max-tokens ' + imgMax;

  // TurboQuant Metal checkpoint flags carried over from the A5000 workbench.
  if (ckpt && ckpt !== '0') cmd += ' --ctx-checkpoints ' + ckpt;
  if (ckptEvery && ckptEvery !== '0') cmd += ' --checkpoint-min-step ' + ckptEvery;
  if (cacheRam) cmd += ' --cache-ram ' + cacheRam;
  if (ropeSc) cmd += ' --rope-scaling yarn --rope-scale ' + ropeSc;
  if (yarnCtx) cmd += ' --yarn-orig-ctx ' + yarnCtx;
  cmd += ' --override-kv ' + (moe ? 'qwen35moe' : 'qwen35') + '.context_length=int:' + ctx;
  // note: GGML_CUDA_GRAPH_CACHE_MAX is now a no-op on the resync build (upstream native graph cache)

  el.textContent = cmd;
}

// ── updateVramEstimate (debounced) ────────────────────────────────────────────
function updateVramEstimate() {
  if (_vramEstimateTimer) clearTimeout(_vramEstimateTimer);
  _vramEstimateTimer = setTimeout(_doVramEstimate, 400);
}

async function _doVramEstimate() {
  var modelEl = document.getElementById('cfg-model');
  var kvKEl   = document.getElementById('cfg-kv-k');
  var kvVEl   = document.getElementById('cfg-kv-v');
  var el      = document.getElementById('vramEstimate');
  if (!el) return;
  var modelPath = modelEl ? modelEl.value : '';
  var kvK = kvKEl ? kvKEl.value : 'q8_0';
  var kvV = kvVEl ? kvVEl.value : 'turbo4';
  if (!modelPath) return;

  el.innerHTML = '<span class="muted">Calculating…</span>';
  var r = await api('/api/vram-calc', 'POST', { path: modelPath, kv_k: kvK, kv_v: kvV, dflash: false });
  if (!r || r.error || !r.estimates) { el.innerHTML = '<span class="muted">—</span>'; return; }

  el.innerHTML =
    '<table class="data" style="font-size:11px;margin-top:6px">' +
    '<thead><tr><th>Ctx</th><th>Model</th><th>KV</th><th>GPU total</th><th>Free@peak</th></tr></thead>' +
    '<tbody>' +
    r.estimates.map(function(e) {
      var cls = e.free_peak >= 1024 ? 'success-text' : (e.free_peak >= 200 ? 'warn-text' : 'error-text');
      return '<tr>' +
        '<td>' + fmt(e.ctx) + '</td>' +
        '<td>' + fmtMib(e.model) + '</td>' +
        '<td>' + fmtMib(e.kv) + '</td>' +
        '<td>' + fmtMib(e.gpu_total) + '</td>' +
        '<td class="' + cls + '">' + fmtMib(e.free_peak) + '</td>' +
        '</tr>';
    }).join('') +
    '</tbody></table>';
}

// ── loadRawEnv / saveRawEnv ───────────────────────────────────────────────────
async function loadRawEnv() {
  var data = await api('/api/env-raw');
  var el = document.getElementById('rawEnv');
  if (el) el.value = (data && !data.error && data.content) ? data.content : '';
}

async function saveRawEnv() {
  var el = document.getElementById('rawEnv');
  if (!el) return;
  if (!(await confirmDialog('Save raw environment file? This will overwrite all settings.'))) return;
  var r = await api('/api/env-raw', 'PUT', { content: el.value });
  if (r && r.error) return;
  toast('Environment file saved!', 'success');
}

// ── refreshPower ──────────────────────────────────────────────────────────────
async function refreshPower() {
  _loadPowerStatus();
  _loadPowerLimit();
  var results = await Promise.all([
    api('/api/energy/total'),
    api('/api/gpu/detail'),
    api('/api/gpu/thermal'),
    api('/api/gpu/clocks'),
  ]);
  var energy  = results[0];
  var gpu     = results[1];
  var thermal = results[2];
  var clocks  = results[3];

  // Power stats
  if (energy && !energy.error) {
    setVal('gpuPower',    energy.gpu_power_w  != null ? energy.gpu_power_w.toFixed(1) : '—');
    setVal('totalEnergy', energy.total_wh     != null ? energy.total_wh.toFixed(3)   : '—');
    // If there's a sysPower element
    var sysPEl = document.getElementById('sysPower');
    if (sysPEl) sysPEl.textContent = energy.system_power_w != null ? energy.system_power_w.toFixed(1) : '—';
    var totalPEl = document.getElementById('totalPower');
    if (totalPEl) totalPEl.textContent = energy.total_power_w != null ? energy.total_power_w.toFixed(1) : '—';
  }

  // GPU details table
  var gpuDetailsEl = document.getElementById('gpuDetails');
  if (gpuDetailsEl && gpu && !gpu.error) {
    var tempStr = '—';
    if (thermal && !thermal.error && thermal.current_c != null) {
      var tPct = thermal.pct;
      var tColor = tPct > 90 ? 'var(--red)' : (tPct > 75 ? 'var(--yellow)' : 'var(--green)');
      tempStr = '<span style="color:' + tColor + '">' + thermal.current_c + '°C (' + tPct + '%)</span>';
    }
    var coreClk = (clocks && !clocks.error && clocks.graphics_mhz) ? clocks.graphics_mhz + ' MHz' : (gpu.clock_mhz ? gpu.clock_mhz + ' MHz' : '—');
    var memClk  = (clocks && !clocks.error && clocks.memory_mhz)   ? clocks.memory_mhz  + ' MHz' : (gpu.memory_clock_mhz ? gpu.memory_clock_mhz + ' MHz' : '—');

    var rows = [
      ['GPU',        gpu.name || '—'],
      ['VRAM',       (gpu.vram_used_mib || '—') + ' / ' + (gpu.vram_total_mib || '—') + ' MiB'],
      ['Compute',    (gpu.util_pct != null ? gpu.util_pct : '—') + '%'],
      ['Mem BW',     (gpu.memory_util_pct != null ? gpu.memory_util_pct : '—') + '%'],
      ['Temp',       tempStr],
      ['Core Clock', coreClk],
      ['Mem Clock',  memClk],
      ['Fan',        gpu.fan_rpm != null ? gpu.fan_rpm + ' RPM' : '—'],
      ['Power',      (gpu.power_w != null ? gpu.power_w : '—') + ' W'],
    ];
    gpuDetailsEl.innerHTML =
      '<table style="width:100%;border-collapse:collapse">' +
      rows.map(function(row) {
        return '<tr>' +
          '<td style="padding:5px 4px;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:40%">' + row[0] + '</td>' +
          '<td style="padding:5px 4px;font-weight:600">' + row[1] + '</td>' +
          '</tr>';
      }).join('') +
      '</table>';
  }
}

// ── loadGpuProcesses ──────────────────────────────────────────────────────────
async function loadGpuProcesses() {
  var procs = await api('/api/gpu/processes');
  var el = document.getElementById('gpuProcs');
  if (!el) return;
  if (!procs || procs.error || !Array.isArray(procs) || !procs.length) {
    el.innerHTML = '<span class="muted">No GPU compute processes.</span>';
    return;
  }
  el.innerHTML =
    '<table class="data" style="font-size:12px">' +
    '<thead><tr><th>PID</th><th>Process</th><th>VRAM</th></tr></thead>' +
    '<tbody>' +
    procs.map(function(p) {
      return '<tr><td class="font-mono">' + p.pid + '</td><td>' + p.name + '</td><td class="font-mono">' + p.memory_mib + ' MiB</td></tr>';
    }).join('') +
    '</tbody></table>';
}

// ── loadSystemInfo ────────────────────────────────────────────────────────────
async function loadSystemInfo() {
  var r = await api('/api/system/info');
  var el = document.getElementById('sysInfo');
  if (!el || !r || r.error) return;
  var uh = Math.floor((r.uptime_secs || 0) / 3600);
  var um = Math.floor(((r.uptime_secs || 0) % 3600) / 60);
  var ramPct = r.ram_total_mib > 0 ? ((r.ram_used_mib / r.ram_total_mib) * 100).toFixed(1) : '—';
  var rows = [
    ['CPU Usage', r.cpu_pct + '% (' + r.cpu_cores + ' cores)'],
    ['CPU Temp',  r.cpu_temp_c != null ? r.cpu_temp_c + '°C' : '—'],
    ['RAM',       Math.round(r.ram_used_mib).toLocaleString() + ' / ' + Math.round(r.ram_total_mib).toLocaleString() + ' MiB (' + ramPct + '%)'],
    ['Available', Math.round(r.ram_available_mib).toLocaleString() + ' MiB'],
    ['Cached',    Math.round(r.ram_cached_mib).toLocaleString() + ' MiB'],
    ['Wired',     Math.round(r.wired_mib || 0).toLocaleString() + ' MiB'],
    ['Compressed', Math.round(r.compressed_mib || 0).toLocaleString() + ' MiB'],
    ['Swap',      Math.round(r.swap_total_mib - r.swap_free_mib).toLocaleString() + ' / ' + Math.round(r.swap_total_mib).toLocaleString() + ' MiB'],
    ['Metal ceiling', r.wired_limit_mib ? Math.round(r.wired_limit_mib).toLocaleString() + ' MiB' : 'macOS default'],
    ['LLM sleep guard', r.server_mode ? 'active (display may sleep)' : 'off'],
    ['Power source', r.power_source || '—'],
    ['Uptime',    uh + 'h ' + um + 'm'],
  ];
  el.innerHTML =
    '<table class="kv-table">' +
    rows.map(function(row) {
      return '<tr><td>' + row[0] + '</td><td>' + row[1] + '</td></tr>';
    }).join('') +
    '</table>';
}

// ── systemPower ───────────────────────────────────────────────────────────────
async function _loadPowerLimit() {
  var info = document.getElementById('powerLimitInfo');
  var inp  = document.getElementById('powerLimitInput');
  var sld  = document.getElementById('powerLimitSlider');
  var btn  = document.getElementById('powerLimitApply');
  if (!info) return;
  var p = await api('/api/memory/metal-limit');
  if (!p || p.error) { info.textContent = 'Could not read Metal memory ceiling.'; return; }

  // Set input/slider bounds + current value from the GPU's reported range.
  if (inp && sld && p.min_mib != null && p.max_mib != null) {
    inp.min = sld.min = p.min_mib;
    inp.max = sld.max = p.max_mib;
    inp.step = sld.step = p.step_mib || 256;
    if (document.activeElement !== inp && document.activeElement !== sld) {
      inp.value = sld.value = (p.current_mib || 28672);
    }
    // Keep slider and number input in sync (attach once).
    if (!sld._wired) {
      sld.addEventListener('input', function () { inp.value = sld.value; });
      inp.addEventListener('input', function () { sld.value = inp.value; });
      sld._wired = true;
    }
  }

  var sudoNote = p.sudo_ok
    ? '<span class="success-text">✓ privileged helper ready</span>'
    : '<span class="warn-text">⚠ memory helper is not installed</span>';
  if (btn) btn.disabled = !p.sudo_ok;

  info.innerHTML =
    'Current: <b style="color:var(--text)">' + (p.current_mib ? p.current_mib + ' MiB' : 'macOS default') + '</b>' +
    ' &nbsp;·&nbsp; Range: ' + p.min_mib + '–' + p.max_mib + ' MiB' +
    ' &nbsp;·&nbsp; <span class="muted">not persistent across reboot</span><br>' + sudoNote;
}

async function setPowerLimit(mode) {
  var resultEl = document.getElementById('powerLimitResult');
  var inp = document.getElementById('powerLimitInput');
  var limit;
  if (mode === 'default') {
    limit = 0;
  } else {
    limit = parseInt(inp ? inp.value : '0', 10);
    if (!limit) { if (resultEl) resultEl.innerHTML = '<span class="error-text">Enter a MiB ceiling.</span>'; return; }
  }
  var restart = mode === 'restart';
  var label = limit ? limit + ' MiB' : 'the macOS default';
  if (!(await confirmDialog('Set the Metal wired-memory ceiling to ' + label + (restart ? ' and restart the model?' : '?')))) return;
  if (resultEl) resultEl.innerHTML = '<span style="color:var(--muted)">Applying…</span>';
  var r = await api('/api/memory/metal-limit', 'POST', { limit_mib: limit, restart_runtime: restart });
  if (r && r.ok) {
    if (resultEl) resultEl.innerHTML = '<span class="success-text">✓ Metal ceiling set to ' + (r.current_mib || 'macOS default') + (r.current_mib ? ' MiB' : '') + (r.runtime_restarted ? '; model restarted' : '') + '</span>';
  } else {
    if (resultEl) resultEl.innerHTML = '<span class="error-text">' + (r && r.error ? r.error : 'Failed') + '</span>';
  }
  _loadPowerLimit(); loadSystemInfo();
}

function applyMemoryPreset(limit) {
  var inp = document.getElementById('powerLimitInput');
  var sld = document.getElementById('powerLimitSlider');
  if (inp) inp.value = limit;
  if (sld) sld.value = limit;
}

async function leaveLlmMode() {
  if (!(await confirmDialog(
    'Return to normal macOS mode? This stops the active LLM, releases its unified-memory allocation, restores the default Metal wired-memory limit, and disables the server sleep guard. The MacBook lid must be OPEN.',
    { title: 'Normal macOS mode', okLabel: 'Release GPU memory' }
  ))) return;
  var resultEl = document.getElementById('powerLimitResult');
  var dashboardResult = document.getElementById('normalModeStatus');
  var buttons = document.querySelectorAll('[data-normal-macos-button]');
  buttons.forEach(function(button) { button.disabled = true; });
  if (resultEl) resultEl.textContent = 'Returning to normal macOS mode…';
  if (dashboardResult) dashboardResult.textContent = 'Releasing model and reserved Metal memory…';
  try {
    var r = await api('/api/system/llm-mode/normal', 'POST');
    if (r && r.ok) {
      var success = '✓ Normal macOS mode active — inference stopped and GPU reservation released';
      if (resultEl) resultEl.innerHTML = '<span class="success-text">' + success + '</span>';
      if (dashboardResult) dashboardResult.innerHTML = '<span class="success-text">' + success + '</span>';
      toast('Normal macOS mode restored', 'success');
    } else if (dashboardResult) {
      dashboardResult.innerHTML = '<span class="error-text">' +
        (r && r.error ? r.error : 'Unable to restore normal macOS mode') +
        '</span>';
    }
  } finally {
    buttons.forEach(function(button) { button.disabled = false; });
  }
  _loadPowerLimit(); _loadPowerStatus(); refreshAll(); loadSystemInfo();
}

async function stopModelKeepServerMode() {
  if (!(await confirmDialog('Stop llama.cpp and release model memory while keeping SSH/server mode awake?'))) return;
  var resultEl = document.getElementById('powerLimitResult');
  if (resultEl) resultEl.textContent = 'Stopping model; preserving sleep guard…';
  var r = await api('/api/service/stop', 'POST');
  if (r && !r.error) {
    if (resultEl) resultEl.innerHTML = '<span class="success-text">✓ Model stopped; SSH sleep guard remains active</span>';
    toast('Model stopped; server mode remains awake', 'success');
  }
  _loadPowerLimit(); refreshAll(); loadSystemInfo();
}

async function _loadPowerStatus() {
  var s = await api('/api/system/llm-mode');
  var el = document.getElementById('powerStatusRow');
  if (!el) return;
  if (!s || s.error) { el.textContent = ''; return; }
  el.innerHTML = '<table class="kv-table">' +
    '<tr><td>Mode</td><td>' + (s.enabled ? '<span class="success-text">LLM server</span>' : 'normal macOS') + '</td></tr>' +
    '<tr><td>Metal ceiling</td><td>' + (s.metal_limit_mib ? s.metal_limit_mib + ' MiB' : 'macOS default') + '</td></tr>' +
    '<tr><td>Lid</td><td>' + (s.clamshell_closed ? '<span class="warn-text">closed</span>' : 'open') + '</td></tr>' +
    '<tr><td>Sleep guard</td><td>' + (s.caffeinate_active ? 'active for server mode' : 'off') + '</td></tr>' +
    '<tr><td>Power</td><td>' + (s.power_source || '—') + '</td></tr>' +
    '<tr><td>Persistence</td><td>runtime only; resets at reboot</td></tr></table>';
}

// ── runQuickBench ─────────────────────────────────────────────────────────────
async function runQuickBench() {
  var el = document.getElementById('quickBenchResults');
  if (!el) return;
  el.innerHTML = '<div class="spinner"></div> Running 50-token test…';
  var r = await api('/api/benchmark/quick');
  if (!r || r.error || !r.ok) {
    el.innerHTML = '<div style="color:var(--red);margin-top:8px">' + (r && r.error ? r.error : 'Benchmark failed — is service running?') + '</div>';
    return;
  }
  el.innerHTML =
    '<div class="grid-2 gap-2" style="margin-top:12px">' +
    '<div class="stat-card"><div class="label">Latency</div><div class="value text-accent">' + r.latency_ms + '</div><div class="unit">ms</div></div>' +
    '<div class="stat-card"><div class="label">Speed</div><div class="value text-green">' + (r.tok_per_sec || '—') + '</div><div class="unit">tok/s</div></div>' +
    '</div>';
}

async function runMlxNeedle() {
  var resultEl = document.getElementById('mlxNeedleResults');
  var targetEl = document.getElementById('mlxNeedleTokens');
  if (!resultEl) return;
  var target = parseInt(targetEl && targetEl.value) || 32768;
  resultEl.innerHTML = '<div class="spinner"></div> Starting persistent probe…';
  var r = await api('/api/bench/mlx-needle', 'POST', { target_tokens: target });
  if (!r || r.error || !r.task_id) {
    resultEl.innerHTML = '<span class="error-text">' + (r && r.error ? r.error : 'Unable to start probe') + '</span>';
    return;
  }
  var tid = r.task_id;
  var poll = setInterval(async function() {
    var task = await api('/api/tasks/' + tid);
    if (!task || task.error) { clearInterval(poll); return; }
    if (task.status === 'running') {
      resultEl.textContent = 'Running ' + fmt(target) + '-token retrieval probe. You may leave this tab; task ID: ' + tid;
      return;
    }
    clearInterval(poll);
    var x = task.result || {};
    if (task.status === 'done' && x.needle_found) {
      resultEl.innerHTML = '<span class="success-text">✓ Needle found</span> · ' +
        fmt(x.actual_tokens || target) + ' tokens · ' + (x.elapsed_seconds || '—') + ' s' +
        '<div class="font-sm muted">Saved: ' + (x.path || 'benchmark-results') + '</div>';
    } else {
      resultEl.innerHTML = '<span class="error-text">Probe ' + task.status + '</span> · ' +
        (x.error || (x.needle_found === false ? 'needle was not retrieved' : 'see task log'));
    }
    loadTasks();
  }, 3000);
}

// ── runSweep ──────────────────────────────────────────────────────────────────
function _renderSweepTable(results) {
  return '<table class="data" style="font-size:12px;margin-top:8px">' +
    '<thead><tr><th>Ctx</th><th>Status</th><th>Idle VRAM</th><th>Peak free</th><th>Prompt t/s</th><th>Gen t/s</th></tr></thead>' +
    '<tbody>' +
    results.map(function(row) {
      var ok = row.status === 'ok';
      return '<tr>' +
        '<td>' + fmt(row.ctx) + '</td>' +
        '<td>' + (ok ? '<span class="success-text">✓</span>' : '<span class="error-text">' + row.status + '</span>') + '</td>' +
        '<td>' + (row.idle_vram ? fmtMib(row.idle_vram) : '—') + '</td>' +
        '<td>' + (row.peak_free ? fmtMib(row.peak_free) : '—') + '</td>' +
        '<td>' + (row.prompt_tps ? row.prompt_tps.toFixed(1) : '—') + '</td>' +
        '<td>' + (row.gen_tps  ? row.gen_tps.toFixed(1) : '—') + '</td>' +
        '</tr>';
    }).join('') +
    '</tbody></table>';
}

async function runSweep() {
  var ctxEl = document.getElementById('sweepCtx');
  var resultsEl = document.getElementById('sweepResults');
  if (!resultsEl) return;
  var ctxs = (ctxEl ? ctxEl.value : '131072').split(',')
    .map(function(s) { return parseInt(s.trim()); })
    .filter(function(n) { return n && !isNaN(n); });
  if (!ctxs.length) { toast('Enter one or more context sizes (comma-separated)', 'error'); return; }

  var progEl = document.getElementById('sweepProgress');
  var barEl  = document.getElementById('sweepBar');
  var stepEl = document.getElementById('sweepStep');
  if (progEl) progEl.style.display = 'block';
  if (barEl)  barEl.style.width = '0%';
  if (stepEl) stepEl.textContent = 'Starting sweep…';
  resultsEl.innerHTML = '';

  var r = await api('/api/bench/sweep', 'POST', { contexts: ctxs, tokens: 38000 });
  if (!r || r.error || !r.task_id) {
    if (progEl) progEl.style.display = 'none';
    resultsEl.innerHTML = '<span style="color:var(--red)">' + (r && r.error ? r.error : 'Failed to start sweep') + '</span>';
    return;
  }

  var tid = r.task_id;
  var interval = setInterval(async function() {
    var t = await api('/api/tasks/' + tid);
    if (!t || t.error) { clearInterval(interval); return; }
    var pct = t.progress != null ? t.progress : 0;
    if (barEl)  barEl.style.width = pct + '%';
    if (stepEl) {
      var step = (t.step != null && t.total_steps) ? ('[' + t.step + '/' + t.total_steps + '] ') : '';
      var last = (t.log || []).slice(-1)[0] || '';
      stepEl.textContent = step + last;
    }
    if (t.status === 'running' || t.status === 'stalled') return;
    clearInterval(interval);
    if (barEl) barEl.style.width = '100%';
    if (t.result && t.result.results) {
      resultsEl.innerHTML = _renderSweepTable(t.result.results);
      // Plot the just-completed run.
      var canvas = document.getElementById('sweepChart');
      if (canvas) {
        canvas.style.display = 'block';
        drawSweepChart('sweepChart', [{ results: t.result.results, label: 'this run', color: '#22d3ee' }]);
      }
      loadRecentResults();
    } else {
      resultsEl.innerHTML = '<div class="error-text">Sweep ' + t.status + (t.log ? ': ' + t.log.slice(-1)[0] : '') + '</div>';
    }
    setTimeout(function() { if (progEl) progEl.style.display = 'none'; }, 1500);
  }, 2000);
}

// ── testGen / testPrompt / testCodebase ───────────────────────────────────────
async function testGen() {
  var el = document.getElementById('genResult');
  if (!el) return;
  el.innerHTML = '<div class="spinner"></div>';
  var tokens = parseInt((document.getElementById('genTokens') || {}).value) || 100;
  var r = await api('/api/test/gen', 'POST', { tokens: tokens });
  el.innerHTML = fmtTimings(r);
}

async function testPrompt() {
  var el = document.getElementById('promptResult');
  if (!el) return;
  el.innerHTML = '<div class="spinner"></div>';
  var tokens = parseInt((document.getElementById('promptTokens') || {}).value) || 4096;
  var r = await api('/api/test/prompt', 'POST', { tokens: tokens });
  el.innerHTML = fmtTimings(r);
}

async function testCodebase() {
  var el = document.getElementById('cbResult');
  if (!el) return;
  el.innerHTML = '<div class="spinner"></div>';
  var tokens = parseInt((document.getElementById('cbTokens') || {}).value) || 16000;
  var gen    = parseInt((document.getElementById('cbGen')    || {}).value) || 200;
  var r = await api('/api/test/codebase', 'POST', { tokens: tokens, gen: gen });
  el.innerHTML = fmtTimings(r);
}

// ── countTokens ───────────────────────────────────────────────────────────────
async function countTokens() {
  var textEl   = document.getElementById('tokenCountText');
  var resultEl = document.getElementById('tokenCountResult');
  if (!textEl || !resultEl) return;
  var text = textEl.value.trim();
  if (!text) { toast('Enter some text first', 'error'); return; }
  var r = await api('/api/tokens/count?text=' + encodeURIComponent(text));
  if (!r || r.error) { resultEl.textContent = 'Error'; return; }
  resultEl.textContent = (r.count || '—') + ' tokens (model: ' + (r.model || '?') + ')';
}

// ── calcVRAM ──────────────────────────────────────────────────────────────────
async function calcVRAM() {
  var modelEl  = document.getElementById('calcModel');
  var ctxEl    = document.getElementById('calcCtx');
  var kvEl     = document.getElementById('calcKv');
  var resultEl = document.getElementById('calcResult');
  if (!resultEl) return;

  var modelPath = modelEl ? modelEl.value : '';
  var ctx = parseInt(ctxEl ? ctxEl.value : '131072') || 131072;
  var kv  = kvEl ? kvEl.value : 'turbo4';

  if (!modelPath) { toast('Select a model first', 'error'); return; }

  resultEl.innerHTML = '<span class="muted">Calculating…</span>';
  var r = await api('/api/vram-calc', 'POST', { path: modelPath, kv_k: kv, kv_v: kv });
  if (!r || r.error || !r.estimates) {
    resultEl.innerHTML = '<span style="color:var(--red)">' + (r && r.error ? r.error : 'Error') + '</span>';
    return;
  }

  var row = r.estimates.find(function(e) { return e.ctx >= ctx; }) || r.estimates[r.estimates.length - 1];
  if (!row) { resultEl.innerHTML = '<span class="muted">No estimates.</span>'; return; }

  var cls = row.free_peak >= 1024 ? 'text-green' : (row.free_peak >= 200 ? 'text-yellow' : 'text-red');
  resultEl.innerHTML =
    '<div class="grid-2 gap-2" style="margin-top:12px">' +
    '<div class="stat-card"><div class="label">Context</div><div class="value text-accent">' + (row.ctx / 1024).toFixed(0) + 'K</div></div>' +
    '<div class="stat-card"><div class="label">Free @ peak</div><div class="value ' + cls + '">' + Math.round(row.free_peak) + '</div><div class="unit">MiB</div></div>' +
    '<div class="stat-card"><div class="label">Model MiB</div><div class="value">' + Math.round(row.model || 0) + '</div></div>' +
    '<div class="stat-card"><div class="label">KV MiB</div><div class="value">' + Math.round(row.kv || 0) + '</div></div>' +
    '</div>';
}

// ── loadRecentResults ─────────────────────────────────────────────────────────
async function loadRecentResults() {
  var results = await api('/api/results/recent');
  var el = document.getElementById('recentResults');
  if (!el) return;
  if (!results || results.error || !Array.isArray(results) || !results.length) {
    el.innerHTML = '<span class="muted">No results yet.</span>';
    return;
  }
  el.innerHTML =
    '<table class="data" style="font-size:12px">' +
    '<thead><tr><th style="width:28px"></th><th>File</th><th>Size</th><th>Date</th><th>Export</th></tr></thead>' +
    '<tbody>' +
    results.map(function(r) {
      var n = _esc(r.name);
      return '<tr>' +
        '<td><input type="checkbox" class="cmp-pick" value="' + n + '"></td>' +
        '<td><button onclick="viewResult(\'' + n + '\')" style="background:none;border:none;color:var(--cyan);cursor:pointer;padding:0;font-size:12px">' + r.name + '</button></td>' +
        '<td class="font-mono">' + Math.round((r.size || 0) / 1024) + ' KB</td>' +
        '<td class="font-mono">' + new Date((r.mtime || 0) * 1000).toLocaleString() + '</td>' +
        '<td>' +
          '<a href="/api/results/' + n + '/csv" class="muted" style="font-size:11px;text-decoration:none;margin-right:8px" title="Download CSV">CSV</a>' +
          '<a href="/api/results/' + n + '" class="muted" style="font-size:11px;text-decoration:none" title="Raw JSON" target="_blank">JSON</a>' +
        '</td>' +
        '</tr>';
    }).join('') +
    '</tbody></table>';
}

// ── compareSelected ─ overlay up to 2 ticked saved runs on the sweep chart ─────
var _CMP_COLORS = ['#22d3ee', '#f97316'];
async function compareSelected() {
  var picks = Array.prototype.slice.call(document.querySelectorAll('.cmp-pick:checked'))
    .map(function(c) { return c.value; });
  if (!picks.length) { toast('Tick 1–2 result rows first', 'error'); return; }
  if (picks.length > 2) { toast('Compare at most 2 runs', 'error'); return; }
  var series = [];
  for (var i = 0; i < picks.length; i++) {
    var data = await api('/api/results/' + picks[i]);
    if (!data || data.error || !data.results) { toast('Cannot load ' + picks[i], 'error'); continue; }
    series.push({ results: data.results, label: picks[i].replace(/^sweep-|\.json$/g, ''), color: _CMP_COLORS[i] });
  }
  if (!series.length) return;
  var canvas = document.getElementById('sweepChart');
  if (canvas) { canvas.style.display = 'block'; drawSweepChart('sweepChart', series); }
}

// ── viewResult ────────────────────────────────────────────────────────────────
async function viewResult(name) {
  var data = await api('/api/results/' + name);
  if (!data || data.error) { toast('Cannot load result: ' + name, 'error'); return; }
  // Markdown result (research report): show the text, not JSON.
  if (data.kind === 'markdown') {
    var elm = document.getElementById('recentResults');
    if (!elm) return;
    var ex = elm.querySelector('pre.result-expand');
    if (ex) ex.remove();
    var pm = document.createElement('pre');
    pm.className = 'result-expand';
    pm.style.cssText = 'background:rgba(0,0,0,.4);padding:12px;border-radius:6px;font-size:12px;max-height:480px;overflow:auto;margin-top:8px;white-space:pre-wrap';
    pm.textContent = data.markdown;
    elm.appendChild(pm);
    return;
  }
  // If this is a sweep result, plot it on the chart.
  if (data.results && data.results.length) {
    var canvas = document.getElementById('sweepChart');
    if (canvas) { canvas.style.display = 'block'; drawSweepChart('sweepChart', [{ results: data.results, label: name.replace(/^sweep-|\.json$/g, ''), color: '#22d3ee' }]); }
  }
  var el = document.getElementById('recentResults');
  if (!el) return;
  // Remove existing expanded pre if any
  var existing = el.querySelector('pre.result-expand');
  if (existing) existing.remove();
  var pre = document.createElement('pre');
  pre.className = 'result-expand';
  pre.style.cssText = 'background:rgba(0,0,0,.4);padding:12px;border-radius:6px;font-size:11px;max-height:360px;overflow:auto;margin-top:8px;white-space:pre-wrap;word-break:break-all';
  pre.textContent = JSON.stringify(data, null, 2);
  el.appendChild(pre);
}

// ── loadLogs ──────────────────────────────────────────────────────────────────
async function loadLogs() {
  var filterEl = document.getElementById('logFilter');
  var linesEl  = document.getElementById('logLines');
  var contentEl = document.getElementById('logContent');
  if (!contentEl) return;

  var filter = filterEl ? filterEl.value : 'all';
  var lines  = linesEl  ? parseInt(linesEl.value) || 50 : 50;

  var svcEl = document.getElementById('logService');
  var svc = svcEl ? svcEl.value : 'qwen';
  var r = (svc === 'qwen')
    ? await api('/api/logs?lines=' + lines)
    : await api('/api/services/stack/' + svc + '/logs?lines=' + lines);
  if (!r || r.error) { contentEl.textContent = '(failed to load logs)'; return; }

  var content = r.content || '(no logs)';
  if (filter !== 'all') {
    var filterFn;
    if (filter === 'checkpoint') filterFn = function(l) { return l.toLowerCase().includes('checkpoint'); };
    else if (filter === 'error') filterFn = function(l) { return l.toLowerCase().includes('error'); };
    else if (filter === 'warning') filterFn = function(l) { return l.toLowerCase().includes('warn'); };
    else filterFn = function() { return true; };
    content = content.split('\n').filter(filterFn).join('\n') || '(no matching lines)';
  }
  contentEl.textContent = content;
  contentEl.scrollTop = contentEl.scrollHeight;
}

// ── Live log follow (SSE with polling fallback) ───────────────────────────────
var _logStream = null;

function toggleAutoLogs(checked) {
  if (_autoLogsInterval) { clearInterval(_autoLogsInterval); _autoLogsInterval = null; }
  if (_logStream) { _logStream.close(); _logStream = null; }
  var contentEl = document.getElementById('logContent');
  if (!checked) { toast('Live log follow off', 'info'); return; }

  var svcEl = document.getElementById('logService');
  var svc = svcEl ? svcEl.value : 'qwen';
  if (svc !== 'qwen' || typeof EventSource === 'undefined') {
    _autoLogsInterval = setInterval(loadLogs, 5000);
    toast('Auto-refresh logs on (5s)', 'info');
    return;
  }

  // The runtime log streams over server-sent events and survives model restarts.
  _logStream = new EventSource('/api/logs/stream');
  var buffer = [];
  _logStream.onmessage = function (e) {
    var msg;
    try { msg = JSON.parse(e.data); } catch (_) { return; }
    if (msg.reset) buffer = [];
    if (msg.lines && msg.lines.length) buffer = buffer.concat(msg.lines).slice(-800);
    if (contentEl) {
      var atBottom = contentEl.scrollTop + contentEl.clientHeight >= contentEl.scrollHeight - 30;
      contentEl.textContent = buffer.join('\n') || '(no log output yet)';
      if (atBottom) contentEl.scrollTop = contentEl.scrollHeight;
    }
  };
  _logStream.onerror = function () {
    // Fall back to polling if the stream drops and the browser gives up.
    if (_logStream && _logStream.readyState === EventSource.CLOSED) {
      _logStream = null;
      _autoLogsInterval = setInterval(loadLogs, 5000);
    }
  };
  toast('Live log follow on', 'success');
}

// ── loadSettings ─────────────────────────────────────────────────────────────
async function loadSettings() {
  await Promise.all([
    loadProfiles(),
    loadBuildStatus(),
    loadRawEnv(),
    loadDiskUsage(),
    checkSystemd(),
    loadAuthStatus(),
  ]);
}

// ── saveProfile / loadProfileCfg / deleteProfile ──────────────────────────────
async function saveProfile() {
  var nameEl = document.getElementById('profileName');
  var name = nameEl ? nameEl.value.trim() : '';
  if (!name) { toast('Enter a profile name', 'error'); return; }
  var r = await api('/api/profiles', 'POST', { name: name });
  if (r && r.error) return;
  toast('Profile saved: ' + name, 'success');
  loadProfiles();
}

async function loadProfileCfg(name) {
  if (!(await confirmDialog('Load profile "' + name + '" and restart service?'))) return;
  var r = await api('/api/profiles/' + encodeURIComponent(name) + '/load', 'POST');
  if (r && r.error) return;
  toast('Profile "' + name + '" loaded — service restarting…', 'info');
  setTimeout(refreshAll, 3000);
}

async function deleteProfile(name) {
  if (!(await confirmDialog('Delete profile "' + name + '"?'))) return;
  var r = await api('/api/profiles/' + encodeURIComponent(name), 'DELETE');
  if (r && r.error) return;
  toast('Deleted profile: ' + name, 'success');
  loadProfiles();
}

async function loadProfiles() {
  var profiles = await api('/api/profiles');
  var el = document.getElementById('profilesList');
  if (!el) return;
  if (!profiles || profiles.error || !Array.isArray(profiles) || !profiles.length) {
    el.innerHTML = '<span class="muted">No saved profiles.</span>';
    return;
  }
  el.innerHTML =
    '<table class="data" style="font-size:12px;margin-top:8px">' +
    '<thead><tr><th>Name</th><th>Engine / model</th><th>Context / KV</th><th>Description</th><th>Actions</th></tr></thead>' +
    '<tbody>' +
    profiles.map(function(p) {
      var name = typeof p === 'string' ? p : p.name;
      var model = p && p.model ? _esc(p.model) : '—';
      var draft = p && p.draft ? '<br><span class="muted">draft: ' + _esc(p.draft) + '</span>' : '';
      var kv = p && (p.kv_k || p.kv_v) ? _esc(String(p.kv_k || '—') + '/' + String(p.kv_v || '—')) : '—';
      var detail = p && p.modified
        ? '<br><span class="muted">saved ' + new Date(p.modified * 1000).toLocaleString() + '</span>'
        : '';
      return '<tr>' +
        '<td>' + _esc(name) + detail + '</td>' +
        '<td><span class="pill purple" style="font-size:10px">' + _esc((p && p.engine) || '—') + '</span><br>' + model + draft + '</td>' +
        '<td>' + _esc(String((p && p.ctx) || '—')) + '<br><span class="muted">' + kv + '</span></td>' +
        '<td class="muted" style="max-width:440px">' + _esc((p && p.description) || '') + '</td>' +
        '<td>' +
          '<button onclick="loadProfileCfg(\'' + _esc(name) + '\')" class="btn-success" style="font-size:11px;padding:2px 5px">Load</button> ' +
          ((p && p.built_in) ? '' : '<button onclick="deleteProfile(\'' + _esc(name) + '\')" class="btn-danger" style="font-size:11px;padding:2px 5px">Delete</button>') +
        '</td>' +
        '</tr>';
    }).join('') +
    '</tbody></table>';
}

// ── exportConfig ──────────────────────────────────────────────────────────────
async function exportConfig() {
  var r = await api('/api/config/export', 'POST');
  if (!r || r.error) return;
  var blob = new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'llm-config-' + new Date().toISOString().slice(0, 10) + '.json';
  document.body.appendChild(a);
  a.click();
  setTimeout(function() { URL.revokeObjectURL(url); a.remove(); }, 1000);
  toast('Config exported', 'success');
}

// ── importConfig ──────────────────────────────────────────────────────────────
async function importConfig() {
  var input = document.createElement('input');
  input.type = 'file';
  input.accept = 'application/json,.json';
  input.onchange = async function() {
    var file = input.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = async function(e) {
      try {
        var cfg = JSON.parse(e.target.result);
        var r = await api('/api/config/import', 'POST', cfg);
        if (r && r.error) return;
        toast('Config imported successfully', 'success');
        loadConfig();
      } catch (err) {
        toast('Invalid JSON file: ' + err.message, 'error');
      }
    };
    reader.readAsText(file);
  };
  input.click();
}

// ── setHfToken ────────────────────────────────────────────────────────────────
async function setHfToken() {
  var el = document.getElementById('hfTokenInput');
  if (!el) return;
  var token = el.value.trim();
  if (!token) { toast('Enter a HuggingFace token', 'error'); return; }
  if (!token.startsWith('hf_')) { toast('Token should start with hf_', 'error'); return; }
  var r = await api('/api/hf/token', 'PUT', { token: token });
  if (r && r.error) return;
  toast('HF token saved', 'success');
  el.value = '';
  loadHfCliStatus();
}

// ── loadBuildStatus ───────────────────────────────────────────────────────────
async function loadBuildStatus() {
  var r = await api('/api/build/status');
  var el = document.getElementById('buildTable');
  if (!el || !r || r.error) return;
  el.innerHTML = Object.entries(r).map(function(kv) {
    return '<tr><td>' + kv[0] + '</td><td>' + kv[1] + '</td></tr>';
  }).join('');
}

// ── pullAndRebuild ────────────────────────────────────────────────────────────
async function pullAndRebuild() {
  if (!(await confirmDialog('Pull latest code and rebuild llama.cpp? This takes 10-15 minutes.'))) return;
  var r = await api('/api/build/rebuild', 'POST');
  if (!r || r.error || !r.task_id) { toast('Failed to start rebuild', 'error'); return; }
  var logEl = document.getElementById('rebuildLog');
  toast('Rebuild started', 'info');
  var interval = setInterval(async function() {
    var t = await api('/api/tasks/' + r.task_id);
    if (!t || t.error) { clearInterval(interval); return; }
    if (logEl) logEl.textContent = (t.log || []).join('\n');
    if (t.status !== 'running' && t.status !== 'stalled') {
      clearInterval(interval);
      toast('Rebuild ' + t.status, t.status === 'done' ? 'success' : 'error');
      loadBuildStatus();
    }
  }, 1500);
}

// ── clearCache ────────────────────────────────────────────────────────────────
async function clearCache(kind) {
  var msgs = { run: 'Clear .run/ directory?', logs: 'Clear log files?', os: 'Drop OS page cache? (needs sudo)', all: 'Nuke ALL caches? This cannot be undone.' };
  var msg = msgs[kind] || ('Clear ' + kind + ' cache?');
  if (!(await confirmDialog(msg))) return;
  var r = await api('/api/cache/' + kind, 'DELETE');
  if (r && r.error) return;
  toast('Cache cleared: ' + kind, 'success');
}

// ── launchd service status (endpoint retained for UI compatibility) ──────────
async function checkSystemd() {
  var results = await Promise.all([api('/api/systemd/status'), api('/api/manager/health')]);
  var status = results[0];
  var health = results[1];
  var el = document.getElementById('systemdStatus');
  if (!el) return;

  if (!status || status.error) {
    el.innerHTML = '<span class="muted">Could not load launchd status</span>';
  } else {
    var isActive = status.active;
    el.innerHTML =
      '<div style="margin-bottom:8px">' +
      '<span class="pill ' + (isActive ? 'green' : 'muted') + '">' + (isActive ? 'active' : 'inactive') + '</span>' +
      '</div>' +
      (status.status ? '<pre style="background:#000;padding:8px;border-radius:4px;font-size:11px;max-height:200px;overflow:auto;white-space:pre-wrap">' + status.status + '</pre>' : '');
  }

  // Drift warning
  if (health && !health.error && health.drift && health.drift > 2) {
    var warnEl = document.getElementById('warningRail');
    if (warnEl) {
      warnEl.style.display = 'flex';
      if (!warnEl.innerHTML.includes('drift')) {
        warnEl.innerHTML += '<span class="pill yellow">⚠ launchd drift ' + health.drift.toFixed(1) + 's</span>';
      }
    }
  }
}

// ── restart/disable launchd controller ────────────────────────────────────────
async function installSystemd() {
  var r = await api('/api/systemd/install', 'POST');
  var el = document.getElementById('systemdStatus');
  if (el) el.innerHTML = '<div class="success-text">' + ((r && r.message) ? r.message : 'Installed') + '</div>';
  toast('launchd service is installed and enabled', 'success');
}

async function uninstallSystemd() {
  if (!(await confirmDialog('Disable launchd auto-start? This is blocked from the web UI.'))) return;
  var r = await api('/api/systemd/uninstall', 'POST');
  var el = document.getElementById('systemdStatus');
  if (el) el.innerHTML = '<div class="muted">' + ((r && r.message) ? r.message : 'Uninstalled') + '</div>';
  toast('launchd request completed', 'info');
}

// ── loadDiskUsage ─────────────────────────────────────────────────────────────
async function loadDiskUsage() {
  var r = await api('/api/disk/usage');
  var el = document.getElementById('diskUsage');
  if (!el || !r || r.error) return;
  var pct = r.pct || 0;
  var pctColor = pct > 90 ? 'var(--red)' : (pct > 75 ? 'var(--yellow)' : 'var(--green)');
  el.innerHTML =
    '<div style="margin-bottom:6px">' +
    '<div class="progress" style="height:8px;border-radius:4px;background:rgba(255,255,255,.1);overflow:hidden">' +
    '<div style="height:100%;width:' + pct + '%;background:' + pctColor + ';border-radius:4px;transition:width .4s"></div>' +
    '</div>' +
    '</div>' +
    '<table class="kv-table">' +
    '<tr><td>Mount</td><td>' + (r.mount || '/') + '</td></tr>' +
    '<tr><td>Used</td><td>' + fmtSize(r.used) + '</td></tr>' +
    '<tr><td>Available</td><td>' + fmtSize(r.avail) + '</td></tr>' +
    '<tr><td>Total</td><td>' + fmtSize(r.total) + '</td></tr>' +
    '<tr><td>Usage</td><td style="color:' + pctColor + '">' + pct.toFixed(1) + '%</td></tr>' +
    '</table>';
}

// ── exportMetrics ─────────────────────────────────────────────────────────────
async function exportMetrics() {
  var r = await api('/api/metrics/export');
  if (!r || r.error) return;
  var blob = new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'llm-metrics-' + new Date().toISOString().slice(0, 19).replace(/:/g, '-') + '.json';
  document.body.appendChild(a);
  a.click();
  setTimeout(function() { URL.revokeObjectURL(url); a.remove(); }, 1000);
  toast('Metrics exported', 'success');

  var resultEl = document.getElementById('metricsExportResult');
  if (resultEl) resultEl.innerHTML = '<span class="success-text">Exported ' + new Date().toLocaleTimeString() + '</span>';
}

// ── loadAuthStatus ────────────────────────────────────────────────────────────
async function loadAuthStatus() {
  var el = document.getElementById('authStatus');
  if (!el) return;
  el.innerHTML =
    '<tr><td>Auth enabled</td><td>' +
    (_globalAuthEnabled ? '<span class="success-text">yes</span>' : '<span class="warn-text">no (loopback only)</span>') +
    '</td></tr>' +
    '<tr><td>Bind host</td><td>' + _globalBindHost + '</td></tr>';
}

// ── Telemetry health (mactop) ─────────────────────────────────────────────────
var _telemetryOk = null;

async function loadTelemetryHealth() {
  var r = await api('/api/telemetry/health');
  var el = document.getElementById('telemetryHealth');
  if (!r || r.error) return;
  _telemetryOk = !!r.available;
  if (!el) return;
  var state = r.available
    ? '<span class="pill green"><span class="badge-dot green"></span> live</span>'
    : '<span class="pill red"><span class="badge-dot red"></span> unavailable</span>';
  el.innerHTML =
    '<table class="kv-table">' +
    '<tr><td>Sensor stream</td><td>' + state + '</td></tr>' +
    '<tr><td>Source</td><td>' + (r.source || '—') + '</td></tr>' +
    '<tr><td>mactop</td><td>' + (r.mactop_installed ? (r.mactop_pid ? 'running, pid ' + r.mactop_pid : 'installed, supervised by manager') : 'not installed') + '</td></tr>' +
    '</table>' +
    (!r.available ? '<div class="font-sm muted mt-2">GPU power/temperature tiles show — until the sampler is back. The manager retries every ' + Math.round(r.retry_seconds || 60) + 's.</div>' : '');
}

async function restartTelemetry() {
  var r = await api('/api/telemetry/restart', 'POST');
  if (r && r.ok) toast('Telemetry sampler restart requested', 'success');
  setTimeout(loadTelemetryHealth, 2500);
}

// ── resetEnergy ───────────────────────────────────────────────────────────────
async function resetEnergy() {
  if (!(await confirmDialog('Reset cumulative energy counter?'))) return;
  var r = await api('/api/energy/reset', 'POST');
  if (r && r.error) return;
  toast('Energy counter reset', 'success');
  refreshPower();
}

// ── Bootstrap from /api/ui/bootstrap ─────────────────────────────────────────
async function _loadBootstrap() {
  var s = await api('/api/ui/bootstrap');
  if (!s || s.error) return;
  _globalAuthEnabled = !!s.auth_enabled;
  _globalBindHost = s.bind_host || '127.0.0.1';
}

// ── DOMContentLoaded init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  initTabs();

  // Load bootstrap for auth/host globals
  _loadBootstrap().then(function() {
    loadAuthStatus();
  });

  // Initial data load
  refreshAll();
  loadModels();
  loadConfig();
  loadServices();
  loadVramChart();

  // charts.js chart timers
  if (typeof startVramChartTimer  === 'function') startVramChartTimer();
  if (typeof startPowerChartTimer === 'function') startPowerChartTimer();

  // Main refresh interval: status + session stats
  setInterval(refreshAll,   4000);
  setInterval(refreshStats, 2500);

  // Active-tab specific polling
  setInterval(function() {
    if (document.hidden) return;
    var powerTab  = document.getElementById('tab-power');
    if (powerTab && powerTab.classList.contains('active')) {
      refreshPower();
      loadGpuProcesses();
      loadSystemInfo();
      loadTelemetryHealth();
    }
  }, 5000);

  setInterval(function() {
    var modelsTab = document.getElementById('tab-models');
    if (modelsTab && modelsTab.classList.contains('active')) {
      loadModels();
      loadTasks();
    }
  }, 10000);

  // Startup drift warning
  api('/api/manager/health').then(function(mh) {
    if (mh && !mh.error && mh.drift && mh.drift > 2) {
      var rail = document.getElementById('warningRail');
      if (rail) {
        rail.style.display = 'flex';
        rail.innerHTML = (rail.innerHTML || '') + '<span class="pill yellow">⚠ Manager drift detected at startup (' + mh.drift.toFixed(1) + 's)</span>';
      }
      toast('Warning: manager drift ' + mh.drift.toFixed(1) + 's — check launchd', 'error');
    }
  });
});

// ════════════════════════════════════════════════════════════════════════
// MONITOR (Grafana-style dashboard) — tiles, charts, events, settings
// ════════════════════════════════════════════════════════════════════════

var _monitor = {
  window: '15m',
  eventsWindow: '1h',
  timerTiles: null,
  timerCharts: null,
  timerEvents: null,
  lastSnapshot: null,
};

function loadMonitor() {
  _monitorLoadSampleRate();
  _monitorLoadProfileBaselines();
  monitorRefreshNow();
  if (_monitor.timerTiles)  clearInterval(_monitor.timerTiles);
  if (_monitor.timerCharts) clearInterval(_monitor.timerCharts);
  if (_monitor.timerEvents) clearInterval(_monitor.timerEvents);
  _monitor.timerTiles  = setInterval(_monitorPollTiles,  3000);
  _monitor.timerCharts = setInterval(_monitorPollCharts, 15000);
  _monitor.timerEvents = setInterval(_monitorPollEvents, 30000);
}

function monitorStop() {
  if (_monitor.timerTiles)  { clearInterval(_monitor.timerTiles);  _monitor.timerTiles  = null; }
  if (_monitor.timerCharts) { clearInterval(_monitor.timerCharts); _monitor.timerCharts = null; }
  if (_monitor.timerEvents) { clearInterval(_monitor.timerEvents); _monitor.timerEvents = null; }
}

function monitorRefreshNow() {
  _monitorPollTiles();
  _monitorPollCharts();
  _monitorPollEvents();
  _monitorLoadProfileBaselines();
}

function setMonitorWindow(w) {
  _monitor.window = w;
  _updateWindowBtns('monitorWindowBtns', w);
  _monitorPollCharts();
}

function setMonitorEventsWindow(w) {
  _monitor.eventsWindow = w;
  _updateWindowBtns('monitorEventsWindowBtns', w);
  _monitorPollEvents();
}

async function _monitorLoadSampleRate() {
  var r = await api('/api/dashboard/sampling-rate');
  if (r && !r.error) {
    var sel = document.getElementById('monitorSampleRate');
    if (sel) sel.value = String(r.sample_rate_s);
  }
}

async function _monitorSetSampleRate(secs) {
  var r = await api('/api/dashboard/sampling-rate', 'POST', { sample_rate_s: parseInt(secs, 10) });
  if (r && !r.error) {
    toast('Sample rate set to ' + r.sample_rate_s + 's', 'success');
  }
}

function _fmtBytes(mib) {
  if (mib == null) return '—';
  if (mib >= 1024) return (mib / 1024).toFixed(1);
  return String(mib);
}

function _fmtGiB(mib) {
  return mib == null ? '—' : (mib / 1024).toFixed(2);
}

async function _monitorLoadProfileBaselines() {
  var r = await api('/api/dashboard/profile-memory-baselines');
  var el = document.getElementById('profileMemoryBaselines');
  if (!el || !r || r.error || !Array.isArray(r.profiles)) return;
  var rows = r.profiles.map(function(p) {
    return '<tr>' +
      '<td><b>' + p.profile + '</b><br><span class="muted">' + Math.round(p.context / 1024) + 'K · ' + p.kv + '</span></td>' +
      '<td class="font-mono">' + _fmtGiB(p.system_free_mib) + ' GiB</td>' +
      '<td class="font-mono">' + _fmtGiB(p.unified_available_mib) + ' GiB</td>' +
      '<td class="font-mono">' + _fmtGiB(p.gpu_pool_free_mib) + ' GiB</td>' +
      '<td class="font-mono">' + _fmtGiB(p.metal_headroom_mib) + ' GiB</td>' +
      '</tr>';
  }).join('');
  el.innerHTML = '<div style="overflow-x:auto"><table class="data" style="font-size:11px">' +
    '<thead><tr><th>Profile</th><th>System free</th><th>Unified available</th><th>GPU pool free</th><th>Metal headroom</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table></div>' +
    '<p class="font-sm muted mt-2">Measured at ' + _fmtGiB(r.measured_metal_ceiling_mib) + ' GiB Metal ceiling. Values overlap and fluctuate; they are not additive.</p>';
}

function _fmtUptime(secs) {
  if (secs == null || secs < 0) return '—';
  if (secs < 60) return secs + 's';
  if (secs < 3600) return Math.floor(secs / 60) + 'm ' + (secs % 60) + 's';
  if (secs < 86400) return Math.floor(secs / 3600) + 'h ' + Math.floor((secs % 3600) / 60) + 'm';
  return Math.floor(secs / 86400) + 'd ' + Math.floor((secs % 86400) / 3600) + 'h';
}

function _fmtAgo(unix) {
  if (!unix) return '—';
  var s = Math.floor(Date.now() / 1000) - unix;
  return _fmtUptime(s) + ' ago';
}

async function _monitorPollTiles() {
  var s = await api('/api/dashboard/snapshot');
  if (!s || s.error) return;
  _monitor.lastSnapshot = s;

  setVal('monGpuUtil',    s.gpu_util != null ? Math.round(s.gpu_util) : '—');
  setVal('monVram',       s.vram_used_mib != null ? _fmtBytes(s.vram_used_mib) : '—');
  if (s.vram_total_mib) {
    var unit = document.getElementById('monVramUnit');
    if (unit) unit.textContent = 'of ' + (s.vram_total_mib / 1024).toFixed(0) + ' GiB';
  }
  setVal('monPower',      s.gpu_power_w != null ? Math.round(s.gpu_power_w) : '—');
  setVal('monTemp',       s.gpu_temp != null ? s.gpu_temp : '—');
  setVal('monEvalTokS',   s.counters && s.counters.last_eval_tok_s ? s.counters.last_eval_tok_s.toFixed(1) : '—');
  setVal('monPromptTokS', s.counters && s.counters.last_prompt_tok_s ? s.counters.last_prompt_tok_s.toFixed(0) : '—');
  setVal('monCpu',        s.cpu_pct != null ? s.cpu_pct.toFixed(0) : '—');
  setVal('monRam',        s.mem_used_mib != null ? _fmtBytes(s.mem_used_mib) : '—');
  if (s.mem_total_mib) {
    var ru = document.getElementById('monRamUnit');
    if (ru) ru.textContent = 'of ' + (s.mem_total_mib / 1024).toFixed(0) + ' GiB';
  }
  setVal('monSystemFree',       _fmtGiB(s.mem_free_mib));
  setVal('monUnifiedAvailable', _fmtGiB(s.mem_available_mib));
  setVal('monGpuPoolFree',      _fmtGiB(s.vram_free_mib));
  setVal('monMetalHeadroom',    _fmtGiB(s.metal_headroom_mib));

  // Counters table
  var c = s.counters || {};
  setVal('ctrRequests',     c.requests_total != null ? c.requests_total.toLocaleString() : '—');
  setVal('ctrErrors',       c.errors_total != null ? c.errors_total : (c.errors_5xx_total != null ? c.errors_5xx_total : '—'));
  setVal('ctrForced',       c.forced_reprocess_total != null ? c.forced_reprocess_total : '—');
  setVal('ctrCkptCreated',  c.checkpoints_created_total != null ? c.checkpoints_created_total : '—');
  setVal('ctrCkptRestored', c.checkpoints_restored_total != null ? c.checkpoints_restored_total : '—');
  setVal('ctrPrefillStep',   c.last_prefill_step ? c.last_prefill_step.toLocaleString() : '—');
  setVal('ctrTargetDepth',   c.last_target_depth ? c.last_target_depth.toLocaleString() : '—');
  setVal('ctrDraftAcceptance', c.dflash_acceptance_ratio ? (c.dflash_acceptance_ratio * 100).toFixed(1) + '%' : '—');
  setVal('ctrDraftTokensCycle', c.dflash_tokens_per_cycle ? c.dflash_tokens_per_cycle.toFixed(2) : '—');
  setVal('ctrUptime',       _fmtUptime(Math.floor(Date.now() / 1000) - (c.session_start_ts || 0)));

  var qwen = (s.services && s.services['local-llm-qwen.service']) || {};
  setVal('ctrLlamaUptime',   qwen.active_since || '—');
  setVal('ctrLlamaRestarts', qwen.n_restarts != null ? qwen.n_restarts : '—');
  setVal('ctrPcie', (s.pcie_gen != null && s.pcie_width != null) ? ('Gen ' + s.pcie_gen + ' x' + s.pcie_width) : '—');
  setVal('ctrPstate', s.gpu_pstate || '—');
  setVal('ctrFan',  s.gpu_fan != null ? (s.gpu_fan + '%') : '—');
  setVal('ctrLoad', (s.load_1 != null) ? (s.load_1.toFixed(2) + ' / ' + s.load_5.toFixed(2) + ' / ' + s.load_15.toFixed(2)) : '—');

  var lu = document.getElementById('monitorLastUpdate');
  if (lu) lu.textContent = 'updated ' + new Date().toLocaleTimeString();
}

async function _monitorPollCharts() {
  var win = _monitor.window;
  var metrics = 'gpu_util,gpu_power_w,vram_used_mib,gpu_mem_util,cpu_pct,mem_used_mib,mem_free_mib,mem_available_mib,vram_free_mib,metal_headroom_mib,eval_tok_s,prompt_tok_s';
  var r = await api('/api/dashboard/timeseries?window=' + win + '&metrics=' + metrics);
  if (!r || r.error || !r.timestamps) return;
  var ts = r.timestamps;
  var s = r.series;

  drawMonitorChart('chartGpu', ts, s, [
    { key: 'gpu_util',    color: '#22d3ee', axis: 'left',  label: 'GPU Util' },
    { key: 'gpu_power_w', color: '#facc15', axis: 'right', label: 'Power W' },
  ], { leftMax: 100, leftFmt: function(v) { return Math.round(v) + '%'; }, rightFmt: function(v) { return Math.round(v) + 'W'; } });

  drawMonitorChart('chartVram', ts, s, [
    { key: 'vram_used_mib', color: '#a855f7', axis: 'left',  label: 'VRAM MiB' },
    { key: 'gpu_mem_util',  color: '#34d399', axis: 'right', label: 'Mem Util %' },
  ], { rightMax: 100, leftFmt: function(v) { return (v / 1024).toFixed(1) + 'G'; }, rightFmt: function(v) { return Math.round(v) + '%'; } });

  drawMonitorChart('chartCpu', ts, s, [
    { key: 'cpu_pct',      color: '#22d3ee', axis: 'left',  label: 'CPU' },
    { key: 'mem_used_mib', color: '#a855f7', axis: 'right', label: 'RAM' },
  ], { leftMax: 100, leftFmt: function(v) { return Math.round(v) + '%'; }, rightFmt: function(v) { return (v / 1024).toFixed(1) + 'G'; } });

  drawMonitorChart('chartTokS', ts, s, [
    { key: 'eval_tok_s',   color: '#34d399', axis: 'left', label: 'Eval' },
    { key: 'prompt_tok_s', color: '#22d3ee', axis: 'right', label: 'Prefill' },
  ], { leftFmt: function(v) { return v.toFixed(0); }, rightFmt: function(v) { return v.toFixed(0); } });

  drawMonitorChart('chartFreeMemory', ts, s, [
    { key: 'mem_free_mib',       color: '#22d3ee', axis: 'left', label: 'System free' },
    { key: 'mem_available_mib',  color: '#34d399', axis: 'left', label: 'Unified available' },
    { key: 'vram_free_mib',      color: '#a855f7', axis: 'left', label: 'GPU pool free' },
    { key: 'metal_headroom_mib', color: '#facc15', axis: 'left', label: 'Metal headroom' },
  ], { leftMax: 32768, leftFmt: function(v) { return (v / 1024).toFixed(0) + 'G'; } });
}

async function _monitorPollEvents() {
  var r = await api('/api/dashboard/events?window=' + _monitor.eventsWindow + '&limit=200');
  var el = document.getElementById('monitorEvents');
  if (!el) return;
  if (!r || r.error) { el.innerHTML = '<span style="color:var(--muted)">No events</span>'; return; }
  if (!r.events || !r.events.length) { el.innerHTML = '<span style="color:var(--muted)">No events in this window</span>'; return; }
  var KIND_COLORS = {
    error:           '#ef4444',
    forced_reprocess:'#facc15',
    service_state:   '#22d3ee',
    service_restart: '#a855f7',
  };
  el.innerHTML = r.events.map(function(ev) {
    var t = new Date(ev.ts * 1000).toLocaleString();
    var color = KIND_COLORS[ev.kind] || '#94a3b8';
    var safe = (ev.message || '').replace(/[<>&]/g, function(c) { return ({'<':'&lt;','>':'&gt;','&':'&amp;'})[c]; });
    return '<div style="margin:2px 0;padding:3px 6px;border-left:3px solid ' + color + ';background:rgba(255,255,255,0.02)">' +
           '<span style="color:#64748b">' + t + '</span> ' +
           '<span class="pill" style="background:' + color + '22;color:' + color + ';font-size:10px;padding:1px 6px;margin:0 4px">' + ev.kind + '</span>' +
           '<span>' + safe + '</span></div>';
  }).join('');
}

document.addEventListener('DOMContentLoaded', function() {
  if (typeof initWindowBtns === 'function') {
    initWindowBtns('monitorWindowBtns', function(w) { setMonitorWindow(w); });
    initWindowBtns('monitorEventsWindowBtns', function(w) { setMonitorEventsWindow(w); });
  }
  var sel = document.getElementById('monitorSampleRate');
  if (sel) {
    sel.addEventListener('change', function() { _monitorSetSampleRate(sel.value); });
  }
});

// ════════════════════════════════════════════════════════════════════════
// CHAT PLAYGROUND — streams straight through the manager's /api/chat proxy
// ════════════════════════════════════════════════════════════════════════

var _chat = { messages: [], controller: null, busy: false };

function chatFocus() {
  var input = document.getElementById('chatInput');
  if (input) input.focus();
}

function _chatRender() {
  var box = document.getElementById('chatMessages');
  if (!box) return;
  if (!_chat.messages.length) {
    box.innerHTML = '<div class="chat-empty muted">Send a prompt to the running model. Nothing here is saved.</div>';
    return;
  }
  box.innerHTML = _chat.messages.map(function (m) {
    var safe = (m.content || '').replace(/[<>&]/g, function (c) { return ({'<':'&lt;','>':'&gt;','&':'&amp;'})[c]; });
    return '<div class="chat-msg ' + (m.role === 'user' ? 'user' : 'assistant') + '">' +
      '<div class="chat-role">' + (m.role === 'user' ? 'You' : 'Model') + '</div>' +
      '<div class="chat-body">' + (safe || '<span class="muted">…</span>') + '</div></div>';
  }).join('');
  box.scrollTop = box.scrollHeight;
}

function chatClear() {
  chatStop();
  _chat.messages = [];
  _chatRender();
}

function chatStop() {
  if (_chat.controller) { _chat.controller.abort(); _chat.controller = null; }
  _chat.busy = false;
  var send = document.getElementById('chatSend');
  var stop = document.getElementById('chatStop');
  if (send) send.disabled = false;
  if (stop) stop.style.display = 'none';
}

async function chatSend() {
  if (_chat.busy) return;
  var input = document.getElementById('chatInput');
  var text = input ? input.value.trim() : '';
  if (!text) return;
  var sys = (document.getElementById('chatSystem') || {}).value || '';
  var temp = parseFloat((document.getElementById('chatTemp') || {}).value) || 0.7;
  var maxTok = parseInt((document.getElementById('chatMaxTok') || {}).value) || 1024;

  input.value = '';
  _chat.messages.push({ role: 'user', content: text });
  var reply = { role: 'assistant', content: '' };
  _chat.messages.push(reply);
  _chatRender();

  var payload = {
    messages: (sys.trim() ? [{ role: 'system', content: sys.trim() }] : []).concat(
      _chat.messages.slice(0, -1).map(function (m) { return { role: m.role, content: m.content }; })),
    temperature: temp, max_tokens: maxTok, stream: true,
  };

  _chat.busy = true;
  _chat.controller = new AbortController();
  var send = document.getElementById('chatSend');
  var stop = document.getElementById('chatStop');
  if (send) send.disabled = true;
  if (stop) stop.style.display = '';
  var t0 = Date.now(), tokens = 0;

  try {
    var resp = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload), signal: _chat.controller.signal,
    });
    if (!resp.ok) {
      var errText = await resp.text();
      var detail; try { detail = JSON.parse(errText).detail; } catch (_) { detail = errText; }
      throw new Error(detail || ('HTTP ' + resp.status));
    }
    var reader = resp.body.getReader();
    var decoder = new TextDecoder();
    var pending = '';
    for (;;) {
      var chunk = await reader.read();
      if (chunk.done) break;
      pending += decoder.decode(chunk.value, { stream: true });
      var lines = pending.split('\n');
      pending = lines.pop();
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim();
        if (!line.startsWith('data:')) continue;
        var data = line.slice(5).trim();
        if (data === '[DONE]') continue;
        var obj; try { obj = JSON.parse(data); } catch (_) { continue; }
        if (obj.error) throw new Error(typeof obj.error === 'string' ? obj.error : JSON.stringify(obj.error));
        var delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
        if (delta && delta.content) {
          reply.content += delta.content;
          tokens++;
          _chatRender();
        }
      }
    }
    var secs = (Date.now() - t0) / 1000;
    var statsEl = document.getElementById('chatStats');
    if (statsEl) statsEl.textContent = '~' + tokens + ' chunks in ' + secs.toFixed(1) + 's (' + (tokens / Math.max(secs, 0.1)).toFixed(1) + ' tok/s)';
  } catch (e) {
    if (e.name !== 'AbortError') {
      reply.content += (reply.content ? '\n\n' : '') + '⚠ ' + (e.message || 'request failed');
      toast(e.message || 'Chat request failed', 'error');
    }
    _chatRender();
  } finally {
    chatStop();
  }
}

document.addEventListener('DOMContentLoaded', function () {
  var input = document.getElementById('chatInput');
  if (input) {
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); chatSend(); }
    });
  }
  _chatRender();
});
