// charts.js — Canvas chart rendering for VRAM and GPU power history
// Loaded after components.js, before app.js

var _vramChartWindow = '15m';
var _powerChartWindow = '15m';
var _vramChartTimer = null;
var _powerChartTimer = null;
var _lastVramData = null;
var _lastPowerData = null;

// ── Palette from CSS variables (single source of truth in styles.css) ───────
function _cssVar(name, fallback) {
  var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
var CHART_COLORS = {
  cyan:   function() { return _cssVar('--cyan',   '#22d3ee'); },
  purple: function() { return _cssVar('--purple', '#a78bfa'); },
  yellow: function() { return _cssVar('--yellow', '#fbbf24'); },
  green:  function() { return _cssVar('--green',  '#34d399'); },
  orange: function() { return _cssVar('--orange', '#f97316'); },
};

// ── Shared hover tooltip + crosshair ────────────────────────────────────────
function _chartTooltipEl() {
  var el = document.getElementById('chartTooltip');
  if (!el) {
    el = document.createElement('div');
    el.id = 'chartTooltip';
    el.className = 'chart-tooltip';
    el.style.opacity = '0';
    document.body.appendChild(el);
  }
  return el;
}

// Charts register {points:[{x, html}], redraw(crosshairX)} on the canvas; this
// wires one mousemove handler per canvas that finds the nearest point.
function _wireChartHover(canvas) {
  if (canvas._hoverWired) return;
  canvas._hoverWired = true;
  canvas.addEventListener('mousemove', function (e) {
    var st = canvas._chart;
    if (!st || !st.points || !st.points.length) return;
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var best = 0, bestD = Infinity;
    for (var i = 0; i < st.points.length; i++) {
      var d = Math.abs(st.points[i].x - x);
      if (d < bestD) { bestD = d; best = i; }
    }
    var p = st.points[best];
    var tip = _chartTooltipEl();
    tip.innerHTML = p.html;
    tip.style.opacity = '1';
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 10) + 'px';
    tip.style.top = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 10) + 'px';
    if (st.redraw && st.crosshairX !== p.x) { st.crosshairX = p.x; st.redraw(p.x); }
  });
  canvas.addEventListener('mouseleave', function () {
    _chartTooltipEl().style.opacity = '0';
    var st = canvas._chart;
    if (st && st.redraw) { st.crosshairX = null; st.redraw(null); }
  });
}

function _drawCrosshair(ctx, x, padTop, innerH) {
  if (x == null) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x, padTop);
  ctx.lineTo(x, padTop + innerH);
  ctx.stroke();
  ctx.restore();
}

// Only poll when the page is visible and the canvas is actually on screen.
function _chartVisible(canvas) {
  return !document.hidden && canvas && canvas.offsetParent !== null;
}

// ── Generic chart renderer ──────────────────────────────────────────────────
function _drawChart(canvas, data, opts, crosshairX) {
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var dpr = window.devicePixelRatio || 1;
  var cssW = canvas.offsetWidth || 600;
  var cssH = canvas.offsetHeight || 120;
  canvas.width  = cssW * dpr;
  canvas.height = cssH * dpr;
  canvas.style.height = cssH + 'px';
  ctx.scale(dpr, dpr);
  var W = cssW, H = cssH;

  var pad = opts.pad || { top: 8, bot: 22, left: 48, right: 8 };
  var iw = W - pad.left - pad.right;
  var ih = H - pad.top - pad.bot;

  ctx.clearRect(0, 0, W, H);

  if (!data || data.length === 0) {
    ctx.fillStyle = '#56697a';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No data', W / 2, H / 2);
    return;
  }

  var yMax = opts.yMax || Math.max.apply(null, data.map(opts.primaryFn)) || 1;
  var tMin = data[0].t, tMax = data[data.length - 1].t;
  var tRange = Math.max(tMax - tMin, 1);
  var toX = function(t) { return pad.left + ((t - tMin) / tRange) * iw; };
  var toY = function(v) { return pad.top + ih - (v / yMax) * ih; };

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 0.5;
  [0.25, 0.5, 0.75, 1.0].forEach(function(pct) {
    var y = pad.top + ih - pct * ih;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    var labelVal = opts.yLabelFn ? opts.yLabelFn(yMax * pct) : Math.round(yMax * pct).toString();
    ctx.fillStyle = 'rgba(255,255,255,0.28)';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(labelVal, pad.left - 4, y + 3);
  });

  // Secondary line (optional)
  if (opts.secondaryFn && data.some(function(d) { return opts.secondaryFn(d) != null; })) {
    ctx.strokeStyle = opts.secondaryColor || '#a855f7';
    ctx.lineWidth = 1.2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    data.forEach(function(d, i) {
      var v = opts.secondaryFn(d);
      if (v == null) return;
      var x = toX(d.t), y = toY(v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Primary filled line
  ctx.strokeStyle = opts.primaryColor || '#22d3ee';
  ctx.lineWidth = 1.8;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  data.forEach(function(d, i) {
    var x = toX(d.t), y = toY(opts.primaryFn(d) || 0);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under primary line
  if (opts.fill !== false) {
    var fillColor = opts.primaryColor || '#22d3ee';
    ctx.lineTo(toX(data[data.length - 1].t), pad.top + ih);
    ctx.lineTo(toX(data[0].t), pad.top + ih);
    ctx.closePath();
    // Convert hex to rgba with alpha
    if (fillColor.startsWith('#')) {
      var r = parseInt(fillColor.slice(1,3),16), g = parseInt(fillColor.slice(3,5),16), b = parseInt(fillColor.slice(5,7),16);
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.12)';
    } else {
      ctx.fillStyle = fillColor;
    }
    ctx.fill();
  }

  // Stats element
  if (opts.statsEl) {
    var vals = data.map(opts.primaryFn);
    var minV = Math.min.apply(null, vals), maxV = Math.max.apply(null, vals);
    var avgV = vals.reduce(function(a, b) { return a + b; }, 0) / vals.length;
    var curV = vals[vals.length - 1];
    var f = opts.statsFormat || function(v) { return Math.round(v); };
    opts.statsEl.innerHTML =
      '<span class="metric-label">Now</span> <span class="metric">' + f(curV) + '</span> &nbsp; ' +
      '<span class="metric-label">Avg</span> <span class="metric">' + f(avgV) + '</span> &nbsp; ' +
      '<span class="metric-label">Min</span> <span class="metric">' + f(minV) + '</span> &nbsp; ' +
      '<span class="metric-label">Max</span> <span class="metric">' + f(maxV) + '</span> &nbsp; ' +
      '<span class="muted" style="font-size:10px">' + data.length + ' pts</span>';
  }

  // Crosshair overlay + hover registration
  _drawCrosshair(ctx, crosshairX, pad.top, ih);
  var fmtVal = opts.statsFormat || function(v) { return Math.round(v); };
  canvas._chart = {
    points: data.map(function(d) {
      var lines = ['<b>' + _fmtTime(d.t) + '</b>',
                   (opts.primaryLabel || 'value') + ': ' + fmtVal(opts.primaryFn(d) || 0)];
      if (opts.secondaryFn && opts.secondaryFn(d) != null) {
        lines.push((opts.secondaryLabel || 'secondary') + ': ' + fmtVal(opts.secondaryFn(d)));
      }
      return { x: toX(d.t), html: lines.join('<br>') };
    }),
    redraw: function(cx) { _drawChart(canvas, data, opts, cx); },
    crosshairX: crosshairX,
  };
  _wireChartHover(canvas);
}

// ── VRAM chart ──────────────────────────────────────────────────────────────
function loadVramChart(force) {
  var canvas = document.getElementById('vramChart');
  if (!force && !_chartVisible(canvas)) return;
  api('/api/vram-history?window=' + _vramChartWindow).then(function(r) {
    if (!r || r.error || !r.data) return;
    _lastVramData = r.data;
    if (!canvas) return;
    var totalMiB = (r.data[0] && r.data[0].total) || 32768;
    _drawChart(canvas, r.data, {
      primaryFn:    function(d) { return d.used || 0; },
      secondaryFn:  function(d) { return d.app || 0; },
      primaryColor: CHART_COLORS.cyan(),
      secondaryColor: CHART_COLORS.purple(),
      primaryLabel: 'Used',
      secondaryLabel: 'App VRAM',
      yMax: totalMiB,
      yLabelFn: function(v) { return Math.round(v / 1024) + 'G'; },
      statsEl: document.getElementById('vramChartStats'),
      statsFormat: function(v) { return Math.round(v) + ' MiB'; },
    });
  });
}

function startVramChartTimer() {
  if (_vramChartTimer) clearInterval(_vramChartTimer);
  _vramChartTimer = setInterval(loadVramChart, 5000);
  loadVramChart();
}

function setVramWindow(w) {
  _vramChartWindow = w;
  _updateWindowBtns('vramWindowBtns', w);
  loadVramChart();
}

// ── Power chart ─────────────────────────────────────────────────────────────
function loadPowerChart(force) {
  var canvas = document.getElementById('powerChart');
  if (!force && !_chartVisible(canvas)) return;
  api('/api/power/history?window=' + _powerChartWindow).then(function(r) {
    if (!r || r.error || !r.data) return;
    _lastPowerData = r.data;
    if (!canvas) return;
    var statsEl = document.getElementById('powerChartStats');
    _drawChart(canvas, r.data, {
      primaryFn:    function(d) { return d.power_w || 0; },
      primaryColor: CHART_COLORS.orange(),
      primaryLabel: 'GPU power',
      fill: true,
      yMax: null,
      statsEl: statsEl,
      statsFormat: function(v) { return v.toFixed(1) + ' W'; },
    });
  });
}

function startPowerChartTimer() {
  if (_powerChartTimer) clearInterval(_powerChartTimer);
  _powerChartTimer = setInterval(loadPowerChart, 5000);
  loadPowerChart();
}

function setPowerWindow(w) {
  _powerChartWindow = w;
  _updateWindowBtns('powerWindowBtns', w);
  loadPowerChart();
}

// ── Window button wiring ────────────────────────────────────────────────────
function _updateWindowBtns(containerId, activeWindow) {
  var container = document.getElementById(containerId);
  if (!container) return;
  container.querySelectorAll('.window-btn').forEach(function(btn) {
    var isActive = btn.dataset.window === activeWindow;
    btn.className = 'window-btn pill ' + (isActive ? 'cyan' : 'muted');
  });
}

function initWindowBtns(containerId, setterFn) {
  var container = document.getElementById(containerId);
  if (!container) return;
  container.addEventListener('click', function(e) {
    var btn = e.target.closest('.window-btn');
    if (!btn) return;
    setterFn(btn.dataset.window);
  });
}

// ── Resize handler ──────────────────────────────────────────────────────────
window.addEventListener('resize', function() {
  if (_lastVramData) loadVramChart();
  if (_lastPowerData) loadPowerChart();
});

// ── Init window buttons after DOM ready ────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  initWindowBtns('vramWindowBtns',  function(w) { setVramWindow(w); });
  initWindowBtns('powerWindowBtns', function(w) { setPowerWindow(w); });
});

// ════════════════════════════════════════════════════════════════════════
// MONITOR (Grafana-style) chart helpers
// ════════════════════════════════════════════════════════════════════════

// Generic monitor chart: dual-axis line, accepts {timestamps:[], series:{name:[v,...]}}.
// `lines` is an array of {key, color, axis:'left'|'right', label}.
function drawMonitorChart(canvasId, ts, series, lines, opts, crosshairX) {
  opts = opts || {};
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;
  var dpr = window.devicePixelRatio || 1;
  var cssW = canvas.offsetWidth || 600;
  var cssH = canvas.offsetHeight || 160;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  canvas.style.height = cssH + 'px';
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  var W = cssW, H = cssH;
  var pad = { top: 8, bot: 22, left: 50, right: 50 };
  var iw = W - pad.left - pad.right;
  var ih = H - pad.top - pad.bot;
  ctx.clearRect(0, 0, W, H);

  if (!ts || ts.length === 0) {
    ctx.fillStyle = '#56697a';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No data', W / 2, H / 2);
    return;
  }
  var tMin = ts[0], tMax = ts[ts.length - 1];
  var tRange = Math.max(tMax - tMin, 1);
  var toX = function(t) { return pad.left + ((t - tMin) / tRange) * iw; };

  // Compute axis ranges per side
  function axisMax(keys) {
    var max = 0;
    keys.forEach(function(k) {
      var arr = series[k] || [];
      arr.forEach(function(v) { if (v != null && v > max) max = v; });
    });
    return max || 1;
  }
  var leftKeys = lines.filter(function(l) { return (l.axis || 'left') === 'left'; }).map(function(l) { return l.key; });
  var rightKeys = lines.filter(function(l) { return l.axis === 'right'; }).map(function(l) { return l.key; });
  var leftMax = opts.leftMax || axisMax(leftKeys);
  var rightMax = opts.rightMax || (rightKeys.length ? axisMax(rightKeys) : 1);

  // Grid + left/right axis labels
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 0.5;
  [0, 0.25, 0.5, 0.75, 1.0].forEach(function(pct) {
    var y = pad.top + ih - pct * ih;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.32)';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    var lFmt = opts.leftFmt || function(v) { return Math.round(v); };
    ctx.fillText(lFmt(leftMax * pct), pad.left - 4, y + 3);
    if (rightKeys.length) {
      ctx.textAlign = 'left';
      var rFmt = opts.rightFmt || function(v) { return Math.round(v); };
      ctx.fillText(rFmt(rightMax * pct), W - pad.right + 4, y + 3);
    }
  });

  // Time labels (first/last)
  ctx.fillStyle = 'rgba(255,255,255,0.32)';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(_fmtTime(tMin), pad.left, H - 6);
  ctx.textAlign = 'right';
  ctx.fillText(_fmtTime(tMax), W - pad.right, H - 6);

  // Draw lines
  lines.forEach(function(l) {
    var arr = series[l.key] || [];
    var max = (l.axis === 'right') ? rightMax : leftMax;
    var toY = function(v) { return pad.top + ih - (v / max) * ih; };
    ctx.strokeStyle = l.color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    var started = false;
    for (var i = 0; i < ts.length; i++) {
      var v = arr[i];
      if (v == null) continue;
      var x = toX(ts[i]), y = toY(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else { ctx.lineTo(x, y); }
    }
    ctx.stroke();
  });

  // Crosshair overlay + hover registration
  _drawCrosshair(ctx, crosshairX, pad.top, ih);
  canvas._chart = {
    points: ts.map(function(t, i) {
      var rows = ['<b>' + _fmtTime(t) + '</b>'];
      lines.forEach(function(l) {
        var v = (series[l.key] || [])[i];
        if (v == null) return;
        var f = (l.axis === 'right') ? (opts.rightFmt || Math.round) : (opts.leftFmt || Math.round);
        rows.push('<span style="color:' + l.color + '">■</span> ' + (l.label || l.key) + ': ' + f(v));
      });
      return { x: toX(t), html: rows.join('<br>') };
    }),
    redraw: function(cx) { drawMonitorChart(canvasId, ts, series, lines, opts, cx); },
    crosshairX: crosshairX,
  };
  _wireChartHover(canvas);
}

function _fmtTime(unix) {
  var d = new Date(unix * 1000);
  var hh = String(d.getHours()).padStart(2, '0');
  var mm = String(d.getMinutes()).padStart(2, '0');
  return hh + ':' + mm;
}

// ════════════════════════════════════════════════════════════════════════
// SWEEP chart: throughput (gen/prompt tok/s) vs context size.
// Purpose-built sibling of drawMonitorChart — same canvas/DPR/grid style,
// but the x-axis is context size (K tokens), not wall-clock time.
// `series` is an array of {results:[{ctx,gen_tps,prompt_tps,status}], label, color}
// so two saved runs can be overlaid for comparison.
// ════════════════════════════════════════════════════════════════════════
function drawSweepChart(canvasId, series) {
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;
  var dpr = window.devicePixelRatio || 1;
  var cssW = canvas.offsetWidth || 600;
  var cssH = canvas.offsetHeight || 200;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  canvas.style.height = cssH + 'px';
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  var W = cssW, H = cssH;
  var pad = { top: 10, bot: 26, left: 50, right: 12 };
  var iw = W - pad.left - pad.right;
  var ih = H - pad.top - pad.bot;
  ctx.clearRect(0, 0, W, H);

  // Flatten all ok points across all series to compute axis ranges.
  var allPts = [];
  series.forEach(function(s) {
    (s.results || []).forEach(function(r) {
      if (r.status === 'ok' && r.ctx != null) {
        allPts.push({ ctx: r.ctx, gen: r.gen_tps || 0, prompt: r.prompt_tps || 0 });
      }
    });
  });
  if (!allPts.length) {
    ctx.fillStyle = '#56697a';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No sweep data', W / 2, H / 2);
    return;
  }
  var ctxVals = allPts.map(function(p) { return p.ctx; });
  var cMin = Math.min.apply(null, ctxVals), cMax = Math.max.apply(null, ctxVals);
  var cRange = Math.max(cMax - cMin, 1);
  var yMax = Math.max.apply(null, allPts.map(function(p) { return Math.max(p.gen, p.prompt); })) || 1;
  yMax *= 1.1;
  var toX = function(c) { return pad.left + ((c - cMin) / cRange) * iw; };
  var toY = function(v) { return pad.top + ih - (v / yMax) * ih; };

  // Grid + y labels (tok/s)
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 0.5;
  [0, 0.25, 0.5, 0.75, 1.0].forEach(function(pct) {
    var y = pad.top + ih - pct * ih;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.32)';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText((yMax * pct).toFixed(0), pad.left - 4, y + 3);
  });

  // X labels (context in K) — first / mid / last
  ctx.fillStyle = 'rgba(255,255,255,0.32)';
  ctx.font = '9px JetBrains Mono, monospace';
  [cMin, (cMin + cMax) / 2, cMax].forEach(function(c, i) {
    ctx.textAlign = i === 0 ? 'left' : (i === 2 ? 'right' : 'center');
    ctx.fillText(Math.round(c / 1024) + 'K', toX(c), H - 8);
  });

  // Draw each series: gen (solid) + prompt (dashed), in the series color.
  series.forEach(function(s) {
    var pts = (s.results || []).filter(function(r) { return r.status === 'ok' && r.ctx != null; })
      .sort(function(a, b) { return a.ctx - b.ctx; });
    if (!pts.length) return;
    var color = s.color || '#22d3ee';
    // gen tok/s — solid line + dots
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
    ctx.setLineDash([]);
    ctx.beginPath();
    pts.forEach(function(r, i) {
      var x = toX(r.ctx), y = toY(r.gen_tps || 0);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    pts.forEach(function(r) {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(toX(r.ctx), toY(r.gen_tps || 0), 2.5, 0, 6.2832); ctx.fill();
    });
    // prompt tok/s — dashed, dimmer
    ctx.strokeStyle = color; ctx.globalAlpha = 0.45; ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    pts.forEach(function(r, i) {
      var x = toX(r.ctx), y = toY(r.prompt_tps || 0);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  });

  // Legend
  var lx = pad.left + 6, ly = pad.top + 6;
  ctx.textAlign = 'left';
  ctx.font = '9px JetBrains Mono, monospace';
  series.forEach(function(s, i) {
    var color = s.color || '#22d3ee';
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly + i * 13, 10, 3);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText((s.label || 'run') + '  (solid=gen, dash=prompt t/s)', lx + 16, ly + i * 13 + 4);
  });
}
