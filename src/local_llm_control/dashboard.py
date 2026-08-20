"""Dependency-free dashboard served by the control API."""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Local LLM Control</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { max-width: 980px; margin: 32px auto; padding: 0 18px; background:#0b0d10; color:#e8eaed; }
    header,.card { background:#15191f; border:1px solid #2b313a; border-radius:12px; padding:18px; margin:14px 0; }
    h1,h2 { margin:0 0 12px; } h1 { font-size:1.45rem; } h2 { font-size:1.05rem; }
    .status { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
    .metric { background:#0f1216; border-radius:8px; padding:11px; }
    .metric small { display:block; color:#99a1ab; margin-bottom:4px; }
    .profiles { display:grid; gap:10px; }
    .profile { display:grid; grid-template-columns:1fr auto; gap:12px; align-items:center; background:#0f1216; padding:12px; border-radius:8px; }
    .profile p { margin:4px 0 0; color:#aeb5bf; font-size:.9rem; }
    button { border:0; border-radius:7px; padding:9px 14px; font-weight:650; cursor:pointer; background:#3b82f6; color:white; }
    button.stop { background:#dc3545; } button:disabled { opacity:.45; cursor:wait; }
    pre { overflow:auto; max-height:330px; white-space:pre-wrap; background:#090b0e; padding:12px; border-radius:8px; font-size:.78rem; }
    .ok { color:#55d187; } .off { color:#aeb5bf; } .error { color:#ff7070; }
  </style>
</head>
<body>
  <header><h1>Local LLM Control</h1><div id="message" class="off">Connecting…</div></header>
  <section class="card"><h2>Runtime</h2><div id="status" class="status"></div></section>
  <section class="card"><h2>Profiles</h2><div id="profiles" class="profiles"></div></section>
  <section class="card"><h2>Logs</h2><pre id="logs">No runtime selected.</pre></section>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let busy = false;
async function json(url, options) {
  const r = await fetch(url, options); const body = await r.json();
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`); return body;
}
async function act(url) {
  if (busy) return; busy = true; document.querySelectorAll('button').forEach(b => b.disabled=true);
  $('message').textContent = 'Working…';
  try { await json(url, {method:'POST'}); await refresh(); }
  catch (e) { $('message').textContent=e.message; $('message').className='error'; }
  finally { busy=false; document.querySelectorAll('button').forEach(b => b.disabled=false); }
}
async function refresh() {
  try {
    const [profiles, state] = await Promise.all([json('/v1/profiles'), json('/v1/runtime')]);
    $('message').textContent = state.running ? `${state.profile} is ready` : 'Controller ready — no model loaded';
    $('message').className = state.running ? 'ok' : 'off';
    $('status').innerHTML = [
      ['State',state.running?'Running':'Stopped'], ['Profile',state.profile||'—'],
      ['PID',state.pid||'—'], ['Runtime RSS',state.process_rss_gib == null?'—':`${state.process_rss_gib} GiB`],
      ['Available RAM',`${state.memory.available_gib} GiB`], ['System memory',`${state.memory.used_percent}% used`]
    ].map(([a,b])=>`<div class="metric"><small>${esc(a)}</small>${esc(b)}</div>`).join('');
    $('profiles').innerHTML = profiles.map(p => `<div class="profile"><div><strong>${esc(p.name)}</strong><p>${esc(p.description)} · ${esc(p.engine)} · port ${esc(p.port)}</p></div><button onclick="act('/v1/runtime/${encodeURIComponent(p.name)}/start')" ${state.running?'disabled':''}>Start</button></div>`).join('') +
      `<button class="stop" onclick="act('/v1/runtime/stop')" ${state.running?'':'disabled'}>Stop current model</button>`;
    const logs = await json('/v1/runtime/logs?lines=120'); $('logs').textContent = logs.lines.join('\n') || 'No runtime log available.';
  } catch(e) { $('message').textContent=e.message; $('message').className='error'; }
}
refresh(); setInterval(()=>{ if(!busy) refresh(); }, 3000);
</script>
</body></html>"""
