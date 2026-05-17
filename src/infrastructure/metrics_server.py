"""Lightweight monitoring dashboard server.

Serves a dark-theme HTML dashboard at http://<host>:8765/
with live metrics (auto-refresh every 2 s) and a History tab to
review past-day resource usage stored in SQLite.

Endpoints:
    GET /                   — HTML dashboard
    GET /api/metrics        — live JSON snapshot
    GET /api/history?date=  — historical rows for YYYY-MM-DD
    GET /api/dates          — list of dates that have data
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

from loguru import logger

# ── Embedded HTML dashboard ───────────────────────────────────────────────────
_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Smart Office — Monitoring</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg:#0d1117;--surface:#161b22;--border:#30363d;
    --text:#e6edf3;--muted:#8b949e;
    --green:#3fb950;--yellow:#d29922;--red:#f85149;
    --blue:#58a6ff;--purple:#bc8cff;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}
  header{background:var(--surface);border-bottom:1px solid var(--border);padding:14px 24px;display:flex;align-items:center;justify-content:space-between}
  header h1{font-size:16px;font-weight:600;letter-spacing:.3px}
  header h1 span{color:var(--blue)}
  #status-bar{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted)}
  #dot{width:8px;height:8px;border-radius:50%;background:var(--green)}
  #dot.stale{background:var(--yellow)}
  nav{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;display:flex;gap:0}
  .tab-btn{background:none;border:none;color:var(--muted);font-size:13px;padding:10px 16px;cursor:pointer;border-bottom:2px solid transparent;transition:color .15s}
  .tab-btn:hover{color:var(--text)}
  .tab-btn.active{color:var(--blue);border-bottom-color:var(--blue)}
  .tab{display:none;padding:20px 24px;max-width:1400px;margin:0 auto}
  .tab.active{display:block}
  .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}
  .kpi{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px 18px}
  .kpi .label{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin-bottom:6px}
  .kpi .value{font-size:28px;font-weight:700;line-height:1}
  .kpi .sub{font-size:12px;color:var(--muted);margin-top:4px}
  .kpi.alert .value{color:var(--red)}.kpi.warn .value{color:var(--yellow)}.kpi.ok .value{color:var(--green)}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px 18px}
  .card h2{font-size:13px;font-weight:600;color:var(--muted);margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px}
  .cam-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
  .cam-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
  .cam-card .cam-title{font-size:12px;font-weight:600;color:var(--muted);margin-bottom:8px}
  .cam-fps{font-size:32px;font-weight:700;color:var(--blue);line-height:1}
  .cam-fps.low{color:var(--red)}.cam-fps.mid{color:var(--yellow)}
  .cam-drops{font-size:12px;color:var(--muted);margin-top:4px}
  .alert-box{background:var(--surface);border:1px solid #f8514944;border-radius:8px;padding:14px 18px;margin-bottom:20px;display:none}
  .alert-box h2{font-size:13px;font-weight:600;color:var(--red);margin-bottom:10px;text-transform:uppercase}
  .alert-item{font-size:13px;color:var(--text);padding:6px 0;border-bottom:1px solid var(--border)}
  .alert-item:last-child{border-bottom:none}
  /* History tab */
  .hist-controls{display:flex;align-items:center;gap:12px;margin-bottom:20px;flex-wrap:wrap}
  .hist-controls label{font-size:13px;color:var(--muted)}
  select,input[type=date]{background:var(--surface);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px;font-size:13px;outline:none}
  select:focus,input[type=date]:focus{border-color:var(--blue)}
  .btn{background:var(--blue);color:#000;border:none;padding:7px 16px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer}
  .btn:hover{opacity:.85}
  #hist-status{font-size:13px;color:var(--muted)}
  footer{text-align:center;color:var(--muted);font-size:11px;padding:24px;border-top:1px solid var(--border)}
</style>
</head>
<body>
<header>
  <h1>Smart Office <span>/ Monitoring</span></h1>
  <div id="status-bar"><div id="dot"></div><span id="status-text">Connecting...</span></div>
</header>
<nav>
  <button class="tab-btn active" onclick="showTab('live',this)">Live</button>
  <button class="tab-btn" onclick="showTab('history',this)">History</button>
</nav>

<!-- ══ LIVE TAB ══════════════════════════════════════════════════════════ -->
<div id="tab-live" class="tab active">
  <div class="grid">
    <div class="kpi" id="kpi-cpu"><div class="label">CPU</div><div class="value" id="v-cpu">—</div><div class="sub">utilization</div></div>
    <div class="kpi" id="kpi-ram"><div class="label">RAM</div><div class="value" id="v-ram">—</div><div class="sub" id="v-ram-sub">—</div></div>
    <div class="kpi" id="kpi-gpu"><div class="label">GPU</div><div class="value" id="v-gpu">—</div><div class="sub" id="v-gpu-sub">—</div></div>
    <div class="kpi" id="kpi-vram"><div class="label">VRAM</div><div class="value" id="v-vram">—</div><div class="sub" id="v-vram-sub">—</div></div>
  </div>

  <div id="alert-box" class="alert-box">
    <h2>Active Alerts</h2>
    <div id="alert-list"></div>
  </div>

  <div class="charts">
    <div class="card"><h2>System Resources (%)</h2><canvas id="chart-sys" height="140"></canvas></div>
    <div class="card"><h2>Inference Latency (ms)</h2><canvas id="chart-lat" height="140"></canvas></div>
  </div>
  <div class="card" style="margin-bottom:20px"><h2>Per-Camera FPS</h2><canvas id="chart-fps" height="90"></canvas></div>
  <div class="cam-grid" id="cam-grid"></div>
</div>

<!-- ══ HISTORY TAB ════════════════════════════════════════════════════════ -->
<div id="tab-history" class="tab">
  <div class="hist-controls">
    <label>Date</label>
    <input type="date" id="hist-date"/>
    <select id="hist-dates-sel"><option value="">— available dates —</option></select>
    <button class="btn" onclick="loadHistory()">Load</button>
    <span id="hist-status"></span>
  </div>
  <div class="charts">
    <div class="card"><h2>CPU &amp; RAM (%)</h2><canvas id="h-sys" height="160"></canvas></div>
    <div class="card"><h2>GPU Utilization &amp; VRAM (%)</h2><canvas id="h-gpu" height="160"></canvas></div>
  </div>
  <div class="card" style="margin-bottom:20px"><h2>Inference Latency (ms)</h2><canvas id="h-lat" height="110"></canvas></div>
  <div class="card" style="margin-bottom:20px"><h2>Per-Camera FPS</h2><canvas id="h-fps" height="110"></canvas></div>
</div>

<footer>Smart Office Monitoring Dashboard &nbsp;·&nbsp; Live refreshes every 2 s &nbsp;·&nbsp; History retained 30 days</footer>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'history') loadAvailableDates();
}

// ── Chart helpers ─────────────────────────────────────────────────────────────
const WINDOW = 90;
function mkDataset(label, color, data) {
  return {label, data: data||Array(WINDOW).fill(null),
    borderColor:color, backgroundColor:color+'18',
    tension:.3, fill:true, pointRadius:0, borderWidth:1.5};
}
function baseOpts(yMax, unit) {
  return {responsive:true, maintainAspectRatio:true, animation:false,
    plugins:{legend:{labels:{color:'#8b949e',boxWidth:12,font:{size:11}}}},
    scales:{
      x:{ticks:{display:false},grid:{color:'#30363d'}},
      y:{min:0,max:yMax||undefined,
         ticks:{color:'#8b949e',font:{size:11},callback:v=>v+(unit||'')},
         grid:{color:'#30363d'}}}};
}
function push(chart, dsIdx, val) {
  chart.data.datasets[dsIdx].data.push(val);
  if (chart.data.datasets[dsIdx].data.length > WINDOW)
    chart.data.datasets[dsIdx].data.shift();
  if (dsIdx === 0) {
    chart.data.labels.push('');
    if (chart.data.labels.length > WINDOW) chart.data.labels.shift();
  }
}

// Live charts
const sysChart = new Chart(document.getElementById('chart-sys'), {
  type:'line', data:{labels:Array(WINDOW).fill(''),
    datasets:[mkDataset('CPU %','#58a6ff'),mkDataset('RAM %','#bc8cff'),mkDataset('GPU %','#3fb950')]},
  options:baseOpts(100,'%')});

const latChart = new Chart(document.getElementById('chart-lat'), {
  type:'line', data:{labels:Array(WINDOW).fill(''),
    datasets:[mkDataset('YOLO ms','#f78166'),mkDataset('ArcFace ms','#ffa657')]},
  options:baseOpts(null,'ms')});

const fpsChart = new Chart(document.getElementById('chart-fps'), {
  type:'bar', data:{labels:[],datasets:[]},
  options:{responsive:true,maintainAspectRatio:true,animation:false,
    plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#8b949e',font:{size:11}},grid:{color:'#30363d'}},
            y:{min:0,ticks:{color:'#8b949e',font:{size:11},callback:v=>v+' fps'},grid:{color:'#30363d'}}}}});

// ── KPI helpers ───────────────────────────────────────────────────────────────
function kpiState(id, pct, warnAt, alertAt) {
  const el = document.getElementById(id);
  el.classList.remove('ok','warn','alert');
  el.classList.add(pct >= alertAt ? 'alert' : pct >= warnAt ? 'warn' : 'ok');
}
function fpsColor(fps) {
  return fps < 5 ? '#f85149' : fps < 15 ? '#d29922' : '#58a6ff';
}

// ── Live polling ──────────────────────────────────────────────────────────────
let lastOk = 0;

async function fetchLive() {
  try {
    const d = await fetch('/api/metrics').then(r => r.json());
    lastOk = Date.now();
    document.getElementById('dot').className = '';
    document.getElementById('status-text').textContent =
      'Live · ' + new Date(d.timestamp*1000).toLocaleTimeString();

    const cpu = d.cpu_percent||0;
    document.getElementById('v-cpu').textContent = cpu.toFixed(0)+'%';
    kpiState('kpi-cpu', cpu, 70, 90);

    const mem = d.memory||{};
    document.getElementById('v-ram').textContent = (mem.percent||0).toFixed(0)+'%';
    document.getElementById('v-ram-sub').textContent =
      (mem.used_gb||0).toFixed(1)+' / '+(mem.total_gb||0).toFixed(1)+' GB';
    kpiState('kpi-ram', mem.percent||0, 75, 90);

    const gpu = d.gpu;
    if (gpu) {
      document.getElementById('v-gpu').textContent = gpu.util_percent+'%';
      document.getElementById('v-gpu-sub').textContent = gpu.name||'GPU';
      kpiState('kpi-gpu', gpu.util_percent, 80, 95);
      document.getElementById('v-vram').textContent = gpu.mem_percent.toFixed(0)+'%';
      document.getElementById('v-vram-sub').textContent =
        (gpu.mem_used_mb/1024).toFixed(1)+' / '+(gpu.mem_total_mb/1024).toFixed(1)+' GB';
      kpiState('kpi-vram', gpu.mem_percent, 80, 90);
    } else {
      ['v-gpu','v-vram'].forEach(id => document.getElementById(id).textContent='N/A');
    }

    push(sysChart,0,cpu); push(sysChart,1,mem.percent||0);
    push(sysChart,2,gpu?gpu.util_percent:null);
    sysChart.update();

    const inf = d.inference||{};
    push(latChart,0,inf.yolo_avg_ms||0); push(latChart,1,inf.arcface_avg_ms||0);
    latChart.update();

    const cameras = d.cameras||{};
    const ks = Object.keys(cameras).sort((a,b)=>+a-+b);
    fpsChart.data.labels = ks.map(k=>'cam'+k);
    if (!fpsChart.data.datasets.length)
      fpsChart.data.datasets.push({data:[],backgroundColor:[],borderRadius:4});
    fpsChart.data.datasets[0].data = ks.map(k=>cameras[k].fps);
    fpsChart.data.datasets[0].backgroundColor = ks.map(k=>fpsColor(cameras[k].fps));
    fpsChart.update();

    const grid = document.getElementById('cam-grid');
    grid.innerHTML = ks.map(k => {
      const cam=cameras[k], fps=cam.fps;
      const cls=fps<5?'low':fps<15?'mid':'';
      return `<div class="cam-card">
        <div class="cam-title">Camera ${k}</div>
        <div class="cam-fps ${cls}">${fps.toFixed(1)}</div>
        <div class="cam-drops">fps &nbsp;·&nbsp; ${cam.frame_drops} drops</div>
      </div>`;
    }).join('');

    const alerts=[];
    if (cpu>=90) alerts.push('CPU critical: '+cpu.toFixed(0)+'%');
    if ((mem.percent||0)>=90) alerts.push('RAM critical: '+(mem.percent||0).toFixed(0)+'%');
    if (gpu&&gpu.mem_percent>=90) alerts.push('GPU VRAM critical: '+gpu.mem_percent.toFixed(0)+'%');
    ks.forEach(k=>{ const fps=cameras[k].fps; if(fps>0&&fps<5) alerts.push('Camera '+k+' low FPS: '+fps.toFixed(1)); });
    const box=document.getElementById('alert-box');
    box.style.display=alerts.length?'block':'none';
    document.getElementById('alert-list').innerHTML=alerts.map(a=>'<div class="alert-item">⚠ '+a+'</div>').join('');
  } catch {
    if (Date.now()-lastOk>10000) {
      document.getElementById('dot').className='stale';
      document.getElementById('status-text').textContent='Connection lost';
    }
  }
}
fetchLive();
setInterval(fetchLive, 2000);

// ── History ───────────────────────────────────────────────────────────────────
let hSys=null, hGpu=null, hLat=null, hFps=null;

function histLineOpts(unit) {
  return {responsive:true,maintainAspectRatio:true,animation:false,
    plugins:{legend:{labels:{color:'#8b949e',boxWidth:12,font:{size:11}}}},
    scales:{
      x:{ticks:{color:'#8b949e',font:{size:10},maxTicksLimit:12,maxRotation:0},grid:{color:'#30363d'}},
      y:{min:0,ticks:{color:'#8b949e',font:{size:11},callback:v=>v+(unit||'')},grid:{color:'#30363d'}}}};
}
function mkHistDs(label, color, data) {
  return {label, data, borderColor:color, backgroundColor:color+'18',
    tension:.3, fill:false, pointRadius:0, borderWidth:1.5};
}

async function loadAvailableDates() {
  try {
    const dates = await fetch('/api/dates').then(r=>r.json());
    const sel = document.getElementById('hist-dates-sel');
    sel.innerHTML = '<option value="">— available dates —</option>' +
      dates.map(d=>`<option value="${d}">${d}</option>`).join('');
    sel.onchange = () => {
      if (sel.value) { document.getElementById('hist-date').value = sel.value; }
    };
    // Default to today
    const today = new Date().toISOString().slice(0,10);
    document.getElementById('hist-date').value = today;
  } catch {}
}

async function loadHistory() {
  const date = document.getElementById('hist-date').value;
  if (!date) return;
  document.getElementById('hist-status').textContent = 'Loading…';
  try {
    const rows = await fetch('/api/history?date='+date).then(r=>r.json());
    if (!rows.length) { document.getElementById('hist-status').textContent = 'No data for '+date; return; }
    document.getElementById('hist-status').textContent = rows.length+' samples';

    // Downsample to max 300 points for readability
    const step = Math.max(1, Math.floor(rows.length/300));
    const sampled = rows.filter((_,i)=>i%step===0);

    const ts = sampled.map(r => new Date(r.ts*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}));
    const cpu = sampled.map(r => r.cpu_percent??null);
    const ram = sampled.map(r => r.ram_percent??null);
    const gpuU = sampled.map(r => r.gpu_util??null);
    const gpuM = sampled.map(r => r.gpu_mem_pct??null);
    const yolo = sampled.map(r => r.yolo_ms??null);
    const arc  = sampled.map(r => r.arcface_ms??null);

    // All camera keys found in the data
    const camKeys = [...new Set(sampled.flatMap(r=>Object.keys(r.cameras||{})))].sort((a,b)=>+a-+b);
    const camColors = ['#58a6ff','#3fb950','#ffa657','#f78166','#bc8cff','#39d353'];

    // Rebuild charts
    if (hSys) hSys.destroy();
    hSys = new Chart(document.getElementById('h-sys'), {
      type:'line', data:{labels:ts, datasets:[mkHistDs('CPU %','#58a6ff',cpu),mkHistDs('RAM %','#bc8cff',ram)]},
      options:histLineOpts('%')});

    if (hGpu) hGpu.destroy();
    hGpu = new Chart(document.getElementById('h-gpu'), {
      type:'line', data:{labels:ts, datasets:[mkHistDs('GPU util %','#3fb950',gpuU),mkHistDs('VRAM %','#ffa657',gpuM)]},
      options:histLineOpts('%')});

    if (hLat) hLat.destroy();
    hLat = new Chart(document.getElementById('h-lat'), {
      type:'line', data:{labels:ts, datasets:[mkHistDs('YOLO ms','#f78166',yolo),mkHistDs('ArcFace ms','#ffa657',arc)]},
      options:histLineOpts('ms')});

    if (hFps) hFps.destroy();
    hFps = new Chart(document.getElementById('h-fps'), {
      type:'line', data:{labels:ts, datasets:camKeys.map((k,i)=>
        mkHistDs('cam'+k, camColors[i%camColors.length], sampled.map(r=>(r.cameras[k]||{}).fps??null)))},
      options:histLineOpts(' fps')});

  } catch(e) {
    document.getElementById('hist-status').textContent = 'Error: '+e.message;
  }
}
</script>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):
    _metrics = None
    _store = None
    _camera_indices: List[int] = []

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._ok("application/json", json.dumps({"status": "ok", "timestamp": time.time()}).encode())

        elif path in ("/", "/index.html"):
            self._ok("text/html; charset=utf-8", _DASHBOARD_HTML.encode())

        elif path == "/api/metrics":
            try:
                snap = self.__class__._metrics.snapshot(self.__class__._camera_indices)
                self._ok("application/json", json.dumps(snap, default=float).encode())
            except Exception as e:
                self._err(500, str(e))

        elif path == "/api/history":
            store = self.__class__._store
            if store is None:
                self._ok("application/json", b"[]")
                return
            qs = parse_qs(parsed.query)
            date = (qs.get("date") or [""])[0]
            try:
                rows = store.get_day(date) if date else []
                self._ok("application/json", json.dumps(rows, default=float).encode())
            except Exception as e:
                self._err(500, str(e))

        elif path == "/api/dates":
            store = self.__class__._store
            try:
                dates = store.available_dates() if store else []
                self._ok("application/json", json.dumps(dates).encode())
            except Exception as e:
                self._err(500, str(e))

        elif path.startswith("/images/"):
            filename = os.path.basename(path)
            image_dir = os.environ.get(
                "SO_LOCAL_IMAGE_DIR",
                "/app/volumes/storage/person-tracking/images"
            )
            file_path = os.path.join(image_dir, filename)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    data = f.read()
                self._ok("image/jpeg", data)
            else:
                self._err(404, "Image not found")

        elif path == "/videos" or path == "/videos/":
            video_dir = "/app/volumes/storage/person-tracking"
            videos = sorted(
                f for f in os.listdir(video_dir)
                if f.endswith(".mp4") and not f.startswith("camera")
            )
            links = "".join(
                f'<li><a href="/videos/{v}" download>{v}</a></li>' for v in videos
            )
            html = f"<html><body><h2>Output Videos</h2><ul>{links}</ul></body></html>"
            self._ok("text/html; charset=utf-8", html.encode())

        elif path.startswith("/videos/"):
            filename = os.path.basename(path)
            video_dir = "/app/volumes/storage/person-tracking"
            file_path = os.path.join(video_dir, filename)
            if os.path.isfile(file_path) and filename.endswith(".mp4"):
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(os.path.getsize(file_path)))
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with open(file_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        self.wfile.write(chunk)
            else:
                self._err(404, "Video not found")

        else:
            self._err(404, "Not Found")

    def _ok(self, ct: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code: int, msg: str) -> None:
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ("200", "304"):
            logger.debug(f"[MetricsDashboard] {fmt % args}")


class MetricsDashboardServer:
    """Runs the HTTP monitoring dashboard + optional history store in daemon threads.

    Usage::

        server = MetricsDashboardServer(metrics, store, camera_indices=[0,1], port=8765)
        server.start()
        ...
        server.stop()
    """

    def __init__(
        self,
        metrics,
        store=None,
        camera_indices: Optional[List[int]] = None,
        port: int = 8765,
    ):
        self._metrics = metrics
        self._store = store
        self._camera_indices = camera_indices or []
        self._port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def update_camera_indices(self, indices: List[int]) -> None:
        _Handler._camera_indices = indices
        if self._store:
            self._store.update_camera_indices(indices)

    def start(self) -> None:
        _Handler._metrics = self._metrics
        _Handler._store = self._store
        _Handler._camera_indices = self._camera_indices

        if self._store:
            self._store.start()

        self._server = HTTPServer(("0.0.0.0", self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="metrics-dashboard",
        )
        self._thread.start()
        logger.info(f"Metrics dashboard running at http://0.0.0.0:{self._port}/")

    def stop(self) -> None:
        if self._store:
            self._store.stop()
        if self._server:
            self._server.shutdown()
        logger.info("Metrics dashboard stopped")
