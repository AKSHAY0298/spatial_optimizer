"""
Web-based interactive UI for 6G Tower Placement Optimization.

Opens a browser tab with sliders. Click "Run" → backend runs the pipeline →
new map + metrics appear. Zero dependencies beyond core/ and stdlib.
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs

import matplotlib
matplotlib.use("Agg")  # non-GUI backend — avoids tkinter thread warnings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from core.data import find_cities_file
from core.pipeline import run

CITIES_FILE = find_cities_file()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _km_to_degrees_lon(km: float, lat: float) -> float:
    return km / (111.320 * math.cos(math.radians(lat)))

def _km_to_degrees_lat(km: float) -> float:
    return km / 110.574


def _render_map_image(result) -> bytes:
    """Run the pipeline, render the map to a PNG, return raw bytes."""
    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    cities = result.cities
    labels = result.labels
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(unique_labels), 1))

    for label in unique_labels:
        mask = labels == label
        subset = cities[mask]
        if label < 0:
            color, mkr, sz, zo = "#888888", "x", 22, 2
        else:
            color, mkr, sz, zo = cmap(label % 20), "o", 20, 2
        kw = dict(c=[color], marker=mkr, s=sz, linewidths=0.5, zorder=zo)
        if label >= 0:
            kw["edgecolors"] = "white"
        ax.scatter(subset["longitude"], subset["latitude"], **kw)

    selected = result.optimization.selected_candidates
    for tower in selected:
        rx = _km_to_degrees_lon(tower.radius_km, tower.latitude)
        ry = _km_to_degrees_lat(tower.radius_km)
        fc = "#00e676" if tower.tower_type == "dense" else "#ffab00"
        ax.add_patch(mpatches.Ellipse(
            (tower.longitude, tower.latitude), 2 * rx, 2 * ry,
            lw=0.8, ec=fc, fc=fc, alpha=0.12, zorder=1))
        ax.add_patch(mpatches.Ellipse(
            (tower.longitude, tower.latitude), 2 * rx, 2 * ry,
            lw=0.6, ec=fc, fc="none", alpha=0.45, zorder=3))

    tlons = [t.longitude for t in selected]
    tlats = [t.latitude for t in selected]
    ax.scatter(tlons, tlats, c="#ff1744", marker="^", s=70,
               linewidths=0.6, edgecolors="white", zorder=5)

    pv = result.optimization.pair_values
    sel = set(result.optimization.selected_indices.tolist())
    for pi, (li, ri) in enumerate(result.interference_pairs):
        if li in sel and ri in sel and pv[pi] > 0.5:
            a, b = result.candidates[li], result.candidates[ri]
            ax.plot([a.longitude, b.longitude], [a.latitude, b.latitude],
                    color="#ff1744", lw=1.2, ls="--", alpha=0.7, zorder=4)

    rp = result.radius_plan
    handles = [
        mpatches.Patch(fc="#00e676", alpha=0.35, ec="#00e676",
                       label=f"Dense {rp.dense_radius_km:.0f} km"),
        mpatches.Patch(fc="#ffab00", alpha=0.35, ec="#ffab00",
                       label=f"Sparse {rp.sparse_radius_km:.0f} km"),
        plt.Line2D([], [], color="#ff1744", ls="--", lw=1.2, label="Interference"),
        plt.Line2D([], [], color="#ff1744", marker="^", ls="None",
                   markersize=7, mec="white", label="Tower"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=8,
                    facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
                    framealpha=0.9, ncol=2)
    leg.get_frame().set_linewidth(0.5)

    opt = result.optimization
    ax.set_title(
        f"{len(selected)} towers  |  Coverage {opt.coverage_ratio:.0%}  |  "
        f"Objective {opt.objective_value:.2f}",
        color="white", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Longitude", color="white", fontsize=10, fontweight="bold")
    ax.set_ylabel("Latitude", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_aspect("equal")
    ax.set_xlim(5.5, 15.5)
    ax.set_ylim(47.0, 55.0)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _metrics_dict(result) -> dict:
    opt = result.optimization
    rp = result.radius_plan
    cost = sum(c.cost for c in opt.selected_candidates)
    return {
        "dense_radius": f"{rp.dense_radius_km:.0f}",
        "sparse_radius": f"{rp.sparse_radius_km:.0f}",
        "towers": len(opt.selected_candidates),
        "coverage": f"{opt.coverage_ratio:.1%}",
        "cost": f"{cost:.2f}",
        "clusters": result.cluster_profile.cluster_count,
        "noise": result.cluster_profile.noise_count,
        "candidates": len(result.candidates),
        "solver": f"{opt.status}",
        "objective": f"{opt.objective_value:.2f}",
    }


# ---------------------------------------------------------------------------
# HTML page (single-page app)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>6G Tower Placement</title>
<style>
  :root {
    --bg: #1a1a2e; --panel: #16213e; --accent: #00bcd4;
    --green: #00e676; --red: #ff1744; --text: #e0e0e0; --muted: #888;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
         color: var(--text); height: 100vh; display: flex; overflow: hidden; }

  /* ---- left panel ---- */
  #panel {
    width: 300px; min-width: 300px; background: var(--panel); padding: 24px 20px;
    display: flex; flex-direction: column; gap: 18px; border-right: 1px solid #333;
  }
  #panel h1 { font-size: 1.2rem; color: var(--accent); text-align: center;
              letter-spacing: 0.5px; margin-bottom: 2px; }
  .param { display: flex; flex-direction: column; gap: 4px; }
  .param label { font-size: 0.82rem; font-weight: 600; color: #ccc;
                 display: flex; justify-content: space-between; }
  .param label span { color: var(--accent); font-weight: 700; }
  input[type=range] { -webkit-appearance: none; width: 100%; height: 6px;
    background: #333; border-radius: 3px; outline: none; cursor: pointer; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none;
    width: 18px; height: 18px; border-radius: 50%; background: var(--accent);
    cursor: pointer; border: 2px solid #fff; }

  #btn {
    padding: 12px; font-size: 0.95rem; font-weight: 700; letter-spacing: 0.5px;
    border: none; border-radius: 6px; background: var(--accent); color: #111;
    cursor: pointer; transition: 0.15s;
  }
  #btn:hover { background: #00e5ff; }
  #btn:active { transform: scale(0.97); }
  #btn.loading { opacity: 0.6; pointer-events: none; }

  /* ---- metrics card ---- */
  #metrics {
    background: var(--bg); border-radius: 8px; padding: 14px 16px;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 0.8rem; line-height: 1.7; color: #ccc; flex: 1; overflow-y: auto;
  }
  #metrics .val { color: var(--accent); font-weight: 600; }
  #metrics .good { color: var(--green); }

  /* ---- right side: map ---- */
  #map-area {
    flex: 1; display: flex; align-items: center; justify-content: center;
    padding: 20px; position: relative;
  }
  #map-area img { max-width: 100%; max-height: 100%; border-radius: 6px;
                   box-shadow: 0 0 30px rgba(0,0,0,0.5); }
  #spinner {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    font-size: 2rem; color: var(--accent); pointer-events: none; display: none;
  }
</style>
</head>
<body>

<div id="panel">
  <h1>6G Tower Placement</h1>

  <div class="param">
    <label>eps-km <span id="v-eps">45</span></label>
    <input type="range" id="eps" min="10" max="100" value="45" step="1">
  </div>
  <div class="param">
    <label>min-samples <span id="v-min">2</span></label>
    <input type="range" id="min_s" min="1" max="10" value="2" step="1">
  </div>
  <div class="param">
    <label>&alpha; (cost weight) <span id="v-alpha">0.50</span></label>
    <input type="range" id="alpha" min="0" max="1" value="0.5" step="0.01">
  </div>
  <div class="param">
    <label>&beta; (coverage incentive) <span id="v-beta">0.25</span></label>
    <input type="range" id="beta" min="0" max="1" value="0.25" step="0.01">
  </div>
  <div class="param">
    <label>time-limit <span id="v-time">60 s</span></label>
    <input type="range" id="time" min="5" max="300" value="60" step="5">
  </div>

  <button id="btn" onclick="run()">Run Optimization</button>

  <div id="metrics"><span class="val">Click Run</span> to start</div>
</div>

<div id="map-area">
  <img id="map" src="" alt="Map will appear after run">
  <div id="spinner">Running&#8230;</div>
</div>

<script>
  // --- live slider value display ---
  for (const id of ['eps','min_s','alpha','beta','time']) {
    const sl = document.getElementById(id);
    const disp = document.getElementById('v-'+id);
    const fmt = id==='alpha'||id==='beta' ? v => v.toFixed(2) :
                id==='time' ? v => v+' s' : v => v;
    sl.oninput = () => { disp.textContent = fmt(Number(sl.value)); };
  }

  async function run() {
    const btn = document.getElementById('btn');
    const map = document.getElementById('map');
    const mdiv = document.getElementById('metrics');
    const spin = document.getElementById('spinner');
    btn.classList.add('loading');
    btn.textContent = 'Running...';
    spin.style.display = 'block';
    map.style.opacity = '0.4';

    const params = new URLSearchParams({
      eps: document.getElementById('eps').value,
      min_samples: document.getElementById('min_s').value,
      alpha: document.getElementById('alpha').value,
      beta: document.getElementById('beta').value,
      time_limit: document.getElementById('time').value,
    });

    try {
      const resp = await fetch('/run?' + params.toString());
      const data = await resp.json();
      map.src = 'data:image/png;base64,' + data.image;
      map.style.opacity = '1';
      mdiv.innerHTML = data.metrics;
    } catch(e) {
      mdiv.innerHTML = '<span style="color:#ff5252">Error: '+e.message+'</span>';
    } finally {
      spin.style.display = 'none';
      btn.classList.remove('loading');
      btn.textContent = 'Run Optimization';
    }
  }

  // --- initial run ---
  run();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # quiet

    def do_GET(self):
        if self.path.startswith("/run"):
            qs = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            def _p(key, cast):
                return cast(qs[key][0]) if key in qs else None
            try:
                result = run(
                    CITIES_FILE,
                    eps_km=_p("eps", float) or 45.0,
                    min_samples=_p("min_samples", int) or 2,
                    alpha=_p("alpha", float) or 0.5,
                    beta=_p("beta", float) or 0.0,
                    time_limit=_p("time_limit", float) or 60.0,
                    radius_mode="milp_pair_search",
                )
                img_bytes = _render_map_image(result)
                img_b64 = base64.b64encode(img_bytes).decode()
                m = _metrics_dict(result)
                metrics_html = (
                    f"r_dense  = <span class='val'>{m['dense_radius']} km</span><br>"
                    f"r_sparse = <span class='val'>{m['sparse_radius']} km</span><br>"
                    f"Towers   = <span class='val'>{m['towers']}</span><br>"
                    f"Coverage = <span class='val good'>{m['coverage']}</span><br>"
                    f"Cost     = <span class='val'>{m['cost']}</span><br>"
                    f"Clusters = <span class='val'>{m['clusters']}</span><br>"
                    f"Noise    = <span class='val'>{m['noise']}</span><br>"
                    f"Candidates = <span class='val'>{m['candidates']}</span><br>"
                    f"Solver   = <span class='val'>{m['solver']}</span>"
                )
                resp = json.dumps({"image": img_b64, "metrics": metrics_html})
                self._json(resp)
            except Exception as e:
                self._json(json.dumps({"error": str(e)}), code=500)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())

    def _json(self, body: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())


def main():
    port = 8765
    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"UI running at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
