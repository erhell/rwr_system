#!/usr/bin/env python3
"""Spin panel: a local web page to run the cube-spin policies on the real hand and record the measured speed.

Start it in a terminal with the ROS setup (the hand controller runs in another terminal):
    mamba activate rwr_faive_lab_env && source ~/ros_ws/install/setup.zsh
    python ~/Documents/RWR/spin_panel.py            # then open http://localhost:8765
Stop it with Ctrl+C (a running policy is stopped too).

Each run starts experiments/policy_inference_cube_spin_v2.py with the chosen model, speed and direction and logs to
rwr_system/logs/policy_runs/<model>_<direction>_<speed>[_clipped][_dry]_<date>.csv like spin.zsh. Every run and the values
typed in on the page (measured speed, drops, note) are kept in rwr_system/logs/policy_runs/real_hand_results.csv.
Only listens on localhost.

The policy's last action is fed back unclipped by default, as the node did before 2026-11-11 (the policies spun
faster on the real hand that way); the "clip" checkbox clips it to [-1, 1] like the training. Results: variant
"unclipped" / "clipped" (older rows: "old" = unclipped, "new" = clipped).
"""
import collections
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path.home()
RUNS_DIR = HOME / "Documents/RWR/rwr_faive_lab/logs/rsl_rl/orca_v2_repose"
LOG_DIR = Path(os.environ.get("SPIN_PANEL_LOG_DIR", HOME / "Documents/RWR/rwr_system/logs/policy_runs"))
RESULTS = LOG_DIR / "real_hand_results.csv"
PORT = int(os.environ.get("SPIN_PANEL_PORT", "8765"))
NODE = "experiments/lib/experiments/policy_inference_cube_spin_v2.py"

# name, run folder, group, what the sim says (spin speed about the commanded axis at 1.5-2.5 rad/s commands)
MODELS = [
    ("r31", "2026-11-11_15-44-24_r31", "Joint-only policies", "sim: 1.88 rad/s, 4.4% drops; lag matched to the real hand"),
    ("r33B", "2026-11-13_02-59-19_r33B", "Joint-only policies", "sim: 2.10 rad/s, 7.0% drops; GRU student, sensitive to extra lag"),
    ("r30", "2026-11-10_18-39-03_r30", "Joint-only policies", "sim: 1.61 rad/s, 7.4% drops (with its lag); older motor fits"),
    ("r29", "2026-11-11_00-15-49_r29", "Joint-only policies", "sim: 1.79 rad/s, 6.1% drops; no lag training"),
    ("r28", "2026-11-09_20-39-13_r28", "Joint-only policies", "sim: 1.47 rad/s, 5.4% drops"),
    ("r21", "2026-11-08_20-44-06_r21_sysid_16k", "Joint-only policies", "r21_sysid_16k, trained at 0.9-1.5: sim 1.36 rad/s there, 1.29 at 1.5-2.5"),
    ("r33A", "2026-11-12_15-58-48_r33A", "Joint-only policies", "sim: 1.87 rad/s, 8.8% drops; GRU student, worse than r31 in sim"),
    ("r32", "2026-11-11_16-01-53_r32", "Camera policies, run without the camera", "sim with the camera off: 1.95 rad/s, 5.2% drops"),
    ("r33cB", "2026-11-13_02-59-19_r33cB", "Camera policies, run without the camera", "sim with the camera off: 2.14 rad/s, 5.4% drops; GRU"),
    ("r33cBd", "2026-11-13_07-22-25_r33cBd", "Camera policies, run without the camera", "sim with the camera off: 2.02 rad/s, 10.0% drops; GRU, delay-robust"),
    ("r26_8k", "2026-11-08_12-51-25_r26_8k", "Camera policies, run without the camera", "trained with the camera off in 20% of episodes, up to 2.0 rad/s; before the sysID fits"),
]
MODEL_BY_NAME = {m[0]: m for m in MODELS}
SPEEDS = [1.0, 1.3, 1.6, 2.0]
FIELDS = ["id", "date", "time", "model", "run_folder", "direction", "speed", "goal_ang_vel", "variant", "fixed_wrist_deg", "dry_run",
          "run_seconds", "duration_s", "exit_code", "measured_rad_s", "drops", "note", "csv"]


def find_node() -> Path:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix and (Path(prefix).parent / NODE).exists():
            return Path(prefix).parent / NODE
        if prefix and (Path(prefix) / NODE.split("/", 1)[1]).exists():
            return Path(prefix) / NODE.split("/", 1)[1]
    sys.exit("policy node not found: run this in a terminal with `source ~/ros_ws/install/setup.zsh`")


class Panel:
    def __init__(self, node: Path):
        self.node = node
        self.lock = threading.Lock()
        self.proc = None
        self.current = None  # the running (or last) run's record
        self.lines = collections.deque(maxlen=300)
        self.version = 0
        self.results = self.load()
        self.estop_msg = ""

    # ---------- results file ----------
    def load(self):
        if not RESULTS.exists():
            return []
        with open(RESULTS, newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]

    def save(self):  # called with the lock held
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = RESULTS.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.results)
        tmp.replace(RESULTS)
        self.version += 1

    # ---------- runs ----------
    def start(self, cfg):
        name = cfg.get("model")
        if name not in MODEL_BY_NAME:
            raise ValueError(f"unknown model {name!r}")
        speed = float(cfg.get("speed"))
        if not 0.0 < speed <= 5.0:
            raise ValueError("speed must be between 0 and 5 rad/s")
        direction = cfg.get("direction")
        if direction not in ("ccw", "cw", "flip"):
            raise ValueError("direction must be ccw, cw or flip")
        seconds = float(cfg.get("seconds") or 0)
        if not 0.0 <= seconds <= 3600:
            raise ValueError("run time must be between 0 and 3600 s")
        clip, dry = bool(cfg.get("clip")), bool(cfg.get("dry"))
        fix_wrist = bool(cfg.get("fix_wrist"))
        wrist_deg = float(cfg.get("wrist_deg", 15.0))
        if fix_wrist and not -65.0 <= wrist_deg <= 35.0:
            raise ValueError("the wrist angle must be between -65 and 35 deg")
        _, folder, _, _ = MODEL_BY_NAME[name]
        policy = RUNS_DIR / folder / "exported/policy.pt"
        if not policy.exists():
            raise ValueError(f"{policy} not found")
        goal = -speed if direction == "cw" else speed
        flip = 8.0 if direction == "flip" else 0.0
        now = datetime.now()
        csv_path = LOG_DIR / (f"{name}_{direction}_{speed:g}{'_clipped' if clip else ''}{f'_wrist{wrist_deg:g}' if fix_wrist else ''}{'_dry' if dry else ''}"
                              f"_{now:%Y%m%d_%H%M%S}.csv")
        cmd = [str(self.node), "--ros-args",
               "-p", f"policy_ckpt_path:={policy}",
               "-p", f"goal_ang_vel:={goal:.3f}",
               "-p", f"goal_flip_period_s:={flip:.1f}",
               "-p", f"clip_last_action:={'true' if clip else 'false'}",
               "-p", f"run_seconds:={seconds:.1f}",
               "-p", f"fix_wrist:={'true' if fix_wrist else 'false'}",
               "-p", f"fixed_wrist_deg:={wrist_deg:.1f}",
               "-p", f"csv_log_path:={csv_path}",
               "-p", f"dry_run:={'true' if dry else 'false'}"]
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("a run is still going: stop it first")
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.lines.clear()
            self.lines.append("$ " + " ".join(cmd))
            # own session: Ctrl+C in this terminal does not reach the node; stop() sends it SIGINT instead
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            self.current = {
                "id": f"{now:%Y%m%d_%H%M%S}", "date": f"{now:%Y-%m-%d}", "time": f"{now:%H:%M:%S}",
                "model": name, "run_folder": folder, "direction": direction, "speed": f"{speed:g}",
                "goal_ang_vel": f"{goal:g}", "variant": "clipped" if clip else "unclipped",
                "fixed_wrist_deg": f"{wrist_deg:g}" if fix_wrist else "", "dry_run": "yes" if dry else "no",
                "run_seconds": f"{seconds:g}", "duration_s": "", "exit_code": "", "measured_rad_s": "", "drops": "",
                "note": "", "csv": str(csv_path), "_start": time.time(),
            }
            self.results.append(self.current)
            self.save()
            threading.Thread(target=self._reader, args=(self.proc, self.current), daemon=True).start()
        return self.current["id"]

    def _reader(self, proc, run):
        for line in proc.stdout:
            line = line.rstrip()
            if "FutureWarning" in line or "warnings.warn(" in line:
                continue
            self.lines.append(line)
        code = proc.wait()
        with self.lock:
            run["duration_s"] = f"{time.time() - run['_start']:.1f}"
            run["exit_code"] = str(code)
            self.lines.append(f"--- run ended after {run['duration_s']} s (exit code {code})")
            self.save()

    def stop(self):
        with self.lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return False
        for sig, wait in ((signal.SIGINT, 6), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return True
            try:
                proc.wait(wait)
                return True
            except subprocess.TimeoutExpired:
                continue
        return True

    def estop(self):
        self.stop()
        try:
            out = subprocess.run(["ros2", "service", "call", "/right/hand/estop", "std_srvs/srv/SetBool", "{data: true}"],
                                 capture_output=True, text=True, timeout=20)
            self.estop_msg = (out.stdout + out.stderr).strip().splitlines()[-1:] or ["sent"]
            self.estop_msg = self.estop_msg[0]
        except Exception as e:  # noqa: BLE001
            self.estop_msg = f"estop call failed: {e}"
        self.lines.append(f"--- E-STOP: {self.estop_msg}")
        return self.estop_msg

    def update(self, data):
        with self.lock:
            for r in self.results:
                if r["id"] == data.get("id"):
                    for k in ("measured_rad_s", "drops", "note"):
                        if k in data:
                            r[k] = str(data[k]).strip()
                    self.save()
                    return True
        return False

    def delete(self, run_id):
        with self.lock:
            if self.current is not None and self.current["id"] == run_id and self.proc and self.proc.poll() is None:
                raise RuntimeError("cannot delete the running run")
            n = len(self.results)
            self.results = [r for r in self.results if r["id"] != run_id]
            if len(self.results) != n:
                self.save()

    def state(self):
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            cur = None
            if self.current is not None:
                cur = {k: v for k, v in self.current.items() if not k.startswith("_")}
                cur["elapsed_s"] = round(time.time() - self.current["_start"], 1) if running else None
            recent = list(self.lines)[-40:]
            waiting = running and any("Waiting for hand state" in l for l in recent[-3:])
            return {"running": running, "waiting_for_hand": waiting, "current": cur, "log": recent,
                    "version": self.version, "results": [{k: v for k, v in r.items() if not k.startswith("_")}
                                                         for r in self.results],
                    "estop": self.estop_msg}


PANEL: Panel = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/config":
            self._send(200, json.dumps({"models": [{"name": m[0], "folder": m[1], "group": m[2], "sim": m[3]}
                                                   for m in MODELS], "speeds": SPEEDS, "results_file": str(RESULTS)}))
        elif self.path == "/api/state":
            self._send(200, json.dumps(PANEL.state()))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/start":
                out = {"id": PANEL.start(data)}
            elif self.path == "/api/stop":
                out = {"stopped": PANEL.stop()}
            elif self.path == "/api/estop":
                out = {"estop": PANEL.estop()}
            elif self.path == "/api/result":
                out = {"saved": PANEL.update(data)}
            elif self.path == "/api/delete":
                PANEL.delete(data.get("id"))
                out = {"deleted": True}
            else:
                return self._send(404, json.dumps({"error": "not found"}))
            self._send(200, json.dumps(out))
        except (ValueError, RuntimeError) as e:
            self._send(400, json.dumps({"error": str(e)}))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spin Panel</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mute:#667085;--line:#dde1e7;--acc:#2f6fdf;--accfg:#fff;--ok:#1f8a4c;
--warn:#b7791f;--bad:#c0392b;--cell:#eef1f5;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--fg:#e6e8ec;--mute:#9aa3b2;--line:#2e343e;
--acc:#5b8ff0;--accfg:#0d1117;--ok:#3fb27a;--warn:#d9a441;--bad:#e5675a;--cell:#262b34}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0}h2{font-size:15px;margin:0 0 10px}
.top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.pill{padding:3px 10px;border-radius:99px;font-size:13px;background:var(--cell);color:var(--mute)}
.pill.run{background:var(--ok);color:#fff}.pill.wait{background:var(--warn);color:#fff}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media (max-width:860px){.grid2{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
label{display:block;font-size:12px;color:var(--mute);margin:10px 0 4px}
select,input[type=number],input[type=text]{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);
border-radius:6px;padding:6px 8px;width:100%}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.seg button,.btn{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;
padding:6px 12px;cursor:pointer}
.seg button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.btn.go{background:var(--ok);color:#fff;border-color:var(--ok);font-weight:600;padding:9px 22px}
.btn.stop{background:var(--warn);color:#fff;border-color:var(--warn)}.btn.es{background:var(--bad);color:#fff;border-color:var(--bad);font-weight:700}
.btn:disabled{opacity:.45;cursor:default}
.hint{font-size:12px;color:var(--mute);margin-top:4px}
.big{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:13px}.kv span:nth-child(odd){color:var(--mute)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mute);font-weight:500;position:sticky;top:0;background:var(--card)}
td input{padding:3px 6px;width:100%}td.num{font-variant-numeric:tabular-nums;text-align:right}
.matrix td,.matrix th{text-align:center;padding:4px}.matrix td.m{cursor:pointer;background:var(--cell);border:2px solid var(--card);
border-radius:6px;min-width:58px;font-variant-numeric:tabular-nums}.matrix td.m:hover{outline:2px solid var(--acc)}
.matrix td.sel{outline:2px solid var(--acc)}.matrix th.mod{text-align:left}
.log{font:12px/1.4 var(--mono);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;
max-height:260px;overflow:auto;white-space:pre-wrap}
.scroll{max-height:420px;overflow:auto}.x{color:var(--bad);cursor:pointer;border:0;background:none;font-size:15px}
.err{color:var(--bad);font-size:13px;min-height:18px}
</style></head><body><main>
<div class="top"><h1>Spin panel</h1><span id="pill" class="pill">idle</span><span class="hint" id="resfile"></span></div>
<div class="grid2">
 <div class="card"><h2>Run</h2>
  <label>Model</label><select id="model"></select><div class="hint" id="simnote"></div>
  <label>Commanded speed (rad/s)</label>
  <div class="row"><div class="seg" id="speeds"></div><input id="speed" type="number" step="0.1" min="0.1" max="5" style="width:90px"></div>
  <label>Direction (seen from above)</label>
  <div class="seg row" id="dirs"><button data-d="ccw">CCW ⟲</button><button data-d="cw">CW ⟳</button><button data-d="flip">flip every 8 s</button></div>
  <label>Run time (s, 0 = until Stop)</label><input id="secs" type="number" min="0" max="3600" step="5" value="30" style="width:110px">
  <div class="row" style="margin-top:10px"><label style="margin:0"><input type="checkbox" id="clip"> clip the previous actions to [-1, 1] (as in training; default off = as before 11-11)</label></div>
  <div class="row"><label style="margin:0"><input type="checkbox" id="fixw" checked> fix the wrist at</label><input id="wdeg" type="number" value="15" step="1" min="-65" max="35" style="width:70px">°<span class="hint">(+ = fingertips towards the palm = up; −65…35; off = the policy moves the wrist)</span></div>
  <div class="row"><label style="margin:0"><input type="checkbox" id="dry"> dry run (nothing sent to the hand)</label></div>
  <div class="row" style="margin-top:14px"><button class="btn go" id="start">Start</button><button class="btn stop" id="stop">Stop</button>
   <span style="flex:1"></span><button class="btn es" id="estop">E-STOP</button></div>
  <div class="err" id="err"></div>
 </div>
 <div class="card"><h2 id="curtitle">Current run</h2>
  <div id="cur" class="hint">No run yet.</div>
  <div id="meas" style="display:none">
   <label>Real measured speed (rad/s)</label>
   <div class="row"><input id="m_val" type="number" step="0.01" min="0" style="width:120px"><span id="ratio" class="hint"></span></div>
   <div class="hint">or count: <input id="m_turns" type="number" step="0.25" min="0" style="width:80px"> turns in
    <input id="m_secs" type="number" step="0.5" min="0" style="width:80px"> s → <button class="btn" id="m_calc">fill in</button></div>
   <label>Drops</label><input id="m_drops" type="number" min="0" step="1" style="width:120px">
   <label>Note</label><input id="m_note" type="text" placeholder="e.g. smooth, cube slipped at the thumb, …">
   <div class="row" style="margin-top:10px"><button class="btn go" id="m_save">Save</button><span id="m_ok" class="hint"></span></div>
  </div>
 </div>
</div>
<div class="card"><h2>Test grid</h2><div class="hint" style="margin-bottom:8px">Runs with <b id="gridmode">unclipped previous actions, wrist fixed at +15°</b> (follows the checkboxes above). Latest measured speed per model, speed and direction
 (rad/s; colour = measured / commanded). “run” = run without a measured value. Click a cell to select it.</div>
 <div class="scroll"><table class="matrix" id="matrix"></table></div></div>
<div class="card"><h2>All runs</h2><div class="hint" style="margin-bottom:8px">Edit measured speed, drops and note directly; saved when you leave the field.</div>
 <div class="scroll"><table id="runs"></table></div></div>
<div class="card"><h2>Node output</h2><div class="log" id="log"></div></div>
</main>
<script>
window.onerror=(msg,src,line)=>{const p=document.getElementById('err');if(p)p.textContent=`page error: ${msg} (line ${line})`;};
const $=id=>document.getElementById(id);
let cfg=null, st=null, sel={model:null,speed:1.0,dir:'ccw'}, lastVersion=-1, shownRun=null;
const fmt=(x,d=2)=>x===''||x==null||isNaN(+x)?'':(+x).toFixed(d);
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json(); if(!r.ok) throw new Error(j.error||r.status); return j;}
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
const isClipped=r=>r.variant==='clipped'||r.variant==='new';
const wristKey=r=>r.fixed_wrist_deg?String(+r.fixed_wrist_deg):'';
const curWrist=()=>$('fixw').checked?String(+$('wdeg').value):'';
const wristText=w=>w===''?'wrist moved by the policy':`wrist fixed at ${+w>0?'+':''}${w}°`;
function gridMode(){$('gridmode').textContent=($('clip').checked?'clipped':'unclipped')+' previous actions, '+wristText(curWrist());renderMatrix();}
function ratioColor(r){if(!isFinite(r))return'';const h=Math.max(0,Math.min(120,r*120));return`hsl(${h} 60% 45% / .35)`;}
function setSel(p){Object.assign(sel,p);$('model').value=sel.model;$('speed').value=sel.speed;
 document.querySelectorAll('#speeds button').forEach(b=>b.classList.toggle('on',+b.dataset.s===+sel.speed));
 document.querySelectorAll('#dirs button').forEach(b=>b.classList.toggle('on',b.dataset.d===sel.dir));
 const m=cfg.models.find(m=>m.name===sel.model);$('simnote').textContent=m?m.sim:'';renderMatrix();}
async function init(){
 cfg=await api('/api/config');$('resfile').textContent='results: '+cfg.results_file;
 let html='',g=null;for(const m of cfg.models){if(m.group!==g){if(g)html+='</optgroup>';g=m.group;html+=`<optgroup label="${esc(g)}">`;}
  html+=`<option value="${m.name}">${m.name}</option>`;}$('model').innerHTML=html+'</optgroup>';
 $('speeds').innerHTML=cfg.speeds.map(s=>`<button data-s="${s}">${s.toFixed(1)}</button>`).join('');
 $('speeds').onclick=e=>{if(e.target.dataset.s)setSel({speed:+e.target.dataset.s});};
 $('dirs').onclick=e=>{if(e.target.dataset.d)setSel({dir:e.target.dataset.d});};
 $('model').onchange=()=>setSel({model:$('model').value});$('speed').onchange=()=>setSel({speed:+$('speed').value});
 $('start').onclick=start;$('clip').onchange=gridMode;$('fixw').onchange=gridMode;$('wdeg').onchange=gridMode;$('stop').onclick=()=>api('/api/stop',{}).then(poll);
 $('estop').onclick=()=>{api('/api/estop',{}).then(poll).catch(e=>$('err').textContent=e.message);};
 $('m_calc').onclick=()=>{const t=+$('m_turns').value,s=+$('m_secs').value;if(t>0&&s>0){$('m_val').value=(t*2*Math.PI/s).toFixed(2);showRatio();}};
 $('m_val').oninput=showRatio;$('m_save').onclick=saveMeas;
 setSel({model:cfg.models[0].name});poll();setInterval(poll,1000);}
async function start(){$('err').textContent='';
 try{await api('/api/start',{model:sel.model,speed:+$('speed').value,direction:sel.dir,seconds:+$('secs').value,clip:$('clip').checked,fix_wrist:$('fixw').checked,wrist_deg:+$('wdeg').value,dry:$('dry').checked});
  shownRun=null;poll();}catch(e){$('err').textContent=e.message;}}
function showRatio(){const c=st&&st.current;if(!c)return;const v=+$('m_val').value;
 $('ratio').textContent=v>0?`= ${(v/(+c.speed)*100).toFixed(0)} % of the commanded ${c.speed} rad/s`:'';}
async function saveMeas(){const c=st.current;await api('/api/result',{id:c.id,measured_rad_s:$('m_val').value,drops:$('m_drops').value,note:$('m_note').value});
 $('m_ok').textContent='saved ✓';setTimeout(()=>$('m_ok').textContent='',2000);poll();}
async function poll(){try{st=await api('/api/state');}catch(e){$('pill').textContent='panel server not reachable';$('pill').className='pill wait';return;}
 const c=st.current;$('pill').className='pill'+(st.waiting_for_hand?' wait':st.running?' run':'');
 $('pill').textContent=st.waiting_for_hand?'waiting for hand state: controller running? hand connected?':
  st.running?`running ${c.model} ${c.direction} ${c.speed} rad/s · ${c.elapsed_s.toFixed(0)} s`+(+c.run_seconds?` / ${c.run_seconds} s`:''):'idle';
 $('start').disabled=st.running;$('stop').disabled=!st.running;
 if(c){$('curtitle').textContent=st.running?'Current run':'Last run';
  $('cur').innerHTML=`<div class="big">${esc(c.model)} · ${c.direction==='cw'?'CW ⟳':c.direction==='ccw'?'CCW ⟲':'flip'} · ${esc(c.speed)} rad/s</div>
  <div class="kv" style="margin-top:6px"><span>status</span><span>${st.running?'running '+c.elapsed_s.toFixed(0)+' s':'ended after '+esc(c.duration_s)+' s (exit '+esc(c.exit_code)+')'}</span>
  <span>node</span><span>${isClipped(c)?'previous actions clipped (as in training)':'previous actions unclipped (as before 11-11)'}, ${wristText(wristKey(c))}${c.dry_run==='yes'?', DRY RUN':''}</span><span>log</span><span>${esc(c.csv.split('/').pop())}</span></div>`;
  $('meas').style.display='';
  if(shownRun!==c.id){shownRun=c.id;$('m_val').value=c.measured_rad_s;$('m_drops').value=c.drops;$('m_note').value=c.note;
   $('m_turns').value='';$('m_secs').value=+c.run_seconds||'';showRatio();}
  if(!st.running&&!$('m_secs').value&&c.duration_s)$('m_secs').value=(+c.run_seconds||+c.duration_s);}
 const log=$('log'),atEnd=log.scrollTop+log.clientHeight>=log.scrollHeight-5;log.textContent=st.log.join('\n');if(atEnd)log.scrollTop=log.scrollHeight;
 if(st.version!==lastVersion&&!$('runs').contains(document.activeElement)){lastVersion=st.version;renderRuns();renderMatrix();}}
function latest(model,speed,dir){let best=null;for(const r of st.results){if(r.model===model&&+r.speed===+speed&&r.direction===dir&&r.dry_run!=='yes'&&isClipped(r)===$('clip').checked&&wristKey(r)===curWrist()){
 if(!best||(r.measured_rad_s!==''&&(best.measured_rad_s===''||r.id>best.id))||(r.measured_rad_s===''&&best.measured_rad_s===''&&r.id>best.id))best=r;}}return best;}
function renderMatrix(){if(!cfg||!st)return;const dirs=['ccw','cw'];let h='<tr><th></th>'+cfg.speeds.map(s=>`<th colspan="2">${s.toFixed(1)}</th>`).join('')+'</tr>';
 h+='<tr><th></th>'+cfg.speeds.map(()=>'<th>CCW</th><th>CW</th>').join('')+'</tr>';
 for(const m of cfg.models){h+=`<tr><th class="mod">${m.name}</th>`;for(const s of cfg.speeds)for(const d of dirs){const r=latest(m.name,s,d);
  let txt='',bg='';if(r){if(r.measured_rad_s!==''){txt=fmt(r.measured_rad_s);bg=ratioColor(+r.measured_rad_s/s);}else txt='run';}
  const on=sel.model===m.name&&+sel.speed===s&&sel.dir===d;
  h+=`<td class="m${on?' sel':''}" style="background:${bg}" data-m="${m.name}" data-s="${s}" data-d="${d}" title="${m.name} ${d} ${s}">${txt}</td>`;}h+='</tr>';}
 $('matrix').innerHTML=h;$('matrix').onclick=e=>{const t=e.target.closest('td.m');if(t)setSel({model:t.dataset.m,speed:+t.dataset.s,dir:t.dataset.d});};}
function renderRuns(){const rows=[...st.results].reverse();let h='<tr><th>time</th><th>model</th><th>dir</th><th>cmd</th><th>node</th><th>s</th><th>measured rad/s</th><th>%</th><th>drops</th><th>note</th><th></th></tr>';
 for(const r of rows){const pct=r.measured_rad_s!==''?(+r.measured_rad_s/+r.speed*100).toFixed(0):'';
  h+=`<tr data-id="${r.id}"><td>${esc(r.date.slice(5))} ${esc(r.time.slice(0,5))}</td><td>${esc(r.model)}</td><td>${esc(r.direction)}</td><td class="num">${esc(r.speed)}</td>
  <td>${isClipped(r)?'clipped ':''}${wristKey(r)!==''?'wrist '+(+wristKey(r)>0?'+':'')+wristKey(r)+'° ':''}${r.dry_run==='yes'?'dry':''}</td><td class="num">${esc(fmt(r.duration_s,0))}</td>
  <td><input data-k="measured_rad_s" type="number" step="0.01" value="${esc(r.measured_rad_s)}" style="width:90px"></td><td class="num">${pct}</td>
  <td><input data-k="drops" type="number" min="0" value="${esc(r.drops)}" style="width:60px"></td>
  <td><input data-k="note" type="text" value="${esc(r.note)}"></td><td><button class="x" title="delete this run">×</button></td></tr>`;}
 $('runs').innerHTML=h;
 $('runs').onchange=async e=>{const tr=e.target.closest('tr');if(!tr||!e.target.dataset.k)return;
  await api('/api/result',{id:tr.dataset.id,[e.target.dataset.k]:e.target.value});if(st.current&&st.current.id===tr.dataset.id)shownRun=null;};
 $('runs').onclick=async e=>{if(!e.target.classList.contains('x'))return;const tr=e.target.closest('tr');
  if(confirm('Delete this run from the results? (its CSV log stays)')){try{await api('/api/delete',{id:tr.dataset.id});poll();}catch(err){alert(err.message);}}};}
init();
</script></body></html>
"""


def main():
    global PANEL
    PANEL = Panel(find_node())
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Spin panel on http://localhost:{PORT}  (results: {RESULTS}; Ctrl+C to quit)")

    def quit_(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, quit_)
    signal.signal(signal.SIGTERM, quit_)
    try:
        server.serve_forever()
    finally:
        if PANEL.stop():
            print("stopped the running policy")
        server.server_close()


if __name__ == "__main__":
    main()
