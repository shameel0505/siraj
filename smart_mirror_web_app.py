"""
Interactive AI Smart Mirror Web Studio (MacBook Live Camera + Pre-Recorded Video Mode)
Run with: python smart_mirror_web_app.py
Then open: http://localhost:8090
"""
import os
import time
import shutil
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
import uvicorn

app = FastAPI(title="Siraj AI Smart Mirror Studio")

os.makedirs("assets/uploads", exist_ok=True)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

_vton_client = None

def get_vton_client():
    global _vton_client
    if _vton_client is None:
        _vton_client = Client("fashn-ai/fashn-vton-1.5")
    return _vton_client

@app.get("/board/{name}")
def get_board(name: str):
    allowed = {
        "comparison": "smart_mirror_comparison_board.jpg",
        "occlusion": "pose_occlusion_study.jpg",
        "demo_video": "smart_mirror_demo.mp4",
    }
    path = allowed.get(name)
    if path and os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.post("/api/live_tryon")
async def live_tryon(
    frame_base64: str = Form(...),
    garment_file: str = Form("garment6.webp"),
    category: str = Form("one-pieces"),
    photo_type: str = Form("model"),
):
    """Receives a live camera frame (base64 JPEG) from MacBook webcam and renders the selected dress."""
    try:
        header, encoded = frame_base64.split(",", 1) if "," in frame_base64 else ("", frame_base64)
        img_bytes = base64.b64decode(encoded)
        ts = int(time.time() * 1000)
        person_path = os.path.join("assets/uploads", f"webcam_capture_{ts}.jpg")
        with open(person_path, "wb") as f:
            f.write(img_bytes)

        g_path = os.path.join("assets/garments", garment_file)
        client = get_vton_client()
        res = client.predict(
            person_image=handle_file(person_path),
            garment_image=handle_file(g_path),
            category=category,
            garment_photo_type=photo_type,
            num_timesteps=30,
            guidance_scale=1.5,
            seed=42,
            segmentation_free=True,
            api_name="/try_on",
        )
        out_src = res["path"] if isinstance(res, dict) else res
        out_name = f"tryon_live_{ts}.png"
        out_dst = os.path.join("assets/uploads", out_name)
        shutil.copy(out_src, out_dst)
        return {"status": "ok", "result_url": f"/assets/uploads/{out_name}", "capture_url": f"/assets/uploads/webcam_capture_{ts}.jpg"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/upload_garment")
async def upload_garment(file: UploadFile = File(...)):
    safe_name = os.path.basename(file.filename.replace(" ", "_"))
    dst = os.path.join("assets/garments", safe_name)
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "ok", "garment_file": safe_name, "url": f"/assets/garments/{safe_name}"}

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Siraj AI Smart Mirror — Live Camera & Studio</title>
<style>
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; background: #0b0d14; color: #f1f5f9; }
  header { padding: 16px 28px; background: #121521; border-bottom: 1px solid #25293e; display: flex; justify-content: space-between; align-items: center; }
  h1 { margin: 0; font-size: 19px; color: #38bdf8; letter-spacing: 0.4px; }
  .badge { background: #1e293b; border: 1px solid #38bdf8; color: #7dd3fc; padding: 5px 12px; border-radius: 99px; font-size: 12px; }
  .container { display: grid; grid-template-columns: 440px 1fr; gap: 24px; padding: 24px 28px; max-width: 1440px; margin: 0 auto; }
  .mirror-frame { position: relative; width: 420px; height: 640px; border-radius: 22px; overflow: hidden; border: 3px solid #38bdf8; box-shadow: 0 0 40px rgba(56,189,248,0.22); background: #000; }
  .mirror-frame video, .mirror-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .mirror-frame video { transform: scaleX(-1); }
  #resultOverlay { position: absolute; inset: 0; display: none; z-index: 4; }
  .hud-top { position: absolute; top: 14px; left: 14px; right: 14px; background: rgba(15,18,30,0.82); backdrop-filter: blur(8px); padding: 10px 14px; border-radius: 12px; z-index: 6; border: 1px solid rgba(56,189,248,0.35); display: flex; justify-content: space-between; align-items: center; font-size: 13px; }
  .stability-bar { width: 90px; height: 8px; background: #334155; border-radius: 4px; overflow: hidden; }
  .stability-fill { height: 100%; width: 0%; background: #22c55e; transition: width 0.15s; }
  .scan-line { position: absolute; left: 0; width: 100%; height: 6px; background: #fde047; box-shadow: 0 0 28px 10px rgba(56,189,248,0.9); top: 0; display: none; z-index: 5; }
  @keyframes scanMove { 0% { top: 0%; } 100% { top: 98%; } }
  .scanning .scan-line { display: block; animation: scanMove 1.3s ease-in-out infinite alternate; }
  .controls { display: flex; gap: 10px; margin-top: 12px; width: 420px; }
  .btn { flex: 1; background: linear-gradient(135deg, #0ea5e9, #6366f1); color: #fff; border: none; padding: 11px 14px; border-radius: 10px; font-weight: 600; cursor: pointer; font-size: 13px; }
  .btn.secondary { background: #1e2438; border: 1px solid #334155; }
  .panel { background: #151826; border: 1px solid #262b42; border-radius: 16px; padding: 20px; }
  .garments { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; }
  .g-card { background: #1c2033; border: 2px solid transparent; border-radius: 10px; padding: 8px; cursor: pointer; text-align: center; transition: 0.18s; }
  .g-card.active, .g-card:hover { border-color: #38bdf8; background: #232942; }
  .g-card img { width: 100%; height: 120px; object-fit: contain; border-radius: 6px; background: #fff; }
  .g-card span { display: block; font-size: 12px; margin-top: 6px; color: #e2e8f0; font-weight: 500; }
  .g-card small { font-size: 10px; color: #94a3b8; }
  .board-img { width: 100%; border-radius: 12px; margin-top: 12px; border: 1px solid #2e334d; }
</style>
</head>
<body>
<header>
  <h1>SIRAJ AI SMART MIRROR — Live Camera & Retail Prototype</h1>
  <span class="badge">Cloud GPU: NVIDIA A10G (FASHN-VTON-1.5)</span>
</header>

<div class="container">
  <div>
    <div class="mirror-frame" id="mirrorBox">
      <div class="hud-top">
        <span id="statusText">Demo Mode (Click Start Camera for Live Mirror)</span>
        <div class="stability-bar"><div class="stability-fill" id="stabFill"></div></div>
      </div>
      <video id="webcamVid" autoplay playsinline muted style="display:none"></video>
      <img id="resultOverlay" src="/assets/tryon_navy_gown_clean.png" style="display:block" onerror="this.src='/assets/tryon_navy_gown.png'">
      <div class="scan-line"></div>
    </div>

    <div class="controls">
      <button class="btn" onclick="startCamera()">🎥 Start MacBook Camera</button>
      <button class="btn" onclick="triggerLiveTryOn()">✨ Magic Lock Now</button>
      <button class="btn secondary" id="autoLockBtn" onclick="toggleAutoLock()">Auto-Lock: OFF</button>
    </div>
    <canvas id="motionCanvas" width="160" height="240" style="display:none"></canvas>
    <canvas id="captureCanvas" style="display:none"></canvas>
  </div>

  <div class="panel">
    <h2 style="margin-top:0; font-size:18px;">1. Store Garment Catalog (Select Outfit for Live Camera or Demo)</h2>
    <div class="garments" id="garmentGrid">
      <div class="g-card active" onclick="selectGarment('garment6.webp', 'one-pieces', 'model', '/assets/tryon_navy_gown_clean.png', this)">
        <img src="/assets/garments/garment6.webp">
        <span>Navy Sequin Gown</span><small>Full Dress (Model)</small>
      </div>
      <div class="g-card" onclick="selectGarment('garment1.jpeg', 'one-pieces', 'model', '/assets/tryon_red_dress_clean.png', this)">
        <img src="/assets/garments/garment1.jpeg">
        <span>Red Puff Mini Dress</span><small>Full Dress (Model)</small>
      </div>
      <div class="g-card" onclick="selectGarment('garment4.webp', 'one-pieces', 'model', '/assets/tryon_floral_set_clean.png', this)">
        <img src="/assets/garments/garment4.webp">
        <span>Floral Halter Set</span><small>One-Piece / 2-Piece</small>
      </div>
      <div class="g-card" onclick="selectGarment('garment3.jpeg', 'tops', 'flat-lay', null, this)">
        <img src="/assets/garments/garment3.jpeg">
        <span>Navy Eyelet Crop Top</span><small>Top (Flat-Lay)</small>
      </div>
      <div class="g-card" onclick="selectGarment('00126_00.jpg', 'tops', 'flat-lay', null, this)">
        <img src="/assets/garments/00126_00.jpg">
        <span>FILA Colorblock Tee</span><small>Top (Flat-Lay)</small>
      </div>
      <div class="g-card" onclick="selectGarment('00057_00.jpg', 'tops', 'flat-lay', null, this)">
        <img src="/assets/garments/00057_00.jpg">
        <span>Lee Graphic Tee</span><small>Top (Flat-Lay)</small>
      </div>
      <div class="g-card" onclick="selectGarment('garment2.webp', 'tops', 'model', null, this)">
        <img src="/assets/garments/garment2.webp">
        <span>Rick & Morty Tee</span><small>Top (Model)</small>
      </div>
      <div class="g-card" onclick="selectGarment('garment5.jpeg', 'bottoms', 'flat-lay', null, this)">
        <img src="/assets/garments/garment5.jpeg">
        <span>White Ruffled Skirt</span><small>Bottom (Flat-Lay)</small>
      </div>
    </div>

    <h3 style="margin-top:22px; font-size:16px;">2. Pre-Rendered 4-Panel Benchmark Board</h3>
    <img class="board-img" src="/board/comparison">

    <h3 style="margin-top:22px; font-size:16px;">3. Pose Self-Occlusion Study</h3>
    <img class="board-img" src="/board/occlusion">
  </div>
</div>

<script>
let selectedGarment = { file: 'garment6.webp', category: 'one-pieces', photoType: 'model' };
let cameraActive = false;
let autoLock = false;
let isRendering = false;
let stillFrames = 0;
let prevFrameData = null;

function selectGarment(file, category, photoType, preRenderedUrl, el) {
  document.querySelectorAll('.g-card').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  selectedGarment = { file, category, photoType };

  if (!cameraActive && preRenderedUrl) {
    const box = document.getElementById('mirrorBox');
    const overlay = document.getElementById('resultOverlay');
    box.classList.add('scanning');
    document.getElementById('statusText').innerText = 'Scanning: ' + file + '...';
    setTimeout(() => {
      overlay.style.display = 'block';
      overlay.src = preRenderedUrl;
      box.classList.remove('scanning');
      document.getElementById('statusText').innerText = 'Locked: ' + file;
    }, 650);
  } else if (cameraActive) {
    document.getElementById('resultOverlay').style.display = 'none';
    document.getElementById('statusText').innerText = 'Selected ' + file + ' — Pose to Lock!';
    stillFrames = 0;
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1080 }, height: { ideal: 1920 }, facingMode: 'user' },
      audio: false
    });
    const vid = document.getElementById('webcamVid');
    vid.srcObject = stream;
    vid.style.display = 'block';
    document.getElementById('resultOverlay').style.display = 'none';
    cameraActive = true;
    document.getElementById('statusText').innerText = 'LIVE 60FPS MIRROR | Ready';
    requestAnimationFrame(monitorMotionLoop);
  } catch (err) {
    alert('Could not access camera: ' + err.message);
  }
}

function toggleAutoLock() {
  autoLock = !autoLock;
  const btn = document.getElementById('autoLockBtn');
  btn.innerText = 'Auto-Lock: ' + (autoLock ? 'ON' : 'OFF');
  btn.style.borderColor = autoLock ? '#22c55e' : '#334155';
  stillFrames = 0;
}

function monitorMotionLoop() {
  if (!cameraActive) return;
  const vid = document.getElementById('webcamVid');
  const mc = document.getElementById('motionCanvas');
  const ctx = mc.getContext('2d');
  if (vid.readyState >= 2 && !isRendering) {
    ctx.drawImage(vid, 0, 0, mc.width, mc.height);
    const curr = ctx.getImageData(0, 0, mc.width, mc.height).data;
    if (prevFrameData) {
      let diff = 0;
      for (let i = 0; i < curr.length; i += 16) {
        diff += Math.abs(curr[i] - prevFrameData[i]);
      }
      const avgDiff = diff / (curr.length / 16);
      const stability = Math.max(0, Math.min(100, (1 - avgDiff / 18) * 100));
      document.getElementById('stabFill').style.width = stability + '%';

      if (autoLock && document.getElementById('resultOverlay').style.display === 'none') {
        if (stability > 82) {
          stillFrames++;
          if (stillFrames > 28) {
            stillFrames = 0;
            triggerLiveTryOn();
          }
        } else {
          stillFrames = 0;
        }
      }
    }
    prevFrameData = curr;
  }
  requestAnimationFrame(monitorMotionLoop);
}

async function triggerLiveTryOn() {
  if (!cameraActive) {
    alert('Start your MacBook camera first, or click the garment cards in Demo Mode!');
    return;
  }
  if (isRendering) return;
  isRendering = true;

  const vid = document.getElementById('webcamVid');
  const cc = document.getElementById('captureCanvas');
  cc.width = vid.videoWidth || 720;
  cc.height = vid.videoHeight || 960;
  const ctx = cc.getContext('2d');
  // Mirror horizontally to match reflection
  ctx.translate(cc.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(vid, 0, 0, cc.width, cc.height);
  const dataUrl = cc.toDataURL('image/jpeg', 0.92);

  const box = document.getElementById('mirrorBox');
  box.classList.add('scanning');
  document.getElementById('statusText').innerText = 'AI Materializing ' + selectedGarment.file + '...';

  const form = new FormData();
  form.append('frame_base64', dataUrl);
  form.append('garment_file', selectedGarment.file);
  form.append('category', selectedGarment.category);
  form.append('photo_type', selectedGarment.photoType);

  try {
    const resp = await fetch('/api/live_tryon', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.status === 'ok') {
      const overlay = document.getElementById('resultOverlay');
      overlay.src = data.result_url;
      overlay.style.display = 'block';
      document.getElementById('statusText').innerText = 'LOCKED HD REFLECTION (Click dress to unlock)';
    } else {
      document.getElementById('statusText').innerText = 'Error: ' + data.message;
    }
  } catch (e) {
    document.getElementById('statusText').innerText = 'Network error: ' + e.message;
  } finally {
    box.classList.remove('scanning');
    isRendering = false;
  }
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090)
