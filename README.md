# Siraj AI Smart Mirror — Virtual Try-On Research & Retail Studio

Complete research blueprint, benchmark study, and live webcam / pre-recorded video Smart Mirror prototype.

## Quick Start on MacBook (With Built-In FaceTime Camera)

```bash
# 1. Clone the repository and enter the folder
cd siraj

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the Live Smart Mirror Web Studio
python smart_mirror_web_app.py
```

Then open **`http://localhost:8090`** in Safari or Chrome on your MacBook:
1. Click **🎥 Start MacBook Camera** to see your live 60 FPS mirror feed.
2. Pick any garment from the **Store Garment Catalog** on the right.
3. Click **✨ Magic Lock Now** (or toggle **Auto-Lock: ON** and stand still for 1 second) — the glowing Scan Wave will sweep down the mirror while the cloud **NVIDIA A10G (24GB)** GPU renders the outfit onto your reflection!

---

## Included Research & Study Documents
- [`VTON_Smart_Mirror_Master_Study.md`](VTON_Smart_Mirror_Master_Study.md) — Complete encyclopedia of open-source VTON models (`CatVTON`, `Leffa`, `IDM-VTON`, `ViViD`, `CatV2TON`, `FASHN-VTON-1.5`), preprocessing pipelines (`Meta Sapiens`, `DWpose`, `DensePose`), datasets, legal licensing traps, and retail store BOM.
- [`Live_Mirror_Magic_Lock_Deep_Dive.md`](Live_Mirror_Magic_Lock_Deep_Dive.md) — Millisecond-by-millisecond engineering breakdown of the Dual-Loop Smart Mirror architecture.
- [`smart_mirror_simulator.py`](smart_mirror_simulator.py) — Offline simulator that runs the pipeline on `assets/videos/source_video.mp4` and generates `smart_mirror_demo.mp4`, `smart_mirror_comparison_board.jpg`, and `pose_occlusion_study.jpg`.
