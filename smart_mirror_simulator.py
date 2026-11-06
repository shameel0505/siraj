import cv2
import numpy as np
import os

def load_and_fit(path, target_w, target_h):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Missing image: {path}")
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_hud(frame, status_text, active_idx, thumbnails, motion_val, scan_y=None):
    h, w = frame.shape[:2]
    out = frame.copy()

    # 1. Top Smart Mirror Status Bar
    draw_rounded_rect(out, (20, 20), (w - 20, 88), (20, 20, 28), -1, 0.78)
    cv2.putText(out, "SIRAJ AI SMART MIRROR v1.0", (38, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 220, 110), 1, cv2.LINE_AA)
    cv2.putText(out, status_text, (38, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 255, 180) if scan_y is None else (100, 220, 255), 1, cv2.LINE_AA)

    # Motion stability bar on top right
    bar_w = 130
    bx1, by1 = w - bar_w - 38, 58
    cv2.rectangle(out, (bx1, by1), (bx1 + bar_w, by1 + 14), (70, 70, 80), -1)
    fill_w = int(np.clip(1.0 - (motion_val / 25.0), 0.05, 1.0) * bar_w)
    bar_col = (80, 240, 120) if fill_w > 95 else (80, 180, 255)
    cv2.rectangle(out, (bx1, by1), (bx1 + fill_w, by1 + 14), bar_col, -1)
    cv2.putText(out, "POSE LOCK STABILITY", (bx1 - 15, by1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 220), 1, cv2.LINE_AA)

    # 2. Floating Store Dress Carousel on Right Side
    tw, th = 96, 128
    start_y = 130
    for idx, (label, thumb) in enumerate(thumbnails):
        y1 = start_y + idx * (th + 38)
        x1 = w - tw - 24
        is_sel = (idx == active_idx)
        border_col = (80, 235, 255) if is_sel else (90, 90, 100)
        border_thick = 3 if is_sel else 1

        draw_rounded_rect(out, (x1 - 6, y1 - 6), (x1 + tw + 6, y1 + th + 24),
                          (30, 35, 50) if is_sel else (20, 20, 25), -1, 0.82)
        out[y1:y1+th, x1:x1+tw] = thumb
        cv2.rectangle(out, (x1 - 2, y1 - 2), (x1 + tw + 2, y1 + th + 2), border_col, border_thick)
        cv2.putText(out, label, (x1, y1 + th + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255) if is_sel else (180, 180, 180), 1, cv2.LINE_AA)

    # 3. Sci-Fi Glowing Scan Wave Line
    if scan_y is not None and 0 <= scan_y < h:
        glow = out.copy()
        y_top = max(0, scan_y - 18)
        y_bot = min(h, scan_y + 18)
        cv2.rectangle(glow, (0, y_top), (w, y_bot), (255, 230, 90), -1)
        cv2.addWeighted(glow, 0.32, out, 0.68, 0, out)
        cv2.line(out, (0, scan_y), (w, scan_y), (255, 255, 200), 3, cv2.LINE_AA)

    return out

def build_demo():
    cap = cv2.VideoCapture("assets/videos/source_video.mp4")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    h, w = frames[0].shape[:2]
    base_frame = frames[0]

    # Load clean stance try-on outputs (or fallback to standard try-on outputs)
    def pick_file(clean_p, std_p):
        return clean_p if os.path.exists(clean_p) else std_p

    navy_img = load_and_fit(pick_file("assets/tryon_navy_gown_clean.png", "assets/tryon_navy_gown.png"), w, h)
    red_img = load_and_fit(pick_file("assets/tryon_red_dress_clean.png", "assets/tryon_red_dress.png"), w, h)
    floral_img = load_and_fit(pick_file("assets/tryon_floral_set_clean.png", "assets/tryon_floral_set.png"), w, h)

    # Load thumbnails for floating carousel
    tw, th = 96, 128
    thumbnails = [
        ("1. Navy Gown", load_and_fit("assets/garments/garment6.webp", tw, th)),
        ("2. Red Dress", load_and_fit("assets/garments/garment1.jpeg", tw, th)),
        ("3. Floral Set", load_and_fit("assets/garments/garment4.webp", tw, th)),
    ]

    # 1. Build 4-Panel Comparison Board (Original + 3 Dresses)
    def add_banner(img, title, subtitle):
        canvas = img.copy()
        draw_rounded_rect(canvas, (15, h - 95), (w - 15, h - 18), (18, 18, 26), -1, 0.84)
        cv2.putText(canvas, title, (32, h - 58), cv2.FONT_HERSHEY_DUPLEX, 0.78, (100, 235, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (32, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
        return canvas

    p0 = add_banner(base_frame, "ORIGINAL VIDEO INPUT", "Mustard Long-Sleeve + Black Pants")
    p1 = add_banner(navy_img, "TRY-ON #1: SEQUIN GOWN", "Replaced Top & Pants + Sheer Mesh + Slit")
    p2 = add_banner(red_img, "TRY-ON #2: RED MINI DRESS", "Long Sleeves & Pants -> Bare Arms & Legs!")
    p3 = add_banner(floral_img, "TRY-ON #3: FLORAL 2-PIECE", "Halter Crop Top + Floral Wide Pants")

    board = np.hstack([p0, p1, p2, p3])
    cv2.imwrite("smart_mirror_comparison_board.jpg", board)
    print("Saved: smart_mirror_comparison_board.jpg")

    # 2. Build Pose Occlusion Engineering Study (Frame #41 Hands Blocking Torso vs Frame #0 Clean Stance)
    if os.path.exists("assets/tryon_floral_set_clean.png") and os.path.exists("assets/tryon_floral_set.png"):
        f41_orig = load_and_fit("assets/videos/locked_frame.jpg", w, h)
        f41_try = load_and_fit("assets/tryon_floral_set.png", w, h)
        c1 = add_banner(f41_orig, "BAD LOCK (FRAME #41)", "Hands Raised Blocking Chest")
        c2 = add_banner(f41_try, "HAND OCCLUSION ARTIFACT", "Wrists Collide With Halter Top")
        c3 = add_banner(base_frame, "SMART LOCK (FRAME #0)", "Arms Clear of Torso Box")
        c4 = add_banner(floral_img, "CLEAN SMART-LOCK OUTPUT", "Zero Hand Occlusion + Clean Waist")
        occ_board = np.hstack([c1, c2, c3, c4])
        cv2.imwrite("pose_occlusion_study.jpg", occ_board)
        print("Saved: pose_occlusion_study.jpg")

    # 3. Build Full Smart Mirror Experience Video (smart_mirror_demo.mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter("smart_mirror_demo.mp4", fourcc, 24.0, (w, h))

    # Segment A: Live Mirror Movement (Frames 25 -> 0 reversed so she moves then comes to a natural stop at Frame 0!)
    live_seq = frames[:36][::-1]
    for i, f in enumerate(live_seq):
        prev = live_seq[max(0, i - 3)]
        mot = float(np.mean(np.abs(f.astype(float) - prev.astype(float))))
        hud = draw_hud(f, "LIVE 60FPS MIRROR | Detecting Pose Stability...", 0, thumbnails, mot)
        out_vid.write(hud)

    # Segment B: Scan Wave Transitions Across All 3 Dresses
    renders = [
        (0, "AUTO-LOCK: Materializing Navy Sequin Gown...", "LOCKED: Navy Sequin Gown (QR Ready)", navy_img),
        (1, "SWIPED DRESS #2: Materializing Red Mini Dress...", "LOCKED: Red Puff-Sleeve Mini Dress", red_img),
        (2, "SWIPED DRESS #3: Materializing Floral 2-Piece...", "LOCKED: Floral Halter Jumpsuit", floral_img),
    ]

    prev_img = base_frame
    for idx, scan_msg, lock_msg, target_img in renders:
        # 28-frame (1.15s) top-to-bottom glowing Scan Wave
        scan_steps = 28
        for s in range(scan_steps):
            scan_y = int((s / float(scan_steps - 1)) * h)
            comp = prev_img.copy()
            comp[:scan_y, :] = target_img[:scan_y, :]
            hud = draw_hud(comp, scan_msg, idx, thumbnails, 1.2, scan_y=scan_y)
            out_vid.write(hud)

        # Hold locked HD reflection for 32 frames (~1.3s)
        for _ in range(32):
            hud = draw_hud(target_img, lock_msg, idx, thumbnails, 0.8, scan_y=None)
            out_vid.write(hud)

        prev_img = target_img

    out_vid.release()
    print("Saved: smart_mirror_demo.mp4")

if __name__ == "__main__":
    build_demo()
