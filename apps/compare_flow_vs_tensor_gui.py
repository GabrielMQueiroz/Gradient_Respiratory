"""
Interactive Real-Time Side-by-Side Visualizer:
Optical Flow vs. Structure Tensor Eigenvalues & Eigenvectors.

Displays a live 4-panel visual dashboard comparing:
  [Top-Left]     Original Video / Moving Frame with Difference Overlay
  [Top-Right]    Farneback Dense Optical Flow (HSV Colorwheel + Quiver)
  [Bottom-Left]  Dominant Eigenvalue lambda_1 (Adaptive Edge Energy)
  [Bottom-Right] Eigenvector Field: v_1 (Normal, Red) & v_2 (Tangent, Cyan) + Coherence

Keyboard Controls:
  [SPACE] - Pause / Resume live stream
  [M]     - Toggle Display Mode: 4-Way Grid <-> Side-by-Side Dual <-> Full View
  [S]     - Save snapshot comparison to experiments/figures/snapshot_flow_vs_tensor.png
  [Q/ESC] - Quit

Usage:
  # Synthetic animated pattern (default, no webcam needed):
  python apps/compare_flow_vs_tensor_gui.py

  # Live Webcam feed:
  python apps/compare_flow_vs_tensor_gui.py --webcam 0

  # Video file:
  python apps/compare_flow_vs_tensor_gui.py --video path/to/video.mp4
"""

import sys
import argparse
import time
import numpy as np
import cv2

from sttensor import gradients, tensor, eigen, filters


def flow_to_hsv_bgr(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
    """Convert 2D flow (u, v) into BGR image using HSV color wheel."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    if max_mag is None:
        p99 = np.percentile(mag, 99)
        max_mag = p99 if p99 > 1e-4 else 1.0
    hsv[..., 2] = np.clip((mag / max_mag) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_quivers(img_bgr: np.ndarray, u: np.ndarray, v: np.ndarray, step: int = 16,
                 color=(0, 255, 255), scale: float = 2.0, min_len: float = 0.5):
    """Draw sparse 2D vector arrows on an image."""
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            dx = u[y, x] * scale
            dy = v[y, x] * scale
            length = np.sqrt(dx**2 + dy**2)
            if length >= min_len:
                pt1 = (x, y)
                pt2 = (int(np.clip(x + dx, 0, W - 1)), int(np.clip(y + dy, 0, H - 1)))
                cv2.arrowedLine(out, pt1, pt2, color, 1, tipLength=0.3)
    return out


def generate_animated_frame(t_idx: int, size: int = 256) -> np.ndarray:
    """Generate dynamic synthetic frame with translating and rotating textured disk."""
    y, x = np.mgrid[:size, :size]
    
    # Static background pattern
    bg = 128.0 + 35.0 * np.sin(2 * np.pi * x / 30.0) * np.cos(2 * np.pi * y / 30.0)
    bg += np.random.normal(0, 2, size=(size, size))
    
    # Animated object motion (circular trajectory)
    theta = 2 * np.pi * (t_idx % 120) / 120.0
    cx = size // 2 + int(40 * np.cos(theta))
    cy = size // 2 + int(25 * np.sin(theta))
    radius = 45
    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = dist <= radius
    
    # Object texture
    obj_tex = 135.0 + 50.0 * np.sin(2 * np.pi * (x - cx) / 8.0) * np.sin(2 * np.pi * (y - cy) / 8.0)
    obj_tex += np.random.normal(0, 2, size=(size, size))
    
    frame = np.where(mask, obj_tex, bg).astype(np.float32)
    return np.clip(frame, 0, 255)


def add_panel_label(img: np.ndarray, text: str, subtext: str = None, color=(0, 255, 0)):
    """Add high-contrast header banner to panel."""
    h, w = img.shape[:2]
    # Header bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if subtext:
        cv2.putText(img, subtext, (w - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def run_visualizer(source_type="synthetic", source_arg=None):
    print("=" * 65)
    print("  Optical Flow vs. Structure Tensor Eigen-Analysis Visualizer")
    print("=" * 65)
    print("  Controls:")
    print("    [SPACE]   - Pause / Resume")
    print("    [M]       - Toggle view mode (4-Way Grid / Side-by-Side Dual)")
    print("    [S]       - Save comparison snapshot")
    print("    [Q / ESC] - Quit")
    print("=" * 65)
    
    cap = None
    t_idx = 0
    size = 256
    
    if source_type == "webcam":
        cap = cv2.VideoCapture(int(source_arg))
        if not cap.isOpened():
            print(f"Error: Could not open webcam index {source_arg}")
            return
    elif source_type == "video":
        cap = cv2.VideoCapture(source_arg)
        if not cap.isOpened():
            print(f"Error: Could not open video {source_arg}")
            return

    # Read first frame
    if cap is not None:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Error: Could not read frame from source.")
            return
        frame_prev = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (size, size)).astype(np.float32)
    else:
        frame_prev = generate_animated_frame(0, size=size)

    paused = False
    view_mode = 0  # 0: 4-Way Grid, 1: Dual (Flow vs Tensor)
    fps_ema = 30.0
    
    window_name = "Side-by-Side: Optical Flow vs. Eigenvalues/Eigenvectors"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    while True:
        t_start = time.perf_counter()
        
        if not paused:
            t_idx += 1
            if cap is not None:
                ret, frame_bgr = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                frame_curr = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (size, size)).astype(np.float32)
            else:
                frame_curr = generate_animated_frame(t_idx, size=size)
        else:
            frame_curr = frame_prev.copy()

        # -------------------------------------------------------------
        # 1. OPTICAL FLOW (Farneback)
        # -------------------------------------------------------------
        flow = cv2.calcOpticalFlowFarneback(
            frame_prev.astype(np.uint8),
            frame_curr.astype(np.uint8),
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        flow_hsv_bgr = flow_to_hsv_bgr(flow)
        flow_quivers = draw_quivers(flow_hsv_bgr, flow[..., 0], flow[..., 1], step=14, color=(255, 255, 255), scale=2.5)

        # -------------------------------------------------------------
        # 2. STRUCTURE TENSOR & EIGEN-DECOMPOSITION
        # -------------------------------------------------------------
        Gx_t, Gy_t = gradients.sobel(frame_curr)
        Gx_tm1, Gy_tm1 = gradients.sobel(frame_prev)
        Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))

        T_adapt = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=1.0)
        eigvals, eigvecs = eigen.decompose_tensor(T_adapt)

        l1 = eigvals[..., 0]
        coh = eigen.coherence(eigvals)

        # Minor eigenvector v2 is contour tangent (v2x, v2y)
        v2_x = eigvecs[..., 0, 1]
        v2_y = eigvecs[..., 1, 1]
        # Major eigenvector v1 is normal to boundary (v1x, v1y)
        v1_x = eigvecs[..., 0, 0]
        v1_y = eigvecs[..., 1, 0]

        # Visualizations
        l1_norm = np.clip((l1 / (np.percentile(l1, 99) + 1e-4)) * 255, 0, 255).astype(np.uint8)
        l1_colormap = cv2.applyColorMap(l1_norm, cv2.COLORMAP_VIRIDIS)

        coh_norm = np.clip(coh * 255, 0, 255).astype(np.uint8)
        coh_colormap = cv2.applyColorMap(coh_norm, cv2.COLORMAP_MAGMA)

        # Eigenvector overlay on coherence map
        eig_display = coh_colormap.copy()
        # Draw major eigenvector v1 (Red = boundary normal)
        eig_display = draw_quivers(eig_display, v1_x, v1_y, step=14, color=(0, 0, 255), scale=6.0, min_len=0.1)
        # Draw minor eigenvector v2 (Cyan = contour tangent)
        eig_display = draw_quivers(eig_display, v2_x, v2_y, step=14, color=(255, 255, 0), scale=6.0, min_len=0.1)

        # Panel 1: Original + Temporal difference
        raw_diff = np.abs(frame_curr - frame_prev)
        diff_norm = np.clip((raw_diff / (np.percentile(raw_diff, 99) + 1e-4)) * 255, 0, 255).astype(np.uint8)
        diff_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
        input_bgr = cv2.cvtColor(frame_curr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        p1 = cv2.addWeighted(input_bgr, 0.6, diff_bgr, 0.4, 0)

        # Panel 2: Optical Flow
        p2 = flow_quivers

        # Panel 3: Dominant Eigenvalue lambda1
        p3 = l1_colormap

        # Panel 4: Eigenvector Field (Normal v1 & Tangent v2)
        p4 = eig_display

        # Annotate panels
        p1 = add_panel_label(p1, "1. Input & Raw Diff |It - It-1|", "Temporal change", (255, 255, 255))
        p2 = add_panel_label(p2, "2. Optical Flow (Farneback)", "Smeared field (u, v)", (0, 165, 255))
        p3 = add_panel_label(p3, "3. Dominant Eigenvalue lambda_1", "Sharp edge energy", (0, 255, 128))
        p4 = add_panel_label(p4, "4. Eigenvectors: v1(Red) & v2(Cyan)", "Coherence C heatmap", (255, 255, 0))

        # Compose Layout
        if view_mode == 0:
            top_row = np.hstack([p1, p2])
            bot_row = np.hstack([p3, p4])
            display_grid = np.vstack([top_row, bot_row])
        else:
            # Dual: Optical Flow vs Dominant Eigenvalue
            display_grid = np.hstack([p2, p3])

        # Status Bar
        t_elapsed = time.perf_counter() - t_start
        fps_instant = 1.0 / (t_elapsed + 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * fps_instant
        status_text = f"FPS: {fps_ema:.1f} | Frame: {t_idx} | Mode: {'4-Way Grid' if view_mode == 0 else 'Dual Comparison'} | [SPACE] Pause [M] Mode [S] Snapshot [Q] Quit"
        cv2.putText(display_grid, status_text, (12, display_grid.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display_grid)

        if not paused:
            frame_prev = frame_curr.copy()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('m'), ord('M')):
            view_mode = (view_mode + 1) % 2
        elif key in (ord('s'), ord('S')):
            snap_path = "experiments/figures/snapshot_flow_vs_tensor.png"
            cv2.imwrite(snap_path, display_grid)
            print(f"Saved snapshot to {snap_path}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Visualizer: Optical Flow vs. Structure Tensor Eigen-Analysis")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device index (e.g. 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    args = parser.parse_args()

    if args.webcam is not None:
        run_visualizer(source_type="webcam", source_arg=args.webcam)
    elif args.video is not None:
        run_visualizer(source_type="video", source_arg=args.video)
    else:
        run_visualizer(source_type="synthetic")


if __name__ == "__main__":
    main()
