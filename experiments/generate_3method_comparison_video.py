"""
Generate 3-Method Synchronized Comparison Video:
[1] Optical Flow (Farnebäck)
[2] Fixed-Scale Structure Tensor (Fixed-sigma smearing)
[3] Adaptive + Directional Gradient Difference Tensor (Our edge-preserving method)

Exports a side-by-side composite MP4 video comparing all three methods
frame-by-frame with synchronized colormaps, labels, and vector field overlays.

Usage:
    python experiments/generate_3method_comparison_video.py --input videos/translating_disk_sample.mp4 --output videos/comparison_3_methods_disk.mp4
    python experiments/generate_3method_comparison_video.py --input videos/respiratory_sample.mp4 --output videos/comparison_3_methods_respiratory.mp4
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2

from sttensor import gradients, tensor, eigen, filters


def flow_to_bgr(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
    """Convert 2D flow (u, v) into a BGR colorwheel representation."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    if max_mag is None:
        p99 = np.percentile(mag, 99)
        max_mag = p99 if p99 > 1e-4 else 1.0
    hsv[..., 2] = np.clip((mag / max_mag) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_quivers_bgr(img: np.ndarray, u: np.ndarray, v: np.ndarray, step: int = 14,
                     color=(0, 255, 255), scale: float = 3.0, min_len: float = 0.4):
    """Draw sparse vector arrows on an image."""
    H, W = img.shape[:2]
    out = img.copy()
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            dx = u[y, x] * scale
            dy = v[y, x] * scale
            length = np.sqrt(dx**2 + dy**2)
            if length >= min_len:
                pt1 = (x, y)
                pt2 = (int(np.clip(x + dx, 0, W - 1)), int(np.clip(y + dy, 0, H - 1)))
                cv2.arrowedLine(out, pt1, pt2, color, 1, tipLength=0.35)
    return out


def normalize_to_colormap(arr: np.ndarray, colormap=cv2.COLORMAP_VIRIDIS, max_val: float = None) -> np.ndarray:
    """Normalize 2D array to 0-255 and apply an OpenCV colormap."""
    if max_val is None:
        p99 = np.percentile(arr, 99)
        max_val = p99 if p99 > 1e-6 else 1.0
    arr_norm = np.clip((arr / max_val) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(arr_norm, colormap)


def render_comparison_video(input_video_path: str, output_video_path: str,
                            max_dim: int = 320, fps_override: float = None,
                            fixed_sigma: float = 2.5, adaptive_sigma_max: float = 3.0,
                            k: float = 1.0, dir_sigma: float = 1.5):
    """
    Read input video, process each consecutive frame pair with the 3 methods,
    and write a synchronized side-by-side composite video.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps_override or orig_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    ret, frame0_bgr = cap.read()
    if not ret:
        raise RuntimeError("Empty video file.")

    gray0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H_orig, W_orig = gray0.shape
    scale = max_dim / float(max(H_orig, W_orig)) if max(H_orig, W_orig) > max_dim else 1.0
    H = int(H_orig * scale)
    W = int(W_orig * scale)
    gray_prev = cv2.resize(gray0, (W, H))

    # Output dimensions: 3 panels horizontally + bottom profile panel
    panel_w = W
    panel_h = H
    header_h = 36
    profile_h = 90
    total_w = panel_w * 3
    total_h = header_h + panel_h + profile_h

    # Ensure output directory exists
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        # Fallback to AVI
        alt_path = str(Path(output_video_path).with_suffix('.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(alt_path, fourcc, fps, (total_w, total_h))
        output_video_path = alt_path

    print(f"Generating 3-Method Comparison Video:")
    print(f"  Input:  {input_video_path} ({total_frames} frames)")
    print(f"  Output: {output_video_path} ({total_w}x{total_h} @ {fps:.1f} FPS)")

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        frame_idx += 1
        gray_curr = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (W, H))

        # ---------------------------------------------------------
        # Method 1: Optical Flow (Farnebäck)
        # ---------------------------------------------------------
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev.astype(np.uint8), gray_curr.astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        panel1 = flow_to_bgr(flow)
        panel1 = draw_quivers_bgr(panel1, flow[..., 0], flow[..., 1], step=max(W // 16, 8), color=(255, 255, 255), scale=2.5)

        # ---------------------------------------------------------
        # Method 2: Fixed-Scale Structure Tensor (Gaussian smearing)
        # ---------------------------------------------------------
        Gx_t, Gy_t = gradients.sobel(gray_curr)
        Gx_tm1, Gy_tm1 = gradients.sobel(gray_prev)
        Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))

        T_fixed = tensor.structure_tensor_from_difference(Dx, Dy, sigma=fixed_sigma, adaptive=False)
        eigvals_fixed, _ = eigen.decompose_tensor(T_fixed)
        l1_fixed = eigvals_fixed[..., 0]
        panel2 = normalize_to_colormap(l1_fixed, cv2.COLORMAP_VIRIDIS)

        # ---------------------------------------------------------
        # Method 3: Adaptive + Directional Gradient Difference Tensor (Ours)
        # ---------------------------------------------------------
        T_adapt = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=adaptive_sigma_max, k=k)
        eigvals_adapt, eigvecs_adapt = eigen.decompose_tensor(T_adapt)
        coh = eigen.coherence(eigvals_adapt)

        # Directional smoothing along minor eigenvector v2 (tangent to contour)
        T_dir = filters.directional_smooth(T_adapt, eigvecs_adapt, sigma=dir_sigma)
        eigvals_dir, _ = eigen.decompose_tensor(T_dir)
        l1_dir = eigvals_dir[..., 0]
        panel3 = normalize_to_colormap(l1_dir, cv2.COLORMAP_VIRIDIS)

        # Overlay minor eigenvector v2 (cyan contour tangents) and v1 (red boundary normals)
        v2_x = eigvecs_adapt[..., 0, 1]
        v2_y = eigvecs_adapt[..., 1, 1]
        v1_x = eigvecs_adapt[..., 0, 0]
        v1_y = eigvecs_adapt[..., 1, 0]
        step_q = max(W // 18, 8)
        
        # Mask where motion is significant
        mask_q = (coh > 0.35) & (l1_dir > 0.05 * np.max(l1_dir))
        v2_x_masked = np.where(mask_q, v2_x, 0)
        v2_y_masked = np.where(mask_q, v2_y, 0)
        v1_x_masked = np.where(mask_q, v1_x, 0)
        v1_y_masked = np.where(mask_q, v1_y, 0)
        
        panel3 = draw_quivers_bgr(panel3, v1_x_masked, v1_y_masked, step=step_q, color=(0, 0, 255), scale=5.0, min_len=0.2)
        panel3 = draw_quivers_bgr(panel3, v2_x_masked, v2_y_masked, step=step_q, color=(255, 255, 0), scale=5.0, min_len=0.2)

        # ---------------------------------------------------------
        # Header Banners
        # ---------------------------------------------------------
        def add_banner(p, title, subtitle, bg_color=(30, 30, 30), text_color=(0, 255, 255)):
            header = np.full((header_h, panel_w, 3), bg_color, dtype=np.uint8)
            cv2.putText(header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2, cv2.LINE_AA)
            if subtitle:
                cv2.putText(header, subtitle, (panel_w - 120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            return np.vstack([header, p])

        p1_labeled = add_banner(panel1, "[1] Optical Flow (Farneback)", "Smeared Halo", text_color=(0, 165, 255))
        p2_labeled = add_banner(panel2, f"[2] Fixed Tensor (sigma={fixed_sigma})", "Blurred Edge", text_color=(0, 255, 128))
        p3_labeled = add_banner(panel3, "[3] Adaptive + Directional (Ours)", "Sharp + v1/v2", text_color=(255, 255, 0))

        # Horizontal assembly of 3 panels
        top_composite = np.hstack([p1_labeled, p2_labeled, p3_labeled])

        # ---------------------------------------------------------
        # Bottom Panel: Real-time 1D Edge Cross-Section Profile
        # ---------------------------------------------------------
        profile_canvas = np.full((profile_h, total_w, 3), (25, 25, 25), dtype=np.uint8)
        row_y = H // 2

        # Extract 1D cross-sections across row_y
        p_flow = flow_mag[row_y, :]
        p_fixed = l1_fixed[row_y, :]
        p_dir = l1_dir[row_y, :]

        # Normalize to [0, 1]
        def norm01(s):
            mx = np.max(s)
            return s / mx if mx > 1e-6 else s

        nf = norm01(p_flow)
        nx = norm01(p_fixed)
        nd = norm01(p_dir)

        # Plot waveforms on bottom canvas
        plot_h = profile_h - 26
        plot_top = 22

        # Draw grid line
        cv2.line(profile_canvas, (0, plot_top + plot_h), (total_w, plot_top + plot_h), (60, 60, 60), 1)
        cv2.line(profile_canvas, (0, plot_top + plot_h // 2), (total_w, plot_top + plot_h // 2), (45, 45, 45), 1)

        # Draw legend & title
        cv2.putText(profile_canvas, f"Synchronized 1D Edge Profile at row y={row_y} | Frame {frame_idx}/{total_frames}:",
                    (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(profile_canvas, "- - Optical Flow (Smeared)", (total_w - 480, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(profile_canvas, "-- Fixed Tensor (Blurred)", (total_w - 310, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.putText(profile_canvas, "- Adaptive+Dir (Ours)", (total_w - 145, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 2, cv2.LINE_AA)

        # Map W points to total_w
        xs = np.linspace(20, total_w - 20, W).astype(int)
        for j in range(len(xs) - 1):
            x1, x2 = xs[j], xs[j + 1]
            # Flow (Orange)
            y_f1 = int(plot_top + plot_h - nf[j] * (plot_h - 4))
            y_f2 = int(plot_top + plot_h - nf[j + 1] * (plot_h - 4))
            cv2.line(profile_canvas, (x1, y_f1), (x2, y_f2), (0, 140, 255), 1)

            # Fixed (Green)
            y_x1 = int(plot_top + plot_h - nx[j] * (plot_h - 4))
            y_x2 = int(plot_top + plot_h - nx[j + 1] * (plot_h - 4))
            cv2.line(profile_canvas, (x1, y_x1), (x2, y_x2), (0, 255, 100), 1)

            # Adaptive+Dir (Yellow)
            y_d1 = int(plot_top + plot_h - nd[j] * (plot_h - 4))
            y_d2 = int(plot_top + plot_h - nd[j + 1] * (plot_h - 4))
            cv2.line(profile_canvas, (x1, y_d1), (x2, y_d2), (0, 255, 255), 2)

        # Final composite frame
        composite = np.vstack([top_composite, profile_canvas])
        writer.write(composite)

        gray_prev = gray_curr

    cap.release()
    writer.release()
    print(f"Successfully generated comparison video: {output_video_path}")
    return output_video_path


def main():
    parser = argparse.ArgumentParser(description="Render 3-Method Synchronized Comparison Video")
    parser.add_argument("--input", type=str, default="videos/translating_disk_sample.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="videos/comparison_3_methods_disk.mp4", help="Output video path")
    parser.add_argument("--max_dim", type=int, default=300, help="Internal frame resolution")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS override")
    args = parser.parse_args()

    render_comparison_video(args.input, args.output, max_dim=args.max_dim, fps_override=args.fps)


if __name__ == "__main__":
    main()
