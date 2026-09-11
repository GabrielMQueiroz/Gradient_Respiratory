"""
Side-by-Side Comparison: Structure Tensor Eigenvalues & Eigenvectors vs. Optical Flow.

This script directly compares:
1. Optical Flow Methods:
   - Dense Farneback Optical Flow: Vector field (u, v), flow magnitude, HSV direction color-wheel.
   - Sparse Lucas-Kanade Flow: Shi-Tomasi feature tracking vectors.
2. Structure Tensor Eigen-Analysis:
   - Gradient Difference Tensor: D = grad(I_t) - grad(I_t-1)
   - Major Eigenvalue (lambda_1): Motion edge energy map
   - Minor Eigenvalue (lambda_2): Dispersion / isotropy
   - Orientation Coherence (C): Anisotropy gating in [0, 1]
   - Major Eigenvector (v_1): Normal to moving edge boundary
   - Minor Eigenvector (v_2): Tangent along moving edge contour
   - Tensor Orientation HSV: Hue = orientation angle, Sat = coherence, Val = lambda_1
3. 1D Spatial Boundary Profile:
   - Direct cross-section comparing Farneback flow magnitude vs. lambda_1 edge localization (FWHM).

Usage:
    python experiments/compare_optical_flow_vs_tensor.py
    python experiments/compare_optical_flow_vs_tensor.py --image1 path1.png --image2 path2.png
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sttensor import gradients, tensor, eigen, filters
from sttensor.benchmarks import setup_publication_style, ensure_output_dirs


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """
    Convert 2D optical flow (u, v) into an RGB image using the standard optical flow HSV colorwheel.
    Hue = direction, Saturation = 1.0, Value = normalized magnitude.
    """
    u = flow[..., 0]
    v = flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v)
    
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # OpenCV Hue is 0-179
    hsv[..., 1] = 255
    max_mag = np.percentile(mag, 99) if np.percentile(mag, 99) > 1e-4 else 1.0
    hsv[..., 2] = np.clip((mag / max_mag) * 255, 0, 255).astype(np.uint8)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def generate_benchmark_pair(size=256, radius=45, vel=(4, 2), texture_freq=8):
    """
    Generate a pair of synthetic frames with known displacement (vel)
    containing complex inner texture and background texture.
    """
    np.random.seed(42)
    y, x = np.mgrid[:size, :size]
    
    # Background texture
    bg = 128.0 + 40.0 * np.sin(2 * np.pi * x / 32.0) * np.cos(2 * np.pi * y / 32.0)
    bg += np.random.normal(0, 3, size=(size, size))
    
    # Object texture
    obj_tex = 130.0 + 55.0 * np.sin(2 * np.pi * x / texture_freq) * np.sin(2 * np.pi * y / texture_freq)
    obj_tex += np.random.normal(0, 3, size=(size, size))
    
    c0 = (size // 2, size // 2)
    c1 = (c0[0] + vel[0], c0[1] + vel[1])
    
    dist0 = np.sqrt((x - c0[0])**2 + (y - c0[1])**2)
    dist1 = np.sqrt((x - c1[0])**2 + (y - c1[1])**2)
    
    mask0 = dist0 <= radius
    mask1 = dist1 <= radius
    
    frame0 = np.where(mask0, obj_tex, bg).astype(np.float32)
    
    # Object with translated texture in frame 1
    obj_tex1 = 130.0 + 55.0 * np.sin(2 * np.pi * (x - vel[0]) / texture_freq) * np.sin(2 * np.pi * (y - vel[1]) / texture_freq)
    obj_tex1 += np.random.normal(0, 3, size=(size, size))
    frame1 = np.where(mask1, obj_tex1, bg).astype(np.float32)
    
    frame0 = np.clip(frame0, 0, 255)
    frame1 = np.clip(frame1, 0, 255)
    
    return frame0, frame1, c0, c1, radius, vel


def compute_all_comparisons(frame_prev, frame_curr):
    """
    Compute Optical Flow (Farneback + Lucas-Kanade) and
    Structure Tensor Eigen-decomposition side-by-side.
    """
    H, W = frame_prev.shape
    
    # -------------------------------------------------------------
    # 1. OPTICAL FLOW (Farneback dense)
    # -------------------------------------------------------------
    # Parameters: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2
    flow_farneback = cv2.calcOpticalFlowFarneback(
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
    flow_mag = np.sqrt(flow_farneback[..., 0]**2 + flow_farneback[..., 1]**2)
    flow_rgb = flow_to_hsv(flow_farneback)
    
    # Lucas-Kanade sparse feature tracking
    prev_uint8 = frame_prev.astype(np.uint8)
    curr_uint8 = frame_curr.astype(np.uint8)
    p0 = cv2.goodFeaturesToTrack(prev_uint8, maxCorners=120, qualityLevel=0.03, minDistance=10)
    lk_points_prev = []
    lk_points_curr = []
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_uint8, curr_uint8, p0, None, winSize=(15, 15), maxLevel=2)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        lk_points_prev = good_old
        lk_points_curr = good_new

    # -------------------------------------------------------------
    # 2. STRUCTURE TENSOR & EIGENVALUES / EIGENVECTORS
    # -------------------------------------------------------------
    # Gradients & difference of gradients
    Gx_t, Gy_t = gradients.sobel(frame_curr)
    Gx_tm1, Gy_tm1 = gradients.sobel(frame_prev)
    Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))
    
    # Adaptive Structure Tensor
    T_adapt = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=1.0)
    eigvals, eigvecs = eigen.decompose_tensor(T_adapt)
    
    lambda1 = eigvals[..., 0]
    lambda2 = eigvals[..., 1]
    coh = eigen.coherence(eigvals)
    
    # Major eigenvector v1 (normal to edge)
    v1_x = eigvecs[..., 0, 0]
    v1_y = eigvecs[..., 1, 0]
    
    # Minor eigenvector v2 (tangent along edge)
    v2_x = eigvecs[..., 0, 1]
    v2_y = eigvecs[..., 1, 1]
    
    # Orientation angle in [0, pi)
    theta_v1 = eigen.orientation(eigvecs, modulo_pi=True)
    
    # Directional smoothed tensor
    T_dir = filters.directional_smooth(T_adapt, eigvecs, sigma=1.5)
    eigvals_dir, _ = eigen.decompose_tensor(T_dir)
    lambda1_dir = eigvals_dir[..., 0]

    return {
        "flow_farneback": flow_farneback,
        "flow_mag": flow_mag,
        "flow_rgb": flow_rgb,
        "lk_prev": lk_points_prev,
        "lk_curr": lk_points_curr,
        "Dx": Dx,
        "Dy": Dy,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda1_dir": lambda1_dir,
        "coherence": coh,
        "v1_x": v1_x,
        "v1_y": v1_y,
        "v2_x": v2_x,
        "v2_y": v2_y,
        "theta_v1": theta_v1,
    }


def create_side_by_side_figure(frame_prev, frame_curr, data, out_path_pdf, out_path_png):
    """
    Generate comprehensive 8-panel comparison figure.
    """
    setup_publication_style()
    
    H, W = frame_prev.shape
    fig = plt.figure(figsize=(16.5, 12), dpi=150)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.35)
    
    # Grid for sparse quivers (step = 10)
    step = 10
    y_q, x_q = np.mgrid[step//2:H:step, step//2:W:step]
    
    # -------------------------------------------------------------
    # Panel (1, 1): Input Frames & Temporal Difference
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    raw_diff = np.abs(frame_curr - frame_prev)
    im1 = ax1.imshow(raw_diff, cmap='hot', origin='upper')
    ax1.set_title('(a) Raw Difference $|I_t - I_{t-1}|$', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis('off')
    
    # -------------------------------------------------------------
    # Panel (1, 2): Optical Flow Magnitude (Farneback)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(data["flow_mag"], cmap='inferno', origin='upper')
    ax2.set_title(r'(b) Optical Flow Magnitude $\|\mathbf{v}\|$', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    
    # -------------------------------------------------------------
    # Panel (1, 3): Optical Flow Vector Field (Farneback HSV + Quiver)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(data["flow_rgb"], origin='upper')
    # Overlay flow quivers
    u_q = data["flow_farneback"][step//2:H:step, step//2:W:step, 0]
    v_q = data["flow_farneback"][step//2:H:step, step//2:W:step, 1]
    mag_q = np.sqrt(u_q**2 + v_q**2)
    valid = mag_q > 0.5
    ax3.quiver(x_q[valid], y_q[valid], u_q[valid], v_q[valid], color='white',
               scale=40, width=0.005, headwidth=4)
    ax3.set_title('(c) Optical Flow Field (HSV + Quiver)', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # -------------------------------------------------------------
    # Panel (2, 1): Major Eigenvalue lambda_1 (Motion Edge Energy)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(data["lambda1"], cmap='viridis', origin='upper')
    ax4.set_title(r'(d) Dominant Eigenvalue $\lambda_1$ (Ours)', fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.axis('off')
    
    # -------------------------------------------------------------
    # Panel (2, 2): Orientation Coherence C
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(data["coherence"], cmap='magma', vmin=0, vmax=1, origin='upper')
    ax5.set_title(r'(e) Tensor Coherence $C = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2}$', fontsize=11, fontweight='bold')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.axis('off')
    
    # -------------------------------------------------------------
    # Panel (2, 3): Eigenvector Field v1 (Normal) & v2 (Tangent)
    # -------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(data["lambda1"], cmap='gray', origin='upper', alpha=0.4)
    
    v1x_q = data["v1_x"][step//2:H:step, step//2:W:step]
    v1y_q = data["v1_y"][step//2:H:step, step//2:W:step]
    v2x_q = data["v2_x"][step//2:H:step, step//2:W:step]
    v2y_q = data["v2_y"][step//2:H:step, step//2:W:step]
    c_q = data["coherence"][step//2:H:step, step//2:W:step]
    l1_q = data["lambda1"][step//2:H:step, step//2:W:step]
    
    # Plot where coherence and energy are active
    edge_active = (c_q > 0.4) & (l1_q > 0.1 * data["lambda1"].max())
    
    # Major eigenvector v1 (red arrows: normal to edge)
    ax6.quiver(x_q[edge_active], y_q[edge_active], v1x_q[edge_active], v1y_q[edge_active],
               color='red', scale=22, width=0.007, headwidth=3, label=r'$\mathbf{v}_1$ (Normal)')
    # Minor eigenvector v2 (cyan arrows: contour tangent)
    ax6.quiver(x_q[edge_active], y_q[edge_active], v2x_q[edge_active], v2y_q[edge_active],
               color='cyan', scale=22, width=0.005, headwidth=3, label=r'$\mathbf{v}_2$ (Tangent)')
    
    ax6.set_title(r'(f) Eigenvectors: $\mathbf{v}_1$ (Normal) vs $\mathbf{v}_2$ (Tangent)', fontsize=11, fontweight='bold')
    ax6.legend(loc='lower right', fontsize=8, framealpha=0.8)
    ax6.axis('off')
    
    # -------------------------------------------------------------
    # Panel (3, 1): Directional Smoothed lambda_1 (Sharper Localization)
    # -------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(data["lambda1_dir"], cmap='viridis', origin='upper')
    ax7.set_title(r'(g) Directional Smoothed $\lambda_1$ (Along $\mathbf{v}_2$)', fontsize=11, fontweight='bold')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    ax7.axis('off')
    
    # -------------------------------------------------------------
    # Panel (3, 2 & 3 combined): 1D Cross-Section Spatial Boundary Profile
    # -------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 1:])
    row = H // 2
    
    # Normalize profiles to [0, 1] for direct width comparison
    p_flow = data["flow_mag"][row, :]
    p_flow_norm = (p_flow - p_flow.min()) / (p_flow.max() - p_flow.min() + 1e-6)
    
    p_l1 = data["lambda1"][row, :]
    p_l1_norm = (p_l1 - p_l1.min()) / (p_l1.max() - p_l1.min() + 1e-6)
    
    p_l1_dir = data["lambda1_dir"][row, :]
    p_l1_dir_norm = (p_l1_dir - p_l1_dir.min()) / (p_l1_dir.max() - p_l1_dir.min() + 1e-6)
    
    p_diff = raw_diff[row, :]
    p_diff_norm = (p_diff - p_diff.min()) / (p_diff.max() - p_diff.min() + 1e-6)
    
    x_axis = np.arange(W)
    
    ax8.plot(x_axis, p_diff_norm, ':', color='gray', alpha=0.7, label=r'Raw Diff $|I_t - I_{t-1}|$ (Noisy)')
    ax8.plot(x_axis, p_flow_norm, '-', color='tab:orange', linewidth=2.0, label=r'Farneback Flow (Smeared boundary)')
    ax8.plot(x_axis, p_l1_norm, '--', color='tab:blue', linewidth=2.0, label=r'Adaptive Tensor $\lambda_1$ (Sharp edge)')
    ax8.plot(x_axis, p_l1_dir_norm, '-', color='tab:green', linewidth=2.2, label=r'Directional $\lambda_1$ Along $\mathbf{v}_2$ (Ultra-sharp)')
    
    # Annotate region of interest around the edge
    peak_x = np.argmax(p_l1_norm[W//2:]) + W//2
    ax8.set_xlim(peak_x - 35, peak_x + 35)
    ax8.set_ylim(-0.05, 1.1)
    ax8.set_xlabel('Spatial X Coordinate (pixels)', fontsize=11)
    ax8.set_ylabel('Normalized Response', fontsize=11)
    ax8.set_title(r'(h) 1D Moving Edge Profile: Smearing Comparison at $y = %d$' % row, fontsize=11, fontweight='bold')
    ax8.grid(True, linestyle='--', alpha=0.5)
    ax8.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    fig.suptitle('Side-by-Side Comparison: Optical Flow vs. Eigenvalues & Eigenvectors of Gradient Difference Tensor',
                 fontsize=14, fontweight='bold', y=0.98)
    
    fig.savefig(out_path_pdf, bbox_inches='tight')
    fig.savefig(out_path_png, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Generated side-by-side figure saved to:\n  {out_path_pdf}\n  {out_path_png}")


def main():
    parser = argparse.ArgumentParser(description="Side-by-side comparison of Optical Flow vs. Structure Tensor Eigen-decomposition")
    parser.add_argument("--image1", type=str, default=None, help="Path to first frame (grayscale or BGR)")
    parser.add_argument("--image2", type=str, default=None, help="Path to second frame (grayscale or BGR)")
    args = parser.parse_args()
    
    dirs = ensure_output_dirs()
    
    if args.image1 and args.image2:
        print(f"Loading custom image pair: {args.image1} and {args.image2}")
        img1 = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img2 = cv2.imread(args.image2, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        print("Generating synthetic moving textured disc pair...")
        img1, img2, c0, c1, radius, vel = generate_benchmark_pair(size=256, radius=45, vel=(4, 2), texture_freq=8)
        
    print("Computing Optical Flow and Structure Tensor Eigen-decomposition...")
    data = compute_all_comparisons(img1, img2)
    
    out_pdf = dirs['figures'] / "fig_flow_vs_tensor_comparison.pdf"
    out_png = dirs['figures'] / "fig_flow_vs_tensor_comparison.png"
    
    create_side_by_side_figure(img1, img2, data, out_pdf, out_png)
    
    # Also copy to artifacts directory for interactive UI viewing
    art_dir = Path(r"C:\Users\AetherCheeta\.gemini\antigravity\brain\f5936b9d-8dc2-49b2-9c3a-5e8a1f05b2a4")
    art_dir.mkdir(parents=True, exist_ok=True)
    art_png = art_dir / "fig_flow_vs_tensor_comparison.png"
    
    import shutil
    shutil.copyfile(out_png, art_png)
    print(f"Copied to artifact directory: {art_png}")


if __name__ == "__main__":
    main()
