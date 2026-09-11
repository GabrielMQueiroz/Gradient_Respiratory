"""
Script to programmatically generate 'notebooks/data_distribution_and_comparison.ipynb'
using nbformat.

Includes:
1. Interactive 3-Method Synchronized Video Player & Comparator:
   - [1] Optical Flow (Farnebäck)
   - [2] Fixed-Scale Structure Tensor (Gaussian smearing)
   - [3] Adaptive + Directional Gradient Difference Tensor (Ours)
   - Synchronized 1D Edge Cross-Section Profile across row y
   - Play/Pause animation controls, scrubbing, resolution selection
   - One-click export of 3-method synchronized comparison MP4 video
2. High-Visibility Parameter Sweep Explorer:
   - Interactive parameter selection: Fixed sigma, Adaptive k, Sigma max, Directional sigma
   - Visual Multiples Strip (side-by-side spatial thumbnails with shared colormap)
   - Quantitative Dual-Metric Curves (FWHM Edge Spread vs. Orientation Coherence)
   - Synchronized 1D Waterfall Profiles showing edge boundary transition from sharp to smeared
3. Statistical Distributions of the Data:
   - Histograms & KDE of lambda1 vs Flow Magnitude
   - Coherence C density distribution
   - Minor eigenvalue lambda2 (dispersion)
   - Polar Rose Plot of motion angles
   - Quantiles summary table
4. Full-Video Temporal Dynamics & Spectral Analysis:
   - Temporal evolution waveforms across all video frames
   - Frequency spectrum (PSD) for respiratory/periodic motion extraction
"""

import nbformat as nbf
from pathlib import Path

def build():
    nb = nbf.v4.new_notebook()
    cells = []

    # -------------------------------------------------------------
    # Cell 0: Header & Introduction (Markdown)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
r"""# Spatio-Temporal Structure Tensor: 3-Method Video Comparator & Parameter Sweeps

Welcome to the interactive exploration notebook for **Spatio-Temporal Gradient Difference Tensors**!

### Key Features in this Notebook:
1. **Synchronized 3-Method Video Player**:
   - **Method 1: Optical Flow (Farnebäck)** — Velocity field $(u, v)$, flow magnitude, and HSV direction colorwheel.
   - **Method 2: Fixed-Scale Structure Tensor** — Gaussian smoothed tensor demonstrating boundary smearing ($\sim 11\text{ px}$).
   - **Method 3: Adaptive + Directional Tensor (Ours)** — Edge-preserving $\sigma(x, y)$ scale + directional filtering along minor eigenvector $\mathbf{v}_2$.
   - **Real-Time 1D Edge Cross-Section Profile**: Directly inspect boundary sharpness frame-by-frame.
   - **Play / Pause / Scrub** live animation controls, plus one-click MP4 video export!
2. **High-Visibility Parameter Sweep Explorer**:
   - Sweep Fixed Gaussian $\sigma$, Adaptive Sensitivity $k$, Maximum Scale $\sigma_{\max}$, or Directional $\sigma_{\text{dir}}$.
   - **Visual Multiples Strip**: Spatial thumbnails side by side with identical colormaps.
   - **Quantitative Curves**: FWHM (edge spread in pixels) vs. Orientation Coherence ($C$) with Pareto-optimal guidance.
   - **Waterfall 1D Profile Plot**: Overlaid edge cross-sections revealing the exact transition from sharp to smeared.
3. **Statistical Data Distributions**:
   - Histograms & KDE of $\lambda_1$ vs. Flow Magnitude, Coherence $C \in [0, 1]$ density, and Polar Rose Plots.
4. **Full-Video Dynamics & Frequency Spectrum**:
   - Temporal waveforms and Power Spectral Density (PSD) in RPM/Hz for periodic/respiratory extraction.
"""
    ))

    # -------------------------------------------------------------
    # Cell 1: Environment Setup & Imports (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(
r"""# Setup and imports
import os
import glob
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

import ipywidgets as widgets
from IPython.display import display, clear_output

# Import sttensor package
from sttensor import gradients, tensor, eigen, filters, multiscale
from sttensor.benchmarks import setup_publication_style, compute_fwhm

# Configure publication-quality style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("sttensor library and dependencies successfully loaded!")
"""
    ))

    # -------------------------------------------------------------
    # Cell 2: Core Processing & 3-Method Engine (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(
r"""# Core Video Loading & 3-Method Processing Engine

def find_available_videos():
    # Discover all video files in the repository
    exts = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    videos = []
    for ext in exts:
        videos.extend(glob.glob(f'videos/**/{ext}', recursive=True))
        videos.extend(glob.glob(f'**/{ext}', recursive=True))
    videos = sorted(list(set(videos)))
    return videos if videos else ['videos/translating_disk_sample.mp4', 'videos/respiratory_sample.mp4']


def load_video_frames(video_path, max_frames=250, max_dim=256):
    # Load video frames and downscale if requested
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        
        h, w = gray.shape
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
            
        frames.append(gray.astype(np.float32))
        
    cap.release()
    print(f"Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]} px @ {fps:.1f} FPS) from {video_path}")
    return frames, fps


def compute_3method_analysis(frame_prev, frame_curr, fixed_sigma=2.5, adaptive_sigma_max=3.0, k=1.0, dir_sigma=1.5):
    # Compute all 3 motion analysis methods synchronously:
    # 1. Optical Flow (Farneback)
    # 2. Fixed-Scale Structure Tensor (Gaussian smearing)
    # 3. Adaptive + Directional Gradient Difference Tensor (Ours)
    H, W = frame_prev.shape
    
    # -------------------------------------------------------------
    # Method 1: Optical Flow (Farneback)
    # -------------------------------------------------------------
    flow = cv2.calcOpticalFlowFarneback(
        frame_prev.astype(np.uint8), frame_curr.astype(np.uint8),
        None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    flow_ang = np.mod(np.arctan2(flow[..., 1], flow[..., 0]), 2 * np.pi)
    
    # Optical flow HSV visualization
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = (flow_ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    p99_flow = np.percentile(flow_mag, 99) if np.percentile(flow_mag, 99) > 1e-4 else 1.0
    hsv[..., 2] = np.clip((flow_mag / p99_flow) * 255, 0, 255).astype(np.uint8)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # -------------------------------------------------------------
    # Shared Gradient Differencing: D = grad(I_t) - grad(I_t-1)
    # -------------------------------------------------------------
    Gx_t, Gy_t = gradients.sobel(frame_curr)
    Gx_tm1, Gy_tm1 = gradients.sobel(frame_prev)
    Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))
    
    # -------------------------------------------------------------
    # Method 2: Fixed-Scale Structure Tensor (Fixed Gaussian sigma)
    # -------------------------------------------------------------
    T_fixed = tensor.structure_tensor_from_difference(Dx, Dy, sigma=fixed_sigma, adaptive=False)
    eigvals_fixed, _ = eigen.decompose_tensor(T_fixed)
    lambda1_fixed = eigvals_fixed[..., 0]
    coh_fixed = eigen.coherence(eigvals_fixed)
    
    # -------------------------------------------------------------
    # Method 3: Adaptive + Directional Gradient Difference Tensor (Ours)
    # -------------------------------------------------------------
    T_adapt = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=adaptive_sigma_max, k=k)
    eigvals_adapt, eigvecs_adapt = eigen.decompose_tensor(T_adapt)
    coh_adapt = eigen.coherence(eigvals_adapt)
    
    # Directional smoothing along minor eigenvector v2 (tangent to edge)
    T_dir = filters.directional_smooth(T_adapt, eigvecs_adapt, sigma=dir_sigma)
    eigvals_dir, _ = eigen.decompose_tensor(T_dir)
    lambda1_dir = eigvals_dir[..., 0]
    
    # Eigenvectors: v1 (major, normal) and v2 (minor, tangent)
    v1_x = eigvecs_adapt[..., 0, 0]
    v1_y = eigvecs_adapt[..., 1, 0]
    v2_x = eigvecs_adapt[..., 0, 1]
    v2_y = eigvecs_adapt[..., 1, 1]
    
    return {
        "raw_diff": np.abs(frame_curr - frame_prev),
        "flow": flow,
        "flow_mag": flow_mag,
        "flow_ang": flow_ang,
        "flow_rgb": flow_rgb,
        "lambda1_fixed": lambda1_fixed,
        "coh_fixed": coh_fixed,
        "lambda1_dir": lambda1_dir,
        "lambda2_dir": eigvals_dir[..., 1],
        "coh_adapt": coh_adapt,
        "v1_x": v1_x, "v1_y": v1_y,
        "v2_x": v2_x, "v2_y": v2_y,
    }
"""
    ))

    # -------------------------------------------------------------
    # Cell 3: Interactive Synchronized 3-Method Video Player (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(
r"""# Interactive Synchronized 3-Method Video Player & Comparator

available_videos = find_available_videos()

# Control Widgets
w_video = widgets.Dropdown(options=available_videos, value=available_videos[0] if available_videos else None,
                           description='Video:', layout=widgets.Layout(width='45%'))
w_custom = widgets.Text(value='', placeholder='Or custom path (e.g. videos/plane_at_night.mp4)',
                        description='Custom:', layout=widgets.Layout(width='45%'))
w_res = widgets.Dropdown(options=[160, 256, 320, 384], value=256, description='Resolution:', layout=widgets.Layout(width='25%'))

w_frame = widgets.IntSlider(value=10, min=1, max=100, step=1, description='Frame (t):', layout=widgets.Layout(width='60%'))
w_row = widgets.IntSlider(value=128, min=10, max=240, step=1, description='Profile Row y:', layout=widgets.Layout(width='35%'))

w_fixed_sigma = widgets.FloatSlider(value=2.5, min=0.5, max=6.0, step=0.5, description='Fixed Sigma:', layout=widgets.Layout(width='30%'))
w_k = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.2, description='Adaptive k:', layout=widgets.Layout(width='30%'))
w_dir_sigma = widgets.FloatSlider(value=1.5, min=0.5, max=4.0, step=0.5, description='Dir Sigma:', layout=widgets.Layout(width='30%'))

w_quivers = widgets.Checkbox(value=True, description='Overlay Vector Quivers')
w_play = widgets.Play(value=10, min=1, max=100, step=1, interval=100, description="Press play")
widgets.jslink((w_play, 'value'), (w_frame, 'value'))

btn_load = widgets.Button(description='Load Video', button_style='primary', icon='refresh')
btn_export_mp4 = widgets.Button(description='Export 3-Method MP4 Video', button_style='success', icon='video-camera')
out_player = widgets.Output()

frames_cache = []
fps_cache = 30.0

def load_selected_video(b=None):
    global frames_cache, fps_cache
    with out_player:
        clear_output(wait=True)
        path = w_custom.value.strip() or w_video.value
        try:
            frames_cache, fps_cache = load_video_frames(path, max_frames=300, max_dim=w_res.value)
            n_max = max(len(frames_cache) - 1, 1)
            w_frame.max = n_max
            w_play.max = n_max
            w_frame.value = min(15, n_max)
            w_row.max = frames_cache[0].shape[0] - 5
            w_row.value = frames_cache[0].shape[0] // 2
            render_frame()
        except Exception as e:
            print(f"Error loading video: {e}")

btn_load.on_click(load_selected_video)

def render_frame(*args):
    global frames_cache
    if not frames_cache or len(frames_cache) < 2:
        return
    
    t = w_frame.value
    frame_prev = frames_cache[t - 1]
    frame_curr = frames_cache[t]
    row_y = min(w_row.value, frame_curr.shape[0] - 1)
    
    data = compute_3method_analysis(
        frame_prev, frame_curr,
        fixed_sigma=w_fixed_sigma.value,
        adaptive_sigma_max=3.0,
        k=w_k.value,
        dir_sigma=w_dir_sigma.value
    )
    
    with out_player:
        clear_output(wait=True)
        H, W = frame_curr.shape
        fig = plt.figure(figsize=(16, 8.5))
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.3, 1.0], hspace=0.28, wspace=0.22)
        
        # -----------------------------------------------------
        # Top Row: 3 Synchronized Methods
        # -----------------------------------------------------
        # Panel 1: Optical Flow (Farneback)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(data['flow_rgb'])
        if w_quivers.value:
            step = max(H // 16, 8)
            y_q, x_q = np.mgrid[step//2:H:step, step//2:W:step]
            u_q = data['flow'][step//2:H:step, step//2:W:step, 0]
            v_q = data['flow'][step//2:H:step, step//2:W:step, 1]
            mag_q = np.sqrt(u_q**2 + v_q**2)
            valid = mag_q > 0.25
            ax1.quiver(x_q[valid], y_q[valid], u_q[valid], v_q[valid], color='white', scale=30, headwidth=4)
        ax1.axhline(row_y, color='white', linestyle=':', alpha=0.7)
        ax1.set_title("[1] Optical Flow (Farnebäck)\nSmeared Velocity Field (u, v)", fontsize=11, fontweight='bold', color='tab:orange')
        ax1.axis('off')
        
        # Panel 2: Fixed-Scale Structure Tensor
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(data['lambda1_fixed'], cmap='viridis')
        ax2.axhline(row_y, color='white', linestyle=':', alpha=0.7)
        ax2.set_title(f"[2] Fixed-Scale Structure Tensor ($\sigma={w_fixed_sigma.value:.1f}$)\nBoundary Smearing Dilemma", fontsize=11, fontweight='bold', color='tab:green')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')
        
        # Panel 3: Adaptive + Directional Tensor (Ours)
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(data['lambda1_dir'], cmap='viridis')
        if w_quivers.value:
            step = max(H // 16, 8)
            y_q, x_q = np.mgrid[step//2:H:step, step//2:W:step]
            c_q = data['coh_adapt'][step//2:H:step, step//2:W:step]
            l1_q = data['lambda1_dir'][step//2:H:step, step//2:W:step]
            active = (c_q > 0.3) & (l1_q > 0.05 * data['lambda1_dir'].max())
            
            v1x = data['v1_x'][step//2:H:step, step//2:W:step]
            v1y = data['v1_y'][step//2:H:step, step//2:W:step]
            v2x = data['v2_x'][step//2:H:step, step//2:W:step]
            v2y = data['v2_y'][step//2:H:step, step//2:W:step]
            ax3.quiver(x_q[active], y_q[active], v1x[active], v1y[active], color='red', scale=22, label=r'$\mathbf{v}_1$ (Normal)')
            ax3.quiver(x_q[active], y_q[active], v2x[active], v2y[active], color='cyan', scale=22, label=r'$\mathbf{v}_2$ (Tangent)')
            ax3.legend(loc='lower right', fontsize=8, framealpha=0.8)
            
        ax3.axhline(row_y, color='white', linestyle=':', alpha=0.7)
        ax3.set_title("[3] Adaptive + Directional Tensor (Ours)\nSharp Boundary + Contour Eigenvectors", fontsize=11, fontweight='bold', color='tab:blue')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.axis('off')
        
        # -----------------------------------------------------
        # Bottom Row: Synchronized 1D Edge Cross-Section Profile
        # -----------------------------------------------------
        ax_prof = fig.add_subplot(gs[1, :])
        x_axis = np.arange(W)
        
        p_flow = data['flow_mag'][row_y, :]
        p_fixed = data['lambda1_fixed'][row_y, :]
        p_adapt = data['lambda1_dir'][row_y, :]
        p_diff = data['raw_diff'][row_y, :]
        
        def norm(s):
            mx = np.max(s)
            return s / mx if mx > 1e-6 else s
            
        ax_prof.plot(x_axis, norm(p_diff), ':', color='gray', alpha=0.6, label='Raw Diff $|I_t - I_{t-1}|$ (Noisy)')
        ax_prof.plot(x_axis, norm(p_flow), '-', color='tab:orange', linewidth=2.2, label='[1] Farnebäck Flow (Smeared boundary halo)')
        ax_prof.plot(x_axis, norm(p_fixed), '--', color='tab:green', linewidth=2.0, label=f'[2] Fixed Tensor $\sigma={w_fixed_sigma.value:.1f}$ (Blurred edge)')
        ax_prof.plot(x_axis, norm(p_adapt), '-', color='tab:blue', linewidth=2.4, label='[3] Adaptive + Directional (Ours: Sharp confined edge)')
        
        ax_prof.set_xlabel('Spatial X Coordinate (pixels)', fontsize=11)
        ax_prof.set_ylabel('Normalized Amplitude', fontsize=11)
        ax_prof.set_title(f"Synchronized 1D Moving Edge Profile Comparison at row $y = {row_y}$ (Frame {t})", fontsize=12, fontweight='bold')
        ax_prof.set_ylim(-0.05, 1.08)
        ax_prof.grid(True, linestyle='--', alpha=0.5)
        ax_prof.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()

# Export video button listener
def on_export_clicked(b):
    from experiments.generate_3method_comparison_video import render_comparison_video
    path = w_custom.value.strip() or w_video.value
    out_name = f"videos/comparison_3_methods_{Path(path).stem}.mp4"
    print(f"Exporting synchronized comparison video to {out_name}...")
    render_comparison_video(path, out_name, max_dim=w_res.value, fixed_sigma=w_fixed_sigma.value,
                            adaptive_sigma_max=3.0, k=w_k.value, dir_sigma=w_dir_sigma.value)
    print("Done! Video saved to:", out_name)

btn_export_mp4.on_click(on_export_clicked)

# Observers
w_frame.observe(render_frame, names='value')
w_row.observe(render_frame, names='value')
w_fixed_sigma.observe(render_frame, names='value')
w_k.observe(render_frame, names='value')
w_dir_sigma.observe(render_frame, names='value')
w_quivers.observe(render_frame, names='value')

# Layout
ui_player = widgets.VBox([
    widgets.HBox([w_video, w_custom, btn_load]),
    widgets.HBox([w_res, w_play, w_frame]),
    widgets.HBox([w_row, w_fixed_sigma, w_k, w_dir_sigma]),
    widgets.HBox([w_quivers, btn_export_mp4]),
    out_player
])

display(ui_player)
load_selected_video()
"""
    ))

    # -------------------------------------------------------------
    # Cell 4: High-Visibility Parameter Sweep Explorer (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(
r"""# High-Visibility Parameter Sweep Explorer

w_sweep_param = widgets.Dropdown(
    options=[
        ('Fixed Gaussian Sigma (smearing sweep)', 'fixed_sigma'),
        ('Adaptive Sensitivity k (scale gating sweep)', 'adaptive_k'),
        ('Directional Smoothing Sigma (tangent diffusion)', 'dir_sigma')
    ],
    value='fixed_sigma',
    description='Sweep Param:',
    layout=widgets.Layout(width='55%')
)

btn_run_sweep = widgets.Button(description='Run Parameter Sweep', button_style='info', icon='sliders')
out_sweep = widgets.Output()

def execute_parameter_sweep(b=None):
    global frames_cache
    if not frames_cache or len(frames_cache) < 2:
        print("Please load a video first in Cell 3!")
        return
        
    t = w_frame.value
    frame_prev = frames_cache[t - 1]
    frame_curr = frames_cache[t]
    center_y = w_row.value
    
    param_type = w_sweep_param.value
    
    # Gradient differences
    Gx_t, Gy_t = gradients.sobel(frame_curr)
    Gx_tm1, Gy_tm1 = gradients.sobel(frame_prev)
    Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))
    
    if param_type == 'fixed_sigma':
        values = [0.5, 1.0, 2.0, 3.0, 4.5, 6.0]
        titles = [f"Fixed $\sigma={v:.1f}$" for v in values]
        param_label = r"Fixed Gaussian Scale $\sigma$ (px)"
    elif param_type == 'adaptive_k':
        values = [0.1, 0.3, 0.8, 1.5, 3.0, 6.0]
        titles = [f"Adaptive $k={v:.1f}$" for v in values]
        param_label = r"Adaptive Sensitivity $k$"
    else:
        values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.5]
        titles = [f"Dir $\sigma_{{dir}}={v:.1f}$" for v in values]
        param_label = r"Directional Scale $\sigma_{dir}$ along $\mathbf{v}_2$"
        
    maps = []
    fwhms = []
    coherences = []
    profiles = []
    
    for v in values:
        if param_type == 'fixed_sigma':
            T = tensor.structure_tensor_from_difference(Dx, Dy, sigma=v, adaptive=False)
            eigvals, _ = eigen.decompose_tensor(T)
            l1 = eigvals[..., 0]
            coh = eigen.coherence(eigvals)
        elif param_type == 'adaptive_k':
            T = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.5, k=v)
            eigvals, _ = eigen.decompose_tensor(T)
            l1 = eigvals[..., 0]
            coh = eigen.coherence(eigvals)
        else:
            T = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=1.0)
            eigvals, eigvecs = eigen.decompose_tensor(T)
            T_dir = filters.directional_smooth(T, eigvecs, sigma=v)
            eigvals_dir, _ = eigen.decompose_tensor(T_dir)
            l1 = eigvals_dir[..., 0]
            coh = eigen.coherence(eigvals_dir)
            
        prof = l1[center_y, :]
        fwhms.append(compute_fwhm(prof))
        mask = l1 > 0.1 * l1.max()
        coherences.append(float(np.mean(coh[mask])) if np.any(mask) else 0.0)
        maps.append(l1)
        profiles.append(prof)
        
    with out_sweep:
        clear_output(wait=True)
        fig = plt.figure(figsize=(17, 10))
        gs = gridspec.GridSpec(3, len(values), figure=fig, height_ratios=[1.3, 1.0, 1.0], hspace=0.36, wspace=0.22)
        
        # -----------------------------------------------------
        # Row 1: High-Visibility Multiples Strip (Spatial Maps)
        # -----------------------------------------------------
        vmax = np.percentile(maps[1], 99.5) if len(maps) > 1 else 1.0
        for i, (m, t_str) in enumerate(zip(maps, titles)):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(m, cmap='viridis', vmax=vmax)
            ax.set_title(f"{t_str}\nFWHM: {fwhms[i]:.1f} px", fontsize=10, fontweight='bold')
            ax.axhline(center_y, color='red', linestyle=':', alpha=0.5)
            ax.axis('off')
            
        # -----------------------------------------------------
        # Row 2: Metric Progression Curves
        # -----------------------------------------------------
        ax_fwhm = fig.add_subplot(gs[1, :len(values)//2])
        ax_fwhm.plot(values, fwhms, 'o-', color='tab:blue', linewidth=2.2, markersize=8)
        ax_fwhm.set_xlabel(param_label, fontsize=11)
        ax_fwhm.set_ylabel('Edge Spread FWHM (pixels)', fontsize=11)
        ax_fwhm.set_title("Edge Localization Sharpness (Lower = Sharper)", fontsize=11, fontweight='bold')
        ax_fwhm.grid(True, linestyle='--', alpha=0.5)
        
        ax_coh = fig.add_subplot(gs[1, len(values)//2:])
        ax_coh.plot(values, coherences, 's-', color='tab:purple', linewidth=2.2, markersize=8)
        ax_coh.set_xlabel(param_label, fontsize=11)
        ax_coh.set_ylabel('Orientation Coherence ($C$)', fontsize=11)
        ax_coh.set_ylim(0, 1.05)
        ax_coh.set_title("Orientation Coherence Response (Higher = Anisotropic)", fontsize=11, fontweight='bold')
        ax_coh.grid(True, linestyle='--', alpha=0.5)
        
        # -----------------------------------------------------
        # Row 3: 1D Edge Cross-Section Waterfall Overlay
        # -----------------------------------------------------
        ax_waterfall = fig.add_subplot(gs[2, :])
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(values)))
        x_axis = np.arange(len(profiles[0]))
        
        for i, (p, col, val) in enumerate(zip(profiles, colors, values)):
            p_norm = p / (np.max(p) + 1e-6)
            ax_waterfall.plot(x_axis, p_norm, color=col, linewidth=2.0, label=f'{val:.1f}')
            
        ax_waterfall.set_xlabel('Spatial X Coordinate (pixels)', fontsize=11)
        ax_waterfall.set_ylabel('Normalized Response', fontsize=11)
        ax_waterfall.set_title(f"1D Edge Waterfall Profile across Sweep at row y={center_y}", fontsize=11, fontweight='bold')
        ax_waterfall.grid(True, linestyle='--', alpha=0.5)
        ax_waterfall.legend(title=param_label, loc='upper right', ncol=len(values), fontsize=9)
        
        plt.tight_layout()
        plt.show()

btn_run_sweep.on_click(execute_parameter_sweep)
ui_sweep = widgets.VBox([
    widgets.HBox([w_sweep_param, btn_run_sweep]),
    out_sweep
])
display(ui_sweep)
"""
    ))

    # -------------------------------------------------------------
    # Cell 5: Statistical Data Distributions (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(
r"""# Statistical Data Distribution Analysis

btn_dist = widgets.Button(description='Plot Data Distributions', button_style='warning', icon='bar-chart')
out_dist = widgets.Output()

def plot_data_distributions(b=None):
    global frames_cache
    if not frames_cache or len(frames_cache) < 2:
        print("Please load a video first in Cell 3!")
        return
        
    t = w_frame.value
    data = compute_3method_analysis(frames_cache[t-1], frames_cache[t])
    
    with out_dist:
        clear_output(wait=True)
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)
        
        l1_adapt = data['lambda1_dir'].flatten()
        l1_fixed = data['lambda1_fixed'].flatten()
        coh_flat = data['coh_adapt'].flatten()
        flow_mag_flat = data['flow_mag'].flatten()
        
        # 1. lambda1 Comparison: Adaptive vs. Fixed (Log Scale)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(l1_adapt[l1_adapt > 0], bins=40, kde=True, ax=ax1, color='tab:blue', log_scale=(True, False), label='Adaptive (Ours)')
        sns.histplot(l1_fixed[l1_fixed > 0], bins=40, kde=True, ax=ax1, color='tab:green', log_scale=(True, False), alpha=0.5, label='Fixed Scale')
        ax1.set_title(r"Dominant Eigenvalue $\lambda_1$ Distribution (Log Scale)", fontsize=11, fontweight='bold')
        ax1.set_xlabel(r"$\lambda_1$ (Motion Edge Energy)")
        ax1.set_ylabel("Pixel Count")
        ax1.legend()
        
        # 2. Optical Flow Magnitude Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(flow_mag_flat[flow_mag_flat > 0.05], bins=40, kde=True, ax=ax2, color='tab:orange')
        ax2.set_title(r"Optical Flow Magnitude $\|\mathbf{v}\|$ Distribution", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Velocity (px/frame)")
        ax2.set_ylabel("Pixel Count")
        
        # 3. Orientation Coherence C Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        sns.histplot(coh_flat, bins=40, kde=True, ax=ax3, color='tab:purple', stat='density')
        ax3.set_xlim(0, 1.0)
        ax3.set_title(r"Orientation Coherence $C = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2}$ Density", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Coherence $C$ (0=noise, 1=sharp edge)")
        ax3.set_ylabel("Density")
        
        # 4. Scatter: lambda1 vs Flow Magnitude
        ax4 = fig.add_subplot(gs[1, 0])
        idx = np.random.choice(len(l1_adapt), size=min(3000, len(l1_adapt)), replace=False)
        ax4.scatter(flow_mag_flat[idx], l1_adapt[idx], alpha=0.35, s=12, color='tab:blue')
        ax4.set_xlabel(r"Optical Flow Magnitude $\|\mathbf{v}\|$")
        ax4.set_ylabel(r"Adaptive $\lambda_1$")
        ax4.set_yscale('log')
        ax4.set_title(r"Correlation: $\lambda_1$ vs. Flow Magnitude", fontsize=11, fontweight='bold')
        
        # 5. Dispersion / Minor Eigenvalue lambda2
        ax5 = fig.add_subplot(gs[1, 1])
        l2_flat = data['lambda2_dir'].flatten()
        sns.histplot(l2_flat[l2_flat > 0], bins=40, kde=True, ax=ax5, color='tab:red', log_scale=(True, False))
        ax5.set_title(r"Minor Eigenvalue $\lambda_2$ (Isotropic Dispersion)", fontsize=11, fontweight='bold')
        ax5.set_xlabel(r"$\lambda_2$")
        ax5.set_ylabel("Pixel Count")
        
        # 6. Polar Rose Plot: Motion Angle
        ax6 = fig.add_subplot(gs[1, 2], projection='polar')
        valid_angles = data['flow_ang'][data['flow_mag'] > 0.4]
        if len(valid_angles) > 0:
            counts, bin_edges = np.histogram(valid_angles, bins=36, range=(0, 2*np.pi))
            widths = np.diff(bin_edges)
            ax6.bar(bin_edges[:-1], counts, width=widths, color='tab:orange', alpha=0.7, edgecolor='black')
        ax6.set_title("Polar Motion Angles", fontsize=11, fontweight='bold', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Quantiles Table
        df_stats = pd.DataFrame({
            "Adaptive $\lambda_1$ (Ours)": pd.Series(l1_adapt).describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
            "Fixed $\sigma$ $\lambda_1$": pd.Series(l1_fixed).describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
            "Orientation Coherence $C$": pd.Series(coh_flat).describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
            "Flow Magnitude $\|\mathbf{v}\|$": pd.Series(flow_mag_flat).describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
        })
        display(df_stats.round(4))

btn_dist.on_click(plot_data_distributions)
display(widgets.VBox([btn_dist, out_dist]))
"""
    ))

    # -------------------------------------------------------------
    # Cell 6: Full-Video Temporal Evolution & Dynamics (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(
r"""# Full-Video Temporal Waveforms & Motion Spectrum

btn_temp = widgets.Button(description='Analyze Full Video Dynamics', button_style='success', icon='area-chart')
out_temp = widgets.Output()

def run_temporal_analysis(b=None):
    global frames_cache, fps_cache
    if not frames_cache or len(frames_cache) < 4:
        print("Please load a video first in Cell 3!")
        return
        
    with out_temp:
        clear_output(wait=True)
        n = len(frames_cache)
        t_axis = np.arange(1, n) / float(fps_cache)
        
        m_l1_adapt, p95_l1_adapt = [], []
        m_l1_fixed = []
        m_flow = []
        m_coh = []
        
        print(f"Tracking motion dynamics across all {n} frames...")
        for i in range(1, n):
            Gx_t, Gy_t = gradients.sobel(frames_cache[i])
            Gx_tm1, Gy_tm1 = gradients.sobel(frames_cache[i - 1])
            Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))
            
            # Adaptive
            T_a = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=1.0)
            eig_a, _ = eigen.decompose_tensor(T_a)
            l1_a = eig_a[..., 0]
            m_l1_adapt.append(np.mean(l1_a))
            p95_l1_adapt.append(np.percentile(l1_a, 95))
            m_coh.append(np.mean(eigen.coherence(eig_a)))
            
            # Fixed
            T_f = tensor.structure_tensor_from_difference(Dx, Dy, sigma=2.5, adaptive=False)
            eig_f, _ = eigen.decompose_tensor(T_f)
            m_l1_fixed.append(np.mean(eig_f[..., 0]))
            
            # Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                frames_cache[i-1].astype(np.uint8), frames_cache[i].astype(np.uint8),
                None, 0.5, 2, 11, 2, 5, 1.1, 0
            )
            m_flow.append(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
            
        fig, axes = plt.subplots(3, 1, figsize=(15, 8.5), sharex=True)
        
        # 1. lambda1 Waveforms
        axes[0].plot(t_axis, m_l1_adapt, color='tab:blue', linewidth=2.0, label=r'Mean Adaptive $\lambda_1(t)$')
        axes[0].plot(t_axis, p95_l1_adapt, color='tab:purple', linestyle='--', label=r'95th Percentile $\lambda_1(t)$')
        axes[0].plot(t_axis, m_l1_fixed, color='tab:green', alpha=0.7, label=r'Mean Fixed $\lambda_1(t)$')
        axes[0].set_ylabel(r"Eigenvalue $\lambda_1$")
        axes[0].set_title("Temporal Evolution of Spatio-Temporal Motion Energy Across Video", fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right')
        
        # 2. Coherence
        axes[1].plot(t_axis, m_coh, color='tab:purple', linewidth=2.0, label='Mean Orientation Coherence $C(t)$')
        axes[1].set_ylabel("Coherence $C$")
        axes[1].set_ylim(0, 1.0)
        axes[1].legend(loc='upper right')
        
        # 3. Flow Magnitude
        axes[2].plot(t_axis, m_flow, color='tab:orange', linewidth=2.0, label=r'Mean Flow Magnitude $\|\mathbf{v}\|(t)$')
        axes[2].set_xlabel("Time (seconds)", fontsize=11)
        axes[2].set_ylabel("Velocity (px/frame)", fontsize=11)
        axes[2].legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Power Spectral Density (PSD)
        freqs = np.fft.rfftfreq(len(m_l1_adapt), d=(1.0 / fps_cache))
        psd_l1 = np.abs(np.fft.rfft(m_l1_adapt - np.mean(m_l1_adapt)))**2
        psd_flow = np.abs(np.fft.rfft(m_flow - np.mean(m_flow)))**2
        
        rpm_axis = freqs * 60.0
        valid = (rpm_axis >= 4.0) & (rpm_axis <= 50.0)
        
        if np.any(valid):
            fig_psd, ax_psd = plt.subplots(figsize=(12, 4))
            ax_psd.plot(rpm_axis[valid], psd_l1[valid] / (psd_l1[valid].max() + 1e-6), color='tab:blue', linewidth=2.0, label=r'$\lambda_1$ Spectrum')
            ax_psd.plot(rpm_axis[valid], psd_flow[valid] / (psd_flow[valid].max() + 1e-6), color='tab:orange', linestyle='--', label='Optical Flow Spectrum')
            
            peak_rpm = rpm_axis[valid][np.argmax(psd_l1[valid])]
            ax_psd.axvline(peak_rpm, color='red', linestyle=':', label=f'Peak Frequency: {peak_rpm:.1f} RPM')
            ax_psd.set_xlabel("Frequency (Cycles / Breaths per Minute - RPM)", fontsize=11)
            ax_psd.set_ylabel("Normalized Power", fontsize=11)
            ax_psd.set_title("Frequency Domain Spectrum (Respiratory & Periodic Motion Extraction)", fontsize=12, fontweight='bold')
            ax_psd.legend()
            plt.tight_layout()
            plt.show()

btn_temp.on_click(run_temporal_analysis)
display(widgets.VBox([btn_temp, out_temp]))
"""
    ))

    nb.cells = cells
    out_file = Path("notebooks/data_distribution_and_comparison.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook: {out_file}")

    # Copy to root as well
    root_file = Path("data_distribution_and_comparison.ipynb")
    with open(root_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully copied to root: {root_file}")


if __name__ == "__main__":
    build()
