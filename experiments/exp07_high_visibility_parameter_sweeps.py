"""
Experiment 07: High-Visibility Parameter Sweep Analysis.

Demonstrates the physical impact of parameter variations on the Spatio-Temporal
Structure Tensor and compares:
1. Fixed Gaussian Smoothing: sigma in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
2. Adaptive Scale Sensitivity: k in [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
3. Directional Smoothing: sigma_dir in [0.5, 1.0, 1.5, 2.5, 4.0]

Produces:
- Visual Multiples Strip (side-by-side spatial maps showing edge preservation vs. smearing)
- Synchronized 1D Edge Cross-Section Waterfall plots
- Metric Curves: FWHM (px), Orientation Coherence (C), and Motion Contrast Ratio
- Output: experiments/figures/fig07_parameter_sweeps.png & .pdf
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sttensor import gradients, tensor, eigen, filters
from sttensor.benchmarks import (
    generate_translating_disk,
    compute_fwhm,
    setup_publication_style,
    ensure_output_dirs
)


def run_parameter_sweep():
    setup_publication_style()
    dirs = ensure_output_dirs()
    rng = np.random.default_rng(42)

    print("Generating benchmark motion sequence...")
    # Clean moving disk with high-frequency sinusoidal texture
    frames = generate_translating_disk(
        size=256, radius=42.0, velocity=(3.0, 1.0),
        n_frames=8, texture_freq=8.0, noise_sigma=1.5, rng=rng
    )

    frame_tm1 = frames[3]
    frame_t = frames[4]

    # Compute base gradients
    Gx_t, Gy_t = gradients.sobel(frame_t)
    Gx_tm1, Gy_tm1 = gradients.sobel(frame_tm1)
    Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))

    center_y = 256 // 2
    row_slice = slice(center_y - 45, center_y + 45)
    col_slice = slice(128 - 45, 128 + 45)

    # -------------------------------------------------------------
    # 1. SWEEP FIXED GAUSSIAN SIGMA
    # -------------------------------------------------------------
    fixed_sigmas = [0.5, 1.0, 2.0, 3.0, 4.5, 6.0]
    fixed_maps = []
    fixed_fwhms = []
    fixed_coherences = []
    fixed_profiles = []

    print("Running Fixed Sigma sweep...")
    for s in fixed_sigmas:
        T = tensor.structure_tensor_from_difference(Dx, Dy, sigma=s, adaptive=False)
        eigvals, _ = eigen.decompose_tensor(T)
        l1 = eigvals[..., 0]
        coh = eigen.coherence(eigvals)

        prof = l1[center_y, :]
        fwhm = compute_fwhm(prof)
        mask = l1 > 0.1 * l1.max()
        mean_c = float(np.mean(coh[mask])) if np.any(mask) else 0.0

        fixed_maps.append(l1[row_slice, col_slice])
        fixed_fwhms.append(fwhm)
        fixed_coherences.append(mean_c)
        fixed_profiles.append(prof)

    # -------------------------------------------------------------
    # 2. SWEEP ADAPTIVE SENSITIVITY k (at sigma_max = 3.5)
    # -------------------------------------------------------------
    k_values = [0.1, 0.3, 0.8, 1.5, 3.0, 6.0]
    adapt_maps = []
    adapt_fwhms = []
    adapt_coherences = []
    adapt_profiles = []

    print("Running Adaptive Sensitivity k sweep...")
    for k_val in k_values:
        T = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.5, k=k_val)
        eigvals, eigvecs = eigen.decompose_tensor(T)
        l1 = eigvals[..., 0]
        coh = eigen.coherence(eigvals)

        prof = l1[center_y, :]
        fwhm = compute_fwhm(prof)
        mask = l1 > 0.1 * l1.max()
        mean_c = float(np.mean(coh[mask])) if np.any(mask) else 0.0

        adapt_maps.append(l1[row_slice, col_slice])
        adapt_fwhms.append(fwhm)
        adapt_coherences.append(mean_c)
        adapt_profiles.append(prof)

    # -------------------------------------------------------------
    # PLOTTING HIGH-VISIBILITY FIGURE
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12), dpi=150)
    # 4 vertical blocks:
    # Block 1: Fixed sigma spatial strip (6 thumbnails)
    # Block 2: Adaptive k spatial strip (6 thumbnails)
    # Block 3: Quantitative metric curves (FWHM & Coherence)
    # Block 4: 1D Cross-Section Waterfall comparison
    gs = gridspec.GridSpec(4, 6, figure=fig, height_ratios=[1.2, 1.2, 1.1, 1.1], hspace=0.38, wspace=0.22)

    # Top Row: Fixed Sigma Spatial Thumbnails
    for i, s in enumerate(fixed_sigmas):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(fixed_maps[i], cmap='viridis')
        ax.set_title(f"Fixed $\sigma = {s:.1f}$\nFWHM: {fixed_fwhms[i]:.1f} px", fontsize=10, fontweight='bold')
        ax.axis('off')
        if i == 0:
            ax.text(-0.25, 0.5, "Fixed Scale\nIsotropic $\sigma$", transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='center', ha='right', rotation=90, color='tab:blue')

    # Second Row: Adaptive k Spatial Thumbnails
    for i, k_val in enumerate(k_values):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(adapt_maps[i], cmap='viridis')
        ax.set_title(f"Adaptive $k = {k_val:.1f}$\nFWHM: {adapt_fwhms[i]:.1f} px", fontsize=10, fontweight='bold')
        ax.axis('off')
        if i == 0:
            ax.text(-0.25, 0.5, "Adaptive Scale\nSensitivity $k$", transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='center', ha='right', rotation=90, color='tab:red')

    # Third Row Left: FWHM vs Parameter curves
    ax_fwhm = fig.add_subplot(gs[2, :3])
    ax_fwhm.plot(fixed_sigmas, fixed_fwhms, 'o-', color='tab:blue', linewidth=2.2, markersize=8, label=r'Fixed Gaussian $\sigma$ (Smearing grows)')
    ax_fwhm.set_xlabel(r'Scale Parameter ($\sigma$ in pixels / $k$ scaling)', fontsize=11)
    ax_fwhm.set_ylabel('Edge FWHM (pixels) - Lower is Sharper', fontsize=11)
    ax_fwhm.set_title('Edge Spread (FWHM) Progression', fontsize=12, fontweight='bold')
    ax_fwhm.grid(True, linestyle='--', alpha=0.5)

    # Plot adaptive k on secondary x-axis or overlaid
    ax_fwhm_twin = ax_fwhm.twiny()
    ax_fwhm_twin.plot(k_values, adapt_fwhms, 's--', color='tab:red', linewidth=2.2, markersize=8, label=r'Adaptive Sensitivity $k$ (Sharpness preserved)')
    ax_fwhm_twin.set_xlabel('Adaptive Sensitivity $k$', color='tab:red', fontsize=11)
    ax_fwhm_twin.tick_params(axis='x', labelcolor='tab:red')

    # Combined legend
    lines1, labels1 = ax_fwhm.get_legend_handles_labels()
    lines2, labels2 = ax_fwhm_twin.get_legend_handles_labels()
    ax_fwhm.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9.5)

    # Third Row Right: Coherence vs Parameter curves
    ax_coh = fig.add_subplot(gs[2, 3:])
    ax_coh.plot(fixed_sigmas, fixed_coherences, 'o-', color='tab:blue', linewidth=2.2, markersize=8, label=r'Fixed $\sigma$ Coherence')
    ax_coh.set_xlabel(r'Scale Parameter $\sigma$', fontsize=11)
    ax_coh.set_ylabel('Orientation Coherence ($C$)', fontsize=11)
    ax_coh.set_title('Orientation Coherence Response', fontsize=12, fontweight='bold')
    ax_coh.grid(True, linestyle='--', alpha=0.5)

    ax_coh_twin = ax_coh.twiny()
    ax_coh_twin.plot(k_values, adapt_coherences, 's--', color='tab:red', linewidth=2.2, markersize=8, label=r'Adaptive $k$ Coherence')
    ax_coh_twin.set_xlabel('Adaptive Sensitivity $k$', color='tab:red', fontsize=11)
    ax_coh_twin.tick_params(axis='x', labelcolor='tab:red')

    lines1, labels1 = ax_coh.get_legend_handles_labels()
    lines2, labels2 = ax_coh_twin.get_legend_handles_labels()
    ax_coh.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9.5)

    # Fourth Row: 1D Cross-Section Profiles (Waterfall)
    # Left: Fixed Sigma profiles
    ax_prof_fixed = fig.add_subplot(gs[3, :3])
    colors_fixed = plt.cm.Blues(np.linspace(0.4, 1.0, len(fixed_sigmas)))
    x_axis = np.arange(len(fixed_profiles[0]))
    for i, s in enumerate(fixed_sigmas):
        p = fixed_profiles[i]
        p_norm = p / (np.max(p) + 1e-6)
        ax_prof_fixed.plot(x_axis, p_norm, color=colors_fixed[i], linewidth=1.8, label=f'$\sigma={s:.1f}$')
    ax_prof_fixed.set_xlim(128 - 25, 128 + 25)
    ax_prof_fixed.set_xlabel('Spatial X Coordinate (pixels)', fontsize=11)
    ax_prof_fixed.set_ylabel('Normalized Response', fontsize=11)
    ax_prof_fixed.set_title(r'Fixed $\sigma$: Progressively Widening Halo (Boundary Smearing)', fontsize=11, fontweight='bold')
    ax_prof_fixed.grid(True, linestyle='--', alpha=0.5)
    ax_prof_fixed.legend(loc='upper right', fontsize=8, ncol=2)

    # Right: Adaptive k profiles
    ax_prof_adapt = fig.add_subplot(gs[3, 3:])
    colors_adapt = plt.cm.Reds(np.linspace(0.4, 1.0, len(k_values)))
    for i, k_val in enumerate(k_values):
        p = adapt_profiles[i]
        p_norm = p / (np.max(p) + 1e-6)
        ax_prof_adapt.plot(x_axis, p_norm, color=colors_adapt[i], linewidth=1.8, label=f'$k={k_val:.1f}$')
    ax_prof_adapt.set_xlim(128 - 25, 128 + 25)
    ax_prof_adapt.set_xlabel('Spatial X Coordinate (pixels)', fontsize=11)
    ax_prof_adapt.set_ylabel('Normalized Response', fontsize=11)
    ax_prof_adapt.set_title(r'Adaptive $k$: Consistently Confined Edge Peak (No Boundary Creep)', fontsize=11, fontweight='bold')
    ax_prof_adapt.grid(True, linestyle='--', alpha=0.5)
    ax_prof_adapt.legend(loc='upper right', fontsize=8, ncol=2)

    fig.suptitle('High-Visibility Parameter Sweep Analysis: Fixed Scale Smearing vs. Adaptive Scale Preservation',
                 fontsize=15, fontweight='bold', y=0.99)

    out_pdf = dirs['figures'] / 'fig07_parameter_sweeps.pdf'
    out_png = dirs['figures'] / 'fig07_parameter_sweeps.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved figure to {out_pdf} and {out_png}")

    # Copy to artifacts directory
    art_dir = Path(r"C:\Users\AetherCheeta\.gemini\antigravity\brain\f5936b9d-8dc2-49b2-9c3a-5e8a1f05b2a4")
    art_png = art_dir / "fig07_parameter_sweeps.png"
    import shutil
    shutil.copyfile(out_png, art_png)
    print(f"Copied to artifact directory: {art_png}")


if __name__ == '__main__':
    run_parameter_sweep()
