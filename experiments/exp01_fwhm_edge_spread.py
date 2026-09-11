"""
Experiment 01: FWHM Edge Spread Analysis

This script demonstrates that adaptive and directional filtering
preserve edge localization while fixed Gaussian smoothing smears edges.
It measures the Full Width at Half Maximum (FWHM) of the major eigenvalue (λ₁)
cross-sections for different structure tensor filtering methods.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sttensor import gradients, tensor, eigen, filters
from sttensor.benchmarks import (
    generate_translating_disk,
    compute_fwhm,
    setup_publication_style,
    ensure_output_dirs
)

def main():
    setup_publication_style()
    ensure_output_dirs()

    # Configuration
    vel_x = 3.0
    size = 256
    radius = 40.0
    center_y = size // 2
    n_frames = 10
    
    rng = np.random.default_rng(42)
    frames = generate_translating_disk(
        size=size,
        radius=radius,
        velocity=(vel_x, 0.0),
        n_frames=n_frames,
        texture_freq=8.0,
        rng=rng
    )

    frame_4 = frames[4]
    frame_5 = frames[5]

    # Compute Sobel gradients
    gx_4, gy_4 = gradients.sobel(frame_4, axis='both')
    gx_5, gy_5 = gradients.sobel(frame_5, axis='both')

    # Compute gradient difference
    dx, dy = gradients.gradient_difference((gx_5, gy_5), (gx_4, gy_4))

    # Compute variants
    # (a) Fixed Gaussian σ=1.0
    T_a = tensor.structure_tensor_from_difference(dx, dy, sigma=1.0)
    
    # (b) Fixed Gaussian σ=2.0
    T_b = tensor.structure_tensor_from_difference(dx, dy, sigma=2.0)
    
    # (c) Fixed Gaussian σ=3.0
    T_c = tensor.structure_tensor_from_difference(dx, dy, sigma=3.0)
    
    # (d) Adaptive Gaussian (sigma_max=3.0, k=1.0)
    T_d = tensor.structure_tensor_from_difference(dx, dy, adaptive=True, sigma_max=3.0, k=1.0)
    
    # (e) Directional smooth (sigma=1.5)
    # Directional smooth needs eigenvectors. We first get raw tensor without smoothing, 
    # decompose it, and then apply directional_smooth.
    T_raw = tensor.structure_tensor_from_difference(dx, dy, sigma=None)
    _, eigvecs_raw = eigen.decompose_tensor(T_raw)
    T_e = filters.directional_smooth(T_raw, eigvecs_raw, sigma=1.5)

    variants = [
        (r'Fixed $\sigma=1.0$', T_a),
        (r'Fixed $\sigma=2.0$', T_b),
        (r'Fixed $\sigma=3.0$', T_c),
        ('Adaptive', T_d),
        ('Directional', T_e),
    ]

    styles = [
        {'color': 'C0', 'linestyle': '-'},
        {'color': 'C1', 'linestyle': '--'},
        {'color': 'C2', 'linestyle': ':'},
        {'color': 'C3', 'linestyle': '-.'},
        {'color': 'C4', 'linestyle': '-'},
    ]

    # Ground truth edge position for frame 5 (which is the 'current' frame when we difference 5-4)
    # The disk starts at size/2, and moves vel_x per frame.
    # At frame 5, cx = 128 + 5 * 3.0 = 143.
    # The leading edge is at cx + radius = 143 + 40 = 183.
    # We will use this to plot a vertical line. Wait, wait.
    # The edge is actually smeared between frame 4 and 5.
    cx_5 = size / 2.0 + 5 * vel_x
    edge_gt = cx_5 + radius - vel_x / 2.0  # Center of movement between 4 and 5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    results = []
    
    for (name, T), style in zip(variants, styles):
        eigvals, _ = eigen.decompose_tensor(T)
        lambda1 = eigvals[..., 0]
        
        # Extract horizontal 1D cross-section through center
        profile = lambda1[center_y, :]
        
        # We want to measure the peak near the leading edge.
        # Mask out the trailing edge (which is at cx - radius)
        profile_leading = np.zeros_like(profile)
        # Search region: [cx, end]
        start_idx = int(cx_5)
        profile_leading[start_idx:] = profile[start_idx:]
        
        fwhm = compute_fwhm(profile_leading)
        peak_lambda1 = np.max(profile_leading)
        
        results.append((name, fwhm, peak_lambda1))
        print(f"Variant: {name:20s} | FWHM: {fwhm:.2f} px | Peak lambda_1: {peak_lambda1:.2f}")

        # Plot profile centered around the edge to zoom in
        x_axis = np.arange(size)
        ax1.plot(x_axis, profile_leading, label=name, **style)

    ax1.axvline(edge_gt, color='black', linestyle='--', label='Ground Truth Edge')
    ax1.set_xlim(edge_gt - 20, edge_gt + 20)
    ax1.set_title('Cross-section of λ₁ at Leading Edge')
    ax1.set_xlabel('X coordinate (pixels)')
    ax1.set_ylabel('Magnitude (λ₁)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Bar chart
    names = [r[0] for r in results]
    fwhms = [r[1] for r in results]
    
    bars = ax2.bar(names, fwhms, color=['C0', 'C1', 'C2', 'C3', 'C4'])
    ax2.set_title('FWHM Comparison (lower is better)')
    ax2.set_ylabel('FWHM (pixels)')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.tight_layout()
    
    # Save figure and CSV
    fig_path = Path("experiments/figures/fig01_fwhm_comparison.pdf")
    csv_path = Path("experiments/results/fwhm_table.csv")
    
    fig.savefig(fig_path)
    print(f"Saved figure to {fig_path}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'fwhm_px', 'peak_lambda1'])
        for name, fwhm, peak_val in results:
            writer.writerow([name, fwhm, peak_val])
    print(f"Saved results to {csv_path}")

if __name__ == '__main__':
    main()
