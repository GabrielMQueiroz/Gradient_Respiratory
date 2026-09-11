"""
Experiment 02: Pareto Frontier of Spatio-Temporal Gradient Tensors.
Plots Edge Localization Error vs Orientation Coherence to show that adaptive
filters achieve the Pareto-optimal corner (low error, high coherence).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sttensor import gradients, tensor, eigen, filters
from sttensor.benchmarks import (
    generate_translating_disk, 
    compute_localization_error, 
    setup_publication_style, 
    ensure_output_dirs
)

def run_experiment():
    # Setup
    setup_publication_style()
    dirs = ensure_output_dirs()
    rng = np.random.default_rng(42)
    
    # Generate data
    radius = 40
    velocity = (3, 0)
    center_start = (128, 128)
    frames = generate_translating_disk(
        size=256, 
        radius=radius, 
        velocity=velocity, 
        n_frames=10, 
        texture_freq=8, 
        noise_sigma=0.0,
        rng=rng
    )
    
    frame_t = frames[5]
    frame_tm1 = frames[4]
    
    # Compute base gradients
    grad_t = gradients.sobel(frame_t)
    grad_tm1 = gradients.sobel(frame_tm1)
    Dx, Dy = gradients.gradient_difference(grad_t, grad_tm1)
    
    row_idx = 256 // 2
    
    # True edge position at frame 5 (t=5). Center started at 128, velocity is (3, 0).
    # Center at t=5 is (128 + 5*3, 128 + 5*0) = (143, 128).
    # Leading edge is at x = 143 + 40 = 183.
    true_edge_x = 128 + 5 * velocity[0] + radius
    
    # 1. Sweep fixed Gaussian sigma
    sigmas = np.linspace(0.5, 5.0, 20)
    fixed_errors = []
    fixed_coherences = []
    
    print("Evaluating fixed Gaussian sigmas:")
    for sig in sigmas:
        T = tensor.structure_tensor_from_difference(Dx, Dy, sigma=sig, adaptive=False)
        eigvals, _ = eigen.decompose_tensor(T)
        l1 = eigvals[..., 0]
        C = eigen.coherence(eigvals)
        
        # Localization error
        profile = l1[row_idx, :]
        err = compute_localization_error(profile, true_edge_x)
        
        # Mean coherence in motion region
        mask = l1 > 0.1 * l1.max()
        mean_C = C[mask].mean() if np.any(mask) else 0.0
        
        fixed_errors.append(err)
        fixed_coherences.append(mean_C)
        print(f"Sigma: {sig:.2f} | Error: {err:.4f} | Coherence: {mean_C:.4f}")
        
    # 2. Adaptive Gaussian
    T_adapt = tensor.structure_tensor_from_difference(Dx, Dy, sigma=None, adaptive=True, sigma_max=3.0, k=1.0)
    eigvals_a, _ = eigen.decompose_tensor(T_adapt)
    l1_a = eigvals_a[..., 0]
    err_a = compute_localization_error(l1_a[row_idx, :], true_edge_x)
    mask_a = l1_a > 0.1 * l1_a.max()
    C_a = eigen.coherence(eigvals_a)[mask_a].mean()
    print(f"Adaptive (max_sigma=3.0, k=1.0) | Error: {err_a:.4f} | Coherence: {C_a:.4f}")
    
    # 3. Directional Smooth
    T_base = tensor.structure_tensor_from_difference(Dx, Dy, sigma=1.0, adaptive=False)
    _, eigvecs_base = eigen.decompose_tensor(T_base)
    T_dir = filters.directional_smooth(T_base, eigvecs_base, sigma=1.5)
    eigvals_d, _ = eigen.decompose_tensor(T_dir)
    l1_d = eigvals_d[..., 0]
    err_d = compute_localization_error(l1_d[row_idx, :], true_edge_x)
    mask_d = l1_d > 0.1 * l1_d.max()
    C_d = eigen.coherence(eigvals_d)[mask_d].mean()
    print(f"Directional (sigma=1.5) | Error: {err_d:.4f} | Coherence: {C_d:.4f}")
    
    # 4. Combined (Adaptive + Directional)
    _, eigvecs_a = eigen.decompose_tensor(T_adapt)
    T_comb = filters.directional_smooth(T_adapt, eigvecs_a, sigma=1.5)
    eigvals_c, _ = eigen.decompose_tensor(T_comb)
    l1_c = eigvals_c[..., 0]
    err_c = compute_localization_error(l1_c[row_idx, :], true_edge_x)
    mask_c = l1_c > 0.1 * l1_c.max()
    C_c = eigen.coherence(eigvals_c)[mask_c].mean()
    print(f"Combined (Adaptive+Dir) | Error: {err_c:.4f} | Coherence: {C_c:.4f}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Fixed sigma points connected by line
    norm = Normalize(vmin=sigmas.min(), vmax=sigmas.max())
    cmap = plt.cm.viridis
    
    for i in range(len(sigmas) - 1):
        ax.plot([fixed_errors[i], fixed_errors[i+1]], 
                [fixed_coherences[i], fixed_coherences[i+1]], 
                '-', color='gray', alpha=0.5, zorder=1)
                
    sc = ax.scatter(fixed_errors, fixed_coherences, c=sigmas, cmap=cmap, norm=norm, 
                    s=50, zorder=2, label=r'Fixed $\sigma$')
    
    # Add colorbar for sigmas
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label(r'Fixed Gaussian $\sigma$')
    
    # Special points
    ax.plot(err_a, C_a, '*', markersize=15, color='orange', markeredgecolor='k', 
            label='Adaptive', zorder=3)
    ax.plot(err_d, C_d, '*', markersize=15, color='green', markeredgecolor='k', 
            label='Directional', zorder=3)
    ax.plot(err_c, C_c, '*', markersize=15, color='red', markeredgecolor='k', 
            label='Combined', zorder=3)
            
    # Arrow pointing to the Pareto-optimal Combined method
    ax.annotate('Pareto Optimal\n(Combined: Low Error, High Coherence)', 
                xy=(err_c, C_c),
                xytext=(err_c + 8, C_c - 0.02),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=7),
                fontsize=9.5, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
                
    ax.set_xlabel('Edge Localization Error (pixels)')
    ax.set_ylabel('Mean Orientation Coherence ($C$)')
    ax.set_title('Pareto Frontier of Spatial Smoothing')
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    out_pdf = dirs['figures'] / 'fig02_pareto_frontier.pdf'
    out_png = dirs['figures'] / 'fig02_pareto_frontier.png'
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"Saved figure to {out_pdf} and {out_png}")
    plt.close(fig)

    # Save CSV table
    import pandas as pd
    rows = []
    for s, err, coh in zip(sigmas, fixed_errors, fixed_coherences):
        rows.append({"method": f"Fixed sigma={s:.2f}", "localization_error_px": err, "coherence": coh})
    rows.append({"method": "Adaptive", "localization_error_px": err_a, "coherence": C_a})
    rows.append({"method": "Directional", "localization_error_px": err_d, "coherence": C_d})
    rows.append({"method": "Combined", "localization_error_px": err_c, "coherence": C_c})
    df = pd.DataFrame(rows)
    csv_path = dirs['results'] / 'pareto_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

if __name__ == '__main__':
    run_experiment()
