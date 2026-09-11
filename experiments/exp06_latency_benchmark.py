"""
Experiment 06: Latency Benchmark
Profiles per-frame computational cost at various resolutions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sttensor import gradients, tensor, eigen, filters
from sttensor.benchmarks import measure_latency, setup_publication_style, ensure_output_dirs

def main():
    setup_publication_style()
    ensure_output_dirs()

    rng = np.random.default_rng(42)

    resolutions = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    
    stages = [
        'Sobel Gradients',
        'Gradient Difference',
        'Tensor (Fixed)',
        'Tensor (Adaptive)',
        'Eigendecomposition',
        'Directional Filtering',
        'Full Pipeline'
    ]
    
    results = []

    for H, W in resolutions:
        print(f"Profiling resolution {H}x{W}...")
        frame_t = rng.random((H, W))
        frame_tm1 = rng.random((H, W))

        grad_t = gradients.sobel(frame_t)
        grad_tm1 = gradients.sobel(frame_tm1)
        Dx, Dy = gradients.gradient_difference(grad_t, grad_tm1)
        T_fixed = tensor.structure_tensor_from_difference(Dx, Dy, sigma=2.0)
        T_adapt = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0)
        eigvals, eigvecs = eigen.decompose_tensor(T_adapt)

        def stage_a():
            gradients.sobel(frame_t)

        def stage_b():
            gradients.gradient_difference(grad_t, grad_tm1)

        def stage_c():
            tensor.structure_tensor_from_difference(Dx, Dy, sigma=2.0)

        def stage_d():
            tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0)

        def stage_e():
            eigen.decompose_tensor(T_adapt)

        def stage_f():
            filters.directional_smooth(T_adapt, eigvecs, sigma=1.5)

        def stage_g():
            gt = gradients.sobel(frame_t)
            gtm1 = gradients.sobel(frame_tm1)
            dx, dy = gradients.gradient_difference(gt, gtm1)
            t_ad = tensor.structure_tensor_from_difference(dx, dy, adaptive=True, sigma_max=3.0)
            evals, evecs = eigen.decompose_tensor(t_ad)
            filters.directional_smooth(t_ad, evecs, sigma=1.5)

        funcs = [stage_a, stage_b, stage_c, stage_d, stage_e, stage_f, stage_g]

        for stage_name, func in zip(stages, funcs):
            mean_ms, std_ms = measure_latency(func, n_trials=50, warmup=5)
            results.append({
                'resolution': f"{H}x{W}",
                'stage': stage_name,
                'mean_ms': mean_ms,
                'std_ms': std_ms
            })

    df = pd.DataFrame(results)
    
    csv_path = os.path.join('experiments', 'results', 'latency_table.csv')
    df.to_csv(csv_path, index=False)
    
    print("\nLatency Benchmark Results:")
    print(df.to_string())

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stacked_stages = [
        'Sobel Gradients',
        'Gradient Difference',
        'Tensor (Adaptive)',
        'Eigendecomposition',
        'Directional Filtering'
    ]
    
    res_labels = [f"{H}x{W}" for H, W in resolutions]
    x = np.arange(len(res_labels))
    bottoms = np.zeros(len(res_labels))
    
    for stage in stacked_stages:
        stage_means = df[df['stage'] == stage]['mean_ms'].values
        ax.bar(x, stage_means, label=stage, bottom=bottoms)
        bottoms += stage_means

    ax.set_xticks(x)
    ax.set_xticklabels(res_labels)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Per-Frame Computational Cost by Stage')
    ax.legend(title='Pipeline Stage')
    
    plt.tight_layout()
    fig_path = os.path.join('experiments', 'figures', 'fig06_latency.pdf')
    plt.savefig(fig_path)
    plt.close()

if __name__ == '__main__':
    main()
