import streamlit as st
import time
import numpy as np
import pandas as pd

from sttensor import gradients, tensor, eigen, benchmarks
from sttensor.optical_flow import farneback_flow

st.title("System & Library Benchmark Dashboard")
st.markdown("""
Empirical latency profiling across tensor pipeline components and optical flow baselines.
""")

resolutions = [64, 128, 256, 512]
n_trials = st.slider("Trials per benchmark", 5, 30, 10, 5)

if st.button("Run Full Benchmark Suite", type="primary"):
    progress = st.progress(0, text="Benchmarking in progress...")

    results = []
    total_steps = len(resolutions)

    for i, res in enumerate(resolutions):
        progress.progress((i + 1) / total_steps, text=f"Benchmarking resolution {res}x{res}...")

        f0 = (np.random.rand(res, res) * 255).astype(np.uint8)
        f1 = (np.random.rand(res, res) * 255).astype(np.uint8)
        f0_f = f0.astype(np.float64)
        f1_f = f1.astype(np.float64)

        # 1. Sobel Gradients
        t_sobel, _ = benchmarks.measure_latency(gradients.sobel, f1_f, n_trials=n_trials)

        # 2. Gradient Difference
        gx0, gy0 = gradients.sobel(f0_f)
        gx1, gy1 = gradients.sobel(f1_f)
        t_gdiff, _ = benchmarks.measure_latency(
            gradients.gradient_difference, (gx1, gy1), (gx0, gy0), n_trials=n_trials
        )
        dx, dy = gx1 - gx0, gy1 - gy0

        # 3. Structure Tensor (Unsmoothed)
        t_traw, _ = benchmarks.measure_latency(
            tensor.structure_tensor_from_difference, dx, dy, sigma=None, n_trials=n_trials
        )

        # 4. Structure Tensor (Fixed Gaussian)
        t_tfixed, _ = benchmarks.measure_latency(
            tensor.structure_tensor_from_difference, dx, dy, sigma=1.5, n_trials=n_trials
        )

        # 5. Structure Tensor (Adaptive)
        t_tadapt, _ = benchmarks.measure_latency(
            tensor.structure_tensor_from_difference, dx, dy, adaptive=True, sigma_max=3.0, k=1.0, n_trials=n_trials
        )

        # 6. Eigen Decomposition
        T = tensor.structure_tensor_from_difference(dx, dy, sigma=1.5)
        t_eigen, _ = benchmarks.measure_latency(eigen.decompose_tensor, T, n_trials=n_trials)

        # 7. Fast Coherence
        t_fast_coh, _ = benchmarks.measure_latency(eigen.fast_coherence, T, n_trials=n_trials)

        # 8. Farneback Flow
        t_flow, _ = benchmarks.measure_latency(farneback_flow, f0, f1, n_trials=n_trials)

        results.append({
            "Resolution": f"{res}x{res}",
            "Sobel (ms)": t_sobel,
            "Gradient Diff (ms)": t_gdiff,
            "Raw Tensor (ms)": t_traw,
            "Fixed Tensor (ms)": t_tfixed,
            "Adaptive Tensor (ms)": t_tadapt,
            "Eigen Decomposition (ms)": t_eigen,
            "Fast Coherence (ms)": t_fast_coh,
            "Farnebäck Flow (ms)": t_flow,
        })

    progress.empty()
    st.success("Benchmarks complete.")

    df_results = pd.DataFrame(results).set_index("Resolution")
    st.dataframe(df_results)
    st.bar_chart(df_results[["Fixed Tensor (ms)", "Adaptive Tensor (ms)", "Farnebäck Flow (ms)"]])
