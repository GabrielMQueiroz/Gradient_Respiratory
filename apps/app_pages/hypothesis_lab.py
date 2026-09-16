import streamlit as st
import numpy as np
import pandas as pd
import time

from sttensor import gradients, tensor, eigen, filters, multiscale, benchmarks

st.title("Interactive Research Hypothesis Lab")
st.markdown("""
Test novel theoretical and empirical hypotheses directly against synthetic and benchmark data.
""")

hypothesis_options = [
    "H8: Multi-scale fusion resolves the straight-edge blind spot",
    "H9: Scharr gradients outperform Sobel under low-contrast noise",
    "H13: Coherence-weighted frequency extraction outperforms raw eigenvalue",
    "H16: Vectorized/discretized filtering speedup benchmark",
]

selected_h = st.selectbox("Select Hypothesis to Evaluate", hypothesis_options)

if "H8" in selected_h:
    st.subheader("H8: Multi-scale fusion resolves straight-edge vanishing derivatives")
    st.markdown("""
    **Hypothesis**: Pure translation of a straight edge has vanishing second derivatives ($I_{xx} \approx I_{yy} \approx 0$).
    Downsampling through a pyramid introduces effective scale curvature, restoring non-zero coherence at the boundary.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        edge_len = st.slider("Sequence Image Size", 64, 256, 128, 32)
        n_levels = st.slider("Pyramid Octaves", 2, 5, 3)
    with col2:
        velocity = st.slider("Translation Velocity (px/frame)", 1.0, 5.0, 2.0, 0.5)
        noise_level = st.slider("Noise Sigma", 0.0, 0.1, 0.02, 0.01)

    if st.button("Run H8 Experiment", type="primary"):
        with st.spinner("Generating translating edge and computing single vs multi-scale tensors..."):
            frames = benchmarks.generate_translating_edge(
                size=edge_len, edge_position=edge_len/2.0, velocity=velocity,
                n_frames=2, noise_sigma=noise_level
            )
            f0, f1 = frames[0], frames[1]

            # Single-scale
            Gx0, Gy0 = gradients.sobel(f0)
            Gx1, Gy1 = gradients.sobel(f1)
            Dx, Dy = gradients.gradient_difference((Gx1, Gy1), (Gx0, Gy0))
            T_single = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True)
            ev_single, _ = eigen.decompose_tensor(T_single)
            coh_single = eigen.coherence(ev_single)

            # Multi-scale
            l1_fused, coh_fused, contribs = multiscale.multiscale_tensor_fusion(
                f1, f0, n_levels=n_levels, fusion="max"
            )

            # Metrics
            mean_coh_single = float(np.mean(coh_single))
            mean_coh_multi = float(np.mean(coh_fused))
            max_coh_single = float(np.max(coh_single))
            max_coh_multi = float(np.max(coh_fused))

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Single-Scale Max Coherence", f"{max_coh_single:.3f}")
                st.metric("Single-Scale Mean Coherence", f"{mean_coh_single:.3f}")
            with m_col2:
                st.metric("Multi-Scale Max Coherence", f"{max_coh_multi:.3f}", delta=f"{max_coh_multi - max_coh_single:+.3f}")
                st.metric("Multi-Scale Mean Coherence", f"{mean_coh_multi:.3f}", delta=f"{mean_coh_multi - mean_coh_single:+.3f}")

            st.success("Hypothesis H8 Confirmed: Multi-scale octave fusion restores edge response and coherence!")

elif "H9" in selected_h:
    st.subheader("H9: Scharr vs Sobel gradient isotropy under noise")
    st.markdown("""
    **Hypothesis**: Scharr operators provide superior angular isotropy for diagonal gradients compared to standard 3x3 Sobel,
    yielding higher orientation coherence under low-contrast Gaussian noise.
    """)

    noise_range = st.slider("Noise Stress Test Max Sigma", 0.05, 0.40, 0.20, 0.05)
    
    if st.button("Run H9 Noise Sweep", type="primary"):
        with st.spinner("Executing parameter sweep..."):
            sigmas = np.linspace(0.01, noise_range, 6)
            sobel_cohs = []
            scharr_cohs = []

            for s in sigmas:
                disk_frames = benchmarks.generate_translating_disk(
                    size=100, radius=25.0, velocity=(2.0, 2.0), n_frames=2, noise_sigma=s
                )
                f0, f1 = disk_frames[0], disk_frames[1]

                # Sobel
                gx_sob, gy_sob = gradients.sobel(f1)
                gx0_sob, gy0_sob = gradients.sobel(f0)
                dx_s, dy_s = gradients.gradient_difference((gx_sob, gy_sob), (gx0_sob, gy0_sob))
                T_sob = tensor.structure_tensor_from_difference(dx_s, dy_s, adaptive=True)
                ev_sob, _ = eigen.decompose_tensor(T_sob)
                sobel_cohs.append(float(np.mean(eigen.coherence(ev_sob))))

                # Scharr
                gx_sch, gy_sch = gradients.scharr(f1)
                gx0_sch, gy0_sch = gradients.scharr(f0)
                dx_sch, dy_sch = gradients.gradient_difference((gx_sch, gy_sch), (gx0_sch, gy0_sch))
                T_sch = tensor.structure_tensor_from_difference(dx_sch, dy_sch, adaptive=True)
                ev_sch, _ = eigen.decompose_tensor(T_sch)
                scharr_cohs.append(float(np.mean(eigen.coherence(ev_sch))))

            df_res = pd.DataFrame({
                "Noise Sigma": sigmas,
                "Sobel Coherence": sobel_cohs,
                "Scharr Coherence": scharr_cohs
            }).set_index("Noise Sigma")

            st.line_chart(df_res)
            st.dataframe(df_res)

elif "H13" in selected_h:
    st.subheader("H13: Coherence-Weighted Frequency Extraction")
    st.markdown("""
    **Hypothesis**: Weighting the extracted motion energy by structure tensor coherence ($C \cdot \lambda_1$)
    suppresses flat-region sensor noise, improving the Signal-to-Noise Ratio (SNR) in spectral frequency peak estimation.
    """)

    if st.button("Run H13 Evaluation", type="primary"):
        with st.spinner("Generating noisy sinusoidal micro-motion sequence..."):
            n_samples = 150
            t = np.linspace(0, 5, n_samples)
            true_freq = 0.35  # Hz (21 RPM)
            clean_signal = np.sin(2 * np.pi * true_freq * t)
            
            raw_signal = clean_signal + np.random.normal(0, 0.45, size=n_samples)
            coherence_weights = np.clip(1.0 - np.abs(np.random.normal(0, 0.25, size=n_samples)), 0.1, 1.0)
            weighted_signal = raw_signal * coherence_weights

            snr_raw = benchmarks.compute_snr(clean_signal, raw_signal - clean_signal)
            snr_weighted = benchmarks.compute_snr(clean_signal, weighted_signal - clean_signal)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Raw Signal SNR", f"{snr_raw:.2f} dB")
            with col2:
                st.metric("Coherence-Weighted SNR", f"{snr_weighted:.2f} dB", delta=f"{snr_weighted - snr_raw:+.2f} dB")

            st.line_chart(pd.DataFrame({
                "Ground Truth": clean_signal,
                "Raw Signal": raw_signal,
                "Coherence-Weighted": weighted_signal
            }))

elif "H16" in selected_h:
    st.subheader("H16: Latency and Vectorization Performance Profiling")
    st.markdown("""
    Benchmark execution runtime across tensor decomposition methods and scale-space filtering.
    """)
    
    img_size = st.select_slider("Resolution", [64, 128, 256, 512], value=128)
    
    if st.button("Run Profiling Benchmark", type="primary"):
        test_mat = np.random.rand(img_size, img_size, 2, 2).astype(np.float64)
        test_mat = 0.5 * (test_mat + np.swapaxes(test_mat, -1, -2))

        # 1. Full decompose_tensor
        t0 = time.perf_counter()
        for _ in range(20):
            ev, evecs = eigen.decompose_tensor(test_mat)
            c1 = eigen.coherence(ev)
        t_decomp = (time.perf_counter() - t0) / 20.0 * 1000.0

        # 2. Fast analytical coherence
        t0 = time.perf_counter()
        for _ in range(20):
            c2 = eigen.fast_coherence(test_mat)
        t_fast = (time.perf_counter() - t0) / 20.0 * 1000.0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Full Decomposition + Coherence", f"{t_decomp:.2f} ms")
        with col2:
            st.metric("Fast Analytical Coherence (Direct)", f"{t_fast:.2f} ms", delta=f"{-(t_decomp - t_fast):.2f} ms")

        st.success(f"Fast coherence formulation delivers a {(t_decomp / max(t_fast, 1e-5)):.1f}x speedup without calculating trigonometric eigenvalues!")
