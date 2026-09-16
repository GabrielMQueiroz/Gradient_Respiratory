import streamlit as st
import numpy as np
import pandas as pd

from sttensor import benchmarks, gradients, tensor, eigen

st.title("Parameter Space & Pareto Explorer")
st.markdown("""
Explore how hyperparameters ($\sigma, k, \sigma_{\max}$) modulate edge localization, noise suppression, and orientation coherence.
""")

col1, col2 = st.columns(2)
with col1:
    disk_radius = st.slider("Disk Radius (px)", 15.0, 45.0, 25.0, 5.0)
    noise_sigma = st.slider("Background Gaussian Noise", 0.0, 0.20, 0.05, 0.02)
with col2:
    sigma_max_range = st.slider("Max Adaptive Sigma ($\sigma_{\max}$)", 1.0, 5.0, 3.0, 0.5)
    k_range = st.slider("Sensitivity Gain ($k$)", 0.5, 4.0, 1.5, 0.5)

if st.button("Generate Parameter Surface Analysis", type="primary"):
    with st.spinner("Computing sweep over fixed vs adaptive parameters..."):
        # Synthetic disk motion
        frames = benchmarks.generate_translating_disk(
            size=128, radius=disk_radius, velocity=(2.0, 0.0), n_frames=2, noise_sigma=noise_sigma
        )
        f0, f1 = frames[0], frames[1]
        Gx0, Gy0 = gradients.sobel(f0)
        Gx1, Gy1 = gradients.sobel(f1)
        Dx, Dy = gradients.gradient_difference((Gx1, Gy1), (Gx0, Gy0))

        # Sweep fixed sigmas
        fixed_sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        fwhms_fixed = []
        cohs_fixed = []

        mid_y = 64
        for s in fixed_sigmas:
            T = tensor.structure_tensor_from_difference(Dx, Dy, sigma=s)
            ev, _ = eigen.decompose_tensor(T)
            l1 = ev[..., 0]
            profile = l1[mid_y, :] / (np.max(l1[mid_y, :]) + 1e-6)
            fwhms_fixed.append(benchmarks.compute_fwhm(profile))
            cohs_fixed.append(float(np.mean(eigen.coherence(ev))))

        # Sweep adaptive k
        k_values = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        fwhms_adapt = []
        cohs_adapt = []

        for kv in k_values:
            T = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=sigma_max_range, k=kv)
            ev, _ = eigen.decompose_tensor(T)
            l1 = ev[..., 0]
            profile = l1[mid_y, :] / (np.max(l1[mid_y, :]) + 1e-6)
            fwhms_adapt.append(benchmarks.compute_fwhm(profile))
            cohs_adapt.append(float(np.mean(eigen.coherence(ev))))

        st.subheader("Trade-off Comparison: Smearing (FWHM) vs Coherence")

        df_fixed = pd.DataFrame({
            "Fixed Sigma": fixed_sigmas,
            "FWHM (pixels - lower is sharper)": fwhms_fixed,
            "Mean Coherence": cohs_fixed
        }).set_index("Fixed Sigma")

        df_adapt = pd.DataFrame({
            "Adaptive Sensitivity (k)": k_values,
            "FWHM (pixels - lower is sharper)": fwhms_adapt,
            "Mean Coherence": cohs_adapt
        }).set_index("Adaptive Sensitivity (k)")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Fixed Gaussian Smoothing (Smearing Creep)")
            st.dataframe(df_fixed)
            st.line_chart(df_fixed)
        with col_b:
            st.markdown("#### Adaptive Smoothing (Boundary Preservation)")
            st.dataframe(df_adapt)
            st.line_chart(df_adapt)
