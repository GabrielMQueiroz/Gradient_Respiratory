import streamlit as st
import os
import glob
import tempfile
import cv2
import numpy as np
import pandas as pd

from sttensor import gradients, tensor, eigen, filters
from sttensor.optical_flow import farneback_flow, flow_to_colorwheel
from sttensor.benchmarks import compute_fwhm

st.title("Interactive 3-Method Video Comparison")
st.markdown("""
This page interactively evaluates the 3 methods demonstrated across the research videos:
1. **Farnebäck Optical Flow** (Dense motion vector field)
2. **Fixed-Scale Structure Tensor** (Isotropic Gaussian smearing baseline)
3. **Adaptive + Directional Tensor** (Proposed edge-preserving formulation)
""")

# Sidebar settings
st.sidebar.subheader("Method Parameters")
sigma_fixed = st.sidebar.slider("Fixed Gaussian Sigma", 0.5, 6.0, 2.5, 0.5)
sigma_max = st.sidebar.slider("Adaptive Max Sigma", 1.0, 6.0, 3.0, 0.5)
k_sensitivity = st.sidebar.slider("Edge Sensitivity (k)", 0.1, 5.0, 1.5, 0.1)
dir_smooth = st.sidebar.checkbox("Apply Directional Smoothing", value=True)

# Video selection
videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "videos"))
video_paths = glob.glob(os.path.join(videos_dir, "*.mp4"))
video_dict = {os.path.basename(v): v for v in video_paths}
video_dict["Custom Upload..."] = "UPLOAD"

selected_choice = st.selectbox("Select Sample Video", options=list(video_dict.keys()))

input_path = None
if selected_choice == "Custom Upload...":
    uploaded = st.file_uploader("Upload video (MP4/AVI)", type=["mp4", "avi"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        input_path = tfile.name
elif selected_choice in video_dict:
    input_path = video_dict[selected_choice]

if input_path and os.path.exists(input_path):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_idx = st.slider("Inspect Frame Pair", 1, max(1, total_frames - 1), 10)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ret1, f_prev = cap.read()
    ret2, f_curr = cap.read()
    cap.release()

    if ret1 and ret2:
        # Grayscale
        g_prev = cv2.cvtColor(f_prev, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g_curr = cv2.cvtColor(f_curr, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 1. Optical Flow
        flow = farneback_flow(f_prev, f_curr)
        flow_bgr = flow_to_colorwheel(flow)
        flow_mag = np.hypot(flow[..., 0], flow[..., 1])

        # 2. Fixed Tensor
        Gx, Gy = gradients.sobel(g_curr)
        Gx_prev, Gy_prev = gradients.sobel(g_prev)
        Dx, Dy = gradients.gradient_difference((Gx, Gy), (Gx_prev, Gy_prev))
        
        T_fixed = tensor.structure_tensor_from_difference(Dx, Dy, sigma=sigma_fixed)
        ev_fixed, _ = eigen.decompose_tensor(T_fixed)
        l1_fixed = ev_fixed[..., 0]
        coh_fixed = eigen.coherence(ev_fixed)

        # 3. Adaptive Tensor
        T_adapt = tensor.structure_tensor_from_difference(
            Dx, Dy, adaptive=True, sigma_max=sigma_max, k=k_sensitivity
        )
        ev_adapt, evecs_adapt = eigen.decompose_tensor(T_adapt)
        
        if dir_smooth:
            T_adapt_smooth = filters.directional_smooth(T_adapt, evecs_adapt, sigma=1.5)
            ev_adapt, _ = eigen.decompose_tensor(T_adapt_smooth)
            
        l1_adapt = ev_adapt[..., 0]
        coh_adapt = eigen.coherence(ev_adapt)

        # Visualizations
        st.subheader(f"Side-by-Side Method Analysis (Frames {frame_idx-1} -> {frame_idx})")
        col1, col2, col3 = st.columns(3)

        def to_colormap(arr):
            p99 = np.percentile(arr, 99)
            norm = np.clip(arr / (p99 if p99 > 1e-5 else 1.0) * 255.0, 0, 255).astype(np.uint8)
            return cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

        with col1:
            st.markdown("**1. Optical Flow (Farnebäck)**")
            st.image(cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption(f"Mean Flow Mag: {np.mean(flow_mag):.2f} px")

        with col2:
            st.markdown(f"**2. Fixed Tensor ($\sigma={sigma_fixed}$)**")
            fixed_vis = to_colormap(l1_fixed)
            st.image(cv2.cvtColor(fixed_vis, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption(f"Mean Coherence: {np.mean(coh_fixed):.3f}")

        with col3:
            st.markdown("**3. Adaptive Gradient Difference Tensor (Ours)**")
            adapt_vis = to_colormap(l1_adapt)
            st.image(cv2.cvtColor(adapt_vis, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption(f"Mean Coherence: {np.mean(coh_adapt):.3f}")

        # Quantitative cross-section profile
        st.subheader("Spatial Edge Cross-Section (FWHM Analysis)")
        mid_y = height // 2
        line_fixed = l1_fixed[mid_y, :]
        line_adapt = l1_adapt[mid_y, :]
        
        # Normalize profiles
        p_fixed = line_fixed / (np.max(line_fixed) + 1e-6)
        p_adapt = line_adapt / (np.max(line_adapt) + 1e-6)
        
        fwhm_fixed = compute_fwhm(p_fixed)
        fwhm_adapt = compute_fwhm(p_adapt)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Fixed Tensor FWHM (Smearing)", f"{fwhm_fixed:.2f} px")
        with col_m2:
            st.metric("Adaptive Tensor FWHM (Edge Localization)", f"{fwhm_adapt:.2f} px", delta=f"{fwhm_adapt - fwhm_fixed:.2f} px", delta_color="inverse")

        df_profiles = pd.DataFrame({
            "Fixed Tensor (Smears Boundary)": p_fixed,
            "Adaptive Tensor (Sharp Boundary)": p_adapt,
        })
        st.line_chart(df_profiles)
