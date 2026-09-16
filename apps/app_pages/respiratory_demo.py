import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import glob

# Ensure we can import the live application classes
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mediapipe_chest_pose_deconvolution import (
    RespirationConfig, MediaPipeChestExtractor, EulerianEngine
)

st.title("Pose-Anchored Eulerian Respiratory Deconvolution")
st.markdown("""
Extract sub-pixel physiological breathing signals from video while decoupling gross bodily motion.
""")

# Parameter tuning in sidebar
st.sidebar.subheader("Eulerian Parameters")
alpha = st.sidebar.slider("Gain (alpha)", 1.0, 80.0, 35.0, 1.0)
tau = st.sidebar.slider("Anti-Blooming Tau", 1.0, 30.0, 8.0, 1.0)
gamma_fast = st.sidebar.slider("Temporal Gamma (gamma)", 0.02, 0.50, 0.20, 0.02)
grad_scale = st.sidebar.slider("Contrast Sensitivity", 0.5, 3.0, 1.2, 0.1)
freeze_th = st.sidebar.slider("Pose Freeze Threshold", 0.0, 10.0, 3.5, 0.5)

# Video selection
videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "videos"))
repo_videos = glob.glob(os.path.join(videos_dir, "*.mp4")) + glob.glob(os.path.join(videos_dir, "*.avi"))
video_options = {os.path.basename(v): v for v in repo_videos}
video_options["Upload Custom..."] = "UPLOAD"

selected_video = st.selectbox("Select Video", list(video_options.keys()))

input_path = None
if selected_video == "Upload Custom...":
    uploaded_file = st.file_uploader("Upload an MP4", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        input_path = tfile.name
else:
    input_path = video_options[selected_video]

if st.button("Process Video Sequence", type="primary") and input_path:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Failed to open video source.")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        cfg = RespirationConfig()
        cfg.alpha = alpha
        cfg.soft_clamp = tau
        cfg.gamma_fast = gamma_fast
        cfg.gradient_scale = grad_scale

        extractor = MediaPipeChestExtractor()
        extractor.freeze_threshold = freeze_th
        engine = EulerianEngine(cfg.patch_width, cfg.patch_height)

        macro_motion_y = []
        micro_motion_s = []

        progress_bar = st.progress(0, text="Processing video frames...")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Process Frame
            patch, quad_pts, has_pose, _, all_landmarks, sternum_info = extractor.extract_patch(
                frame, cfg.patch_width, cfg.patch_height
            )
            mag_patch, relief_patch, _, scalar, _, _ = engine.process(patch, cfg)

            # Track Macro and Micro
            if quad_pts is not None:
                center_y = float(np.mean(quad_pts[:, 1]))
                macro_motion_y.append(center_y)
            else:
                macro_motion_y.append(macro_motion_y[-1] if macro_motion_y else height / 2.0)

            micro_motion_s.append(scalar)

            # Overlay
            display_frame = frame.copy()
            if cfg.overlay_chest and has_pose and quad_pts is not None:
                src_pts = np.array([
                    [0, 0],
                    [cfg.patch_width - 1, 0],
                    [cfg.patch_width - 1, cfg.patch_height - 1],
                    [0, cfg.patch_height - 1]
                ], dtype=np.float32)
                try:
                    inv_mat = cv2.getPerspectiveTransform(src_pts, quad_pts)
                    warped = cv2.warpPerspective(relief_patch, inv_mat, (width, height))
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, quad_pts.astype(np.int32), 255)
                    mask_3ch = cv2.merge([mask, mask, mask])
                    display_frame = np.where(mask_3ch > 0, cv2.addWeighted(warped, 0.88, display_frame, 0.12, 0), display_frame)
                except Exception:
                    pass
                cv2.polylines(display_frame, [quad_pts.astype(np.int32)], True, (0, 240, 168), 2, cv2.LINE_AA)

            writer.write(display_frame)

            if total_frames > 0 and frame_idx % 5 == 0:
                progress_bar.progress(min(1.0, frame_idx / total_frames), text=f"Processing: {frame_idx}/{total_frames} frames")

        writer.release()
        cap.release()
        progress_bar.empty()

        st.success("Video processing complete.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Enhanced Eulerian Output")
            try:
                st.video(out_file)
            except Exception:
                st.warning("Video codec fallback required to render in browser.")

        with col2:
            st.subheader("Signal Decoupling Telemetry")
            st.markdown("**1. Macro-Motion (Gross Subject Sway)**")
            macro_arr = np.array(macro_motion_y)
            if len(macro_arr) > 0:
                macro_arr = macro_arr - macro_arr[0]
            st.line_chart(pd.DataFrame({"Chest Displacement (pixels)": macro_arr}))

            st.markdown("**2. Micro-Motion (Eulerian Respiratory Waveform)**")
            st.line_chart(pd.DataFrame({"Respiratory Signal Amplitude": micro_motion_s}))
