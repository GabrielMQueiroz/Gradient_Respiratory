"""
Rotation-Invariant Real-Time Eulerian Motion Magnification with MediaPipe Pose
=============================================================================
Linux-Optimized Multi-Backend Pose Deconvolution with Dedicated GTK Controls
"""

import sys
import os
import time
import math
import collections
import subprocess

# Pre-flight NumPy ABI Compatibility Check
try:
    import numpy as np
    np_ver = tuple(map(int, np.__version__.split('.')[:2]))
    if np_ver[0] >= 2:
        print("\n" + "!" * 76)
        print(f" [WARNING] NumPy 2.x Detected ({np.__version__})")
        print(" If MediaPipe C-extensions fail to load, run in your terminal:")
        print("     conda activate respiration")
        print("     pip install \"numpy<2\"")
        print("!" * 76 + "\n")
except Exception:
    pass

import cv2

# Linux GUI Initialization helper: starts GTK background event thread
if sys.platform.startswith("linux"):
    try:
        cv2.startWindowThread()
    except Exception:
        pass


class RespirationConfig:
    """Runtime parameters and algorithm gain thresholds."""
    def __init__(self):
        # Processing resolution for the normalized chest patch
        self.patch_width = 240
        self.patch_height = 180
        
        # Eulerian amplification parameters (adjustable via trackbars & hotkeys)
        self.alpha = 35.0          # Eulerian amplification gain (1 to 80)
        self.artifact_gate = 0.22  # Motion energy threshold for artifact rejection (0.01 to 1.0)
        self.soft_clamp = 8.0      # Tanh saturation ceiling (anti-blooming) (1 to 30)
        self.gamma_fast = 0.20     # Fast IIR pole (0.02 to 0.50)
        self.gradient_scale = 1.2  # Edge Contrast Sensitivity (0.5 to 3.0x)
        self.coherence_th = 0.08   # Structure tensor edge coherence threshold (0.0 to 0.40)
        self.gate_mode = 0         # 0 = Soft AGC, 1 = Hard Freeze, 2 = Bypass
        
        # Chest Viewport Mode: 0 = Motion Band Relief, 1 = Color EVM, 2 = Coherence Map
        self.chest_view_mode = 0
        self.overlay_chest = True  # Project active deconvolution mode directly onto chest quad
        
        # Physiological limits
        self.min_rpm = 6.0         # 0.10 Hz
        self.max_rpm = 42.0        # 0.70 Hz
        self.buffer_len = 180      # ~6 seconds at 30 FPS
        self.show_skeleton = True  # Toggle skeleton rendering overlay


MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def robust_model_resolver():
    """
    Downloads and resolves Google's MediaPipe PoseLandmarker Lite model.
    Uses Linux-native curl with fallback to SSL-unverified urllib.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "models", MODEL_FILENAME),
        os.path.join(os.path.dirname(script_dir), "models", MODEL_FILENAME),
        os.path.join(script_dir, MODEL_FILENAME),
        os.path.join(os.path.dirname(script_dir), MODEL_FILENAME),
        os.path.expanduser(f"~/.cache/mediapipe/{MODEL_FILENAME}"),
        f"/tmp/{MODEL_FILENAME}"
    ]

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            return path

    dest_path = candidates[1] if os.path.exists(os.path.dirname(candidates[1])) else candidates[0]
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[INFO] Downloading PoseLandmarker Lite model (~3.5MB) to:\n       {dest_path}")

    # 1. Primary: Linux curl (handles proxies, system certificates, and redirections)
    try:
        res = subprocess.run(["curl", "-sL", "-f", "-o", dest_path, MODEL_URL], timeout=45)
        if res.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            print("[INFO] Model successfully acquired via curl.")
            return dest_path
    except Exception as e:
        print(f"[DEBUG] curl download bypassed: {e}")

    # 2. Secondary: wget
    try:
        res = subprocess.run(["wget", "-q", "-O", dest_path, MODEL_URL], timeout=45)
        if res.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            print("[INFO] Model successfully acquired via wget.")
            return dest_path
    except Exception:
        pass

    # 3. Tertiary: Python urllib with SSL unverified context
    try:
        import urllib.request
        import ssl
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
        with urllib.request.urlopen(req, context=ctx, timeout=45) as resp:
            with open(dest_path, "wb") as f:
                f.write(resp.read())
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            print("[INFO] Model successfully acquired via urllib.")
            return dest_path
    except Exception as err:
        print(f"[ERROR] Could not automatically download model: {err}")

    return None

class MediaPipeChestExtractor:
    """
    Extracts an anatomically normalized chest patch from body joints.
    Supports Modern Tasks Vision, Legacy Solutions, and Thoracic geometric fallback.
    """
    def __init__(self):
        self.backend = "NONE"
        self.detector = None
        self.is_tasks_api = False
        self.last_quad = None
        self.smoothing = 0.65  # EMA smoothing factor for bounding quad stability
        self.status_msg = "INITIALIZING POSE DETECTOR..."

        # 1. Modern MediaPipe Tasks Vision (v0.10.15+)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_file = robust_model_resolver()
            if model_file and os.path.exists(model_file):
                with open(model_file, "rb") as f:
                    model_bytes = f.read()

                # Passing raw buffer avoids any Linux C++ file path/permission issues
                base_options = python.BaseOptions(model_asset_buffer=model_bytes)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.25,
                    min_pose_presence_confidence=0.25,
                    min_tracking_confidence=0.25
                )
                self.detector = vision.PoseLandmarker.create_from_options(options)
                self.backend = "MEDIAPIPE_TASKS"
                self.is_tasks_api = True
                self.status_msg = "MEDIAPIPE TASKS ONLINE"
                print("[INFO] Initialized Modern MediaPipe Tasks Vision (PoseLandmarker Lite).")
        except Exception as e:
            print(f"[DEBUG] MediaPipe Tasks init notice: {e}")

        # 2. Legacy MediaPipe Solutions (v0.10.14 or older)
        if self.backend == "NONE":
            try:
                mp_pose = None
                try:
                    import mediapipe.python.solutions.pose as _sub_pose
                    mp_pose = _sub_pose
                except Exception:
                    pass

                if mp_pose is None:
                    import mediapipe as mp
                    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                        mp_pose = mp.solutions.pose

                if mp_pose is not None:
                    self.detector = mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=0,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        min_detection_confidence=0.30,
                        min_tracking_confidence=0.30
                    )
                    self.backend = "MEDIAPIPE_SOLUTIONS"
                    self.is_tasks_api = False
                    self.status_msg = "MEDIAPIPE SOLUTIONS ONLINE"
                    print("[INFO] Initialized Legacy MediaPipe Solutions Pose pipeline.")
            except Exception as e:
                pass

        # 3. Fallback: OpenCV Haar Cascade Face / Upper-Body
        if self.backend == "NONE":
            print("\n" + "=" * 74)
            print("[INFO] MediaPipe inactive. Running OpenCV Thoracic Estimator fallback.")
            print("=" * 74 + "\n")
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self.face_cascade = None
            self.status_msg = "OPENCV THORACIC FALLBACK"
        else:
            self.face_cascade = None

    def calculate_chest_quad(self, landmarks, frame_w, frame_h):
        """
        Calculates oriented sternum/chest quad from landmark coordinates.
        Supports both full body (shoulders+hips) and seated desk (shoulders+neck) setups.
        """
        def get_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * frame_w, lm.y * frame_h], dtype=np.float32)

        p_ls = get_pt(11)  # Left shoulder
        p_rs = get_pt(12)  # Right shoulder
        p_nose = get_pt(0) # Nose reference

        shoulder_vec = p_ls - p_rs
        shoulder_width = float(np.linalg.norm(shoulder_vec))

        if shoulder_width < 14.0:
            return None, "SEARCHING POSE (TOO DISTANT)...", None

        mid_shoulder = (p_ls + p_rs) * 0.5
        shoulder_dir = shoulder_vec / (shoulder_width + 1e-6)

        # Calculate torso downward vector
        v_neck = mid_shoulder - p_nose
        ortho_down = np.array([-shoulder_dir[1], shoulder_dir[0]], dtype=np.float32)
        if np.dot(ortho_down, v_neck) < 0:
            ortho_down = -ortho_down

        # Evaluate hip detection
        hips_detected = False
        torso_vector_vis = ortho_down * (shoulder_width * 0.45)
        if len(landmarks) > 24:
            try:
                lm_lh = landmarks[23]
                lm_rh = landmarks[24]
                vis_lh = getattr(lm_lh, 'visibility', 1.0) or 1.0
                vis_rh = getattr(lm_rh, 'visibility', 1.0) or 1.0
                if vis_lh > 0.25 and vis_rh > 0.25:
                    p_lh = get_pt(23)
                    p_rh = get_pt(24)
                    mid_hip = (p_lh + p_rh) * 0.5
                    torso_vec = mid_hip - mid_shoulder
                    torso_len = float(np.linalg.norm(torso_vec))
                    if torso_len > shoulder_width * 0.35:
                        torso_dir = torso_vec / (torso_len + 1e-6)
                        chest_center = mid_shoulder + torso_dir * (torso_len * 0.36)
                        chest_half_w = shoulder_width * 0.38
                        chest_half_h = torso_len * 0.22
                        down_dir = torso_dir
                        torso_vector_vis = torso_vec * 0.5
                        hips_detected = True
                        mode_label = "POSE LOCKED (FULL BODY)"
            except Exception:
                hips_detected = False

        if not hips_detected:
            down_dir = ortho_down
            chest_center = mid_shoulder + down_dir * (shoulder_width * 0.42)
            chest_half_w = shoulder_width * 0.38
            chest_half_h = shoulder_width * 0.30
            mode_label = "POSE LOCKED (UPPER TORSO / DESK)"

        # 4 corners of oriented bounding quad
        p1 = chest_center - shoulder_dir * chest_half_w - down_dir * chest_half_h
        p2 = chest_center + shoulder_dir * chest_half_w - down_dir * chest_half_h
        p3 = chest_center + shoulder_dir * chest_half_w + down_dir * chest_half_h
        p4 = chest_center - shoulder_dir * chest_half_w + down_dir * chest_half_h
        quad = np.array([p1, p2, p3, p4], dtype=np.float32)

        # EMA quad stabilization against high-frequency jitter
        if self.last_quad is not None:
            quad = self.smoothing * quad + (1.0 - self.smoothing) * self.last_quad
        self.last_quad = quad

        return quad, mode_label, (mid_shoulder, chest_center, torso_vector_vis)

    def extract_patch(self, frame_bgr, target_w, target_h):
        """
        Runs pose inference and extracts affine chest ROI patch.
        Returns: (patch, quad_pts, has_pose, label_text, all_landmarks, sternum_info)
        """
        h, w = frame_bgr.shape[:2]
        all_landmarks = []

        # 1. MediaPipe Tasks Inference
        if self.backend == "MEDIAPIPE_TASKS":
            try:
                import mediapipe as mp
                frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = self.detector.detect(mp_image)

                if res and res.pose_landmarks and len(res.pose_landmarks) > 0:
                    lms = res.pose_landmarks[0]
                    # Convert normalized landmarks to pixel coords for visualization
                    for lm in lms:
                        all_landmarks.append((int(lm.x * w), int(lm.y * h)))

                    quad, label, sternum_info = self.calculate_chest_quad(lms, w, h)
                    if quad is not None:
                        dst_quad = np.array([
                            [0, 0], [target_w - 1, 0],
                            [target_w - 1, target_h - 1], [0, target_h - 1]
                        ], dtype=np.float32)
                        mat = cv2.getPerspectiveTransform(quad, dst_quad)
                        patch = cv2.warpPerspective(frame_bgr, mat, (target_w, target_h), flags=cv2.INTER_LINEAR)
                        return patch, quad, True, label, all_landmarks, sternum_info
            except Exception as e:
                pass

        # 2. MediaPipe Solutions Inference
        elif self.backend == "MEDIAPIPE_SOLUTIONS":
            try:
                frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                frame_rgb.flags.writeable = False
                res = self.detector.process(frame_rgb)

                if res and res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    for lm in lms:
                        all_landmarks.append((int(lm.x * w), int(lm.y * h)))

                    quad, label, sternum_info = self.calculate_chest_quad(lms, w, h)
                    if quad is not None:
                        dst_quad = np.array([
                            [0, 0], [target_w - 1, 0],
                            [target_w - 1, target_h - 1], [0, target_h - 1]
                        ], dtype=np.float32)
                        mat = cv2.getPerspectiveTransform(quad, dst_quad)
                        patch = cv2.warpPerspective(frame_bgr, mat, (target_w, target_h), flags=cv2.INTER_LINEAR)
                        return patch, quad, True, label, all_landmarks, sternum_info
            except Exception as e:
                pass

        # 3. Haar Cascade Face/Thoracic Fallback
        if self.face_cascade is not None:
            try:
                gray_small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (320, 240))
                faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.2, minNeighbors=4)
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    scale_x, scale_y = w / 320.0, h / 240.0
                    fx, fy, fw, fh = fx * scale_x, fy * scale_y, fw * scale_x, fh * scale_y

                    cx = fx + fw * 0.5
                    cy = fy + fh * 2.1
                    rw = fw * 1.8
                    rh = fh * 1.6

                    x1 = max(0, int(cx - rw * 0.5))
                    y1 = max(0, int(cy - rh * 0.5))
                    x2 = min(w, int(cx + rw * 0.5))
                    y2 = min(h, int(cy + rh * 0.5))

                    if x2 > x1 and y2 > y1:
                        patch = cv2.resize(frame_bgr[y1:y2, x1:x2], (target_w, target_h))
                        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                        # Synthesize anchor keypoints for face fallback visualization
                        synthetic_pts = [
                            (int(cx), int(fy + fh * 0.5)),
                            (int(cx - rw * 0.45), int(cy - rh * 0.35)),
                            (int(cx + rw * 0.45), int(cy - rh * 0.35))
                        ]
                        return patch, pts, True, "HAAR THORACIC ESTIMATOR", synthetic_pts, None
            except Exception:
                pass

        # 4. Default Static Thoracic Box
        cx, cy = w // 2, int(h * 0.55)
        rw, rh = int(w * 0.35), int(h * 0.38)
        x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        x2, y2 = min(w, x1 + rw), min(h, y1 + rh)
        patch = cv2.resize(frame_bgr[y1:y2, x1:x2], (target_w, target_h))
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        return patch, pts, False, "STATIC CHEST ESTIMATOR (POSITION BODY)", [], None


def draw_full_skeleton(frame, landmarks):
    """Draws anatomical skeletal connections and landmark nodes."""
    if not landmarks or len(landmarks) < 13:
        return

    # Skeletal kinematic chain pairs
    pairs = [
        # Upper body and shoulders
        (11, 12),
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        # Torso
        (11, 23), (12, 24),  # Torso edges
        (23, 24),            # Hip bridge
        # Facial contours
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10)
    ]

    num_lms = len(landmarks)
    for p1, p2 in pairs:
        if p1 < num_lms and p2 < num_lms:
            pt1 = landmarks[p1]
            pt2 = landmarks[p2]
            cv2.line(frame, pt1, pt2, (0, 225, 180), 2, cv2.LINE_AA)

    # Render anatomical nodes
    for idx, (px, py) in enumerate(landmarks):
        if idx in [11, 12]:  # Shoulder anchors (highlighted)
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 9, (0, 160, 255), 2, cv2.LINE_AA)
        elif idx in [23, 24]:  # Hips
            cv2.circle(frame, (px, py), 5, (255, 200, 0), -1, cv2.LINE_AA)
        elif idx == 0:  # Nose
            cv2.circle(frame, (px, py), 4, (0, 255, 120), -1, cv2.LINE_AA)
        elif idx in [13, 14, 15, 16]:  # Arms
            cv2.circle(frame, (px, py), 4, (0, 200, 255), -1, cv2.LINE_AA)
        elif idx < 11:  # Facial points
            cv2.circle(frame, (px, py), 2, (180, 255, 200), -1, cv2.LINE_AA)

class EulerianEngine:
    """
    Rotation-Invariant Structure Tensor Coherence and Anti-Blooming Eulerian Deconvolution.
    """
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.iir_slow = np.zeros((height, width), dtype=np.float32)
        self.iir_fast = np.zeros((height, width), dtype=np.float32)
        self.is_initialized = False

    def reset(self):
        self.is_initialized = False

    def process(self, patch_bgr, cfg):
        """
        Applies Eulerian Magnification, Motion Band Relief synthesis, and motion metrics.
        Returns: (magnified_bgr, relief_bgr, coherence_map, motion_scalar, motion_energy, is_gated)
        """
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if not self.is_initialized or self.iir_slow.shape != gray.shape:
            self.iir_slow = gray.copy()
            self.iir_fast = gray.copy()
            self.is_initialized = True
            empty_coh = np.zeros_like(gray)
            neutral_relief = np.full_like(patch_bgr, 128)
            return patch_bgr.copy(), neutral_relief, empty_coh, 0.0, 0.0, False

        # Dual-IIR Temporal Respiration Bandpass Filter (Linked poles matching HTML engine)
        gamma_slow = cfg.gamma_fast * 0.15
        self.iir_slow += gamma_slow * (gray - self.iir_slow)
        self.iir_fast += cfg.gamma_fast * (gray - self.iir_fast)
        bandpass = self.iir_fast - self.iir_slow

        # Spatial Gradients Ix, Iy with Edge Contrast Sensitivity scaling
        g_scale = cfg.gradient_scale * 0.125
        ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * g_scale
        iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * g_scale

        # Structure Tensor 3x3 local neighborhood integration
        jxx = cv2.boxFilter(ix * ix, -1, (3, 3))
        jyy = cv2.boxFilter(iy * iy, -1, (3, 3))
        jxy = cv2.boxFilter(ix * iy, -1, (3, 3))

        # Rotation-Invariant Coherence Formula: C = ((Jxx - Jyy)^2 + 4Jxy^2) / ((Jxx + Jyy)^2 + eps)
        diff = jxx - jyy
        trace = jxx + jyy
        num = (diff * diff) + (4.0 * jxy * jxy)
        den = (trace * trace) + 1e-4

        coherence = np.clip(num / den, 0.0, 1.0)
        coherence_mask = np.where(coherence >= cfg.coherence_th, coherence, 0.0)

        # Motion Energy & Gross Motion Gate Evaluation
        motion_energy = float(np.mean(np.abs(bandpass)))
        is_gated = (cfg.gate_mode != 2) and (motion_energy > cfg.artifact_gate)

        effective_alpha = cfg.alpha
        if is_gated:
            if cfg.gate_mode == 1:  # Hard Freeze
                effective_alpha = 0.0
            else:                   # Soft AGC: quadratically scales down excess energy
                excess = motion_energy / max(1e-4, cfg.artifact_gate)
                effective_alpha = cfg.alpha / (1.0 + excess * excess)

        # Anti-Blooming Soft Limiter (Hyperbolic Tangent with expanded tau range)
        tau = max(1.0, cfg.soft_clamp)
        clamped_motion = tau * np.tanh(bandpass / tau)
        magnified_delta = effective_alpha * coherence_mask * clamped_motion

        # 1. Synthesize magnified output image (Color EVM)
        out_bgr = patch_bgr.astype(np.float32)
        out_bgr[:, :, 0] += magnified_delta * 1.15  # Blue channel
        out_bgr[:, :, 1] += magnified_delta * 0.95  # Green channel
        out_bgr[:, :, 2] += magnified_delta         # Red channel
        out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)

        # 2. Synthesize High-Gain Motion Band Relief (Phase-Contrast / Schlieren Embossed View)
        # Matches HTML app: diffVis = 128 + clampedMotion * 18
        diff_vis = np.clip(128.0 + clamped_motion * 18.0, 0, 255).astype(np.uint8)
        relief_bgr = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2BGR)

        # Respiratory scalar deconvolution over oriented chest
        motion_scalar = float(np.mean(coherence_mask * clamped_motion))

        return out_bgr, relief_bgr, coherence, motion_scalar, motion_energy, is_gated

class GoertzelFilterBank:
    """Per-frame incremental frequency estimator (O(1) per bin)."""
    def __init__(self, fps=30.0, min_rpm=6.0, max_rpm=42.0, num_bins=36):
        self.fps = fps
        self.freqs = np.linspace(min_rpm / 60.0, max_rpm / 60.0, num_bins)
        self.omegas = 2 * np.pi * self.freqs / fps
        self.coeffs = 2 * np.cos(self.omegas)
        self.s1 = np.zeros(num_bins, dtype=np.float32)
        self.s2 = np.zeros(num_bins, dtype=np.float32)
        self.smoothed_energy = np.zeros(num_bins, dtype=np.float32)
        self.alpha_smooth = 0.05  # exponential smoothing for frequency tracking

    def update(self, sample):
        s0 = sample + self.coeffs * self.s1 - self.s2
        self.s2 = self.s1
        self.s1 = s0
        energy = self.s1**2 + self.s2**2 - self.coeffs * self.s1 * self.s2
        self.smoothed_energy = (1 - self.alpha_smooth) * self.smoothed_energy + self.alpha_smooth * energy
        max_idx = np.argmax(self.smoothed_energy)
        return self.freqs[max_idx] * 60.0, self.smoothed_energy

class RespirationTelemetry:
    """Buffers 1D respiratory signal and calculates physiological metrics via Goertzel Filter."""
    def __init__(self, buffer_len=180, fps=30.0):
        self.buffer_len = buffer_len
        self.fps = fps
        self.raw_buffer = collections.deque([0.0]*buffer_len, maxlen=buffer_len)
        self.filtered_buffer = collections.deque([0.0]*buffer_len, maxlen=buffer_len)
        self.baseline = 0.0
        self.filtered_val = 0.0
        self.current_rpm = 0.0
        self.current_freq = 0.0
        self.snr_db = 0.0
        self.confidence = 0
        self.goertzel = GoertzelFilterBank(fps=fps)

    def push(self, sample, is_gated, gate_mode):
        if is_gated and gate_mode == 1:  # Hard freeze smooth decay
            self.filtered_val *= 0.92
            self.raw_buffer.append(self.baseline)
            self.filtered_buffer.append(self.filtered_val)
        else:
            self.baseline += (sample - self.baseline) * 0.04
            detrended = sample - self.baseline
            self.filtered_val += (detrended - self.filtered_val) * 0.38
            self.raw_buffer.append(sample)
            self.filtered_buffer.append(self.filtered_val)

        # Update Goertzel Filter Bank every frame (no FFT latency)
        rpm, energies = self.goertzel.update(self.filtered_val)
        self.current_rpm = rpm
        self.current_freq = rpm / 60.0
        
        peak_energy = np.max(energies)
        noise_floor = np.mean(energies) + 1e-5
        if peak_energy > 1e-4:
            self.snr_db = 10.0 * math.log10(max(1.0, peak_energy / noise_floor))
            self.confidence = int(np.clip(self.snr_db * 4.8, 10, 99))


def render_waveform_plot(telemetry, cfg, is_gated, width=400, height=130):
    """Draws real-time deconvolved respiration oscilloscope."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (12, 14, 20)

    for y in range(25, height, 25):
        cv2.line(canvas, (0, y), (width, y), (30, 36, 48), 1)

    pts = np.array(telemetry.filtered_buffer, dtype=np.float32)
    max_val = max(0.04, float(np.max(np.abs(pts))))
    center_y = height // 2
    scale_y = (height * 0.42) / max_val

    num_pts = len(pts)
    line_coords = []
    for i in range(num_pts):
        x = int(i * (width - 1) / (num_pts - 1))
        y = int(center_y - pts[i] * scale_y)
        y = np.clip(y, 2, height - 3)
        line_coords.append((x, y))

    color = (30, 180, 250) if not is_gated else (20, 160, 245)
    for i in range(1, len(line_coords)):
        cv2.line(canvas, line_coords[i - 1], line_coords[i], color, 2, cv2.LINE_AA)

    if line_coords:
        cv2.circle(canvas, line_coords[-1], 4, (0, 240, 168), -1)

    label = "DECONVOLVED S(t)" if not is_gated else "MOTION REJECT GATED"
    cv2.putText(canvas, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 180), 1, cv2.LINE_AA)
    return canvas


def render_controls_dashboard(cfg, telemetry, is_gated, width=460, height=460):
    """
    Renders an active graphical control surface into the 'Respiration Controls' window.
    This guarantees GTK/X11 on Linux paints the window and keeps trackbars responsive.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (14, 16, 24)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 36), (20, 26, 38), -1)
    cv2.line(panel, (0, 36), (width, 36), (0, 240, 168), 1)
    cv2.putText(panel, "EULERIAN RESPIRATION PARAMETERS", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 168), 1, cv2.LINE_AA)

    # Parameter card renderer helper
    def draw_param(y_pos, name, val_str, norm_val, key_str, bar_color):
        cv2.putText(panel, name, (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, val_str, (215, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"[{key_str}]", (width - 68, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 140, 160), 1, cv2.LINE_AA)

        bar_y = y_pos + 5
        cv2.rectangle(panel, (16, bar_y), (width - 16, bar_y + 5), (30, 38, 50), -1)
        fill_w = int((width - 32) * np.clip(norm_val, 0.0, 1.0))
        cv2.rectangle(panel, (16, bar_y), (16 + fill_w, bar_y + 5), bar_color, -1)

    draw_param(56, "Gain Alpha (a)", f"{cfg.alpha:.0f}x", cfg.alpha / 80.0, "A / Z", (0, 240, 168))
    draw_param(92, "Edge Contrast (S)", f"{cfg.gradient_scale:.1f}x", (cfg.gradient_scale - 0.5) / 2.5, "E / W", (255, 180, 50))
    draw_param(128, "Temporal Gamma (g)", f"{cfg.gamma_fast:.2f}", (cfg.gamma_fast - 0.02) / 0.48, "T / G", (0, 220, 255))
    draw_param(164, "Anti-Bloom Tau (t)", f"{cfg.soft_clamp:.1f}", cfg.soft_clamp / 30.0, "D / C", (0, 210, 255))
    draw_param(200, "Artifact Gate (E)", f"{cfg.artifact_gate:.2f}", cfg.artifact_gate / 1.0, "S / X", (0, 165, 255))
    draw_param(236, "Coherence Gate", f"{cfg.coherence_th:.2f}", cfg.coherence_th / 0.40, "F / V", (180, 220, 50))

    mode_names = ["SOFT AGC", "HARD FREEZE", "BYPASS"]
    mode_colors = [(0, 240, 168), (0, 140, 255), (180, 180, 180)]
    cur_m = cfg.gate_mode
    draw_param(272, "Gate Strategy", mode_names[cur_m], (cur_m + 1) / 3.0, "M", mode_colors[cur_m])

    chest_modes = ["MOTION BAND", "COLOR EVM", "COHERENCE"]
    chest_colors = [(255, 200, 50), (0, 240, 168), (0, 180, 255)]
    cur_cm = cfg.chest_view_mode
    overlay_txt = "ON" if cfg.overlay_chest else "OFF"
    draw_param(308, f"Chest View ({overlay_txt})", chest_modes[cur_cm], (cur_cm + 1) / 3.0, "B / O", chest_colors[cur_cm])

    # Live Telemetry Snapshot Card
    cv2.rectangle(panel, (14, 335), (width - 14, 400), (22, 28, 42), -1)
    cv2.rectangle(panel, (14, 335), (width - 14, 400), (45, 55, 75), 1)

    rpm_text = f"{telemetry.current_rpm:.1f} RPM" if telemetry.confidence >= 25 else "--.- RPM"
    gate_str = "MOTION GATED" if is_gated else "TELEMETRY STABLE"
    gate_col = (0, 140, 255) if is_gated else (0, 240, 168)

    cv2.putText(panel, f"RATE: {rpm_text}", (24, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 168), 2, cv2.LINE_AA)
    cv2.putText(panel, f"STATUS: {gate_str}", (215, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.40, gate_col, 1, cv2.LINE_AA)
    cv2.putText(panel, f"SNR: {telemetry.snr_db:.1f} dB | CONF: {telemetry.confidence}%", (24, 388),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 180, 200), 1, cv2.LINE_AA)

    # Shortcut Hints Footer
    cv2.putText(panel, "[B] Mode | [O] Overlay | [E/W] Contrast | [T/G] Gamma | [A/Z] Alpha | [Q] Quit",
                (14, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 150, 175), 1, cv2.LINE_AA)

    return panel


def main():
    print("=" * 76)
    print("  Rotation-Invariant Real-Time Eulerian Respiration Monitor")
    print("  Optimized for Linux (GTK/X11) with MediaPipe Pose Guidance")
    print("=" * 76)
    print(" Press 'q' or ESC to exit.")
    print(" Adjust parameters using trackbars or keyboard hotkeys.\n")

    cfg = RespirationConfig()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam (/dev/video0). Check camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    extractor = MediaPipeChestExtractor()
    engine = EulerianEngine(cfg.patch_width, cfg.patch_height)
    telemetry = RespirationTelemetry(buffer_len=cfg.buffer_len, fps=30.0)

    main_win = "Eulerian Chest Monitor"
    ctrl_win = "Respiration Controls"

    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(main_win, 860, 680)

    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctrl_win, 460, 520)

    # Arrange windows side-by-side on Linux desktops
    try:
        cv2.moveWindow(main_win, 40, 40)
        cv2.moveWindow(ctrl_win, 920, 40)
    except Exception:
        pass

    # OpenCV Interactive Trackbars attached to 'Respiration Controls'
    cv2.createTrackbar("Gain Alpha", ctrl_win, int(cfg.alpha), 80, lambda v: setattr(cfg, 'alpha', max(1.0, float(v))))
    cv2.createTrackbar("Contrast Sens (x10)", ctrl_win, int(cfg.gradient_scale * 10), 30, lambda v: setattr(cfg, 'gradient_scale', max(0.5, v / 10.0)))
    cv2.createTrackbar("Temporal Gamma (x100)", ctrl_win, int(cfg.gamma_fast * 100), 50, lambda v: setattr(cfg, 'gamma_fast', max(0.02, v / 100.0)))
    cv2.createTrackbar("Anti-Bloom Tau", ctrl_win, int(cfg.soft_clamp), 30, lambda v: setattr(cfg, 'soft_clamp', max(1.0, float(v))))
    cv2.createTrackbar("Artifact Gate (x100)", ctrl_win, int(cfg.artifact_gate * 100), 100, lambda v: setattr(cfg, 'artifact_gate', max(0.01, v / 100.0)))
    cv2.createTrackbar("Coherence Gate (x100)", ctrl_win, int(cfg.coherence_th * 100), 40, lambda v: setattr(cfg, 'coherence_th', v / 100.0))
    cv2.createTrackbar("Gate Mode (0:Soft 1:Hard 2:Byp)", ctrl_win, cfg.gate_mode, 2, lambda v: setattr(cfg, 'gate_mode', int(v)))
    cv2.createTrackbar("Chest View (0:Rel 1:EVM 2:Coh)", ctrl_win, cfg.chest_view_mode, 2, lambda v: setattr(cfg, 'chest_view_mode', int(v)))

    prev_time = time.time()
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        t_now = time.time()
        fps = 1.0 / max(1e-4, t_now - prev_time)
        prev_time = t_now

        # 1. Pose Inference and Anatomical Chest Patch Extraction
        patch, quad_pts, has_pose, label_text, all_landmarks, sternum_info = extractor.extract_patch(
            frame, cfg.patch_width, cfg.patch_height
        )

        # 2. Eulerian Amplification, Motion Band Relief & Motion Energy Analysis
        mag_patch, relief_patch, coherence, scalar, energy, is_gated = engine.process(patch, cfg)

        # 3. Respiration Telemetry & Spectral Analysis (Goertzel handles per-frame updates)
        telemetry.push(scalar, is_gated, cfg.gate_mode)

        # 4. Viewport Skeleton and Reticle Composition
        display_frame = frame.copy()

        # Render full 33-point pose skeleton
        if cfg.show_skeleton and all_landmarks:
            draw_full_skeleton(display_frame, all_landmarks)

        # Draw stabilized chest extraction polygon
        quad_int = quad_pts.astype(np.int32)
        poly_color = (0, 240, 168) if has_pose else (0, 140, 255)

        # 4a. Warp selected deconvolution mode directly onto the chest quad in live video
        if cfg.overlay_chest and has_pose and quad_pts is not None:
            if cfg.chest_view_mode == 0:
                selected_patch = relief_patch
            elif cfg.chest_view_mode == 1:
                selected_patch = mag_patch
            else:
                pip_coh_temp = (coherence * 255).astype(np.uint8)
                selected_patch = cv2.applyColorMap(pip_coh_temp, cv2.COLORMAP_VIRIDIS)

            # Map normalized patch coordinates to live chest quad
            src_pts = np.array([
                [0, 0], [cfg.patch_width - 1, 0],
                [cfg.patch_width - 1, cfg.patch_height - 1], [0, cfg.patch_height - 1]
            ], dtype=np.float32)
            try:
                inv_mat = cv2.getPerspectiveTransform(src_pts, quad_pts)
                warped = cv2.warpPerspective(selected_patch, inv_mat, (display_frame.shape[1], display_frame.shape[0]))

                mask = np.zeros((display_frame.shape[0], display_frame.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(mask, quad_int, 255)
                mask_3ch = cv2.merge([mask, mask, mask])

                alpha_blend = 0.88
                blended = cv2.addWeighted(warped, alpha_blend, display_frame, 1.0 - alpha_blend, 0)
                display_frame = np.where(mask_3ch > 0, blended, display_frame)
            except Exception:
                pass

        cv2.polylines(display_frame, [quad_int], isClosed=True, color=poly_color, thickness=2, lineType=cv2.LINE_AA)

        for pt in quad_int:
            cv2.circle(display_frame, tuple(pt), 4, (0, 255, 255), -1, cv2.LINE_AA)

        # Draw torso vector & sternum center if available
        if sternum_info is not None:
            mid_sh, stern_center, torso_vec = sternum_info
            p_sh = (int(mid_sh[0]), int(mid_sh[1]))
            p_st = (int(stern_center[0]), int(stern_center[1]))
            cv2.circle(display_frame, p_st, 7, (0, 240, 168), -1, cv2.LINE_AA)
            cv2.circle(display_frame, p_st, 10, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.arrowedLine(display_frame, p_sh, (int(p_sh[0] + torso_vec[0]), int(p_sh[1] + torso_vec[1])),
                            (0, 255, 200), 2, cv2.LINE_AA, tipLength=0.2)

        # Pose Tracking Status Label
        cv2.putText(display_frame, label_text, (quad_int[0][0], max(24, quad_int[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, poly_color, 1, cv2.LINE_AA)

        # 5. Picture-in-Picture Panels (Motion Band Relief + EVM + Coherence + Oscilloscope)
        h_pip = 120
        # 5a. Motion Band Relief (See-through paper / Micro-Gradient mode)
        pip_relief = cv2.resize(relief_patch, (150, h_pip))
        rel_border = (0, 240, 168) if cfg.chest_view_mode == 0 else (50, 60, 75)
        cv2.rectangle(pip_relief, (0, 0), (149, h_pip - 1), rel_border, 2 if cfg.chest_view_mode == 0 else 1)
        cv2.putText(pip_relief, "MOTION BAND", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 168), 1)

        # 5b. Eulerian Magnification (Color EVM)
        pip_mag = cv2.resize(mag_patch, (150, h_pip))
        mag_border = (0, 240, 168) if cfg.chest_view_mode == 1 else (50, 60, 75)
        cv2.rectangle(pip_mag, (0, 0), (149, h_pip - 1), mag_border, 2 if cfg.chest_view_mode == 1 else 1)
        cv2.putText(pip_mag, f"EVM {cfg.alpha:.0f}x", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 168), 1)

        # 5c. Structure Tensor Coherence
        pip_coh = (coherence * 255).astype(np.uint8)
        pip_coh_bgr = cv2.applyColorMap(pip_coh, cv2.COLORMAP_VIRIDIS)
        pip_coh_bgr = cv2.resize(pip_coh_bgr, (150, h_pip))
        coh_border = (0, 240, 168) if cfg.chest_view_mode == 2 else (50, 60, 75)
        cv2.rectangle(pip_coh_bgr, (0, 0), (149, h_pip - 1), coh_border, 2 if cfg.chest_view_mode == 2 else 1)
        cv2.putText(pip_coh_bgr, "COHERENCE C", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # 5d. Oscilloscope (fills remaining width to total 640px)
        wave_img = render_waveform_plot(telemetry, cfg, is_gated, width=190, height=h_pip)

        # Top Diagnostic HUD Banner
        hud_bar = np.zeros((46, display_frame.shape[1], 3), dtype=np.uint8)
        hud_bar[:] = (10, 12, 18)

        rpm_str = f"{telemetry.current_rpm:.1f} RPM" if telemetry.confidence >= 25 else "--.- RPM"
        gate_status = "MOTION GATED" if is_gated else "STABLE"
        gate_color = (0, 140, 255) if is_gated else (0, 240, 168)

        cv2.putText(hud_bar, f"RESPIRATION: {rpm_str}", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 240, 168), 2)
        cv2.putText(hud_bar, f"BACKEND: {extractor.backend}", (310, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 200, 220), 1)
        cv2.putText(hud_bar, f"STATUS: {gate_status} ({fps:.0f} FPS)", (310, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gate_color, 1)

        # Assemble Full Main Viewport
        top_view = np.vstack([hud_bar, display_frame])
        bottom_strip = np.hstack([pip_relief, pip_mag, pip_coh_bgr, wave_img])
        if bottom_strip.shape[1] != top_view.shape[1]:
            bottom_strip = cv2.resize(bottom_strip, (top_view.shape[1], h_pip))

        final_composite = np.vstack([top_view, bottom_strip])

        # Render Controls Dashboard Image (ensures GTK window is mapped on Linux)
        ctrl_canvas = render_controls_dashboard(cfg, telemetry, is_gated, width=460, height=460)

        # Present frames to both windows
        cv2.imshow(main_win, final_composite)
        cv2.imshow(ctrl_win, ctrl_canvas)

        # Keyboard event handler & hotkeys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in [ord('b'), ord('B'), 9]:  # 'b' or TAB: switch chest view mode
            cfg.chest_view_mode = (cfg.chest_view_mode + 1) % 3
            cv2.setTrackbarPos("Chest View (0:Rel 1:EVM 2:Coh)", ctrl_win, cfg.chest_view_mode)
        elif key in [ord('o'), ord('O')]:
            cfg.overlay_chest = not cfg.overlay_chest
        elif key in [ord('e'), ord('E')]:
            cfg.gradient_scale = min(3.0, cfg.gradient_scale + 0.1)
            cv2.setTrackbarPos("Contrast Sens (x10)", ctrl_win, int(cfg.gradient_scale * 10))
        elif key in [ord('w'), ord('W')]:
            cfg.gradient_scale = max(0.5, cfg.gradient_scale - 0.1)
            cv2.setTrackbarPos("Contrast Sens (x10)", ctrl_win, int(cfg.gradient_scale * 10))
        elif key in [ord('t'), ord('T')]:
            cfg.gamma_fast = min(0.50, cfg.gamma_fast + 0.02)
            cv2.setTrackbarPos("Temporal Gamma (x100)", ctrl_win, int(cfg.gamma_fast * 100))
        elif key in [ord('g'), ord('G')]:
            cfg.gamma_fast = max(0.02, cfg.gamma_fast - 0.02)
            cv2.setTrackbarPos("Temporal Gamma (x100)", ctrl_win, int(cfg.gamma_fast * 100))
        elif key in [ord('a'), ord('A')]:
            cfg.alpha = min(80.0, cfg.alpha + 2.0)
            cv2.setTrackbarPos("Gain Alpha", ctrl_win, int(cfg.alpha))
        elif key in [ord('z'), ord('Z')]:
            cfg.alpha = max(1.0, cfg.alpha - 2.0)
            cv2.setTrackbarPos("Gain Alpha", ctrl_win, int(cfg.alpha))
        elif key in [ord('s'), ord('S')]:
            cfg.artifact_gate = min(1.0, cfg.artifact_gate + 0.02)
            cv2.setTrackbarPos("Artifact Gate (x100)", ctrl_win, int(cfg.artifact_gate * 100))
        elif key in [ord('x'), ord('X')]:
            cfg.artifact_gate = max(0.01, cfg.artifact_gate - 0.02)
            cv2.setTrackbarPos("Artifact Gate (x100)", ctrl_win, int(cfg.artifact_gate * 100))
        elif key in [ord('d'), ord('D')]:
            cfg.soft_clamp = min(30.0, cfg.soft_clamp + 1.0)
            cv2.setTrackbarPos("Anti-Bloom Tau", ctrl_win, int(cfg.soft_clamp))
        elif key in [ord('c'), ord('C')]:
            cfg.soft_clamp = max(1.0, cfg.soft_clamp - 1.0)
            cv2.setTrackbarPos("Anti-Bloom Tau", ctrl_win, int(cfg.soft_clamp))
        elif key in [ord('f'), ord('F')]:
            cfg.coherence_th = min(0.40, cfg.coherence_th + 0.02)
            cv2.setTrackbarPos("Coherence Gate (x100)", ctrl_win, int(cfg.coherence_th * 100))
        elif key in [ord('v'), ord('V')]:
            cfg.coherence_th = max(0.00, cfg.coherence_th - 0.02)
            cv2.setTrackbarPos("Coherence Gate (x100)", ctrl_win, int(cfg.coherence_th * 100))
        elif key in [ord('m'), ord('M')]:
            cfg.gate_mode = (cfg.gate_mode + 1) % 3
            cv2.setTrackbarPos("Gate Mode (0:Soft 1:Hard 2:Byp)", ctrl_win, cfg.gate_mode)
        elif key in [ord('h'), ord('H')]:
            cfg.show_skeleton = not cfg.show_skeleton
        elif key in [ord('r'), ord('R')]:
            cfg.alpha = 35.0
            cfg.gradient_scale = 1.2
            cfg.gamma_fast = 0.20
            cfg.soft_clamp = 8.0
            cfg.artifact_gate = 0.22
            cfg.coherence_th = 0.08
            cfg.gate_mode = 0
            cfg.chest_view_mode = 0
            cv2.setTrackbarPos("Gain Alpha", ctrl_win, int(cfg.alpha))
            cv2.setTrackbarPos("Contrast Sens (x10)", ctrl_win, int(cfg.gradient_scale * 10))
            cv2.setTrackbarPos("Temporal Gamma (x100)", ctrl_win, int(cfg.gamma_fast * 100))
            cv2.setTrackbarPos("Anti-Bloom Tau", ctrl_win, int(cfg.soft_clamp))
            cv2.setTrackbarPos("Artifact Gate (x100)", ctrl_win, int(cfg.artifact_gate * 100))
            cv2.setTrackbarPos("Coherence Gate (x100)", ctrl_win, int(cfg.coherence_th * 100))
            cv2.setTrackbarPos("Gate Mode (0:Soft 1:Hard 2:Byp)", ctrl_win, cfg.gate_mode)
            cv2.setTrackbarPos("Chest View (0:Rel 1:EVM 2:Coh)", ctrl_win, cfg.chest_view_mode)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()