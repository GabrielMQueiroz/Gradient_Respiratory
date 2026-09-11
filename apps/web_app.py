import os
import cv2
import time
import json
import threading
from flask import Flask, render_template, Response, jsonify, request
import numpy as np

# Import the core classes from the original pose deconvolution script
try:
    from .mediapipe_chest_pose_deconvolution import (
        RespirationConfig,
        MediaPipeChestExtractor,
        EulerianEngine,
        RespirationTelemetry,
        draw_full_skeleton
    )
except (ImportError, ValueError):
    from mediapipe_chest_pose_deconvolution import (
        RespirationConfig,
        MediaPipeChestExtractor,
        EulerianEngine,
        RespirationTelemetry,
        draw_full_skeleton
    )

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Initialize global Eulerian Engine instances
cfg = RespirationConfig()
extractor = MediaPipeChestExtractor()
engine = EulerianEngine(cfg.patch_width, cfg.patch_height)
telemetry = RespirationTelemetry(buffer_len=cfg.buffer_len, fps=30.0)

# Global telemetry state shared between the background processing loop and the Flask API
telemetry_data = {
    "rpm": 0.0,
    "confidence": 0,
    "gate_status": False,
    "motion_energy": 0.0,
    "snr": 0.0,
    "fps": 0,
    "coherence": 0.0,
    "waveform": [],
    "freqs": [],
    "energies": []
}

# The video capture lock and frame buffer
frame_lock = threading.Lock()
current_frame_jpg = None

def processing_loop():
    global current_frame_jpg
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Check camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        t_now = time.time()
        fps = 1.0 / max(1e-4, t_now - prev_time)
        prev_time = t_now
        
        # 1. Pose Inference and Anatomical Chest Patch Extraction
        patch, quad_pts, has_pose, label_text, all_landmarks, sternum_info = extractor.extract_patch(
            frame, cfg.patch_width, cfg.patch_height
        )
        
        # 2. Eulerian Amplification & Motion Energy Analysis
        mag_patch, coherence, scalar, energy, is_gated = engine.process(patch, cfg)
        
        # 3. Respiration Telemetry
        telemetry.push(scalar, is_gated, cfg.gate_mode)
        
        # Update shared telemetry dictionary for the web dashboard API
        telemetry_data["rpm"] = telemetry.current_rpm
        telemetry_data["confidence"] = telemetry.confidence
        telemetry_data["gate_status"] = is_gated
        telemetry_data["motion_energy"] = energy
        telemetry_data["snr"] = telemetry.snr_db
        telemetry_data["fps"] = int(fps)
        telemetry_data["coherence"] = float(np.mean(coherence))
        telemetry_data["waveform"] = list(telemetry.filtered_buffer)
        telemetry_data["freqs"] = list(telemetry.goertzel.freqs * 60.0) # in RPM
        telemetry_data["energies"] = list(telemetry.goertzel.smoothed_energy)
        
        # 4. Viewport skeleton and reticle composition
        display_frame = frame.copy()
        
        if cfg.show_skeleton and all_landmarks:
            draw_full_skeleton(display_frame, all_landmarks)
            
        quad_int = quad_pts.astype(np.int32)
        poly_color = (0, 240, 168) if has_pose else (0, 140, 255)
        cv2.polylines(display_frame, [quad_int], isClosed=True, color=poly_color, thickness=2, lineType=cv2.LINE_AA)
        
        for pt in quad_int:
            cv2.circle(display_frame, tuple(pt), 4, (0, 255, 255), -1, cv2.LINE_AA)
            
        if sternum_info is not None:
            mid_sh, stern_center, torso_vec = sternum_info
            p_sh = (int(mid_sh[0]), int(mid_sh[1]))
            p_st = (int(stern_center[0]), int(stern_center[1]))
            cv2.circle(display_frame, p_st, 7, (0, 240, 168), -1, cv2.LINE_AA)
            cv2.arrowedLine(display_frame, p_sh, (int(p_sh[0] + torso_vec[0]), int(p_sh[1] + torso_vec[1])),
                            (0, 255, 200), 2, cv2.LINE_AA, tipLength=0.2)
                            
        # Pose Tracking Status Label
        cv2.putText(display_frame, label_text, (quad_int[0][0], max(24, quad_int[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, poly_color, 1, cv2.LINE_AA)

        # Picture-in-Picture Panels (Magnified Patch + Coherence)
        pip_patch = cv2.resize(mag_patch, (160, 120))
        cv2.rectangle(pip_patch, (0, 0), (159, 119), (0, 240, 168), 1)
        cv2.putText(pip_patch, f"EVM {cfg.alpha:.0f}x", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 168), 1)

        pip_coh = (coherence * 255).astype(np.uint8)
        pip_coh_bgr = cv2.applyColorMap(pip_coh, cv2.COLORMAP_VIRIDIS)
        pip_coh_bgr = cv2.resize(pip_coh_bgr, (160, 120))
        cv2.rectangle(pip_coh_bgr, (0, 0), (159, 119), (180, 180, 40), 1)
        cv2.putText(pip_coh_bgr, "COHERENCE C(x,y)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        # Assemble composite display (putting PiP into the main frame)
        h, w = display_frame.shape[:2]
        display_frame[h-120:h, 0:160] = pip_patch
        display_frame[h-120:h, 160:320] = pip_coh_bgr

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            with frame_lock:
                current_frame_jpg = buffer.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        with frame_lock:
            frame = current_frame_jpg
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def get_telemetry():
    return jsonify(telemetry_data)

@app.route('/update_param', methods=['POST'])
def update_param():
    data = request.json
    if 'alpha' in data:
        cfg.alpha = float(data['alpha'])
    if 'motion_gate' in data:
        cfg.artifact_gate = float(data['motion_gate'])
    if 'gate_mode' in data:
        cfg.gate_mode = int(data['gate_mode'])
    if 'soft_clamp' in data:
        cfg.soft_clamp = float(data['soft_clamp'])
    if 'coherence_th' in data:
        cfg.coherence_th = float(data['coherence_th'])
    if 'bandpass_gamma' in data:
        cfg.gamma_fast = float(data['bandpass_gamma'])
    
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Start the background OpenCV processing loop
    t = threading.Thread(target=processing_loop)
    t.daemon = True
    t.start()
    
    # Start the Flask web server
    print("=========================================================================")
    print(" Flask Eulerian Deconvolution UI Server Started")
    print(" Access the dashboard at: http://127.0.0.1:5000")
    print("=========================================================================")
    app.run(host='0.0.0.0', port=5000, debug=False)
