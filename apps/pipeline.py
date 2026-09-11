import cv2
import numpy as np
import time
import threading
import queue
import collections
import math
import mediapipe as mp

class GoertzelFilterBank:
    def __init__(self, fps, min_rpm=5, max_rpm=30, num_bins=25):
        self.fps = fps
        self.freqs = np.linspace(min_rpm / 60.0, max_rpm / 60.0, num_bins)
        
        # Precompute constants
        self.omegas = 2 * np.pi * self.freqs / fps
        self.coeffs = 2 * np.cos(self.omegas)
        
        # State variables
        self.s1 = np.zeros(num_bins)
        self.s2 = np.zeros(num_bins)
        
        self.smoothed_energy = np.zeros(num_bins)
        self.alpha_smooth = 0.1  # exponential smoothing for energy

    def update(self, sample):
        # Update state for each frequency
        s0 = sample + self.coeffs * self.s1 - self.s2
        self.s2 = self.s1
        self.s1 = s0
        
        # Compute current energy (sliding Goertzel magnitude squared)
        energy = self.s1**2 + self.s2**2 - self.coeffs * self.s1 * self.s2
        
        self.smoothed_energy = (1 - self.alpha_smooth) * self.smoothed_energy + self.alpha_smooth * energy
        
        # Find dominant frequency
        max_idx = np.argmax(self.smoothed_energy)
        return self.freqs[max_idx] * 60.0

def compute_structure_tensor_lambda1(image):
    # Same as synthetic validation
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    Ixx = cv2.GaussianBlur(gx**2, (5, 5), 1.0)
    Iyy = cv2.GaussianBlur(gy**2, (5, 5), 1.0)
    Ixy = cv2.GaussianBlur(gx*gy, (5, 5), 1.0)
    
    trace = Ixx + Iyy
    det_term = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)
    lambda1 = 0.5 * (trace + det_term)
    
    return np.mean(lambda1)

class RespiratoryTracker:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        
        # Shared UI State
        self.lock = threading.Lock()
        self.ui_state = {
            'rpm': 0.0,
            'lambda_history': collections.deque(maxlen=100),
            'degraded_rate': False,
            'mode': 'initializing', # 'texture' or 'displacement'
            'polarity': 1,
            'is_skin': False,
            'latest_frame': None,
            'roi_box': None,
            'goertzel_freqs': None,
            'goertzel_energy': None
        }
        
    def capture_thread(self):
        cap = cv2.VideoCapture(0)
        # Verify webcam opened successfully
        if not cap.isOpened():
            self.running = False
            return
            
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.ui_state['latest_frame'] = frame.copy()
            
            # Overwrite policy
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(frame)
                
        cap.release()

    def processing_thread(self):
        target_fps = 30.0
        frame_time = 1.0 / target_fps
        
        goertzel = GoertzelFilterBank(fps=target_fps)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Calibration state
        calibration_frames = 0
        calibration_duration = 15 * int(target_fps)
        calib_lambda = collections.deque(maxlen=30 * int(target_fps))
        calib_centroid = collections.deque(maxlen=30 * int(target_fps))
        frames_since_snr_check = 0
        snr_check_interval = 30 * int(target_fps)
        
        # Skin classification cache
        last_skin_classification_time = 0
        is_skin = False
        
        # Motion artifact state
        centroid_history = collections.deque(maxlen=15)
        motion_threshold = 10.0
        
        while self.running:
            start_t = time.time()
            
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            roi_box = None
            cy_val = 0.0
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Check visibility
                if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
                    x1 = int(min(l_sh.x, r_sh.x) * w)
                    x2 = int(max(l_sh.x, r_sh.x) * w)
                    
                    if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
                        y1 = int(min(l_sh.y, r_sh.y) * h)
                        y2 = int(max(l_hip.y, r_hip.y) * h)
                    else:
                        y1 = int(min(l_sh.y, r_sh.y) * h)
                        y2 = int(y1 + (x2 - x1)) # Rough square if hips missing
                        
                    # Expand slightly and clamp
                    padding_x = int((x2 - x1) * 0.1)
                    padding_y = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - padding_x)
                    x2 = min(w, x2 + padding_x)
                    y1 = max(0, y1 - padding_y)
                    y2 = min(h, y2 + padding_y)
                    
                    if x2 > x1 and y2 > y1:
                        roi_box = (x1, y1, x2, y2)
                        cy_val = (y1 + y2) / 2.0
            
            if roi_box is None:
                roi_box = (w//4, h//4, 3*w//4, 3*h//4)
                cy_val = h / 2.0
                cx_val = w / 2.0
            else:
                x1, y1, x2, y2 = roi_box
                cx_val = (x1 + x2) / 2.0
                
            centroid_history.append((cx_val, cy_val))
            
            if len(centroid_history) == centroid_history.maxlen:
                c_arr = np.array(centroid_history)
                std_x = np.std(c_arr[:, 0])
                std_y = np.std(c_arr[:, 1])
                
                if std_x > motion_threshold or std_y > motion_threshold:
                    self.ui_state['mode'] = 'motion artifact'
                    # flush goertzel buffers per plan
                    goertzel.s1[:] = 0
                    goertzel.s2[:] = 0
                    # Skip processing this frame
                    elapsed = time.time() - start_t
                    if elapsed > frame_time:
                        self.ui_state['degraded_rate'] = True
                    else:
                        self.ui_state['degraded_rate'] = False
                        time.sleep(frame_time - elapsed)
                    continue
                    
            if not results.pose_landmarks:
                self.ui_state['mode'] = 'face camera'
                goertzel.s1[:] = 0
                goertzel.s2[:] = 0
                
            x1, y1, x2, y2 = roi_box
            roi = frame[y1:y2, x1:x2]
            
            # 1. Skin vs Clothing Classification (cached)
            current_time = time.time()
            if current_time - last_skin_classification_time > 1.0 and roi.size > 0:
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)
                skin_ratio = np.sum(mask > 0) / (mask.size + 1e-5)
                is_skin = skin_ratio > 0.6
                last_skin_classification_time = current_time
                self.ui_state['is_skin'] = is_skin
            
            if roi.size == 0:
                continue
                
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_downsampled = cv2.resize(gray_roi, (0,0), fx=0.5, fy=0.5)
            
            # 2. Extract signals
            l1_val = compute_structure_tensor_lambda1(roi_downsampled)
            
            if calibration_frames < calibration_duration:
                # Phase 0: Calibration
                self.ui_state['mode'] = 'calibrating'
                calib_lambda.append(l1_val)
                calib_centroid.append(cy_val)
                calibration_frames += 1
                
                if calibration_frames == calibration_duration:
                    l_arr = np.array(calib_lambda)
                    c_arr = np.array(calib_centroid)
                    l_norm = l_arr - np.mean(l_arr)
                    c_norm = c_arr - np.mean(c_arr)
                    
                    l_std = np.std(l_arr)
                    c_std = np.std(c_arr)
                    
                    if l_std < 1e-5 or c_std < 1e-5:
                        self.ui_state['polarity'] = 1
                        self.ui_state['mode'] = 'displacement'
                    else:
                        corr = np.correlate(l_norm, c_norm)[0]
                        self.ui_state['polarity'] = 1 if corr > 0 else -1
                        
                        freqs = np.fft.rfftfreq(len(l_norm), d=1.0/target_fps)
                        fft_mags = np.abs(np.fft.rfft(l_norm))
                        valid_bins = (freqs >= 0.08) & (freqs <= 0.5)
                        snr = np.sum(fft_mags[valid_bins]) / (np.sum(fft_mags) + 1e-5)
                        
                        threshold = 0.4 if is_skin else 0.2
                        if snr > threshold:
                            self.ui_state['mode'] = 'texture'
                        else:
                            self.ui_state['mode'] = 'displacement'
                    frames_since_snr_check = 0
            else:
                calib_lambda.append(l1_val)
                calib_centroid.append(cy_val)
                frames_since_snr_check += 1
                
                if frames_since_snr_check >= snr_check_interval:
                    l_arr = np.array(calib_lambda)
                    l_norm = l_arr - np.mean(l_arr)
                    freqs = np.fft.rfftfreq(len(l_norm), d=1.0/target_fps)
                    fft_mags = np.abs(np.fft.rfft(l_norm))
                    valid_bins = (freqs >= 0.08) & (freqs <= 0.5)
                    snr = np.sum(fft_mags[valid_bins]) / (np.sum(fft_mags) + 1e-5)
                    
                    threshold = 0.4 if is_skin else 0.2
                    if snr > threshold:
                        self.ui_state['mode'] = 'texture'
                    else:
                        self.ui_state['mode'] = 'displacement'
                    
                    frames_since_snr_check = 0

                if self.ui_state['mode'] == 'texture':
                    signed_val = l1_val * self.ui_state['polarity']
                    rpm = goertzel.update(signed_val)
                    self.ui_state['rpm'] = rpm
                else:
                    rpm = goertzel.update(cy_val)
                    self.ui_state['rpm'] = rpm
                
            with self.lock:
                self.ui_state['lambda_history'].append(l1_val)
                self.ui_state['roi_box'] = roi_box
                self.ui_state['goertzel_freqs'] = goertzel.freqs.copy()
                self.ui_state['goertzel_energy'] = goertzel.smoothed_energy.copy()
            
            elapsed = time.time() - start_t
            if elapsed > frame_time:
                self.ui_state['degraded_rate'] = True
            else:
                self.ui_state['degraded_rate'] = False
                time.sleep(frame_time - elapsed)
                
        pose.close()

    def ui_thread(self):
        cv2.namedWindow('Respiratory Tracker', cv2.WINDOW_NORMAL)
        while self.running:
            frame = self.ui_state.get('latest_frame', None)
            if frame is None:
                time.sleep(0.03)
                continue
            
            disp = frame.copy()
            h, w = disp.shape[:2]
            
            mode = self.ui_state['mode']
            rpm = self.ui_state['rpm']
            is_skin = self.ui_state['is_skin']
            
            cv2.putText(disp, f"Mode: {mode} (Skin: {is_skin})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if mode not in ('calibrating', 'initializing'):
                cv2.putText(disp, f"RPM: {rpm:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            if self.ui_state['degraded_rate']:
                cv2.putText(disp, "SLOW", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            roi_box = self.ui_state.get('roi_box')
            if roi_box is not None:
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
            with self.lock:
                history = list(self.ui_state['lambda_history'])
                goertzel_freqs = self.ui_state.get('goertzel_freqs')
                goertzel_energy = self.ui_state.get('goertzel_energy')
                
            if len(history) > 0:
                h_min = min(history)
                h_max = max(history)
                
                graph_w = 200
                graph_h = 100
                graph_x = w - graph_w - 10
                graph_y = h - graph_h - 10
                
                cv2.rectangle(disp, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (50, 50, 50), -1)
                
                if h_max > h_min:
                    points = []
                    x_denominator = max(1, len(history) - 1)
                    for i, val in enumerate(history):
                        px = int(graph_x + (i / x_denominator) * graph_w)
                        py = int(graph_y + graph_h - ((val - h_min) / (h_max - h_min)) * graph_h)
                        points.append((px, py))
                    
                    if len(points) > 1:
                        for i in range(1, len(points)):
                            cv2.line(disp, points[i-1], points[i], (0, 255, 255), 2)
                            
            if goertzel_freqs is not None and goertzel_energy is not None:
                spec_w = 200
                spec_h = 100
                spec_x = 10
                spec_y = h - spec_h - 10
                
                cv2.rectangle(disp, (spec_x, spec_y), (spec_x + spec_w, spec_y + spec_h), (50, 50, 50), -1)
                e_max = np.max(goertzel_energy)
                if e_max > 0:
                    num_bins = len(goertzel_energy)
                    bin_w = spec_w / num_bins
                    for i, e in enumerate(goertzel_energy):
                        bar_h = int((e / e_max) * spec_h)
                        bx1 = int(spec_x + i * bin_w)
                        by1 = spec_y + spec_h - bar_h
                        bx2 = int(spec_x + (i + 1) * bin_w)
                        by2 = spec_y + spec_h
                        color = (0, 0, 255) if e == e_max else (255, 0, 255)
                        cv2.rectangle(disp, (bx1, by1), (bx2, by2), color, -1)
                        
            cv2.imshow('Respiratory Tracker', disp)
            if cv2.waitKey(33) & 0xFF == 27:
                self.running = False
                break
                
        cv2.destroyAllWindows()

    def start(self):
        t1 = threading.Thread(target=self.capture_thread, daemon=True)
        t2 = threading.Thread(target=self.processing_thread, daemon=True)
        t3 = threading.Thread(target=self.ui_thread, daemon=True)
        t1.start()
        t2.start()
        t3.start()
        
        try:
            while True:
                time.sleep(0.1)
                if not self.running:
                    break
        except KeyboardInterrupt:
            self.running = False
        finally:
            t1.join()
            t2.join()
            t3.join()

if __name__ == "__main__":
    tracker = RespiratoryTracker()
    tracker.start()
