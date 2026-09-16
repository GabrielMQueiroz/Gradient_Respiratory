import os
import sys
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

# Import the existing live classes so we don't duplicate code
from mediapipe_chest_pose_deconvolution import (
    RespirationConfig, MediaPipeChestExtractor, EulerianEngine, draw_full_skeleton
)

def select_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Input Video for Eulerian Processing",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

def select_save_file(default_name):
    root = tk.Tk()
    root.withdraw()
    return filedialog.asksaveasfilename(
        title="Save Enhanced Video As",
        initialfile=default_name,
        defaultextension=".mp4",
        filetypes=[("MP4 Video", "*.mp4")]
    )

def main():
    print("======================================================")
    print("  Eulerian Respiration - Offline Video Generator")
    print("======================================================")
    
    # 1. Ask for Input File
    input_path = select_file()
    if not input_path:
        print("[INFO] No input file selected. Exiting.")
        return

    # 2. Ask for Output File
    out_name = os.path.basename(input_path).rsplit('.', 1)[0] + "_eulerian.mp4"
    output_path = select_save_file(out_name)
    if not output_path:
        print("[INFO] No output file selected. Exiting.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open {input_path}")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 3. Setup the pipeline
    cfg = RespirationConfig()
    # Boost alpha a bit for offline generation since it's cleaner
    cfg.alpha = 40.0
    
    extractor = MediaPipeChestExtractor()
    engine = EulerianEngine(cfg.patch_width, cfg.patch_height)
    
    print(f"\n[INFO] Processing: {os.path.basename(input_path)}")
    print(f"[INFO] Outputting to: {os.path.basename(output_path)}")
    print(f"[INFO] Resolution: {width}x{height} @ {fps:.1f} FPS")
    print("[INFO] Showing live preview while generating. Press 'q' in the window to stop.")
    print("-" * 54)

    cv2.namedWindow("Eulerian Video Maker Preview", cv2.WINDOW_NORMAL)
    
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Extract Patch using MediaPipe
        patch, quad_pts, has_pose, label_text, all_landmarks, sternum_info = extractor.extract_patch(
            frame, cfg.patch_width, cfg.patch_height
        )

        # Eulerian Process
        mag_patch, relief_patch, coherence, scalar, energy, is_gated = engine.process(patch, cfg)

        display_frame = frame.copy()

        # Skeleton
        if cfg.show_skeleton and all_landmarks:
            draw_full_skeleton(display_frame, all_landmarks)

        # Warp Relief Patch onto Full Frame
        if cfg.overlay_chest and has_pose and quad_pts is not None:
            selected_patch = relief_patch if cfg.chest_view_mode == 0 else mag_patch
            
            src_pts = np.array([
                [0, 0], [cfg.patch_width - 1, 0],
                [cfg.patch_width - 1, cfg.patch_height - 1], [0, cfg.patch_height - 1]
            ], dtype=np.float32)
            
            try:
                inv_mat = cv2.getPerspectiveTransform(src_pts, quad_pts)
                warped = cv2.warpPerspective(selected_patch, inv_mat, (width, height))

                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, quad_pts.astype(np.int32), 255)
                mask_3ch = cv2.merge([mask, mask, mask])

                alpha_blend = 0.88
                blended = cv2.addWeighted(warped, alpha_blend, display_frame, 1.0 - alpha_blend, 0)
                display_frame = np.where(mask_3ch > 0, blended, display_frame)
            except Exception:
                pass
                
            quad_int = quad_pts.astype(np.int32)
            cv2.polylines(display_frame, [quad_int], True, (0, 240, 168), 2, cv2.LINE_AA)

        # Draw UI
        cv2.putText(display_frame, "EULERIAN RESPIRATION DECONVOLUTION", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 168), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"MODE: MOTION RELIEF | GAIN: {cfg.alpha}x | TAU: {cfg.soft_clamp}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Show Fluidity Live
        cv2.imshow("Eulerian Video Maker Preview", display_frame)
        writer.write(display_frame)
        
        # Check for interrupt
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Cancelled by user.")
            break
        
        # Terminal Progress
        if frame_idx % 15 == 0:
            percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            sys.stdout.write(f"\r[INFO] Progress: {frame_idx}/{total_frames} frames ({percent:.1f}%) processed...")
            sys.stdout.flush()

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"\n[INFO] Finished in {elapsed:.1f} seconds. Saved to {output_path}")

if __name__ == '__main__':
    main()
