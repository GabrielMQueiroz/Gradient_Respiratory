import os
import time
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from sttensor import gradients
from sttensor import tensor
from sttensor import eigen
from sttensor import filters

try:
    from sttensor import benchmarks
except ImportError:
    pass

def generate_textured_disk(frames=10, size=256, radius=40, vel=(3, 0), texture_freq=8):
    """Generates a textured disk translating over time."""
    np.random.seed(42)
    seq = []
    masks = []
    x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
    # background texture
    bg = np.sin(x_grid / texture_freq) * np.cos(y_grid / texture_freq) * 127 + 128
    bg += np.random.normal(0, 5, size=(size, size))
    for i in range(frames):
        center = (size//2 + i*vel[0], size//2 + i*vel[1])
        dist = np.sqrt((x_grid - center[0])**2 + (y_grid - center[1])**2)
        mask = dist <= radius
        masks.append(mask)
        # object texture
        obj_tex = np.sin(x_grid / (texture_freq/2.0)) * np.cos(y_grid / (texture_freq/2.0)) * 127 + 128
        obj_tex += np.random.normal(0, 5, size=(size, size))
        frame = np.where(mask, obj_tex, bg).astype(np.float32)
        frame = np.clip(frame, 0, 255)
        seq.append(frame)
    return np.array(seq), np.array(masks)

def compute_fwhm(response_1d, start_idx=140):
    if np.max(response_1d[start_idx:]) == 0:
        return 0
    peak_offset = np.argmax(response_1d[start_idx:])
    peak_idx = start_idx + peak_offset
    half_max = response_1d[peak_idx] / 2.0
    
    left_idx = peak_idx
    while left_idx > 0 and response_1d[left_idx] >= half_max:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < len(response_1d) - 1 and response_1d[right_idx] >= half_max:
        right_idx += 1
        
    return right_idx - left_idx

def compute_contrast(response, gt_mask):
    bg_mask = ~gt_mask
    motion_mean = np.mean(response[gt_mask])
    bg_mean = np.mean(response[bg_mask])
    return motion_mean / (bg_mean + 1e-6)

def time_method(func, n_iters=100):
    # warmup
    _ = func()
    start = time.perf_counter()
    for _ in range(n_iters):
        func()
    end = time.perf_counter()
    return (end - start) * 1000 / n_iters

def main():
    """
    Experiment 04: Baseline Comparison
    Quantitative comparison against established baselines on synthetic sequences with known ground truth.
    """
    # 2. Generate sequence
    size = 256
    seq, masks = generate_textured_disk(frames=10, size=size, radius=40, vel=(3, 0), texture_freq=8)
    
    # 3. Take frames 4 and 5
    frame4 = seq[4]
    frame5 = seq[5]
    mask4 = masks[4]
    mask5 = masks[5]
    
    gt_mask = np.logical_xor(mask5, mask4)
    
    results = []
    methods = [
        "Raw temporal diff", 
        "3x3 Structure tensor", 
        "Farneback Flow", 
        "Fixed-sigma tensor", 
        "Adaptive tensor", 
        "Adaptive+Directional"
    ]
    fwhms = []
    contrasts = []
    runtimes = []
    
    center_row = size // 2
    
    # a. Raw temporal difference
    def method_a():
        diff = np.abs(gradients.temporal_difference(frame5, frame4))
        return diff
    resp_a = method_a()
    fwhms.append(compute_fwhm(resp_a[center_row, :]))
    contrasts.append(compute_contrast(resp_a, gt_mask))
    runtimes.append(time_method(method_a))
    
    # b. 3x3 Structure tensor
    def method_b():
        Ix, Iy = gradients.sobel(frame5)
        It = gradients.temporal_difference(frame5, frame4)
        J_tt = gaussian_filter(It*It, sigma=2.0)
        return J_tt
    resp_b = method_b()
    fwhms.append(compute_fwhm(resp_b[center_row, :]))
    contrasts.append(compute_contrast(resp_b, gt_mask))
    runtimes.append(time_method(method_b))
    
    # c. Farneback optical flow
    def method_c():
        flow = cv2.calcOpticalFlowFarneback(frame4, frame5, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return mag
    resp_c = method_c()
    fwhms.append(compute_fwhm(resp_c[center_row, :]))
    contrasts.append(compute_contrast(resp_c, gt_mask))
    runtimes.append(time_method(method_c))
    
    # d. Fixed-sigma gradient difference tensor
    def method_d():
        Gx_5, Gy_5 = gradients.sobel(frame5)
        Gx_4, Gy_4 = gradients.sobel(frame4)
        Dx, Dy = gradients.gradient_difference((Gx_5, Gy_5), (Gx_4, Gy_4))
        T_fixed = tensor.structure_tensor_from_difference(Dx, Dy, sigma=2.0, adaptive=False)
        eigvals = eigen.decompose_tensor(T_fixed)[0]
        return eigvals[..., 0]
    resp_d = method_d()
    fwhms.append(compute_fwhm(resp_d[center_row, :]))
    contrasts.append(compute_contrast(resp_d, gt_mask))
    runtimes.append(time_method(method_d))
    
    # e. Adaptive gradient difference tensor
    def method_e():
        Gx_5, Gy_5 = gradients.sobel(frame5)
        Gx_4, Gy_4 = gradients.sobel(frame4)
        Dx, Dy = gradients.gradient_difference((Gx_5, Gy_5), (Gx_4, Gy_4))
        T_adapt = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=1.0)
        eigvals = eigen.decompose_tensor(T_adapt)[0]
        return eigvals[..., 0]
    resp_e = method_e()
    fwhms.append(compute_fwhm(resp_e[center_row, :]))
    contrasts.append(compute_contrast(resp_e, gt_mask))
    runtimes.append(time_method(method_e))
    
    # f. Adaptive + directional
    def method_f():
        Gx_5, Gy_5 = gradients.sobel(frame5)
        Gx_4, Gy_4 = gradients.sobel(frame4)
        Dx, Dy = gradients.gradient_difference((Gx_5, Gy_5), (Gx_4, Gy_4))
        T_adapt = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=1.0)
        eigvals_ad, eigvecs_ad = eigen.decompose_tensor(T_adapt)
        T_dir = filters.directional_smooth(T_adapt, eigvecs_ad, sigma=2.0)
        eigvals_dir = eigen.decompose_tensor(T_dir)[0]
        return eigvals_dir[..., 0]
    resp_f = method_f()
    fwhms.append(compute_fwhm(resp_f[center_row, :]))
    contrasts.append(compute_contrast(resp_f, gt_mask))
    runtimes.append(time_method(method_f))
    
    # 8. Print to stdout
    print(f"{'Method':<25} | {'FWHM (px)':<10} | {'Contrast':<10} | {'Time (ms)':<10}")
    print("-" * 65)
    for m, f_val, c, t in zip(methods, fwhms, contrasts, runtimes):
        print(f"{m:<25} | {f_val:<10.2f} | {c:<10.2f} | {t:<10.2f}")
        
    # 7. Create table as CSV
    os.makedirs(os.path.join("experiments", "results"), exist_ok=True)
    csv_path = os.path.join("experiments", "results", "baseline_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "FWHM", "Contrast", "Runtime_ms"])
        for m, f_val, c, t in zip(methods, fwhms, contrasts, runtimes):
            writer.writerow([m, f_val, c, t])
            
    # 6. Grouped bar chart figure
    os.makedirs(os.path.join("experiments", "figures"), exist_ok=True)
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.5
    
    ax.bar(x, fwhms, width, color='skyblue', edgecolor='black')
    ax.set_ylabel('Edge FWHM (pixels)')
    ax.set_title('Edge Localization Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    
    plt.tight_layout()
    fig_path = os.path.join("experiments", "figures", "fig04_baseline_bars.pdf")
    plt.savefig(fig_path)
    plt.close()

if __name__ == "__main__":
    main()
