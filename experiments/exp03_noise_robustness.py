"""
Experiment 03: Noise Robustness of Spatio-Temporal Gradient Difference Tensors

This script measures the robustness of different structure tensor smoothing 
methods (Fixed, Adaptive, Directional) against varying levels of Gaussian noise.
It addresses reviewer concerns regarding second-order derivative noise amplification
by demonstrating how adaptive filtering maintains higher SNR and lower false 
positive rates in background regions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sttensor
from sttensor.benchmarks import generate_translating_disk

def main():
    # Setup directories
    os.makedirs('experiments/figures', exist_ok=True)
    os.makedirs('experiments/results', exist_ok=True)

    # 1. Setup parameters
    noise_levels = [0, 5, 10, 15, 20, 25, 30]
    num_frames = 10
    shape = (256, 256)
    radius = 40
    velocity = (3, 0)
    
    # 2. Generate clean sequence
    clean_frames = generate_translating_disk(
        size=256,
        radius=radius, 
        velocity=velocity, 
        n_frames=num_frames, 
        texture_freq=0.05
    )
    
    # Scale frames to 0-255 to match the noise scale
    clean_frames = [f * 255.0 for f in clean_frames]
    
    H, W = shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Disk centers for frames 4 and 5 (to define masks)
    # Using the logic from our generate_textured_disk function
    cx_4 = W / 2 + 4 * velocity[0] - (num_frames * velocity[0]) / 2
    cy_4 = H / 2 + 4 * velocity[1] - (num_frames * velocity[1]) / 2
    
    cx_5 = W / 2 + 5 * velocity[0] - (num_frames * velocity[0]) / 2
    cy_5 = H / 2 + 5 * velocity[1] - (num_frames * velocity[1]) / 2
    
    # Mean center for the gradient difference D (between 4 and 5)
    cx = (cx_4 + cx_5) / 2
    cy = (cy_4 + cy_5) / 2
    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    signal_mask = np.abs(dist - radius) <= 5
    noise_mask = dist > 60

    methods = [r'Fixed ($\sigma=2.0$)', 'Adaptive', 'Directional']
    
    results = {
        'noise_level': [],
        'method': [],
        'snr_db': [],
        'fpr': []
    }
    
    thresholds = {}
    rng = np.random.default_rng(42)

    for sigma_n in noise_levels:
        print(f"Processing noise level: {sigma_n}")
        
        # Add noise to frames 4 and 5
        frame4 = clean_frames[4].copy()
        frame5 = clean_frames[5].copy()
        
        if sigma_n > 0:
            frame4 += rng.normal(0, sigma_n, shape).astype(np.float32)
            frame5 += rng.normal(0, sigma_n, shape).astype(np.float32)
            
        # Compute spatial gradients
        grad4 = sttensor.sobel(frame4, axis='both')
        grad5 = sttensor.sobel(frame5, axis='both')
        
        # Compute gradient difference D
        Dx, Dy = sttensor.gradient_difference(grad5, grad4)
        
        for method_name in methods:
            if method_name == r'Fixed ($\sigma=2.0$)':
                T = sttensor.structure_tensor_from_difference(Dx, Dy, sigma=2.0, adaptive=False)
                eigvals, _ = sttensor.decompose_tensor(T)
                
            elif method_name == 'Adaptive':
                T = sttensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=0.0, alpha=1.0)
                eigvals, _ = sttensor.decompose_tensor(T)
                
            elif method_name == 'Directional':
                T_raw = sttensor.structure_tensor_from_difference(Dx, Dy, sigma=None, adaptive=False)
                _, eigvecs_raw = sttensor.decompose_tensor(T_raw)
                T = sttensor.directional_smooth(T_raw, eigvecs_raw, sigma=2.0)
                eigvals, _ = sttensor.decompose_tensor(T)
                
            # lambda_1 is the largest eigenvalue (channel 0 in sttensor)
            lambda1 = eigvals[..., 0]
            
            # Store threshold at clean level
            if sigma_n == 0:
                thresholds[method_name] = 0.1 * np.max(lambda1)
            
            # Compute metrics
            signal_power = np.mean(lambda1[signal_mask]**2)
            noise_power = np.mean(lambda1[noise_mask]**2)
            
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            fpr = np.mean(lambda1[noise_mask] > thresholds[method_name])
            
            results['noise_level'].append(sigma_n)
            results['method'].append(method_name)
            results['snr_db'].append(snr)
            results['fpr'].append(fpr)
            
            print(f"  {method_name}: SNR={snr:.2f}dB, FPR={fpr:.4f}")

    # 4. Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('experiments/results/noise_metrics.csv', index=False)

    # 5. Create figure
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    markers = {r'Fixed ($\sigma=2.0$)': 'o-', 'Adaptive': 's-', 'Directional': '^-'}
    colors = {r'Fixed ($\sigma=2.0$)': '#1f77b4', 'Adaptive': '#2ca02c', 'Directional': '#ff7f0e'}
    
    for method_name in methods:
        mask = df['method'] == method_name
        ax1.plot(df[mask]['noise_level'], df[mask]['snr_db'], 
                 markers[method_name], color=colors[method_name], label=method_name, linewidth=2)
        ax2.plot(df[mask]['noise_level'], df[mask]['fpr'], 
                 markers[method_name], color=colors[method_name], label=method_name, linewidth=2)
        
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Signal-to-Noise Ratio vs Noise Level')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    ax2.set_xlabel(r'Noise Level ($\sigma_n$)')
    ax2.set_ylabel('False Positive Rate')
    ax2.set_title('False Positive Rate in Background Region')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('experiments/figures/fig03_noise_snr.pdf', bbox_inches='tight', dpi=300)
    print("Results saved to experiments/results/noise_metrics.csv and experiments/figures/fig03_noise_snr.pdf")

if __name__ == '__main__':
    main()
