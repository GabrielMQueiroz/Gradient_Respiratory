"""
Experiment 05: Qualitative Respiratory Monitoring Demo.

Demonstrates the sttensor pipeline on a synthetic respiratory motion scenario:
a textured patch oscillating sinusoidally (simulating chest wall motion).
Compares three extraction methods:
  (a) Raw temporal difference energy
  (b) Structure tensor λ₁ modulation
  (c) Full adaptive tensor pipeline (adaptive + directional)

Since no reference respiratory sensor is available, this uses synthetic
ground-truth motion to show signal fidelity and spectral clarity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sttensor import gradients, tensor, eigen, filters
from sttensor.benchmarks import setup_publication_style, ensure_output_dirs


def generate_breathing_sequence(
    size: int = 128,
    n_frames: int = 300,
    fps: float = 30.0,
    resp_rate_hz: float = 0.25,  # 15 RPM
    amplitude: float = 2.0,  # pixels of displacement
    texture_freq: float = 6.0,
    noise_sigma: float = 3.0,
    rng=None,
):
    """
    Generate a synthetic breathing sequence: a textured patch that oscillates
    vertically with sinusoidal displacement at a known respiratory frequency.

    Returns
    -------
    frames : list of np.ndarray
        List of (size, size) float64 frames.
    gt_signal : np.ndarray
        Ground-truth displacement signal (1D, length n_frames).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t = np.arange(n_frames) / fps
    gt_displacement = amplitude * np.sin(2 * np.pi * resp_rate_hz * t)

    y_base, x_base = np.ogrid[:size, :size]
    frames = []

    for i in range(n_frames):
        dy = gt_displacement[i]
        # Shifted texture (sinusoidal grid)
        texture = 128.0 + 60.0 * np.sin(
            2 * np.pi * texture_freq * x_base / size
        ) * np.sin(
            2 * np.pi * texture_freq * (y_base - dy) / size
        )
        frame = texture.astype(np.float64)
        if noise_sigma > 0:
            frame += rng.normal(0, noise_sigma, (size, size))
        frames.append(frame)

    return frames, gt_displacement, t


def extract_raw_temporal_diff(frames):
    """Extract motion signal via raw |I_t - I_{t-1}| spatial mean."""
    signal = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i] - frames[i - 1])
        signal.append(np.mean(diff))
    return np.array(signal)


def extract_lambda1_modulation(frames):
    """Extract motion signal via mean λ₁ of spatial structure tensor."""
    signal = []
    for i in range(len(frames)):
        gx, gy = gradients.sobel(frames[i])
        T = tensor.structure_tensor(gx, gy, sigma=2.0)
        eigvals, _ = eigen.decompose_tensor(T)
        signal.append(np.mean(eigvals[..., 0]))
    return np.array(signal)


def extract_adaptive_tensor(frames):
    """Extract motion signal via adaptive gradient difference tensor pipeline."""
    signal = []
    for i in range(1, len(frames)):
        grad_t = gradients.sobel(frames[i])
        grad_tm1 = gradients.sobel(frames[i - 1])
        Dx, Dy = gradients.gradient_difference(grad_t, grad_tm1)

        T = tensor.structure_tensor_from_difference(
            Dx, Dy, adaptive=True, sigma_max=3.0, k=1.0
        )
        eigvals, eigvecs = eigen.decompose_tensor(T)

        # Directional smooth
        T_dir = filters.directional_smooth(T, eigvecs, sigma=1.5)
        eigvals_dir, _ = eigen.decompose_tensor(T_dir)

        # Coherence-weighted λ₁
        coh = eigen.coherence(eigvals_dir)
        weighted = coh * eigvals_dir[..., 0]
        signal.append(np.mean(weighted))

    return np.array(signal)


def run_experiment():
    setup_publication_style()
    dirs = ensure_output_dirs()
    rng = np.random.default_rng(42)

    print("Generating synthetic breathing sequence...")
    frames, gt_signal, t = generate_breathing_sequence(
        size=128, n_frames=300, fps=30.0,
        resp_rate_hz=0.25, amplitude=2.0,
        texture_freq=6.0, noise_sigma=3.0, rng=rng
    )

    print("Extracting signals...")
    sig_raw = extract_raw_temporal_diff(frames)
    sig_lambda1 = extract_lambda1_modulation(frames)
    sig_adaptive = extract_adaptive_tensor(frames)

    # Align time axes (raw and adaptive are 1 sample shorter)
    t_diff = t[1:]
    gt_diff = gt_signal[1:]

    # Normalize all signals to [-1, 1] for comparison
    def normalize(s):
        s = s - np.mean(s)
        mx = np.max(np.abs(s))
        return s / mx if mx > 0 else s

    sig_raw_n = normalize(sig_raw)
    sig_lambda1_n = normalize(sig_lambda1[1:])  # align with diff-based
    sig_adaptive_n = normalize(sig_adaptive)
    gt_n = normalize(gt_diff)

    # Compute spectral density
    fps = 30.0
    freqs = np.fft.rfftfreq(len(sig_raw), d=1.0 / fps)

    psd_raw = np.abs(np.fft.rfft(sig_raw_n)) ** 2
    psd_lambda1 = np.abs(np.fft.rfft(sig_lambda1_n)) ** 2
    psd_adaptive = np.abs(np.fft.rfft(sig_adaptive_n)) ** 2
    psd_gt = np.abs(np.fft.rfft(gt_n)) ** 2

    # Correlation with ground truth
    corr_raw = np.corrcoef(gt_n, sig_raw_n)[0, 1]
    corr_lambda1 = np.corrcoef(gt_n, sig_lambda1_n)[0, 1]
    corr_adaptive = np.corrcoef(gt_n, sig_adaptive_n)[0, 1]

    print(f"\nCorrelation with ground truth:")
    print(f"  Raw temporal diff:    r = {corr_raw:.4f}")
    print(f"  lambda_1 modulation: r = {corr_lambda1:.4f}")
    print(f"  Adaptive tensor:     r = {corr_adaptive:.4f}")

    # Find peak frequency
    resp_band = (freqs >= 0.1) & (freqs <= 0.5)
    peak_raw = freqs[resp_band][np.argmax(psd_raw[resp_band])]
    peak_lambda1 = freqs[resp_band][np.argmax(psd_lambda1[resp_band])]
    peak_adaptive = freqs[resp_band][np.argmax(psd_adaptive[resp_band])]

    print(f"\nPeak frequency (GT = 0.25 Hz / 15 RPM):")
    print(f"  Raw temporal diff:    {peak_raw:.3f} Hz ({peak_raw * 60:.1f} RPM)")
    print(f"  lambda_1 modulation: {peak_lambda1:.3f} Hz ({peak_lambda1 * 60:.1f} RPM)")
    print(f"  Adaptive tensor:     {peak_adaptive:.3f} Hz ({peak_adaptive * 60:.1f} RPM)")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})

    # Time-series
    ax = axes[0]
    ax.plot(t_diff, gt_n, "k--", alpha=0.5, linewidth=1.0, label="Ground truth")
    ax.plot(t_diff, sig_raw_n, alpha=0.7, label=f"Raw temporal diff (r={corr_raw:.3f})")
    ax.plot(t_diff, sig_lambda1_n, alpha=0.7, label=rf"$\lambda_1$ modulation (r={corr_lambda1:.3f})")
    ax.plot(t_diff, sig_adaptive_n, alpha=0.7, label=f"Adaptive tensor (r={corr_adaptive:.3f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Respiratory Signal Extraction Comparison")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 10)

    # PSD
    ax = axes[1]
    ax.plot(freqs * 60, psd_gt / psd_gt.max(), "k--", alpha=0.5, label="Ground truth")
    ax.plot(freqs * 60, psd_raw / psd_raw.max(), alpha=0.7, label="Raw temporal diff")
    ax.plot(freqs * 60, psd_lambda1 / psd_lambda1.max(), alpha=0.7, label=r"$\lambda_1$ modulation")
    ax.plot(freqs * 60, psd_adaptive / psd_adaptive.max(), alpha=0.7, label="Adaptive tensor")
    ax.set_xlabel("Frequency (RPM)")
    ax.set_ylabel("Normalized PSD")
    ax.set_xlim(0, 60)
    ax.axvline(15, color="red", linestyle=":", alpha=0.5, label="GT = 15 RPM")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_path = dirs["figures"] / "fig05_respiratory.pdf"
    fig.savefig(out_path)
    print(f"\nSaved figure to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    run_experiment()
