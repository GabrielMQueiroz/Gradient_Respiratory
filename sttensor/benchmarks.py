import time
import os
import numpy as np
import matplotlib as mpl
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any

def measure_latency(
    func: Callable,
    *args: Any,
    n_trials: int = 100,
    warmup: int = 5,
    **kwargs: Any,
) -> Tuple[float, float]:
    """
    Measure the execution time of a function.
    
    Args:
        func: The function to benchmark.
        *args: Arguments to pass to the function.
        n_trials: Number of execution trials.
        warmup: Number of warmup executions before measuring.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Tuple containing mean execution time and standard deviation in milliseconds.
    """
    for _ in range(warmup):
        func(*args, **kwargs)
        
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # convert to ms
        
    return float(np.mean(times)), float(np.std(times))

def compute_fwhm(profile_1d: np.ndarray) -> float:
    """
    Compute Full Width at Half Maximum of a 1D signal profile.
    
    Args:
        profile_1d: 1D numpy array containing the signal profile.
        
    Returns:
        The FWHM width in pixel units.
    """
    peak_idx = np.argmax(profile_1d)
    peak_val = profile_1d[peak_idx]
    baseline = np.min(profile_1d)
    half_max = baseline + (peak_val - baseline) / 2.0
    
    # Find left crossing
    left_idx = peak_idx
    while left_idx > 0 and profile_1d[left_idx] > half_max:
        left_idx -= 1
        
    # Interpolate left crossing
    if profile_1d[left_idx] <= half_max and profile_1d[left_idx + 1] > half_max:
        x1, y1 = left_idx, profile_1d[left_idx]
        x2, y2 = left_idx + 1, profile_1d[left_idx + 1]
        left_crossing = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    else:
        left_crossing = float(left_idx)
        
    # Find right crossing
    right_idx = peak_idx
    while right_idx < len(profile_1d) - 1 and profile_1d[right_idx] > half_max:
        right_idx += 1
        
    # Interpolate right crossing
    if profile_1d[right_idx] <= half_max and profile_1d[right_idx - 1] > half_max:
        x1, y1 = right_idx - 1, profile_1d[right_idx - 1]
        x2, y2 = right_idx, profile_1d[right_idx]
        right_crossing = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    else:
        right_crossing = float(right_idx)
        
    return right_crossing - left_crossing

def compute_snr(signal_region: np.ndarray, noise_region: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.
    
    Args:
        signal_region: Numpy array of the signal region.
        noise_region: Numpy array of the noise region.
        
    Returns:
        SNR in decibels (dB).
    """
    eps = 1e-10
    signal_power = np.mean(signal_region ** 2)
    noise_power = np.mean(noise_region ** 2)
    return 10.0 * np.log10(signal_power / (noise_power + eps))

def compute_localization_error(detected_edge_profile: np.ndarray, gt_edge_position: float) -> float:
    """
    Compute distance from the peak of detected edge profile to ground truth edge position.
    
    Args:
        detected_edge_profile: 1D array representing edge response.
        gt_edge_position: Ground truth edge position in pixels.
        
    Returns:
        Localization error in pixels.
    """
    peak_idx = np.argmax(detected_edge_profile)
    return abs(peak_idx - gt_edge_position)

def setup_publication_style() -> None:
    """
    Configure matplotlib for publication quality figures.
    """
    mpl.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'font.family': 'serif',
    })

def ensure_output_dirs() -> dict:
    """
    Create standard output directories for experiments.
    
    Returns:
        Dict with 'figures' and 'results' keys pointing to Path objects.
    """
    base_dir = Path("experiments")
    figures_dir = base_dir / "figures"
    results_dir = base_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"figures": figures_dir, "results": results_dir}

def generate_translating_disk(
    size: int = 256,
    radius: float = 40.0,
    velocity: Tuple[float, float] = (3.0, 0.0),
    n_frames: int = 10,
    texture_freq: float = 0.0,
    noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """
    Generate a sequence of frames with a translating disk.
    
    Args:
        size: Size of the square frame.
        radius: Radius of the disk.
        velocity: (vx, vy) velocity of the disk per frame.
        n_frames: Number of frames to generate.
        texture_freq: Frequency of sinusoidal texture. 0 for solid disk.
        noise_sigma: Standard deviation of Gaussian noise.
        rng: Random number generator.
        
    Returns:
        List of 2D float64 frames.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    frames = []
    y, x = np.ogrid[:size, :size]
    
    center_x = size / 2.0
    center_y = size / 2.0
    
    for i in range(n_frames):
        cx = center_x + i * velocity[0]
        cy = center_y + i * velocity[1]
        
        dist_sq = (x - cx)**2 + (y - cy)**2
        mask = dist_sq <= radius**2
        
        frame = np.zeros((size, size), dtype=np.float64)
        
        if texture_freq > 0:
            texture = 0.5 * (1 + np.sin(2 * np.pi * texture_freq * x) * np.sin(2 * np.pi * texture_freq * y))
            frame[mask] = texture[mask]
        else:
            frame[mask] = 1.0
            
        if noise_sigma > 0:
            frame += rng.normal(0, noise_sigma, size=(size, size))
            
        frames.append(frame)
        
    return frames

def generate_translating_edge(
    size: int = 256,
    edge_position: float = 128.0,
    velocity: float = 3.0,
    n_frames: int = 10,
    edge_type: str = 'step',
    noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """
    Generate frames with a vertical edge translating horizontally.
    
    Args:
        size: Height and width of the frame.
        edge_position: Initial position of the edge.
        velocity: Horizontal velocity of the edge.
        n_frames: Number of frames.
        edge_type: 'step' or 'smooth' (Gaussian edge).
        noise_sigma: Standard deviation of Gaussian noise.
        rng: Random number generator.
        
    Returns:
        List of 2D float64 frames.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    frames = []
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    for i in range(n_frames):
        pos = edge_position + i * velocity
        
        if edge_type == 'step':
            frame = np.zeros((size, size), dtype=np.float64)
            frame[X >= pos] = 1.0
        elif edge_type == 'smooth':
            # Use error function or sigmoid for smooth edge
            sigma = 2.0
            frame = 0.5 * (1 + np.tanh((X - pos) / sigma))
        else:
            raise ValueError(f"Unknown edge_type: {edge_type}")
            
        if noise_sigma > 0:
            frame += rng.normal(0, noise_sigma, size=(size, size))
            
        frames.append(frame)
        
    return frames
