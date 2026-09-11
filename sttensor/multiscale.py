"""
Multi-scale pyramid extension for gradient difference tensors.

Addresses the straight-edge limitation: under pure translation of a locally
straight edge, the second spatial derivatives vanish (I_xx ≈ I_yy ≈ I_xy ≈ 0),
causing D = ∇(∂_t I) → 0 even under significant physical velocity.

By computing D at multiple spatial scales (octaves) via Gaussian pyramid
downsampling, coarser levels introduce effective curvature into otherwise
straight edges, ensuring at least one pyramid level produces a non-zero
response.

Fusion strategies:
    - 'max': Per-pixel maximum λ₁ across scales (preserves sharpest response).
    - 'sum': Summed λ₁ across scales (energy accumulation).
    - 'weighted': Scale-weighted sum favoring finer resolutions.
"""

from typing import List, Optional, Tuple, Literal
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from .gradients import sobel, gradient_difference
from .tensor import structure_tensor_from_difference
from .eigen import decompose_tensor, coherence


def build_gaussian_pyramid(
    frame: np.ndarray,
    n_levels: int = 4,
    sigma_anti_alias: float = 1.0,
) -> List[np.ndarray]:
    """
    Build a Gaussian pyramid by iterative smoothing and 2× downsampling.

    Parameters
    ----------
    frame : np.ndarray
        2D grayscale image of shape (H, W).
    n_levels : int, default=4
        Number of pyramid levels (including the original resolution).
    sigma_anti_alias : float, default=1.0
        Gaussian sigma applied before each 2× downsampling step to prevent aliasing.

    Returns
    -------
    List[np.ndarray]
        Pyramid levels from finest (original) to coarsest.
        Level 0 = original, level k has shape approximately (H/2^k, W/2^k).
    """
    frame_arr = np.asarray(frame, dtype=np.float64)
    if frame_arr.ndim != 2:
        raise ValueError(
            f"frame must be 2D, got shape {frame_arr.shape}."
        )
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}.")

    pyramid = [frame_arr]
    current = frame_arr

    for _ in range(1, n_levels):
        if current.shape[0] < 4 or current.shape[1] < 4:
            break  # Too small to downsample further
        smoothed = gaussian_filter(current, sigma=sigma_anti_alias)
        downsampled = smoothed[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled

    return pyramid


def multiscale_gradient_difference(
    frame_t: np.ndarray,
    frame_tm1: np.ndarray,
    n_levels: int = 4,
    sigma_anti_alias: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute gradient differences D = ∇I_t - ∇I_{t-1} at multiple pyramid scales.

    Parameters
    ----------
    frame_t : np.ndarray
        Current frame at time t, shape (H, W).
    frame_tm1 : np.ndarray
        Previous frame at time t-1, shape (H, W).
    n_levels : int, default=4
        Number of pyramid levels.
    sigma_anti_alias : float, default=1.0
        Anti-aliasing sigma before downsampling.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (Dx, Dy) gradient difference pairs at each scale.
        Level 0 is at original resolution, level k at ~(H/2^k, W/2^k).
    """
    pyr_t = build_gaussian_pyramid(frame_t, n_levels, sigma_anti_alias)
    pyr_tm1 = build_gaussian_pyramid(frame_tm1, n_levels, sigma_anti_alias)

    # Ensure matching number of levels
    n = min(len(pyr_t), len(pyr_tm1))

    diffs = []
    for level in range(n):
        ft = pyr_t[level]
        ftm1 = pyr_tm1[level]

        # Ensure same shape (handle odd pixel counts)
        min_h = min(ft.shape[0], ftm1.shape[0])
        min_w = min(ft.shape[1], ftm1.shape[1])
        ft = ft[:min_h, :min_w]
        ftm1 = ftm1[:min_h, :min_w]

        grad_t = sobel(ft)
        grad_tm1 = sobel(ftm1)
        dx, dy = gradient_difference(grad_t, grad_tm1)
        diffs.append((dx, dy))

    return diffs


def multiscale_tensor_fusion(
    frame_t: np.ndarray,
    frame_tm1: np.ndarray,
    n_levels: int = 4,
    sigma_anti_alias: float = 1.0,
    adaptive: bool = True,
    sigma_max: float = 3.0,
    k: float = 1.0,
    fusion: Literal["max", "sum", "weighted"] = "max",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradient difference tensors at multiple scales and fuse eigenvalue
    responses back to the original resolution.

    This addresses the straight-edge limitation: at the original scale,
    a perfectly straight translating edge produces D ≈ 0. At coarser scales,
    the effective edge curvature increases (due to downsampling smoothing),
    producing nonzero gradient differences that capture the motion.

    Parameters
    ----------
    frame_t : np.ndarray
        Current frame at time t, shape (H, W).
    frame_tm1 : np.ndarray
        Previous frame at time t-1, shape (H, W).
    n_levels : int, default=4
        Number of pyramid levels.
    sigma_anti_alias : float, default=1.0
        Anti-aliasing sigma before downsampling.
    adaptive : bool, default=True
        Whether to use adaptive Gaussian filtering at each level.
    sigma_max : float, default=3.0
        Maximum sigma for adaptive filtering.
    k : float, default=1.0
        Sensitivity factor for adaptive filtering.
    fusion : str, default='max'
        Fusion strategy:
        - 'max': Per-pixel maximum λ₁ across scales.
        - 'sum': Summed λ₁ across scales.
        - 'weighted': Scale-weighted sum (finer scales weighted more heavily).

    Returns
    -------
    lambda1_fused : np.ndarray
        Fused dominant eigenvalue map at original resolution, shape (H, W).
    coherence_fused : np.ndarray
        Fused coherence map at original resolution, shape (H, W).
    level_contributions : np.ndarray
        Array of shape (n_levels, H, W) showing each level's λ₁ contribution
        (upsampled to original resolution).
    """
    frame_t_arr = np.asarray(frame_t, dtype=np.float64)
    frame_tm1_arr = np.asarray(frame_tm1, dtype=np.float64)

    if frame_t_arr.ndim != 2 or frame_tm1_arr.ndim != 2:
        raise ValueError("Both frames must be 2D arrays.")
    if frame_t_arr.shape != frame_tm1_arr.shape:
        raise ValueError(
            f"Frame shapes must match: {frame_t_arr.shape} vs {frame_tm1_arr.shape}."
        )

    H, W = frame_t_arr.shape

    # Compute gradient differences at each scale
    diffs = multiscale_gradient_difference(
        frame_t_arr, frame_tm1_arr, n_levels, sigma_anti_alias
    )

    n = len(diffs)
    lambda1_levels = np.zeros((n, H, W), dtype=np.float64)
    coherence_levels = np.zeros((n, H, W), dtype=np.float64)

    for level, (dx, dy) in enumerate(diffs):
        # Compute structure tensor at this scale
        T = structure_tensor_from_difference(
            dx, dy,
            adaptive=adaptive,
            sigma_max=sigma_max,
            k=k,
        )

        # Eigen decomposition
        eigvals, _ = decompose_tensor(T)
        l1 = eigvals[..., 0]
        coh = coherence(eigvals)

        # Upsample to original resolution using bilinear interpolation
        if l1.shape != (H, W):
            zoom_factors = (H / l1.shape[0], W / l1.shape[1])
            l1_up = zoom(l1, zoom_factors, order=1)
            coh_up = zoom(coh, zoom_factors, order=1)
            # Ensure exact shape match (zoom may differ by ±1 pixel)
            l1_up = l1_up[:H, :W]
            coh_up = coh_up[:H, :W]
            # Pad if needed
            if l1_up.shape[0] < H or l1_up.shape[1] < W:
                pad_h = H - l1_up.shape[0]
                pad_w = W - l1_up.shape[1]
                l1_up = np.pad(l1_up, ((0, pad_h), (0, pad_w)), mode="edge")
                coh_up = np.pad(coh_up, ((0, pad_h), (0, pad_w)), mode="edge")
        else:
            l1_up = l1
            coh_up = coh

        # Scale-normalize: coarser levels have different energy scales
        # Normalize by 4^level to account for 2× downsampling in each dimension
        # (gradient magnitudes scale with spatial resolution)
        scale_factor = 4.0 ** level
        lambda1_levels[level] = l1_up / max(scale_factor, 1.0)
        coherence_levels[level] = coh_up

    # Fusion
    if fusion == "max":
        lambda1_fused = np.max(lambda1_levels, axis=0)
        # For coherence, take the coherence from the scale that had max λ₁
        best_level = np.argmax(lambda1_levels, axis=0)
        coherence_fused = np.take_along_axis(
            coherence_levels, best_level[np.newaxis, ...], axis=0
        )[0]
    elif fusion == "sum":
        lambda1_fused = np.sum(lambda1_levels, axis=0)
        # Weighted average coherence
        weights = lambda1_levels / (np.sum(lambda1_levels, axis=0, keepdims=True) + 1e-10)
        coherence_fused = np.sum(weights * coherence_levels, axis=0)
    elif fusion == "weighted":
        # Finer scales get exponentially more weight
        scale_weights = np.array([2.0 ** (-level) for level in range(n)])
        scale_weights /= scale_weights.sum()
        lambda1_fused = np.sum(
            lambda1_levels * scale_weights[:, np.newaxis, np.newaxis], axis=0
        )
        weights = (
            lambda1_levels * scale_weights[:, np.newaxis, np.newaxis]
        )
        weights /= np.sum(weights, axis=0, keepdims=True) + 1e-10
        coherence_fused = np.sum(weights * coherence_levels, axis=0)
    else:
        raise ValueError(f"Unknown fusion strategy '{fusion}'. Use 'max', 'sum', or 'weighted'.")

    return lambda1_fused, coherence_fused, lambda1_levels
