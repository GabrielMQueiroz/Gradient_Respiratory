"""
Module for adaptive, directional, and spatial filtering of gradients and structure tensors.
"""

from typing import Optional
import numpy as np
from scipy.ndimage import convolve, gaussian_filter

from .eigen import decompose_tensor, coherence


def background_mask(
    frame_t: np.ndarray,
    background: np.ndarray,
    threshold: float = 10.0,
) -> np.ndarray:
    """
    Compute a boolean mask highlighting moving or changed pixels:
    |frame_t - background| > threshold.

    This mask can be used to zero out spatial gradients or gradient differences
    before structure tensor computation, suppressing background noise.

    Parameters
    ----------
    frame_t : np.ndarray
        Current 2D frame of shape (H, W).
    background : np.ndarray
        Background model or reference frame of shape (H, W).
    threshold : float, default=10.0
        Intensity threshold.

    Returns
    -------
    np.ndarray
        Boolean mask array of shape (H, W) where True indicates change.

    Raises
    ------
    ValueError
        If inputs are not 2D arrays or have mismatched shapes.
    """
    ft = np.asarray(frame_t)
    bg = np.asarray(background)

    if ft.ndim != 2 or bg.ndim != 2:
        raise ValueError(
            f"frame_t and background must both be 2D arrays. Got {ft.shape} and {bg.shape}."
        )
    if ft.shape != bg.shape:
        raise ValueError(
            f"Shape mismatch: frame_t {ft.shape} vs background {bg.shape}."
        )

    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold}.")

    ft_f = ft if np.issubdtype(ft.dtype, np.floating) else ft.astype(np.float64)
    bg_f = bg if np.issubdtype(bg.dtype, np.floating) else bg.astype(np.float64)

    return np.abs(ft_f - bg_f) > threshold


def adaptive_gaussian_filter(
    T: np.ndarray,
    sigma_max: float = 3.0,
    k: float = 1.0,
    alpha: Optional[float] = None,
    num_scales: int = 16,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply adaptive Gaussian smoothing to structure tensor T.

    The local smoothing scale sigma at each pixel is inversely proportional
    to the local tensor trace (gradient energy):
        sigma(x, y) = sigma_max / (1 + k * trace(T(x, y)))

    Where trace(T) -> 0 (uniform background), sigma -> sigma_max, strongly
    suppressing noise.
    Where trace(T) >> 0 (sharp moving edges), sigma -> 0, preserving spatial
    edge localization without smearing across boundaries.

    Parameters
    ----------
    T : np.ndarray
        Structure tensor array of shape (H, W, 2, 2).
    sigma_max : float, default=3.0
        Maximum Gaussian standard deviation (applied in low-energy / flat regions).
    k : float, default=1.0
        Sensitivity factor for adaptive sigma.
    alpha : Optional[float], default=None
        If provided, normalizes k by the maximum trace:
        k = alpha / (max(trace) + 1e-8), making sensitivity dimensionless.
    num_scales : int, default=16
        Number of discrete scale levels used for scale-space interpolation.
    mode : str, default='reflect'
        Boundary condition for Gaussian filtering ('reflect', 'constant', 'nearest', 'mirror', 'wrap').

    Returns
    -------
    np.ndarray
        Adaptively smoothed structure tensor array of shape (H, W, 2, 2).
    """
    T_arr = np.asarray(T)
    if T_arr.ndim != 4 or T_arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"T must have shape (H, W, 2, 2), but got shape {T_arr.shape}."
        )

    if sigma_max < 0.0:
        raise ValueError(f"sigma_max must be non-negative, got {sigma_max}.")
    if k < 0.0:
        raise ValueError(f"k must be non-negative, got {k}.")
    if alpha is not None and alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")

    if T_arr.size == 0 or sigma_max == 0.0 or num_scales <= 1:
        return T_arr.copy()

    # Ensure float64
    T_float = T_arr.astype(np.float64, copy=False) if np.issubdtype(T_arr.dtype, np.floating) else T_arr.astype(np.float64)

    # Compute trace: Tr(T) = Txx + Tyy
    trace = T_float[..., 0, 0] + T_float[..., 1, 1]
    # Symmetrize off-diagonal
    Txy = 0.5 * (T_float[..., 0, 1] + T_float[..., 1, 0])
    Txx = T_float[..., 0, 0]
    Tyy = T_float[..., 1, 1]

    if alpha is not None:
        max_trace = float(np.max(trace))
        k = alpha / (max_trace + 1e-8)

    # Compute spatially-varying sigma(x, y)
    sigmas = sigma_max / (1.0 + k * np.maximum(trace, 0.0))

    # Scale-space bank: discrete scales from 0.0 to sigma_max
    scale_levels = np.linspace(0.0, sigma_max, num_scales)
    delta_s = sigma_max / (num_scales - 1)

    H, W = trace.shape
    components = [Txx, Txy, Tyy]
    filtered_banks = []

    for comp in components:
        bank = np.empty((num_scales, H, W), dtype=np.float64)
        bank[0] = comp
        for s_idx in range(1, num_scales):
            bank[s_idx] = gaussian_filter(comp, sigma=scale_levels[s_idx], mode=mode)
        filtered_banks.append(bank)

    # Scale index and interpolation weights
    idx = np.clip((sigmas / delta_s).astype(np.intp), 0, num_scales - 2)
    weight = (sigmas - scale_levels[idx]) / delta_s
    weight = np.clip(weight, 0.0, 1.0)

    out = np.empty_like(T_float)
    smoothed_comps = []
    for bank in filtered_banks:
        v0 = np.take_along_axis(bank, idx[np.newaxis, ...], axis=0)[0]
        v1 = np.take_along_axis(bank, (idx + 1)[np.newaxis, ...], axis=0)[0]
        smoothed_comps.append((1.0 - weight) * v0 + weight * v1)

    out[..., 0, 0] = smoothed_comps[0]
    out[..., 0, 1] = smoothed_comps[1]
    out[..., 1, 0] = smoothed_comps[1]
    out[..., 1, 1] = smoothed_comps[2]

    return out


def directional_smooth(
    T: np.ndarray,
    eigvecs: np.ndarray,
    sigma: float = 1.5,
    sigma_perp: Optional[float] = None,
    num_orientations: int = 8,
    mode: str = "reflect",
    blend_coherence: bool = True,
) -> np.ndarray:
    """
    Smooth tensor along the minor eigenvector direction (parallel to edges).

    Uses a steerable oriented anisotropic Gaussian filter bank:
        G_{sigma_parallel, sigma_perp}(x, y)
    with sigma_parallel = sigma and sigma_perp << sigma_parallel.

    Filtering along the edge direction (minor eigenvector v2) reduces noise
    while preserving edge localization and avoiding cross-edge smearing.

    When blend_coherence is True (default), anisotropic filtering is blended
    with isotropic filtering (G_{sigma_perp}) according to local coherence:
        T_smooth = coherence * T_anisotropic + (1 - coherence) * T_isotropic.
    This prevents artificial smearing in unoriented / flat background regions
    where eigenvalues are degenerate.

    Parameters
    ----------
    T : np.ndarray
        Structure tensor array of shape (H, W, 2, 2).
    eigvecs : np.ndarray
        Eigenvectors array of shape (H, W, 2, 2) where column 1 is minor eigenvector v2.
    sigma : float, default=1.5
        Smoothing standard deviation along the edge (parallel direction).
    sigma_perp : Optional[float], default=None
        Smoothing standard deviation across the edge (perpendicular direction).
        If None, defaults to max(0.5, sigma / 3.0).
    num_orientations : int, default=8
        Number of discrete orientation angles used for the steerable filter bank.
    mode : str, default='reflect'
        Boundary condition mode for convolution.
    blend_coherence : bool, default=True
        Whether to blend anisotropic filtering with isotropic filtering based on
        tensor coherence to prevent directional bias in unoriented background regions.

    Returns
    -------
    np.ndarray
        Directionally smoothed structure tensor of shape (H, W, 2, 2).
    """
    T_arr = np.asarray(T)
    evec_arr = np.asarray(eigvecs)

    if T_arr.ndim != 4 or T_arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"T must have shape (H, W, 2, 2), but got {T_arr.shape}."
        )
    if evec_arr.ndim != 4 or evec_arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"eigvecs must have shape (H, W, 2, 2), but got {evec_arr.shape}."
        )
    if T_arr.shape[:2] != evec_arr.shape[:2]:
        raise ValueError(
            f"Spatial dimensions mismatch between T {T_arr.shape[:2]} and eigvecs {evec_arr.shape[:2]}."
        )

    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")
    if sigma_perp is not None and sigma_perp <= 0.0:
        raise ValueError(f"sigma_perp must be positive, got {sigma_perp}.")
    if num_orientations < 1:
        raise ValueError(f"num_orientations must be at least 1, got {num_orientations}.")

    if T_arr.size == 0 or sigma == 0.0:
        return T_arr.copy()

    if sigma_perp is None:
        sigma_perp = max(0.5, sigma / 3.0)

    # Minor eigenvector v2 is column 1 (v2_x, v2_y)
    v2_x = evec_arr[..., 0, 1]
    v2_y = evec_arr[..., 1, 1]

    # Orientation angle of v2 in [0, pi)
    theta = np.mod(np.arctan2(v2_y, v2_x), np.pi)

    # Ensure float64
    T_float = T_arr.astype(np.float64, copy=False) if np.issubdtype(T_arr.dtype, np.floating) else T_arr.astype(np.float64)
    Txx = T_float[..., 0, 0]
    Txy = 0.5 * (T_float[..., 0, 1] + T_float[..., 1, 0])
    Tyy = T_float[..., 1, 1]

    # Precompute anisotropic Gaussian kernels for num_orientations
    radius = int(np.ceil(3.0 * sigma))
    if radius < 1:
        radius = 1
    y_grid, x_grid = np.mgrid[-radius:radius + 1, -radius:radius + 1]

    orientations = np.linspace(0.0, np.pi, num_orientations, endpoint=False)
    delta_theta = np.pi / num_orientations

    H, W = theta.shape
    components = [Txx, Txy, Tyy]
    filtered_banks = []

    for comp in components:
        bank = np.empty((num_orientations, H, W), dtype=np.float64)
        for k, ori in enumerate(orientations):
            u = x_grid * np.cos(ori) + y_grid * np.sin(ori)
            v = -x_grid * np.sin(ori) + y_grid * np.cos(ori)
            kernel = np.exp(-0.5 * (u**2 / (sigma**2) + v**2 / (sigma_perp**2)))
            k_sum = kernel.sum()
            if k_sum > 0:
                kernel /= k_sum
            bank[k] = convolve(comp, kernel, mode=mode)
        filtered_banks.append(bank)

    # Circular interpolation between orientation bins in [0, pi)
    idx0 = (np.floor(theta / delta_theta).astype(np.intp)) % num_orientations
    idx1 = (idx0 + 1) % num_orientations
    weight = (theta - idx0 * delta_theta) / delta_theta
    weight = np.clip(weight, 0.0, 1.0)

    out = np.empty_like(T_float)
    smoothed_comps = []
    for bank in filtered_banks:
        v0 = np.take_along_axis(bank, idx0[np.newaxis, ...], axis=0)[0]
        v1 = np.take_along_axis(bank, idx1[np.newaxis, ...], axis=0)[0]
        smoothed_comps.append((1.0 - weight) * v0 + weight * v1)

    if blend_coherence:
        evals, _ = decompose_tensor(T_float)
        coh = coherence(evals)
        for c_idx, comp in enumerate(components):
            iso = gaussian_filter(comp, sigma=sigma_perp, mode=mode)
            smoothed_comps[c_idx] = coh * smoothed_comps[c_idx] + (1.0 - coh) * iso

    out[..., 0, 0] = smoothed_comps[0]
    out[..., 0, 1] = smoothed_comps[1]
    out[..., 1, 0] = smoothed_comps[1]
    out[..., 1, 1] = smoothed_comps[2]

    return out
