"""
Module for computing 2x2 structure tensors from spatial gradients and gradient differences.
"""

from typing import Optional
import numpy as np
from scipy.ndimage import gaussian_filter

from .filters import adaptive_gaussian_filter


def structure_tensor(
    Gx: np.ndarray,
    Gy: np.ndarray,
    sigma: Optional[float] = None,
    adaptive: bool = False,
    sigma_max: float = 3.0,
    k: float = 1.0,
    alpha: Optional[float] = None,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Compute 2x2 structure tensor from spatial gradients:
    T = [[Gx^2, Gx*Gy], [Gx*Gy, Gy^2]], optionally smoothed.

    Parameters
    ----------
    Gx : np.ndarray
        Horizontal spatial gradient array of shape (H, W).
    Gy : np.ndarray
        Vertical spatial gradient array of shape (H, W).
    sigma : Optional[float], default=None
        Fixed Gaussian smoothing standard deviation.
        If None and adaptive=False, no smoothing is applied (raw outer product).
    adaptive : bool, default=False
        If True, applies adaptive Gaussian smoothing where local sigma is inversely
        proportional to local tensor trace: sigma(x, y) = sigma_max / (1 + k * Tr(T)).
    sigma_max : float, default=3.0
        Maximum Gaussian sigma for adaptive smoothing.
    k : float, default=1.0
        Sensitivity factor for adaptive smoothing.
    alpha : Optional[float], default=None
        If provided, normalizes k by maximum trace: k = alpha / (max(trace) + 1e-8).
    mode : str, default='reflect'
        Boundary condition mode for smoothing convolution.

    Returns
    -------
    np.ndarray
        Structure tensor array T of shape (H, W, 2, 2) where:
        T[..., 0, 0] = Txx
        T[..., 0, 1] = Txy
        T[..., 1, 0] = Txy
        T[..., 1, 1] = Tyy

    Raises
    ------
    ValueError
        If Gx and Gy are not 2D arrays or have mismatched shapes, or if sigma < 0.
    """
    if sigma is not None and sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")

    gx = np.asarray(Gx)
    gy = np.asarray(Gy)

    if gx.ndim != 2 or gy.ndim != 2:
        raise ValueError(
            f"Gx and Gy must be 2D arrays. Got shapes {gx.shape} and {gy.shape}."
        )
    if gx.shape != gy.shape:
        raise ValueError(
            f"Gx and Gy must have identical shapes. Got {gx.shape} vs {gy.shape}."
        )

    gx_f = gx.astype(np.float64, copy=False) if np.issubdtype(gx.dtype, np.floating) else gx.astype(np.float64)
    gy_f = gy.astype(np.float64, copy=False) if np.issubdtype(gy.dtype, np.floating) else gy.astype(np.float64)

    # Compute outer products
    txx = gx_f * gx_f
    txy = gx_f * gy_f
    tyy = gy_f * gy_f

    H, W = gx.shape
    T = np.empty((H, W, 2, 2), dtype=np.float64)
    T[..., 0, 0] = txx
    T[..., 0, 1] = txy
    T[..., 1, 0] = txy
    T[..., 1, 1] = tyy

    if adaptive:
        T = adaptive_gaussian_filter(T, sigma_max=sigma_max, k=k, alpha=alpha, mode=mode)
    elif sigma is not None and sigma > 0.0:
        txx_smooth = gaussian_filter(txx, sigma=sigma, mode=mode)
        txy_smooth = gaussian_filter(txy, sigma=sigma, mode=mode)
        tyy_smooth = gaussian_filter(tyy, sigma=sigma, mode=mode)
        T[..., 0, 0] = txx_smooth
        T[..., 0, 1] = txy_smooth
        T[..., 1, 0] = txy_smooth
        T[..., 1, 1] = tyy_smooth

    return T


def structure_tensor_from_difference(
    Dx: np.ndarray,
    Dy: np.ndarray,
    sigma: Optional[float] = None,
    adaptive: bool = False,
    sigma_max: float = 3.0,
    k: float = 1.0,
    alpha: Optional[float] = None,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Compute structure tensor from gradient difference D = ∇I_t - ∇I_{t-1}.

    Parameters
    ----------
    Dx : np.ndarray
        Horizontal gradient difference array of shape (H, W).
    Dy : np.ndarray
        Vertical gradient difference array of shape (H, W).
    sigma : Optional[float], default=None
        Fixed Gaussian smoothing standard deviation.
        If None and adaptive=False, no smoothing is applied (raw outer product).
    adaptive : bool, default=False
        If True, applies adaptive Gaussian smoothing based on local trace.
    sigma_max : float, default=3.0
        Maximum Gaussian sigma for adaptive smoothing.
    k : float, default=1.0
        Sensitivity factor for adaptive smoothing.
    alpha : Optional[float], default=None
        If provided, normalizes k by maximum trace: k = alpha / (max(trace) + 1e-8).
    mode : str, default='reflect'
        Boundary condition mode for smoothing convolution.

    Returns
    -------
    np.ndarray
        Structure tensor array T of shape (H, W, 2, 2).
    """
    return structure_tensor(
        Gx=Dx,
        Gy=Dy,
        sigma=sigma,
        adaptive=adaptive,
        sigma_max=sigma_max,
        k=k,
        alpha=alpha,
        mode=mode,
    )
