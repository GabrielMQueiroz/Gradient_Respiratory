"""
Module for eigen-decomposition, coherence analysis, and orientation estimation
of 2x2 symmetric structure tensors.
"""

from typing import Tuple
import numpy as np


def decompose_tensor(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform closed-form analytical eigen-decomposition of 2x2 structure tensors.

    Avoids iterative numerical solvers by using the exact closed-form algebraic
    solution for 2x2 real symmetric matrices.

    Parameters
    ----------
    T : np.ndarray
        Structure tensor array of shape (..., 2, 2), typically (H, W, 2, 2).

    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues array of shape (..., 2) sorted in descending order
        (lambda1 >= lambda2 >= 0).
    eigvecs : np.ndarray
        Corresponding orthonormal eigenvectors of shape (..., 2, 2).
        Column 0 (eigvecs[..., :, 0]) corresponds to major eigenvalue lambda1.
        Column 1 (eigvecs[..., :, 1]) corresponds to minor eigenvalue lambda2.

    Raises
    ------
    ValueError
        If T does not have last two dimensions equal to (2, 2).
    """
    T_arr = np.asarray(T)
    if T_arr.ndim < 2 or T_arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"T must have shape (..., 2, 2), but got shape {T_arr.shape}."
        )

    # Ensure float64 for numerical precision
    T_float = T_arr.astype(np.float64, copy=False) if np.issubdtype(T_arr.dtype, np.floating) else T_arr.astype(np.float64)

    a = T_float[..., 0, 0]
    # Symmetrize off-diagonal entries
    b = 0.5 * (T_float[..., 0, 1] + T_float[..., 1, 0])
    c = T_float[..., 1, 1]

    trace = a + c
    delta = np.hypot(a - c, 2.0 * b)

    lambda1 = 0.5 * (trace + delta)
    lambda2 = 0.5 * (trace - delta)

    # Structure tensors are positive semi-definite; clamp slight negative values from float precision
    lambda1 = np.maximum(lambda1, 0.0)
    lambda2 = np.maximum(lambda2, 0.0)

    eigvals = np.stack([lambda1, lambda2], axis=-1)

    # Eigenvector angle for major eigenvector v1:
    # 2*b = (a - c)*tan(2*theta) => theta = 0.5 * arctan2(2*b, a - c)
    theta = 0.5 * np.arctan2(2.0 * b, a - c)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    eigvecs = np.empty_like(T_float)
    # Major eigenvector v1 = [cos(theta), sin(theta)]^T (column 0)
    eigvecs[..., 0, 0] = cos_t
    eigvecs[..., 1, 0] = sin_t
    # Minor eigenvector v2 = [-sin(theta), cos(theta)]^T (column 1)
    eigvecs[..., 0, 1] = -sin_t
    eigvecs[..., 1, 1] = cos_t

    return eigvals, eigvecs


def coherence(eigvals: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute structure tensor coherence measure:
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)

    Values range in [0, 1]:
    - Near 1: Highly anisotropic / 1D oriented structure (strong line or edge).
    - Near 0: Isotropic structure or flat region (no dominant orientation).

    Parameters
    ----------
    eigvals : np.ndarray
        Eigenvalues array of shape (..., 2) sorted descending.
    eps : float, default=1e-6
        Small regularization constant to avoid division by zero in flat regions.

    Returns
    -------
    np.ndarray
        Coherence array of shape (...) with values in [0, 1].

    Raises
    ------
    ValueError
        If eigvals does not have last dimension equal to 2.
    """
    ev_arr = np.asarray(eigvals)
    if ev_arr.ndim < 1 or ev_arr.shape[-1] != 2:
        raise ValueError(
            f"eigvals must have shape (..., 2), but got {ev_arr.shape}."
        )

    l1 = ev_arr[..., 0]
    l2 = ev_arr[..., 1]

    coh = (l1 - l2) / (l1 + l2 + eps)
    return np.clip(coh, 0.0, 1.0)


def orientation(eigvecs: np.ndarray, modulo_pi: bool = False) -> np.ndarray:
    """
    Compute orientation angle (in radians) of the major eigenvector v1.

    Parameters
    ----------
    eigvecs : np.ndarray
        Eigenvectors array of shape (..., 2, 2) where column 0 is the major eigenvector.
    modulo_pi : bool, default=False
        If True, wraps the orientation angle into the interval (-pi/2, pi/2]
        to represent unoriented directional lines. Default returns atan2(v1y, v1x)
        in (-pi, pi].

    Returns
    -------
    np.ndarray
        Orientation angles in radians of shape (...).

    Raises
    ------
    ValueError
        If eigvecs does not have last two dimensions equal to (2, 2).
    """
    evec_arr = np.asarray(eigvecs)
    if evec_arr.ndim < 2 or evec_arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"eigvecs must have shape (..., 2, 2), but got {evec_arr.shape}."
        )

    v1_x = evec_arr[..., 0, 0]
    v1_y = evec_arr[..., 1, 0]

    angle = np.arctan2(v1_y, v1_x)
    if modulo_pi:
        angle = np.mod(angle + 0.5 * np.pi, np.pi) - 0.5 * np.pi
    return angle
