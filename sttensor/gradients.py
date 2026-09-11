"""
Module for computing spatial gradients, temporal differences, and gradient differences.
"""

from typing import Tuple, Union
import numpy as np
from scipy.ndimage import sobel as scipy_sobel


def sobel(
    frame: np.ndarray,
    axis: str = "both",
    mode: str = "reflect",
    cval: float = 0.0,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Sobel gradients of a 2D frame.

    Parameters
    ----------
    frame : np.ndarray
        2D grayscale image array of shape (H, W).
    axis : str, default='both'
        Gradient axis: 'x' (horizontal gradient Gx), 'y' (vertical gradient Gy),
        or 'both' (returns tuple (Gx, Gy)).
    mode : str, default='reflect'
        Boundary condition mode for convolution ('reflect', 'constant', 'nearest',
        'mirror', 'wrap').
    cval : float, default=0.0
        Value to fill past edges if mode is 'constant'.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        If axis is 'both', returns (Gx, Gy).
        If axis is 'x', returns Gx.
        If axis is 'y', returns Gy.

    Raises
    ------
    ValueError
        If `frame` is not a 2D array or `axis` is not one of ('x', 'y', 'both').
    """
    frame_arr = np.asarray(frame)
    if frame_arr.ndim != 2:
        raise ValueError(
            f"frame must be a 2D grayscale image array, but got shape {frame_arr.shape} "
            f"with {frame_arr.ndim} dimensions."
        )

    # Cast integer or boolean arrays to float64 to avoid overflow/truncation
    if not np.issubdtype(frame_arr.dtype, np.floating):
        frame_arr = frame_arr.astype(np.float64)

    axis_lower = axis.lower()
    if axis_lower not in ("x", "y", "both"):
        raise ValueError(
            f"Invalid axis '{axis}'. Must be one of 'x', 'y', or 'both'."
        )

    # Note: axis 1 corresponds to columns (x), axis 0 corresponds to rows (y)
    if axis_lower == "x":
        return scipy_sobel(frame_arr, axis=1, mode=mode, cval=cval)
    elif axis_lower == "y":
        return scipy_sobel(frame_arr, axis=0, mode=mode, cval=cval)
    else:
        gx = scipy_sobel(frame_arr, axis=1, mode=mode, cval=cval)
        gy = scipy_sobel(frame_arr, axis=0, mode=mode, cval=cval)
        return gx, gy


def temporal_difference(frame_t: np.ndarray, frame_tm1: np.ndarray) -> np.ndarray:
    """
    Compute pixel-wise difference between two frames: I_t - I_{t-1}.

    Parameters
    ----------
    frame_t : np.ndarray
        Current frame at time t, shape (H, W).
    frame_tm1 : np.ndarray
        Previous frame at time t-1, shape (H, W).

    Returns
    -------
    np.ndarray
        Difference array (I_t - I_{t-1}) of shape (H, W) in floating-point.

    Raises
    ------
    ValueError
        If inputs have mismatched shapes or are not 2D arrays.
    """
    ft = np.asarray(frame_t)
    ftm1 = np.asarray(frame_tm1)

    if ft.ndim != 2 or ftm1.ndim != 2:
        raise ValueError(
            f"Both frames must be 2D arrays. Got shapes {ft.shape} and {ftm1.shape}."
        )
    if ft.shape != ftm1.shape:
        raise ValueError(
            f"Frames must have identical shapes. Got {ft.shape} vs {ftm1.shape}."
        )

    ft_f = ft if np.issubdtype(ft.dtype, np.floating) else ft.astype(np.float64)
    ftm1_f = ftm1 if np.issubdtype(ftm1.dtype, np.floating) else ftm1.astype(np.float64)

    return ft_f - ftm1_f


def gradient_difference(
    grad_t: Tuple[np.ndarray, np.ndarray],
    grad_tm1: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute difference of spatial gradients: D = ∇I_t - ∇I_{t-1}.

    Parameters
    ----------
    grad_t : Tuple[np.ndarray, np.ndarray]
        Spatial gradients (Gx_t, Gy_t) at time t.
    grad_tm1 : Tuple[np.ndarray, np.ndarray]
        Spatial gradients (Gx_{t-1}, Gy_{t-1}) at time t-1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Gradient differences (Dx, Dy) where:
        Dx = Gx_t - Gx_{t-1}
        Dy = Gy_t - Gy_{t-1}

    Raises
    ------
    ValueError
        If inputs are not 2-tuples or if gradient component shapes do not match.
    """
    if len(grad_t) != 2 or len(grad_tm1) != 2:
        raise ValueError(
            "grad_t and grad_tm1 must each be a 2-tuple of (Gx, Gy) arrays."
        )

    gx_t, gy_t = np.asarray(grad_t[0]), np.asarray(grad_t[1])
    gx_tm1, gy_tm1 = np.asarray(grad_tm1[0]), np.asarray(grad_tm1[1])

    if gx_t.shape != gx_tm1.shape or gy_t.shape != gy_tm1.shape or gx_t.shape != gy_t.shape:
        raise ValueError(
            f"Gradient components shape mismatch: "
            f"grad_t: ({gx_t.shape}, {gy_t.shape}), grad_tm1: ({gx_tm1.shape}, {gy_tm1.shape})"
        )

    gx_t_f = gx_t if np.issubdtype(gx_t.dtype, np.floating) else gx_t.astype(np.float64)
    gy_t_f = gy_t if np.issubdtype(gy_t.dtype, np.floating) else gy_t.astype(np.float64)
    gx_tm1_f = gx_tm1 if np.issubdtype(gx_tm1.dtype, np.floating) else gx_tm1.astype(np.float64)
    gy_tm1_f = gy_tm1 if np.issubdtype(gy_tm1.dtype, np.floating) else gy_tm1.astype(np.float64)

    dx = gx_t_f - gx_tm1_f
    dy = gy_t_f - gy_tm1_f

    return dx, dy
