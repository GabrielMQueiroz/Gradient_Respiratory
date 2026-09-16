"""
Optical Flow baselines module.

Provides standardized wrappers for classical optical flow methods (Farneback, Lucas-Kanade)
to enable direct, fair comparisons against structure tensor representations.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def farneback_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> np.ndarray:
    """
    Compute dense optical flow using Gunnar Farnebäck's algorithm.

    Parameters
    ----------
    prev_frame : np.ndarray
        Previous grayscale frame (H, W) or color frame (H, W, 3).
    curr_frame : np.ndarray
        Current grayscale frame (H, W) or color frame (H, W, 3).

    Returns
    -------
    flow : np.ndarray
        Dense flow field array of shape (H, W, 2) where flow[..., 0] is horizontal (u)
        and flow[..., 1] is vertical (v) displacement in pixels.
    """
    prev_gray = prev_frame if prev_frame.ndim == 2 else cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = curr_frame if curr_frame.ndim == 2 else cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    if prev_gray.dtype != np.uint8:
        prev_gray = np.clip(prev_gray, 0, 255).astype(np.uint8)
    if curr_gray.dtype != np.uint8:
        curr_gray = np.clip(curr_gray, 0, 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )
    return flow


def flow_to_colorwheel(flow: np.ndarray, max_mag: Optional[float] = None) -> np.ndarray:
    """
    Convert 2D optical flow (u, v) into a standard BGR colorwheel visualization.

    Parameters
    ----------
    flow : np.ndarray
        Array of shape (H, W, 2).
    max_mag : Optional[float]
        Normalization magnitude. If None, uses 99th percentile of flow magnitude.

    Returns
    -------
    bgr : np.ndarray
        Colorwheel visualization image of shape (H, W, 3), dtype uint8.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
    hsv[..., 1] = 255

    if max_mag is None:
        p99 = float(np.percentile(mag, 99))
        max_mag = p99 if p99 > 1e-4 else 1.0

    hsv[..., 2] = np.clip((mag / max_mag) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_divergence_and_curl(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute divergence (div = du/dx + dv/dy) and curl/vorticity (curl = dv/dx - du/dy)
    of a 2D optical flow field.
    """
    u = flow[..., 0]
    v = flow[..., 1]

    du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    du_dy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    dv_dx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)

    div = du_dx + dv_dy
    curl = dv_dx - du_dy
    return div, curl
