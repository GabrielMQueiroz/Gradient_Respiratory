"""
Module for video loading and tensor visualization utilities.
"""

from typing import Iterator
import os
import numpy as np

from .eigen import decompose_tensor, coherence, orientation

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def load_video(path: str, as_float: bool = True) -> Iterator[np.ndarray]:
    """
    Yield grayscale frames from a video file.

    Parameters
    ----------
    path : str
        Path to the video file.
    as_float : bool, default=True
        If True, yields frames as float32 arrays in the range [0.0, 255.0].
        If False, yields frames as uint8 arrays in the range [0, 255].

    Yields
    ------
    frame : np.ndarray
        2D grayscale frame array of shape (H, W).

    Raises
    ------
    ImportError
        If opencv-python (cv2) is not installed.
    FileNotFoundError
        If the specified video file path does not exist.
    RuntimeError
        If the video file could not be opened by OpenCV.
    """
    if not _HAS_CV2:
        raise ImportError(
            "opencv-python is required for video I/O. Please install it via 'pip install opencv-python'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: '{path}'")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: '{path}'")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame.ndim == 3 and frame.shape[2] >= 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                gray = frame[..., 0]
            else:
                gray = frame

            if as_float:
                yield gray.astype(np.float32)
            else:
                yield gray
    finally:
        cap.release()


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Pure NumPy vector conversion from HSV to uint8 RGB.
    h: hue in degrees [0, 360)
    s: saturation in [0, 1]
    v: value in [0, 1]
    """
    c = v * s
    x = c * (1.0 - np.abs(np.mod(h / 60.0, 2.0) - 1.0))
    m = v - c

    h_norm = np.mod(h, 360.0)
    sector = (h_norm // 60).astype(int)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Sector 0: 0 - 60
    mask = sector == 0
    r[mask] = c[mask]
    g[mask] = x[mask]

    # Sector 1: 60 - 120
    mask = sector == 1
    r[mask] = x[mask]
    g[mask] = c[mask]

    # Sector 2: 120 - 180
    mask = sector == 2
    g[mask] = c[mask]
    b[mask] = x[mask]

    # Sector 3: 180 - 240
    mask = sector == 3
    g[mask] = x[mask]
    b[mask] = c[mask]

    # Sector 4: 240 - 300
    mask = sector == 4
    r[mask] = x[mask]
    b[mask] = c[mask]

    # Sector 5: 300 - 360
    mask = sector >= 5
    r[mask] = c[mask]
    b[mask] = x[mask]

    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def visualize_tensor(T: np.ndarray, mode: str = "magnitude") -> np.ndarray:
    """
    Visualize structure tensor as an RGB image.

    Parameters
    ----------
    T : np.ndarray
        Structure tensor array of shape (H, W, 2, 2).
    mode : str, default='magnitude'
        Visualization mode:
        - 'magnitude': Displays tensor trace / gradient energy.
        - 'orientation': Encodes major eigenvector orientation as Hue,
          coherence as Saturation, and energy as Value.
        - 'coherence': Displays degree of local anisotropy / coherence in [0, 1].

    Returns
    -------
    np.ndarray
        RGB visualization array of shape (H, W, 3) with dtype uint8.

    Raises
    ------
    ValueError
        If mode is not one of ('magnitude', 'orientation', 'coherence')
        or if T has invalid shape.
    """
    T_arr = np.asarray(T)
    if T_arr.ndim != 4 or T_arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"T must have shape (H, W, 2, 2), but got {T_arr.shape}."
        )

    mode_lower = mode.lower()
    if mode_lower not in ("magnitude", "orientation", "coherence"):
        raise ValueError(
            f"Invalid mode '{mode}'. Supported modes are 'magnitude', 'orientation', 'coherence'."
        )

    H, W = T_arr.shape[:2]
    if H == 0 or W == 0:
        return np.empty((H, W, 3), dtype=np.uint8)

    eigvals, eigvecs = decompose_tensor(T_arr)
    l1 = eigvals[..., 0]
    l2 = eigvals[..., 1]
    trace = l1 + l2

    if mode_lower == "magnitude":
        trace_clean = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(trace_clean) if trace_clean.size > 0 else 0.0
        norm = trace_clean / (max_val + 1e-8) if max_val > 0 else np.zeros_like(trace_clean)
        norm_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)

        if _HAS_CV2:
            colormap_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_VIRIDIS)
            return cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)
        else:
            return np.stack([norm_u8, norm_u8, norm_u8], axis=-1)

    elif mode_lower == "coherence":
        coh = coherence(eigvals)
        coh_clean = np.nan_to_num(coh, nan=0.0, posinf=0.0, neginf=0.0)
        norm_u8 = np.clip(coh_clean * 255.0, 0, 255).astype(np.uint8)

        if _HAS_CV2:
            colormap_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_MAGMA)
            return cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)
        else:
            return np.stack([norm_u8, norm_u8, norm_u8], axis=-1)

    else:  # orientation
        # Major eigenvector angle in radians
        angles = orientation(eigvecs)
        # Orientation of undirected line is modulo pi: map [0, pi) to [0, 360)
        h_deg = np.mod(angles, np.pi) / np.pi * 360.0

        # Saturation: local coherence
        coh = coherence(eigvals)
        s = np.clip(coh, 0.0, 1.0)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

        # Value: normalized energy
        trace_clean = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(trace_clean) if trace_clean.size > 0 else 0.0
        v = np.clip(trace_clean / (max_val + 1e-8), 0.0, 1.0) if max_val > 0 else np.zeros_like(trace_clean)

        return _hsv_to_rgb(h_deg, s, v)
