import os
import tempfile
import numpy as np
import pytest
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from sttensor.tensor import structure_tensor
from sttensor.utils import load_video, visualize_tensor
import sttensor.utils as utils


def test_visualize_tensor_modes():
    H, W = 30, 30
    gx = np.zeros((H, W))
    gy = np.zeros((H, W))
    gx[:, 15] = 20.0

    T = structure_tensor(gx, gy, sigma=1.0)

    # 1. Magnitude
    vis_mag = visualize_tensor(T, mode="magnitude")
    assert vis_mag.shape == (H, W, 3)
    assert vis_mag.dtype == np.uint8

    # 2. Orientation
    vis_ori = visualize_tensor(T, mode="orientation")
    assert vis_ori.shape == (H, W, 3)
    assert vis_ori.dtype == np.uint8

    # 3. Coherence
    vis_coh = visualize_tensor(T, mode="coherence")
    assert vis_coh.shape == (H, W, 3)
    assert vis_coh.dtype == np.uint8


def test_visualize_tensor_numpy_fallback_and_empty():
    H, W = 20, 20
    T = structure_tensor(np.ones((H, W)), np.zeros((H, W)))

    # Test pure NumPy fallback (simulating cv2 not installed)
    orig_has_cv2 = utils._HAS_CV2
    try:
        utils._HAS_CV2 = False
        vis_mag = visualize_tensor(T, mode="magnitude")
        vis_coh = visualize_tensor(T, mode="coherence")
        vis_ori = visualize_tensor(T, mode="orientation")
        assert vis_mag.shape == (H, W, 3)
        assert vis_coh.shape == (H, W, 3)
        assert vis_ori.shape == (H, W, 3)
    finally:
        utils._HAS_CV2 = orig_has_cv2

    # Empty tensor
    T_empty = np.empty((0, 0, 2, 2))
    assert visualize_tensor(T_empty).shape == (0, 0, 3)


def test_visualize_tensor_invalid():
    T = np.zeros((10, 10, 2, 2))
    with pytest.raises(ValueError, match="Invalid mode"):
        visualize_tensor(T, mode="invalid_mode")

    with pytest.raises(ValueError, match="shape"):
        visualize_tensor(np.zeros((10, 10, 3)))


@pytest.mark.skipif(not _HAS_CV2, reason="opencv-python not installed")
def test_load_video_synthetic():
    # Create a small temporary video file
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        temp_path = f.name

    try:
        H, W = 64, 64
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(temp_path, fourcc, 10.0, (W, H), isColor=False)

        # Write 5 frames
        for i in range(5):
            frame = (np.ones((H, W), dtype=np.uint8) * (i * 40))
            writer.write(frame)
        writer.release()

        # Test loading as float
        frames_float = list(load_video(temp_path, as_float=True))
        assert len(frames_float) == 5
        assert frames_float[0].shape == (H, W)
        assert frames_float[0].dtype == np.float32

        # Test loading as uint8
        frames_u8 = list(load_video(temp_path, as_float=False))
        assert len(frames_u8) == 5
        assert frames_u8[0].dtype == np.uint8

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_video_file_not_found():
    with pytest.raises(FileNotFoundError):
        list(load_video("non_existent_video_file_12345.mp4"))
