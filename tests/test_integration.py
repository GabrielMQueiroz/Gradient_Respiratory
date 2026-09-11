import numpy as np
import pytest
from sttensor import gradients, tensor, eigen, filters, utils
import sttensor as st


def test_section_5_usage_examples():
    # Synthetic frames simulating moving object
    H, W = 64, 64
    np.random.seed(42)
    background_frame = np.random.uniform(50, 100, (H, W)).astype(np.float32)

    frame_tm1 = background_frame.copy()
    # Add a square object at (20, 20)
    frame_tm1[20:30, 20:30] += 80.0

    frame_t = background_frame.copy()
    # Object translated by 2 pixels right and down: (22, 22)
    frame_t[22:32, 22:32] += 80.0

    # Example 1: Basic gradient difference and tensor
    Gx_t, Gy_t = gradients.sobel(frame_t)
    Gx_tm1, Gy_tm1 = gradients.sobel(frame_tm1)

    Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))
    assert Dx.shape == (H, W)
    assert Dy.shape == (H, W)

    T = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=2.0)
    assert T.shape == (H, W, 2, 2)

    eigvals, eigvecs = eigen.decompose_tensor(T)
    assert eigvals.shape == (H, W, 2)
    assert eigvecs.shape == (H, W, 2, 2)

    coh = eigen.coherence(eigvals)
    ori = eigen.orientation(eigvecs)
    assert coh.shape == (H, W)
    assert ori.shape == (H, W)

    # Example 2: Background subtraction mask
    mask = filters.background_mask(frame_t, background_frame, threshold=15.0)
    assert mask.shape == (H, W)
    assert mask.dtype == bool

    Dx_masked = Dx * mask
    Dy_masked = Dy * mask
    T_masked = tensor.structure_tensor_from_difference(Dx_masked, Dy_masked, adaptive=True)
    assert T_masked.shape == (H, W, 2, 2)

    # Example 3: Directional smoothing
    T_raw = tensor.structure_tensor_from_difference(Dx, Dy, sigma=None)
    eigvals_raw, eigvecs_raw = eigen.decompose_tensor(T_raw)
    T_smooth = filters.directional_smooth(T_raw, eigvecs_raw, sigma=1.5)
    assert T_smooth.shape == (H, W, 2, 2)

    # Example 4: Visualization
    rgb_mag = utils.visualize_tensor(T_smooth, mode="magnitude")
    rgb_ori = utils.visualize_tensor(T_smooth, mode="orientation")
    rgb_coh = utils.visualize_tensor(T_smooth, mode="coherence")
    assert rgb_mag.shape == (H, W, 3)
    assert rgb_ori.shape == (H, W, 3)
    assert rgb_coh.shape == (H, W, 3)


def test_top_level_namespace_access():
    assert hasattr(st, "sobel")
    assert hasattr(st, "temporal_difference")
    assert hasattr(st, "gradient_difference")
    assert hasattr(st, "structure_tensor")
    assert hasattr(st, "structure_tensor_from_difference")
    assert hasattr(st, "decompose_tensor")
    assert hasattr(st, "coherence")
    assert hasattr(st, "orientation")
    assert hasattr(st, "adaptive_gaussian_filter")
    assert hasattr(st, "directional_smooth")
    assert hasattr(st, "background_mask")
    assert hasattr(st, "load_video")
    assert hasattr(st, "visualize_tensor")
