import numpy as np
import pytest
from sttensor.tensor import structure_tensor, structure_tensor_from_difference


def test_structure_tensor_unsmoothed():
    H, W = 16, 16
    gx = np.full((H, W), 3.0)
    gy = np.full((H, W), 4.0)

    T = structure_tensor(gx, gy, sigma=None, adaptive=False)
    assert T.shape == (H, W, 2, 2)

    # Check outer product values:
    # Txx = 9, Txy = 12, Tyy = 16
    assert np.allclose(T[..., 0, 0], 9.0)
    assert np.allclose(T[..., 0, 1], 12.0)
    assert np.allclose(T[..., 1, 0], 12.0)
    assert np.allclose(T[..., 1, 1], 16.0)

    # Rank 1: det(T) = Txx*Tyy - Txy^2 = 9*16 - 144 = 0
    det = T[..., 0, 0] * T[..., 1, 1] - T[..., 0, 1] * T[..., 1, 0]
    assert np.allclose(det, 0.0, atol=1e-12)


def test_structure_tensor_fixed_smoothing():
    np.random.seed(123)
    gx = np.random.randn(32, 32)
    gy = np.random.randn(32, 32)

    T_raw = structure_tensor(gx, gy, sigma=None)
    T_smooth = structure_tensor(gx, gy, sigma=1.5)

    # Smoothed tensor must maintain symmetry
    assert np.allclose(T_smooth[..., 0, 1], T_smooth[..., 1, 0])

    # Smoothing should reduce variance of tensor components
    assert np.var(T_smooth[..., 0, 0]) < np.var(T_raw[..., 0, 0])

    # Rank increases from 1 to 2 in general noisy field
    det_smooth = T_smooth[..., 0, 0] * T_smooth[..., 1, 1] - T_smooth[..., 0, 1] ** 2
    assert np.all(det_smooth >= -1e-12)


def test_structure_tensor_adaptive():
    H, W = 40, 40
    gx = np.zeros((H, W))
    gy = np.zeros((H, W))

    # A sharp vertical edge with high gradient
    gx[:, 20] = 50.0

    # Background has slight noise
    gx += np.random.uniform(-0.1, 0.1, (H, W))
    gy += np.random.uniform(-0.1, 0.1, (H, W))

    T_adapt = structure_tensor(gx, gy, adaptive=True, sigma_max=3.0, k=1.0)
    assert T_adapt.shape == (H, W, 2, 2)
    assert np.allclose(T_adapt[..., 0, 1], T_adapt[..., 1, 0])

    # At the sharp edge, trace should be largely preserved (sigma -> 0)
    # Compare with fixed heavy smoothing sigma=3.0
    T_fixed = structure_tensor(gx, gy, sigma=3.0)

    edge_trace_adapt = T_adapt[:, 20, 0, 0] + T_adapt[:, 20, 1, 1]
    edge_trace_fixed = T_fixed[:, 20, 0, 0] + T_fixed[:, 20, 1, 1]

    # Adaptive trace at the edge must be significantly higher than fixed smoothing
    assert np.mean(edge_trace_adapt) > np.mean(edge_trace_fixed)


def test_structure_tensor_from_difference():
    dx = np.ones((10, 10)) * 2.0
    dy = np.ones((10, 10)) * -1.0

    T1 = structure_tensor_from_difference(dx, dy, sigma=1.0)
    T2 = structure_tensor(dx, dy, sigma=1.0)
    assert np.allclose(T1, T2)


def test_structure_tensor_invalid_shapes():
    with pytest.raises(ValueError):
        structure_tensor(np.zeros((10, 10)), np.zeros((10, 12)))

    with pytest.raises(ValueError):
        structure_tensor(np.zeros((10, 10, 1)), np.zeros((10, 10, 1)))

    with pytest.raises(ValueError, match="sigma"):
        structure_tensor(np.zeros((10, 10)), np.zeros((10, 10)), sigma=-1.0)


def test_structure_tensor_alpha_parameter():
    gx = np.ones((10, 10)) * 2.0
    gy = np.ones((10, 10)) * 3.0

    T_adapt = structure_tensor(gx, gy, adaptive=True, alpha=1.0)
    assert T_adapt.shape == (10, 10, 2, 2)

    T_diff = structure_tensor_from_difference(gx, gy, adaptive=True, alpha=1.0)
    assert np.allclose(T_adapt, T_diff)
