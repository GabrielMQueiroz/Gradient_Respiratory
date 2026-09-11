import numpy as np
import pytest
from sttensor.filters import background_mask, adaptive_gaussian_filter, directional_smooth
from sttensor.tensor import structure_tensor
from sttensor.eigen import decompose_tensor


def test_background_mask():
    f1 = np.zeros((10, 10))
    f2 = np.zeros((10, 10))
    f2[3:7, 3:7] = 20.0

    mask = background_mask(f2, f1, threshold=10.0)
    assert mask.shape == (10, 10)
    assert mask.dtype == bool

    expected = np.zeros((10, 10), dtype=bool)
    expected[3:7, 3:7] = True
    assert np.array_equal(mask, expected)


def test_background_mask_errors():
    with pytest.raises(ValueError):
        background_mask(np.zeros((5, 5)), np.zeros((5, 6)))
    with pytest.raises(ValueError):
        background_mask(np.zeros((5, 5, 2)), np.zeros((5, 5, 2)))


def test_adaptive_gaussian_filter_edge_preservation():
    """
    Verify that adaptive filtering preserves a sharp edge while smoothing
    the background, whereas standard Gaussian filter smears the edge.
    """
    H, W = 50, 50
    # Create an image with a vertical step edge and background noise
    np.random.seed(42)
    gx = np.zeros((H, W))
    gy = np.zeros((H, W))

    # Sharp vertical edge at x=25 (gradient along x)
    gx[:, 25] = 100.0
    # Add noise across the field
    noise = np.random.normal(0, 1.0, (H, W))
    gx += noise
    gy += noise

    T_raw = structure_tensor(gx, gy, sigma=None)

    # Adaptive filter with sigma_max=4.0
    T_adapt = adaptive_gaussian_filter(T_raw, sigma_max=4.0, k=1.0)
    # Fixed filter with same sigma=4.0
    T_fixed = structure_tensor(gx, gy, sigma=4.0)

    # 1. Background noise check (away from edge, outside 3*sigma radius, e.g. x in [0..10])
    bg_raw_var = np.var(T_raw[:, 0:10, 0, 0])
    bg_adapt_var = np.var(T_adapt[:, 0:10, 0, 0])
    # Adaptive filter should smooth the background and reduce variance
    assert bg_adapt_var < bg_raw_var

    # 2. Edge sharpness check at x=25
    edge_peak_raw = np.mean(T_raw[:, 25, 0, 0])
    edge_peak_adapt = np.mean(T_adapt[:, 25, 0, 0])
    edge_peak_fixed = np.mean(T_fixed[:, 25, 0, 0])

    # The adaptive filter should retain most of the peak edge intensity,
    # significantly higher than fixed smoothing which smears it into neighbors
    assert edge_peak_adapt > edge_peak_fixed * 2.0
    assert edge_peak_adapt > edge_peak_raw * 0.7


def test_adaptive_gaussian_filter_alpha():
    H, W = 20, 20
    T = np.ones((H, W, 2, 2)) * 10.0
    T[..., 0, 1] = 0.0
    T[..., 1, 0] = 0.0

    T_smooth = adaptive_gaussian_filter(T, sigma_max=2.0, alpha=1.0)
    assert T_smooth.shape == (H, W, 2, 2)
    assert np.allclose(T_smooth[..., 0, 1], T_smooth[..., 1, 0])


def test_adaptive_gaussian_filter_zero_sigma():
    T = np.random.randn(10, 10, 2, 2)
    T_copy = adaptive_gaussian_filter(T, sigma_max=0.0)
    assert np.allclose(T, T_copy)


def test_directional_smooth():
    """
    Test directional smoothing along the minor eigenvector v2.
    For a vertical edge at x=15:
    - Gradient points horizontally: v1 = [1, 0]
    - Edge runs vertically: v2 = [0, 1]
    Directional smooth with sigma along v2 should smooth along columns (y)
    while keeping the horizontal profile across columns (x) sharp.
    """
    H, W = 40, 40
    np.random.seed(123)

    # Edge at column 20 with variations along the edge (rows)
    gx = np.zeros((H, W))
    # Noisy signal along the edge
    gx[:, 20] = 50.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, H))
    gy = np.zeros((H, W))

    T_raw = structure_tensor(gx, gy, sigma=None)
    _, eigvecs = decompose_tensor(T_raw)

    T_dir = directional_smooth(T_raw, eigvecs, sigma=2.0, sigma_perp=0.3)

    assert T_dir.shape == (H, W, 2, 2)
    assert np.allclose(T_dir[..., 0, 1], T_dir[..., 1, 0])

    # Along the edge (y-direction at x=20), variance should be smoothed/reduced
    raw_col_var = np.var(T_raw[:, 20, 0, 0])
    dir_col_var = np.var(T_dir[:, 20, 0, 0])
    assert dir_col_var < raw_col_var

    # Across the edge (x-direction), the adjacent pixels should remain near zero (no cross-smearing)
    assert np.mean(T_dir[:, 18, 0, 0]) < 5.0
    assert np.mean(T_dir[:, 22, 0, 0]) < 5.0


def test_filters_invalid_shapes():
    with pytest.raises(ValueError):
        adaptive_gaussian_filter(np.zeros((10, 10, 2)))

    with pytest.raises(ValueError):
        directional_smooth(np.zeros((10, 10, 2, 2)), np.zeros((10, 10, 2)))

    with pytest.raises(ValueError):
        directional_smooth(np.zeros((10, 10, 2, 2)), np.zeros((12, 10, 2, 2)))


def test_directional_smooth_horizontal_and_diagonal():
    """
    Test directional smoothing on horizontal and diagonal edges to ensure
    rotational symmetry and prevent smearing into the background.
    """
    H, W = 40, 40
    np.random.seed(123)

    # 1. Horizontal edge at row 20
    gx = np.zeros((H, W))
    gy = np.zeros((H, W))
    gy[20, :] = 50.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, W))

    T_raw = structure_tensor(gx, gy, sigma=None)
    _, eigvecs = decompose_tensor(T_raw)

    T_dir = directional_smooth(T_raw, eigvecs, sigma=2.0, sigma_perp=0.3)

    # Along the horizontal edge (x-direction at row 20), variance should be smoothed
    raw_row_var = np.var(T_raw[20, :, 1, 1])
    dir_row_var = np.var(T_dir[20, :, 1, 1])
    assert dir_row_var < raw_row_var

    # Across the edge (y-direction), adjacent pixels must remain near zero without smearing
    assert np.mean(T_dir[18, :, 1, 1]) < 5.0
    assert np.mean(T_dir[22, :, 1, 1]) < 5.0
    assert np.allclose(T_dir[18, :, 1, 1], 0.0)

    # 2. Diagonal edge
    gx_d = np.zeros((H, W))
    gy_d = np.zeros((H, W))
    for i in range(H):
        gx_d[i, i] = 50.0 / np.sqrt(2)
        gy_d[i, i] = -50.0 / np.sqrt(2)

    T_raw_d = structure_tensor(gx_d, gy_d, sigma=None)
    _, eigvecs_d = decompose_tensor(T_raw_d)
    T_dir_d = directional_smooth(T_raw_d, eigvecs_d, sigma=2.0, sigma_perp=0.3)

    # Peak on diagonal, sharp cutoff 2 pixels away
    assert T_dir_d[20, 20, 0, 0] > 1000.0
    assert T_dir_d[18, 20, 0, 0] < 1.0


def test_filters_parameter_validation_and_empty():
    # Empty tensor handling
    T_empty = np.empty((0, 0, 2, 2))
    assert adaptive_gaussian_filter(T_empty).shape == (0, 0, 2, 2)
    assert directional_smooth(T_empty, T_empty).shape == (0, 0, 2, 2)

    # Negative parameters
    T = np.zeros((10, 10, 2, 2))
    evecs = np.zeros((10, 10, 2, 2))

    with pytest.raises(ValueError, match="sigma_max"):
        adaptive_gaussian_filter(T, sigma_max=-1.0)
    with pytest.raises(ValueError, match="k"):
        adaptive_gaussian_filter(T, k=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        adaptive_gaussian_filter(T, alpha=-1.0)
    with pytest.raises(ValueError, match="threshold"):
        background_mask(np.zeros((5, 5)), np.zeros((5, 5)), threshold=-1.0)
    with pytest.raises(ValueError, match="sigma_perp"):
        directional_smooth(T, evecs, sigma=1.0, sigma_perp=-0.5)
    with pytest.raises(ValueError, match="num_orientations"):
        directional_smooth(T, evecs, num_orientations=0)
