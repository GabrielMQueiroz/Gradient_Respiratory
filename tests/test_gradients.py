import numpy as np
import pytest
from sttensor.gradients import sobel, temporal_difference, gradient_difference


def test_sobel_axes():
    img = np.zeros((20, 20), dtype=np.float32)
    # Vertical ramp (gradient along y)
    img[:, :] = np.arange(20)[:, None]

    gx, gy = sobel(img, axis="both")
    assert gx.shape == (20, 20)
    assert gy.shape == (20, 20)
    # Gx should be 0 away from borders
    assert np.allclose(gx[2:-2, 2:-2], 0.0)
    # Gy should be positive (derivative along y)
    assert np.all(gy[2:-2, 2:-2] > 0.0)

    # Test single axis calls
    gx_single = sobel(img, axis="x")
    gy_single = sobel(img, axis="y")
    assert np.allclose(gx, gx_single)
    assert np.allclose(gy, gy_single)


def test_sobel_modes():
    img = np.random.randn(15, 15)
    gx_refl, gy_refl = sobel(img, mode="reflect")
    gx_const, gy_const = sobel(img, mode="constant", cval=0.0)
    # Interior should be identical
    assert np.allclose(gx_refl[1:-1, 1:-1], gx_const[1:-1, 1:-1])
    assert np.allclose(gy_refl[1:-1, 1:-1], gy_const[1:-1, 1:-1])


def test_sobel_dtype_support():
    img_u8 = np.zeros((10, 10), dtype=np.uint8)
    img_u8[:, 5:] = 255
    gx, gy = sobel(img_u8)
    assert gx.dtype == np.float64
    assert np.all(gx[:, 5] > 0)


def test_sobel_invalid_inputs():
    with pytest.raises(ValueError, match="2D grayscale"):
        sobel(np.zeros((10, 10, 3)))

    with pytest.raises(ValueError, match="Invalid axis"):
        sobel(np.zeros((10, 10)), axis="z")


def test_temporal_difference():
    f1 = np.full((10, 10), 100, dtype=np.uint8)
    f2 = np.full((10, 10), 150, dtype=np.uint8)

    diff = temporal_difference(f2, f1)
    assert diff.shape == (10, 10)
    assert np.allclose(diff, 50.0)

    # Test negative result (would underflow in uint8)
    diff_neg = temporal_difference(f1, f2)
    assert np.allclose(diff_neg, -50.0)


def test_temporal_difference_shape_mismatch():
    with pytest.raises(ValueError, match="mismatch|shapes"):
        temporal_difference(np.zeros((10, 10)), np.zeros((10, 12)))


def test_gradient_difference():
    gx1 = np.ones((8, 8)) * 2.0
    gy1 = np.ones((8, 8)) * 3.0
    gx2 = np.ones((8, 8)) * 5.0
    gy2 = np.ones((8, 8)) * 1.0

    dx, dy = gradient_difference((gx2, gy2), (gx1, gy1))
    assert np.allclose(dx, 3.0)
    assert np.allclose(dy, -2.0)


def test_gradient_difference_invalid():
    with pytest.raises(ValueError):
        gradient_difference((np.ones((4, 4)),), (np.ones((4, 4)), np.ones((4, 4))))

    with pytest.raises(ValueError):
        gradient_difference(
            (np.ones((4, 4)), np.ones((4, 4))),
            (np.ones((4, 5)), np.ones((4, 5)))
        )


def test_clairaut_schwarz_linearity():
    """
    Test linearity: ∇(I_t - I_{t-1}) == ∇I_t - ∇I_{t-1}
    """
    np.random.seed(42)
    frame_t = np.random.uniform(0, 255, (30, 30))
    frame_tm1 = np.random.uniform(0, 255, (30, 30))

    # Path 1: Sobel of temporal difference
    diff = temporal_difference(frame_t, frame_tm1)
    gx_diff, gy_diff = sobel(diff)

    # Path 2: Difference of Sobel gradients
    g_t = sobel(frame_t)
    g_tm1 = sobel(frame_tm1)
    dx, dy = gradient_difference(g_t, g_tm1)

    assert np.allclose(gx_diff, dx, atol=1e-10)
    assert np.allclose(gy_diff, dy, atol=1e-10)
