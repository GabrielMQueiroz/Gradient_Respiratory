"""
Unit tests for sttensor.multiscale module.
"""

import numpy as np
import pytest
from sttensor.multiscale import (
    build_gaussian_pyramid,
    multiscale_gradient_difference,
    multiscale_tensor_fusion,
)


def test_build_gaussian_pyramid():
    img = np.random.rand(64, 64)
    pyr = build_gaussian_pyramid(img, n_levels=3)
    assert len(pyr) == 3
    assert pyr[0].shape == (64, 64)
    assert pyr[1].shape == (32, 32)
    assert pyr[2].shape == (16, 16)


def test_multiscale_gradient_difference():
    f1 = np.random.rand(64, 64)
    f2 = np.random.rand(64, 64)
    diffs = multiscale_gradient_difference(f1, f2, n_levels=3)
    assert len(diffs) == 3
    for dx, dy in diffs:
        assert dx.shape == dy.shape


def test_multiscale_tensor_fusion():
    f1 = np.random.rand(64, 64)
    f2 = np.random.rand(64, 64)
    l1_fused, coh_fused, contribs = multiscale_tensor_fusion(f1, f2, n_levels=3, fusion="max")
    assert l1_fused.shape == (64, 64)
    assert coh_fused.shape == (64, 64)
    assert contribs.shape == (3, 64, 64)
    assert np.all(coh_fused >= 0.0)
    assert np.all(coh_fused <= 1.0)
