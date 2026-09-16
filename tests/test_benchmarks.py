"""
Unit tests for sttensor.benchmarks module.
"""

import numpy as np
import pytest
from sttensor.benchmarks import (
    compute_fwhm,
    compute_snr,
    compute_localization_error,
    measure_latency,
    generate_translating_disk,
    generate_translating_edge,
)


def test_compute_fwhm_gaussian():
    # Gaussian profile with known sigma -> FWHM ~ 2.355 * sigma
    sigma = 3.0
    x = np.linspace(-15, 15, 301)
    profile = np.exp(-0.5 * (x / sigma) ** 2)
    fwhm = compute_fwhm(profile)
    expected_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    # Step size is (30 / 300) = 0.1
    fwhm_scaled = fwhm * 0.1
    assert np.isclose(fwhm_scaled, expected_fwhm, rtol=0.05)


def test_compute_snr():
    signal = np.ones((50, 50)) * 5.0
    noise = np.ones((50, 50)) * 0.5
    snr_db = compute_snr(signal, noise)
    # 10 * log10(25 / 0.25) = 10 * log10(100) = 20 dB
    assert np.isclose(snr_db, 20.0, atol=0.1)


def test_compute_localization_error():
    profile = np.zeros(100)
    profile[45] = 10.0
    profile[46] = 8.0
    err = compute_localization_error(profile, gt_edge_position=45.0)
    assert np.isclose(err, 0.0)


def test_measure_latency():
    def dummy_func(x, factor=1):
        return (x ** 2) * factor

    mean_ms, std_ms = measure_latency(dummy_func, 10, factor=2, n_trials=10, warmup=2)
    assert mean_ms >= 0.0
    assert std_ms >= 0.0


def test_generate_translating_disk():
    frames = generate_translating_disk(size=64, radius=15.0, velocity=(2.0, 0.0), n_frames=5)
    assert len(frames) == 5
    assert frames[0].shape == (64, 64)
    # Check that disk actually moved
    assert not np.array_equal(frames[0], frames[-1])


def test_generate_translating_edge():
    frames = generate_translating_edge(size=64, edge_position=30.0, velocity=2.0, n_frames=4)
    assert len(frames) == 4
    assert frames[0].shape == (64, 64)
    assert not np.array_equal(frames[0], frames[-1])
