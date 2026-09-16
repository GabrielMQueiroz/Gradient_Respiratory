"""
Unit tests for sttensor.optical_flow and sttensor.eulerian.
"""

import numpy as np
import pytest
from sttensor.optical_flow import farneback_flow, flow_to_colorwheel, flow_divergence_and_curl
from sttensor.eulerian import EulerianConfig, EulerianEngine


def test_optical_flow_basic():
    f1 = np.zeros((64, 64), dtype=np.uint8)
    f2 = np.zeros((64, 64), dtype=np.uint8)
    f1[20:40, 20:40] = 255
    f2[20:40, 25:45] = 255  # shifted by 5 pixels

    flow = farneback_flow(f1, f2)
    assert flow.shape == (64, 64, 2)
    
    bgr = flow_to_colorwheel(flow)
    assert bgr.shape == (64, 64, 3)
    assert bgr.dtype == np.uint8

    div, curl = flow_divergence_and_curl(flow)
    assert div.shape == (64, 64)
    assert curl.shape == (64, 64)


def test_eulerian_engine():
    engine = EulerianEngine(64, 64)
    cfg = EulerianConfig(alpha=20.0, tau=5.0)

    f1 = np.ones((64, 64, 3), dtype=np.uint8) * 128
    f2 = np.ones((64, 64, 3), dtype=np.uint8) * 135

    out1, rel1, coh1, s1, e1, g1 = engine.process(f1, cfg)
    assert out1.shape == (64, 64, 3)
    
    out2, rel2, coh2, s2, e2, g2 = engine.process(f2, cfg)
    assert out2.shape == (64, 64, 3)
    assert rel2.shape == (64, 64, 3)
    assert coh2.shape == (64, 64)
    assert isinstance(s2, float)
    assert isinstance(e2, float)
    assert isinstance(g2, (bool, np.bool_))
