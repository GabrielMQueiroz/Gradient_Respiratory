"""
Eulerian Motion Magnification and Structural Deconvolution Module.

Provides pose-agnostic Eulerian processing with:
- Dual-IIR temporal bandpass filtering (isolates physiological micro-motion)
- Structure tensor coherence weighting / gating (suppresses flat noise, tracks contours)
- Non-linear anti-blooming limiter (hyperbolic tangent dynamic compression)
- Color EVM magnification & Phase-Contrast Relief synthesis
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class EulerianConfig:
    """Configuration parameters for the Eulerian processing engine."""
    def __init__(
        self,
        alpha: float = 35.0,
        tau: float = 8.0,
        gamma_fast: float = 0.20,
        gamma_slow_ratio: float = 0.15,
        gradient_scale: float = 1.2,
        coherence_th: float = 0.08,
        artifact_gate: float = 0.22,
        gate_mode: int = 0,  # 0: Soft AGC, 1: Hard Freeze, 2: Bypass
    ):
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.soft_clamp = float(tau)
        self.gamma_fast = float(gamma_fast)
        self.gamma_slow = float(gamma_fast * gamma_slow_ratio)
        self.gradient_scale = float(gradient_scale)
        self.coherence_th = float(coherence_th)
        self.artifact_gate = float(artifact_gate)
        self.gate_mode = int(gate_mode)


class EulerianEngine:
    """
    Pose-agnostic rotation-invariant Eulerian Motion Magnification and Deconvolution.
    Operates on any region of interest (ROI) or full frame.
    """
    def __init__(self, width: int = 240, height: int = 180):
        self.w = width
        self.h = height
        self.iir_slow = np.zeros((height, width), dtype=np.float32)
        self.iir_fast = np.zeros((height, width), dtype=np.float32)
        self.is_initialized = False

    def reset(self):
        """Reset internal temporal filter state buffers."""
        self.is_initialized = False

    def process(
        self,
        patch_bgr: np.ndarray,
        cfg: Optional[EulerianConfig] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
        """
        Process a single image patch or frame.

        Parameters
        ----------
        patch_bgr : np.ndarray
            BGR image patch (H, W, 3) or grayscale (H, W).
        cfg : Optional[EulerianConfig]
            Runtime configuration parameters.

        Returns
        -------
        magnified_bgr : np.ndarray
            Motion-magnified image patch (uint8 BGR).
        relief_bgr : np.ndarray
            Phase-contrast motion relief embossed visualization (uint8 BGR).
        coherence_map : np.ndarray
            Structure tensor coherence map in [0, 1].
        motion_scalar : float
            Deconvolved 1D motion scalar (mean coherence-weighted displacement).
        motion_energy : float
            Mean absolute temporal bandpass energy.
        is_gated : bool
            Whether motion energy exceeded the artifact gate.
        """
        if cfg is None:
            cfg = EulerianConfig()

        if patch_bgr.ndim == 2:
            gray = patch_bgr.astype(np.float32)
            patch_3ch = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            patch_3ch = patch_bgr

        # Initialize IIR filter poles on first run or shape change
        if not self.is_initialized or self.iir_slow.shape != gray.shape:
            self.iir_slow = gray.copy()
            self.iir_fast = gray.copy()
            self.is_initialized = True
            empty_coh = np.zeros_like(gray)
            neutral_relief = np.full_like(patch_3ch, 128)
            return patch_3ch.copy(), neutral_relief, empty_coh, 0.0, 0.0, False

        # 1. Dual-IIR Temporal Respiration Bandpass Filter
        self.iir_slow += cfg.gamma_slow * (gray - self.iir_slow)
        self.iir_fast += cfg.gamma_fast * (gray - self.iir_fast)
        bandpass = self.iir_fast - self.iir_slow

        # 2. Spatial Gradients Ix, Iy with Edge Contrast Sensitivity scaling
        g_scale = cfg.gradient_scale * 0.125
        ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * g_scale
        iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * g_scale

        # 3. Structure Tensor 3x3 local neighborhood integration
        jxx = cv2.boxFilter(ix * ix, -1, (3, 3))
        jyy = cv2.boxFilter(iy * iy, -1, (3, 3))
        jxy = cv2.boxFilter(ix * iy, -1, (3, 3))

        # 4. Rotation-Invariant Coherence Formula:
        diff = jxx - jyy
        trace = jxx + jyy
        num = (diff * diff) + (4.0 * jxy * jxy)
        den = (trace * trace) + 1e-4

        coherence = np.clip(num / den, 0.0, 1.0)
        coherence_mask = np.where(coherence >= cfg.coherence_th, coherence, 0.0)

        # 5. Motion Energy & Gross Motion Gate Evaluation
        motion_energy = float(np.mean(np.abs(bandpass)))
        is_gated = (cfg.gate_mode != 2) and (motion_energy > cfg.artifact_gate)

        effective_alpha = cfg.alpha
        if is_gated:
            if cfg.gate_mode == 1:  # Hard Freeze
                effective_alpha = 0.0
            else:  # Soft AGC: quadratically scales down excess energy
                excess = motion_energy / max(1e-4, cfg.artifact_gate)
                effective_alpha = cfg.alpha / (1.0 + excess * excess)

        # 6. Anti-Blooming Soft Limiter (Hyperbolic Tangent)
        tau = max(1.0, cfg.soft_clamp)
        clamped_motion = tau * np.tanh(bandpass / tau)
        magnified_delta = effective_alpha * coherence_mask * clamped_motion

        # 7. Synthesize magnified output image (Color EVM)
        out_bgr = patch_3ch.astype(np.float32)
        out_bgr[:, :, 0] += magnified_delta * 1.15  # Blue channel
        out_bgr[:, :, 1] += magnified_delta * 0.95  # Green channel
        out_bgr[:, :, 2] += magnified_delta         # Red channel
        out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)

        # 8. Synthesize High-Gain Motion Band Relief (Phase-Contrast / Schlieren Embossed View)
        diff_vis = np.clip(128.0 + clamped_motion * 18.0, 0, 255).astype(np.uint8)
        relief_bgr = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2BGR)

        # 9. Coherence-weighted motion scalar
        motion_scalar = float(np.mean(coherence_mask * clamped_motion))

        return out_bgr, relief_bgr, coherence, motion_scalar, motion_energy, is_gated
