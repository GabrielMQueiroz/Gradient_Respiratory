# sttensor: Spatio-Temporal Gradient and Tensor Analysis

[![CI](https://github.com/GabrielMQueiroz/Gradient_Respiratory/actions/workflows/ci.yml/badge.svg)](https://github.com/GabrielMQueiroz/Gradient_Respiratory/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Pattern%20Recognition-green.svg)](paper/main.tex)

## 1. Overview

`sttensor` is a Python library for computing and manipulating spatio-temporal gradients, structure tensors, and their eigen-decompositions from image sequences. It provides modular access to each intermediate result—Sobel gradients, temporal differences, gradient differences, structure tensors, eigenvalues, eigenvectors—and supports adaptive filtering to preserve spatial localization of moving edges.

The library is designed for computer vision tasks such as motion analysis, dynamic texture segmentation, physiological Eulerian video analysis, and feature tracking under varying velocities.

## 2. Installation

### Basic Installation
```bash
git clone https://github.com/GabrielMQueiroz/Gradient_Respiratory.git
cd Gradient_Respiratory
pip install -e .
```

### Optional Extras
```bash
# With video I/O support (OpenCV)
pip install -e ".[video]"

# With dashboard and apps (Flask, OpenCV)
pip install -e ".[apps]"

# Development dependencies (pytest, OpenCV)
pip install -e ".[dev]"

# All dependencies
pip install -e ".[all]"
```


## 3. Core Concepts

- **Spatial gradient**: Sobel operator applied to a single frame.
- **Temporal difference**: Pixel-wise difference between consecutive frames.
- **Gradient difference** (`D`): Difference between spatial gradients of consecutive frames.  
  $mathbf{D} = \nabla I_t - \nabla I_{t-1} = \nabla (I_t - I_{t-1})$
- **Structure tensor** (`T`): Outer product of gradients, optionally smoothed.  
  $mathbf{T} = G_\sigma * (\mathbf{D} \mathbf{D}^T) \quad \text{or} \quad \mathbf{T} = G_\sigma * (\nabla I \nabla I^T$
- **Eigen-decomposition**: Eigenvalues $lambda_1 \ge \lambda_2 \ge 0$ and eigenvectors of the 2×2 tensor.
- **Adaptive filtering**: Gaussian smoothing with variance inversely proportional to local tensor trace, or directional smoothing along eigenvectors, to avoid smearing moving edges.

## 4. API Reference

### 4.1 Module `sttensor.gradients`

```python
def sobel(frame: np.ndarray, axis: str = 'both') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Sobel gradients of a 2D frame.
    
    Parameters:
        frame (np.ndarray): 2D grayscale image.
        axis (str): 'x', 'y', or 'both'. If 'both', returns (Gx, Gy).
    
    Returns:
        Gx, Gy or single gradient.
    """

def temporal_difference(frame_t: np.ndarray, frame_tm1: np.ndarray) -> np.ndarray:
    """
    Compute pixel-wise difference between two frames.
    
    Returns:
        I_t - I_{t-1}
    """

def gradient_difference(grad_t: Tuple[np.ndarray, np.ndarray],
                        grad_tm1: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute difference of spatial gradients: D = ∇I_t - ∇I_{t-1}.
    
    Returns:
        (Dx, Dy)
    """
```

### 4.2 Module `sttensor.tensor`

```python
def structure_tensor(Gx: np.ndarray, Gy: np.ndarray,
                     sigma: float = None,
                     adaptive: bool = False,
                     sigma_max: float = 3.0,
                     k: float = 1.0) -> np.ndarray:
    """
    Compute 2x2 structure tensor from spatial gradients.
    
    Parameters:
        Gx, Gy: Spatial gradients.
        sigma: Fixed Gaussian sigma. If None and adaptive=False, no smoothing.
        adaptive: If True, use adaptive sigma based on local trace.
        sigma_max: Maximum sigma for adaptive filtering.
        k: Sensitivity factor for adaptive sigma.
    
    Returns:
        T: (H, W, 2, 2) array of structure tensors.
    """

def structure_tensor_from_difference(Dx: np.ndarray, Dy: np.ndarray,
                                     sigma: float = None,
                                     adaptive: bool = False,
                                     sigma_max: float = 3.0,
                                     k: float = 1.0) -> np.ndarray:
    """
    Compute structure tensor from gradient difference D = ∇I_t - ∇I_{t-1}.
    Same parameters as structure_tensor.
    """
```

### 4.3 Module `sttensor.eigen`

```python
def decompose_tensor(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition of 2x2 structure tensors.
    
    Parameters:
        T: (H, W, 2, 2) array.
    
    Returns:
        eigvals: (H, W, 2) sorted descending.
        eigvecs: (H, W, 2, 2) corresponding eigenvectors (columns).
    """

def coherence(eigvals: np.ndarray) -> np.ndarray:
    """
    Compute coherence measure: (λ1 - λ2) / (λ1 + λ2 + eps).
    """

def orientation(eigvecs: np.ndarray) -> np.ndarray:
    """
    Compute orientation angle (radians) of the major eigenvector.
    """
```

### 4.4 Module `sttensor.filters`

```python
def adaptive_gaussian_filter(T: np.ndarray,
                             sigma_max: float = 3.0,
                             k: float = 1.0) -> np.ndarray:
    """
    Apply adaptive Gaussian smoothing to structure tensor T.
    Sigma at each pixel = sigma_max / (1 + k * trace(T)).
    """

def directional_smooth(T: np.ndarray,
                       eigvecs: np.ndarray,
                       sigma: float = 1.5) -> np.ndarray:
    """
    Smooth tensor along the minor eigenvector direction (parallel to edges).
    Uses steerable filters.
    """

def background_mask(frame_t: np.ndarray,
                    background: np.ndarray,
                    threshold: float = 10.0) -> np.ndarray:
    """
    Boolean mask: |frame_t - background| > threshold.
    Can be used to zero out D before tensor computation.
    """
```

### 4.5 Module `sttensor.utils`

```python
def load_video(path: str) -> Iterator[np.ndarray]:
    """Yield grayscale frames from video file."""

def visualize_tensor(T: np.ndarray, mode: str = 'magnitude') -> np.ndarray:
    """
    Visualize tensor as RGB image.
    mode: 'magnitude', 'orientation', 'coherence'.
    """
```

## 5. Usage Examples

### Basic gradient difference and tensor

```python
import cv2
import numpy as np
from sttensor import gradients, tensor, eigen

# Load two consecutive frames
frame_t = cv2.imread('frame_t.png', 0).astype(np.float32)
frame_tm1 = cv2.imread('frame_tm1.png', 0).astype(np.float32)

# Sobel gradients
Gx_t, Gy_t = gradients.sobel(frame_t)
Gx_tm1, Gy_tm1 = gradients.sobel(frame_tm1)

# Gradient difference
Dx, Dy = gradients.gradient_difference((Gx_t, Gy_t), (Gx_tm1, Gy_tm1))

# Structure tensor from D with adaptive smoothing
T = tensor.structure_tensor_from_difference(Dx, Dy, adaptive=True, sigma_max=3.0, k=2.0)

# Eigen-decomposition
eigvals, eigvecs = eigen.decompose_tensor(T)

# Coherence and orientation
coh = eigen.coherence(eigvals)
ori = eigen.orientation(eigvecs)
```

### Background subtraction mask

```python
from sttensor import filters

mask = filters.background_mask(frame_t, background_frame, threshold=15)
Dx_masked = Dx * mask
Dy_masked = Dy * mask
T = tensor.structure_tensor_from_difference(Dx_masked, Dy_masked, adaptive=True)
```

### Directional smoothing

```python
from sttensor import filters, tensor, eigen

T_raw = tensor.structure_tensor_from_difference(Dx, Dy, sigma=None)
eigvals, eigvecs = eigen.decompose_tensor(T_raw)
T_smooth = filters.directional_smooth(T_raw, eigvecs, sigma=1.5)
```

## 6. Mathematical Foundation

### 6.1 Why Difference of Gradients?

The simple temporal difference $I_t - I_{t-1}$ answers the question *"did the pixel change?"*. It does not distinguish between:
- An object that moved (shifted edge).
- An object that changed intensity (illumination).
- Sensor noise.

The **difference between gradients** answers a different question: *"did the local structure change its orientation or magnitude?"*.

Formally, let $I(x,y,t)$ be the intensity. The spatial gradient is:
$$\nabla I(x,y,t) = \begin{bmatrix} I_x \\ I_y \end{bmatrix}$$

Expanding in a Taylor series in time around $t-1$:
$$\nabla I(x,y,t) = \nabla I(x,y,t-1) + \frac{\partial}{\partial t} \nabla I(x,y,t-1) \cdot \Delta t + O(\Delta t^2)$$

With $\Delta t = 1$:
$$\mathbf{D}(x,y,t) = \frac{\partial}{\partial t} \nabla I(x,y,t-1) + O(\Delta t^2)$$

By equality of mixed derivatives (Clairaut-Schwarz):
$$\frac{\partial}{\partial t} \nabla I = \nabla \frac{\partial I}{\partial t} = \nabla I_t$$

Therefore:
$$\mathbf{D} = \nabla I_t = \begin{bmatrix} I_{xt} \\ I_{yt} \end{bmatrix}$$

The operator is the **gradient of the temporal derivative**—equivalently, the **temporal derivative of the gradient**.

### 6.2 Adaptive $\sigma$ Criterion

To preserve localization, define $\sigma$ as a function of the local trace **before** smoothing:
$$\sigma(x,y) = \frac{\sigma_{max}}{1 + k \cdot \text{Tr}(\mathbf{T}(x,y))}$$

- Where $\|\mathbf{D}\| \to 0$ (background): $\sigma \to \sigma_{max}$. Strong smoothing removes noise.
- Where $\|\mathbf{D}\| \gg 0$ (moving edge): $\sigma \to 0$. No smoothing, localization preserved.

### 6.3 Directional Smoothing

Steers an anisotropic Gaussian along the minor eigenvector $v_2$ (parallel to edges):
$$G_{\sigma_\parallel, \sigma_\perp}(x,y) = \frac{1}{2\pi \sigma_\parallel \sigma_\perp} \exp\left( -\frac{(x \cdot \mathbf{v}_2)^2}{2\sigma_\parallel^2} - \frac{(x \cdot \mathbf{v}_1)^2}{2\sigma_\perp^2} \right)$$
with $\sigma_\parallel \gg \sigma_\perp$.

## 7. Repository Layout

```text
Gradient_Respiratory/
├── .github/                          # CI/CD workflows and issue/PR templates
│   ├── workflows/ci.yml              # Multi-OS, multi-Python automated test matrix
│   ├── ISSUE_TEMPLATE/               # Bug report & feature request templates
│   └── pull_request_template.md      # Pull request checklist & template
├── sttensor/                         # Core Python library package
│   ├── __init__.py                   # Package exports and version
│   ├── eigen.py                      # Tensor decomposition & coherence metrics
│   ├── filters.py                    # Spatial & directional adaptive filters
│   ├── gradients.py                  # Spatial & temporal gradient differences
│   ├── multiscale.py                 # Multi-scale pyramid & fusion
│   ├── tensor.py                     # Structure tensor computation
│   ├── benchmarks.py                 # FWHM edge spread & latency benchmarks
│   └── utils.py                      # Video I/O & tensor visualization
├── tests/                            # Comprehensive pytest suite (39 tests)
│   ├── test_eigen.py                 # Eigenvalue/vector decomposition tests
│   ├── test_filters.py               # Adaptive & directional filtering tests
│   ├── test_gradients.py             # Sobel & gradient difference tests
│   ├── test_integration.py           # End-to-end video pipeline integration tests
│   ├── test_tensor.py                # Structure tensor computation tests
│   └── test_utils.py                 # Utility & normalization tests
├── apps/                             # Physiological extraction applications & dashboards
│   ├── mediapipe_chest_pose_deconvolution.py # Eulerian Deconvolution with Pose Tracking
│   ├── pipeline.py                   # Structure Tensor Eigenvalue Modulation pipeline
│   ├── web_app.py                    # Real-time Flask dashboard (http://localhost:5000)
│   ├── templates/index.html          # Dashboard web UI
│   └── standalone/                   # Browser-only client-side demos
├── notebooks/                        # Interactive Jupyter notebooks
│   ├── build_notebook.py             # Script to regenerate notebooks
│   └── data_distribution_and_comparison.ipynb # Visual exploration & distribution analysis
├── experiments/                      # Reproducible research experiments
│   ├── exp01_fwhm_edge_spread.py     # Edge spread measurement
│   ├── exp02_pareto_frontier.py      # Noise vs. smearing trade-off
│   ├── exp03_noise_robustness.py     # SNR stress tests
│   ├── exp04_baseline_comparison.py  # Optical flow & standard tensor comparison
│   ├── exp05_respiratory_demo.py     # Eulerian respiratory signal extraction
│   ├── exp06_latency_benchmark.py    # Runtime latency profiling
│   └── synthetic_validation.py       # Full synthetic benchmark suite
├── paper/                            # Research manuscript & bibliography
│   ├── main.tex                      # Manuscript LaTeX source
│   └── bibliography.bib              # BibTeX references
├── models/                           # Pretrained task models (pose_landmarker_lite.task)
├── docs/                             # Comprehensive mathematical specifications
│   └── GRADIENT_TEMPORAL_DIFFERENCE.md # Theoretical derivation & proof
├── videos/                           # Sample test sequences
├── pyproject.toml                    # PEP 621 package build configuration
├── setup.py                          # Setup configuration wrapper
├── CITATION.cff                      # Machine-readable academic citation metadata
├── CONTRIBUTING.md                   # Contribution guidelines
├── CODE_OF_CONDUCT.md                # Contributor Covenant Code of Conduct
└── LICENSE                           # MIT License
```

## 8. Running Applications & Experiments

### Running Tests
```bash
pytest
```

### Running Experiments
```bash
# Synthetic validation suite
python experiments/synthetic_validation.py

# Latency and edge-spread benchmarks
python experiments/exp01_fwhm_edge_spread.py
python experiments/exp06_latency_benchmark.py
```

### Running Applications & Dashboards
```bash
# Flask Telemetry Dashboard (opens on http://localhost:5000)
python apps/web_app.py

# Eulerian Pose Deconvolution CLI
python apps/mediapipe_chest_pose_deconvolution.py

# Real-time Structure Tensor Pipeline
python apps/pipeline.py
```

## 9. Citation

If you use this library, the adaptive structure tensor formulation, or the Eulerian deconvolution methodology in your research, please cite our paper:

```bibtex
@article{queiroz2026adaptive,
  title={Adaptive Spatio-Temporal Gradient Difference Tensors for Edge-Preserving Motion Analysis},
  author={Queiroz, Gabriel M.},
  journal={Pattern Recognition},
  year={2026},
  url={https://github.com/GabrielMQueiroz/Gradient_Respiratory}
}
```

A machine-readable citation format is also provided in [`CITATION.cff`](CITATION.cff).

## 10. Contributing

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting a pull request.

## 11. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

