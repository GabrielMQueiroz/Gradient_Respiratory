"""
sttensor: Spatio-Temporal Gradient and Tensor Analysis

A Python library for computing and manipulating spatio-temporal gradients,
structure tensors, and their eigen-decompositions from image sequences.
"""

from . import gradients
from . import tensor
from . import eigen
from . import filters
from . import utils
from . import multiscale
from . import eulerian
from . import optical_flow
from . import benchmarks

try:
    from . import nn
    from .nn import SGEMDLayer
    _HAS_NN = True
except ImportError:
    _HAS_NN = False

from .gradients import (
    sobel,
    temporal_difference,
    gradient_difference,
)
from .tensor import (
    structure_tensor,
    structure_tensor_from_difference,
)
from .eigen import (
    decompose_tensor,
    coherence,
    fast_coherence,
    orientation,
)
from .filters import (
    adaptive_gaussian_filter,
    directional_smooth,
    background_mask,
)
from .utils import (
    load_video,
    visualize_tensor,
)
from .multiscale import (
    build_gaussian_pyramid,
    multiscale_gradient_difference,
    multiscale_tensor_fusion,
)
from .eulerian import (
    EulerianConfig,
    EulerianEngine,
)
from .optical_flow import (
    farneback_flow,
    flow_to_colorwheel,
    flow_divergence_and_curl,
)

__version__ = "0.1.1"

__all__ = [
    "gradients",
    "tensor",
    "eigen",
    "filters",
    "utils",
    "multiscale",
    "eulerian",
    "optical_flow",
    "benchmarks",
    # gradients
    "sobel",
    "temporal_difference",
    "gradient_difference",
    # tensor
    "structure_tensor",
    "structure_tensor_from_difference",
    # eigen
    "decompose_tensor",
    "coherence",
    "fast_coherence",
    "orientation",
    # filters
    "adaptive_gaussian_filter",
    "directional_smooth",
    "background_mask",
    # utils
    "load_video",
    "visualize_tensor",
    # multiscale
    "build_gaussian_pyramid",
    "multiscale_gradient_difference",
    "multiscale_tensor_fusion",
    # eulerian
    "EulerianConfig",
    "EulerianEngine",
    # optical_flow
    "farneback_flow",
    "flow_to_colorwheel",
    "flow_divergence_and_curl",
]

if _HAS_NN:
    __all__.extend(["nn", "SGEMDLayer"])
