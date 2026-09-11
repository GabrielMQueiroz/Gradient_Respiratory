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

__version__ = "0.1.0"

__all__ = [
    "gradients",
    "tensor",
    "eigen",
    "filters",
    "utils",
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
    "orientation",
    # filters
    "adaptive_gaussian_filter",
    "directional_smooth",
    "background_mask",
    # utils
    "load_video",
    "visualize_tensor",
    # multiscale
    "multiscale",
    "build_gaussian_pyramid",
    "multiscale_gradient_difference",
    "multiscale_tensor_fusion",
]
