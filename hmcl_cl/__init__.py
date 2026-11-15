"""Dynamic-curvature hyperbolic continual learning: Euclidean, EWC, HMCL-style and dynamic curvature (ours)."""

from .data import GPUAugment, build_stream, load_from_kaggle
from .geometry import DynamicCurvature, FullyHyperbolicLinear, pairwise_dist
from .models import ContinualLearningModel, ConvEncoder
from .regularizers import EWC, MemoryProjector
from .train import (METHOD_ALIASES, METHOD_LABELS, DYNAMIC_FAMILY, build_parser, make_args,
                    print_table, run_experiment)

__all__ = [
    "GPUAugment", "build_stream", "load_from_kaggle", "DynamicCurvature", "FullyHyperbolicLinear",
    "pairwise_dist", "ContinualLearningModel", "ConvEncoder", "EWC", "MemoryProjector",
    "METHOD_ALIASES", "METHOD_LABELS", "DYNAMIC_FAMILY", "build_parser", "make_args", "print_table",
    "run_experiment",
]
