"""Adaptive-grid moment-conserving sectional PBM solver.

Public API:

    from adaptivehmmc import (
        SolverOptions, ProcessOptions, InitialDistribution, NumericalOptions,
        default_options, solve, SolveResult,
    )

Supported processes (set via ``ProcessOptions.type`` and ``.kernel``):

    * coalescence  / constant
    * breakage     / binary_equal_volume
    * growth       / proportional
"""

from .options import (
    InitialDistribution,
    NumericalOptions,
    ProcessOptions,
    SolverOptions,
    default_options,
)
from .solve import SolveResult, solve

__all__ = [
    "InitialDistribution",
    "NumericalOptions",
    "ProcessOptions",
    "SolverOptions",
    "SolveResult",
    "default_options",
    "solve",
]

__version__ = "0.2.0"
