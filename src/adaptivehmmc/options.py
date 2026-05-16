"""Solver option dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np


@dataclass
class InitialDistribution:
    """Initial-distribution specification.

    type:
        ``"benchmark"`` populates the categories from the analytical benchmark
        density ``3 N0 / v0 * d^2 * exp(-d^3 / v0)``; ``"custom"`` uses the
        provided ``population`` (and optionally ``pivots``).
    """

    type: str = "benchmark"
    total_number: float = 1.0
    volume_scale: float = 1.0e-3
    pivots: Optional[np.ndarray] = None
    population: Optional[np.ndarray] = None


@dataclass
class ProcessOptions:
    """Process description.

    Supported (type, kernel) pairs:

    * ``("coalescence", "constant")``
    * ``("breakage", "binary_equal_volume")``
    * ``("growth", "proportional")``
    """

    type: str = "coalescence"
    kernel: str = "constant"
    rate: float = 1.0


@dataclass
class NumericalOptions:
    """Numerical-solver settings."""

    rel_tol: float = 1.0e-6
    abs_tol: float = 1.0e-9
    max_step: Optional[float] = None
    store_history: bool = True
    num_steps: int = 500  # fixed-step count, used by the growth path only


@dataclass
class SolverOptions:
    """Top-level solver options.

    Convenient construction::

        opts = default_options()
        opts.num_classes = 30
        opts.adaptive = True
        opts.process.type = "breakage"
        opts.process.kernel = "binary_equal_volume"
    """

    time_span: Sequence[float] = (0.0, 1000.0)
    num_classes: int = 20
    num_moments: int = 6

    diameter_min: float = 0.0
    diameter_max: float = 1.0
    grid_ratio: float = 1.3

    adaptive: bool = True
    f_max: float = 2.5
    f_mult: float = 1.4

    initial_distribution: InitialDistribution = field(default_factory=InitialDistribution)
    process: ProcessOptions = field(default_factory=ProcessOptions)
    solver: NumericalOptions = field(default_factory=NumericalOptions)

    def copy(self) -> "SolverOptions":
        """Return a deep copy of the options struct."""
        return replace(
            self,
            initial_distribution=replace(self.initial_distribution),
            process=replace(self.process),
            solver=replace(self.solver),
        )


def default_options() -> SolverOptions:
    """Return a fresh ``SolverOptions`` with defaults matching the manuscript benchmark."""
    return SolverOptions()
