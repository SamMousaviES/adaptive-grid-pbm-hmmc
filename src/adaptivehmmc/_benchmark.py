"""Analytical benchmark initial distribution."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from ._redistribution import solve_moment_weights


def benchmark_initial_population(
    pivots: np.ndarray,
    widths: np.ndarray,
    num_moments: int,
    total_number: float,
    volume_scale: float,
) -> np.ndarray:
    """Discretize the analytical benchmark density used in the manuscript.

    Density: ``n(d) = 3 N0 / v0 * d^2 * exp(-d^3 / v0)``.
    """
    pivots = np.asarray(pivots, dtype=float)
    widths = np.asarray(widths, dtype=float)
    num_classes = pivots.size
    moment_orders = np.arange(num_moments)
    population = np.zeros(num_classes)

    def density(x, k):
        return 3.0 * total_number / volume_scale * x ** 2 * np.exp(-x ** 3 / volume_scale) * x ** k

    for i in range(num_classes):
        lower = max(pivots[i] - widths[i] / 2.0, 0.0)
        upper = pivots[i] + widths[i] / 2.0

        local_moments = np.array([
            quad(density, max(lower, np.finfo(float).tiny), upper, args=(k,), epsabs=1e-14, epsrel=1e-10)[0]
            for k in moment_orders
        ])

        # Choose a centered stencil around the current pivot.
        center_index = i
        first = max(center_index - num_moments // 2, 0)
        last = first + num_moments - 1
        if last >= num_classes:
            last = num_classes - 1
            first = num_classes - num_moments

        stencil = slice(first, last + 1)
        weights = solve_moment_weights(pivots[stencil], local_moments, moment_orders)
        population[stencil] += weights

    initial_number = population.sum()
    if initial_number > 0:
        population = population * total_number / initial_number

    return population
