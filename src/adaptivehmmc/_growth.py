"""Proportional-growth machinery."""

from __future__ import annotations

import numpy as np

from ._redistribution import local_stencil_start, solve_moment_weights


def build_growth_projection_matrix(
    pivots: np.ndarray, growth_factor: float, moment_orders: np.ndarray
) -> np.ndarray:
    """Build the per-step projection matrix ``M`` so that ``Y_{n+1} = M @ Y_n``.

    Each source pivot ``d_s`` is moved along the analytical characteristic to
    ``d_s * growth_factor`` and redistributed onto the current grid by local
    moment matching. The matrix only depends on dimensionless pivot ratios, so
    it survives uniform rescaling unchanged.
    """
    pivots = np.asarray(pivots, dtype=float)
    n = pivots.size
    s = moment_orders.size
    M = np.zeros((n, n))

    for source in range(n):
        target = pivots[source] * growth_factor
        first = local_stencil_start(pivots, target, s)
        stencil = slice(first, first + s)
        target_moments = target ** moment_orders
        weights = solve_moment_weights(pivots[stencil], target_moments, moment_orders)
        M[stencil, source] = weights

    return M


def growth_projection_step(
    population: np.ndarray,
    pivots: np.ndarray,
    growth_factor: float,
    moment_orders: np.ndarray,
) -> np.ndarray:
    """Apply one growth-projection time step (convenience wrapper around the matrix builder)."""
    M = build_growth_projection_matrix(pivots, growth_factor, moment_orders)
    return M @ np.asarray(population, dtype=float).ravel()
