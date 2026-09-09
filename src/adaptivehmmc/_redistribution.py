"""Moment-matching redistribution primitives."""

from __future__ import annotations

import numpy as np


def local_stencil_start(pivots: np.ndarray, target: float, stencil_size: int) -> int:
    """Return the index of the first pivot in the local stencil around ``target``."""
    n = pivots.size
    mask = pivots < target
    if not mask.any():
        left = 0
    else:
        left = int(np.flatnonzero(mask)[-1])
    first = max(left - stencil_size // 2 + 1, 0)
    last = first + stencil_size - 1
    if last >= n:
        first = n - stencil_size
    return int(first)


def solve_moment_weights(pivots: np.ndarray, target_moments: np.ndarray, moment_orders: np.ndarray) -> np.ndarray:
    """Solve the local moment-matching linear system.

    Returns the weight vector ``w`` satisfying ``A w = m`` with
    ``A[r, s] = pivots[s] ** moment_orders[r]`` and ``m = target_moments``.
    """
    pivots = np.asarray(pivots, dtype=float)
    target_moments = np.asarray(target_moments, dtype=float)
    moment_orders = np.asarray(moment_orders, dtype=float)
    scale = max(float(np.max(np.abs(pivots))), np.finfo(float).tiny)
    scaled_pivots = pivots / scale
    scaled_moments = target_moments[: pivots.size] / scale**moment_orders
    matrix = scaled_pivots[None, :] ** moment_orders[:, None]
    return np.linalg.solve(matrix, scaled_moments)


def redistribute_population(
    population0: np.ndarray,
    pivots0: np.ndarray,
    moment_orders: np.ndarray,
    f_max: float,
):
    """Uniformly rescale the grid and project the population onto it.

    Returns ``(population, pivots, was_adapted)``.
    """
    from ._moments import d43_from_moments, moment_vector

    population0 = np.asarray(population0, dtype=float)
    pivots0 = np.asarray(pivots0, dtype=float)
    diagnostic_moments = moment_vector(pivots0, population0, 5)
    d43_value = d43_from_moments(diagnostic_moments)
    target_upper = f_max * d43_value

    if not np.isfinite(target_upper) or target_upper <= 0:
        return population0.copy(), pivots0.copy(), False

    scale = target_upper / pivots0[-1]
    pivots = pivots0 * scale
    population = np.zeros_like(population0)
    stencil_size = moment_orders.size

    for i, source_pivot in enumerate(pivots0):
        source_population = population0[i]
        first = local_stencil_start(pivots, source_pivot, stencil_size)
        stencil = slice(first, first + stencil_size)
        source_moments = source_pivot ** moment_orders
        weights = solve_moment_weights(pivots[stencil], source_moments, moment_orders)
        population[stencil] += source_population * weights

    return population, pivots, True
