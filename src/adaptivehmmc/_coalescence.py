"""Constant-kernel coalescence machinery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._redistribution import local_stencil_start, solve_moment_weights


@dataclass
class CoalescenceTable:
    """Precomputed product-redistribution weights and stencil pointers."""

    weights: np.ndarray   # (num_classes, num_classes, stencil_size)
    first_index: np.ndarray   # (num_classes, num_classes) int
    stencil_size: int
    num_classes: int


def build_coalescence_table(pivots: np.ndarray, moment_orders: np.ndarray) -> CoalescenceTable:
    """Precompute the product redistribution table for ordered pairs (i, j)."""
    pivots = np.asarray(pivots, dtype=float)
    n = pivots.size
    s = moment_orders.size
    weights = np.zeros((n, n, s))
    first_index = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i, n):
            product_pivot = (pivots[i] ** 3 + pivots[j] ** 3) ** (1.0 / 3.0)
            first = local_stencil_start(pivots, product_pivot, s)
            stencil = slice(first, first + s)

            product_moments = product_pivot ** moment_orders
            local_weights = solve_moment_weights(pivots[stencil], product_moments, moment_orders)

            volume = np.sum(local_weights * pivots[stencil] ** 3)
            if volume != 0.0:
                local_weights = local_weights * ((pivots[i] ** 3 + pivots[j] ** 3) / volume)

            weights[i, j, :] = local_weights
            weights[j, i, :] = local_weights
            first_index[i, j] = first
            first_index[j, i] = first

    return CoalescenceTable(
        weights=weights,
        first_index=first_index,
        stencil_size=s,
        num_classes=n,
    )


def coalescence_rhs(_t: float, y: np.ndarray, table: CoalescenceTable, kernel_rate) -> np.ndarray:
    """Right-hand side for scalar or pairwise coalescence kernels."""
    y = np.asarray(y, dtype=float).ravel()
    kernel_rate = np.asarray(kernel_rate, dtype=float)
    scalar_kernel = kernel_rate.ndim == 0
    n = table.num_classes
    dydt = np.zeros(n)

    for i in range(n):
        yi = y[i]
        for j in range(i, n):
            yj = y[j]
            rate_coefficient = kernel_rate if scalar_kernel else kernel_rate[i, j]
            rate = rate_coefficient * yi * yj
            event_rate = 0.5 * rate if i == j else rate

            first = table.first_index[i, j]
            for k in range(table.stencil_size):
                destination = first + k
                if destination < n:
                    dydt[destination] += event_rate * table.weights[i, j, k]

            dydt[i] -= event_rate
            dydt[j] -= event_rate

    return dydt
