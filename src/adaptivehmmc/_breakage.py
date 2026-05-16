"""Binary equal-volume breakage machinery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._redistribution import local_stencil_start, solve_moment_weights


@dataclass
class BreakageTable:
    """Precomputed daughter redistribution matrix."""

    B: np.ndarray   # (num_classes, num_classes)
    num_classes: int


def build_breakage_table(pivots: np.ndarray, moment_orders: np.ndarray) -> BreakageTable:
    """Daughter redistribution table for binary equal-volume breakage.

    Each mother ``d_j`` fragments into two daughters at ``d_j / 2^(1/3)``. The
    two daughters contribute a moment vector ``2 (d_j / 2^(1/3))^q``,
    distributed by local moment matching on a stencil around the daughter
    position. The construction depends only on dimensionless pivot ratios and
    is therefore invariant under uniform grid scaling.
    """
    pivots = np.asarray(pivots, dtype=float)
    n = pivots.size
    s = moment_orders.size
    B = np.zeros((n, n))
    inv_two = 2.0 ** (1.0 / 3.0)

    for j in range(n):
        daughter_pivot = pivots[j] / inv_two
        target_moments = 2.0 ** (1.0 - moment_orders / 3.0) * pivots[j] ** moment_orders
        first = local_stencil_start(pivots, daughter_pivot, s)
        stencil = slice(first, first + s)
        weights = solve_moment_weights(pivots[stencil], target_moments, moment_orders)

        volume = np.sum(weights * pivots[stencil] ** 3)
        if volume != 0.0:
            weights = weights * (pivots[j] ** 3 / volume)

        B[stencil, j] = weights

    return BreakageTable(B=B, num_classes=n)


def breakage_rhs(_t: float, y: np.ndarray, table: BreakageTable, rate: float) -> np.ndarray:
    """Right-hand side ``dY/dt = rate * (B - I) @ Y`` for binary breakage."""
    y = np.asarray(y, dtype=float).ravel()
    return rate * (table.B @ y - y)
