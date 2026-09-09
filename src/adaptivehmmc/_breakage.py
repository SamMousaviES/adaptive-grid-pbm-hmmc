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


def build_beta_breakage_table(pivots: np.ndarray, moment_orders: np.ndarray) -> BreakageTable:
    """Build the CT beta-daughter redistribution table.

    For a mother pivot ``d_j``, the daughter density is
    ``180 d^2 / d_j^3 * (d^3 / d_j^3)^2 * (1 - d^3 / d_j^3)^2`` on
    ``0 <= d <= d_j``. Each daughter interval is integrated analytically and
    then distributed on a local moment-matching stencil. The resulting table
    depends only on pivot ratios and is invariant under uniform scaling.
    """
    pivots = np.asarray(pivots, dtype=float)
    moment_orders = np.asarray(moment_orders, dtype=float)
    n = pivots.size
    s = moment_orders.size
    B = np.zeros((n, n))

    for mother, mother_pivot in enumerate(pivots):
        for daughter in range(mother + 1):
            lower = 0.0 if daughter == 0 else 0.5 * (pivots[daughter - 1] + pivots[daughter])
            upper = mother_pivot if daughter == mother else 0.5 * (pivots[daughter] + pivots[daughter + 1])
            lower_ratio = lower / mother_pivot
            upper_ratio = upper / mother_pivot
            moments = 180.0 * mother_pivot**moment_orders * (
                (upper_ratio ** (moment_orders + 9.0) - lower_ratio ** (moment_orders + 9.0))
                / (moment_orders + 9.0)
                - 2.0
                * (upper_ratio ** (moment_orders + 12.0) - lower_ratio ** (moment_orders + 12.0))
                / (moment_orders + 12.0)
                + (upper_ratio ** (moment_orders + 15.0) - lower_ratio ** (moment_orders + 15.0))
                / (moment_orders + 15.0)
            )
            first = local_stencil_start(pivots, pivots[daughter], s)
            stencil = slice(first, first + s)
            B[stencil, mother] += solve_moment_weights(
                pivots[stencil], moments, moment_orders
            )

        volume = np.sum(B[:, mother] * pivots**3)
        if volume != 0.0:
            B[:, mother] *= pivots[mother] ** 3 / volume

    return BreakageTable(B=B, num_classes=n)


def build_alopaeus_beta_breakage_table(
    pivots: np.ndarray, moment_orders: np.ndarray
) -> BreakageTable:
    """Build the binary beta-daughter table used by Alopaeus (2022).

    The published density has coefficient 90 and integrates to one daughter.
    Binary breakage contributes two daughters, so its HMMC birth moments are
    twice the published density moments. This is algebraically identical to
    the coefficient-180 table assembled by :func:`build_beta_breakage_table`.
    """
    return build_beta_breakage_table(pivots, moment_orders)


def breakage_rhs(_t: float, y: np.ndarray, table: BreakageTable, rate) -> np.ndarray:
    """Right-hand side for scalar or pivot-dependent binary-breakage rates."""
    y = np.asarray(y, dtype=float).ravel()
    weighted_population = np.asarray(rate, dtype=float) * y
    return table.B @ weighted_population - weighted_population
