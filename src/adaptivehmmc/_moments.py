"""Diameter-moment helpers."""

from __future__ import annotations

import numpy as np


def moment_vector(pivots: np.ndarray, population: np.ndarray, num_moments: int) -> np.ndarray:
    """Return ``[M_0, M_1, ..., M_{num_moments-1}]`` where ``M_k = sum(Y_i d_i^k)``."""
    pivots = np.asarray(pivots, dtype=float)
    population = np.asarray(population, dtype=float)
    orders = np.arange(num_moments)
    powers = pivots[:, None] ** orders[None, :]
    return powers.T @ population


def d43_from_moments(moments: np.ndarray) -> float:
    """Compute the volume-weighted mean diameter ``d_{43} = M_4 / M_3``."""
    moments = np.asarray(moments, dtype=float)
    if moments.size < 5 or abs(moments[3]) < np.finfo(float).eps:
        return float("nan")
    return float(moments[4] / moments[3])
