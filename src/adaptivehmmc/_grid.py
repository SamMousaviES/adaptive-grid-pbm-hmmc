"""Grid construction utilities."""

from __future__ import annotations

import numpy as np


def make_grid(num_classes: int, diameter_min: float, diameter_max: float, grid_ratio: float):
    """Build a monotone pivot grid with geometric interval widths."""
    if grid_ratio == 1:
        weights = np.ones(num_classes)
    else:
        weights = grid_ratio ** np.arange(num_classes)

    widths = weights / weights.sum() * (diameter_max - diameter_min)
    edges = diameter_min + np.concatenate(([0.0], np.cumsum(widths)))
    pivots = 0.5 * (edges[:-1] + edges[1:])
    return pivots, widths


def widths_from_pivots(pivots: np.ndarray) -> np.ndarray:
    """Recover positive interval widths from an arbitrary monotone pivot vector."""
    pivots = np.asarray(pivots, dtype=float)
    n = pivots.size
    widths = np.empty(n)
    if n == 1:
        widths[0] = max(2.0 * pivots[0], 1.0e-12)
        return widths

    midpoints = 0.5 * (pivots[:-1] + pivots[1:])
    widths[0] = 2.0 * (midpoints[0] - pivots[0])
    widths[-1] = 2.0 * (pivots[-1] - midpoints[-1])
    for i in range(1, n - 1):
        widths[i] = midpoints[i] - midpoints[i - 1]
    return np.maximum(widths, 1.0e-12)
