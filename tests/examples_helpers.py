"""Small helpers shared by tests that mirror the example-script initial conditions."""

from __future__ import annotations

import numpy as np


def geometric_grid(num_classes, diameter_min, diameter_max, grid_ratio):
    weights = grid_ratio ** np.arange(num_classes)
    widths = weights / weights.sum() * (diameter_max - diameter_min)
    edges = diameter_min + np.concatenate(([0.0], np.cumsum(widths)))
    pivots = 0.5 * (edges[:-1] + edges[1:])
    return pivots, widths


def lognormal_third_moment_normalized(pivots, widths, center, sigma):
    raw = np.exp(-0.5 * (np.log(pivots / center) / sigma) ** 2) / pivots
    raw *= widths
    raw /= raw.sum()
    third_moment = np.sum(raw * pivots ** 3)
    return raw / third_moment


def lognormal_normalized(pivots, widths, center, sigma):
    raw = np.exp(-0.5 * (np.log(pivots / center) / sigma) ** 2) / pivots
    raw *= widths
    return raw / raw.sum()
