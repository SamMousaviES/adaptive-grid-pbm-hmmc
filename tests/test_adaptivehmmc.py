"""Unit tests for the adaptivehmmc package."""

from __future__ import annotations

import numpy as np
import pytest

import adaptivehmmc


def coalescence_reference(time: float, initial_moments: np.ndarray, rate: float) -> np.ndarray:
    initial_number = initial_moments[0]
    factor = 2.0 / (2.0 + rate * initial_number * time)
    orders = np.arange(initial_moments.size)
    return initial_moments * factor ** (1.0 - orders / 3.0)


def test_package_loads():
    opts = adaptivehmmc.default_options()
    assert opts.num_classes == 20
    opts.time_span = (0.0, 5.0)
    result = adaptivehmmc.solve(opts)
    assert result.moments.shape[1] > 1
    assert result.time.size == result.moments.shape[1]


def test_volume_moment_conserved():
    opts = adaptivehmmc.default_options()
    opts.time_span = (0.0, 200.0)
    opts.num_classes = 16
    opts.adaptive = True
    result = adaptivehmmc.solve(opts)
    rel_volume_change = abs(result.moments[3, -1] - result.moments[3, 0]) / abs(result.moments[3, 0])
    assert rel_volume_change < 1.0e-5


def test_adaptive_improves_over_narrow_fixed_grid():
    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 600.0)
    base.num_classes = 16
    base.f_mult = 1.4

    fixed_opts = base.copy()
    fixed_opts.adaptive = False
    fixed = adaptivehmmc.solve(fixed_opts)

    adaptive_opts = base.copy()
    adaptive_opts.adaptive = True
    adaptive = adaptivehmmc.solve(adaptive_opts)

    reference = coalescence_reference(base.time_span[-1], fixed.moments[:, 0], base.process.rate)
    fixed_err = np.max(np.abs((fixed.moments[:, -1] - reference) / reference))
    adaptive_err = np.max(np.abs((adaptive.moments[:, -1] - reference) / reference))
    assert adaptive_err < fixed_err


def test_redistribution_preserves_moments():
    opts = adaptivehmmc.default_options()
    opts.time_span = (0.0, 200.0)
    opts.num_classes = 16
    opts.adaptive = True
    result = adaptivehmmc.solve(opts)

    duplicates = np.where(np.diff(result.time) == 0)[0]
    assert duplicates.size > 0
    idx = duplicates[0]
    before = result.moments[:, idx]
    after = result.moments[:, idx + 1]
    rel_difference = np.max(np.abs(after - before) / np.maximum(np.abs(before), np.finfo(float).eps))
    assert rel_difference < 1.0e-5


def test_invalid_options_error():
    opts = adaptivehmmc.default_options()
    opts.num_classes = 3
    opts.num_moments = 6
    with pytest.raises(ValueError):
        adaptivehmmc.solve(opts)


def test_breakage_matches_analytical_with_adaptation():
    from examples_helpers import geometric_grid, lognormal_third_moment_normalized

    pivots, widths = geometric_grid(21, 0.0, 100.0, 1.2)
    initial_population = lognormal_third_moment_normalized(pivots, widths, 85.0, 0.5)

    opts = adaptivehmmc.default_options()
    opts.time_span = (0.0, 50.0)
    opts.num_moments = 6
    opts.process.type = "breakage"
    opts.process.kernel = "binary_equal_volume"
    opts.f_max = 1.45
    opts.f_mult = 1.25
    opts.adaptive = True
    opts.initial_distribution.type = "custom"
    opts.initial_distribution.pivots = pivots
    opts.initial_distribution.population = initial_population
    opts.solver.rel_tol = 1.0e-8
    opts.solver.abs_tol = 1.0e-10

    result = adaptivehmmc.solve(opts)

    orders = np.arange(opts.num_moments)
    initial_moments = result.moments[:, 0]
    reference = initial_moments * np.exp((2.0 ** (1.0 - orders / 3.0) - 1.0) * 50.0)
    err = np.max(np.abs((result.moments[:, -1] - reference) / reference))
    assert err < 0.05  # 5 % over six moments and 50 dimensionless time units
    assert result.report.num_adaptations >= 1


def test_growth_matches_analytical_with_adaptation():
    from examples_helpers import geometric_grid, lognormal_normalized

    pivots, widths = geometric_grid(31, 0.0, 2.0, 1.15)
    initial_population = lognormal_normalized(pivots, widths, 0.45, 0.18)

    opts = adaptivehmmc.default_options()
    opts.time_span = (0.0, 3.0)
    opts.num_moments = 6
    opts.process.type = "growth"
    opts.process.kernel = "proportional"
    opts.f_max = 6.0
    opts.f_mult = 1.25
    opts.adaptive = True
    opts.initial_distribution.type = "custom"
    opts.initial_distribution.pivots = pivots
    opts.initial_distribution.population = initial_population
    opts.solver.num_steps = 500

    result = adaptivehmmc.solve(opts)

    orders = np.arange(opts.num_moments)
    initial_moments = result.moments[:, 0]
    reference = initial_moments * np.exp(orders * 3.0)
    err = np.max(np.abs((result.moments[:, -1] - reference) / reference))
    assert err < 0.10
    assert result.report.num_adaptations >= 1
