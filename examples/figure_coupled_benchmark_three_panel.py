"""Coupled-process validation figures for the adaptive HMMC method.

The coalescence--growth benchmark has an analytical reference for a constant
coalescence kernel and proportional growth. The coalescence--breakage benchmark
uses a refined fixed grid as its reference because binary equal-volume breakage
combined with coalescence has no closed-form PSD solution used here.
"""

from __future__ import annotations

import sys
import time
from math import exp, log
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples"))

import adaptivehmmc  # noqa: E402
from adaptivehmmc._benchmark import benchmark_initial_population  # noqa: E402
from adaptivehmmc._breakage import breakage_rhs, build_breakage_table  # noqa: E402
from adaptivehmmc._coalescence import build_coalescence_table, coalescence_rhs  # noqa: E402
from adaptivehmmc._growth import build_growth_projection_matrix  # noqa: E402
from adaptivehmmc._grid import widths_from_pivots  # noqa: E402
from adaptivehmmc._redistribution import redistribute_population  # noqa: E402
from adaptivehmmc.solve import (  # noqa: E402
    SolveResult,
    _append_state,
    _make_history,
    _ode_solve,
    _pack_result,
    _should_adapt,
)
from figure_breakage_benchmark_three_panel import (  # noqa: E402
    FIGS,
    PsdCase,
    build_rows_and_curves,
    geometric_grid,
    plot_three_panel_case_validation,
    result_cdf_at,
)


def initial_state(num_classes, diameter_min, diameter_max, total_number, volume_scale):
    pivots, widths = geometric_grid(num_classes, diameter_min, diameter_max, 1.0)
    population = benchmark_initial_population(
        pivots, widths, 6, total_number, volume_scale
    )
    return pivots, widths, population


def coupled_options(pivots, population, final_time, f_max, f_mult, adaptive):
    options = adaptivehmmc.default_options()
    options.time_span = (0.0, final_time)
    options.num_classes = pivots.size
    options.num_moments = 6
    options.adaptive = adaptive
    options.f_max = f_max
    options.f_mult = f_mult
    options.initial_distribution.type = "custom"
    options.initial_distribution.pivots = pivots
    options.initial_distribution.population = population
    options.solver.rel_tol = 1.0e-7
    options.solver.abs_tol = 1.0e-10
    options.solver.max_step = 0.1
    return options


def run_coalescence_breakage(options, coalescence_rate, breakage_rate):
    pivots = np.asarray(options.initial_distribution.pivots, dtype=float)
    population = np.asarray(options.initial_distribution.population, dtype=float)
    orders = np.arange(options.num_moments)
    coalescence_table = build_coalescence_table(pivots, orders)
    breakage_table = build_breakage_table(pivots, orders)

    def rhs(t, y):
        return (
            coalescence_rhs(t, y, coalescence_table, coalescence_rate)
            + breakage_rhs(t, y, breakage_table, breakage_rate)
        )

    result = _ode_solve(
        options,
        pivots,
        widths_from_pivots(pivots),
        population,
        orders,
        rhs,
    )
    return SolveResult(options=options, **result)


def run_coalescence_growth(
    options, coalescence_rate, growth_rate, num_steps
):
    """Use Strang splitting for constant-kernel coalescence and proportional growth."""
    pivots = np.asarray(options.initial_distribution.pivots, dtype=float).copy()
    population = np.asarray(options.initial_distribution.population, dtype=float).copy()
    orders = np.arange(options.num_moments)
    coalescence_table = build_coalescence_table(pivots, orders)
    history = _make_history(options.num_classes, options.num_moments)
    _append_state(history, 0.0, population, pivots, options.num_moments, options.f_max)
    adaptation_times = []
    total_steps = 0
    started = time.perf_counter()
    dt = options.time_span[-1] / num_steps
    half_growth = exp(0.5 * growth_rate * dt)
    current_time = 0.0

    for _ in range(num_steps):
        growth_matrix = build_growth_projection_matrix(pivots, half_growth, orders)
        population = growth_matrix @ population
        solution = solve_ivp(
            lambda t, y: coalescence_rhs(t, y, coalescence_table, coalescence_rate),
            (current_time, current_time + dt),
            population,
            method="BDF",
            rtol=options.solver.rel_tol,
            atol=options.solver.abs_tol,
            max_step=dt,
        )
        if not solution.success:
            raise RuntimeError(f"Coupled coalescence-growth integration failed: {solution.message}")
        total_steps += solution.t.size
        population = growth_matrix @ solution.y[:, -1]
        current_time += dt
        _append_state(history, current_time, population, pivots, options.num_moments, options.f_max)

        if options.adaptive and _should_adapt(population, pivots, options):
            population, pivots, was_adapted = redistribute_population(
                population, pivots, orders, options.f_max
            )
            if was_adapted:
                adaptation_times.append(current_time)
                _append_state(
                    history, current_time, population, pivots, options.num_moments, options.f_max
                )

    return SolveResult(
        options=options,
        **_pack_result(history, adaptation_times, total_steps, started, options),
    )


def refined_reference_case(result):
    pivots = np.asarray(result.pivots[:, -1], dtype=float)
    widths = widths_from_pivots(pivots)
    population = np.maximum(np.asarray(result.population[:, -1], dtype=float), 0.0)
    density = population / widths
    cdf = result_cdf_at(result, pivots, moment_power=3)

    def reference_density(diameter):
        return np.interp(diameter, pivots, density, left=0.0, right=0.0)

    def reference_quantile(probability):
        return float(np.interp(probability, np.r_[0.0, cdf], np.r_[0.5 * pivots[0], pivots]))

    def reference_moments(times, _initial_moments):
        return np.vstack(
            [np.interp(times, result.time, values) for values in result.moments]
        )

    return reference_density, reference_quantile, reference_moments


def coalescence_breakage_case():
    total_number = 0.025
    volume_scale = 0.04
    final_time = 10.0
    coalescence_rate = 4.0
    breakage_rate = 2.5
    diameter_min = 1.0e-3
    diameter_max = 1.0
    f_max = 2.5
    f_mult = 1.1

    pivots, _, population = initial_state(
        21, diameter_min, diameter_max, total_number, volume_scale
    )
    fixed = run_coalescence_breakage(
        coupled_options(pivots, population, final_time, f_max, f_mult, False),
        coalescence_rate,
        breakage_rate,
    )
    adaptive = run_coalescence_breakage(
        coupled_options(pivots, population, final_time, f_max, f_mult, True),
        coalescence_rate,
        breakage_rate,
    )
    reference_pivots, _, reference_population = initial_state(
        81, diameter_min, diameter_max, total_number, volume_scale
    )
    reference = run_coalescence_breakage(
        coupled_options(reference_pivots, reference_population, final_time, f_max, f_mult, False),
        coalescence_rate,
        breakage_rate,
    )
    density, quantile, moments = refined_reference_case(reference)
    case = PsdCase(
        name="Coalescence + breakage",
        fixed=fixed,
        adaptive=adaptive,
        reference_density=density,
        reference_quantile=quantile,
        reference_moments=moments,
        figure_basename="coupled_coalescence_breakage_moment_validation",
        grid_ylog=True,
        moment_clip_negative=False,
        log_x=True,
    )
    case.reference_label = "Refined-grid reference"
    return case


def coalescence_growth_case():
    total_number = 0.025
    volume_scale = 0.04
    final_time = 10.0
    coalescence_rate = 5.0
    growth_rate = 0.12
    f_max = 2.5
    f_mult = 1.3

    pivots, _, population = initial_state(21, 1.0e-3, 1.0, total_number, volume_scale)
    fixed = run_coalescence_growth(
        coupled_options(pivots, population, final_time, f_max, f_mult, False),
        coalescence_rate,
        growth_rate,
        40,
    )
    adaptive = run_coalescence_growth(
        coupled_options(pivots, population, final_time, f_max, f_mult, True),
        coalescence_rate,
        growth_rate,
        40,
    )
    factor = lambda t: 2.0 / (2.0 + coalescence_rate * total_number * t)

    def reference_density(diameter):
        scale = exp(growth_rate * final_time)
        a = factor(final_time)
        diameter = np.asarray(diameter, dtype=float)
        return (
            3.0 * total_number * a**2 * diameter**2
            * np.exp(-a * diameter**3 / (volume_scale * scale**3))
            / (volume_scale * scale**3)
        )

    def reference_quantile(probability):
        scale = exp(growth_rate * final_time)
        return scale * (-volume_scale / factor(final_time) * log(1.0 - probability)) ** (1.0 / 3.0)

    def reference_moments(times, initial_moments):
        times = np.asarray(times, dtype=float)
        orders = np.arange(initial_moments.size)
        return initial_moments[:, None] * np.exp(growth_rate * times[None, :] * orders[:, None]) * (
            factor(times)[None, :] ** (1.0 - orders[:, None] / 3.0)
        )

    return PsdCase(
        name="Coalescence + growth",
        fixed=fixed,
        adaptive=adaptive,
        reference_density=reference_density,
        reference_quantile=reference_quantile,
        reference_moments=reference_moments,
        figure_basename="coupled_coalescence_growth_moment_validation",
        grid_ylog=False,
        moment_clip_negative=True,
        log_x=False,
    )


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    cases = [coalescence_breakage_case(), coalescence_growth_case()]
    rows, curves = build_rows_and_curves(cases)
    for curve in curves:
        print(f"Saved {FIGS / (plot_three_panel_case_validation(curve) + '.pdf')}")
    for row in rows:
        print(
            f"{row['case']}: fixed CDF error={100.0 * row['fixed_max']:.2f}%, "
            f"adaptive CDF error={100.0 * row['adaptive_max']:.2f}%, "
            f"adaptations={row['adaptations']}"
        )
    for case in cases:
        assert np.all(np.isfinite(case.fixed.moments))
        assert np.all(np.isfinite(case.adaptive.moments))


if __name__ == "__main__":
    main()
