"""Three-panel validation for the Alopaeus coalescence and breakage kernels.

The liquid-liquid properties and closure coefficients follow Alopaeus (2022)
and the corresponding legacy MATLAB implementation. The coalescence product
and beta-daughter tables are built once; only the absolute-size-dependent rate
coefficients are refreshed after grid scaling.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples"))

import adaptivehmmc  # noqa: E402
from adaptivehmmc._breakage import (  # noqa: E402
    breakage_rhs,
    build_alopaeus_beta_breakage_table,
)
from adaptivehmmc._coalescence import build_coalescence_table, coalescence_rhs  # noqa: E402
from adaptivehmmc._alopaeus_kernels import (  # noqa: E402
    AlopaeusKernelParameters,
    alopaeus_breakage_rate,
    alopaeus_coalescence_kernel,
)
from adaptivehmmc._grid import widths_from_pivots  # noqa: E402
from adaptivehmmc._redistribution import redistribute_population  # noqa: E402
from adaptivehmmc.solve import (  # noqa: E402
    SolveResult,
    _append_state,
    _make_adaptation_event,
    _make_history,
    _pack_result,
)
from figure_breakage_benchmark_three_panel import (  # noqa: E402
    FIGS,
    PsdCase,
    build_rows_and_curves,
    geometric_grid,
    plot_three_panel_case_validation,
    result_cdf_at,
)


def alopaeus_initial_state(num_classes, diameter_min, diameter_max, parameters):
    """Return the existing log-normal initial condition at the specified holdup."""
    pivots, widths = geometric_grid(num_classes, diameter_min, diameter_max, 1.30)
    center = 10.0e-6
    sigma = 0.20
    raw = np.exp(-0.5 * (np.log(pivots / center) / sigma) ** 2) / pivots
    raw *= widths
    total_number = parameters.dispersed_holdup / ((np.pi / 6.0) * center**3)
    return pivots, widths, total_number * raw / raw.sum()


def alopaeus_options(pivots, population, final_time, adaptive):
    options = adaptivehmmc.default_options()
    options.time_span = (0.0, final_time)
    options.num_classes = pivots.size
    options.num_moments = 6
    options.adaptive = adaptive
    options.f_max = 4
    options.f_mult = 1.3
    options.initial_distribution.type = "custom"
    options.initial_distribution.pivots = pivots
    options.initial_distribution.population = population
    options.solver.rel_tol = 1.0e-6
    options.solver.abs_tol = 1.0e-3
    options.solver.max_step = 100.0
    return options


def run_alopaeus_coalescence_breakage(options, parameters):
    """Solve coalescence and breakage, updating physical rates after adaptation."""
    pivots = np.asarray(options.initial_distribution.pivots, dtype=float).copy()
    population = np.asarray(options.initial_distribution.population, dtype=float).copy()
    orders = np.arange(options.num_moments)
    coalescence_table = build_coalescence_table(pivots, orders)
    breakage_table = build_alopaeus_beta_breakage_table(pivots, orders)
    history = _make_history(options.num_classes, options.num_moments)
    _append_state(history, 0.0, population, pivots, options.num_moments, options.f_max)

    adaptation_times = []
    total_steps = 0
    current_time = 0.0
    final_time = float(options.time_span[-1])
    started = time.perf_counter()

    while current_time < final_time:
        coalescence_kernel = alopaeus_coalescence_kernel(pivots, parameters)
        breakage_frequency = alopaeus_breakage_rate(pivots, parameters)

        def rhs(t, y):
            return (
                coalescence_rhs(t, y, coalescence_table, coalescence_kernel)
                + breakage_rhs(t, y, breakage_table, breakage_frequency)
            )

        events = _make_adaptation_event(pivots, options, current_time) if options.adaptive else None
        solution = solve_ivp(
            rhs,
            (current_time, final_time),
            population,
            method="BDF",
            rtol=options.solver.rel_tol,
            atol=options.solver.abs_tol,
            max_step=options.solver.max_step,
            events=events,
        )
        if not solution.success:
            raise RuntimeError(f"Alopaeus-kernel integration failed: {solution.message}")

        total_steps += solution.t.size
        for index in range(1, solution.t.size):
            _append_state(
                history,
                solution.t[index],
                solution.y[:, index],
                pivots,
                options.num_moments,
                options.f_max,
            )

        current_time = float(solution.t[-1])
        population = solution.y[:, -1].copy()
        triggered = events is not None and solution.t_events[0].size > 0
        if not triggered or current_time >= final_time:
            break

        population, pivots, was_adapted = redistribute_population(
            population, pivots, orders, options.f_max
        )
        if not was_adapted:
            break
        adaptation_times.append(current_time)
        _append_state(history, current_time, population, pivots, options.num_moments, options.f_max)

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
        return np.vstack([np.interp(times, result.time, values) for values in result.moments])

    return reference_density, reference_quantile, reference_moments


def alopaeus_case():
    parameters = AlopaeusKernelParameters()
    final_time = 3600.0
    pivots, _, population = alopaeus_initial_state(21, 1.0e-6, 1.0e-3, parameters)

    fixed = run_alopaeus_coalescence_breakage(
        alopaeus_options(pivots, population, final_time, adaptive=False), parameters
    )
    adaptive = run_alopaeus_coalescence_breakage(
        alopaeus_options(pivots, population, final_time, adaptive=True), parameters
    )

    reference_pivots, _, reference_population = alopaeus_initial_state(
        81, 1.0e-7, 2.0e-3, parameters
    )
    reference = run_alopaeus_coalescence_breakage(
        alopaeus_options(reference_pivots, reference_population, final_time, adaptive=False),
        parameters,
    )
    density, quantile, moments = refined_reference_case(reference)
    case = PsdCase(
        name="Size-dependent coalescence + breakage",
        fixed=fixed,
        adaptive=adaptive,
        reference_density=density,
        reference_quantile=quantile,
        reference_moments=moments,
        figure_basename="alopaeus_coalescence_breakage_moment_validation",
        grid_ylog=True,
        moment_clip_negative=True,
        log_x=True,
    )
    case.reference_label = "Refined-grid reference"
    case.diameter_scale = 1.0e3
    case.diameter_unit = "mm"
    case.continuous_numerical_cdf = True
    return case


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    case = alopaeus_case()
    row, curve = build_rows_and_curves([case])
    output = plot_three_panel_case_validation(curve[0])
    assert np.all(np.isfinite(case.fixed.moments))
    assert np.all(np.isfinite(case.adaptive.moments))
    print(f"Saved {FIGS / (output + '.pdf')}")
    print(
        "Size-dependent coalescence + breakage: "
        f"fixed CDF error={100.0 * row[0]['fixed_max']:.2f}%, "
        f"adaptive CDF error={100.0 * row[0]['adaptive_max']:.2f}%, "
        f"adaptations={row[0]['adaptations']}"
    )
    reference_final = case.reference_moments(
        np.array([case.fixed.time[-1]]), case.fixed.moments[:, 0]
    )[:, 0]
    reference_d43 = reference_final[4] / reference_final[3]
    fixed_d43 = case.fixed.moments[4, -1] / case.fixed.moments[3, -1]
    adaptive_d43 = case.adaptive.moments[4, -1] / case.adaptive.moments[3, -1]
    print(
        f"final d43 reference={1.0e3 * reference_d43:.6f} mm, "
        f"fixed error={100.0 * abs(fixed_d43 / reference_d43 - 1.0):.2f}%, "
        f"adaptive error={100.0 * abs(adaptive_d43 / reference_d43 - 1.0):.2f}%"
    )


if __name__ == "__main__":
    main()
