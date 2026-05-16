"""Main solver entry point."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ._benchmark import benchmark_initial_population
from ._breakage import BreakageTable, breakage_rhs, build_breakage_table
from ._coalescence import CoalescenceTable, build_coalescence_table, coalescence_rhs
from ._grid import make_grid, widths_from_pivots
from ._growth import build_growth_projection_matrix
from ._moments import d43_from_moments, moment_vector
from ._redistribution import redistribute_population
from .options import SolverOptions


@dataclass
class SolveReport:
    final_time: float = 0.0
    num_classes: int = 0
    num_moments: int = 0
    num_adaptations: int = 0
    total_ode_steps: int = 0
    cpu_time: float = 0.0
    initial_largest_pivot: float = 0.0
    final_largest_pivot: float = 0.0
    final_moments: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class SolveResult:
    """Time history and diagnostics from :func:`solve`."""

    time: np.ndarray
    population: np.ndarray     # (num_classes, num_times)
    pivots: np.ndarray         # (num_classes, num_times)
    moments: np.ndarray        # (num_moments, num_times)
    d43: np.ndarray
    target_upper_pivot: np.ndarray
    adaptation_times: np.ndarray
    report: SolveReport
    options: SolverOptions


def solve(options: Optional[SolverOptions] = None) -> SolveResult:
    """Solve a population balance for the configured process.

    Parameters
    ----------
    options:
        :class:`SolverOptions` instance. If omitted, ``default_options()`` is
        used.
    """
    if options is None:
        from .options import default_options

        options = default_options()
    options = _validate(options)

    pivots, widths, population = _initialize_state(options)
    moment_orders = np.arange(options.num_moments)

    process_type = options.process.type
    if process_type == "coalescence":
        table = build_coalescence_table(pivots, moment_orders)
        rhs = _make_coalescence_rhs(table, options.process.rate)
        result = _ode_solve(options, pivots, widths, population, moment_orders, rhs)
    elif process_type == "breakage":
        table = build_breakage_table(pivots, moment_orders)
        rhs = _make_breakage_rhs(table, options.process.rate)
        result = _ode_solve(options, pivots, widths, population, moment_orders, rhs)
    elif process_type == "growth":
        result = _growth_solve(options, pivots, widths, population, moment_orders)
    else:
        raise ValueError(f"Unsupported process type: {process_type!r}")

    return SolveResult(options=options, **result)


def _make_coalescence_rhs(table: CoalescenceTable, rate: float):
    def rhs(t, y):
        return coalescence_rhs(t, y, table, rate)

    return rhs


def _make_breakage_rhs(table: BreakageTable, rate: float):
    def rhs(t, y):
        return breakage_rhs(t, y, table, rate)

    return rhs


def _validate(options: SolverOptions) -> SolverOptions:
    """Lightweight runtime checks mirroring the MATLAB validateOptions logic."""
    options = options.copy()

    ts = np.asarray(options.time_span, dtype=float)
    if ts.size < 2 or ts[0] < 0 or ts[-1] <= ts[0]:
        raise ValueError("time_span must contain at least two increasing nonnegative times.")
    options.time_span = tuple(ts.tolist())

    if not _is_int(options.num_classes) or options.num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2.")
    if not _is_int(options.num_moments) or options.num_moments < 1 or options.num_moments > options.num_classes:
        raise ValueError("num_moments must be a positive integer at most num_classes.")

    if options.diameter_min < 0 or options.diameter_max <= options.diameter_min:
        raise ValueError("diameter_max must exceed nonnegative diameter_min.")
    if options.grid_ratio < 1:
        raise ValueError("grid_ratio must be at least 1.")
    if options.f_max <= 0 or options.f_mult <= 1:
        raise ValueError("f_max must be positive and f_mult > 1.")

    process_type = options.process.type.lower()
    kernel = options.process.kernel.lower()
    options.process.type = process_type
    options.process.kernel = kernel

    valid_pairs = {
        "coalescence": "constant",
        "breakage": "binary_equal_volume",
        "growth": "proportional",
    }
    if process_type not in valid_pairs:
        raise ValueError(
            f"process.type must be one of {list(valid_pairs.keys())}, got {process_type!r}."
        )
    if kernel != valid_pairs[process_type]:
        raise ValueError(
            f"process.kernel for {process_type!r} must be {valid_pairs[process_type]!r}, got {kernel!r}."
        )
    if options.process.rate <= 0:
        raise ValueError("process.rate must be positive.")

    init = options.initial_distribution
    init.type = init.type.lower()
    if init.type not in {"benchmark", "custom"}:
        raise ValueError(f"initial_distribution.type must be benchmark or custom, got {init.type!r}.")
    if init.type == "custom":
        if init.population is None:
            raise ValueError("Custom initial_distribution requires a population vector.")
        init.population = np.asarray(init.population, dtype=float).ravel()
        if init.pivots is not None:
            init.pivots = np.asarray(init.pivots, dtype=float).ravel()
            if init.pivots.size != init.population.size:
                raise ValueError("Custom pivots and population must have the same length.")
            options.num_classes = init.population.size
            if options.num_moments > options.num_classes:
                raise ValueError("num_moments cannot exceed the custom population length.")
        elif init.population.size != options.num_classes:
            raise ValueError("Custom population length must match num_classes.")

    if options.solver.rel_tol <= 0 or options.solver.abs_tol <= 0:
        raise ValueError("Solver tolerances must be positive.")
    if options.solver.max_step is not None and options.solver.max_step <= 0:
        raise ValueError("solver.max_step must be positive when set.")
    if process_type == "growth":
        if not _is_int(options.solver.num_steps) or options.solver.num_steps < 1:
            raise ValueError("solver.num_steps must be a positive integer for growth.")

    return options


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _initialize_state(options: SolverOptions):
    init = options.initial_distribution
    if init.type == "custom" and init.pivots is not None:
        pivots = np.asarray(init.pivots, dtype=float).ravel()
        population = np.asarray(init.population, dtype=float).ravel()
        widths = widths_from_pivots(pivots)
        return pivots, widths, population

    pivots, widths = make_grid(options.num_classes, options.diameter_min,
                               options.diameter_max, options.grid_ratio)

    if init.type == "custom":
        population = np.asarray(init.population, dtype=float).ravel()
    elif init.type == "benchmark":
        population = benchmark_initial_population(pivots, widths, options.num_moments,
                                                  init.total_number, init.volume_scale)
    else:
        raise ValueError(f"Unsupported initial_distribution.type: {init.type!r}")

    return pivots, widths, population


def _make_history(num_classes: int, num_moments: int):
    return {
        "time": [],
        "population": [],
        "pivots": [],
        "moments": [],
        "d43": [],
        "target_upper_pivot": [],
        "_num_classes": num_classes,
        "_num_moments": num_moments,
    }


def _append_state(history, t, population, pivots, num_moments, f_max):
    moments = moment_vector(pivots, population, max(num_moments, 5))
    d43_value = d43_from_moments(moments)
    history["time"].append(float(t))
    history["population"].append(np.asarray(population, dtype=float).copy())
    history["pivots"].append(np.asarray(pivots, dtype=float).copy())
    history["moments"].append(np.asarray(moments[:num_moments], dtype=float).copy())
    history["d43"].append(float(d43_value))
    history["target_upper_pivot"].append(float(f_max * d43_value))


def _finalize_history(history):
    return {
        "time": np.asarray(history["time"], dtype=float),
        "population": np.asarray(history["population"], dtype=float).T,
        "pivots": np.asarray(history["pivots"], dtype=float).T,
        "moments": np.asarray(history["moments"], dtype=float).T,
        "d43": np.asarray(history["d43"], dtype=float),
        "target_upper_pivot": np.asarray(history["target_upper_pivot"], dtype=float),
    }


def _pack_result(history, adaptation_times, total_steps, t0, options):
    arrays = _finalize_history(history)
    report = SolveReport(
        final_time=float(arrays["time"][-1]),
        num_classes=options.num_classes,
        num_moments=options.num_moments,
        num_adaptations=len(adaptation_times),
        total_ode_steps=int(total_steps),
        cpu_time=float(_time.perf_counter() - t0),
        initial_largest_pivot=float(arrays["pivots"][-1, 0]),
        final_largest_pivot=float(arrays["pivots"][-1, -1]),
        final_moments=arrays["moments"][:, -1].copy(),
    )
    return {
        **arrays,
        "adaptation_times": np.asarray(adaptation_times, dtype=float),
        "report": report,
    }


def _make_adaptation_event(pivots: np.ndarray, options: SolverOptions, segment_start: float):
    """Return a scipy.integrate.solve_ivp event callable.

    The event flags `value` zero-crossings; we expose a smooth surrogate that
    becomes nonpositive exactly when the d43-based criterion is satisfied.
    """
    f_max = options.f_max
    f_mult = options.f_mult
    largest = pivots[-1]
    upper_band = f_mult * largest
    lower_band = largest / f_mult

    def event(t, y):
        if t - segment_start <= 1.0e-9:
            return 1.0
        moments = moment_vector(pivots, y, 5)
        d43_value = d43_from_moments(moments)
        target = f_max * d43_value
        if not np.isfinite(target) or target <= 0:
            return 1.0
        # Trigger when target leaves the admissible band: become non-positive.
        upper_excess = target - upper_band       # >0 when above the band
        lower_excess = lower_band - target       # >0 when below the band
        margin = max(upper_excess, lower_excess)
        # event sign: positive while inside band, zero at boundary.
        return -margin

    event.terminal = True
    event.direction = 0
    return event


def _ode_solve(options, pivots, widths, population, moment_orders, rhs):
    t0_value, tf_value = options.time_span[0], options.time_span[-1]
    history = _make_history(options.num_classes, options.num_moments)
    _append_state(history, t0_value, population, pivots, options.num_moments, options.f_max)

    adaptation_times = []
    total_steps = 0
    perf_start = _time.perf_counter()

    current_time = t0_value
    current_pivots = pivots
    current_population = population
    _ = widths   # widths are not consulted on the ODE path after initialization

    while current_time < tf_value:
        events = None
        if options.adaptive:
            events = _make_adaptation_event(current_pivots, options, current_time)

        kwargs = dict(
            t_span=(current_time, tf_value),
            y0=current_population,
            method="BDF",
            rtol=options.solver.rel_tol,
            atol=options.solver.abs_tol,
            dense_output=False,
        )
        if options.solver.max_step is not None:
            kwargs["max_step"] = options.solver.max_step
        if events is not None:
            kwargs["events"] = events

        sol = solve_ivp(rhs, **kwargs)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        ts = sol.t
        ys = sol.y
        total_steps += ts.size

        if ts.size > 1:
            start_index = 1 if len(history["time"]) > 0 else 0
            if options.solver.store_history:
                for k in range(start_index, ts.size):
                    _append_state(history, ts[k], ys[:, k], current_pivots,
                                  options.num_moments, options.f_max)
            else:
                _append_state(history, ts[-1], ys[:, -1], current_pivots,
                              options.num_moments, options.f_max)

        current_time = float(ts[-1])
        current_population = ys[:, -1].copy()

        triggered = events is not None and sol.t_events is not None and \
                    len(sol.t_events) > 0 and sol.t_events[0].size > 0
        if not options.adaptive or current_time >= tf_value or not triggered:
            break

        new_population, new_pivots, was_adapted = redistribute_population(
            current_population, current_pivots, moment_orders, options.f_max
        )
        if not was_adapted:
            break

        current_population = new_population
        current_pivots = new_pivots
        adaptation_times.append(current_time)
        _append_state(history, current_time, current_population, current_pivots,
                      options.num_moments, options.f_max)

    return _pack_result(history, adaptation_times, total_steps, perf_start, options)


def _growth_solve(options, pivots, widths, population, moment_orders):
    t0_value, tf_value = options.time_span[0], options.time_span[-1]
    num_steps = options.solver.num_steps
    dt = (tf_value - t0_value) / num_steps
    growth_factor = float(np.exp(options.process.rate * dt))

    history = _make_history(options.num_classes, options.num_moments)
    _append_state(history, t0_value, population, pivots, options.num_moments, options.f_max)

    adaptation_times = []
    perf_start = _time.perf_counter()

    current_time = t0_value
    current_pivots = pivots
    current_population = population
    _ = widths

    projection_matrix = build_growth_projection_matrix(current_pivots, growth_factor, moment_orders)

    for _step in range(num_steps):
        current_population = projection_matrix @ current_population
        current_time += dt

        if options.solver.store_history:
            _append_state(history, current_time, current_population, current_pivots,
                          options.num_moments, options.f_max)

        if options.adaptive and _should_adapt(current_population, current_pivots, options):
            new_population, new_pivots, was_adapted = redistribute_population(
                current_population, current_pivots, moment_orders, options.f_max
            )
            if was_adapted:
                current_population = new_population
                current_pivots = new_pivots
                # Projection matrix is scale-invariant under uniform pivot rescaling,
                # so it can be reused. (Recompute defensively only if the user changed
                # the relative grid layout, which uniform scaling does not.)
                adaptation_times.append(current_time)
                if options.solver.store_history:
                    _append_state(history, current_time, current_population, current_pivots,
                                  options.num_moments, options.f_max)

    if not options.solver.store_history:
        _append_state(history, current_time, current_population, current_pivots,
                      options.num_moments, options.f_max)

    return _pack_result(history, adaptation_times, num_steps, perf_start, options)


def _should_adapt(population, pivots, options):
    moments = moment_vector(pivots, population, 5)
    d43_value = d43_from_moments(moments)
    target = options.f_max * d43_value
    if not np.isfinite(target) or target <= 0:
        return False
    largest = pivots[-1]
    return target > options.f_mult * largest or target < largest / options.f_mult
