"""Reproduce the sum-kernel aggregation test of Kumar and Ramkrishna (1995).

The paper uses particle volume ``v``.  This solver uses a diameter coordinate,
so this script sets ``d = v**(1/3)`` and evaluates the published kernel as
``q(d_i, d_j) = k * (d_i**3 + d_j**3)``.  The printed initial mean volume and
the reported final number ratio are inconsistent under the printed kernel;
therefore ``k`` is calibrated to the reported ``N(40)/N(0) = 0.135``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i1e

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
FIGS = REPO / "figures_out"
sys.path.insert(0, str(REPO / "src"))

from adaptivehmmc._coalescence import build_coalescence_table, coalescence_rhs  # noqa: E402
from adaptivehmmc._redistribution import redistribute_population  # noqa: E402
from adaptivehmmc.options import default_options  # noqa: E402
from adaptivehmmc.solve import (  # noqa: E402
    SolveResult,
    _append_state,
    _make_adaptation_event,
    _make_history,
    _pack_result,
)


FINAL_TIME = 40.0
INITIAL_MEAN_VOLUME = 1.0e-2
PUBLISHED_NUMBER_RATIO = 0.135
SUM_KERNEL_RATE = -np.log(PUBLISHED_NUMBER_RATIO) / (INITIAL_MEAN_VOLUME * FINAL_TIME)


def volume_geometric_pivots(num_classes: int, volume_min: float, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Return diameter pivots and volume-cell edges for a geometric volume grid."""
    edges = volume_min * ratio ** np.arange(num_classes + 1)
    volume_pivots = np.sqrt(edges[:-1] * edges[1:])
    return volume_pivots ** (1.0 / 3.0), edges


def exponential_population(volume_edges: np.ndarray, mean_volume: float) -> np.ndarray:
    """Integrate n(v,0)=exp(-v/v0)/v0 exactly over each volume cell."""
    return np.exp(-volume_edges[:-1] / mean_volume) - np.exp(-volume_edges[1:] / mean_volume)


def analytical_number_ratio(times: np.ndarray) -> np.ndarray:
    """Exact total-number ratio for the additive kernel."""
    return np.exp(-SUM_KERNEL_RATE * INITIAL_MEAN_VOLUME * np.asarray(times, dtype=float))


def analytical_number_density(volumes: np.ndarray) -> np.ndarray:
    """Scott's exact density for the exponential initial condition and sum kernel."""
    volumes = np.asarray(volumes, dtype=float)
    ratio = PUBLISHED_NUMBER_RATIO
    delta = 1.0 - ratio
    x = volumes / INITIAL_MEAN_VOLUME
    argument = 2.0 * np.sqrt(delta) * x
    log_density = (
        np.log(ratio / np.sqrt(delta) / INITIAL_MEAN_VOLUME)
        - (2.0 - ratio) * x
        + argument
        + np.log(i1e(argument))
        - np.log(x)
    )
    return np.exp(log_density)


def analytical_oversize_number(volumes: np.ndarray) -> np.ndarray:
    """Numerically integrate the exact density from each volume to infinity."""
    volumes = np.asarray(volumes, dtype=float)
    density = analytical_number_density(volumes)
    intervals = 0.5 * (density[:-1] + density[1:]) * np.diff(volumes)
    return np.r_[np.cumsum(intervals[::-1])[::-1], 0.0]


def digitized_kumar_moving_pivot_r15() -> tuple[np.ndarray, np.ndarray]:
    """Digitized separated-tail points for the moving-pivot r=1.5 curve in Fig. 4.

    The source figure is logarithmic on both axes. Pixel coordinates were read
    from the local PDF at 300 dpi after calibration against its labelled ticks.
    Only the separated tail is retained because all five published curves
    overlap at small volume.
    """
    pixels = np.array(
        [
            [330.0, 370.0], [360.0, 393.0], [390.0, 431.0],
            [420.0, 473.0], [450.0, 526.0], [480.0, 589.0],
            [510.0, 662.0], [540.0, 730.0], [570.0, 807.0],
            [600.0, 882.0], [620.0, 938.0],
        ]
    )
    log_volume = -1.0 + (pixels[:, 0] - 28.0) * 5.0 / (934.0 - 28.0)
    log_oversize = (173.0 - pixels[:, 1]) / ((1037.0 - 173.0) / 15.0)
    return 10.0**log_volume, 10.0**log_oversize


def options_for(pivots: np.ndarray, population: np.ndarray, adaptive: bool):
    options = default_options()
    options.time_span = (0.0, FINAL_TIME)
    options.num_classes = pivots.size
    # M_0 and M_3 correspond to number and volume in the diameter coordinate.
    options.num_moments = 6
    options.adaptive = adaptive
    options.f_max = 4
    options.f_mult = 1.1
    options.initial_distribution.type = "custom"
    options.initial_distribution.pivots = pivots
    options.initial_distribution.population = population
    options.solver.rel_tol = 1.0e-7
    options.solver.abs_tol = 1.0e-11
    options.solver.max_step = 0.1
    return options


def sum_kernel(pivots: np.ndarray) -> np.ndarray:
    volumes = np.asarray(pivots, dtype=float) ** 3
    return SUM_KERNEL_RATE * (volumes[:, None] + volumes[None, :])


def run_sum_kernel(options) -> SolveResult:
    """Solve the sum-kernel PBE and refresh the absolute-size kernel after scaling."""
    pivots = np.asarray(options.initial_distribution.pivots, dtype=float).copy()
    population = np.asarray(options.initial_distribution.population, dtype=float).copy()
    orders = np.arange(options.num_moments)
    table = build_coalescence_table(pivots, orders)
    history = _make_history(options.num_classes, options.num_moments)
    _append_state(history, 0.0, population, pivots, options.num_moments, options.f_max)

    adaptation_times = []
    total_steps = 0
    current_time = 0.0
    started = time.perf_counter()

    while current_time < FINAL_TIME:
        kernel = sum_kernel(pivots)

        def rhs(t, y):
            return coalescence_rhs(t, y, table, kernel)

        events = _make_adaptation_event(pivots, options, current_time) if options.adaptive else None
        solution = solve_ivp(
            rhs,
            (current_time, FINAL_TIME),
            population,
            method="BDF",
            rtol=options.solver.rel_tol,
            atol=options.solver.abs_tol,
            max_step=options.solver.max_step,
            events=events,
        )
        if not solution.success:
            raise RuntimeError(f"Sum-kernel integration failed: {solution.message}")

        total_steps += solution.t.size
        for index in range(1, solution.t.size):
            _append_state(history, solution.t[index], solution.y[:, index], pivots, options.num_moments, options.f_max)

        current_time = float(solution.t[-1])
        population = solution.y[:, -1].copy()
        triggered = events is not None and solution.t_events[0].size > 0
        if not triggered or current_time >= FINAL_TIME:
            break

        population, pivots, was_adapted = redistribute_population(population, pivots, orders, options.f_max)
        if not was_adapted:
            break
        adaptation_times.append(current_time)
        _append_state(history, current_time, population, pivots, options.num_moments, options.f_max)

    return SolveResult(options=options, **_pack_result(history, adaptation_times, total_steps, started, options))


def moment_history(result: SolveResult, orders: np.ndarray) -> np.ndarray:
    population = np.asarray(result.population, dtype=float)
    pivots = np.asarray(result.pivots, dtype=float)
    return np.stack([np.sum(population * pivots**order, axis=0) for order in orders])


def oversize_number(result: SolveResult, volumes: np.ndarray) -> np.ndarray:
    pivot_volumes = result.pivots[:, -1] ** 3
    population = np.maximum(result.population[:, -1], 0.0)
    cumulative = np.cumsum(population[::-1])[::-1]
    index = np.searchsorted(pivot_volumes, volumes, side="left")
    values = np.zeros_like(volumes)
    valid = index < cumulative.size
    values[valid] = cumulative[index[valid]]
    return values


def front_volume(result: SolveResult, oversize_threshold: float = 1.0e-10) -> float:
    """Return the largest pivot whose cumulative oversize number exceeds a threshold."""
    pivot_volumes = result.pivots[:, -1] ** 3
    cumulative = np.cumsum(np.maximum(result.population[:, -1], 0.0)[::-1])[::-1]
    above = np.flatnonzero(cumulative >= oversize_threshold)
    return float(pivot_volumes[above[-1]]) if above.size else float(pivot_volumes[0])


def run_cases() -> tuple[SolveResult, SolveResult]:
    # The coarse grid starts narrowly enough to require global rescaling.
    coarse_pivots, coarse_edges = volume_geometric_pivots(52, 4.0e-10, 1.5)
    coarse_population = exponential_population(coarse_edges, INITIAL_MEAN_VOLUME)
    fixed = run_sum_kernel(options_for(coarse_pivots, coarse_population, adaptive=False))
    adaptive = run_sum_kernel(options_for(coarse_pivots, coarse_population, adaptive=True))
    return fixed, adaptive


def plot_results(fixed: SolveResult, adaptive: SolveResult) -> Path:
    fig, (ax_moments, ax_grid, ax_oversize) = plt.subplots(3, 1, figsize=(5.0, 7.7))
    colors = {"fixed": "#8c1f1f", "adaptive": "#1a619c", "reference": "black"}

    orders = np.array([0, 3])
    fixed_moments = moment_history(fixed, orders)
    adaptive_moments = moment_history(adaptive, orders)
    analytical_times = np.linspace(0.0, FINAL_TIME, 240)
    analytical_ratios = np.vstack(
        (analytical_number_ratio(analytical_times), np.ones_like(analytical_times))
    )
    for index, order in enumerate(orders):
        ax_moments.semilogy(analytical_times, analytical_ratios[index], "-", color=plt.cm.tab10(index), lw=1.2, label=rf"$M_{order}$")
        ax_moments.semilogy(fixed.time, fixed_moments[index] / fixed_moments[index, 0], "--", color=plt.cm.tab10(index), lw=0.9)
        marker_index = np.unique(np.round(np.linspace(0, adaptive.time.size - 1, 28)).astype(int))
        ax_moments.semilogy(adaptive.time[marker_index], adaptive_moments[index, marker_index] / adaptive_moments[index, 0], "o", color=plt.cm.tab10(index), markerfacecolor="white", markersize=2.6)
    ax_moments.set_ylabel(r"$M_q/M_q(0)$")
    ax_moments.set_xlabel("Time")
    ax_moments.grid(True, which="major", ls=":", alpha=0.6)
    moment_legend = ax_moments.legend(loc="upper left", ncol=2, fontsize=6.7, title="moment")
    ax_moments.add_artist(moment_legend)
    ax_moments.legend(
        handles=[
            plt.Line2D([], [], color=colors["reference"], lw=1.2, label="Kumar--Ramkrishna analytical"),
            plt.Line2D([], [], color=colors["fixed"], ls="--", lw=1.0, label="Fixed HMMC"),
            plt.Line2D([], [], color=colors["adaptive"], marker="o", markerfacecolor="white", lw=0, markersize=4, label="Adaptive HMMC"),
        ],
        loc="lower right",
        fontsize=6.7,
    )

    ax_grid.plot(
        analytical_times,
        INITIAL_MEAN_VOLUME / analytical_number_ratio(analytical_times),
        "-",
        color=colors["reference"],
        lw=1.2,
        label=r"analytical $M_3/M_0$",
    )
    for label, result, style in (("fixed", fixed, "--"), ("adaptive", adaptive, "-")):
        number_and_volume = moment_history(result, np.array([0, 3]))
        mean_volume = number_and_volume[1] / number_and_volume[0]
        ax_grid.plot(result.time, mean_volume, style, color=colors[label], lw=1.1, label=rf"{label} $M_3/M_0$")
    ax_grid.step(fixed.time, fixed.pivots[-1] ** 3, where="post", color="#7f7f7f", ls="--", lw=0.9, label="fixed $v_N$")
    ax_grid.step(adaptive.time, adaptive.pivots[-1] ** 3, where="post", color="#007358", lw=1.0, label="adaptive $v_N$")
    ax_grid.set_yscale("log")
    ax_grid.set_xlabel("Time")
    ax_grid.set_ylabel(r"Volume, $v=d^3$")
    ax_grid.grid(True, which="major", ls=":", alpha=0.6)
    ax_grid.legend(loc="best", fontsize=6.5)

    volumes = np.geomspace(1.0e-4, 1.0e4, 1000)
    ax_oversize.semilogx(volumes, analytical_oversize_number(volumes), "-", color=colors["reference"], lw=1.3, label="Kumar--Ramkrishna analytical")
    digitized_volume, digitized_oversize = digitized_kumar_moving_pivot_r15()
    ax_oversize.plot(
        digitized_volume,
        digitized_oversize,
        "s",
        color="#6a3d9a",
        markerfacecolor="white",
        markersize=3.5,
        label=r"Kumar--Ramkrishna moving pivot, $r=1.5$ (digitized)",
    )
    ax_oversize.step(volumes, oversize_number(fixed, volumes), where="post", color=colors["fixed"], ls="--", lw=1.0, label="Fixed HMMC")
    ax_oversize.step(volumes, oversize_number(adaptive, volumes), where="post", color=colors["adaptive"], lw=1.1, label="Adaptive HMMC")
    ax_oversize.set_yscale("log")
    ax_oversize.set_ylim(1.0e-15, 1.1)
    ax_oversize.set_xlabel(r"Particle volume, $v$")
    ax_oversize.set_ylabel(r"Number of particles larger than $v$")
    ax_oversize.grid(True, which="major", ls=":", alpha=0.6)
    ax_oversize.legend(loc="best", fontsize=6.7)

    fig.tight_layout(h_pad=0.8)
    FIGS.mkdir(parents=True, exist_ok=True)
    output = FIGS / "kumar_ramkrishna_sum_kernel_reproduction.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    fixed, adaptive = run_cases()
    assert np.all(np.isfinite(fixed.moments))
    assert np.all(np.isfinite(adaptive.moments))
    assert abs(adaptive.moments[0, -1] / adaptive.moments[0, 0] - PUBLISHED_NUMBER_RATIO) < 2.0e-3
    output = plot_results(fixed, adaptive)
    analytical_volumes = np.geomspace(1.0e-4, 1.0e4, 20000)
    analytical_oversize = analytical_oversize_number(analytical_volumes)
    analytical_front = analytical_volumes[np.flatnonzero(analytical_oversize >= 1.0e-10)[-1]]
    print(f"Saved {output}")
    print(
        "Kumar--Ramkrishna sum kernel: "
        f"fixed N/N0={fixed.moments[0, -1] / fixed.moments[0, 0]:.4f}, "
        f"adaptive N/N0={adaptive.moments[0, -1] / adaptive.moments[0, 0]:.4f}, "
        f"analytical N/N0={PUBLISHED_NUMBER_RATIO:.4f}, "
        f"adaptations={adaptive.adaptation_times.size}"
    )
    print(
        "Front volume at CON=1e-10: "
        f"fixed={front_volume(fixed):.3g}, "
        f"adaptive={front_volume(adaptive):.3g}, "
        f"analytical={analytical_front:.3g}"
    )


if __name__ == "__main__":
    main()
