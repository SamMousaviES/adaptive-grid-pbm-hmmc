"""Reproduce manuscript Figure: growth_moment_validation.

This manuscript example uses a size-independent growth law,

    G(d) = G0,

where characteristics translate as ``d(t) = d(0) + G0 t``. The growth step
projects each translated source pivot onto the current grid by local moment
matching. When adaptation is enabled, the grid is still updated by the current
uniform-scaling rule driven by ``d43``.

The analytical moment reference for a translated initial distribution is

    M_q(t) = sum_{r=0}^q binom(q, r) (G0 t)^(q-r) M_r(0).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

from adaptivehmmc._moments import d43_from_moments, moment_vector
from adaptivehmmc._redistribution import (
    local_stencil_start,
    solve_moment_weights,
)
from _paths import output_dir


@dataclass
class GrowthHistory:
    time: np.ndarray
    population: np.ndarray
    pivots: np.ndarray
    moments: np.ndarray
    d43: np.ndarray
    target_upper_pivot: np.ndarray
    adaptation_times: np.ndarray


def geometric_grid(num_classes, diameter_min, diameter_max, grid_ratio):
    weights = grid_ratio ** np.arange(num_classes)
    widths = weights / weights.sum() * (diameter_max - diameter_min)
    edges = diameter_min + np.concatenate(([0.0], np.cumsum(widths)))
    pivots = 0.5 * (edges[:-1] + edges[1:])
    return pivots, widths


def lognormal_on_grid(pivots, widths, center, sigma):
    raw = np.exp(-0.5 * (np.log(pivots / center) / sigma) ** 2) / pivots
    raw *= widths
    return raw / raw.sum()


def analytical_delta_l_moments(initial_moments, orders, displacement):
    values = np.zeros_like(orders, dtype=float)
    for q_index, q in enumerate(orders):
        values[q_index] = sum(
            comb(int(q), r) * displacement ** (int(q) - r) * initial_moments[r]
            for r in range(int(q) + 1)
        )
    return values


def solve_nonnegative_moment_weights(pivots, target_moments, moment_orders):
    """Approximate moment matching with nonnegative redistribution weights."""
    pivots = np.asarray(pivots, dtype=float)
    target_moments = np.asarray(target_moments, dtype=float)
    moment_orders = np.asarray(moment_orders)
    matrix = pivots[None, :] ** moment_orders[:, None]
    target = target_moments[: pivots.size]
    scale = np.maximum(np.abs(target), np.finfo(float).eps)
    weights, _ = nnls(matrix / scale[:, None], target / scale)
    weight_sum = weights.sum()
    if weight_sum > np.finfo(float).eps:
        weights = weights / weight_sum
    else:
        weights[np.argmin(np.abs(pivots - target_moments[1]))] = 1.0
    return weights


def redistribute_population_nonnegative(population0, pivots0, moment_orders, f_max):
    """Uniformly rescale the grid and project with nonnegative weights."""
    population0 = np.asarray(population0, dtype=float)
    pivots0 = np.asarray(pivots0, dtype=float)
    diagnostic_moments = moment_vector(pivots0, population0, 5)
    d43_value = d43_from_moments(diagnostic_moments)
    target_upper = f_max * d43_value

    if not np.isfinite(target_upper) or target_upper <= 0:
        return population0.copy(), pivots0.copy(), False

    pivots = pivots0 * (target_upper / pivots0[-1])
    population = np.zeros_like(population0)
    stencil_size = moment_orders.size

    for source, source_pivot in enumerate(pivots0):
        first = local_stencil_start(pivots, source_pivot, stencil_size)
        stencil = slice(first, first + stencil_size)
        source_moments = source_pivot ** moment_orders
        weights = solve_nonnegative_moment_weights(
            pivots[stencil], source_moments, moment_orders
        )
        population[stencil] += population0[source] * weights

    return np.maximum(population, 0.0), pivots, True


def build_delta_l_projection_matrix(
    pivots, displacement, moment_orders, *, enforce_nonnegative
):
    pivots = np.asarray(pivots, dtype=float)
    n = pivots.size
    stencil_size = moment_orders.size
    matrix = np.zeros((n, n))

    for source in range(n):
        target = pivots[source] + displacement
        first = local_stencil_start(pivots, target, stencil_size)
        stencil = slice(first, first + stencil_size)
        target_moments = target ** moment_orders
        if enforce_nonnegative:
            weights = solve_nonnegative_moment_weights(
                pivots[stencil], target_moments, moment_orders
            )
        else:
            weights = solve_moment_weights(pivots[stencil], target_moments, moment_orders)
        matrix[stencil, source] = weights

    return matrix


def append_state(history, time, population, pivots, num_moments, f_max):
    moments = moment_vector(pivots, population, max(num_moments, 5))
    d43_value = d43_from_moments(moments)
    history["time"].append(float(time))
    history["population"].append(np.asarray(population, dtype=float).copy())
    history["pivots"].append(np.asarray(pivots, dtype=float).copy())
    history["moments"].append(moments[:num_moments].copy())
    history["d43"].append(float(d43_value))
    history["target_upper_pivot"].append(float(f_max * d43_value))


def should_adapt(population, pivots, f_max, f_mult):
    moments = moment_vector(pivots, population, 5)
    d43_value = d43_from_moments(moments)
    target = f_max * d43_value
    if not np.isfinite(target) or target <= 0:
        return False
    return target > f_mult * pivots[-1] or target < pivots[-1] / f_mult


def run_delta_l_growth(
    pivots0,
    population0,
    *,
    final_time,
    num_steps,
    growth_rate,
    num_moments,
    f_max,
    f_mult,
    adaptive,
    enforce_nonnegative,
):
    moment_orders = np.arange(num_moments)
    dt = final_time / num_steps
    displacement_per_step = growth_rate * dt

    history = {
        "time": [],
        "population": [],
        "pivots": [],
        "moments": [],
        "d43": [],
        "target_upper_pivot": [],
    }
    adaptation_times = []

    time = 0.0
    pivots = np.asarray(pivots0, dtype=float).copy()
    population = np.asarray(population0, dtype=float).copy()
    append_state(history, time, population, pivots, num_moments, f_max)

    for _ in range(num_steps):
        # McCabe growth is an additive characteristic shift, so the projection
        # matrix must be rebuilt after pivot rescaling.
        projection = build_delta_l_projection_matrix(
            pivots,
            displacement_per_step,
            moment_orders,
            enforce_nonnegative=enforce_nonnegative,
        )
        population = projection @ population
        if enforce_nonnegative:
            population = np.maximum(population, 0.0)
        time += dt
        append_state(history, time, population, pivots, num_moments, f_max)

        if adaptive and should_adapt(population, pivots, f_max, f_mult):
            if enforce_nonnegative:
                population, pivots, was_adapted = redistribute_population_nonnegative(
                    population, pivots, moment_orders, f_max
                )
            else:
                from adaptivehmmc._redistribution import redistribute_population

                population, pivots, was_adapted = redistribute_population(
                    population, pivots, moment_orders, f_max
                )
            if was_adapted:
                adaptation_times.append(time)
                append_state(history, time, population, pivots, num_moments, f_max)

    return GrowthHistory(
        time=np.asarray(history["time"]),
        population=np.asarray(history["population"]).T,
        pivots=np.asarray(history["pivots"]).T,
        moments=np.asarray(history["moments"]).T,
        d43=np.asarray(history["d43"]),
        target_upper_pivot=np.asarray(history["target_upper_pivot"]),
        adaptation_times=np.asarray(adaptation_times),
    )


def main() -> None:
    num_classes = 61
    num_moments = 6
    grid_ratio = 1
    diameter_min = 0.0
    diameter_max = 3.0
    final_time = 10.0
    num_steps = 100
    growth_rate = 0.5
    f_max = 2
    f_mult = 1.1
    lognorm_center = 0.45
    lognorm_sigma = 0.18

    pivots, widths = geometric_grid(
        num_classes, diameter_min, diameter_max, grid_ratio
    )
    initial_population = lognormal_on_grid(
        pivots, widths, lognorm_center, lognorm_sigma
    )
    initial_moments = moment_vector(pivots, initial_population, num_moments)

    print("Running fixed-grid McCabe delta-L growth ...")
    fixed = run_delta_l_growth(
        pivots,
        initial_population,
        final_time=final_time,
        num_steps=num_steps,
        growth_rate=growth_rate,
        num_moments=num_moments,
        f_max=f_max,
        f_mult=f_mult,
        adaptive=False,
        enforce_nonnegative=True,
    )

    print("Running current uniform-scaling adaptive McCabe delta-L growth ...")
    adaptive = run_delta_l_growth(
        pivots,
        initial_population,
        final_time=final_time,
        num_steps=num_steps,
        growth_rate=growth_rate,
        num_moments=num_moments,
        f_max=f_max,
        f_mult=f_mult,
        adaptive=True,
        enforce_nonnegative=True,
    )

    orders = np.arange(num_moments)
    final_reference = analytical_delta_l_moments(
        initial_moments, orders, growth_rate * final_time
    )
    fixed_err = 100.0 * np.max(
        np.abs((fixed.moments[:, -1] - final_reference) / final_reference)
    )
    adaptive_err = 100.0 * np.max(
        np.abs((adaptive.moments[:, -1] - final_reference) / final_reference)
    )
    print(f"Fixed-grid max moment error:    {fixed_err:.3e} %")
    print(f"Adaptive-grid max moment error: {adaptive_err:.3e} %")
    print(f"Adaptation events: {adaptive.adaptation_times.size}")

    time_reference = np.linspace(0.0, final_time, 240)
    reference = np.column_stack(
        [
            analytical_delta_l_moments(initial_moments, orders, growth_rate * time)
            for time in time_reference
        ]
    )
    reference_d43 = reference[4, :] / reference[3, :]

    fig, (ax_mom, ax_grid) = plt.subplots(2, 1, figsize=(5.0, 5.8))
    colors = plt.get_cmap("tab10")
    marker_idx = np.unique(
        np.round(np.linspace(0, adaptive.time.size - 1, 28)).astype(int)
    )

    for i in range(num_moments):
        ax_mom.semilogy(
            time_reference,
            reference[i, :] / initial_moments[i],
            "-",
            color=colors(i),
            lw=1.2,
            label=f"$M_{i}$",
        )
        ax_mom.semilogy(
            fixed.time,
            fixed.moments[i, :] / initial_moments[i],
            "--",
            color=colors(i),
            lw=0.9,
        )
        ax_mom.semilogy(
            adaptive.time[marker_idx],
            adaptive.moments[i, marker_idx] / initial_moments[i],
            "o",
            color=colors(i),
            markerfacecolor="white",
            markersize=2.8,
            lw=0.7,
        )

    ax_mom.set_xlabel("Time (s)")
    ax_mom.set_ylabel(r"$M_q / M_q(0)$")
    ax_mom.grid(True, which="major", ls=":", alpha=0.6)
    ax_mom.legend(loc="upper left", ncol=3, fontsize=7)

    ax_grid.plot(
        time_reference,
        reference_d43,
        "-",
        color="black",
        lw=1.2,
        label=r"analytical $d_{43}$",
    )
    ax_grid.plot(
        fixed.time,
        fixed.d43,
        "--",
        color="#8c1f1f",
        lw=1.0,
        label=r"fixed $d_{43}$",
    )
    ax_grid.plot(
        adaptive.time,
        adaptive.d43,
        "-",
        color="#1a619c",
        lw=1.1,
        label=r"adaptive $d_{43}$",
    )
    ax_grid.step(
        adaptive.time,
        adaptive.pivots[-1, :],
        where="post",
        color="#007358",
        lw=1.1,
        label=r"adaptive $d_N$",
    )
    ax_grid.axhline(
        fixed.pivots[-1, 0],
        ls="-.",
        color="0.55",
        lw=0.9,
        label=r"fixed $d_N$",
    )
    for t_event in adaptive.adaptation_times:
        ax_grid.axvline(t_event, ls=":", color="0.70", lw=0.6)
    ax_grid.set_xlabel("Time (s)")
    ax_grid.set_ylabel(r"Diameter, $d$ (mm)")
    ax_grid.grid(True, which="major", ls=":", alpha=0.6)
    ax_grid.legend(loc="upper left", fontsize=7)

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "growth_moment_validation.pdf", bbox_inches="tight")
    fig.savefig(
        out / "growth_moment_validation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved growth_moment_validation to {out}")


if __name__ == "__main__":
    main()
