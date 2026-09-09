"""Generate a focused PSD comparison for testing the reviewer-response idea.

This script is a duplicate/sandbox version of the PSD reconstruction diagnostic.
It keeps the manuscript untouched and produces a clearer comparison based on the
volume-weighted PSD, f_V(d). The plotted cases are selected to test
where adaptive rescaling should be advantageous over a fixed grid:

* the existing constant-kernel coalescence benchmark,
* a binary breakage benchmark, and
* proportional growth, where uniform grid rescaling matches the characteristic
  motion of the distribution.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import exp, log, sqrt
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
from scipy.special import gammaln, ndtr, ndtri

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
FIGS = REPO / "figures_out"
TABLES = FIGS / "tables"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples"))

import adaptivehmmc  # noqa: E402
from figure_growth_benchmark import analytical_delta_l_moments, run_delta_l_growth  # noqa: E402


@dataclass
class PsdCase:
    name: str
    fixed: object
    adaptive: object
    reference_density: Callable[[np.ndarray], np.ndarray]
    reference_quantile: Callable[[float], float]
    reference_moments: Callable[[np.ndarray, np.ndarray], np.ndarray]
    figure_basename: str
    grid_ylog: bool
    moment_clip_negative: bool
    log_x: bool


def geometric_grid(num_classes, diameter_min, diameter_max, grid_ratio):
    weights = grid_ratio ** np.arange(num_classes)
    widths = weights / weights.sum() * (diameter_max - diameter_min)
    edges = diameter_min + np.concatenate(([0.0], np.cumsum(widths)))
    pivots = 0.5 * (edges[:-1] + edges[1:])
    return pivots, widths


def lognormal_on_grid(pivots, widths, center, sigma, *, normalize_to):
    raw = np.exp(-0.5 * (np.log(pivots / center) / sigma) ** 2) / pivots
    raw *= widths
    if normalize_to == "number":
        return raw / raw.sum()
    if normalize_to == "third_moment":
        third_moment = np.sum(raw * pivots**3)
        return raw / third_moment
    raise ValueError(f"Unknown normalization: {normalize_to!r}")


def lognormal_pdf(diameter, center, sigma):
    diameter = np.asarray(diameter, dtype=float)
    density = np.zeros_like(diameter)
    mask = diameter > 0.0
    z = np.log(diameter[mask] / center) / sigma
    density[mask] = (
        np.exp(-0.5 * z**2)
        / (diameter[mask] * sigma * sqrt(2.0 * np.pi))
    )
    return density


def lognormal_cdf(diameter, center, sigma):
    diameter = np.asarray(diameter, dtype=float)
    values = np.zeros_like(diameter)
    mask = diameter > 0.0
    values[mask] = ndtr(np.log(diameter[mask] / center) / sigma)
    values[diameter == np.inf] = 1.0
    return values


def cell_edges_from_pivots(pivots):
    pivots = np.asarray(pivots, dtype=float)
    if pivots.size < 2:
        raise ValueError("At least two pivots are required.")

    edges = np.empty(pivots.size + 1)
    edges[1:-1] = 0.5 * (pivots[:-1] + pivots[1:])
    edges[0] = pivots[0] - 0.5 * (pivots[1] - pivots[0])
    edges[-1] = pivots[-1] + 0.5 * (pivots[-1] - pivots[-2])
    edges[0] = max(edges[0], np.finfo(float).tiny)
    return edges


def result_volume_psd_at(result, points, *, moment_power):
    pivots = np.asarray(result.pivots[:, -1], dtype=float)
    population = np.maximum(np.asarray(result.population[:, -1], dtype=float), 0.0)
    order = np.argsort(pivots)
    pivots = pivots[order]
    edges = cell_edges_from_pivots(pivots)
    widths = np.diff(edges)
    weights = population[order] * pivots**moment_power
    total = weights.sum()
    if total <= np.finfo(float).eps:
        return np.zeros_like(points, dtype=float)

    density = weights / (total * widths)
    values = np.zeros_like(points, dtype=float)
    indices = np.searchsorted(edges, points, side="right") - 1
    valid = (indices >= 0) & (indices < density.size)
    values[valid] = density[indices[valid]]
    return values


def result_volume_psd_at_pivots(result, *, moment_power):
    pivots = np.asarray(result.pivots[:, -1], dtype=float)
    population = np.maximum(np.asarray(result.population[:, -1], dtype=float), 0.0)
    order = np.argsort(pivots)
    pivots = pivots[order]
    edges = cell_edges_from_pivots(pivots)
    weights = population[order] * pivots**moment_power
    total = weights.sum()
    if total <= np.finfo(float).eps:
        return pivots, np.zeros_like(pivots, dtype=float)
    return pivots, weights / (total * np.diff(edges))


def reference_volume_psd(case, points):
    points = np.asarray(points, dtype=float)
    density = case.reference_density(points) * points**3
    total = np.trapz(density, points)
    if total <= np.finfo(float).eps:
        return np.zeros_like(points)
    return density / total


def comparison_points(case):
    lower = float(case.reference_quantile(1.0e-6))
    upper = float(case.reference_quantile(1.0 - 1.0e-6))
    upper = max(
        upper,
        1.05 * float(np.max(case.fixed.pivots[:, -1])),
        1.05 * float(np.max(case.adaptive.pivots[:, -1])),
    )
    if case.log_x:
        lower = max(lower, upper * 1.0e-8, 1.0e-14)
        return np.geomspace(lower, upper, 2000)
    return np.linspace(0.0, upper, 2000)


def psd_errors(numerical, reference, points):
    absolute = np.abs(numerical - reference)
    max_abs = float(np.max(absolute))
    l1_abs = float(np.trapz(absolute, points))
    return max_abs, l1_abs


def moment_history(result, num_moments, *, clip_negative):
    orders = np.arange(num_moments)
    population = np.asarray(result.population, dtype=float)
    if clip_negative:
        population = np.maximum(population, 0.0)
    moments = np.zeros((num_moments, result.time.size))
    for i in range(result.time.size):
        pivots = result.pivots[:, i]
        moments[:, i] = (
            population[:, i, None] * pivots[:, None] ** orders[None, :]
        ).sum(axis=0)
    return moments


def d43_from_history(moments):
    with np.errstate(divide="ignore", invalid="ignore"):
        return moments[4, :] / moments[3, :]


def positive_for_log(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 0.0, values, np.nan)


def max_relative_error(values, reference):
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    scale = np.maximum(np.abs(reference), np.finfo(float).eps)
    return float(np.max(np.abs((values - reference) / scale)))


def coalescence_case():
    num_classes = 21
    num_moments = 6
    grid_ratio = 1.0
    diameter_min = 1e-3
    diameter_max = 1.0
    simulation_time = 10.0
    rate = 150.0
    f_max = 3.0
    f_mult = 1.3
    lognorm_center = 0.31
    lognorm_sigma = 0.1

    base = adaptivehmmc.default_options()
    pivots, widths = geometric_grid(
        num_classes,
        diameter_min,
        diameter_max,
        grid_ratio,
    )
    initial_population = lognormal_on_grid(
        pivots,
        widths,
        lognorm_center,
        lognorm_sigma,
        normalize_to="number",
    )
    total_volume = (
        base.initial_distribution.total_number
        * base.initial_distribution.volume_scale
    )
    volume_scale = np.sum(initial_population * pivots**3) / np.sum(initial_population)
    total_number = total_volume / volume_scale

    base.time_span = (0.0, simulation_time)
    base.num_classes = num_classes
    base.num_moments = num_moments
    base.grid_ratio = grid_ratio
    base.diameter_min = diameter_min
    base.diameter_max = diameter_max
    base.f_max = f_max
    base.process.type = "coalescence"
    base.process.kernel = "constant"
    base.process.rate = rate
    base.initial_distribution.total_number = total_number
    base.initial_distribution.volume_scale = volume_scale

    fixed_opts = base.copy()
    fixed_opts.adaptive = False
    fixed = adaptivehmmc.solve(fixed_opts)

    adaptive_opts = base.copy()
    adaptive_opts.adaptive = True
    adaptive_opts.f_mult = f_mult
    adaptive = adaptivehmmc.solve(adaptive_opts)

    total_number = base.initial_distribution.total_number
    volume_scale = base.initial_distribution.volume_scale
    rate = base.process.rate
    final_time = base.time_span[-1]
    factor = 2.0 / (2.0 + rate * total_number * final_time)

    def reference_density(diameter):
        diameter = np.asarray(diameter, dtype=float)
        return (
            3.0
            * total_number
            * factor**2
            / volume_scale
            * diameter**2
            * np.exp(-factor * diameter**3 / volume_scale)
        )

    def reference_quantile(probability):
        return (-volume_scale / factor * log(1.0 - probability)) ** (1.0 / 3.0)

    def reference_moments(times, initial_moments):
        times = np.asarray(times, dtype=float)
        initial_moments = np.asarray(initial_moments, dtype=float)
        orders = np.arange(initial_moments.size)
        factors = 2.0 / (2.0 + rate * initial_moments[0] * times)
        return initial_moments[:, None] * factors[None, :] ** (
            1.0 - orders[:, None] / 3.0
        )

    return PsdCase(
        name="Coalescence",
        fixed=fixed,
        adaptive=adaptive,
        reference_density=reference_density,
        reference_quantile=reference_quantile,
        reference_moments=reference_moments,
        figure_basename="psd_superiority_coalescence_moment_validation",
        grid_ylog=False,
        moment_clip_negative=True,
        log_x=False,
    )


def breakage_case():
    num_classes = 21
    num_moments = 6
    grid_ratio = 1.3
    diameter_min = 1e-3
    diameter_max = 1.0
    simulation_time = 10.0
    rate = 2
    f_max = 1.1
    f_mult = 1.5
    lognorm_center = 0.51
    lognorm_sigma = 0.1

    pivots, widths = geometric_grid(
        num_classes,
        diameter_min,
        diameter_max,
        grid_ratio,
    )
    initial_population = lognormal_on_grid(
        pivots,
        widths,
        lognorm_center,
        lognorm_sigma,
        normalize_to="third_moment",
    )

    base = adaptivehmmc.default_options()
    base.time_span = (0.0, simulation_time)
    base.num_classes = num_classes
    base.num_moments = num_moments
    base.diameter_min = diameter_min
    base.diameter_max = diameter_max
    base.grid_ratio = grid_ratio
    base.f_max = f_max
    base.f_mult = f_mult
    base.process.type = "breakage"
    base.process.kernel = "binary_equal_volume"
    base.process.rate = rate
    base.initial_distribution.type = "custom"
    base.initial_distribution.pivots = pivots
    base.initial_distribution.population = initial_population
    base.solver.rel_tol = 1.0e-8
    base.solver.abs_tol = 1.0e-10

    fixed_opts = base.copy()
    fixed_opts.adaptive = False
    fixed = adaptivehmmc.solve(fixed_opts)

    adaptive_opts = base.copy()
    adaptive_opts.adaptive = True
    adaptive = adaptivehmmc.solve(adaptive_opts)

    tau = rate * simulation_time
    alpha = 2.0 ** (-1.0 / 3.0)
    lambda_generation = 2.0 * tau
    k_max = int(np.ceil(lambda_generation + 12.0 * sqrt(lambda_generation + 1.0)))
    generations = np.arange(k_max + 1)
    log_pmf = (
        -lambda_generation
        + generations * log(lambda_generation)
        - gammaln(generations + 1.0)
    )
    pmf = np.exp(log_pmf)
    pmf = pmf / pmf.sum()
    generation_scale = alpha**generations

    initial_number = 1.0 / exp(
        3.0 * log(lognorm_center) + 0.5 * (3.0 * lognorm_sigma) ** 2
    )
    final_number = initial_number * exp(tau)

    def normalized_number_cdf(diameter):
        diameter = np.asarray(diameter, dtype=float)
        terms = [
            weight
            * lognormal_cdf(diameter / scale, lognorm_center, lognorm_sigma)
            for weight, scale in zip(pmf, generation_scale)
        ]
        return np.sum(np.stack(terms), axis=0)

    def reference_density(diameter):
        diameter = np.asarray(diameter, dtype=float)
        terms = [
            weight
            * lognormal_pdf(diameter / scale, lognorm_center, lognorm_sigma)
            / scale
            for weight, scale in zip(pmf, generation_scale)
        ]
        return final_number * np.sum(np.stack(terms), axis=0)

    def reference_quantile(probability):
        lower = 1.0e-14
        upper = lognorm_center * 10.0
        while normalized_number_cdf(np.asarray([upper]))[0] < probability:
            upper *= 10.0
        for _ in range(90):
            middle = sqrt(lower * upper)
            if normalized_number_cdf(np.asarray([middle]))[0] < probability:
                lower = middle
            else:
                upper = middle
        return upper

    def reference_moments(times, initial_moments):
        times = np.asarray(times, dtype=float)
        initial_moments = np.asarray(initial_moments, dtype=float)
        orders = np.arange(initial_moments.size)
        tau_values = rate * times
        exponents = (2.0 ** (1.0 - orders[:, None] / 3.0) - 1.0) * tau_values
        return initial_moments[:, None] * np.exp(exponents)

    return PsdCase(
        name="Breakage",
        fixed=fixed,
        adaptive=adaptive,
        reference_density=reference_density,
        reference_quantile=reference_quantile,
        reference_moments=reference_moments,
        figure_basename="psd_superiority_breakage_moment_validation",
        grid_ylog=True,
        moment_clip_negative=False,
        log_x=True,
    )


def growth_case():
    num_classes = 21
    num_moments = 6
    grid_ratio = 1.1
    diameter_min = 1e-3
    diameter_max = 1.0
    simulation_time = 10.0
    num_steps = 10
    rate = 0.1
    f_max = 3.3
    f_mult = 1.1
    lognorm_center = 0.51
    lognorm_sigma = 0.2

    pivots, widths = geometric_grid(
        num_classes,
        diameter_min,
        diameter_max,
        grid_ratio,
    )
    initial_population = lognormal_on_grid(
        pivots,
        widths,
        lognorm_center,
        lognorm_sigma,
        normalize_to="number",
    )

    fixed = run_delta_l_growth(
        pivots,
        initial_population,
        final_time=simulation_time,
        num_steps=num_steps,
        growth_rate=rate,
        num_moments=num_moments,
        f_max=f_max,
        f_mult=f_mult,
        adaptive=False,
        enforce_nonnegative=True,
    )

    adaptive = run_delta_l_growth(
        pivots,
        initial_population,
        final_time=simulation_time,
        num_steps=num_steps,
        growth_rate=rate,
        num_moments=num_moments,
        f_max=f_max,
        f_mult=f_mult,
        adaptive=True,
        enforce_nonnegative=True,
    )

    displacement = rate * simulation_time

    def reference_density(diameter):
        shifted = np.asarray(diameter, dtype=float) - displacement
        return lognormal_pdf(shifted, lognorm_center, lognorm_sigma)

    def reference_quantile(probability):
        return displacement + lognorm_center * exp(lognorm_sigma * ndtri(probability))

    def reference_moments(times, initial_moments):
        times = np.asarray(times, dtype=float)
        initial_moments = np.asarray(initial_moments, dtype=float)
        orders = np.arange(initial_moments.size)
        return np.column_stack(
            [
                analytical_delta_l_moments(
                    initial_moments,
                    orders,
                    rate * time,
                )
                for time in times
            ]
        )

    return PsdCase(
        name="Growth",
        fixed=fixed,
        adaptive=adaptive,
        reference_density=reference_density,
        reference_quantile=reference_quantile,
        reference_moments=reference_moments,
        figure_basename="psd_superiority_growth_moment_validation",
        grid_ylog=False,
        moment_clip_negative=False,
        log_x=False,
    )


def build_rows_and_curves(cases):
    rows = []
    curves = []
    for case in cases:
        points = comparison_points(case)
        reference = reference_volume_psd(case, points)
        fixed = result_volume_psd_at(case.fixed, points, moment_power=3)
        adaptive = result_volume_psd_at(case.adaptive, points, moment_power=3)
        fixed_max, fixed_l1 = psd_errors(fixed, reference, points)
        adaptive_max, adaptive_l1 = psd_errors(
            adaptive,
            reference,
            points,
        )
        rows.append(
            {
                "case": case.name,
                "fixed_max": fixed_max,
                "fixed_l1": fixed_l1,
                "adaptive_max": adaptive_max,
                "adaptive_l1": adaptive_l1,
                "ratio": fixed_max / adaptive_max,
                "adaptations": int(case.adaptive.adaptation_times.size),
            }
        )
        curves.append(
            {
                "case": case,
                "points": points,
                "reference": reference,
                "fixed": fixed,
                "adaptive": adaptive,
            }
        )
    return rows, curves


def plot_three_panel_case_validation(curve):
    case = curve["case"]
    num_moments = case.fixed.moments.shape[0]
    initial_moments = np.asarray(case.fixed.moments[:, 0], dtype=float)
    time_reference = np.linspace(
        float(case.fixed.time[0]),
        float(case.fixed.time[-1]),
        240,
    )
    reference_moments = case.reference_moments(time_reference, initial_moments)
    fixed_moments = moment_history(
        case.fixed,
        num_moments,
        clip_negative=case.moment_clip_negative,
    )
    adaptive_moments = moment_history(
        case.adaptive,
        num_moments,
        clip_negative=case.moment_clip_negative,
    )
    reference_d43 = d43_from_history(reference_moments)
    fixed_d43 = d43_from_history(fixed_moments)
    adaptive_d43 = d43_from_history(adaptive_moments)

    points = curve["points"]
    reference_psd = curve["reference"]
    fixed_psd = curve["fixed"]
    adaptive_psd = curve["adaptive"]
    fixed_pivots, fixed_pivot_psd = result_volume_psd_at_pivots(
        case.fixed,
        moment_power=3,
    )
    adaptive_pivots, adaptive_pivot_psd = result_volume_psd_at_pivots(
        case.adaptive,
        moment_power=3,
    )

    basename = case.figure_basename.replace(
        "_moment_validation",
        "_three_panel_psd_validation",
    )
    fig, (ax_mom, ax_grid, ax_psd) = plt.subplots(3, 1, figsize=(5.0, 7.7))
    moment_colors = plt.get_cmap("tab10")
    marker_idx = np.unique(
        np.round(np.linspace(0, case.adaptive.time.size - 1, 28)).astype(int)
    )

    positive_values = []
    for i in range(num_moments):
        reference_ratio = reference_moments[i, :] / initial_moments[i]
        fixed_ratio = fixed_moments[i, :] / initial_moments[i]
        adaptive_ratio = adaptive_moments[i, marker_idx] / initial_moments[i]
        positive_values.extend([reference_ratio, fixed_ratio, adaptive_ratio])

        ax_mom.semilogy(
            time_reference,
            positive_for_log(reference_ratio),
            "-",
            color=moment_colors(i),
            lw=1.2,
            label=f"$M_{i}$",
        )
        ax_mom.semilogy(
            case.fixed.time,
            positive_for_log(fixed_ratio),
            "--",
            color=moment_colors(i),
            lw=0.9,
        )
        ax_mom.semilogy(
            case.adaptive.time[marker_idx],
            positive_for_log(adaptive_ratio),
            "o",
            color=moment_colors(i),
            markerfacecolor="white",
            markersize=2.6,
            lw=0.7,
        )

    positive = np.concatenate(
        [values[np.isfinite(values) & (values > 0.0)] for values in positive_values]
    )
    if positive.size:
        ax_mom.set_ylim(
            bottom=max(1.0e-8, float(np.min(positive)) * 0.55),
            top=float(np.max(positive)) * 1.45,
        )
    ax_mom.set_title(case.name)
    ax_mom.set_xlabel("Time (s)")
    ax_mom.set_ylabel(r"$M_q/M_q(0)$")
    ax_mom.grid(True, which="major", ls=":", alpha=0.6)
    leg1 = ax_mom.legend(loc="upper left", ncol=3, fontsize=6.7, title="moment")
    style_handles = [
        plt.Line2D([], [], color="k", ls="-", lw=1.2, label="Analytical"),
        plt.Line2D([], [], color="k", ls="--", lw=0.9, label="Fixed grid"),
        plt.Line2D(
            [],
            [],
            color="k",
            marker="o",
            ls="None",
            markerfacecolor="white",
            markersize=4,
            label="Adaptive grid",
        ),
    ]
    ax_mom.add_artist(leg1)
    style_loc = "lower right" if case.name == "Coalescence" else "lower left"
    ax_mom.legend(handles=style_handles, loc=style_loc, fontsize=6.7)

    ax_grid.plot(
        time_reference,
        reference_d43,
        "-",
        color="black",
        lw=1.2,
        label=r"analytical $d_{43}$",
    )
    ax_grid.plot(
        case.fixed.time,
        fixed_d43,
        "--",
        color="#8c1f1f",
        lw=1.0,
        label=r"fixed $d_{43}$",
    )
    ax_grid.plot(
        case.adaptive.time,
        adaptive_d43,
        "-",
        color="#1a619c",
        lw=1.1,
        label=r"adaptive $d_{43}$",
    )
    ax_grid.step(
        case.adaptive.time,
        case.adaptive.pivots[-1, :],
        where="post",
        color="#007358",
        lw=1.1,
        label=r"adaptive $d_N$",
    )
    ax_grid.step(
        case.adaptive.time,
        case.adaptive.pivots[0, :],
        where="post",
        color="#007358",
        lw=0.9,
        ls=":",
        label=r"adaptive $d_1$",
    )
    ax_grid.axhline(
        case.fixed.pivots[-1, 0],
        ls="-.",
        color="0.55",
        lw=0.9,
        label=r"fixed $d_N$",
    )
    ax_grid.axhline(
        case.fixed.pivots[0, 0],
        ls=":",
        color="0.55",
        lw=0.9,
        label=r"fixed $d_1$",
    )
    if case.grid_ylog:
        ax_grid.set_yscale("log")
    ax_grid.set_xlabel("Time (s)")
    ax_grid.set_ylabel(r"Diameter, $d$ (mm)")
    ax_grid.grid(True, which="major", ls=":", alpha=0.6)
    ax_grid.legend(loc="best", fontsize=6.7)

    ax_psd.plot(
        points,
        positive_for_log(reference_psd),
        "-",
        color="black",
        lw=1.3,
        label="Analytical",
    )
    ax_psd.step(
        points,
        positive_for_log(fixed_psd),
        where="post",
        color="#8c1f1f",
        lw=1.0,
        ls="--",
        label="Fixed grid",
    )
    ax_psd.plot(
        fixed_pivots,
        positive_for_log(fixed_pivot_psd),
        "o",
        color="#8c1f1f",
        markerfacecolor="white",
        markersize=2.6,
        lw=0.7,
        label="_nolegend_",
    )
    ax_psd.step(
        points,
        positive_for_log(adaptive_psd),
        where="post",
        color="#1a619c",
        lw=1.1,
        label="Adaptive grid",
    )
    ax_psd.plot(
        adaptive_pivots,
        positive_for_log(adaptive_pivot_psd),
        "o",
        color="#1a619c",
        markerfacecolor="white",
        markersize=2.6,
        lw=0.7,
        label="_nolegend_",
    )
    if case.log_x:
        ax_psd.set_xscale("log")
    positive_psd = np.concatenate(
        [
            values[np.isfinite(values) & (values > 0.0)]
            for values in (
                reference_psd,
                fixed_psd,
                adaptive_psd,
                fixed_pivot_psd,
                adaptive_pivot_psd,
            )
        ]
    )
    ax_psd.set_yscale("log")
    if positive_psd.size:
        top = float(np.max(positive_psd)) * 2.2
        ax_psd.set_ylim(
            bottom=max(float(np.min(positive_psd)) * 0.45, top * 1.0e-7),
            top=top,
        )
    ax_psd.set_xlim(points[0], points[-1])
    ax_psd.set_xlabel(r"Diameter, $d$ (mm)")
    ax_psd.set_ylabel(r"Volume PSD, $f_V(d)$ (mm$^{-1}$)")
    ax_psd.grid(True, which="major", ls=":", alpha=0.6)
    ax_psd.legend(loc="best", fontsize=6.7)

    fig.tight_layout(h_pad=0.75)
    fig.savefig(FIGS / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return basename


def plot_moment_validation(case):
    num_moments = case.fixed.moments.shape[0]
    initial_moments = np.asarray(case.fixed.moments[:, 0], dtype=float)
    time_reference = np.linspace(
        float(case.fixed.time[0]),
        float(case.fixed.time[-1]),
        240,
    )
    reference = case.reference_moments(time_reference, initial_moments)
    fixed_moments = moment_history(
        case.fixed,
        num_moments,
        clip_negative=case.moment_clip_negative,
    )
    adaptive_moments = moment_history(
        case.adaptive,
        num_moments,
        clip_negative=case.moment_clip_negative,
    )

    reference_d43 = d43_from_history(reference)
    fixed_d43 = d43_from_history(fixed_moments)
    adaptive_d43 = d43_from_history(adaptive_moments)

    final_reference = reference[:, -1]
    fixed_error = max_relative_error(fixed_moments[:, -1], final_reference)
    adaptive_error = max_relative_error(adaptive_moments[:, -1], final_reference)

    fig, (ax_mom, ax_grid) = plt.subplots(2, 1, figsize=(5.0, 5.8))
    colors = plt.get_cmap("tab10")
    marker_idx = np.unique(
        np.round(np.linspace(0, case.adaptive.time.size - 1, 30)).astype(int)
    )

    all_ratios = []
    for i in range(num_moments):
        reference_ratio = reference[i, :] / initial_moments[i]
        fixed_ratio = fixed_moments[i, :] / initial_moments[i]
        adaptive_ratio = adaptive_moments[i, marker_idx] / initial_moments[i]
        all_ratios.extend([reference_ratio, fixed_ratio, adaptive_ratio])

        ax_mom.semilogy(
            time_reference,
            positive_for_log(reference_ratio),
            "-",
            color=colors(i),
            lw=1.2,
            label=f"$M_{i}$",
        )
        ax_mom.semilogy(
            case.fixed.time,
            positive_for_log(fixed_ratio),
            "--",
            color=colors(i),
            lw=0.9,
        )
        ax_mom.semilogy(
            case.adaptive.time[marker_idx],
            positive_for_log(adaptive_ratio),
            "o",
            color=colors(i),
            markerfacecolor="white",
            markersize=2.8,
            lw=0.7,
        )

    positive = np.concatenate(
        [values[np.isfinite(values) & (values > 0.0)] for values in all_ratios]
    )
    if positive.size:
        ax_mom.set_ylim(
            bottom=max(1.0e-8, float(np.min(positive)) * 0.55),
            top=float(np.max(positive)) * 1.45,
        )
    ax_mom.set_xlabel("Time (s)")
    ax_mom.set_ylabel(r"$M_q/M_q(0)$")
    ax_mom.grid(True, which="major", ls=":", alpha=0.6)
    leg1 = ax_mom.legend(loc="upper left", ncol=3, fontsize=7, title="moment")
    style_handles = [
        plt.Line2D([], [], color="k", ls="-", lw=1.2, label="Analytical"),
        plt.Line2D([], [], color="k", ls="--", lw=0.9, label="Fixed grid"),
        plt.Line2D(
            [],
            [],
            color="k",
            marker="o",
            ls="None",
            markerfacecolor="white",
            markersize=4,
            label="Adaptive grid",
        ),
    ]
    ax_mom.add_artist(leg1)
    style_loc = "lower right" if case.name == "Coalescence" else "lower left"
    ax_mom.legend(handles=style_handles, loc=style_loc, fontsize=7)

    ax_grid.plot(
        time_reference,
        reference_d43,
        "-",
        color="black",
        lw=1.2,
        label=r"analytical $d_{43}$",
    )
    ax_grid.plot(
        case.fixed.time,
        fixed_d43,
        "--",
        color="#8c1f1f",
        lw=1.0,
        label=r"fixed $d_{43}$",
    )
    ax_grid.plot(
        case.adaptive.time,
        adaptive_d43,
        "-",
        color="#1a619c",
        lw=1.1,
        label=r"adaptive $d_{43}$",
    )
    ax_grid.step(
        case.adaptive.time,
        case.adaptive.pivots[-1, :],
        where="post",
        color="#007358",
        lw=1.1,
        label=r"adaptive $d_N$",
    )
    ax_grid.axhline(
        case.fixed.pivots[-1, 0],
        ls="-.",
        color="0.55",
        lw=0.9,
        label=r"fixed $d_N$",
    )
    for t_event in case.adaptive.adaptation_times:
        ax_grid.axvline(t_event, ls=":", color="0.70", lw=0.6)
    if case.grid_ylog:
        ax_grid.set_yscale("log")
    ax_grid.set_xlabel("Time (s)")
    ax_grid.set_ylabel(r"Diameter, $d$ (mm)")
    ax_grid.grid(True, which="major", ls=":", alpha=0.6)
    grid_loc = "upper right" if case.grid_ylog else "upper left"
    ax_grid.legend(loc=grid_loc, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGS / f"{case.figure_basename}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{case.figure_basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "case": case.name,
        "fixed_error": fixed_error,
        "adaptive_error": adaptive_error,
        "adaptations": int(case.adaptive.adaptation_times.size),
        "figure": case.figure_basename,
    }


def write_table(rows):
    body = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Case & Fixed $L_1(f_V)$ [\%] & Adaptive $L_1(f_V)$ [\%] & Fixed/adaptive & Adaptations \\",
        r"\midrule",
    ]
    for row in rows:
        ratio = row["fixed_l1"] / row["adaptive_l1"]
        body.append(
            f"{row['case']} & "
            f"{100.0 * row['fixed_l1']:.2f} & "
            f"{100.0 * row['adaptive_l1']:.2f} & "
            f"{ratio:.2f} & "
            f"{row['adaptations']} \\\\"
        )
    body.extend([r"\bottomrule", r"\end{tabular}"])
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "psd_superiority_psd_errors.tex").write_text(
        "\n".join(body),
        encoding="utf-8",
    )


def write_moment_table(rows):
    body = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Case & Fixed max moment error [\%] & Adaptive max moment error [\%] & Adaptations \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(
            f"{row['case']} & "
            f"{100.0 * row['fixed_error']:.3g} & "
            f"{100.0 * row['adaptive_error']:.3g} & "
            f"{row['adaptations']} \\\\"
        )
    body.extend([r"\bottomrule", r"\end{tabular}"])
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "psd_superiority_moment_errors.tex").write_text(
        "\n".join(body),
        encoding="utf-8",
    )


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    cases = [coalescence_case(), breakage_case(), growth_case()]
    rows, curves = build_rows_and_curves(cases)
    three_panel_figures = [
        plot_three_panel_case_validation(curve)
        for curve in curves
    ]
    write_table(rows)

    print("Volume-PSD superiority errors:")
    for row in rows:
        print(
            f"{row['case']:21s} "
            f"fixed_L1={100.0 * row['fixed_l1']:7.3f}% "
            f"adaptive_L1={100.0 * row['adaptive_l1']:7.3f}% "
            f"ratio={row['fixed_l1'] / row['adaptive_l1']:5.2f} "
            f"adaptations={row['adaptations']}"
        )
    for basename in three_panel_figures:
        print(f"Saved three-panel figure to {FIGS / f'{basename}.pdf'}")
    print(f"Saved table to {TABLES / 'psd_superiority_psd_errors.tex'}")


if __name__ == "__main__":
    main()
