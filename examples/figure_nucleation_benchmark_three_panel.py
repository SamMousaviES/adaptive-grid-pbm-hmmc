"""Standalone seeded nucleation and nucleation + constant-growth verification.

PBE: dn/dt + G dn/dd = J h(d), h a unit-area triangular density on [0, 2a].
Initial condition: n(d,0) = N0 h(d). Constant J supplies new particles at their
physical birth sizes even after grid adaptation. No aggregation or breakage.

Literature basis (local PDFs in Misc/Literature/Adaptive Grid):
* Hounslow, Ryall & Marshall (1988), AIChE J. 34, 1821-1832, Eqs. (3),
  (10), (45)-(46): nucleation source and constant-size growth verification.
* Qamar et al. (2007), Computers & Chemical Engineering 31, 1296-1311,
  Eqs. (2.27)-(2.29): source/boundary treatment of nucleation and growth.
* Sewerin & Rigopoulos (2017), Chemical Engineering Science 168, 250-270,
  Section 6.3, preceding Eq. (65): triangular nucleation source on [0,2a].

The hat source shape is borrowed, but the seeded initial condition and constant
rates below are chosen verification inputs, NOT literature-fitted kinetics.
Units: d in mm, time in s, N0 in particles per arbitrary suspension-volume unit,
J in particles per that volume unit per s. No supersaturation balance is solved.

Reuses the existing nonnegative moment-fit growth/redistribution routines. These
preserve particle number but approximate higher moments; this is NOT a claim
of exact six-moment preservation or positivity of unconstrained HMMC.

Run: python examples/figure_nucleation_benchmark_three_panel.py
Add --verify for time-step and class-count refinement, without extra figures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from math import comb
from pathlib import Path
import sys

import matplotlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

matplotlib.use("Agg")
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "examples")]

from adaptivehmmc._redistribution import local_stencil_start  # noqa: E402
from figure_growth_benchmark import (  # noqa: E402
    GrowthHistory, append_state, build_delta_l_projection_matrix,
    redistribute_population_nonnegative, should_adapt,
    solve_nonnegative_moment_weights,
)
from figure_breakage_benchmark_three_panel import (  # noqa: E402
    FIGS, PsdCase, geometric_grid, plot_three_panel_case_validation, result_cdf_at,
)


@dataclass(frozen=True)
class Parameters:
    num_classes: int = 21
    num_moments: int = 6
    diameter_min: float = 0.0
    diameter_max: float = 1.0
    grid_ratio: float = 1.0  # Ratio of consecutive diameter-cell widths.
    initial_number: float = 1.0
    nuclei_size: float = 0.1
    nucleation_rate: float = 0.5
    growth_rate: float = 0.2
    final_time: float = 10.0
    num_steps: int = 10
    f_max: float = 1.8
    f_mult: float = 1.1


GAUSS_X, GAUSS_W = leggauss(5)


def hat_density(d, a):
    d = np.asarray(d, dtype=float)
    return np.maximum(1.0 - np.abs(d / a - 1.0), 0.0) / a


def hat_cdf(d, a):
    z = np.clip(np.asarray(d, dtype=float) / a, 0.0, 2.0)
    return np.where(z <= 1.0, 0.5 * z**2, 1.0 - 0.5 * (2.0 - z)**2)


def analytical_density(d, time, p):
    if p.growth_rate == 0.0:
        return (p.initial_number + p.nucleation_rate * time) * hat_density(d, p.nuclei_size)
    shift = p.growth_rate * time
    return (
        p.initial_number * hat_density(np.asarray(d) - shift, p.nuclei_size)
        + p.nucleation_rate / p.growth_rate
        * (hat_cdf(d, p.nuclei_size) - hat_cdf(np.asarray(d) - shift, p.nuclei_size))
    )


def density_knots(time, p):
    source = np.array([0.0, p.nuclei_size, 2.0 * p.nuclei_size])
    return np.unique(np.r_[source, source + p.growth_rate * time])


def cumulative_moment(points, power, time, p):
    """Integrate each polynomial segment exactly by five-point Gaussian quadrature."""
    points = np.asarray(points, dtype=float)
    values = np.zeros_like(points)
    knots = density_knots(time, p)
    for lower, upper in zip(knots[:-1], knots[1:]):
        half_width = 0.5 * (np.clip(points, lower, upper) - lower)
        x = lower + half_width[..., None] * (1.0 + GAUSS_X)
        values += half_width * np.sum(
            GAUSS_W * analytical_density(x, time, p) * x**power, axis=-1
        )
    return values


def birth_moments(p):
    # Triangular density is linear on each half; all six moments integrate exactly.
    x = np.concatenate([0.5 * p.nuclei_size * (1 + GAUSS_X),
                        p.nuclei_size + 0.5 * p.nuclei_size * (1 + GAUSS_X)])
    w = np.tile(GAUSS_W, 2) * 0.5 * p.nuclei_size * hat_density(x, p.nuclei_size)
    return np.array([np.sum(w * x**q) for q in range(p.num_moments)])


def analytical_moments(times, p):
    """Translate seeds and integrate over all birth ages, without time discretization."""
    times = np.atleast_1d(times)
    birth = birth_moments(p)
    moments = np.zeros((p.num_moments, times.size))
    for q in range(p.num_moments):
        for r in range(q + 1):
            k = q - r
            coefficient = comb(q, r) * birth[r] * p.growth_rate**k
            moments[q] += coefficient * (
                p.initial_number * times**k
                + p.nucleation_rate * times**(k + 1) / (k + 1)
            )
    return moments


def project_density(pivots, density, knots, orders):
    """Fit interval moments on local stencils using the existing nonnegative solver."""
    population = np.zeros_like(pivots)
    edges = np.unique(np.r_[knots, pivots[(pivots > knots[0]) & (pivots < knots[-1])]])
    for lower, upper in zip(edges[:-1], edges[1:]):
        x = lower + 0.5 * (upper - lower) * (1.0 + GAUSS_X)
        mass = 0.5 * (upper - lower) * GAUSS_W * density(x)
        moments = np.array([np.sum(mass * x**q) for q in orders])
        if moments[0] <= 0.0:
            continue
        mean = moments[1] / moments[0]
        first = local_stencil_start(pivots, mean, orders.size)
        stencil = slice(first, first + orders.size)
        weights = solve_nonnegative_moment_weights(pivots[stencil], moments / moments[0], orders)
        population[stencil] += moments[0] * weights
    return population


def run_case(p, adaptive):
    if not (p.num_classes >= p.num_moments >= 5 and p.num_steps > 0
            and p.nuclei_size > 0 and p.final_time > 0 and p.initial_number > 0
            and p.growth_rate >= 0 and p.nucleation_rate >= 0 and p.f_mult > 1
            and p.f_max > 0 and p.diameter_max > 2 * p.nuclei_size):
        raise ValueError("Invalid nucleation benchmark parameters")
    pivots, _ = geometric_grid(p.num_classes, p.diameter_min, p.diameter_max, p.grid_ratio)
    orders = np.arange(p.num_moments)
    population = project_density(pivots, lambda x: analytical_density(x, 0.0, p),
                                 density_knots(0.0, p), orders)
    history = {key: [] for key in ("time", "population", "pivots", "moments", "d43", "target_upper_pivot")}
    append_state(history, 0.0, population, pivots, p.num_moments, p.f_max)
    adaptations = []
    dt = p.final_time / p.num_steps
    source_parameters = replace(p, initial_number=0.0)
    rebuild = True
    for step in range(p.num_steps):
        if rebuild:
            projection = (build_delta_l_projection_matrix(
                pivots, p.growth_rate * dt, orders, enforce_nonnegative=True
            ) if p.growth_rate else np.eye(p.num_classes))
            # Newly born particles have ages 0..dt: integrate their growth exactly.
            births = project_density(
                pivots, lambda x: analytical_density(x, dt, source_parameters),
                density_knots(dt, source_parameters), orders,
            )
            rebuild = False
        population = projection @ population + births
        time = (step + 1) * dt
        append_state(history, time, population, pivots, p.num_moments, p.f_max)
        if adaptive and should_adapt(population, pivots, p.f_max, p.f_mult):
            population, pivots, changed = redistribute_population_nonnegative(
                population, pivots, orders, p.f_max
            )
            if changed:
                adaptations.append(time)
                append_state(history, time, population, pivots, p.num_moments, p.f_max)
                rebuild = True
    return GrowthHistory(**{key: np.array(values).T if key in ("population", "pivots", "moments")
                            else np.array(values) for key, values in history.items()},
                         adaptation_times=np.array(adaptations))


def make_case(p):
    name = "Nucleation + growth" if p.growth_rate else "Nucleation"
    fixed, adaptive = run_case(p, False), run_case(p, True)
    upper = 2 * p.nuclei_size + p.growth_rate * p.final_time
    total_number = p.initial_number + p.nucleation_rate * p.final_time

    def quantile(probability):
        return brentq(lambda d: cumulative_moment(d, 0, p.final_time, p) / total_number - probability,
                      0.0, upper)

    case = PsdCase(
        name, fixed, adaptive, lambda d: analytical_density(d, p.final_time, p), quantile,
        lambda times, initial: analytical_moments(times, p),
        "nucleation_growth_moment_validation" if p.growth_rate else "nucleation_moment_validation",
        grid_ylog=False, moment_clip_negative=False, log_x=False,
    )
    return case


def curve_and_diagnostics(case, p):
    max_d = max(2 * p.nuclei_size + p.growth_rate * p.final_time,
                case.fixed.pivots[-1, -1], case.adaptive.pivots[-1, -1]) * 1.05
    # Include both sides of every discrete CDF jump in the error calculation.
    pivots = np.r_[case.fixed.pivots[:, -1], case.adaptive.pivots[:, -1]]
    points = np.unique(np.r_[np.linspace(0, max_d, 4000), pivots, np.nextafter(pivots, 0)])
    ref_moments = analytical_moments([p.final_time], p)[:, 0]
    ref_cdf = cumulative_moment(points, 3, p.final_time, p) / ref_moments[3]
    curve = dict(case=case, points=points, reference=ref_cdf)
    report = dict(parameters=asdict(p), projection="nonnegative approximate six-moment fit")
    for label in ("fixed", "adaptive"):
        result = getattr(case, label)
        curve[label] = result_cdf_at(result, points, moment_power=3)
        relative_error = result.moments[:, -1] / ref_moments - 1.0
        exact_number = p.initial_number + p.nucleation_rate * result.time
        assert np.all(np.isfinite(result.population)) and np.min(result.population) >= 0
        assert np.allclose(result.moments[0], exact_number, rtol=1e-9, atol=1e-12)
        report[label] = dict(
            final_moment_relative_errors=relative_error.tolist(),
            final_d43_relative_error=float(result.d43[-1] / (ref_moments[4] / ref_moments[3]) - 1),
            max_volume_cdf_error=float(np.max(np.abs(curve[label] - ref_cdf))),
            adaptations=int(result.adaptation_times.size),
            minimum_population=float(np.min(result.population)),
            birth_probability_below_first_pivot=float(hat_cdf(result.pivots[0, -1], p.nuclei_size)),
            final_pivots_in_birth_band=int(np.count_nonzero(result.pivots[:, -1] < 2 * p.nuclei_size)),
        )
    # Independent piecewise quadrature checks the closed moment formulas.
    for time in (0.0, p.final_time / 2, p.final_time):
        exact = analytical_moments([time], p)[:, 0]
        integrated = np.array([cumulative_moment(max_d, q, time, p) for q in range(p.num_moments)])
        assert np.allclose(exact, integrated, rtol=1e-10, atol=1e-13)
    return curve, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    reports = {}
    pure_nucleation = replace(
        Parameters(),
        grid_ratio=1.08,
        nuclei_size=0.08,
        growth_rate=0.0,
        num_steps=100,
        f_max=2.5,
        f_mult=1.3,
    )
    for p in (pure_nucleation, Parameters()):
        print(f"Running nucleation with G={p.growth_rate:g} mm/s", flush=True)
        case = make_case(p)
        curve, report = curve_and_diagnostics(case, p)
        output = plot_three_panel_case_validation(curve)
        for label in ("fixed", "adaptive"):
            values = report[label]
            print(f"  {label}: CDF error={100*values['max_volume_cdf_error']:.3f}%, "
                  f"d43 error={100*values['final_d43_relative_error']:.3f}%, "
                  f"adaptations={values['adaptations']}", flush=True)
        if args.verify:
            report["refinement"] = {}
            for label, refined in (("half_dt", replace(p, num_steps=2*p.num_steps)),
                                   ("double_classes", replace(p, num_classes=2*p.num_classes,
                                    grid_ratio=np.sqrt(p.grid_ratio)))):
                fine_case = make_case(refined)
                _, fine_report = curve_and_diagnostics(fine_case, refined)
                report["refinement"][label] = fine_report
                print(f"  {label} adaptive CDF error="
                      f"{100*fine_report['adaptive']['max_volume_cdf_error']:.3f}%", flush=True)
        reports[case.name] = report
        print(f"Saved {FIGS / (output + '.pdf')}", flush=True)
    (FIGS / "nucleation_benchmark_diagnostics.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
