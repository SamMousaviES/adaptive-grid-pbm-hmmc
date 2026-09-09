"""Generate the f_max sensitivity table for the coalescence benchmark."""

from __future__ import annotations

import numpy as np
from scipy.special import gammainc

import adaptivehmmc
from _paths import table_dir
from figure_benchmark_tradeoff import coalescence_reference, write_table


def volume_cdf_error(result, factor: float, volume_scale: float) -> float:
    pivots = np.asarray(result.pivots[:, -1], dtype=float)
    population = np.maximum(np.asarray(result.population[:, -1], dtype=float), 0.0)
    weights = population * pivots**3
    numerical = np.cumsum(weights) / weights.sum()
    points = np.unique(np.concatenate(([0.0], pivots, [1.05 * pivots[-1]])))
    indices = np.searchsorted(pivots, points, side="right") - 1
    values = np.zeros_like(points)
    valid = indices >= 0
    values[valid] = numerical[indices[valid]]
    values[points >= pivots[-1]] = 1.0
    reference = gammainc(2.0, factor * points**3 / volume_scale)
    return 100.0 * float(np.max(np.abs(values - reference)))


def main() -> None:
    f_max_values = np.array([1.5, 2.0, 2.5, 3.5, 5.0])
    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 1000.0)
    base.num_classes = 20
    base.num_moments = 6
    base.grid_ratio = 1.3
    base.diameter_min = 0.0
    base.diameter_max = 1.0
    base.f_mult = 1.4
    base.adaptive = True

    rows = []
    for f_max in f_max_values:
        options = base.copy()
        options.f_max = float(f_max)
        result = adaptivehmmc.solve(options)
        initial_moments = result.moments[:, 0]
        reference = coalescence_reference(
            base.time_span[-1], initial_moments, base.process.rate
        )
        d43 = result.moments[4, -1] / result.moments[3, -1]
        d43_ref = reference[4] / reference[3]
        factor = 2.0 / (2.0 + base.process.rate * initial_moments[0] * base.time_span[-1])
        cdf_error = volume_cdf_error(result, factor, initial_moments[3] / initial_moments[0])
        rows.append(
            f"{f_max:.1f} & {100.0 * abs((d43 - d43_ref) / d43_ref):.4f} & "
            f"{cdf_error:.2f} & {result.report.num_adaptations}"
        )

    write_table(
        table_dir() / "fmax_sensitivity.tex",
        r"$f_{\max}$ & Final $d_{43}$ error [\%] & Final $\Delta_V$ [\%] & Adaptations",
        rows,
        column_spec="lrrr",
    )
    print(f"Saved fmax_sensitivity.tex to {table_dir()}")


if __name__ == "__main__":
    main()
