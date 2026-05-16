"""Reproduce manuscript Table: category_comparison.tex.

Sweeps the category count for the constant-kernel coalescence benchmark and
reports the maximum relative moment error vs the analytical reference and
the CPU time, for fixed and adaptive grids at equal category count.
"""

from __future__ import annotations

import numpy as np

import adaptivehmmc
from _paths import table_dir


def coalescence_reference(time, initial_moments, rate):
    initial_number = initial_moments[0]
    factor = 2.0 / (2.0 + rate * initial_number * time)
    orders = np.arange(initial_moments.size)
    return initial_moments * factor ** (1.0 - orders / 3.0)


def write_table(path, header, rows):
    body = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    body.extend(row + r" \\" for row in rows)
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    category_counts = [10, 20, 30, 40]

    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 1000.0)
    base.num_moments = 6
    base.grid_ratio = 1.3
    base.diameter_min = 0.0
    base.diameter_max = 1.0
    base.f_max = 2.5
    base.f_mult = 1.4

    fixed_err = np.zeros(len(category_counts))
    adaptive_err = np.zeros(len(category_counts))
    fixed_cpu = np.zeros(len(category_counts))
    adaptive_cpu = np.zeros(len(category_counts))
    adaptive_events = np.zeros(len(category_counts), dtype=int)

    for k, nc in enumerate(category_counts):
        print(f"Category count = {nc} (fixed + adaptive) ...")
        fixed_opts = base.copy()
        fixed_opts.num_classes = nc
        fixed_opts.adaptive = False
        fixed = adaptivehmmc.solve(fixed_opts)

        adaptive_opts = base.copy()
        adaptive_opts.num_classes = nc
        adaptive_opts.adaptive = True
        adaptive = adaptivehmmc.solve(adaptive_opts)

        reference = coalescence_reference(base.time_span[-1], fixed.moments[:, 0], base.process.rate)
        fixed_err[k]    = 100.0 * np.max(np.abs((fixed.moments[:, -1] - reference) / reference))
        adaptive_err[k] = 100.0 * np.max(np.abs((adaptive.moments[:, -1] - reference) / reference))
        fixed_cpu[k] = fixed.report.cpu_time
        adaptive_cpu[k] = adaptive.report.cpu_time
        adaptive_events[k] = adaptive.report.num_adaptations

    tdir = table_dir()
    rows = [
        f"{nc} & {fe:.3f} & {ae:.3f} & {fc:.3f} & {ac:.3f} & {ev}"
        for nc, fe, ae, fc, ac, ev in zip(
            category_counts, fixed_err, adaptive_err,
            fixed_cpu, adaptive_cpu, adaptive_events
        )
    ]
    write_table(
        tdir / "category_comparison.tex",
        r"Categories & Fixed err. [\%] & Adaptive err. [\%] & "
        r"Fixed CPU [s] & Adaptive CPU [s] & Adaptive events",
        rows,
    )

    print(f"Saved category_comparison.tex to {tdir}")


if __name__ == "__main__":
    main()
