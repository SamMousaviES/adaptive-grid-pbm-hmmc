"""Reproduce manuscript Table: category_comparison.tex.

Sweeps the category count for the constant-kernel coalescence benchmark and
reports representative CPU times for fixed and adaptive grids at equal
category count.
"""

from __future__ import annotations

import numpy as np

import adaptivehmmc
from _paths import table_dir


def write_table(path, header, rows):
    body = [
        r"\begin{tabular}{lrrr}",
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
    repeats = 5

    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 1000.0)
    base.num_moments = 6
    base.grid_ratio = 1.3
    base.diameter_min = 0.0
    base.diameter_max = 1.0
    base.f_max = 2.5
    base.f_mult = 1.4

    fixed_cpu = np.zeros(len(category_counts))
    adaptive_cpu = np.zeros(len(category_counts))

    for k, nc in enumerate(category_counts):
        print(f"Category count = {nc} (fixed + adaptive, {repeats} repeats) ...")
        fixed_runs = np.zeros(repeats)
        adaptive_runs = np.zeros(repeats)
        for r in range(repeats):
            fixed_opts = base.copy()
            fixed_opts.num_classes = nc
            fixed_opts.adaptive = False
            fixed = adaptivehmmc.solve(fixed_opts)

            adaptive_opts = base.copy()
            adaptive_opts.num_classes = nc
            adaptive_opts.adaptive = True
            adaptive = adaptivehmmc.solve(adaptive_opts)

            fixed_runs[r] = fixed.report.cpu_time
            adaptive_runs[r] = adaptive.report.cpu_time

        fixed_cpu[k] = np.median(fixed_runs)
        adaptive_cpu[k] = np.median(adaptive_runs)

    tdir = table_dir()
    rows = [
        f"{nc} & {fc:.3f} & {ac:.3f} & \\textcolor{{blue}}{{{ac / fc:.2f}}}"
        for nc, fc, ac in zip(category_counts, fixed_cpu, adaptive_cpu)
    ]
    write_table(
        tdir / "category_comparison.tex",
        r"Categories & Fixed CPU [s] & Adaptive CPU [s] & \textcolor{blue}{Ratio}",
        rows,
    )

    print(f"Saved category_comparison.tex to {tdir}")


if __name__ == "__main__":
    main()
