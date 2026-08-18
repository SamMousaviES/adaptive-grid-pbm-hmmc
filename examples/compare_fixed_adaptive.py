"""Compare fixed and adaptive CPU time at equal class count."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import adaptivehmmc
from _paths import output_dir


def coalescence_reference(time: float, initial_moments: np.ndarray, rate: float) -> np.ndarray:
    initial_number = initial_moments[0]
    factor = 2.0 / (2.0 + rate * initial_number * time)
    orders = np.arange(initial_moments.size)
    return initial_moments * factor ** (1.0 - orders / 3.0)


def repeated_solve(base, adaptive_enabled, repeats, label):
    results = []
    cpu_times = np.zeros(repeats)
    for i in range(repeats):
        opts = base.copy()
        opts.adaptive = adaptive_enabled
        result = adaptivehmmc.solve(opts)
        results.append(result)
        cpu_times[i] = result.report.cpu_time
        print(f"{label} run {i + 1}/{repeats}: {cpu_times[i]:.3f} s")

    median_idx = int(np.argsort(cpu_times)[repeats // 2])
    return results[median_idx], cpu_times


def main() -> None:
    repeats = 5
    category_counts = np.array([10, 20, 30, 40])

    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 1000.0)
    base.f_mult = 1.4

    fixed_median_cpu = np.zeros(category_counts.size)
    adaptive_median_cpu = np.zeros(category_counts.size)
    fixed_error = np.zeros(category_counts.size)
    adaptive_error = np.zeros(category_counts.size)

    for k, nc in enumerate(category_counts):
        print(f"Category count = {nc}")
        opts = base.copy()
        opts.num_classes = int(nc)

        fixed, fixed_cpu = repeated_solve(opts, False, repeats, "Fixed-grid")
        adaptive, adaptive_cpu = repeated_solve(opts, True, repeats, "Adaptive-grid")

        reference = coalescence_reference(opts.time_span[-1], fixed.moments[:, 0], opts.process.rate)
        fixed_error[k] = 100.0 * np.max(np.abs((fixed.moments[:, -1] - reference) / reference))
        adaptive_error[k] = 100.0 * np.max(np.abs((adaptive.moments[:, -1] - reference) / reference))
        fixed_median_cpu[k] = np.median(fixed_cpu)
        adaptive_median_cpu[k] = np.median(adaptive_cpu)

        print(f"Fixed-grid max moment error:    {fixed_error[k]:.3f} %")
        print(f"Adaptive-grid max moment error: {adaptive_error[k]:.3f} %")
        print(f"Fixed-grid median CPU:          {fixed_median_cpu[k]:.3f} s")
        print(f"Adaptive-grid median CPU:       {adaptive_median_cpu[k]:.3f} s")
        print(f"Adaptive/fixed CPU ratio:       {adaptive_median_cpu[k] / fixed_median_cpu[k]:.2f}")

    cpu_ratio = adaptive_median_cpu / fixed_median_cpu

    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    fixed_line, = ax1.plot(category_counts, fixed_median_cpu, "o--",
                           color="#8c1f1f", lw=1.6, ms=6, label="Fixed-grid CPU")
    adaptive_line, = ax1.plot(category_counts, adaptive_median_cpu, "s-",
                              color="#8d6e63", lw=1.6, ms=6, label="Adaptive-grid CPU")
    ax1.set_xlabel("Number of categories")
    ax1.set_ylabel("CPU time [s]")
    ax1.set_xticks(category_counts)
    ax1.grid(True, ls=":", alpha=0.6)

    ax2 = ax1.twinx()
    ratio_line, = ax2.plot(category_counts, cpu_ratio, "d-.",
                           color="#1565c0", lw=1.6, ms=6,
                           label="Adaptive/fixed ratio")
    ax2.set_ylabel("CPU ratio", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")

    ax1.legend(handles=[fixed_line, adaptive_line, ratio_line], loc="lower center",
               bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "compare_fixed_adaptive.pdf", bbox_inches="tight")
    fig.savefig(out / "compare_fixed_adaptive.png", dpi=300, bbox_inches="tight")
    print(f"Saved compare_fixed_adaptive.{{pdf,png}} to {out}")


if __name__ == "__main__":
    main()
