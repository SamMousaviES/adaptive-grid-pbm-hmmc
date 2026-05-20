"""Reproduce manuscript Figure: benchmark_tradeoff and Table: benchmark_tradeoff.tex.

Sweeps the hysteresis factor f_mult on the constant-kernel coalescence
benchmark with 20 categories. The accuracy diagnostic is the final relative
error in d43. The figure reports CPU time and adaptation count vs f_mult on
twin y-axes.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import adaptivehmmc
from _paths import output_dir, table_dir


def coalescence_reference(time: float, initial_moments: np.ndarray, rate: float) -> np.ndarray:
    initial_number = initial_moments[0]
    factor = 2.0 / (2.0 + rate * initial_number * time)
    orders = np.arange(initial_moments.size)
    return initial_moments * factor ** (1.0 - orders / 3.0)


def write_table(path, header, rows):
    body = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    body.extend(row + r" \\" for row in rows)
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    expansion_factors = np.arange(1.1, 1.85, 0.1)
    n_f = expansion_factors.size
    repeats = 20

    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 1000.0)
    base.num_classes = 20
    base.num_moments = 6
    base.grid_ratio = 1.3
    base.diameter_min = 0.0
    base.diameter_max = 1.0
    base.f_max = 2.5
    base.adaptive = True

    d43_error = np.zeros(n_f)
    cpu_time = np.zeros(n_f)
    adaptations = np.zeros(n_f, dtype=int)
    ode_steps = np.zeros(n_f, dtype=int)

    for k, fm in enumerate(expansion_factors):
        print(f"Adaptive coalescence, f_mult = {fm:.1f} ({repeats} repeats) ...")
        results = []
        cpu_samples = np.zeros(repeats)
        for r in range(repeats):
            opts = base.copy()
            opts.f_mult = float(fm)
            result = adaptivehmmc.solve(opts)
            results.append(result)
            cpu_samples[r] = result.report.cpu_time
            print(f"  run {r + 1}/{repeats}: {cpu_samples[r]:.3f} s")

        median_idx = int(np.argsort(cpu_samples)[repeats // 2])
        result = results[median_idx]
        initial_moments = result.moments[:, 0]
        reference = coalescence_reference(base.time_span[-1], initial_moments, base.process.rate)
        d43 = result.moments[4, -1] / result.moments[3, -1]
        d43_ref = reference[4] / reference[3]
        d43_error[k] = 100.0 * abs((d43 - d43_ref) / d43_ref)
        cpu_time[k] = np.median(cpu_samples)
        adaptations[k] = result.report.num_adaptations
        ode_steps[k] = result.report.total_ode_steps

    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    cpu_line, = ax1.plot(expansion_factors, cpu_time, "s--",
                         color="#8d6e63", lw=1.6, ms=6, label="Median CPU time")
    ax1.set_xlabel(r"Grid expansion factor $f_{\mathrm{mult}}$")
    ax1.set_ylabel("Median CPU time [s]", color="#8d6e63")
    ax1.tick_params(axis="y", labelcolor="#8d6e63")
    ax1.grid(True, ls=":", alpha=0.6)

    ax2 = ax1.twinx()
    adapt_line, = ax2.plot(expansion_factors, adaptations, "d-.",
                           color="#1565c0", lw=1.6, ms=6, label="Adaptations")
    ax2.set_ylabel("Adaptation count", color="#1565c0")
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(axis="y", labelcolor="#1565c0")

    ax1.legend(handles=[cpu_line, adapt_line], loc="lower center",
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "benchmark_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(out / "benchmark_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tdir = table_dir()
    rows = [
        f"{fm:.1f} & {err:.4f} & {cpu:.3f} & {int(steps)} & {int(adapt)}"
        for fm, err, cpu, steps, adapt in zip(
            expansion_factors, d43_error, cpu_time, ode_steps, adaptations
        )
    ]
    write_table(
        tdir / "benchmark_tradeoff.tex",
        r"$f_{\mathrm{mult}}$ & Final $d_{43}$ error [\%] & Median CPU time [s] & ODE steps & Adaptations",
        rows,
    )

    print(f"Saved benchmark_tradeoff.{{pdf,png}} to {out}")
    print(f"Saved benchmark_tradeoff.tex to {tdir}")


if __name__ == "__main__":
    main()
