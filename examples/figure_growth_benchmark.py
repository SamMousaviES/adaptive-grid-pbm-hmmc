"""Reproduce manuscript Figure: growth_moment_validation.

Pure proportional growth ``G(d) = k_g d``. The analytical characteristic
solution is ``d(t) = d(0) exp(k_g t)`` and the moments satisfy
``M_q(tau)/M_q(0) = exp(q tau)``. The numerical solver advances each pivot
along this characteristic and projects to the current grid by local moment
matching; uniform grid scaling follows the dilation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import adaptivehmmc
from _paths import output_dir


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


def analytical_moments(initial_moments, orders, tau):
    return initial_moments * np.exp(orders * tau)


def main() -> None:
    num_classes = 31
    num_moments = 6
    grid_ratio = 1.15
    diameter_min = 0.0
    diameter_max = 2.0
    final_tau = 3.0
    num_steps = 500
    rate = 1.0
    f_max = 6.0
    f_mult = 1.25
    lognorm_center = 0.45
    lognorm_sigma = 0.18

    pivots, widths = geometric_grid(num_classes, diameter_min, diameter_max, grid_ratio)
    initial_population = lognormal_on_grid(pivots, widths, lognorm_center, lognorm_sigma)

    base = adaptivehmmc.default_options()
    base.time_span = (0.0, final_tau / rate)
    base.num_classes = num_classes
    base.num_moments = num_moments
    base.diameter_min = diameter_min
    base.diameter_max = diameter_max
    base.grid_ratio = grid_ratio
    base.f_max = f_max
    base.f_mult = f_mult
    base.process.type = "growth"
    base.process.kernel = "proportional"
    base.process.rate = rate
    base.initial_distribution.type = "custom"
    base.initial_distribution.pivots = pivots
    base.initial_distribution.population = initial_population
    base.solver.num_steps = num_steps

    print("Running fixed-grid growth ...")
    fixed_opts = base.copy()
    fixed_opts.adaptive = False
    fixed = adaptivehmmc.solve(fixed_opts)

    print("Running adaptive-grid growth ...")
    adaptive_opts = base.copy()
    adaptive_opts.adaptive = True
    adaptive = adaptivehmmc.solve(adaptive_opts)

    initial_moments = fixed.moments[:, 0]
    orders = np.arange(num_moments)
    analytical_final = analytical_moments(initial_moments, orders, final_tau)

    fixed_err = 100.0 * np.max(np.abs((fixed.moments[:, -1] - analytical_final) / analytical_final))
    adaptive_err = 100.0 * np.max(np.abs((adaptive.moments[:, -1] - analytical_final) / analytical_final))
    print(f"Fixed-grid max moment error:    {fixed_err:.3e} %")
    print(f"Adaptive-grid max moment error: {adaptive_err:.3e} %")
    print(f"Adaptation events: {adaptive.report.num_adaptations}")

    tau_reference = np.linspace(0.0, final_tau, 240)
    reference = np.column_stack([
        analytical_moments(initial_moments, orders, tau) for tau in tau_reference
    ])
    reference_d43 = reference[4, :] / reference[3, :]

    fig, (ax_mom, ax_grid) = plt.subplots(2, 1, figsize=(4.5, 5.5))

    colors = plt.get_cmap("tab10")
    marker_idx = np.unique(np.round(np.linspace(0, adaptive.time.size - 1, 28)).astype(int))

    for i in range(num_moments):
        ax_mom.semilogy(tau_reference, reference[i, :] / initial_moments[i],
                        "-", color=colors(i), lw=1.2, label=f"$M_{i}$")
        ax_mom.semilogy(fixed.time * rate, fixed.moments[i, :] / initial_moments[i],
                        "--", color=colors(i), lw=0.9)
        ax_mom.semilogy(adaptive.time[marker_idx] * rate,
                        adaptive.moments[i, marker_idx] / initial_moments[i],
                        "o", color=colors(i), markerfacecolor="white",
                        markersize=2.8, lw=0.7)
    ax_mom.set_xlabel(r"Dimensionless time, $\tau = k_g t$")
    ax_mom.set_ylabel(r"$M_q / M_q(0)$")
    ax_mom.grid(True, which="major", ls=":", alpha=0.6)
    ax_mom.legend(loc="upper left", ncol=3, fontsize=7)

    ax_grid.semilogy(tau_reference, reference_d43, "-", color="black", lw=1.2,
                     label=r"analytical $d_{43}$")
    ax_grid.semilogy(fixed.time * rate, fixed.d43, "--", color="#8c1f1f",
                     lw=1.0, label=r"fixed $d_{43}$")
    ax_grid.semilogy(adaptive.time * rate, adaptive.d43, "-", color="#1a619c",
                     lw=1.1, label=r"adaptive $d_{43}$")
    ax_grid.step(adaptive.time * rate, adaptive.pivots[-1, :], where="post",
                 color="#007358", lw=1.1, label=r"adaptive $d_N$")
    ax_grid.axhline(fixed.pivots[-1, 0], ls="-.", color="0.55", lw=0.9,
                    label=r"fixed $d_N$")
    for t_event in adaptive.adaptation_times:
        ax_grid.axvline(t_event * rate, ls=":", color="0.70", lw=0.6)
    ax_grid.set_xlabel(r"Dimensionless time, $\tau = k_g t$")
    ax_grid.set_ylabel("Diameter")
    ax_grid.grid(True, which="major", ls=":", alpha=0.6)
    ax_grid.legend(loc="upper left", fontsize=7)

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "growth_moment_validation.pdf", bbox_inches="tight")
    fig.savefig(out / "growth_moment_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved growth_moment_validation to {out}")


if __name__ == "__main__":
    main()
