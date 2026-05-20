"""Reproduce manuscript Figure breakage_moment_validation.

Binary equal-volume breakage with constant breakage frequency: the moments
satisfy ``M_q(t)/M_q(0) = exp(g_b * (2^(1-q/3)-1) * t)``. The case is chosen
so that d43 decreases enough to trigger downward uniform grid scaling, while
the breakage table remains reusable thanks to the scale-similar daughter law.
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
    """Initial log-normal population, normalized so M_3 = 1."""
    raw = np.exp(-0.5 * (np.log(pivots / center) / sigma) ** 2) / pivots
    raw *= widths
    raw /= raw.sum()
    third_moment = np.sum(raw * pivots ** 3)
    return raw / third_moment


def analytical_moments(initial_moments, orders, tau):
    return initial_moments * np.exp((2.0 ** (1.0 - orders / 3.0) - 1.0) * tau)


def moment_history(result, num_moments):
    orders = np.arange(num_moments)
    population = result.population
    moments = np.zeros((num_moments, result.time.size))
    for i in range(result.time.size):
        pivots = result.pivots[:, i]
        moments[:, i] = (
            population[:, i, None] * pivots[:, None] ** orders[None, :]
        ).sum(axis=0)
    return moments


def d43_from_history(moments):
    return moments[4, :] / moments[3, :]


def main() -> None:
    num_classes = 21
    num_moments = 6
    grid_ratio = 1
    diameter_min = 0.0
    diameter_max = 100.0
    simulation_time = 10  # s
    rate = 3.7
    final_tau = rate * simulation_time
    f_max = 1.3
    f_mult = 1.25
    lognorm_center = 85.0
    lognorm_sigma = 0.5

    pivots, widths = geometric_grid(num_classes, diameter_min, diameter_max, grid_ratio)
    initial_population = lognormal_on_grid(pivots, widths, lognorm_center, lognorm_sigma)

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

    print("Running fixed-grid breakage ...")
    fixed_opts = base.copy()
    fixed_opts.adaptive = False
    fixed = adaptivehmmc.solve(fixed_opts)

    print("Running adaptive-grid breakage ...")
    adaptive_opts = base.copy()
    adaptive_opts.adaptive = True
    adaptive = adaptivehmmc.solve(adaptive_opts)

    initial_moments = fixed.moments[:, 0]
    orders = np.arange(num_moments)
    analytical_final = analytical_moments(initial_moments, orders, final_tau)

    fixed_moments = moment_history(fixed, num_moments)
    adaptive_moments = moment_history(adaptive, num_moments)
    fixed_d43 = d43_from_history(fixed_moments)
    adaptive_d43 = d43_from_history(adaptive_moments)

    fixed_err = 100.0 * np.max(np.abs((fixed_moments[:, -1] - analytical_final) / analytical_final))
    adaptive_err = 100.0 * np.max(np.abs((adaptive_moments[:, -1] - analytical_final) / analytical_final))
    print(f"Fixed-grid max moment error:    {fixed_err:.3e} %")
    print(f"Adaptive-grid max moment error: {adaptive_err:.3e} %")
    print(f"Adaptation events: {adaptive.report.num_adaptations}")

    time_reference = np.linspace(0.0, simulation_time, 240)
    tau_reference = rate * time_reference
    reference = np.column_stack([
        analytical_moments(initial_moments, orders, tau) for tau in tau_reference
    ])

    reference_d43 = reference[4, :] / reference[3, :]

    fig, (ax_mom, ax_grid) = plt.subplots(2, 1, figsize=(5.0, 5.8))
    colors = plt.get_cmap("tab10")
    marker_idx = np.unique(np.round(np.linspace(0, adaptive.time.size - 1, 30)).astype(int))

    for i in range(num_moments):
        ax_mom.semilogy(time_reference, reference[i, :] / initial_moments[i],
                        "-", color=colors(i), lw=1.2, label=f"$M_{i}$")
        ax_mom.semilogy(fixed.time, fixed_moments[i, :] / initial_moments[i],
                        "--", color=colors(i), lw=0.9)
        ax_mom.semilogy(adaptive.time[marker_idx],
                        adaptive_moments[i, marker_idx] / initial_moments[i],
                        "o", color=colors(i), markerfacecolor="white",
                        markersize=2.8, lw=0.7)

    ax_mom.set_xlabel("Time (s)")
    ax_mom.set_ylabel(r"$M_q / M_q(0)$")
    ax_mom.grid(True, which="major", ls=":", alpha=0.6)
    leg1 = ax_mom.legend(loc="upper left", ncol=3, fontsize=7, title="moment")

    # Style legend (analytical/fixed/adaptive)
    style_handles = [
        plt.Line2D([], [], color="k", ls="-", lw=1.2, label="Analytical"),
        plt.Line2D([], [], color="k", ls="--", lw=0.9, label="Fixed grid"),
        plt.Line2D([], [], color="k", marker="o", ls="None",
                   markerfacecolor="white", markersize=4, label="Adaptive grid"),
    ]
    ax_mom.add_artist(leg1)
    ax_mom.legend(handles=style_handles, loc="lower left", fontsize=7)

    ax_grid.plot(time_reference, reference_d43, "-", color="black", lw=1.2,
                 label=r"analytical $d_{43}$")
    ax_grid.plot(fixed.time, fixed_d43, "--", color="#8c1f1f", lw=1.0,
                 label=r"fixed $d_{43}$")
    ax_grid.plot(adaptive.time, adaptive_d43, "-", color="#1a619c", lw=1.1,
                 label=r"adaptive $d_{43}$")
    ax_grid.step(adaptive.time, adaptive.pivots[-1, :], where="post",
                 color="#007358", lw=1.1, label=r"adaptive $d_N$")
    ax_grid.axhline(fixed.pivots[-1, 0], ls="-.", color="0.55", lw=0.9,
                    label=r"fixed $d_N$")
    for t_event in adaptive.adaptation_times:
        ax_grid.axvline(t_event, ls=":", color="0.70", lw=0.6)
    ax_grid.set_xlabel("Time (s)")
    ax_grid.set_ylabel(r"Diameter, $d$ (mm)")
    ax_grid.set_yscale("log")
    ax_grid.grid(True, ls=":", alpha=0.6)
    ax_grid.legend(loc="upper right", fontsize=7)

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "breakage_moment_validation.pdf", bbox_inches="tight")
    fig.savefig(out / "breakage_moment_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved breakage_moment_validation to {out}")


if __name__ == "__main__":
    main()
