"""Reproduce manuscript Figure coalescence_benchmark.

Single fixed-grid run plus one adaptive run at f_mult = 1.4 on the
constant-kernel coalescence benchmark with 20 categories and unit initial
diameter range. The upper panel compares normalized analytical, fixed-grid,
and adaptive-grid moments. The lower panel shows d43 and upper-pivot histories.
"""

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


def nonnegative_moment_history(result, num_moments: int) -> np.ndarray:
    orders = np.arange(num_moments)
    population = np.maximum(result.population, 0.0)
    moments = np.zeros((num_moments, result.time.size))
    for i in range(result.time.size):
        pivots = result.pivots[:, i]
        moments[:, i] = (
            population[:, i, None] * pivots[:, None] ** orders[None, :]
        ).sum(axis=0)
    return moments


def d43_from_history(moments: np.ndarray) -> np.ndarray:
    return moments[4, :] / moments[3, :]


def d43_diagnostics(result, initial_moments, rate, f_max):
    n_t = result.time.size
    d43_ref = np.zeros(n_t)
    target_ref = np.zeros(n_t)
    ratio = np.zeros(n_t)
    for i in range(n_t):
        reference = coalescence_reference(result.time[i], initial_moments, rate)
        d43_ref[i] = reference[4] / reference[3]
        target_ref[i] = f_max * d43_ref[i]
        ratio[i] = target_ref[i] / result.pivots[-1, i]
    return d43_ref, target_ref, ratio


def main() -> None:
    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 10.0)
    base.num_classes = 20
    base.num_moments = 6
    base.grid_ratio = 1.3
    base.diameter_min = 0.0
    base.diameter_max = 1.0
    base.f_max = 2.5
    base.process.rate = 100.0

    fixed_opts = base.copy()
    fixed_opts.adaptive = False
    print("Running fixed-grid coalescence ...")
    fixed = adaptivehmmc.solve(fixed_opts)

    adaptive_factor = 1.4
    adaptive_opts = base.copy()
    adaptive_opts.adaptive = True
    adaptive_opts.f_mult = adaptive_factor
    print(f"Running adaptive coalescence (f_mult={adaptive_factor}) ...")
    adaptive = adaptivehmmc.solve(adaptive_opts)

    initial_moments = fixed.moments[:, 0]
    rate = base.process.rate
    fixed_moments = nonnegative_moment_history(fixed, base.num_moments)
    adaptive_moments = nonnegative_moment_history(adaptive, base.num_moments)
    fixed_d43 = d43_from_history(fixed_moments)
    adaptive_d43 = d43_from_history(adaptive_moments)

    _, _, ratio_fixed = d43_diagnostics(
        fixed, initial_moments, rate, base.f_max
    )

    hit_idx = np.argmax(ratio_fixed >= 1.0)
    hit_time = float(fixed.time[hit_idx]) if ratio_fixed[hit_idx] >= 1.0 else float("nan")

    time_reference = np.linspace(base.time_span[0], base.time_span[-1], 240)
    reference = np.column_stack([
        coalescence_reference(t, initial_moments, rate) for t in time_reference
    ])
    reference_d43 = reference[4, :] / reference[3, :]

    fig1, (ax_mom, ax_grid) = plt.subplots(2, 1, figsize=(5.0, 5.8))
    colors = plt.get_cmap("tab10")
    marker_idx = np.unique(
        np.round(np.linspace(0, adaptive.time.size - 1, 30)).astype(int)
    )

    for i in range(base.num_moments):
        ax_mom.plot(
            time_reference,
            reference[i, :] / initial_moments[i],
            "-",
            color=colors(i),
            lw=1.2,
            label=f"$M_{i}$",
        )
        ax_mom.plot(
            fixed.time,
            fixed_moments[i, :] / initial_moments[i],
            "--",
            color=colors(i),
            lw=0.9,
        )
        ax_mom.plot(
            adaptive.time[marker_idx],
            adaptive_moments[i, marker_idx] / initial_moments[i],
            "o",
            color=colors(i),
            markerfacecolor="white",
            markersize=2.8,
            lw=0.7,
        )

    ax_mom.set_xlabel("Time (s)")
    ax_mom.set_ylabel(r"$M_q / M_q(0)$")
    ax_mom.set_yscale("log")
    ax_mom.set_ylim(bottom=1.0e-3)
    ax_mom.grid(True, which="major", ls=":", alpha=0.6)
    leg1 = ax_mom.legend(loc="upper left", ncol=3, fontsize=7, title="moment")
    style_handles = [
        plt.Line2D([], [], color="k", ls="-", lw=1.2, label="Analytical"),
        plt.Line2D([], [], color="k", ls="--", lw=0.9, label="Fixed grid"),
        plt.Line2D([], [], color="k", marker="o", ls="None",
                   markerfacecolor="white", markersize=4, label="Adaptive grid"),
    ]
    ax_mom.add_artist(leg1)
    ax_mom.legend(handles=style_handles, loc="lower right", fontsize=7)

    ax_grid.plot(time_reference, reference_d43, "-", color="black", lw=1.2,
                 label=r"analytical $d_{43}$")
    ax_grid.plot(fixed.time, fixed_d43, "--", color="#8c1f1f", lw=1.0,
                 label=r"fixed $d_{43}$")
    ax_grid.plot(adaptive.time, adaptive_d43, "-", color="#1a619c", lw=1.1,
                 label=r"adaptive $d_{43}$")
    #ax_grid.plot(time_reference, base.f_max * reference_d43, ":",
    #             color="#007358", lw=1.2, label=r"$f_{\max} d_{43}^{\mathrm{ref}}$")
    ax_grid.axhline(fixed.pivots[-1, 0], ls="-.", color="0.55", lw=0.9,
                    label=r"fixed $d_N$")
    ax_grid.step(adaptive.time, adaptive.pivots[-1, :], where="post",
                 color="#007358", lw=1.1, label=r"adaptive $d_N$")
    for t_event in adaptive.adaptation_times:
        ax_grid.axvline(t_event, ls=":", color="0.75", lw=0.7)
    if np.isfinite(hit_time):
        ax_grid.axvline(hit_time, ls=":", color="0.25", lw=1.2)
    ax_grid.set_xlabel("Time (s)")
    ax_grid.set_ylabel(r"Diameter, $d$ (mm)")
    ax_grid.grid(True, ls=":", alpha=0.6)
    ax_grid.legend(loc="upper left", fontsize=7)

    out = output_dir()
    fig1.tight_layout()
    fig1.savefig(out / "coalescence_benchmark.pdf", bbox_inches="tight")
    fig1.savefig(out / "coalescence_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    print(f"Saved coalescence_benchmark to {out}")


if __name__ == "__main__":
    main()
