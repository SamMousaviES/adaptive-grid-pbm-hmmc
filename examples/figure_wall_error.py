"""Reproduce manuscript Figures wall_error_time and wall_distribution_snapshots.

Single fixed-grid run plus two adaptive runs at f_mult = 1.2 and 1.4 on the
constant-kernel coalescence benchmark with 20 categories and unit initial
diameter range. Reports max moment error vs analytical, the
d_max^star / d_N proximity ratio, and three fixed-grid distribution snapshots
straddling the wall hit.
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


def error_and_ratio(result, initial_moments, rate, f_max):
    n_t = result.time.size
    max_err = np.zeros(n_t)
    ratio = np.zeros(n_t)
    for i in range(n_t):
        reference = coalescence_reference(result.time[i], initial_moments, rate)
        rel = (result.moments[:, i] - reference) / reference
        max_err[i] = 100.0 * np.max(np.abs(rel))
        d43_ref = reference[4] / reference[3]
        ratio[i] = (f_max * d43_ref) / result.pivots[-1, i]
    return max_err, ratio


def main() -> None:
    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 1000.0)
    base.num_classes = 20
    base.num_moments = 6
    base.grid_ratio = 1.3
    base.diameter_min = 0.0
    base.diameter_max = 1.0
    base.f_max = 2.5

    fixed_opts = base.copy()
    fixed_opts.adaptive = False
    print("Running fixed-grid coalescence ...")
    fixed = adaptivehmmc.solve(fixed_opts)

    adaptive_factors = [1.2, 1.4]
    adaptive_runs = []
    for fm in adaptive_factors:
        opts = base.copy()
        opts.adaptive = True
        opts.f_mult = fm
        print(f"Running adaptive coalescence (f_mult={fm}) ...")
        adaptive_runs.append(adaptivehmmc.solve(opts))

    initial_moments = fixed.moments[:, 0]
    rate = base.process.rate

    err_fixed, ratio_fixed = error_and_ratio(fixed, initial_moments, rate, base.f_max)
    err_adaptive = []
    ratio_adaptive = []
    for r in adaptive_runs:
        e, ra = error_and_ratio(r, initial_moments, rate, base.f_max)
        err_adaptive.append(e)
        ratio_adaptive.append(ra)

    hit_idx = np.argmax(ratio_fixed >= 1.0)
    hit_time = float(fixed.time[hit_idx]) if ratio_fixed[hit_idx] >= 1.0 else float("nan")

    case_colors = ["#bf1f1f", "#1a7338", "#1f52b8"]
    case_styles = ["-", "-", "--"]
    case_names = ["Fixed grid"] + [f"Adaptive $f_{{mult}}={fm}$" for fm in adaptive_factors]

    fig1, axes = plt.subplots(2, 1, figsize=(7.5, 5.5))

    ax1 = axes[0]
    ax1.semilogy(fixed.time, err_fixed, case_styles[0], color=case_colors[0],
                 lw=1.8, label=case_names[0])
    for k, r in enumerate(adaptive_runs):
        ax1.semilogy(r.time, err_adaptive[k], case_styles[k + 1],
                     color=case_colors[k + 1], lw=1.8, label=case_names[k + 1])
    if np.isfinite(hit_time):
        ax1.axvline(hit_time, ls=":", color="0.25", lw=1.2)
    ax1.set_ylabel("Max. moment error [%]")
    ax1.set_title("Time evolution of moment error")
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2 = axes[1]
    ax2.plot(fixed.time, np.ones_like(fixed.time), "k-", lw=1.2)
    ax2.plot(fixed.time, ratio_fixed, case_styles[0], color=case_colors[0],
             lw=1.8, label=case_names[0])
    for k, r in enumerate(adaptive_runs):
        ax2.plot(r.time, ratio_adaptive[k], case_styles[k + 1],
                 color=case_colors[k + 1], lw=1.8, label=case_names[k + 1])
    if np.isfinite(hit_time):
        ax2.axvline(hit_time, ls=":", color="0.25", lw=1.2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"$d_{\max}^{\star,\mathrm{ref}} / d_N$")
    ax2.set_title("Boundary proximity relative to the reference target")
    ax2.grid(True, ls=":", alpha=0.6)

    out = output_dir()
    fig1.tight_layout()
    fig1.savefig(out / "wall_error_time.pdf", bbox_inches="tight")
    fig1.savefig(out / "wall_error_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    if np.isnan(hit_time):
        snapshot_times = [0.0, 0.65 * base.time_span[-1], base.time_span[-1]]
    else:
        snapshot_times = [max(0.0, hit_time - 100.0), hit_time, base.time_span[-1]]

    fixed_dmax = float(fixed.pivots[-1, 0])
    fig2, snap_axes = plt.subplots(1, len(snapshot_times),
                                   figsize=(3.0 * len(snapshot_times), 3.2))
    if len(snapshot_times) == 1:
        snap_axes = [snap_axes]

    for k, t_target in enumerate(snapshot_times):
        idx = int(np.argmin(np.abs(fixed.time - t_target)))
        sizes = fixed.pivots[:, idx]
        pop = fixed.population[:, idx]
        volumes = pop * sizes ** 3
        volume_fraction = volumes / volumes.sum()

        ax = snap_axes[k]
        markerline, stemlines, baseline = ax.stem(sizes, volume_fraction,
                                                  linefmt="k-", markerfmt="ko", basefmt=" ")
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=0.9)
        ax.plot(sizes[-1], volume_fraction[-1], "o",
                markersize=7, color="#bf1f1f")
        ax.axvline(fixed_dmax, ls="--", color="black", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Diameter")
        if k == 0:
            ax.set_ylabel("Volume fraction")
        ax.set_title(f"Fixed grid, t = {fixed.time[idx]:.0f}")
        ax.set_xlim(sizes.min() / np.sqrt(base.grid_ratio),
                    fixed_dmax * np.sqrt(base.grid_ratio))
        ax.set_ylim(0, max(0.05, 1.15 * float(volume_fraction.max())))
        ax.grid(True, which="both", axis="x", ls=":", alpha=0.6)

    fig2.tight_layout()
    fig2.savefig(out / "wall_distribution_snapshots.pdf", bbox_inches="tight")
    fig2.savefig(out / "wall_distribution_snapshots.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved wall_error_time and wall_distribution_snapshots to {out}")


if __name__ == "__main__":
    main()
