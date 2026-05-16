"""Reproduce manuscript Figure: redistribution_example.

Schematic illustration of the adaptive redistribution step. The pivot set is
uniformly rescaled and the population of one source class is projected onto a
nearby stencil on the scaled grid by local moment matching. Positions and
weights are chosen for visual clarity; the redistribution mechanism itself is
what the package implements internally.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _paths import output_dir


def main() -> None:
    old_grid = np.array([1.0, 1.45, 2.10, 3.05, 4.40, 6.35])
    scale = 1.22
    new_grid = scale * old_grid

    source_idx = 3
    source_x = old_grid[source_idx]
    source_y = 1.0
    stencil_idx = np.array([1, 2, 3])
    weights = np.array([0.18, 0.64, 0.18])

    old_color = "#37474f"
    new_color = "#1b5e20"
    source_color = "#c62828"
    accent = "#ef6c00"
    muted = "#b0bec5"

    fig, axes = plt.subplots(
        2, 1,
        figsize=(6.2, 4.3),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.35]},
    )
    fig.subplots_adjust(hspace=0.28)

    ax = axes[0]
    y_old = 0.58
    y_new = 0.18
    ax.hlines(y_old, old_grid.min() - 0.35, old_grid.max() + 0.35, color=muted, lw=1.0)
    ax.hlines(y_new, new_grid.min() - 0.35, new_grid.max() + 0.35, color=muted, lw=1.0)
    ax.plot(old_grid, np.full_like(old_grid, y_old), "o", ms=5.8, color=old_color)
    ax.plot(new_grid, np.full_like(new_grid, y_new), "s", ms=5.8, color=new_color)
    ax.plot(source_x, y_old, "o", ms=8.0, color=source_color, zorder=4)
    ax.annotate(
        "",
        xy=(new_grid[-1], 0.83),
        xytext=(old_grid[-1], 0.83),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=accent),
    )
    ax.text(0.02, y_old + 0.10, "old pivots",
            transform=ax.get_yaxis_transform(), color=old_color, fontsize=9.5)
    ax.text(0.02, y_new - 0.17, "scaled pivots",
            transform=ax.get_yaxis_transform(), color=new_color, fontsize=9.5)
    ax.text(0.52, 0.87,
            r"uniform scaling:  $\tilde{d}_i=\lambda d_i$",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=10.5, color=accent)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)

    ax = axes[1]
    ax.vlines(old_grid, 0.0, 0.18, color=muted, lw=1.0, alpha=0.75)
    ax.plot(old_grid, np.full_like(old_grid, 0.18), "o", ms=4.8, color=muted, alpha=0.75)
    ax.vlines(source_x, 0.0, source_y, color=source_color, lw=2.2)
    ax.plot(source_x, source_y, "o", ms=8.0, color=source_color, zorder=5)

    ax.vlines(new_grid, 0.0, 0.12, color=muted, lw=1.0, alpha=0.75)
    ax.plot(new_grid, np.full_like(new_grid, 0.12), "s", ms=4.8, color=muted, alpha=0.75)
    ax.vlines(new_grid[stencil_idx], 0.0, weights, color=new_color, lw=2.2)
    ax.plot(new_grid[stencil_idx], weights, "s", ms=7.0, color=new_color, zorder=5)

    for idx, weight in zip(stencil_idx, weights):
        ax.annotate(
            "",
            xy=(new_grid[idx], weight + 0.04),
            xytext=(source_x, source_y - 0.06),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#78909c", alpha=0.95),
        )
        ax.text(new_grid[idx], weight + 0.08,
                rf"$\tilde{{w}}={weight:.2f}$",
                ha="center", fontsize=8.8, color=new_color)

    ax.text(source_x, source_y + 0.10, r"one old class $Y_s$",
            ha="center", color=source_color, fontsize=10)
    ax.text(new_grid[stencil_idx].mean(), 0.88,
            "local moment-matching stencil",
            ha="center", color=new_color, fontsize=10)
    ax.annotate(
        "",
        xy=(new_grid[stencil_idx[-1]], 0.80),
        xytext=(new_grid[stencil_idx[0]], 0.80),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color=new_color),
    )

    ax.set_xlabel("Pivot diameter", fontsize=10.5)
    ax.set_ylabel("Class population", fontsize=10.5)
    ax.set_xlim(old_grid.min() - 0.45, new_grid.max() + 0.55)
    ax.set_ylim(-0.03, 1.18)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=9)

    out = output_dir()
    fig.savefig(out / "redistribution_example.pdf", bbox_inches="tight")
    fig.savefig(out / "redistribution_example.png", dpi=300, bbox_inches="tight")
    print(f"Saved {out}/redistribution_example.{{pdf,png}}")


if __name__ == "__main__":
    main()
