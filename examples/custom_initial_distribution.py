"""Solve coalescence with a user-defined pivot grid and initial population."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import adaptivehmmc
from _paths import output_dir


def main() -> None:
    pivots = np.linspace(0.05, 1.0, 16)
    population = np.exp(-0.5 * ((pivots - 0.25) / 0.08) ** 2)
    population = population / population.sum()

    options = adaptivehmmc.default_options()
    options.time_span = (0.0, 200.0)
    options.num_moments = 6
    options.adaptive = True
    options.initial_distribution.type = "custom"
    options.initial_distribution.pivots = pivots
    options.initial_distribution.population = population

    result = adaptivehmmc.solve(options)

    print(f"Custom case final largest pivot: {result.pivots[-1, -1]:.6g}")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.stem(result.pivots[:, 0], result.population[:, 0],
            linefmt="k:", markerfmt="ko", basefmt=" ", label="Initial")
    ax.stem(result.pivots[:, -1], result.population[:, -1],
            linefmt="k-", markerfmt="ks", basefmt=" ", label="Final")
    ax.set_xlabel("Diameter")
    ax.set_ylabel("Class population")
    ax.set_title("Custom initial distribution")
    ax.legend()

    out = output_dir() / "custom_initial_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
