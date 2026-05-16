"""Reproduce manuscript Figure: bad_grid_example.

A log-normal droplet distribution discretized on a fixed geometric grid whose
upper boundary lies below the active size range. Mass beyond ``d_max``
accumulates in the last category, producing a volumetric pile-up at the
largest pivot. Pure illustration; no solver call.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from _paths import output_dir


def main() -> None:
    N = 10
    d_med = 300e-6
    sigma = 0.55

    def npdf(d):
        return 1.0 / (d * sigma * np.sqrt(2 * np.pi)) * \
            np.exp(-0.5 * ((np.log(d) - np.log(d_med)) / sigma) ** 2)

    dmin = 60e-6
    dmax = 400e-6
    x_range = (40, 600)
    y_max = 0.85

    r = (dmax / dmin) ** (1.0 / (N - 1))
    d = dmin * r ** np.arange(N)
    dlo = d / np.sqrt(r)
    dhi = d * np.sqrt(r)

    Y = np.array([
        quad(npdf, max(lo, 1e-10), hi, epsabs=1e-14, epsrel=1e-8)[0]
        for lo, hi in zip(dlo, dhi)
    ])
    Y[-1] += quad(npdf, dhi[-1], 100 * dmax, epsabs=1e-14, epsrel=1e-8)[0]

    vol = (Y * d ** 3) / np.sum(Y * d ** 3)

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for i in range(N):
        ax.bar(
            dlo[i] * 1e6,
            vol[i],
            width=(dhi[i] - dlo[i]) * 1e6,
            align="edge",
            color="black",
            edgecolor="white",
            linewidth=0.6,
        )
    ax.bar(
        dlo[-1] * 1e6,
        vol[-1],
        width=(dhi[-1] - dlo[-1]) * 1e6,
        align="edge",
        fill=False,
        edgecolor="black",
        linewidth=1.6,
    )
    ax.plot([dmin * 1e6, dmin * 1e6], [0, y_max], "k--", lw=0.6)
    ax.plot([dmax * 1e6, dmax * 1e6], [0, y_max], "k--", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(*x_range)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"Diameter ($\mu$m)", fontsize=9)
    ax.set_ylabel(r"Volume fraction, $Y_i d_i^3 / M_3$", fontsize=9)
    ax.tick_params(labelsize=8)

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "bad_grid_example.pdf", bbox_inches="tight")
    fig.savefig(out / "bad_grid_example.png", dpi=300, bbox_inches="tight")
    print(f"Saved {out}/bad_grid_example.{{pdf,png}}")


if __name__ == "__main__":
    main()
