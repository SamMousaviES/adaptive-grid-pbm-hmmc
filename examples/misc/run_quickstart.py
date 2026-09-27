"""Basic adaptive-grid coalescence example."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "examples")]

import adaptivehmmc
from _paths import output_dir


def main() -> None:
    options = adaptivehmmc.default_options()
    options.time_span = (0.0, 300.0)
    options.num_classes = 20
    options.adaptive = True

    result = adaptivehmmc.solve(options)

    print(f"Final d43:           {result.d43[-1]:.6g}")
    print(f"Adaptation events:   {result.report.num_adaptations}")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(result.time, result.d43, "k-", lw=1.5, label=r"$d_{43}$")
    ax.plot(result.time, result.pivots[-1, :], "k--", lw=1.2,
            label="largest pivot")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Diameter, $d$ [arb. units]")
    ax.set_title("Adaptive HMMC quickstart")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="upper left")

    out = output_dir() / "run_quickstart.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
