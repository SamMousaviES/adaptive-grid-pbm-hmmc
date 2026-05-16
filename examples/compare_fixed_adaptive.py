"""Compare fixed and adaptive grids at equal class count for constant-kernel coalescence."""

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


def moment_error_history(result, final_reference, final_time, rate):
    initial_moments = result.moments[:, 0]
    errors = np.zeros(result.time.size)
    for i, t in enumerate(result.time):
        reference = coalescence_reference(t, initial_moments, rate)
        if t == final_time:
            reference = final_reference
        errors[i] = 100.0 * np.max(np.abs((result.moments[:, i] - reference) / reference))
    return errors


def main() -> None:
    base = adaptivehmmc.default_options()
    base.time_span = (0.0, 1000.0)
    base.num_classes = 20
    base.f_mult = 1.4

    fixed_opts = base.copy()
    fixed_opts.adaptive = False
    fixed = adaptivehmmc.solve(fixed_opts)

    adaptive_opts = base.copy()
    adaptive_opts.adaptive = True
    adaptive = adaptivehmmc.solve(adaptive_opts)

    reference = coalescence_reference(base.time_span[-1], fixed.moments[:, 0], base.process.rate)
    fixed_err = 100.0 * np.max(np.abs((fixed.moments[:, -1] - reference) / reference))
    adaptive_err = 100.0 * np.max(np.abs((adaptive.moments[:, -1] - reference) / reference))
    print(f"Fixed-grid max moment error:    {fixed_err:.3f} %")
    print(f"Adaptive-grid max moment error: {adaptive_err:.3f} %")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(fixed.time,
                moment_error_history(fixed, reference, base.time_span[-1], base.process.rate),
                "k--", lw=1.5, label="Fixed grid")
    ax.semilogy(adaptive.time,
                moment_error_history(adaptive, reference, base.time_span[-1], base.process.rate),
                "k-", lw=1.5, label="Adaptive grid")
    ax.set_xlabel("Time")
    ax.set_ylabel("Max. moment error [%]")
    ax.set_title("Fixed versus adaptive grid")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="upper left")

    out = output_dir() / "compare_fixed_adaptive.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
