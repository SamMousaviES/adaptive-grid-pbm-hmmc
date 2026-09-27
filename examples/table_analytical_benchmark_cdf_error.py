"""Reproduce the seven-case final volume-CDF error summary table."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "examples")]

from figure_breakage_benchmark_three_panel import (  # noqa: E402
    TABLES,
    breakage_case,
    build_rows_and_curves,
    coalescence_case,
    growth_case,
)
from figure_coupled_benchmark_three_panel import (  # noqa: E402
    coalescence_breakage_case,
    coalescence_growth_case,
)
from figure_nucleation_benchmark_three_panel import (  # noqa: E402
    Parameters,
    curve_and_diagnostics,
    make_case,
)


def psd_row(label, case):
    row = build_rows_and_curves([case])[0][0]
    return label, row["fixed_max"], row["adaptive_max"]


def nucleation_row(label, parameters):
    _, report = curve_and_diagnostics(make_case(parameters), parameters)
    return (
        label,
        report["fixed"]["max_volume_cdf_error"],
        report["adaptive"]["max_volume_cdf_error"],
    )


def benchmark_rows():
    pure_nucleation = replace(
        Parameters(),
        grid_ratio=1.08,
        nuclei_size=0.08,
        growth_rate=0.0,
        num_steps=100,
        f_max=2.5,
        f_mult=1.3,
    )
    return [
        psd_row("Coalescence", coalescence_case()),
        psd_row("Breakage", breakage_case()),
        psd_row("Growth", growth_case()),
        nucleation_row("Nucleation", pure_nucleation),
        psd_row("Coalescence--breakage", coalescence_breakage_case()),
        nucleation_row("Nucleation--growth", Parameters()),
        psd_row("Coalescence--growth", coalescence_growth_case()),
    ]


def write_table(rows):
    body = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Case & Fixed [\%] & Adaptive [\%] & Ratio \\",
        r"\midrule",
    ]
    for label, fixed_error, adaptive_error in rows:
        body.append(
            f"{label} & {100.0 * fixed_error:.2f} & "
            f"{100.0 * adaptive_error:.2f} & "
            f"{fixed_error / adaptive_error:.2f} \\\\"
        )
    body.extend([r"\bottomrule", r"\end{tabular}"])
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "analytical_benchmark_cdf_error.tex"
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def main():
    rows = benchmark_rows()
    path = write_table(rows)
    for label, fixed_error, adaptive_error in rows:
        print(
            f"{label}: fixed={100.0 * fixed_error:.2f}%, "
            f"adaptive={100.0 * adaptive_error:.2f}%, "
            f"ratio={fixed_error / adaptive_error:.2f}"
        )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
