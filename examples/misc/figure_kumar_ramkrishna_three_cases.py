"""Three literature aggregation tests from Kumar and Ramkrishna (1995)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples"))

import figure_kumar_ramkrishna_sum_kernel as base  # noqa: E402


RATE = 5.0
V0 = 1.0e-2
FIGS = REPO / "figures_out"


def configure_case(name: str, final_time: float, ratio: float):
    """Configure the shared HMMC runner in the volume coordinate of the paper."""
    base.FINAL_TIME = final_time
    base.SUM_KERNEL_RATE = RATE
    base.PUBLISHED_NUMBER_RATIO = ratio
    if name == "Constant kernel":
        base.sum_kernel = lambda pivots: np.full((pivots.size, pivots.size), RATE)
    elif name == "Sum kernel":
        base.sum_kernel = lambda pivots: RATE * (pivots[:, None] ** 3 + pivots[None, :] ** 3)
    elif name == "Product kernel":
        base.sum_kernel = lambda pivots: RATE * np.outer(pivots**3, pivots**3)
    else:
        raise ValueError(name)


def run_case(name: str, final_time: float, ratio: float, grid_ratio: float):
    configure_case(name, final_time, ratio)
    if name == "Constant kernel":
        num_classes, volume_min = 52, 1.0e-5
    elif name == "Sum kernel":
        num_classes, volume_min = 52, 4.0e-10
    else:
        num_classes, volume_min = 24, 1.0e-5
    pivots, edges = base.volume_geometric_pivots(num_classes, volume_min, grid_ratio)
    population = base.exponential_population(edges, V0)

    adaptive_options = base.options_for(pivots, population, adaptive=True)
    adaptive_options.f_mult = 1.8
    adaptive_options.solver.max_step = 10.0
    adaptive = base.run_sum_kernel(adaptive_options)
    return adaptive


def constant_oversize(volumes: np.ndarray, final_number_ratio: float) -> np.ndarray:
    return final_number_ratio * np.exp(-final_number_ratio * volumes / V0)


def digitized_product_analytical() -> tuple[np.ndarray, np.ndarray]:
    """Solid analytical curve in Kumar and Ramkrishna Fig. 5, digitized from the PDF."""
    pixels = np.array(
        [
            [0.0, 445.0], [40.0, 464.0], [80.0, 489.0], [120.0, 517.0],
            [160.0, 565.0], [180.0, 594.0], [200.0, 626.0], [220.0, 670.0],
            [240.0, 718.0], [260.0, 786.0], [280.0, 856.0], [300.0, 948.0],
            [320.0, 1050.0], [340.0, 1186.0], [360.0, 1277.0],
        ]
    )
    log_volume = -1.0 + pixels[:, 0] / 226.0
    log_oversize = (383.0 - pixels[:, 1]) / 45.0
    return 10.0**log_volume, 10.0**log_oversize


def kumar_ramkrishna_moving_pivot(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Digitized Kumar--Ramkrishna moving-pivot values supplied for Figs. 3--5."""
    if name == "Constant kernel":
        values = np.array(
            [
                [1.0284075923927483, 0.014204884343026846],
                [1.7776346718366873, 0.011192032080403258],
                [2.8781274008180238, 0.007577835901531032],
                [4.610860416819193, 0.003993796124122047],
                [8.524000744600595, 0.0010476849528008039],
                [12.524073371953246, 2.7346820474930287e-4],
                [17.049770036830303, 5.833191724170737e-5],
                [21.997481005719933, 6.166672896065145e-6],
                [26.576250948264516, 6.844069440484578e-7],
                [32.11725392153662, 6.217618547376269e-8],
                [37.549817447466154, 6.895690449402578e-9],
                [41.558805756864096, 8.442906926490484e-10],
                [49.698489441396056, 5.678944922601486e-11],
                [54.441088734734365, 4.4302936972406234e-12],
                [63.051639601713795, 1.717019924740158e-13],
                [69.17204290313329, 4.6820128408687765e-15],
                [78.40688448887798, 1.4121362175066987e-16],
                [86.01781669795396, 3.850648328628328e-18],
                [93.23446591583699, 2.3383157945086133e-19],
                [98.8204433751507, 2.014814772669878e-20],
            ]
        )
        return values[:, 0], values[:, 1], r"Kumar--Ramkrishna moving pivot, $r=1.4$"
    if name == "Sum kernel":
        values = np.array(
            [
                [0.1062366565852201, 0.08555167225579544],
                [0.20223895011640378, 0.05212511969222633],
                [0.4448705516365457, 0.02693511901324116],
                [0.7046704106420016, 0.015769481062355678],
                [1.1920547801619092, 0.00783379397860714],
                [1.8884673546640574, 0.003586812890932603],
                [3.0319185309680186, 0.0011357129537852854],
                [4.619047215991081, 3.0534381798662202e-4],
                [6.590383131347541, 6.971146461521435e-5],
                [7.821477771226336, 2.714003395196424e-5],
                [10.731453160747419, 3.790557534555214e-6],
                [12.740571627766693, 7.981866409687263e-7],
                [15.12512582873489, 1.8242868421991428e-7],
                [17.264320792464083, 3.3979345464846384e-8],
                [19.705608467259207, 6.593720691211815e-9],
                [22.793700156493404, 9.218558584174563e-10],
                [25.337695513297977, 2.4831070908180796e-10],
                [27.800102604589878, 5.915354070700481e-11],
                [30.105974457927765, 1.2462894095236722e-11],
                [32.608439675168576, 1.9710674345010693e-12],
                [33.94416214555273, 5.311327168341115e-13],
                [36.27581677260702, 1.3735463347754444e-13],
                [37.754705503087116, 5.136817325070879e-14],
                [39.82541828342133, 1.1277000306952625e-14],
                [43.14787946571043, 1.0908180517478213e-15],
            ]
        )
        return values[:, 0], values[:, 1], r"Kumar--Ramkrishna moving pivot, $r=1.2$"

    values = np.array(
        [
            [0.1079754239580052, 0.03131142411808953],
            [0.18542008769743765, 0.007471723332559836],
            [0.39772729288697295, 4.715774804920915e-4],
            [0.5824096796378505, 6.9524342603195e-5],
            [0.7047283341962567, 2.1569291798364832e-5],
            [0.86174847491319, 4.8607001936768675e-6],
            [1.0207549428038072, 1.1549149505151934e-6],
            [1.2090610665399335, 2.466633986187373e-7],
            [1.3297395417900817, 7.644052715736188e-8],
            [1.43184746156137, 2.9310881773643414e-8],
            [1.6255122755967462, 6.257058907318986e-9],
            [1.8067396559345148, 1.6527145541132224e-9],
            [1.965971266303437, 4.1378100982978525e-10],
            [2.207924033908543, 4.913919562953569e-11],
            [2.4277441230853563, 7.220574928748023e-12],
            [2.669318284101839, 9.042131602573003e-13],
            [2.9350746118745077, 1.3286621385543421e-13],
            [3.0932270308988747, 2.0582265601560103e-14],
            [3.510337239739049, 1.360108228562967e-15],
            [3.699426307733994, 1.997578505598528e-16],
            [3.940806619306978, 4.4944176259194867e-17],
            [4.023289532056125, 8.613605051052308e-18],
            [4.285169708736006, 1.199555784632204e-18],
            [4.710565265487344, 7.512638611022342e-20],
            [4.809553351330411, 1.8795123335393984e-20],
        ]
    )
    return values[:, 0], values[:, 1], r"Kumar--Ramkrishna moving pivot, $r=1.2$"


def main() -> None:
    cases = [
        ("Constant kernel", 100.0, 4.0e-3, 1.4, 1.0e4),
        ("Sum kernel", 40.0, 0.135, 1.5, 1.0e4),
        ("Product kernel", 500.0, 0.875, 1.5, 1.0e3),
    ]
    results = []
    for name, final_time, ratio, grid_ratio, volume_max in cases:
        print(f"Running {name}...")
        adaptive = run_case(name, final_time, ratio, grid_ratio)
        results.append((name, final_time, ratio, adaptive, volume_max))

    fig, axes = plt.subplots(3, 1, figsize=(5.0, 8.2), sharex=False)
    colors = {"adaptive": "#1a619c", "reference": "black"}
    for axis, (name, final_time, ratio, adaptive, volume_max) in zip(axes, results):
        plot_volumes = np.geomspace(1.0e-4, volume_max, 1200)
        if name == "Constant kernel":
            reference_volumes = plot_volumes
            reference = constant_oversize(reference_volumes, ratio)
            reference_label = "Analytical"
        elif name == "Sum kernel":
            configure_case(name, final_time, ratio)
            reference_volumes = plot_volumes
            reference = base.analytical_oversize_number(reference_volumes)
            reference_label = "Analytical"
        else:
            reference_volumes, reference = digitized_product_analytical()
            reference_label = "Kumar--Ramkrishna analytical (digitized)"

        axis.semilogx(reference_volumes, reference, "-", color=colors["reference"], lw=1.3, label=reference_label)
        kumar_volume, kumar_oversize, kumar_label = kumar_ramkrishna_moving_pivot(name)
        axis.plot(
            kumar_volume,
            kumar_oversize,
            "-s",
            color="#6a3d9a",
            markerfacecolor="white",
            markersize=3.6,
            label=kumar_label,
        )
        axis.step(plot_volumes, base.oversize_number(adaptive, plot_volumes), where="post", color=colors["adaptive"], lw=1.1, label="Adaptive HMMC")
        adaptive_volumes = adaptive.pivots[:, -1] ** 3
        adaptive_oversize = base.oversize_number(adaptive, adaptive_volumes)
        positive = adaptive_oversize > 0.0
        axis.plot(
            adaptive_volumes[positive],
            adaptive_oversize[positive],
            "o",
            color=colors["adaptive"],
            markerfacecolor="white",
            markersize=2.8,
        )
        axis.set_yscale("log")
        axis.set_ylim(1.0e-15, 1.1)
        axis.set_title(f"{name}, $t={final_time:g}$")
        axis.set_ylabel(r"Number of particles larger than $v$")
        axis.grid(True, which="major", ls=":", alpha=0.6)
        axis.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel(r"Particle volume, $v$")
    fig.tight_layout(h_pad=0.8)
    FIGS.mkdir(parents=True, exist_ok=True)
    output = FIGS / "kumar_ramkrishna_three_kernel_oversize.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    assert all(np.all(np.isfinite(result[3].moments)) for result in results)
    print(f"Saved {output}")
    for name, _time, ratio, adaptive, _volume_max in results:
        print(
            f"{name}: adaptive N/N0={adaptive.moments[0, -1] / adaptive.moments[0, 0]:.4f}, "
            f"target={ratio:.4f}, adaptations={adaptive.adaptation_times.size}"
        )


if __name__ == "__main__":
    main()
