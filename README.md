# Adaptive HMMC for PBMs (Python)

`adaptivehmmc` is a Python package for solving moment-conserving sectional
population balance equations with an optional uniform adaptive-grid
rescaling. The number of size classes is fixed; the pivot grid is uniformly
rescaled when the active size range moves, and the population is projected
onto the rescaled pivots by local moment matching.

Supported processes:

| process       | kernel                  | notes                                              |
|---------------|-------------------------|----------------------------------------------------|
| `coalescence` | `constant`              | constant-kernel coalescence; analytical reference  |
| `breakage`    | `binary_equal_volume`   | two daughters at `d / 2^(1/3)`; scale-similar table |
| `growth`      | `proportional`          | `G(d) = k_g d`; fixed-step projection solver       |

## Installation

```bash
pip install .
```

For development and to run the examples and tests:

```bash
pip install -e .[dev]
```

Requirements: Python >= 3.9, NumPy >= 1.22, SciPy >= 1.9. Matplotlib is needed
only for the example scripts; pytest only for the test suite.

## Quick start

```python
import adaptivehmmc

options = adaptivehmmc.default_options()
options.time_span = (0.0, 1000.0)
options.num_classes = 20
options.adaptive = True

result = adaptivehmmc.solve(options)

print(f"Final d43: {result.d43[-1]:.4g}")
print(f"Adaptations: {result.report.num_adaptations}")
```

The result is a `SolveResult` dataclass with NumPy arrays:

| attribute            | shape                | meaning                                   |
|----------------------|----------------------|-------------------------------------------|
| `time`               | `(T,)`               | time samples (every solver step)          |
| `population`         | `(num_classes, T)`   | class populations                         |
| `pivots`             | `(num_classes, T)`   | pivot diameters                           |
| `moments`            | `(num_moments, T)`   | diameter moments `M_0 ... M_{num_moments-1}` |
| `d43`                | `(T,)`               | volume-weighted mean diameter `M_4 / M_3` |
| `target_upper_pivot` | `(T,)`               | adaptive target `f_max * d43`             |
| `adaptation_times`   | `(num_events,)`      | times where the grid was rescaled         |
| `report`             | `SolveReport`        | scalar diagnostics (CPU, step counts)     |
| `options`            | `SolverOptions`      | validated options used for the run        |

## Choosing a process

```python
options = adaptivehmmc.default_options()

# Coalescence (default)
options.process.type = "coalescence"
options.process.kernel = "constant"
options.process.rate = 1.0

# Breakage
options.process.type = "breakage"
options.process.kernel = "binary_equal_volume"

# Proportional growth
options.process.type = "growth"
options.process.kernel = "proportional"
options.solver.num_steps = 500
```

For breakage the manuscript uses a log-normal initial population on a custom
geometric grid; `examples/figure_breakage_benchmark_three_panel.py` shows how
to pass this through `options.initial_distribution`. The McCabe delta-L growth
benchmark in `examples/figure_growth_benchmark.py` is implemented as a
standalone manuscript example because this translated-characteristic case is
not part of the public `adaptivehmmc.solve` API.

## Method overview

The solver represents the number distribution on a finite set of diameter
pivots. Coalescence products and breakage daughters are mapped to nearby
pivots by local moment matching with an arbitrary preserved moment set
(default `M_0 ... M_5`). When adaptation is enabled the largest pivot target is

```
d_max_star = f_max * d43
```

and the grid is uniformly rescaled when this target leaves the admissible
band defined by `f_mult`. After each scaling event the existing population
is projected onto the new pivots using the same local moment-matching idea.
Uniform scaling is used deliberately: it preserves the relative pivot
layout, so the precomputed redistribution tables stay valid across
adaptation events.

## Reproducing the manuscript figures and tables

Each generated numerical manuscript artifact has a dedicated example script
in `examples/`. Outputs are written to `figures_out/` (PDFs and PNGs) and
`figures_out/tables/` (LaTeX tables). Static descriptive and parameter tables
remain in the manuscript source.

| Manuscript artifact | Script |
|---|---|
| Fig. `bad_grid_example` | `examples/figure_bad_grid_example.py` |
| Fig. `redistribution_example` | `examples/figure_redistribution_example.py` |
| Figs. `psd_superiority_{coalescence,breakage,growth}_three_panel_validation` | `examples/figure_breakage_benchmark_three_panel.py` |
| Figs. `nucleation_three_panel_validation`, `nucleation_growth_three_panel_validation` | `examples/figure_nucleation_benchmark_three_panel.py` |
| Figs. `coupled_coalescence_{breakage,growth}_three_panel_validation` | `examples/figure_coupled_benchmark_three_panel.py` |
| Fig. `alopaeus_coalescence_breakage_three_panel_validation` | `examples/figure_alopaeus_coalescence_breakage_three_panel.py` |
| Fig. + Table `benchmark_tradeoff` | `examples/figure_benchmark_tradeoff.py` |
| Table `category_comparison` | `examples/table_category_comparison.py` |
| Table `fmax_sensitivity` | `examples/table_fmax_sensitivity.py` |
| Table `analytical_benchmark_cdf_error` | `examples/table_analytical_benchmark_cdf_error.py` |

To regenerate every manuscript artifact:

```bash
python examples/reproduce_manuscript.py
```

The runner adds the local `src/` directory to `PYTHONPATH`, so it works from a
fresh checkout before installation. Individual example scripts can also be run
directly after `pip install -e .[dev]`. The small helper `_paths.py` simply
locates the `figures_out/` output directory.

Additional validation and exploratory scripts are retained under
`examples/misc/`. The Coulaloglou--Tavlarides physical-kernel comparison is
available as `examples/figure_ct_coalescence_breakage_three_panel.py`.

## Tests

```bash
pytest
```

The suite checks package loading, volume-moment conservation, that the
adaptive grid beats a narrow fixed grid, moment preservation across
redistribution events, invalid-option handling, and end-to-end accuracy of
the breakage and growth benchmarks.

## Citation

If you use this software, its examples, or results produced with it, please
cite the accompanying publication:

> Ville Alopaeus and Mahdi Mousavi (2026). "An Adaptive-Grid Strategy for
> HMMC Population Balance Equations." *Chemical Engineering Science*, article
> 125217. <https://doi.org/10.1016/j.ces.2026.125217>

BibTeX:

```bibtex
@article{Alopaeus2026AdaptiveGrid,
  author  = {Alopaeus, Ville and Mousavi, Mahdi},
  title   = {An Adaptive-Grid Strategy for {HMMC} Population Balance Equations},
  journal = {Chemical Engineering Science},
  year    = {2026},
  pages   = {125217},
  doi     = {10.1016/j.ces.2026.125217},
  url     = {https://doi.org/10.1016/j.ces.2026.125217}
}
```

GitHub's **Cite this repository** function uses the matching metadata in
`CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE`.
