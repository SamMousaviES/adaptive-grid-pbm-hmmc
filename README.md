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

Requirements: Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.9. Matplotlib is needed
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

For breakage and growth the manuscript uses a log-normal initial population
on a custom geometric grid; the `examples/figure_breakage_benchmark.py` and
`examples/figure_growth_benchmark.py` scripts show how to pass these in via
`options.initial_distribution`.

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

## Reproducing the manuscript figures

Each manuscript figure has a dedicated example script in `examples/`.
Outputs are written to `figures_out/` (PDFs and PNGs) and
`figures_out/tables/` (LaTeX tables).

| Manuscript artifact                                       | Script                                       |
|-----------------------------------------------------------|----------------------------------------------|
| Fig. `bad_grid_example`                                   | `examples/figure_bad_grid_example.py`        |
| Fig. `redistribution_example`                             | `examples/figure_redistribution_example.py`  |
| Figs. `wall_error_time`, `wall_distribution_snapshots`    | `examples/figure_wall_error.py`              |
| Figs. `breakage_moment_validation`, `breakage_grid_error` | `examples/figure_breakage_benchmark.py`      |
| Fig. `growth_moment_validation`                           | `examples/figure_growth_benchmark.py`        |
| Fig. + Table `benchmark_tradeoff`                         | `examples/figure_benchmark_tradeoff.py`      |
| Table `category_comparison`                               | `examples/table_category_comparison.py`      |

To regenerate everything:

```bash
cd examples
python figure_bad_grid_example.py
python figure_redistribution_example.py
python figure_wall_error.py
python figure_breakage_benchmark.py
python figure_growth_benchmark.py
python figure_benchmark_tradeoff.py
python table_category_comparison.py
```

The example scripts import `adaptivehmmc` from the installed package; the
small helper `_paths.py` simply locates the `figures_out/` output directory.

## Tests

```bash
pytest
```

The suite checks package loading, volume-moment conservation, that the
adaptive grid beats a narrow fixed grid, moment preservation across
redistribution events, invalid-option handling, and end-to-end accuracy of
the breakage and growth benchmarks.

## Citation

If you use this package in academic work, please cite the accompanying
publication or use the metadata in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE`.
