# Adaptive HMMC for PBMs

Adaptive HMMC for PBMs is a small MATLAB package for solving constant-kernel
coalescence population balance models with a moment-conserving sectional grid.
The package keeps the number of size classes fixed and can uniformly rescale
the pivot grid when the active size range moves.

This distribution is a cleaned core-solver package intended for reuse and
publication. It does not include manuscript build files, exploratory figures,
or temporary research scripts.

## Requirements

- MATLAB R2020b or newer
- No required third-party toolboxes

## Installation

Clone or download the repository and add the package root to the MATLAB path:

```matlab
addpath("Git_dist")
```

If `Git_dist` is itself the repository root, add that folder:

```matlab
addpath(pwd)
```

## Quick Start

```matlab
addpath("Git_dist")

options = adaptivehmmc.defaultOptions();
options.timeSpan = [0, 1000];
options.numClasses = 20;
options.adaptive = true;

result = adaptivehmmc.solve(options);

fprintf("Final d43: %.4g\n", result.d43(end));
fprintf("Adaptation events: %d\n", result.report.numAdaptations);
```

The primary output fields are:

- `result.time`: time vector
- `result.population`: class populations, one column per saved time
- `result.pivots`: pivot locations, one column per saved time
- `result.moments`: transported moments
- `result.d43`: characteristic diameter `M4/M3`
- `result.targetUpperPivot`: adaptive target `fMax*d43`
- `result.adaptationTimes`: times where the grid was rescaled
- `result.report`: summary diagnostics

## Method Overview

The solver represents the number distribution on a finite set of diameter
pivots. Coalescence products are mapped to nearby pivots by local moment
matching. When adaptation is enabled, the largest pivot target is

```text
d_max_star = fMax * d43
```

and the grid is uniformly rescaled when this target leaves the admissible band
defined by `fMult`. After each scaling event, the existing population is
projected onto the new pivots using the same local moment-matching idea.

Uniform grid scaling is used deliberately: it preserves the relative pivot
layout and is compatible with reusable redistribution logic.

## Examples

Run from MATLAB:

```matlab
run("examples/run_quickstart.m")
run("examples/compare_fixed_adaptive.m")
run("examples/custom_initial_distribution.m")
```

## Tests

Run the test suite from the package root:

```matlab
runtests("tests")
```

The tests check package loading, example execution, volume conservation,
adaptive-grid improvement over a narrow fixed grid, moment preservation during
redistribution, invalid-option handling, and no-plot solver execution.

## Citation

If you use this package in academic work, cite the accompanying publication or
use the metadata in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE`.
