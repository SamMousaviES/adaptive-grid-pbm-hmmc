# Nucleation verification examples

Run from `Git_dist`:

```powershell
python examples/figure_nucleation_benchmark_three_panel.py --verify
```

The script creates two PDF/PNG pairs in `figures_out`, plus
`nucleation_benchmark_diagnostics.json`. It does not update the manuscript or
reviewer responses. Both figures reuse the manuscript's three-panel plotting
function: normalized moments, diameter/grid histories, and final volume CDF.

## Literature basis

The following local papers were consulted under `Misc/Literature/Adaptive Grid`:

- Hounslow, Ryall and Marshall (1988), *A discretized population balance for
  nucleation, growth, and aggregation*, AIChE Journal 34, 1821-1832. Equations
  (3), (10), (45)-(46) introduce a birth source, its moment balance, and the
  constant nucleation/constant growth verification problem. Their ideal source
  is a Dirac delta at zero size.
- Qamar et al. (2007), *Adaptive high-resolution schemes for multidimensional
  population balances in crystallization processes*, Computers & Chemical
  Engineering 31, 1296-1311. Equations (2.27)-(2.29) discuss the nucleation source
  and its equivalent inflow treatment when growth is present.
- Sewerin and Rigopoulos (2017), *An explicit adaptive grid approach for the
  numerical solution of the population balance equation*, Chemical Engineering
  Science 168, 250-270. Section 6.3, immediately before Eq. (65), uses a
  triangular source on `[0, 2 l_nuc]`. Section 5.2 discusses keeping adequate
  resolution in the birth interval as the rest of the population grows.

The examples adopt the triangular source shape, with a seeded initial state
and chosen constant rates. They do not reproduce the literature's chemical
kinetics, BaSO4 parameter set, or experimental data. These are analytical
verification cases; they do not by themselves establish engineering validation.

## Equations and parameters

Let `h(d) = max(1 - abs(d/a - 1), 0)/a`. Its integral is one, its support is
`[0, 2a]`, and its mean is `a`. Both cases solve

```text
dn/dt + G dn/dd = J h(d),       n(d,0) = N0 h(d).
```

The default values are `a = 0.08 mm`, `N0 = 1`, `J = 0.5 per second`,
`T = 10 s`, `f_max = 2.5`, `f_mult = 1.3`, and 100 steps. Number quantities
are per an arbitrary common suspension-volume unit. There are 41 categories
initially covering `[0, 1] mm`, with consecutive cell-width ratio 1.08.
These are geometric **cell widths**, not a constant ratio between diameter pivots.
The initial seed population makes `M_q(0)` nonzero and permits the same moment
normalization as the other manuscript figures.

For nucleation alone, `G = 0`:

```text
n(d,t) = (N0 + J t) h(d)
M_q(t) / M_q(0) = 1 + J t / N0
```

All normalized analytical moments overlap, `d43` stays constant, and the
normalized CDF is stationary. The adaptive grid makes one initial range
adjustment. Sustained grid motion or a large adaptive advantage is not expected.

For nucleation plus growth, `G = 0.12 mm/s`. With `H` the CDF of `h`:

```text
n(d,t) = N0 h(d-Gt) + (J/G) [H(d) - H(d-Gt)]
```

Writing `b_r = integral d^r h(d) dd`, the exact moments are

```text
M_q(t) = sum_{r=0}^q binom(q,r) b_r G^(q-r)
         [N0 t^(q-r) + J t^(q-r+1)/(q-r+1)].
```

The final analytical volume CDF is evaluated by piecewise polynomial
quadrature divided by the exact `M3`. Its value and the moment formulas are
independent of the numerical time step and pivot grid. Nucleation is continuous
throughout the growth calculation. Every adaptation rebuilds the birth-source
projection at the original physical birth sizes, not at rescaled birth sizes.

## Numerical method and limits

The solver reuses the nonnegative approximate moment fitting and characteristic
projection from `figure_growth_benchmark.py`. This is the positivity-constrained
variant: it conserves number, but does not preserve all six moments exactly.
The original unconstrained HMMC weight solver has not been changed. This choice
must be disclosed if these figures are later used in the manuscript.

Existing particles grow by one characteristic step. Particles born during that
step are integrated over their birth ages from zero to the step duration, so
new particles are not all assigned a full step of growth. The source projection
is rebuilt whenever the grid changes. Both approaches start with identical
pivots/populations and use identical rates and time steps.

No signed populations are silently clipped for these figures: the numerical
method constructs nonnegative populations, and the script checks positivity and
the exact number balance at every saved time. The CDF error samples both sides
of every numerical jump. It also records the fraction of the birth density below
the first pivot and the count of pivots inside the birth interval.

## Initial results

Errors below are percentages; CDF error is the maximum absolute difference
between the numerical and analytical volume CDF. Diameter errors are absolute
relative errors at the final time.

| Case | Fixed CDF | Adaptive CDF | Fixed d43 | Adaptive d43 | Adaptations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nucleation | 9.21 | 7.58 | 0.22 | 0.17 | 1 |
| Nucleation + growth | 81.88 | 24.98 | 22.38 | 1.26 | 9 |

Halving the time step changes the adaptive CDF error to 7.57% and 25.68%,
respectively. Doubling to 82 classes (with cell-width ratio `sqrt(1.08)` to
preserve the stretched-grid profile) reduces these errors to 3.78% and 20.30%.
These checks show remaining spatial/reconstruction error; they do not establish
full distribution convergence. The normalized distributions must be examined
alongside the raw moments: for growth, the adaptive `M3` error is about -21.57%
on 41 classes and -7.41% on 82 classes. A small error in the ratio `M4/M3` alone
does not demonstrate accurate volume prediction.

All input parameters and per-moment errors, including the refinement runs, are
recorded in the generated JSON file. No claim of exact moment conservation or
validated crystallization kinetics should be drawn from these provisional plots.
