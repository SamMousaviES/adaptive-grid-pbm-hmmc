# Changelog

## Unreleased

- Added all current manuscript validation figures to the reproduction runner.
- Added a generator for the seven-case final volume-CDF error table.
- Corrected local-source paths so examples run reliably from a fresh checkout.
- Updated the README artifact-to-script index to match the accepted manuscript.

## 0.2.0

- Rewritten as a pip-installable Python package built on NumPy and SciPy.
- Dataclass-based options API (`SolverOptions`, `ProcessOptions`, etc.).
- `adaptivehmmc.solve` dispatches on `process.type`:
  - `coalescence` / `constant`
  - `breakage` / `binary_equal_volume`
  - `growth` / `proportional`
- Adaptive-grid rescaling driven by a SciPy `solve_ivp` event for the
  ODE-based processes and by a fixed-step projection loop for growth.
- Example scripts reproducing every manuscript figure and table.
- pytest test suite covering volume conservation, breakage and growth
  benchmark accuracy, and adaptive-grid improvements.

## 0.1.0 - 2026-05-13

- Initial MATLAB distribution (`+adaptivehmmc`) with constant-kernel
  coalescence and adaptive uniform grid scaling.
