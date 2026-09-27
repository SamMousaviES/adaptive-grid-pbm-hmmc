"""Regenerate all current manuscript figures and numerical tables.

Outputs are written to ``figures_out/`` and ``figures_out/tables/``. Each
generator runs in a separate Python process so it remains independently usable.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent

SCRIPTS = [
    "figure_bad_grid_example.py",
    "figure_redistribution_example.py",
    "figure_breakage_benchmark_three_panel.py",
    "figure_nucleation_benchmark_three_panel.py",
    "figure_coupled_benchmark_three_panel.py",
    "figure_alopaeus_coalescence_breakage_three_panel.py",
    "figure_benchmark_tradeoff.py",
    "table_category_comparison.py",
    "table_fmax_sensitivity.py",
    "table_analytical_benchmark_cdf_error.py",
]


def main() -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )

    for name in SCRIPTS:
        script = EXAMPLES_DIR / name
        print(f"\n=== Running {name} ===", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=EXAMPLES_DIR,
            env=env,
            check=True,
        )

    print(f"\nManuscript artifacts saved under {REPO_ROOT / 'figures_out'}")


if __name__ == "__main__":
    main()
