"""Regenerate all manuscript figures and LaTeX tables.

The outputs are written to ``Git_dist/figures_out`` and
``Git_dist/figures_out/tables``. The script runs each artifact generator in a
separate Python process so the examples remain usable independently.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent

SCRIPTS = [
    EXAMPLES_DIR / "figure_bad_grid_example.py",
    EXAMPLES_DIR / "figure_redistribution_example.py",
    SCRIPT_DIR / "figure_coalescence_benchmark.py",
    EXAMPLES_DIR / "table_category_comparison.py",
    SCRIPT_DIR / "figure_breakage_benchmark.py",
    EXAMPLES_DIR / "figure_growth_benchmark.py",
    EXAMPLES_DIR / "figure_benchmark_tradeoff.py",
    EXAMPLES_DIR / "table_fmax_sensitivity.py",
]


def main() -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )

    for script in SCRIPTS:
        print(f"\n=== Running {script.name} ===", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
            env=env,
            check=True,
        )

    print(f"\nManuscript artifacts saved under {REPO_ROOT / 'figures_out'}")


if __name__ == "__main__":
    main()
