"""Shared helpers used by the example scripts."""

from __future__ import annotations

from pathlib import Path


def output_dir() -> Path:
    """Return the figures_out directory (creating it if needed)."""
    here = Path(__file__).resolve().parent
    out = here.parent / "figures_out"
    out.mkdir(parents=True, exist_ok=True)
    return out


def table_dir() -> Path:
    """Return the figures_out/tables directory (creating it if needed)."""
    out = output_dir() / "tables"
    out.mkdir(parents=True, exist_ok=True)
    return out
