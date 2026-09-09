"""Coulaloglou--Tavlarides liquid-liquid dispersion rate closures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CTParameters:
    """Physical properties and original CT coefficients in SI units."""

    rho_continuous: float = 992.8
    rho_dispersed: float = 824.0
    mu_continuous: float = 9.3e-4
    surface_tension: float = 3.812e-2
    dissipation_rate: float = 3.0e-3
    dispersed_holdup: float = 0.1
    coalescence_c1: float = 2.17e-4
    coalescence_c2: float = 2.28e13
    breakage_c3: float = 4.87e-3
    breakage_c4: float = 5.52e-2


def ct_coalescence_kernel(pivots: np.ndarray, parameters: CTParameters) -> np.ndarray:
    """Return the CT pairwise coalescence kernel in m^3 s^-1."""
    diameter_i, diameter_j = np.meshgrid(pivots, pivots, indexing="ij")
    pair_sum = diameter_i + diameter_j
    collision = (
        parameters.coalescence_c1
        * parameters.dissipation_rate ** (1.0 / 3.0)
        * pair_sum**2
        * (diameter_i ** (2.0 / 3.0) + diameter_j ** (2.0 / 3.0)) ** 0.5
        / (1.0 + parameters.dispersed_holdup)
    )
    drainage = np.exp(
        -parameters.coalescence_c2
        * parameters.mu_continuous
        * parameters.rho_continuous
        * parameters.dissipation_rate
        * (diameter_i * diameter_j / pair_sum) ** 4
        / (
            parameters.surface_tension**2
            * (1.0 + parameters.dispersed_holdup) ** 3
        )
    )
    return collision * drainage


def ct_breakage_rate(pivots: np.ndarray, parameters: CTParameters) -> np.ndarray:
    """Return the CT breakage frequency in s^-1."""
    pivots = np.asarray(pivots, dtype=float)
    prefactor = (
        parameters.breakage_c3
        * parameters.dissipation_rate ** (1.0 / 3.0)
        / ((1.0 + parameters.dispersed_holdup) * pivots ** (2.0 / 3.0))
    )
    barrier = np.exp(
        -parameters.breakage_c4
        * parameters.surface_tension
        * (1.0 + parameters.dispersed_holdup) ** 2
        / (
            parameters.rho_dispersed
            * parameters.dissipation_rate ** (2.0 / 3.0)
            * pivots ** (5.0 / 3.0)
        )
    )
    return prefactor * barrier
