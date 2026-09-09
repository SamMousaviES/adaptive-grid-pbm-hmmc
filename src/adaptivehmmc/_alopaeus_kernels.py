"""Size-dependent coalescence and breakage kernels from Alopaeus (2022)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc


@dataclass(frozen=True)
class AlopaeusKernelParameters:
    """Physical properties and closure coefficients in SI units."""

    rho_continuous: float = 992.8
    rho_dispersed: float = 824.0
    mu_continuous: float = 9.3e-4
    mu_dispersed: float = 5.0e-3
    surface_tension: float = 3.812e-2
    dissipation_rate: float = 3.0e-3
    dispersed_holdup: float = 0.1
    collision_c3: float = 0.433e-3
    breakage_c7: float = 0.986
    breakage_c8: float = 0.892e-3
    breakage_c9: float = 0.2
    efficiency_c11_prime: float = 2.71446


def alopaeus_coalescence_kernel(
    pivots: np.ndarray, parameters: AlopaeusKernelParameters
) -> np.ndarray:
    """Return collision frequency times coalescence efficiency in m^3 s^-1."""
    diameter_i, diameter_j = np.meshgrid(pivots, pivots, indexing="ij")
    pair_sum = diameter_i + diameter_j
    collision_frequency = (
        parameters.collision_c3
        * parameters.dissipation_rate ** (1.0 / 3.0)
        * pair_sum**2
        * (diameter_i ** (2.0 / 3.0) + diameter_j ** (2.0 / 3.0)) ** 0.5
        / (1.0 + parameters.dispersed_holdup)
    )
    exponent = -(
        parameters.efficiency_c11_prime
        * parameters.mu_continuous
        / (
            parameters.rho_continuous
            * parameters.dissipation_rate ** (1.0 / 3.0)
            * pair_sum ** (2.0 / 3.0)
        )
    )
    efficiency = (
        1.0 + 0.26144 * parameters.mu_dispersed / parameters.mu_continuous
    ) ** exponent
    return collision_frequency * efficiency


def alopaeus_breakage_rate(
    pivots: np.ndarray, parameters: AlopaeusKernelParameters
) -> np.ndarray:
    """Return the Alopaeus-model breakage frequency in s^-1."""
    pivots = np.asarray(pivots, dtype=float)
    surface_term = (
        parameters.breakage_c8
        * parameters.surface_tension
        / (
            parameters.rho_continuous
            * parameters.dissipation_rate ** (2.0 / 3.0)
            * pivots ** (5.0 / 3.0)
        )
    )
    viscosity_term = (
        parameters.breakage_c9
        * parameters.mu_dispersed
        / (
            np.sqrt(parameters.rho_continuous * parameters.rho_dispersed)
            * parameters.dissipation_rate ** (1.0 / 3.0)
            * pivots ** (4.0 / 3.0)
        )
    )
    return (
        parameters.breakage_c7
        * parameters.dissipation_rate ** (1.0 / 3.0)
        * erfc(np.sqrt(surface_term + viscosity_term))
    )
