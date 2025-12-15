"""
Strain-rate helper models.

NOTE
----
The *fixed DIF* study in `railway_simulator.studies.strain_rate_sensitivity` does NOT
require these functions. This module is provided for future, physically-based
rate-dependent studies.

Concrete DIFs below follow the conservative empirical laws used in your
`cdp-generator` implementation (exponents 0.014 / 0.018 etc.).
"""
from __future__ import annotations

import math


def concrete_dif_compression(strain_rate_s: float) -> float:
    """
    Compressive DIF.

    Reference rate: eps_dot_s = 3e-5 1/s

    For eps_dot <= 30 1/s:
      DIF = (eps_dot/eps_dot_s)^0.014

    For eps_dot > 30 1/s:
      DIF = 0.012 * (eps_dot/eps_dot_s)^(1/3)
    """
    eps_dot = abs(float(strain_rate_s))
    eps_dot_s = 3e-5
    if eps_dot <= 0:
        return 1.0
    if eps_dot <= 30.0:
        dif = (eps_dot / eps_dot_s) ** 0.014
    else:
        dif = 0.012 * (eps_dot / eps_dot_s) ** (1.0 / 3.0)
    return max(1.0, float(dif))


def concrete_dif_tension(strain_rate_s: float) -> float:
    """
    Tensile DIF.

    Reference rate: eps_dot_s = 1e-6 1/s

    For eps_dot <= 10 1/s:
      DIF = (eps_dot/eps_dot_s)^0.018

    For eps_dot > 10 1/s:
      DIF = 0.0062 * (eps_dot/eps_dot_s)^(1/3)
    """
    eps_dot = abs(float(strain_rate_s))
    eps_dot_s = 1e-6
    if eps_dot <= 0:
        return 1.0
    if eps_dot <= 10.0:
        dif = (eps_dot / eps_dot_s) ** 0.018
    else:
        dif = 0.0062 * (eps_dot / eps_dot_s) ** (1.0 / 3.0)
    return max(1.0, float(dif))


def steel_dif_cowper_symonds(strain_rate_s: float, D: float = 40.4, q: float = 5.0) -> float:
    """
    Cowper-Symonds DIF for steel:
      DIF = 1 + (eps_dot / D)^(1/q)
    """
    eps_dot = abs(float(strain_rate_s))
    if eps_dot <= 0:
        return 1.0
    return float(1.0 + (eps_dot / float(D)) ** (1.0 / float(q)))
