"""
AstroChaos -- lightweight Polynomial Chaos Expansion (PCE) for uncertainty propagation.

Public API::

    from AstroChaos import normal, hermite, regression, covariance, covariance_from_state
    from AstroChaos import montecarlo_covariance, montecarlo_convergence
"""

from AstroChaos.Normal import normal
from AstroChaos.HermiteBase import (
    hermite,
    hermite_probabilists,
    multi_indices,
    Poly,
)
from AstroChaos.FitRegression import regression
from AstroChaos.Covariance import covariance, covariance_from_state
from AstroChaos.Montecarlo import montecarlo_covariance, montecarlo_convergence

__all__ = [
    "normal",
    "hermite",
    "hermite_probabilists",
    "multi_indices",
    "Poly",
    "regression",
    "covariance",
    "covariance_from_state",
    "montecarlo_covariance",
    "montecarlo_convergence",
]

__version__ = "0.1.0"
