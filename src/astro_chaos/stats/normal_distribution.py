from math import erf, sqrt, pi, comb

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfinv


class NormalDistribution:
    """
    Vectorized IID Normal distribution.

    Parameters:
        mean : float
        std  : float
        d    : int = 1   (number of the dimensions of the problem)
    """

    def __init__(self, mean=0.0, std=1.0, d=1):
        self.mean = float(mean)
        self.std = float(std)
        self.d = int(d)

    def sample(self, n, rule="random"):
        """
        Returns samples of shape (d, n).
        Sobol works better withn power of 2^k (256, 512, 1024, 2048, …)"""

        if rule == "random":
            rng = np.random.default_rng()
            return rng.normal(self.mean, self.std, size=(self.d, n))

        elif rule == "sobol":
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=self.d, scramble=True)
            u = sampler.random(n).T  # shape = (d, n)
            return self.ppf(u)

        else:
            raise ValueError(f"Sampling rule '{rule}' not supported.")

    def _check_shape(self, samples: NDArray[np.float64]):
        if samples.ndim not in (1, 2):
            raise ValueError(f"Samples shape '{samples.shape}' not supported.")
        if samples.shape[0] != self.d:
            shape = self.d, samples.shape[1] if samples.ndim == 2 else self.d
            raise ValueError(f"Samples should have (shape {shape}")

    def pdf(self, samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        PDF (multivariate independent)

        x shape = (d, n) or (d,)
        returns pdf for each sample.
        """

        samples = np.asarray(samples)
        self._check_shape(samples)

        z = (samples - self.mean) / self.std
        pdf_uni = np.exp(-0.5 * z * z) / (self.std * sqrt(2 * pi))

        # multivariate independent: product along axis 0
        return np.prod(pdf_uni, axis=0)

    def cdf(self, samples: NDArray[np.float64]) -> float:
        """
        CDF (univariate only)  # TODO: does this mean that the shape is 1D?
        """
        samples = np.asarray(samples)
        self._check_shape(samples)

        z = (samples - self.mean) / (self.std * sqrt(2))
        return 0.5 * (1 + erf(z))

    def ppf(self, u):
        """
        Vectorized inverse CDF

        u shape = (d, n)
        """
        return self.mean + self.std * sqrt(2) * erfinv(2 * u - 1)

    def mom(self, n_elements: int) -> float:
        """
        Raw moment of univariate Normal distribution.
        """
        n_elements = int(n_elements)
        moment = 0.0

        for i in range(0, n_elements + 1):
            if i % 2 == 1:
                continue
            # double factorial for even i
            df = np.prod(np.arange(1, i, 2)) if i > 0 else 1

            moment += (
                    comb(n_elements, i)
                    * (self.mean ** (n_elements - i))
                    * (self.std ** i)
                    * df
            )

        return moment

    icdf = ppf  # alias
