import numpy as np
from math import sqrt, pi, comb
from scipy.special import erf, erfinv  # erf/erfinv are vectorized (NumPy-aware)


class normal:
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

    # ----------------------------------------------------
    # Sampling
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # PDF (multivariate independent)
    # ----------------------------------------------------
    def pdf(self, x):
        """
        x shape = (d, n) or (d,)
        returns pdf for each sample.
        """

        x = np.asarray(x)
        z = (x - self.mean) / self.std
        pdf_uni = np.exp(-0.5 * z * z) / (self.std * sqrt(2 * pi))

        # multivariate independent: product along axis 0
        return np.prod(pdf_uni, axis=0)

    # ----------------------------------------------------
    # CDF (univariate, vectorized over arrays)
    # ----------------------------------------------------
    def cdf(self, x):
        """Standard-Normal-style CDF, evaluated element-wise on scalars or arrays."""
        x = np.asarray(x, dtype=float)
        z = (x - self.mean) / (self.std * sqrt(2))
        return 0.5 * (1 + erf(z))

    # ----------------------------------------------------
    # Vectorized inverse CDF
    # ----------------------------------------------------
    def ppf(self, u):
        """
        u shape = (d, n)
        """
        return self.mean + self.std * sqrt(2) * erfinv(2 * u - 1)

    icdf = ppf  # alias

    # ----------------------------------------------------
    # Raw moment (same for each component)
    # ----------------------------------------------------
    def mom(self, k):
        """
        Raw moment of univariate Normal distribution.
        """
        k = int(k)
        moment = 0.0

        for i in range(0, k + 1):
            if i % 2 == 1:
                continue
            # double factorial for even i
            df = np.prod(np.arange(1, i, 2)) if i > 0 else 1

            moment += (
                    comb(k, i)
                    * (self.mean ** (k - i))
                    * (self.std ** i)
                    * df
            )

        return moment
