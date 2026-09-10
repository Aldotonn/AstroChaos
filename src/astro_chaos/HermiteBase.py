import numpy as np


def hermite_probabilists(n):
    # H_0 = 1
    if n == 0:
        return lambda x: np.ones_like(x)

    # H_1 = x
    if n == 1:
        return lambda x: x

    # Ricorrenza
    H0 = lambda x: np.ones_like(x)
    H1 = lambda x: x

    polys = [H0, H1]

    for k in range(1, n):
        def make(k):
            return lambda x, k=k: x*polys[k](x) - k*polys[k-1](x)
        polys.append(make(k))
    return polys[n]


def multi_indices(d, p):
    indices = []

    def build(prefix, remaining_dim):
        if remaining_dim == d:
            if sum(prefix) <= p:
                indices.append(tuple(prefix))
            return
        for k in range(p+1):
            build(prefix+[k], remaining_dim+1)

    build([], 0)
    return indices


class Poly:
    def __init__(self, funcs):
        self.funcs = funcs  # list of callables

    def __call__(self, x):
        # x shape = (d, n_samples)
        out = np.ones(x.shape[1])
        for i, f in enumerate(self.funcs):
            out *= f(x[i])
        return out


def hermite(order, dist):
    d = dist.d  # number of dimensions

    # 1. Precompute univariate Hermite
    hermites = [hermite_probabilists(n) for n in range(order+1)]

    # 2. Generate multiindices
    alphas = multi_indices(d, order)

    # 3. Build polynomials
    basis = []
    for alpha in alphas:
        funcs = [hermites[k] for k in alpha]   # one poly for each dimension
        basis.append(Poly(funcs))

    return basis
