# AstroChaos

**Lightweight Polynomial Chaos Expansion (PCE) for uncertainty propagation.**

AstroChaos builds a cheap *surrogate model* of how an input uncertainty (described by a
covariance matrix) propagates through an expensive, possibly black-box, dynamical system.
Instead of running a huge Monte Carlo campaign to estimate the output covariance, it fits a
polynomial expansion from a small, well-chosen set of samples and reads the statistics off the
expansion coefficients.

Originally developed for **cislunar collision-risk analysis** (propagating spacecraft state
uncertainty under high-fidelity multi-body dynamics), the core is fully generic: it works with
**any** propagator that maps an input vector to an output vector.

---

## Why PCE instead of Monte Carlo?

| | Monte Carlo | Polynomial Chaos (AstroChaos) |
|---|---|---|
| Samples for a smooth 6-D problem | $10^4$–$10^6$ propagations | $\sim 10^2$ propagations |
| Output | covariance only (slow convergence, $O(1/\sqrt{N})$) | full surrogate: covariance **and** a cheap evaluable model |
| Re-evaluation at new points | re-run the propagator | evaluate a polynomial (near-free) |

The idea — replacing a costly stochastic simulation with a low-cost surrogate to estimate
covariance, sensitivities and moments — is the same one used for **uncertainty quantification
and risk estimation** well beyond astrodynamics.

---

## Features

- **Probabilist Hermite basis** in arbitrary stochastic dimension `d`, total-degree truncation up to order `p`.
- **Smart sampling**: i.i.d. Gaussian or scrambled **Sobol** quasi-random sequences (faster convergence).
- **Regression (non-intrusive) PCE fit** via least squares — treats the propagator as a black box.
- **Output covariance** reconstructed directly from the PCE coefficients.
- **One-call API** (`covariance_from_state`) and a fully manual pipeline for control.
- **Monte Carlo cross-check** utilities (Welford online covariance + convergence metrics).
- Pure **NumPy / SciPy**, no heavy dependencies.

---

## Installation

```bash
git clone <your-repo-url> AstroChaos
cd AstroChaos
pip install -r requirements.txt
```

AstroChaos is imported as a package; make sure the folder *containing* `AstroChaos/` is on your
`PYTHONPATH` (the bundled examples handle this automatically).

---

## Quick start

The only thing you provide is a `propagator(t_span, x0)` that returns
`(t_vec, state_vec, *_)`, where `state_vec[-1, :6]` is the final state. A complete, runnable
example with a toy linear propagator lives in [`example/example_minimal.py`](example/example_minimal.py):

```python
import numpy as np
from AstroChaos.Covariance import covariance_from_state

# your black-box propagator: (t_span, x0) -> (t_vec, state_vec, *_)
def propagator(t_span, x0):
    ...
    return t_vec, state_vec, None, None

r_mean = np.array([1.0, 0.5, -0.2, 0.01, -0.02, 0.03])   # nominal state (6,)
Sigma0 = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])    # initial covariance (6x6)

Sigma_f = covariance_from_state(
    Sigma0, r_mean, t_span=[0.0, 2.0],
    d=6, p_order=3, r=1.5,          # stochastic dim, PCE order, oversampling ratio
    propagator=propagator,
)
print(Sigma_f)   # propagated 6x6 covariance at final time
```

For full control (custom basis, reusing the surrogate, sensitivity studies) use the manual
pipeline shown in [`example/example_open_use.py`](example/example_open_use.py).

---

## API overview

| Module | Key object | Purpose |
|---|---|---|
| `Normal.py` | `normal(mean, std, d)` | Gaussian germ: `sample` (random/Sobol), `pdf`, `cdf`, `ppf`, raw moments. |
| `HermiteBase.py` | `hermite(order, dist)` | Builds the multivariate probabilist-Hermite basis (total-degree ≤ `order`). |
| `FitRegression.py` | `regression(basis, Xi, Ya)` | Least-squares PCE fit → `(pce_model, C, Psi)`: a callable surrogate + coefficients + design matrix. |
| `Covariance.py` | `covariance(Psi, C)` / `covariance_from_state(...)` | Output covariance from PCE coefficients; end-to-end one-call helper. |
| `Montecarlo.py` | `montecarlo_covariance(...)`, `montecarlo_convergence(...)` | Reference Monte Carlo covariance (online Welford) and convergence diagnostics. |

---

## Project structure

```
AstroChaos/
├── Normal.py            # Gaussian germ distribution (sampling, moments)
├── HermiteBase.py       # Hermite polynomial basis + multi-indices
├── FitRegression.py     # non-intrusive (regression) PCE fit
├── Covariance.py        # covariance from PCE coefficients (+ one-call helper)
├── Montecarlo.py        # Monte Carlo reference & convergence checks
└── example/
    ├── example_minimal.py            # self-contained, runs out of the box
    ├── example_open_use.py           # manual pipeline (needs a real propagator)
    └── example_closed_implementation.py  # one-call API (needs a real propagator)
```

---

## Notes & limitations

- Surrogate accuracy depends on PCE order `p` and the oversampling ratio `r` (number of samples
  `M ≈ r · n_basis`). Increase them for strongly non-linear dynamics.
- The basis is the **probabilist** Hermite family; the covariance step uses an empirical
  normalisation of the basis norms, so accuracy improves with more samples.
- Current germ is Gaussian (`Normal`); other input distributions would need the matching
  orthogonal family (e.g. Legendre for uniform inputs).

---

## Author

**Aldo Tonnini** — MSc Aerospace Engineering, ISAE-Supaero.
Built as part of research on cislunar debris risk and Lunar Gateway NRHO safety
(IAC 2025 / IAC 2026).
