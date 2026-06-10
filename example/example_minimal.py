"""
Minimal, self-contained AstroChaos example.

Runs out of the box (only NumPy + SciPy) with a toy *linear* propagator, so it
needs none of the heavy astrodynamics machinery used by the other examples.

For a linear map  x(tf) = Phi @ x0  the propagated covariance has a closed form,
    Sigma_f = Phi @ Sigma0 @ Phi^T,
which we use here to check the PCE result.

Run:  python example/example_minimal.py
"""

import os
import sys

import numpy as np
from scipy.linalg import expm

# Make "import AstroChaos.*" work regardless of the current working directory:
# add the folder that CONTAINS the AstroChaos package to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AstroChaos.Covariance import covariance_from_state


# --- toy black-box propagator -------------------------------------------------
# Required signature: propagator(t_span, x0) -> (t_vec, state_vec, *_)
# with state_vec of shape (T, 6); the final state is state_vec[-1, :6].
_rng = np.random.default_rng(0)
_A = _rng.normal(scale=0.1, size=(6, 6))   # arbitrary linear dynamics matrix


def toy_propagator(t_span, x0):
    tf = t_span[-1]
    Phi = expm(_A * tf)
    x0 = np.asarray(x0).reshape(6)
    final = Phi @ x0
    state_vec = np.vstack([x0, final])     # (2, 6): initial and final state
    t_vec = np.array([t_span[0], tf])
    return t_vec, state_vec, None, None


def main():
    # nominal state and initial covariance
    r_mean = np.array([1.0, 0.5, -0.2, 0.01, -0.02, 0.03])
    Sigma0 = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    t_span = [0.0, 2.0]

    # --- AstroChaos: propagate the covariance with a PCE surrogate ---
    Sigma_pce = covariance_from_state(
        Sigma0, r_mean, t_span,
        d=6, p_order=3, r=1.5,
        propagator=toy_propagator,
    )

    # --- analytic ground truth for the linear map ---
    Phi = expm(_A * t_span[-1])
    Sigma_true = Phi @ Sigma0 @ Phi.T

    rel = np.linalg.norm(Sigma_pce - Sigma_true, "fro") / np.linalg.norm(Sigma_true, "fro")

    np.set_printoptions(precision=3, suppress=True)
    print("PCE-propagated covariance:\n", Sigma_pce)
    print("\nRelative Frobenius error vs analytic truth: {:.2e}".format(rel))


if __name__ == "__main__":
    main()
