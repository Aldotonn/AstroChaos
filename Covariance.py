import numpy as np
from AstroChaos.Normal import normal
from AstroChaos.HermiteBase import hermite
from AstroChaos.FitRegression import regression
from math import ceil





def covariance(psi_a, C_a):
    phi2_emp_a = (psi_a ** 2).mean(axis=0)  # length Nt
    phi2_emp_a[0] = 0.0  # drop constant term for covariance

    Sigma_pce = C_a.T @ np.diag(phi2_emp_a) @ C_a

    return Sigma_pce



def covariance_from_state(Cov_initial, init_state, t_span, d, p_order, r, propagator):
    """
    :param Cov_initial:initial covariance matrix (6,6) or (3,3)
    :param init_state: initial state vector shape (6,) or (3,)
    :param t_span: time of the simulation for your propagator
    :param d: stochastic dimension of the problem
    :param p_order: order of the PCE
    :param r: over sampling ratio in [1.2, 3.0]
    :param propagator: insert your propagtor
    :return:
    -covariance matrix at final time


    """
    ## --- CHOLESKY DECOMPOSITION --- ##
    L6 = np.linalg.cholesky(Cov_initial + 1e-16 * np.eye(6))  # small jitter for safety
    # --- distribution and basis ---
    xi_dist = normal(0, 1, d)
    basis = hermite(p_order, xi_dist)  # orthonormal Hermite
    Nt = len(basis)  # number of basis terms (== comb(d+p, p))
    M = max(Nt, int(ceil(r * Nt)))  # ensure M ≥ Nt
    # --- draw samples and map to initial conditions ---
    Xi = xi_dist.sample(M, rule="sobol")  # shape (d, M)
    X0 = init_state[:, None] + L6 @ Xi  # shape (6, M)
    Ya = np.zeros((M,6))
    for j in range(M):
        t_vec, state_vec, _, _ = propagator(t_span, X0[:, j])
        Ya[j,:] = state_vec[-1, :6]


    # Fit PCE at this snapshot (same basis and Xi)
    poly_a, C_a, Psi_a = regression(basis, Xi, Ya)

    Sigma_pce_time = covariance(Psi_a, C_a)

    return Sigma_pce_time