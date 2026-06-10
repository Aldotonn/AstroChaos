import numpy as np


# ---------------------------------------------------------------------------
# Vectorized Welford update over all K epochs at once.
#   n_arr  : (K,)        sample counts per epoch
#   mu_arr : (K, 6)      running means per epoch
#   M2_arr : (K, 6, 6)   running co-moment accumulators per epoch
#   X      : (K, 6)      one new observation per epoch (same Monte Carlo draw,
#                        sampled at the K evaluation epochs)
# Updates the accumulators in place (rank-1 covariance update per epoch).
# ---------------------------------------------------------------------------
def _welford_update_all(n_arr, mu_arr, M2_arr, X):
    n_arr += 1
    delta = X - mu_arr                       # (K, 6) -- before mean update
    mu_arr += delta / n_arr[:, None]
    delta2 = X - mu_arr                      # (K, 6) -- after mean update
    M2_arr += np.einsum("ki,kj->kij", delta, delta2)


def _build_covariances(K, n_arr, M2_arr):
    """Turn co-moment accumulators into sample covariances (M2 / (n-1))."""
    covs = []
    for a in range(K):
        if n_arr[a] > 1:
            covs.append(M2_arr[a] / (n_arr[a] - 1))
        else:
            covs.append(np.full((6, 6), np.nan))
    return covs


def montecarlo_covariance(K, Sigma6, r_mean, propagator, t_span, N1, rng):
    """
    ---------------
    Monte Carlo simulation to obtain the state covariance at K equispaced epochs,
    using the (numerically stable) online Welford method.
    ---------------

        :param K: number of epochs at which the covariance is evaluated
        :param Sigma6: initial covariance (6, 6)
        :param r_mean: initial (mean) state (6,)
        :param propagator: propagator, called as propagator(t_span, x0) ->
                           (t_vec, state_vec, *_), state_vec of shape (T, >=6)
        :param t_span: propagation horizon, in the format the propagator expects
        :param N1: number of Monte Carlo samples
        :param rng: numpy random Generator (fixes the simulation output)
        :return: list of K covariance matrices (6, 6), one per epoch
    """
    # ---------- Welford accumulators (per epoch) ----------
    n_k = np.zeros(K, dtype=np.int64)         # counts per epoch
    mu_k = np.zeros((K, 6), dtype=np.float64)  # running mean per epoch
    M2_k = np.zeros((K, 6, 6), dtype=np.float64)  # co-moment per epoch

    # ---------- draws and Cholesky factor ----------
    Xi = rng.standard_normal((6, N1))                  # xi ~ N(0, I)
    L6 = np.linalg.cholesky(Sigma6 + 1e-16 * np.eye(6))

    # epoch indices into the trajectory; set after the first propagation
    t_idx_eval = None

    # ---------- propagate the N1 samples ----------
    for j_loc in range(N1):
        x0 = r_mean + L6 @ Xi[:, j_loc]
        t_vec, state_vec, _, _ = propagator(t_span, x0)

        if t_idx_eval is None:
            Tlen = state_vec.shape[0]
            t_idx_eval = np.linspace(0, Tlen - 1, K, dtype=int)

        X = state_vec[t_idx_eval, :6]          # (K, 6): this draw at every epoch
        _welford_update_all(n_k, mu_k, M2_k, X)

    return _build_covariances(K, n_k, M2_k)


def montecarlo_convergence(K, Sigma6, r_mean, propagator, t_span, N1, rng):
    """
    ---------------
    Assess whether N1 Monte Carlo samples are enough to produce a converged
    covariance, by comparing the covariance built from N1 samples against the one
    built from N2 = 5*N1 samples, per epoch.
    ---------------

        :param K: number of epochs at which the covariance is evaluated
        :param Sigma6: initial covariance (6, 6)
        :param r_mean: initial (mean) state (6,)
        :param propagator: propagator, called as propagator(t_span, x0) ->
                           (t_vec, state_vec, *_)
        :param t_span: propagation horizon, in the format the propagator expects
        :param N1: number of Monte Carlo samples for the reference (smaller) run
        :param rng: numpy random Generator (fixes the simulation output)
        :return: (covs_N1, converged, metrics)
                 covs_N1  : list of K covariance matrices from N1 samples
                 converged: bool, True if all relative metrics are below threshold
                 metrics  : dict of max relative changes (diag / Frobenius / trace / eig)
    """
    # ---------- accumulators for the full N2 run ----------
    n_k = np.zeros(K, dtype=np.int64)
    mu_k = np.zeros((K, 6), dtype=np.float64)
    M2_k = np.zeros((K, 6, 6), dtype=np.float64)

    # ---------- snapshot accumulators for the first N1 samples ----------
    n_k_N1 = np.zeros(K, dtype=np.int64)
    mu_k_N1 = np.zeros((K, 6), dtype=np.float64)
    M2_k_N1 = np.zeros((K, 6, 6), dtype=np.float64)

    N2 = 5 * N1
    Xi = rng.standard_normal((6, N2))                  # xi ~ N(0, I)
    L6 = np.linalg.cholesky(Sigma6 + 1e-16 * np.eye(6))

    t_idx_eval = None

    # ---------- first N1 samples: feed both the N1 snapshot and the N2 run ----------
    for j_loc in range(N1):
        x0 = r_mean + L6 @ Xi[:, j_loc]
        t_vec, state_vec, _, _ = propagator(t_span, x0)

        if t_idx_eval is None:
            Tlen = state_vec.shape[0]
            t_idx_eval = np.linspace(0, Tlen - 1, K, dtype=int)

        X = state_vec[t_idx_eval, :6]
        _welford_update_all(n_k, mu_k, M2_k, X)
        _welford_update_all(n_k_N1, mu_k_N1, M2_k_N1, X)

    # ---------- remaining samples up to N2: feed only the N2 run ----------
    for j_loc in range(N1, N2):
        x0 = r_mean + L6 @ Xi[:, j_loc]
        t_vec, state_vec, _, _ = propagator(t_span, x0)

        if t_idx_eval is None:                          # safety (N1 == 0)
            Tlen = state_vec.shape[0]
            t_idx_eval = np.linspace(0, Tlen - 1, K, dtype=int)

        X = state_vec[t_idx_eval, :6]
        _welford_update_all(n_k, mu_k, M2_k, X)

    covs_N1 = _build_covariances(K, n_k_N1, M2_k_N1)
    covs_N2 = _build_covariances(K, n_k, M2_k)

    # ============================================================
    #   CONVERGENCE METRICS (per epoch) between N1 and N2
    #   - diagonals, Frobenius, trace, eigenvalues
    # ============================================================
    eps_den = 1e-12
    thr = 0.05  # threshold for convergence (5%)

    max_diag_rel = 0.0
    max_fro_rel = 0.0
    max_tr_rel = 0.0
    max_eig_rel = 0.0

    for a in range(K):
        P1 = covs_N1[a]
        P2 = covs_N2[a]
        if not np.isfinite(P1).all() or not np.isfinite(P2).all():
            continue

        # diagonal
        rel_diag = np.abs(np.diag(P2) - np.diag(P1)) / np.maximum(np.abs(np.diag(P2)), eps_den)
        max_diag_rel = max(max_diag_rel, np.max(rel_diag))

        # Frobenius
        fro_rel = np.linalg.norm(P2 - P1, "fro") / max(np.linalg.norm(P2, "fro"), eps_den)
        max_fro_rel = max(max_fro_rel, fro_rel)

        # trace
        tr_rel = abs(np.trace(P2) - np.trace(P1)) / max(abs(np.trace(P2)), eps_den)
        max_tr_rel = max(max_tr_rel, tr_rel)

        # eigenvalues
        e1 = np.linalg.eigvalsh(P1)
        e2 = np.linalg.eigvalsh(P2)
        eig_rel = np.abs(e2 - e1) / np.maximum(np.abs(e2), eps_den)
        max_eig_rel = max(max_eig_rel, np.max(eig_rel))

    converged = (max_diag_rel <= thr) and (max_fro_rel <= thr) and (max_tr_rel <= thr)

    metrics = {
        "max_diag_rel": max_diag_rel,
        "max_fro_rel": max_fro_rel,
        "max_tr_rel": max_tr_rel,
        "max_eig_rel": max_eig_rel,
        "threshold": thr,
    }

    return covs_N1, converged, metrics
