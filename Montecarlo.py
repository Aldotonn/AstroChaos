import numpy as np



def MonteCarloConvergence(K, Sigma6, r_mean, propagator, t_span, N1, rng):

    # ---------- Welford accumulators (per epoch) ----------
    # counts per epoch
    n_k   = np.zeros(K, dtype=np.int64)
    # running mean per epoch (K x 6)
    mu_k  = np.zeros((K, 6), dtype=np.float64)
    # second-moment accumulators per epoch (K x 6 x 6)
    M2_k  = np.zeros((K, 6, 6), dtype=np.float64)
    # ---------- Helper: update Welford for vector x at epoch k ----------


    # snapshots at N1 (to get Sigma_N1 without a second run)
    n_k_N1  = None
    mu_k_N1 = None
    M2_k_N1 = None

    N2 = 5 * N1
    Xi  = rng.standard_normal((6, N2))  # ξ ~ N(0,I)

    # ---------- Cholesky ----------
    L6 = np.linalg.cholesky(Sigma6 + 1e-16*np.eye(6))


    def welford_update(k, x):
        # x is (6,)
        n_old   = n_k[k]
        n_new   = n_old + 1
        n_k[k]  = n_new

        delta   = x - mu_k[k]           # before mean update
        mu_k[k] = mu_k[k] + delta / n_new
        delta2  = x - mu_k[k]           # after mean update
        # rank-1 update for covariance accumulator
        M2_k[k] = M2_k[k] + np.outer(delta, delta2)

    def welford_update_arrays(n_arr, mu_arr, M2_arr, a_idx, x):
        n_old = n_arr[a_idx]
        n_new = n_old + 1
        n_arr[a_idx] = n_new
        delta  = x - mu_arr[a_idx]
        mu_arr[a_idx] = mu_arr[a_idx] + delta / n_new
        delta2 = x - mu_arr[a_idx]
        M2_arr[a_idx] = M2_arr[a_idx] + np.outer(delta, delta2)

    # ---------- PROPAGATE N1 for first montecarlo ----------
    for j_loc in range(N1):
        x0 = r_mean + L6 @ Xi[:, j_loc]
        t_vec, state_vec, _, _ = propagator(t_span, x0)
        if t_idx_eval is None:
            Tlen = state_vec.shape[0]
            t_idx_eval = np.linspace(0, Tlen-1, K, dtype=int)

        for a, idx in enumerate(t_idx_eval):
            x = state_vec[idx, :]
            # we update N1 and partially N2
            welford_update_arrays(n_k,     mu_k,     M2_k,     a, x)
            welford_update_arrays(n_k_N1,  mu_k_N1,  M2_k_N1,  a, x)

    # ---------- PROPAGATE N2-N1 elements to get N2 covariance----------
    for j_loc in range(N1, N2):
        x0 = r_mean + L6 @ Xi[:, j_loc]
        t_vec, state_vec, _, _ = propagator(t_span, x0)
        if t_idx_eval is None:
            Tlen = state_vec.shape[0]
            t_idx_eval = np.linspace(0, Tlen-1, K, dtype=int)

        for a, idx in enumerate(t_idx_eval):
            x = state_vec[idx, :]
            welford_update_arrays(n_k, mu_k, M2_k, a, x)

    # ----------  ----------
    def build_covariances(n_arr, M2_arr):
        """ utility to build covariances """
        covs = []
        for a in range(K):
            if n_arr[a] > 1:
                covs.append(M2_arr[a] / (n_arr[a] - 1))
            else:
                covs.append(np.full((6,6), np.nan))
        return covs

    covs_N1 = build_covariances(n_k_N1, M2_k_N1)
    covs_N2 = build_covariances(n_k, M2_k)
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

    # ------------------------------------------------------------
    #   convergence
    # ------------------------------------------------------------
    converged = (max_diag_rel <= thr) and (max_fro_rel <= thr) and (max_tr_rel <= thr)

    metrics = {
        "max_diag_rel": max_diag_rel,
        "max_fro_rel": max_fro_rel,
        "max_tr_rel": max_tr_rel,
        "max_eig_rel": max_eig_rel,
        "threshold": thr
    }

    return covs_N1, converged, metrics



