import numpy as np

def regression(basis, Xi, Ya):
    """
    basis : list of Poly objects (each has __call__(Xi) -> array(M,))
    Xi    : array shape (d, M)
    Ya    : array shape (M,) or (M, n_outputs)
    """

    Xi = np.asarray(Xi)
    Ya = np.asarray(Ya)

    M = Xi.shape[1]
    P = len(basis)

    # ---------------------------------------------------------
    # Build evaluation matrix Psi
    # ---------------------------------------------------------
    Psi = np.zeros((M, P))

    for j, poly in enumerate(basis):
        Psi[:, j] = poly(Xi)   # poly returns shape (M,)

    # ---------------------------------------------------------
    # Solve least squares
    # ---------------------------------------------------------
    # If Ya is shape (M,), convert to (M,1)
    if Ya.ndim == 1:
        Ya = Ya.reshape(-1, 1)

    C, _, _, _ = np.linalg.lstsq(Psi, Ya, rcond=None)  # shape (P, n_outputs)

    # ---------------------------------------------------------
    # Build PCE model evaluator
    # ---------------------------------------------------------
    def pce_model(X):
        """
        Evaluate the PCE at X.
        X must have shape (d, N)
        """
        X = np.asarray(X)
        N = X.shape[1]

        Y_pred = np.zeros((N, C.shape[1]))

        for j, poly in enumerate(basis):
            Y_pred += poly(X).reshape(N, 1) * C[j]

        # return shape (N,) if scalar output
        if Y_pred.shape[1] == 1:
            return Y_pred[:, 0]
        return Y_pred


    return pce_model, C, Psi
