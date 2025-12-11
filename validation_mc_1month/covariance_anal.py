import numpy as np
import glob
import math
import matplotlib.pyplot as plt

# -----------------------------
# Parallel-Welford combiner
# -----------------------------
def combine_triplet(nA, muA, M2A, nB, muB, M2B):
    """
    Combine two (n, mu, M2) triplets for a single epoch.
    Shapes:
      n* : scalar
      mu*: (6,)
      M2*: (6,6)
    """
    if nA == 0:         # identity
        return nB, muB.copy(), M2B.copy()
    if nB == 0:
        return nA, muA.copy(), M2A.copy()
    n = nA + nB
    delta = muB - muA
    mu = muA + (nB / n) * delta
    M2 = M2A + M2B + np.outer(delta, delta) * (nA * nB / n)
    return n, mu, M2

def combine_epochwise(A, B):
    """
    Combine two dicts that each contain arrays:
      'n_k'   (K,), 'mu_k' (K,6), 'M2_k' (K,6,6)
    Returns a merged dict with same keys.
    """
    K = A['n_k'].shape[0]
    out = {
        'n_k':   np.zeros(K, dtype=np.int64),
        'mu_k':  np.zeros((K,6), dtype=np.float64),
        'M2_k':  np.zeros((K,6,6), dtype=np.float64),
    }
    for a in range(K):
        n, mu, M2 = combine_triplet(
            int(A['n_k'][a]), A['mu_k'][a], A['M2_k'][a],
            int(B['n_k'][a]), B['mu_k'][a], B['M2_k'][a]
        )
        out['n_k'][a]  = n
        out['mu_k'][a] = mu
        out['M2_k'][a] = M2
    return out

def cov_from(n, M2):
    """Unbiased sample covariance from Welford triplet."""
    C = np.full((6,6), np.nan)
    return M2 / (n - 1) if n > 1 else C

# -----------------------------
# Load partials and sort
# -----------------------------
files = sorted(glob.glob("mc_partial_job*_of_*.npz"))
if not files:
    raise RuntimeError("No mc_partial_job*.npz files found.")

partials = []
meta = None
for f in files:
    dat = np.load(f)
    # core totals (≤ N2). For N1 curves you can repeat with *_N1
    piece = {
        'n_k':  dat['n_k'],
        'mu_k': dat['mu_k'],
        'M2_k': dat['M2_k'],
    }
    partials.append(piece)
    if meta is None:
        meta = {
            'K': int(dat['K'][()]),
            't_idx_eval': dat['t_idx_eval'],
            'r_mean': dat['r_mean'],
            'Sigma6': dat['Sigma6'],
            'et0': float(dat['et0']),
            'et_final': float(dat['et_final']),
            'N1': int(dat['N1'][()]),
            'N2': int(dat['N2'][()]),
        }

K = meta['K']
N1 = meta['N1']
N2 = meta['N2']
epoch_idx = (K-1)                 # pick the mid-arc epoch for scalar plots

print(f"K = {K}, epoch_idx = {epoch_idx}, N1 = {N1}, N2 = {N2}")

# -----------------------------
# Build cumulative merges
# -----------------------------
cumulatives = []   # list of dicts like {'n_k','mu_k','M2_k'} after merging first i files
acc = {
    'n_k':  np.zeros(K, dtype=np.int64),
    'mu_k': np.zeros((K,6)),
    'M2_k': np.zeros((K,6,6)),
}
for piece in partials:
    acc = combine_epochwise(acc, piece)
    cumulatives.append({
        'n_k':  acc['n_k'].copy(),
        'mu_k': acc['mu_k'].copy(),
        'M2_k': acc['M2_k'].copy(),
    })

# Reference = final cumulative (largest N, i.e. N2)
ref = cumulatives[-1]

Sigma_ref_all = np.array([cov_from(ref['n_k'][a], ref['M2_k'][a]) for a in range(K)])
mu_ref_all    = ref['mu_k']

Sigma_ref = Sigma_ref_all[epoch_idx]
mu_ref    = mu_ref_all[epoch_idx]

tr_ref      = np.trace(Sigma_ref)
lambda_ref  = np.linalg.eigvalsh(Sigma_ref).max()

# -----------------------------
# Error metrics vs N (per epoch)
# -----------------------------
Ns = []                 # total samples used up to this cumulative step (at selected epoch)
errF = []               # relative Frobenius norm ||Σ(N)-Σ_ref||_F / ||Σ_ref||_F
tr_vals = []            # trace(Σ(N))
lambda_max = []         # largest eigenvalue of Σ(N)
vol_ratio = []          # sqrt(det Σ(N)) / sqrt(det Σ_ref)
mean_err = []           # ||μ(N)-μ_ref||_2 / max(1,||μ_ref||_2)

for cum in cumulatives:
    n = int(cum['n_k'][epoch_idx])
    if n < 2:
        continue
    SigmaN = cov_from(n, cum['M2_k'][epoch_idx])
    muN    = cum['mu_k'][epoch_idx]

    Ns.append(n)

    # Frobenius relative error
    num = np.linalg.norm(SigmaN - Sigma_ref, ord='fro')
    den = max(1e-16, np.linalg.norm(Sigma_ref, ord='fro'))
    errF.append(num / den)

    tr_vals.append(np.trace(SigmaN))

    w = np.linalg.eigvalsh(SigmaN)
    lambda_max.append(w.max())

    # volume ratio ~ (det Σ)^{1/2}
    detN = max(1e-300, np.linalg.det(SigmaN))
    detR = max(1e-300, np.linalg.det(Sigma_ref))
    vol_ratio.append(math.sqrt(detN / detR))

    # mean relative error
    den_mu = max(1.0, np.linalg.norm(mu_ref))
    mean_err.append(np.linalg.norm(muN - mu_ref) / den_mu)

Ns = np.array(Ns, dtype=float)
errF = np.array(errF)
tr_vals = np.array(tr_vals)
lambda_max = np.array(lambda_max)
vol_ratio = np.array(vol_ratio)
mean_err = np.array(mean_err)

# -----------------------------
# Locate N1 in Ns
# -----------------------------
idx_N1 = np.argmin(np.abs(Ns - N1))

print("\n=== Metrics at N1 ===")
print(f"N1 (target)           : {N1}")
print(f"N at idx_N1 in curve  : {Ns[idx_N1]}")
print(f"Frobenius rel. error  : {errF[idx_N1]:.4e}")
print(f"Mean rel. error       : {mean_err[idx_N1]:.4e}")
print(f"Trace(Σ(N1))          : {tr_vals[idx_N1]:.6e}")
print(f"λ_max(Σ(N1))          : {lambda_max[idx_N1]:.6e}")
print(f"Vol ratio √det        : {vol_ratio[idx_N1]:.6e}")
print("Reference (N2) trace  :", f"{tr_ref:.6e}")
print("Reference (N2) λ_max  :", f"{lambda_ref:.6e}")
print("======================\n")

# -----------------------------
# Plots (log–log where appropriate)
# -----------------------------

# 1) Covariance convergence (Frobenius)
plt.figure()
plt.loglog(Ns, errF, marker='o', label='Rel. Frobenius error of Σ')
# reference slope ~ N^{-1/2}
c0 = errF[-1] * (Ns[-1]**0.5)      # fit a 1/sqrt(N) guide through last point
plt.loglog(Ns, c0 / np.sqrt(Ns), linestyle='--', label='~ N^{-1/2}')

# highlight N1
plt.scatter([Ns[idx_N1]], [errF[idx_N1]],
            s=120, marker='o', edgecolors='k', facecolors='none',
            label=f'N1 ≈ {int(Ns[idx_N1]):d}')

plt.xlabel('Samples N (epoch idx = %d)' % epoch_idx)
plt.ylabel('Relative error ||Σ(N)-Σ_ref||_F / ||Σ_ref||_F')
plt.legend()
plt.title('Covariance convergence')

# 2) Mean convergence
plt.figure()
plt.semilogx(Ns, mean_err, marker='o')
plt.scatter([Ns[idx_N1]], [mean_err[idx_N1]],
            s=120, marker='o', edgecolors='k', facecolors='none')
plt.xlabel('Samples N (epoch idx = %d)' % epoch_idx)
plt.ylabel('Relative mean error ||μ(N)-μ_ref||/||μ_ref||')
plt.title('Mean convergence')

# 3) Trace of covariance vs N
plt.figure()
plt.semilogx(Ns, tr_vals, marker='o', label='trace(Σ(N))')
plt.axhline(tr_ref, linestyle='--', label='trace(Σ_ref)')

plt.scatter([Ns[idx_N1]], [tr_vals[idx_N1]],
            s=120, marker='o', edgecolors='k', facecolors='none',
            label='N1')

plt.xlabel('Samples N (epoch idx = %d)' % epoch_idx)
plt.ylabel('trace(Σ)')
plt.title('Trace of covariance vs N')
plt.legend()

# 4) Largest eigenvalue
plt.figure()
plt.semilogx(Ns, lambda_max, marker='o', label='λ_max(Σ(N))')
plt.axhline(lambda_ref, linestyle='--', label='λ_max(Σ_ref)')

plt.scatter([Ns[idx_N1]], [lambda_max[idx_N1]],
            s=120, marker='o', edgecolors='k', facecolors='none',
            label='N1')

plt.xlabel('Samples N (epoch idx = %d)' % epoch_idx)
plt.ylabel('Largest eigenvalue of Σ')
plt.title('Spectral convergence')
plt.legend()

# 5) Ellipsoid volume ratio
plt.figure()
plt.semilogx(Ns, vol_ratio, marker='o', label='√det(Σ(N)) / √det(Σ_ref)')
plt.axhline(1.0, linestyle='--', label='Reference (N2)')

plt.scatter([Ns[idx_N1]], [vol_ratio[idx_N1]],
            s=120, marker='o', edgecolors='k', facecolors='none',
            label='N1')

plt.xlabel('Samples N (epoch idx = %d)' % epoch_idx)
plt.ylabel('√det(Σ(N)) / √det(Σ_ref)')
plt.title('Ellipsoid volume ratio')
plt.legend()

plt.show()

from src.init.constants import MOON_OMEGA_MEAN

omega = MOON_OMEGA_MEAN              # [rad/s]

# span totale in tempo adimensionale (qui = Tpr = 7 giorni * omega)
T_nd_tot = float(meta['et_final'] - meta['et0'])

# gli indici di tempo usati negli snapshot MC
t_idx = meta['t_idx_eval'].astype(int)
max_idx = int(t_idx.max())           # ≈ Tlen-1 (ultimo step dell'integrazione)

# tempo adimensionale relativo di ogni snapshot (0 → T_nd_tot)
tau_snap = T_nd_tot * (t_idx / max_idx)

# converto in tempo reale [s] e poi in ore
t_snap_sec   = tau_snap / omega
t_snap_hours = t_snap_sec / 3600.0

print("Snapshot times [hours]:", t_snap_hours)
print("Range: %.3f h → %.3f h (%.3f days)" %
      (t_snap_hours[0], t_snap_hours[-1], t_snap_hours[-1]/24.0))

# tempo fisico corrispondente all'epoch_idx usato nella convergenza
t_epoch_hours = t_snap_hours[epoch_idx]
print("epoch_idx = %d → t = %.3f h = %.3f days" %
      (epoch_idx, t_epoch_hours, t_epoch_hours/24.0))
