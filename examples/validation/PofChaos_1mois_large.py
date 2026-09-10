import numpy as np
import spiceypy as sp
from math import comb, ceil
from src.init.constants import MOON_SMA, MOON_OMEGA_MEAN
from src.coc.synodic_j2000 import synodic_to_j2000, j2000_to_synodic
from src.init.primary import Primary
from src.init.cr3bp import Cr3bp
from src.init.ephemeris import Ephemeris
from src.propagation.ephemeris_propagator import EphemerisSHPropagator
from src.init.load_kernels import load_kernels_moon_pot
from astro_chaos.Normal import normal
from astro_chaos.HermiteBase import hermite
from astro_chaos.FitRegression import regression
from astro_chaos.Covariance import covariance


if not hasattr(np, "bool"):  # compatibility for NumPy >=1.24
    np.bool = np.bool_

load_kernels_moon_pot()

## --- INITIAL STATE --- ##

r_mean = np.array([
    9.87848892e-01,  -1.47371321e-06,  4.91889975e-03,   # [x, y, z] [adim]
    1.60283536e+00,  9.84390573e-01,  4.13915823e-05    # [vx, vy, vz] [adim]
])

## --- PROP INIT --- ##
cr3bp = Cr3bp(Primary.EARTH, Primary.MOON)
ephemeris = Ephemeris((Primary.EARTH, Primary.MOON, Primary.SUN),
                      moon=("GRGM0100", 50, 50, False))
prop = EphemerisSHPropagator(
    ephemeris,
    with_stm=True,
    obs=Primary.EARTH,
    t_c=1. / MOON_OMEGA_MEAN,
    l_c=MOON_SMA,
    time_steps=1000000
)

T = 375190.2  # [s]
#Tpr = (40*269.09*60*MOON_OMEGA_MEAN)
#Tpr = 7 *24* 60*60*MOON_OMEGA_MEAN
Tpr = 2 * np.pi
t_span = [0, Tpr]
L = 384400

## --- ROTATION INTO J2000 ---##
et0 = sp.str2et('2019 JUN 18 12:00:00.000')*MOON_OMEGA_MEAN
et_final = et0 + Tpr

# DOPO (ok con Numba):
t0_nd = np.array([0.0], dtype=np.float64)  # array 1D float64, non lista int
state_matrix = np.ascontiguousarray(
    np.array(r_mean, dtype=np.float64).reshape(1, 6)
)

t_patch, state_0 = synodic_to_j2000(
    t0_nd,
    state_matrix,
    float(et0),
    cr3bp,
    cr3bp.m1,
    adim=True,
    bary_from_spice=True
)

r_mean = state_0.reshape(6,)
print(state_0)

## --- POSITION COVARIANCE --- ##

sigma_r = 10.0/L  # [km] standard deviation
rho_xy, rho_xz, rho_yz = 0.6/L, 0.3/L, 0.5/L

P_rr = np.array([
    [sigma_r**2,         rho_xy*sigma_r**2, rho_xz*sigma_r**2],
    [rho_xy*sigma_r**2,  sigma_r**2,        rho_yz*sigma_r**2],
    [rho_xz*sigma_r**2,  rho_yz*sigma_r**2, sigma_r**2]
])

## --- VELOCITY COVARIANCE --- ##

v_nom = r_mean[3:]            # nominal velocity components [km/s]
sigma_v = 0.02 * np.abs(v_nom)  # 2% relative uncertainty
rho_vxy, rho_vxz, rho_vyz = 0.2, 0.2, 0.2

P_vv = np.array([
    [sigma_v[0]**2,              rho_vxy*sigma_v[0]*sigma_v[1], rho_vxz*sigma_v[0]*sigma_v[2]],
    [rho_vxy*sigma_v[0]*sigma_v[1], sigma_v[1]**2,              rho_vyz*sigma_v[1]*sigma_v[2]],
    [rho_vxz*sigma_v[0]*sigma_v[2], rho_vyz*sigma_v[1]*sigma_v[2], sigma_v[2]**2]
])

## --- CROSS TERMS (pos–vel) --- ##
P_rv = np.zeros((3, 3))
P_vr = P_rv.T

## --- FULL 6×6 COVARIANCE MATRIX --- ##

Sigma6 = np.block([
    [P_rr, P_rv],
    [P_vr, P_vv]
])

print("Full covariance matrix Sigma6 [km² | km²/s²]:\n", Sigma6)

## --- CHOLESKY DECOMPOSITION --- ##

L6 = np.linalg.cholesky(Sigma6 + 1e-16*np.eye(6))  # small jitter for safety

# --- stochastic dimension and order ---
d = 6
p_order = 5  # choose once and keep it consistent

# --- distribution and basis ---
xi_dist = normal(0, 1, d)
basis = hermite(p_order, xi_dist)  # orthonormal Hermite
Nt = len(basis)  # number of basis terms (== comb(d+p, p))

r = 3  # over-sampling ratio in [1.2, 2.0]
M = max(Nt, int(ceil(r*Nt)))  # ensure M ≥ Nt
print(f"p={p_order}, Nt={Nt}, M={M}")  # for d=6: p=3 → Nt=84 → M≈126

# --- draw samples and map to initial conditions ---
Xi = xi_dist.sample(M, rule="sobol")  # shape (d, M)
X0 = r_mean[:, None] + L6 @ Xi        # shape (6, M)
print("Number of terms in basis =", len(basis))

## --- Propagation of Random Samples --- ##

tf = Tpr  # or any scalar final time you chose above
Yf = np.empty((6, M), dtype=float)

def propagate_to_tf(x0_vec, et0, et_final):
    t_vec, state_vec, _, _ = prop.propagate([et0, et_final], x0_vec)
    return state_vec[-1, :6]

for j in range(M):
    Yf[:, j] = propagate_to_tf(X0[:, j], et0, et_final)

print(Yf.shape)  # (6, M), each column is state at t = tf

# Fit AND get design matrix (basis evaluations) back
poly, C, Psi = regression(basis, Xi, Yf.T)
# here X0 for the sanity test
# Psi has shape (M, Nt): each column = Φ_k(ξ^(j)) evaluated at your samples
# Covariance from PCE coefficients with weights
Sigma_pce = covariance(Psi, C)
print("diag Σ_PCE:", Sigma_pce)
print("sqrt diag Σ_PCE (std):", np.sqrt(np.diag(Sigma_pce)))

# ============================================================
# VALIDATION: compare PCE covariance vs MC (Welford) at tf
# ============================================================

import glob
import math

# --- Welford combiner (same as in reducer) ---
def combine_triplet(nA, muA, M2A, nB, muB, M2B):
    if nA == 0:
        return nB, muB.copy(), M2B.copy()
    if nB == 0:
        return nA, muA.copy(), M2A.copy()

    n = nA + nB
    delta = muB - muA
    mu = muA + (nB / n) * delta
    M2 = M2A + M2B + np.outer(delta, delta) * (nA * nB / n)
    return n, mu, M2

def combine_epochwise(A, B):
    K = A['n_k'].shape[0]
    out = {
        'n_k':  np.zeros(K, dtype=np.int64),
        'mu_k': np.zeros((K, 6), dtype=np.float64),
        'M2_k': np.zeros((K, 6, 6), dtype=np.float64),
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
    return M2 / (n - 1) if n > 1 else np.full((6, 6), np.nan)

# --- 1) Load and merge MC partials ---
files_mc = sorted(glob.glob("mc_partial_job*_of_*.npz"))
if not files_mc:
    raise RuntimeError("No mc_partial_job*.npz files found for MC validation.")

partials_mc = []
meta_mc = None

for f in files_mc:
    dat = np.load(f)
    piece = {
        'n_k':  dat['n_k'],
        'mu_k': dat['mu_k'],
        'M2_k': dat['M2_k'],
    }
    partials_mc.append(piece)
    if meta_mc is None:
        meta_mc = {
            'K':          int(dat['K'][()]),
            't_idx_eval': dat['t_idx_eval'],
            'N2':         int(dat['N2'][()]),
        }

K_mc = meta_mc['K']
acc_mc = {
    'n_k':  np.zeros(K_mc, dtype=np.int64),
    'mu_k': np.zeros((K_mc, 6)),
    'M2_k': np.zeros((K_mc, 6, 6)),
}

for piece in partials_mc:
    acc_mc = combine_epochwise(acc_mc, piece)

# --- 2) Take MC covariance at final epoch (same tf) ---
epoch_idx_mc = K_mc - 1  # last epoch = final time
n_mc   = int(acc_mc['n_k'][epoch_idx_mc])
mu_mc  = acc_mc['mu_k'][epoch_idx_mc]
M2_mc  = acc_mc['M2_k'][epoch_idx_mc]
Sigma_mc = cov_from(n_mc, M2_mc)

# --- 3) Compare Σ_PCE vs Σ_MC ---
Sigma_diff = Sigma_pce - Sigma_mc
fro_mc = np.linalg.norm(Sigma_mc, ord='fro')
rel_err_F = np.linalg.norm(Sigma_diff, ord='fro') / max(1e-16, fro_mc)

det_pce = max(1e-300, np.linalg.det(Sigma_pce))
det_mc  = max(1e-300, np.linalg.det(Sigma_mc))
vol_ratio_pce_mc = math.sqrt(det_pce / det_mc)

print("\n=== PCE vs MC covariance at final time tf ===")
print("N_MC (final epoch) =", n_mc)
print("Frobenius rel. error =", rel_err_F)
print("Volume ratio √det(Σ_PCE)/√det(Σ_MC) =", vol_ratio_pce_mc)
print("trace Σ_MC =", np.trace(Sigma_mc))
print("trace Σ_PCE =", np.trace(Sigma_pce))
print("=============================================\n")
