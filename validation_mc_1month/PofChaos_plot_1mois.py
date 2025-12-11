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
from AstroChaos.Normal import normal
from AstroChaos.HermiteBase import hermite
from AstroChaos.FitRegression import regression
from AstroChaos.Covariance import covariance
import glob
import math
import matplotlib.pyplot as plt


# ============================================================
#  VALIDATION: compare PCE covariance vs MC (Welford) nel tempo
# ============================================================

import json

def save_matrix_json(filename, matrix):
    """
    Save a numpy matrix (or array) to a JSON file.
    Converts to a Python list to ensure JSON compatibility.
    """
    with open(filename, "w") as f:
        json.dump(matrix.tolist(), f, indent=2)

# --- Welford combiner (come prima) ---
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
    Kloc = A['n_k'].shape[0]
    out = {
        'n_k':   np.zeros(Kloc, dtype=np.int64),
        'mu_k':  np.zeros((Kloc,6), dtype=np.float64),
        'M2_k':  np.zeros((Kloc,6,6), dtype=np.float64),
    }
    for a in range(Kloc):
        n, mu, M2 = combine_triplet(
            int(A['n_k'][a]), A['mu_k'][a], A['M2_k'][a],
            int(B['n_k'][a]), B['mu_k'][a], B['M2_k'][a]
        )
        out['n_k'][a]  = n
        out['mu_k'][a] = mu
        out['M2_k'][a] = M2
    return out

def cov_from(n, M2):
    return M2 / (n - 1) if n > 1 else np.full((6,6), np.nan)

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
            'K': int(dat['K'][()]),
            't_idx_eval': dat['t_idx_eval'],
            'N2': int(dat['N2'][()]),
        }

K_mc = meta_mc['K']


acc_mc = {
    'n_k':  np.zeros(K_mc, dtype=np.int64),
    'mu_k': np.zeros((K_mc,6)),
    'M2_k': np.zeros((K_mc,6,6)),
}
for piece in partials_mc:
    acc_mc = combine_epochwise(acc_mc, piece)

# --- 2) MC covariance time series Σ_MC(t_k) ---
Sigma_mc_time = np.zeros((K_mc, 6, 6))
for a in range(K_mc):
    n_a  = int(acc_mc['n_k'][a])
    M2_a = acc_mc['M2_k'][a]
    Sigma_mc_time[a] = cov_from(n_a, M2_a)

# MC covariance al tempo finale tf (per mantenere le stampe finali)
epoch_idx_mc = K_mc - 1
Sigma_mc = Sigma_mc_time[epoch_idx_mc]
n_mc   = int(acc_mc['n_k'][epoch_idx_mc])
mu_mc  = acc_mc['mu_k'][epoch_idx_mc]


#####------PCE--------#####

if not hasattr(np, "bool"):  # compatibility for NumPy >=1.24
    np.bool = np.bool_

load_kernels_moon_pot()

## --- INITIAL STATE --- ##

r_mean = np.array([
    9.87848892e-01,  -1.47371321e-06,  4.91889975e-03,   # [x, y, z] [adim]
    1.60283536e+00,   9.84390573e-01,  4.13915823e-05    # [vx, vy, vz] [adim]
])

## --- PROP INIT --- ##

cr3bp = Cr3bp(Primary.EARTH, Primary.MOON)
ephemeris = Ephemeris((Primary.EARTH, Primary.MOON, Primary.SUN),
                      moon=("GRGM0100", 50, 50, False))
prop = EphemerisSHPropagator(ephemeris,
                             with_stm=True,
                             obs=Primary.EARTH,
                             t_c=1. / MOON_OMEGA_MEAN,
                             l_c=MOON_SMA,
                             time_steps=1000000)
T = 375190.2  # [s]
#Tpr = (182*24*60*60*MOON_OMEGA_MEAN)
#Tpr = 7 *24* 60*60*MOON_OMEGA_MEAN
Tpr = 2 * np.pi
t_span = [0, Tpr]
L = 384400

## --- ROTATION INTO J2000 ---##

et0 = sp.str2et('2019 JUN 18 12:00:00.000')*MOON_OMEGA_MEAN
et_final = et0 + Tpr
# DOPO (ok con Numba):
t0_nd = np.array([0.0], dtype=np.float64) # array 1D float64, non lista int
state_matrix = np.ascontiguousarray(
    np.array(r_mean, dtype=np.float64).reshape(1, 6)
)

t_patch, state_0 = synodic_to_j2000(
    t0_nd, state_matrix, float(et0), cr3bp, cr3bp.m1,
    adim=True, bary_from_spice=True
)

r_mean = state_0.reshape(6,)
print(state_0)

## --- POSITION COVARIANCE --- ##
sigma_r = 10.0/L # [km] standard deviation
rho_xy, rho_xz, rho_yz = 0.6/L, 0.3/L, 0.5/L

P_rr = np.array([
    [sigma_r**2,        rho_xy*sigma_r**2, rho_xz*sigma_r**2],
    [rho_xy*sigma_r**2, sigma_r**2,        rho_yz*sigma_r**2],
    [rho_xz*sigma_r**2, rho_yz*sigma_r**2, sigma_r**2       ]
])

## --- VELOCITY COVARIANCE --- ##
v_nom = r_mean[3:]                     # nominal velocity components [km/s]
sigma_v = 0.02 * np.abs(v_nom)         # 2% relative uncertainty
rho_vxy, rho_vxz, rho_vyz = 0.2, 0.2, 0.2

P_vv = np.array([
    [sigma_v[0]**2,             rho_vxy*sigma_v[0]*sigma_v[1], rho_vxz*sigma_v[0]*sigma_v[2]],
    [rho_vxy*sigma_v[0]*sigma_v[1], sigma_v[1]**2,            rho_vyz*sigma_v[1]*sigma_v[2]],
    [rho_vxz*sigma_v[0]*sigma_v[2], rho_vyz*sigma_v[1]*sigma_v[2], sigma_v[2]**2           ]
])

## --- CROSS TERMS (pos–vel) --- ##
P_rv = np.zeros((3,3))
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
basis = hermite(p_order, xi_dist)    # orthonormal Hermite
Nt = len(basis)               # number of basis terms (== comb(d+p, p))
r  = 3                          # over-sampling ratio in [1.2, 2.0]
M  = max(Nt, int(ceil(r*Nt))) # ensure M ≥ Nt

print(f"p={p_order}, Nt={Nt}, M={M}")   # for d=6: p=5 → Nt, etc.

# --- draw samples and map to initial conditions ---
Xi = xi_dist.sample(M, rule="sobol")   # shape (d, M)
X0 = r_mean[:, None] + L6 @ Xi        # shape (6, M)

print("Number of terms in basis =", len(basis))


# ============================================================
#  RECUPERO GRIGLIA TEMPORALE (K, t_idx_eval) DALLA MC
#  (la usiamo per PCE, stessi istanti della MC)
# ============================================================

files_mc_meta = sorted(glob.glob("mc_partial_job*_of_*.npz"))
if not files_mc_meta:
    raise RuntimeError("No mc_partial_job*.npz files found to get time snapshots.")

meta0 = np.load(files_mc_meta[0])
K = int(meta0['K'][()])
t_idx_eval = meta0['t_idx_eval'].astype(int)
print(f"PCE: user snapshot grid K={K} from MC partials.")

T_nd_tot = float(meta0['et_final'] - meta0['et0'])

t_idx = meta0['t_idx_eval'].astype(int)
max_idx = int(t_idx.max())           # ≈ Tlen-1 (ultimo step dell'integrazione)

# tempo adimensionale relativo di ogni snapshot (0 → T_nd_tot)
tau_snap = T_nd_tot * (t_idx / max_idx)
t_snap_abs = et0 + tau_snap
# ============================================================
#  Propagation of Random Samples  -> stati ai K istanti
# ============================================================

# Y[a, :, j] = stato (6) della realizzazione j all'istante t_idx_eval[a]
Y = np.empty((K, 6, M), dtype=float)



print("Propagating PCE samples at all snapshots...")
for j in range(M):
    if (j+1) % 10 == 0 or j == 0:
        print(f"  sample {j+1}/{M}")
    t_vec, state_vec, _, _ = prop.propagate([et0, et_final], X0[:, j])

    if j == 0:
        if t_idx_eval.max() >= len(t_vec):
            raise RuntimeError("t_idx_eval exceeds length of t_vec from prop.propagate!")

    # trova indici in t_vec più vicini ai tempi tau_snap
    idx = np.searchsorted(t_vec, t_snap_abs)
    idx = np.clip(idx, 0, len(t_vec) - 1)
    # salva stati pce agli stessi istanti della MC
    Y[:, :, j] = state_vec[idx, :6]

print("Y.shape (K, 6, M) =", Y.shape)


# ============================================================
#  PCE COVARIANCE Σ_PCE(t_k) FOR EACH SNAPSHOT
# ============================================================

Sigma_pce_time = np.zeros((K, 6, 6))

print("Computing PCE covariance time series...")
for a in range(K):
    # target for regression at snapshot a: shape (M, 6)
    Ya = Y[a,:,:].T   # (6, M) -> (M, 6)

    # Fit PCE at this snapshot (same basis and Xi)
    poly_a, C_a, Psi_a =  regression(basis, Xi, Ya)

    Sigma_pce_time[a] = covariance(Psi_a, C_a)

print("Sigma_pce_time.shape =", Sigma_pce_time.shape)

# Covarianza PCE al tempo finale (per mantenere il check che avevi già)
Sigma_pce = Sigma_pce_time[-1]
print("diag Σ_PCE(tf):", np.diag(Sigma_pce))
print("sqrt diag Σ_PCE(tf) (std):", np.sqrt(np.diag(Sigma_pce)))



# --- 3) Frobenius norm vs time ---
Delta_time = Sigma_pce_time - Sigma_mc_time          # (K, 6, 6)
fro_err = np.linalg.norm(Delta_time, axis=(1, 2))   # |ΔΣ|_F per istante

fro_mc = np.linalg.norm(Sigma_mc_time, axis=(1, 2))
rel_err_time = np.where(fro_mc > 0, fro_err / fro_mc, np.nan)

print("\n=== Frobenius error over time ===")
print("max ||ΔΣ(t)||_F =", np.nanmax(fro_err))
print("mean||ΔΣ(t)||_F =", np.nanmean(fro_err))

# --- 4) plot errore di Frobenius vs snapshot index (o vs tempo se lo ricostruisci) ---
plt.figure()
plt.plot(
    np.arange(K),
    rel_err_time,
    label=r"$\|\Sigma_{\rm PCE}(t_k)-\Sigma_{\rm MC}(t_k)\|_F \,/\, \|\Sigma_{\rm MC}(t_k)\|_F$"
)
plt.xlabel("Snapshot index k")
plt.ylabel("Frobenius norm of covariance difference")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("frobenius_error_vs_time.png", dpi=200)
plt.show()

# --- 5) Riepilogo finale al tempo tf (come avevi prima) ---
Sigma_diff_tf = Sigma_pce - Sigma_mc


# Save full time series (K snapshots)
save_matrix_json("Sigma_MC_time.json",  Sigma_mc_time)
save_matrix_json(f"Sigma_PCE_time_order{p_order}.json", Sigma_pce_time)

print("Saved: Sigma_MC_time.json and Sigma_PCE_time.json")

save_matrix_json("Sigma_MC_tf.json",  Sigma_mc)
save_matrix_json(f"Sigma_PCE_tf_order{p_order}.json", Sigma_pce)

print("Saved: Sigma_MC_tf.json and Sigma_PCE_tf.json")


fro_mc_tf = np.linalg.norm(Sigma_mc, ord='fro')
rel_err_F_tf = np.linalg.norm(Sigma_diff_tf, ord='fro') / max(1e-16, fro_mc_tf)

det_pce = max(1e-300, np.linalg.det(Sigma_pce))
det_mc  = max(1e-300, np.linalg.det(Sigma_mc))
vol_ratio_pce_mc = math.sqrt(det_pce / det_mc)

print("\n=== PCE vs MC covariance at final time tf ===")
print("N_MC (final epoch)                  =", n_mc)
print("Frobenius rel. error at tf          =", rel_err_F_tf)
print("Volume ratio √det(Σ_PCE)/√det(Σ_MC) =", vol_ratio_pce_mc)
print("trace Σ_MC(tf)                      =", np.trace(Sigma_mc))
print("trace Σ_PCE(tf)                     =", np.trace(Sigma_pce))
print("=============================================\n")
print("Frobenius error vs time saved in frobenius_error_vs_time.png")

import logging
logging.getLogger('comtypes').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

import pyttsx3
engine = pyttsx3.init()
engine.say("Ho finito!")
engine.runAndWait()