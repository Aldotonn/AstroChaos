import numpy as np
import spiceypy as sp
from math import comb, ceil
from src.init.constants import MOON_SMA, MOON_OMEGA_MEAN
from src.coc.synodic_j2000 import synodic_to_j2000
from src.init.primary import Primary
from src.init.cr3bp import Cr3bp
from src.init.ephemeris import Ephemeris
from src.propagation.ephemeris_propagator import EphemerisSHPropagator
from src.init.load_kernels import load_kernels_moon_pot
from AstroChaos.Normal import normal
from AstroChaos.HermiteBase import hermite
from AstroChaos.FitRegression import regression
from AstroChaos.Covariance import covariance

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
                             l_c=MOON_SMA)
T = 375190.2  # [s]

Tpr = 2 * np.pi
t_span = [0, Tpr]
L = 384400

## --- ROTATION INTO J2000 ---##

et0 = sp.str2et('2019 JUN 18 12:00:00.000')*MOON_OMEGA_MEAN
et_final = et0 + Tpr

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
p_order = 3  # choose once and keep it consistent

# --- distribution and basis ---
xi_dist = normal(0, 1, d)
basis = hermite(p_order, xi_dist)    # orthonormal Hermite
Nt = len(basis)               # number of basis terms (== comb(d+p, p))
r  = 3                          # over-sampling ratio in [1.2, 2.0]
M  = max(Nt, int(ceil(r*Nt))) # ensure M ≥ Nt

# --- draw samples and map to initial conditions ---
Xi = xi_dist.sample(M, rule="sobol")   # shape (d, M)
X0 = r_mean[:, None] + L6 @ Xi        # shape (6, M)
Ya = np.zeros((M,6))

for j in range(M):
    t_vec, state_vec, _, _ = prop.propagate([et0, et_final], X0[:, j])
    Ya[j, :] = state_vec[-1, :6]



# Fit PCE at this snapshot (same basis and Xi)
poly_a, C_a, Psi_a =  regression(basis, Xi, Ya)

Sigma_pce_time = covariance(Psi_a, C_a)

print(Sigma_pce_time)