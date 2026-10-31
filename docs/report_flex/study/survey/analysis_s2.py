# Survey session 2: closed-form numbers used in NOTES.md (no simulation).
#  A. Why third-party validations miss F1 (StVK lambda doubled): effective modulus ratio (lambda'=2 lambda vs lambda)
#     for each homogeneous load case. Uniaxial STRESS (free lateral faces: bending, cantilevers, unconfined blocks,
#     the nu=0 checks of MuJoCo's test and of badinkajink/hand) vs uniaxial STRAIN (laterally confined: our thin bonded
#     layer) vs hydrostatic vs shear.
#  B. Our layer: confined bounds M A/t, vertex mass, thickness-mode frequencies and omega*h at h = 0.5 ms
#     (roqsim's resolution limit omega*h <= 0.3; discrete adds Rayleigh damping equal to h: zeta=(beta+h)omega/2).
#  C. Arithmetic checks of numbers quoted from sources.
# usage: python analysis_s2.py -> analysis_s2.json
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
def Eu(lam, mu):  # uniaxial-stress modulus
    return mu * (3 * lam + 2 * mu) / (lam + mu)
A = {}
for nu in (0.0, 0.3, 0.4, 0.45, 0.49, 0.499):
    lam, mu = lame(1.0, nu)
    A[str(nu)] = dict(
        uniaxial_stress_E_ratio=round(Eu(2 * lam, mu) / Eu(lam, mu), 4),
        uniaxial_strain_M_ratio=round((2 * lam + 2 * mu) / (lam + 2 * mu), 4),
        hydrostatic_K_ratio=round((2 * lam + 2 * mu / 3) / (lam + 2 * mu / 3), 4),
        shear_ratio=1.0)
out["A_F1_ratio_by_load_case"] = A
E, rho, Lx, Ly, t, h = 1e6, 965.0, 0.051, 0.071, 0.0015, 5e-4
B = {}
for nu in (0.45, 0.49):
    lam, mu = lame(E, nu)
    M, M2 = lam + 2 * mu, 2 * lam + 2 * mu
    cp, cp2, cs = math.sqrt(M / rho), math.sqrt(M2 / rho), math.sqrt(mu / rho)
    wp, wp2, ws = [math.pi / 2 * c / t for c in (cp, cp2, cs)]   # fixed-free thickness modes
    B[str(nu)] = dict(K_confined_N_per_um=round(M * Lx * Ly / t * 1e-6, 3), K_confined_2lambda_N_per_um=round(M2 * Lx * Ly / t * 1e-6, 3),
                      K_shear_GA_t_N_per_um=round(mu * Lx * Ly / t * 1e-6, 4),
                      thickness_P_mode_kHz=round(wp / 2 / math.pi / 1e3, 2), thickness_P_mode_2lambda_kHz=round(wp2 / 2 / math.pi / 1e3, 2),
                      thickness_S_mode_kHz=round(ws / 2 / math.pi / 1e3, 2),
                      omega_h_P=round(wp * h, 1), omega_h_P_2lambda=round(wp2 * h, 1), omega_h_S=round(ws * h, 1),
                      zeta_discrete_numerical_S_mode_if_resolved=round(h * ws / 2, 2))
mass = rho * Lx * Ly * t
B["layer_mass_g"] = round(mass * 1e3, 3)
for g in ((11, 15, 3), (21, 29, 4), (41, 57, 5)):
    B[f"mean_vertex_mass_mg_{g[0]}x{g[1]}x{g[2]}"] = round(mass / (g[0] * g[1] * g[2]) * 1e6, 3)
out["B_our_layer"] = B
C = {}
C["badinkajink_block_uniaxial_nu0_um"] = 1.0 * 3e-3 / (1e7 * 16e-6) * 1e6   # F t/(E A), 4x4x3 mm, 1 N, E 10 MPa
lam, mu = lame(1e7, 0.45)
C["badinkajink_cap_cp_2lambda_m_s"] = round(math.sqrt((2 * lam + 2 * mu) / 1200), 1)
out["C_checks"] = C
json.dump(out, open(os.path.join(HERE, "analysis_s2.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
