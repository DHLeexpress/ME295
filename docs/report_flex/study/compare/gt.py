"""Unified ground-truth accessor for the compare task.

Sources (read only):
  ../gt_fem/results.json       finite-pad Q2 FEM (C1, C1f, C2, C3, C6, C6f; C4 axisymmetric)
  ../gt_transform/results.json infinite bonded layer, exact transform/BEM (C4, C5, C6, closed forms)
Policy (from the two READMEs and check_gt.py): C1, C1f, C2, C3 from gt_fem (pad edges matter);
C4, C5, C6, C6f from gt_transform (finite-pad effect < 2e-4, more accurate); gt_fem kept for cross-checks.
All taxel maps are returned as arrays [j][i] (7 rows along y, 5 columns along x); +z points into the layer.
Stiffness in N/um, forces per um of imposed displacement.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_FEM = None
_TR = None

A, B, T, E = 0.051, 0.071, 0.0015, 1e6


def fem():
    global _FEM
    if _FEM is None:
        _FEM = json.load(open(os.path.join(ROOT, "gt_fem", "results.json")))
    return _FEM


def tr():
    global _TR
    if _TR is None:
        _TR = json.load(open(os.path.join(ROOT, "gt_transform", "results.json")))
    return _TR


def nukey_fem(nu):
    return f"nu{nu:.3f}"


def nukey_tr(nu):
    return f"nu={nu:g}"


def closed(nu):
    G = E / (2 * (1 + nu))
    M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    return dict(G=G, M=M, MAt=M * A * B / T * 1e-6, GAt=G * A * B / T * 1e-6)


def get(case, nu):
    """Return dict(K, K_err, src, Fz_map (7x5, fraction of total normal force at the contact face) or None,
    uz_map (7x5, mean u_z/delta) or None, extra...)."""
    case = {"C6ns": "C6"}.get(case, case)
    out = dict(case=case, nu=nu)
    if case in ("C1", "C1f", "C2", "C3"):
        c = fem()["cases"][case][nukey_fem(nu)]
        out.update(K=c["K_N_per_um"], K_err=c["K_err_N_per_um"], src="gt_fem")
        F = np.array(c["taxel_force_N_per_um"])          # (7,5,3)
        comp = 1 if case == "C2" else 2
        out["F_map"] = F[:, :, comp] / F[:, :, comp].sum()
        out["F_map_err"] = np.array(c["taxel_force_err_N_per_um"])[:, :, comp] / F[:, :, comp].sum()
        out["uz_map"] = np.array(c["taxel_mean_disp_per_delta"])[:, :, 2]
        out["profile_y_mm"] = np.array(c["profile_y_mm"])
        out["uz_x0"] = np.array(c["u_free_at_x0_per_delta"])[:, 2]
        out["profile_err_max"] = c["profile_err_max"]
        if case == "C3":
            out["Fy_over_Fz"] = c["F_xyz_N_per_um"][1] / c["F_xyz_N_per_um"][2]
        return out
    if case in ("C6", "C6f"):
        sub = "noslip" if case == "C6" else "frictionless"
        c = tr()[nukey_tr(nu)]["C6"][sub]
        out.update(K=c["K_N_per_um"], K_err=c["K_N_per_um"] * max(c["rel_err_est"], 1e-5), src="gt_transform")
        tm = c["taxel_maps"]
        Fs = np.array(tm["surface_contact_force_per_um_N"]).transpose(1, 0, 2)   # -> [j][i][3]
        out["F_map"] = Fs[:, :, 2] / Fs[:, :, 2].sum()
        out["uz_map"] = np.array(tm["mean_surface_displacement_per_delta"]).transpose(1, 0, 2)[:, :, 2]
        out["neigh_edge_uz"] = float(np.mean([out["uz_map"][3, 1], out["uz_map"][3, 3],
                                              out["uz_map"][2, 2], out["uz_map"][4, 2]]))
        # cross-check value from gt_fem
        cf = fem()["cases"][case][nukey_fem(nu)]
        out["K_fem"], out["K_fem_err"] = cf["K_N_per_um"], cf["K_err_N_per_um"]
        out["uz_map_fem"] = np.array(cf["taxel_mean_disp_per_delta"])[:, :, 2]
        return out
    if case.startswith("C4"):
        # C4 / C4f: frictionless a=5 mm; C4ns: bonded; suffix _a2.5 for the small punch
        a = "a=2.5mm" if "a2.5" in case else "a=5mm"
        sub = "bonded_noslip" if "ns" in case else "frictionless"
        c = tr()[nukey_tr(nu)]["C4"][a][sub]
        out.update(K=c["K_N_per_um"], K_err=c["K_N_per_um"] * max(c["rel_err_est"], 1e-6), src="gt_transform")
        return out
    if case.startswith("C5"):
        R = "R=25mm" if "R25" in case else "R=10mm"
        c = tr()[nukey_tr(nu)]["C5"][R]
        out.update(delta_um=np.array(c["delta_um"]), F_N=np.array(c["F_N"]),
                   a_mm=np.array(c["contact_radius_mm"]), hertz_F_N=np.array(c["hertz_F_N"]), src="gt_transform")
        return out
    raise KeyError(case)


def c5_F(delta_um, nu=0.45, R="R10"):
    g = get("C5" + ("_R25" if R == "R25" else ""), nu)
    return float(np.interp(delta_um, g["delta_um"], g["F_N"])), float(np.interp(delta_um, g["delta_um"], g["a_mm"]))


def neigh_edge(uzmap):
    """mean u_z/delta of the 4 edge neighbours of the centre taxel (i=2, j=3), map [j][i]."""
    u = np.asarray(uzmap)
    return float(np.mean([u[3, 1], u[3, 3], u[2, 2], u[4, 2]]))


if __name__ == "__main__":
    for nu in (0.45, 0.49, 0.499):
        for c in ("C1", "C1f", "C2", "C3", "C6", "C6f", "C4", "C4ns", "C4_a2.5"):
            g = get(c, nu)
            print(nu, c, round(g["K"], 6), f"{g['K_err']:.1e}", g["src"])
        print(nu, "C5 F(30um), a", c5_F(30, nu))
