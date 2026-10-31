# P0 patch tests from MuJoCo's own stiffness (FD of qfrc_spring, unpinned pad): impose exact homogeneous strain
# fields u = H x on ALL points and compare the strain energy 1/2 u'Ku with the exact W V (target E, nu):
#   uniaxial strain eps_zz=1: W = M/2;  simple shear u_x = y (gamma_xy=1): W = G/2;  hydrostatic eps=I: W = 9 K_b/2.
# Also interior nodal force residual |K u|_interior / |K u|_max (should be 0 for an exact patch test).
# Usage: python patch_check.py [nproc]  -> patch.json
import sys, os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

FIELDS = {"uniaxial_strain_zz": np.diag([0., 0., 1.]),
          "simple_shear_xy": np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
          "hydrostatic": np.eye(3)}


def exact_W(name, nu):
    lam, mu, M = lame(E_TARGET, nu)
    return {"uniaxial_strain_zz": M / 2, "simple_shear_xy": mu / 2, "hydrostatic": 4.5 * (lam + 2 * mu / 3)}[name]


def one(c):
    m, d, _ = load(c, pinned=False)
    P = points(m, d); el = elements(m, P)
    K, info = fd_stiffness(m, d, P, el)
    X = P["X"]; V = LX * LY * T
    g = P["g"]; cnt = P["cnt"]
    interior = np.all((g > 0) & (g < cnt - 1), axis=1)
    rec = dict(id=cfg_id(c), config=describe(c))
    for nm, H in FIELDS.items():
        u = (X @ H.T).ravel()
        f = K @ u
        U = 0.5 * u @ f
        fr = f.reshape(-1, 3)
        res = float(np.abs(fr[interior]).max() / np.abs(fr).max()) if interior.any() else None
        rec[nm] = dict(energy_ratio=float(U / (exact_W(nm, c["nu"]) * V)), interior_force_rel=res)
    return rec


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    from driver import configs
    cs = [c for c in configs() if c["nt"] <= 4 or (c["nt"] == 8 and c.get("nz", 0) in (3, 7)) or
          (c["nt"] == 8 and c.get("cz", 0) == 3)]
    with ProcessPoolExecutor(nproc) as ex:
        res = list(ex.map(one, cs))
    json.dump(to_json(res), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch.json"), "w"), indent=1)
    for r in res:
        print(r["id"], *["%s %.6f" % (k[:6], r[k]["energy_ratio"]) for k in FIELDS])
