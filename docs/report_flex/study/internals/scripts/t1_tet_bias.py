"""Item 1b: does the grid tetrahedralisation (6 tets per hex, one body diagonal for all hexes) bias the response?

flexcomp type="grid" splits every hex into 6 tets around the SAME body diagonal (user_mesh.cc:2703, cube2tets,
diagonal from local corner (1,0,0) to (0,1,1)) - a Kuhn/Freudenthal triangulation, not mirror symmetric.
We build the same layer as flexcomp type="direct" with (a) MuJoCo's pattern (checked identical to the grid)
and (b) the reflected ("J1", Union-Jack) pattern (each hex mirrored by the parity of its indices), take
MuJoCo's own assembled stiffness (efm_K / h^2 at rest, world-fixed parent) and solve static load cases on
the centre taxel (10.2 x 10.14 mm), back face pinned:
  normal load  -> spurious tangential displacement of the loaded patch (normal->shear cross-talk) and
                  mirror asymmetry of u_z;
  tangential load along (1,1)/sqrt2 vs (1,-1)/sqrt2 -> compliance ratio (1 for an unbiased mesh).
Material: (E, nu) = (1 MPa, 0.45) entered through the lambda remap (common.remap_full), so K is the
correct linear-elastic stiffness of the discretisation. Outputs: out/t1_tet_bias.json
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import mujoco

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load, dump, remap_full

A, B, T = 0.051, 0.071, 0.0015
CUBE2TETS = [[0, 3, 1, 7], [0, 1, 4, 7], [1, 3, 2, 7], [1, 2, 6, 7], [1, 5, 4, 7], [1, 6, 5, 7]]
CORNERS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def mesh(count, pattern):
    nx, ny, nz = count
    sp_ = (A / (nx - 1), B / (ny - 1), T / (nz - 1))
    pts = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                pts.append([sp_[0] * (ix - 0.5 * (nx - 1)), sp_[1] * (iy - 0.5 * (ny - 1)), sp_[2] * (iz - 0.5 * (nz - 1))])
    gid = lambda i, j, k: i * ny * nz + j * nz + k
    tets = []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                flip = (ix & 1, iy & 1, iz & 1) if pattern == "j1" else (0, 0, 0)
                vert = []
                for c in CORNERS:
                    cc = [c[a] ^ flip[a] for a in range(3)]
                    vert.append(gid(ix + cc[0], iy + cc[1], iz + cc[2]))
                for t in CUBE2TETS:
                    tets.append([vert[v] for v in t])
    return np.array(pts), np.array(tets)


def conforming(tets):
    faces = {}
    for t in tets:
        for f in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
            key = tuple(sorted(t[f]))
            faces[key] = faces.get(key, 0) + 1
    return max(faces.values()) <= 2


def build(count, pattern, E=1e6, nu=0.45):
    pts, tets = mesh(count, pattern)
    young, poisson = remap_full(E, nu)
    nz = count[2]
    pin = [i for i in range(len(pts)) if i % nz == nz - 1]  # back face (iz = nz-1)
    xml = f"""<mujoco><option integrator="discrete" solver="CG" timestep="0.0005" gravity="0 0 0"/>
<worldbody><body name="pdms" pos="0 0 0.1">
<flexcomp name="f" type="direct" dim="3" radius="0" mass="0.0052414"
 point="{' '.join(f'{v:.9g}' for v in pts.ravel())}" element="{' '.join(str(v) for v in tets.ravel())}">
<elasticity young="{young}" poisson="{poisson}" damping="0"/><contact contype="0" conaffinity="0"/>
<pin id="{' '.join(map(str, pin))}"/></flexcomp></body></worldbody></mujoco>"""
    m, _ = load(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    h = m.opt.timestep
    rows, cols, vals = [], [], []
    for i in range(m.nv):
        n, a = d.efm_K_rownnz[i], d.efm_K_rowadr[i]
        rows += [i] * n
        cols += list(d.efm_K_colind[a:a + n])
        vals += list(d.efm_K_val[a:a + n])
    K = sp.csr_matrix((np.array(vals) / h ** 2, (rows, cols)), shape=(m.nv, m.nv))
    X = d.flexvert_xpos[:m.flex_vertnum[0]].copy() - np.array([0, 0, 0.1])
    vb = m.flex_vertbodyid[:m.flex_vertnum[0]]
    dof = np.array([m.body_dofadr[b] if m.body_dofnum[b] == 3 else -1 for b in vb])
    return m, K, X, dof, tets, conforming(tets)


def solve_patch(K, X, dof, direction, count):
    """Uniform traction (unit total force) on the centre taxel of the free face (z = -T/2)."""
    nx, ny, nz = count
    face = np.isclose(X[:, 2], -T / 2) & (dof >= 0)
    hx, hy = A / (nx - 1), B / (ny - 1)
    # tributary area of each face vertex inside the patch (exact for a uniform traction on bilinear tiles)
    px, py = 0.0102 / 2, (B / 7) / 2
    def overlap(c, hw, h):
        lo, hi = max(c - h / 2, -hw), min(c + h / 2, hw)
        return max(0.0, hi - lo)
    f = np.zeros(K.shape[0])
    w = np.zeros(len(X))
    for v in np.where(face)[0]:
        w[v] = overlap(X[v, 0], px, hx) * overlap(X[v, 1], py, hy)
    w /= w.sum()
    for v in np.where(w > 0)[0]:
        f[dof[v]:dof[v] + 3] += w[v] * np.asarray(direction)
    u = spla.spsolve(K.tocsc(), f)
    U = np.zeros((len(X), 3))
    for v in range(len(X)):
        if dof[v] >= 0:
            U[v] = u[dof[v]:dof[v] + 3]
    patch = w > 0
    return U, w, patch, face


def mirror_asym(U, X, face):
    """max |u_z(x,y) - u_z(-x,y)| / max|u_z| over the free face (and same for y and xy mirrors)."""
    Xf, Uf = X[face], U[face]
    key = {(round(x * 1e7), round(y * 1e7)): uz for (x, y, _), uz in zip(Xf, Uf[:, 2])}
    out = {}
    for name, sx, sy in (("x", -1, 1), ("y", 1, -1), ("xy", -1, -1)):
        diffs = [abs(uz - key[(round(sx * x * 1e7), round(sy * y * 1e7))]) for (x, y, _), uz in zip(Xf, Uf[:, 2])
                 if (round(sx * x * 1e7), round(sy * y * 1e7)) in key]
        out[name] = float(max(diffs) / np.abs(Uf[:, 2]).max())
    return out


def analyse(count, pattern):
    m, K, X, dof, tets, conf = build(count, pattern)
    res = dict(count=count, pattern=pattern, conforming=bool(conf), nelem=int(len(tets)))
    U, w, patch, face = solve_patch(K, X, dof, (0, 0, 1), count)
    uz = (w[patch] * U[patch, 2]).sum()
    res["normal_load"] = dict(patch_uz_um_per_N=float(uz * 1e6),
                              crosstalk_ux_over_uz=float((w[patch] * U[patch, 0]).sum() / uz),
                              crosstalk_uy_over_uz=float((w[patch] * U[patch, 1]).sum() / uz),
                              mirror_asym=mirror_asym(U, X, face))
    comp = {}
    for name, dirn in (("t(1,1)", (1 / np.sqrt(2), 1 / np.sqrt(2), 0)), ("t(1,-1)", (1 / np.sqrt(2), -1 / np.sqrt(2), 0)),
                       ("t(1,0)", (1, 0, 0)), ("t(0,1)", (0, 1, 0))):
        U, w, patch, face = solve_patch(K, X, dof, dirn, count)
        ut = (w[patch, None] * U[patch]).sum(0)
        comp[name] = dict(compliance_um_per_N=float(ut[:2] @ np.array(dirn[:2]) * 1e6),
                          uz_over_ut=float(ut[2] / (ut[:2] @ np.array(dirn[:2]))))
    comp["ratio_(1,1)/(1,-1)"] = comp["t(1,1)"]["compliance_um_per_N"] / comp["t(1,-1)"]["compliance_um_per_N"]
    res["tangential_load"] = comp
    return res


def main():
    # the direct "kuhn" mesh must reproduce flexcomp type="grid": compare element sets
    from c1_contact_census import layer_xml  # noqa
    pts, tets = mesh((11, 15, 3), "kuhn")
    xml = layer_xml('<geom type="sphere" size="0.001" pos="1 1 1" contype="0" conaffinity="0"/>', 1.0)
    mg, _ = load(xml)
    tg = mg.flex_elem[:4 * mg.flex_elemnum[0]].reshape(-1, 4)
    same = {tuple(sorted(t)) for t in tg} == {tuple(sorted(t)) for t in tets}
    out = dict(direct_kuhn_equals_grid=bool(same), cases=[])
    print("direct 'kuhn' mesh == flexcomp grid elements:", same)
    for count in ((11, 15, 3), (21, 29, 5)):
        for pattern in ("kuhn", "j1"):
            r = analyse(count, pattern)
            out["cases"].append(r)
            print(r)
    p = dump("t1_tet_bias.json", out)
    print("wrote", p)


if __name__ == "__main__":
    main()
