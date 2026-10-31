"""axi_fem.py -- axisymmetric 9-node (Q2) finite elements for a bonded layer under a centred circular punch (C4).

Independent of fem_layer.py (new code, gt_fem 2026-10-30).  Units mm, N, MPa.
Geometry: disc of radius R (lateral face traction-free), thickness t; z = 0 free face, z = t bonded (u = 0).
Unknowns (u_r, u_z) per node; strains (e_rr, e_zz, e_tt, g_rz) = (u_r,r, u_z,z, u_r/r, u_r,z + u_z,r).
Integrals include the 2 pi r weight, so reactions are total forces.
form="full": displacement Q2.  form="mixed": Q2/P1disc (B-bar with the dilatation projected, in the r-weighted
L2 inner product, onto span{1, r - rc, z - zc} per element; = Q2/P1disc with the pressure condensed).
Quadrature: 4 x 4 Gauss (the integrands contain 1/r from the hoop strain; nodes on the axis r = 0 have u_r = 0).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def q2_1d(xi):
    xi = np.asarray(xi, float)
    N = np.stack([0.5 * xi * (xi - 1), 1 - xi ** 2, 0.5 * xi * (xi + 1)], -1)
    dN = np.stack([xi - 0.5, -2 * xi, xi + 0.5], -1)
    return N, dN


def graded(lo, hi, sing, hmin, hmax, g):
    """Breaks on [lo, hi]: size hmin at the points in `sing`, growing by g, capped at hmax."""
    xs = np.linspace(lo, hi, 200001)
    d = np.min(np.abs(xs[:, None] - np.asarray(sing)[None, :]), 1) if len(sing) else np.full(xs.size, np.inf)
    h = np.minimum(hmax, hmin + (g - 1.0) * d)            # size grows linearly with distance -> ratio ~ g
    s = np.concatenate([[0.0], np.cumsum(0.5 * (1 / h[1:] + 1 / h[:-1]) * np.diff(xs))])
    n = max(1, int(np.ceil(s[-1])))
    b = np.interp(np.linspace(0, s[-1], n + 1), s, xs)
    b[0], b[-1] = lo, hi
    return b


def with_lines(lo, hi, lines, sing, hmin, hmax, g):
    """Graded breaks that contain the given lines exactly (each sub-interval graded separately)."""
    pts = sorted(set([lo, hi] + [l for l in lines if lo < l < hi]))
    out = [np.array([lo])]
    for a, b in zip(pts[:-1], pts[1:]):
        out.append(graded(a, b, [s for s in sing if a - 1e-12 <= s <= b + 1e-12], hmin, hmax, g)[1:])
    return np.concatenate(out)


def subdivide(b, m):
    return np.concatenate([np.linspace(b[i], b[i + 1], m + 1)[:-1] for i in range(len(b) - 1)] + [[b[-1]]])


class AxiGrid:
    def __init__(self, br, bz):
        self.br, self.bz = np.asarray(br, float), np.asarray(bz, float)
        self.ner, self.nez = len(br) - 1, len(bz) - 1
        self.nr, self.nz = 2 * self.ner + 1, 2 * self.nez + 1
        self.r = np.empty(self.nr); self.r[0::2] = br; self.r[1::2] = 0.5 * (br[1:] + br[:-1])
        self.z = np.empty(self.nz); self.z[0::2] = bz; self.z[1::2] = 0.5 * (bz[1:] + bz[:-1])
        self.nnode = self.nr * self.nz
        self.ndof = 2 * self.nnode

    def node(self, i, k):
        return i * self.nz + k


def stiffness(g, E, nu, form="full", ng=4):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu)); kap = lam + 2 * mu / 3
    xg, wg = np.polynomial.legendre.leggauss(ng)
    N1, dN1 = q2_1d(xg)                                    # (ng, 3)
    ie, ke = np.meshgrid(np.arange(g.ner), np.arange(g.nez), indexing="ij")
    ie, ke = ie.ravel(), ke.ravel()
    ne = ie.size
    r0, r1 = g.br[ie], g.br[ie + 1]; z0, z1 = g.bz[ke], g.bz[ke + 1]
    hr, hz = r1 - r0, z1 - z0
    # element nodes (a = 3*ar + az), global dofs
    ar, az = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    ar, az = ar.ravel(), az.ravel()
    nodes = g.node(2 * ie[:, None] + ar[None, :], 2 * ke[:, None] + az[None, :])       # (ne, 9)
    dofs = np.stack([2 * nodes, 2 * nodes + 1], -1).reshape(ne, 18)                 # (ur, uz) interleaved
    Ke = np.zeros((ne, 18, 18))
    if form == "mixed":
        Gp = np.zeros((ne, 3, 3)); Bp = np.zeros((ne, 3, 18))
    D = np.array([[lam + 2 * mu, lam, lam, 0], [lam, lam + 2 * mu, lam, 0], [lam, lam, lam + 2 * mu, 0],
                  [0, 0, 0, mu]])
    Ddev = mu * np.array([[4 / 3, -2 / 3, -2 / 3, 0], [-2 / 3, 4 / 3, -2 / 3, 0], [-2 / 3, -2 / 3, 4 / 3, 0],
                          [0, 0, 0, 1]])
    Dk = Ddev if form == "mixed" else D
    for p in range(ng):
        for q in range(ng):
            Nr, dNr = N1[p][ar], dN1[p][ar] * 2 / hr[:, None]           # (ne, 9)
            Nz, dNz = N1[q][az], dN1[q][az] * 2 / hz[:, None]
            N = Nr * Nz
            Nr_ = dNr * Nz; Nz_ = Nr * dNz
            r = 0.5 * (r0 + r1) + 0.5 * hr * xg[p]
            z = 0.5 * (z0 + z1) + 0.5 * hz * xg[q]
            w = wg[p] * wg[q] * 0.25 * hr * hz * 2 * np.pi * r
            B = np.zeros((ne, 4, 18))
            B[:, 0, 0::2] = Nr_
            B[:, 1, 1::2] = Nz_
            B[:, 2, 0::2] = N / r[:, None]
            B[:, 3, 0::2] = Nz_
            B[:, 3, 1::2] = Nr_
            Ke += w[:, None, None] * np.einsum("eik,ij,ejl->ekl", B, Dk, B)
            if form == "mixed":
                phi = np.stack([np.ones(ne), (r - 0.5 * (r0 + r1)) / hr, (z - 0.5 * (z0 + z1)) / hz], 1)
                Gp += w[:, None, None] * phi[:, :, None] * phi[:, None, :]
                div = B[:, 0] + B[:, 1] + B[:, 2]
                Bp += w[:, None, None] * phi[:, :, None] * div[:, None, :]
    if form == "mixed":
        Ke += kap * np.einsum("epk,epq,eql->ekl", Bp, np.linalg.inv(Gp), Bp)
    I = np.repeat(dofs, 18, 1).ravel(); J = np.tile(dofs, (1, 18)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (I, J)), shape=(g.ndof, g.ndof))
    return K


def solve(K, fixed, vals):
    n = K.shape[0]
    fixed = np.asarray(fixed); vals = np.asarray(vals, float)
    U = np.zeros(n); U[fixed] = vals
    mask = np.ones(n, bool); mask[fixed] = False
    free = np.nonzero(mask)[0]
    Kff = K[free][:, free].tocsc()
    rhs = -(K @ U)[free]
    lu = spla.splu(Kff, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))  # K_ff SPD
    U[free] = lu.solve(rhs)
    R = K @ U
    res = np.linalg.norm(R[free]) / max(np.linalg.norm(rhs), 1e-300)
    return U, R, res


def punch_bcs(g, a, bc="frictionless", delta=1.0, lateral="free"):
    I, Kk = np.meshgrid(np.arange(g.nr), np.arange(g.nz), indexing="ij")
    I, Kk = I.ravel(), Kk.ravel()
    nd = g.node(I, Kk)
    fx, fv = [], []
    back = nd[Kk == g.nz - 1]
    fx += [2 * back, 2 * back + 1]; fv += [np.zeros(back.size)] * 2
    axis = nd[(I == 0) & (Kk < g.nz - 1)]
    fx += [2 * axis]; fv += [np.zeros(axis.size)]
    pun = nd[(Kk == 0) & (g.r[I] <= a + 1e-9)]
    fx += [2 * pun + 1]; fv += [np.full(pun.size, delta)]
    if bc == "noslip":
        q = pun[g.r[pun // g.nz] > 0]
        fx += [2 * q]; fv += [np.zeros(q.size)]
    if lateral == "roller":
        q = nd[(I == g.nr - 1) & (Kk < g.nz - 1)]
        fx += [2 * q]; fv += [np.zeros(q.size)]
    d = np.concatenate(fx); v = np.concatenate(fv)
    d, ix = np.unique(d, return_index=True)
    return d, v[ix], pun
