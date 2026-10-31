"""fem_layer.py -- static, small-strain, linear-elastic finite-element reference for a thin layer
bonded to a rigid plate (the PDMS skin of the blade), written independently of MuJoCo.

Discretisation
  * 27-node (triquadratic Lagrange, "Q2") hexahedra on tensor-product box meshes (graded per axis).
  * form="full" : displacement formulation, exact integration (= full 3x3x3 Gauss on boxes).
    The global matrix is built from Kronecker products of 1D quadratic-element matrices, which is
    algebraically identical to element-by-element 3x3x3 Gauss assembly on box elements
    (checked against `stiffness_elementwise`, an independent isoparametric implementation).
  * form="mixed": Q2/P1disc mixed element (B-bar): the volumetric strain is L2-projected onto
    discontinuous linear polynomials {1, xi, eta, zeta} per element and the pressure is condensed;
    deviatoric part fully integrated.  Inf-sup stable, locking-free as nu -> 0.5.
Units: mm, N, MPa (= N/mm^2).  Stiffness in N/mm (1 N/mm = 1e-3 N/um).

Review log of this copy (gt_fem, 2026-10-30; original: ME295 scratchpad fem_ref/fem_layer.py, kept here as
fem_layer_orig.py).  Checked, not trusted: see verify_fem.py / verify_fem.json.
  * stiffness(): derivation re-done by hand: a(u,v) = int lam div u div v + 2 mu eps(u):eps(v) gives the
    (i,j) block lam G_ij + mu G_ji + delta_ij mu sum_k G_kk with G_ij[A,B] = int d_i N_A d_j N_B -- as coded.
    Mixed (B-bar) form: deviatoric part -2mu/3 G_ij + mu G_ji + delta_ij mu L plus kappa * (P1disc projection of
    div u)^2; {1, xi, eta, zeta} are L2-orthogonal on a box, with norms vol and vol/3 -- as coded.
  * Kronecker assembly == element-by-element isoparametric assembly to 1e-15 (both forms, nu up to 0.499).
  * NDCholesky == SuperLU to 1e-13 on a 25k-DOF layer problem.
  * CHANGED graded_breaks(): the cumulative mesh density was integrated on a uniform 4001-point sample, which
    is coarser than hmin for hmin < (hi-lo)/4000 (the first elements were then wrong); it is now integrated on
    a sample that is geometric near the singular points (first element size checked in verify_fem.py).
  * CHANGED solve_dirichlet(): removed the duplicated set-up for method="nd" (results unchanged), added
    `threads` (BLAS threads, via threadpoolctl) and the reaction sum check.
  * ADDED: separable body-force loads and tensor-product evaluation of u_h and grad u_h (method of manufactured
    solutions in verify_fem.py), node/DOF helpers, face-traction maps on either face.
Coordinates: x across (51 mm), y along (71 mm), z through the thickness; z = 0 free face,
z = t bonded face.  Node index A = (ia*ny + ib)*nz + ic; DOF = 3*A + component.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------------- 1D quadratic element
_GX, _GW = np.polynomial.legendre.leggauss(4)          # exact for the polynomial degrees used here


def q2_basis(xi):
    """Quadratic Lagrange functions on [-1, 1] with nodes (-1, 0, 1): values and d/dxi."""
    xi = np.asarray(xi, dtype=float)
    N = np.stack([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)], axis=-1)
    dN = np.stack([xi - 0.5, -2.0 * xi, xi + 0.5], axis=-1)
    return N, dN


_N, _dN = q2_basis(_GX)
REF_M = np.einsum("g,ga,gb->ab", _GW, _N, _N)        # int N_a N_b dxi
REF_K = np.einsum("g,ga,gb->ab", _GW, _dN, _dN)      # int N_a' N_b' dxi
REF_D = np.einsum("g,ga,gb->ab", _GW, _dN, _N)       # int N_a' N_b dxi
REF_E0 = np.einsum("g,ga->a", _GW, _N)               # int N_a dxi
REF_E0p = np.einsum("g,ga->a", _GW, _dN)             # int N_a' dxi
REF_E1 = np.einsum("g,g,ga->a", _GW, _GX, _N)        # int xi N_a dxi
REF_E1p = np.einsum("g,g,ga->a", _GW, _GX, _dN)      # int xi N_a' dxi


def lame(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu, lam + 2.0 * mu / 3.0


def confined_modulus(E, nu):
    return E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))


def shear_modulus(E, nu):
    return E / (2.0 * (1.0 + nu))


class Mesh1D:
    """Quadratic 1D mesh: element boundaries `breaks`; nodes = boundaries + midpoints."""

    def __init__(self, breaks):
        b = np.asarray(breaks, dtype=float)
        if np.any(np.diff(b) <= 0):
            raise ValueError("breaks must increase")
        self.breaks, self.ne, self.h = b, len(b) - 1, np.diff(b)
        self.n = 2 * self.ne + 1
        x = np.empty(self.n)
        x[0::2] = b
        x[1::2] = 0.5 * (b[:-1] + b[1:])
        self.x = x
        self.conn = 2 * np.arange(self.ne)[:, None] + np.arange(3)[None, :]

    def mat(self, kind, elems=None):
        """Global n x n matrix: 'M' int phi_a phi_b, 'K' int phi_a' phi_b', 'D' int phi_a' phi_b."""
        h = self.h[:, None, None]
        if kind == "M":
            e = 0.5 * h * REF_M
        elif kind == "K":
            e = (2.0 / h) * REF_K
        elif kind == "D":
            e = np.ones_like(h) * REF_D
        else:
            raise KeyError(kind)
        conn = self.conn
        if elems is not None:
            e, conn = e[elems], conn[elems]
        rows = np.broadcast_to(conn[:, :, None], e.shape).ravel()
        cols = np.broadcast_to(conn[:, None, :], e.shape).ravel()
        return sp.csr_matrix((e.ravel(), (rows, cols)), shape=(self.n, self.n))

    def efun(self, kind):
        """Element x node functionals: E0 int_e phi_a, E0p int_e phi_a', E1 int_e xi phi_a,
        E1p int_e xi phi_a' (xi = local coordinate in [-1, 1])."""
        h = self.h[:, None]
        v = {"E0": 0.5 * h * REF_E0, "E0p": np.ones_like(h) * REF_E0p,
             "E1": 0.5 * h * REF_E1, "E1p": np.ones_like(h) * REF_E1p}[kind]
        rows = np.repeat(np.arange(self.ne), 3)
        return sp.csr_matrix((v.ravel(), (rows, self.conn.ravel())), shape=(self.ne, self.n))

    def weights(self, lo, hi, tol=1e-9):
        """w[a] = int_{[lo,hi]} phi_a dx; [lo,hi] must be a union of whole elements."""
        sel = (self.breaks[:-1] >= lo - tol) & (self.breaks[1:] <= hi + tol)
        w = np.zeros(self.n)
        if not sel.any():
            return w
        np.add.at(w, self.conn[sel].ravel(), (0.5 * self.h[sel, None] * REF_E0).ravel())
        # sanity: covered length must equal hi - lo
        if abs(self.h[sel].sum() - (hi - lo)) > 1e-7:
            raise ValueError("interval [%g, %g] is not a union of mesh elements" % (lo, hi))
        return w

    def node(self, x, tol=1e-7):
        i = int(np.argmin(np.abs(self.x - x)))
        if abs(self.x[i] - x) > tol:
            raise ValueError("no node at %g" % x)
        return i

    def eval_basis(self, x0):
        """(node indices, values) of the global basis at point x0."""
        e = int(np.clip(np.searchsorted(self.breaks, x0, side="right") - 1, 0, self.ne - 1))
        xi = 2.0 * (x0 - self.breaks[e]) / self.h[e] - 1.0
        N, _ = q2_basis(xi)
        return self.conn[e], N


# ----------------------------------------------------------------------------- mesh generation
def graded_breaks(lo, hi, lines, singular, hmin, hmax, growth):
    """Element boundaries on [lo, hi] that contain every coordinate in `lines`.
    Target size h(x) = min(hmax, hmin + (growth - 1) * dist(x, singular)) (geometric grading
    towards the singular lines; hmax where there are none).  Between consecutive lines the number of
    elements is n = ceil(int dx / h(x)) and the boundaries are equidistributed in int dx / h(x).
    The density integral is evaluated on a sample that is geometric near the singular points (exact to
    the sampling of a piecewise-smooth 1/h; checked against the closed form in verify_fem.py)."""
    pts = sorted(set([float(lo), float(hi)] + [float(p) for p in lines if lo + 1e-9 < p < hi - 1e-9]))
    sing = np.asarray([s for s in singular if lo - 1e-9 <= s <= hi + 1e-9], dtype=float)

    def hfun(x):
        if sing.size == 0:
            return np.full_like(x, hmax)
        d = np.min(np.abs(x[:, None] - sing[None, :]), axis=1)
        return np.minimum(hmax, hmin + (growth - 1.0) * d)

    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        s = [np.linspace(a, b, 20001)]
        for c in sing:
            d = hmin * np.geomspace(1e-4, max(1e-3, 1e3 * (b - a) / hmin), 4000)
            s += [c - d, c + d, [c]]
        s = np.unique(np.clip(np.concatenate(s), a, b))
        dens = 1.0 / hfun(s)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(s))])
        n = max(1, int(np.ceil(cum[-1] - 1e-6)))
        xb = np.interp(np.linspace(0.0, cum[-1], n + 1), cum, s)
        xb[0], xb[-1] = a, b
        out.extend(xb[1:].tolist())
    return np.asarray(out)


def subdivide(breaks, m):
    """Split every element into m equal elements (nested, geometrically similar refinement)."""
    b = np.asarray(breaks, dtype=float)
    if m == 1:
        return b.copy()
    out = [b[0]]
    for a, c in zip(b[:-1], b[1:]):
        out.extend((a + (c - a) * np.arange(1, m + 1) / m).tolist())
    return np.asarray(out)


class Grid3D:
    def __init__(self, bx, by, bz):
        self.m1 = [Mesh1D(bx), Mesh1D(by), Mesh1D(bz)]
        self.nx, self.ny, self.nz = (m.n for m in self.m1)
        self.N = self.nx * self.ny * self.nz
        self.ndof = 3 * self.N
        self.nel = self.m1[0].ne * self.m1[1].ne * self.m1[2].ne

    def node(self, ia, ib, ic):
        return (np.asarray(ia) * self.ny + np.asarray(ib)) * self.nz + np.asarray(ic)

    def coords(self):
        X, Y, Z = np.meshgrid(self.m1[0].x, self.m1[1].x, self.m1[2].x, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _kron3(a, b, c):
    return sp.kron(sp.kron(a, b, format="csr"), c, format="csr")


def _interleave(blocks):
    """3x3 list of N x N sparse blocks -> 3N x 3N matrix with DOF = 3*node + component."""
    rows, cols, vals = [], [], []
    for i in range(3):
        for j in range(3):
            B = blocks[i][j].tocoo()
            rows.append(3 * B.row.astype(np.int64) + i)
            cols.append(3 * B.col.astype(np.int64) + j)
            vals.append(B.data)
    n = 3 * blocks[0][0].shape[0]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def _interleave_cols(cols):
    """list of 3 (m x N) matrices -> m x 3N with column 3*node + component."""
    rows_, cols_, vals_ = [], [], []
    for i, C in enumerate(cols):
        C = C.tocoo()
        rows_.append(C.row)
        cols_.append(3 * C.col.astype(np.int64) + i)
        vals_.append(C.data)
    return sp.csr_matrix((np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))),
                         shape=(cols[0].shape[0], 3 * cols[0].shape[1]))


def gradient_products(grid):
    """G[i][j][A, B] = int d_i N_A d_j N_B dV as Kronecker products of 1D matrices."""
    mats = [{k: m.mat(k) for k in "MKD"} for m in grid.m1]
    G = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            fs = []
            for d in range(3):
                if d == i and d == j:
                    fs.append(mats[d]["K"])
                elif d == i:
                    fs.append(mats[d]["D"])
                elif d == j:
                    fs.append(mats[d]["D"].T.tocsr())
                else:
                    fs.append(mats[d]["M"])
            G[i][j] = _kron3(*fs)
    return G


def volumetric_projection(grid):
    """V[(A,i),(B,j)] = sum_e sum_k (int_e q_k d_i N_A)(int_e q_k d_j N_B) / int_e q_k^2,
    q = {1, xi, eta, zeta} on each element (P1disc projection of div u)."""
    F = [{k: m.efun(k) for k in ("E0", "E0p", "E1", "E1p")} for m in grid.m1]
    vol = np.kron(np.kron(grid.m1[0].h, grid.m1[1].h), grid.m1[2].h)
    V = None
    for k in range(4):
        cols = []
        for i in range(3):
            fs = []
            for d in range(3):
                if d == i:
                    fs.append(F[d]["E1p"] if k == d + 1 else F[d]["E0p"])
                else:
                    fs.append(F[d]["E1"] if k == d + 1 else F[d]["E0"])
            cols.append(_kron3(*fs))
        Bk = _interleave_cols(cols)
        w = 1.0 / (vol * (1.0 if k == 0 else 1.0 / 3.0))
        term = (Bk.T @ sp.diags(w) @ Bk).tocsr()
        V = term if V is None else V + term
    return V


def stiffness(grid, E, nu, form="full"):
    lam, mu, kappa = lame(E, nu)
    G = gradient_products(grid)
    L = G[0][0] + G[1][1] + G[2][2]
    blocks = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if form == "full":
                B = lam * G[i][j] + mu * G[j][i]
            elif form == "mixed":
                B = (-2.0 * mu / 3.0) * G[i][j] + mu * G[j][i]
            else:
                raise KeyError(form)
            if i == j:
                B = B + mu * L
            blocks[i][j] = B
    K = _interleave(blocks)
    if form == "mixed":
        K = K + kappa * volumetric_projection(grid)
    K.sum_duplicates()
    return K.tocsr()


# ----------------------------------------------------------------------------- independent check
def _hex27_tables():
    gx, gw = np.polynomial.legendre.leggauss(3)
    N1, dN1 = q2_basis(gx)
    Nv = np.einsum("ia,jb,kc->ijkabc", N1, N1, N1).reshape(27, 27)
    dX = np.einsum("ia,jb,kc->ijkabc", dN1, N1, N1).reshape(27, 27)
    dY = np.einsum("ia,jb,kc->ijkabc", N1, dN1, N1).reshape(27, 27)
    dZ = np.einsum("ia,jb,kc->ijkabc", N1, N1, dN1).reshape(27, 27)
    W = np.einsum("i,j,k->ijk", gw, gw, gw).ravel()
    xi = np.stack(np.meshgrid(gx, gx, gx, indexing="ij"), -1).reshape(27, 3)
    return Nv, np.stack([dX, dY, dZ], axis=1), W, xi        # dN: (gp, 3, node)


def _dmat(E, nu):
    lam, mu, kappa = lame(E, nu)
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] = lam + 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    m = np.r_[1, 1, 1, 0, 0, 0].astype(float)
    return D, D - kappa * np.outer(m, m), kappa


def stiffness_elementwise(grid, E, nu, form="full", coords=None):
    """Element-by-element isoparametric assembly (27-node hex, 3x3x3 Gauss, Voigt B-matrix,
    explicit P1 projection for form='mixed').  Slow; for verification on small meshes, also on
    distorted meshes (pass `coords`)."""
    D, Ddev, kappa = _dmat(E, nu)
    Nv, dNr, W, xig = _hex27_tables()
    X = grid.coords() if coords is None else coords
    nex, ney, nez = (m.ne for m in grid.m1)
    loc = np.array([(a, b, c) for a in range(3) for b in range(3) for c in range(3)])
    R, C, V = [], [], []
    for ex in range(nex):
        for ey in range(ney):
            for ez in range(nez):
                ids = grid.node(2 * ex + loc[:, 0], 2 * ey + loc[:, 1], 2 * ez + loc[:, 2])
                Xe = X[ids]
                Ke = np.zeros((81, 81))
                Gq = np.zeros((4, 81))
                H = np.zeros((4, 4))
                for g in range(27):
                    J = dNr[g] @ Xe
                    dN = np.linalg.solve(J, dNr[g])
                    dv = W[g] * np.linalg.det(J)
                    B = np.zeros((6, 81))
                    B[0, 0::3], B[1, 1::3], B[2, 2::3] = dN[0], dN[1], dN[2]
                    B[3, 1::3], B[3, 2::3] = dN[2], dN[1]
                    B[4, 0::3], B[4, 2::3] = dN[2], dN[0]
                    B[5, 0::3], B[5, 1::3] = dN[1], dN[0]
                    if form == "full":
                        Ke += B.T @ D @ B * dv
                    else:
                        Ke += B.T @ Ddev @ B * dv
                        q = np.r_[1.0, xig[g]]
                        Gq += np.outer(q, B[0] + B[1] + B[2]) * dv
                        H += np.outer(q, q) * dv
                if form == "mixed":
                    Ke += kappa * Gq.T @ np.linalg.solve(H, Gq)
                dofs = (3 * ids[:, None] + np.arange(3)).ravel()
                R.append(np.repeat(dofs, 81))
                C.append(np.tile(dofs, 81))
                V.append(Ke.ravel())
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                         shape=(grid.ndof, grid.ndof))


# ----------------------------------------------------------------------------- loads
def face_load(grid, axis, side, traction, lo=None, hi=None):
    """Consistent nodal loads of a uniform traction (3-vector) on the face `axis` = side
    (side 0 = min, 1 = max), optionally restricted to the rectangle lo <= (other coords) <= hi."""
    fs = []
    k = 0
    for d in range(3):
        m = grid.m1[d]
        if d == axis:
            e = np.zeros(m.n)
            e[0 if side == 0 else m.n - 1] = 1.0
            fs.append(e)
        else:
            a = m.breaks[0] if lo is None else lo[k]
            b = m.breaks[-1] if hi is None else hi[k]
            fs.append(m.weights(a, b))
            k += 1
    w = np.kron(np.kron(fs[0], fs[1]), fs[2])
    f = np.zeros(grid.ndof)
    for i in range(3):
        f[i::3] = w * traction[i]
    return f


# ----------------------------------------------------------------------------- solve
def nd_columns(nx, ny, leaf=3):
    """Geometric nested dissection of the (ia, ib) node-column grid of a Q2 tensor mesh.
    Separators are full-thickness planes of nodes at even (element-boundary) indices, so the
    two halves are not coupled.  Returns the column visiting order (separators last)."""
    order = []
    stack = [(0, nx, 0, ny, False)]
    # iterative post-order: (a0, a1, b0, b1, expanded)
    out_stack = []

    def split(a0, a1, b0, b1):
        na, nb = a1 - a0, b1 - b0
        if na <= leaf and nb <= leaf:
            return None
        if na >= nb:
            c = [s for s in range(a0 + 1, a1 - 1) if s % 2 == 0]
            if not c:
                return None
            s = min(c, key=lambda s: abs(s - (a0 + a1 - 1) / 2.0))
            return ((a0, s, b0, b1), (s + 1, a1, b0, b1), [(s, ib) for ib in range(b0, b1)])
        c = [s for s in range(b0 + 1, b1 - 1) if s % 2 == 0]
        if not c:
            return None
        s = min(c, key=lambda s: abs(s - (b0 + b1 - 1) / 2.0))
        return ((a0, a1, b0, s), (a0, a1, s + 1, b1), [(ia, s) for ia in range(a0, a1)])

    def rec(a0, a1, b0, b1):
        if a1 <= a0 or b1 <= b0:
            return
        sp_ = split(a0, a1, b0, b1)
        if sp_ is None:
            order.extend((ia, ib) for ia in range(a0, a1) for ib in range(b0, b1))
            return
        rec(*sp_[0])
        rec(*sp_[1])
        order.extend(sp_[2])

    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    rec(0, nx, 0, ny)
    o = np.asarray(order, dtype=np.int64)
    assert len(o) == nx * ny and np.unique(o[:, 0] * ny + o[:, 1]).size == nx * ny
    return o


def nd_dof_perm(grid, leaf=3):
    cols = nd_columns(grid.nx, grid.ny, leaf)
    ic = np.arange(grid.nz)
    nodes = ((cols[:, 0][:, None] * grid.ny + cols[:, 1][:, None]) * grid.nz + ic[None, :]).ravel()
    return (3 * nodes[:, None] + np.arange(3)[None, :]).ravel()


def nd_tree(nx, ny, leaf=4):
    """Nested-dissection tree of the (ia, ib) column grid, nodes in post-order.  Each node:
    dict(region=(a0, a1, b0, b1), cols=(k, 2) array of the columns eliminated at this node
    (separator, or the whole box for a leaf), children=[indices])."""
    nodes = []

    def split(a0, a1, b0, b1):
        na, nb = a1 - a0, b1 - b0
        if na <= leaf and nb <= leaf:
            return None
        for along_a in ((True, False) if na >= nb else (False, True)):
            lo, hi = (a0, a1) if along_a else (b0, b1)
            c = [s for s in range(lo + 1, hi - 1) if s % 2 == 0]
            if c:
                s = min(c, key=lambda s: abs(s - (lo + hi - 1) / 2.0))
                return along_a, s
        return None

    def rec(a0, a1, b0, b1):
        sp_ = split(a0, a1, b0, b1)
        if sp_ is None:
            cols = np.array([(ia, ib) for ia in range(a0, a1) for ib in range(b0, b1)], dtype=np.int64)
            nodes.append(dict(region=(a0, a1, b0, b1), cols=cols, children=[]))
            return len(nodes) - 1
        along_a, s = sp_
        if along_a:
            c1 = rec(a0, s, b0, b1)
            c2 = rec(s + 1, a1, b0, b1)
            cols = np.array([(s, ib) for ib in range(b0, b1)], dtype=np.int64)
        else:
            c1 = rec(a0, a1, b0, s)
            c2 = rec(a0, a1, s + 1, b1)
            cols = np.array([(ia, s) for ia in range(a0, a1)], dtype=np.int64)
        nodes.append(dict(region=(a0, a1, b0, b1), cols=cols, children=[c1, c2]))
        return len(nodes) - 1

    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    rec(0, nx, 0, ny)
    allc = np.concatenate([nd["cols"] for nd in nodes])
    assert allc.shape[0] == nx * ny and np.unique(allc[:, 0] * ny + allc[:, 1]).size == nx * ny
    return nodes


class NDCholesky:
    """Multifrontal sparse Cholesky for K_ff on a Q2 tensor grid, using the geometric nested
    dissection of the in-plane column grid (full-thickness separators at element boundaries).
    Dense fronts are factorised with LAPACK/BLAS-3 (potrf, trsm, gemm).  No 32-bit index limit.
    (Rewritten 2026-10-30 for speed: Schur complement by gemm instead of syrk + tril symmetrisation,
    extend-add by contiguous index runs instead of np.ix_ scatter; results identical to the original
    to round-off, checked in verify_fem.py V5.)"""

    def __init__(self, Kff, free, grid, leaf=4):
        import scipy.linalg as sla
        n = free.size
        self.n = n
        g2f = -np.ones(grid.ndof, dtype=np.int64)
        g2f[free] = np.arange(n)
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        colfree = g2f.reshape(nx, ny, nz * 3)                 # free index per (column, ic, comp)
        A = Kff.tocsr()
        tree = nd_tree(nx, ny, leaf)

        def cols_dofs(cols):
            if len(cols) == 0:
                return np.zeros(0, dtype=np.int64)
            d = colfree[cols[:, 0], cols[:, 1]].ravel()
            return d[d >= 0]

        def bounding(region):
            """Columns adjacent to the region, line by line (each line contiguous, corners once)."""
            a0, a1, b0, b1 = region
            blo, bhi = max(b0 - 1, 0), min(b1, ny - 1)
            alo, ahi = max(a0 - 1, 0), min(a1, nx - 1)
            bl = []
            if a0 - 1 >= 0:
                bl += [(a0 - 1, ib) for ib in range(blo, bhi + 1)]
            if a1 <= nx - 1:
                bl += [(a1, ib) for ib in range(blo, bhi + 1)]
            if b0 - 1 >= 0:
                bl += [(ia, b0 - 1) for ia in range(alo, ahi + 1)]
            if b1 <= ny - 1:
                bl += [(ia, b1) for ia in range(alo, ahi + 1)]
            seen, out = set(), []
            for c in bl:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            return np.asarray(out, dtype=np.int64).reshape(-1, 2)

        def runs(p):
            """Split index vector p into maximal runs of consecutive integers: (start_in_p, start, length)."""
            if p.size == 0:
                return []
            br = np.nonzero(np.diff(p) != 1)[0] + 1
            st = np.r_[0, br]
            en = np.r_[br, p.size]
            return list(zip(st.tolist(), p[st].tolist(), (en - st).tolist()))

        loc = -np.ones(n, dtype=np.int64)
        updates = {}
        self.fronts = []
        self.flops = 0.0
        self.max_front = 0
        for k, nd in enumerate(tree):
            el = cols_dofs(nd["cols"])
            bd = cols_dofs(bounding(nd["region"]))
            I = np.concatenate([el, bd])
            ne, nb = el.size, bd.size
            self.max_front = max(self.max_front, ne + nb)
            F = np.zeros((ne + nb, ne + nb))
            if ne:
                sub = A[el][:, I].toarray()
                F[:ne, :] = sub
                F[ne:, :ne] = sub[:, ne:].T
            loc[I] = np.arange(ne + nb)
            for c in nd["children"]:
                cb, U = updates.pop(c)
                p = loc[cb]
                if np.any(p < 0):
                    raise RuntimeError("update index not in parent front")
                R = runs(p)
                if len(R) <= 64:
                    for (si, pi, li) in R:
                        for (sj, pj, lj) in R:
                            F[pi:pi + li, pj:pj + lj] += U[si:si + li, sj:sj + lj]
                else:
                    F[np.ix_(p, p)] += U
                del U
            loc[I] = -1
            if ne == 0:
                updates[k] = (bd, F[ne:, ne:])
                self.fronts.append((el, bd, np.zeros((0, 0)), np.zeros((0, nb))))
                continue
            L11 = sla.cholesky(F[:ne, :ne], lower=True, overwrite_a=True, check_finite=False)
            if nb:
                W = sla.solve_triangular(L11, F[:ne, ne:], lower=True, check_finite=False)
                C = F[ne:, ne:]
                C -= W.T @ W
                updates[k] = (bd, np.ascontiguousarray(C))
            else:
                W = np.zeros((ne, 0))
            del F
            self.flops += ne ** 3 / 3.0 + ne * ne * nb + ne * nb * nb
            self.fronts.append((el, bd, L11, W))
        assert not updates
        self.nnz_L = int(sum(f[2].shape[0] * (f[2].shape[0] + 1) // 2 + f[3].size for f in self.fronts))

    def solve(self, B):
        import scipy.linalg as sla
        B = np.array(B, dtype=float, copy=True)
        vec = B.ndim == 1
        if vec:
            B = B[:, None]
        for el, bd, L11, W in self.fronts:              # forward: L y = b
            if el.size == 0:
                continue
            y = sla.solve_triangular(L11, B[el], lower=True, check_finite=False)
            B[el] = y
            if bd.size:
                B[bd] -= W.T @ y
        for el, bd, L11, W in reversed(self.fronts):    # backward: L^T x = y
            if el.size == 0:
                continue
            r = B[el]
            if bd.size:
                r = r - W @ B[bd]
            B[el] = sla.solve_triangular(L11, r, lower=True, trans="T", check_finite=False)
        return B[:, 0] if vec else B


def solve_dirichlet(K, fixed, values, f=None, grid=None, method="nd", threads=None):
    """Solve K u = f + r with u[fixed] = values (columns of `values` = load cases sharing the
    constraint set).  K_ff is SPD (displacement and B-bar forms).
    method="nd": multifrontal Cholesky on the grid's geometric nested dissection (NDCholesky, needs `grid`);
    method="superlu": scipy SuperLU (splu), symmetric mode, no pivoting, with the same nested-dissection
    ordering when `grid` is given (else MMD on A + A^T).
    Returns u, reactions r = K u - f (zero at free DOFs up to round-off), info."""
    import time
    from contextlib import nullcontext
    try:
        from threadpoolctl import threadpool_limits
        ctx = threadpool_limits(limits=threads) if threads else nullcontext()
    except ImportError:                                     # pragma: no cover
        ctx = nullcontext()
    n = K.shape[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    nrhs = values.shape[1]
    fixed = np.asarray(fixed, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[fixed] = False
    U = np.zeros((n, nrhs))
    U[fixed, :] = values
    F = None if f is None else np.asarray(f, dtype=float).reshape(n, -1)
    with ctx:
        t0 = time.time()
        if method == "nd":
            if grid is None:
                raise ValueError("method='nd' needs the grid")
            free = np.nonzero(mask)[0]
            Kff = K[free][:, free].tocsr()
            t1 = time.time()
            fac = NDCholesky(Kff, free, grid)
            nnz_fac = fac.nnz_L
            extra = dict(max_front=int(fac.max_front), gflop=round(fac.flops / 1e9, 1))
            solver = "multifrontal Cholesky, geometric nested dissection"
        elif method == "superlu":
            if grid is not None:
                order = nd_dof_perm(grid)
                free = order[mask[order]]
                permc = "NATURAL"
            else:
                free = np.nonzero(mask)[0]
                permc = "MMD_AT_PLUS_A"
            Kff = K[free][:, free].tocsc()
            t1 = time.time()
            fac = spla.splu(Kff, permc_spec=permc, diag_pivot_thresh=0.0,
                            options=dict(SymmetricMode=True))
            nnz_fac = int(fac.L.nnz + fac.U.nnz)
            extra = {}
            solver = "SuperLU " + permc
        else:
            raise KeyError(method)
        rhs = -(K @ U)[free, :]
        if F is not None:
            rhs = rhs + F[free, :]
        t2 = time.time()
        U[free, :] = fac.solve(rhs)
        t3 = time.time()
    R = K @ U
    if F is not None:
        R = R - F
    res = np.linalg.norm(R[free, :]) / max(np.linalg.norm(R[fixed, :]) if fixed.size else 0.0,
                                           np.linalg.norm(F) if F is not None else 0.0, 1e-300)
    info = dict(n_dof=int(n), n_free=int(free.size), nnz_K=int(K.nnz),
                nnz_factor=int(nnz_fac), t_extract_s=round(t1 - t0, 3),
                t_factor_s=round(t2 - t1, 3), t_solve_s=round(t3 - t2, 3),
                rel_residual_free=float(res), solver=solver, **extra)
    return U, R, info


# ----------------------------------------------------------------------------- face post-processing
def unfold_face(F, axis, parity, additive):
    """Mirror a face field F[ia, ib, comp] about the plane index 0 of `axis` (0: x, 1: y).
    parity[comp] = +1/-1; additive=True for nodal forces (plane nodes collect both halves)."""
    P = np.asarray(parity, dtype=float)
    n = F.shape[axis]
    inner = np.take(F, np.arange(1, n), axis=axis)
    mirrored = np.flip(inner, axis=axis) * P
    plane = np.take(F, [0], axis=axis)
    if additive:
        plane = plane + plane * P
    return np.concatenate([mirrored, plane, inner], axis=axis)


def unfold_breaks(b):
    b = np.asarray(b, dtype=float)
    return np.concatenate([-b[:0:-1], b])


class FaceField:
    """Free-face (z = 0) fields on the full 51 x 71 face: nodal reactions -> L2-projected traction
    on the platen rectangle, per-taxel integrals, profiles, point values."""

    def __init__(self, bx, by, U, R):
        self.mx, self.my = Mesh1D(bx), Mesh1D(by)
        self.U, self.R = U, R                        # (nx, ny, 3)

    def traction(self, rect):
        """Consistent traction t_h (nodal values) on the platen rectangle (x0, x1, y0, y1):
        M_platen t_h = R (Galerkin flux recovery)."""
        x0, x1, y0, y1 = rect
        ex = np.nonzero((self.mx.breaks[:-1] >= x0 - 1e-9) & (self.mx.breaks[1:] <= x1 + 1e-9))[0]
        ey = np.nonzero((self.my.breaks[:-1] >= y0 - 1e-9) & (self.my.breaks[1:] <= y1 + 1e-9))[0]
        ix = np.arange(2 * ex[0], 2 * ex[-1] + 3)
        iy = np.arange(2 * ey[0], 2 * ey[-1] + 3)
        Mx = self.mx.mat("M", ex)[ix][:, ix].toarray()
        My = self.my.mat("M", ey)[iy][:, iy].toarray()
        T = np.zeros_like(self.R)
        for c in range(3):
            Rc = self.R[np.ix_(ix, iy, [c])][:, :, 0]
            T[np.ix_(ix, iy, [c])] = np.linalg.solve(Mx, np.linalg.solve(My, Rc.T).T)[:, :, None]
        self.T, self.rect = T, rect
        return T

    def integrate(self, field, x0, x1, y0, y1):
        wx, wy = self.mx.weights(x0, x1), self.my.weights(y0, y1)
        return np.einsum("a,b,abc->c", wx, wy, field)

    def taxel_forces(self, xb, yb):
        """F[i, j, c] = int over taxel (i, j) of the traction (within the platen)."""
        x0, x1, y0, y1 = self.rect
        out = np.zeros((len(xb) - 1, len(yb) - 1, 3))
        for i in range(len(xb) - 1):
            for j in range(len(yb) - 1):
                a, b = max(xb[i], x0), min(xb[i + 1], x1)
                c, d = max(yb[j], y0), min(yb[j + 1], y1)
                if b - a > 1e-9 and d - c > 1e-9:
                    out[i, j] = self.integrate(self.T, a, b, c, d)
        return out

    def taxel_mean_disp(self, xb, yb):
        out = np.zeros((len(xb) - 1, len(yb) - 1, 3))
        for i in range(len(xb) - 1):
            for j in range(len(yb) - 1):
                A = (xb[i + 1] - xb[i]) * (yb[j + 1] - yb[j])
                out[i, j] = self.integrate(self.U, xb[i], xb[i + 1], yb[j], yb[j + 1]) / A
        return out

    def point(self, field, x0, y0):
        ix, Nx = self.mx.eval_basis(x0)
        iy, Ny = self.my.eval_basis(y0)
        return np.einsum("a,b,abc->c", Nx, Ny, field[np.ix_(ix, iy)])


# ----------------------------------------------------------------------------- manufactured solutions (verification)
# A separable term is (coef, ((k_x, p_x), (k_y, p_y), (k_z, p_z))) = coef * prod_d sin(k_d x_d + p_d).
def sep_diff(term, d):
    """d/dx_d of a separable sine term (d/dx sin(kx+p) = k sin(kx+p+pi/2))."""
    c, f = term
    f = list(f)
    k, p = f[d]
    f[d] = (k, p + 0.5 * np.pi)
    return (c * k, tuple(f))


def sep_eval(terms, X, Y, Z):
    """Evaluate a sum of separable terms on the tensor grid X x Y x Z (1D arrays) -> (nX, nY, nZ)."""
    out = np.zeros((len(X), len(Y), len(Z)))
    for c, ((kx, px), (ky, py), (kz, pz)) in terms:
        out += c * np.einsum("a,b,c->abc", np.sin(kx * X + px), np.sin(ky * Y + py), np.sin(kz * Z + pz))
    return out


def mms_body_force(u_terms, E, nu):
    """f_i = -(lam + mu) d_i (div u) - mu lap u_i for u_i = sum of separable terms (list of 3 lists)."""
    lam, mu, _ = lame(E, nu)
    div = [sep_diff(t, k) for k in range(3) for t in u_terms[k]]
    f = []
    for i in range(3):
        fi = [(-(lam + mu) * c, fn) for c, fn in (sep_diff(t, i) for t in div)]
        for j in range(3):
            fi += [(-mu * c, fn) for c, fn in (sep_diff(sep_diff(t, j), j) for t in u_terms[i])]
        f.append(fi)
    return f


def _gauss_1d(mesh, ng):
    """Gauss points/weights on every element of a Mesh1D and the (dense) basis / derivative matrices."""
    gx, gw = np.polynomial.legendre.leggauss(ng)
    N, dN = q2_basis(gx)
    a, h = mesh.breaks[:-1], mesh.h
    X = (a[:, None] + 0.5 * h[:, None] * (gx[None, :] + 1.0)).ravel()
    W = (0.5 * h[:, None] * gw[None, :]).ravel()
    B = np.zeros((X.size, mesh.n))
    D = np.zeros((X.size, mesh.n))
    rows = np.arange(X.size).reshape(mesh.ne, ng)
    for e in range(mesh.ne):
        B[np.ix_(rows[e], mesh.conn[e])] = N
        D[np.ix_(rows[e], mesh.conn[e])] = dN * (2.0 / h[e])
    return X, W, B, D


def sep_load(grid, f_terms, ng=10):
    """Consistent load vector F[(A, i)] = int f_i N_A dV of separable body forces (Gauss, ng points/dir)."""
    G = [_gauss_1d(m, ng) for m in grid.m1]
    F = np.zeros(grid.ndof)
    for i in range(3):
        acc = np.zeros((grid.nx, grid.ny, grid.nz))
        for c, fn in f_terms[i]:
            v = [G[d][2].T @ (G[d][1] * np.sin(fn[d][0] * G[d][0] + fn[d][1])) for d in range(3)]
            acc += c * np.einsum("a,b,c->abc", *v)
        F[i::3] = acc.ravel()
    return F


def tensor_eval(grid, U, ng=5, deriv=None):
    """Values (or the derivative along axis `deriv`) of a nodal field U (ndof,) at the tensor Gauss grid:
    returns X, Y, Z (1D), W (3D weights), V (3, nX, nY, nZ)."""
    G = [_gauss_1d(m, ng) for m in grid.m1]
    V = []
    for i in range(3):
        Ui = U[i::3].reshape(grid.nx, grid.ny, grid.nz)
        mats = [G[d][3] if deriv == d else G[d][2] for d in range(3)]
        V.append(np.einsum("pa,qb,rc,abc->pqr", mats[0], mats[1], mats[2], Ui, optimize=True))
    W = np.einsum("a,b,c->abc", G[0][1], G[1][1], G[2][1])
    return G[0][0], G[1][0], G[2][0], W, np.asarray(V)
