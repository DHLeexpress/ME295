# Shared machinery for bench_static: build the 51 x 71 x 1.5 mm PDMS pad as a MuJoCo 3.17 flex, extract
# its elastic stiffness from MuJoCo by finite differences of qfrc_spring, cross-check it, and solve the
# static benchmark cases with exact Dirichlet / penalty boundary conditions (no contact).
# Units: SI everywhere (m, N, Pa, kg, s). Stiffness reported in N/um = 1e-6 N/m.
import os, time, json, math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import mujoco

# ----------------------------------------------------------------------------------------------------------
# geometry, material
LX, LY, T = 0.051, 0.071, 0.0015          # pad size (m): x across the blade, y along the finger, z thickness
AREA = LX * LY
RHO = 965.0
MASS = RHO * LX * LY * T                   # 0.0052414 kg, as in the robot's pdms_left.xml
E_TARGET = 1e6
H_STEP = 5e-4                              # robot timestep (s)
NTX, NTY = 5, 7                            # taxels
TX, TY = LX / NTX, LY / NTY                # 10.2 mm, 10.142857 mm
ZFREE, ZBACK = -T / 2, T / 2               # free face at z=-t/2 (as in the robot PDMS frame), back face pinned
TOL = 1e-9                                 # geometric tolerance (m)


def lame(E, nu):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return lam, mu, lam + 2 * mu


def remap_full(E, nu):
    """Inputs (young, poisson) for dof=full such that MuJoCo's doubled lambda equals the target lambda (F1)."""
    G = E / (2 * (1 + nu))
    nu_in = nu / (2 * (1 - nu))
    return 2 * G * (1 + nu_in), nu_in


def exact_G1(nu, E=E_TARGET):
    return lame(E, nu)[2] * AREA / T       # N/m


def shear_GA_t(nu, E=E_TARGET):
    return lame(E, nu)[1] * AREA / T       # N/m


# ----------------------------------------------------------------------------------------------------------
# configurations
VARIANTS = {"full_corr": "full", "full_raw": "full", "trilinear": "trilinear", "quadratic": "quadratic"}


def cfg_id(c):
    if c["variant"].startswith("full"):
        return f'{c["variant"]}_nt{c["nt"]}_nz{c["nz"]}_nu{c["nu"]}'
    return f'{c["variant"]}_nt{c["nt"]}_cz{c["cz"]}_nu{c["nu"]}'


def describe(c):
    """Derived configuration: grid counts, inputs passed to MuJoCo, spacings."""
    v = c["variant"]; nt = c["nt"]; nu = c["nu"]
    cx, cy = NTX * nt, NTY * nt
    out = dict(c)
    out["dof"] = VARIANTS[v]
    if v.startswith("full"):
        nz = c["nz"]; count = [cx + 1, cy + 1, nz]; cz = nz - 1; order = 1; cells = None
    else:
        order = 1 if v == "trilinear" else 2
        cz = c["cz"]; cells = [cx, cy, cz]
        count = [cx * order + 1, cy * order + 1, cz * order + 1]   # vertices coincide with nodes
    if v == "full_corr":
        young, poisson = remap_full(E_TARGET, nu)
    else:
        young, poisson = E_TARGET, nu
    out.update(order=order, count=count, cellcount=cells, young_in_Pa=young, poisson_in=poisson,
               E_target_Pa=E_TARGET, nu_target=nu,
               elem_dx_mm=LX / cx * 1e3, elem_dy_mm=LY / cy * 1e3, elem_dz_mm=T / cz * 1e3,
               elements_through_thickness=cz)
    out["aspect_dx_over_dz"] = out["elem_dx_mm"] / out["elem_dz_mm"]
    # node spacing (interpolated: nodes are order times denser than cells)
    out["node_dx_mm"] = LX / (count[0] - 1) * 1e3
    out["node_dz_mm"] = T / (count[2] - 1) * 1e3
    return out


def make_xml(c, pinned, gravity=False, young=None, poisson=None):
    dd = describe(c)
    count = dd["count"]
    sp_ = [LX / (count[0] - 1), LY / (count[1] - 1), T / (count[2] - 1)]
    young = dd["young_in_Pa"] if young is None else young
    poisson = dd["poisson_in"] if poisson is None else poisson
    cellattr = ""
    if dd["cellcount"] is not None:
        cc = dd["cellcount"]; cellattr = f'cellcount="{cc[0]} {cc[1]} {cc[2]}"'
    pin = ""
    if pinned:   # back face = last z index (z = +t/2); vertices coincide with nodes for interpolated flexes
        pin = f'<pin gridrange="0 0 {count[2]-1} {count[0]-1} {count[1]-1} {count[2]-1}"/>'
    g = "0 0 -9.81" if gravity else "0 0 0"
    return f"""<mujoco model="pad">
  <option gravity="{g}" integrator="discrete" solver="CG" timestep="{H_STEP}"/>
  <worldbody>
    <body name="pad">
      <flexcomp name="pad" type="grid" dim="3" count="{count[0]} {count[1]} {count[2]}"
                spacing="{sp_[0]:.17g} {sp_[1]:.17g} {sp_[2]:.17g}" dof="{dd['dof']}" {cellattr}
                mass="{MASS:.17g}" radius="0">
        <elasticity young="{young:.17g}" poisson="{poisson:.17g}" damping="0"/>
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        {pin}
      </flexcomp>
    </body>
  </worldbody>
</mujoco>"""


def load(c, pinned, **kw):
    t0 = time.time()
    m = mujoco.MjModel.from_xml_string(make_xml(c, pinned, **kw))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, time.time() - t0


# ----------------------------------------------------------------------------------------------------------
# points (= flex vertices for dof=full, = interpolation nodes for trilinear/quadratic)
def points(m, d):
    interp = int(m.flex_interp[0]) != 0
    if interp:
        n = int(m.flex_nodenum[0]); a = int(m.flex_nodeadr[0])
        bodies = np.array(m.flex_nodebodyid[a:a + n])
        X = np.array(m.flex_node0[a:a + n]).reshape(n, 3)
    else:
        n = int(m.flex_vertnum[0]); a = int(m.flex_vertadr[0])
        bodies = np.array(m.flex_vertbodyid[a:a + n])
        X = np.array(d.flexvert_xpos[a:a + n]).reshape(n, 3)
    dofadr = np.full(n, -1); qadr = np.full(n, -1); mass = np.zeros(n)
    for i, b in enumerate(bodies):
        if m.body_dofnum[b] == 3:
            dofadr[i] = m.body_dofadr[b]; qadr[i] = m.jnt_qposadr[m.body_jntadr[b]]; mass[i] = m.body_mass[b]
            # the three slide axes must be world x, y, z
            for k in range(3):
                ax = m.jnt_axis[m.body_jntadr[b] + k]
                assert np.allclose(ax, np.eye(3)[k]), "unexpected joint axis"
    lo = X.min(0); hi = X.max(0)
    cnt = np.array([len(np.unique(np.round(X[:, k], 9))) for k in range(3)])
    g = np.rint((X - lo) / (hi - lo) * (cnt - 1)).astype(int)
    assert np.allclose(lo + g / (cnt - 1) * (hi - lo), X, atol=1e-9), "points not on a grid"
    order = int(abs(m.flex_interp[0])) if interp else 1
    if interp:
        cells = np.array(m.flex_cellnum[0])
        nn = cells * order + 1
        assert np.all(nn == cnt)
        # flexcomp node order: idx = gi*ny*nz + gj*nz + gk
        assert np.all(g[:, 0] * cnt[1] * cnt[2] + g[:, 1] * cnt[2] + g[:, 2] == np.arange(n))
    return dict(n=n, X=X, g=g, cnt=cnt, dofadr=dofadr, qadr=qadr, mass=mass, interp=interp, order=order,
                bodies=bodies)


def elements(m, P):
    """Point lists of the elastic elements: tets (full) or cells (interpolated)."""
    if not P["interp"]:
        ne = int(m.flex_elemnum[0]); a = int(m.flex_elemdataadr[0])
        return np.array(m.flex_elem[a:a + 4 * ne]).reshape(ne, 4)
    o = P["order"]; nx, ny, nz = P["cnt"]; cx, cy, cz = (P["cnt"] - 1) // o
    li = np.arange(o + 1)
    loc = (li[:, None, None] * ny * nz + li[None, :, None] * nz + li[None, None, :]).ravel()
    ci, cj, ck = np.meshgrid(np.arange(cx), np.arange(cy), np.arange(cz), indexing="ij")
    base = (ci.ravel() * o) * ny * nz + (cj.ravel() * o) * nz + ck.ravel() * o
    return base[:, None] + loc[None, :]


def adjacency(P, elem):
    n = P["n"]; k = elem.shape[1]
    r = np.repeat(elem, k, axis=1).ravel(); c = np.tile(elem, (1, k)).ravel()
    A = sp.csr_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(n, n))
    A.data[:] = 1
    return A


# ----------------------------------------------------------------------------------------------------------
# finite-difference stiffness with graph colouring
def passive_force(m, d, qpos):
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_flex(m, d); mujoco.mj_passive(m, d)
    return d.qfrc_spring.copy()


def colouring(P, reach):
    p = 2 * reach + 1
    g = P["g"]; pz = min(p, P["cnt"][2])
    return (g[:, 0] % p) + p * (g[:, 1] % p) + p * p * (g[:, 2] % pz)


def fd_stiffness(m, d, P, elem, h=1e-8):
    """K (3n x 3n, point-major, world xyz components) = -d qfrc_spring / d x, central differences.
    Only points with dofs are perturbed; rows/cols of points without dofs are left empty."""
    n = P["n"]; reach = 1 if P["order"] == 1 else 2
    A = adjacency(P, elem)
    col = colouring(P, reach)
    free = P["dofadr"] >= 0
    q0 = d.qpos.copy()
    rows, cols, vals = [], [], []
    out_max = 0.0; in_max = 0.0; nev = 0
    t0 = time.time()
    for c in np.unique(col):
        mem = np.where((col == c) & free)[0]
        if mem.size == 0:
            continue
        B = A[:, mem].tocsr()
        cnt = np.diff(B.indptr)
        if cnt.max() > 1:
            raise RuntimeError("colouring invalid")
        has = np.where((cnt == 1) & free)[0]
        own = mem[B.indices[B.indptr[has]]]
        no = np.where((cnt == 0) & free)[0]
        for k in range(3):
            qp = q0.copy(); qp[P["qadr"][mem] + k] += h
            fp = passive_force(m, d, qp)
            qm = q0.copy(); qm[P["qadr"][mem] + k] -= h
            fm = passive_force(m, d, qm)
            nev += 2
            dF = -(fp - fm) / (2 * h)
            for r in range(3):
                v = dF[P["dofadr"][has] + r]
                rows.append(3 * has + r); cols.append(3 * own + k); vals.append(v)
                in_max = max(in_max, np.abs(v).max())
                if no.size:
                    out_max = max(out_max, np.abs(dF[P["dofadr"][no] + r]).max())
    passive_force(m, d, q0)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    K = sp.csr_matrix((vals, (rows, cols)), shape=(3 * n, 3 * n))
    info = dict(fd_step_m=h, fd_evaluations=nev, fd_colours=int(np.unique(col[free]).size), fd_time_s=time.time() - t0,
                fd_out_of_pattern_rel=out_max / in_max)
    return K, info


def rigid_modes(X):
    c = X.mean(0); Y = X - c; n = X.shape[0]
    R = np.zeros((3 * n, 6))
    for k in range(3):
        R[k::3, k] = 1.0
    for k in range(3):
        w = np.eye(3)[k]
        R[:, 3 + k] = np.cross(w, Y).ravel()
    return R


def k_checks(K, X):
    Kt = K.T.tocsr()
    D = (K - Kt)
    sym = abs(D).max() / abs(K).max() if D.nnz else 0.0
    symF = sp.linalg.norm(D) / sp.linalg.norm(K)
    R = rigid_modes(X)
    absK = abs(K)
    null = [float(np.linalg.norm(K @ R[:, i]) / np.linalg.norm(absK @ np.abs(R[:, i]))) for i in range(6)]
    return dict(sym_maxabs_rel=float(sym), sym_frob_rel=float(symF), nullspace_rel=null,
                nullspace_rel_max=float(max(null)))


# ----------------------------------------------------------------------------------------------------------
# independent stiffness assemblies (cross-checks of the FD extraction)
STRETCH_EDGES = [(0, 1), (1, 2), (2, 0), (2, 3), (0, 3), (1, 3)]   # engine_core_util.h mj_stretchEdges[1]


def k_full_from_model(m, P, elem):
    """K = sum_t 2 G_t' k_t G_t from MuJoCo's packed per-tet metric (flex_stiffness), energy s'ks/4."""
    ne = elem.shape[0]
    a = int(m.flex_stiffnessadr[0])
    packed = np.array(m.flex_stiffness[a:a + 24 * ne]).reshape(ne, 24)[:, :21]
    k = np.zeros((ne, 6, 6)); iu = np.triu_indices(6)
    k[:, iu[0], iu[1]] = packed; k[:, iu[1], iu[0]] = packed
    X = P["X"]
    G = np.zeros((ne, 6, 12))
    for e, (i, j) in enumerate(STRETCH_EDGES):
        dvec = X[elem[:, i]] - X[elem[:, j]]
        G[:, e, 3 * i:3 * i + 3] = dvec; G[:, e, 3 * j:3 * j + 3] = -dvec
    Ke = 2 * np.einsum("tei,tef,tfj->tij", G, k, G)
    dofs = (3 * elem[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 12)
    r = np.repeat(dofs, 12, axis=1).ravel(); c = np.tile(dofs, (1, 12)).ravel()
    return sp.csr_matrix((Ke.ravel(), (r, c)), shape=(3 * P["n"], 3 * P["n"]))


def phi1(s, i, o):
    if o == 1:
        return 1 - s if i == 0 else s
    return [2 * s * s - 3 * s + 1, 4 * (s - s * s), 2 * s * s - s][i]


def dphi1(s, i, o):
    if o == 1:
        return -1.0 if i == 0 else 1.0
    return [4 * s - 3, 4 * (1 - 2 * s), 4 * s - 1][i]


def hex_stiffness(dx, dy, dz, E, nu, o, nq=None):
    """Standard Lagrange hex (order o) linear-elastic stiffness, Gauss-Legendre (o+1)^3 (as MuJoCo)."""
    lam, mu, _ = lame(E, nu)
    nq = o + 1 if nq is None else nq
    xg, wg = np.polynomial.legendre.leggauss(nq); s = (xg + 1) / 2; w = wg / 2
    nb = o + 1; n = nb ** 3
    K = np.zeros((3 * n, 3 * n))
    J = np.array([dx, dy, dz])
    for a_ in range(nq):
        for b_ in range(nq):
            for c_ in range(nq):
                dv = w[a_] * w[b_] * w[c_] * dx * dy * dz
                Gr = np.zeros((n, 3)); idx = 0
                for i in range(nb):
                    for j in range(nb):
                        for l in range(nb):
                            Gr[idx] = [dphi1(s[a_], i, o) * phi1(s[b_], j, o) * phi1(s[c_], l, o),
                                       phi1(s[a_], i, o) * dphi1(s[b_], j, o) * phi1(s[c_], l, o),
                                       phi1(s[a_], i, o) * phi1(s[b_], j, o) * dphi1(s[c_], l, o)]
                            idx += 1
                Gr = Gr / J
                # B matrix (6 x 3n), Voigt with engineering shear
                B = np.zeros((6, 3 * n))
                B[0, 0::3] = Gr[:, 0]; B[1, 1::3] = Gr[:, 1]; B[2, 2::3] = Gr[:, 2]
                B[3, 1::3] = Gr[:, 2]; B[3, 2::3] = Gr[:, 1]
                B[4, 0::3] = Gr[:, 2]; B[4, 2::3] = Gr[:, 0]
                B[5, 0::3] = Gr[:, 1]; B[5, 1::3] = Gr[:, 0]
                Dm = np.zeros((6, 6)); Dm[:3, :3] = lam; Dm[np.arange(3), np.arange(3)] += 2 * mu
                Dm[3, 3] = Dm[4, 4] = Dm[5, 5] = mu
                K += B.T @ Dm @ B * dv
    return K


def k_interp_own(P, elem, E, nu):
    o = P["order"]; cnt = P["cnt"]; cells = (cnt - 1) // o
    dx, dy, dz = LX / cells[0], LY / cells[1], T / cells[2]
    Ke = hex_stiffness(dx, dy, dz, E, nu, o)
    ne, npe = elem.shape
    dofs = (3 * elem[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 3 * npe)
    r = np.repeat(dofs, 3 * npe, axis=1).ravel(); c = np.tile(dofs, (1, 3 * npe)).ravel()
    v = np.tile(Ke.ravel(), ne)
    return sp.csr_matrix((v, (r, c)), shape=(3 * P["n"], 3 * P["n"]))


def efm_stiffness(m, d):
    """MuJoCo's own assembled effective stiffness CSR (h^2 K for damping 0), nv x nv."""
    nv = m.nv
    rownnz = np.array(d.efm_K_rownnz); rowadr = np.array(d.efm_K_rowadr)
    colind = np.array(d.efm_K_colind); val = np.array(d.efm_K_val)
    rows = np.repeat(np.arange(nv), rownnz)
    idx = np.concatenate([np.arange(rowadr[i], rowadr[i] + rownnz[i]) for i in range(nv)]) if nv else np.zeros(0, int)
    return sp.csr_matrix((val[idx], (rows, colind[idx])), shape=(nv, nv))


# ----------------------------------------------------------------------------------------------------------
# free face: surface interpolation, taxel integrals, sample points
def taxel_of(x, y):
    i = np.clip(np.floor((x + LX / 2) / TX).astype(int), 0, NTX - 1)
    j = np.clip(np.floor((y + LY / 2) / TY).astype(int), 0, NTY - 1)
    return i, j


def surface(P, elem):
    """Free-face (z=-t/2) surface elements: list of (point ids, shape-function evaluator, corner coords)."""
    X = P["X"]; onf = P["g"][:, 2] == 0
    if not P["interp"]:
        tris = []
        for tet in elem:
            f = [v for v in tet if onf[v]]
            if len(f) == 3:
                tris.append(f)
        tris = np.array(tris)
        return dict(kind="tri", conn=tris)
    o = P["order"]; nx, ny, nz = P["cnt"]; cx, cy = (nx - 1) // o, (ny - 1) // o
    li = np.arange(o + 1)
    quads = []
    for ci in range(cx):
        for cj in range(cy):
            ids = [((ci * o + a) * ny + (cj * o + b)) * nz + 0 for a in li for b in li]
            quads.append(ids)
    return dict(kind="quad", order=o, conn=np.array(quads))


def surface_integrals(P, S):
    """Wn[a, T] = integral over taxel T of the free-face shape function of point a (exact)."""
    X = P["X"]; n = P["n"]
    W = np.zeros((n, NTX * NTY))
    if S["kind"] == "tri":
        for tri in S["conn"]:
            p = X[tri][:, :2]
            area = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
            cxy = p.mean(0); i, j = taxel_of(cxy[0], cxy[1])
            W[tri, i * NTY + j] += area / 3
    else:
        o = S["order"]
        w1 = np.array([0.5, 0.5]) if o == 1 else np.array([1 / 6, 2 / 3, 1 / 6])
        w2 = np.outer(w1, w1).ravel()
        for q in S["conn"]:
            p = X[q][:, :2]
            dx = p[:, 0].max() - p[:, 0].min(); dy = p[:, 1].max() - p[:, 1].min()
            cxy = p.mean(0); i, j = taxel_of(cxy[0], cxy[1])
            W[q, i * NTY + j] += w2 * dx * dy
    return W


def surface_samples(P, S, inside, target):
    """Sample points (midpoint rule, sub-spacing <= target) on the free face where inside(x,y) is true.
    Returns (point ids per sample [ns, npe], shape values [ns, npe], weights [ns])."""
    X = P["X"]
    ids, vals, wts = [], [], []
    if S["kind"] == "tri":
        for tri in S["conn"]:
            p = X[tri][:, :2]
            lo = p.min(0); hi = p.max(0)
            ns = max(2, int(math.ceil(max(hi - lo) / target)))
            u = (np.arange(ns) + 0.5) / ns
            gx, gy = np.meshgrid(lo[0] + u * (hi[0] - lo[0]), lo[1] + u * (hi[1] - lo[1]), indexing="ij")
            q = np.c_[gx.ravel(), gy.ravel()]
            # barycentric coordinates
            Tm = np.array([[p[1, 0] - p[0, 0], p[2, 0] - p[0, 0]], [p[1, 1] - p[0, 1], p[2, 1] - p[0, 1]]])
            lam12 = np.linalg.solve(Tm, (q - p[0]).T).T
            l0 = 1 - lam12.sum(1)
            Lb = np.c_[l0, lam12]
            inn = (Lb >= -1e-12).all(1)
            # cell area / ns^2 per sample; each sample belongs to exactly one triangle of its square cell
            dA = (hi[0] - lo[0]) * (hi[1] - lo[1]) / ns ** 2
            keep = inn & inside(q[:, 0], q[:, 1])
            if keep.any():
                ids.append(np.tile(tri, (keep.sum(), 1))); vals.append(Lb[keep]); wts.append(np.full(keep.sum(), dA))
    else:
        o = S["order"]
        for qd in S["conn"]:
            p = X[qd][:, :2]
            lo = p.min(0); hi = p.max(0)
            ns = max(2, int(math.ceil(max(hi - lo) / target)))
            u = (np.arange(ns) + 0.5) / ns
            su, tu = np.meshgrid(u, u, indexing="ij"); su = su.ravel(); tu = tu.ravel()
            qx = lo[0] + su * (hi[0] - lo[0]); qy = lo[1] + tu * (hi[1] - lo[1])
            keep = inside(qx, qy)
            if not keep.any():
                continue
            Nv = np.stack([phi1(su[keep], a, o) * phi1(tu[keep], b, o) for a in range(o + 1) for b in range(o + 1)], 1)
            dA = (hi[0] - lo[0]) * (hi[1] - lo[1]) / ns ** 2
            ids.append(np.tile(qd, (keep.sum(), 1))); vals.append(Nv); wts.append(np.full(keep.sum(), dA))
    if not ids:
        return None
    return np.concatenate(ids), np.concatenate(vals), np.concatenate(wts)


# ----------------------------------------------------------------------------------------------------------
# static solves
def solve_dirichlet(K, fixed, uval):
    """K u = f with u[fixed] = uval and f = 0 on the free dofs. Returns u and reactions R = K u."""
    N = K.shape[0]
    mask = np.zeros(N, bool); mask[fixed] = True
    fr = np.where(~mask)[0]
    u = np.zeros(N); u[fixed] = uval
    if fr.size:
        Kff = K[fr][:, fr].tocsc()
        rhs = -(K[fr][:, fixed] @ uval)
        lu = spla.splu(Kff, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        u[fr] = lu.solve(rhs)
    R = K @ u
    resid = np.abs(R[fr]).max() / max(np.abs(R[fixed]).max(), 1e-300) if fr.size else 0.0
    return u, R, float(resid)


def solve_penalty(K, fixed, uval, ids, vals, wts, target_z, kappa):
    """min 1/2 u'Ku + kappa/2 sum_q w_q (N_q u_z - target)^2, u[fixed]=uval (back face)."""
    N = K.shape[0]
    ns, npe = ids.shape
    rz = np.repeat(np.arange(ns), npe); cz = (3 * ids + 2).ravel()
    Nm = sp.csr_matrix((vals.ravel(), (rz, cz)), shape=(ns, N))
    Kp = (K + kappa * (Nm.T @ sp.diags(wts) @ Nm)).tocsr()
    f = kappa * (Nm.T @ (wts * target_z))
    mask = np.zeros(N, bool); mask[fixed] = True
    fr = np.where(~mask)[0]
    u = np.zeros(N); u[fixed] = uval
    lu = spla.splu(Kp[fr][:, fr].tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    u[fr] = lu.solve(f[fr] - Kp[fr][:, fixed] @ uval)
    uz = Nm @ u
    F = float(kappa * np.sum(wts * (target_z - uz)))
    gap = float(np.max(np.abs(target_z - uz)) / abs(target_z))
    return u, F, gap


def to_json(o):
    if isinstance(o, dict):
        return {k: to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_json(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o
