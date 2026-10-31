"""Shared helpers for the flex-internals checks (MuJoCo 3.17.0 python bindings).

Conventions: SI units; displacements imposed on the dof-carrying bodies (flex vertices for dof="full",
interpolation nodes for trilinear/quadratic); forces read back from qfrc_spring (or from constraint rows).
"""
import json
import os
import warnings

import numpy as np
import mujoco

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
os.makedirs(OUT, exist_ok=True)

# the layer of the study
LAYER = dict(A=0.051, B=0.071, T=0.0015, E=1e6, nu=0.45, rho=965.0)


def lame(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def remap_full(E, nu):
    """Inputs (young, poisson) that make MuJoCo's dof="full" StVK (lambda doubled) reproduce (E, nu)."""
    G = E / (2 * (1 + nu))
    nu_in = nu / (2 * (1 - nu))
    return 2 * G * (1 + nu_in), nu_in


def iso_C(lam, mu):
    """6x6 Voigt stiffness (order 11,22,33,23,13,12; engineering shear strains)."""
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


VPAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def voigt_strain(k, eps):
    """Symmetric small-strain tensor for Voigt component k (engineering shear)."""
    e = np.zeros((3, 3))
    i, j = VPAIRS[k]
    if i == j:
        e[i, i] = eps
    else:
        e[i, j] = e[j, i] = eps / 2
    return e


def dof_bodies(m):
    """Bodies carrying exactly 3 slide dofs (flex vertex or node bodies)."""
    out = []
    for b in range(m.nbody):
        if m.body_dofnum[b] == 3 and m.body_jntnum[b] == 3 and \
                all(m.jnt_type[m.body_jntadr[b] + k] == mujoco.mjtJoint.mjJNT_SLIDE for k in range(3)):
            out.append(b)
    return np.array(out, dtype=int)


class Rig:
    """Impose world displacements on slide bodies and read world forces."""

    def __init__(self, m, d=None):
        self.m = m
        self.d = d if d is not None else mujoco.MjData(m)
        self.d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, self.d)
        self.bodies = dof_bodies(m)
        self.ax = {}
        for b in self.bodies:
            A = np.zeros((3, 3))
            for k in range(3):
                A[:, k] = self.d.xaxis[m.body_jntadr[b] + k]
            self.ax[b] = A
        self.X0 = np.array([self.d.xpos[b].copy() for b in self.bodies])
        self.vert0 = self.d.flexvert_xpos.copy()

    def set_disp(self, U, forward=True):
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for i, b in enumerate(self.bodies):
            q = np.linalg.solve(self.ax[b], U[i])
            for k in range(3):
                d.qpos[m.jnt_qposadr[m.body_jntadr[b] + k]] = q[k]
        if forward:
            mujoco.mj_forward(m, d)

    def forces(self, vec):
        F = np.zeros((len(self.bodies), 3))
        for i, b in enumerate(self.bodies):
            a = self.m.body_dofadr[b]
            F[i] = self.ax[b] @ vec[a:a + 3]
        return F


def load(xml):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = mujoco.MjModel.from_xml_string(xml)
    return m, [str(x.message) for x in w]


def compile_spec(spec):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = spec.compile()
    return m, [str(x.message) for x in w]


# ---------------------------------------------------------------- reference finite elements
def tet_grads(X, tets):
    """Reference shape-function gradients (ne,4,3) and signed volumes of linear tets."""
    X0 = X[tets[:, 0]]
    Dm = np.stack([X[tets[:, 1]] - X0, X[tets[:, 2]] - X0, X[tets[:, 3]] - X0], axis=2)  # (ne,3,3) cols
    Dinv = np.linalg.inv(Dm)
    g123 = Dinv  # row a-1 of Dm^{-1} is grad N_a
    g0 = -g123.sum(axis=1)
    G = np.concatenate([g0[:, None, :], g123], axis=1)
    vol = np.linalg.det(Dm) / 6
    return G, vol


def tet_force_basis(X, x, tets, model="svk"):
    """Nodal internal forces of linear tets for unit material parameters.

    Returns (f_mu, f_lam), each (n,3), such that f = mu*f_mu + lam*f_lam for
      svk: S = 2 mu E + lam tr(E) I, P = F S
      snh: P = mu (F - J F^-T) + (lam + mu) (J - 1) J F^-T   (MuJoCo's simplified SNH, see engine_core_util.h)
    """
    G, vol = tet_grads(X, tets)
    V = np.abs(vol)
    xe = x[tets]  # (ne,4,3)
    F = np.einsum("eai,eaj->eij", xe, G)  # sum_a x_a (x) gradN_a
    out = []
    if model == "svk":
        I = np.eye(3)
        E = 0.5 * (np.einsum("eki,ekj->eij", F, F) - I)
        trE = np.trace(E, axis1=1, axis2=2)
        P_mu = np.einsum("eik,ekj->eij", F, 2 * E)
        P_lam = F * trE[:, None, None]
        parts = [P_mu, P_lam]
    else:
        J = np.linalg.det(F)
        FinvT = np.transpose(np.linalg.inv(F), (0, 2, 1))
        P_lam = ((J - 1) * J)[:, None, None] * FinvT
        P_mu = F - J[:, None, None] * FinvT + P_lam
        parts = [P_mu, P_lam]
    n = x.shape[0]
    for P in parts:
        fe = -V[:, None, None] * np.einsum("eij,eaj->eai", P, G)  # (ne,4,3)
        f = np.zeros((n, 3))
        np.add.at(f, tets.reshape(-1), fe.reshape(-1, 3))
        out.append(f)
    return out


def tri_force_basis(X, x, tris, thick):
    """Membrane (plane-stress-form) SVK forces of linear triangles: f = mu*f_mu + lam2*f_lam with
    S = 2 mu E + lam2 tr(E) I (2x2, in the reference plane), P = F S; volume = area * thick."""
    n = x.shape[0]
    f_mu = np.zeros((n, 3))
    f_lam = np.zeros((n, 3))
    for t in tris:
        X0, X1, X2 = X[t]
        e1 = X1 - X0
        e2 = X2 - X0
        nrm = np.cross(e1, e2)
        A = 0.5 * np.linalg.norm(nrm)
        a1 = e1 / np.linalg.norm(e1)
        a2 = np.cross(nrm / np.linalg.norm(nrm), a1)
        Dm = np.array([[e1 @ a1, e2 @ a1], [e1 @ a2, e2 @ a2]])  # 2x2
        Dinv = np.linalg.inv(Dm)
        G = np.vstack([-(Dinv[0] + Dinv[1]), Dinv[0], Dinv[1]])  # (3 nodes, 2)
        xs = x[t]
        F = xs.T @ G  # (3,2)
        E = 0.5 * (F.T @ F - np.eye(2))
        for P, acc in [(F @ (2 * E), f_mu), (F * np.trace(E), f_lam)]:
            fe = -A * thick * (P @ G.T).T  # (3 nodes, 3)
            for a in range(3):
                acc[t[a]] += fe[a]
    return f_mu, f_lam


def fit_two(f, b1, b2, mask=None):
    """Least-squares f ~ c1 b1 + c2 b2; returns (c1, c2, relative residual)."""
    if mask is not None:
        f, b1, b2 = f[mask], b1[mask], b2[mask]
    A = np.stack([b1.ravel(), b2.ravel()], axis=1)
    c, *_ = np.linalg.lstsq(A, f.ravel(), rcond=None)
    res = np.linalg.norm(A @ c - f.ravel()) / max(np.linalg.norm(f.ravel()), 1e-300)
    return float(c[0]), float(c[1]), float(res)


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return os.path.abspath(p)
