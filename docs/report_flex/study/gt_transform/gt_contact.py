"""Contact problems on the exact bonded-layer Green's function (gt_transform.py).

Conventions as in gt_transform.py: z into the layer, p = compressive pressure, u_z > 0 = indentation.
Origin at the centre of the free face.  SI units.

A) Axisymmetric, frictionless, normal (C4 flat punch, C5 sphere):
   Fredholm integral equation of the 2nd kind (Copson/Sneddon representation; the classical route of
   Lebedev & Ufliand and of Hayes et al. 1972).  Writing H0[p](k) = int_0^a g(tau) cos(k tau) dtau,
       g(tau) + (2/pi) int_0^a K(tau,s) g(s) ds = (2/pi) mu/(1-nu) * h*(tau),
       K(tau,s) = [H(tau-s) + H(tau+s)]/2,  H(x) = int_0^oo dzz(k t) cos(k x) dk,
       dzz = mu k C_zz/(1-nu) - 1 (exponentially decaying),
       h*(tau) = (2/pi)... Abel transform of the prescribed surface displacement h(r):
           flat punch h = delta        -> h* = delta
           sphere h = delta - r^2/2R  -> h* = delta - tau^2/R
   F = 2 pi int_0^a g ;  p(r) = g(a)/sqrt(a^2-r^2) - int_r^a g'(tau)/sqrt(tau^2-r^2) dtau.
   Sphere: unknown a fixed by g(a) = 0 (bounded pressure); delta(a) = g2(a)/g1(a).
   Solved by Gauss-Legendre Nystrom (exponential convergence).
B) Axisymmetric ring boundary elements (independent check; also the BONDED/no-slip flat punch):
   piecewise-constant p (and q_r) on annuli, collocation at ring mid-radii.
C) 3-D rectangular boundary elements (C6 square punch, frictionless and no-slip) with quarter symmetry,
   influence = half-space closed form + layer correction (gt_transform.rect_layer); post-processing
   (taxel means, base reactions) by exact Fourier series of the piecewise-constant tractions on a large
   periodic cell.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, special

import gt_transform as gt
from gt_transform import Layer, PI

TAXEL_W = 10.2e-3
TAXEL_H = 71e-3 / 7.0


def taxel_bounds(i, j):
    x1 = -25.5e-3 + TAXEL_W * i
    y1 = -35.5e-3 + TAXEL_H * j
    return x1, x1 + TAXEL_W, y1, y1 + TAXEL_H


# =============================================================================== A) Fredholm
class Fredholm:
    """Nystrom solver for the axisymmetric normal contact on the bonded layer."""

    def __init__(self, L: Layer, a, N=48, umax=32.0, du=0.05, ngl=12):
        self.L, self.a, self.N = L, a, N
        x, w = np.polynomial.legendre.leggauss(N)
        self.tau = 0.5 * a * (x + 1.0)
        self.w = 0.5 * a * w
        u, wu = gt.gl_nodes(umax, du, ngl)
        self.ku = u / L.t
        self.wk = wu / L.t * gt.kernel_parts(u, L.nu)['dzz']     # weights for int dzz(kt) (.) dk
        self.c0 = 2.0 / PI * L.mu / (1.0 - L.nu)
        Kmat = self._K(self.tau, self.tau)
        self.A = np.eye(N) + (2.0 / PI) * Kmat * self.w[None, :]
        self.lu = linalg.lu_factor(self.A)

    def _H(self, x, deriv=False):
        X = np.outer(np.abs(np.ravel(x)), self.ku)
        if not deriv:
            return (np.cos(X) @ self.wk).reshape(np.shape(x))
        sgn = np.sign(np.ravel(x))
        return (sgn * (-(np.sin(X) * self.ku[None, :]) @ self.wk)).reshape(np.shape(x))

    def _K(self, tau, s, deriv=False):
        T, S = np.meshgrid(tau, s, indexing='ij')
        return 0.5 * (self._H(T - S, deriv) + self._H(T + S, deriv))

    def solve(self, rhs_fun):
        """rhs_fun(tau) = h*(tau) (Abel-transformed prescribed displacement).  Returns node values g."""
        return linalg.lu_solve(self.lu, self.c0 * rhs_fun(self.tau))

    def interp(self, g, tau, rhs_fun, deriv=False, rhs_deriv=None):
        """Nystrom interpolation of g (or g') at arbitrary tau in [0, a]."""
        tau = np.atleast_1d(np.asarray(tau, float))
        if not deriv:
            return self.c0 * rhs_fun(tau) - (2.0 / PI) * self._K(tau, self.tau) @ (self.w * g)
        return self.c0 * rhs_deriv(tau) - (2.0 / PI) * self._K(tau, self.tau, deriv=True) @ (self.w * g)

    def force(self, g):
        return 2.0 * PI * np.sum(self.w * g)

    def surface_w(self, g, r, rhs_fun, nq=64):
        """Surface displacement u_z(r) for any r (r <= a returns the computed field, a check)."""
        L = self.L
        r = np.atleast_1d(np.asarray(r, float))
        x, wq = np.polynomial.legendre.leggauss(nq)
        hs = np.empty(r.size)
        for i, rr in enumerate(r):
            th_max = np.arcsin(min(1.0, self.a / rr)) if rr > 0 else PI / 2
            if rr == 0:
                hs[i] = PI / 2 * self.interp(g, 0.0, rhs_fun)[0]
                continue
            th = 0.5 * th_max * (x + 1.0)
            hs[i] = 0.5 * th_max * np.sum(wq * self.interp(g, rr * np.sin(th), rhs_fun))
        # correction  int dzz(kt) J0(kr) P0(k) dk,  P0(k) = int g cos(k s) ds
        P0 = np.cos(np.outer(self.ku, self.tau)) @ (self.w * g)
        corr = special.j0(np.outer(r, self.ku)) @ (self.wk * P0)
        return (1.0 - L.nu) / L.mu * (hs + corr)

    def pressure(self, g, r, rhs_fun, rhs_deriv, nq=64):
        """p(r) = g(a)/sqrt(a^2-r^2) - int_r^a g'(tau)/sqrt(tau^2-r^2) dtau   (r < a)."""
        r = np.atleast_1d(np.asarray(r, float))
        x, wq = np.polynomial.legendre.leggauss(nq)
        ga = self.interp(g, self.a, rhs_fun)[0]
        out = np.empty(r.size)
        for i, rr in enumerate(r):
            smax = np.sqrt(self.a ** 2 - rr ** 2)
            s = 0.5 * smax * (x + 1.0)
            tt = np.sqrt(rr ** 2 + s ** 2)
            gp = self.interp(g, tt, rhs_fun, deriv=True, rhs_deriv=rhs_deriv)
            out[i] = ga / smax - 0.5 * smax * np.sum(wq * gp / tt)
        return out


def flat_punch_fredholm(L, a, N=48):
    fr = Fredholm(L, a, N)
    one = lambda t: np.ones_like(t)
    g = fr.solve(one)
    return fr, g, fr.force(g)          # K = F / delta  (delta = 1 m, linear)


def sphere_fredholm(L, R, a, N=48, fr=None):
    """Returns delta(a), F(a), node values g1, g2 (g = delta g1 - g2), Fredholm object."""
    if fr is None:
        fr = Fredholm(L, a, N)
    one = lambda t: np.ones_like(t)
    quad = lambda t: t ** 2 / R
    g1 = fr.solve(one)
    g2 = fr.solve(quad)
    g1a = fr.interp(g1, a, one)[0]
    g2a = fr.interp(g2, a, quad)[0]
    delta = g2a / g1a
    F = fr.force(delta * g1 - g2)
    return delta, F, g1, g2, fr


# =============================================================================== B) ring BEM
def graded_edges(a, n, kind='sine'):
    v = np.linspace(0.0, 1.0, n + 1)
    if kind == 'sine':
        return a * np.sin(0.5 * PI * v)
    return a * v


def ring_flat_punch(L, a, n, bonded=False):
    s = graded_edges(a, n)
    rc = 0.5 * (s[:-1] + s[1:])
    M = gt.ring_influence_matrices(L, rc, s, bonded=bonded)
    area = PI * (s[1:] ** 2 - s[:-1] ** 2)
    if not bonded:
        p = linalg.solve(M['wp'], np.ones(n))
        return np.sum(p * area), dict(s=s, p=p)
    A = np.block([[M['wp'], M['wq']], [M['up'], M['uq']]])
    b = np.concatenate([np.ones(n), np.zeros(n)])
    x = linalg.solve(A, b)
    p, q = x[:n], x[n:]
    return np.sum(p * area), dict(s=s, p=p, q=q)


def ring_sphere(L, R, delta, rmax, n, maxit=100):
    """Frictionless sphere by ring BEM with an active set (p >= 0, gap >= 0)."""
    s = np.linspace(0.0, rmax, n + 1)
    rc = 0.5 * (s[:-1] + s[1:])
    A = gt.ring_influence_matrices(L, rc, s)['wp']
    h = delta - rc ** 2 / (2 * R)
    act = h > 0
    for _ in range(maxit):
        p = np.zeros(n)
        p[act] = linalg.solve(A[np.ix_(act, act)], h[act])
        w = A @ p
        new = act.copy()
        new[act & (p < 0)] = False
        new[(~act) & (w < h)] = True
        if np.array_equal(new, act):
            break
        act = new
    area = PI * (s[1:] ** 2 - s[:-1] ** 2)
    a_c = s[1:][act].max() if act.any() else 0.0
    return np.sum(p * area), a_c, dict(s=s, p=p)


# =============================================================================== C) rectangular BEM
def quarter_mesh(hx, hy, n):
    xe = graded_edges(hx, n)
    ye = graded_edges(hy, n)
    X1, Y1 = np.meshgrid(xe[:-1], ye[:-1], indexing='ij')
    X2, Y2 = np.meshgrid(xe[1:], ye[1:], indexing='ij')
    return X1.ravel(), X2.ravel(), Y1.ravel(), Y2.ravel()


_ALL = ('zz', 'zx', 'zy', 'xz', 'xx', 'xy', 'yz', 'yy')


def influence_blocks(L, G, x0, y0, X1, X2, Y1, Y2, noslip, chunk=512):
    """Quarter-symmetric influence blocks at observation points (x0, y0) for unknowns on the quarter
    elements.  Frictionless: returns {'zp'}.  No-slip: 9 blocks z/x/y <- p/qx/qy."""
    no, ne = x0.size, X1.size
    names = ['zp'] if not noslip else ['zp', 'zx', 'zy', 'xp', 'xx', 'xy', 'yp', 'yx', 'yy']
    out = {k: np.zeros((no, ne)) for k in names}
    comps = ('zz',) if not noslip else _ALL
    for i0 in range(0, no, chunk):
        sl = slice(i0, min(no, i0 + chunk))
        xo = x0[sl, None]
        yo = y0[sl, None]
        for sx in (1, -1):
            a1, a2 = (X1, X2) if sx == 1 else (-X2, -X1)
            for sy in (1, -1):
                b1, b2 = (Y1, Y2) if sy == 1 else (-Y2, -Y1)
                if G is None:      # half-space (no layer correction)
                    r = gt.rect_halfspace(L, xo, yo, a1[None, :], a2[None, :], b1[None, :], b2[None, :])
                else:
                    r = gt.rect_layer(L, G, xo, yo, a1[None, :], a2[None, :], b1[None, :], b2[None, :],
                                      comps=comps)
                out['zp'][sl] += r['zz']
                if noslip:
                    out['zx'][sl] += sx * r['zx']
                    out['zy'][sl] += sy * r['zy']
                    out['xp'][sl] += r['xz']
                    out['xx'][sl] += sx * r['xx']
                    out['xy'][sl] += sy * r['xy']
                    out['yp'][sl] += r['yz']
                    out['yx'][sl] += sx * r['xy']
                    out['yy'][sl] += sy * r['yy']
    return out


def square_punch(L, G, n, noslip, hx=TAXEL_W / 2, hy=TAXEL_H / 2):
    """Rigid square punch, unit indentation (delta = 1 m; linear).  Returns dict with K and the quarter
    tractions."""
    X1, X2, Y1, Y2 = quarter_mesh(hx, hy, n)
    xc, yc = 0.5 * (X1 + X2), 0.5 * (Y1 + Y2)
    B = influence_blocks(L, G, xc, yc, X1, X2, Y1, Y2, noslip)
    ne = X1.size
    area = (X2 - X1) * (Y2 - Y1)
    if not noslip:
        p = linalg.solve(B['zp'], np.ones(ne))
        qx = qy = np.zeros(ne)
    else:
        A = np.block([[B['zp'], B['zx'], B['zy']], [B['xp'], B['xx'], B['xy']], [B['yp'], B['yx'], B['yy']]])
        b = np.concatenate([np.ones(ne), np.zeros(2 * ne)])
        x = linalg.solve(A, b)
        p, qx, qy = x[:ne], x[ne:2 * ne], x[2 * ne:]
    return dict(K=4.0 * np.sum(p * area), X1=X1, X2=X2, Y1=Y1, Y2=Y2, p=p, qx=qx, qy=qy, n=n,
                noslip=noslip)


def full_from_quarter(sol):
    """Expand quarter tractions to the full punch (4 images with symmetry signs)."""
    X1, X2, Y1, Y2 = sol['X1'], sol['X2'], sol['Y1'], sol['Y2']
    a1, a2, b1, b2, P, QX, QY = [], [], [], [], [], [], []
    for sx in (1, -1):
        for sy in (1, -1):
            a1.append(X1 if sx == 1 else -X2)
            a2.append(X2 if sx == 1 else -X1)
            b1.append(Y1 if sy == 1 else -Y2)
            b2.append(Y2 if sy == 1 else -Y1)
            P.append(sol['p'])
            QX.append(sx * sol['qx'])
            QY.append(sy * sol['qy'])
    c = np.concatenate
    return c(a1), c(a2), c(b1), c(b2), c(P), c(QX), c(QY)


def surface_field_realspace(L, G, sol, xo, yo):
    """u_x, u_y, u_z at arbitrary surface points (real-space influence; exact closed form + correction)."""
    a1, a2, b1, b2, P, QX, QY = full_from_quarter(sol)
    xo = np.asarray(xo, float)[:, None]
    yo = np.asarray(yo, float)[:, None]
    comps = _ALL if sol['noslip'] else ('zz', 'xz', 'yz')
    r = gt.rect_layer(L, G, xo, yo, a1[None], a2[None], b1[None], b2[None], comps=comps)
    uz = r['zz'] @ P
    ux = r['xz'] @ P
    uy = r['yz'] @ P
    if sol['noslip']:
        uz += r['zx'] @ QX + r['zy'] @ QY
        ux += r['xx'] @ QX + r['xy'] @ QY
        uy += r['xy'] @ QX + r['yy'] @ QY
    return ux, uy, uz


def _ft_1d(k, e1, e2):
    """int_{e1}^{e2} exp(-i k x) dx for k (Nk,) and intervals (Ne,) -> (Nk, Ne)."""
    K = k[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (np.exp(-1j * K * e1[None]) - np.exp(-1j * K * e2[None])) / (1j * K)
    z = k == 0
    out[z, :] = (e2 - e1)[None, :]
    return out


def taxel_maps_spectral(L, sol, Lp=0.1024, Ng=2048):
    """Per-taxel integrals on the 5 x 7 grid via exact Fourier series on a periodic cell of side Lp:
    surface contact force (3 comps), base reaction force (3 comps), mean surface displacement (3 comps).
    Exact for the piecewise-constant tractions up to (i) truncation at |k| < pi Ng/Lp and (ii) periodic
    images at distance Lp (both negligible: kernels decay exponentially; reported as checks)."""
    a1, a2, b1, b2, P, QX, QY = full_from_quarter(sol)
    m = np.fft.fftfreq(Ng, d=1.0 / Ng)
    k = 2 * PI * m / Lp
    Xe = _ft_1d(k, a1, a2)
    Ye = _ft_1d(k, b1, b2)
    Ph = (Xe * P[None]) @ Ye.T
    QXh = (Xe * QX[None]) @ Ye.T
    QYh = (Xe * QY[None]) @ Ye.T
    KX, KY = np.meshgrid(k, k, indexing='ij')
    KK = np.hypot(KX, KY)
    with np.errstate(invalid='ignore', divide='ignore'):
        cx = np.where(KK > 0, KX / KK, 0.0)
        cy = np.where(KK > 0, KY / KK, 0.0)
    kpos = np.where(KK > 0, KK, 1.0)
    C = gt.compliance(kpos, L)
    Pp = gt.kernel_parts(kpos * L.t, L.nu)
    z = KK == 0
    for key, val in (('Czz', L.t / L.M), ('CL', L.t / L.mu), ('CT', L.t / L.mu), ('Cc', 0.0)):
        C[key] = np.where(z, val, C[key])
    for key, val in (('Bzz', 1.0), ('Bzx', 0.0), ('Bxz', 0.0), ('BLL', 1.0), ('BTT', 1.0)):
        Pp[key] = np.where(z, val, Pp[key])
    kq = cx * QXh + cy * QYh
    Uz = -1j * C['Cc'] * kq + C['Czz'] * Ph
    Ux = C['CL'] * cx * kq + C['CT'] * (QXh - cx * kq) + 1j * C['Cc'] * cx * Ph
    Uy = C['CL'] * cy * kq + C['CT'] * (QYh - cy * kq) + 1j * C['Cc'] * cy * Ph
    Pb = Pp['Bzz'] * Ph - 1j * Pp['Bzx'] * kq
    Tbx = Pp['BLL'] * cx * kq + Pp['BTT'] * (QXh - cx * kq) - 1j * Pp['Bxz'] * cx * Ph
    Tby = Pp['BLL'] * cy * kq + Pp['BTT'] * (QYh - cy * kq) - 1j * Pp['Bxz'] * cy * Ph
    out = {key: np.zeros((5, 7, 3)) for key in ('surface_force', 'base_force', 'mean_u')}
    fields = dict(surface_force=(QXh, QYh, Ph), base_force=(Tbx, Tby, Pb), mean_u=(Ux, Uy, Uz))
    for i in range(5):
        for j in range(7):
            x1, x2, y1, y2 = taxel_bounds(i, j)
            tx = np.conj(_ft_1d(k, np.array([x1]), np.array([x2])))[:, 0]
            ty = np.conj(_ft_1d(k, np.array([y1]), np.array([y2])))[:, 0]
            for key, (fx, fy, fz) in fields.items():
                for c, f in enumerate((fx, fy, fz)):
                    v = np.real(tx @ f @ ty) / Lp ** 2
                    out[key][i, j, c] = v / (TAXEL_W * TAXEL_H) if key == 'mean_u' else v
    total = dict(surface=np.real(Ph[0, 0]), base=np.real(Pb[0, 0]))
    return out, total


# =============================================================================== spectral check of rect correction
def rect_correction_spectral(L, x0, y0, x1, x2, y1, y2, umax=25.0, du=0.05, ngl=12, nth=512):
    """Layer correction (total - half-space) for all 9 components of a uniform unit traction on a
    rectangle, by direct 2-D Fourier quadrature in polar coordinates of the (exponentially decaying)
    kernel corrections.  Independent of the real-space tables/splines used by gt.rect_correction."""
    u, wu = gt.gl_nodes(umax, du, ngl)
    k = u / L.t
    wk = wu / L.t
    kc = gt.k_times_corrections(k, L)        # k * dC
    th = 2 * PI * np.arange(nth) / nth
    KX = np.outer(k, np.cos(th))
    KY = np.outer(k, np.sin(th))
    cx, cy = np.cos(th)[None, :], np.sin(th)[None, :]

    def ft(kk, e1, e2):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(np.abs(kk) > 1e-12, (np.exp(-1j * kk * e1) - np.exp(-1j * kk * e2)) / (1j * kk),
                            e2 - e1)
    R = ft(KX, x1, x2) * ft(KY, y1, y2) * np.exp(1j * (KX * x0 + KY * y0))
    # d^2k = k dk dth ; dC = kc / k  -> integrand weight wk * kc (the k's cancel)
    W = (wk[:, None] * (2 * PI / nth)) / (2 * PI) ** 2
    dzz, dL, dT, dc = (kc[n][:, None] for n in ('zz', 'L', 'T', 'c'))

    def S(f):
        return np.real(np.sum(W * f * R))
    return dict(zz=S(dzz), xx=S(dL * cx * cx + dT * (1 - cx * cx)), yy=S(dL * cy * cy + dT * (1 - cy * cy)),
                xy=S((dL - dT) * cx * cy), xz=S(1j * dc * cx), yz=S(1j * dc * cy),
                zx=S(-1j * dc * cx), zy=S(-1j * dc * cy))
