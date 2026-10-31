"""Exact transform-space solution for a linear-elastic layer of thickness t perfectly bonded
to a rigid base, loaded on its free surface by normal and tangential tractions.

Conventions (used everywhere in this directory)
------------------------------------------------
* z points INTO the layer; free surface z = 0, bonded base z = t.
* Surface loads applied to the body at z = 0: in-plane traction q = (q_x, q_y) and the
  COMPRESSIVE pressure p (traction +p e_z, i.e. pushing into the layer).
  => sigma_xz(0) = -q_x, sigma_yz(0) = -q_y, sigma_zz(0) = -p.
* Displacement u_z > 0 = into the layer (indentation positive).
* 2-D Fourier transform f^(k) = int f(x) exp(-i k.x) dA ; k = |k|, u = k t, kh = k/|k|.
  Surface compliance (derived with Papkovich-Neuber potentials, checked with sympy in
  derive_sympy.py and with an independent Chebyshev/ODE solve below):
      u^_a = [C_L kh_a kh_b + C_T (d_ab - kh_a kh_b)] q^_b + i C_c kh_a p^
      u^_z = -i C_c kh_b q^_b + C_zz p^
      D'   = 2 kap cosh 2u + 1 + kap^2 + 4u^2 ,     kap = 3 - 4 nu
      C_zz = (1-nu)/(mu k) [2 kap sinh 2u - 4u] / D'        (Aleksandrov kernel)
      C_L  = (1-nu)/(mu k) [2 kap sinh 2u + 4u] / D'
      C_c  = [kap (kap-1)(cosh 2u - 1) - 4u^2] / (2 mu k D')
      C_T  = tanh(u) / (mu k)
  Half-space limits (u->oo): C_zz = C_L = (1-nu)/(mu k), C_T = 1/(mu k), C_c = (1-2nu)/(2 mu k)
  (Fourier transforms of Boussinesq/Cerruti).  Thin limits (u->0): C_zz -> t/M, C_L,C_T -> t/mu.
* Axisymmetric (Hankel) form: with P0 = H0[p], Q1 = H1[q_r] (H_n[f](k) = int f J_n(kr) r dr)
      U1 = C_L Q1 - C_c P0 ,   W0 = -C_c Q1 + C_zz P0 ,  u_r = H1^-1[U1], w = H0^-1[W0].
* Base tractions = force per area exerted BY the layer ON the rigid base (+z = into base):
      p_b^ = B_zz p^ - i B_zx kh.q^ ,  tau_b^ = [B_LL kh kh + B_TT (1 - kh kh)] q^ - i B_xz kh p^
      B_zz = (kap+1)((1+kap)cosh u + 2u sinh u)/D'   B_zx = (kap+1)((kap-1) sinh u + 2u cosh u)/D'
      B_xz = (kap+1)((1-kap) sinh u + 2u cosh u)/D'  B_LL = (kap+1)((1+kap)cosh u - 2u sinh u)/D'
      B_TT = 1/cosh u
Real-space point-load Green's functions = half-space closed form + smooth correction obtained
by numerical Hankel transform of the (exponentially decaying) kernel corrections.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import special, linalg
from scipy.interpolate import CubicSpline

PI = np.pi


# ----------------------------------------------------------------------------- material
@dataclass(frozen=True)
class Layer:
    E: float = 1.0e6       # Pa
    nu: float = 0.45
    t: float = 1.5e-3      # m

    @property
    def mu(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def M(self):           # P-wave / confined (oedometric) modulus
        return self.lam + 2.0 * self.mu

    @property
    def Kb(self):
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def Estar(self):       # plane-strain modulus E/(1-nu^2)
        return self.E / (1.0 - self.nu ** 2)

    @property
    def kap(self):
        return 3.0 - 4.0 * self.nu


# ----------------------------------------------------------------------------- kernels
def _sinh2u_minus_2u(u):
    """sinh(2u) - 2u without cancellation."""
    u = np.asarray(u, float)
    out = np.empty_like(u)
    s = u < 0.2
    x = 2.0 * u[s]
    x2 = x * x
    # series x^3/3! + x^5/5! + ... (enough terms for x < 0.4)
    term = x * x2 / 6.0
    acc = term.copy()
    for n in range(2, 12):
        term = term * x2 / ((2 * n) * (2 * n + 1))
        acc = acc + term
    out[s] = acc
    out[~s] = np.sinh(2.0 * u[~s]) - 2.0 * u[~s]
    return out


def kernel_parts(u, nu):
    """Dimensionless kernels as functions of u = k t (arrays), all multiplied by e^{-2u}
    in numerator and denominator where needed for stability.
    Returns dict with
      Lzz = mu k C_zz/(1-nu),  LL = mu k C_L/(1-nu),  Lc = 2 mu k C_c,  LT = mu k C_T,
      dzz = Lzz - 1, dL = LL - 1, dc = Lc - (1-2nu), dT = LT - 1   (corrections, no cancellation)
      Bzz, Bzx, Bxz, BLL, BTT (base transfer functions, dimensionless)
    """
    u = np.asarray(u, float)
    kap = 3.0 - 4.0 * nu
    e2 = np.exp(-2.0 * u)
    e4 = e2 * e2
    Dn = kap * (1.0 + e4) + (1.0 + kap ** 2 + 4.0 * u ** 2) * e2           # D' e^{-2u}
    num_zz = (2.0 * kap * _sinh2u_minus_2u(u) + 4.0 * u * (kap - 1.0)) * e2  # [2kap sinh2u - 4u] e^{-2u}
    num_L = kap * (1.0 - e4) + 4.0 * u * e2                                  # [2kap sinh2u + 4u] e^{-2u}
    sh = -np.expm1(-2.0 * u)                                                 # 1 - e^{-2u}
    num_c = 0.5 * kap * (kap - 1.0) * sh * sh - 4.0 * u ** 2 * e2            # [kap(kap-1)(cosh2u-1)-4u^2] e^{-2u}
    Lzz = num_zz / Dn
    LL = num_L / Dn
    Lc = num_c / Dn
    LT = np.tanh(u)
    dzz = -(2.0 * kap * e2 + (1.0 + 2.0 * u) ** 2 + kap ** 2) * e2 / Dn
    dL = -(2.0 * kap * e2 + (1.0 - 2.0 * u) ** 2 + kap ** 2) * e2 / Dn
    dc = -0.5 * ((kap - 1.0) * ((kap + 1.0) ** 2 + 4.0 * u ** 2) + 8.0 * u ** 2) * e2 / Dn
    dT = -2.0 * e2 / (1.0 + e2)
    # base transfer: numerators ~ e^{u}; multiply by e^{-2u}:  cosh u e^{-2u} = (e^{-u}+e^{-3u})/2
    em1 = np.exp(-u)
    em3 = em1 * e2
    chu = 0.5 * (em1 + em3)
    shu = 0.5 * (em1 - em3)
    Bzz = (kap + 1.0) * ((1.0 + kap) * chu + 2.0 * u * shu) / Dn
    Bzx = (kap + 1.0) * ((kap - 1.0) * shu + 2.0 * u * chu) / Dn
    Bxz = (kap + 1.0) * ((1.0 - kap) * shu + 2.0 * u * chu) / Dn
    BLL = (kap + 1.0) * ((1.0 + kap) * chu - 2.0 * u * shu) / Dn
    BTT = 2.0 * em1 / (1.0 + e2)
    return dict(Lzz=Lzz, LL=LL, Lc=Lc, LT=LT, dzz=dzz, dL=dL, dc=dc, dT=dT,
                Bzz=Bzz, Bzx=Bzx, Bxz=Bxz, BLL=BLL, BTT=BTT)


def compliance(k, L: Layer):
    """Dimensional surface compliances C_zz, C_L, C_c, C_T [m/Pa * m^2 per (m^2) -> m^3/N]
    for k > 0 (array)."""
    k = np.asarray(k, float)
    P = kernel_parts(k * L.t, L.nu)
    mu, nu = L.mu, L.nu
    return dict(Czz=(1 - nu) / (mu * k) * P['Lzz'], CL=(1 - nu) / (mu * k) * P['LL'],
                Cc=P['Lc'] / (2 * mu * k), CT=P['LT'] / (mu * k))


def k_times_corrections(k, L: Layer):
    """k * (C - C_halfspace) for each kernel (finite at k = 0, decays like u^2 e^{-2u})."""
    P = kernel_parts(np.asarray(k, float) * L.t, L.nu)
    mu, nu = L.mu, L.nu
    return dict(zz=(1 - nu) / mu * P['dzz'], L=(1 - nu) / mu * P['dL'],
                c=P['dc'] / (2 * mu), T=P['dT'] / mu)


# ----------------------------------------------------------------------------- independent ODE routes
def cheb(N):
    """Chebyshev differentiation matrix and Gauss-Lobatto points (Trefethen, Spectral Methods in MATLAB)."""
    x = np.cos(PI * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    Dm = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    Dm = Dm - np.diag(Dm.sum(axis=1))
    return Dm, x


def ode_chebyshev(kk, L: Layer, N=96):
    """Solve the Navier ODEs in z directly (no potentials) for one wavenumber kk.
    Real formulation: U_x = i V, U_z = W.
      mu V'' - (lam+2mu) k^2 V + (lam+mu) k W' = 0
      (lam+2mu) W'' - mu k^2 W - (lam+mu) k V' = 0
      z=0: mu (V' + k W) = -q_r ,  -lam k V + (lam+2mu) W' = -p ;   z=t: V = W = 0
    SH: U_y'' = k^2 U_y, mu U_y'(0) = -q_y, U_y(t) = 0.
    Returns the same quantities as the closed forms: Czz, Cc, CL, CT, Bzz, Bxz, Bzx, BLL, BTT."""
    lam, mu, t = L.lam, L.mu, L.t
    Dm, xi = cheb(N)
    # z = (1 - xi) t / 2 : xi = 1 -> z = 0 (row 0), xi = -1 -> z = t (row N)
    Dz = -(2.0 / t) * Dm
    D2 = Dz @ Dz
    I = np.eye(N + 1)
    n = N + 1
    A = np.zeros((2 * n, 2 * n))
    A[:n, :n] = mu * D2 - (lam + 2 * mu) * kk ** 2 * I
    A[:n, n:] = (lam + mu) * kk * Dz
    A[n:, :n] = -(lam + mu) * kk * Dz
    A[n:, n:] = (lam + 2 * mu) * D2 - mu * kk ** 2 * I
    # boundary rows
    A[0, :] = 0.0
    A[0, :n] = mu * Dz[0]
    A[0, n] = mu * kk                        # mu (V' + k W) at z=0
    A[n, :] = 0.0
    A[n, 0] = -lam * kk
    A[n, n:] = (lam + 2 * mu) * Dz[0]        # -lam k V + (lam+2mu) W' at z = 0
    A[N, :] = 0.0
    A[N, N] = 1.0                            # V(t) = 0
    A[n + N, :] = 0.0
    A[n + N, n + N] = 1.0                    # W(t) = 0
    out = {}
    for name, (qr, p) in dict(p=(0.0, 1.0), q=(1.0, 0.0)).items():
        b = np.zeros(2 * n)
        b[0] = -qr
        b[n] = -p
        s = np.linalg.solve(A, b)
        V, W = s[:n], s[n:]
        out[name] = (V[0], W[0], mu * (Dz[N] @ V), (lam + 2 * mu) * (Dz[N] @ W))
    # p = 1 :  W(0) = Czz, V(0) = Cc ;  B_xz = mu V'(t),  B_zz = -(lam+2mu) W'(t)
    # q = i (q_r = 1): V(0) = CL, W(0) = Cc ; B_LL = -mu V'(t), B_zx = -(lam+2mu) W'(t)
    res = dict(Czz=out['p'][1], Cc_from_p=out['p'][0], CL=out['q'][0], Cc_from_q=out['q'][1],
               Bzz=-out['p'][3], Bxz=out['p'][2], BLL=-out['q'][2], Bzx=-out['q'][3])
    # SH
    B = mu * (D2 - kk ** 2 * I)
    B[0, :] = mu * Dz[0]
    B[N, :] = 0.0
    B[N, N] = 1.0
    b = np.zeros(n)
    b[0] = -1.0
    Uy = np.linalg.solve(B, b)
    res['CT'] = Uy[0]
    res['BTT'] = -mu * (Dz[N] @ Uy)
    return res


def ode_expm(kk, L: Layer):
    """Third route: first-order state-space y=[V, W, S, T] (S = sigma_xz / i, T = sigma_zz),
    y' = A y, transfer matrix expm(A t). Accurate for k t <~ 10."""
    lam, mu, t = L.lam, L.mu, L.t
    l2m = lam + 2 * mu
    A = np.array([[0.0, -kk, 1.0 / mu, 0.0],
                  [lam * kk / l2m, 0.0, 0.0, 1.0 / l2m],
                  [4 * mu * (lam + mu) * kk ** 2 / l2m, 0.0, 0.0, -lam * kk / l2m],
                  [0.0, 0.0, kk, 0.0]])
    T = linalg.expm(A * t)
    res = {}
    for name, (qr, p) in dict(p=(0.0, 1.0), q=(1.0, 0.0)).items():
        S0, T0 = -qr, -p
        # V(t) = T[0,0]V0 + T[0,1]W0 + T[0,2]S0 + T[0,3]T0 = 0, same for W
        M2 = T[:2, :2]
        rhs = -(T[:2, 2] * S0 + T[:2, 3] * T0)
        V0, W0 = np.linalg.solve(M2, rhs)
        res[name] = (V0, W0)
    return dict(Czz=res['p'][1], Cc_from_p=res['p'][0], CL=res['q'][0], Cc_from_q=res['q'][1])


# ----------------------------------------------------------------------------- Hankel machinery
def gl_nodes(umax, du, ngl=16):
    """Composite Gauss-Legendre nodes/weights on [0, umax] with panels du."""
    g, w = np.polynomial.legendre.leggauss(ngl)
    edges = np.arange(0.0, umax + 1e-12, du)
    a, b = edges[:-1], edges[1:]
    x = (0.5 * (b - a)[:, None] * g[None, :] + 0.5 * (a + b)[:, None]).ravel()
    ww = (0.5 * (b - a)[:, None] * w[None, :]).ravel()
    return x, ww


class GreenTables:
    """Real-space correction Green's functions (point loads, unit force) on a radial grid,
    plus base-traction Green's functions.  All in SI units.

    Correction (total minus half-space) for SURFACE displacement, observation minus source = (xi, eta), r:
      u_z <- p   : gzz(r)
      u_a <- q_b : a(r) d_ab - f2(r) [[xi^2-eta^2, 2 xi eta],[2 xi eta, eta^2-xi^2]]_ab
      u_a <- p   : -xi_a f1(r)          u_z <- q_b : +xi_b f1(r)
    Base tractions (total; smooth, no half-space part):
      p_b <- p   : bzz(r)                p_b <- q_b : xi_b h_zx(r)   (h = b(r)/r)
      tau_a <- p : xi_a h_xz(r)          tau_a <- q_b : ab(r) d_ab - f2b(r) [[..same..]]
    """

    def __init__(self, L: Layer, rmax=0.064, dr=None, umax_s=25.0, umax_b=42.0, du=0.1, ngl=12):
        self.L = L
        t = L.t
        if dr is None:
            dr = t / 100.0
        r = np.arange(0.0, rmax + 0.5 * dr, dr)
        self.r = r
        mu, nu = L.mu, L.nu
        # surface corrections
        u, w = gl_nodes(umax_s, du, ngl)
        k = u / t
        wk = w / t
        P = kernel_parts(u, nu)
        kdzz = (1 - nu) / mu * P['dzz']        # k * dCzz
        kdL = (1 - nu) / mu * P['dL']
        kdc = P['dc'] / (2 * mu)
        kdT = P['dT'] / mu
        self.gzz = self._hankel(r, k, wk * kdzz, 0) / (2 * PI)
        self.a = self._hankel(r, k, wk * (kdL + kdT), 0) / (4 * PI)
        self.f2 = self._hankel_over(r, k, wk * (kdL - kdT), 2) / (4 * PI)
        self.f1 = self._hankel_over(r, k, wk * kdc, 1) / (2 * PI)
        # base tractions (total).  integrand B(k) * J_n * k  -> weights include k
        u2, w2 = gl_nodes(umax_b, du, ngl)
        k2 = u2 / t
        wk2 = w2 / t
        P2 = kernel_parts(u2, nu)
        self.bzz = self._hankel(r, k2, wk2 * k2 * P2['Bzz'], 0) / (2 * PI)
        self.hzx = self._hankel_over(r, k2, wk2 * k2 * P2['Bzx'], 1) / (2 * PI)
        self.hxz = self._hankel_over(r, k2, wk2 * k2 * P2['Bxz'], 1) / (2 * PI)
        self.ab = self._hankel(r, k2, wk2 * k2 * (P2['BLL'] + P2['BTT']), 0) / (4 * PI)
        self.f2b = self._hankel_over(r, k2, wk2 * k2 * (P2['BLL'] - P2['BTT']), 2) / (4 * PI)
        self.spl = {name: CubicSpline(r, getattr(self, name)) for name in
                    ['gzz', 'a', 'f2', 'f1', 'bzz', 'hzx', 'hxz', 'ab', 'f2b']}

    @staticmethod
    def _jn(n, X):
        if n == 0:
            return special.j0(X)
        if n == 1:
            return special.j1(X)
        return special.jv(n, X)

    @staticmethod
    def _hankel(r, k, wfac, n, chunk=256):
        out = np.empty(r.size)
        for i in range(0, r.size, chunk):
            rr = r[i:i + chunk]
            out[i:i + chunk] = GreenTables._jn(n, np.outer(rr, k)) @ wfac
        return out

    @staticmethod
    def _hankel_over(r, k, wfac, n, chunk=256):
        """int wfac J_n(k r) / r^n ; at r = 0 uses the limit k^n/(2^n n!)."""
        out = np.empty(r.size)
        for i in range(0, r.size, chunk):
            rr = r[i:i + chunk]
            X = np.outer(rr, k)
            with np.errstate(invalid='ignore', divide='ignore'):
                Jr = GreenTables._jn(n, X) / rr[:, None] ** n
            z = rr == 0.0
            if np.any(z):
                Jr[z, :] = k[None, :] ** n / (2.0 ** n * special.factorial(n))
            out[i:i + chunk] = Jr @ wfac
        return out

    def __call__(self, name, rr):
        return self.spl[name](rr)


# ----------------------------------------------------------------------------- half-space closed forms
def boussinesq_point(L: Layer, r):
    """Half-space, unit compressive normal point force: u_z (into solid), u_r (inward negative)."""
    mu, nu = L.mu, L.nu
    return (1 - nu) / (2 * PI * mu * r), -(1 - 2 * nu) / (4 * PI * mu * r)


def disk_uniform_halfspace(L: Layer, r, b):
    """Uniform unit pressure on a disk radius b, half-space: w(r) via complete elliptic integrals
    (Timoshenko & Goodier sec. 139 form; verified numerically in validate.py)."""
    r = np.asarray(r, float)
    mu, nu = L.mu, L.nu
    out = np.empty_like(r)
    inn = r <= b
    out[inn] = (2 * b / PI) * special.ellipe((r[inn] / b) ** 2)
    ro = r[~inn]
    m = (b / ro) ** 2
    out[~inn] = (2 * ro / PI) * (special.ellipe(m) - (1 - m) * special.ellipkm1(1 - m))
    return (1 - nu) / mu * out


def sneddon_flat_punch_K(L: Layer, a):
    return 2.0 * L.Estar * a


def hertz(L: Layer, R, delta):
    a = np.sqrt(R * delta)
    F = 4.0 / 3.0 * L.Estar * np.sqrt(R) * delta ** 1.5
    return F, a


# rectangle antiderivatives (double primitives) -------------------------------------
def _F_inv_r(xi, eta):
    """d2F/dxi deta = 1/rho"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ax = np.abs(xi)
        ay = np.abs(eta)
        t1 = np.where(ax > 0, xi * np.arcsinh(eta / np.where(ax > 0, ax, 1.0)), 0.0)
        t2 = np.where(ay > 0, eta * np.arcsinh(xi / np.where(ay > 0, ay, 1.0)), 0.0)
    return t1 + t2


def _F_xx_r3(xi, eta):
    """d2F/dxi deta = xi^2/rho^3  ->  F = eta asinh(xi/|eta|)"""
    ay = np.abs(eta)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(ay > 0, eta * np.arcsinh(xi / np.where(ay > 0, ay, 1.0)), 0.0)


def _F_xy_r3(xi, eta):
    """xi eta / rho^3 -> F = -rho"""
    return -np.sqrt(xi * xi + eta * eta)


def _F_x_r2(xi, eta):
    """xi/rho^2 -> F = (eta/2) ln(rho^2) + xi atan(eta/xi)"""
    r2 = xi * xi + eta * eta
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(r2 > 0, 0.5 * eta * np.log(np.where(r2 > 0, r2, 1.0)), 0.0)
        t2 = np.where(xi != 0, xi * np.arctan(eta / np.where(xi != 0, xi, 1.0)), 0.0)
    return t1 + t2


def _corner_sum(F, x0, y0, x1, x2, y1, y2):
    """int_{x1}^{x2} int_{y1}^{y2} g(x0-x', y0-y') dx'dy' given double primitive F of g."""
    a1, a2 = x0 - x2, x0 - x1
    b1, b2 = y0 - y2, y0 - y1
    return F(a2, b2) - F(a1, b2) - F(a2, b1) + F(a1, b1)


def rect_halfspace(L: Layer, x0, y0, x1, x2, y1, y2):
    """Half-space surface displacement at (x0,y0) due to uniform unit tractions on the
    rectangle [x1,x2]x[y1,y2].  Returns dict of the 3x3 response: key 'zz' = u_z <- p,
    'xx' = u_x <- q_x, 'yy', 'xy' (= u_x <- q_y = u_y <- q_x), 'xz' (u_x <- p), 'yz' (u_y <- p),
    'zx' (u_z <- q_x) = -'xz', 'zy' = -'yz'."""
    mu, nu = L.mu, L.nu
    I1 = _corner_sum(_F_inv_r, x0, y0, x1, x2, y1, y2)
    Ixx = _corner_sum(_F_xx_r3, x0, y0, x1, x2, y1, y2)
    Iyy = _corner_sum(lambda a, b: _F_xx_r3(b, a), x0, y0, x1, x2, y1, y2)
    Ixy = _corner_sum(_F_xy_r3, x0, y0, x1, x2, y1, y2)
    Ix = _corner_sum(_F_x_r2, x0, y0, x1, x2, y1, y2)
    Iy = _corner_sum(lambda a, b: _F_x_r2(b, a), x0, y0, x1, x2, y1, y2)
    c0 = 1.0 / (2 * PI * mu)
    cc = (1 - 2 * nu) / (4 * PI * mu)
    return dict(zz=(1 - nu) * c0 * I1,
                xx=c0 * ((1 - nu) * I1 + nu * Ixx), yy=c0 * ((1 - nu) * I1 + nu * Iyy),
                xy=c0 * nu * Ixy,
                xz=-cc * Ix, yz=-cc * Iy, zx=cc * Ix, zy=cc * Iy)


_G2, _W2 = np.polynomial.legendre.leggauss(2)
_G3, _W3 = np.polynomial.legendre.leggauss(3)


def rect_correction(G: GreenTables, x0, y0, x1, x2, y1, y2, order=2, comps=('zz',)):
    """Layer correction (smooth) for the rectangle influence, Gauss order x order per element."""
    g, w = (_G2, _W2) if order == 2 else (_G3, _W3) if order == 3 else np.polynomial.legendre.leggauss(order)
    hx = 0.5 * (x2 - x1)
    hy = 0.5 * (y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    out = {c: 0.0 for c in comps}
    for i, gi in enumerate(g):
        for j, gj in enumerate(g):
            xi = x0 - (cx + hx * gi)
            eta = y0 - (cy + hy * gj)
            r = np.sqrt(xi * xi + eta * eta)
            wt = w[i] * w[j] * hx * hy
            need_a = any(c in comps for c in ('xx', 'yy'))
            need_f2 = any(c in comps for c in ('xx', 'yy', 'xy'))
            need_f1 = any(c in comps for c in ('xz', 'yz', 'zx', 'zy'))
            if 'zz' in comps:
                out['zz'] = out['zz'] + wt * G('gzz', r)
            if need_a:
                a = G('a', r)
            if need_f2:
                f2 = G('f2', r)
                d2 = xi * xi - eta * eta
            if need_f1:
                f1 = G('f1', r)
            if 'xx' in comps:
                out['xx'] = out['xx'] + wt * (a - f2 * d2)
            if 'yy' in comps:
                out['yy'] = out['yy'] + wt * (a + f2 * d2)
            if 'xy' in comps:
                out['xy'] = out['xy'] + wt * (-2.0 * f2 * xi * eta)
            if 'xz' in comps:
                out['xz'] = out['xz'] + wt * (-xi * f1)
            if 'yz' in comps:
                out['yz'] = out['yz'] + wt * (-eta * f1)
            if 'zx' in comps:
                out['zx'] = out['zx'] + wt * (xi * f1)
            if 'zy' in comps:
                out['zy'] = out['zy'] + wt * (eta * f1)
    return out


def rect_layer(L: Layer, G: GreenTables, x0, y0, x1, x2, y1, y2, comps=('zz',), order=2):
    hs = rect_halfspace(L, x0, y0, x1, x2, y1, y2)
    co = rect_correction(G, x0, y0, x1, x2, y1, y2, order=order, comps=comps)
    return {c: hs[c] + co[c] for c in comps}


# ----------------------------------------------------------------------------- axisymmetric ring elements
def _I00_ring(r, s):
    """int_0^oo J0(k s) J0(k r) dk = 2 K(m)/(pi (r+s)),  m = 4 r s/(r+s)^2 (log-singular at r=s)."""
    p = ((r - s) / (r + s)) ** 2          # 1 - m, computed without cancellation
    return 2.0 * special.ellipkm1(p) / (PI * (r + s))


def _I11_ring(r, s):
    """int_0^oo J1(k s) J1(k r) dk = Q_{1/2}(chi)/(pi sqrt(r s)),  chi = (r^2+s^2)/(2 r s).
    Q_{1/2}(chi) = chi kk K(m) - (2/kk) E(m), m = kk^2 = 2/(chi+1) = 4rs/(r+s)^2   (m > 0.5)
                 = (pi/2)(2 chi)^(-3/2) 2F1(5/4, 3/4; 2; chi^-2)                     (m <= 0.5)
    (both forms checked against direct quadrature in validate.py)."""
    r = np.asarray(r, float)
    s = np.asarray(s, float)
    r, s = np.broadcast_arrays(r, s)
    p = ((r - s) / (r + s)) ** 2
    m = 1.0 - p
    chi = (r * r + s * s) / (2 * r * s)
    out = np.empty(r.shape)
    hi = m > 0.5
    kk = np.sqrt(m[hi])
    out[hi] = chi[hi] * kk * special.ellipkm1(p[hi]) - (2.0 / kk) * special.ellipe(m[hi])
    lo = ~hi
    out[lo] = (PI / 2) * (2 * chi[lo]) ** (-1.5) * special.hyp2f1(1.25, 0.75, 2.0, chi[lo] ** -2)
    return out / (PI * np.sqrt(r * s))


def _ring_quad(f, r, s1, s2, n=32):
    """int_{s1}^{s2} f(r, s) s ds where f may be log-singular at s = r.
    Split at r when s1 < r < s2; cubic clustering s = a + L v^3 toward the (near-)singular end."""
    r, s1, s2 = (np.array(a, float) for a in np.broadcast_arrays(r, s1, s2))
    g, w = np.polynomial.legendre.leggauss(n)
    v = 0.5 * (g + 1.0)
    wv = 0.5 * w

    def seg(rr, a, b, cl_at_a):
        Lr = b - a
        res = np.zeros(a.shape)
        for vi, wi in zip(v, wv):
            s = np.where(cl_at_a, a + Lr * vi ** 3, b - Lr * vi ** 3)
            res += wi * 3 * vi ** 2 * Lr * f(rr, s) * s
        return res

    out = np.zeros(r.shape)
    inside = (r > s1) & (r < s2)
    o = ~inside
    if np.any(o):
        out[o] = seg(r[o], s1[o], s2[o], r[o] <= s1[o])
    if np.any(inside):
        ri = r[inside]
        out[inside] = (seg(ri, s1[inside], ri, np.zeros(ri.shape, bool)) +
                       seg(ri, ri, s2[inside], np.ones(ri.shape, bool)))
    return out


def _disk_w(r, b):
    """int_0^oo (b J1(kb)/k) J0(kr) dk  (vectorised over r and b; 0 for b = 0)."""
    r, b = (np.array(a, float) for a in np.broadcast_arrays(r, b))
    o = np.zeros(r.shape)
    inn = (r <= b) & (b > 0)
    o[inn] = (2 * b[inn] / PI) * special.ellipe((r[inn] / b[inn]) ** 2)
    oo = (r > b) & (b > 0)
    m = (b[oo] / r[oo]) ** 2
    o[oo] = (2 * r[oo] / PI) * (special.ellipe(m) - (1 - m) * special.ellipkm1(1 - m))
    return o


def ring_influence_halfspace(L: Layer, r, s1, s2, which):
    """Half-space response at radius r to UNIFORM unit traction on the annulus [s1, s2].
    which: 'wp' (w <- p), 'up' (u_r <- p), 'wq' (w <- q_r), 'uq' (u_r <- q_r)."""
    mu, nu = L.mu, L.nu
    r, s1, s2 = (np.array(a, float) for a in np.broadcast_arrays(r, s1, s2))
    if which == 'wp':
        return (1 - nu) / mu * (_disk_w(r, s2) - _disk_w(r, s1))
    if which == 'up':      # u_r = -(1-2nu)/(2mu) * b int J1(kb)J1(kr)/k dk ; that integral = (1/2) min/max
        def B(b):
            return np.where(r < b, 0.5 * r, 0.5 * b * b / np.maximum(r, 1e-300))
        return -(1 - 2 * nu) / (2 * mu) * (B(s2) - B(s1))
    if which == 'wq':      # w = -(1-2nu)/(2mu) int Q1 J0 dk = -(1-2nu)/(2mu) (b - r)_+
        def Cq(b):
            return np.maximum(b - r, 0.0)
        return -(1 - 2 * nu) / (2 * mu) * (Cq(s2) - Cq(s1))
    if which == 'uq':      # u_r = (1-nu)/mu int_s1^s2 s I11(s, r) ds
        return (1 - nu) / mu * _ring_quad(_I11_ring, r, s1, s2)
    raise ValueError(which)


def _struve(n, x):
    """scipy.special.struve with a guard: scipy 1.17 returns NaN at isolated arguments
    (found: struve(0, x) near its zero x = 25.765365...); such points are replaced by the mean of the two
    neighbours x(1 +- 1e-6) (H_n is smooth; absolute error ~1e-11)."""
    v = special.struve(n, x)
    bad = ~np.isfinite(v)
    if np.any(bad):
        xb = x[bad]
        v[bad] = 0.5 * (special.struve(n, xb * (1 + 1e-6)) + special.struve(n, xb * (1 - 1e-6)))
    return v


def ring_transforms(k, s1, s2, which):
    """Hankel transforms of uniform unit traction on annuli [s1,s2] (vectors) at k (vector):
    'P0' = int J0(ks) s ds = [s J1(ks)/k] ;  'Q1' = int J1(ks) s ds = (pi/(2k^2)) [x (J1 H0 - J0 H1)](x=ks)."""
    K = np.asarray(k, float)[None, :]
    s1 = np.asarray(s1, float)
    s2 = np.asarray(s2, float)
    if which == 'P0':
        def F(s):
            return s[:, None] * special.j1(s[:, None] * K) / K
        return F(s2) - F(s1)
    if which == 'Q1':
        def F(s):
            x = s[:, None] * K
            return (PI / (2 * K * K)) * x * (special.j1(x) * _struve(0, x) - special.j0(x) * _struve(1, x))
        return F(s2) - F(s1)
    raise ValueError(which)


def ring_influence_matrices(L: Layer, rc, s, bonded=False, umax=25.0, du=0.1, ngl=12):
    """Influence matrices for piecewise-constant tractions on annuli [s_j, s_j+1] evaluated at
    radii rc.  Returns dict with 'wp' (and 'up','wq','uq' if bonded):
       half-space part (closed form / ring quadrature) + layer correction (Hankel integral of the
       exponentially decaying kernel correction, done as a matrix product over Gauss nodes)."""
    rc = np.asarray(rc, float)
    s = np.asarray(s, float)
    s1, s2 = s[:-1], s[1:]
    R, S1 = np.meshgrid(rc, s1, indexing='ij')
    _, S2 = np.meshgrid(rc, s2, indexing='ij')
    u, w = gl_nodes(umax, du, ngl)
    k = u / L.t
    wk = w / L.t
    kc = k_times_corrections(k, L)
    J0 = special.j0(np.outer(rc, k))
    P0 = ring_transforms(k, s1, s2, 'P0')
    out = {}
    out['wp'] = ring_influence_halfspace(L, R, S1, S2, 'wp') + (J0 * (wk * kc['zz'])) @ P0.T
    if bonded:
        J1 = special.j1(np.outer(rc, k))
        Q1 = ring_transforms(k, s1, s2, 'Q1')
        out['up'] = ring_influence_halfspace(L, R, S1, S2, 'up') - (J1 * (wk * kc['c'])) @ P0.T
        out['wq'] = ring_influence_halfspace(L, R, S1, S2, 'wq') - (J0 * (wk * kc['c'])) @ Q1.T
        out['uq'] = ring_influence_halfspace(L, R, S1, S2, 'uq') + (J1 * (wk * kc['L'])) @ Q1.T
    return out


# ----------------------------------------------------------------------------- simple closed forms (P0, G1)
def closed_forms(L: Layer, A=51e-3 * 71e-3):
    mu, lam, M, Kb = L.mu, L.lam, L.M, L.Kb
    t = L.t
    eps = 1.0
    return dict(
        mu_G=mu, lam=lam, M=M, K_bulk=Kb, Estar=L.Estar,
        P0_uniaxial_strain=dict(sigma_zz_per_strain=M, sigma_xx_per_strain=lam, energy_density_per_strain2=M / 2,
                                ratio_if_lambda_doubled=(2 * lam + 2 * mu) / M),
        P0_simple_shear=dict(sigma_xz_per_gamma=mu, energy_density_per_gamma2=mu / 2, ratio_if_lambda_doubled=1.0),
        P0_hydrostatic=dict(mean_stress_per_strain=3 * Kb, energy_density_per_strain2=9 * Kb / 2,
                            ratio_if_lambda_doubled=(6 * lam + 2 * mu) / (3 * lam + 2 * mu)),
        G1_confined_K_N_per_m=M * A / t, G1_confined_K_N_per_um=M * A / t * 1e-6,
        G1_confined_K_lambda_doubled_N_per_um=(2 * lam + 2 * mu) * A / t * 1e-6,
        simple_shear_infinite_layer_K_N_per_um=mu * A / t * 1e-6,
        A_m2=A, t_m=t,
        F1_workaround_inputs=dict(poisson_in=L.nu / (2 * (1 - L.nu)),
                                  young_in=2 * mu * (1 + L.nu / (2 * (1 - L.nu)))),
    )
