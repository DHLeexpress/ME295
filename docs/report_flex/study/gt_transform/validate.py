"""Validation of gt_transform.py (independent of MuJoCo).  Writes validate.json + PNGs.

V1  closed-form kernels vs Chebyshev collocation of the Navier ODEs (no potentials) and vs the
    4x4 matrix-exponential transfer matrix, for several k t and nu.
V2  limits: k t -> oo (half-space: Boussinesq/Cerruti transforms) and k t -> 0 (t/M, t/G).
V3  real-space tables: total point-load response integrates to C(0) (t/M, t/G, 1 for base),
    exponential decay of the total, table resolution test (coarse vs fine Hankel quadrature).
V4  half-space rectangle closed forms (Love-type double primitives) vs brute-force quadrature.
V5  ring-load closed forms (disk/elliptic, I00, I11) vs quadrature of Bessel products.
V6  annulus Hankel transforms (Struve form) vs Gauss quadrature.
V7  forward check: uniform pressure on a 10.2 x 10.14 mm square on the LAYER, surface u_z at
    several points, by (a) rect_layer (half-space closed form + correction) and (b) a direct
    2-D Fourier quadrature of the full kernel C_zz(k) with the half-space tail added analytically.
"""
import json
import sys
import time
import numpy as np
from scipy import integrate, special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
import gt_transform as g

out = {}
NUS = [0.45, 0.49, 0.499]
T = 1.5e-3

# ---------------------------------------------------------------- V1
v1 = {}
uts = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
for nu in NUS:
    L = g.Layer(nu=nu)
    rows = []
    for ut in uts:
        kk = ut / L.t
        C = g.compliance(np.array([kk]), L)
        P = g.kernel_parts(np.array([ut]), nu)
        ch = g.ode_chebyshev(kk, L, N=110)
        ch2 = g.ode_chebyshev(kk, L, N=150)
        cf = dict(Czz=C['Czz'][0], CL=C['CL'][0], Cc=C['Cc'][0], CT=C['CT'][0],
                  Bzz=P['Bzz'][0], Bzx=P['Bzx'][0], Bxz=P['Bxz'][0], BLL=P['BLL'][0], BTT=P['BTT'][0])
        scale = {'Czz': cf['CL'], 'CL': cf['CL'], 'Cc': cf['CL'], 'CT': cf['CT'],
                 'Bzz': 1, 'Bzx': 1, 'Bxz': 1, 'BLL': 1, 'BTT': 1}
        row = dict(kt=ut)
        for key in ['Czz', 'CL', 'CT', 'Bzz', 'Bzx', 'Bxz', 'BLL', 'BTT']:
            row['cheb_' + key] = abs(ch[key] - cf[key]) / abs(scale[key])
        row['cheb_Cc(p)'] = abs(ch['Cc_from_p'] - cf['Cc']) / abs(cf['CL'])
        row['cheb_Cc(q)'] = abs(ch['Cc_from_q'] - cf['Cc']) / abs(cf['CL'])
        row['cheb_N110_vs_N150'] = abs(ch['Czz'] - ch2['Czz']) / abs(cf['Czz'])
        if ut <= 5.0:
            ex = g.ode_expm(kk, L)
            row['expm_Czz'] = abs(ex['Czz'] - cf['Czz']) / abs(cf['Czz'])
            row['expm_CL'] = abs(ex['CL'] - cf['CL']) / abs(cf['CL'])
            row['expm_Cc'] = abs(ex['Cc_from_p'] - cf['Cc']) / abs(cf['CL'])
        rows.append(row)
    v1[str(nu)] = rows
    worst = max(max(v for kk_, v in r.items() if kk_ != 'kt') for r in rows)
    print(f'V1 nu={nu}: worst relative difference (closed form vs ODE routes) = {worst:.2e}')
out['V1_kernel_vs_ode'] = v1

# ---------------------------------------------------------------- V2
v2 = {}
for nu in NUS:
    L = g.Layer(nu=nu)
    big = np.array([60.0]) / L.t
    small = np.array([1e-5]) / L.t
    Cb = g.compliance(big, L)
    Cs = g.compliance(small, L)
    mu = L.mu
    v2[str(nu)] = dict(
        halfspace_Czz=float(Cb['Czz'][0] / ((1 - nu) / (mu * big[0]))),
        halfspace_CL=float(Cb['CL'][0] / ((1 - nu) / (mu * big[0]))),
        halfspace_CT=float(Cb['CT'][0] / (1 / (mu * big[0]))),
        halfspace_Cc=float(Cb['Cc'][0] / ((1 - 2 * nu) / (2 * mu * big[0]))),
        thin_Czz_over_t_by_M=float(Cs['Czz'][0] / (L.t / L.M)),
        thin_CL_over_t_by_mu=float(Cs['CL'][0] / (L.t / mu)),
        thin_CT_over_t_by_mu=float(Cs['CT'][0] / (L.t / mu)))
    print('V2', nu, v2[str(nu)])
out['V2_limits'] = v2

# ---------------------------------------------------------------- V3
v3 = {}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for nu in NUS:
    L = g.Layer(nu=nu)
    t0 = time.time()
    G = g.GreenTables(L)
    tb = time.time() - t0
    Gf = g.GreenTables(L, umax_s=32.0, umax_b=55.0, du=0.05, ngl=16, dr=L.t / 150)
    r = G.r
    mu = L.mu
    with np.errstate(divide='ignore'):
        hs_zz = (1 - nu) / (2 * np.pi * mu * r)
        hs_a = (2 - nu) / (4 * np.pi * mu * r)
    tot_zz = hs_zz + G.gzz
    tot_a = hs_a + G.a
    # moments: int 2 pi r G dr = C(0).  Half-space part integrated analytically up to rmax:
    rmax = r[-1]
    m_zz = (1 - nu) / mu * rmax + integrate.simpson(2 * np.pi * r * G.gzz, x=r)
    # in-plane: int (a I - b Q) dA -> isotropic part int a dA = (C_L(0)+C_T(0))/2 = t/mu
    m_a = (2 - nu) / (2 * mu) * rmax + integrate.simpson(2 * np.pi * r * G.a, x=r)
    m_b = integrate.simpson(2 * np.pi * r * G.bzz, x=r)
    m_ab = integrate.simpson(2 * np.pi * r * G.ab, x=r)
    # far-field: total at r = 20 t relative to the half-space value there
    i20 = np.searchsorted(r, 20 * L.t)
    i10 = np.searchsorted(r, 10 * L.t)
    res = dict(
        table_build_s=tb,
        moment_zz_over_t_by_M=float(m_zz / (L.t / L.M)),
        moment_inplane_over_t_by_mu=float(m_a / (L.t / mu)),
        base_pressure_total=float(m_b), base_shear_isotropic_total=float(m_ab),
        total_zz_at_10t_over_hs=float(tot_zz[i10] / hs_zz[i10]),
        total_zz_at_20t_over_hs=float(tot_zz[i20] / hs_zz[i20]),
        total_a_at_20t_over_hs=float(tot_a[i20] / hs_a[i20]),
    )
    # table accuracy: compare default vs fine tables on a random set of radii
    rr = np.linspace(0, 0.06, 997)
    for name in ['gzz', 'a', 'f2', 'f1', 'bzz', 'hzx', 'hxz', 'ab', 'f2b']:
        d = np.max(np.abs(G(name, rr) - Gf(name, rr)))
        sc = np.max(np.abs(Gf(name, rr)))
        res['table_maxdiff_rel_' + name] = float(d / sc)
    v3[str(nu)] = res
    print('V3', nu, {k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in res.items()})
    rm = r[1:] / L.t
    axs[0].semilogy(rm, np.abs(tot_zz[1:]) * 2 * np.pi * mu * L.t, label=f'layer total, nu={nu}')
    axs[1].plot(rm, G.bzz[1:] * L.t ** 2, label=f'nu={nu}')
    axs[2].plot(rm, G.hxz[1:] * r[1:] * L.t ** 2, label=f'nu={nu}')
axs[0].semilogy(rm, 2 * np.pi * (1 - 0.45) / (2 * np.pi * rm), 'k--', label='half-space (nu=0.45)')
axs[0].set_xlabel('r / t'); axs[0].set_ylabel('|u_z| 2 pi mu t / F'); axs[0].set_xlim(0, 25)
axs[0].set_title('surface u_z, unit normal point force'); axs[0].legend(fontsize=8)
axs[1].set_xlabel('r / t'); axs[1].set_ylabel('base pressure * t^2 / F'); axs[1].set_xlim(0, 8)
axs[1].set_title('base pressure, unit normal point force'); axs[1].legend(fontsize=8)
axs[2].set_xlabel('r / t'); axs[2].set_ylabel('radial base shear * t^2 / F'); axs[2].set_xlim(0, 8)
axs[2].set_title('base shear (outward +), unit normal point force'); axs[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig('fig_green_realspace.png', dpi=130); plt.close(fig)
out['V3_realspace_tables'] = v3

# ---------------------------------------------------------------- V4 rectangle half-space
L = g.Layer(nu=0.45)
mu, nu = L.mu, L.nu
x1, x2, y1, y2 = -1.0e-3, 2.0e-3, -0.5e-3, 1.5e-3
pts = [(4e-3, 0.3e-3), (0.5e-3, 3.0e-3), (-2.5e-3, -1.7e-3), (0.2e-3, 0.4e-3), (1.9e-3, -0.45e-3)]
v4 = []
for (x0, y0) in pts:
    cf = g.rect_halfspace(L, np.array(x0), np.array(y0), x1, x2, y1, y2)

    def kern(comp):
        def f(yp, xp):
            xi, eta = x0 - xp, y0 - yp
            r = np.hypot(xi, eta)
            c0 = 1 / (2 * np.pi * mu)
            cc = (1 - 2 * nu) / (4 * np.pi * mu)
            return dict(zz=(1 - nu) * c0 / r, xx=c0 * ((1 - nu) / r + nu * xi ** 2 / r ** 3),
                        yy=c0 * ((1 - nu) / r + nu * eta ** 2 / r ** 3), xy=c0 * nu * xi * eta / r ** 3,
                        xz=-cc * xi / r ** 2, yz=-cc * eta / r ** 2)[comp]
        return f
    row = dict(x0=x0, y0=y0)
    inside = (x1 < x0 < x2) and (y1 < y0 < y2)
    for comp in ['zz', 'xx', 'yy', 'xy', 'xz', 'yz']:
        if inside:   # split at the singular point so it sits on sub-rectangle corners
            val = 0.0
            for (a, b) in [(x1, x0), (x0, x2)]:
                for (c, d) in [(y1, y0), (y0, y2)]:
                    val += integrate.dblquad(kern(comp), a, b, c, d, epsabs=1e-16, epsrel=1e-11)[0]
        else:
            val = integrate.dblquad(kern(comp), x1, x2, y1, y2, epsabs=1e-16, epsrel=1e-11)[0]
        row[comp] = float(abs(val - cf[comp]) / max(abs(cf['zz']), 1e-30))
    v4.append(row)
print('V4 rectangle closed forms, max rel diff vs dblquad:', max(max(v for k_, v in r.items() if k_ not in ('x0', 'y0')) for r in v4))
out['V4_rect_halfspace'] = v4

# ---------------------------------------------------------------- V5 ring closed forms
v5 = {}
# disk uniform pressure: compare with radial integration of the ring kernel I00 (log singular)
for (r, b) in [(0.0, 1.0), (0.3, 1.0), (0.999, 1.0), (1.5, 1.0), (4.0, 1.0)]:
    cf = g._disk_w(np.array(r), np.array(b))
    if r == 0:
        num = b
    else:
        f = lambda s: s * 2 * special.ellipk(4 * r * s / (r + s) ** 2) / (np.pi * (r + s))
        pts_ = [r] if r < b else None
        num = integrate.quad(f, 0, b, points=pts_, limit=400, epsabs=1e-14)[0]
    v5[f'disk_w r={r} b={b}'] = float(abs(cf - num) / abs(num))


def bessel_prod_num(n, r, s, eps=0.0):
    # int_0^oo J_n(ks) J_n(kr) dk by splitting into many panels + Shanks-free brute force with a
    # very long range and tail averaging (product decays like 1/k)
    f = lambda k: special.jv(n, k * s) * special.jv(n, k * r)
    tot = 0.0
    edges = np.linspace(0, 4000.0, 40001)
    gq, gw = np.polynomial.legendre.leggauss(12)
    a, bb = edges[:-1], edges[1:]
    kq = (0.5 * (bb - a)[:, None] * gq + 0.5 * (a + bb)[:, None])
    wq = 0.5 * (bb - a)[:, None] * gw
    vals = (f(kq) * wq).sum(axis=1)
    cs = np.cumsum(vals)
    # tail ~ oscillatory with slowly decaying amplitude: average the partial sums over the last part
    return cs[-20000:].mean()


for (r, s) in [(0.5, 1.0), (0.9, 1.0), (2.0, 0.7), (0.05, 1.0)]:
    v5[f'I00 r={r} s={s}'] = float(abs(g._I00_ring(np.array(r), np.array(s)) - bessel_prod_num(0, r, s)) /
                                   abs(g._I00_ring(np.array(r), np.array(s))))
    v5[f'I11 r={r} s={s}'] = float(abs(g._I11_ring(np.array(r), np.array(s)) - bessel_prod_num(1, r, s)) /
                                   abs(g._I11_ring(np.array(r), np.array(s))))
# I11 branch continuity at m = 0.5
rs = np.linspace(0.05, 0.4, 2000)
vals = g._I11_ring(rs, np.ones_like(rs))
v5['I11 branch smoothness max |2nd diff|/value'] = float(np.max(np.abs(np.diff(vals, 2))) / np.max(np.abs(vals)))
# annulus u_r <- q_r: ring quadrature with log singularity vs adaptive quad
for (r, s1, s2) in [(0.5, 0.2, 1.0), (0.2, 0.25, 0.3), (1.3, 0.2, 1.0)]:
    cf = g._ring_quad(g._I11_ring, np.array(r), np.array(s1), np.array(s2))
    f = lambda s: s * g._I11_ring(np.array(r), np.array(s))
    pts_ = [r] if s1 < r < s2 else None
    num = integrate.quad(f, s1, s2, points=pts_, limit=400, epsabs=1e-15)[0]
    v5[f'ring_quad I11 r={r} [{s1},{s2}]'] = float(abs(cf - num) / abs(num))
print('V5', {k_: f'{v:.1e}' for k_, v in v5.items()})
out['V5_ring_closed_forms'] = v5

# ---------------------------------------------------------------- V6 annulus transforms
kk = np.linspace(0.1, 60.0, 50)
s1, s2 = np.array([0.3, 0.0]), np.array([0.9, 0.2])
Q1 = g.ring_transforms(kk, s1, s2, 'Q1')
gq, gw = np.polynomial.legendre.leggauss(200)
num = np.zeros_like(Q1)
for i in range(2):
    s = 0.5 * (s2[i] - s1[i]) * gq + 0.5 * (s1[i] + s2[i])
    num[i] = (special.j1(np.outer(kk, s)) * s * 0.5 * (s2[i] - s1[i]) * gw).sum(axis=1)
out['V6_annulus_Q1_struve_vs_quad'] = float(np.max(np.abs(Q1 - num)) / np.max(np.abs(num)))
print('V6 Q1 Struve vs quadrature:', out['V6_annulus_Q1_struve_vs_quad'])

# ---------------------------------------------------------------- V7 forward check on the layer
v7 = {}
for nu in [0.45, 0.499]:
    L = g.Layer(nu=nu)
    G = g.GreenTables(L)
    ax_, ay_ = 5.1e-3, 71e-3 / 14
    obs = [(0.0, 0.0), (3.0e-3, 2.0e-3), (5.1e-3, 0.0), (7.0e-3, 0.0), (0.0, 12.0e-3), (15.3e-3, 0.0)]
    # (a) split method, element subdivided into 40x40 sub-rectangles for the correction quadrature
    nsub = 40
    xs = np.linspace(-ax_, ax_, nsub + 1)
    ys = np.linspace(-ay_, ay_, nsub + 1)
    X1, Y1 = np.meshgrid(xs[:-1], ys[:-1], indexing='ij')
    X2, Y2 = np.meshgrid(xs[1:], ys[1:], indexing='ij')
    res = {}
    for (x0, y0) in obs:
        hs = g.rect_halfspace(L, np.array(x0), np.array(y0), -ax_, ax_, -ay_, ay_)['zz']
        co = g.rect_correction(G, x0, y0, X1, X2, Y1, Y2, order=3, comps=('zz',))['zz'].sum()
        a_val = hs + co
        # (b) direct 2-D Fourier quadrature of the FULL kernel on |k| < Kc + analytic half-space tail.
        #     u(x0) = 1/(4pi^2) int C_zz(k) Phat(k) e^{ik.x0} d^2k ; Phat = 4 sin(kx ax) sin(ky ay)/(kx ky)
        #     polar coords; tail |k| > Kc uses C_zz -> C_hs (correction < 1e-30 there) and the
        #     half-space tail is obtained as  hs_total - hs(|k|<Kc).
        Kc = 30.0 / L.t
        nr, nth = 6000, 512
        kr, wr = np.polynomial.legendre.leggauss(nr)
        kr = 0.5 * Kc * (kr + 1); wr = 0.5 * Kc * wr
        th = (np.arange(nth) + 0.5) * (np.pi / 2) / nth       # quarter, symmetric integrand
        wth = (np.pi / 2) / nth
        Kx = np.outer(kr, np.cos(th)); Ky = np.outer(kr, np.sin(th))
        with np.errstate(invalid='ignore', divide='ignore'):
            Ph = 4 * np.sin(Kx * ax_) * np.sin(Ky * ay_) / (Kx * Ky)
        cosx = np.cos(Kx * x0) * np.cos(Ky * y0)      # even part (integrand symmetric in quadrants)
        Czz = g.compliance(kr, L)['Czz']
        Chs = (1 - nu) / (L.mu * kr)
        integ_full = (Czz[:, None] * Ph * cosx).sum(axis=1) * wth * 4
        integ_hs = (Chs[:, None] * Ph * cosx).sum(axis=1) * wth * 4
        full_in = (integ_full * kr * wr).sum() / (4 * np.pi ** 2)
        hs_in = (integ_hs * kr * wr).sum() / (4 * np.pi ** 2)
        b_val = full_in + (hs - hs_in)
        res[f'({x0*1e3:.1f},{y0*1e3:.1f})mm'] = dict(split=a_val, fourier=b_val, rel=float(abs(a_val - b_val) / abs(a_val)))
    v7[str(nu)] = res
    print('V7', nu, {k_: f"{v['rel']:.1e}" for k_, v in res.items()})
out['V7_forward_square_patch'] = v7

json.dump(out, open('validate.json', 'w'), indent=1, default=float)

# ---------------------------------------------------------------- kernel figure
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
u = np.logspace(-2, 1.5, 400)
for nu in NUS:
    P = g.kernel_parts(u, nu)
    Lr = g.Layer(nu=nu)
    axs[0].loglog(u, P['Lzz'], label=f'Czz mu k/(1-nu), nu={nu}')
    axs[1].loglog(u, P['LL'], label=f'C_L, nu={nu}')
    axs[2].semilogx(u, P['Lc'] / (1 - 2 * nu), label=f'C_c / C_c(half-space), nu={nu}')
    axs[0].loglog(u, u * (1 - 2 * nu) / (2 * (1 - nu) ** 2), ':', lw=0.8)
axs[1].loglog(u, np.tanh(u), 'k--', label='C_T mu k = tanh(kt)')
axs[1].loglog(u, u / (1 - 0.45), 'k:', lw=0.8, label='t/mu (thin, nu=0.45)')
for a_ in axs[:2]:
    a_.axhline(1, color='k', lw=0.6)
    a_.set_xlabel('k t'); a_.legend(fontsize=7)
axs[0].set_title('normal kernel (=1: half-space; dotted: t/M)')
axs[1].set_title('tangential kernels')
axs[2].set_xlabel('k t'); axs[2].axhline(0, color='k', lw=0.5); axs[2].legend(fontsize=7)
axs[2].set_title('normal-tangential coupling (sign: + = Boussinesq-like inward)')
fig.tight_layout(); fig.savefig('fig_kernels.png', dpi=130); plt.close(fig)
print('done')
