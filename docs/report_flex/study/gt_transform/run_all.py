"""Run every ground-truth case (C4, C5, C6 + closed forms + literature checks) for nu in {0.45, 0.49, 0.499}.
Writes results.json, runs/*.pkl (raw), PNG figures (made by plots.py).
Usage:  OMP_NUM_THREADS=8 python run_all.py [--workers 6]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gt_transform as gt          # noqa: E402
import gt_contact as gc            # noqa: E402

NUS = [0.45, 0.49, 0.499]
T = 1.5e-3
RUNS = os.path.join(HERE, 'runs')
os.makedirs(RUNS, exist_ok=True)
C6_N_FRIC = [12, 24, 48, 96]
C6_N_NOSLIP = [12, 24, 48, 96]


def richardson(vals, ns):
    """Observed order and extrapolated value from the last three refinements (factor-2 meshes)."""
    v = np.asarray(vals, float)
    if v.size < 3:
        return dict(extrapolated=float(v[-1]), order=None, rel_err_est=None)
    d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
    rho = d1 / d2 if d2 != 0 else np.inf
    p = np.log2(abs(rho)) if rho > 0 else None
    if p is None or not np.isfinite(p):
        return dict(extrapolated=float(v[-1]), order=None, rel_err_est=float(abs(d2 / v[-1])))
    ext = v[-1] + d2 / (2 ** p - 1)
    return dict(extrapolated=float(ext), order=float(p), rel_err_est=float(abs(ext - v[-1]) / abs(ext)))


# ======================================================================== axisymmetric (C4, C5, literature)
def task_axi(nu):
    L = gt.Layer(nu=nu, t=T)
    out = dict(nu=nu)
    one = lambda x: np.ones_like(x)
    zero = lambda x: np.zeros_like(x)
    M, Es = L.M, L.Estar
    # ---------------- C4 flat circular punch
    c4 = {}
    for a in (5e-3, 2.5e-3):
        d = {}
        Ks = {N: gc.flat_punch_fredholm(L, a, N)[2] for N in (24, 48, 96)}
        fr, g, K = gc.flat_punch_fredholm(L, a, 96)
        d['frictionless'] = dict(K_N_per_um=K * 1e-6, method='Fredholm 2nd kind, Gauss-Legendre Nystrom N=96',
                                 rel_err_est=abs(Ks[48] - Ks[96]) / Ks[96] + 1e-13,
                                 K_by_N={str(n): v * 1e-6 for n, v in Ks.items()})
        # ring-BEM cross-check (independent discretisation)
        ns = [100, 200, 400, 800]
        Kr = [gc.ring_flat_punch(L, a, n)[0] for n in ns]
        rr = richardson(Kr, ns)
        d['frictionless']['ring_BEM_check'] = dict(n=ns, K_N_per_um=[k * 1e-6 for k in Kr],
                                                   extrapolated_N_per_um=rr['extrapolated'] * 1e-6,
                                                   order=rr['order'],
                                                   rel_diff_extrap_vs_fredholm=abs(rr['extrapolated'] - K) / K)
        Kb = []
        sols = []
        for n in ns:
            kb, s = gc.ring_flat_punch(L, a, n, bonded=True)
            Kb.append(kb)
            sols.append(s)
        rb = richardson(Kb, ns)
        d['bonded_noslip'] = dict(K_N_per_um=rb['extrapolated'] * 1e-6, method='ring BEM (p, q_r piecewise const.), '
                                  'sine-graded, Richardson over n=100..800', n=ns, K_by_n=[k * 1e-6 for k in Kb],
                                  order=rb['order'], rel_err_est=rb['rel_err_est'],
                                  ratio_to_frictionless=rb['extrapolated'] / K)
        # profiles (frictionless, per unit indentation)
        r_in = np.linspace(0, a * 0.995, 60)
        p_r = fr.pressure(g, r_in, one, zero)
        r_out = np.concatenate([np.linspace(0, a, 21), np.linspace(a * 1.01, 4 * a, 80)])
        w = fr.surface_w(g, r_out, one)
        d['profile_frictionless_per_unit_delta'] = dict(r_m=r_out.tolist(), uz_over_delta=w.tolist(),
                                                        r_in_m=r_in.tolist(), p_Pa_per_m=p_r.tolist(),
                                                        check_max_abs_uz_err_under_punch=float(np.max(np.abs(w[:21] - 1))))
        d['halfspace_sneddon_K_N_per_um'] = gt.sneddon_flat_punch_K(L, a) * 1e-6
        d['winkler_comparison_K_N_per_um(k=M/t, labelled comparison only)'] = M * np.pi * a * a / T * 1e-6
        d['kappa_c=K(1-nu^2)/(2Ea)'] = K / (2 * Es * a)
        c4[f'a={a*1e3:g}mm'] = d
    out['C4'] = c4
    out['C5'] = task_c5(nu)
    return out


def task_c5(nu):
    """C5 sphere: for each target delta solve delta(a) = target for the contact radius a (brentq on the
    Fredholm solution with g(a) = 0), then F.  Error estimate: N = 24 vs N = 48 at the same a, plus |delta(a_root) - target|."""
    from scipy.optimize import brentq
    L = gt.Layer(nu=nu, t=T)
    M = L.M
    c5 = {}
    dgrid = np.arange(1, 31) * 1e-6
    for R in (10e-3, 25e-3):
        a_at, F_at, errF = [], [], []
        for dt in dgrid:
            ah = np.sqrt(R * dt)
            f = lambda a: gc.sphere_fredholm(L, R, a, 24)[0] - dt     # N=24 is converged to ~1e-15 (checked)
            a_root = brentq(f, 0.3 * ah, 2.0 * ah, xtol=1e-13 * ah, rtol=1e-12)
            dd, FF, *_ = gc.sphere_fredholm(L, R, a_root, 48)
            dd2, FF2, *_ = gc.sphere_fredholm(L, R, a_root, 24)
            a_at.append(a_root)
            F_at.append(FF)
            errF.append(abs(FF2 - FF) / FF + abs(dd - dt) / dt)
        a_at, F_at = np.array(a_at), np.array(F_at)
        Fh, ah = gt.hertz(L, R, dgrid)
        # pressure positivity and profile at 30 um
        a30 = float(a_at[-1])
        d30, F30, g1, g2, fr = gc.sphere_fredholm(L, R, a30, 48)
        gg = d30 * g1 - g2
        rhs = lambda x: d30 - x ** 2 / R
        rhsd = lambda x: -2 * x / R
        r_in = np.linspace(0, a30 * 0.999, 50)
        pr = fr.pressure(gg, r_in, rhs, rhsd)
        r_out = np.concatenate([np.linspace(0, a30, 21), np.linspace(a30 * 1.01, 8e-3, 80)])
        wz = fr.surface_w(gg, r_out, rhs)
        Fr, ar, _ = gc.ring_sphere(L, R, d30, 1.3 * a30, 600)
        c5[f'R={R*1e3:g}mm'] = dict(
            delta_um=(dgrid * 1e6).tolist(), F_N=F_at.tolist(), contact_radius_mm=(a_at * 1e3).tolist(),
            method='Fredholm 2nd kind (Nystrom N=48), contact radius from g(a)=0 by root finding at each delta',
            rel_err_est_F=float(max(errF)), rel_err_est_a=1e-10,
            hertz_F_N=Fh.tolist(), hertz_a_mm=(ah * 1e3).tolist(),
            kappa_s=(F_at / Fh).tolist(),
            winkler_comparison_F_N=(M / T * np.pi * R * dgrid ** 2).tolist(),
            winkler_note='Winkler k=M/t: F = k pi R delta^2 (labelled comparison only)',
            at_30um=dict(delta_um=d30 * 1e6, a_mm=a30 * 1e3, F_N=F30, min_pressure_Pa=float(pr.min()), p0_Pa=float(pr[0]),
                         r_m=r_out.tolist(), uz_m=wz.tolist(), r_in_m=r_in.tolist(), p_Pa=pr.tolist(),
                         check_max_rel_uz_err_inside=float(np.max(np.abs(wz[:21] - (d30 - r_out[:21] ** 2 / (2 * R)))) / d30),
                         ring_BEM_active_set_check=dict(n=600, F_N=Fr, a_mm=ar * 1e3, rel_diff_F=abs(Fr - F30) / F30,
                                                        note='ring width 1.3a/600 limits the contact-radius resolution')))
    return c5
    return out


def task_literature():
    """kappa_s(alpha) (sphere) vs Argatov et al. arXiv:1203.0918 Table 1 (based on Hayes et al. 1972);
    kappa_c(alpha) (flat punch) vs their asymptotic model (7)."""
    tab = json.load(open(os.path.join(HERE, 'lit', 'argatov_table1.json')))
    alphas = np.array(tab['alpha'])
    pub = np.array(tab['kappa_s'])
    mine = np.zeros_like(pub)
    R = 10e-3
    for j, nu in enumerate(tab['nu']):
        L = gt.Layer(nu=nu, t=T)
        for i, al in enumerate(alphas):
            a = al * T
            dd, FF, *_ = gc.sphere_fredholm(L, R, a, 64)
            mine[i, j] = FF / gt.hertz(L, R, dd)[0]
    # kappa_c and asymptotic model (7): a_m = (-1)^m/(2^{2m} (m!)^2) int_0^oo [1 - L(l)] l^{2m} dl
    kc = {}
    for nu in [0.3, 0.4, 0.45, 0.49, 0.499, 0.5]:
        L = gt.Layer(nu=nu, t=T)
        u, w = gt.gl_nodes(40.0, 0.05, 12)
        dzz = gt.kernel_parts(u, nu)['dzz']             # = L(u) - 1
        a0 = np.sum(w * (-dzz))
        a1 = -0.25 * np.sum(w * (-dzz) * u ** 2)
        als = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 3.333333, 1.666667, 4.0, 8.0])
        exact = np.array([gc.flat_punch_fredholm(L, al * T, 96)[2] / (2 * L.Estar * al * T) for al in als])
        q = 2 * a0 / np.pi
        asym = 1 + als * q + als ** 2 * q ** 2 + als ** 3 * (q ** 3 + 8 * a1 / (3 * np.pi)) + \
            als ** 4 * (q ** 4 + 32 * a0 * a1 / (3 * np.pi ** 2))
        kc[str(nu)] = dict(alpha=als.tolist(), kappa_c_exact=exact.tolist(), kappa_c_asymptotic_eq7=asym.tolist(),
                           a0=a0, a1=a1)
    # bonded (no-slip) flat punch on a half-space: Mossakovskii/Spence ratio (1-nu) ln(3-4nu)/(1-2nu)
    moss = {}
    for nu in (0.0, 0.3, 0.45, 0.49):
        L = gt.Layer(nu=nu, t=100.0)
        r = gc.ring_flat_punch(L, 5e-3, 800, bonded=True)[0] / gc.ring_flat_punch(L, 5e-3, 800)[0]
        moss[str(nu)] = dict(ring_BEM_ratio_n800=r, closed_form=(1 - nu) * np.log(3 - 4 * nu) / (1 - 2 * nu))
    return dict(mossakovskii_bonded_halfspace=moss, kappa_s=dict(source=tab['source'], nu=tab['nu'], alpha=alphas.tolist(), published=pub.tolist(),
                             this_work=mine.tolist(),
                             max_abs_diff=float(np.max(np.abs(mine - pub))),
                             max_rel_diff=float(np.max(np.abs(mine - pub) / pub))),
                kappa_c=kc)


# ======================================================================== C6 square taxel punch
def task_c6(args):
    nu, noslip, n, post = args
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    L = gt.Layer(nu=nu, t=T)
    t0 = time.time()
    G = gt.GreenTables(L) if n is not None else None
    sol = gc.square_punch(L, G, n if n is not None else 48, noslip)
    sol['time_solve_s'] = time.time() - t0
    sol['nu'] = nu
    if post:
        maps, tot = gc.taxel_maps_spectral(L, sol, Ng=2048)
        maps1, _ = gc.taxel_maps_spectral(L, sol, Ng=1024)
        sol['maps'] = maps
        sol['maps_Ng1024'] = maps1
        sol['spectral_totals'] = tot
        x = np.linspace(0, 25.5e-3, 256)
        y = np.linspace(0, 35.5e-3, 256)
        ux, uy, uz = gc.surface_field_realspace(L, G, sol, x, np.zeros_like(x))
        vx, vy, vz = gc.surface_field_realspace(L, G, sol, np.zeros_like(y), y)
        sol['profile_x'] = dict(x=x, ux=ux, uy=uy, uz=uz)
        sol['profile_y'] = dict(y=y, ux=vx, uy=vy, uz=vz)
        # 2-D surface map for the figure (quarter, coarse)
        xg = np.linspace(0, 20e-3, 81)
        yg = np.linspace(0, 25e-3, 101)
        XG, YG = np.meshgrid(xg, yg, indexing='ij')
        _, _, uzg = gc.surface_field_realspace(L, G, sol, XG.ravel(), YG.ravel())
        sol['map2d'] = dict(x=xg, y=yg, uz=uzg.reshape(XG.shape))
    sol['time_total_s'] = time.time() - t0
    tag = f"c6_nu{nu}_{'noslip' if noslip else 'fric'}_n{n if n is not None else 'HS48'}"
    pickle.dump(sol, open(os.path.join(RUNS, tag + '.pkl'), 'wb'))
    return tag, sol['K'], sol['time_total_s']


def main_c5():
    """Recompute only C5 (after the session-2 fix) and update runs/axi_lit.pkl."""
    res = pickle.load(open(os.path.join(RUNS, 'axi_lit.pkl'), 'rb'))
    ctx = get_context('spawn')
    with ctx.Pool(len(NUS)) as pool:
        for nu, c5 in zip(NUS, pool.map(task_c5, NUS)):
            res[f'nu={nu}']['C5'] = c5
    pickle.dump(res, open(os.path.join(RUNS, 'axi_lit.pkl'), 'wb'))
    print('C5 updated')


def main():
    if '--c5' in sys.argv:
        return main_c5()
    workers = 6
    if '--workers' in sys.argv:
        workers = int(sys.argv[sys.argv.index('--workers') + 1])
    t0 = time.time()
    results = dict(meta=dict(date='2026-10-30', E_Pa=1e6, t_m=T, units='SI; stiffness in N/um',
                             conventions='origin at the centre of the free face; z into the layer; u_z>0 = indentation',
                             layer='infinite layer of thickness t perfectly bonded to a rigid base (exact Green function)'))
    jobs_c6 = []
    for nu in NUS:
        for n in C6_N_FRIC:
            jobs_c6.append((nu, False, n, n in (48, 96)))
        for n in C6_N_NOSLIP:
            jobs_c6.append((nu, True, n, n in (48, 96)))
        jobs_c6.append((nu, False, None, False))     # half-space square punch (reference limit)
    # biggest first
    jobs_c6.sort(key=lambda j: -((j[2] or 48) ** 2 * (3 if j[1] else 1)))
    ctx = get_context('spawn')
    with ctx.Pool(workers) as pool:
        axi_async = [pool.apply_async(task_axi, (nu,)) for nu in NUS]
        lit_async = pool.apply_async(task_literature)
        c6_res = pool.map_async(task_c6, jobs_c6, chunksize=1)
        for nu, ar in zip(NUS, axi_async):
            results[f'nu={nu}'] = ar.get()
            print(f'axi nu={nu} done {time.time()-t0:.0f}s', flush=True)
        results['literature'] = lit_async.get()
        print(f'literature done {time.time()-t0:.0f}s', flush=True)
        for tag, K, tt in c6_res.get():
            print(f'{tag}: K={K*1e-6:.6f} N/um ({tt:.0f}s)', flush=True)
    pickle.dump(results, open(os.path.join(RUNS, 'axi_lit.pkl'), 'wb'))
    print(f'all jobs done {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
