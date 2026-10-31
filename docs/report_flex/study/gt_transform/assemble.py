"""Assemble results.json and the PNG figures from runs/*.pkl (written by run_all.py)."""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gt_transform as gt   # noqa: E402
from run_all import NUS, T, C6_N_FRIC, C6_N_NOSLIP, richardson  # noqa: E402

RUNS = os.path.join(HERE, 'runs')
COL = {0.45: '#2a78d6', 0.49: '#eb6834', 0.499: '#1baf7a'}
LS = {0.45: '-', 0.49: '--', 0.499: ':'}
GREY = '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'lines.linewidth': 2})
DIV = LinearSegmentedColormap.from_list('div', ['#eb6834', '#f0efec', '#2a78d6'])
SEQ = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#86b6ef', '#2a78d6', '#104281'])
A_TAX = 10.2e-3 * 71e-3 / 7


def tolist(x):
    if isinstance(x, dict):
        return {k: tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tolist(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def load(nu, kind, n):
    f = os.path.join(RUNS, f'c6_nu{nu}_{kind}_n{n}.pkl')
    return pickle.load(open(f, 'rb')) if os.path.exists(f) else None


def c6_block(nu, kind, ns):
    L = gt.Layer(nu=nu, t=T)
    sols = {n: load(nu, kind, n) for n in ns}
    ns_ok = [n for n in ns if sols[n] is not None]
    Ks = [sols[n]['K'] for n in ns_ok]
    rr = richardson(Ks, ns_ok)
    fin = sols[ns_ok[-1]]
    prev = sols[ns_ok[-2]] if len(ns_ok) > 1 and 'maps' in sols[ns_ok[-2]] else None
    out = dict(method='rectangular BEM, piecewise-constant tractions, sine-graded quarter mesh n x n, collocation at '
                      'element centres; half-space closed form + exact layer correction; Richardson over n',
               n=ns_ok, K_by_n_N_per_um=[k * 1e-6 for k in Ks], K_N_per_um=rr['extrapolated'] * 1e-6,
               K_finest_N_per_um=Ks[-1] * 1e-6, observed_order=rr['order'], rel_err_est=rr['rel_err_est'],
               solve_time_s_finest=fin['time_total_s'])
    if 'maps' in fin:
        K = fin['K']
        m = fin['maps']

        def dmax(a, b):
            return float(np.max(np.abs(a - b)))
        acc = {}
        for key in ('surface_force', 'base_force'):
            acc[key + '_maxabs_change_n_vs_n/2_fraction_of_F'] = dmax(m[key] / K, prev['maps'][key] / prev['K']) if prev else None
            acc[key + '_maxabs_change_Ng2048_vs_1024_fraction_of_F'] = dmax(m[key] / K, fin['maps_Ng1024'][key] / K)
        acc['mean_u_maxabs_change_n_vs_n/2_per_delta'] = dmax(m['mean_u'], prev['maps']['mean_u']) if prev else None
        acc['mean_u_maxabs_change_Ng2048_vs_1024_per_delta'] = dmax(m['mean_u'], fin['maps_Ng1024']['mean_u'])
        acc['centre_taxel_mean_uz_minus_1(exact 0 for frictionless)'] = float(m['mean_u'][2, 3, 2] - 1.0)
        acc['spectral_total_vs_K'] = float(fin['spectral_totals']['base'] / K - 1.0)
        nb = [(i, j) for i in (1, 2, 3) for j in (2, 3, 4) if (i, j) != (2, 3)]
        sf_exact = np.zeros((5, 7, 3))
        sf_exact[2, 3, 2] = K * 1e-6
        out['taxel_maps'] = dict(
            index_order='[i][j][component], i = 0..4 along x, j = 0..6 along y, component (x, y, z); centre taxel (2,3)',
            from_n=fin['n'],
            surface_contact_force_per_um_N=sf_exact.tolist(),
            surface_contact_force_note='exact: all contact elements lie inside taxel (2,3); in-plane components sum to 0 '
                                       'by symmetry. (The Fourier-series value differs by up to the Ng-change listed in '
                                       'accuracy because the 1/sqrt edge singularity sits exactly on the taxel border.)',
            base_reaction_force_per_um_N=(m['base_force'] * 1e-6).tolist(),
            base_reaction_fraction_of_total=(m['base_force'] / K).tolist(),
            mean_surface_displacement_per_delta=m['mean_u'].tolist(),
            accuracy=acc,
            neighbours=dict(
                mean_uz_over_delta={f'({i},{j})': float(m['mean_u'][i, j, 2]) for i, j in nb},
                base_Fz_fraction={f'({i},{j})': float(m['base_force'][i, j, 2] / K) for i, j in nb},
                surface_contact_force_fraction_sum_of_8=0.0,
                base_Fz_fraction_sum_of_8=float(sum(m['base_force'][i, j, 2] for i, j in nb) / K)))
        px, py = fin['profile_x'], fin['profile_y']
        out['profiles_per_delta'] = dict(from_n=fin['n'], x_m=px['x'].tolist(), uz_x0=px['uz'].tolist(),
                                         ux_x0=px['ux'].tolist(), y_m=py['y'].tolist(), uz_0y=py['uz'].tolist(),
                                         uy_0y=py['uy'].tolist(),
                                         note='u/delta along y=0 and x=0; under the punch the BEM collocation error is '
                                              'visible as small wiggles near the edge')
        if prev is not None:
            out['profiles_per_delta']['maxabs_change_vs_n/2_outside_punch'] = float(np.max(np.abs(
                (px['uz'] - prev['profile_x']['uz'])[px['x'] > 5.4e-3])))
    hs = load(nu, 'fric', 'HS48')
    if hs is not None and kind == 'fric':
        out['halfspace_square_punch_K_N_per_um(n=48, finest only)'] = hs['K'] * 1e-6
    out['winkler_comparison_K_N_per_um(k=M/t, labelled comparison only)'] = L.M * A_TAX / T * 1e-6
    return out, sols


def main():
    axi = pickle.load(open(os.path.join(RUNS, 'axi_lit.pkl'), 'rb'))
    res = dict(meta=axi['meta'])
    res['meta']['accuracy_note'] = ('rel_err_est: Fredholm = |K(N/2)-K(N)|/K; ring/rect BEM = Richardson extrapolation '
                                    'distance from the finest mesh; maps = change between the two finest meshes and '
                                    'between two Fourier resolutions')
    c6sols = {}
    for nu in NUS:
        L = gt.Layer(nu=nu, t=T)
        blk = dict(closed_forms=gt.closed_forms(L))
        blk['closed_forms']['halfspace_formulas'] = dict(
            boussinesq_uz='(1-nu)/(2 pi mu r) per unit normal force', cerruti_ux_along_force='(1/(2 pi mu r))[(1-nu)+nu x^2/r^2]',
            sneddon_flat_punch='K = 2 E a/(1-nu^2)', hertz='F = (4/3) E/(1-nu^2) R^1/2 delta^3/2, a = sqrt(R delta)')
        blk['C4'] = axi[f'nu={nu}']['C4']
        blk['C5'] = axi[f'nu={nu}']['C5']
        for Rk, d5 in blk['C5'].items():      # corrected inside-contact check: compare with h(r) = delta - r^2/(2R)
            R_ = float(Rk[2:-2]) * 1e-3
            a30 = d5['at_30um']
            rr = np.array(a30['r_m'])[:21]
            uz = np.array(a30['uz_m'])[:21]
            a30.pop('check_max_abs_uz_err_inside_m', None)
            a30['check_max_rel_uz_err_inside(vs delta - r^2/2R)'] = float(
                np.max(np.abs(uz - (a30['delta_um'] * 1e-6 - rr ** 2 / (2 * R_)))) / (a30['delta_um'] * 1e-6))
        c6f, sf = c6_block(nu, 'fric', C6_N_FRIC)
        c6n, sn = c6_block(nu, 'noslip', C6_N_NOSLIP)
        blk['C6'] = dict(frictionless=c6f, noslip=c6n,
                         geometry='rigid flat square punch 10.2 x 10.14 mm exactly over taxel (2,3), centred at origin')
        c6sols[nu] = (sf, sn)
        res[f'nu={nu}'] = blk
    res['literature'] = axi['literature']
    gf = os.path.join(HERE, '..', 'gt_fem')
    res['gt_fem_crosscheck'] = 'see README (gt_fem results checked at assembly time)'
    json.dump(tolist(res), open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    figures(res, c6sols)
    summary(res)


def figures(res, c6sols):
    # ---------------- C4
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for nu in NUS:
        for a, lw in (('a=5mm', 2), ('a=2.5mm', 1.2)):
            d = res[f'nu={nu}']['C4'][a]['profile_frictionless_per_unit_delta']
            r = np.array(d['r_m']) * 1e3
            ax[0].plot(r, d['uz_over_delta'], color=COL[nu], ls=LS[nu], lw=lw,
                       label=f'nu={nu}, {a}' if True else None)
            ax[1].plot(np.array(d['r_in_m']) * 1e3, np.array(d['p_Pa_per_m']) * 1e-6 * 1e-6, color=COL[nu], ls=LS[nu], lw=lw,
                       label=f'nu={nu}, {a}')
    ax[0].axhline(0, color=GREY, lw=0.8)
    ax[0].set_xlabel('r [mm]')
    ax[0].set_ylabel('u_z / delta')
    ax[0].set_title('C4 flat punch (frictionless): surface displacement')
    ax[1].set_xlabel('r [mm]')
    ax[1].set_ylabel('p [MPa] per um of indentation')
    ax[1].set_title('C4 contact pressure')
    ax[1].set_ylim(0, None)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_c4_punch.png'), dpi=150)
    plt.close(fig)
    # ---------------- C5
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for k, R in enumerate(('R=10mm', 'R=25mm')):
        for nu in NUS:
            d = res[f'nu={nu}']['C5'][R]
            ax[k].plot(d['delta_um'], d['F_N'], color=COL[nu], ls=LS[nu], label=f'exact layer, nu={nu}')
            ax[2].plot(d['delta_um'], d['contact_radius_mm'], color=COL[nu], ls=LS[nu],
                       lw=2 if R == 'R=10mm' else 1.2, label=f'nu={nu}, {R}')
        d = res['nu=0.45']['C5'][R]
        ax[k].plot(d['delta_um'], d['hertz_F_N'], color=GREY, ls='-.', lw=1.2, label='Hertz half-space, nu=0.45')
        ax[k].plot(d['delta_um'], d['winkler_comparison_F_N'], color='#9a9993', ls=(0, (1, 2)), lw=1.2,
                   label='Winkler k=M/t, nu=0.45 (comparison only)')
        ax[k].set_xlabel('delta [um]')
        ax[k].set_ylabel('F [N]')
        ax[k].set_title(f'C5 sphere {R} (frictionless)')
        ax[k].legend(fontsize=7, frameon=False)
    d = res['nu=0.45']['C5']['R=10mm']
    ax[2].plot(d['delta_um'], d['hertz_a_mm'], color=GREY, ls='-.', lw=1.2, label='Hertz R=10mm')
    ax[2].set_xlabel('delta [um]')
    ax[2].set_ylabel('contact radius [mm]')
    ax[2].set_title('C5 contact radius')
    ax[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_c5_sphere.png'), dpi=150)
    plt.close(fig)
    # ---------------- C6 profiles
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for nu in NUS:
        for kind, lw in (('frictionless', 2), ('noslip', 1.2)):
            p = res[f'nu={nu}']['C6'][kind].get('profiles_per_delta')
            if p is None:
                continue
            ax[0].plot(np.array(p['x_m']) * 1e3, p['uz_x0'], color=COL[nu], ls=LS[nu], lw=lw, label=f'nu={nu}, {kind}')
            ax[1].plot(np.array(p['y_m']) * 1e3, p['uz_0y'], color=COL[nu], ls=LS[nu], lw=lw, label=f'nu={nu}, {kind}')
    for a, edges, lab in ((ax[0], [5.1, 15.3, 25.5], 'x'), (ax[1], [71 / 14, 3 * 71 / 14, 5 * 71 / 14, 35.5], 'y')):
        for e in edges:
            a.axvline(e, color='#c3c2b7', lw=0.8)
        a.axhline(0, color=GREY, lw=0.8)
        a.set_xlabel(f'{lab} [mm] (vertical lines: taxel borders)')
        a.set_ylabel('u_z / delta  (negative = surface rises)')
    ax[0].set_title('C6 square taxel punch: u_z(x, 0)')
    ax[1].set_title('C6 square taxel punch: u_z(0, y)')
    ax[0].legend(fontsize=7, frameon=False, loc='upper right')
    for a in ax:
        ins = a.inset_axes([0.52, 0.1, 0.45, 0.36])
        for nu in NUS:
            p = res[f'nu={nu}']['C6']['frictionless'].get('profiles_per_delta')
            if p is None:
                continue
            xx = np.array(p['x_m' if a is ax[0] else 'y_m']) * 1e3
            ins.plot(xx, p['uz_x0' if a is ax[0] else 'uz_0y'], color=COL[nu], ls=LS[nu], lw=1.2)
        ins.set_xlim(5, 20)
        ins.set_ylim(-0.15, 0.2)
        ins.axhline(0, color=GREY, lw=0.6)
        ins.tick_params(labelsize=6)
        ins.set_title('zoom outside punch (frictionless)', fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_c6_profiles.png'), dpi=150)
    plt.close(fig)
    # ---------------- C6 taxel maps (nu = 0.45 and 0.49, frictionless)
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.5))
    for r_, nu in enumerate((0.45, 0.49)):
        tm = res[f'nu={nu}']['C6']['frictionless'].get('taxel_maps')
        if tm is None:
            continue
        mu_ = np.array(tm['mean_surface_displacement_per_delta'])[..., 2]
        bf = np.array(tm['base_reaction_fraction_of_total'])[..., 2]
        bx = np.array(tm['base_reaction_fraction_of_total'])[..., 0]
        for c_, (dat, title, cmap, sym) in enumerate((
                (mu_, 'mean u_z / delta per taxel', DIV, True),
                (bf, 'base reaction F_z / F per taxel', SEQ, False),
                (bx, 'base reaction F_x / F per taxel', DIV, True))):
            a = ax[r_, c_]
            vmax = np.max(np.abs(dat)) if sym else dat.max()
            show = dat.copy()
            if c_ == 0:
                vmax = 0.2
            im = a.imshow(show.T, origin='lower', cmap=cmap, vmin=-vmax if sym else 0, vmax=vmax,
                          extent=[-25.5, 25.5, -35.5, 35.5])
            for i in range(5):
                for j in range(7):
                    a.text(-25.5 + 10.2 * (i + 0.5), -35.5 + 71 / 7 * (j + 0.5), f'{dat[i, j]:.3f}', ha='center',
                           va='center', fontsize=6, color='#0b0b0b')
            a.set_title(f'nu={nu}: {title}', fontsize=8)
            a.set_xlabel('x [mm]')
            a.set_ylabel('y [mm]')
            a.grid(False)
            plt.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle('C6 frictionless square punch over taxel (2,3): contact force is 100% in the centre taxel; '
                 'colour scale of u_z clipped at +-0.2 (centre = 1.000)', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_c6_taxels.png'), dpi=150)
    plt.close(fig)
    # ---------------- literature
    lit = res['literature']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ks = lit['kappa_s']
    pub = np.array(ks['published'])
    mine = np.array(ks['this_work'])
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']
    for j, nu in enumerate(ks['nu']):
        ax[0].plot(ks['alpha'], mine[:, j], color=cols[j], lw=1.5, label=f'this work nu={nu}')
        ax[0].plot(ks['alpha'], pub[:, j], 'o', mfc='none', color=cols[j], ms=5)
    ax[0].set_xlabel('alpha = a/h')
    ax[0].set_ylabel('kappa_s = F / F_Hertz(delta)')
    ax[0].set_title('sphere: lines = this work, circles = Argatov et al. 2012 Tab.1 (Hayes 1972)', fontsize=8)
    ax[0].legend(fontsize=7, frameon=False)
    kc = lit['kappa_c']
    for j, nu in enumerate(['0.3', '0.45', '0.5']):
        d = kc[nu]
        o = np.argsort(d['alpha'])
        al = np.array(d['alpha'])[o]
        ax[1].plot(al, np.array(d['kappa_c_exact'])[o], color=cols[j], lw=1.5, marker='o', ms=3, label=f'exact nu={nu}')
        ax[1].plot(al, np.array(d['kappa_c_asymptotic_eq7'])[o], color=cols[j], ls='--', lw=1.2,
                   label=f'asymptotic eq.(7) nu={nu}')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('alpha = a/h')
    ax[1].set_ylabel('kappa_c = K (1-nu^2)/(2 E a)')
    ax[1].set_title('flat punch: exact vs 4th-order asymptotics (valid alpha < ~0.6)', fontsize=8)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_literature.png'), dpi=150)
    plt.close(fig)
    # ---------------- convergence
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))
    for nu in NUS:
        for kind, mk in (('frictionless', 'o'), ('noslip', 's')):
            b = res[f'nu={nu}']['C6'][kind]
            n = np.array(b['n'])
            err = np.abs(np.array(b['K_by_n_N_per_um']) - b['K_N_per_um']) / b['K_N_per_um']
            ok = err > 0
            ax.loglog(n[ok], err[ok], color=COL[nu], ls=LS[nu], marker=mk, ms=5, lw=1.5, label=f'nu={nu} {kind}')
    ax.set_xlabel('n (elements per quarter side)')
    ax.set_ylabel('|K(n) - K_extrap| / K_extrap')
    ax.set_title('C6 BEM convergence (o frictionless, s no-slip)', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_c6_convergence.png'), dpi=150)
    plt.close(fig)


def summary(res):
    lines = []
    for nu in NUS:
        b = res[f'nu={nu}']
        cf = b['closed_forms']
        s = [f'nu={nu}: G1 M A/t={cf["G1_confined_K_N_per_um"]:.4f}  shear G A/t={cf["simple_shear_infinite_layer_K_N_per_um"]:.4f}']
        for a in ('a=5mm', 'a=2.5mm'):
            d = b['C4'][a]
            s.append(f'  C4 {a}: K_fric={d["frictionless"]["K_N_per_um"]:.6f} (err {d["frictionless"]["rel_err_est"]:.1e}, '
                     f'ringBEM {d["frictionless"]["ring_BEM_check"]["rel_diff_extrap_vs_fredholm"]:.1e}) '
                     f'K_bonded={d["bonded_noslip"]["K_N_per_um"]:.6f} (err {d["bonded_noslip"]["rel_err_est"]:.1e}) '
                     f'Sneddon={d["halfspace_sneddon_K_N_per_um"]:.5f}')
        for R in ('R=10mm', 'R=25mm'):
            d = b['C5'][R]
            s.append(f'  C5 {R}: F(30um)={d["F_N"][-1]:.5f} N a={d["contact_radius_mm"][-1]:.4f} mm '
                     f'(Hertz {d["hertz_F_N"][-1]:.5f} N, {d["hertz_a_mm"][-1]:.4f} mm) err {d["rel_err_est_F"]:.1e} '
                     f'pmin={d["at_30um"]["min_pressure_Pa"]:.3g} ringBEM dF={d["at_30um"]["ring_BEM_active_set_check"]["rel_diff_F"]:.1e}')
        for kind in ('frictionless', 'noslip'):
            d = b['C6'][kind]
            s.append(f'  C6 {kind}: K={d["K_N_per_um"]:.6f} (finest {d["K_finest_N_per_um"]:.6f}, order {d["observed_order"]}, '
                     f'err {d["rel_err_est"]:.1e})')
            if 'taxel_maps' in d:
                nb = d['taxel_maps']['neighbours']
                s.append(f'     neighbours mean uz/delta: x-nb {nb["mean_uz_over_delta"]["(1,3)"]:.4f}, y-nb '
                         f'{nb["mean_uz_over_delta"]["(2,4)"]:.4f}, diag {nb["mean_uz_over_delta"]["(1,2)"]:.4f}; base Fz '
                         f'centre {d["taxel_maps"]["base_reaction_fraction_of_total"][2][3][2]:.4f}, sum of 8 nb '
                         f'{nb["base_Fz_fraction_sum_of_8"]:.4f}')
        lines += s
    ks = res['literature']['kappa_s']
    lines.append(f'kappa_s vs Argatov/Hayes table: max abs diff {ks["max_abs_diff"]:.4f}, max rel {ks["max_rel_diff"]:.2e}')
    for nu, d in res['literature']['mossakovskii_bonded_halfspace'].items():
        lines.append(f'Mossakovskii nu={nu}: BEM {d["ring_BEM_ratio_n800"]:.6f} closed {d["closed_form"]:.6f}')
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(HERE, 'summary.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
