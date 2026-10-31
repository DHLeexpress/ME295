"""C6/C6f cross-check against EVERY gt-fem run level (../gt_fem/runs/*.json, read only), including the level-3 runs
that finished after gt_fem/results.json (20:42) was assembled.  Writes compare_gt_fem_levels.json, prints a table."""
import json, os, glob, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
me = json.load(open(os.path.join(HERE, 'results.json')))
rows = []
for fn in sorted(glob.glob(os.path.join(HERE, '..', 'gt_fem', 'runs', 'C6*_nu*_m*.json'))):
    mt = re.match(r'(C6f?)_(full|mixed)_nu([\d.]+)_m(\d)\.json', os.path.basename(fn))
    if not mt: continue
    case, form, nu, lev = mt.group(1), mt.group(2), float(mt.group(3)), int(mt.group(4))
    f = json.load(open(fn))
    kind = 'frictionless' if case == 'C6f' else 'noslip'
    m = me[f'nu={nu:g}']['C6'][kind]
    Kf = f['K_energy_N_per_mm'] * 1e-3; Km = m['K_N_per_um']
    bf = np.array(f['taxel_force_on_blade_N_per_mm']) * 1e-3 / Kf
    bm = np.transpose(np.array(m['taxel_maps']['base_reaction_fraction_of_total']), (1, 0, 2))
    uf = np.array(f['taxel_mean_disp_per_delta'])
    um = np.transpose(np.array(m['taxel_maps']['mean_surface_displacement_per_delta']), (1, 0, 2))
    px = np.array(f['profile_x_mm']) * 1e-3; uzf = np.array(f['u_free_at_y0_per_delta'])[:, 2]
    xm = np.array(m['profiles_per_delta']['x_m']); sel = (xm > 6e-3) & (xm < 25e-3)
    prof = float(np.max(np.abs(np.interp(xm[sel], px, uzf) - np.array(m['profiles_per_delta']['uz_x0'])[sel])))
    rows.append(dict(case=case, nu=nu, form=form, level=lev, n_dof=f['n_dof'], K_fem=Kf, K_transform=Km,
                     rel_diff_K=(Kf - Km) / Km, base_frac_maxabs_diff=float(np.max(np.abs(bf - bm))),
                     mean_disp_maxabs_diff=float(np.max(np.abs(uf - um))), profile_uz_maxabs_diff=prof))
json.dump(rows, open(os.path.join(HERE, 'compare_gt_fem_levels.json'), 'w'), indent=1)
print(f"{'case':4s} {'nu':>5s} {'form':5s} L {'ndof':>8s} {'K_fem':>9s} {'K_gt':>9s} {'dK/K':>9s} {'dBase':>8s} {'dDisp':>8s} {'dProf':>8s}")
for r in sorted(rows, key=lambda r: (r['case'], r['nu'], r['form'], r['level'])):
    print(f"{r['case']:4s} {r['nu']:5.3f} {r['form']:5s} {r['level']} {r['n_dof']:8d} {r['K_fem']:9.6f} {r['K_transform']:9.6f} "
          f"{r['rel_diff_K']:+9.2e} {r['base_frac_maxabs_diff']:8.1e} {r['mean_disp_maxabs_diff']:8.1e} {r['profile_uz_maxabs_diff']:8.1e}")
