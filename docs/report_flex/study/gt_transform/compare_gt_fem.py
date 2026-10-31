"""Cross-check of C4 and C6 against the finite-pad FEM of the gt-fem agent (../gt_fem/results.json, read only).
Writes compare_gt_fem.json and prints a summary."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
me = json.load(open(os.path.join(HERE, 'results.json')))
fem = json.load(open(os.path.join(HERE, '..', 'gt_fem', 'results.json')))
out = {}
for case, kind in (('C6f', 'frictionless'), ('C6', 'noslip')):
    for nu in (0.45, 0.49, 0.499):
        f = fem['cases'][case][f'nu{nu:.3f}']
        m = me[f'nu={nu}']['C6'][kind]
        tm = m['taxel_maps']
        Kf, Km = f['K_N_per_um'], m['K_N_per_um']
        bf = np.array(f['taxel_force_on_blade_N_per_um']) / Kf                     # [j][i][c]
        bm = np.transpose(np.array(tm['base_reaction_fraction_of_total']), (1, 0, 2))
        uf = np.array(f['taxel_mean_disp_per_delta'])
        um = np.transpose(np.array(tm['mean_surface_displacement_per_delta']), (1, 0, 2))
        px = np.array(f['profile_x_mm']) * 1e-3
        uzf = np.array(f['u_free_at_y0_per_delta'])[:, 2]
        pm = m['profiles_per_delta']
        xm = np.array(pm['x_m'])
        sel = (xm > 6.0e-3) & (xm < 25e-3)
        uzf_i = np.interp(xm[sel], px, uzf)
        out[f'{case} nu={nu}'] = dict(
            K_fem=Kf, K_fem_err=f['K_err_N_per_um'], K_transform=Km, K_transform_err=m['rel_err_est'] * Km,
            rel_diff_K=(Kf - Km) / Km,
            base_force_fraction_maxabs_diff=float(np.max(np.abs(bf - bm))),
            base_Fz_centre=(float(bf[3, 2, 2]), float(bm[3, 2, 2])),
            mean_disp_per_delta_maxabs_diff=float(np.max(np.abs(uf - um))),
            mean_uz_xneighbour=(float(uf[3, 1, 2]), float(um[3, 1, 2])),
            profile_uz_y0_maxabs_diff_6_to_25mm=float(np.max(np.abs(uzf_i - np.array(pm['uz_x0'])[sel]))),
            fem_profile_err_reported=f['profile_err_max']['u_free_at_y0_per_delta'])
c4 = fem.get('C4_axisymmetric', {})
out['C4_note'] = 'C4 compared from gt_fem/summary_table.md (axisymmetric FEM) in README'
json.dump(out, open(os.path.join(HERE, 'compare_gt_fem.json'), 'w'), indent=1)
for k, v in out.items():
    if isinstance(v, dict):
        print(k, {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()})
