# Print a compact table from official_examples.json (written by official_examples.py) -> official_examples_table.txt
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'official_examples.json')))
dofn = {0: 'full', 1: 'trilin', 2: 'quad'}
lines = [f"MuJoCo {d['mujoco']}; each model stepped {d['sim_time_per_model']} s from its initial state (no control input)",
         f"{'file':26s} {'integrator':12s} {'solver':6s} {'h':>6s} | {'flex':10s} d {'dof':6s} {'nvert':>5s} {'cells':>6s} {'young':>7s} {'nu':>4s} {'damp':>6s} {'rad_mm':>6s} eq {'solref':>13s} {'solimp[0:3]':>19s} mov {'CFL':>6s} {'lmin_mm':>7s} {'us/step':>8s} div"]
for r in d['rows']:
    o = r.get('option', {})
    st = r.get('step', {})
    for f in r.get('flexes', []):
        cfl = f.get('cfl')
        cflstr = f"{cfl:6.2f}" if (cfl is not None and f['interp'] == 0 and f.get('rho_kg_m3', 0) > 0) else '     -'
        lm = f"{f['l_min_m']*1e3:7.1f}" if ('l_min_m' in f and f['interp'] == 0) else '      -'
        cells = 'x'.join(str(c) for c in f['cellnum']) if f['cellnum'] else '-'
        integ = o.get('integrator', '').replace('mjINT_', '')
        solver = o.get('solver', '').replace('mjSOL_', '')
        lines.append(f"{r['file'][:26]:26s} {integ[:12]:12s} {solver[:6]:6s} {o.get('timestep', 0):6.0e} | {f['name'][:10]:10s} {f['dim']} "
                     f"{dofn.get(f['interp'], str(f['interp'])):6s} {f['nvert']:5d} {cells:>6s} {f['young']:7.0e} {f['poisson']:4.2f} "
                     f"{f['damping']:6.0e} {f['radius']*1e3:6.1f} {f['edgeequality']:2d} {str([round(x, 4) for x in f['solref']]):>13s} "
                     f"{str([round(x, 4) for x in f['solimp'][:3]]):>19s} {'Y' if f['attached_to_moving_body'] else 'n':3s} {cflstr} {lm} "
                     f"{st.get('us_per_step', float('nan')):8.0f} {st.get('diverged_at_step')}")
txt = "\n".join(lines)
open(os.path.join(HERE, 'official_examples_table.txt'), 'w').write(txt + "\n")
print(txt)
