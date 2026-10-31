"""Check the friction cone in the CONTACT frame for segment 4 (book corner), where the pad-frame ratio |F_t|/F_z
reached 3.1 during the slide. Re-runs sim.run(3) to step 1620 (covers the worst steps 1391-1615) with a wrapper on
sim.contact_forces_pad that records, per step, max over contacts of |f_tangential|/f_normal (contact frame) and the
tilt of the contact normals from the pad normal.  Output: test/cone/cone.json"""
import json, os
import numpy as np
import sim
rec = []
orig = sim.contact_forces_pad
def wrap(m, d, flexid=0):
    out = orig(m, d, flexid)
    n = d.ncon
    if n:
        adr = np.asarray(d.contact.efc_address[:n])
        f = np.array([d.efc_force[a:a + 3] for a in adr])
        nz = np.asarray(d.contact.frame[:n]).reshape(-1, 3, 3)[:, 0, 2]
        ok = f[:, 0] > 1e-6
        r = np.hypot(f[ok, 1], f[ok, 2]) / f[ok, 0] if ok.any() else np.zeros(1)
        rec.append([float(d.time), int(n), float(r.max()), float(np.degrees(np.arccos(np.clip(np.abs(nz), 0, 1))).max())])
    return out
sim.contact_forces_pad = wrap
sim.SOLREF = (3e-6, 1.0)            # as prod2 (sim.py CLI --solref0 3e-6)
out = os.path.join(sim.HERE, "test", "cone")
sim.run(3, [40, 56, 3], 3000, out, max_steps=1620)
a = np.array(rec)
res = dict(n_steps_with_contact=len(a), max_contactframe_ft_over_fn=float(a[:, 2].max()),
           frac_steps_ratio_gt_1p001=float((a[:, 2] > 1.001).mean()), max_normal_tilt_deg=float(a[:, 3].max()),
           median_normal_tilt_deg_last200=float(np.median(a[-200:, 3])))
json.dump(res, open(os.path.join(out, "cone.json"), "w"), indent=1)
print(res)
