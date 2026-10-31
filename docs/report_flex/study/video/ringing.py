"""Ringing / chatter statistics per segment from prod2/seg<k>.npz -> prod2/ringing.json (README item 4)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, os.environ.get("RUNS", "prod2"))
out = {}
for k in range(8):
    z = np.load(os.path.join(R, f"seg{k}.npz"), allow_pickle=True); ph = z["phase"]; h, s = ph == 2, ph == 3
    out[k] = dict(hold_uzmax_std_um=float(z["uz_max"][h].std() * 1e6), hold_uzmax_p2p_um=float(np.ptp(z["uz_max"][h]) * 1e6),
                  hold_Fz_std_N=float(z["F"][h, 2].std()), slide_Fz_std_N=float(z["F"][s, 2].std()))
    print(k, {a: round(b, 4) for a, b in out[k].items()})
json.dump(out, open(os.path.join(R, "ringing.json"), "w"), indent=1)
