import sys, time, resource, numpy as np
import cases
m = int(sys.argv[1]); thr = int(sys.argv[2]); form = sys.argv[3] if len(sys.argv) > 3 else "full"
nu = float(sys.argv[4]) if len(sys.argv) > 4 else 0.45
out = cases.run("C1", nu, form, m, threads=thr)
print("Kn [N/um] = %.7f  F = %s  maxrss %.1f GB" % (out["F_platen_N_per_mm"][2] / 1e3, out["F_platen_N_per_mm"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
print("solver", out["solver"])
