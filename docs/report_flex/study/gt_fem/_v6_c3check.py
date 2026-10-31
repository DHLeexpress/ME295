# V6 follow-up: C3 half vs full model with identical (mirrored) meshes (x = 0 forced as a mesh line in the full model)
import numpy as np, cases, fem_layer as fl
p = dict(hmin=0.3, hmax=4.0, hzmax=0.75)
orig = fl.graded_breaks
cases.CASES["C3_full"] = dict(cases.CASES["C3"], symx=None); cases.MESH["C3_full"] = cases.MESH["C3"]
r1 = cases.run("C3", 0.45, "full", 1, p=p, verbose=False)
r2 = cases.run("C3_full", 0.45, "full", 1, p=p, verbose=False)
def gb(lo, hi, lines, sing, *a):
    return orig(lo, hi, list(lines) + ([0.0] if lo < 0 < hi and abs(hi - 25.5) < 1e-9 else []), sing, *a)
fl.graded_breaks = gb
r3 = cases.run("C3_full", 0.45, "full", 1, p=p, verbose=False)
for n, r in (("half", r1), ("full (own mesh)", r2), ("full (mirrored mesh)", r3)):
    print(n, r["n_elem"], r["F_platen_N_per_mm"], "taxel max", np.max(np.abs(r["taxel_force_N_per_mm"])))
T1, T3 = np.array(r1["taxel_force_N_per_mm"]), np.array(r3["taxel_force_N_per_mm"])
print("half vs mirrored-full: F rel diff %.1e, taxel rel diff %.1e" % (
    np.max(np.abs(np.array(r1["F_platen_N_per_mm"]) - r3["F_platen_N_per_mm"])) / abs(r3["F_platen_N_per_mm"][2]),
    np.max(np.abs(T1 - T3)) / np.max(np.abs(T3))))
